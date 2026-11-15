# Cosmos3-Edge 评测（Phase W · W2）—— reasoner n=184 跨代对比

`nvidia/Cosmos3-Edge`（4B 全模态世界模型 = 2B VL reasoner + rectified-flow 扩散专家，MoT 单塔）。
部署链路与六道 JetPack 6.2 兼容墙见本仓 `docs/PROBLEMS.md` §K/§L 与 `docs/DEPLOYMENT.md`。
呈现页：<https://ranpin.github.io/cosmos-edge/cosmos3.html>（跨代精度对比专篇）。

> **迁移说明（2026-11-03 迁入本仓；2026-11-15 CR2 篇亦迁入）**：本目录原在 alpamayo-edge
> （2026-10-29 拆仓时跨代对比曾留彼处）。CR2-8B 侧对照数据以 `ref_cr2/` 副本随迁；2026-11-15
> 按 NVIDIA 模型家族重组仓库后，CR2 全套资产的 canonical 已在本仓 `cr2/eval/accuracy/driving/`
> （`ref_cr2/` 副本按实测记录铁律原样保留、不再回写）。policy 动作链（W3/W4）在本篇
> `../policy/`；reasoning 自由描述 demo 见
> <https://ranpin.github.io/cosmos-edge/cosmos3-demo/>。

## Reasoner 侧：n=184 驾驶域基准（2026-10-16，W2）

**协议与 CR2 报告页（本仓 `site/cr2.html`）§1.11/§1.12 完全一致**（同帧、同题、同标答、贪心、cap 12、6 帧 @448），
唯一变量是模型。输入直接复用 `bench/ood_eval/ood_input_goat.json`（messages 格式模型无关，
runtime 用引擎目录内自己的 chat template）。

| 模型 | 正确率 | Wilson 95%CI | vs CR2-8B FP16 | E2E/请求 |
|---|---|---|---|---|
| CR2-8B FP16（goat） | 51.6% | [44.5,58.7] | — | ~1.2s |
| CR2-8B INT4（dog） | 50.5% | [43.4,57.7] | −1.1（p=0.815 ns） | 0.68s |
| CR2-8B INT8（dog） | 47.8% | [40.7,55.0] | −3.8（p=0.143 ns） | 0.50s |
| **Cosmos3-Edge 2B FP16（goat）** | **42.4%** | [35.5,49.6] | **−9.2 pts（b/c=29/12，p=0.0115 显著）** | **0.91s** |

- 184/184 全 EOS、零解析失败、字母分布均衡（A17 B22 C20 D23 E29 F26 G19 H28，无坍缩）。
- 随机基线 12.5% ⇒ 2B reasoner 达 3.4× chance，有真实场景理解力。
- **结论（跨代验证项目核心发现）**：新一代 2B（Cosmos3，2026-07）在驾驶域显著输给上一代 8B
  的量化版（CR2 INT8/INT4）——「缩小模型不免费、量化几乎免费」在第二个模型家族上成立。
- 引擎：FP16，llm 3.7G + visual 940M（分解注意力版；插件版 visual 亦已建成，性能对比待做）。

## 文件

| 文件 | 说明 |
|---|---|
| `cosmos3_edge_fp16_outputs_n184.json` | reasoner n=184 原始输出（实测记录） |
| `cosmos3_viz.py` | 跨代对比图 + policy 帧不敏感图生成（`FIG_THEME=dark` 出暗色变体，输出 `site/figures/cosmos3[/dark]/`） |
| `ref_cr2/` | CR2-8B 侧对照数据副本（manifest_n184 / driving_results / orin_fp16_outputs_n184 / score_mc.py），2026-11-03 逐字节拷贝自 alpamayo-edge；2026-11-15 起 canonical 在本仓 `cr2/eval/accuracy/driving/` |
| 打分 | `ref_cr2/score_mc.py` 的解析/Wilson/McNemar（与 CR2 篇打分同源同实现） |

## 2B 量化臂：工具链侦察结论（2026-11-15，源码级）

跨代矩阵为 8B×{FP16, INT8, INT4} + 2B×{FP16}——**2B 的 INT4/INT8 臂缺失是工具链约束，不是未做**。
面试评审（interview-expert 四遍审查）把该缺格判为唯一淘汰级风险后，做了半天预注册侦察闸门，
判据 =「TRTEdge 0.10.1 的 cosmos3 reasoning 导出是否接受 int4_awq 且校准路径可执行」，结论 = **否**：

| # | 证据（dog 上 `trtedge-0.10.1/` 源码实查） | 含义 |
|---|---|---|
| 1 | `tensorrt_edgellm/quantization/` 全模块 grep cosmos3 **零命中** | AWQ/SQ 校准驱动不认识 cosmos3 架构，产不出量化 checkpoint |
| 2 | `models/cosmos3/export.py::export_cosmos3_components` 签名只有 `dtype=torch.float16`，无任何量化参数 | policy 组件导出无量化路径 |
| 3 | reasoning 导出走 `_export_llm(model_type="cosmos3_edge")`，只能消费已量化 checkpoint | 链条断在第一环（无量化 checkpoint 可产） |
| 4 | reasoner text 塔 = `Cosmos3ReasonerCausalLM(CausalLM)` + `Relu2MLP`（自定义 MoT，非 Qwen 系） | R1 式「repack 进受支持架构再量化」旁路架构上不成立（R1 可行是因与 qwen3_vl 逐层同构） |
| 5 | 上游 main（0.12+）changelog：cosmos3 只有「policy 部署指引扩展 / reasoner 视频输入」，无量化；且 0.11+ 支持矩阵已删 JetPack 6.2 行 | 升级工具链也解不开（Orin JP6.2 被钉在 0.10.x） |

**理论旁路（评估后放弃）**：用 modelopt.torch 对自定义 MoT 手工做 AWQ 校准并产出 TRTEdge 兼容的
column-packed int32 checkpoint ≈ 为 cosmos3 重写一遍 quantize.py 驱动——ViT INT8 的五连败先例
（cr2/docs/PROBLEMS.md §I1）表明此类管线工程风险高，为一个矩阵格不值得。

**对结论的影响（诚实口径）**：「量化几乎免费」在 8B（CR2）与 10B（R1，alpamayo-edge 仓）规模上已验证，
**2B 规模上的量化行为未测**。方向推断（非实测）：CR2 家族先例 2B INT4 比 8B INT4 再低 8.6 pts
（p=0.005）⇒ 小模型对量化更敏感，若 Cosmos3-2B 同理，跨代差只会更大、本页结论方向不变。
