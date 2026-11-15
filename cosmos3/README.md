# Cosmos3 篇 —— Cosmos3-Edge 世界模型

`nvidia/Cosmos3-Edge`（4B 全模态世界模型 = 2B VL reasoner + rectified-flow 扩散动作专家，MoT 单塔）
在 Jetson AGX Orin 上的部署，覆盖两条能力线：**AV 驾驶场景理解（reasoning）** 与
**DROID 机器人操作（policy 动作链）**，以及与上一代 CR2-8B 的**跨代精度对比**。
全部为 Orin 实机推理输出，非模拟。

🎬 **在线 Demo（reasoning 侧 · 真实驾驶场景自由描述）**：<https://ranpin.github.io/cosmos-edge/cosmos3-demo/>

📊 **跨代精度对比专篇（n=184 驾驶域）**：<https://ranpin.github.io/cosmos-edge/cosmos3.html>

## 核心结论

- **AV reasoning 达 3.4× 随机基线**：n=184 驾驶域 MC **42.4%**（Wilson CI [35.5,49.6]），
  184/184 全 EOS、零解析失败——该口径用作同条件跨代对比；自由描述质量见在线 demo。
- **跨代反例**：新一代 2B reasoner 在驾驶域**显著输给**上一代 8B 的量化版（−9.2 pts，McNemar
  p=0.0115）⇒「缩小不免费、量化几乎免费」在第二个模型上同向成立（CR2 侧数据在 `../cr2/eval/accuracy/driving/`）。
- **DROID policy 部署打通 + 诚实边界**：四组件引擎建成、冒烟通过（action [1,16,10] 全 finite）；
  但 W4 实测 turnkey 输出由 **diffusion seed 主导（52%）**、帧/prompt/CFG 三个 conditioning 轴仅
  1–2%，根因未定 ⇒ 定位为「部署打通 + conditioning 弱耦合边界」，demo 锚定 reasoning 侧。
- **六道 JetPack 6.2 兼容墙**逐个打通（CuTe DSL / nlohmann / XQA JIT / ViTAttentionPlugin /
  TRT-11-only 算子），核心解法 = 分解注意力/RoPE 补丁 + 导出前数值对拍。
- 谱系：Cosmos3-Edge backbone 正是 **Alpamayo2-Edge 的底座**（2-Edge 跨代评测在 alpamayo-edge 仓）。

## 目录

```
docs/           DEPLOYMENT.md（六道墙 + 分解补丁 + 动作表示）/ PROBLEMS §K–L / figures（decomposed-attn）
eval/reasoning/ reasoner n=184 驾驶域跨代对比（原始输出 + cosmos3_viz.py + ref_cr2/ 对照副本）
eval/policy/    DROID 动作链：W3 帧敏感性 + W4 conditioning/seed 敏感性（脚本 + 证据 JSON）
scripts/orin/   Phase W 部署链：TRTEdge 0.10.1 自建 / 引擎构建 / 分解注意力补丁 / policy 冒烟
```
