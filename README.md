# Cosmos-Edge

**NVIDIA Cosmos 家族模型在 Jetson AGX Orin（sm_87）上的边缘部署全史**——同一套 TensorRT-Edge-LLM
管线与评测方法学量到家族两代模型上，仓库按篇分治：**`cr2/`**（Cosmos-Reason2-8B/2B 推理 VLM 的
INT4/INT8 量化部署，方法学建立之地）与 **`cosmos3/`**（Cosmos3-Edge 4B 全模态世界模型，跨代反例与
policy 能力边界）。全部为 Orin 实机实测，非模拟。零预算 · 零训练。

🎬 **在线 Demo（Cosmos3-Edge 2B 驾驶场景自由描述，真实 Orin 推理输出）**：<https://ranpin.github.io/cosmos-edge/cosmos3-demo/>

⛏️ **在线 Demo（CR2-8B INT4 离线批量场景挖掘：8 clip 批作业 → 结构化事件标签 + 实测吞吐/能耗）**：<https://ranpin.github.io/cosmos-edge/cr2-demo/>

📊 **跨代精度对比专篇（n=184 驾驶域，Cosmos3-Edge 2B vs CR2-8B 三精度）**：<https://ranpin.github.io/cosmos-edge/cosmos3.html>

📄 **CR2 篇主报告（8B 量化部署实战，图 1–22 / 表 1–13）**：<https://ranpin.github.io/cosmos-edge/cr2.html>

> **仓库沿革**：2026-10-29 Cosmos3-Edge 工作自 alpamayo-edge 拆出独立成仓（cosmos3-edge）；
> 2026-11-15 按 NVIDIA 模型家族重组——CR2 量化部署全套迁入并改名 **cosmos-edge**，同日目录按
> `cr2/` / `cosmos3/` 两篇分治。Alpamayo 家族（R1 / 1.5 / 2-Edge 轨迹 VLA）在姊妹仓
> [alpamayo-edge](https://github.com/ranpin/alpamayo-edge)；两仓共享设备资产（dog/goat 两台 Orin，
> 布局见 [device_layout.md](https://github.com/ranpin/alpamayo-edge/blob/main/docs/device_layout.md)）
> 与 TRTEdge 0.10.1 构建脚本（`cosmos3/scripts/orin/w0b_build_trtedge_0101.sh`，两仓并存）。

## 家族地图

| | Cosmos 家族（本仓） | Alpamayo 家族（[alpamayo-edge](https://github.com/ranpin/alpamayo-edge)） |
|---|---|---|
| 模型 | Cosmos-Reason2-8B/2B（推理 VLM）、Cosmos3-Edge 4B（世界模型） | Alpamayo-R1-10B、1.5、2-Edge（AV 轨迹 VLA） |
| 谱系关联 | CR2-8B **正是 Alpamayo 1.5 的 VLM backbone**；Cosmos3-Edge backbone **正是 Alpamayo2-Edge 的底座** | 建在 Cosmos 家族 backbone 之上的 AV 栈 |
| 主线结论 | 「激进量化 + 保留大模型」（推理域）；跨代 2B 显著输 8B 量化版 | R1 backbone INT4「量化近乎免费」延伸到轨迹域；轨迹域「缩小免费」（2-Edge 2.65B ≈ R1 10B） |

## 核心结果

### CR2 篇（`cr2/`）—— Cosmos-Reason2-8B/2B 量化部署

- **量化既是提速也是 32GB 边缘的唯一可部署方案**：FP16 引擎在 32GB Orin 上 build（峰值 54.8GB）与
  载入双双 OOM；INT4 decode 比 FP16 快 2.66×、E2E 2.5×，引擎 4.8GB。
- **量化基本无精度损失**（官方基准 n=210 + 自建驾驶域 n=184 双证据，McNemar 全不显著 + TOST 等价检验）；
  **缩小模型不免费**：CR2-2B 掉 8.6–11.0 pts（p=0.005 显著）⇒ 部署策略 =「激进量化 + 保留大模型」。
- **选型随负载形状翻转**：输出 token < 输入 token / 83 时 INT8 更快（n=210 基准实测 INT8 快 1.43×）。
- **已到硬件上限**：roofline + kernel 级剖析 + 手写 GEMV 对照——decode 达可达带宽 97–100%，
  手写 kernel 无空间；lm_head INT4 落地后 TPOT 再 +19.6%。
- 报告页含 §1.13–1.14（R1 INT4 / Alpamayo2-Edge 跨代）两章时间线延伸，资产在 alpamayo-edge 仓。

### Cosmos3 篇（`cosmos3/`）—— Cosmos3-Edge 世界模型

- **AV reasoning 达 3.4× 随机基线**：n=184 驾驶域 MC 42.4%——该口径用作同条件跨代对比；自由描述质量见在线 demo（真实 Orin 输出）。
- **跨代反例**：新一代 2B reasoner 在驾驶域**显著输给**上一代 8B 的量化版（−9.2 pts，p=0.0115）
  ⇒「缩小不免费、量化几乎免费」在第二个模型上同向成立（2B 量化臂因 TRTEdge 0.10.1
  工具链约束缺失，源码级核实见 `cosmos3/eval/reasoning/README.md`）。
- **DROID policy 动作链部署打通 + 诚实边界**：四组件引擎建成、冒烟通过（action [1,16,10] 全 finite）；
  但 W4 实测 turnkey 输出由 diffusion seed 主导（52%）、conditioning 三轴仅 1–2%，根因未定
  ⇒ 定位为「部署打通 + conditioning 弱耦合边界」，demo 锚定 reasoning 侧。

## 仓库结构

```
├── cr2/                        ── CR2 篇：Cosmos-Reason2-8B/2B 量化部署 ──
│   ├── README.md                 篇导读（核心结论 + 目录）
│   ├── cloud/                    Kaggle T4×2 量化脚本全套
│   ├── data/                     驾驶域 n=184 评测集构建
│   ├── docs/                     METHODOLOGY / 结果报告 / 工程日志 / PROBLEMS §A–J / figures
│   ├── eval/                     accuracy / perf / profile / kernels / vision / io_example + viz 脚本
│   ├── outputs/                  产物 SHA256 + archive/（原始日志，本地未跟踪）
│   └── scripts/                  quantize.sh + orin/（CR2 设备脚本）
├── cosmos3/                    ── Cosmos3 篇：Cosmos3-Edge 世界模型 ──
│   ├── README.md                 篇导读（核心结论 + 目录）
│   ├── docs/                     DEPLOYMENT（六道墙 + 分解补丁）/ PROBLEMS §K–L / figures
│   ├── eval/                     reasoning/（n184 跨代）+ policy/（W3/W4 动作链）
│   └── scripts/orin/             Phase W 部署链（TRTEdge 0.10.1 / 引擎构建 / 补丁 / 冒烟）
├── site/                       呈现站点（gh-pages 源）
│   ├── index.html                家族总览
│   ├── cr2.html                  CR2 篇主报告（图 1–22 / 表 1–13）
│   ├── cosmos3.html              跨代精度对比专篇
│   ├── cosmos3-demo/             reasoning 自由描述 demo（4 公开 clip）
│   └── figures/{cr2,cosmos3}/    站点插图（亮色 + dark/ 暗色变体）
└── .claude/skills/             edge-quantize-deploy（量化部署工作流 skill）
```

## 边缘部署要点

- **TRTEdge**：CR2 篇 0.9.0→0.10.1，Cosmos3 篇 0.10.1（JetPack 6.2 属 Compatible 档，0.11+ 已删该行）。
- **Jetson Orin sm_87 量化选项空间已穷举**：骨干 int4_awq/int8_sq、lm_head int4_awq 可用；
  视觉编码器/KV cache 官方仅 fp8（sm_87 无）⇒ W4A8 与 fp8 路线在此硬件上不存在（cr2/docs/METHODOLOGY.md §3.5）。
- Cosmos3 的六道 JetPack 6.2 兼容墙与分解注意力/RoPE 补丁见 [cosmos3/docs/DEPLOYMENT.md](cosmos3/docs/DEPLOYMENT.md)。

## 许可与数据来源

- 代码 **MIT**（见 LICENSE）。
- 模型：`nvidia/Cosmos-Reason2-8B`/`-2B`（gated，已授权）、`nvidia/Cosmos3-Edge`（OpenMDW1.1，非 gated）。
- 基准：官方 `nvidia/Cosmos-Reason1-Benchmark`（robovqa/robofail，CR2 无官方 benchmark）；
  驾驶域 n=184 自建自 gated `nvidia/PhysicalAI-Autonomous-Vehicles`（人工核验标注，逐 clip 原始视频不入库）。
- reasoning demo 帧取自 dgural 公开样本（PhysicalAI-AV 公开 100-clip 子集）。
- `cosmos3/eval/reasoning/ref_cr2/` 为 2026-11-03 迁移期的 CR2 对照数据副本；2026-11-15 起 canonical
  在 `cr2/eval/accuracy/driving/`，副本按实测记录铁律原样保留。
