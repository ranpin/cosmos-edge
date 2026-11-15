# CR2 篇 —— Cosmos-Reason2-8B/2B 量化部署

Cosmos 家族推理 VLM（`nvidia/Cosmos-Reason2-8B` / `-2B`）的 INT4 AWQ / INT8 SmoothQuant 量化部署
与五轴实测（Jetson AGX Orin sm_87 · TensorRT-Edge-LLM）。**方法学建立之地**：五轴评测 · roofline
顶格 · 预注册闸门 · TOST 等价检验均在本篇建立，后被 Cosmos3 篇与姊妹仓
[alpamayo-edge](https://github.com/ranpin/alpamayo-edge)（R1 backbone INT4）复用。

📄 **主报告（图 1–22 / 表 1–13）**：<https://ranpin.github.io/cosmos-edge/cr2.html>

⛏️ **在线 Demo（离线批量场景挖掘：8 clip 批作业 → 结构化事件标签 + 实测吞吐/能耗）**：<https://ranpin.github.io/cosmos-edge/cr2-demo/>

## 核心结论

- **量化是 32GB 边缘的部署前提**：FP16 build（峰值 54.8G）/载入双 OOM；INT4 decode 快 2.66×、引擎 4.8G。
- **量化基本无精度损失**（官方基准 n=210/510 + 自建驾驶域 n=184，McNemar 全不显著 + TOST 等价）；
  **缩小到 2B 显著掉点**（−8.6~11.0 pts，p=0.005）⇒ 部署策略 =「激进量化 + 保留大模型」。
- **选型随负载形状翻转**：输出 token < 输入 token / 83 时 INT8 更快（n=210 实测 1.43×）。
- **decode 已达硬件上限**：达可达带宽 97–100%，手写 GEMV 对照证否 kernel 优化空间（`eval/kernels/`）；
  lm_head INT4 落地后 TPOT 再 +19.6%。
- 谱系：CR2-8B 正是 **Alpamayo 1.5 的 VLM backbone**；报告页 §1.13–1.14 为 R1 INT4 / Alpamayo2-Edge
  两章时间线延伸（资产在 alpamayo-edge 仓）。

## 目录

```
cloud/     Kaggle T4×2 量化脚本全套（INT4/INT8/lm_head/2B/ViT 侦察/FP16 参考）
data/      驾驶域 n=184 评测集构建（PhysicalAI-AV，gated）
docs/      METHODOLOGY / edge_deploy_status（结果报告）/ orin_build_notes（工程日志）/
           PROBLEMS §A–J / FINDINGS / M0 / M5 / plan / next_plan / figures（图片唯一真源）
eval/      评测资产（见 eval/README.md）：accuracy / perf / profile / kernels / vision / io_example + viz 脚本
outputs/   量化产物 SHA256（tgz 大件已归档 scm）+ archive/（原始日志，本地未跟踪）
scripts/   quantize.sh + orin/（CR2 设备脚本：引擎构建 / 功耗 / Pareto / 持续负载）
```
