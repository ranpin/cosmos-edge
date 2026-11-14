# CR2 篇评测资产

Cosmos-Reason2-8B/2B 量化部署的全部评测资产（报告页 `site/cr2.html`，方法与口径 `docs/METHODOLOGY.md`）。

| 目录 / 脚本 | 内容 |
|---|---|
| `accuracy/` | 精度轴：12-prompt 探针 + 官方基准 n=210/510（robovqa/robofail）+ 驾驶域 n=184 + 分辨率 Pareto + 校准集消融 + TOST/统计校正（各带 README） |
| `perf/` | 性能轴：三精度 TTFT/TPOT/TPS、len4096 对照、lm_head、31min 持续负载、延迟预算 |
| `profile/` | Nsight kernel 级剖析（decode 时间构成） |
| `kernels/` | 手写 W4A16 GEMV 微基准（TRT 达可达带宽 97% 的证否实验） |
| `vision/` | 视觉编码器扫描（1/2/4/6 帧线性律 + io 逐字请求响应） |
| `io_example/` | 真实逐字 I/O 实例（freeform + robovqa 6 帧） |
| `roofline.py` | roofline 分析（脊点 210 FLOP/byte，decode 访存墙 / prefill 计算墙） |
| `bench_viz.py` / `pareto_viz.py` / `profile_viz.py` / `driving_viz.py` / `experiment_logic.py` 等 | 站点图 1–22 的确定性重渲脚本（数据驱动，无硬编码数字） |

> 打分可复现：`accuracy/benchmark/score_mc.py` 逐一复现历史全部数字（含 CI 与 McNemar p）。
> 驾驶域 n=184 的 manifest 与构建脚本在 `../data/`；跨代对比（Cosmos3 侧）在 `../../cosmos3/eval/reasoning/`。
