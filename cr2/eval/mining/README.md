# CR2-8B INT4 离线批量场景挖掘（应用演示评测）

「0.1–2 Hz 事件级用例」定位的落地演示：CR2-8B INT4（backbone AWQ + lm_head INT4，最终部署形态）
在 orin-dog 上对 **dgural 公开样本** 8 段驾驶 clip 做一次批作业挖掘，输出结构化事件标签，
并实测吞吐与能耗。演示页：<https://ranpin.github.io/cosmos-edge/cr2-demo/>（2026-11-15 上线）。

## 协议

- **输入**：8 clips × 6 帧（camera_front_wide_120fov，长边 448、视频内均匀采样；4 段复用
  cosmos3-demo 帧、4 段本次自 dgural 视频抽帧）+ 挖掘指令（约定 JSON 键与取值域，事件词表与
  n=184 基准 8 事件簇同源）。
- **推理**：`llm_inference`（TRTEdge 0.9.0 共享安装），engines/8b_lmhead/llm + engines/int4/visual，
  greedy（temp=0, top_k=1）、max_generate_length=300、**batch_size=1**。
- **测量**：`run_pareto.sh` 同款功率窗口对齐采集（tegrastats 三轨之和，首请求开始后启动、
  进程退出即停）；逐请求 E2E 从日志时间戳解析（`parse_mining.py`，口径同
  `../accuracy/driving/pareto/parse_pareto.py`）。

## 结果（2026-11-15 实测）

| 项 | 值 |
|---|---|
| 批作业墙钟（纯处理窗口） | 20.2 s（8 clips / 48 帧） |
| E2E 均值 / 中位 | 2.52 s / 2.56 s per clip |
| 平均功率（三轨和，81 样本） | 50.1 W（峰值 62.8 W） |
| 能耗 / clip | 126.2 J（= P̄ × wall ÷ n） |
| 吞吐（连续运行外推） | ~1428 clips/hour（= 3600 ÷ E2E 均值） |
| JSON 合规率 | 8/8（贪心解码） |

事件检测的**带标注可靠性证据**不在本目录——见 `../accuracy/driving/RESULTS.json` 的分簇正确率
（n=184 OOD 基准：WORK_ZONES INT4 82.6% 最强、CYCLISTS 34.8% 最弱）。本目录的标签是模型输出，
dgural 公开样本无人工真值，仅作功能演示。

## 已知约束（如实记录）

- **batch_size=4 会失败**：视觉引擎预处理 cuSeqlens 超限（4480 > maxHW 4096，qwenViTRunner.cpp:368）
  ——6 帧/请求下并发批处理受视觉引擎容量限制，本次作业 batch_size=1 顺序处理。离线挖掘关心吞吐
  而非单请求延迟，约束不影响用例；要并发需重建更大 maxHW 的视觉引擎（未做）。
- 引擎口径与 n=184 INT4 臂不同：本目录用 8b_lmhead（lm_head INT4），n=184 用 engines/int4
  （lm_head fp16）——演示页表 2 已标注。

## 文件

| 文件 | 说明 |
|---|---|
| `build_input.py` | 输入 JSON 构造（8 请求、挖掘 prompt、dog 绝对路径） |
| `mining_input.json` | 本次运行的输入（实测记录，dog 路径原样保留） |
| `mining_out.json` | 8 条原始输出（实测记录） |
| `run_pareto_mining_dgural.log` / `pw_pareto_mining_dgural.log` / `run_mining.log` | 运行日志 / tegrastats 功率 / epoch 标记（实测证据） |
| `parse_mining.py` | 解析 → `mining_results.json`（演示页 DATA 的唯一数据源） |
| `mining_results.json` | 解析产物：逐 clip 标签 + E2E + 功率/能耗汇总 |
