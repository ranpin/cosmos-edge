# 设备端脚本 —— Cosmos3 篇（Phase W：Cosmos3-Edge 部署链，2026-10-15~19）

从两台 Orin 归档到仓库的原始脚本。设备上的副本在各自的
`chenrunbin/alpamayo-edge/scripts/`（设备目录名是历史遗留、不随仓库改名，布局见
[alpamayo-edge 仓 docs/device_layout.md](https://github.com/ranpin/alpamayo-edge/blob/main/docs/device_layout.md)）。

六道 JetPack 6.2 兼容墙与打通见 [`../../docs/PROBLEMS.md`](../../docs/PROBLEMS.md) §K/§L；
部署摘要见 [`../../docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md)。
CR2 篇设备脚本在 [`../../../cr2/scripts/orin/`](../../../cr2/scripts/orin/)。

| 脚本 | 设备 | 作用 | 产出/结论 |
|---|---|---|---|
| `w0b_build_trtedge_0101.sh` | dog | 自建 TRTEdge v0.10.1（共享 0.9.0 安装不动），`-DBUILD_EXPERIMENTAL_MODELS=ON` | K1（CuTe DSL 缺 sm_87+cuda12）暴露处；**与 alpamayo-edge 仓共享脚本（两仓并存）** |
| `w0c_goat_export_env.sh` | goat | python 导出环境（torch 2.13.0 / transformers 5.14.1 / onnx 1.19.0 硬 pin），与 gate4 venv 隔离 | export 是 CPU 任务 |
| `w1_goat_engine_build.sh` | goat | TRTEdge 0.10.1 C++ 构建 + reasoning 引擎构建（dog GPU 被 system_agent 占 26.5G ⇒ 改 goat）；需先打 vector_types.h 补丁（K3） | reasoning 引擎（分解注意力版） |
| `w1_cosmos3_engine_build.sh` | dog | reasoning 引擎构建（dog 版，与 goat 同源，maxInputLen 2048 / KV 4096） | 插件路径须指 0.10.1 自建 build |
| `apply_decomposed_attn_patch.py` | — | **K4 解法**：visual 导出加分解注意力路径（基础 ONNX 算子块对角 ragged self-attn） | 绕开 ViTAttentionPlugin 无 legacy 后端 + TRT_Attention 需 TRT≥11 |
| `apply_policy_decomposed_patch.py` | — | **K6 解法**：policy 的 und_prefill/gen 加 `decomposed_attention` + `decomposed_rope`，`USE_DECOMPOSED_POLICY=1` 门控 | 绕开 ONNX 标准 Attention/RotaryEmbedding 需 TRT≥11 |
| `test_policy_decomposed.py` | CPU | 导出前数值对拍（attention vs SDPA、rope vs LLaMA 全维公式），6 项全过 | err~1e-7 / err=0，避免 ~1h 导出后才发现语义 bug |
| `w2_policy_build.sh` | goat | policy 四组件分组件构建（und_prefill → gen → 重导出补 vae → vae） | vae_encoder 首次被杀（exit=120，内存压力） |
| `w2_policy_rebuild_decomposed.sh` | goat | 分解版重导 und_prefill+gen 再建引擎，分段执行保磁盘余量（起始 free 6.3G） | 四组件引擎建成 |
| `w2_policy_smoke.sh` | goat | `cosmos3_policy_inference` 冒烟（官方 UMI 首帧 + "pick up the object"，4 步扩散） | exit=0、action [1,16,10] 全 finite、5154ms/round；**只验管线不验语义** |

> W3 帧无关复测脚本在 [`../../eval/policy/w3_frame_sensitivity.py`](../../eval/policy/w3_frame_sensitivity.py)、
> W4 conditioning/seed 敏感性复测在
> [`../../eval/policy/w4_conditioning_sensitivity.py`](../../eval/policy/w4_conditioning_sensitivity.py)
> （归评测资产，不是设备构建脚本）。
