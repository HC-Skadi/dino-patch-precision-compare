# CUDA FlashAttention 2 与 NPU SDPA 精度比较

日期：2026-11-10。测试入口为 `compare_glm_attention.py`，第一轮仅测试算子，不加载 GLM 权重，不修改正式推理代码或安装依赖。

## 固定测试口径

- batch1，96 个 Q heads、8 个 KV heads，每个 head 128 维，对应 GLM-4.5-Air 的 GQA 配置。
- 默认 BF16、64 个 KV token，dropout=0、scale=1/sqrt(128)。两端先在 CPU 生成并舍入同一份 Q/K/V，再上传到设备；原始输入和舍入输入分别记录 SHA256。
- 输入属于固定合成的 attention 夹具，视为 RoPE 之后的 Q/K/V，不代表真实模型激活分布。
- CUDA 直接调用 `flash_attn.flash_attn_func`，要求 FlashAttention 2 >=2.1、<3，无其他后端回退。
- NPU 使用已安装 Transformers 的 `sdpa_attention_forward`，保留其 GQA、KV 重复与 mask 处理逻辑，记录版本和该入口源码摘要。历史 GLM 环境采用 Transformers 5.13.0。
- 输出保存为 FP16/BF16 原始 uint16 位模式，避免转换为 FP32 后扩大文件，也不重新量化输出。

| 场景 | Q 长度 | KV 长度 | CUDA FA2 | NPU SDPA |
|---|---:|---:|---|---|
| prefill | 64 | 64 | causal=True | 无显式 mask，因果 attention |
| decode | 1 | 64 | causal=True，右下对齐，允许全部历史 KV | 无显式 mask，is_causal=False |
| chunk_decode | 4 | 64 | causal=True，右下对齐 | 显式右下对齐布尔 mask，is_causal=False |

decode 不能直接照搬 PyTorch SDPA 的 `is_causal=True`。当 Q、KV 长度不同，FA2 >=2.1 使用右下对齐；PyTorch 的普通因果三角形不能表达相同的缓存位置。单 token decode 应允许读取全部已有 KV，多 token decode 使用绝对位置构造 mask。[FlashAttention 官方说明](https://github.com/Dao-AILab/flash-attention/blob/main/README.md#21-change-behavior-of-causal-flag)

这三个场景仅模拟已有 KV 的 attention 计算，不执行 cache 写入、QKV 投影、RoPE 或完整生成。初轮默认只执行每个场景一次；`--timing` 可选增加 3 次预热和 10 次短同步计时，包含接口内的布局转换和 mask 准备，不能称为纯 kernel 耗时。

## NPU 执行

将仓库代码放到 `/home/hc` 下的实验目录，在已有环境执行。输出目录必须是新目录：

```bash
source /home/hc/envs/gns83/env-npu.sh
cd /home/hc/diagnostics/glm_attention_compare_20261110
python3 -B -u compare_glm_attention.py \
  --device npu:7 --dtype bf16 --length 64 \
  --out glm_attn_npu_results
```

无需传输模型权重或大份输入数据。两端按照固定算法生成输入，并在比较时核对两份输入摘要。需要传输的 NPU 文件仅为 `glm_attn_npu_results/`，其中输出的未压缩原始位数据合计约 1.70 MB，npz 压缩后的实际大小以文件为准。

## A100 执行

更新同一版本脚本，将完整 `glm_attn_npu_results/` 放到仓库目录，再执行：

```bash
cd /workspace/dino-patch-precision-compare
git pull --ff-only
OUT="glm_attn_a100_results_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
python3 -u compare_glm_attention.py \
  --device cuda:0 --dtype bf16 --length 64 \
  --out "$OUT" --compare glm_attn_npu_results
```

CUDA 环境需要已有 PyTorch CUDA、NumPy 和 FlashAttention 2。脚本不会安装依赖，也不会用 CUDA SDPA 代替缺失的 FlashAttention 2。

## 如何读取结果

- `comparison.csv`：NPU SDPA 对未修改 CUDA FA2 的 max_abs、RMSE、相对 L2 差，三个场景分别记录，这是主要结果。
- `accuracy.csv`：各设备对同精度输入的 CPU FP64 参考，辅助区分原生运算误差，不代替跨设备结果。
- `report.json`：版本、入口、输入与脚本摘要、形状、dtype 和输出摘要。
- `outputs.npz`：原始低精度输出位模式，可离线复核。

短序列结果只说明本夹具上的算子一致性。确认这一步后，再增加序列长度或使用真实模型同层、同事件的 post-RoPE Q/K/V，不同时修改权重、投影、RoPE 或输出头。

## 当前状态

脚本已实现并上传到 node02 的 `/home/hc/diagnostics/glm_attention_compare_20261110/`。9 项 CPU 单元测试全部通过，日志见 `GLM_ATTENTION_CPU_TEST.log`，覆盖输入身份、GQA、布局、缓存位置掩码、原始位存储、比较方向和已安装 Transformers 5.13.0 的 SDPA 分发逻辑。日志中的零差异表来自模拟数组，仅用于验证比较程序。

2026-11-10 已在 node02 的 Ascend 910B3、NPU 7 上完成 BF16、batch1、KV 长度 64 的三个场景。使用现有 PyTorch / torch_npu 2.9.0 和 Transformers 5.13.0 环境，没有安装依赖，没有加载模型权重，每个场景仅执行一次设备计算，没有额外计时。

下表以同一份 BF16 舍入输入的 CPU FP64 输出为参考：

| 场景 | 最大绝对差 | RMSE | 相对 L2 差（%） |
|---|---:|---:|---:|
| prefill | 9.66765e-3 | 6.62179e-4 | 0.184370 |
| decode | 2.56257e-3 | 4.17240e-4 | 0.213317 |
| chunk_decode | 3.85818e-3 | 4.22886e-4 | 0.210901 |

原始输出、精度指标和环境记录保存在 `glm_attn_npu_results/`，运行日志为 `GLM_ATTENTION_NPU_RUN.log`。远程目录为 `/home/hc/diagnostics/glm_attention_compare_20261110/glm_attn_npu_results/`。日志中仅出现 PyTorch 精度开关 API 的弃用提示，三个场景均正常完成。

CUDA FA2 尚未执行。本轮结果用于建立 NPU 基准，CUDA / NPU 的直接差异需要按上述 A100 命令生成 `comparison.csv` 后判断。
