# BF16 核心算子与 Patch embedding 比较

更新时间：2026-11-10。

BF16 沿用 FP32 / FP16 的固定 `inputs.npz`，只增加计算精度，不改变输入生成规则、待测算子或正式模型。原入口默认仍执行 FP32 / FP16，BF16 使用独立结果目录。

## 已完成的 NPU 测试

node02，Ascend 910B3，`npu:7`，PyTorch 2.9.0、torch_npu 2.9.0。五项小规模 CPU 测试通过，六项核心算子和三种 Patch 路径均已在 NPU 执行。

以下核心结果以“输入和权重先舍入到 BF16，再用 CPU FP64 计算”的输出为参考，尚未涉及 A100：

| 算子 | 最大绝对差 | RMSE | 相对 L2 差（%） |
|---|---:|---:|---:|
| Linear | 3.90540e-3 | 6.48735e-4 | 0.165781 |
| LayerNorm | 1.49760e-2 | 1.66676e-3 | 0.166443 |
| erf GELU | 8.36736e-3 | 2.05606e-3 | 0.165581 |
| 原生 GELU | 8.19435e-3 | 1.73696e-3 | 0.139883 |
| SDPA | 8.04199e-4 | 1.16911e-4 | 0.222254 |
| Patch：Linear＋独立 bias | 1.51158e-2 | 1.30683e-3 | 0.235077 |

三种 BF16 Patch 实现，以同一台 NPU 的原生 Conv2d 为参考：

| 候选 | 最大绝对差 | RMSE | 逐元素完全一致 |
|---|---:|---:|---:|
| Linear＋独立 bias | 0.015625 | 0.00159025 | 73.96978% |
| 带 bias 的 Linear | 0.00390625 | 0.0000179506 | 99.98423% |

这些结果只说明带 bias 的 Linear 更接近 NPU Conv2d。此前 FP16 的 A100 对比中，独立 bias 路径更接近 A100，因此 BF16 也需要单独核对 A100 输出，不能直接沿用 NPU 内部比较的结论。

BF16 的指数范围与 FP32 相同，有效精度低于 FP16。在这组输入上，BF16 对 FP64 参考的误差整体更大；这不等于完整模型的任务精度结论。本轮没有执行整模型 BF16 或性能测试，也没有改变模型默认精度。

## A100 执行命令

在已有测试目录中执行：

```bash
cd /workspace/dino-patch-precision-compare
bash run_a100_bf16.sh
```

脚本先执行六项 A100 BF16 核心算子，再比较已保存的三种 NPU BF16 Patch 输出与 A100 BF16 原生 Conv2d。不安装依赖，每项算子只执行一次。

需要与新版脚本一起传递的文件：

- `compare_core_ops.py`
- `compare_patch_precision.py`
- `compare_patch_to_a100.py`
- `run_a100_bf16.sh`
- `npu_bf16_results/`
- `patch_npu_bf16_results/`

继续使用目录中已有的 `inputs.npz`；不要重新生成输入。输入 SHA256：`8e542442baa74a4db2697293f0826b5c536f25bb5a190bfd9fdb99feebb37502`。

## 结果文件

| 路径 | 内容 |
|---|---|
| `a100_bf16_results/accuracy.csv` | A100 BF16 与同精度输入的 CPU FP64 参考 |
| `a100_bf16_results/comparison.csv` | A100 BF16 与 NPU BF16 的核心算子比较 |
| `patch_a100_bf16_comparison/comparison.csv` | 三种 NPU BF16 Patch 与 A100 BF16 Conv2d 比较 |
| `npu_bf16_results/report.json` | NPU 核心结果、原始 FP32 输入参考指标及执行版本 |
| `patch_npu_bf16_results/report.json` | NPU 三种 Patch 的原始报告 |

BF16 设备输出转为 FP32 后保存为 NumPy 数组，转换可以精确保留 BF16 数值，避免 NumPy 不支持原生 BF16 的限制。报告同时记录实际设备输出 dtype。

## NPU 复现

实验目录：`/home/hc/diagnostics/dino_core_ops_bf16_compare_20261110`。

```bash
source /home/hc/envs/gns83/env-npu.sh
cd /home/hc/diagnostics/dino_core_ops_bf16_compare_20261110
python3 -B -u compare_core_ops.py \
  --device npu:7 --dtypes bf16 --out npu_bf16_results_new
python3 -B -u compare_patch_precision.py \
  --device npu:7 --dtypes bf16 --out patch_npu_bf16_results_new
```

旧结果保持原样。CUDA 的 FP16 / BF16 reduced-precision reduction 开关保留环境设置，并记录在新报告中。
