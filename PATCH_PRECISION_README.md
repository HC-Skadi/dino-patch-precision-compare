# Patch embedding 精度比较

本轮只测试 Patch embedding，复用此前核心算子测试的同一份输入与权重：图像 `[1, 3, 512, 512]`，权重 `[384, 3, 16, 16]`，patch=16。输入为固定合成数据，没有修改模型或权重文件。

## 三种实现

1. `conv2d`：原生 `F.conv2d(image, weight, bias, stride=16)`。
2. `linear_separate_bias`：图像块重排，执行无 bias 的 Linear，再单独加 bias；对应当前交付代码。
3. `linear_with_bias`：图像块重排后，将 bias 一并交给 Linear。该名称描述调用方式，没有宣称全部步骤已经融合成一个硬件 kernel。

每种实现分别运行 FP32、FP16，关闭 TF32/HF32。每种精度、每种实现只执行一次，当前未测量性能或完整模型输出。

## 已完成的 NPU 比较

node02，`npu:7`，Ascend 910B3，PyTorch 2.9.0、torch_npu 2.9.0。

| 精度 | 候选实现与 NPU 原生 Conv2d 比较 | 最大绝对差 | RMSE | 逐元素完全一致 |
| --- | --- | ---: | ---: | ---: |
| FP32 | Linear＋独立 bias | 2.14577e-6 | 2.11864e-7 | 7.51% |
| FP32 | 带 bias 的 Linear | 2.14577e-6 | 2.11617e-7 | 7.48% |
| FP16 | Linear＋独立 bias | 1.953125e-3 | 1.96377e-4 | 74.07% |
| FP16 | 带 bias 的 Linear | 9.765625e-4 | 7.22410e-6 | 99.84% |

FP32 的差异很小，即使逐元素一致率较低，也不能据此判断精度很差。FP16 下，将 bias 交给 Linear 后，结果明显更接近 NPU 原生 Conv2d。

相对同精度舍入输入的 CPU FP64 Conv2d 参考：

| FP16 实现 | 最大绝对差 | RMSE | 相对 L2 差 |
| --- | ---: | ---: | ---: |
| NPU 原生 Conv2d | 9.67955e-4 | 1.15477e-4 | 0.020772% |
| Linear＋独立 bias | 1.88182e-3 | 1.61762e-4 | 0.029098% |
| 带 bias 的 Linear | 9.67955e-4 | 1.15477e-4 | 0.020772% |

这些结果表明 bias 的计算顺序会影响低精度舍入。与 NPU 原生 Conv2d 更接近、与 CPU FP64 参考更接近，以及与 A100 的实际输出更接近，是三种不同的比较。本轮没有据此更改交付包的默认实现。

当前 Linear＋独立 bias 的 FP32、FP16 输出，均与此前 NPU 核心算子测试中的 `patch_embed` 输出逐元素完全相同。

## 直接与已有 A100 输出比较

你之前运行 `compare_core_ops.py` 已经把 A100 原生 Conv2d 输出保存到了 `a100_results/outputs.npz`。无需再次进行 A100 推理。

把以下新增文件复制到已有的 A100 测试目录，保留原来的 `compare_core_ops.py` 和 `a100_results/`：

- `compare_patch_precision.py`
- `compare_patch_to_a100.py`
- 整个 `patch_npu_results/`

在 A100 测试目录中执行：

```bash
python3 compare_patch_to_a100.py
```

它会直接读取已保存的 NPU 三种实现，与已保存的 A100 原生 Conv2d 比较。结果写入 `patch_a100_comparison/comparison.csv` 和 `comparison.json`。

如果目录名称不同，可以指定：

```bash
python3 compare_patch_to_a100.py \
  --npu-results patch_npu_results \
  --a100-results a100_results \
  --out patch_a100_comparison
```

## 重新执行算子测试

NPU：

```bash
source /home/hc/envs/gns83/env-npu.sh
cd /home/hc/diagnostics/dino_core_ops_compare_20261109
python3 compare_patch_precision.py --device npu:7 --out patch_npu_results_new
```

A100，如果需要另外测试三种 CUDA 实现：

```bash
python3 compare_patch_precision.py \
  --device cuda:0 \
  --out patch_a100_results \
  --compare patch_npu_results
```

后一条命令中的跨设备比较采用同名实现对同名实现；如果目的是对齐 A100 原生 Conv2d，请使用前面的 `compare_patch_to_a100.py`。

## 记录

- `patch_npu_results/report.json`：原始结果、环境、输入与输出摘要，以及最大差对应的位置和数值。
- `same_device_comparison.csv`：三种 NPU 路径之间的比较。
- `reference_accuracy.csv`：三种路径与 CPU FP64 参考的比较。
- `outputs.npz`：两种精度、三种实现的原始输出。
- `VERIFICATION.json`：文件摘要检查，以及与此前 NPU Patch 输出的一致性检查。
- `PATCH_NPU_RUN.log`：本轮执行日志。

本轮 NPU 测试已经结束，没有进行全量模型实验。合成算子测试不直接代表真实数据的任务精度。
