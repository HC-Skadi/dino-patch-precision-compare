# DINO 算子与 Patch embedding 精度比较

用于比较 A100 与 Ascend NPU 的 FP32、FP16、BF16 数值结果。仓库包含独立测试脚本、固定合成输入和已完成的 NPU 输出，不包含原始模型权重或真实业务数据。

FP32 / FP16 的 NPU 基准于 2026-11-09 在 Ascend 910B3 上完成。BF16 使用独立入口和输出目录，保留原有结果；具体命令与测试记录见 `BF16_README.md`。

## 在 A100 上运行

进入仓库，在已有 PyTorch CUDA 和 NumPy 的环境中执行：

```bash
bash run_a100.sh
```

默认使用 `cuda:0`。如果需要指定另一张卡：

```bash
bash run_a100.sh cuda:1
```

脚本会完成两件事：

1. 测试 A100 上的核心算子，并与已保存的 NPU 输出比较。
2. 使用刚生成的 A100 原生 Conv2d 输出，对照 NPU 的三种 Patch embedding 实现。

不会安装依赖，不会运行完整模型，也不会重新生成输入。每种算子、每种精度只执行一次，不测量性能。

## 在 A100 上增加 BF16 比较

```bash
bash run_a100_bf16.sh
```

这条命令仅测试 BF16，使用同一份 `inputs.npz`，并比较已保存的 BF16 NPU 输出。结果分别保存到 `a100_bf16_results/` 和 `patch_a100_bf16_comparison/`，不会覆盖此前的 FP32 / FP16 结果。

## 查看结果

| 文件 | 内容 |
| --- | --- |
| `a100_results/comparison.csv` | A100 与 NPU 核心算子的直接比较 |
| `patch_a100_comparison/comparison.csv` | NPU 三种 Patch 实现与 A100 原生 Conv2d 的比较 |
| `a100_results/accuracy.csv` | A100 输出与 CPU FP64 参考的比较 |
| `a100_results/report.json` | A100 型号、软件版本、精度开关、输出 dtype 与文件摘要 |

比较指标包括最大绝对差、平均绝对差、RMSE 和相对 L2 差。Patch 专项额外记录逐元素完全一致比例及最大差对应的位置。

**相对 L2 差的单位是百分比，不能直接换算为完整模型的物理能量误差。**

## 测试范围

核心算子共 6 项，原入口执行 FP32、FP16，BF16 入口单独执行 BF16：

- Linear
- LayerNorm
- GELU：与当前模型相同的 erf 表达式
- GELU：原生实现
- SDPA attention
- Patch embedding

Patch 专项包含 3 种实现：

- `conv2d`：原生卷积，包含 bias。
- `linear_separate_bias`：图像块重排后执行无 bias 的 Linear，再单独加 bias；对应当前 NPU 交付实现。
- `linear_with_bias`：图像块重排后，将 bias 一并交给 Linear。

第三种名称描述调用方式，没有宣称图像重排与全部计算已经融合为单个硬件 kernel。

## 已完成的 NPU Patch 结果

下表均以同一台 NPU 上的原生 Conv2d 为基准：

| 精度 | 候选实现 | 最大绝对差 | RMSE | 逐元素完全一致 |
| --- | --- | ---: | ---: | ---: |
| FP32 | Linear＋独立 bias | 2.14577e-6 | 2.11864e-7 | 7.51% |
| FP32 | 带 bias 的 Linear | 2.14577e-6 | 2.11617e-7 | 7.48% |
| FP16 | Linear＋独立 bias | 1.953125e-3 | 1.96377e-4 | 74.07% |
| FP16 | 带 bias 的 Linear | 9.765625e-4 | 7.22410e-6 | 99.84% |

FP32 数值差很小，不能仅凭逐元素一致率判断精度。FP16 下 bias 的计算顺序会影响舍入；最终选择需要参考 A100 的实际输出，当前模型默认实现未因此修改。

## 复用已有 A100 结果

如果已经在本目录完成核心算子测试，并保留了 `a100_results/`，只需要执行：

```bash
python3 compare_patch_to_a100.py
```

这条命令仅比较已保存的数组，不再进行 GPU 推理。

## 手动执行与复现

A100 核心算子：

```bash
python3 compare_core_ops.py --device cuda:0 --out a100_results --compare npu_results
```

NPU 核心算子，在已有 torch_npu 环境中执行：

```bash
python3 compare_core_ops.py --device npu:7 --out npu_results_new
```

NPU 三种 Patch 实现：

```bash
python3 compare_patch_precision.py --device npu:7 --out patch_npu_results_new
```

FP32 测试关闭 CUDA TF32 和 NPU HF32。SDPA 使用 PyTorch 自动选择的设备实现；CUDA 的 FP16 reduced-precision reduction 开关保留环境默认值，并记录在报告中。

CPU FP64 仅用于生成数值参考，待测算子在指定设备上完整执行。输入和测试权重固定保存在 `inputs.npz` 中，两端不得分别重新生成。

## 文件说明

- `compare_core_ops.py`：独立核心算子测试与跨设备比较。
- `compare_patch_precision.py`：三种 Patch 实现的同设备比较。
- `compare_patch_to_a100.py`：保存的 NPU Patch 输出与保存的 A100 原生 Conv2d 比较。
- `run_a100_bf16.sh`、`BF16_README.md`：BF16 执行入口与测试记录。
- [最新适配方案](LATEST_ADAPTATION_PLAN.md)：第六模型当前实现、精度选择、实验结论与下一步安排；本仓库不包含完整模型工作包。
- `inputs.npz`：固定合成输入和测试权重。
- `npu_results/`、`patch_npu_results/`：原始 NPU 数组、指标、环境和检查记录。
- `npu_bf16_results/`、`patch_npu_bf16_results/`：同一份输入的 BF16 NPU 结果。
- `NPU_RUN.log`、`PATCH_NPU_RUN.log`：NPU 实验日志。
- [核心算子说明](CORE_OPS_README.md)、[Patch 专项说明](PATCH_PRECISION_README.md)：详细测试口径。
- `SHA256SUMS`：发布时的文件校验清单。

新生成的 A100 结果及复现实验结果默认被 Git 忽略，避免误提交。执行报告中的脚本摘要对应当次实验版本；新增 BF16 支持没有改写旧实验报告或数组。
