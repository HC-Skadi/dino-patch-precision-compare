# DINO 核心算子：A100 与 NPU 精度比较

新增 Patch embedding 专项精度比较，包含原生 Conv2d、Linear＋独立 bias 和带 bias 的 Linear。NPU 测试已完成；详见同目录的 `PATCH_PRECISION_README.md`，可以直接复用已有 A100 输出进行比较。

一个独立脚本，测试 Linear、LayerNorm、GELU、SDPA attention 和 patch embedding。GELU 分为当前模型使用的 erf 实现与原生实现，共 6 个测试项，每项分别运行 FP32、FP16。

NPU 测试已经完成。把本目录复制到 A100 后，运行一次命令即可获得两台设备的输出差异。脚本只依赖 PyTorch、NumPy；NPU 端额外使用 torch_npu，无需模型权重或 Transformers。

## 在 A100 上运行

需要一起复制：`compare_core_ops.py`、`inputs.npz` 和整个 `npu_results/`。进入复制后的目录，在已经安装 PyTorch CUDA 的环境中执行：

```bash
python3 -u compare_core_ops.py \
  --device cuda:0 \
  --out a100_results \
  --compare npu_results
```

脚本会先运行 A100 测试，再直接比较 A100 与已保存的 NPU 输出。默认自动读取脚本旁边的 `inputs.npz`，不要重新生成这份输入。

主要查看：

- `a100_results/comparison.csv`：A100/NPU 直接比较，包含最大绝对差、平均绝对差、RMSE、相对 L2 差。
- `a100_results/comparison.json`：同样的比较结果，以及两端报告位置、输入和脚本摘要。
- `a100_results/accuracy.csv`：A100 与 CPU FP64 数值参考的比较。
- `a100_results/report.json`：设备名称、软件版本、精度开关、输出形状和 dtype。

## 已完成的 NPU 测试

设备为 node02 宿主机 `npu:7`，Ascend 910B3，PyTorch 2.9.0、torch_npu 2.9.0。没有启动容器、安装依赖或修改模型权重。每个测试项每种精度只执行一次，没有进行性能测试或批量扫描。

下表比较的是 NPU 输出与 CPU FP64 参考，参考输入已经按相应的 FP32/FP16 精度舍入。它体现算子运算差异，不包含额外的 FP32 到 FP16 输入转换误差。

| 测试项 | FP32 最大绝对差 | FP16 最大绝对差 |
| --- | ---: | ---: |
| Linear | 8.09592e-7 | 4.88388e-4 |
| LayerNorm | 7.95245e-7 | 1.80828e-3 |
| GELU：erf 实现 | 1.05576e-6 | 1.92242e-3 |
| GELU：原生实现 | 4.73517e-4 | 1.43779e-3 |
| SDPA attention | 1.36038e-7 | 1.15243e-4 |
| Patch embedding | 1.74566e-6 | 1.88182e-3 |

12 项输出全部为有限值。FP32 原生 GELU 与精确 GELU 参考的差异明显高于 erf 实现；当前模型使用的是 erf 实现，原生实现作为定位用的对照项。

这些数值尚未涉及 A100/NPU 直接比较。FP16 某个算子的最大绝对差也不能直接换算为完整模型的物理能量误差。

## 测试内容

输入是固定合成数据，维度参考第六模型的配置，batch=1，不读取模型权重或真实中间特征：

| 测试项 | 实现与主要尺寸 |
| --- | --- |
| Linear | `F.linear`，输入 `[1, 1029, 384]`，权重 `[1152, 384]`，包含 bias |
| LayerNorm | 最后一维 384，`eps=1e-5`，包含缩放和偏置 |
| GELU：erf | 与当前适配代码相同：`x * 0.5 * (1 + erf(x / sqrt(2)))`，输入 `[1, 1029, 1536]` |
| GELU：原生 | `F.gelu(x, approximate="none")`，使用与 erf 实现完全相同的输入 |
| SDPA | Q/K/V 均为 `[1, 6, 1029, 64]`，无 mask、无 dropout、非因果；由 PyTorch 自动选择设备实现 |
| Patch embedding | 图像 `[1, 3, 512, 512]`，patch=16，输出通道 384；CUDA 使用 Conv2d，NPU 使用当前适配代码的 Linear 加独立 bias |

Patch embedding 比较的是当前两端的实际实现路径，包含它们的运算和舍入边界差异。其他测试项使用相同的 PyTorch 调用或相同的 erf 表达式。

FP32 测试明确关闭 CUDA TF32 和 NPU HF32。CUDA FP16 的 reduced-precision reduction 开关按环境默认值运行，并记录到报告中。SDPA 采用自动后端选择，没有强制切换为数学实现。

## 如何理解指标

- `comparison.csv`：直接比较两端同一 dtype、同一输入的输出。这里的 `relative_l2_pct` 是整体张量的相对 L2 差，单位为百分比。
- `accuracy.csv` 的 `max_abs`、`rmse`、`relative_l2_pct`：与同精度舍入输入的 CPU FP64 参考比较，侧重算子本身的运算差异。
- `accuracy.csv` 的 `max_abs_vs_original_fp64`、`relative_l2_pct_vs_original_fp64`：与原始 FP32 输入对应的 CPU FP64 参考比较，FP16 结果同时包含输入、权重转换与运算的影响。

CPU FP64 仅用于生成数值参考；待测算子仍完整地在指定的 CUDA/NPU 设备上执行。各算子独立测试，没有串联成完整模型。

## 在 NPU 上复现

```bash
source /home/hc/envs/gns83/env-npu.sh
cd /home/hc/diagnostics/dino_core_ops_compare_20261109
python3 -u compare_core_ops.py --device npu:7 --out npu_results_new
```

默认运行全部 6 项和 FP32、FP16。`--cases`、`--dtypes` 可以缩小范围；进行直接比较时两端需要使用相同选项。

远程目录：`/home/hc/diagnostics/dino_core_ops_compare_20261109`。

`inputs.npz`、脚本和 NPU 输出的 SHA256 已核对一致；脚本也已通过小尺寸 CPU 检查，覆盖全部 12 项、相同输出比较、不同输入拒绝比较及指标公式检查。
