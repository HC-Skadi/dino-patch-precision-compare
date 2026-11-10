# SDPA 精度控制实验

日期：2026-11-10。目标是判断提高 SDPA 计算精度能否降低 A100 / NPU 差异，再决定是否接入完整模型。本轮不修改正式模型、原始权重或默认精度。

## 已知现象

同一份合成 Q/K/V，原生 SDPA 的跨设备相对 L2 差为 FP16 0.029874%、BF16 0.238487%，约相差 8 倍。这个指标描述 attention 输出张量差异，无法直接换算为整模型能量误差。

## 第一阶段：只测试 SDPA

两端继续使用已有 `inputs.npz` 的 Q/K/V，形状均为 `[1, 6, 1029, 64]`。FP16、BF16 分别测试，保留同精度舍入后的输入；无 mask、无 dropout、非因果，关闭 TF32 / HF32 和 autocast。

| 组别 | Q/K/V 进入 SDPA 的精度 | SDPA 输出 | 用途 |
|---|---|---|---|
| A：native | FP16 或 BF16 | 与输入相同 | 复现现有差异 |
| B：fp32_compute | 将 A 的同一份输入转换为 FP32 | 保留 FP32 | 隔离低精度 attention 运算的影响 |
| C：fp32_cast_back | 与 B 相同 | 将 B 的输出转换回原精度 | 判断可接入模型的修改是否仍然有效 |

B 不能直接使用原始 FP32 Q/K/V，否则会同时改变输入舍入和 attention 计算两个变量。B、C 的精度报告使用同一次 FP32 SDPA 结果，确保二者仅有输出转换的区别。

实际适配的主要判据是“NPU C 对未修改的 CUDA A”。B 对 B、C 对 C 用于诊断，不能只因为两端一起提高精度后接近，就判定 NPU 已经对齐原 CUDA 实现。

### 三个待验证的解释

1. 低精度 attention 运算或后端计算顺序主导差异：B 的跨设备误差应明显低于 A。
2. 输出转换放大舍入边界差异：B 很接近，但 C 的差异明显上升。
3. 输入舍入限制计算精度：B 相对“同精度输入的 FP64 参考”很接近，但舍入后的 FP64 参考仍偏离原始 FP32 输入的 FP64 参考。

这三个解释分别对应 B、C 和 `input_rounding.csv`，不增加其他算子或修改模型。

### 记录指标

- A100 / NPU：最大绝对差、RMSE、相对 L2 差，分别记录 A、B、C。
- 原 CUDA 对齐：将 NPU A、NPU C 分别与同精度的 CUDA A 比较，这是候选修改的主要判据。
- 各设备对 CPU FP64：使用同一份已经舍入的 Q/K/V，避免将输入误差混入运算误差。
- 输入舍入：比较舍入输入与原始 FP32 输入各自的 CPU FP64 attention 输出。
- 性能：仅计时 A、C，每组预热 3 次、同步测量 10 次；C 包含 Q/K/V 转换、SDPA 和输出转换的全部开销。

所有待测 SDPA 都在指定设备执行。CPU FP64 只用于离线参考，不参与模型推理路径。

### NPU 命令

将脚本和 `compare_core_ops.py` 放到 `/home/hc` 下的实验目录，继续使用原输入。以下输出目录必须是新目录：

```bash
source /home/hc/envs/gns83/env-npu.sh
cd /home/hc/diagnostics/sdpa_precision_experiment_20261110
python3 -B -u compare_sdpa_precision.py \
  --device npu:7 --dtypes fp16 bf16 \
  --inputs /home/hc/diagnostics/dino_core_ops_compare_20261109/inputs.npz \
  --out /home/hc/diagnostics/sdpa_precision_experiment_20261110/sdpa_npu_results \
  --timing --warmup 3 --repeat 10
```

### A100 命令

将同版本脚本和完整 `sdpa_npu_results/` 放到已有 A100 测试目录，复用已有的 `inputs.npz`：

```bash
cd /workspace/dino-patch-precision-compare
git pull --ff-only
python3 -u compare_sdpa_precision.py \
  --device cuda:0 --dtypes fp16 bf16 \
  --out sdpa_a100_results \
  --compare sdpa_npu_results \
  --timing --warmup 3 --repeat 10
```

### 如何决定下一步

- B 明显改善、NPU C 对原 CUDA A 也改善：进入完整模型实验，候选为“低精度主干＋FP32 SDPA＋转换回原精度”。
- B 改善、C 仍接近 A：输出转换是值得继续检查的环节，暂不宣称模型精度已经改善。
- B 没有明显改善：停止这个候选，下一次再比较两端显式 FP32 attention 算法，不同时修改 Patch 或 GELU。

“明显改善”先按相对 L2 差降低至少 4 倍进行候选筛选，最大差和 RMSE 一起检查。这个倍数是实验筛选标准，不是任务验收阈值。

## 第二阶段：完整模型单批验证

第一阶段通过后，再创建包外实验副本，只替换 NPU SDPA 调用，CUDA 原生模型保持不变。初轮从已有合成事件固定选取 16 个，比较原 FP32 基准、原全量 FP16 / BF16、仅 NPU SDPA 提升 FP32 三组。两套 Pass1 / Pass2 权重、Patch、GELU、输出头和预处理全部保持不变。合成事件用于数值定位，真实任务精度仍需真实数据。

优先冻结每个阶段的图像和条件输入。Pass2 使用同一份筛选事件及条件，先隔离本阶段计算差异；确认后才使用各组自己的 Pass1 预测，恢复完整两阶段链路测试。

记录第十二层归一化 CLS、最终分类、早退分类、物理能量和方向，并执行相同 batch16 的短同步前向计时。能量头不读取 FiLM 条件，Pass2 能量差应沿图像到 CLS 的路径分析。

只有候选的任务输出得到改善，再评估额外耗时是否值得。当前没有既定的业务验收阈值，不将算子级改善直接标记为模型精度通过。

## 第一阶段 NPU 实测

2026-11-10 已在 node02 的 Ascend 910B3、npu:7 完成短测试。使用原固定输入，PyTorch / torch_npu 2.9.0，预热 3 次、同步测量 10 次。

以下是 NPU 输出对“同精度舍入输入的 CPU FP64 参考”的相对 L2 差，单位为百分比，尚未涉及 A100：

| 输入精度 | A：原生 SDPA | B：保留 FP32 输出 | C：转换回原精度 |
|---|---:|---:|---:|
| FP16 | 0.0277682% | 0.0000261625% | 0.0206525% |
| BF16 | 0.222254% | 0.0000239524% | 0.165267% |

| 输入精度 | A 平均耗时 | C 平均耗时 | SDPA 耗时变化 |
|---|---:|---:|---:|
| FP16 | 0.2381 ms | 0.2993 ms | 约 +26% |
| BF16 | 0.2397 ms | 0.3078 ms | 约 +28% |

保留 FP32 输出时运算误差明显降低，转换回低精度后仍有输出舍入误差。C 对 FP64 参考的相对 L2 差比 A 降低约 25.6%，但这不能直接证明已经更接近 CUDA 原路径。耗时变化只描述本次 SDPA 算子，不代表完整模型增加同样比例。

原始数组和报告位于 `sdpa_npu_results/`，执行日志为 `SDPA_NPU_RUN.log`。A100 命令可以直接读取这份结果；两端完成后，重点查看 `alignment_to_cuda_native.csv`。

`sdpa_npu_results/VERIFICATION.json` 记录输入、脚本和输出摘要核验。FP16 / BF16 的 A 输出分别与此前核心算子实验逐元素相同，C 与 B 转换回目标精度的结果逐元素相同，确认原基准复现正确且 B、C 仅有输出转换的区别。

## 文件与状态

- `compare_sdpa_precision.py`：三组 SDPA 控制实验与短计时。
- `test_sdpa_precision.py`：小规模 CPU 测试，核对输入输出 dtype、autocast、FP64 参考和报告生成。
- `report.json`：版本、输入与输出 SHA256、实际 dtype 和各项指标。
- `comparison.csv`：同组跨设备差异，仅在 A100 指定 NPU 结果后生成。
- `alignment_to_cuda_native.csv`：NPU 原路径和候选路径分别对照未修改的 CUDA 原路径，这是实际适配的主要结果。
- `accuracy.csv`、`input_rounding.csv`、`timing.csv`：运算精度、输入舍入与计时。

六项小规模 CPU 测试全部通过，日志见 `SDPA_CPU_TEST.log`。第一阶段 NPU 测试已完成，A100 对比等待执行；第二阶段完整模型实验尚未启动，正式模型代码和默认精度保持不变。
