# 最新适配方案：第六模型 CellAware DINO

更新时间：2026-11-10。

当前方案：完整十二层、WebDataset 两阶段推理、原始权重保持不变；NPU 复用 DINO 的 Patch 线性投影、独立 bias、显式 erf GELU、SDPA 和 dtype 对齐。正式默认保持 FP32，FP16 / BF16 单独评估。

## 1. 工作版本与范围

本地工作包：`/Users/hc/Documents/Codex/2026-10-28/11/outputs/deliver_6models_20261109`。

node02 工作包：`/home/hc/diagnostics/dino_cellaware_full12_20261109/package`。直接使用宿主机 Transformers 5.13.0、PyTorch / torch_npu 2.9.0，不启动容器，不修改共享依赖。

| 模型 | 当前默认模式 | 本轮处理 |
|---|---|---|
| dino_multitask | 全量 FP16 | 保留五模型 v4 的适配 |
| km2a_c5 | 全量 BF16 | 保留五模型 v4 的适配 |
| km2a_ene_core | FP32 | 保留兼容 KNN、自定义算子与 FA 复用 |
| km2a_dir_v5 | FP32 | 保留 packed 融合 FA |
| weaver_pid | 全量 BF16 | 保留非融合 attention |
| dino_multitask_cellaware | FP32 | 本次新增和优化的第六模型 |

前五模型没有重新测试新版宿主机环境，历史成绩与第六模型分别记录。旧三层截断实验仅供参考，不作为当前入口。

## 2. 保持的模型与数据语义

```text
配套 img.tar + meta.tar
  -> Pass1：原 Run 0 权重，完整十二层
  -> 按原 cell 阈值筛选，保存 Pass1 预测与事件身份
  -> Pass2：原 Pass2 权重，完整十二层，仅处理保留事件
  -> 最终分类、早退分类、能量、方向
```

模型内部的特征来源：

| 分支 | 输入特征 |
|---|---|
| 早退分类 | 第三层原始 CLS，经 early_norm 与 FiLM |
| 最终分类、方向 | 第十二层归一化 patch，经空间头与 FiLM |
| 能量 | 第十二层归一化 CLS，经原能量头，不读取 FiLM 条件 |

第三层分支保留，但没有根据早退分类结果跳过后九层。Pass1 预测能量参与 Pass2 条件，会影响分类和方向；Pass2 能量不通过该条件分支，能量差应从自身图像特征路径分析。

预处理沿用原 PIL / torchvision 的 Resize512、CenterCrop、ToTensor、ImageNet Normalize。元数据保持 schema v6 和训练口径的 nhit；Pass2 第三维条件来自同事件 Pass1 预测能量。

Pass2 原始 safetensors 的全部 242 个张量严格加载。适配 Transformers 5.13.0 时仅映射 `backbone.layer.*` 到 `backbone.model.layer.*`，没有改写权重文件或增加输出头。

## 3. 当前已经采用的优化

| 项目 | 当前实现与选择 |
|---|---|
| Patch embedding | 非重叠 patch 重排，NPU 无 bias Linear 后独立加 bias；CUDA 保留 Conv2d |
| GELU | 两端使用相同的显式 erf 表达式，没有使用 fast_gelu |
| Attention | SDPA，保持原语义，尚未单独提高 SDPA 计算精度 |
| dtype | Linear / LayerNorm 输入与参数类型对齐 |
| 空间头 | 保留 3×3 Conv2d、GroupNorm、SiLU 与 FiLM，不套用 Patch 改写 |
| 推理 | eval、inference_mode；性能测试采用设备常驻输入和同步完整前向 |

正式 Patch 代码：`/Users/hc/Documents/Codex/2026-10-28/11/outputs/deliver_6models_20261109/src/five_models/runtime.py`。

当前 Patch 优化包含图像块重排、矩阵乘和独立加法，尚未实现单个自定义融合 kernel。第六模型也没有接入 ene_core 的 KNN 算子。

## 4. 整模型 FP16 实验结论

同一批 64 个独立合成 WDS 事件，node02 单卡 npu:7，每阶段预热 3 次、同步测量 10 次。最新 FP16 复测的输出与上轮逐元素相同。

| 阶段 | FP32 前向 | FP16 前向 | FP32 / FP16 吞吐 |
|---|---:|---:|---:|
| Pass1 | 203.71 ms | 114.13 ms | 314.17 / 560.75 事件/秒 |
| Pass2 | 203.63 ms | 114.04 ms | 314.29 / 561.21 事件/秒 |

FP16 约有 1.78 倍的纯前向加速，PyTorch 分配峰值从 4.701 GiB 降至 2.350 GiB。该计时不包含加载、预处理、传输、筛选或输出复制，不能作为端到端吞吐。

以下比较为 NPU FP16 对 NPU FP32，尚未涉及 CUDA 整模型：

| 指标 | Pass1 | Pass2 |
|---|---:|---:|
| 最终分类标签一致 | 62/64 | 64/64 |
| 早退分类标签一致 | 64/64 | 64/64 |
| 物理能量最大相对差 | 1.611955% | 7.313360% |
| 方向最大角差 | 0.02889034 度 | 0.01364422 度 |

仅将能量头恢复为原始 FP32 权重和 FP32 运算后，Pass2 最大能量相对差仍为 7.317031%，没有改善。因此全量 FP16 和这个混合精度方案都不设为正式默认。

## 5. 最新 A100 / NPU Patch 结论

2026-11-10 用户回传的控制台结果，采用相同固定合成输入，以 A100 FP16 原生 Conv2d 为参考：

| NPU 路径 | 最大绝对差 | RMSE | 逐元素完全一致 |
|---|---:|---:|---:|
| 原生 Conv2d | 0.00195312 | 0.000196341 | 73.96774% |
| Linear＋独立 bias | 0.00195312 | 0.0000259546 | 99.16941% |
| 带 bias 的 Linear | 0.00195312 | 0.000196343 | 73.96978% |

独立 bias 的 RMSE 约低 7.56 倍，当前实现最接近这次 A100 基准，继续保留。FP32 三种实现的最大跨设备差均约为 3e-6。

核心算子中，FP16 SDPA 跨设备相对 L2 差为 0.02987%，当前 Patch 为 0.00467%。SDPA 适合进入下一项实验，但目前没有证据将完整模型能量差归因到它。

A100 原始输出目前位于用户机器的 `/workspace/dino-patch-precision-compare/a100_results`，本地汇总采用用户回传结果，尚未同步这些原始数组。

## 6. 新增 BF16 实验

BF16 核心算子和三种 Patch 路径已在 node02 完成。沿用同一份输入，没有执行完整模型 BF16 或性能测试。

相对同精度舍入输入的 CPU FP64 参考，BF16 Linear / LayerNorm 的相对 L2 差约 0.166%，SDPA 约 0.222%，当前 Patch 约 0.235%。这些属于参考精度指标，不能与跨设备指标直接混用。

用户已回传 A100 / NPU 的 BF16 核心算子结果：SDPA 跨设备相对 L2 差为 0.238487%，约为 FP16 0.029874% 的 8 倍；显式 erf GELU 逐元素相同，当前 Patch 的相对 L2 差为 0.00853799%。下一项实验聚焦 SDPA，不同时修改其他算子。

BF16 带 bias Linear 与 NPU Conv2d 的逐元素一致率为 99.98423%，独立 bias 为 73.96978%。最终选择仍需要 A100 BF16 对比，当前正式 Patch 实现保持不变。

执行脚本与 NPU 数组：`/Users/hc/Documents/Codex/2026-10-28/11/outputs/dino-patch-precision-compare`。A100 使用 `bash run_a100_bf16.sh`，详细说明见同目录的 `BF16_README.md`。

## 7. SDPA 控制实验与下一步

三组独立实验已经实现：A 使用原生低精度 SDPA；B 将同一份已经舍入的 Q/K/V 转为 FP32，保留 FP32 输出；C 将 B 的同一次输出转换回原精度。固定输入形状为 `[1, 6, 1029, 64]`，分别测试 FP16、BF16。

NPU 短测试已完成。相对同精度输入的 CPU FP64 参考，C 的相对 L2 差比 A 降低约 25.6%；FP16 为 0.0277682% → 0.0206525%，BF16 为 0.222254% → 0.165267%。C 的 SDPA 平均耗时分别增加约 26%、28%，不代表整模型的耗时变化。B 保留 FP32 输出时，相对 L2 差分别为 0.0000261625%、0.0000239524%。

这份结果尚不能判断候选更接近 A100。实际适配以“NPU C 对未修改的 CUDA A”为主要判据，避免同时修改两端后偏离原 CUDA 基准。详细命令与记录见 `SDPA_EXPERIMENT.md`。

1. 在 A100 执行三组 SDPA 实验，对照仓库的 `sdpa_npu_results/`，重点查看 `alignment_to_cuda_native.csv`。
2. 只有 NPU C 对原 CUDA A 明显改善，才进行完整模型单批验证。固定 16 个合成事件，保持 CUDA 原模型、Patch、GELU 和输出头不变，仅替换 NPU SDPA，比较 CLS、分类、能量、方向和纯前向耗时。Pass2 先冻结同一事件和条件输入，再恢复完整两阶段链路。
3. 如果能量差仍然明显，比较同事件的逐层骨干输出，定位差异增大的层，再决定局部 FP32 范围。
4. 提供真实 WDS 后，先完成 FP32 的 A100 / NPU 整模型一致性验证，再决定 FP16 / BF16 是否满足任务要求。首轮使用一个小批次，不自动开展全量实验。

各实验分别保留同一输入、权重和模式的身份记录。整模型降精度比较、同精度跨设备比较和 FP64 参考比较分别汇报，不使用算子相对 L2 差代替物理能量差。

## 8. 记录位置

- 原始三份附件：`/Users/hc/Documents/Codex/2026-10-28/11/outputs/dino_cellaware_reference_20261109/originals`。
- FP32 整模型基准：`/Users/hc/Documents/Codex/2026-10-28/11/outputs/dino_cellaware_analysis_20261109/batch64_fp32_20261109_161233/SUMMARY.json`。
- FP16 复测：`/Users/hc/Documents/Codex/2026-10-28/11/outputs/dino_cellaware_analysis_20261109/batch64_fp16_repeat_20261109_162446/README.md`。
- FP32 能量头实验：`/Users/hc/Documents/Codex/2026-10-28/11/outputs/dino_cellaware_analysis_20261109/batch64_mixed_energy_fp32_20261109_170512/README.md`。
- BF16 记录：`/Users/hc/Documents/Codex/2026-10-28/11/outputs/dino-patch-precision-compare/BF16_README.md`。
- SDPA 控制实验：`/Users/hc/Documents/Codex/2026-10-28/11/outputs/dino-patch-precision-compare/SDPA_EXPERIMENT.md`，原始 NPU 结果位于同目录的 `sdpa_npu_results/`。

本次完成独立 BF16 算子实验和第一阶段 NPU SDPA 控制实验，完整模型 SDPA 实验尚未启动。正式模型代码、原始权重、输入数据和默认精度保持不变，没有生成新的交付压缩包。
