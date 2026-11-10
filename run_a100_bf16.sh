#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -u compare_core_ops.py \
  --device "${1:-cuda:0}" --dtypes bf16 \
  --out a100_bf16_results \
  --compare npu_bf16_results

python3 -u compare_patch_to_a100.py \
  --dtypes bf16 \
  --npu-results patch_npu_bf16_results \
  --a100-results a100_bf16_results \
  --out patch_a100_bf16_comparison
