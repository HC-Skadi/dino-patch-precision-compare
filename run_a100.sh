#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -u compare_core_ops.py \
  --device "${1:-cuda:0}" \
  --out a100_results \
  --compare npu_results

python3 -u compare_patch_to_a100.py \
  --npu-results patch_npu_results \
  --a100-results a100_results \
  --out patch_a100_comparison
