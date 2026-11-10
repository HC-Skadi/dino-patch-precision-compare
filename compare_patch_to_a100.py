"""Compare three saved NPU patch paths with the existing A100 Conv2d output."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_patch_precision import VARIANTS, detailed_metrics
from compare_core_ops import sha256, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-results", type=Path, default=Path("patch_npu_results"))
    parser.add_argument("--a100-results", type=Path, default=Path("a100_results"))
    parser.add_argument("--out", type=Path, default=Path("patch_a100_comparison"))
    args = parser.parse_args()
    npu = json.loads((args.npu_results / "report.json").read_text())
    cuda = json.loads((args.a100_results / "report.json").read_text())
    if npu["inputs_sha256"] != cuda["inputs_sha256"]:
        raise ValueError("The saved NPU and A100 inputs are different")
    if not npu["device"].startswith("npu") or not cuda["device"].startswith("cuda"):
        raise ValueError("Use saved NPU results and CUDA results")
    if cuda["patch_implementation"] != "conv2d":
        raise ValueError("The A100 baseline must be the original Conv2d path")
    for folder, report in ((args.npu_results, npu), (args.a100_results, cuda)):
        if sha256(folder / "outputs.npz") != report["outputs_sha256"]:
            raise ValueError(f"Output checksum mismatch: {folder}")
    rows = []
    print("Saved NPU outputs vs saved A100 Conv2d; no new device inference")
    print(f"{'dtype':<6} {'NPU variant':<23} {'max_abs':>12} {'RMSE':>12} {'equal(%)':>12}")
    with np.load(args.npu_results / "outputs.npz", allow_pickle=False) as candidate, \
            np.load(args.a100_results / "outputs.npz", allow_pickle=False) as baseline:
        for dtype in ("fp32", "fp16"):
            for variant in VARIANTS:
                row = dict(dtype=dtype, npu_variant=variant, a100_reference="conv2d",
                           **detailed_metrics(candidate[f"{dtype}__{variant}"],
                                              baseline[f"{dtype}__patch_embed"]))
                rows.append(row)
                print(f"{dtype:<6} {variant:<23} {row['max_abs']:12.5e} "
                      f"{row['rmse']:12.5e} {row['elementwise_equal_pct']:12.5f}")
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(npu_report=str(args.npu_results / "report.json"),
                  a100_report=str(args.a100_results / "report.json"),
                  inputs_sha256=npu["inputs_sha256"], rows=rows,
                  script_sha256=sha256(__file__),
                  compared_outputs_sha256=dict(npu=npu["outputs_sha256"], cuda=cuda["outputs_sha256"]))
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.out / "comparison.csv", rows)
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
