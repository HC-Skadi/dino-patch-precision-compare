"""Compare fixed DINO operator inputs on CUDA and Ascend, without model weights."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch
import torch.nn.functional as F

CASES = ("linear", "layer_norm", "gelu_erf", "gelu_native", "sdpa", "patch_embed")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
DEFAULT_DTYPES = ("fp32", "fp16")
SPEC = dict(batch=1, tokens=1029, hidden=384, intermediate=1536, heads=6,
            head_dim=64, image=512, patch=16, layer_norm_eps=1e-5, seed=20261109)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_inputs(path):
    rng = np.random.default_rng(SPEC["seed"])

    def normal(shape, scale=1.0):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    b, t, d = SPEC["batch"], SPEC["tokens"], SPEC["hidden"]
    data = dict(
        x=normal((b, t, d)),
        linear_weight=normal((3 * d, d), 0.02), linear_bias=normal((3 * d,), 0.02),
        ln_weight=1.0 + normal((d,), 0.05), ln_bias=normal((d,), 0.02),
        gelu_x=normal((b, t, SPEC["intermediate"]), 1.8),
        q=normal((b, SPEC["heads"], t, SPEC["head_dim"])),
        k=normal((b, SPEC["heads"], t, SPEC["head_dim"])),
        v=normal((b, SPEC["heads"], t, SPEC["head_dim"])),
        image=normal((b, 3, SPEC["image"], SPEC["image"])),
        patch_weight=normal((d, 3, SPEC["patch"], SPEC["patch"]), 0.02),
        patch_bias=normal((d,), 0.02),
    )
    data["gelu_x"].reshape(-1)[:25] = np.linspace(-6, 6, 25, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def operation(name, x, reference=False):
    if name == "linear":
        return F.linear(x["x"], x["linear_weight"], x["linear_bias"])
    if name == "layer_norm":
        return F.layer_norm(x["x"], (SPEC["hidden"],), x["ln_weight"], x["ln_bias"],
                            SPEC["layer_norm_eps"])
    if name == "gelu_erf" or (name == "gelu_native" and reference):
        value = x["gelu_x"]
        return value * 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))
    if name == "gelu_native":
        return F.gelu(x["gelu_x"], approximate="none")
    if name == "sdpa":
        if reference:
            logits = (x["q"] @ x["k"].transpose(-2, -1)) / math.sqrt(SPEC["head_dim"])
            return torch.softmax(logits, dim=-1) @ x["v"]
        return F.scaled_dot_product_attention(x["q"], x["k"], x["v"],
                                             dropout_p=0.0, is_causal=False)
    if name == "patch_embed":
        image, weight, bias = x["image"], x["patch_weight"], x["patch_bias"]
        if image.device.type != "npu":
            return F.conv2d(image, weight, bias, stride=SPEC["patch"])
        # Match the current DINO NPU adapter's separate bias-rounding boundary.
        b, c, h, w = image.shape
        p = SPEC["patch"]
        oh, ow = h // p, w // p
        patches = image.reshape(b, c, oh, p, ow, p).permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(b, oh * ow, -1).contiguous()
        out = F.linear(patches, weight.reshape(SPEC["hidden"], -1), None) + bias
        return out.permute(0, 2, 1).reshape(b, SPEC["hidden"], oh, ow)
    raise ValueError(name)


def metrics(candidate, reference):
    a, b = candidate.astype(np.float64), reference.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Output shape mismatch: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite operator output")
    diff = a - b
    return dict(max_abs=float(np.abs(diff).max()), mean_abs=float(np.abs(diff).mean()),
                rmse=float(np.sqrt(np.mean(diff * diff))),
                relative_l2_pct=float(np.linalg.norm(diff.reshape(-1)) /
                                      max(np.linalg.norm(b.reshape(-1)), 1e-30) * 100.0))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_saved(current, other, output_dir):
    prior = json.loads((other / "report.json").read_text())
    if current["inputs_sha256"] != prior["inputs_sha256"] or current["spec"] != prior["spec"]:
        raise ValueError("Copy the same inputs.npz and script to both machines before comparing")
    if sha256(other / "outputs.npz") != prior["outputs_sha256"]:
        raise ValueError("The comparison outputs do not match their saved report")
    rows = []
    with np.load(output_dir / "outputs.npz", allow_pickle=False) as candidate, \
            np.load(other / "outputs.npz", allow_pickle=False) as reference:
        if set(candidate.files) != set(reference.files):
            raise ValueError("Run the same --cases and --dtypes on both machines")
        print(f"\nCross-device: {current['device']} vs {prior['device']}")
        print(f"{'dtype':<6} {'operator':<13} {'max_abs':>12} {'RMSE':>12} {'rel_L2(%)':>12}")
        for key in candidate.files:
            dtype, name = key.split("__", 1)
            row = dict(dtype=dtype, operator=name, **metrics(candidate[key], reference[key]))
            rows.append(row)
            print(f"{dtype:<6} {name:<13} {row['max_abs']:12.5e} {row['rmse']:12.5e} "
                  f"{row['relative_l2_pct']:12.5e}")
    result = dict(candidate_device=current["device"], reference_device=prior["device"],
                  candidate_report=str(output_dir / "report.json"),
                  reference_report=str(other / "report.json"),
                  inputs_sha256=current["inputs_sha256"],
                  candidate_script_sha256=current["script_sha256"],
                  reference_script_sha256=prior["script_sha256"], rows=rows)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(output_dir / "comparison.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="npu:7, cuda:0 or cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().with_name("inputs.npz"))
    parser.add_argument("--compare", type=Path, help="Other machine's result directory")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DEFAULT_DTYPES))
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    if device.type not in ("npu", "cuda", "cpu"):
        parser.error("Use npu, cuda or cpu")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    flags = {}
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        flags = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                     fp16_reduced_precision_reduction=
                     torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                     bf16_reduced_precision_reduction=
                     torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "npu":
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        flags = dict(matmul_allow_hf32=torch.npu.matmul.allow_hf32,
                     conv_allow_hf32=torch.npu.conv.allow_hf32)
        device_name = torch.npu.get_device_name(device)
    else:
        device_name = "CPU smoke test"
    if not args.inputs.exists():
        if args.compare:
            parser.error("Copy inputs.npz from the NPU test before running --compare")
        create_inputs(args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.inputs, allow_pickle=False) as stored:
        cpu = {key: torch.from_numpy(stored[key].copy()) for key in stored.files}
    original = {key: value.double() for key, value in cpu.items()}
    outputs, rows, original_refs, metadata = {}, [], {}, {}
    print(f"Device: {device} ({device_name}); FP32 TF32/HF32 disabled")
    print(f"{'dtype':<6} {'operator':<13} {'max_abs':>12} {'RMSE':>12} {'rel_L2(%)':>12}")
    with torch.inference_mode():
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]
            rounded = {key: value.to(dtype) for key, value in cpu.items()}
            same_input_fp64 = {key: value.double() for key, value in rounded.items()}
            inputs = {key: value.to(device) for key, value in rounded.items()}
            for name in args.cases:
                if name not in original_refs:
                    original_refs[name] = operation(name, original, reference=True).numpy()
                reference = (original_refs[name] if dtype_name == "fp32" else
                             operation(name, same_input_fp64, reference=True).numpy())
                out = operation(name, inputs)
                if device.type != "cpu":
                    getattr(torch, device.type).synchronize(device)
                key = f"{dtype_name}__{name}"
                outputs[key] = out.float().cpu().numpy()
                metadata[key] = dict(shape=list(out.shape), output_dtype=str(out.dtype))
                same_metrics = metrics(outputs[key], reference)
                original_metrics = metrics(outputs[key], original_refs[name])
                row = dict(dtype=dtype_name, operator=name, **same_metrics,
                           max_abs_vs_original_fp64=original_metrics["max_abs"],
                           relative_l2_pct_vs_original_fp64=original_metrics["relative_l2_pct"])
                rows.append(row)
                print(f"{dtype_name:<6} {name:<13} {row['max_abs']:12.5e} {row['rmse']:12.5e} "
                      f"{row['relative_l2_pct']:12.5e}", flush=True)
                del out
            del inputs
    np.savez_compressed(args.out / "outputs.npz", **outputs)
    report = dict(device=str(device), device_name=device_name, spec=SPEC,
                  torch=torch.__version__, numpy=np.__version__, cuda_version=torch.version.cuda,
                  torch_npu=torch_npu.__version__ if device.type == "npu" else None,
                  precision_flags=flags, cases=args.cases, dtypes=args.dtypes,
                  sdpa_selection="automatic PyTorch backend; no mask, no dropout, noncausal",
                  patch_implementation="linear + separate bias" if device.type == "npu" else "conv2d",
                  reference="CPU FP64 applied to the same dtype-rounded inputs; original FP32 inputs also reported",
                  input_kind="fixed synthetic operator fixtures, not model activations",
                  inputs_sha256=sha256(args.inputs), outputs_sha256=sha256(args.out / "outputs.npz"),
                  script_sha256=sha256(__file__), output_metadata=metadata, rows=rows)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.out / "accuracy.csv", rows)
    if args.compare:
        compare_saved(report, args.compare, args.out)
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
