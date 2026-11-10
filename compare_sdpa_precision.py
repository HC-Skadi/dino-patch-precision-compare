"""Isolate SDPA computation from low-precision input and output rounding."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch
import torch.nn.functional as F

from compare_core_ops import DTYPES, compare_saved, metrics, sha256, write_csv


VARIANTS = ("native", "fp32_compute", "fp32_cast_back")


def reference_sdpa(q, k, v):
    scale = q.shape[-1] ** -0.5
    return torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1) @ v


def sdpa_forward(q, k, v, variant):
    with torch.autocast(device_type=q.device.type, enabled=False):
        if variant == "native":
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        if variant not in VARIANTS:
            raise ValueError(variant)
        output = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), dropout_p=0.0, is_causal=False)
    return output.to(q.dtype) if variant == "fp32_cast_back" else output


def synchronize(device):
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def measure(qkv, variant, warmup, repeat):
    device = qkv[0].device
    for _ in range(warmup):
        output = sdpa_forward(*qkv, variant)
        del output
    synchronize(device)
    durations = []
    for _ in range(repeat):
        synchronize(device)
        start = time.perf_counter()
        output = sdpa_forward(*qkv, variant)
        synchronize(device)
        durations.append((time.perf_counter() - start) * 1000)
        del output
    return dict(mean_ms=float(np.mean(durations)), median_ms=float(np.median(durations)),
                min_ms=min(durations), warmup=warmup, repeat=repeat)


def compare_cuda_native(current, other, output_dir):
    prior = json.loads((other / "report.json").read_text())
    devices = {current["device"].split(":")[0], prior["device"].split(":")[0]}
    if devices != {"cuda", "npu"}:
        return
    cuda, npu = ((output_dir, other) if current["device"].startswith("cuda")
                 else (other, output_dir))
    rows = []
    print("\nNPU candidates vs unchanged CUDA native SDPA")
    print(f"{'dtype':<6} {'NPU variant':<18} {'max_abs':>12} {'RMSE':>12} {'rel_L2(%)':>12}")
    with np.load(cuda / "outputs.npz", allow_pickle=False) as reference, \
            np.load(npu / "outputs.npz", allow_pickle=False) as candidate:
        for dtype in current["dtypes"]:
            for variant in ("native", "fp32_cast_back"):
                row = dict(dtype=dtype, npu_variant=variant, cuda_reference="native",
                           **metrics(candidate[f"{dtype}__{variant}"], reference[f"{dtype}__native"]))
                rows.append(row)
                print(f"{dtype:<6} {variant:<18} {row['max_abs']:12.5e} "
                      f"{row['rmse']:12.5e} {row['relative_l2_pct']:12.5e}")
    (output_dir / "alignment_to_cuda_native.json").write_text(json.dumps(dict(rows=rows), indent=2) + "\n")
    write_csv(output_dir / "alignment_to_cuda_native.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="npu:7, cuda:0 or cpu")
    parser.add_argument("--inputs", type=Path, default=Path(__file__).with_name("inputs.npz"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=["fp16", "bf16"])
    parser.add_argument("--compare", type=Path, help="Other device's SDPA experiment outputs")
    parser.add_argument("--timing", action="store_true", help="Short synchronized timing")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("Use nonnegative warmup and positive repeat")
    if args.device.startswith("npu"):
        import torch_npu
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    elif device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        parser.error("Use npu, cuda or cpu")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if device.type == "npu":
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        device_name = torch.npu.get_device_name(device)
        flags = dict(matmul_allow_hf32=False, conv_allow_hf32=False)
    elif device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device_name = torch.cuda.get_device_name(device)
        flags = dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                     fp16_reduced_precision_reduction=
                     torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                     bf16_reduced_precision_reduction=
                     torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    else:
        device_name, flags = "CPU smoke test", {}
    with np.load(args.inputs, allow_pickle=False) as stored:
        original = [torch.from_numpy(stored[key].copy()) for key in ("q", "k", "v")]
    args.out.mkdir(parents=True, exist_ok=False)
    outputs, rows, metadata, rounding, timing = {}, [], {}, [], []
    print(f"Device: {device} ({device_name}); fixed Q/K/V, no mask or dropout")
    print("Accuracy reference: CPU FP64 using the same dtype-rounded Q/K/V")
    print(f"{'dtype':<6} {'variant':<18} {'max_abs':>12} {'RMSE':>12} {'rel_L2(%)':>12}")
    with torch.inference_mode():
        original_reference = reference_sdpa(*(value.double() for value in original)).numpy()
        for dtype_name in args.dtypes:
            rounded = [value.to(DTYPES[dtype_name]) for value in original]
            reference = reference_sdpa(*(value.double() for value in rounded)).numpy()
            rounding.append(dict(dtype=dtype_name, **metrics(reference, original_reference)))
            qkv = [value.to(device) for value in rounded]
            native = sdpa_forward(*qkv, "native")
            promoted = sdpa_forward(*qkv, "fp32_compute")
            # Both diagnostic outputs use the exact same promoted SDPA result.
            candidates = dict(native=native, fp32_compute=promoted,
                              fp32_cast_back=promoted.to(DTYPES[dtype_name]))
            synchronize(device)
            for variant, value in candidates.items():
                key = f"{dtype_name}__{variant}"
                array = value.float().cpu().numpy()
                outputs[key] = array
                row = dict(dtype=dtype_name, variant=variant, **metrics(array, reference))
                rows.append(row)
                metadata[key] = dict(shape=list(value.shape), output_dtype=str(value.dtype),
                                     sdpa_input_dtype=str(qkv[0].dtype if variant == "native" else torch.float32))
                print(f"{dtype_name:<6} {variant:<18} {row['max_abs']:12.5e} "
                      f"{row['rmse']:12.5e} {row['relative_l2_pct']:12.5e}", flush=True)
            if args.timing:
                for variant in ("native", "fp32_cast_back"):
                    row = dict(dtype=dtype_name, variant=variant,
                               **measure(qkv, variant, args.warmup, args.repeat))
                    timing.append(row)
                    print(f"Timing {dtype_name} {variant}: {row['mean_ms']:.4f} ms", flush=True)
            del candidates, native, promoted, qkv
    np.savez_compressed(args.out / "outputs.npz", **outputs)
    report = dict(device=str(device), device_name=device_name, torch=torch.__version__,
                  torch_npu=torch_npu.__version__ if device.type == "npu" else None,
                  numpy=np.__version__, cuda_version=torch.version.cuda,
                  spec=dict(qkv_shapes=[list(value.shape) for value in original],
                            mask=None, dropout=0.0, causal=False, variants=list(VARIANTS)),
                  dtypes=args.dtypes, precision_flags=flags,
                  backend_selection="automatic PyTorch SDPA; autocast disabled",
                  reference="CPU FP64 using the same dtype-rounded Q/K/V",
                  input_kind="existing fixed synthetic operator fixture, not model activations",
                  inputs_sha256=sha256(args.inputs), outputs_sha256=sha256(args.out / "outputs.npz"),
                  script_sha256=sha256(__file__), helper_script_sha256=sha256(Path(__file__).with_name("compare_core_ops.py")),
                  output_metadata=metadata, rows=rows,
                  input_rounding_vs_original_fp64=rounding, timing=timing,
                  timing_scope="device-resident SDPA call; promoted path includes Q/K/V casts and output cast")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.out / "accuracy.csv", rows)
    write_csv(args.out / "input_rounding.csv", rounding)
    if timing:
        write_csv(args.out / "timing.csv", timing)
    if args.compare:
        compare_saved(report, args.compare, args.out)
        compare_cuda_native(report, args.compare, args.out)
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
