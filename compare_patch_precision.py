"""Compare original Conv2d and two equivalent patch Linear paths."""
import argparse
import json
from pathlib import Path

from compare_core_ops import DTYPES, compare_saved, metrics, sha256, write_csv
import numpy as np
import torch
import torch.nn.functional as F

VARIANTS = ("conv2d", "linear_separate_bias", "linear_with_bias")


def patch_forward(image, weight, bias, variant):
    kh, kw = weight.shape[-2:]
    if variant == "conv2d":
        return F.conv2d(image, weight, bias, stride=(kh, kw))
    batch, channels, height, width = image.shape
    oh, ow = height // kh, width // kw
    patches = image.reshape(batch, channels, oh, kh, ow, kw).permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(batch, oh * ow, -1).contiguous()
    matrix = weight.reshape(weight.shape[0], -1)
    if variant == "linear_separate_bias":
        result = F.linear(patches, matrix, None) + bias
    elif variant == "linear_with_bias":
        result = F.linear(patches, matrix, bias)
    else:
        raise ValueError(variant)
    return result.permute(0, 2, 1).reshape(batch, weight.shape[0], oh, ow)


def detailed_metrics(candidate, reference):
    result = metrics(candidate, reference)
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    index = np.unravel_index(int(diff.argmax()), diff.shape)
    result.update(p99_abs=float(np.percentile(diff, 99)),
                  elementwise_equal_pct=float(np.mean(candidate == reference) * 100),
                  max_diff_index=[int(value) for value in index],
                  candidate_at_max_diff=float(candidate[index]),
                  reference_at_max_diff=float(reference[index]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="npu:7, cuda:0 or cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().with_name("inputs.npz"))
    parser.add_argument("--compare", type=Path, help="Other device's patch result directory")
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if device.type == "npu":
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        device_name = torch.npu.get_device_name(device)
        flags = dict(matmul_allow_hf32=False, conv_allow_hf32=False)
    elif device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        device_name = torch.cuda.get_device_name(device)
        flags = dict(matmul_allow_tf32=False, cudnn_allow_tf32=False,
                     fp16_reduced_precision_reduction=
                     torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    elif device.type == "cpu":
        device_name, flags = "CPU smoke test", {}
    else:
        parser.error("Use npu, cuda or cpu")
    with np.load(args.inputs, allow_pickle=False) as stored:
        cpu = [torch.from_numpy(stored[key].copy())
               for key in ("image", "patch_weight", "patch_bias")]
    args.out.mkdir(parents=True, exist_ok=True)
    outputs, pairs, reference_rows, output_metadata = {}, [], [], {}
    print(f"Device: {device} ({device_name}); same inputs and weights, TF32/HF32 disabled")
    print(f"{'dtype':<6} {'candidate vs reference':<48} {'max_abs':>12} {'RMSE':>12} {'equal(%)':>12}")
    with torch.inference_mode():
        for dtype_name in args.dtypes:
            rounded = [value.to(DTYPES[dtype_name]) for value in cpu]
            reference = patch_forward(*(value.double() for value in rounded), "conv2d").numpy()
            inputs = [value.to(device) for value in rounded]
            stage = {}
            for variant in VARIANTS:
                out = patch_forward(*inputs, variant)
                if device.type != "cpu":
                    getattr(torch, device.type).synchronize(device)
                stage[variant] = out.float().cpu().numpy()
                key = f"{dtype_name}__{variant}"
                outputs[key] = stage[variant]
                output_metadata[key] = dict(shape=list(out.shape), output_dtype=str(out.dtype))
                reference_rows.append(dict(dtype=dtype_name, variant=variant,
                                           **detailed_metrics(stage[variant], reference)))
            for candidate, baseline in (("linear_separate_bias", "conv2d"),
                                         ("linear_with_bias", "conv2d"),
                                         ("linear_with_bias", "linear_separate_bias")):
                row = dict(dtype=dtype_name, candidate=candidate, reference=baseline,
                           **detailed_metrics(stage[candidate], stage[baseline]))
                pairs.append(row)
                label = f"{candidate} vs {baseline}"
                print(f"{dtype_name:<6} {label:<48} {row['max_abs']:12.5e} "
                      f"{row['rmse']:12.5e} {row['elementwise_equal_pct']:12.5f}", flush=True)
            del inputs
    np.savez_compressed(args.out / "outputs.npz", **outputs)
    report = dict(device=str(device), device_name=device_name, torch=torch.__version__,
                  torch_npu=torch_npu.__version__ if device.type == "npu" else None,
                  numpy=np.__version__, cuda_version=torch.version.cuda, precision_flags=flags,
                  spec=dict(image_shape=list(cpu[0].shape), weight_shape=list(cpu[1].shape),
                            bias_shape=list(cpu[2].shape), variants=list(VARIANTS)),
                  input_kind="fixed synthetic fixture shared with the core operator comparison",
                  reference="CPU FP64 Conv2d using the same dtype-rounded input and weights",
                  inputs_sha256=sha256(args.inputs), outputs_sha256=sha256(args.out / "outputs.npz"),
                  script_sha256=sha256(__file__), helper_script_sha256=sha256(Path(__file__).with_name("compare_core_ops.py")),
                  output_metadata=output_metadata, same_device_comparisons=pairs,
                  reference_accuracy=reference_rows)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.out / "same_device_comparison.csv", pairs)
    write_csv(args.out / "reference_accuracy.csv", reference_rows)
    if args.compare:
        compare_saved(report, args.compare, args.out)
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
