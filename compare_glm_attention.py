"""Compare real CUDA FlashAttention 2 with the GLM Transformers NPU SDPA entry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from compare_core_ops import metrics, sha256, write_csv


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
HEADS, KV_HEADS, HEAD_DIM = 96, 8, 128


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for name, value in zip(("q", "k", "v"), tensors):
        digest.update(name.encode())
        digest.update(str(list(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_fixture(length, dtype, seed):
    rng = np.random.default_rng(seed)
    shapes = [(1, heads, length, HEAD_DIM) for heads in (HEADS, KV_HEADS, KV_HEADS)]
    original = [torch.from_numpy(rng.standard_normal(shape).astype(np.float32)) for shape in shapes]
    rounded = [value.to(DTYPES[dtype]) for value in original]
    return rounded, tensor_hash(original), tensor_hash(rounded)


def cases(q, k, v):
    return {"prefill": (q, k, v), "decode": (q[:, :, -1:, :].contiguous(), k, v),
            "chunk_decode": (q[:, :, -4:, :].contiguous(), k, v)}


def bottom_right_mask(query_length, kv_length, device):
    queries = torch.arange(query_length, device=device) + kv_length - query_length
    keys = torch.arange(kv_length, device=device)
    return (keys[None, :] <= queries[:, None])[None, None, :, :]


def sdpa_mask_and_causal(query_length, kv_length, device):
    if query_length == kv_length:
        return None, True
    if query_length == 1:
        return None, False
    # Cached multi-token queries need their absolute positions, not a top-left triangle.
    return bottom_right_mask(query_length, kv_length, device), False


def attention_forward(q, k, v, backend, implementation):
    if backend == "cuda":
        return implementation(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                              v.transpose(1, 2).contiguous(), dropout_p=0.0,
                              softmax_scale=q.shape[-1] ** -0.5, causal=True)
    if backend != "npu":
        raise ValueError(backend)
    mask, causal = sdpa_mask_and_causal(q.shape[2], k.shape[2], q.device)
    module = SimpleNamespace(num_key_value_groups=q.shape[1] // k.shape[1], is_causal=True)
    output, _ = implementation(module, q, k, v, mask, dropout=0.0,
                               scaling=q.shape[-1] ** -0.5, is_causal=causal)
    return output


def reference_attention(q, k, v):
    q, k, v = (value.double() for value in (q, k, v))
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    scores = (q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5
    allowed = bottom_right_mask(q.shape[2], k.shape[2], q.device)
    output = torch.softmax(scores.masked_fill(~allowed, -torch.inf), -1) @ v
    return output.transpose(1, 2).contiguous().numpy()


def output_bits(value):
    return value.detach().cpu().contiguous().view(torch.uint16).numpy().copy()


def decode_bits(array, dtype):
    if array.dtype != np.uint16:
        raise ValueError("Expected exact uint16 output bits")
    return torch.from_numpy(array.copy()).view(DTYPES[dtype]).float().numpy()


def synchronize(device):
    getattr(torch, device.type).synchronize(device)


def compare_results(current, other, output_dir):
    prior = json.loads((other / "report.json").read_text())
    for key in ("spec", "dtype", "inputs_fp32_sha256", "inputs_rounded_sha256",
                "script_sha256", "helper_script_sha256", "output_layout"):
        if current[key] != prior[key]:
            raise ValueError(f"Comparison mismatch: {key}; use the same script and fixture")
    if {current["device"].split(":")[0], prior["device"].split(":")[0]} != {"cuda", "npu"}:
        raise ValueError("Compare a CUDA FlashAttention 2 run with an NPU HF SDPA run")
    if sha256(other / "outputs.npz") != prior["outputs_sha256"]:
        raise ValueError("Saved comparison outputs do not match their report")
    cuda, npu = ((output_dir, other) if current["device"].startswith("cuda") else (other, output_dir))
    rows = []
    print("\nNPU Transformers SDPA vs unchanged CUDA FlashAttention 2")
    print(f"{'case':<14} {'max_abs':>12} {'RMSE':>12} {'rel_L2(%)':>12}")
    with np.load(cuda / "outputs.npz", allow_pickle=False) as reference, \
            np.load(npu / "outputs.npz", allow_pickle=False) as candidate:
        for name in ("prefill", "decode", "chunk_decode"):
            row = dict(case=name, dtype=current["dtype"],
                       **metrics(decode_bits(candidate[name], current["dtype"]),
                                 decode_bits(reference[name], current["dtype"])))
            rows.append(row)
            print(f"{name:<14} {row['max_abs']:12.5e} {row['rmse']:12.5e} "
                  f"{row['relative_l2_pct']:12.5e}")
    write_csv(output_dir / "comparison.csv", rows)
    (output_dir / "comparison.json").write_text(json.dumps(dict(rows=rows), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="cuda:0 or npu:7")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.length < 8:
        parser.error("Use --length >= 8 to keep cached queries distinct from prefill")
    if args.device.startswith("npu"):
        import torch_npu
    device = torch.device(args.device)
    backend = device.type
    if backend not in ("cuda", "npu"):
        parser.error("Use cuda or npu; this entry has no CPU fallback")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    getattr(torch, backend).set_device(device)
    if backend == "cuda":
        import flash_attn
        from flash_attn import flash_attn_func

        version = tuple(map(int, re.match(r"(\d+)\.(\d+)", flash_attn.__version__).groups()))
        if not ((2, 1) <= version < (3, 0)):
            raise ValueError("Use FlashAttention 2 >= 2.1 for bottom-right causal alignment")
        implementation = flash_attn_func
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        versions = dict(flash_attn=flash_attn.__version__, transformers=None, torch_npu=None)
        backend_info = dict(entry="flash_attn.flash_attn_func", causal_alignment="bottom_right")
    else:
        import transformers
        from transformers.integrations import sdpa_attention

        implementation = sdpa_attention.sdpa_attention_forward
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        versions = dict(flash_attn=None, transformers=transformers.__version__, torch_npu=torch_npu.__version__)
        backend_info = dict(entry="transformers.integrations.sdpa_attention.sdpa_attention_forward",
                            hf_sdpa_source_sha256=sha256(sdpa_attention.__file__),
                            kernel_selection="torch_npu SDPA dispatch; no kernel profiling in this test")
    fixture, original_hash, rounded_hash = make_fixture(args.length, args.dtype, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, arrays, metadata, timing = [], {}, {}, []
    print(f"Device: {device} ({getattr(torch, backend).get_device_name(device)})")
    print(f"Backend: {backend_info['entry']}; dtype: {args.dtype}; Q heads=96, KV heads=8, head_dim=128")
    print("Accuracy reference: CPU FP64 on the identical dtype-rounded Q/K/V")
    print(f"{'case':<14} {'max_abs':>12} {'RMSE':>12} {'rel_L2(%)':>12}")
    with torch.inference_mode():
        for name, qkv_cpu in cases(*fixture).items():
            reference = reference_attention(*qkv_cpu)
            qkv = tuple(value.to(device) for value in qkv_cpu)
            output = attention_forward(*qkv, backend, implementation)
            synchronize(device)
            expected_shape = (qkv[0].shape[0], qkv[0].shape[2], HEADS, HEAD_DIM)
            if tuple(output.shape) != expected_shape or output.dtype != DTYPES[args.dtype]:
                raise ValueError(f"Unexpected output: {output.shape}, {output.dtype}")
            arrays[name] = output_bits(output)
            row = dict(case=name, dtype=args.dtype, **metrics(output.float().cpu().numpy(), reference))
            rows.append(row)
            metadata[name] = dict(q_shape=list(qkv[0].shape), kv_shape=list(qkv[1].shape),
                                  output_shape=list(output.shape), output_dtype=str(output.dtype))
            print(f"{name:<14} {row['max_abs']:12.5e} {row['rmse']:12.5e} {row['relative_l2_pct']:12.5e}", flush=True)
            if args.timing:
                for _ in range(3):
                    attention_forward(*qkv, backend, implementation)
                synchronize(device)
                durations = []
                for _ in range(10):
                    start = time.perf_counter()
                    attention_forward(*qkv, backend, implementation)
                    synchronize(device)
                    durations.append((time.perf_counter() - start) * 1000)
                timing.append(dict(case=name, mean_ms=float(np.mean(durations)), warmup=3, repeat=10))
            del output, qkv
    np.savez_compressed(args.out / "outputs.npz", **arrays)
    report = dict(device=str(device), torch=torch.__version__, numpy=np.__version__,
                  cuda_version=torch.version.cuda, **versions, backend=backend_info, dtype=args.dtype,
                  spec=dict(batch=1, q_heads=HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM,
                            length=args.length, seed=args.seed, dropout=0.0, scale=HEAD_DIM ** -0.5,
                            causal_alignment="bottom_right", cases=["prefill", "decode", "chunk_decode"]),
                  input_kind="fixed synthetic post-RoPE operator fixture; not captured model activations",
                  inputs_fp32_sha256=original_hash, inputs_rounded_sha256=rounded_hash,
                  outputs_sha256=sha256(args.out / "outputs.npz"),
                  script_sha256=sha256(__file__), helper_script_sha256=sha256(Path(__file__).with_name("compare_core_ops.py")),
                  output_layout="BSHD", storage="uint16 original FP16/BF16 bits, no output requantization",
                  output_metadata=metadata, rows=rows, timing=timing,
                  scope="attention only; no model weights, QKV projections, RoPE, cache writes or generation",
                  timing_scope="device-resident adapter call including layout copies and mask preparation")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.out / "accuracy.csv", rows)
    if timing:
        write_csv(args.out / "timing.csv", timing)
    if args.compare:
        compare_results(report, args.compare, args.out)
    print(f"\nSaved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
