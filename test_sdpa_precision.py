"""CPU checks for the SDPA experiment's precision boundaries and reports."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import compare_sdpa_precision as experiment


class SDPAPrecisionTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(17)
        self.original = [torch.randn(1, 2, 7, 4, generator=generator) for _ in range(3)]

    def test_output_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16):
            qkv = [value.to(dtype) for value in self.original]
            for variant in experiment.VARIANTS:
                with self.subTest(dtype=dtype, variant=variant):
                    result = experiment.sdpa_forward(*qkv, variant)
                    self.assertEqual(result.dtype, torch.float32 if variant == "fp32_compute" else dtype)
                    self.assertEqual(result.shape, qkv[0].shape)

    def test_cast_back_matches_promoted_result(self):
        qkv = [value.bfloat16() for value in self.original]
        promoted = experiment.sdpa_forward(*qkv, "fp32_compute")
        restored = experiment.sdpa_forward(*qkv, "fp32_cast_back")
        self.assertTrue(torch.equal(promoted.bfloat16(), restored))

    def test_outer_autocast_does_not_lower_promoted_output(self):
        qkv = [value.bfloat16() for value in self.original]
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = experiment.sdpa_forward(*qkv, "fp32_compute")
        self.assertEqual(result.dtype, torch.float32)

    def test_fp64_reference_matches_pytorch_sdpa(self):
        qkv = [value.double() for value in self.original]
        actual = experiment.reference_sdpa(*qkv)
        expected = torch.nn.functional.scaled_dot_product_attention(*qkv)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_alignment_preserves_cuda_native_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            npu, cuda = root / "npu", root / "cuda"
            npu.mkdir()
            cuda.mkdir()
            native = np.array([1.0, 2.0], dtype=np.float32)
            np.savez_compressed(cuda / "outputs.npz", bf16__native=native,
                                bf16__fp32_cast_back=native * 10)
            np.savez_compressed(npu / "outputs.npz", bf16__native=native * 2,
                                bf16__fp32_cast_back=native)
            (npu / "report.json").write_text(json.dumps(dict(device="npu:7")))
            report = dict(device="cuda:0", dtypes=["bf16"])
            with contextlib.redirect_stdout(io.StringIO()):
                experiment.compare_cuda_native(report, npu, cuda)
            rows = json.loads((cuda / "alignment_to_cuda_native.json").read_text())["rows"]
            self.assertGreater(rows[0]["relative_l2_pct"], 0)
            self.assertEqual(rows[1]["relative_l2_pct"], 0)

    def test_cli_reports_and_saved_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "inputs.npz"
            np.savez_compressed(inputs, **{
                key: value.numpy() for key, value in zip(("q", "k", "v"), self.original)
            })
            for name in ("first", "second"):
                args = ["compare_sdpa_precision.py", "--device", "cpu", "--inputs", str(inputs),
                        "--out", str(root / name), "--timing", "--warmup", "0", "--repeat", "1"]
                if name == "second":
                    args += ["--compare", str(root / "first")]
                with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                    experiment.main()
            report = json.loads((root / "second" / "report.json").read_text())
            self.assertEqual(len(report["rows"]), 6)
            self.assertEqual(len(report["timing"]), 4)
            self.assertEqual(len(report["input_rounding_vs_original_fp64"]), 2)
            comparison = json.loads((root / "second" / "comparison.json").read_text())
            self.assertTrue(all(row["max_abs"] == 0 for row in comparison["rows"]))


if __name__ == "__main__":
    unittest.main()
