"""Small CPU checks for BF16 support and saved-output comparison."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import compare_core_ops as core
import compare_patch_precision as patch_precision
import compare_patch_to_a100 as patch_comparison


class BF16SupportTests(unittest.TestCase):
    def test_existing_defaults_are_preserved(self):
        self.assertEqual(core.DEFAULT_DTYPES, ("fp32", "fp16"))
        self.assertEqual(core.DTYPES["bf16"], torch.bfloat16)

    def test_bf16_values_survive_float32_numpy_storage(self):
        value = torch.tensor([-3.14159, 0.0012345, 1.23456], dtype=torch.bfloat16)
        stored = value.float().numpy()
        self.assertTrue(torch.equal(torch.from_numpy(stored).bfloat16(), value))

    def test_all_core_operators_return_bf16(self):
        sizes = dict(batch=1, tokens=5, hidden=4, intermediate=8, heads=2,
                     head_dim=2, image=8, patch=4)
        with tempfile.TemporaryDirectory() as folder, patch.dict(core.SPEC, sizes):
            path = Path(folder) / "inputs.npz"
            core.create_inputs(path)
            with np.load(path, allow_pickle=False) as stored:
                inputs = {key: torch.from_numpy(stored[key]).bfloat16()
                          for key in stored.files}
            for name in core.CASES:
                with self.subTest(operator=name):
                    output = core.operation(name, inputs)
                    self.assertEqual(output.dtype, torch.bfloat16)
                    self.assertTrue(bool(output.isfinite().all()))

    def test_all_patch_variants_return_bf16(self):
        generator = torch.Generator().manual_seed(23)
        image = torch.randn(1, 3, 8, 8, generator=generator).bfloat16()
        weight = torch.randn(4, 3, 4, 4, generator=generator).bfloat16()
        bias = torch.randn(4, generator=generator).bfloat16()
        for variant in patch_precision.VARIANTS:
            with self.subTest(variant=variant):
                output = patch_precision.patch_forward(image, weight, bias, variant)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(tuple(output.shape), (1, 4, 2, 2))

    def test_saved_bf16_patch_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            npu, cuda, out = (root / name for name in ("npu", "cuda", "out"))
            for path in (npu, cuda):
                path.mkdir()
            reference = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
            np.savez_compressed(cuda / "outputs.npz", bf16__patch_embed=reference)
            np.savez_compressed(npu / "outputs.npz", **{
                "bf16__" + name: reference + index * 0.015625
                for index, name in enumerate(patch_precision.VARIANTS)
            })
            for path, device in ((npu, "npu:7"), (cuda, "cuda:0")):
                report = dict(device=device, inputs_sha256="shared-input",
                              outputs_sha256=core.sha256(path / "outputs.npz"),
                              patch_implementation="conv2d")
                (path / "report.json").write_text(json.dumps(report))
            args = ["compare_patch_to_a100.py", "--dtypes", "bf16",
                    "--npu-results", str(npu), "--a100-results", str(cuda),
                    "--out", str(out)]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                patch_comparison.main()
            result = json.loads((out / "comparison.json").read_text())
            self.assertEqual(result["dtypes"], ["bf16"])
            self.assertEqual(len(result["rows"]), 3)
            self.assertEqual(result["rows"][0]["elementwise_equal_pct"], 100.0)
            self.assertEqual(result["rows"][2]["max_abs"], 0.03125)


if __name__ == "__main__":
    unittest.main()
