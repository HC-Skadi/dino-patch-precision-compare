import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F

import compare_glm_attention as experiment


class GLMAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(11)
        self.q = torch.from_numpy(rng.normal(size=(1, 4, 8, 8))).to(torch.bfloat16)
        self.k = torch.from_numpy(rng.normal(size=(1, 2, 8, 8))).to(torch.bfloat16)
        self.v = torch.from_numpy(rng.normal(size=(1, 2, 8, 8))).to(torch.bfloat16)

    def test_cached_mask_uses_bottom_right_positions(self):
        mask = experiment.bottom_right_mask(2, 5, "cpu")[0, 0]
        self.assertEqual(mask.tolist(), [[True, True, True, True, False], [True] * 5])
        mask, causal = experiment.sdpa_mask_and_causal(1, 5, "cpu")
        self.assertIsNone(mask)
        self.assertFalse(causal)

    def test_prefill_decode_and_chunk_reference_consistency(self):
        outputs = {name: experiment.reference_attention(*values)
                   for name, values in experiment.cases(self.q, self.k, self.v).items()}
        np.testing.assert_allclose(outputs["decode"], outputs["prefill"][:, -1:], atol=1e-14)
        np.testing.assert_allclose(outputs["chunk_decode"], outputs["prefill"][:, -4:], atol=1e-14)

    def test_reference_matches_explicit_cpu_sdpa(self):
        for _, qkv in experiment.cases(self.q, self.k, self.v).items():
            q, k, v = (value.double() for value in qkv)
            mask = experiment.bottom_right_mask(q.shape[2], k.shape[2], "cpu")
            output = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1),
                                                   v.repeat_interleave(2, 1), attn_mask=mask)
            np.testing.assert_allclose(output.transpose(1, 2).numpy(),
                                       experiment.reference_attention(*qkv), atol=1e-14)

    def test_cuda_receives_bshd_and_causal_flag(self):
        flash = Mock(side_effect=lambda q, k, v, **kwargs: q)
        output = experiment.attention_forward(self.q, self.k, self.v, "cuda", flash)
        q, k, v = flash.call_args.args
        self.assertEqual(q.shape, (1, 8, 4, 8))
        self.assertEqual(k.shape, (1, 8, 2, 8))
        self.assertTrue(q.is_contiguous() and k.is_contiguous() and v.is_contiguous())
        self.assertTrue(flash.call_args.kwargs["causal"])
        self.assertEqual(output.shape, q.shape)

    def test_npu_receives_matching_masks_and_gqa_groups(self):
        sdpa = Mock(side_effect=lambda module, q, k, v, mask, **kwargs: (q.transpose(1, 2), None))
        for name, qkv in experiment.cases(self.q, self.k, self.v).items():
            experiment.attention_forward(*qkv, "npu", sdpa)
            module, q, k, v, mask = sdpa.call_args.args
            self.assertEqual(module.num_key_value_groups, 2)
            self.assertEqual(sdpa.call_args.kwargs["is_causal"], name == "prefill")
            if name == "chunk_decode":
                self.assertEqual(mask.dtype, torch.bool)
                self.assertEqual(mask.shape, (1, 1, 4, 8))
            else:
                self.assertIsNone(mask)

    def test_output_bits_preserve_bf16_and_fp16(self):
        for name, dtype in experiment.DTYPES.items():
            original = self.q.to(dtype)
            restored = experiment.decode_bits(experiment.output_bits(original), name)
            np.testing.assert_array_equal(restored, original.float().numpy())

    def test_installed_hf_sdpa_gqa_and_causal_dispatch(self):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        def kernel(q, k, v, **kwargs):
            return torch.zeros_like(q)

        with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=kernel) as sdpa:
            for name, qkv in experiment.cases(self.q, self.k, self.v).items():
                experiment.attention_forward(*qkv, "npu", sdpa_attention_forward)
                kwargs = sdpa.call_args.kwargs
                self.assertEqual(kwargs["is_causal"], name == "prefill")
                if name == "chunk_decode":
                    self.assertEqual(sdpa.call_args.args[1].shape[1], self.q.shape[1])
                    self.assertNotIn("enable_gqa", kwargs)
                else:
                    self.assertTrue(kwargs["enable_gqa"])

    def test_fixture_hash_and_rounding_are_repeatable(self):
        one, fp32_hash, rounded_hash = experiment.make_fixture(8, "bf16", 11)
        two, other_fp32_hash, other_rounded_hash = experiment.make_fixture(8, "bf16", 11)
        self.assertEqual((fp32_hash, rounded_hash), (other_fp32_hash, other_rounded_hash))
        self.assertTrue(all(value.dtype == torch.bfloat16 for value in one))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(one, two)))

    def test_comparison_keeps_cuda_as_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            cuda, npu = Path(folder) / "cuda", Path(folder) / "npu"
            cuda.mkdir()
            npu.mkdir()
            saved = {name: experiment.output_bits(self.q) for name in ("prefill", "decode", "chunk_decode")}
            for directory in (cuda, npu):
                np.savez_compressed(directory / "outputs.npz", **saved)
            report = dict(spec={}, dtype="bf16", inputs_fp32_sha256="one", inputs_rounded_sha256="two",
                          script_sha256="three", helper_script_sha256="four", output_layout="BSHD")
            prior = dict(report, device="npu:7", outputs_sha256=experiment.sha256(npu / "outputs.npz"))
            (npu / "report.json").write_text(json.dumps(prior))
            experiment.compare_results(dict(report, device="cuda:0"), npu, cuda)
            rows = json.loads((cuda / "comparison.json").read_text())["rows"]
            self.assertTrue(all(row["relative_l2_pct"] == 0 for row in rows))
            prior["inputs_rounded_sha256"] = "different"
            (npu / "report.json").write_text(json.dumps(prior))
            with self.assertRaisesRegex(ValueError, "inputs_rounded_sha256"):
                experiment.compare_results(dict(report, device="cuda:0"), npu, cuda)


if __name__ == "__main__":
    unittest.main()
