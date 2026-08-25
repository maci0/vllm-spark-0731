#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@unittest.skipUnless(torch is not None, "torch is not installed")
class TestPackedGatherMqaLogits(unittest.TestCase):
    def test_packed_gather_mqa_logits_matches_reference(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        from sm12x_b12x_kernels import packed_gather_mqa_logits

        torch.manual_seed(0)
        n_pages, page_size, D = 4, 64, 128
        M, H = 2, 8
        max_model_len = n_pages * page_size

        # Build packed K-then-scale data: per page, 64 tokens of fp8 K then 64 fp32 scales.
        k_fp8 = (torch.randn(n_pages * page_size, D) * 0.1).to(torch.float8_e4m3fn)
        scale = torch.rand(n_pages * page_size) + 0.5
        packed = torch.zeros(n_pages, page_size * (D + 4), dtype=torch.uint8)
        packed[:, : page_size * D] = k_fp8.view(torch.uint8).reshape(n_pages, page_size * D)
        packed[:, page_size * D :] = scale.view(torch.uint8).reshape(n_pages, page_size * 4)

        # kv_cache holds the same raw bytes, viewed as [n_pages, 64, 132].
        kv_cache = packed.view(n_pages, page_size, D + 4).clone()

        q_fp8 = (torch.randn(M, H, D) * 0.1).to(torch.float8_e4m3fn)
        w = torch.randn(M, H)
        block_tables = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32)
        context_lens = torch.tensor([256, 192], dtype=torch.int32)

        logits = packed_gather_mqa_logits(
            (q_fp8, None), kv_cache, w, context_lens, block_tables, max_model_len
        )
        self.assertIsNotNone(logits, "function returned None")
        assert logits is not None
        self.assertEqual(logits.shape, (M, max_model_len), f"shape {logits.shape}")

        # Reference: score = q @ (k * scale).T ; logits = w @ relu(score), masked by ctx.
        k_real = k_fp8.to(torch.float32) * scale.view(-1, 1)
        k_real = k_real.view(n_pages, page_size, D)
        ref = torch.full((M, max_model_len), float("-inf"))
        for m in range(M):
            k_m = k_real[block_tables[m]].reshape(-1, D)
            score = q_fp8[m].to(torch.float32) @ k_m.T
            lm = (w[m].unsqueeze(0) @ torch.relu(score)).squeeze(0)
            n = int(context_lens[m])
            ref[m, :n] = lm[:n]

        valid = torch.isfinite(ref)
        diff = (logits[valid] - ref[valid]).abs().max().item()
        self.assertLess(diff, 1e-3, "MISMATCH")


if __name__ == "__main__":
    unittest.main()
