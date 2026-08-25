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
class TestExpandBlockTable(unittest.TestCase):
    def test_identity_when_already_64(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        from sm12x_b12x_kernels import expand_block_table_to_page64

        tables = torch.tensor([[0, 3, 7]], dtype=torch.int32)
        out = expand_block_table_to_page64(tables, block_size=64)
        self.assertTrue(torch.equal(out, tables))

    def test_256_to_four_kernel_pages(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        from sm12x_b12x_kernels import expand_block_table_to_page64

        tables = torch.tensor([[2, 5]], dtype=torch.int32)
        out = expand_block_table_to_page64(tables, block_size=256)
        self.assertEqual(tuple(out.shape), (1, 8))
        self.assertEqual(out[0].tolist(), [8, 9, 10, 11, 20, 21, 22, 23])

    def test_rejects_non_multiple(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        from sm12x_b12x_kernels import expand_block_table_to_page64

        tables = torch.tensor([[1]], dtype=torch.int32)
        # 200 is not a multiple of the 64-token kernel page size (192 would
        # be: 192 // 64 == 3, a valid expansion).
        with self.assertRaises(ValueError):
            expand_block_table_to_page64(tables, block_size=200)

    def test_trim_drops_last_page_at_schedule_threshold(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        from sm12x_b12x_kernels import trim_page_table_skip_schedule

        tables = torch.arange(1024, dtype=torch.int32).view(1, 1024)
        out = trim_page_table_skip_schedule(tables)
        self.assertEqual(tuple(out.shape), (1, 1023))
        self.assertEqual(int(out[0, -1]), 1022)
        small = torch.arange(16, dtype=torch.int32).view(1, 16)
        self.assertTrue(torch.equal(trim_page_table_skip_schedule(small), small))


@unittest.skipUnless(torch is not None, "torch is not installed")
class TestUsableB12xSchedule(unittest.TestCase):
    def test_rejects_none_and_wrong_layout(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        from sm12x_b12x_kernels import _usable_b12x_schedule

        device = torch.device("cpu")
        self.assertFalse(_usable_b12x_schedule(None, device))
        bad_dtype = torch.zeros((8, 2), dtype=torch.int64)
        self.assertFalse(_usable_b12x_schedule(bad_dtype, device))
        bad_width = torch.zeros((8, 3), dtype=torch.int32)
        self.assertFalse(_usable_b12x_schedule(bad_width, device))
        too_short = torch.zeros((1, 2), dtype=torch.int32)
        self.assertFalse(_usable_b12x_schedule(too_short, device))

    def test_accepts_sms_plus_one_int32(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        from sm12x_b12x_kernels import _usable_b12x_schedule

        device = torch.device("cpu")
        ok = torch.zeros((11, 2), dtype=torch.int32)
        self.assertTrue(_usable_b12x_schedule(ok, device))


@unittest.skipUnless(torch is not None, "torch is not installed")
class TestConsumeVllmPagedSchedule(unittest.TestCase):
    def test_only_single_row_uses_schedule(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        from sm12x_b12x_kernels import _consume_vllm_paged_schedule

        self.assertTrue(
            _consume_vllm_paged_schedule(
                need_sched=True, q_rows=1, schedule_ok=True
            )
        )
        self.assertFalse(
            _consume_vllm_paged_schedule(
                need_sched=True, q_rows=6, schedule_ok=True
            )
        )
        self.assertFalse(
            _consume_vllm_paged_schedule(
                need_sched=True, q_rows=8, schedule_ok=True
            )
        )
        self.assertFalse(
            _consume_vllm_paged_schedule(
                need_sched=True, q_rows=48, schedule_ok=True
            )
        )
        self.assertFalse(
            _consume_vllm_paged_schedule(
                need_sched=True, q_rows=1, schedule_ok=False
            )
        )
        self.assertFalse(
            _consume_vllm_paged_schedule(
                need_sched=False, q_rows=1, schedule_ok=True
            )
        )


@unittest.skipUnless(torch is not None, "torch is not installed")
class TestPackIndexerPages(unittest.TestCase):
    def test_k_then_scale_layout(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        from sm12x_b12x_kernels import _PACKED_PAGE_BYTES, pack_indexer_k_pages

        pages, block, width = 2, 64, 132
        cache = torch.zeros(pages, block, 1, width, dtype=torch.uint8)
        cache[0, 0, 0, 0] = 7
        cache[0, 0, 0, 128] = 9
        packed = pack_indexer_k_pages(cache)
        self.assertIsNotNone(packed)
        assert packed is not None
        self.assertEqual(tuple(packed.shape), (pages, _PACKED_PAGE_BYTES))
        self.assertEqual(int(packed[0, 0]), 7)
        self.assertEqual(int(packed[0, 64 * 128]), 9)

    def test_from_ids_matches_full_pack_subset(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        from sm12x_b12x_kernels import pack_indexer_k_pages, pack_indexer_k_pages_from_ids

        pages, block, width = 3, 64, 132
        cache = torch.arange(pages * block * width, dtype=torch.uint8).view(
            pages, block, width
        )
        full = pack_indexer_k_pages(cache)
        self.assertIsNotNone(full)
        assert full is not None
        full = full.clone()
        ids = torch.tensor([[2, 0]], dtype=torch.int32)
        compact = pack_indexer_k_pages_from_ids(cache, ids)
        self.assertIsNotNone(full)
        self.assertIsNotNone(compact)
        assert full is not None and compact is not None
        self.assertEqual(tuple(compact.shape), (2, full.shape[1]))
        self.assertTrue(torch.equal(compact[0], full[2]))
        self.assertTrue(torch.equal(compact[1], full[0]))


@unittest.skipUnless(torch is not None, "torch is not installed")
class TestPackedIndexerInsert(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches" / "files"))
        import sm12x_b12x_kernels as k

        k._packed_sidecars.clear()
        k._insert_ok_logged = False

    def test_sync_matches_full_pack_at_written_slots(self) -> None:
        from sm12x_b12x_kernels import (
            lookup_packed_indexer_k,
            pack_indexer_k_pages,
            sync_packed_indexer_k,
        )

        cache = torch.zeros(2, 256, 1, 132, dtype=torch.uint8)
        cache[0, 0, 0, 0] = 7
        cache[0, 0, 0, 128] = 9
        cache[1, 64, 0, 1] = 3
        cache[1, 64, 0, 128] = 11
        slots = torch.tensor([0, 256 + 64], dtype=torch.int32)
        packed = sync_packed_indexer_k(cache, slots)
        self.assertIsNotNone(packed)
        assert packed is not None
        full = pack_indexer_k_pages(cache)
        self.assertIsNotNone(full)
        assert full is not None
        full = full.clone()
        self.assertEqual(tuple(packed.shape), (8, full.shape[1]))
        self.assertTrue(torch.equal(packed[0], full[0]))
        self.assertTrue(torch.equal(packed[5], full[5]))
        looked = lookup_packed_indexer_k(cache)
        self.assertIsNotNone(looked)
        assert looked is not None
        self.assertTrue(torch.equal(looked[0], packed[0]))

    def test_negative_slots_do_not_clobber_page0(self) -> None:
        from sm12x_b12x_kernels import sync_packed_indexer_k

        cache = torch.zeros(1, 256, 1, 132, dtype=torch.uint8)
        cache[0, 0, 0, 0] = 4
        cache[0, 0, 0, 128] = 5
        first = sync_packed_indexer_k(cache, torch.tensor([0], dtype=torch.int32))
        self.assertIsNotNone(first)
        assert first is not None
        page0 = first[0].clone()
        again = sync_packed_indexer_k(cache, torch.tensor([-1, -1], dtype=torch.int32))
        self.assertIsNotNone(again)
        assert again is not None
        self.assertTrue(torch.equal(again[0], page0))


if __name__ == "__main__":
    unittest.main()
