"""Unit tests for prepare_priors shard selection and merge ordering."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    SPEC = importlib.util.spec_from_file_location(
        "prepare_priors", ROOT / "scripts" / "prepare_priors.py"
    )
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
    HAS_DEPS = True
except ImportError:
    # torch/transformers are unavailable in the lightweight CI environment
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "prepare_priors needs torch/transformers")
class ShardRowIndicesTests(unittest.TestCase):
    def test_round_robin_partition_is_exhaustive_and_disjoint(self):
        all_indices = []
        for shard in range(4):
            all_indices.extend(MODULE.shard_row_indices(10, shard, 4))
        self.assertEqual(sorted(all_indices), list(range(10)))

    def test_single_shard_owns_everything(self):
        self.assertEqual(MODULE.shard_row_indices(7, 0, 1), list(range(7)))

    def test_invalid_shard_spec_raises(self):
        with self.assertRaises(ValueError):
            MODULE.shard_row_indices(10, 4, 4)
        with self.assertRaises(ValueError):
            MODULE.shard_row_indices(10, 0, 0)


@unittest.skipUnless(HAS_DEPS, "prepare_priors needs torch/transformers")
class MergeDumpRecordsTests(unittest.TestCase):
    def test_merge_restores_original_row_order_and_renumbers(self):
        shard_a = [{"idx": 4, "n": 0}, {"idx": 0, "n": 1}]
        shard_b = [{"idx": 1, "n": 0}, {"idx": 3, "n": 1}]
        merged = MODULE.merge_dump_records(shard_a + shard_b)
        self.assertEqual([rec["idx"] for rec in merged], [0, 1, 3, 4])
        self.assertEqual([rec["n"] for rec in merged], [0, 1, 2, 3])

    def test_merge_keeps_token_payloads(self):
        records = [{"idx": 2, "n": 0, "tokens": [{"tok": "a"}]},
                   {"idx": 1, "n": 0, "tokens": [{"tok": "b"}]}]
        merged = MODULE.merge_dump_records(records)
        self.assertEqual(merged[0]["tokens"], [{"tok": "b"}])
        self.assertEqual(merged[1]["tokens"], [{"tok": "a"}])


if __name__ == "__main__":
    unittest.main()
