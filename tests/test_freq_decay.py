"""Unit tests for the frequency-decay extrapolation reallocation.

Covers: freq-table loading, the budget-matching invariant of
redistribute_by_freq, non-eligible neutrality, the damping direction,
and the SelfDistillationConfig validations for vd_freq_decay.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

UTILS_SPEC = importlib.util.spec_from_file_location(
    "lookaway_utils", ROOT / "verl" / "workers" / "actor" / "lookaway_utils.py"
)
UTILS = importlib.util.module_from_spec(UTILS_SPEC)
UTILS_SPEC.loader.exec_module(UTILS)

sys.path.insert(0, str(ROOT))
from verl.workers.config.actor import SelfDistillationConfig  # noqa: E402


def make_freq_file(tmpdir, vocab_size, counts):
    path = Path(tmpdir) / "token_freq.json"
    path.write_text(json.dumps({"vocab_size": vocab_size, "counts": counts}))
    return path


class LoadFreqTableTests(unittest.TestCase):
    def test_loads_counts_into_vocab_sized_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = make_freq_file(tmpdir, 10, {"3": 7, "8": 120})
            table = UTILS.load_freq_table(str(path))
            self.assertEqual(table.shape, (10,))
            self.assertEqual(table[3].item(), 7.0)
            self.assertEqual(table[8].item(), 120.0)
            self.assertEqual(table[0].item(), 0.0)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            UTILS.load_freq_table("/nonexistent/token_freq.json")


class RedistributeByFreqTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # vocab: id 1 = high frequency (n=4000), id 2 = low frequency (n=40)
        self.freq = torch.zeros(8)
        self.freq[1] = 4000.0
        self.freq[2] = 40.0
        # batch of 2 responses x 3 tokens; valid mask drops pad column
        self.response_ids = torch.tensor([[1, 1, 2], [2, 1, 0]])
        self.valid = torch.tensor([[True, True, True], [True, True, False]])
        ext = torch.zeros(2, 3)
        ext[0, 0] = 0.5
        ext[0, 2] = 0.5
        ext[1, 0] = 0.5
        self.ext = ext

    def test_budget_total_is_preserved(self):
        ext_new, ratio = UTILS.redistribute_by_freq(self.ext, self.freq, self.response_ids, self.valid)
        self.assertAlmostEqual(ratio, 1.0, places=5)
        self.assertAlmostEqual(
            ext_new[self.valid].sum().item(), self.ext[self.valid].sum().item(), places=5
        )

    def test_non_eligible_positions_stay_zero(self):
        ext_new, _ = UTILS.redistribute_by_freq(self.ext, self.freq, self.response_ids, self.valid)
        self.assertEqual(ext_new[0, 1].item(), 0.0)  # was zero, stays zero
        self.assertEqual(ext_new[1, 2].item(), 0.0)  # invalid pad position stays zero

    def test_low_frequency_gains_relative_to_high_frequency(self):
        ext_new, _ = UTILS.redistribute_by_freq(self.ext, self.freq, self.response_ids, self.valid)
        # identical raw ext (0.5) on ids 1 (n=4000) and 2 (n=40): the low-frequency
        # position must end up with more redistributed budget than the high-frequency one.
        self.assertGreater(ext_new[0, 2].item(), ext_new[0, 0].item())
        self.assertGreater(ext_new[1, 0].item(), ext_new[0, 0].item())

    def test_zero_budget_batch_is_neutral(self):
        ext = torch.zeros(2, 3)
        ext_new, _ = UTILS.redistribute_by_freq(ext, self.freq, self.response_ids, self.valid)
        self.assertTrue(torch.all(ext_new == 0).item())


class ConfigValidationTests(unittest.TestCase):
    def _base_kwargs(self):
        return dict(teacher_always_on=True, teacher_image_key="bbox_images",
                    teacher_neg_image_key="neg_bbox_images",
                    vd_gamma=1.0, vd_dual_signal=True, vd_targeted=True, vd_prior_file="/tmp/p.json")

    def test_freq_decay_requires_targeted(self):
        kwargs = self._base_kwargs()
        kwargs.update(vd_targeted=False, vd_prior_file=None, vd_freq_decay=True, vd_freq_file="/tmp/f.json")
        with self.assertRaises(ValueError):
            SelfDistillationConfig(**kwargs)

    def test_freq_decay_requires_file(self):
        kwargs = self._base_kwargs()
        kwargs.update(vd_freq_decay=True, vd_freq_file=None)
        with self.assertRaises(ValueError):
            SelfDistillationConfig(**kwargs)

    def test_freq_decay_with_file_and_targeted_is_accepted(self):
        kwargs = self._base_kwargs()
        kwargs.update(vd_freq_decay=True, vd_freq_file="/tmp/f.json")
        cfg = SelfDistillationConfig(**kwargs)
        self.assertTrue(cfg.vd_freq_decay)

    def test_flag_defaults_to_off(self):
        cfg = SelfDistillationConfig(teacher_always_on=True, teacher_image_key="bbox_images",
                                     teacher_neg_image_key="neg_bbox_images",
                                     vd_gamma=1.0, vd_dual_signal=True, vd_targeted=True,
                                     vd_prior_file="/tmp/p.json")
        self.assertFalse(cfg.vd_freq_decay)
        self.assertIsNone(cfg.vd_freq_file)


if __name__ == "__main__":
    unittest.main()
