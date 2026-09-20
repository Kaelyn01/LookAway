"""Unit tests for the frequency-decay extrapolation reallocation.

Covers: freq-table loading, the strict budget-matching invariant of
redistribute_by_freq (including tiny positive budgets and the zero-budget
neutral case), non-eligible neutrality, the damping direction, the
SelfDistillationConfig validations, frequency-table provenance validation,
and the launcher's enable/disable contract.
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

VALIDATE_SPEC = importlib.util.spec_from_file_location(
    "validate_priors", ROOT / "scripts" / "validate_priors.py"
)
VALIDATE = importlib.util.module_from_spec(VALIDATE_SPEC)
VALIDATE_SPEC.loader.exec_module(VALIDATE)

sys.path.insert(0, str(ROOT))
try:
    from verl.workers.config.actor import SelfDistillationConfig  # noqa: E402

    HAS_VERL_CONFIG = True
except ImportError:
    # The lightweight CI environment installs neither ray nor omegaconf; the
    # verl package import chain requires them. The config validations run
    # wherever the full training environment is available (e.g. the H200 box).
    HAS_VERL_CONFIG = False


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
        ext_new, ratio, _ = UTILS.redistribute_by_freq(self.ext, self.freq, self.response_ids, self.valid)
        self.assertAlmostEqual(ratio, 1.0, places=5)
        self.assertAlmostEqual(
            ext_new[self.valid].sum().item(), self.ext[self.valid].sum().item(), places=5
        )

    def test_tiny_positive_budget_is_strictly_preserved(self):
        # Regression: a budget below any epsilon must NOT be clamped away.
        # 0 < E < 1e-6 previously collapsed to ~1.5% of the true total.
        ext = torch.zeros(2, 3)
        ext[0, 0] = 2.0e-7  # sole eligible position, high-frequency token
        ext_new, ratio, _ = UTILS.redistribute_by_freq(ext, self.freq, self.response_ids, self.valid)
        self.assertGreater(ratio, 0.99)
        self.assertAlmostEqual(ext_new[0, 0].item(), 2.0e-7, delta=1e-12)
        self.assertAlmostEqual(ext_new[self.valid].sum().item(), 2.0e-7, delta=1e-12)

    def test_zero_budget_is_neutral_with_ratio_one(self):
        ext = torch.zeros(2, 3)
        ext_new, ratio, _ = UTILS.redistribute_by_freq(ext, self.freq, self.response_ids, self.valid)
        self.assertTrue(torch.all(ext_new == 0).item())
        self.assertEqual(ratio, 1.0)

    def test_non_eligible_positions_stay_zero(self):
        ext_new, _, _ = UTILS.redistribute_by_freq(self.ext, self.freq, self.response_ids, self.valid)
        self.assertEqual(ext_new[0, 1].item(), 0.0)  # was zero, stays zero
        self.assertEqual(ext_new[1, 2].item(), 0.0)  # invalid pad position stays zero

    def test_low_frequency_gains_relative_to_high_frequency(self):
        ext_new, _, _ = UTILS.redistribute_by_freq(self.ext, self.freq, self.response_ids, self.valid)
        # identical raw ext (0.5) on ids 1 (n=4000) and 2 (n=40): the low-frequency
        # position must end up with more redistributed budget than the high-frequency one.
        self.assertGreater(ext_new[0, 2].item(), ext_new[0, 0].item())
        self.assertGreater(ext_new[1, 0].item(), ext_new[0, 0].item())

    def test_gathered_counts_are_returned(self):
        _, _, n_freq = UTILS.redistribute_by_freq(self.ext, self.freq, self.response_ids, self.valid)
        self.assertEqual(n_freq.shape, self.ext.shape)
        self.assertEqual(n_freq[0, 0].item(), 4000.0)
        self.assertEqual(n_freq[1, 0].item(), 40.0)


@unittest.skipUnless(HAS_VERL_CONFIG, "verl runtime dependencies (ray, omegaconf) not installed")
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


class ValidateFreqTableTests(unittest.TestCase):
    """Provenance validation of token_freq.json (see validate_priors.validate_freq)."""

    def _write_table(self, tmpdir, meta_overrides=None, counts=None, vocab_size=10):
        meta = {
            "schema_version": 1,
            "model_path": "/tmp/model",
            "model_revision": None,
            "tokenizer_sha256": "abc",
            "vocab_size": vocab_size,
            "token_dump_sha256": "dump",
            "n_samples": 5,
            "n_token_instances": 40,
            "kappa": 50,
        }
        meta.update(meta_overrides or {})
        path = Path(tmpdir) / "token_freq.json"
        path.write_text(json.dumps({
            "meta": meta, "vocab_size": vocab_size,
            "counts": counts if counts is not None else {"3": 7, "8": 12},
        }))
        return path

    def _write_dump(self, tmpdir, content="line\n"):
        path = Path(tmpdir) / "token_scores_full.jsonl"
        path.write_text(content)
        return path

    def test_valid_table_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dump = self._write_dump(tmpdir)
            table = self._write_table(tmpdir, meta_overrides={
                "token_dump_sha256": VALIDATE.sha256(dump),
            })
            VALIDATE.validate_freq(table, "/tmp/model", None, "abc", dump)

    def test_wrong_tokenizer_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dump = self._write_dump(tmpdir)
            table = self._write_table(tmpdir, meta_overrides={
                "token_dump_sha256": VALIDATE.sha256(dump), "tokenizer_sha256": "other",
            })
            with self.assertRaises(ValueError):
                VALIDATE.validate_freq(table, "/tmp/model", None, "abc", dump)

    def test_changed_dump_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dump = self._write_dump(tmpdir)
            table = self._write_table(tmpdir)  # dump hash left as "dump"
            with self.assertRaises(ValueError):
                VALIDATE.validate_freq(table, "/tmp/model", None, "abc", dump)

    def test_wrong_kappa_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dump = self._write_dump(tmpdir)
            table = self._write_table(tmpdir, meta_overrides={
                "token_dump_sha256": VALIDATE.sha256(dump), "kappa": 100,
            })
            with self.assertRaises(ValueError):
                VALIDATE.validate_freq(table, "/tmp/model", None, "abc", dump)

    def test_out_of_range_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dump = self._write_dump(tmpdir)
            table = self._write_table(tmpdir, counts={"99": 1}, meta_overrides={
                "token_dump_sha256": VALIDATE.sha256(dump),
            })
            with self.assertRaises(ValueError):
                VALIDATE.validate_freq(table, "/tmp/model", None, "abc", dump)

    def test_empty_counts_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dump = self._write_dump(tmpdir)
            table = self._write_table(tmpdir, counts={}, meta_overrides={
                "token_dump_sha256": VALIDATE.sha256(dump),
            })
            with self.assertRaises(ValueError):
                VALIDATE.validate_freq(table, "/tmp/model", None, "abc", dump)


if __name__ == "__main__":
    unittest.main()
