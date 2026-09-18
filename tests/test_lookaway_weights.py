import importlib.util
import unittest
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "verl/workers/actor/lookaway_utils.py"
SPEC = importlib.util.spec_from_file_location("lookaway_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NormalizeWeightsTests(unittest.TestCase):
    def test_missing_negative_view_keeps_baseline_weight(self):
        raw = torch.tensor([[2.0, 4.0, 9.0], [7.0, 8.0, 9.0]])
        valid = torch.tensor([[True, True, False], [False, False, False]])
        result = MODULE.normalize_vd_weights(raw, valid)
        self.assertTrue(torch.allclose(result[0, :2].mean(), torch.tensor(1.0)))
        self.assertTrue(torch.equal(result[1], torch.ones(3)))
        self.assertEqual(result[0, 2].item(), 1.0)

    def test_no_counterfactual_tokens_returns_neutral_weights(self):
        raw = torch.tensor([[2.0, 4.0]])
        result = MODULE.normalize_vd_weights(raw, torch.zeros_like(raw, dtype=torch.bool))
        self.assertTrue(torch.equal(result, torch.ones_like(raw)))


class TokenDivergenceTests(unittest.TestCase):
    def test_kl_endpoints_match_main_loss_directions(self):
        student = torch.log(torch.tensor([[[0.8, 0.2]]]))
        teacher = torch.log(torch.tensor([[[0.3, 0.7]]]))
        forward = torch.nn.functional.kl_div(student, teacher, reduction="none", log_target=True).sum(-1)
        reverse = torch.nn.functional.kl_div(teacher, student, reduction="none", log_target=True).sum(-1)
        self.assertTrue(torch.allclose(MODULE.token_distillation_divergence(student, teacher, 0.0), forward))
        self.assertTrue(torch.allclose(MODULE.token_distillation_divergence(student, teacher, 1.0), reverse))

    def test_midpoint_is_finite_and_nonnegative(self):
        student = torch.log(torch.tensor([[[0.8, 0.2]]]))
        teacher = torch.log(torch.tensor([[[0.3, 0.7]]]))
        result = MODULE.token_distillation_divergence(student, teacher, 0.5)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue((result >= 0).all())


if __name__ == "__main__":
    unittest.main()
