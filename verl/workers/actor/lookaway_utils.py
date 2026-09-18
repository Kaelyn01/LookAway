"""Small tensor helpers for LookAway token weighting."""

import math

import torch
import torch.nn.functional as F


def token_distillation_divergence(
    student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Match the per-token KL/JSD direction used by the distillation loss."""
    if alpha == 0.0:
        return F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(-1)
    if alpha == 1.0:
        return F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True).sum(-1)

    mixture = torch.logsumexp(
        torch.stack(
            [student_log_probs + math.log(1.0 - alpha), teacher_log_probs + math.log(alpha)]
        ),
        dim=0,
    )
    student_kl = F.kl_div(mixture, student_log_probs, reduction="none", log_target=True).sum(-1)
    teacher_kl = F.kl_div(mixture, teacher_log_probs, reduction="none", log_target=True).sum(-1)
    return torch.lerp(student_kl, teacher_kl, alpha)


def normalize_vd_weights(raw_weights: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean-normalize available counterfactual weights and leave unavailable rows neutral."""
    valid_mask = valid_mask.bool()
    valid_count = valid_mask.sum().to(raw_weights.dtype)
    valid_sum = raw_weights.masked_select(valid_mask).sum().clamp(min=1.0)
    normalized = raw_weights * (valid_count / valid_sum)
    return torch.where(valid_mask, normalized, torch.ones_like(raw_weights))
