"""Small tensor helpers for LookAway token weighting."""

import json
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


def load_freq_table(freq_file: str) -> torch.Tensor:
    """Frequency decay: vocab-indexed corpus token counts from the frozen dump.

    The JSON holds {"vocab_size": int, "counts": {token_id: n}} as produced by
    scripts/build_freq_table.py over the training-side generation dump.
    """
    with open(freq_file, encoding="utf-8") as stream:
        blob = json.load(stream)
    counts = torch.zeros(int(blob["vocab_size"]), dtype=torch.float32)
    for key, value in blob["counts"].items():
        counts[int(key)] = float(value)
    return counts


def redistribute_by_freq(ext: torch.Tensor, freq_counts: torch.Tensor, response_ids: torch.Tensor,
                         valid_mask: torch.Tensor, kappa: float = 50.0) -> "tuple[torch.Tensor, float, torch.Tensor]":
    """Frequency decay: strictly reallocate the extrapolation budget.

    e'_t = E * e_t * a_t / sum_j e_j * a_j with a_t = 1/sqrt(n_t + kappa) over
    the normalization scope (valid_mask). The budget total E, the base weights,
    and non-eligible positions are unchanged -- including tiny positive budgets
    (no clamping: the scale uses the true sums, computed in float32). A
    zero-budget input returns all zeros (a_t > 0 wherever e_t > 0, so the
    weighted sum cannot vanish while the total is positive). kappa defaults to
    50 to match the empirical-Bayes pseudo-count used when building the token
    priors. Returns (ext_new, budget_ratio, n_freq): budget_ratio == 1.0 means
    the total was preserved exactly (1.0 is also returned for zero-budget
    inputs, which are neutral by definition; callers distinguish them via
    whether the total is zero); n_freq is the gathered per-position count so
    callers need not gather it again.
    """
    n_freq = freq_counts.to(ext.device)[response_ids.long()]
    a_fd = torch.rsqrt(n_freq + kappa)
    e_w = ext * a_fd
    e_total = ext[valid_mask].float().sum().item()
    if e_total <= 0.0:
        return torch.zeros_like(ext), 1.0, n_freq
    e_weighted = e_w[valid_mask].float().sum().item()
    ext_new = e_w * (e_total / e_weighted)
    budget_ratio = ext_new[valid_mask].float().sum().item() / e_total
    return ext_new, budget_ratio, n_freq
