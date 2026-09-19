"""Off-policy V-trace on [T, B] player-decision tensors (SPEC 5.3).

All continuation masks use torch.where (spec S2): a NaN in a padding row of v_preds
can never propagate into a real row (NaN * 0.0 == NaN, but where() is a select).
"""
from __future__ import annotations

import torch


def importance_coeffs(
    pi_taken: torch.Tensor, behavior_prob: torch.Tensor, rho_clip: float, c_clip: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """rho and c coefficients from taken-action probabilities."""
    ratio = pi_taken / behavior_prob.clamp(min=1e-8)
    rho = ratio.clamp(max=rho_clip)
    c = ratio.clamp(max=c_clip)
    return rho, c


def vtrace(
    v_preds: torch.Tensor,
    r_reg: torch.Tensor,
    rho: torch.Tensor,
    c: torch.Tensor,
    hand_lengths: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Backward recursion; returns v_targets [T, B]."""
    T, B = v_preds.shape
    v_targets = torch.zeros_like(v_preds)
    zeros = torch.zeros(B, dtype=v_preds.dtype, device=v_preds.device)
    t_idx = torch.arange(T, device=v_preds.device).unsqueeze(1).expand(T, B)
    for t in reversed(range(T)):
        cont_next = (t_idx[t] + 1 < hand_lengths)
        if t + 1 < T:
            next_v = torch.where(cont_next, v_preds[t + 1], zeros)
            next_vt = torch.where(cont_next, v_targets[t + 1], zeros)
        else:
            next_v = zeros
            next_vt = zeros
        delta = rho[t] * (r_reg[t] + gamma * next_v - v_preds[t])
        cont_curr = (t_idx[t] < hand_lengths - 1)
        v_targets[t] = v_preds[t] + delta + gamma * c[t] * torch.where(
            cont_curr, next_vt - next_v, zeros
        )
    return v_targets


def bootstrap_next_value(
    v_targets: torch.Tensor, hand_lengths: torch.Tensor
) -> torch.Tensor:
    """v_targets shifted by one, zeroed where there is no next decision (SPEC 5.4 td)."""
    T, B = v_targets.shape
    shifted = torch.cat([v_targets[1:], torch.zeros(1, B, dtype=v_targets.dtype,
                                                    device=v_targets.device)], dim=0)
    t_idx = torch.arange(T, device=v_targets.device).unsqueeze(1).expand(T, B)
    has_next = (t_idx + 1 < hand_lengths.unsqueeze(0))
    return torch.where(has_next, shifted, torch.zeros_like(shifted))


def decision_mask(hand_lengths: torch.Tensor, T: int) -> torch.Tensor:
    """[T, B] float mask of real decisions (including terminal)."""
    t_idx = torch.arange(T).unsqueeze(1).expand(T, hand_lengths.shape[0])
    return (t_idx < hand_lengths.unsqueeze(0)).float()
