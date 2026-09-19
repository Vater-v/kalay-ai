"""Canonical NeuRD loss with the per-action regularizer (SPEC 5.4-5.5, fix S1).

The regularizer must enter the *force* through an action-dependent term
tau*(log pi_ref(a) - log pi(a)) inside q_vr — a per-infoset KL scalar cancels in
q - V at critic convergence and leaves the policy unregularized (audit of v6.0,
empirically: pure best response instead of the QRE).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_NEG_INF = -1e9


def masked_softmax(raw_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(legal_mask == 1.0, raw_logits, torch.full_like(raw_logits, _NEG_INF))
    return F.softmax(masked, dim=-1)


@dataclass
class NeuRDDiagnostics:
    force_gated_fraction: float  # share of legal-action slots whose force was zeroed by the Z gate
    rail_fraction: float         # share of slots where |adv| hit the A_MAX clip


def neurd_policy_loss(
    raw_logits: torch.Tensor,      # [T, B, K] with grad
    legal_mask: torch.Tensor,      # [T, B, K]
    pi: torch.Tensor,              # [T, B, K] current probs (grad ok; used detached)
    pi_ref: torch.Tensor,          # [T, B, K] anchor probs
    v_preds: torch.Tensor,         # [T, B] value predictions (detached)
    td: torch.Tensor,              # [T, B] TD residual with PLAIN reward (SPEC S5)
    action_taken: torch.Tensor,    # [T, B]
    behavior_prob: torch.Tensor,   # [T, B]
    dec_mask: torch.Tensor,        # [T, B] float
    num_valid: torch.Tensor,       # scalar >= 1
    tau: float,
    z_thresh: float,
    a_max: float,
    mu_floor: float,
) -> tuple[torch.Tensor, NeuRDDiagnostics]:
    T, B, K = raw_logits.shape
    act_oh = F.one_hot(action_taken, num_classes=K).to(raw_logits.dtype)

    log_pi = torch.log(pi.clamp(min=1e-9))
    log_ref = torch.log(pi_ref.clamp(min=1e-9))

    # q_vr: per-action regularizer + TD (importance-scaled) at the taken action
    q_vr = v_preds.unsqueeze(-1) + tau * (log_ref - log_pi)
    mu = behavior_prob.clamp(min=mu_floor)
    q_vr = q_vr + act_oh * (td / mu).unsqueeze(-1)
    q_vr = q_vr * legal_mask

    adv = q_vr - (pi.detach() * q_vr).sum(dim=-1, keepdim=True)
    rail = (adv.abs() >= a_max).float()
    adv = adv.clamp(min=-a_max, max=a_max).detach()

    # masked centering (OpenSpiel #1156 fix): differentiable mean over legal logits
    legal_count = legal_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean_logit = (raw_logits * legal_mask).sum(dim=-1, keepdim=True) / legal_count
    z_centered = raw_logits - mean_logit

    can_decrease = (z_centered.detach() > -z_thresh).to(raw_logits.dtype)
    can_increase = (z_centered.detach() < z_thresh).to(raw_logits.dtype)
    force = can_decrease * torch.clamp(adv, max=0.0) + can_increase * torch.clamp(adv, min=0.0)

    gated = ((adv != 0) & (force == 0)).float()
    slot_mask = legal_mask * dec_mask.unsqueeze(-1)
    n_slots = slot_mask.sum().clamp(min=1.0)
    diags = NeuRDDiagnostics(
        force_gated_fraction=float((gated * slot_mask).sum() / n_slots),
        rail_fraction=float((rail * slot_mask).sum() / n_slots),
    )
    loss = -(slot_mask * z_centered * force).sum() / num_valid
    return loss, diags
