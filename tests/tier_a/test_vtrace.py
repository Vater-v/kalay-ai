"""Tier A (frozen): V-trace recursion, padding isolation, NaN safety (SPEC 5.3)."""
from __future__ import annotations

import pytest
import torch

from kalay.core.vtrace import bootstrap_next_value, decision_mask, vtrace


def run_vtrace(v_preds, r_reg, lengths, rho=None, c=None, gamma=1.0):
    T, B = v_preds.shape
    rho = torch.ones(T, B) if rho is None else rho
    c = torch.ones(T, B) if c is None else c
    return vtrace(v_preds, r_reg, rho, c, lengths, gamma)


@pytest.mark.tier_a
def test_hand_computed_values():
    # T=3, single hand with L=2; gamma=1, rho=c=1 -> v = sum of future r_reg
    v_preds = torch.tensor([[0.4], [0.6], [999.0]])
    r_reg = torch.tensor([[0.0], [1.0], [0.0]])
    lengths = torch.tensor([2])
    v_targets = run_vtrace(v_preds, r_reg, lengths)
    assert v_targets[0, 0].item() == pytest.approx(1.0)  # r0 + r1
    assert v_targets[1, 0].item() == pytest.approx(1.0)  # terminal: r1
    # td for the policy: reward + gamma*v_next - v  (r0 = 0 here)
    next_v = bootstrap_next_value(v_targets, lengths)
    td = r_reg + next_v - v_preds
    assert td[0, 0].item() == pytest.approx(0.0 + 1.0 - 0.4)
    assert td[1, 0].item() == pytest.approx(1.0 - 0.6)  # no leak from the padding row


@pytest.mark.tier_a
def test_no_padding_leak_finite_case():
    # padding rows carry garbage but finite values: nothing may leak into real rows
    v_preds = torch.tensor([[0.0], [0.0], [123.0]])
    r_reg = torch.tensor([[0.5], [0.5], [-7.0]])
    lengths = torch.tensor([2])
    v_targets = run_vtrace(v_preds, r_reg, lengths)
    assert v_targets[1, 0].item() == pytest.approx(0.5)
    next_v = bootstrap_next_value(v_targets, lengths)
    assert next_v[1, 0].item() == pytest.approx(0.0)


@pytest.mark.tier_a
def test_nan_in_padding_cannot_poison_real_rows():
    # NaN * 0 == NaN, so masks must be selects (torch.where), not multiplications
    v_preds = torch.tensor([[0.4], [0.6], [float("nan")]])
    r_reg = torch.tensor([[0.0], [1.0], [float("nan")]])
    lengths = torch.tensor([2])
    v_targets = run_vtrace(v_preds, r_reg, lengths)
    assert torch.isfinite(v_targets[:2, 0]).all()
    next_v = bootstrap_next_value(v_targets, lengths)
    assert torch.isfinite(next_v[:2, 0]).all()


@pytest.mark.tier_a
def test_truncation_coefficients():
    # rho=c=0 zeroes the TD error: v_target == v_preds on real rows
    v_preds = torch.tensor([[0.3], [0.7]])
    r_reg = torch.tensor([[2.0], [2.0]])
    lengths = torch.tensor([2])
    v_targets = run_vtrace(v_preds, r_reg, lengths,
                           rho=torch.zeros(2, 1), c=torch.zeros(2, 1))
    assert v_targets[0, 0].item() == pytest.approx(0.3)
    assert v_targets[1, 0].item() == pytest.approx(0.7)


@pytest.mark.tier_a
def test_decision_mask_shape_and_values():
    lengths = torch.tensor([3, 1])
    m = decision_mask(lengths, T=3)
    assert m.tolist() == [[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
