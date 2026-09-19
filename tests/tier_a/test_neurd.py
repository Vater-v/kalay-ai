"""Tier A (frozen): NeuRD loss invariants (SPEC 5.4-5.5)."""
from __future__ import annotations

import pytest
import torch

from kalay.core.neurd import masked_softmax, neurd_policy_loss


def build(T=1, B=1, K=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, B, K, generator=g)
    mask = torch.zeros(T, B, K)
    mask[..., :2] = 1.0  # two legal actions
    pi = masked_softmax(logits, mask)
    pi_ref = masked_softmax(torch.randn(T, B, K, generator=g), mask)
    v = torch.zeros(T, B)
    td = torch.zeros(T, B)
    act = torch.zeros(T, B, dtype=torch.int64)
    mu = torch.ones(T, B)
    dec = torch.ones(T, B)
    num_valid = torch.tensor(1.0)
    return logits, mask, pi, pi_ref, v, td, act, mu, dec, num_valid


@pytest.mark.tier_a
def test_masked_centering_survives_illegal_logit_drift():
    # v6.0 bug #2 regression: illegal logits drifted to +30 must not close the Z gate
    logits = torch.tensor([[[0.5, 0.0, 30.0, 30.0]]], requires_grad=True)
    mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    pi = masked_softmax(logits.detach(), mask)
    pi_ref = pi.clone()
    v = torch.zeros(1, 1)
    td = torch.tensor([[-1.0]])  # taken action (index 0) is bad
    act = torch.zeros(1, 1, dtype=torch.int64)
    mu = torch.ones(1, 1)
    dec = torch.ones(1, 1)
    num_valid = torch.tensor(1.0)
    loss, _ = neurd_policy_loss(logits, mask, pi, pi_ref, v, td, act, mu, dec,
                                num_valid, tau=0.1, z_thresh=3.0, a_max=100.0,
                                mu_floor=1e-3)
    loss.backward()
    # negative force on the taken action must produce a nonzero gradient
    assert logits.grad[0, 0, 0] != 0.0


@pytest.mark.tier_a
def test_loss_is_shift_invariant():
    logits, mask, pi, pi_ref, v, td, act, mu, dec, num_valid = build()
    loss_a, _ = neurd_policy_loss(logits, mask, pi, pi_ref, v, td, act, mu, dec,
                                  num_valid, 0.1, 3.0, 100.0, 1e-3)
    shifted = logits + 10.0
    loss_b, _ = neurd_policy_loss(shifted, mask, pi, pi_ref, v, td, act, mu, dec,
                                  num_valid, 0.1, 3.0, 100.0, 1e-3)
    assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-5)


@pytest.mark.tier_a
def test_per_action_regularizer_pulls_toward_anchor():
    # pi drifted from the anchor on action 0: with no TD signal the regularizer alone
    # must push action 0's logit down and action 1's logit up
    logits, mask, pi, pi_ref, v, td, act, mu, dec, num_valid = build()
    pi = torch.tensor([[[0.9, 0.1, 0.0]]])
    pi_ref = torch.tensor([[[0.5, 0.5, 0.0]]])
    logits = torch.tensor([[[0.5, -0.5, 0.0]]], requires_grad=True)
    loss, _ = neurd_policy_loss(logits, mask, pi, pi_ref, v, td, act, mu, dec,
                                num_valid, tau=0.5, z_thresh=3.0, a_max=100.0,
                                mu_floor=1e-3)
    loss.backward()
    # loss = -sum legal * z_centered * force; gradient descent on z = +force direction
    grad = logits.grad[0, 0]
    assert grad[0] > 0  # descending increases loss along z0? no: dL/dz0 = -force0 > 0 means force0 < 0 (push down)
    assert grad[1] < 0  # force1 > 0 (push up)


@pytest.mark.tier_a
def test_z_gate_blocks_forces_at_threshold():
    logits, mask, pi, pi_ref, v, td, act, mu, dec, num_valid = build()
    logits = torch.tensor([[[-10.0, 0.0, 0.0]]])  # taken action far below -Z after centering
    mask = torch.tensor([[[1.0, 1.0, 0.0]]])
    pi = masked_softmax(torch.tensor([[[0.0, 0.0, 0.0]]]), mask)
    pi_ref = pi.clone()
    td = torch.tensor([[-1.0]])  # wants to decrease action 0 further
    logits = logits.detach().requires_grad_(True)
    loss, diags = neurd_policy_loss(logits, mask, pi, pi_ref, v, td, act, mu, dec,
                                    num_valid, 0.1, 3.0, 100.0, 1e-3)
    loss.backward()
    assert logits.grad[0, 0, 0] == 0.0  # gated: cannot decrease below -Z
    assert diags.force_gated_fraction > 0.0


@pytest.mark.tier_a
def test_rail_diagnostic_fires_on_huge_advantage():
    logits, mask, pi, pi_ref, v, td, act, mu, dec, num_valid = build()
    td = torch.tensor([[1000.0]])
    _, diags = neurd_policy_loss(logits, mask, pi, pi_ref, v, td, act, mu, dec,
                                 num_valid, 0.1, 3.0, 100.0, 1e-3)
    assert diags.rail_fraction > 0.0


@pytest.mark.tier_a
def test_masked_softmax_zeros_illegal_and_normalizes():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    probs = masked_softmax(logits, mask)
    assert probs[0, 1] == 0.0
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0, 2] == pytest.approx(probs[0, 0] * float(torch.exp(torch.tensor(2.0))))
