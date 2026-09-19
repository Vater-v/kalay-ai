"""Tier A (frozen): [DECISION]+bidirectional transformer invariants (v3)."""
from __future__ import annotations

import pytest
import torch

from kalay.core.nets_transformer import (
    ACTION,
    CARD,
    DECISION,
    META,
    PolicyTransformerV3,
    TokenConfigV3,
    TokenStreamBatch,
    ValueTransformerV3,
)

KINDS = [DECISION, META, CARD, CARD, CARD, CARD, CARD, ACTION, ACTION, ACTION]


def _cfg(seed=0):
    torch.manual_seed(seed)
    return TokenConfigV3()


def _nets(cfg):
    torch.manual_seed(1)
    return PolicyTransformerV3(cfg), ValueTransformerV3(cfg)


def _batch(num=2, mutate=None, pad_extra=0):
    L0 = len(KINDS)
    g = torch.Generator().manual_seed(2)
    kind = torch.tensor(KINDS).repeat(num, 1)
    card = torch.zeros(num, L0, dtype=torch.long)
    act = torch.zeros(num, L0, dtype=torch.long)
    for j, k in enumerate(KINDS):
        if k == CARD:
            card[:, j] = torch.randint(1, 52, (num,), generator=g)
        elif k == ACTION:
            act[:, j] = torch.randint(0, 3, (num,), generator=g)
    round_id = torch.zeros(num, L0, dtype=torch.long)
    seat = torch.randint(0, 3, (num, L0), generator=g)
    pos_index = torch.arange(L0).repeat(num, 1)
    v_num = torch.randn(num, L0, 4, generator=g).abs() * 0.3
    pad = torch.ones(num, L0, dtype=torch.bool)
    if pad_extra:  # PAD block appended uniformly to every field
        kind = torch.cat([kind, torch.zeros(num, pad_extra, dtype=torch.long)], 1)
        card = torch.cat([card, torch.zeros(num, pad_extra, dtype=torch.long)], 1)
        act = torch.cat([act, torch.zeros(num, pad_extra, dtype=torch.long)], 1)
        round_id = torch.cat([round_id, torch.zeros(num, pad_extra, dtype=torch.long)], 1)
        seat = torch.cat([seat, torch.zeros(num, pad_extra, dtype=torch.long)], 1)
        pos_index = torch.cat([pos_index, torch.zeros(num, pad_extra, dtype=torch.long)], 1)
        v_num = torch.cat([v_num, torch.zeros(num, pad_extra, 4)], 1)
        pad = torch.cat([pad, torch.zeros(num, pad_extra, dtype=torch.bool)], 1)
    b = TokenStreamBatch(kind, card, act, round_id, seat, pos_index, v_num, pad)
    if mutate:
        mutate(b)
    return b


@pytest.mark.tier_a
def test_shapes_determinism_and_gradient_flow():
    cfg = _cfg()
    pol, val = _nets(cfg)
    b = _batch()
    l1, v1 = pol(b), val(b)
    assert l1.shape == (2, cfg.n_actions) and v1.shape == (2,)
    assert torch.equal(l1, pol(b)) and torch.equal(v1, val(b))
    l1.sum().backward()
    assert pol.trunk.backbone.numeric.proj.weight.grad is not None
    assert torch.isfinite(l1).all()


@pytest.mark.tier_a
def test_bidirectional_perturbations_everywhere_matter():
    """Perturbing ANY context token - even a 'later' action - must change the
    readout: full bidirectional graph inside the decision slice."""
    cfg = _cfg()
    pol, _ = _nets(cfg)

    def perturb_late_action(b):
        b.act_type[:, 8] = (b.act_type[:, 8] + 1) % 3

    def perturb_early_card(b):
        b.card[:, 2] = (b.card[:, 2] + 7) % 52 + 1

    base = pol(_batch())
    assert not torch.allclose(base, pol(_batch(mutate=perturb_late_action)), atol=1e-5)
    assert not torch.allclose(base, pol(_batch(mutate=perturb_early_card)), atol=1e-5)


@pytest.mark.tier_a
def test_padding_after_decision_is_invisible():
    cfg = _cfg()
    pol, val = _nets(cfg)
    b1 = _batch()
    b2 = _batch(pad_extra=5)
    torch.testing.assert_close(pol(b1), pol(b2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(val(b1), val(b2), atol=1e-5, rtol=1e-5)


@pytest.mark.tier_a
def test_hole_card_symmetry_shared_position():
    cfg = _cfg()
    pol, val = _nets(cfg)

    def share_pos(b):
        b.pos_index[:, 3] = b.pos_index[:, 2]
        b.seat[:, 3] = b.seat[:, 2]      # same owner
        b.v_num[:, 3] = b.v_num[:, 2]    # cards carry no numerics

    def swap_hole(b):
        c1, c2 = b.card[:, 2].clone(), b.card[:, 3].clone()
        b.card[:, 2], b.card[:, 3] = c2, c1

    b1 = _batch(mutate=share_pos)
    b2 = _batch(mutate=lambda b: (share_pos(b), swap_hole(b)))
    torch.testing.assert_close(pol(b1), pol(b2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(val(b1), val(b2), atol=1e-5, rtol=1e-5)


@pytest.mark.tier_a
def test_meta_numerics_affect_output():
    cfg = _cfg()
    pol, _ = _nets(cfg)

    def scale_meta(b):
        b.v_num[:, 0] *= 3.0

    assert not torch.allclose(pol(_batch()), pol(_batch(mutate=scale_meta)), atol=1e-6)
