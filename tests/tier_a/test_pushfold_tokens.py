"""Tier A (frozen): push-fold token adapter invariants (DESIGN_NET v3)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kalay.games.pushfold_tokens import (
    bb_slice,
    build_batch,
    btn_slice,
    canonical_hole,
)


def _card(rank: int, suit: int) -> int:
    return (rank << 2) | suit


A, K, Q = 12, 11, 10


@pytest.mark.tier_a
def test_canonical_hole_order_and_suitedness():
    # hand-lock: higher card -> suit 0; lower -> 0 (suited) / 1 (offsuit)
    assert canonical_hole((_card(K, 0), _card(A, 1))) == ((A << 2), (K << 2) | 1)
    assert canonical_hole((_card(A, 2), _card(K, 2))) == ((A << 2), (K << 2))  # suited
    assert canonical_hole((_card(A, 3), _card(K, 0))) == ((A << 2), (K << 2) | 1)
    # pairs: equal ranks -> suits 0 and 1 regardless of raw suits
    assert canonical_hole((_card(Q, 3), _card(Q, 1))) == ((Q << 2), (Q << 2) | 1)


@pytest.mark.tier_a
def test_slices_structure():
    b = btn_slice((_card(A, 0), _card(K, 1)))
    assert b["kind"] == [4, 1, 2, 2]  # DECISION, META, CARD, CARD
    assert b["pos"][2] == b["pos"][3]  # hole tokens share the positional index
    assert b["card"][2] > b["card"][3]  # rank-descending
    q = bb_slice((_card(Q, 0), _card(Q, 1)))
    assert q["kind"][-1] == 3  # ACTION
    assert q["act"][-1] == 2   # Bet/Raise
    # push numerics: bet 9.5, pot 11, to_call 9
    v = q["v"][-1]
    assert v[0] == pytest.approx(9.5 / 11.0)
    assert v[1] == pytest.approx(9.5 / 10.0)
    assert v[2] == pytest.approx(math.log(1.0 + 11.0 / 10.0))
    assert v[3] == pytest.approx(9.0 / 11.0)


@pytest.mark.tier_a
def test_infoset_completeness_distinctness():
    """Distinct strategic situations -> distinct token slices; canonical hands
    (same ranks+suitedness, different raw suits) -> identical slices."""
    a = btn_slice((_card(A, 1), _card(K, 0)))   # A(h) K(s)
    b = btn_slice((_card(A, 2), _card(K, 3)))   # A(d) K(c)
    assert a == b, "isomorphic hands must produce identical token slices"
    c = btn_slice((_card(A, 0), _card(K, 0)))   # suited
    assert a != c
    d = btn_slice((_card(A, 0), _card(Q, 1)))
    assert a != d
    assert btn_slice((_card(A, 0), _card(K, 1))) != bb_slice((_card(A, 0), _card(K, 1)))


@pytest.mark.tier_a
def test_build_batch_shapes_and_pads():
    slices = [btn_slice((_card(A, 0), _card(K, 1))),
              bb_slice((_card(Q, 2), _card(Q, 3)))]
    batch = build_batch(slices, hero_seat=[0, 1], opp_seat=[1, 0], max_len=8)
    assert batch.kind.shape == (2, 8)
    assert bool(batch.pad[0, :4].all()) and not bool(batch.pad[0, 4:].any())
    assert bool(batch.pad[1, :5].all()) and not bool(batch.pad[1, 5:].any())
    assert int(batch.kind[:, 0].eq(4).all())  # DECISION first everywhere
    # opponent action carries the opponent seat
    assert batch.seat[1, 4] == 0 and batch.seat[1, 1] == 1


@pytest.mark.tier_a
def test_transformer_consumes_token_batch():
    from kalay.core.nets_transformer import PolicyTransformerV3, TokenConfigV3

    torch.manual_seed(0)
    cfg = TokenConfigV3()
    pol = PolicyTransformerV3(cfg)
    batch = build_batch(
        [btn_slice((_card(A, 0), _card(K, 1)))] * 3,
        hero_seat=[0, 0, 0], opp_seat=[1, 1, 1],
    )
    logits = pol(batch)
    assert logits.shape == (3, cfg.n_actions)
    assert torch.isfinite(logits).all()
