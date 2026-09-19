"""Tier A (frozen): validate the vendored native core (kalay_rs).

Leg 1: the Rust perfect-hash 7-card evaluator vs an INDEPENDENT slow Python
       reference evaluator (5-of-7 via itertools, straightforward rank logic)
       on random deals - exact agreement required.
Leg 2: table equities vs exact full-board enumeration (C(48,5) boards through
       the validated evaluator) on key matchups, within MC precision.
Leg 3: the CFR+ Nash solve: exploitability(nash) ~ 0, zero-sum payoffs,
       directional sanity (AA always, trash never).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from kalay.cards import native


# ---------------------------------------------------------------- leg 1
def _ref_eval5(cards: tuple[int, ...]) -> int:
    """Independent 5-card evaluator. Returns comparable rank (higher = stronger)."""
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    suits = [c & 3 for c in cards]
    flush = len(set(suits)) == 1
    straight, high = _straight(ranks)
    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    if flush and straight:
        return (8, high)
    if groups[0][1] == 4:
        return (7, groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, groups[0][0], groups[1][0])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, high)
    if groups[0][1] == 3:
        return (3, groups[0][0], groups[1][0], groups[2][0])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return (2, groups[0][0], groups[1][0], groups[2][0])
    if groups[0][1] == 2:
        return (1, groups[0][0], groups[1][0], groups[2][0], groups[3][0])
    return (0, *ranks)


def _straight(ranks: list[int]):
    if ranks == [12, 3, 2, 1, 0]:  # A-5 wheel
        return True, 3
    for i in range(4):
        if ranks[i] - 1 != ranks[i + 1]:
            return False, -1
    return True, ranks[0]


def _ref_eval7(cards: list[int]) -> tuple:
    return max(_ref_eval5(c) for c in itertools.combinations(cards, 5))


def _cat(rust_rank: int) -> int:
    """Map the Rust absolute rank (1..7462) to a category 0..8 (higher=stronger).

    Boundaries mirror rs/src/eval.rs HandRank::category() exactly.
    """
    if rust_rank <= 10:
        return 8       # straight flush (incl. royal)
    if rust_rank <= 166:
        return 7       # quads
    if rust_rank <= 322:
        return 6       # full house
    if rust_rank <= 1599:
        return 5       # flush
    if rust_rank <= 1609:
        return 4       # straight
    if rust_rank <= 2467:
        return 3       # trips
    if rust_rank <= 3325:
        return 2       # two pair
    if rust_rank <= 6185:
        return 1       # pair
    return 0           # high card


@pytest.mark.tier_a
def test_rust_evaluator_matches_independent_reference():
    rng = np.random.default_rng(7)
    for _ in range(400):
        cards = rng.choice(52, size=7, replace=False).tolist()
        rust = native.eval_river7(cards)
        ref = _ref_eval7(cards)
        assert _cat(rust) == ref[0], (cards, rust, ref)


@pytest.mark.tier_a
def test_rust_evaluator_orders_correctly():
    rng = np.random.default_rng(11)
    for _ in range(400):
        cards = rng.choice(52, size=14, replace=False).tolist()
        a, b = cards[:7], cards[7:]
        ra, rb = native.eval_river7(a), native.eval_river7(b)
        refa, refb = _ref_eval7(a), _ref_eval7(b)
        assert (ra < rb) == (refa > refb)  # lower Rust = stronger; higher ref = stronger
        assert (ra == rb) == (refa == refb)


# ---------------------------------------------------------------- leg 2
def _exact_equity(hole_a, hole_b) -> float:
    """Exact enumeration over all C(48,5) boards via the validated evaluator."""
    dead = set(hole_a) | set(hole_b)
    rest = [c for c in range(52) if c not in dead]
    score = 0.0
    n = 0
    for board in itertools.combinations(rest, 5):
        ra = native.eval_river7([*hole_a, *board])
        rb = native.eval_river7([*hole_b, *board])
        score += 1.0 if ra < rb else (0.5 if ra == rb else 0.0)
        n += 1
    return score / n


@pytest.mark.tier_a
@pytest.mark.parametrize(
    "name_i,name_j,tol",
    [("AA", "KK", 0.01), ("AKs", "QQ", 0.01), ("22", "AKo", 0.01), ("KQs", "72o", 0.01)],
)
def test_table_equity_vs_exact_enumeration(name_i, name_j, tol):
    names = [native.hand_name(i) for i in range(169)]
    i, j = names.index(name_i), names.index(name_j)
    cards_i = _class_repr_cards(i)
    cards_j = _class_repr_cards(j)
    exact = _exact_equity(cards_i, cards_j)
    table = native.equity_table()[i, j]
    se = math.sqrt(0.25 / 1_000_000)  # table MC precision (1M sims)
    assert abs(table - exact) <= tol + 3 * se, (
        f"{name_i} vs {name_j}: table {table:.4f}, exact {exact:.4f}"
    )


def _class_repr_cards(idx: int) -> list[int]:
    """Representative combo of class idx (matches kalay_rs hand_169 conventions)."""
    name = native.hand_name(idx)
    r1 = "23456789TJQKA".index(name[0])
    r2 = "23456789TJQKA".index(name[1])
    suited = name[2] == "s" if len(name) > 2 else False
    hi, lo = max(r1, r2), min(r1, r2)
    if suited:
        return [(hi << 2), (lo << 2)]
    return [(hi << 2), (lo << 2) | 1]


# ---------------------------------------------------------------- leg 3
@pytest.mark.tier_a
def test_pushfold_nash_solution_quality():
    p, q, value = native.nash_pushfold()
    names = [native.hand_name(i) for i in range(169)]
    nc = native.nashconv(p, q)
    assert nc <= 1e-3, f"CFR+ Nash exploitability {nc} > 1e-3 bb/hand"
    # zero-sum self-play payoffs
    u_btn, u_bb = native.pushfold_solver().payoffs(
        np.ascontiguousarray(p, dtype=np.float32),
        np.ascontiguousarray(q, dtype=np.float32),
    )
    assert u_btn == pytest.approx(value, abs=1e-4)
    assert u_btn + u_bb == pytest.approx(0.0, abs=1e-5)
    # directional anchors
    assert p[names.index("AA")] > 0.99
    assert q[names.index("AA")] > 0.99
    assert p[names.index("72o")] < 0.05
    assert q[names.index("32o")] < 0.05


@pytest.mark.tier_a
def test_exploitability_of_uniform_is_large_and_finite():
    p = np.full(169, 0.5, dtype=np.float32)
    q = np.full(169, 0.5, dtype=np.float32)
    nc = native.nashconv(p, q)
    assert math.isfinite(nc) and nc > 0.1
