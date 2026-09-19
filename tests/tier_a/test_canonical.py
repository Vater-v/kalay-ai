"""Tier A (frozen): S4 canonicalization invariants — SPEC section 3."""
from __future__ import annotations

import itertools
import random

import pytest

from kalay.cards.canonical import StreamingCanonicalizer, canonical_form

RANKS = 4
DECK = [(r, s) for r in range(RANKS) for s in range(4)]
PERMS = [dict(zip(range(4), p)) for p in itertools.permutations(range(4))]


def apply_perm(cards, perm):
    return [(r, perm[s]) for (r, s) in cards]


@pytest.mark.tier_a
def test_orbit_completeness_all_24_perms():
    """canonical(D) == canonical(sigma(D)) for every suit permutation sigma."""
    rng = random.Random(7)
    for _ in range(4000):
        cards = rng.sample(DECK, 7)
        hand, board = frozenset(cards[:2]), tuple(cards[2:])
        base = canonical_form(hand, board)
        for perm in PERMS:
            h2 = frozenset(apply_perm(list(hand), perm))
            b2 = tuple(apply_perm(list(board), perm))
            assert canonical_form(h2, b2) == base


@pytest.mark.tier_a
def test_no_false_merges_exhaustive_flops():
    """Same canonical form => deals are related by a suit permutation."""
    canon_to_deals: dict[tuple, list] = {}
    for hand in itertools.combinations(DECK, 2):
        rest = [c for c in DECK if c not in hand]
        for board in itertools.permutations(rest, 3):
            key = canonical_form(frozenset(hand), board)
            canon_to_deals.setdefault(key, []).append((frozenset(hand), board))
    for group in canon_to_deals.values():
        d0 = group[0]
        for d in group[1:]:
            related = any(
                frozenset(apply_perm(list(d0[0]), p)) == d[0]
                and tuple(apply_perm(list(d0[1]), p)) == d[1]
                for p in PERMS
            )
            assert related


@pytest.mark.tier_a
def test_prefix_stability_across_streets():
    """Append-only: flop tokens never change when turn/river arrive."""
    rng = random.Random(11)
    for _ in range(4000):
        cards = rng.sample(DECK, 7)
        hand, board = frozenset(cards[:2]), list(cards[2:])
        c3 = canonical_form(hand, board[:3])
        c4 = canonical_form(hand, board[:4])
        c5 = canonical_form(hand, board)
        assert c3 == c4[:5] == c5[:5]


@pytest.mark.tier_a
def test_pocket_pair_orbit_collapse():
    """Bug #7 fix: equivalent pair deals collapse; v5 (no collapse) provably splits."""
    hand = frozenset({(12, 0), (12, 1)})  # pocket aces, raw suits 0 and 1
    board_a = ((8, 0), (5, 2), (2, 1))    # one card matches each hole suit
    swap = {0: 1, 1: 0, 2: 2, 3: 3}
    board_b = tuple(apply_perm(list(board_a), swap))
    assert canonical_form(hand, board_a) == canonical_form(hand, board_b)
    # without the collapse the two representations differ (the v5 bug)
    assert canonical_form(hand, board_a, pair_collapse=False) != canonical_form(
        hand, board_b, pair_collapse=False
    )


@pytest.mark.tier_a
def test_suited_hand_uses_three_new_suit_indices():
    """SPEC S4: suited hands leave indices 1,2,3 for new suits; all three can appear."""
    hand = frozenset({(12, 0), (10, 0)})  # suited
    board = ((8, 1), (7, 2), (5, 3), (4, 1), (2, 2))
    form = canonical_form(hand, board)
    used = sorted({s for _, s in form})
    assert used == [0, 1, 2, 3]


@pytest.mark.tier_a
def test_streaming_matches_batch_and_is_append_only():
    rng = random.Random(13)
    for _ in range(2000):
        cards = rng.sample(DECK, 7)
        hand, board = frozenset(cards[:2]), list(cards[2:])
        stream = StreamingCanonicalizer(hand)
        emitted = [stream.add_board_card(c) for c in board]
        assert stream.tokens() == canonical_form(hand, board)
        # every street prefix of the streaming tokens equals the batch canonicalization
        for k in range(1, 6):
            assert tuple(emitted[:k]) == canonical_form(hand, board[:k])[2:]
