"""Hand-locked S4 suit canonicalization (SPEC section 3).

Card = (rank, suit) with raw suit ids 0..3. The canonical form is a full invariant
of the suit-permutation orbit (verified by brute force in tests/tier_a/test_canonical):
greedy hand lock + first-appearance relabeling of new suits + 0<->1 board swap for
pocket pairs. Prefix-stable across streets, so emitted tokens never change (append-only).
"""
from __future__ import annotations

from collections.abc import Iterable

Card = tuple[int, int]


def canonical_form(
    hand: Iterable[Card], board: Iterable[Card], *, pair_collapse: bool = True
) -> tuple[Card, ...]:
    """Canonical (hand + board) token sequence; board order preserved."""
    h = sorted(hand, key=lambda c: (-c[0], -c[1]))
    suited = h[0][1] == h[1][1]
    pair = h[0][0] == h[1][0]
    mapping: dict[int, int] = {h[0][1]: 0}
    if not suited:
        mapping[h[1][1]] = 1
    hand_tokens = ((h[0][0], 0), (h[1][0], 0 if suited else 1))

    nxt = 1 if suited else 2  # smallest unused suit index (SPEC S4: suited -> 1,2,3)
    board_tokens: list[Card] = []
    for r, s in board:
        if s not in mapping:
            mapping[s] = nxt
            nxt += 1
        board_tokens.append((r, mapping[s]))

    if pair and pair_collapse:
        swapped = [(r, 1 - s if s in (0, 1) else s) for r, s in board_tokens]
        if swapped < board_tokens:
            board_tokens = swapped
    return hand_tokens + tuple(board_tokens)


class StreamingCanonicalizer:
    """Street-by-street canonicalization with a locked hand.

    The hand fixes the suit mapping once. Each new board card is canonicalized with
    the persistent mapping; for pocket pairs the deferred 0<->1 swap is re-derived
    from the full board-so-far each time (prefix tokens never change — proven by the
    prefix-stability test).
    """

    def __init__(self, hand: Iterable[Card]) -> None:
        self._hand = sorted(hand, key=lambda c: (-c[0], -c[1]))
        self._suited = self._hand[0][1] == self._hand[1][1]
        self._pair = self._hand[0][0] == self._hand[1][0]
        self._raw_board: list[Card] = []

    @property
    def hand_tokens(self) -> tuple[Card, ...]:
        h = self._hand
        return ((h[0][0], 0), (h[1][0], 0 if self._suited else 1))

    def add_board_card(self, card: Card) -> Card:
        """Append a board card, return its canonical token (past tokens unchanged)."""
        self._raw_board.append(card)
        return canonical_form(self._hand, self._raw_board)[-1]

    def tokens(self) -> tuple[Card, ...]:
        return canonical_form(self._hand, self._raw_board)
