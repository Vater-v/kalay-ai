"""Leduc hold'em per SPEC section 9 (frozen definition).

6-card deck (ranks J/Q/K x 2 suits), one hole card each, antes 1/1.
K=3 actions: 0=fold (legal only facing a bet), 1=check/call, 2=bet/raise
(legal while <2 raises in the round). Round 1 bet=2 (player 0 opens);
public card revealed between rounds (mid-episode chance node); round 2 bet=4
(player 1 opens). Showdown: pair with the board beats high card; equal strength
splits. Observations encode RANKS only - suits are canonicalized away by
construction (hand strength depends only on ranks in Leduc).
Max |net| = 13 chips.
"""
from __future__ import annotations

import numpy as np

from kalay.games.base import GameEnv

_DECK = [(r, s) for r in range(3) for s in range(2)]          # card id -> (rank, suit)
_HOLE_DEALS = [(a, b) for a in range(6) for b in range(6) if a != b]  # 30 ordered deals


class LeducEnv(GameEnv):
    n_players = 2
    n_actions = 3
    obs_dim = 3 + 4 + 2 + 2 + 8 * 4  # hole rank, public(None/J/Q/K), round, player, 8 hist slots

    def __init__(self) -> None:
        self._hole_ids: tuple[int, int] | None = None
        self._public_id: int | None = None
        self._pending_public = False
        self._hist: list[tuple[int, int]] = []
        self._round_start = 0
        self._turn = 0
        self._folded: int | None = None

    def clone(self) -> "LeducEnv":
        env = LeducEnv()
        env._hole_ids = self._hole_ids
        env._public_id = self._public_id
        env._pending_public = self._pending_public
        env._hist = list(self._hist)
        env._round_start = self._round_start
        env._turn = self._turn
        env._folded = self._folded
        return env

    # --- chance: hole deal (leading) and public card (mid-episode) ---
    def is_chance(self) -> bool:
        return self._hole_ids is None or self._pending_public

    def chance_probs(self) -> np.ndarray:
        if self._hole_ids is None:
            return np.full(len(_HOLE_DEALS), 1.0 / len(_HOLE_DEALS))
        return np.full(4, 0.25)  # 6 - 2 hole cards remain

    def step_chance(self, outcome: int) -> None:
        if self._hole_ids is None:
            self._hole_ids = _HOLE_DEALS[outcome]
        else:
            remaining = [c for c in range(6) if c not in self._hole_ids]
            self._public_id = remaining[outcome]
            self._pending_public = False
            self._round_start = len(self._hist)
            self._turn = 1  # round 2 opener (SPEC 9)

    # --- decisions ---
    def current_player(self) -> int:
        if self.is_chance() or self.is_terminal():
            raise RuntimeError("no acting player at this node")
        return self._turn

    def legal_actions_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=np.float32)
        mask[1] = 1.0  # check / call always legal
        if self._facing_bet():
            mask[0] = 1.0  # fold only against a bet (SPEC 9)
        if self._raises() < 2:
            mask[2] = 1.0  # bet / raise
        return mask

    def step(self, action: int) -> None:
        self._hist.append((self.current_player(), action))
        if action == 0:
            self._folded = self._turn
            return
        if self._round_done():
            if self._public_id is None:
                self._pending_public = True
        self._turn = 1 - self._turn

    # --- terminal & rewards ---
    def is_terminal(self) -> bool:
        if self._folded is not None:
            return True
        return self._public_id is not None and not self._pending_public and self._round_done()

    def rewards(self) -> np.ndarray:
        if not self.is_terminal():
            return np.zeros(self.n_players)
        inv = self._investments()
        pot = inv[0] + inv[1]
        rew = np.zeros(self.n_players)
        if self._folded is not None:
            winner = 1 - self._folded
            rew[winner] = pot - inv[winner]
            rew[self._folded] = -inv[self._folded]
            return rew
        r0 = _DECK[self._hole_ids[0]][0]
        r1 = _DECK[self._hole_ids[1]][0]
        rp = _DECK[self._public_id][0]
        s0 = r0 + (3 if r0 == rp else 0)  # pair outranks any high card
        s1 = r1 + (3 if r1 == rp else 0)
        if s0 == s1:
            rew[0] = pot / 2 - inv[0]
            rew[1] = pot / 2 - inv[1]
        else:
            winner = 0 if s0 > s1 else 1
            rew[winner] = pot - inv[winner]
            rew[1 - winner] = -inv[1 - winner]
        return rew

    # --- observation ---
    def obs(self, player: int) -> np.ndarray:
        hole = np.zeros(3, dtype=np.float32)
        hole[_DECK[self._hole_ids[player]][0]] = 1.0
        public = np.zeros(4, dtype=np.float32)
        if self._public_id is not None:
            public[_DECK[self._public_id][0] + 1] = 1.0
        else:
            public[0] = 1.0
        rnd = np.zeros(2, dtype=np.float32)
        rnd[0 if self._public_id is None else 1] = 1.0
        pid = np.zeros(2, dtype=np.float32)
        pid[player] = 1.0
        hist = np.zeros(8 * 4, dtype=np.float32)
        for i, (_, action) in enumerate(self._hist[:8]):
            hist[4 * i + action + 1] = 1.0
        return np.concatenate([hole, public, rnd, pid, hist])

    # --- internals ---
    def _round_hist(self) -> list[tuple[int, int]]:
        return self._hist[self._round_start:]

    def _raises(self) -> int:
        return sum(1 for _, a in self._round_hist() if a == 2)

    def _facing_bet(self) -> bool:
        rh = self._round_hist()
        return bool(rh) and rh[-1][1] == 2

    def _round_done(self) -> bool:
        rh = self._round_hist()
        if len(rh) < 2 or rh[-1][1] != 1:
            return False
        return rh[-2][1] == 1 or any(a == 2 for _, a in rh[:-1])

    def _investments(self) -> list[float]:
        inv = [1.0, 1.0]
        for i, (p, a) in enumerate(self._hist):
            if a == 2:
                bet = 2.0 if (self._public_id is None or i < self._round_start) else 4.0
                inv[p] += bet
        return inv
