"""Preflop Push-Fold NLHE 10bb per SPEC section 9 (frozen definition).

BTN acts: 0=fold, 1=push 10bb. BB vs push: 0=fold, 1=call. Showdown equity comes
from the validated kalay_rs table (the SAME model the exact Nash anchor uses -
the game is defined by that table; its MC precision vs real poker is SE~0.01bb).
Observations are S4-canonical hole encodings (hi/lo ranks + suited/pair flags):
169 classes per seat by construction. The chance space (52*51*50*49 deals) is
not enumerable: analytic stats (SPEC 6) and the native class solver (SPEC 8)
replace the generic tree machinery.
"""
from __future__ import annotations

import numpy as np

from kalay.cards import native
from kalay.core.gamestats import GameStats
from kalay.games.base import GameEnv

_STACK = 10.0
_SB = 0.5
_BB_BLIND = 1.0


class PushFoldEnv(GameEnv):
    n_players = 2
    n_actions = 2  # 0 = fold, 1 = push (BTN) / call (BB)
    obs_dim = 13 + 13 + 1 + 1 + 2

    ANALYTIC_STATS = GameStats(
        n_players=2,
        n_actions=2,
        obs_dim=obs_dim,
        n_infosets=338,  # 169 hole classes x 2 seats
        max_decisions_per_player=1,
        avg_decisions_per_player=1.0,
        max_abs_reward=10.0,
    )

    def __init__(self) -> None:
        self._btn: tuple[int, int] | None = None
        self._bb: tuple[int, int] | None = None
        self._stage = 0  # 0 chance, 1 BTN acts, 2 BB acts, 3 terminal
        self._btn_folded = False
        self._bb_folded = False

    def clone(self) -> "PushFoldEnv":
        env = PushFoldEnv()
        env._btn = self._btn
        env._bb = self._bb
        env._stage = self._stage
        env._btn_folded = self._btn_folded
        env._bb_folded = self._bb_folded
        return env

    # --- chance: the full-deck deal (not enumerable; direct sampling) ---
    def is_chance(self) -> bool:
        return self._stage == 0

    def chance_probs(self) -> np.ndarray:
        raise NotImplementedError(
            "52*51*50*49 chance outcomes; use sample_chance (SPEC 6: analytic stats)"
        )

    def step_chance(self, outcome: int) -> None:
        deck = list(range(52))
        cards = []
        x = outcome
        for n in (52, 51, 50, 49):
            cards.append(deck.pop(x % n))
            x //= n
        self._btn = (cards[0], cards[1])
        self._bb = (cards[2], cards[3])
        self._stage = 1

    def sample_chance(self, rng: np.random.Generator) -> None:
        draw = rng.choice(52, size=4, replace=False)
        self._btn = (int(draw[0]), int(draw[1]))
        self._bb = (int(draw[2]), int(draw[3]))
        self._stage = 1

    def set_deal(self, btn: tuple[int, int], bb: tuple[int, int]) -> None:
        """Test/eval helper: fix a specific deal and open the BTN decision."""
        self._btn = tuple(btn)
        self._bb = tuple(bb)
        self._stage = 1

    # --- decisions ---
    def current_player(self) -> int:
        if self.is_chance() or self.is_terminal():
            raise RuntimeError("no acting player at this node")
        return 0 if self._stage == 1 else 1

    def legal_actions_mask(self) -> np.ndarray:
        return np.ones(self.n_actions, dtype=np.float32)

    def step(self, action: int) -> None:
        if self._stage == 1:
            if action == 0:
                self._btn_folded = True
                self._stage = 3
            else:
                self._stage = 2
        elif self._stage == 2:
            self._bb_folded = action == 0
            self._stage = 3
        else:
            raise RuntimeError("no acting player at this node")

    # --- terminal & rewards ---
    def is_terminal(self) -> bool:
        return self._stage == 3

    def rewards(self) -> np.ndarray:
        if not self.is_terminal():
            return np.zeros(self.n_players)
        if self._btn_folded:
            return np.array([-_SB, _SB])
        if self._bb_folded:
            return np.array([_BB_BLIND, -_BB_BLIND])
        eq = native.pushfold_solver().equity(
            native.hand_index(*self._btn), native.hand_index(*self._bb)
        )
        r_btn = 2.0 * _STACK * eq - _STACK  # pot 20bb, invested 10bb
        return np.array([r_btn, -r_btn])

    # --- observation: S4-canonical hole encoding ---
    def obs(self, player: int) -> np.ndarray:
        cards = self._btn if player == 0 else self._bb
        c1, c2 = sorted(cards, key=lambda c: (-(c >> 2), -(c & 3)))
        hi, lo = c1 >> 2, c2 >> 2
        f = np.zeros(self.obs_dim, dtype=np.float32)
        f[hi] = 1.0
        f[13 + lo] = 1.0
        f[26] = 1.0 if (c1 & 3) == (c2 & 3) else 0.0  # suited
        f[27] = 1.0 if hi == lo else 0.0               # pair
        f[28 if player == 0 else 29] = 1.0
        return f
