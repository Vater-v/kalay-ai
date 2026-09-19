"""Kuhn poker: 3-card deck, one street, fixed bet size.

Actions (abstract, K=2): 0 = passive (check / fold), 1 = aggressive (bet / call).
Ante 1 each; bet size 1. Net terminal rewards in chips, zero-sum:
check-check showdown +-1, bet-call showdown +-2, any fold +-1.
Game value (player 0): -1/18. reward_scale for the engine: 2.0 (max |net|).
"""
from __future__ import annotations

import numpy as np

from kalay.games.base import GameEnv

_DEALS = [(a, b) for a in range(3) for b in range(3) if a != b]  # (card0, card1)


class KuhnPokerEnv(GameEnv):
    n_players = 2
    n_actions = 2
    obs_dim = 3 + 2 + 3 * 3  # card one-hot, player one-hot, 3 history slots one-hot(3)

    def __init__(self) -> None:
        self._cards: tuple[int, int] | None = None
        self._hist: list[tuple[int, int]] = []  # (player, action)

    # --- construction / cloning ---
    def clone(self) -> "KuhnPokerEnv":
        env = KuhnPokerEnv()
        env._cards = self._cards
        env._hist = list(self._hist)
        return env

    # --- chance: the initial deal ---
    def is_chance(self) -> bool:
        return self._cards is None

    def chance_probs(self) -> np.ndarray:
        return np.full(len(_DEALS), 1.0 / len(_DEALS))

    def step_chance(self, outcome: int) -> None:
        self._cards = _DEALS[outcome]

    # --- decisions ---
    def current_player(self) -> int:
        if self.is_chance() or self.is_terminal():
            raise RuntimeError("no acting player at this node")
        return len(self._hist) % 2  # turns strictly alternate

    def legal_actions_mask(self) -> np.ndarray:
        return np.ones(self.n_actions, dtype=np.float32)

    def step(self, action: int) -> None:
        self._hist.append((self.current_player(), action))

    # --- terminal & rewards ---
    def is_terminal(self) -> bool:
        return self._terminal_rewards() is not None

    def _terminal_rewards(self) -> np.ndarray | None:
        h = self._hist
        if not h:
            return None
        rew = np.zeros(self.n_players)
        if h[-1][1] == 0:  # last action passive: fold or check-close
            if len(h) == 1:
                return None  # check by player 0: game continues
            if h[-2][1] == 1:  # bet then fold: the aggressor takes the pot
                folder = h[-1][0]
                rew[folder] = -1.0
                rew[1 - folder] = 1.0
                return rew
            if len(h) == 2:  # check, check -> showdown for pot 2
                return self._showdown(1.0)
            return None
        # last action aggressive: terminal only if it is a call (the previous
        # action was aggressive too: bet-call or check-bet-call)
        if len(h) == 1 or h[-2][1] == 0:
            return None  # opening bet or a bet after a check: game continues
        return self._showdown(2.0)

    def _showdown(self, magnitude: float) -> np.ndarray:
        c0, c1 = self._cards
        rew = np.zeros(self.n_players)
        sign = 1.0 if c0 > c1 else -1.0
        rew[0] = sign * magnitude
        rew[1] = -sign * magnitude
        return rew

    def rewards(self) -> np.ndarray:
        rew = self._terminal_rewards()
        return rew if rew is not None else np.zeros(self.n_players)

    # --- observation ---
    def obs(self, player: int) -> np.ndarray:
        card = np.zeros(3, dtype=np.float32)
        card[self._cards[player]] = 1.0
        pid = np.zeros(2, dtype=np.float32)
        pid[player] = 1.0
        hist = np.zeros(9, dtype=np.float32)
        for i, (_, action) in enumerate(self._hist[:3]):
            hist[3 * i + action] = 1.0
        return np.concatenate([card, pid, hist])
