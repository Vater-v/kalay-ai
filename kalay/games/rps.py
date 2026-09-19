"""Rock-Paper-Scissors: the NeuRD-core smoke game (no infosets, no chance).

Simultaneous move is modelled sequentially with a hidden first action: player 1's
observation never reveals player 0's move. Nash: uniform, value 0.
"""
from __future__ import annotations

import numpy as np

from kalay.games.base import GameEnv

# rows = player 0 action, cols = player 1 action, units of player 0's payoff
PAYOFF = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


class RPSEnv(GameEnv):
    n_players = 2
    n_actions = 3
    obs_dim = 3  # [const, is_player0, is_player1]

    def __init__(self) -> None:
        self._moves: list[int | None] = [None, None]
        self._turn = 0

    def clone(self) -> "RPSEnv":
        env = RPSEnv()
        env._moves = list(self._moves)
        env._turn = self._turn
        return env

    def is_chance(self) -> bool:
        return False

    def chance_probs(self) -> np.ndarray:
        raise RuntimeError("RPS has no chance nodes")

    def step_chance(self, outcome: int) -> None:
        raise RuntimeError("RPS has no chance nodes")

    def current_player(self) -> int:
        return self._turn

    def legal_actions_mask(self) -> np.ndarray:
        return np.ones(self.n_actions, dtype=np.float32)

    def step(self, action: int) -> None:
        self._moves[self._turn] = action
        self._turn += 1

    def is_terminal(self) -> bool:
        return self._moves[1] is not None

    def rewards(self) -> np.ndarray:
        if not self.is_terminal():
            return np.zeros(self.n_players)
        v0 = PAYOFF[self._moves[0], self._moves[1]]
        return np.array([v0, -v0])

    def obs(self, player: int) -> np.ndarray:
        # player id is part of the observation; the opponent's move never is
        return np.array(
            [1.0, 1.0 if player == 0 else 0.0, 1.0 if player == 1 else 0.0],
            dtype=np.float32,
        )
