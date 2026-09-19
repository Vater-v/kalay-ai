"""Game-agnostic environment interface (N-player, imperfect information).

The RL engine consumes only (obs, legal_mask, rewards); it has no poker knowledge.
Chance nodes are explicit so that tabular best-response search (kalay/eval) can take
exact expectations over hidden deals instead of sampling them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GameEnv(ABC):
    n_players: int
    n_actions: int  # number of abstract actions K
    obs_dim: int

    @abstractmethod
    def clone(self) -> "GameEnv":
        """Deep copy for search; must not share mutable state."""

    # --- chance ---
    @abstractmethod
    def is_chance(self) -> bool:
        """True while hidden state is being dealt."""

    @abstractmethod
    def chance_probs(self) -> np.ndarray:
        """Distribution over chance outcomes at this chance node (sums to 1)."""

    @abstractmethod
    def step_chance(self, outcome: int) -> None:
        """Apply a chance outcome."""

    def sample_chance(self, rng: np.random.Generator) -> None:
        """Sample a chance outcome with `rng`.

        Default: enumerate chance_probs(). Games with astronomically many
        outcomes (full-deck deals) override this with direct sampling; their
        chance_probs() may raise, and analytic stats + specialised eval replace
        the generic tree machinery (SPEC 6).
        """
        probs = self.chance_probs()
        self.step_chance(int(rng.choice(len(probs), p=probs)))

    # --- decisions ---
    @abstractmethod
    def current_player(self) -> int:
        """Acting player at a non-chance, non-terminal node."""

    @abstractmethod
    def legal_actions_mask(self) -> np.ndarray:
        """[n_actions] float 0/1."""

    @abstractmethod
    def step(self, action: int) -> None:
        """Apply a decision action (must be legal)."""

    @abstractmethod
    def is_terminal(self) -> bool: ...

    @abstractmethod
    def rewards(self) -> np.ndarray:
        """[n_players] terminal rewards (native units); zeros before terminal."""

    @abstractmethod
    def obs(self, player: int) -> np.ndarray:
        """[obs_dim] observation from `player`'s perspective (no hidden info of others)."""


def reset_env(env: GameEnv, rng: np.random.Generator) -> GameEnv:
    """Resolve all leading chance nodes by sampling with `rng`."""
    while env.is_chance():
        probs = env.chance_probs()
        env.step_chance(int(rng.choice(len(probs), p=probs)))
    return env
