"""TrajectoryBatch: player-centric sub-trajectories packed as [T, B] tensors.

Contracts C1-C6 (SPEC section 4) are enforced at construction time; tier A tests
freeze them. Reward placement (C5) is the #1 integration bug of v6.0, hence runtime
asserts instead of documentation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TrajectoryBatch:
    obs: torch.Tensor            # [T, B, obs_dim] float32
    legal_mask: torch.Tensor     # [T, B, K] float32
    action_taken: torch.Tensor   # [T, B] int64
    behavior_prob: torch.Tensor  # [T, B] float32
    terminal_reward: torch.Tensor# [T, B] float32 (native units, nonzero only at t=L-1)
    hand_lengths: torch.Tensor   # [B] int64
    player_id: torch.Tensor      # [B] int64
    reward_scale: float

    def validate(self, n_players: int | None = None) -> None:
        T, B, K = self.legal_mask.shape
        # C1: shapes and finiteness
        assert self.obs.ndim == 3 and self.obs.shape[:2] == (T, B), "C1 obs shape"
        assert self.action_taken.shape == (T, B), "C1 action shape"
        assert self.behavior_prob.shape == (T, B), "C1 mu shape"
        assert self.terminal_reward.shape == (T, B), "C1 reward shape"
        assert self.hand_lengths.shape == (B,) and self.player_id.shape == (B,), "C1 lengths"
        for t in (self.obs, self.legal_mask, self.behavior_prob, self.terminal_reward):
            assert torch.isfinite(t).all(), "C1 finiteness"
        # C2: lengths
        L = self.hand_lengths
        assert (L >= 1).all() and (L <= T).all(), "C2 hand_lengths in [1, T]"
        # decision mask [T, B]
        t_idx = torch.arange(T, device=self.terminal_reward.device).unsqueeze(1).expand(T, B)
        dec = (t_idx < L.unsqueeze(0))
        # C3: taken actions are legal on real decisions
        legal_taken = self.legal_mask.gather(-1, self.action_taken.unsqueeze(-1)).squeeze(-1)
        assert (legal_taken[dec] == 1.0).all(), "C3 taken action must be legal"
        # C4: behavior probabilities on real decisions
        mu = self.behavior_prob[dec]
        assert (mu > 0).all() and (mu <= 1.0 + 1e-6).all(), "C4 0 < mu <= 1"
        # C5: terminal reward lives ONLY on the last real decision (t = L-1)
        pre_last = dec & (t_idx < L.unsqueeze(0) - 1)
        assert (self.terminal_reward[pre_last] == 0).all(), "C5 reward only at t = L-1"
        assert (self.terminal_reward[~dec] == 0).all(), "C5 padding rows carry no reward"
        # C6: player ids
        if n_players is not None:
            assert (self.player_id >= 0).all() and (self.player_id < n_players).all(), "C6 player_id"


def pack_subtrajectories(
    per_hand: list[dict],
    reward_scale: float,
) -> TrajectoryBatch:
    """Build a batch from collector records.

    Each record: {"obs": [np.ndarray], "mask": [np.ndarray], "act": [int],
    "mu": [float], "reward": float, "player": int}; length >= 1 (C2).
    NumPy-buffered: one `torch.from_numpy` per field instead of per-decision
    tensor construction (the collector hot path, see profile in SPEC history).
    """
    B = len(per_hand)
    T = max(len(r["act"]) for r in per_hand)
    obs_dim = per_hand[0]["obs"][0].shape[0]
    K = per_hand[0]["mask"][0].shape[0]

    obs = np.zeros((T, B, obs_dim), dtype=np.float32)
    mask = np.zeros((T, B, K), dtype=np.float32)
    act = np.zeros((T, B), dtype=np.int64)
    mu = np.zeros((T, B), dtype=np.float32)
    rew = np.zeros((T, B), dtype=np.float32)
    lengths = np.zeros(B, dtype=np.int64)
    pid = np.zeros(B, dtype=np.int64)

    for b, r in enumerate(per_hand):
        L = len(r["act"])
        lengths[b] = L
        pid[b] = r["player"]
        obs[:L, b] = r["obs"]
        mask[:L, b] = r["mask"]
        act[:L, b] = r["act"]
        mu[:L, b] = r["mu"]
        rew[L - 1, b] = r["reward"]

    batch = TrajectoryBatch(
        obs=torch.from_numpy(obs),
        legal_mask=torch.from_numpy(mask),
        action_taken=torch.from_numpy(act),
        behavior_prob=torch.from_numpy(mu),
        terminal_reward=torch.from_numpy(rew),
        hand_lengths=torch.from_numpy(lengths),
        player_id=torch.from_numpy(pid),
        reward_scale=reward_scale,
    )
    batch.validate()
    return batch
