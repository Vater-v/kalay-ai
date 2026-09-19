"""Test-support helpers (used by frozen tier A/B suites)."""
from __future__ import annotations

import numpy as np
import torch

from kalay.core.batch import TrajectoryBatch


def make_valid_batch(
    T: int = 3,
    B: int = 4,
    K: int = 4,
    obs_dim: int = 5,
    seed: int = 0,
    reward_scale: float = 1.0,
) -> TrajectoryBatch:
    """A random but contract-clean batch (C1-C6)."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(T, B, obs_dim)).astype(np.float32)
    mask = (rng.random((T, B, K)) > 0.5).astype(np.float32)
    mask[..., 0] = 1.0
    act = np.zeros((T, B), dtype=np.int64)
    mu = np.zeros((T, B), dtype=np.float32)
    rew = np.zeros((T, B), dtype=np.float32)
    lengths = rng.integers(1, T + 1, size=B).astype(np.int64)
    t_idx = np.arange(T)[:, None]
    dec = t_idx < lengths[None, :]
    for t in range(T):
        for b in range(B):
            legal = np.flatnonzero(mask[t, b])
            act[t, b] = rng.choice(legal) if dec[t, b] else int(legal[0])
            mu[t, b] = rng.uniform(0.2, 1.0) if dec[t, b] else 1.0
    for b in range(B):
        rew[lengths[b] - 1, b] = rng.uniform(-1, 1)
    batch = TrajectoryBatch(
        obs=torch.as_tensor(obs),
        legal_mask=torch.as_tensor(mask),
        action_taken=torch.as_tensor(act),
        behavior_prob=torch.as_tensor(mu),
        terminal_reward=torch.as_tensor(rew),
        hand_lengths=torch.as_tensor(lengths),
        player_id=torch.zeros(B, dtype=torch.int64),
        reward_scale=reward_scale,
    )
    batch.validate()
    return batch
