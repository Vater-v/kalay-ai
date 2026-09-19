"""Self-play data collection into TrajectoryBatch.

Player-centric: each hand is split into per-player sub-trajectories; hands where a
player never acted (walk-overs) are dropped for that player (C2 requires L >= 1).
Probs are memoized per (obs, mask) within one collection call — the net is frozen
during collection, so this is exact.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from kalay.core.batch import TrajectoryBatch, pack_subtrajectories
from kalay.games.base import GameEnv


def collect_batch(
    make_env: Callable[[], GameEnv],
    policy_probs: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_hands: int,
    rng: np.random.Generator,
    reward_scale: float,
) -> TrajectoryBatch:
    records: list[dict] = []
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for _ in range(n_hands):
        env = make_env()
        recs: list[list[tuple]] = [[] for _ in range(env.n_players)]
        while True:
            if env.is_terminal():
                break
            if env.is_chance():  # leading deal OR mid-episode chance (Leduc public card)
                env.sample_chance(rng)
                continue
            p = env.current_player()
            obs = env.obs(p)
            mask = env.legal_actions_mask()
            key = obs.tobytes() + mask.tobytes()
            cached = cache.get(key)
            if cached is None:
                probs_t = policy_probs(obs, mask)
                probs = (
                    probs_t.detach().cpu().numpy().astype(np.float64)
                    if hasattr(probs_t, "detach")
                    else np.asarray(probs_t, dtype=np.float64)
                )
                cum = np.cumsum(probs / probs.sum())
                cum[-1] = 1.0  # guard float drift
                cached = (probs, cum)
                cache[key] = cached
            probs, cum = cached
            action = int(np.searchsorted(cum, rng.random(), side="right"))
            action = min(action, len(probs) - 1)
            recs[p].append((obs, mask, action, float(probs[action])))
            env.step(action)
        rewards = env.rewards()
        for p in range(env.n_players):
            if not recs[p]:
                continue
            records.append(
                {
                    "obs": np.stack([r[0] for r in recs[p]]),
                    "mask": np.stack([r[1] for r in recs[p]]),
                    "act": np.array([r[2] for r in recs[p]], dtype=np.int64),
                    "mu": np.array([r[3] for r in recs[p]], dtype=np.float32),
                    "reward": float(rewards[p]),
                    "player": p,
                }
            )
    return pack_subtrajectories(records, reward_scale)
