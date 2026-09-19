"""Tier A (frozen): v3 token pipeline for push-fold (engine gather path)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from kalay.cards import native
from kalay.core.engine import EngineConfig, RNaDEngine
from kalay.core.vec_pushfold import (
    class_strategies_v3,
    collect_pushfold_v3,
    install_token_batch,
)


def _engine(seed=0):
    cfg = EngineConfig(obs_dim=4, n_actions=2, n_players=2, reward_scale=10.0,
                       net_kind="v3", max_len=8, seed=seed)
    eng = RNaDEngine(cfg)
    install_token_batch(eng)
    return eng


@pytest.mark.tier_a
def test_v3_batch_contracts_and_training():
    eng = _engine()
    rng = np.random.default_rng(0)
    batch = collect_pushfold_v3(eng, 64, rng)
    batch.validate(n_players=2)
    assert batch.slice_ids is not None
    assert int(batch.slice_ids.min()) >= 0
    assert int(batch.slice_ids.max()) < 338
    for _ in range(3):
        m = eng.train_step(collect_pushfold_v3(eng, 64, rng))
    assert np.isfinite(m["loss_policy"]) and np.isfinite(m["loss_value"])


@pytest.mark.tier_a
def test_v3_zero_sum_rewards_and_determinism():
    eng = _engine(seed=3)
    rng = np.random.default_rng(3)
    b1 = collect_pushfold_v3(eng, 128, rng)
    # per-hand zero-sum: BTN row + BB row of the same hand cancel
    rew = b1.terminal_reward[0].numpy()
    players = b1.player_id.numpy()
    # every BTN row either ends alone (fold: -0.5) or pairs with a BB row
    assert abs(rew[players == 0].sum() + rew[players == 1].sum()) < 1e-6
    eng2 = _engine(seed=3)
    rng2 = np.random.default_rng(3)
    losses1 = [eng.train_step(collect_pushfold_v3(eng, 32, rng))["loss_policy"]
               for _ in range(3)]
    losses2 = [eng2.train_step(collect_pushfold_v3(eng2, 32, rng2))["loss_policy"]
               for _ in range(3)]
    assert losses1 == losses2


@pytest.mark.tier_a
def test_v3_class_strategies_shape():
    eng = _engine()
    p, q = class_strategies_v3(eng)
    assert p.shape == (169,) and q.shape == (169,)
    assert ((p >= 0) & (p <= 1)).all()
