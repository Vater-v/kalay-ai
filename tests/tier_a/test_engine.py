"""Tier A (frozen): engine mechanics — train step, anchor bookkeeping, determinism."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from kalay.core.collector import collect_batch
from kalay.core.engine import EngineConfig, RNaDEngine
from kalay.core.gamestats import derive_engine_config
from kalay.games.rps import RPSEnv
from kalay.testing import make_valid_batch


def make_engine(**overrides):
    defaults = dict(obs_dim=5, n_actions=4, hidden=16, seed=0)
    defaults.update(overrides)
    return RNaDEngine(EngineConfig(**defaults))


@pytest.mark.tier_a
def test_train_step_finite_and_returns_metrics():
    engine = make_engine()
    batch = make_valid_batch(T=3, B=4, K=4, obs_dim=5)
    for _ in range(3):
        metrics = engine.train_step(batch)
    for key in ("loss_value", "loss_policy", "tau", "mean_kl_ref",
                "force_gated_fraction", "rail_fraction"):
        assert key in metrics and np.isfinite(metrics[key])


@pytest.mark.tier_a
def test_anchor_update_and_tau_decay():
    engine = make_engine(anchor_period=2, tau=0.1, tau_decay=0.5, tau_min=0.01)
    batch = make_valid_batch(T=2, B=2, K=4, obs_dim=5, seed=1)
    engine.train_step(batch)
    assert engine.tau == pytest.approx(0.1)  # step 1: no anchor yet
    assert len(engine.anchor_state_dicts()) == 2  # initial + current
    engine.train_step(batch)
    assert engine.tau == pytest.approx(0.05)  # step 2: anchor copied, tau decayed
    assert len(engine.anchor_state_dicts()) == 3
    engine.train_step(batch)
    engine.train_step(batch)
    assert engine.tau == pytest.approx(0.025)


@pytest.mark.tier_a
def test_determinism_same_seed_same_batch():
    batch = make_valid_batch(T=3, B=4, K=4, obs_dim=5, seed=3)
    losses_a, losses_b = [], []
    engine_a = make_engine(seed=7)
    engine_b = make_engine(seed=7)
    for _ in range(4):
        losses_a.append(engine_a.train_step(batch)["loss_policy"])
        losses_b.append(engine_b.train_step(batch)["loss_policy"])
    assert losses_a == losses_b


@pytest.mark.tier_a
def test_policy_probs_masked_and_normalized():
    engine = make_engine()
    obs = np.zeros(5, dtype=np.float32)
    mask = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
    probs = engine.policy_probs(obs, mask).numpy()
    assert probs[1] == 0.0
    assert probs.sum() == pytest.approx(1.0)


@pytest.mark.tier_a
def test_collector_produces_valid_player_centric_batches():
    engine = make_engine(obs_dim=RPSEnv.obs_dim, n_actions=RPSEnv.n_actions)
    rng = np.random.default_rng(5)
    batch = collect_batch(RPSEnv, engine.policy_probs, n_hands=16, rng=rng,
                          reward_scale=1.0)
    batch.validate(n_players=2)
    # RPS: every hand gives exactly one decision per player
    assert (batch.hand_lengths == 1).all()
    assert sorted(set(batch.player_id.tolist())) == [0, 1]
    # rewards zero-sum per hand
    p0 = batch.terminal_reward[:, batch.player_id == 0].sum()
    p1 = batch.terminal_reward[:, batch.player_id == 1].sum()
    assert p0 == pytest.approx(-p1)


@pytest.mark.tier_a
def test_three_way_anchor_trigger_semantics():
    """Cumulative drift fires; settle fires only when the drift RATE cools."""
    def feed(engine, kls):
        for kl in kls:
            engine.step_count += 1
            engine._maybe_update_anchor(kl)

    # 1) cumulative trigger
    cfg, _ = derive_engine_config(RPSEnv, budget_steps=100, seed=0)
    cfg.anchor_cum_kl = 2.0
    engine = RNaDEngine(cfg)
    feed(engine, [1.0, 1.0, 1.0])
    assert engine.phases == 1

    # 2) settle trigger: constant drift level => fast/slow EMAs converge => fires
    cfg, _ = derive_engine_config(RPSEnv, budget_steps=100000, seed=0)
    cfg.anchor_cum_kl = 1e9
    cfg.anchor_settle_steps = 3
    engine = RNaDEngine(cfg)
    feed(engine, [0.5] * 500)
    assert engine.phases >= 1

    # 3) rising drift with cumulative disabled: no trigger (fast EMA leads slow)
    cfg, _ = derive_engine_config(RPSEnv, budget_steps=100000, seed=0)
    cfg.anchor_cum_kl = 1e9
    engine = RNaDEngine(cfg)
    feed(engine, [0.01 * t for t in range(1, 51)])
    assert engine.phases == 0


@pytest.mark.tier_a
def test_optimism_flag_runs_finite_and_deterministic():
    results = []
    for _ in range(2):
        cfg, _ = derive_engine_config(RPSEnv, budget_steps=100, seed=3)
        cfg.optimism_beta = 1.0
        engine = RNaDEngine(cfg)
        batch = make_valid_batch(T=1, B=2, K=3, obs_dim=RPSEnv.obs_dim, seed=4)
        losses = [engine.train_step(batch)["loss_policy"] for _ in range(3)]
        results.append(losses)
    assert results[0] == results[1]
    assert all(np.isfinite(x) for x in results[0])
