"""Tier A (frozen): derived-config machinery (SPEC section 6)."""
from __future__ import annotations

import pytest

from kalay.core.engine import RNaDEngine
from kalay.core.gamestats import (
    CLASS_CONSTANTS,
    analyze_game,
    derive_engine_config,
    pi_min_bound,
)
from kalay.games.kuhn import KuhnPokerEnv
from kalay.games.rps import RPSEnv
from kalay.testing import make_valid_batch


@pytest.mark.tier_a
def test_analyzer_exact_statistics():
    kuhn = analyze_game(KuhnPokerEnv)
    assert kuhn.n_infosets == 12
    assert kuhn.max_decisions_per_player == 2
    assert kuhn.max_abs_reward == 2.0
    assert kuhn.n_players == 2 and kuhn.n_actions == 2

    rps = analyze_game(RPSEnv)
    assert rps.n_infosets == 2
    assert rps.max_decisions_per_player == 1
    assert rps.max_abs_reward == 1.0


@pytest.mark.tier_a
def test_pi_min_closed_form_matches_audit_table():
    assert pi_min_bound(2, 3.0) == pytest.approx(0.00247, rel=1e-2)
    assert pi_min_bound(3, 3.0) == pytest.approx(0.00124, rel=1e-2)
    assert pi_min_bound(5, 3.0) == pytest.approx(0.00062, rel=1e-2)
    assert pi_min_bound(12, 3.0) == pytest.approx(0.000225, rel=1e-2)


@pytest.mark.tier_a
def test_derived_config_uses_formulas():
    cfg, stats = derive_engine_config(KuhnPokerEnv, budget_steps=15000)
    assert cfg.anchor_mode == "drift"
    assert cfg.planned_steps == 15000
    assert cfg.reward_scale == stats.max_abs_reward == 2.0
    assert cfg.mu_floor == pytest.approx(pi_min_bound(2, cfg.z_thresh))
    assert cfg.tau == cfg.tau_min == CLASS_CONSTANTS["tau0"]


@pytest.mark.tier_a
def test_drift_anchor_fires_and_tau_anneals():
    cfg, _ = derive_engine_config(RPSEnv, budget_steps=100, seed=0)
    cfg.anchor_cum_kl = 1e-12  # fire essentially every step
    engine = RNaDEngine(cfg)
    batch = make_valid_batch(T=1, B=2, K=3, obs_dim=RPSEnv.obs_dim, seed=1)
    tau0 = engine.tau
    for _ in range(4):
        engine.train_step(batch)
    assert engine.phases >= 2
    assert engine.tau <= tau0
