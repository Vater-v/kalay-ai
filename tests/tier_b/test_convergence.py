"""Tier B (PRE-REGISTERED, frozen in SPEC section 8 BEFORE the first run).

External truth anchors:
  - RPS:  Nash is uniform, NashConv(Nash) = 0 exactly.
  - Kuhn: game value for player 0 is -1/18 (analytic).

Thresholds, budgets and seeds below are FROZEN. If a test fails, the code or the
default hyperparameters are fixed — never these numbers (changing them requires an
explicit SPEC revision).
"""
from __future__ import annotations

import numpy as np
import pytest

from kalay.core.collector import collect_batch
from kalay.core.engine import RNaDEngine
from kalay.core.gamestats import derive_engine_config
from kalay.eval.policy_avg import anchor_average_policy_fn
from kalay.eval.tabular import nash_conv, policy_value
from kalay.games.kuhn import KuhnPokerEnv
from kalay.games.rps import RPSEnv

RPS_STEPS = 4000
KUHN_STEPS = 15000
BATCH_HANDS = 128
SEEDS = [0, 1, 2]
RPS_NASHCONV_MAX = 0.05
KUHN_NASHCONV_MAX = 0.05
KUHN_VALUE_TOL = 0.05  # |EV(avg,avg) - (-1/18)|


def train(env_cls, steps, seed):
    cfg, _stats = derive_engine_config(env_cls, budget_steps=steps, seed=seed)
    engine = RNaDEngine(cfg)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        batch = collect_batch(env_cls, engine.policy_probs, BATCH_HANDS, rng,
                              cfg.reward_scale)
        engine.train_step(batch)
    return engine


@pytest.mark.tier_b
@pytest.mark.parametrize("seed", SEEDS)
def test_rps_anchor_average_policy_near_nash(seed):
    engine = train(RPSEnv, steps=RPS_STEPS, seed=seed)
    fn = anchor_average_policy_fn(engine)
    nc = nash_conv(RPSEnv(), fn)
    assert nc <= RPS_NASHCONV_MAX, f"RPS NashConv {nc:.4f} > {RPS_NASHCONV_MAX}"


@pytest.mark.tier_b
@pytest.mark.parametrize("seed", SEEDS)
def test_kuhn_anchor_average_policy_near_nash(seed):
    engine = train(KuhnPokerEnv, steps=KUHN_STEPS, seed=seed)
    fn = anchor_average_policy_fn(engine)
    nc = nash_conv(KuhnPokerEnv(), fn)
    assert nc <= KUHN_NASHCONV_MAX, f"Kuhn NashConv {nc:.4f} > {KUHN_NASHCONV_MAX}"
    ev = policy_value(KuhnPokerEnv(), fn, player=0)
    assert abs(ev - (-1.0 / 18.0)) <= KUHN_VALUE_TOL, (
        f"Kuhn EV {ev:.4f} deviates from the analytic game value -1/18"
    )
