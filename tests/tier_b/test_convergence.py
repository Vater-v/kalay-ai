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
from kalay.games.leduc import LeducEnv
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


def _uniform_policy(obs, mask):
    m = np.asarray(mask, dtype=np.float64)
    return m / m.sum()


LEDUC_STEPS = 150000
LEDUC_NASHCONV_MAX = 0.15
LEDUC_UNIFORM_RATIO = 0.10  # also <= 10% of the uniform policy's NashConv (unit guard)


@pytest.mark.tier_b
@pytest.mark.parametrize("seed", SEEDS)
def test_leduc_anchor_average_policy_near_nash(seed):
    engine = train(LeducEnv, steps=LEDUC_STEPS, seed=seed)
    fn = anchor_average_policy_fn(engine)
    nc = nash_conv(LeducEnv(), fn)
    assert nc <= LEDUC_NASHCONV_MAX, f"Leduc NashConv {nc:.4f} > {LEDUC_NASHCONV_MAX}"
    uniform_nc = nash_conv(LeducEnv(), _uniform_policy)
    assert nc <= LEDUC_UNIFORM_RATIO * uniform_nc, (
        f"Leduc NashConv {nc:.4f} > 10% of uniform's {uniform_nc:.4f}"
    )


PUSHFOLD_CEILING = 100_000
PUSHFOLD_CHECKPOINT = 10_000
PUSHFOLD_NASHCONV_MAX = 0.15
PUSHFOLD_TV_MAX = 0.05


@pytest.mark.tier_b
@pytest.mark.parametrize("seed", SEEDS)
def test_pushfold_checkpoint_gate(seed):
    """Checkpoint semantics (SPEC 8): pass once both conditions hold at a
    checkpoint AND are reconfirmed at the next one; budget ceiling 100k."""
    from kalay.cards import native
    from kalay.games.pushfold import PushFoldEnv

    cfg, _ = derive_engine_config(PushFoldEnv, budget_steps=PUSHFOLD_CEILING, seed=seed)
    engine = RNaDEngine(cfg)
    rng = np.random.default_rng(seed)
    nash_p, nash_q, _ = native.nash_pushfold()
    conditions_met = False
    passed = False
    for _ in range(PUSHFOLD_CEILING // PUSHFOLD_CHECKPOINT):
        for _ in range(PUSHFOLD_CHECKPOINT):
            batch = collect_batch(PushFoldEnv, engine.policy_probs_np,
                                  BATCH_HANDS, rng, cfg.reward_scale)
            engine.train_step(batch)
        fn = anchor_average_policy_fn(engine)
        p, q = native.class_strategies(fn)
        nc = native.nashconv(p, q)
        ok = (
            nc <= PUSHFOLD_NASHCONV_MAX
            and native.range_tv(p, nash_p) <= PUSHFOLD_TV_MAX
            and native.range_tv(q, nash_q) <= PUSHFOLD_TV_MAX
        )
        print(f"[pushfold seed={seed}] nc={nc:.4f} "
              f"tv_p={native.range_tv(p, nash_p):.4f} "
              f"tv_q={native.range_tv(q, nash_q):.4f}", flush=True)
        if conditions_met and ok:
            passed = True
            break
        conditions_met = ok
    assert passed, "push-fold gate conditions never confirmed at two checkpoints"
