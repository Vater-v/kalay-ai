"""Tier A (frozen): Push-Fold adapter rules and canonical observations (SPEC 9)."""
from __future__ import annotations

import numpy as np
import pytest

from kalay.cards import native
from kalay.core.gamestats import GameStats, derive_engine_config
from kalay.games.pushfold import PushFoldEnv

A = 12  # rank id of Ace (cards encode rank<<2 | suit)


def _card(rank: int, suit: int) -> int:
    return (rank << 2) | suit


@pytest.mark.tier_a
def test_fold_payoffs_exact():
    env = PushFoldEnv()
    env.set_deal((_card(A, 0), _card(A, 1)), (_card(11, 0), _card(11, 1)))
    env.step(0)  # BTN folds AA (legal but dominated)
    np.testing.assert_allclose(env.rewards(), [-0.5, 0.5])
    env = PushFoldEnv()
    env.set_deal((_card(A, 0), _card(A, 1)), (_card(11, 0), _card(11, 1)))
    env.step(1)  # BTN pushes
    assert env.current_player() == 1
    env.step(0)  # BB folds KK
    np.testing.assert_allclose(env.rewards(), [1.0, -1.0])


@pytest.mark.tier_a
def test_showdown_matches_equity_model():
    env = PushFoldEnv()
    aa = (_card(A, 0), _card(A, 1))
    kk = (_card(11, 0), _card(11, 1))
    env.set_deal(aa, kk)
    env.step(1)
    env.step(1)  # call -> showdown
    eq = native.pushfold_solver().equity(
        native.hand_index(*aa), native.hand_index(*kk)
    )
    r = env.rewards()
    assert r[0] == pytest.approx(20.0 * eq - 10.0, abs=1e-5)
    assert r.sum() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.tier_a
def test_observations_are_s4_canonical():
    env = PushFoldEnv()
    # A(h)K(s) offsuit == A(d)K(c) offsuit; differs from A(s)K(s) suited
    env.set_deal((_card(A, 1), _card(11, 0)), (_card(5, 2), _card(2, 3)))
    o1 = env.obs(0)
    env2 = PushFoldEnv()
    env2.set_deal((_card(A, 2), _card(11, 3)), (_card(5, 1), _card(2, 0)))
    o2 = env2.obs(0)
    np.testing.assert_array_equal(o1, o2)
    env3 = PushFoldEnv()
    env3.set_deal((_card(A, 0), _card(11, 0)), (0, 1))
    assert not np.array_equal(o1, env3.obs(0))  # suited differs
    # hole card order invariance
    env4 = PushFoldEnv()
    env4.set_deal((_card(11, 0), _card(A, 1)), (0, 1))
    np.testing.assert_array_equal(o1, env4.obs(0))
    # BB seat observation differs from BTN
    assert not np.array_equal(env.obs(0), env.obs(1))


@pytest.mark.tier_a
def test_analytic_stats_used_by_derive():
    cfg, stats = derive_engine_config(PushFoldEnv, budget_steps=1000)
    assert isinstance(stats, GameStats)
    assert stats.n_infosets == 338
    assert cfg.reward_scale == 10.0
    from kalay.core.gamestats import pi_min_bound
    assert cfg.mu_floor == pytest.approx(pi_min_bound(2, cfg.z_thresh))


@pytest.mark.tier_a
def test_sampled_rollouts_zero_sum_and_bounded():
    rng = np.random.default_rng(3)
    for _ in range(300):
        env = PushFoldEnv()
        env.sample_chance(rng)
        while not env.is_terminal():
            env.step(int(rng.integers(0, 2)))
        rew = env.rewards()
        assert rew.sum() == pytest.approx(0.0, abs=1e-6)
        assert abs(rew).max() <= 10.0 + 1e-9
