"""Tier A (frozen): Leduc rules and analyzer (SPEC section 9)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from kalay.core.gamestats import analyze_game
from kalay.eval.tabular import best_response_value, nash_conv
from kalay.games.base import reset_env
from kalay.games.leduc import _DECK, _HOLE_DEALS, LeducEnv


def deal(hole0, hole1, public=None):
    env = LeducEnv()
    env.step_chance(_HOLE_DEALS.index((hole0, hole1)))
    return env


def play(env, actions, public=None):
    for a in actions:
        if env.is_chance():
            remaining = [c for c in range(6) if c not in env._hole_ids]
            env.step_chance(remaining.index(public))
        env.step(a)
    return env


@pytest.mark.tier_a
def test_analyzer_exact_statistics():
    stats = analyze_game(LeducEnv)
    assert stats.n_infosets == 288          # rank-collapsed (suits canonicalized away)
    assert stats.max_decisions_per_player == 4
    assert stats.max_abs_reward == 7.0
    assert stats.n_players == 2 and stats.n_actions == 3


@pytest.mark.tier_a
def test_showdown_spots():
    # J vs K, public Q, check-down: high card wins pot 2
    np.testing.assert_allclose(play(deal(0, 4), [1, 1, 1, 1], public=3).rewards(), [-1, 1])
    # pair beats high card: J pairs with public J, K does not
    np.testing.assert_allclose(play(deal(0, 4), [1, 1, 1, 1], public=1).rewards(), [1, -1])
    # tie: both hold J (different suits), public K -> split
    np.testing.assert_allclose(play(deal(0, 1), [1, 1, 1, 1], public=4).rewards(), [0, 0])
    # round-1 bet + call: pot 2 + 2 + 2 = 6 -> +-3
    np.testing.assert_allclose(play(deal(0, 4), [2, 1, 1, 1], public=3).rewards(), [-3, 3])
    # round-1 bet + fold: folder invested the ante only
    np.testing.assert_allclose(play(deal(4, 0), [2, 0]).rewards(), [1, -1])


@pytest.mark.tier_a
def test_limit_structure_and_midgame_chance():
    env = deal(0, 4)
    assert not env.is_chance() and env.current_player() == 0
    env.step(2)  # bet (raise #1)
    env.step(2)  # raise (raise #2)
    mask = env.legal_actions_mask()
    assert mask[2] == 0.0          # raises capped at 2 per round
    assert mask[0] == 1.0          # facing a bet: fold legal
    env.step(1)                    # call closes round 1
    assert env.is_chance()         # public card is a mid-episode chance node
    env.step_chance(0)
    assert not env.is_chance()
    assert env.current_player() == 1  # round 2 opener (SPEC 9)
    assert env.legal_actions_mask()[0] == 0.0  # open fold removed (dominated)


@pytest.mark.tier_a
def test_exhaustive_rules_invariants():
    # all hole deals x all action sequences (bounded depth): zero-sum, sane magnitudes
    for holes in [(a, b) for a in range(6) for b in range(6) if a != b]:
        for seq in itertools.product([0, 1, 2], repeat=8):
            env = LeducEnv()
            env.step_chance(_HOLE_DEALS.index(holes))
            for a in seq:
                if env.is_terminal():
                    break
                if env.is_chance():
                    env.step_chance(0)
                if env.legal_actions_mask()[a] == 0:
                    break  # illegal continuation; the env stops being driven
                env.step(a)
            if env.is_terminal():
                rew = env.rewards()
                assert rew.sum() == pytest.approx(0.0)
                assert abs(rew).max() <= 7.0 + 1e-9


@pytest.mark.tier_a
def test_best_response_machinery_on_leduc():
    def uniform(obs, mask):
        m = np.asarray(mask, dtype=np.float64)
        return m / m.sum()

    nc = nash_conv(LeducEnv(), uniform)
    assert np.isfinite(nc) and nc > 0.0
    br0 = best_response_value(LeducEnv(), uniform, 0)
    assert np.isfinite(br0)
