"""Tier A (frozen): game adapter invariants + eval machinery sanity."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from kalay.eval.tabular import best_response_value, nash_conv
from kalay.games.base import reset_env
from kalay.games.kuhn import KuhnPokerEnv, _DEALS
from kalay.games.rps import PAYOFF, RPSEnv


def uniform_policy(obs, mask):
    m = np.asarray(mask, dtype=np.float64)
    return m / m.sum()


@pytest.mark.tier_a
def test_rps_payoffs_and_no_information_leak():
    env = RPSEnv()
    obs_p1_before = env.obs(1).copy()
    env.step(2)  # player 0 plays scissors
    assert env.current_player() == 1
    np.testing.assert_array_equal(env.obs(1), obs_p1_before)  # no leak of the move
    env.step(1)  # paper beats scissors
    np.testing.assert_allclose(env.rewards(), PAYOFF[2, 1] * np.array([1.0, -1.0]))


@pytest.mark.tier_a
def test_kuhn_rules_exhaustive():
    """Enumerate all deals x action sequences: zero-sum, pot arithmetic, alternation."""
    for deal_idx, (c0, c1) in enumerate(_DEALS):
        for seq in itertools.product([0, 1], repeat=3):
            env = KuhnPokerEnv()
            env.step_chance(deal_idx)
            players = []
            for a in seq:
                if env.is_terminal():
                    break
                players.append(env.current_player())
                env.step(a)
            if not env.is_terminal():
                continue
            rew = env.rewards()
            assert rew.sum() == pytest.approx(0.0), "zero-sum"
            assert set(np.abs(rew)) <= {1.0, 2.0}, "chip magnitudes"
            assert players == [i % 2 for i in range(len(players))], "strict alternation"


@pytest.mark.tier_a
def test_kuhn_known_spots():
    def play(deal, actions):
        env = KuhnPokerEnv()
        env.step_chance(_DEALS.index(deal))
        for a in actions:
            env.step(a)
        return env.rewards()

    np.testing.assert_allclose(play((0, 2), [0, 0]), [-1.0, 1.0])   # check-check, J vs K
    np.testing.assert_allclose(play((2, 0), [0, 0]), [1.0, -1.0])
    np.testing.assert_allclose(play((0, 2), [1, 1]), [-2.0, 2.0])   # bet-call
    np.testing.assert_allclose(play((2, 0), [1, 0]), [1.0, -1.0])    # bet-fold
    np.testing.assert_allclose(play((0, 2), [0, 1, 0]), [-1.0, 1.0])  # check-bet-fold
    np.testing.assert_allclose(play((0, 2), [0, 1, 1]), [-2.0, 2.0])  # check-bet-call


@pytest.mark.tier_a
def test_best_response_rps_exact_anchors():
    # Nash (uniform) has NashConv exactly 0
    assert nash_conv(RPSEnv(), uniform_policy) == pytest.approx(0.0, abs=1e-12)
    # both players always playing rock: each BR (paper) gains +1 -> NashConv = 2
    def always_rock(obs, mask):
        return np.array([1.0, 0.0, 0.0])

    assert nash_conv(RPSEnv(), always_rock) == pytest.approx(2.0)


@pytest.mark.tier_a
def test_best_response_kuhn_positive_and_finite():
    nc = nash_conv(KuhnPokerEnv(), uniform_policy)
    assert np.isfinite(nc) and nc > 0.0  # uniform is exploitable
    # a pure aggressive policy is exploitable too, differently
    def always_aggressive(obs, mask):
        return np.array([0.0, 1.0])

    nc2 = nash_conv(KuhnPokerEnv(), always_aggressive)
    assert np.isfinite(nc2) and nc2 > 0.0


@pytest.mark.tier_a
def test_reset_env_resolves_chance_deterministically():
    rng = np.random.default_rng(42)
    env = reset_env(KuhnPokerEnv(), rng)
    assert not env.is_chance()
    assert not env.is_terminal()
