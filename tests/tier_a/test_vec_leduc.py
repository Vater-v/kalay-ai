"""Tier A (frozen): vectorized Leduc collector == sequential LeducEnv.

Drives both state machines through identical (deal, public-card, action)
sequences: actors, legal masks, observations and terminal rewards must match
exactly. Licensed by this test, the vectorized path may replace the sequential
one anywhere without changing contracts (SPEC 6).
"""
from __future__ import annotations

import numpy as np
import pytest

from kalay.core.gamestats import derive_engine_config
from kalay.core.vec_leduc import VecLeducState, collect_leduc_vec
from kalay.games.leduc import _HOLE_DEALS, LeducEnv


class _FixedPickRng:
    """resolve_public asks rng.integers(0, 4, k); we answer deterministically."""

    def __init__(self, value: int) -> None:
        self.value = value

    def integers(self, *args, **kwargs):
        n = args[2] if len(args) >= 3 else kwargs.get("size", 1)
        return np.full(n, self.value, dtype=np.int64)


@pytest.mark.tier_a
def test_state_machine_equivalence_random_plays():
    rng = np.random.default_rng(21)
    for _ in range(300):
        deal = tuple(int(c) for c in rng.choice(6, 2, replace=False))
        pub_choice = int(rng.integers(0, 4))
        env = LeducEnv()
        env.step_chance(_HOLE_DEALS.index(deal))
        st = VecLeducState(1, deals=np.array([deal]))
        fixed = _FixedPickRng(pub_choice)
        for _ in range(12):
            if env.is_terminal():
                assert bool(st.done[0])
                break
            if env.is_chance():
                env.step_chance(pub_choice)
                st.resolve_public(fixed)
                continue
            p_seq = env.current_player()
            assert p_seq == int(st.actor()[0]), "actor mismatch"
            np.testing.assert_array_equal(
                env.legal_actions_mask(), st.legal_mask()[0], err_msg="mask mismatch"
            )
            np.testing.assert_array_equal(
                env.obs(p_seq), st.obs(st.actor())[0], err_msg="obs mismatch"
            )
            legal = np.flatnonzero(env.legal_actions_mask())
            a = int(rng.choice(legal))
            env.step(a)
            st.step(np.array([a]))
        np.testing.assert_allclose(env.rewards(), st.rewards()[0], atol=1e-6)


@pytest.mark.tier_a
def test_known_reward_spots_through_vec_state():
    # bet-call, aggressor (K) wins: [+3, -3]; caller wins: [-3, +3]
    st = VecLeducState(1, deals=np.array([(4, 0)]))
    st.step(np.array([2])); st.step(np.array([1]))
    st.resolve_public(_FixedPickRng(2))  # remaining of (4,0) = [1,2,3,5]; pick idx 2 -> 3 (Q)
    st.step(np.array([1])); st.step(np.array([1]))
    np.testing.assert_allclose(st.rewards()[0], [3.0, -3.0])
    st2 = VecLeducState(1, deals=np.array([(0, 4)]))
    st2.step(np.array([2])); st2.step(np.array([1]))
    st2.resolve_public(_FixedPickRng(2))
    st2.step(np.array([1])); st2.step(np.array([1]))
    np.testing.assert_allclose(st2.rewards()[0], [-3.0, 3.0])


@pytest.mark.tier_a
def test_vec_collect_produces_valid_batches_and_trains():
    import torch

    torch.manual_seed(0)
    from kalay.core.engine import RNaDEngine

    cfg, _ = derive_engine_config(LeducEnv, budget_steps=100, seed=0)
    engine = RNaDEngine(cfg)
    rng = np.random.default_rng(5)
    for _ in range(3):
        batch = collect_leduc_vec(engine, 64, rng)
        batch.validate(n_players=2)
        metrics = engine.train_step(batch)
        assert np.isfinite(metrics["loss_policy"])
