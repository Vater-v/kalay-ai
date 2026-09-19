"""Tier A (frozen): batch contracts C1-C6 (SPEC section 4)."""
from __future__ import annotations

import pytest
import torch

from kalay.core.batch import TrajectoryBatch, pack_subtrajectories
from kalay.testing import make_valid_batch


@pytest.mark.tier_a
def test_valid_batch_passes_and_packs():
    batch = make_valid_batch()
    batch.validate(n_players=2)
    rec = {
        "obs": [batch.obs[0, 0].numpy()],
        "mask": [batch.legal_mask[0, 0].numpy()],
        "act": [int(batch.action_taken[0, 0])],
        "mu": [float(batch.behavior_prob[0, 0])],
        "reward": 0.5,
        "player": 1,
    }
    packed = pack_subtrajectories([rec], reward_scale=2.0)
    assert packed.terminal_reward[0, 0] == pytest.approx(0.5)
    assert packed.player_id[0] == 1


@pytest.mark.tier_a
def test_c1_nan_and_shapes_rejected():
    batch = make_valid_batch()
    bad = replace_field(batch, "obs", batch.obs.clone())
    bad.obs[1, 1, 0] = float("nan")
    with pytest.raises(AssertionError, match="C1"):
        bad.validate()
    with pytest.raises(AssertionError, match="C1"):
        replace_field(batch, "action_taken", batch.action_taken[:, :1]).validate()


@pytest.mark.tier_a
def test_c2_bad_lengths_rejected():
    batch = make_valid_batch(T=3, B=2)
    bad = replace_field(batch, "hand_lengths", batch.hand_lengths.clone())
    bad.hand_lengths[0] = 0
    with pytest.raises(AssertionError, match="C2"):
        bad.validate()
    bad = replace_field(batch, "hand_lengths", batch.hand_lengths.clone())
    bad.hand_lengths[0] = 4  # > T
    with pytest.raises(AssertionError, match="C2"):
        bad.validate()


@pytest.mark.tier_a
def test_c3_illegal_taken_action_rejected():
    batch = make_valid_batch(T=2, B=2, K=4)
    bad = replace_field(batch, "legal_mask", batch.legal_mask.clone())
    bad.legal_mask[0, 0] = 0.0  # make every action illegal at a real decision
    with pytest.raises(AssertionError, match="C3"):
        bad.validate()


@pytest.mark.tier_a
def test_c4_bad_behavior_prob_rejected():
    batch = make_valid_batch(T=2, B=2)
    bad = replace_field(batch, "behavior_prob", batch.behavior_prob.clone())
    bad.behavior_prob[0, 0] = 0.0
    with pytest.raises(AssertionError, match="C4"):
        bad.validate()
    bad = replace_field(batch, "behavior_prob", batch.behavior_prob.clone())
    bad.behavior_prob[0, 0] = 1.5
    with pytest.raises(AssertionError, match="C4"):
        bad.validate()


@pytest.mark.tier_a
def test_c5_reward_misplacement_rejected():
    # reward on a padding row
    batch = make_valid_batch(T=3, B=2)
    bad = replace_field(batch, "terminal_reward", batch.terminal_reward.clone())
    bad.terminal_reward[2, 0] = 1.0  # L[0] <= 2 in this fixture? ensure by force:
    bad.hand_lengths = bad.hand_lengths.clone()
    bad.hand_lengths[0] = 2
    bad.terminal_reward[1, 0] = 0.0
    with pytest.raises(AssertionError, match="C5"):
        bad.validate()
    # reward on a real but non-terminal decision
    bad2 = replace_field(batch, "terminal_reward", batch.terminal_reward.clone())
    bad2.terminal_reward[0, 0] = 1.0
    bad2.hand_lengths = bad2.hand_lengths.clone()
    bad2.hand_lengths[0] = 3
    bad2.terminal_reward[2, 0] = 1.0
    with pytest.raises(AssertionError, match="C5"):
        bad2.validate()


@pytest.mark.tier_a
def test_c6_bad_player_id_rejected():
    batch = make_valid_batch(B=2)
    bad = replace_field(batch, "player_id", batch.player_id.clone())
    bad.player_id[0] = 5
    with pytest.raises(AssertionError, match="C6"):
        bad.validate(n_players=2)


def replace_field(batch: TrajectoryBatch, name: str, value) -> TrajectoryBatch:
    data = dict(batch.__dict__)
    data[name] = value
    return TrajectoryBatch(**data)
