"""Anchor-average (Nash-average) policy for evaluation.

In cyclic games the instantaneous policy keeps orbiting; the average over anchor
phases converges (NeuRD, Hennes et al. 2020). We average probabilities — invariant
to parameterization, unlike weight averaging.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from kalay.core.engine import RNaDEngine
from kalay.core.neurd import masked_softmax


def anchor_average_policy_fn(engine: RNaDEngine):
    net = copy.deepcopy(engine.policy_net).to("cpu").eval()
    snapshots = engine.anchor_state_dicts()

    def fn(obs: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(legal_mask, dtype=torch.float32).unsqueeze(0)
        acc = None
        with torch.no_grad():
            for sd in snapshots:
                net.load_state_dict(sd)
                probs = masked_softmax(net(obs_t), mask_t).squeeze(0)
                acc = probs.clone() if acc is None else acc + probs
        out = (acc / len(snapshots)).numpy()
        return out / out.sum()

    return fn
