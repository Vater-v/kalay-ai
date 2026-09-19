"""Vectorized v3 collection for Push-Fold (token pipeline pilot).

All 338 canonical decision slices (BTN + BB x 169) are packed once into the
engine's token_batch; a single batched forward per collection yields the full
probability table, sampling is a lookup + searchsorted. Rewards use the SAME
validated equity model as the exact Nash anchor (SPEC 9).
"""
from __future__ import annotations

import numpy as np
import torch

from kalay.cards import native
from kalay.core.batch import TrajectoryBatch
from kalay.core.nets_transformer import TokenStreamBatch
from kalay.games.pushfold import PushFoldEnv
from kalay.games.pushfold_tokens import bb_slice, build_batch, btn_slice

N_CLASSES = 169


def install_token_batch(engine) -> None:
    """Pack all 338 slices into engine.token_batch (call once after build)."""
    btn, bb = [], []
    for i in range(N_CLASSES):
        hole = native.class_repr_cards(i)
        btn.append(btn_slice(hole))
        bb.append(bb_slice(hole))
    slices = btn + bb
    hero = [0] * N_CLASSES + [1] * N_CLASSES
    opp = [1] * N_CLASSES + [0] * N_CLASSES
    engine.token_batch = build_batch(slices, hero, opp, max_len=engine.cfg.max_len)


@torch.no_grad()
def slice_probs(engine) -> tuple[np.ndarray, np.ndarray]:
    """(btn_probs [169,2], bb_probs [169,2]) under the current policy."""
    engine.policy_net.eval()
    logits = engine.policy_net(engine.token_batch).cpu().numpy()
    engine.policy_net.train()
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return p[:N_CLASSES], p[N_CLASSES:]


def collect_pushfold_v3(engine, n_hands: int, rng: np.random.Generator) -> TrajectoryBatch:
    p_btn, p_bb = slice_probs(engine)
    deal0 = rng.integers(0, 52, size=(n_hands, 2))  # BTN hole (c1 != c2 handled below)
    ok = deal0[:, 0] != deal0[:, 1]
    while not ok.all():  # resample collisions
        deal0[~ok] = rng.integers(0, 52, size=((~ok).sum(), 2))
        ok = deal0[:, 0] != deal0[:, 1]
    remaining = []
    hole1 = np.empty((n_hands, 2), dtype=np.int64)
    for h in range(n_hands):  # BB hole distinct from BTN's two cards
        pool = [c for c in range(52) if c not in (deal0[h, 0], deal0[h, 1])]
        hole1[h] = rng.choice(pool, size=2, replace=False)
    cls0 = np.array([native.hand_index(a, b) for a, b in deal0])
    cls1 = np.array([native.hand_index(a, b) for a, b in hole1])

    cum0 = np.cumsum(p_btn[cls0], axis=1)
    cum0[:, -1] = 1.0
    a0 = (cum0 < rng.random(n_hands)[:, None]).sum(axis=1).clip(max=1)
    pushed = a0 == 1
    a1 = np.zeros(n_hands, dtype=np.int64)
    if pushed.any():
        cum1 = np.cumsum(p_bb[cls1[pushed]], axis=1)
        cum1[:, -1] = 1.0
        a1[pushed] = (cum1 < rng.random(int(pushed.sum()))[:, None]).sum(axis=1).clip(max=1)

    # rewards: fold paths exact; showdown via the anchor equity model
    rewards = np.zeros((n_hands, 2))
    fold0 = ~pushed
    rewards[fold0, 0] = -0.5
    rewards[fold0, 1] = 0.5
    fold1 = pushed & (a1 == 0)
    rewards[fold1, 0] = 1.0
    rewards[fold1, 1] = -1.0
    show = pushed & (a1 == 1)
    if show.any():
        eq = np.array([native.pushfold_solver().equity(int(i), int(j))
                       for i, j in zip(cls0[show], cls1[show])])
        r = 2.0 * 10.0 * eq - 10.0
        rewards[show, 0] = r
        rewards[show, 1] = -r

    # records: every hand yields a BTN row; pushed hands add a BB row
    rows = []
    for h in range(n_hands):
        rows.append((0, cls0[h], int(a0[h]), float(p_btn[cls0[h], a0[h]]),
                     float(rewards[h, 0])))
        if pushed[h]:
            rows.append((1, N_CLASSES + cls1[h], int(a1[h]),
                         float(p_bb[cls1[h], a1[h]]), float(rewards[h, 1])))
    player = np.array([r[0] for r in rows], dtype=np.int64)
    slice_ids = np.array([r[1] for r in rows], dtype=np.int64)
    act = np.array([r[2] for r in rows], dtype=np.int64)
    mu = np.array([r[3] for r in rows], dtype=np.float32)
    rew = np.array([r[4] for r in rows], dtype=np.float32)
    mask = np.ones((1, len(rows), 2), dtype=np.float32)
    batch = TrajectoryBatch(
        obs=torch.zeros(1, len(rows), 4),          # unused in the v3 path
        legal_mask=torch.as_tensor(mask),
        action_taken=torch.as_tensor(act).unsqueeze(0),
        behavior_prob=torch.as_tensor(mu).unsqueeze(0),
        terminal_reward=torch.as_tensor(rew).unsqueeze(0),
        hand_lengths=torch.ones(len(rows), dtype=torch.int64),
        player_id=torch.as_tensor(player),
        reward_scale=engine.cfg.reward_scale,
        slice_ids=torch.as_tensor(slice_ids).unsqueeze(0),
    )
    batch.validate(n_players=2)
    return batch


def class_strategies_v3(engine) -> tuple[np.ndarray, np.ndarray]:
    """Current-policy (push, call) probabilities over the 169 classes."""
    p_btn, p_bb = slice_probs(engine)
    return p_btn[:, 1].copy(), p_bb[:, 1].copy()


def anchor_avg_class_strategies_v3(engine) -> tuple[np.ndarray, np.ndarray]:
    """Anchor-average (Nash-average) class strategies for a v3 engine."""
    import copy

    net = copy.deepcopy(engine.policy_net).to("cpu").eval()
    snapshots = engine.anchor_state_dicts()
    acc_p = np.zeros(N_CLASSES)
    acc_q = np.zeros(N_CLASSES)
    cpu_batch = _to_cpu(engine.token_batch)
    with torch.no_grad():
        for sd in snapshots:
            net.load_state_dict(sd)
            logits = net(cpu_batch).numpy()
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            acc_p += p[:N_CLASSES, 1]
            acc_q += p[N_CLASSES:, 1]
    return acc_p / len(snapshots), acc_q / len(snapshots)


def _to_cpu(batch: TokenStreamBatch) -> TokenStreamBatch:
    return batch.to("cpu")
