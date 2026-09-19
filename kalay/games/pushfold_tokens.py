"""Token-stream adapter for Push-Fold (v3 pipeline pilot, DESIGN_NET v3).

Decision slices per seat:
  BTN: [DECISION, META, h1, h2]
  BB : [DECISION, META, h1, h2, ACT(push)]
Card ids use the kalay_rs convention (rank<<2 | suit) with S4 hand-locked
canonical suits: A(h)K(s) and A(d)K(c) produce identical ids, A(s)K(s) differs.
Hole tokens share one positional index (hand symmetry). META numerics are
invariant ratios only (no absolute chips). The push action carries the
Fourier-ready numeric vector [bet/pot, bet/stack, log(1+pot/eff), to_call/pot].
"""
from __future__ import annotations

import math

import numpy as np
import torch

from kalay.core.nets_transformer import ACTION, CARD, DECISION, META, TokenStreamBatch
from kalay.games.pushfold import PushFoldEnv

STACK_BB = 10.0
SB_BB = 0.5
BB_BLIND = 1.0


def canonical_hole(cards: tuple[int, int]) -> tuple[int, int]:
    """S4 hand-lock (SPEC 3 rule 1): rank-desc order; h1 -> suit 0, h2 -> suit 0
    (suited) or 1 (offsuit). Isomorphic hands collapse to identical ids."""
    c1, c2 = sorted(cards, key=lambda c: (-(c >> 2), -(c & 3)))
    suited = (c1 & 3) == (c2 & 3)
    return ((c1 >> 2) << 2, ((c2 >> 2) << 2) | (0 if suited else 1))


def btn_slice(hole: tuple[int, int]) -> dict:
    c1, c2 = canonical_hole(hole)
    pot = SB_BB + BB_BLIND
    meta_v = [pot / STACK_BB, 0.0, 2.0 / 2.0, 0.0]  # pot/stack, -, players, -
    return {
        "kind": [DECISION, META, CARD, CARD],
        "card": [0, 0, c1, c2],
        "act": [0, 0, 0, 0],
        "round": [0, 0, 0, 0],
        "seat": [0, 0, 0, 0],  # filled by caller (hero seat)
        "pos": [0, 1, 2, 2],   # hole tokens share index 2
        "v": [meta_v, meta_v, [0.0] * 4, [0.0] * 4],
    }


def bb_slice(hole: tuple[int, int]) -> dict:
    d = btn_slice(hole)
    # opponent shoved 10bb; BB facing the push
    bet = STACK_BB - SB_BB          # BTN adds 9.5 over his 0.5 blind
    pot = STACK_BB + BB_BLIND       # 11bb in the middle after the shove
    to_call = STACK_BB - BB_BLIND   # BB adds 9.0
    act_v = [bet / pot, bet / STACK_BB, math.log(1.0 + pot / STACK_BB), to_call / pot]
    d["kind"].append(ACTION)
    d["card"].append(0)
    d["act"].append(2)  # act type: Bet/Raise
    d["round"].append(0)
    d["seat"].append(0)  # filled by caller (opponent seat)
    d["pos"].append(3)
    d["v"].append(act_v)
    return d


def build_batch(slices: list[dict], hero_seat: list[int], opp_seat: list[int],
                max_len: int = 8) -> TokenStreamBatch:
    """Pack decision slices into a TokenStreamBatch (padding at the end)."""
    n = len(slices)
    kind = torch.zeros(n, max_len, dtype=torch.long)
    card = torch.zeros(n, max_len, dtype=torch.long)
    act = torch.zeros(n, max_len, dtype=torch.long)
    round_id = torch.zeros(n, max_len, dtype=torch.long)
    seat = torch.zeros(n, max_len, dtype=torch.long)
    pos_index = torch.zeros(n, max_len, dtype=torch.long)
    v_num = torch.zeros(n, max_len, 4, dtype=torch.float32)
    pad = torch.zeros(n, max_len, dtype=torch.bool)
    for i, s in enumerate(slices):
        L = len(s["kind"])
        assert L <= max_len
        kind[i, :L] = torch.tensor(s["kind"])
        card[i, :L] = torch.tensor(s["card"])
        act[i, :L] = torch.tensor(s["act"])
        round_id[i, :L] = torch.tensor(s["round"])
        pos_index[i, :L] = torch.tensor(s["pos"])
        v_num[i, :L] = torch.tensor(s["v"], dtype=torch.float32)
        pad[i, :L] = True
        # seat embeddings: META/DECISION/cards carry the hero seat; actions of
        # the opponent carry the opponent seat
        for j, k in enumerate(s["kind"]):
            seat[i, j] = hero_seat[i] if k != ACTION else opp_seat[i]
    return TokenStreamBatch(kind, card, act, round_id, seat, pos_index, v_num, pad)


def class_slices():
    """BTN and BB decision slices for all 169 canonical classes (eval helper)."""
    from kalay.cards.native import class_repr_cards

    btn, bb = [], []
    for i in range(169):
        hole = class_repr_cards(i)
        btn.append(btn_slice(hole))
        bb.append(bb_slice(hole))
    return btn, bb
