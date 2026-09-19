"""Token transformer v3 (DESIGN_NET v3): [DECISION] + full bidirectional.

Owner's take adopted over block-causal v2: each decision is its own slice
[DECISION, META, context...]; attention within a slice is fully bidirectional
(cards/actions/meta all see each other), the DECISION token accumulates the
hand summary and is the ONLY readout (policy/value heads hang off it).
This matches the player-centric [T, B] batch exactly (one row = one infoset),
isolates gradients per decision (critical for NeuRD forces and any future
OMWU/optimistic retrial - the legacy project's notes), and gives richer
intermediate context representations (blocker effects digested in the cards).
Accepted tradeoff: no cross-decision KV reuse; ~3-5x learner FLOPs vs a packed
causal scheme - irrelevant while collection dominates (measured).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

PAD, META, CARD, ACTION, DECISION = 0, 1, 2, 3, 4
N_KINDS = 5


class TokenConfigV3:
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 256, max_len: int = 64, n_cards: int = 52,
                 n_act_types: int = 3, n_rounds: int = 4, n_seats: int = 3,
                 n_num: int = 4, k_freq: int = 8, n_actions: int = 12) -> None:
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.max_len = max_len
        self.n_cards = n_cards
        self.n_act_types = n_act_types
        self.n_rounds = n_rounds
        self.n_seats = n_seats
        self.n_num = n_num
        self.k_freq = k_freq
        self.n_actions = n_actions


class FourierNumerics(nn.Module):
    """gamma(v) = [sin(2^j pi v), cos(2^j pi v)] per feature -> Linear -> d."""

    def __init__(self, cfg: TokenConfigV3) -> None:
        super().__init__()
        self.register_buffer(
            "freqs", torch.tensor([2.0 ** j * math.pi for j in range(cfg.k_freq)]),
        )
        self.proj = nn.Linear(cfg.n_num * 2 * cfg.k_freq, cfg.d_model)

    def forward(self, v: torch.Tensor) -> torch.Tensor:  # [B,L,F] -> [B,L,D]
        ang = v.unsqueeze(-1) * self.freqs  # [B, L, F, K]
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(feats.flatten(start_dim=-2))


class TokenStreamBatch:
    """One decision per row: [DECISION, META, context...].

    DECISION sits FIRST: fixed positional index (0) - the readout slot never
    entangles hand depth into its own position; query-before-content ([CLS]
    pattern); trivial extraction. Owner's directive after questioning the
    end placement. Training cap (owner's deliberate policy): hands with more
    than 16 * n_players actions are SKIPPED entirely (not trained on).
    """

    def __init__(self, kind, card, act_type, round_id, seat, pos_index,
                 v_num, pad) -> None:
        self.kind = kind            # [B, L] long in {PAD,META,CARD,ACTION,DECISION}
        self.card = card            # [B, L] long
        self.act_type = act_type    # [B, L] long
        self.round_id = round_id    # [B, L] long
        self.seat = seat            # [B, L] long
        self.pos_index = pos_index  # [B, L] long (hole cards share one index)
        self.v_num = v_num          # [B, L, n_num] float, invariant ratios
        self.pad = pad              # [B, L] bool (True = real)

    def to(self, device):
        for name in ("kind", "card", "act_type", "round_id", "seat", "pos_index"):
            setattr(self, name, getattr(self, name).to(device))
        self.v_num = self.v_num.to(device)
        self.pad = self.pad.to(device)
        return self


class _TrunkV3(nn.Module):
    def __init__(self, cfg: TokenConfigV3) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = nn.Module()
        self.backbone.kind = nn.Embedding(N_KINDS, cfg.d_model)
        self.backbone.card = nn.Embedding(cfg.n_cards, cfg.d_model)
        self.backbone.act = nn.Embedding(cfg.n_act_types, cfg.d_model)
        self.backbone.round = nn.Embedding(cfg.n_rounds, cfg.d_model)
        self.backbone.seat = nn.Embedding(cfg.n_seats, cfg.d_model)
        self.backbone.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.backbone.numeric = FourierNumerics(cfg)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_ff,
            batch_first=True, norm_first=True, dropout=0.0,
        )  # dropout off: forward determinism is a tier-A invariant
        self.backbone.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.backbone.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, batch: TokenStreamBatch) -> torch.Tensor:
        bb = self.backbone
        is_card = (batch.kind == CARD).unsqueeze(-1)
        token_id_emb = torch.where(is_card, bb.card(batch.card),
                                   bb.act(batch.act_type))
        x = (bb.kind(batch.kind)
             + token_id_emb
             + bb.round(batch.round_id)
             + bb.seat(batch.seat)
             + bb.pos(batch.pos_index)
             + bb.numeric(batch.v_num))
        # full bidirectional attention within the slice; only pads are masked
        h = bb.encoder(x, src_key_padding_mask=~batch.pad)
        return bb.norm(h)


def decision_hidden(h: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Hidden of the DECISION token (position 0) -> [B, D]."""
    return h[:, 0]


class PolicyTransformerV3(nn.Module):
    def __init__(self, cfg: TokenConfigV3) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = _TrunkV3(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.n_actions)

    def forward(self, batch: TokenStreamBatch) -> torch.Tensor:
        h = self.trunk(batch)
        return self.head(decision_hidden(h, batch.pad))  # [B, n_actions]


class ValueTransformerV3(nn.Module):
    def __init__(self, cfg: TokenConfigV3) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = _TrunkV3(cfg)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, batch: TokenStreamBatch) -> torch.Tensor:
        h = self.trunk(batch)
        return self.head(decision_hidden(h, batch.pad)).squeeze(-1)  # [B]
