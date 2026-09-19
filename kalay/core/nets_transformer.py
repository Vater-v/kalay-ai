"""Token-sequence networks for the HU+ ladder (DESIGN_NET v1).

Scaffold behind the same logical interface as the MLPs: policy -> raw logits,
value -> scalar. Causal self-attention over event tokens; the append-only
semantics (outputs at a prefix do not change when future tokens are appended)
is the tier-A-tested invariant that licenses an incremental KV cache later.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TokenConfig:
    def __init__(self, num_tokens: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 256, max_len: int = 64,
                 n_actions: int = 12, twohot_dim: int = 32) -> None:
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.max_len = max_len
        self.n_actions = n_actions
        self.twohot_dim = twohot_dim


class _TokenTrunk(nn.Module):
    def __init__(self, cfg: TokenConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = nn.Module()  # param-group placeholder (wd applies here)
        self.backbone.tok = nn.Embedding(cfg.num_tokens, cfg.d_model)
        self.backbone.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.backbone.bet = nn.Linear(cfg.twohot_dim, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_ff,
            batch_first=True, norm_first=True, dropout=0.0,
        )  # dropout off: forward determinism is a tier-A invariant here
        self.backbone.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.backbone.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, tokens: torch.Tensor, pad: torch.Tensor,
                bet_sensor: torch.Tensor) -> torch.Tensor:
        """tokens [B, L] long (0 = PAD); pad [B, L] bool (True = real);
        bet_sensor [B, L, twohot] float. Returns hidden [B, L, d]."""
        b, length = tokens.shape
        assert length <= self.cfg.max_len
        pos = torch.arange(length, device=tokens.device)
        x = (self.backbone.tok(tokens)
             + self.backbone.pos(pos).unsqueeze(0)
             + self.backbone.bet(bet_sensor))
        causal = nn.Transformer.generate_square_subsequent_mask(length, device=tokens.device)
        key_pad = ~pad  # True marks positions to IGNORE
        h = self.backbone.encoder(x, mask=causal, src_key_padding_mask=key_pad)
        return self.backbone.norm(h)

    @staticmethod
    def last_real(h: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Hidden state at the last real token per row -> [B, d]."""
        lengths = pad.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(1, idx).squeeze(1)


class PolicyTransformer(nn.Module):
    def __init__(self, cfg: TokenConfig) -> None:
        super().__init__()
        self.trunk = _TokenTrunk(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.n_actions)

    def forward(self, tokens: torch.Tensor, pad: torch.Tensor,
                bet_sensor: torch.Tensor) -> torch.Tensor:
        h = self.trunk(tokens, pad, bet_sensor)
        return self.head(self.trunk.last_real(h, pad))  # [B, n_actions]


class ValueTransformer(nn.Module):
    def __init__(self, cfg: TokenConfig) -> None:
        super().__init__()
        self.trunk = _TokenTrunk(cfg)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, tokens: torch.Tensor, pad: torch.Tensor,
                bet_sensor: torch.Tensor) -> torch.Tensor:
        h = self.trunk(tokens, pad, bet_sensor)
        return self.head(self.trunk.last_real(h, pad)).squeeze(-1)  # [B]
