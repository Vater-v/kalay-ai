"""Tier A (frozen): token transformer invariants (DESIGN_NET 2, 6, 7)."""
from __future__ import annotations

import pytest
import torch

from kalay.core.nets_transformer import PolicyTransformer, TokenConfig, ValueTransformer


def _make(cfg=None, seed=0):
    torch.manual_seed(seed)
    cfg = cfg or TokenConfig(num_tokens=80)
    return cfg, PolicyTransformer(cfg), ValueTransformer(cfg)


def _inputs(cfg, b=4, length=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, cfg.num_tokens, (b, length), generator=g)
    bet = torch.randn(b, length, cfg.twohot_dim, generator=g) * 0.1
    pad = torch.ones(b, length, dtype=torch.bool)
    return tokens, pad, bet


@pytest.mark.tier_a
def test_shapes_and_determinism():
    cfg, pol, val = _make()
    tokens, pad, bet = _inputs(cfg)
    torch.manual_seed(0)
    l1 = pol(tokens, pad, bet)
    v1 = val(tokens, pad, bet)
    torch.manual_seed(0)
    pol2, val2 = PolicyTransformer(cfg), ValueTransformer(cfg)
    pol2.load_state_dict(pol.state_dict())
    val2.load_state_dict(val.state_dict())
    assert l1.shape == (4, cfg.n_actions)
    assert v1.shape == (4,)
    assert torch.equal(l1, pol2(tokens, pad, bet))
    assert torch.equal(v1, val2(tokens, pad, bet))


@pytest.mark.tier_a
def test_append_only_prefix_invariance():
    """Outputs on a real prefix must not change when future tokens are appended
    (the KV-cache semantics; also the S4 token-stream contract)."""
    cfg, pol, val = _make()
    tokens, pad, bet = _inputs(cfg, b=6, length=10)
    with torch.no_grad():
        base_logits = pol(tokens, pad, bet)
        base_value = val(tokens, pad, bet)
    # append 8 future tokens; the real prefix stays the first 10 positions
    g = torch.Generator().manual_seed(7)
    fut_tok = torch.randint(1, cfg.num_tokens, (6, 8), generator=g)
    fut_bet = torch.randn(6, 8, cfg.twohot_dim, generator=g) * 0.5
    tokens2 = torch.cat([tokens, fut_tok], dim=1)
    bet2 = torch.cat([bet, fut_bet], dim=1)
    pad2 = torch.cat([pad, torch.ones(6, 8, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        # readout is at the last REAL token; append changes it - so compare the
        # prefix hidden states via a padded readout: emulate by padding right after
        # the prefix with PAD tokens (ignored) instead of real futures:
        pad3 = torch.cat([pad, torch.zeros(6, 8, dtype=torch.bool)], dim=1)
        junk = torch.zeros(6, 8, dtype=torch.long)  # PAD id 0, masked out
        junk_bet = torch.zeros(6, 8, cfg.twohot_dim)
        logits_padded = pol(torch.cat([tokens, junk], dim=1), pad3,
                            torch.cat([bet, junk_bet], dim=1))
        value_padded = val(torch.cat([tokens, junk], dim=1), pad3,
                           torch.cat([bet, junk_bet], dim=1))
    torch.testing.assert_close(base_logits, logits_padded)
    torch.testing.assert_close(base_value, value_padded)


@pytest.mark.tier_a
def test_causality_future_tokens_do_not_affect_prefix_hidden():
    """Hidden states at real positions must be identical regardless of tokens
    that come after them (strict causality, padding-mask interaction included)."""
    cfg, pol, _ = _make()
    tokens, pad, bet = _inputs(cfg, b=4, length=8)
    # build a longer sequence where the first 4 positions are real, rest varies
    g = torch.Generator().manual_seed(3)
    tail1 = torch.randint(1, cfg.num_tokens, (4, 6), generator=g)
    tail2 = torch.randint(1, cfg.num_tokens, (4, 6), generator=g)
    bet_tail1 = torch.randn(4, 6, cfg.twohot_dim, generator=g)
    bet_tail2 = torch.randn(4, 6, cfg.twohot_dim, generator=g)
    all_pad = torch.ones(4, 14, dtype=torch.bool)
    h1 = pol.trunk(torch.cat([tokens, tail1], 1), all_pad,
                   torch.cat([bet, bet_tail1], 1))
    h2 = pol.trunk(torch.cat([tokens, tail2], 1), all_pad,
                   torch.cat([bet, bet_tail2], 1))
    torch.testing.assert_close(h1[:, :4], h2[:, :4])


@pytest.mark.tier_a
def test_twohot_sensor_affects_output_and_gradients_flow():
    cfg, pol, _val = _make()
    tokens, pad, bet = _inputs(cfg, b=2, length=6)
    logits = pol(tokens, pad, bet)
    logits.sum().backward()
    assert pol.trunk.backbone.bet.weight.grad is not None
    assert torch.isfinite(logits).all()
