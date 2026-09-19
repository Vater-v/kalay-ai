"""Game analysis and derived engine configuration (SPEC section 6, "derived profile").

Nothing in the toy ladder is hand-tuned per game:
  - T, reward_scale, infoset counts: measured exactly by DFS enumeration;
  - mu_floor: closed-form softmax floor under the Z threshold;
  - anchor cadence: online drift trigger (accumulated KL to the anchor);
  - tau decay: derived from the training budget so tau reaches tau_min on time.

For games too large to enumerate (full HUNL), the adapter supplies the same stats
analytically; the derivation below does not change.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from kalay.core.engine import EngineConfig
from kalay.games.base import GameEnv


@dataclass(frozen=True)
class GameStats:
    n_players: int
    n_actions: int
    obs_dim: int
    n_infosets: int                     # distinct (player, observation) decision points
    max_decisions_per_player: int       # T
    avg_decisions_per_player: float     # per terminal path
    max_abs_reward: float


def analyze_game(make_env: Callable[[], GameEnv]) -> GameStats:
    """Exact statistics by full enumeration (episodic games of toy scale)."""
    root = make_env()
    infosets: set[tuple[int, bytes]] = set()
    max_depth = 0
    sum_dec = [0] * root.n_players
    n_terminals = 0
    max_abs_r = 0.0

    def walk(e: GameEnv, dec_counts: list[int]) -> None:
        nonlocal max_depth, n_terminals, max_abs_r
        if e.is_chance():
            n = len(e.chance_probs())
            for i in range(n):
                child = e.clone()
                child.step_chance(i)
                walk(child, dec_counts)
            return
        if e.is_terminal():
            rewards = e.rewards()
            max_abs_r = max(max_abs_r, float(abs(rewards).max()))
            n_terminals += 1
            for p in range(e.n_players):
                sum_dec[p] += dec_counts[p]
            return
        pid = e.current_player()
        infosets.add((pid, e.obs(pid).tobytes()))
        counts = list(dec_counts)
        counts[pid] += 1
        max_depth = max(max_depth, counts[pid])
        mask = e.legal_actions_mask()
        for a in range(len(mask)):
            if mask[a] == 0:
                continue
            child = e.clone()
            child.step(a)
            walk(child, counts)

    walk(root, [0] * root.n_players)
    avg = (sum(sum_dec) / (n_terminals * root.n_players)) if n_terminals else 0.0
    return GameStats(
        n_players=root.n_players,
        n_actions=root.n_actions,
        obs_dim=root.obs_dim,
        n_infosets=len(infosets),
        max_decisions_per_player=max_depth,
        avg_decisions_per_player=avg,
        max_abs_reward=max_abs_r,
    )


def pi_min_bound(n_legal: int, z_thresh: float) -> float:
    """Closed-form minimal legal-action probability under the Z threshold."""
    ez, emz = math.exp(z_thresh), math.exp(-z_thresh)
    return emz / ((n_legal - 1) * ez + emz)


# Class-level constants (NOT per-game): the only empirical calibration of the class.
# Calibrated once on the toy class (RPS+Kuhn probes; tier B seeds {0,1,2} guard):
#   - CONSTANT tau (anneal disabled): every probe that drove tau toward ~0 lost
#     the QRE stabilization and the anchor-average degraded (measured on both
#     games). Refinement comes from Nash-averaging over anchor phases, not from
#     small tau at these budgets. Revisit only at HUNL scale (SPEC revision).
#   - lr_policy=1e-3: 3e-4 left Kuhn exploitability 5-15x higher at equal budget.
CLASS_CONSTANTS = dict(
    tau0=0.3,
    tau_min=0.3,            # == tau0: constant regularizer (see note above)
    z_thresh=3.0,
    lr_policy=1e-3,
    lr_value=1e-3,
    anchor_kl_drift=1.0,   # nats of accumulated KL(pi || pi_ref) per anchor phase
)


def derive_engine_config(
    make_env: Callable[[], GameEnv],
    *,
    budget_steps: int,
    seed: int = 0,
    **overrides,
) -> tuple[EngineConfig, GameStats]:
    """Build the engine config for a game from measured stats + class constants."""
    stats = analyze_game(make_env)
    cfg = EngineConfig(
        obs_dim=stats.obs_dim,
        n_actions=stats.n_actions,
        n_players=stats.n_players,
        tau=CLASS_CONSTANTS["tau0"],
        tau_min=CLASS_CONSTANTS["tau_min"],
        z_thresh=CLASS_CONSTANTS["z_thresh"],
        lr_policy=CLASS_CONSTANTS["lr_policy"],
        lr_value=CLASS_CONSTANTS["lr_value"],
        reward_scale=stats.max_abs_reward,
        mu_floor=pi_min_bound(stats.n_actions, CLASS_CONSTANTS["z_thresh"]),
        anchor_mode="drift",
        anchor_kl_drift=CLASS_CONSTANTS["anchor_kl_drift"],
        planned_steps=budget_steps,
        seed=seed,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg, stats
