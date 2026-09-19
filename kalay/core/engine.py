"""R-NaD engine: one train step per SPEC 5.2-5.6 (v6.1).

Order: value step -> policy step -> anchor bookkeeping. The anchor is a hard copy
every `anchor_period` steps (documented simplification vs the reference EMA+alpha).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from kalay.core.batch import TrajectoryBatch
from kalay.core.neurd import masked_softmax, neurd_policy_loss
from kalay.core.nets import PolicyMLP, ValueMLP
from kalay.core.vtrace import bootstrap_next_value, decision_mask, importance_coeffs, vtrace


@dataclass
class EngineConfig:
    obs_dim: int
    n_actions: int
    n_players: int = 2
    hidden: int = 64
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    tau: float = 0.1
    tau_min: float = 0.02
    tau_decay: float = 0.97
    anchor_period: int = 1000
    z_thresh: float = 3.0
    a_max: float = 100.0
    mu_floor: float = 1e-3
    gamma: float = 1.0
    rho_clip: float = 1.0
    c_clip: float = 1.0
    grad_clip: float = 1.0
    reward_scale: float = 1.0
    avg_window: int = 12  # anchor snapshots in the Nash-average window
    anchor_mode: str = "fixed"            # "fixed" (every anchor_period) | "drift"
    anchor_kl_drift: float = 1.0          # legacy drift trigger (superseded by three-way)
    anchor_cum_kl: float = 3.0            # drift mode: high-water accumulated KL per phase
    anchor_settle_steps: int = 300        # drift mode: min phase length for settle trigger
    anchor_cool_tol: float = 0.05         # drift mode: |EMA_fast - EMA_slow| rel. tolerance
    optimism_beta: float = 0.0            # optimistic-gradient extrapolation on policy grads
    planned_steps: int | None = None      # for adaptive tau annealing
    seed: int = 0
    device: str = "cpu"


class RNaDEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.policy_net = PolicyMLP(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.value_net = ValueMLP(cfg.obs_dim, cfg.hidden).to(self.device)
        self.ref_net = copy.deepcopy(self.policy_net)
        for p in self.ref_net.parameters():
            p.requires_grad_(False)
        self.opt_policy = torch.optim.AdamW(
            [
                {"params": self.policy_net.backbone.parameters(), "weight_decay": 1e-2},
                {"params": self.policy_net.head.parameters(), "weight_decay": 0.0},
            ],
            lr=cfg.lr_policy,
        )
        self.opt_value = torch.optim.AdamW(
            [
                {"params": self.value_net.backbone.parameters(), "weight_decay": 1e-2},
                {"params": self.value_net.head.parameters(), "weight_decay": 0.0},
            ],
            lr=cfg.lr_value,
        )
        self.tau = cfg.tau
        self.step_count = 0
        self.phases = 0
        self._kl_since_anchor = 0.0
        self._kl_fast = 0.0
        self._kl_slow = 0.0
        self._last_anchor_step = 0
        self._prev_policy_grads: dict[int, torch.Tensor] = {}
        self._anchor_state_dicts: list[dict] = [self._snapshot()]

    # --- public API ---
    def train_step(self, batch: TrajectoryBatch) -> dict:
        batch.validate(n_players=self.cfg.n_players)
        obs = batch.obs.to(self.device)
        mask = batch.legal_mask.to(self.device)
        act = batch.action_taken.to(self.device)
        mu = batch.behavior_prob.to(self.device)
        rew = batch.terminal_reward.to(self.device)
        lengths = batch.hand_lengths.to(self.device)
        T, B, _ = mask.shape

        raw_logits = self.policy_net(obs)
        pi = masked_softmax(raw_logits, mask)
        with torch.no_grad():
            pi_ref = masked_softmax(self.ref_net(obs), mask)
        v_preds = self.value_net(obs)

        dec_mask = decision_mask(lengths, T).to(self.device)
        num_valid = dec_mask.sum().clamp(min=1.0)

        with torch.no_grad():
            log_pi_c = pi.clamp(min=1e-9).log()
            log_ref_c = pi_ref.clamp(min=1e-9).log()
            kl_scalar = (pi * (log_pi_c - log_ref_c)).sum(dim=-1)  # role of eta_reg_entropy
            r_norm = rew / self.cfg.reward_scale
            r_reg = -self.tau * kl_scalar + r_norm

            pi_taken = pi.gather(-1, act.unsqueeze(-1)).squeeze(-1)
            rho, c = importance_coeffs(pi_taken, mu, self.cfg.rho_clip, self.cfg.c_clip)
            v_targets = vtrace(v_preds, r_reg, rho, c, lengths, self.cfg.gamma)
            next_v = bootstrap_next_value(v_targets, lengths)
            td = r_norm + self.cfg.gamma * next_v - v_preds  # PLAIN reward (SPEC S5)

        # value step
        loss_value = 0.5 * torch.where(
            dec_mask == 1.0, (v_preds - v_targets) ** 2, torch.zeros_like(v_preds)
        ).sum() / num_valid
        self.opt_value.zero_grad(set_to_none=True)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.grad_clip)
        self.opt_value.step()

        # policy step (NeuRD with per-action regularizer)
        loss_policy, diags = neurd_policy_loss(
            raw_logits, mask, pi, pi_ref, v_preds.detach(), td, act, mu,
            dec_mask, num_valid, self.tau, self.cfg.z_thresh, self.cfg.a_max,
            self.cfg.mu_floor,
        )
        self.opt_policy.zero_grad(set_to_none=True)
        loss_policy.backward()
        if self.cfg.optimism_beta > 0.0:
            # optimistic gradient: (1+beta)*g_t - beta*g_{t-1}; raw grads are stashed
            # pre-correction, so consecutive steps see true gradient differences.
            with torch.no_grad():
                for p in self.policy_net.parameters():
                    if p.grad is None:
                        continue
                    snap = p.grad.detach().clone()
                    prev = self._prev_policy_grads.get(id(p))
                    if prev is not None:
                        p.grad.add_(self.cfg.optimism_beta * (p.grad - prev))
                    self._prev_policy_grads[id(p)] = snap
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_clip)
        self.opt_policy.step()

        self.step_count += 1
        metrics = {
            "loss_value": float(loss_value),
            "loss_policy": float(loss_policy),
            "tau": self.tau,
            "mean_kl_ref": float(kl_scalar[dec_mask == 1.0].mean()) if dec_mask.any() else 0.0,
            **diags.__dict__,
        }
        self._maybe_update_anchor(metrics["mean_kl_ref"])
        return metrics

    def policy_probs(self, obs, legal_mask) -> "torch.Tensor":
        """Single-observation probabilities under the current policy."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(legal_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = masked_softmax(self.policy_net(obs_t), mask_t).squeeze(0)
        return probs

    def anchor_state_dicts(self, window: int | None = None) -> list[dict]:
        """Anchor snapshots (Nash-average window) including the current parameters."""
        window = self.cfg.avg_window if window is None else window
        return self._anchor_state_dicts[-window:] + [self._snapshot()]

    # --- internals ---
    def _snapshot(self) -> dict:
        return {
            k: v.detach().clone().cpu() for k, v in self.policy_net.state_dict().items()
        }

    def _maybe_update_anchor(self, mean_kl: float) -> None:
        """Three-way phase trigger (two-timescale discipline, SPEC 5.6).

        1. cumulative drift: the policy moved a lot since the anchor (cyclic games);
        2. settle: the drift RATE has cooled (fast/slow EMA converged) — the inner
           loop has done its immediate work; re-anchoring earlier destroys progress
           in convergent games (measured on Kuhn: nc_avg 0.20 vs 0.05 at 6k steps);
        3. hard cap as a backstop.
        """
        self._kl_since_anchor += mean_kl
        self._kl_fast = 0.90 * self._kl_fast + 0.10 * mean_kl
        self._kl_slow = 0.99 * self._kl_slow + 0.01 * mean_kl
        steps_since = self.step_count - self._last_anchor_step
        if self.cfg.anchor_mode == "drift":
            cooled = abs(self._kl_fast - self._kl_slow) <= self.cfg.anchor_cool_tol * max(
                abs(self._kl_slow), 1e-9
            )
            settled = steps_since >= self.cfg.anchor_settle_steps and cooled
            triggered = (
                self._kl_since_anchor >= self.cfg.anchor_cum_kl
                or settled
                or steps_since >= 4 * self.cfg.anchor_period
            )
        else:
            triggered = self.step_count % self.cfg.anchor_period == 0
        if triggered:
            self.ref_net.load_state_dict(self.policy_net.state_dict())
            self._anchor_state_dicts.append(self._snapshot())
            self._last_anchor_step = self.step_count
            self._kl_since_anchor = 0.0
            self._kl_fast = 0.0
            self._kl_slow = 0.0
            self.phases += 1
            self.tau = self._next_tau()

    def _next_tau(self) -> float:
        """Adaptive anneal: reach tau_min by the planned budget given observed phases."""
        if self.cfg.planned_steps is None:
            return max(self.cfg.tau_min, self.tau * self.cfg.tau_decay)
        frac = min(self.step_count / max(self.cfg.planned_steps, 1), 0.98)
        expected_total = max(self.phases / max(frac, 1e-3), self.phases + 1)
        remaining = max(expected_total - self.phases, 1)
        return max(self.cfg.tau_min, self.tau * (self.cfg.tau_min / self.tau) ** (1.0 / remaining))
