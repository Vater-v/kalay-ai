"""Hypothesis probe: train one config on one game, print exploitability trajectory.

Usage:
  python scripts/probe.py --game kuhn --steps 15000 --seed 0 [--optimism 1.0] [--tag armB]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from kalay.core.collector import collect_batch
from kalay.core.engine import RNaDEngine
from kalay.core.gamestats import derive_engine_config
from kalay.eval.policy_avg import anchor_average_policy_fn
from kalay.eval.tabular import nash_conv, policy_value
from kalay.games.kuhn import KuhnPokerEnv
from kalay.games.leduc import LeducEnv
from kalay.games.pushfold import PushFoldEnv
from kalay.core.vec_leduc import collect_leduc_vec
from kalay.games.rps import RPSEnv

GAMES = {"rps": RPSEnv, "kuhn": KuhnPokerEnv, "leduc": LeducEnv,
         "pushfold": PushFoldEnv}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=GAMES, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimism", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--tau", type=float, default=None,
                    help="override the class tau (constant)")
    ap.add_argument("--sequential", action="store_true",
                    help="force the sequential collector for leduc")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    env_cls = GAMES[args.game]
    cfg, _ = derive_engine_config(env_cls, budget_steps=args.steps, seed=args.seed)
    cfg.optimism_beta = args.optimism
    if args.tau is not None:
        cfg.tau = args.tau
        cfg.tau_min = args.tau
    engine = RNaDEngine(cfg)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    log_every = max(args.steps // 5, 1)
    print(f"[{args.tag}] game={args.game} steps={args.steps} seed={args.seed} "
          f"optimism={args.optimism}", flush=True)
    for i in range(args.steps):
        batch = collect_batch(env_cls, engine.policy_probs, args.batch, rng,
                              cfg.reward_scale)
        m = engine.train_step(batch)
        if (i + 1) % log_every == 0 or i + 1 == args.steps:
            fn = anchor_average_policy_fn(engine)
            if args.game == "pushfold":
                from kalay.cards import native
                pp, qq = native.class_strategies(fn)
                np_, nq_, _ = native.nash_pushfold()
                nc = native.nashconv(pp, qq)
                extra = (f" tv_p {native.range_tv(pp, np_):.4f}"
                         f" tv_q {native.range_tv(qq, nq_):.4f}")
            else:
                nc = nash_conv(env_cls(), fn)
                extra = (f" EV {policy_value(env_cls(), fn, 0):+.4f}"
                         if args.game != "rps" else "")
            print(f"[{args.tag}] step {i+1:6d} | nc_avg {nc:7.4f}{extra} | "
                  f"tau {m['tau']:.3f} | phases {engine.phases} | "
                  f"rail {m['rail_fraction']:.2f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
