"""Vectorized Leduc collection: batch-step all hands as numpy arrays.

Mirrors kalay/games/leduc.py exactly (stake accounting, obs layout, masks,
chance semantics); tier A proves state-machine equivalence by driving both
implementations through identical (deal, action-sequence) pairs. One batched
forward per decision layer instead of per-decision Python stepping.
"""
from __future__ import annotations

import numpy as np

from kalay.core.batch import TrajectoryBatch, pack_subtrajectories

OBS_DIM = 43
MAX_HIST = 8
N_ACTIONS = 3
_HOLE_DEALS = np.array([(a, b) for a in range(6) for b in range(6) if a != b], dtype=np.int64)


class VecLeducState:
    """N parallel Leduc hands; arrays mirror LeducEnv fields."""

    def __init__(self, n: int, rng: np.random.Generator | None = None,
                 deals: np.ndarray | None = None) -> None:
        if deals is None and rng is None:
            raise ValueError("need rng or explicit deals")
        self.n = n
        self.hole = (
            _HOLE_DEALS[rng.integers(0, len(_HOLE_DEALS), n)]
            if deals is None else np.asarray(deals, dtype=np.int64).copy()
        )
        assert self.hole.shape == (n, 2)
        self.public = np.full(n, -1, dtype=np.int64)
        self.round_start = np.zeros(n, dtype=np.int64)
        self.hist = np.zeros((n, MAX_HIST), dtype=np.int64)
        self.hlen = np.zeros(n, dtype=np.int64)
        self.folded = np.full(n, -1, dtype=np.int64)
        self.done = np.zeros(n, dtype=bool)
        self.pending_pub = np.zeros(n, dtype=bool)
        self.revealed = np.zeros(n, dtype=bool)

    def actor(self) -> np.ndarray:
        """Acting player per hand (round 1 alternates from 0; round 2 from 1)."""
        return np.where(self.revealed,
                        (self.hlen - self.round_start + 1) % 2,
                        self.hlen % 2)

    def resolve_public(self, rng: np.random.Generator) -> None:
        m = self.pending_pub & ~self.done
        if not m.any():
            return
        idx = np.flatnonzero(m)
        remaining = np.zeros((len(idx), 4), dtype=np.int64)
        counts = np.zeros(len(idx), dtype=np.int64)
        for c in range(6):
            free = (self.hole[idx, 0] != c) & (self.hole[idx, 1] != c)
            free_rows = np.flatnonzero(free)
            if free_rows.size:
                remaining[free_rows, counts[free]] = c
                counts[free] += 1
        pick = rng.integers(0, 4, len(idx))
        self.public[idx] = remaining[np.arange(len(idx)), pick]
        self.round_start[idx] = self.hlen[idx]
        self.revealed[idx] = True
        self.pending_pub[idx] = False

    def raises_in_round(self) -> np.ndarray:
        rows = np.arange(self.n)
        c2 = np.cumsum(self.hist == 2, axis=1)
        total = c2[rows, np.maximum(self.hlen - 1, 0)]
        before = np.where(self.round_start > 0,
                          c2[rows, np.maximum(self.round_start - 1, 0)], 0)
        return total - before

    def legal_mask(self) -> np.ndarray:
        rows = np.arange(self.n)
        last = self.hist[rows, np.maximum(self.hlen - 1, 0)]
        facing = (self.hlen > self.round_start) & (last == 2)
        mask = np.zeros((self.n, N_ACTIONS), dtype=np.float32)
        mask[:, 1] = 1.0
        mask[facing, 0] = 1.0
        mask[self.raises_in_round() < 2, 2] = 1.0
        return mask

    def round_done(self) -> np.ndarray:
        rows = np.arange(self.n)
        rl = self.hlen - self.round_start
        last = self.hist[rows, np.maximum(self.hlen - 1, 0)]
        second = self.hist[rows, np.maximum(self.hlen - 2, 0)]
        return (rl >= 2) & (last == 1) & ((second == 1) | (self.raises_in_round() > 0))

    def obs(self, players: np.ndarray) -> np.ndarray:
        rows = np.arange(self.n)
        o = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        o[rows, self.hole[rows, players] // 2] = 1.0
        col = np.where(self.public == -1, 0, self.public // 2 + 1)
        o[rows, 3 + col] = 1.0
        o[rows, 7 + (self.public != -1).astype(np.int64)] = 1.0
        o[rows, 9 + players] = 1.0
        for s in range(MAX_HIST):
            valid = self.hlen > s
            if valid.any():
                o[valid, 11 + 4 * s + self.hist[valid, s] + 1] = 1.0
        return o

    def step(self, actions: np.ndarray) -> None:
        rows = np.arange(self.n)
        act = self.actor()
        folding = actions == 0
        self.folded = np.where(folding, act, self.folded)
        self.done |= folding
        placing = np.flatnonzero(~folding)
        self.hist[placing, self.hlen[placing]] = actions[placing]
        self.hlen[placing] += 1
        rd = self.round_done() & ~folding
        self.pending_pub |= rd & ~self.revealed
        self.done |= rd & self.revealed

    def rewards(self) -> np.ndarray:
        """Terminal rewards; stake accounting mirrors LeducEnv._investments."""
        rows = np.arange(self.n)
        inv = np.ones((self.n, 2))
        contrib = np.zeros((self.n, 2))
        stake = np.zeros(self.n)
        stopped = np.zeros(self.n, dtype=bool)
        for s in range(MAX_HIST):
            active = (self.hlen > s) & ~stopped
            if not active.any():
                break
            a = self.hist[:, s]
            p = np.where(self.revealed, (s - self.round_start + 1) % 2, s % 2)
            rnd1 = ~self.revealed | (s < self.round_start)
            bet = np.where(rnd1, 2.0, 4.0)
            newstop = active & (a == 0)
            stake = np.where(active & (a == 2), stake + bet, stake)
            pay = stake - contrib[rows, p]
            domask = active & (a != 0) & (pay > 0)
            contrib[rows, p] += np.where(domask, pay, 0.0)
            inv[rows, p] += np.where(domask, pay, 0.0)
            stopped |= newstop
        pot = inv.sum(axis=1)
        rew = np.zeros((self.n, 2))
        mfold = self.folded != -1
        if mfold.any():
            f = self.folded[mfold]
            rew[mfold, f] = -inv[mfold, f]
            rew[mfold, 1 - f] = pot[mfold] - inv[mfold, 1 - f]
        show = ~mfold
        if show.any():
            r0 = self.hole[show, 0] // 2
            r1 = self.hole[show, 1] // 2
            rp = self.public[show] // 2
            s0 = r0 + 3 * (r0 == rp)
            s1 = r1 + 3 * (r1 == rp)
            v0 = np.where(s0 > s1, pot[show] - inv[show, 0],
                          np.where(s0 < s1, -inv[show, 0], pot[show] / 2 - inv[show, 0]))
            rew[show, 0] = v0
            rew[show, 1] = -v0
        return rew


def collect_leduc_vec(engine, n_hands: int, rng: np.random.Generator) -> TrajectoryBatch:
    """Vectorized player-centric collection for Leduc (obs recorded with masks)."""
    st = VecLeducState(n_hands, rng)
    hands_l: list[np.ndarray] = []
    players_l: list[np.ndarray] = []
    obs_l: list[np.ndarray] = []
    mask_l: list[np.ndarray] = []
    act_l: list[np.ndarray] = []
    mu_l: list[np.ndarray] = []
    while True:
        st.resolve_public(rng)
        live = ~st.done & ~st.pending_pub
        if not live.any():
            break
        players = st.actor()
        obs = st.obs(players)
        mask = st.legal_mask()
        probs = engine.policy_probs_np_batch(obs, mask).astype(np.float64)
        cum = np.cumsum(probs, axis=1)
        cum[:, -1] = 1.0
        r = rng.random(n_hands)
        actions = (cum < r[:, None]).sum(axis=1).clip(max=N_ACTIONS - 1)
        mu = probs[np.arange(n_hands), actions]
        live_rows = np.flatnonzero(live)
        hands_l.append(live_rows)
        players_l.append(players[live_rows])
        obs_l.append(obs[live_rows])
        mask_l.append(mask[live_rows])
        act_l.append(actions[live_rows])
        mu_l.append(mu[live_rows])
        st.step(actions)
    rew = st.rewards()
    hand = np.concatenate(hands_l)
    player = np.concatenate(players_l)
    obs_all = np.concatenate(obs_l)
    mask_all = np.concatenate(mask_l)
    act_all = np.concatenate(act_l)
    mu_all = np.concatenate(mu_l)
    # stable grouping by (hand, player) preserving chronological order
    key = hand * 2 + player
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    records = []
    for h in range(n_hands):
        for p in range(2):
            k = h * 2 + p
            lo = np.searchsorted(key_sorted, k, side="left")
            hi = np.searchsorted(key_sorted, k, side="right")
            if lo == hi:
                continue
            sel = order[lo:hi]
            records.append({
                "obs": obs_all[sel],
                "mask": mask_all[sel],
                "act": act_all[sel],
                "mu": mu_all[sel].astype(np.float32),
                "reward": float(rew[h, p]),
                "player": p,
            })
    return pack_subtrajectories(records, engine.cfg.reward_scale)
