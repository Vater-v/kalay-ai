"""Exact tabular best response and NashConv for small games (SPEC principle 3).

Correct for imperfect information: the best response commits to one action per
INFoset, not per hidden state. Implemented via counterfactual values: for each
infoset I of the BR player and action a, cf(I, a) = sum over states s in I of
opponent-and-chance reach(s) * value(s after a); infosets are resolved bottom-up
(deepest first), which makes the per-infoset greedy argmax exact.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from kalay.games.base import GameEnv

PolicyFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _enumerate_infosets(env: GameEnv, br_player: int, policy_fn: PolicyFn):
    """Group BR decision states by infoset key; returns (groups, depth, legal)."""
    groups: dict[bytes, list[tuple[GameEnv, float]]] = {}
    depth: dict[bytes, int] = {}
    legal: dict[bytes, np.ndarray] = {}

    def walk(e: GameEnv, reach: float, d: int) -> None:
        if e.is_chance():
            probs = e.chance_probs()
            for i in range(len(probs)):
                child = e.clone()
                child.step_chance(i)
                walk(child, reach * float(probs[i]), d)
            return
        if e.is_terminal():
            return
        pid = e.current_player()
        mask = e.legal_actions_mask()
        if pid == br_player:
            key = e.obs(pid).tobytes() + mask.tobytes()
            groups.setdefault(key, []).append((e.clone(), reach))
            depth[key] = max(depth.get(key, 0), d)
            legal.setdefault(key, mask)
            for a in range(len(mask)):
                if mask[a] == 0:
                    continue
                child = e.clone()
                child.step(a)
                walk(child, reach, d + 1)  # BR reach is not counted (counterfactual)
        else:
            probs = np.asarray(policy_fn(e.obs(pid), mask), dtype=np.float64)
            probs = probs / probs.sum()
            for a in range(len(mask)):
                if probs[a] > 0:
                    child = e.clone()
                    child.step(a)
                    walk(child, reach * probs[a], d + 1)

    walk(env, 1.0, 0)
    return groups, depth, legal


def _subtree_value(
    env: GameEnv, br_player: int, policy_fn: PolicyFn, br_action: dict[bytes, int]
) -> float:
    if env.is_chance():
        probs = env.chance_probs()
        total = 0.0
        for i in range(len(probs)):
            child = env.clone()
            child.step_chance(i)
            total += probs[i] * _subtree_value(child, br_player, policy_fn, br_action)
        return total
    if env.is_terminal():
        return float(env.rewards()[br_player])
    pid = env.current_player()
    mask = env.legal_actions_mask()
    if pid == br_player:
        key = env.obs(pid).tobytes() + mask.tobytes()
        child = env.clone()
        child.step(br_action[key])
        return _subtree_value(child, br_player, policy_fn, br_action)
    probs = np.asarray(policy_fn(env.obs(pid), mask), dtype=np.float64)
    probs = probs / probs.sum()
    total = 0.0
    for a in range(len(mask)):
        if probs[a] > 0:
            child = env.clone()
            child.step(a)
            total += probs[a] * _subtree_value(child, br_player, policy_fn, br_action)
    return total


def best_response_value(env: GameEnv, policy_fn: PolicyFn, br_player: int) -> float:
    groups, depth, legal = _enumerate_infosets(env, br_player, policy_fn)
    br_action: dict[bytes, int] = {}
    for key in sorted(groups, key=lambda k: -depth[k]):  # deepest infosets first
        state0, _ = groups[key][0]
        mask = legal[key]
        best_a, best_cf = 0, -np.inf
        for a in range(len(mask)):
            if mask[a] == 0:
                continue
            cf = 0.0
            for s, reach in groups[key]:
                child = s.clone()
                child.step(a)
                cf += reach * _subtree_value(child, br_player, policy_fn, br_action)
            if cf > best_cf:
                best_a, best_cf = a, cf
        br_action[key] = best_a
    return _subtree_value(env, br_player, policy_fn, br_action)


def nash_conv(env: GameEnv, policy_fn: PolicyFn) -> float:
    """Sum over players of best-response values (2p zero-sum: 0 at Nash)."""
    return sum(best_response_value(env, policy_fn, p) for p in range(env.n_players))


def policy_value(env: GameEnv, policy_fn: PolicyFn, player: int) -> float:
    """Expected value of `player` when everyone follows `policy_fn` (self-play)."""

    def value(e: GameEnv) -> float:
        if e.is_chance():
            probs = e.chance_probs()
            return sum(
                probs[i] * value(_cloned_step_chance(e, i)) for i in range(len(probs))
            )
        if e.is_terminal():
            return float(e.rewards()[player])
        pid = e.current_player()
        mask = e.legal_actions_mask()
        probs = np.asarray(policy_fn(e.obs(pid), mask), dtype=np.float64)
        probs = probs / probs.sum()
        return sum(
            probs[a] * value(_cloned_step(e, a))
            for a in range(len(mask))
            if probs[a] > 0
        )

    return value(env)


def _cloned_step_chance(e: GameEnv, i: int) -> GameEnv:
    child = e.clone()
    child.step_chance(i)
    return child


def _cloned_step(e: GameEnv, a: int) -> GameEnv:
    child = e.clone()
    child.step(a)
    return child
