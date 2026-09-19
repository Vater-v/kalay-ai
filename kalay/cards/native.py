"""Native bindings: kalay_rs (Rust evaluator + exact push-fold engine).

The equity table is a generated artifact (fixed seed, documented MC precision);
tier A validates (a) the Rust evaluator against an independent Python reference
evaluator and (b) table entries against exact full-board enumeration.
"""
from __future__ import annotations

import functools
import pathlib

import numpy as np

try:
    import kalay_rs
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "kalay_rs not installed: cd rs && maturin build --release -i python && "
        "pip install target/wheels/*.whl"
    ) from exc

DATA_DIR = pathlib.Path(__file__).parent / "data"
EQUITY_TABLE_1M = DATA_DIR / "pushfold_equity_1m.bin"

N_CLASSES = 169


def hand_index(c1: int, c2: int) -> int:
    return kalay_rs.hand_169(c1, c2)


def hand_name(idx: int) -> str:
    return kalay_rs.hand_169_name_py(idx)


def eval_river7(cards) -> int:
    return kalay_rs.eval_river7(list(cards))


@functools.lru_cache(maxsize=1)
def pushfold_solver(
    stack_bb: float = 10.0, sb: float = 0.5, bb: float = 1.0,
    equity_path: str | None = None,
):
    path = equity_path or str(EQUITY_TABLE_1M)
    return kalay_rs.PushFoldSolver(stack_bb, sb, bb, path)


@functools.lru_cache(maxsize=1)
def nash_pushfold(iterations: int = 200_000):
    solver = pushfold_solver()
    p, q, value = solver.nash(iterations)
    return np.asarray(p), np.asarray(q), value


@functools.lru_cache(maxsize=1)
def equity_table() -> np.ndarray:
    table = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            table[i, j] = pushfold_solver().equity(i, j)
    return table


def nashconv(push: np.ndarray, call: np.ndarray) -> float:
    return pushfold_solver().exploitability(
        np.ascontiguousarray(push, dtype=np.float32),
        np.ascontiguousarray(call, dtype=np.float32),
    )


def class_repr_cards(idx: int) -> tuple[int, int]:
    """Representative combo of class idx (matches kalay_rs hand_169_to_cards)."""
    name = hand_name(idx)
    r1 = "23456789TJQKA".index(name[0])
    r2 = "23456789TJQKA".index(name[1])
    suited = len(name) > 2 and name[2] == "s"
    hi, lo = max(r1, r2), min(r1, r2)
    if suited:
        return (hi << 2, lo << 2)
    return (hi << 2, (lo << 2) | 1)


def class_strategies(policy_fn) -> tuple[np.ndarray, np.ndarray]:
    """(push_probs[169], call_probs[169]) of `policy_fn` over the 169 classes."""
    from kalay.games.pushfold import PushFoldEnv

    mask = np.ones(2, dtype=np.float32)
    p = np.empty(N_CLASSES)
    q = np.empty(N_CLASSES)
    env = PushFoldEnv()
    for i in range(N_CLASSES):
        hole = class_repr_cards(i)
        env.set_deal(hole, (0, 1))
        env._stage = 1
        p[i] = float(np.asarray(policy_fn(env.obs(0), mask))[1])
        env2 = PushFoldEnv()
        env2.set_deal((0, 1), hole)
        env2.step(1)  # BTN pushed; BB to act
        q[i] = float(np.asarray(policy_fn(env2.obs(1), mask))[1])
    return p, q


def range_tv(agent: np.ndarray, nash: np.ndarray) -> float:
    """Prior-weighted total variation distance between two class ranges."""
    prior = np.array([pushfold_solver().prior(i) for i in range(N_CLASSES)])
    return float((prior * np.abs(agent - nash)).sum())
