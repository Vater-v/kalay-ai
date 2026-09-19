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
