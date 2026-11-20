"""Signal Temporal Logic (STL) over discrete, finite traces, with quantitative (robustness) semantics.

A trace maps signal names to equal-length numpy arrays; index t is the step (0.5 s in RecoverDrive). Robustness is
positive when the formula holds and negative when it is violated; its size says by how much. Windows that run past the
end of the trace are cut to the available samples, and `check_always` only evaluates the obligations whose whole window
fits in the trace, so a short trace never counts as a violation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np


@dataclass(frozen=True)
class Pred:
    signal: str
    op: Literal[">=", "<="]
    c: float


@dataclass(frozen=True)
class Not:
    phi: Formula


@dataclass(frozen=True, init=False)
class And:
    args: tuple

    def __init__(self, *args: Formula):
        object.__setattr__(self, "args", tuple(args))


@dataclass(frozen=True, init=False)
class Or:
    args: tuple

    def __init__(self, *args: Formula):
        object.__setattr__(self, "args", tuple(args))


@dataclass(frozen=True)
class Implies:
    lhs: Formula
    rhs: Formula


@dataclass(frozen=True)
class Always:
    a: int
    b: int
    phi: Formula


@dataclass(frozen=True)
class Eventually:
    a: int
    b: int
    phi: Formula


Formula = Union[Pred, Not, And, Or, Implies, Always, Eventually]  # noqa: UP007 - used at runtime in isinstance


def _length(trace: Mapping[str, np.ndarray]) -> int:
    return min((len(v) for v in trace.values()), default=0)


def rho(phi: Formula, trace: Mapping[str, np.ndarray], t: int) -> float:
    """Robustness of `phi` at step t."""
    if isinstance(phi, Pred):
        x = float(trace[phi.signal][t])
        return x - phi.c if phi.op == ">=" else phi.c - x
    if isinstance(phi, Not):
        return -rho(phi.phi, trace, t)
    if isinstance(phi, And):
        return min(rho(p, trace, t) for p in phi.args)
    if isinstance(phi, Or):
        return max(rho(p, trace, t) for p in phi.args)
    if isinstance(phi, Implies):
        return max(-rho(phi.lhs, trace, t), rho(phi.rhs, trace, t))
    if isinstance(phi, (Always, Eventually)):
        n = _length(trace)
        steps = range(t + phi.a, min(t + phi.b, n - 1) + 1)
        values = [rho(phi.phi, trace, s) for s in steps]
        if isinstance(phi, Always):
            return min(values, default=math.inf)
        return max(values, default=-math.inf)
    raise TypeError(f"not an STL formula: {phi!r}")


def horizon(phi: Formula) -> int:
    """How many steps into the future `phi` looks."""
    if isinstance(phi, Pred):
        return 0
    if isinstance(phi, Not):
        return horizon(phi.phi)
    if isinstance(phi, (And, Or)):
        return max((horizon(p) for p in phi.args), default=0)
    if isinstance(phi, Implies):
        return max(horizon(phi.lhs), horizon(phi.rhs))
    if isinstance(phi, (Always, Eventually)):
        return phi.b + horizon(phi.phi)
    raise TypeError(f"not an STL formula: {phi!r}")


@dataclass
class SpecResult:
    robustness: float  # min over obligations; +inf when there is none
    satisfied: bool
    n_obligations: int
    n_violations: int
    worst_t: int | None  # first step with the lowest robustness


def series(phi: Formula, trace: Mapping[str, np.ndarray]) -> np.ndarray:
    """Robustness of each obligation of G(phi): entry t is phi at step t, for every step whose window fits in the
    trace. A top-level And is split (G(p and q) = G(p) and G(q)), so each conjunct is checked up to its own horizon:
    a conjunct without look-ahead is still checked at the last step."""
    n = _length(trace)
    parts = phi.args if isinstance(phi, And) and phi.args else (phi,)
    lasts = [n - 1 - horizon(p) for p in parts]
    return np.array([min(rho(p, trace, t) for p, last in zip(parts, lasts) if t <= last)
                     for t in range(max(lasts) + 1)], float)


def check_always(phi: Formula, trace: Mapping[str, np.ndarray]) -> SpecResult:
    """G(phi) over the trace (see `series` for which steps are obligations)."""
    values = series(phi, trace).tolist()
    if not values:
        return SpecResult(math.inf, True, 0, 0, None)
    worst = int(np.argmin(values))
    return SpecResult(values[worst], values[worst] >= 0, len(values), sum(v < 0 for v in values), worst)
