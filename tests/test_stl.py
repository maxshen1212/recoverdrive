import math

import numpy as np

from recoverdrive.stl import (
    Always,
    And,
    Eventually,
    Implies,
    Not,
    Or,
    Pred,
    SpecResult,
    check_always,
    horizon,
    rho,
)


def test_pred_and_not():
    tr = {"x": np.array([1.0, 3.0])}
    assert rho(Pred("x", ">=", 2), tr, 1) == 1
    assert rho(Not(Pred("x", ">=", 2)), tr, 1) == -1
    assert rho(Pred("x", "<=", 2), tr, 1) == -1
    assert rho(Pred("x", "<=", 2), tr, 0) == 1


def test_always_window_min():
    tr = {"x": np.array([5.0, 1.0, 4.0])}
    assert rho(Always(0, 2, Pred("x", ">=", 0)), tr, 0) == 1
    assert rho(Always(3, 4, Pred("x", ">=", 0)), tr, 0) == math.inf


def test_eventually_window_max():
    tr = {"x": np.array([5.0, 1.0, 4.0])}
    assert rho(Eventually(1, 2, Pred("x", ">=", 0)), tr, 0) == 4
    assert rho(Eventually(3, 4, Pred("x", ">=", 0)), tr, 0) == -math.inf


def test_implies_and_or():
    tr = {"p": np.array([1.0]), "q": np.array([-2.0])}
    p, q = Pred("p", ">=", 0), Pred("q", ">=", 0)
    assert rho(Implies(p, q), tr, 0) == max(-1.0, -2.0)
    assert rho(And(p, q), tr, 0) == -2
    assert rho(Or(p, q), tr, 0) == 1


def test_horizon_nested():
    p = Pred("p", ">=", 0)
    assert horizon(p) == 0
    assert horizon(Implies(p, Eventually(0, 2, p))) == 2
    assert horizon(Always(1, 1, Eventually(0, 3, p))) == 4


def test_check_always_drops_obligations_past_end():
    tr = {"p": np.array([0.0, 0, 0, 0, 1])}
    res = check_always(Eventually(0, 2, Pred("p", ">=", 0.5)), tr)
    assert res.n_obligations == 3
    assert res.n_violations == 2
    assert res.worst_t == 0
    assert res.satisfied is False
    assert res.robustness == -0.5


def test_check_always_splits_top_level_and():
    # G(p and F[0,1] q) == G(p) and G(F[0,1] q): p must still be checked at the last step
    tr = {"p": np.array([1.0, 1, -1]), "q": np.array([1.0, 1, 1])}
    res = check_always(And(Pred("p", ">=", 0), Eventually(0, 1, Pred("q", ">=", 0))), tr)
    assert res.n_obligations == 3 and res.worst_t == 2 and not res.satisfied


def test_empty_trace_satisfied():
    res = check_always(Pred("p", ">=", 0), {"p": np.array([])})
    assert res == SpecResult(math.inf, True, 0, 0, None)
