"""root_brentq is scipy.optimize.brentq: same root, iterations and calls."""
import numpy as np
import pytest

import IMP.bff as bff

scipy_optimize = pytest.importorskip("scipy.optimize")


class _F(bff.MinimizeObjective):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def evaluate(self, x):
        return float(self.fn(x[0]))


CASES = [
    (lambda x: x**3 - 2 * x - 5, 2.0, 3.0),
    (lambda x: np.cos(x) - x, 0.0, 1.0),
    (lambda x: np.exp(x) - 10.0, 0.0, 5.0),
    (lambda x: x - 1e-3, 0.0, 200.0),
    (lambda x: np.tanh(50 * (x - 0.3)), -1.0, 4.0),
]


@pytest.mark.parametrize("fn,a,b", CASES)
def test_matches_scipy_step_for_step(fn, a, b):
    r = bff.root_brentq(_F(fn), a, b)
    s = scipy_optimize.root_scalar(fn, bracket=[a, b])
    assert r.root == s.root
    assert r.iterations == s.iterations
    assert r.function_calls == s.function_calls
    assert r.converged and r.flag == "converged"


def test_same_sign_bracket_raises_value_error():
    with pytest.raises(ValueError, match="different signs"):
        bff.root_brentq(_F(lambda x: x * x + 1), 0.0, 1.0)


def test_iteration_limit_reports_unconverged():
    r = bff.root_brentq(_F(lambda x: x**3 - 2 * x - 5), 2.0, 3.0, maxiter=2)
    assert not r.converged and r.flag == "convergence error"
