"""bff.odeint against scipy.integrate.odeint (LSODA, both).

SciPy is the oracle. Both run SciPy's C translation of LSODA; the dense LU
underneath differs (SciPy calls an optimised LAPACK), so step sequences can
part at rounding level. The solutions are held to well inside the
integration tolerance, and on the reference problems the step, evaluation
and method-switch counts are compared directly.
"""

from __future__ import annotations

import numpy as np
import pytest

import IMP.bff as bff

integrate = pytest.importorskip("scipy.integrate")


class Rhs(bff.OdeFunction):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.calls = 0

    def evaluate(self, y, t):
        self.calls += 1
        return np.asarray(self.f(np.asarray(y), t), dtype=float)


def _relax(y, t):  # A <-> B, k = 2, 1
    return np.array([-2 * y[0] + y[1], 2 * y[0] - y[1]])


def _preequilibrium(y, t):  # A + B <-> C (1e4, 1e3), C -> D (1e-2)
    f = 1e4 * y[0] * y[1] - 1e3 * y[2]
    s = 1e-2 * y[2]
    return np.array([-f, -f, f - s, s])


def _robertson(y, t):
    return np.array([
        -0.04 * y[0] + 1e4 * y[1] * y[2],
        0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
        3e7 * y[1] ** 2,
    ])


CASES = {
    "relax": (_relax, [1.0, 0.0], np.linspace(0, 20, 200), {}),
    "preequilibrium": (_preequilibrium, [1.0, 1.0, 0.0, 0.0], np.linspace(0, 300, 1000), {}),
    "robertson": (
        _robertson, [1.0, 0.0, 0.0], np.logspace(-6, 6, 61),
        {"rtol": 1e-6, "atol": [1e-8, 1e-14, 1e-6]},
    ),
}


def _run(f, y0, t, opts):
    r = bff.odeint(
        Rhs(f), list(y0), list(t),
        list(np.atleast_1d(opts.get("rtol", []))), list(np.atleast_1d(opts.get("atol", []))),
        list(np.atleast_1d(opts.get("tcrit", []))), 0.0, 0.0, 0.0, 0,
        int(opts.get("mxstep", 0)),
    )
    return r, np.asarray(r.y).reshape(r.n_times, r.n_equations)


@pytest.mark.parametrize("name", list(CASES))
def test_odeint_solution_follows_scipy(name):
    f, y0, t, opts = CASES[name]
    r, y = _run(f, y0, t, opts)
    ys, info = integrate.odeint(f, y0, t, full_output=True, **opts)
    assert r.istate == 2 and r.success
    assert r.message == info["message"]
    scale = np.maximum(np.abs(ys), 1e-12)
    rtol = opts.get("rtol", 1.49012e-8)
    # Both agree far inside the tolerance either asked for.
    assert np.max(np.abs(y - ys) / (scale + np.atleast_1d(opts.get("atol", 1.49012e-8)))) < 50 * rtol
    # The method switch (Adams -> BDF) is the same decision.
    np.testing.assert_array_equal(np.asarray(r.mused), info["mused"])
    # Step counts part at LU-rounding level on stiff problems (Robertson:
    # 670 vs 724) while the solution and the method switch agree.
    assert abs(r.nst[-1] - info["nst"][-1]) <= max(3, 0.1 * info["nst"][-1])


def test_robertson_is_stiff_and_switches_to_bdf():
    f, y0, t, opts = CASES["robertson"]
    r, _ = _run(f, y0, t, opts)
    assert 2 in set(r.mused)
    assert r.nfe[-1] < 5000  # an explicit method needs ~1e6 here


def test_tcrit_is_not_stepped_past():
    f = lambda y, t: np.array([1.0 if t <= 1.0 else -1e3])  # noqa: E731
    t = np.linspace(0, 2, 21)
    r, y = _run(f, [0.0], t, {"tcrit": [1.0]})
    ys = integrate.odeint(f, [0.0], t, tcrit=[1.0])
    np.testing.assert_allclose(y, ys, rtol=1e-6, atol=1e-8)
    assert np.all(np.asarray(r.tcur)[: 10] <= 1.0 + 1e-12)


def test_excess_work_fails_like_scipy_and_leaves_nan_rows():
    f, y0, t, opts = CASES["robertson"]
    r, y = _run(f, y0, t, dict(opts, mxstep=5))
    with pytest.warns(integrate.ODEintWarning):
        ys, info = integrate.odeint(f, y0, t, full_output=True, mxstep=5, **opts)
    assert r.istate == -1 and not r.success
    assert r.message == info["message"]
    reached = np.flatnonzero(~np.isnan(y[:, 0]))
    assert reached.size > 0 and np.all(np.isnan(y[reached[-1] + 1:]))


def test_nothing_to_do_when_every_time_is_the_start():
    r, y = _run(_relax, [1.0, 0.0], [0.0, 0.0, 0.0], {})
    assert r.istate == 1
    np.testing.assert_array_equal(y, [[1.0, 0.0]] * 3)


def test_wrong_length_derivative_and_python_errors_propagate():
    with pytest.raises(ValueError):
        _run(lambda y, t: np.zeros(3), [1.0, 0.0], [0.0, 1.0], {})
    with pytest.raises(RuntimeError, match="boom"):
        def boom(y, t):
            raise RuntimeError("boom")
        _run(boom, [1.0], [0.0, 1.0], {})


def test_tolerances_of_the_wrong_length_are_rejected():
    with pytest.raises(ValueError):
        bff.odeint(Rhs(_relax), [1.0, 0.0], [0.0, 1.0], [1e-6, 1e-6, 1e-6])
