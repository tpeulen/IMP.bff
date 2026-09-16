"""The photokinetic bunching factor and the limits it must hit exactly.

``X(tau)`` describes a molecule hopping between the states of a photokinetic
scheme. Two values are fixed by the physics rather than by the numerics:
``X(0)`` is ``sum Q_i^2 p_i / (sum Q_i p_i)^2``, and ``X(tau -> inf)`` is
exactly 1, because a generator conserves probability -- its columns sum to
zero, so its stationary mode has eigenvalue zero and does not decay.

That second one is the interesting test. An eigen-decomposition returns the
stationary eigenvalue as round-off (~1e-18 of the largest rate) rather than as
zero, and treating it as a rate makes ``X`` drift off its own limit *linearly
in tau* -- 3e-9 per second for the rhodamine scheme below, which is invisible
at a microsecond lag and plain at a second.
"""

import numpy as np
import pytest

import IMP.bff as bff

#: A three-state rhodamine-like scheme: ground, excited, triplet (Hz).
DARK = np.array([
    [0.0, 2.5e8, 5.0e5],
    [0.0, 0.0, 0.0],
    [0.0, 2.5e6, 0.0],
])
#: Excitation couples the ground state to the excited one.
EXC = np.array([
    [-1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])
#: Only the excited state emits.
Q = np.array([0.0, 1.0, 0.0])
K_EXC = 1e8


def _bunching(tau):
    return np.asarray(bff.fcs_bunching_factor(
        [float(t) for t in np.atleast_1d(tau)], float(K_EXC),
        [float(v) for v in DARK.ravel()], [float(v) for v in EXC.ravel()],
        DARK.shape[0], [float(v) for v in Q], []), dtype=float)


@pytest.mark.parametrize("tau", [1.0, 1e2, 1e4])
def test_the_limit_is_one_and_stays_there(tau):
    """Long after the scheme has relaxed, X is 1 -- at every long lag alike."""
    assert _bunching(tau)[0] == pytest.approx(1.0, abs=1e-12)


def test_the_limit_does_not_drift_with_the_lag():
    """The stationary mode does not decay, so two long lags agree exactly."""
    near, far = _bunching([1.0, 1e4])
    assert far == pytest.approx(near, abs=1e-12)


def test_zero_lag_is_the_brightness_weighted_ratio():
    """X(0) = sum Q^2 p / (sum Q p)^2, the scheme's own steady state."""
    generator = DARK + K_EXC * EXC
    np.fill_diagonal(generator, 0.0)
    np.fill_diagonal(generator, -generator.sum(axis=0))
    values, vectors = np.linalg.eig(generator)
    stationary = np.real(vectors[:, int(np.argmin(np.abs(values)))])
    p_eq = stationary / stationary.sum()
    expected = float(np.sum(Q**2 * p_eq) / np.sum(Q * p_eq) ** 2)
    assert _bunching(0.0)[0] == pytest.approx(expected, rel=1e-9)
