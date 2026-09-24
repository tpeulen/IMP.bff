"""Fast rates: the dense-exponential path of the FRET network filter.

A gap is propagated by a dense matrix exponential when that is cheaper than
the uniformization series -- a fast rate on a small state space. Before, the
series grew linearly with the rate, and a three-state fit whose optimiser
pushed a rate up along a flat direction crawled for minutes. These pin that
the dense path gives the same likelihood, gradient and occupancies.
"""

import time

import numpy as np
import pytest

import IMP.bff as bff
from test_fret_network_filter import _brute, _setup
from test_fret_network_fit import _fd


def _fast(net, hp, m, rate):
    hp.set_rate(0, 1, rate)
    hp.set_rate(1, 0, 0.7 * rate)
    fast = bff.FRETNetworkModel(hp)
    fast.add_measurement(m, net.get_data(0))
    return fast


@pytest.mark.parametrize("conditional", [True, False])
def test_fast_rates_match_expm(conditional):
    pytest.importorskip("scipy")
    net, hp, m, data = _setup()
    fast = _fast(net, hp, m, 4.0e4)
    fast.set_arrival_model(0, bff.FRET_ARRIVAL_CONDITIONAL if conditional
                           else bff.FRET_ARRIVAL_FULL)
    ref = _brute(hp, m, data, conditional, True)
    np.testing.assert_allclose(fast.segment_log_likelihoods(0), ref, rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("arrival", [bff.FRET_ARRIVAL_CONDITIONAL, bff.FRET_ARRIVAL_FULL])
def test_fast_rates_gradient_matches_finite_differences(arrival):
    net, hp, m, _ = _setup()
    fast = _fast(net, hp, m, 2.0e4)
    fast.set_arrival_model(0, arrival)
    theta = np.asarray(fast.get_theta())
    g = np.asarray(fast.log_likelihood_gradient(theta))
    fd = _fd(fast.log_likelihood_at, theta)
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-4 * np.abs(fd).max())


def test_fast_rates_occupancies_are_time_fractions():
    net, hp, m, _ = _setup()
    fast = _fast(net, hp, m, 3.0e4)
    occ = np.asarray(fast.segment_occupancies(0, "hidden")).reshape(-1, 2)
    np.testing.assert_allclose(occ.sum(1), 1.0, atol=1e-8)
    # Exchange far faster than any gap: each segment sits near the
    # detection-weighted stationary split, not at 0 or 1.
    assert np.all((occ > 0.2) & (occ < 0.8))


def test_a_fast_rate_costs_no_more_than_a_slow_one():
    net, hp, m, _ = _setup()
    times = []
    for rate in (5.0, 5.0e6):
        fast = _fast(net, hp, m, rate)
        start = time.perf_counter()
        for _ in range(5):
            fast.log_likelihood()
        times.append(time.perf_counter() - start)
    assert times[1] < 20.0 * times[0] + 0.05
