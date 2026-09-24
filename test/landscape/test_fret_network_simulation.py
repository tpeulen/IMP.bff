"""Simulated photons of a FRET network measurement follow the model that scores them."""

import numpy as np

import IMP.bff as bff
from test_fret_network_filter import _dyes, _instrument


def _measurement(nb=16, bleach=False):
    process = bff.FRETHiddenProcess(2)
    process.set_rate(0, 1, 6.0)
    process.set_rate(1, 0, 4.0)
    d, a = _dyes(bleach)
    m = bff.FRETMeasurement("pair", d, a, 55.0)
    m.set_instrument(_instrument(nb))
    m.set_state_distance(0, 45.0)
    m.set_state_distance(1, 70.0)
    return process, m


def _options(n=200, duration=1.0, focus=False, seed=3):
    o = bff.FRETSimulationOptions()
    o.n_molecules, o.duration, o.focus, o.seed = n, duration, focus, seed
    return o


def test_photons_follow_each_states_channels_and_microtimes():
    process, m = _measurement(nb=16)
    o = _options(n=400)
    data = bff.simulate_fret_measurement(process, m, o)
    states = np.asarray(bff.simulated_fret_states(process, m, o))
    channels = np.asarray(data.get_channels())
    bins = np.asarray(data.get_microtimes())
    n = m.get_n_states(process)
    C, nb = 2, 16
    lam = np.asarray(m.get_detection_rates(process)).reshape(C, n)
    emission = np.asarray(m.get_emission(process)).reshape(C, nb, n)
    assert len(states) == data.get_n_photons() > 1000
    for s in range(n):
        here = states == s
        if here.sum() < 400:
            continue
        expected = lam[:, s] / lam[:, s].sum()
        observed = np.bincount(channels[here], minlength=C) / here.sum()
        assert np.allclose(observed, expected, atol=4.0 / np.sqrt(here.sum()))
        for c in range(C):
            mask = here & (channels == c)
            if mask.sum() < 400:
                continue
            p = emission[c, :, s] / emission[c, :, s].sum()
            h = np.bincount(bins[mask], minlength=nb) / mask.sum()
            # Total variation well inside what 400+ photons allow.
            assert 0.5 * np.abs(h - p).sum() < 0.12


def test_the_focus_thins_the_signal_not_the_background():
    process, m = _measurement(nb=1)
    flat = bff.simulate_fret_measurement(process, m, _options(n=300))
    focus = bff.simulate_fret_measurement(process, m, _options(n=300, focus=True))
    assert focus.get_n_photons() < 0.8 * flat.get_n_photons()
    # Photons concentrate in the middle of each molecule's segment.
    t = np.asarray(focus.get_macrotimes())
    phase = np.mod(t, 2.0)  # molecules are 2 * duration apart
    middle = np.mean((phase > 0.25) & (phase < 0.75))
    assert middle > 0.6


def test_burst_selection_cuts_at_gaps_and_drops_short_runs():
    t = [0.0, 0.1, 0.2, 1.5, 1.6, 5.0, 5.05, 5.1, 5.15]
    data = bff.FRETPhotonData(t, [0] * len(t), [], [0], [len(t) - 1])
    bursts = bff.select_bursts(data, 0.3, 3)
    assert list(bursts.get_segment_starts()) == [0, 5]
    assert list(bursts.get_segment_stops()) == [2, 8]
    assert bursts.get_n_photons() == len(t)


def test_simulated_bursts_score_best_near_the_truth():
    """The likelihood of simulated bursts peaks near the rates that made them."""
    process, m = _measurement(nb=8)
    data = bff.select_bursts(bff.simulate_fret_measurement(process, m, _options(n=150)), 0.2, 20)
    net = bff.FRETNetworkModel(process)
    net.add_measurement(m, data)
    names = list(net.get_free_parameter_names())
    theta = np.asarray(net.get_theta())
    base = net.log_likelihood_at(list(theta))
    k = names.index("hidden.rate[0->1]")
    for delta in (-1.0, 1.0):
        moved = theta.copy()
        moved[k] += delta  # a factor e away, on the log scale
        assert net.log_likelihood_at(list(moved)) < base
