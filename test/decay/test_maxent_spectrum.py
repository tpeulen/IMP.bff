"""Maximum-entropy distributions from a decay, in the engine.

The MaxEnt TCSPC analysis lives here (moved from tttrlib, whose Skilling-Bryan
engine is vendored as internal/MaxEntQp.h): the grid's decays are TCSPCDecay's
basis and a distance grid's transfer rates are FRETSpectrumNode's, so a MaxEnt
fit meets the same instrument as every other lifetime fit. Known answers on
simulated Poisson data, and the identities that tie the fitted curve to the
solution.
"""

import numpy as np
import pytest

import IMP.bff as bff

N, DT, PERIOD = 512, 0.032, 12.5
X = np.arange(N) * DT
IRF = np.exp(-0.5 * ((X - 1.0) / 0.06) ** 2)


def _decay_from(spectrum, n0, background, seed):
    source = bff.TCSPCDecay("source")
    source.set_number_of_lifetimes(1)
    source.add_output_port("source", bff.GraphPort([0.0], False, True))
    source.set_response_array(np.ascontiguousarray(IRF))
    source.set_timing(DT, PERIOD)
    source.set_convolution_range(N, N)
    source.set_spectrum_from_port(True)
    source.get_input_port("lifetime_spectrum").set_values_array(np.asarray(spectrum, dtype=float))
    source.get_input_port("n0").value = n0
    source.get_input_port("background").value = background
    source.set_normalize_amplitudes(True)
    source.set_instrument_units("counts")
    source.update()
    return np.random.default_rng(seed).poisson(np.asarray(source.get_output_port("source").value)).astype(float)


def _problem(family, y, **values):
    data = bff.FitDataset()
    data.set_values_array(np.ascontiguousarray(y))
    data.set_noise_family(bff.FIT_NOISE_FAMILY_POISSON)
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(IRF))
    spec = bff.ModelSearchSpec.from_name(family)
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", DT)
    spec.set_scalar("period", PERIOD)
    for name, value in values.pop("scalars", {}).items():
        spec.set_scalar(name, value)
    prior_values = np.atleast_1d(values.pop("prior", [0.0])).astype(float)
    prior = bff.GraphPort([0.0])
    spec.set_port("maxent_prior", prior)
    problem = spec.build()
    key = problem.get_structure_keys()[0]
    problem.activate_structure(key)
    for name, value in values.items():
        port = problem.get_parameter(name.replace("__", "."))
        held = port.fixed
        port.fixed = False
        port.value = value
        port.fixed = held
    # The prior after the grid it weighs: a port the caller holds and rewrites.
    prior.set_value_vector(list(prior_values))
    problem.__dict__["_prior_port"] = prior
    return problem, key


def _port(problem, key, node, port):
    return np.asarray(problem.get_structure_port(key, f"{key}.{node}", port), dtype=float)


def _curve(problem, key):
    return np.asarray(problem.get_structure_output(key, problem.get_structure_curve_node(key, "decay")))


def _peaks(p, x, floor=0.1):
    return [(x[i], p[i]) for i in range(1, len(p) - 1) if p[i] > p[i - 1] and p[i] >= p[i + 1] and p[i] > floor * p.max()]


def test_a_grid_is_unit_pairs_linear_or_logarithmic():
    assert np.allclose(bff.SpectrumGrid.points(1.0, 3.0, 3, False), [1.0, 2.0, 3.0])
    assert np.allclose(bff.SpectrumGrid.points(1.0, 100.0, 3, True), [1.0, 10.0, 100.0])
    with pytest.raises(ValueError):
        bff.SpectrumGrid.points(2.0, 1.0, 3, False)


def test_two_lifetimes_are_recovered_with_their_amplitudes():
    y = _decay_from([0.5, 1.0, 0.5, 4.0], 2e5, 5.0, seed=1)
    problem, key = _problem("tcspc_maxent_lifetime", y, instrument__background=5.0, maxent__log10_nu=-4.0,
                            maxent__grid_from=0.2, maxent__grid_to=8.0, maxent__grid_bins=80)
    curve = _curve(problem, key)
    assert np.mean((curve - y) ** 2 / np.maximum(y, 1.0)) < 1.2
    distribution = _port(problem, key, "maxent", "distribution")
    p, tau = distribution[0::2], distribution[1::2]
    p = p / p.sum()
    peaks = _peaks(p, tau)
    assert len(peaks) == 2
    assert abs(peaks[0][0] - 1.0) < 0.15 and abs(peaks[1][0] - 4.0) < 0.2
    assert abs(p[tau < 2.0].sum() - 0.5) < 0.05


def test_two_distances_are_recovered_through_the_fret_rates():
    fret = bff.FRETSpectrumNode("fret")
    fret.build_ports()
    fret.add_output_port("fret", bff.GraphPort([0.0], False, True))
    fret.get_input_port("donor_lifetime_spectrum").set_values_array(np.array([1.0, 4.0]))
    fret.get_input_port("distance_distribution").set_values_array(np.array([0.6, 40.0, 0.4, 65.0]))
    for name, value in (("x_donly", 0.0), ("forster_radius", 52.0), ("tau0", 4.0), ("kappa2", 2.0 / 3.0)):
        fret.get_input_port(name).value = value
    fret.update()
    y = _decay_from(np.asarray(fret.get_output_port("fret").value), 3e5, 5.0, seed=2)
    problem, key = _problem("tcspc_maxent_fret", y, instrument__background=5.0, maxent__log10_nu=-4.0,
                            fret__x_donly=0.0, fret__forster_radius=52.0, fret__tau0=4.0, donor__tau__0=4.0,
                            donor__amplitude__0=1.0, maxent__grid_from=20.0, maxent__grid_to=90.0,
                            maxent__grid_bins=71)
    distribution = _port(problem, key, "maxent", "distribution")
    p, r = distribution[0::2], distribution[1::2]
    p = p / p.sum()
    peaks = _peaks(p, r)
    assert [round(x) for x, _ in peaks] == [40, 65]
    assert abs(p[(r > 32) & (r < 50)].sum() - 0.6) < 0.05
    assert abs(p[(r > 55) & (r < 78)].sum() - 0.4) < 0.05


def test_the_fitted_curve_is_the_solution_on_the_basis_plus_the_instrument():
    """What the inversion took off the data (background, scatter on the prepared
    response) is what the fitted TCSPCDecay adds back."""
    y = _decay_from([0.5, 1.0, 0.5, 4.0], 2e5, 5.0, seed=4)
    problem, key = _problem("tcspc_maxent_lifetime", y, instrument__background=4.0, instrument__scatter=0.8,
                            maxent__grid_bins=40)
    curve = _curve(problem, key)
    basis = _port(problem, key, "basis", "basis").reshape(N, -1)
    response = _port(problem, key, "basis", "prepared_response")
    spectrum = np.asarray(problem.get_structure_output(key, f"{key}.maxent"), dtype=float)
    expected = basis @ spectrum[0::2] + 4.0 + 0.8 * response
    np.testing.assert_allclose(curve, np.maximum(expected, 0.0), rtol=1e-9, atol=1e-9)
    model = expected
    pearson = np.mean((model - y) ** 2 / np.maximum(model, 1.0))
    assert _port(problem, key, "maxent", "chisq_pearson")[0] == pytest.approx(pearson, rel=1e-9)


def test_historic_maxent_lands_on_the_target_chi_square():
    y = _decay_from([0.5, 1.0, 0.5, 4.0], 2e5, 5.0, seed=5)
    problem, key = _problem("tcspc_maxent_lifetime", y, instrument__background=5.0, maxent__grid_bins=60,
                            scalars={"target_chisq": 1.3})
    # Above the least-squares floor of this grid (about 1.06), so reachable.
    assert _port(problem, key, "maxent", "chisq")[0] == pytest.approx(1.3, rel=0.02)
    assert _port(problem, key, "maxent", "nu")[0] > 0.0


def test_a_grouping_is_named():
    node = bff.MaxEntSpectrum("maxent")
    with pytest.raises(ValueError):
        node.set_grouping("distance")


def test_a_prior_pulls_the_distribution_towards_it():
    """The entropy is measured against the prior: a rising prior moves weight to long lifetimes."""
    y = _decay_from([0.5, 1.0, 0.5, 4.0], 2e5, 5.0, seed=6)
    shares = []
    for prior in ([0.0], np.linspace(1.0, 10.0, 20)):
        problem, key = _problem("tcspc_maxent_lifetime", y, prior=prior, instrument__background=5.0,
                                maxent__log10_nu=1.0, maxent__grid_bins=20)
        distribution = _port(problem, key, "maxent", "distribution")
        p, tau = distribution[0::2], distribution[1::2]
        shares.append(p[tau > 3.0].sum() / p.sum())
        counts = _port(problem, key, "maxent", "amplitudes")
        np.testing.assert_allclose(counts / counts.sum(), p / p.sum(), rtol=1e-12)
    assert shares[1] > shares[0] + 0.02


def test_an_empty_fit_window_is_an_empty_distribution_not_an_error():
    y = _decay_from([0.5, 1.0, 0.5, 4.0], 2e5, 5.0, seed=7)
    data = bff.FitDataset()
    data.set_values_array(np.ascontiguousarray(y))
    data.set_noise_family(bff.FIT_NOISE_FAMILY_POISSON)
    data.set_mask_array(np.zeros(N))
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(IRF))
    spec = bff.ModelSearchSpec.from_name("tcspc_maxent_lifetime")
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", DT)
    spec.set_scalar("period", PERIOD)
    spec.set_port("maxent_prior", bff.GraphPort([0.0]))
    problem = spec.build()
    key = problem.get_structure_keys()[0]
    problem.activate_structure(key)
    assert not np.any(_port(problem, key, "maxent", "amplitudes"))
    assert np.all(np.isfinite(_curve(problem, key)))


def test_an_unregularised_periodic_fret_fit_reaches_the_chi_square_floor():
    """At nu = 1e-6 a periodic FRET decay stalled at a reduced chi-square of
    2.73 against a reachable 1.163 (and a distance off by 0.3 A): the bounded
    QP inside the MEM loop never released an amplitude it had clamped, and
    near the positivity floor the entropy curvature kept it there. The decay is
    simulated independently of the engine -- exponentials on a fine grid,
    convolved by numpy, folded over the laser period."""
    n, dt, sub, period = 256, 0.05, 20, 12.8
    tau0, r0, r_true, x_donly = 4.0, 50.0, 45.0, 0.1
    tau_da = 1.0 / (1.0 / tau0 + (1.0 / tau0) * (r0 / r_true) ** 6)
    fine = np.arange(n * sub) * (dt / sub)
    irf_fine = np.exp(-0.5 * ((fine - 1.0) / 0.12) ** 2)
    irf_fine /= irf_fine.sum()
    long = np.arange(30 * n * sub) * (dt / sub)
    full = np.convolve((1 - x_donly) * np.exp(-long / tau_da) + x_donly * np.exp(-long / tau0), irf_fine)
    folded = np.zeros(n * sub)
    np.add.at(folded, np.arange(full.size) % (n * sub), full)
    model = folded.reshape(n, sub).sum(1)
    y = np.random.default_rng(2).poisson(model / model.sum() * 2e6).astype(float)
    irf = irf_fine.reshape(n, sub).sum(1) * 1e5

    def fit(log10_nu):
        data = bff.FitDataset()
        data.set_values_array(np.ascontiguousarray(y))
        data.set_noise_family(bff.FIT_NOISE_FAMILY_POISSON)
        mask = np.zeros(n)
        mask[25:] = 1.0
        data.set_mask_array(np.ascontiguousarray(mask))
        response = bff.FitDataset()
        response.set_values_array(np.ascontiguousarray(irf))
        spec = bff.ModelSearchSpec.from_name("tcspc_maxent_fret")
        spec.set_dataset("decay", data)
        spec.set_dataset("response", response)
        spec.set_scalar("dt", dt)
        spec.set_scalar("period", period)
        spec.set_scalar("max_iterations", 500)
        spec.set_port("maxent_prior", bff.GraphPort([0.0]))
        problem = spec.build()
        key = problem.get_structure_keys()[0]
        problem.activate_structure(key)
        for name, value in (("maxent.log10_nu", log10_nu), ("maxent.grid_from", 30.0), ("maxent.grid_to", 70.0),
                            ("maxent.grid_bins", 81.0), ("fret.tau0", tau0), ("fret.forster_radius", r0),
                            ("fret.kappa2", 2.0 / 3.0), ("fret.x_donly", x_donly), ("donor.tau.0", tau0),
                            ("donor.amplitude.0", 1.0), ("instrument.background", 0.0),
                            ("instrument.response_background", 0.0), ("instrument.scatter", 0.0),
                            ("instrument.timeshift", 0.0)):
            port = problem.get_parameter(name)
            held = port.fixed
            port.fixed = False
            port.value = value
            port.fixed = held
        distribution = _port(problem, key, "maxent", "distribution")
        p, r = distribution[0::2], distribution[1::2]
        return float(_port(problem, key, "maxent", "chisq")[0]), float((p * r).sum() / p.sum())

    chisq, mean = fit(-6.0)
    assert chisq < 1.17
    assert abs(mean - r_true) < 0.1
