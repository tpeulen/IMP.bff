"""S3 tests: the network objective on MCMCSampler and its diagnostics.

Fast-NPS samples the posterior of one dye configuration under its
multi-measurement Gaussian likelihood; its sampler, cross-entropy report
and chi-square consistency check are one MATLAB script. In bff the pieces
are committed infrastructure: the likelihood is
nps_network_log_likelihood(), the sampler is MCMCSampler, and the
convergence estimators live in SamplerDiagnostics.h (Vehtari et al. 2021,
arviz-parity). S3 wires them together:

- `NPSNetworkObjective`, a GraphNode whose update() evaluates the network
  log-likelihood at the configuration its linked input ports carry and
  publishes it on the "chi2" output port (as normalized log L for the
  sampler's log-likelihood reading, or as residual chi2 directly) — so a
  Python-side std::function is never needed and a Markov step never
  crosses into Python;
- `nps_network_chi2`, the chi-square of the active measurements (the
  -2 log L that MCMCSampler's chi2 mode reports), so "the model fits the
  data" and "the sampler moved well" are one number with one definition;
- `nps_cross_entropy` and `nps_mean_mcse` — the sampler report: the
  cross-entropy of the recorded log-likelihoods (Fast-NPS's figure of
  merit; -mean(log L) over the chain) and the Monte-Carlo standard error
  of each coordinate's posterior mean, through the committed
  mcse_mean(); no new estimator is defined here.
"""
import math

import numpy as np
import pytest

import IMP
import IMP.bff as bff


def _world(e_obs=0.6, sigma=0.1):
    """Two iso direct dyes, one FRET measurement."""
    dyes = []
    for _ in range(2):
        dye = bff.NPSNetworkDye()
        dye.dep = 0.0  # r_avg = 0: fully depolarized, kappa^2 = 2/3
        dye.iso = True
        dye.dist_conv = False
        dyes.append(dye)
    meas = bff.NPSMeasurement()
    meas.dye1 = 0
    meas.dye2 = 1
    meas.fret_active = True
    meas.e_avg = e_obs
    meas.e_err = sigma
    meas.r_iso6 = 55.0 ** 6
    return dyes, [meas]


def _config_at(distance):
    return [[0.0, 0.0, 0.0, 0.0, 0.0],
            [distance, 0.0, 0.0, 0.0, 0.0]]


# ---------------------------------------------------------------------------
# chi2 consistency: one number, one definition
# ---------------------------------------------------------------------------

def test_network_chi2_equals_minus_two_log_likelihood():
    dyes, meas = _world()
    config = _config_at(70.0)
    chi2 = bff.nps_network_chi2(config, dyes, meas)
    logl = bff.nps_network_log_likelihood(config, dyes, meas)
    # -2 log L = chi2 - 2 * sum(log(1/(sigma sqrt(2 pi)))) over the
    # active observables: the chi2 is the normalized part only.
    n_active = sum(1 for m in meas for a in (m.fret_active, m.ta_active) if a)
    const = 2.0 * n_active * math.log(meas[0].e_err * math.sqrt(2.0 * math.pi))
    assert chi2 == pytest.approx(-2.0 * logl - const, rel=1e-12)


def test_network_chi2_is_squared_standard_errors_of_the_model():
    dyes, meas = _world(e_obs=0.6, sigma=0.1)
    config = _config_at(70.0)
    e_model = bff.nps_network_fret_efficiency(config, dyes, 0, 1, 55.0 ** 6)
    expected = ((0.6 - e_model) / 0.1) ** 2
    assert bff.nps_network_chi2(config, dyes, meas) == pytest.approx(
        expected, rel=1e-12)


def test_network_chi2_is_zero_when_the_model_matches_the_data():
    dyes, meas = _world()
    config = _config_at(70.0)
    e_model = bff.nps_network_fret_efficiency(config, dyes, 0, 1, 55.0 ** 6)
    meas[0].e_avg = e_model
    assert bff.nps_network_chi2(config, dyes, meas) == pytest.approx(
        0.0, abs=1e-24)


def test_network_chi2_rejects_inactive_and_bad_measurements():
    dyes, meas = _world()
    meas[0].fret_active = False
    meas[0].ta_active = False
    with pytest.raises(IMP.ValueException):
        bff.nps_network_chi2(_config_at(70.0), dyes, meas)
    dyes2, meas2 = _world()
    meas2[0].e_err = -1.0
    with pytest.raises(IMP.ValueException):
        bff.nps_network_chi2(_config_at(70.0), dyes2, meas2)


# ---------------------------------------------------------------------------
# the objective node
# ---------------------------------------------------------------------------

def _linked(port):
    p = bff.GraphPort([0.0])
    p.set_link(port)
    return p


def _objective(distance=70.0):
    """Two dyes -> six coordinate ports (x0..x5); the separation is dye 1's
    x (port x3), matching _config_at. Angles are the node's isotropic seed."""
    dyes, meas = _world()
    node = bff.NPSNetworkObjective(dyes, meas, 2)
    ports = []
    for i, start in enumerate([0.0, 0.0, 0.0,
                               distance, 0.0, 0.0]):
        p = bff.GraphPort([start])
        ports.append(p)
        node.add_input_port("x%d" % i, _linked(p))
    node.add_output_port("chi2", bff.GraphPort([0.0]))
    node.set_output_is_log_likelihood(True)
    return node, ports, dyes, meas


def test_objective_publishes_log_likelihood_with_the_sampler_log_mode():
    """MCMCSampler interprets its log-likelihood port as log L, not -log L."""
    node, ports, dyes, meas = _objective(70.0)
    node.update()
    logl = bff.nps_network_log_likelihood(
        _config_at(70.0), dyes, meas)
    got = node.get_output_port("chi2").get_value()
    assert got == pytest.approx(logl, rel=1e-12)


def test_sampler_records_minus_twice_the_actual_forward_log_likelihood():
    sampler, dyes, meas = _sampler(n_steps=8)
    chain = np.asarray(sampler.get_chain())
    chi2 = np.asarray(sampler.get_chi2())
    for row, recorded in zip(chain[:12], chi2[:12]):
        config = [list(row[:3]) + [0.0, 0.0],
                  list(row[3:]) + [0.0, 0.0]]
        expected = -2.0 * bff.nps_network_log_likelihood(config, dyes, meas)
        assert recorded == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_objective_perfect_agreement_hits_the_oracle_value():
    """sigma = 0.1 with model == data: log L = -log(sigma sqrt(2 pi))
    = 1.383646559789373, the value the PRD pins for this case."""
    dyes, meas = _world()
    config = _config_at(70.0)
    e_model = bff.nps_network_fret_efficiency(config, dyes, 0, 1, 55.0 ** 6)
    meas[0].e_avg = e_model
    node = bff.NPSNetworkObjective(dyes, meas, 2)
    for i, start in enumerate([0.0, 0.0, 0.0, 70.0, 0.0, 0.0]):
        p = bff.GraphPort([start])
        node.add_input_port("x%d" % i, _linked(p))
    node.add_output_port("chi2", bff.GraphPort([0.0]))
    node.set_output_is_log_likelihood(True)
    node.update()
    assert node.get_output_port("chi2").get_value() == pytest.approx(
        1.383646559789373, rel=1e-10)


def test_objective_recomputes_when_a_port_moves():
    node, ports, dyes, meas = _objective(70.0)
    node.update()
    first = node.get_output_port("chi2").get_value()
    ports[3].set_value(90.0)  # the separation port (dye 1 x)
    node.update()
    second = node.get_output_port("chi2").get_value()
    logl = bff.nps_network_log_likelihood(
        _config_at(90.0), dyes, meas)
    assert second == pytest.approx(logl, rel=1e-12)
    assert second != pytest.approx(first, rel=1e-6)


def test_objective_chi2_mode_reports_the_chi_square():
    dyes, meas = _world()
    node = bff.NPSNetworkObjective(dyes, meas, 2)
    for i, start in enumerate([0.0, 0.0, 0.0, 70.0, 0.0, 0.0]):
        p = bff.GraphPort([start])
        node.add_input_port("x%d" % i, _linked(p))
    node.add_output_port("chi2", bff.GraphPort([0.0]))
    # default: the output is read as chi2
    node.update()
    assert node.get_output_port("chi2").get_value() == pytest.approx(
        bff.nps_network_chi2(_config_at(70.0), dyes, meas), rel=1e-12)


# ---------------------------------------------------------------------------
# the sampler loop, end to end, with the diagnostics
# ---------------------------------------------------------------------------

def _sampler(n_steps=250, seed=11):
    node, ports, dyes, meas = _objective(70.0)
    sampler = bff.MCMCSampler("stretch", seed)
    sampler.set_parameter_ports(ports)
    sampler.set_objective(node, "chi2")
    sampler.set_output_is_log_likelihood(True)
    # Seed near the isotropic FRET likelihood optimum: E = 0.6 implies
    # (r/R0)^6 = (1-E)/E = 2/3. This fixture deliberately omits priors
    # and is a wiring check, not a normalizable posterior demonstration.
    rng = np.random.default_rng(4)
    start = [[float(v) for v in rng.normal(0.0, 0.5, 6)] for _ in range(12)]
    for row in start:
        row[3] += 55.0 * (2.0 / 3.0) ** (1.0 / 6.0)
    sampler.set_walker_start(start)
    sampler.run(n_steps)
    return sampler, dyes, meas


def test_sampler_runs_the_network_posterior():
    sampler, dyes, meas = _sampler()
    chain = np.asarray(sampler.get_chain())
    # six coordinates (two dyes x y z), one recorded row per (step, walker)
    assert chain.shape[1] == 6
    assert sampler.get_acceptance_rate() > 0.01
    assert np.all(np.isfinite(sampler.get_log_prob()))


def test_cross_entropy_tracks_the_recorded_log_prob():
    """The tracker is exactly -mean(log L) of the recorded states, which
    in log-likelihood mode is -mean(lnpost): mean(get_log_prob())."""
    sampler, dyes, meas = _sampler()
    ce = bff.nps_cross_entropy(
        sampler.get_chi2(), sampler.get_lnprior())
    assert ce == pytest.approx(-float(np.mean(sampler.get_log_prob())),
                               rel=1e-12)


def test_best_recorded_log_likelihood_respects_gaussian_density_bound():
    """With one fixed-sigma datum, log L cannot exceed -log(sigma sqrt(2pi)).

    This is a direct bound on the normalized density, not a chi2 goodness-
    of-fit test or an assertion about degrees of freedom of the sampler.
    """
    sampler, dyes, meas = _sampler()
    recorded_minus2logl = np.asarray(sampler.get_chi2())
    bound = 2.0 * math.log(meas[0].e_err * math.sqrt(2.0 * math.pi))
    assert np.all(recorded_minus2logl >= bound - 1e-10)
    assert recorded_minus2logl.min() < bound + 0.5


def test_mean_mcse_uses_the_committed_estimator():
    sampler, dyes, meas = _sampler()
    mcse = bff.nps_mean_mcse(sampler.get_chain_of_walker(0))
    # one value per coordinate (six here)
    assert len(mcse) == 6
    assert all(math.isfinite(v) and v >= 0.0 for v in mcse)
    # agrees with the committed Vehtari estimator on the same draws
    # (mcse_mean takes chains x draws: one chain, this coordinate's column)
    rows = np.asarray(sampler.get_chain_of_walker(0))
    from_estimator = bff.mcse_mean([[float(v) for v in rows[:, 0]]])
    assert mcse[0] == pytest.approx(from_estimator, rel=1e-12)
    # The likelihood is translation-invariant, so the absolute positions
    # are flat directions: their MCSEs are huge (the ensemble diffuses),
    # and the tracker reporting that honestly is the point of a
    # diagnostic. The identifiable quantity is the separation, and the
    # chi2-consistency test below pins the sampler on it.


def test_diagnostics_functions_are_exposed():
    """rhat/ess come from SamplerDiagnostics.h; the exposure makes them
    first-class for the sampler report."""
    rng = np.random.default_rng(3)
    chains = [list(map(float, rng.normal(0.0, 1.0, 200))) for _ in range(4)]
    rhat = bff.rhat_rank(chains)
    ess = bff.ess_bulk(chains)
    assert math.isfinite(rhat) and rhat < 1.1
    assert math.isfinite(ess) and ess > 100.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# the objective is a position-only posterior: construction rejects
# orientation-dependent likelihoods (the node fixes m = 0, phi = 0 and
# exposes no angle ports, so any likelihood that reads dye orientation
# would be silently conditioned on that arbitrary orientation)
# ---------------------------------------------------------------------------

def _angle_free_world():
    """dep = 0 dyes: r_avg = 0, fully depolarized, kappa^2 = 2/3 -- the
    one direct-branch chemistry whose likelihood provably ignores the
    frozen angles (avg_k2 = 2/3 for every orientation)."""
    dyes = []
    for _ in range(2):
        dye = bff.NPSNetworkDye()
        dye.dep = 0.0
        dye.iso = False
        dye.dist_conv = False
        dyes.append(dye)
    meas = bff.NPSMeasurement()
    meas.dye1 = 0
    meas.dye2 = 1
    meas.fret_active = True
    meas.e_avg = 0.6
    meas.e_err = 0.1
    meas.r_iso6 = 55.0 ** 6
    return dyes, [meas]


def _ports(node, values):
    ports = []
    for i, start in enumerate(values):
        p = bff.GraphPort([start])
        ports.append(p)
        link = bff.GraphPort([0.0])
        link.set_link(p)
        node.add_input_port("x%d" % i, link)
    return ports


def test_objective_accepts_angle_free_depolarized_direct_dyes():
    dyes, meas = _angle_free_world()
    node = bff.NPSNetworkObjective(dyes, meas, 2)
    _ports(node, [0.0, 0.0, 0.0, 70.0, 0.0, 0.0])
    node.add_output_port("chi2", bff.GraphPort([0.0]))
    node.set_output_is_log_likelihood(True)
    node.update()
    logl = bff.nps_network_log_likelihood(_config_at(70.0), dyes, meas)
    assert node.get_output_port("chi2").get_value() == pytest.approx(
        logl, rel=1e-12)


def test_objective_rejects_oriented_direct_dyes():
    """dep > 0 direct dyes: avg_kappa_2 reads cThT/cTh1/cTh2, which the
    node can never sample (angles frozen at m=0, phi=0)."""
    dyes, meas = _angle_free_world()
    dyes[0].dep = 0.5
    with pytest.raises(IMP.ValueException):
        bff.NPSNetworkObjective(dyes, meas, 2)


def test_objective_rejects_orientation_grid_pairs():
    """Convolved non-iso pairs select their coefficient row by the dyes'
    m/phi -- frozen angles would pin one arbitrary row of the 625."""
    dyes, meas = _angle_free_world()
    for d in dyes:
        d.dist_conv = True
        d.iso = False
    meas[0].set_eff_conv_coeff(
        [[0.4] + [0.0] * 11 for _ in range(625)])
    with pytest.raises(IMP.ValueException):
        bff.NPSNetworkObjective(dyes, meas, 2)


def test_objective_accepts_iso_convoluted_pairs():
    """iso/iso convolved pairs read row 0 only -- angle-free."""
    dyes, meas = _angle_free_world()
    for d in dyes:
        d.dist_conv = True
        d.iso = True
    meas[0].set_eff_conv_coeff([[0.4] + [0.0] * 11])
    node = bff.NPSNetworkObjective(dyes, meas, 2)
    _ports(node, [0.0, 0.0, 0.0, 70.0, 0.0, 0.0])
    node.add_output_port("chi2", bff.GraphPort([0.0]))
    node.update()
    # row 0 is the constant polynomial 0.4
    assert node.get_output_port("chi2").get_value() == pytest.approx(
        bff.nps_network_chi2(_config_at(70.0), dyes, meas), rel=1e-12)


def test_objective_rejects_transfer_anisotropy_on_ordered_dyes():
    """TA = dep1*dep2*(3 cThT^2 - 1)/5 reads the mutual angle; with both
    dyes ordered (dep1*dep2 > 0) it is orientation-dependent."""
    dyes, meas = _angle_free_world()
    dyes[0].dep = 0.5
    dyes[1].dep = 0.5
    meas[0].ta_active = True
    meas[0].r_t_avg = 0.1
    meas[0].r_t_err = 0.02
    with pytest.raises(IMP.ValueException):
        bff.NPSNetworkObjective(dyes, meas, 2)


def test_objective_accepts_transfer_anisotropy_with_one_depolarized_dye():
    """dep1*dep2 = 0: TA is identically zero -- angle-free. FRET must be
    off here: the direct branch needs BOTH dyes depolarized to lose its
    c^2 * dep * (1 - dep) angle terms."""
    dyes, meas = _angle_free_world()
    dyes[0].dep = 0.5  # ordered donor, fully depolarized acceptor
    meas[0].fret_active = False
    meas[0].ta_active = True
    meas[0].r_t_avg = 0.0
    meas[0].r_t_err = 0.02
    node = bff.NPSNetworkObjective(dyes, meas, 2)
    _ports(node, [0.0, 0.0, 0.0, 70.0, 0.0, 0.0])
    node.add_output_port("chi2", bff.GraphPort([0.0]))
    node.set_output_is_log_likelihood(True)
    node.update()
    logl = bff.nps_network_log_likelihood(_config_at(70.0), dyes, meas)
    assert node.get_output_port("chi2").get_value() == pytest.approx(
        logl, rel=1e-12)


def test_objective_rejects_out_of_range_measurement_dyes():
    dyes, meas = _angle_free_world()
    meas[0].dye2 = 5  # no such dye in a 2-dye network
    with pytest.raises(IMP.ValueException):
        bff.NPSNetworkObjective(dyes, meas, 2)


def test_cross_entropy_skips_non_finite_priors():
    """A chain row with a non-finite prior is as unscorable as one with
    chi2 = inf; it must be skipped, not averaged in as NaN."""
    ce = bff.nps_cross_entropy([1.0, 2.0, 3.0],
                               [0.5, float("nan"), -0.5])
    finite = bff.nps_cross_entropy([1.0, 3.0], [0.5, -0.5])
    assert ce == pytest.approx(finite, rel=1e-12)


def test_cross_entropy_rejects_all_unscorable_records():
    with pytest.raises(IMP.ValueException):
        bff.nps_cross_entropy([1.0, 2.0], [float("nan"), float("inf")])
