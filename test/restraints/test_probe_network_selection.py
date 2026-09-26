"""ProbeNetworkSelection: one greedy selector, a weighted mix of term losses.

Pins that the resolution term alone is Olga (select_probe_pairs and
select_probe_positions, bit for bit in the order), that the kinetics term's
rate information is the Schur complement of the pair's per-burst Fisher
information, that the labelling term follows the Labelizer's scores and
rewards reusing a site, and that mixing the terms moves the selection.
"""

import numpy as np
import pytest

import IMP.bff as bff


def _symmetric(rng, n, hi):
    a = rng.uniform(0.0, hi, (n, n))
    a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.0)
    return a


def _olga_case(seed=0, n_frames=30, n_pairs=25):
    rng = np.random.default_rng(seed)
    effs = rng.uniform(0.05, 0.95, (n_frames, n_pairs))
    rmsds = _symmetric(rng, n_frames, 12.0)
    return effs, rmsds


# --- resolution: exactly Olga ---------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_resolution_alone_is_select_probe_pairs(seed):
    effs, rmsds = _olga_case(seed)
    ref_pairs, ref_decay = bff.select_probe_pairs(effs, rmsds, measurement_error=0.05,
                                                  max_pairs=8)
    term = bff.ProbeResolutionTerm(effs, rmsds, 0.05)
    sel = bff.ProbeNetworkSelection(effs.shape[1])
    sel.add_term(term)
    units, losses = sel.select(8)
    np.testing.assert_array_equal(units, ref_pairs)
    np.testing.assert_array_equal(sel.get_selected_pairs(), ref_pairs)
    np.testing.assert_allclose(np.asarray(losses) * term.get_initial_rmsd(), ref_decay,
                               rtol=1e-13)
    assert term.get_expected_rmsd() == pytest.approx(ref_decay[-1], rel=1e-15)


def _dimer_pair_sites(n_sites):
    """Every ordered combination of sites, homotypic ones included."""
    return np.array([(i, j) for i in range(n_sites) for j in range(n_sites)],
                    dtype=np.int32)


@pytest.mark.parametrize("seed", [0, 3])
def test_resolution_alone_is_select_probe_positions(seed):
    n_sites = 6
    ps = _dimer_pair_sites(n_sites)
    effs, rmsds = _olga_case(seed, n_frames=20, n_pairs=len(ps))
    ref_sites, ref_decay = bff.select_probe_positions(
        effs, rmsds, ps, measurement_error=0.05, max_sites=4)
    term = bff.ProbeResolutionTerm(effs, rmsds, 0.05)
    sel = bff.ProbeNetworkSelection(len(ps))
    sel.set_pair_sites(ps)
    sel.add_term(term)
    units, losses = sel.select(4, by_sites=True)
    np.testing.assert_array_equal(units, ref_sites)
    np.testing.assert_allclose(np.asarray(losses) * term.get_initial_rmsd(), ref_decay,
                               rtol=1e-13)


def test_resolution_loss_starts_at_one_and_decays():
    effs, rmsds = _olga_case(4)
    term = bff.ProbeResolutionTerm(effs, rmsds, 0.05)
    assert term.get_loss() == pytest.approx(1.0)
    sel = bff.ProbeNetworkSelection(effs.shape[1])
    sel.add_term(term)
    _, losses = sel.select(6)
    assert np.all(np.diff(losses) <= 1e-12)
    assert 0.0 < losses[-1] < 1.0


# --- kinetics: the rate information of each pair ---------------------------------------

def _three_states():
    """A <-> B <-> C, and A <-> C, rates in 1/ms."""
    hp = bff.FRETHiddenProcess(3)
    for (i, j, k) in [(0, 1, 2.0), (1, 0, 3.0), (1, 2, 1.5), (2, 1, 2.5),
                      (0, 2, 0.5), (2, 0, 0.7)]:
        hp.set_rate(i, j, k)
    return hp


def _template():
    """Steady dyes: one bright state each, so the hidden path is all there is."""
    d = bff.FRETDye("donor")
    d.add_state("bright", 1.0, 0.8, 4.0)
    a = bff.FRETDye("acceptor")
    a.add_state("bright", 1.0, 0.6, 3.0, 1.0)
    m = bff.FRETMeasurement("pair", d, a, 52.0)
    # Bright enough for ~40 photons in a 2 ms burst.
    exc = bff.PhotophysicsCrosstalkMatrix(["p"], ["donor", "acceptor"], [60.0, 1.0])
    em = bff.PhotophysicsCrosstalkMatrix(["donor", "acceptor"], ["g", "r"],
                                         [0.4, 0.03, 0.05, 0.45])
    ins = bff.FRETInstrument(exc, em, 1, 0.25)
    ins.set_background(0, 0.5)
    ins.set_background(1, 0.5)
    m.set_instrument(ins)
    return m


# Distances per state (rows) and candidate (columns). Candidates 0 and 1 see
# A apart from B and C; 2 and 3 see C apart from A and B; 4 sees B apart; 5 is
# the same in every state.
_DISTANCES = np.array([
    [40.0, 41.0, 70.0, 69.0, 70.0, 55.0],
    [70.0, 69.0, 70.0, 69.0, 40.0, 55.0],
    [70.0, 69.0, 40.0, 41.0, 70.0, 55.0],
])


def _options(n=150):
    o = bff.FRETSimulationOptions()
    o.n_molecules = n
    o.duration = 2.0
    o.seed = 7
    return o


@pytest.fixture(scope="module")
def kinetics():
    term = bff.ProbeKineticsTerm(_three_states(), _template(), options=_options(),
                                 n_bursts_per_pair=500.0)
    assert term.add_pairs(_DISTANCES) == 0
    return term


def test_kinetics_information_is_the_schur_complement(kinetics):
    hp = _three_states()
    p = 0
    m = _template()
    for h in range(3):
        m.set_state_distance(h, _DISTANCES[h, p])
    o = _options()
    o.seed = 7 + p
    data = bff.simulate_fret_measurement(hp, m, o)
    net = bff.FRETNetworkModel(hp)
    net.add_measurement(m, data)
    for name in net.get_parameter_names():
        want = name.startswith("hidden.") or name.startswith("pair.mean[")
        if net.get_parameter_free(name) != want:
            net.set_parameter_free(name, want)
    theta = np.asarray(net.get_theta())
    s = np.asarray(net.segment_scores(theta)).reshape(-1, len(theta))
    f = s.T @ s / s.shape[0]
    nk = len(kinetics.get_rate_names())
    fkk, fkd, fdd = f[:nk, :nk], f[:nk, nk:], f[nk:, nk:]
    g = (fkk - fkd @ np.linalg.pinv(fdd) @ fkd.T) * 500.0
    got = np.asarray(kinetics.get_pair_information(p)).reshape(nk, nk)
    np.testing.assert_allclose(got, 0.5 * (g + g.T), rtol=1e-8, atol=1e-8 * np.abs(g).max())
    assert kinetics.get_n_bursts(p) == s.shape[0]


def test_kinetics_static_pair_carries_no_rate_information(kinetics):
    nk = len(kinetics.get_rate_names())
    g = np.asarray(kinetics.get_pair_information(5)).reshape(nk, nk)
    assert np.abs(g).max() < 1e-6 * np.abs(
        np.asarray(kinetics.get_pair_information(0))).max()
    kinetics.reset()
    assert kinetics.get_loss() == pytest.approx(1.0)
    assert kinetics.get_loss_with([5], []) == pytest.approx(1.0, abs=1e-6)
    assert kinetics.get_loss_with([0], []) < 1.0


def test_kinetics_selection_decays_and_skips_the_static_pair(kinetics):
    sel = bff.ProbeNetworkSelection(_DISTANCES.shape[1])
    sel.add_term(kinetics)
    units, losses = sel.select(4)
    assert 5 not in list(units)
    assert np.all(np.diff(losses) <= 1e-12)
    assert np.all(np.asarray(kinetics.get_rate_sigmas()) < np.log(10.0))


def test_mixing_trades_resolution_for_rates(kinetics):
    # Predicted efficiencies per state for the resolution term; three frames
    # are the three states, their RMSDs a line A - B - C.
    e = 1.0 / (1.0 + (_DISTANCES / 52.0) ** 6)
    rmsds = np.array([[0.0, 6.0, 12.0], [6.0, 0.0, 6.0], [12.0, 6.0, 0.0]])
    res = bff.ProbeResolutionTerm(e, rmsds, 0.05)
    only_res = bff.ProbeNetworkSelection(e.shape[1])
    only_res.add_term(res)
    only_res.add_term(kinetics, 0.0)
    only_res.select(2)
    mixed = bff.ProbeNetworkSelection(e.shape[1])
    mixed.add_term(res, 1.0)
    mixed.add_term(kinetics, 1.0)
    mixed.select(2)
    l_res = np.asarray(only_res.get_term_losses())
    l_mix = np.asarray(mixed.get_term_losses())
    assert l_res.shape == (2, 2) and l_mix.shape == (2, 2)
    # The mix is never worse on the weighted sum, and its rates are at least
    # as well determined as with resolution alone.
    assert l_mix[-1].sum() <= l_res[-1].sum() + 1e-12
    assert l_mix[-1, 1] <= l_res[-1, 1] + 1e-12


# --- oligomers: the switch --------------------------------------------------------

def _oligomer_positions(n_protomers, n_sites=3, n_frames=3, seed=0):
    """Protomers placed around a Cn axis, sites scattered on each, per frame."""
    rng = np.random.default_rng(seed)
    local = rng.normal(0.0, 12.0, (n_frames, n_sites, 3)) + np.array([25.0, 0.0, 0.0])
    out = np.empty((n_frames, n_protomers, n_sites, 3))
    for a in range(n_protomers):
        t = 2 * np.pi * a / n_protomers
        rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
        out[:, a] = local @ rot.T
    return out


@pytest.mark.parametrize("n_protomers", [1, 2, 3, 4])
def test_oligomer_rows_and_mixtures(n_protomers):
    x = _oligomer_positions(n_protomers)
    pairs = bff.ProbeOligomerPairs(x)
    n_sites = 3
    ps = np.asarray(pairs.get_pair_sites())
    homotypic = [(i, i) for i in range(n_sites)] if n_protomers > 1 else []
    expected = sorted(homotypic + [(i, j) for i in range(n_sites)
                                   for j in range(i + 1, n_sites)])
    assert sorted(map(tuple, ps.tolist())) == expected
    for r, (i, j) in enumerate(ps.tolist()):
        n = pairs.get_n_components(r)
        if i == j:
            assert n == n_protomers * (n_protomers - 1) // 2
        else:
            assert n == n_protomers * n_protomers   # ordered inter + intra
        d = np.asarray(pairs.get_distances(1, r))
        want = sorted(np.linalg.norm(x[1, a, i] - x[1, b, j])
                      for a in range(n_protomers) for b in range(n_protomers)
                      if (a < b if i == j else True))
        np.testing.assert_allclose(sorted(d), want, rtol=1e-12)
    e = np.asarray(pairs.get_efficiencies(52.0))
    assert e.shape == (3, len(ps))
    r = 0
    d = np.asarray(pairs.get_distances(0, r))
    assert e[0, r] == pytest.approx(np.mean(1.0 / (1.0 + (d / 52.0) ** 6)))


def test_oligomer_without_intra_counts_only_cross_protomer_pairs():
    pairs = bff.ProbeOligomerPairs(_oligomer_positions(3), include_intra=False)
    for r, (i, j) in enumerate(np.asarray(pairs.get_pair_sites()).tolist()):
        assert pairs.get_n_components(r) == (3 if i == j else 6)


def test_trimer_selection_by_sites_with_resolution_and_kinetics():
    x = _oligomer_positions(3, n_sites=4, n_frames=3, seed=2)
    pairs = bff.ProbeOligomerPairs(x)
    ps = np.asarray(pairs.get_pair_sites())
    e = np.asarray(pairs.get_efficiencies(52.0))
    rmsds = np.array([[0.0, 6.0, 12.0], [6.0, 0.0, 6.0], [12.0, 6.0, 0.0]])
    kin = bff.ProbeKineticsTerm(_three_states(), _template(), options=_options(80),
                                n_bursts_per_pair=500.0)
    assert kin.add_oligomer_pairs(pairs, [0, 1, 2]) == 0
    assert kin.get_n_pairs() == len(ps)
    sel = bff.ProbeNetworkSelection(len(ps))
    sel.set_pair_sites(ps)
    sel.add_term(bff.ProbeResolutionTerm(e, rmsds, 0.05))
    sel.add_term(kin)
    sites, losses = sel.select(2, by_sites=True)
    assert len(sites) == 2 and len(set(sites)) == 2
    # Two sites on a trimer measure (i,i), (j,j) and (i,j): three rows.
    assert len(sel.get_selected_pairs()) == 3
    assert np.all(np.diff(losses) <= 1e-12)
    l = np.asarray(sel.get_term_losses())
    assert l[-1, 1] < 1.0


# --- labelling: the Labelizer's scores, and shared sites ------------------------------

def test_labelling_prefers_good_sites_and_reuse():
    keys = ["A10", "A20", "A30", "A40"]
    scores = {"A10": 9.0, "A20": 9.0, "A30": 0.1}   # A40 unscored
    term = bff.ProbeLabellingTerm(scores, keys, mutation_cost=1.0)
    assert term.get_site_cost(0) == pytest.approx((1.0 + 0.1) / 2.0)
    assert term.get_site_cost(2) == pytest.approx((1.0 + 1.0 / 1.1) / 2.0)
    assert not term.get_is_eligible_site(3)
    ps = np.array([[0, 1], [0, 2], [2, 1], [0, 3], [1, 0]], dtype=np.int32)
    sel = bff.ProbeNetworkSelection(len(ps))
    sel.set_pair_sites(ps)
    sel.add_term(term)
    units, _ = sel.select(2)
    # (0,1) first: two good sites; then (1,0) reuses both, costing nothing;
    # pair 3 touches the unscored site and is never selected.
    assert list(units) == [0, 4]
    assert list(term.get_committed_sites()) == [0, 1]


def test_excluded_units_are_never_selected():
    effs, rmsds = _olga_case(5, n_pairs=10)
    sel = bff.ProbeNetworkSelection(effs.shape[1])
    sel.add_term(bff.ProbeResolutionTerm(effs, rmsds, 0.05))
    first, _ = sel.select(1)
    sel.set_excluded([int(first[0])])
    units, _ = sel.select(5)
    assert int(first[0]) not in list(units)


def test_labelizer_scores_feed_the_labelling_term():
    path = bff.get_example_path("structure/T4L/3GUN.pdb")
    # No ConSurf grades ship for T4L, and under the published model a missing
    # term leaves every combined score unavailable; the coordinate-only terms
    # need nothing else (as test_labelizer_ab_male.py does for MalE).
    model = bff.LabelizerParameterList()
    for tag, table in (("se", "N_SE11_MEAN_SURFACE_DIST"), ("cr", "C_CR1_Name"),
                       ("ss", "C_SS1_SS")):
        model.append(bff.LabelizerParameter(tag, table, 1))
    rows = bff.labelizer_score_structure(path, model, bff.LabelizerOptions(), "")
    by_key = dict(bff.labelizer_combined_by_key(rows))
    keys = sorted(by_key)[:8]
    assert keys, "the Labelizer scored no residue"
    term = bff.ProbeLabellingTerm(by_key, keys + ["Z999"])
    assert all(term.get_is_eligible_site(i) for i in range(len(keys)))
    assert not term.get_is_eligible_site(len(keys))
    assert all(0.0 < term.get_site_cost(i) <= 1.0 for i in range(len(keys)))


# --- bad input ------------------------------------------------------------------------

def test_bad_inputs_raise():
    effs, rmsds = _olga_case(6, n_pairs=5)
    with pytest.raises(bff.ValueException):
        bff.ProbeResolutionTerm(effs, rmsds[:-1, :-1], 0.05)
    with pytest.raises(bff.ValueException):
        bff.ProbeNetworkSelection(5).select(2)
    sel = bff.ProbeNetworkSelection(4)
    with pytest.raises(bff.ValueException):
        sel.add_term(bff.ProbeResolutionTerm(effs, rmsds, 0.05))
    with pytest.raises(bff.ValueException):
        sel.set_pair_sites(np.zeros((3, 2), dtype=np.int32))
    term = bff.ProbeKineticsTerm(_three_states(), _template(), options=_options(5))
    with pytest.raises(bff.ValueException):
        term.add_pairs(_DISTANCES[:2])
    with pytest.raises(bff.ValueException):
        term.add_oligomer_pairs(bff.ProbeOligomerPairs(np.zeros((2, 2, 3, 3))), [0, 1])
    with pytest.raises(bff.ValueException):
        bff.ProbeOligomerPairs(np.zeros((2, 2, 3, 2)))
    sel = bff.ProbeNetworkSelection(5)
    sel.add_term(bff.ProbeResolutionTerm(effs, rmsds, 0.05))
    with pytest.raises(bff.ValueException):
        sel.select(2, by_sites=True)
