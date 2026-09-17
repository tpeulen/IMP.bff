"""Exact linear-Gaussian inference: canonical forms and variable elimination (PRD-151).

The contract, in the order a mistake would break it:

* the canonical-form algebra is aGrUM's -- product, division, reduce (evidence
  out of scope ignored), marginalize, fromCLG -- checked against the recorded
  transcription in ``agrum_clg_reference.py`` on random forms whose scopes
  overlap in every arrangement;
* it is also *right*: conditioning is the textbook Gaussian conditional,
  marginalising is slicing the covariance, and the mass survives both;
* a singular precision is carried, not refused: it multiplies, conditions and
  marginalises, names its null space, and only the moments refuse it -- while
  integrating out an unconstrained direction is refused, because it diverges;
* variable elimination over an ``InferenceFactorGraph`` reproduces aGrUM's
  ``CLGVariableElimination.canonicalPosterior`` on three networks, normalised
  and not, and both equal the brute-force conditional of the joint Gaussian;
* the ``"weighted"`` elimination order is aGrUM's default triangulation;
* relevance pruning changes the work, never the answer.
"""

import json
import pathlib

import numpy as np
import pytest

import IMP.bff as bff

FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "data" / "clg_agrum_fixture.json").read_text()
)


def _k(form):
    d = form.get_dimension()
    return np.asarray(form.get_precision()).reshape(d, d)


def _cov(form):
    d = form.get_dimension()
    return np.asarray(form.get_covariance()).reshape(d, d)


def _as_named(entry, rename=str):
    """A fixture form as (names, K, h, g)."""
    return [rename(n) for n in entry["scope"]], np.asarray(entry["K"]), np.asarray(entry["h"]), entry["g"]


def _form(entry, rename=str):
    names, K, h, g = _as_named(entry, rename)
    return bff.InferenceCanonicalForm(names, K.ravel(), h, g)


def _assert_same(form, entry, rename=str, atol=1e-12):
    """Equal as named factors: the scope order may differ, the function may not."""
    names, K, h, g = _as_named(entry, rename)
    got = list(form.get_names())
    assert sorted(got) == sorted(names)
    perm = [got.index(n) for n in names]
    np.testing.assert_allclose(_k(form)[np.ix_(perm, perm)], K, rtol=0, atol=atol)
    np.testing.assert_allclose(np.asarray(form.get_information())[perm], h, rtol=0, atol=atol)
    assert form.get_log_constant() == pytest.approx(g, abs=atol)


def _gaussian(seed=0, d=4):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d))
    return [f"x{i}" for i in range(d)], rng.normal(size=d) * 3.0, a @ a.T + d * np.eye(d)


# --- the algebra against aGrUM ---------------------------------------------------


@pytest.mark.parametrize("case", FIXTURE["form_operations"], ids=lambda c: str(c["f"]["scope"]))
def test_the_algebra_is_agrums(case):
    f, g = _form(case["f"]), _form(case["g"])
    _assert_same(f.product(g), case["product"])
    _assert_same(f.divide(g), case["division"])
    product = f.product(g)
    evidence = case["reduce"]["evidence"]
    _assert_same(product.reduce(list(evidence), list(evidence.values())), case["reduce"]["result"])
    _assert_same(product.marginalize([str(v) for v in case["marginalize"]["variables"]]),
                 case["marginalize"]["result"])
    # reduce ignores a variable the factor does not read (canonicalForm.py 386)
    _assert_same(f.reduce(["99"], [1.0]), case["reduce_out_of_scope"])


def test_from_linear_gaussian_is_fromclg():
    for network in FIXTURE["networks"].values():
        parents = {n["name"]: [] for n in network["nodes"]}
        weights = {n["name"]: [] for n in network["nodes"]}
        for arc in network["arcs"]:
            parents[arc["child"]].append(arc["parent"])
            weights[arc["child"]].append(arc["coef"])
        for node in network["nodes"]:
            name = node["name"]
            form = bff.InferenceCanonicalForm.from_linear_gaussian(
                name, parents[name], node["mu"], node["sigma"], weights[name])
            _assert_same(form, network["factors"][name])


# --- the algebra is right ---------------------------------------------------------


def test_moments_round_trip_and_mass():
    names, mean, cov = _gaussian()
    for mass in (0.0, -12.5, 3.25):
        form = bff.InferenceCanonicalForm.from_moments(names, mean, cov.ravel(), mass)
        np.testing.assert_allclose(form.get_mean(), mean, rtol=1e-12)
        np.testing.assert_allclose(_cov(form), cov, rtol=1e-12)
        np.testing.assert_allclose(_k(form) @ cov, np.eye(4), atol=1e-12)
        assert form.get_log_normalizer() == pytest.approx(mass, abs=1e-12)


def test_conditioning_is_the_textbook_conditional():
    names, mean, cov = _gaussian(seed=4, d=5)
    form = bff.InferenceCanonicalForm.from_moments(names, mean, cov.ravel())
    held = {"x1": 2.0, "x4": -1.5}
    b = [names.index(n) for n in held]
    a = [i for i in range(5) if i not in b]
    v = np.array(list(held.values()))
    s_ab, s_bb = cov[np.ix_(a, b)], cov[np.ix_(b, b)]
    want_mean = mean[a] + s_ab @ np.linalg.solve(s_bb, v - mean[b])
    want_cov = cov[np.ix_(a, a)] - s_ab @ np.linalg.solve(s_bb, cov[np.ix_(b, a)])

    got = form.condition(list(held), list(held.values()))
    assert list(got.get_names()) == [names[i] for i in a]
    err_mean = np.abs(np.asarray(got.get_mean()) - want_mean).max()
    err_cov = np.abs(_cov(got) - want_cov).max()
    assert err_mean < 1e-12 and err_cov < 1e-12, (err_mean, err_cov)
    # And the conditioned form's mass is the density of the held values.
    marginal_b = bff.InferenceCanonicalForm.from_moments(list(held), mean[b], s_bb.ravel())
    assert got.get_log_normalizer() == pytest.approx(marginal_b.get_log_density(v), abs=1e-12)


def test_marginalising_is_slicing_and_keeps_the_mass():
    names, mean, cov = _gaussian(seed=2, d=5)
    form = bff.InferenceCanonicalForm.from_moments(names, mean, cov.ravel(), -7.5)
    keep = ["x3", "x0"]
    idx = [names.index(n) for n in keep]
    marginal = form.marginal(keep)
    assert list(marginal.get_names()) == keep
    assert np.abs(np.asarray(marginal.get_mean()) - mean[idx]).max() < 1e-12
    assert np.abs(_cov(marginal) - cov[np.ix_(idx, idx)]).max() < 1e-12
    assert marginal.get_log_normalizer() == pytest.approx(-7.5, abs=1e-12)
    # aGrUM's spelling names what goes, ChiSurf's what stays; same answer.
    gone = form.marginalize(["x1", "x2", "x4"])
    assert list(gone.get_names()) == ["x0", "x3"]
    np.testing.assert_allclose(_k(gone), _k(form.marginal(["x0", "x3"])), rtol=1e-13)


def test_variables_holding_several_numbers():
    """A 3-number block conditions and marginalises as a block."""
    names, mean, cov = _gaussian(seed=9, d=5)
    block = bff.InferenceCanonicalForm.from_moments(["curve", "tau", "amp"], mean, cov.ravel(), 0.0, [3, 1, 1])
    scalar = bff.InferenceCanonicalForm.from_moments(names, mean, cov.ravel())
    got = block.condition(["curve"], [0.1, 0.2, 0.3]).get_mean()
    want = scalar.condition(["x0", "x1", "x2"], [0.1, 0.2, 0.3]).get_mean()
    np.testing.assert_allclose(got, want, rtol=1e-12)
    np.testing.assert_allclose(_cov(block.marginal(["curve"])), cov[:3, :3], rtol=1e-12)
    assert block.get_offset("tau") == 3


def test_refusals():
    names, mean, cov = _gaussian(seed=6, d=3)
    form = bff.InferenceCanonicalForm.from_moments(names, mean, cov.ravel())
    with pytest.raises(ValueError):
        form.marginal(["x0", "nope"])
    with pytest.raises(ValueError):
        form.condition(["nope"], [1.0])
    with pytest.raises(ValueError, match="unique"):
        bff.InferenceCanonicalForm(["a", "a"], np.eye(2).ravel(), [0.0, 0.0])
    with pytest.raises(ValueError, match="unique"):
        form.marginal(["x0", "x0"])
    with pytest.raises(ValueError):
        bff.InferenceCanonicalForm.from_moments(["a", "b"], [0, 0], [1.0, 1.0, 1.0, 1.0])


def test_the_properness_decision_does_not_depend_on_units():
    """An amplitude of 1e6 beside a lifetime of 1e-9 is an ordinary posterior."""
    cov = np.array([[1e12, 0.3 * 1e6 * 1e-10], [0.3 * 1e6 * 1e-10, 1e-20]])
    form = bff.InferenceCanonicalForm.from_moments(["amp", "tau"], [1e6, 4e-9], cov.ravel())
    assert form.get_is_proper()
    assert form.get_rank() == 2
    np.testing.assert_allclose(_cov(form), cov, rtol=1e-9)


# --- a singular precision ---------------------------------------------------------


def test_a_singular_precision_is_carried_and_explained():
    case = next(c for c in FIXTURE["singular"] if c["name"] == "sum_only")
    f = _form(case["form"])  # constrains a + b only
    assert not f.get_is_proper()
    assert f.get_rank() == 1
    null = np.asarray(f.get_null_space()).reshape(2, -1)
    np.testing.assert_allclose(np.abs(null[:, 0]), [2**-0.5, 2**-0.5], atol=1e-12)
    assert f.get_log_normalizer() == np.inf

    # Integrating b out leaves a flat in a -- aGrUM's toGaussian then fails on inv().
    marg = f.marginalize(["1"])
    _assert_same(marg, case["marginalize_1"])
    assert case["marginal_to_gaussian_raises"]
    assert marg.get_rank() == 0
    np.testing.assert_allclose(marg.get_null_space(), [1.0])
    with pytest.raises(ValueError, match="rank 0 of 1"):
        marg.get_mean()

    # Conditioning or a proper prior makes it a Gaussian again, exactly as aGrUM.
    red = f.reduce(["1"], [0.5])
    _assert_same(red, case["reduce_1"])
    np.testing.assert_allclose(red.get_mean(), case["reduce_1_mean"], atol=1e-12)
    prod = f.product(_form(case["prior_0"]))
    _assert_same(prod, case["product_with_prior"])
    np.testing.assert_allclose(prod.get_mean(), case["product_mean"], atol=1e-12)
    np.testing.assert_allclose(_cov(prod), case["product_cov"], atol=1e-12)


def test_integrating_out_an_unconstrained_direction_is_refused():
    case = next(c for c in FIXTURE["singular"] if c["name"] == "unconstrained")
    assert case["marginalize_1_raises"]  # aGrUM: numpy's LinAlgError, by accident
    with pytest.raises(ValueError, match="diverges"):
        _form(case["form"]).marginalize(["1"])


# --- variable elimination over the factor graph ----------------------------------


def _elimination(network):
    graph = bff.InferenceFactorGraph()
    for i, node in enumerate(network["nodes"]):
        graph.add_variable(node["name"], node["name"], i)
    parents = {n["name"]: [] for n in network["nodes"]}
    weights = {n["name"]: [] for n in network["nodes"]}
    for arc in network["arcs"]:
        parents[arc["child"]].append(arc["parent"])
        weights[arc["child"]].append(arc["coef"])
    for node in network["nodes"]:
        name = node["name"]
        graph.add_factor(f"p({name})", bff.INFERENCE_FACTOR_LIKELIHOOD, [name] + parents[name])
    ve = bff.InferenceGaussianElimination(graph)
    for node in network["nodes"]:
        name = node["name"]
        ve.set_factor(f"p({name})", bff.InferenceCanonicalForm.from_linear_gaussian(
            name, parents[name], node["mu"], node["sigma"], weights[name]))
    return graph, ve


QUERIES = [(name, i) for name, net in FIXTURE["networks"].items() for i in range(len(net["queries"]))]


@pytest.mark.parametrize("name,i", QUERIES)
def test_posteriors_are_agrums_and_the_brute_force_conditional(name, i):
    network = FIXTURE["networks"][name]
    query = network["queries"][i]
    graph, ve = _elimination(network)
    for variable, value in query["evidence"].items():
        ve.set_evidence(variable, [value])
    # aGrUM eliminates in its default triangulation's order; so does bff.
    assert list(graph.get_elimination_order("weighted")) == query["elimination_order"]

    posterior = ve.get_posterior(query["targets"], query["normalized"])
    _assert_same(posterior, query["posterior"], atol=1e-11)
    if query["normalized"]:
        order = [query["targets"].index(n) for n in query["posterior"]["scope"]]
        mean = np.asarray(posterior.get_mean())[order]
        cov = _cov(posterior)[np.ix_(order, order)]
        assert np.abs(mean - query["brute_mean"]).max() < 1e-12
        assert np.abs(cov - query["brute_covariance"]).max() < 1e-12
        assert posterior.get_log_normalizer() == pytest.approx(0.0, abs=1e-12)
        # The unnormalised posterior integrates to p(evidence), as does the whole product.
        whole = ve.get_posterior(query["targets"], False)
        assert whole.get_log_normalizer() == pytest.approx(query["brute_log_evidence"], abs=1e-12)
        assert ve.get_log_evidence() == pytest.approx(query["brute_log_evidence"], abs=1e-12)


def test_an_observed_target_is_refused():
    _, ve = _elimination(FIXTURE["networks"]["diamond"])
    ve.set_evidence("D", [1.0])
    with pytest.raises(ValueError, match="observed"):
        ve.get_posterior(["D"])


def test_relevance_pruning_changes_the_work_not_the_answer():
    """Evidence on the separator cuts the rest of the graph off."""
    network = FIXTURE["networks"]["chain"]  # S0 - S1 - ... - S5
    graph, ve = _elimination(network)
    ve.set_evidence("S2", [0.4])
    assert list(ve.get_relevant_factors(["S0"])) == ["p(S0)", "p(S1)", "p(S2)"]
    pruned = ve.get_posterior(["S0"])
    unpruned = ve.get_posterior(["S0"], False)
    np.testing.assert_allclose(pruned.get_mean(), unpruned.get_mean(), rtol=1e-13)
    np.testing.assert_allclose(_cov(pruned), _cov(unpruned), rtol=1e-13)

    # A factor beyond the evidence may even be non-Gaussian: it is never touched.
    bare = bff.InferenceGaussianElimination(graph)
    for f in ("p(S0)", "p(S1)", "p(S2)"):
        bare.set_factor(f, ve.get_factor(f))
    bare.set_evidence("S2", [0.4])
    np.testing.assert_allclose(bare.get_posterior(["S0"]).get_mean(), pruned.get_mean(), rtol=1e-13)
    with pytest.raises(ValueError, match="no Gaussian"):
        bare.get_log_evidence()


def test_a_factor_cannot_read_outside_its_scope():
    graph, ve = _elimination(FIXTURE["networks"]["collider"])
    with pytest.raises(ValueError, match="does not read"):
        ve.set_factor("p(X)", bff.InferenceCanonicalForm.from_linear_gaussian("X", ["W"], 0.0, 1.0, [1.0]))


# --- the elimination order ----------------------------------------------------------


@pytest.mark.parametrize("case", FIXTURE["orders"], ids=lambda c: f"n{c['n']}")
def test_the_weighted_order_is_agrums_default_triangulation(case):
    graph = bff.InferenceFactorGraph()
    for v in range(case["n"]):
        graph.add_variable(f"v{v}", f"v{v}", v, -1, case["sizes"][v])
    for a, b in case["edges"]:
        graph.add_factor(f"f{a}_{b}", bff.INFERENCE_FACTOR_LIKELIHOOD, [f"v{a}", f"v{b}"])
    for v in range(case["n"]):  # isolated variables need no factor, but give each a prior
        graph.add_factor(f"prior{v}", bff.INFERENCE_FACTOR_PRIOR, [f"v{v}"])
    assert list(graph.get_elimination_order("weighted")) == [f"v{v}" for v in case["order"]]
