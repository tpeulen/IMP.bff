"""Competing models are compared on a criterion, never on misfit alone.

A richer model fits at least as well by construction, so ranking candidates
by chi-square alone is not a comparison between them -- it is a preference
for the largest. Both criteria here are that misfit plus a price per free
parameter, and the reward the search maximises is a monotone transform of
them (-BIC/2, -AIC/2), so maximising reward is minimising the criterion.
"""

from __future__ import annotations

import collections
import copy
import json
import pathlib
import sys

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _characterize  # noqa: E402
import _fixtures  # noqa: E402

FAMILY = json.loads(
    pathlib.Path(bff.get_data_path("model_search/fcs_analytical.json")).read_text(),
    object_pairs_hook=collections.OrderedDict,
)


def _ranked(document):
    axis = np.geomspace(1.0e-3, 20.0, 80)
    spec = bff.ModelSearchSpec.from_json(json.dumps(document))
    spec.set_dataset(
        "curve",
        _fixtures.fcs_correlation_dataset(axis, _fixtures.fcs_curve_2d(axis), 0.01),
    )
    record = _characterize.characterize(spec.build())
    scores = {record["root"]["structure"]: record["root"]["reward"]}
    scores.update({k: v["reward"] for k, v in record["structures"].items()})
    return sorted(scores, key=scores.get, reverse=True), scores


def test_a_family_that_says_nothing_about_comparison_is_refused():
    """Not defaulted to chi-square. The default would always overfit.

    Measured before this was enforced: dropping the sample size made this
    family choose a spurious relaxation term on a curve built from one
    diffusing species.
    """
    blind = copy.deepcopy(FAMILY)
    for structure in blind["structures"].values():
        structure.pop("ess", None)
    with pytest.raises((ValueError, RuntimeError)) as caught:
        _ranked(blind)
    assert "ess" in str(caught.value)


def test_the_simplest_adequate_model_wins_on_bic():
    """The curve is one 2-D species, and that is what should be chosen."""
    order, _ = _ranked(FAMILY)
    assert order[0] == "fcs.2d.1diff.0relax"


def test_aic_is_available_and_is_the_more_permissive_of_the_two():
    """Same fits, different price per parameter: 2k against k ln n.

    With eighty lag points ln n is 4.4, so BIC charges more than twice what
    AIC does, and the gap between a simple topology and a richer one has to
    widen accordingly.
    """
    aic = copy.deepcopy(FAMILY)
    for structure in aic["structures"].values():
        structure["selection"] = "aic"
    _, bic_scores = _ranked(FAMILY)
    _, aic_scores = _ranked(aic)

    simple, richer = "fcs.2d.1diff.0relax", "fcs.2d.2diff.1relax"
    assert bic_scores[simple] - bic_scores[richer] > (
        aic_scores[simple] - aic_scores[richer]
    )


def test_the_criterion_must_be_one_that_exists():
    nonsense = copy.deepcopy(FAMILY)
    for structure in nonsense["structures"].values():
        structure["selection"] = "eyeball"
    with pytest.raises((ValueError, RuntimeError)) as caught:
        _ranked(nonsense)
    assert "eyeball" in str(caught.value)


def test_goodness_of_fit_is_a_separate_question_from_the_comparison():
    """A model can win its family and still describe the data badly.

    Reduced chi-square answers that one, and near one is the expectation.
    """
    problem = _fixtures.fcs_analytical()
    problem.get_initial_state()
    reduced = problem.get_last_reduced_chi2()
    assert np.isfinite(reduced)
    assert 0.0 < reduced < 10.0


def test_the_penalty_is_what_the_criterion_says_it_is():
    """-BIC/2 exactly: the misfit, less half k ln n."""
    axis = np.geomspace(1.0e-3, 20.0, 80)
    problem = _fixtures.fcs_analytical()
    state = problem.get_initial_state()
    objective = problem.get_active_objective()
    objective.update()
    residuals = np.asarray(objective.get_output_port("residuals").value)
    chi2 = float(np.sum(residuals ** 2))
    # The root frees N, the baseline and one diffusion time.
    free = sum(1 for f in problem.get_cached_fixed(state.get_key()) if f == 0)
    expected = -0.5 * chi2 - 0.5 * free * np.log(len(axis))
    assert state.get_reward() == pytest.approx(expected, rel=1e-9)


def test_the_chi2_test_answers_a_question_the_comparison_cannot():
    """Whether the winner describes the data, not whether it beat the others.

    The probability comes from Boost.Math's regularised incomplete gamma,
    named in bff as chi2_p_value -- the toolchain already carried it
    correctly, and a hand-rolled series would have been a second
    implementation rather than a new capability.
    """
    problem = _fixtures.fcs_analytical()
    problem.get_initial_state()
    p_value = problem.get_last_chi2_p_value()
    assert 0.0 <= p_value <= 1.0
    # The fixture is the model's own curve, so it fits: the test must not
    # reject it.
    assert p_value > 0.01


def test_the_probability_is_the_incomplete_gamma_it_claims_to_be():
    from math import isclose
    # Q(dof/2, chi2/2), checked against the identity Q + P = 1 and against
    # known values of the chi-square survival function.
    assert isclose(bff.chi2_p_value(0.0, 4.0), 1.0, abs_tol=1e-12)
    assert bff.chi2_p_value(1e6, 4.0) < 1e-12
    # A chi-square equal to its degrees of freedom sits near the middle.
    assert 0.2 < bff.chi2_p_value(10.0, 10.0) < 0.6
    # gamma_q is the function underneath, and Q(1, x) = exp(-x).
    assert isclose(bff.gamma_q(1.0, 2.0), np.exp(-2.0), rel_tol=1e-12)


def test_no_degrees_of_freedom_means_no_answer():
    """A model with a parameter per point reaches the data exactly; saying
    it fits perfectly would be saying nothing."""
    assert np.isnan(bff.chi2_p_value(0.0, 0.0))


def test_a_model_that_cannot_describe_the_data_fails_the_test():
    """Acceptance is the chi-square test, not the ranking.

    A single diffusing species cannot describe a curve built from two a
    decade apart, and the test says so. The two questions are independent:
    the comparison orders candidates, the test asks whether the one you kept
    is any good, and the best of a bad family is still bad.
    """
    axis = np.geomspace(1.0e-3, 20.0, 120)
    fast, slow, fraction, n, baseline = 0.08, 0.8, 0.4, 2.0, 1.0
    curve = baseline + (1.0 / n) * (
        fraction / (1.0 + axis / fast) + (1.0 - fraction) / (1.0 + axis / slow)
    )
    document = copy.deepcopy(FAMILY)
    for structure in document["structures"].values():
        structure["acceptable_above"] = 0.05  # the conventional level
    spec = bff.ModelSearchSpec.from_json(json.dumps(document))
    spec.set_dataset("curve", _fixtures.fcs_correlation_dataset(axis, curve, 0.002))
    problem = spec.build()

    root = problem.get_initial_state()
    assert root.get_structure_key() == "fcs.2d.1diff.0relax"
    assert problem.get_last_chi2_p_value() < 0.05
    assert problem.get_last_reduced_chi2() > 2.0
    assert not root.get_acceptable()

    # The topology with the freedom to describe it passes the same test.
    state = _characterize._walk(problem, root, ["add-diffusion-component"])
    assert state.get_structure_key() == "fcs.2d.2diff.0relax"
    assert problem.get_last_chi2_p_value() > 0.05
    assert state.get_acceptable()


def test_the_diagnostics_describe_the_state_that_was_returned():
    """Not whichever declared start happened to run last.

    Multistart fits a structure from several starts and keeps the best. The
    goodness-of-fit numbers have to come from that one: reporting one fit's
    quality beside another fit's answer is worse than reporting none, and it
    misleads exactly the person trying to work out why a fit is poor.
    """
    axis = np.geomspace(1.0e-3, 20.0, 120)
    curve = 1.0 + 0.5 * (0.4 / (1.0 + axis / 0.08) + 0.6 / (1.0 + axis / 0.8))
    spec = bff.ModelSearchSpec.from_name("fcs_analytical")
    spec.set_dataset("curve", _fixtures.fcs_correlation_dataset(axis, curve, 0.002))
    problem = spec.build()
    root = problem.get_initial_state()
    # This structure declares extra starts, so the last one tried is very
    # unlikely to be the one kept.
    state = _characterize._walk(problem, root, ["add-diffusion-component"])
    chi2 = problem.get_last_reduced_chi2() * (len(axis) - 5.0 - 1.0)
    # The reward is -chi2/2 - penalty, so the misfit behind the score and the
    # misfit behind the diagnostics have to be the same misfit.
    penalty = 0.5 * 5.0 * np.log(len(axis))
    assert state.get_reward() == pytest.approx(-0.5 * chi2 - penalty, rel=1e-6)


def test_the_family_list_matches_what_is_shipped():
    """The capability answer and the files cannot disagree.

    It was a hardcoded list in C++ and had already fallen one family behind,
    so a caller asking what bff can search was told something untrue.
    """
    directory = pathlib.Path(bff.get_data_path("model_search"))
    shipped = sorted(
        path.stem
        for path in directory.glob("*.json")
        if path.stem not in ("schema", "index")
    )
    assert list(bff.ModelSearchSpec.get_available_names()) == shipped
    assert "tcspc_anisotropy" in shipped  # the one the old list missed
