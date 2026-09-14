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
