"""A model family generating its own training data, and learning from it.

What this learns is *where the parameters start*, not which action to take,
and that choice came from measurement rather than taste: once candidates have
scores of their own the tree search already matches an exhaustive walk on
every fixture and seed, while reaching a topology's optimum from a generic
seed is what fails. See okf/validation/model-search-strategy.md.

The loop needs no hand-written simulator, which is what makes a family added
tomorrow trainable today: a description can already produce the curve it
predicts, so it can be run backwards.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402

STRUCTURE = "fcs.2d.2diff.0relax"
#: Mean squared error of predicting the mean of a uniform [0, 1] target.
PREDICTING_THE_MEAN = 1.0 / 12.0


def _spec():
    axis = np.geomspace(1.0e-3, 20.0, 120)
    fast, slow, fraction, n, baseline = 0.08, 0.8, 0.4, 2.0, 1.0
    curve = baseline + (1.0 / n) * (
        fraction / (1.0 + axis / fast) + (1.0 - fraction) / (1.0 + axis / slow)
    )
    spec = bff.ModelSearchSpec.from_name("fcs_analytical")
    spec.set_dataset("curve", _fixtures.fcs_correlation_dataset(axis, curve, 0.002))
    return spec


def _played(episodes=400, seed=3):
    play = bff.ModelSearchSelfPlay(_spec())
    play.set_structure(STRUCTURE)
    play.set_spread(0.5)
    play.set_poisson(False)  # a correlation curve is not a photon count
    play.generate(episodes, seed)
    return play


def test_a_family_plays_only_the_parameters_its_topology_frees():
    play = _played(20)
    # The relaxation terms are fixed in this topology; proposing them would be
    # proposing a value the model does not use.
    assert set(play.get_target_ids()) == {
        "fcs.N",
        "fcs.baseline",
        "fcs.diffusion_time.1",
        "fcs.diffusion_fraction.1",
        "fcs.diffusion_time.2",
    }


def test_episodes_are_measurements_that_differ():
    """The curves have to vary, or there is nothing in them to learn from.

    They once did not: sampling a baseline across its +/-1e6 bound instead of
    around its declared start made every episode a flat line at 10^6, with
    identical features and a network that could only predict the mean.
    """
    play = _played(32)
    features = np.asarray(play.get_features()).reshape(32, play.get_number_of_features())
    assert np.all(features.std(axis=0)[: play.get_number_of_features() - 1] > 1e-6)
    assert np.isfinite(features).all()


def test_the_same_seed_replays_the_same_episodes():
    first = np.asarray(_played(16, seed=5).get_features())
    again = np.asarray(_played(16, seed=5).get_features())
    assert first == pytest.approx(again)


def test_training_learns_something_rather_than_the_mean():
    """The bar is not a number someone chose; it is what guessing scores."""
    play = _played()
    play.train([64, 64], 800, 0.02)
    error = dict(zip(play.get_target_ids(), play.get_training_error()))
    for determined in ("fcs.N", "fcs.baseline"):
        assert error[determined] < 0.1 * PREDICTING_THE_MEAN, determined


def test_a_proposal_improves_the_seed_where_the_seed_is_weak():
    """Measured against the start it replaces, not against a chosen number.

    An absolute tolerance would have proved nothing here. The declared start
    is computed from the curve and is already within 3% on the amplitude and
    1% on the plateau, so a proposal "near the truth" on those is the seeding
    rule's achievement, not the network's. Where the seed is genuinely weak
    is the diffusion times -- a single characteristic lag stands in for both,
    290% out on the faster one -- and that is the parameter whose basin the
    search keeps missing. So the claim is comparative: the proposal must beat
    the seed there.
    """
    play = _played(600)
    network = play.train([64, 64], 1000, 0.02)
    ids = list(_spec().build().get_parameter_ids())
    proposed = dict(zip(ids, play.propose(network)))

    problem = _spec().build()
    problem.activate_structure(STRUCTURE)
    declared = {name: problem.get_parameter(name).value for name in ids}

    def error(value, truth):
        return abs(value - truth) / abs(truth)

    # The fast species: what a one-component seed cannot see.
    assert error(proposed["fcs.diffusion_time.1"], 0.08) < error(
        declared["fcs.diffusion_time.1"], 0.08
    )
    # And the two species stay ordered, which is what makes them two.
    assert proposed["fcs.diffusion_time.2"] > proposed["fcs.diffusion_time.1"]


def test_a_trained_proposer_is_read_back_through_the_ordinary_network():
    """One MLP implementation: trained here, loaded by IMP.bff.NeuralNet."""
    play = _played(64)
    network = play.train([16], 20, 0.02)
    net = bff.NeuralNet(network)
    assert net.get_n_inputs() == play.get_number_of_features()
    assert net.get_n_outputs() == play.get_number_of_targets()


def test_a_proposal_is_admissible_as_a_start():
    play = _played(64)
    network = play.train([16], 20, 0.02)
    problem = _spec().build()
    # One value per canonical parameter, in registry order, within bounds.
    problem.add_structure_start(STRUCTURE, play.propose(network))


def _policy_play(seed=13, episodes=40, photons=False):
    play = bff.ModelSearchSelfPlay(_spec())
    play.set_spread(0.35)
    play.set_poisson(False)
    return play.generate_policy(episodes, seed)


def test_a_described_search_takes_a_family_agnostic_policy():
    """ChiSurf's described models use this BFF-native problem directly."""
    width = bff.get_policy_state_width() + bff.get_policy_action_width()
    problem = _spec().build()
    problem.set_action_policy(json.dumps({
        "format": "bff.neural_net",
        "layers": [{"n_in": width, "n_out": 1, "weight": [0.01] * width,
                    "bias": [0.0], "activation": "identity"}],
    }))
    root = problem.get_initial_state()
    priors = [action.get_prior() for action in problem.get_actions(root)]

    assert problem.get_has_action_policy()
    assert sum(priors) == pytest.approx(1.0)


def test_policy_episodes_are_labelled_moves_and_include_stopping():
    data = _policy_play()

    assert data.get_number_of_episodes() > 20
    assert data.get_row_width() == (
        bff.get_policy_state_width() + bff.get_policy_action_width())
    assert list(data.get_families()) == ["fcs_analytical"]
    offsets = np.asarray(data.get_offsets())
    labels = np.asarray(data.get_labels())
    assert np.all(labels < np.diff(offsets))
    assert np.isfinite(np.asarray(data.get_rows())).all()
    # Stopping is a decision the data has to teach, not an afterthought.
    assert 0 < data.get_number_of_stop_labels() < data.get_number_of_episodes()


def test_policy_data_pools_across_families_and_round_trips():
    first = _policy_play(seed=1, episodes=10)
    second = _policy_play(seed=2, episodes=10)
    pooled = bff.ModelSearchPolicyData()
    pooled.extend(first)
    pooled.extend(second)

    assert pooled.get_number_of_episodes() == (
        first.get_number_of_episodes() + second.get_number_of_episodes())
    again = bff.ModelSearchPolicyData.from_json(pooled.to_json())
    assert list(again.get_rows()) == list(pooled.get_rows())
    assert list(again.get_labels()) == list(pooled.get_labels())


def test_training_ranks_held_out_moves_and_replays_from_its_seed():
    data = _policy_play(seed=9, episodes=60)
    trained = bff.train_action_policy(data, [16], 200, 0.02, 21, 0.25)

    held = int(0.25 * data.get_number_of_episodes())
    # Half the held-out episodes choose the epoch; the rest are only scored.
    assert trained.get_number_of_validation_episodes() == held - held // 2
    assert 0 <= trained.get_best_epoch() <= 200
    assert np.isfinite(trained.get_baseline_loss())
    assert 0.0 <= trained.get_validation_accuracy() <= 1.0
    assert np.isfinite(trained.get_validation_loss())
    assert list(trained.get_families()) == ["fcs_analytical"]
    # The network is what the search reads.
    problem = _spec().build()
    problem.set_action_policy(trained.get_network())
    again = bff.train_action_policy(data, [16], 200, 0.02, 21, 0.25)
    assert again.get_network() == trained.get_network()
    with pytest.raises(ValueError):
        bff.train_action_policy(data, [8], 1, 0.02, 21, 1.0)


@pytest.mark.skipif(not bff.PhotonExperiment.get_available(),
                    reason="bff was built without TTTRLib")
def test_counts_are_recorded_by_the_photon_engine():
    expected = np.exp(-np.arange(64) / 10.0) * 500.0
    recorded = np.asarray(bff.PhotonExperiment.record_pattern(list(expected), 3))

    assert recorded.size == 64
    assert np.all(recorded == np.round(recorded)) and np.all(recorded >= 0)
    assert abs(recorded.sum() - expected.sum()) < 5 * np.sqrt(expected.sum())
    # Shape follows the pattern: the early channels hold most photons.
    assert recorded[:10].sum() > recorded[30:40].sum() * 5
    assert list(bff.PhotonExperiment.record_pattern(list(expected), 3)) == list(recorded)
