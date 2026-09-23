"""Focused contract tests for the model-independent native MCTS core."""

import json

import numpy as np
import pytest

import IMP.bff as bff


def _problem():
    problem = bff.TabularModelSearchProblem()
    for key, reward, acceptable in (
        ("root", 0.0, False),
        ("small", 2.0, False),
        ("large", 1.0, False),
        ("winner", 5.0, True),
    ):
        problem.add_state(key, reward, acceptable)
    problem.set_initial_state("root")
    problem.add_action("root", "grow-small", "small", 0.5)
    problem.add_action("root", "grow-large", "large", 0.5)
    problem.add_action("small", "refine", "winner", 0.9)
    problem.add_action("small", "undo", "root", 0.1)
    problem.add_action("large", "stop", "large", 1.0, True)
    problem.add_action("winner", "undo", "root", 0.5)
    problem.add_action("winner", "stop", "winner", 0.5, True)
    return problem


def _search(problem, simulations=80):
    config = bff.ModelSearchConfig()
    config.set_number_of_simulations(simulations)
    config.set_reward_scale(1.0)
    config.set_dirichlet_fraction(0.0)
    config.set_seed(7)
    search = bff.ModelSearch(problem)
    search.set_config(config)
    return search


def test_native_search_is_lazy_reproducible_and_selects_best_reward():
    first_problem = _problem()
    first = _search(first_problem).run()
    second = _search(_problem()).run()

    assert first.get_best_state().get_key() == "winner"
    assert first.get_improvement() == 5.0
    assert first.get_acceptable()
    assert first.get_number_of_simulations() == 80
    assert list(first.get_best_path()) == list(second.get_best_path())
    # Only three non-root states exist. Traversals revisit cached nodes rather
    # than re-running the evaluator on every simulation.
    assert first_problem.get_number_of_evaluations() <= 3
    assert first.get_number_of_states_evaluated() == (
        1 + first_problem.get_number_of_evaluations()
    )


def test_cancel_before_run_returns_the_accepted_root_without_evaluation():
    problem = _problem()
    search = _search(problem)
    search.request_cancel()
    result = search.run()

    assert result.get_cancelled()
    assert result.get_number_of_simulations() == 0
    assert result.get_best_state().get_key() == "root"
    assert problem.get_number_of_evaluations() == 0


def _linear_fit_problem():
    x = np.linspace(-2.0, 2.0, 21)
    y = 2.0 * x + 3.0

    model = bff.GraphExpression("linear_model")
    model.set_expression("a*x+b")
    a = bff.GraphPort(2.0)
    b = bff.GraphPort(0.0, True)
    axis = bff.GraphPort(list(x))
    model.add_input_port("a", a)
    model.add_input_port("b", b)
    model.add_input_port("x", axis)
    curve = bff.GraphPort([0.0], False, True)
    model.add_output_port("linear_model", curve)

    objective = bff.FitChiSquared("linear_fit")
    objective.set_data_arrays(np.ascontiguousarray(y), np.ones(y.size))
    model_input = bff.GraphPort([0.0])
    model_input.link = curve
    objective.add_input_port("model", model_input)
    objective.add_output_port("linear_fit", bff.GraphPort(0.0, False, True))
    objective.add_output_port("residuals", bff.GraphPort([0.0], False, True))
    objective._graph = (model, curve, model_input, axis)

    problem = bff.FittingModelSearchProblem()
    problem.add_parameter("slope", a)
    problem.add_parameter("intercept", b)
    ids, ports = ["slope", "intercept"], [a, b]
    problem.add_structure("root", objective, ids, ports, [2.0, 0.0], [0, 1])
    problem.add_structure("with-intercept", objective, ids, ports, [2.0, 1.0], [0, 0])
    # AIC charges one reward unit per free parameter.
    problem.set_structure_selection("root", bff.MODEL_SELECTION_AIC, x.size, 1.0)
    problem.set_structure_selection("with-intercept", bff.MODEL_SELECTION_AIC, x.size, 2.0)
    problem.set_initial_structure("root")
    problem.add_action("root", "enable-intercept", "with-intercept", 0.9)
    problem.add_action("root", "stop", "root", 0.1, True)
    problem.add_action("with-intercept", "disable-intercept", "root", 0.1)
    problem.add_action("with-intercept", "stop", "with-intercept", 0.9, True)
    problem._keepalive = (objective, model, a, b)
    return problem, a, b


def _row_policy(weights=None, bias=0.0):
    """A one-layer move scorer: one weight per feature, one output."""
    width = bff.get_policy_state_width() + bff.get_policy_action_width()
    weights = [0.0] * width if weights is None else list(weights)
    assert len(weights) == width
    return json.dumps({
        "format": "bff.neural_net",
        "layers": [{
            "n_in": width, "n_out": 1, "weight": weights, "bias": [bias],
            "activation": "identity",
        }],
    })


def _favouring(action_feature, strength):
    """A scorer that only reads one action feature."""
    weights = [0.0] * (bff.get_policy_state_width() + bff.get_policy_action_width())
    weights[bff.get_policy_state_width() + action_feature] = strength
    return _row_policy(weights)


TERMINAL, SELF_LOOP, DELTA, ADDS, REMOVES, TARGET_FREE, PRIOR_SHARE = range(7)


def _priors(problem, state):
    return {action.get_key(): action.get_prior() for action in problem.get_actions(state)}


def test_a_policy_multiplies_the_declared_priors_by_its_scores():
    problem, _a, _b = _linear_fit_problem()
    problem.set_action_policy(_favouring(TERMINAL, 8.0))
    root = problem.get_initial_state()
    priors = _priors(problem, root)

    assert problem.get_has_action_policy()
    # 0.1 * e^8 against 0.9: the terminal move now dominates.
    assert priors["stop"] == pytest.approx(0.1 * np.exp(8) / (0.1 * np.exp(8) + 0.9))
    assert sum(priors.values()) == pytest.approx(1.0)


def test_a_score_every_move_shares_leaves_the_declared_priors():
    problem, _a, _b = _linear_fit_problem()
    problem.set_action_policy(_row_policy(bias=3.0))
    root = problem.get_initial_state()

    assert _priors(problem, root) == pytest.approx({"enable-intercept": 0.9, "stop": 0.1})


def test_the_rows_are_the_state_and_move_features_of_the_fitted_residual():
    problem, _a, _b = _linear_fit_problem()
    root = problem.get_initial_state()
    residual = list(problem.get_active_objective().get_residuals())
    actions = problem.get_actions(root)
    width = bff.get_policy_state_width() + bff.get_policy_action_width()

    rows = np.asarray(problem.get_policy_rows("root", residual, actions)).reshape(-1, width)

    state = np.asarray(bff.get_policy_state_features(residual, 1))
    for row, action in zip(rows, actions):
        np.testing.assert_allclose(row[: state.size], state)
        target = action.get_predicted_state_key()
        expected = bff.get_policy_action_features(
            1, problem.get_number_of_free_parameters(target), action.get_terminal(),
            target == "root", action.get_prior())
        np.testing.assert_allclose(row[state.size:], expected)


def test_a_policy_of_the_wrong_shape_is_refused_and_one_can_be_removed():
    problem, _a, _b = _linear_fit_problem()
    wrong = json.dumps({"format": "bff.neural_net", "layers": [{
        "n_in": 3, "n_out": 1, "weight": [0.0] * 3, "bias": [0.0],
        "activation": "identity"}]})
    with pytest.raises(ValueError, match="features"):
        problem.set_action_policy(wrong)

    problem.set_action_policy(_row_policy())
    problem.clear_action_policy()
    assert not problem.get_has_action_policy()
    root = problem.get_initial_state()
    assert _priors(problem, root) == {"enable-intercept": 0.9, "stop": 0.1}


def test_expansion_under_a_policy_leaves_the_graph_alone():
    """Expanding a state must not restore or re-evaluate the fit graph."""
    problem, a, b = _linear_fit_problem()
    problem.set_action_policy(_favouring(ADDS, 1.0))
    root = problem.get_initial_state()
    a.value = 11.0  # a user edit after scoring; b is fixed and stays put
    before = (a.value, b.value, b.fixed)

    problem.get_actions(root)

    assert (a.value, b.value, b.fixed) == before


def test_an_unscored_state_keeps_its_declared_priors():
    problem, _a, _b = _linear_fit_problem()
    problem.set_action_policy(_favouring(TERMINAL, 9.0))
    stranger = bff.ModelSearchState("root@unscored", "root", 0.0)

    assert _priors(problem, stranger) == {"enable-intercept": 0.9, "stop": 0.1}


def test_a_joint_residual_also_shows_its_worst_block_on_its_own():
    rng = np.random.default_rng(2)
    good = rng.normal(size=300)
    bad = 3.0 * np.sin(np.linspace(0, 2 * np.pi, 80)) + rng.normal(size=80)
    joint = list(np.concatenate([good, bad]))
    features = np.asarray(bff.get_policy_state_features(joint, 3, [300, 80]))
    np.testing.assert_allclose(features[32:64], bff.get_residual_profile(list(bad), 32))
    assert features[-1] == pytest.approx(np.log(2))
    single = np.asarray(bff.get_policy_state_features(list(good), 3))
    np.testing.assert_allclose(single[32:64], single[:32])


def test_the_residual_profile_is_compressed_bucket_z_scores():
    profile = bff.get_residual_profile([1.0, 3.0, float("nan"), 5.0, -2.0, -4.0], 3)
    expected = np.arcsinh([4.0 / np.sqrt(2), 5.0, -6.0 / np.sqrt(2)])
    assert list(profile) == pytest.approx(list(expected))
    assert list(bff.get_residual_profile([], 4)) == [0.0] * 4
    with pytest.raises(ValueError):
        bff.get_residual_profile([1.0], 0)


def test_state_features_see_structure_in_a_residual():
    rng = np.random.default_rng(1)
    white = rng.normal(size=400)
    trend = np.sin(np.linspace(0, 3 * np.pi, 400)) * 2 + 0.3 * white
    s = bff.get_policy_state_width()
    w, t = (np.asarray(bff.get_policy_state_features(list(r), 2)) for r in (white, trend))
    assert w.size == t.size == s
    lag, runs = 65, 66  # after two 32-bin profiles and log10 chi2r
    assert t[lag] > 0.8 > abs(w[lag])
    assert t[runs] < -2.0 < w[runs]


def test_live_fit_adapter_caches_masks_and_activates_the_winner():
    problem, a, b = _linear_fit_problem()
    search = _search(problem, simulations=40)
    result = search.run()

    assert result.get_best_state().get_structure_key() == "with-intercept"
    assert abs(a.value - 2.0) < 1e-8
    assert abs(b.value - 3.0) < 1e-8
    assert not a.fixed and not b.fixed
    assert list(problem.get_cached_fixed("root")) == [0, 1]
    assert list(problem.get_cached_fixed(result.get_best_state().get_key())) == [0, 0]
    problem.restore_state("root")
    assert a.value == 2.0 and b.value == 0.0
    assert not a.fixed and b.fixed


def test_live_fit_adapter_cancellation_restores_the_root_snapshot():
    problem, a, b = _linear_fit_problem()
    search = _search(problem, simulations=40)
    search.request_cancel()
    result = search.run()

    assert result.get_cancelled()
    assert result.get_best_state().get_key() == "root"
    assert a.value == 2.0 and b.value == 0.0
    assert not a.fixed and b.fixed


def _expression_objective(name, expression, x, y, owners):
    model = bff.GraphExpression(name + "_model")
    model.set_expression(expression)
    followers = []
    for key, owner in owners.items():
        follower = bff.GraphPort(owner.value)
        follower.link = owner
        model.add_input_port(key, follower)
        followers.append(follower)
    axis = bff.GraphPort(list(x))
    model.add_input_port("x", axis)
    curve = bff.GraphPort([0.0], False, True)
    model.add_output_port(name + "_model", curve)

    objective = bff.FitChiSquared(name)
    objective.set_data_arrays(np.ascontiguousarray(y), np.ones(len(y)))
    model_input = bff.GraphPort([0.0])
    model_input.link = curve
    objective.add_input_port("model", model_input)
    objective.add_output_port(name, bff.GraphPort(0.0, False, True))
    objective.add_output_port("residuals", bff.GraphPort([0.0], False, True))
    objective._graph = (model, followers, axis, curve, model_input)
    return objective


def _multi_structure_linear_problem():
    x = np.linspace(-2.0, 2.0, 25)
    y = 2.0 * x + 3.0
    slope = bff.GraphPort(2.0)
    intercept = bff.GraphPort(0.0, True)
    root_objective = _expression_objective(
        "through_origin", "a*x", x, y, {"a": slope}
    )
    expanded_objective = _expression_objective(
        "with_intercept", "a*x+b", x, y, {"a": slope, "b": intercept}
    )

    problem = bff.FittingModelSearchProblem()
    problem.add_parameter("slope", slope)
    problem.add_parameter("intercept", intercept)
    ids = ["slope", "intercept"]
    ports = [slope, intercept]
    problem.add_structure(
        "root", root_objective, ids, ports, [2.0, 0.0], [0, 1]
    )
    # The deliberately bad slope seed proves that a parameter which remains
    # free is warm-started in the shared registry rather than reset per graph.
    problem.add_structure(
        "expanded",
        expanded_objective,
        ids,
        ports,
        [-100.0, 1.0],
        [0, 0],
    )
    problem.add_structure_node("root", root_objective._graph[0])
    problem.add_structure_node("expanded", expanded_objective._graph[0])
    del root_objective._graph
    del expanded_objective._graph
    problem.set_structure_selection("root", bff.MODEL_SELECTION_BIC, len(x), 1.0)
    problem.set_structure_selection("expanded", bff.MODEL_SELECTION_BIC, len(x), 2.0)
    problem.set_initial_structure("root")
    problem.add_action("root", "add-intercept", "expanded", 1.0)
    problem.add_action("expanded", "stop", "expanded", 1.0, True)
    problem._keepalive = (root_objective, expanded_objective, slope, intercept)
    return problem, slope, intercept, root_objective, expanded_objective


def test_multi_structure_problem_rejects_noncanonical_or_incomplete_state():
    owner = bff.GraphPort(1.0)
    other = bff.GraphPort(1.0)
    objective = _expression_objective(
        "identity_contract", "a*x", [1.0, 2.0], [1.0, 2.0], {"a": owner}
    )
    problem = bff.FittingModelSearchProblem()
    problem.add_parameter("a", owner)

    with np.testing.assert_raises(ValueError):
        problem.add_structure(
            "missing", objective, [], [], [], []
        )
    with np.testing.assert_raises(ValueError):
        problem.add_structure(
            "copied-owner", objective, ["a"], [other], [1.0], [0]
        )
    with np.testing.assert_raises(ValueError):
        problem.add_parameter("second-name", owner)


def test_multi_structure_search_switches_graphs_and_activates_canonical_winner():
    problem, slope, intercept, root_objective, expanded_objective = (
        _multi_structure_linear_problem()
    )
    root = problem.get_initial_state()
    assert list(problem.get_cached_values(root.get_key())) == [2.0, 0.0]
    assert problem.get_active_objective().get_uid() == root_objective.get_uid()

    result = _search(problem, simulations=20).run()

    assert result.get_best_state().get_structure_key() == "expanded"
    assert problem.get_active_structure() == "expanded"
    assert problem.get_active_objective().get_uid() == expanded_objective.get_uid()
    assert abs(slope.value - 2.0) < 1e-8
    assert abs(intercept.value - 3.0) < 1e-8
    assert not slope.fixed and not intercept.fixed

    problem.restore_state("root")
    assert problem.get_active_structure() == "root"
    assert slope.value == 2.0 and intercept.value == 0.0
    assert not slope.fixed and intercept.fixed


def test_multi_structure_cancel_rolls_back_registry_and_active_graph():
    problem, slope, intercept, root_objective, _ = (
        _multi_structure_linear_problem()
    )
    root = problem.get_initial_state()
    action = problem.get_actions(root)[0]
    problem.request_cancel()

    collapsed = problem.evaluate(root, action)

    assert collapsed.get_key() == root.get_key()
    assert problem.get_last_fit_status() == -1
    assert problem.get_last_failure()
    assert problem.get_active_structure() == "root"
    assert problem.get_active_objective().get_uid() == root_objective.get_uid()
    assert slope.value == 2.0 and intercept.value == 0.0
    assert not slope.fixed and intercept.fixed


def test_multi_structure_failed_fit_rolls_back_registry_and_active_graph():
    x = np.array([1.0])
    slope = bff.GraphPort(1.0)
    intercept = bff.GraphPort(0.0, True)
    root_objective = _expression_objective(
        "failure_root", "a*x", x, x, {"a": slope}
    )
    invalid_objective = _expression_objective(
        "failure_target", "a*x+b", x, x, {"a": slope, "b": intercept}
    )
    problem = bff.FittingModelSearchProblem()
    problem.add_parameter("slope", slope)
    problem.add_parameter("intercept", intercept)
    ids = ["slope", "intercept"]
    ports = [slope, intercept]
    problem.add_structure(
        "root", root_objective, ids, ports, [1.0, 0.0], [0, 1]
    )
    # Two free parameters cannot be fitted to one residual.
    problem.add_structure(
        "underdetermined", invalid_objective, ids, ports, [50.0, 20.0], [0, 0]
    )
    # Both are compared, so both have to say on what.
    problem.set_structure_selection("root", bff.MODEL_SELECTION_BIC, len(x), 1.0)
    problem.set_structure_selection(
        "underdetermined", bff.MODEL_SELECTION_BIC, len(x), 2.0
    )
    problem.set_initial_structure("root")
    problem.add_action("root", "fail", "underdetermined", 1.0)
    problem._keepalive = (root_objective, invalid_objective, slope, intercept)
    root = problem.get_initial_state()

    collapsed = problem.evaluate(root, problem.get_actions(root)[0])

    assert collapsed.get_key() == root.get_key()
    assert problem.get_last_fit_status() == 0
    assert problem.get_last_failure()
    assert problem.get_active_structure() == "root"
    assert slope.value == 1.0 and intercept.value == 0.0
    assert not slope.fixed and intercept.fixed


def test_multi_structure_problem_accepts_a_joint_target_objective():
    x = np.linspace(0.1, 8.0, 48)
    lifetime = bff.GraphPort(1.0, True)
    amplitudes = [bff.GraphPort(2.0), bff.GraphPort(5.0)]
    members = []
    for index, amplitude in enumerate(amplitudes):
        y = amplitude.value * np.exp(-x / 3.0)
        members.append(
            _expression_objective(
                "multi_joint_%d" % index,
                "a*exp(-x/t)",
                x,
                y,
                {"a": amplitude, "t": lifetime},
            )
        )
    joint = bff.FitJointChiSquared("multi_joint")
    joint.add_output_port("multi_joint", bff.GraphPort(0.0, False, True))
    joint.add_output_port("residuals", bff.GraphPort([0.0], False, True))
    for member in members:
        joint.add_member(member)

    problem = bff.FittingModelSearchProblem()
    problem.add_parameter("amplitude-0", amplitudes[0])
    problem.add_parameter("amplitude-1", amplitudes[1])
    problem.add_parameter("lifetime", lifetime)
    ids = ["amplitude-0", "amplitude-1", "lifetime"]
    ports = [amplitudes[0], amplitudes[1], lifetime]
    # Both topologies score the *same* two curves and differ only in whether
    # the shared lifetime is free. That is what makes them comparable: no
    # criterion can choose between models fitted to different data, and this
    # fixture used to ask it to -- one structure saw one curve and the other
    # saw two, which only looked like a contest because nothing was charging
    # for parameters.
    problem.add_structure(
        "fixed-lifetime", joint, ids, ports, [2.0, 5.0, 1.0], [0, 0, 1]
    )
    problem.add_structure(
        "joint-free-lifetime", joint, ids, ports, [2.0, 5.0, 2.0], [0, 0, 0]
    )
    for key in ("fixed-lifetime", "joint-free-lifetime"):
        for member in members:
            problem.add_structure_node(key, member._graph[0])
    for member in members:
        del member._graph
    # Same observations for both, so the comparison is between models rather
    # than between datasets; they differ by the one parameter BIC charges for.
    problem.set_structure_selection(
        "fixed-lifetime", bff.MODEL_SELECTION_BIC, 2 * len(x), 2.0
    )
    problem.set_structure_selection(
        "joint-free-lifetime", bff.MODEL_SELECTION_BIC, 2 * len(x), 3.0
    )
    problem.set_initial_structure("fixed-lifetime")
    problem.add_action(
        "fixed-lifetime", "fit-joint", "joint-free-lifetime", 1.0
    )
    problem.add_action(
        "joint-free-lifetime", "stop", "joint-free-lifetime", 1.0, True
    )
    problem._keepalive = (joint, members, amplitudes, lifetime)

    result = _search(problem, simulations=20).run()

    assert result.get_best_state().get_structure_key() == "joint-free-lifetime"
    assert abs(lifetime.value - 3.0) < 1e-5
    assert problem.get_active_objective().get_uid() == joint.get_uid()
