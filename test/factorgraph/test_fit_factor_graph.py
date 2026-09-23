"""A running fit's factor graph, read off its node graph rather than declared.

Two datasets share a lifetime (one port follows the other), each has its own
amplitude, and one baseline is held. The derived graph must say exactly that:
two likelihoods whose only shared variable is the lifetime, a link from the
follower to its master, and the held baseline as evidence rather than as a
variable of the posterior.
"""

import json

import numpy as np

import IMP.bff as bff


def _member(name, y, amplitude, lifetime, baseline, x):
    model = bff.GraphExpression(name + "_model")
    model.set_expression("a*exp(-x/t)+b")
    model.add_input_port("a", amplitude)
    model.add_input_port("t", lifetime)
    model.add_input_port("b", baseline)
    model.add_input_port("x", bff.GraphPort(list(x)))
    curve = bff.GraphPort([0.0], False, True)
    model.add_output_port(name + "_model", curve)
    chi2 = bff.FitChiSquared(name)
    chi2.set_data_arrays(np.ascontiguousarray(y), np.ones(len(y)))
    model_in = bff.GraphPort([0.0])
    model_in.link = curve
    chi2.add_input_port("model", model_in)
    chi2.add_output_port(name, bff.GraphPort(0.0, False, True))
    chi2._graph = (model, curve, model_in)
    return chi2


def _fit():
    x = np.linspace(0.1, 8.0, 40)
    a1, a2 = bff.GraphPort(2.0), bff.GraphPort(5.0)
    t1, t2 = bff.GraphPort(3.0), bff.GraphPort(3.0)
    t2.link = t1
    b = bff.GraphPort(0.1, True)
    members = [
        _member("d1", 2.0 * np.exp(-x / 3.0) + 0.1, a1, t1, b, x),
        _member("d2", 5.0 * np.exp(-x / 3.0) + 0.1, a2, t2, b, x),
    ]
    joint = bff.FitJointChiSquared("joint")
    joint.add_output_port("joint", bff.GraphPort(0.0, False, True))
    for member in members:
        joint.add_member(member)
    joint.update()
    keys = ["a1", "a2", "t1", "t2", "b"]
    ports = [a1, a2, t1, t2, b]
    graph = bff.get_fit_factor_graph(joint, keys, ports)
    graph._keep = (joint, members, ports)
    return graph


def test_the_roles_are_read_off_the_ports():
    graph = _fit()
    assert [graph.get_variable_role(k) for k in ("a1", "a2", "t1", "t2", "b")] == [
        "free", "free", "free", "follower", "fixed"]
    assert graph.get_variable_size("t2") == 0 and graph.get_variable_size("b") == 0


def test_each_dataset_is_a_likelihood_over_what_its_model_reads():
    graph = _fit()
    assert sorted(graph.variables_of("likelihood:d1")) == ["a1", "t1"]
    # d2 reads the follower t2, which is its master t1.
    assert sorted(graph.variables_of("likelihood:d2")) == ["a2", "t1"]
    assert list(graph.get_factor_evidence("likelihood:d1")) == ["b"]
    assert graph.get_factor_kind("link:t2") == bff.INFERENCE_FACTOR_LINK
    assert sorted(graph.variables_of("link:t2")) == ["t1", "t2"]


def test_the_datasets_are_independent_given_the_shared_lifetime():
    graph = _fit()
    assert [list(s) for s in graph.get_separators()][0] == ["t1"]
    assert sorted(graph.affected_fits(["a2"])) == [1]
    assert sorted(graph.affected_fits(["t1"])) == [0, 1]


def test_the_document_round_trips_with_links_and_evidence():
    graph = _fit()
    doc = graph.to_json()
    again = bff.InferenceFactorGraph()
    again.from_json(doc)
    assert json.loads(again.to_json()) == json.loads(doc)
    assert list(again.get_factor_evidence("likelihood:d2")) == ["b"]
