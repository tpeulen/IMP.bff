"""Direction on the factor graph: factors that are mechanisms, do() and twin networks.

A factor that names children is a conditional p(children | rest of its scope).
These tests pin what that adds -- parents, children, descendants, acyclicity,
the mutilated graph of an intervention and the twin network of a counterfactual
-- and what it must not change: the moral graph and everything built on it.
The graph is the mediation model of okf/counterfactuals.md 3.2: an effector L
acts on an active site Y through a hinge M and directly, and a latent W moves
both hinge and active site.
"""
import json

import pytest

from IMP.bff import InferenceFactorGraph, INFERENCE_FACTOR_LIKELIHOOD, INFERENCE_FACTOR_PRIOR


def mediation_graph(directed=True):
    g = InferenceFactorGraph()
    for i, (key, role) in enumerate([("L", "intervention"), ("W", "exogenous"),
                                      ("M", "hinge"), ("Y", "active_site")]):
        g.add_variable(key, key, i, -1, 1, role)
    g.add_factor("f_M", INFERENCE_FACTOR_PRIOR, ["L", "W", "M"])
    g.add_factor("f_Y", INFERENCE_FACTOR_PRIOR, ["L", "W", "M", "Y"])
    g.add_factor("pair1", INFERENCE_FACTOR_LIKELIHOOD, ["M"], 0, 100)
    g.add_factor("pair2", INFERENCE_FACTOR_LIKELIHOOD, ["Y"], 1, 100)
    if directed:
        g.set_factor_children("f_M", ["M"])
        g.set_factor_children("f_Y", ["Y"])
    return g


def test_parents_children_descendants():
    g = mediation_graph()
    assert list(g.get_parents("Y")) == ["L", "W", "M"]
    assert list(g.get_parents("M")) == ["L", "W"]
    assert list(g.get_parents("L")) == []
    assert list(g.get_children("L")) == ["M", "Y"]
    assert list(g.get_children("Y")) == []
    assert list(g.get_descendants(["M"])) == ["M", "Y"]
    assert list(g.get_descendants(["L"])) == ["L", "M", "Y"]
    assert g.get_mechanism_of("Y") == "f_Y"
    assert g.get_mechanism_of("W") == ""
    assert g.get_factor_is_directed("f_M") and not g.get_factor_is_directed("pair1")
    assert list(g.get_exogenous_variables()) == ["W"]
    assert g.get_is_acyclic()


def test_direction_does_not_change_the_moral_graph():
    directed, undirected = mediation_graph(True), mediation_graph(False)
    assert directed.get_cliques() == undirected.get_cliques()
    assert directed.get_treewidth() == undirected.get_treewidth()
    assert directed.get_sampling_blocks() == undirected.get_sampling_blocks()
    assert directed.get_elimination_order() == undirected.get_elimination_order()


def test_reversed_arrows_are_a_different_graph_with_the_same_cliques():
    forward = mediation_graph()
    backward = InferenceFactorGraph()
    for i, key in enumerate(["L", "W", "M", "Y"]):
        backward.add_variable(key, key, i)
    backward.add_factor("f_Y", INFERENCE_FACTOR_PRIOR, ["L", "W", "Y"])
    backward.add_factor("f_M", INFERENCE_FACTOR_PRIOR, ["L", "W", "M", "Y"])
    backward.set_factor_children("f_Y", ["Y"])
    backward.set_factor_children("f_M", ["M"])
    assert list(backward.get_parents("M")) == ["L", "W", "Y"]
    assert list(forward.get_parents("M")) == ["L", "W"]


def test_children_must_be_in_scope_and_have_one_mechanism():
    g = mediation_graph(False)
    with pytest.raises(ValueError):
        g.set_factor_children("f_M", ["Y"])          # Y is not read by f_M
    g.set_factor_children("f_M", ["M"])
    with pytest.raises(ValueError):
        g.set_factor_children("f_Y", ["M"])          # M already has a mechanism
    with pytest.raises(ValueError):
        g.set_factor_children("nope", ["M"])


def test_a_cycle_is_detected():
    g = InferenceFactorGraph()
    for i, key in enumerate("AB"):
        g.add_variable(key, key, i)
    g.add_factor("fA", INFERENCE_FACTOR_PRIOR, ["A", "B"])
    g.add_factor("fB", INFERENCE_FACTOR_PRIOR, ["A", "B"])
    g.set_factor_children("fA", ["A"])
    g.set_factor_children("fB", ["B"])
    assert not g.get_is_acyclic()
    assert "CYCLIC" in g.describe()


def test_intervene_removes_the_mechanism_and_holds_the_variable():
    g = mediation_graph()
    do_m = g.get_intervened(["M"])
    assert "f_M" not in do_m.get_factor_keys()
    assert "f_Y" in do_m.get_factor_keys()
    assert list(do_m.get_intervened_variables()) == ["M"]
    assert list(do_m.get_parents("M")) == []
    assert list(do_m.get_parents("Y")) == ["L", "W", "M"]   # downstream mechanisms stay
    assert "intervened     : M" in do_m.describe()
    # the original is untouched
    assert "f_M" in g.get_factor_keys() and list(g.get_intervened_variables()) == []
    # an exogenous variable has no mechanism: it is only marked as held
    assert list(g.get_intervened(["W"]).get_factor_keys()) == list(g.get_factor_keys())


def test_a_shared_mechanism_cannot_be_cut_for_one_child():
    g = InferenceFactorGraph()
    for i, key in enumerate("XAB"):
        g.add_variable(key, key, i)
    g.add_factor("f", INFERENCE_FACTOR_PRIOR, ["X", "A", "B"])
    g.set_factor_children("f", ["A", "B"])
    with pytest.raises(ValueError):
        g.get_intervened(["A"])
    assert "f" not in g.get_intervened(["A", "B"]).get_factor_keys()


def test_twin_shares_exogenous_inputs_and_copies_only_mechanisms():
    g = mediation_graph()
    twin = g.get_twin(["L"])
    keys = list(twin.get_variable_keys())
    assert keys == ["L", "W", "M", "Y", "L@cf", "M@cf", "Y@cf"]
    # the counterfactual world reads the factual exogenous W and its own treatment
    assert list(twin.get_parents("Y@cf")) == ["W", "L@cf", "M@cf"]
    assert list(twin.get_parents("M@cf")) == ["W", "L@cf"]
    # the counterfactual treatment is intervened on (it had no mechanism to cut)
    assert list(twin.get_intervened_variables()) == ["L@cf"]
    # mechanisms are copied, data likelihoods are not
    factors = list(twin.get_factor_keys())
    assert "f_M@cf" in factors and "f_Y@cf" in factors
    assert "pair1@cf" not in factors and "pair2@cf" not in factors
    assert twin.get_is_acyclic()
    # W couples the worlds: one connected component
    assert len(twin.connected_components()) == 1


def test_json_round_trip_keeps_direction_and_interventions():
    twin = mediation_graph().get_twin(["M"])
    text = twin.to_json()
    doc = json.loads(text)
    assert doc["intervened"] == ["M@cf"]
    assert any(f.get("children") == ["Y@cf"] for f in doc["factors"])
    back = InferenceFactorGraph()
    back.from_json(text)
    assert back.to_json() == text
    assert list(back.get_parents("Y@cf")) == list(twin.get_parents("Y@cf"))


def test_an_undirected_graph_writes_no_direction():
    doc = json.loads(mediation_graph(False).to_json())
    assert "intervened" not in doc
    assert all("children" not in f for f in doc["factors"])
    assert "mechanisms" not in mediation_graph(False).describe()
