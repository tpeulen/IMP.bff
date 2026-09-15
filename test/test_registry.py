"""The registry: what bff can do, by name, as data (PRD-147 amendment A2, written 2026-09-15).

bff's registry is tttrlib's mechanism -- `RegistryCore.h` and `registry_access.py`, vendored and pinned
(test_vendored_headers.py) -- so these tests hold the bff side of that contract: every registered
capability is reachable through `IMP.bff.registry()`, nothing is listed by hand next to it, and the
entry shape is the one a consumer of tttrlib's registry already reads.

Each check can fail: a graph node type added to the factories without an entry (or the reverse), an
entry whose `api` names nothing, a model-search family file missing from the category, a duplicate
key silently overwriting, an entry without the keys a form or a list needs.
"""
import json
import os

import pytest

import IMP.bff as bff

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED = ("name", "label", "summary", "description", "params_schema", "capability", "provider")


@pytest.fixture(scope="module", autouse=True)
def data_path():
    # model-search families are data files; the source tree's data/ is the default in a build tree
    if not os.environ.get("IMP_BFF_DATA") and bff.get_build() == "core":
        os.environ["IMP_BFF_DATA"] = os.path.join(ROOT, "data")
    yield


def test_every_entry_has_the_shape_consumers_read():
    reg = bff.registry()
    assert reg, "empty registry"
    for category, entries in reg.items():
        assert entries, category
        for key, e in entries.items():
            for k in REQUIRED:
                assert k in e, (category, key, k)
            assert e["name"] == key
            assert isinstance(e["params_schema"], dict)
            assert e["provider"] in ("imp.bff", "runtime"), (key, e["provider"])


def test_every_graph_node_type_is_registered_and_nothing_else():
    types = set(bff.GraphNodeRegistry.get_registered_types())
    entries = set(bff.registry("graph_node"))
    assert types == entries, {"factories without entry": sorted(types - entries),
                              "entries without factory": sorted(entries - types)}


def test_a_node_type_registered_at_run_time_appears_in_the_registry():
    name = "RegistryTestRuntimeNode"
    bff.register_node_type(name, lambda n: bff.GraphNode(n))
    e = bff.describe(name)
    assert e["capability"] == "graph_node" and e["provider"] == "runtime"
    test_every_graph_node_type_is_registered_and_nothing_else()


def test_every_api_symbol_resolves():
    for category, entries in bff.registry().items():
        for key, e in entries.items():
            if e.get("api") or e.get("method"):
                target = bff.resolve(key)
                assert target is not None, (category, key)


def test_model_search_category_is_the_shipped_families():
    names = list(bff.ModelSearchSpec.get_available_names())
    assert names, "no shipped families found (IMP_BFF_DATA?)"
    assert set(bff.registry("model_search")) == set(names)
    with open(os.path.join(ROOT, "data", "model_search", "tcspc_fret_gaussian.json")) as fh:
        title = json.load(fh)["title"]
    assert bff.describe("tcspc_fret_gaussian")["label"] == title


def test_duplicates_are_refused_and_provider_defaults_to_bff():
    assert bff.register_algorithm_json("registry_test", "k", json.dumps({"label": "first"}))
    assert not bff.register_algorithm_json("registry_test", "k", json.dumps({"label": "second"}))
    e = bff.describe("k")
    assert e["label"] == "first" and e["provider"] == "imp.bff" and e["capability"] == "registry_test"
    assert not bff.register_algorithm_json("registry_test", "bad", "[1, 2]")


def test_unknown_names_are_refused_with_suggestions():
    with pytest.raises(ValueError, match="did you mean: TCSPCDecay"):
        bff.describe("TCSPCDecai")
    with pytest.raises(ValueError, match="unknown registry category"):
        bff.registry("no_such_category")


def test_categories_json_and_accessors_agree():
    reg = bff.registry()
    assert list(reg) == list(bff.registry_categories())
    assert json.loads(bff.registry_category_json("graph_node")) == reg["graph_node"]
    assert list(bff.registry_keys("graph_node"))[:len(reg["graph_node"])]
