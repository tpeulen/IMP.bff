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


# ------------------------------------------------------------ samplers (PRD-147 step 4, R2)

SAMPLER_KEYS = ("kind", "requires_gradient", "supports_bounds", "population", "default_warmup", "acceptance_rate",
                "statistics", "required_checks", "references", "aliases")


def _default_warmup(entry, steps):
    rule = entry["default_warmup"]
    if rule["rule"] == "clip":
        return min(rule["max"], max(rule["min"], steps // rule["divisor"]))
    return rule["value"]


def test_every_sampler_entry_declares_what_a_caller_needs():
    samplers = bff.registry("sampler")
    assert {"nuts", "stretch", "slice", "de", "metropolis"} <= set(samplers)
    for key, e in samplers.items():
        for k in SAMPLER_KEYS:
            assert k in e, (key, k)
        assert e["kind"] in ("chain", "ensemble")
        assert (e["kind"] == "chain") == (e["population"] == "single"), key
        for name, spec in e["params_schema"].get("properties", {}).items():
            assert spec.get("description") and spec.get("type"), (key, name)


def test_aliases_are_unique_and_do_not_shadow_keys():
    samplers = bff.registry("sampler")
    seen = set(samplers)
    for key, e in samplers.items():
        for a in e["aliases"]:
            assert a not in seen, (key, a)
            seen.add(a)


def test_default_warmup_reproduces_chisurfs_hard_coded_values():
    """chisurf/core/fitting/sampler_bff.py `_default_n_adapt` mirrors the C++ defaults by hand; the
    entries now carry them as data and must give the same numbers (so chisurf's copy can go)."""
    samplers = bff.registry("sampler")
    for steps in (10, 200, 2000, 20000):
        assert _default_warmup(samplers["de"], steps) == min(500, max(50, steps // 4))
        assert _default_warmup(samplers["metropolis"], steps) == min(500, max(100, steps // 20))
        assert _default_warmup(samplers["stretch"], steps) == 0


def test_mcmcsampler_dispatches_by_registry_not_by_name():
    with open(os.path.join(ROOT, "src", "MCMCSampler.cpp")) as fh:
        text = fh.read()
    for name in ("stretch", "slice", "de", "metropolis", "nuts", "ensemble", "blocked"):
        assert f'algorithm_ == "{name}"' not in text and f'algorithm_ != "{name}"' not in text, name
        assert f'algorithm == "{name}"' not in text, name


def test_every_registered_sampler_runs_or_refuses_through_mcmcsampler():
    import sys
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "test", "sampler"))
    from test_sampler import gaussian_graph, MU
    for key, e in bff.registry("sampler").items():
        for name in [key] + list(e["aliases"]):
            s = bff.MCMCSampler(name, 3)
            assert s.get_algorithm() == key
        s = bff.MCMCSampler(key, 3)
        params, objective = gaussian_graph()
        params[0].set_value(MU[0])
        params[1].set_value(MU[1])
        s.set_parameter_ports(params)
        s.set_objective(objective, "chi2")
        if e["requires_gradient"]:
            with pytest.raises(ValueError, match="gradient"):
                s.run(20)
            continue
        s.run(60)
        chain = np.asarray(s.get_chain())
        assert chain.shape[1] == 2 and np.all(np.isfinite(chain)), key
        assert 0.0 <= s.get_acceptance_rate() <= 1.0, key
