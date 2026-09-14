"""Node types are reachable by name, and say what they will not accept.

The registry is the one part of a model description that cannot itself be
data: something has to map the name ``"TCSPCDecay"`` onto the constructor.
Everything a description says *about* a node goes through
``GraphNode.configure``, which each node implements beside its own setters,
so these tests are mostly about the refusals -- a description that misspells
a setting or borrows one from another node type has to fail where it is read.
"""

import pytest

import IMP.bff as bff


def _types():
    return list(bff.GraphNodeRegistry.get_registered_types())


def test_every_registered_type_reports_the_name_it_is_registered_under():
    """Otherwise a saved graph cannot name the node it has to rebuild."""
    for name in _types():
        assert bff.GraphNodeRegistry.create(name).get_node_type() == name


def test_the_numerical_kernels_are_all_reachable():
    assert set(_types()) >= {
        "FCSMdfCurve",
        "FCSSaturationCurve",
        "FRETSpectrumNode",
        "FitChiSquared",
        "FitJointChiSquared",
        "GaussianDistances",
        "GraphExpression",
        "GraphNode",
        "PhotophysicsAnisotropySpectrumNode",
        "PhotophysicsLifetimeSpectrumNode",
        "PolymerDistances",
        "TCSPCDecay",
    }


def test_an_unknown_type_is_refused_and_says_what_it_knows():
    with pytest.raises((ValueError, RuntimeError)) as caught:
        bff.GraphNodeRegistry.create("TCSPCDcay")
    message = str(caught.value)
    assert "TCSPCDcay" in message and "TCSPCDecay" in message


def test_configuring_reaches_the_node_that_was_created():
    """The registry hands back a base-class handle; behaviour is still the
    node's own.  Three lifetimes means three amplitude and lifetime ports,
    which is the node acting, not the registry."""
    decay = bff.GraphNodeRegistry.create("TCSPCDecay", "decay")
    assert not decay.get_input_ports()
    decay.configure(
        '{"number_of_lifetimes": 3, "timing": [0.05, 12.5],'
        ' "convolution_range": [192, 192], "normalize_amplitudes": true}'
    )
    ports = set(decay.get_input_ports())
    assert {"a0", "a1", "a2", "t0", "t1", "t2"} <= ports


def test_an_empty_configuration_is_valid_and_changes_nothing():
    node = bff.GraphNodeRegistry.create("FitChiSquared")
    node.configure("{}")
    node.configure("")


@pytest.mark.parametrize(
    "settings",
    [
        '{"timing": [0.05]}',              # right key, wrong arity
        '{"nmber_of_lifetimes": 3}',       # misspelled key
        '{"number_of_lifetimes": "3"}',    # right key, wrong type
        '{"expression": "x"}',             # a setting of another node type
        '[1, 2, 3]',                       # not an object
        'not json at all',                 # not JSON
    ],
)
def test_a_setting_a_node_has_no_meaning_for_is_refused(settings):
    """Silently ignoring one produces a graph that evaluates happily and
    fits the wrong thing, which is worse than not building at all."""
    with pytest.raises((ValueError, RuntimeError)):
        bff.GraphNodeRegistry.create("TCSPCDecay").configure(settings)


def test_the_refusal_names_the_node_type_and_the_setting():
    with pytest.raises((ValueError, RuntimeError)) as caught:
        bff.GraphNodeRegistry.create("FitChiSquared").configure('{"wavelength": 1.0}')
    message = str(caught.value)
    assert "FitChiSquared" in message and "wavelength" in message


def test_a_measurement_is_not_a_setting():
    """Data, errors and mask reach a node at runtime, never from a
    description: a model outlives any one experiment."""
    with pytest.raises((ValueError, RuntimeError)):
        bff.GraphNodeRegistry.create("FitChiSquared").configure('{"data": [1.0, 2.0]}')


def test_settings_that_only_make_sense_together_are_required_together():
    with pytest.raises((ValueError, RuntimeError)) as caught:
        bff.GraphNodeRegistry.create("FCSSaturationCurve").configure(
            '{"scheme_states": 2}'
        )
    assert "scheme_brightness" in str(caught.value)


def test_a_node_type_without_settings_still_accepts_an_empty_description():
    for name in ("GaussianDistances", "PolymerDistances", "FRETSpectrumNode"):
        bff.GraphNodeRegistry.create(name).configure("{}")
