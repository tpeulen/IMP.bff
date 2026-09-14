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


def test_the_anisotropy_chain_is_reachable_from_a_description():
    """VV/VH anisotropy is a component count like any other.

    A polarisation-resolved decay is a lifetime spectrum, transformed by an
    anisotropy spectrum, convolved by the instrument. Each of the three is a
    registered kernel that takes its settings as data, so the family is a
    file rather than a factory -- including the rotation count, which is an
    axis exactly as the lifetime count is.
    """
    spectrum = bff.GraphNodeRegistry.create("PhotophysicsLifetimeSpectrumNode")
    spectrum.configure('{"number_of_lifetimes": 2, "normalize_amplitudes": true}')
    assert {"a0", "a1", "t0", "t1"} <= set(spectrum.get_input_ports())

    anisotropy = bff.GraphNodeRegistry.create("PhotophysicsAnisotropySpectrumNode")
    anisotropy.configure('{"number_of_rotations": 2, "polarization": "VH"}')
    ports = set(anisotropy.get_input_ports())
    assert {"b0", "rho0", "b1", "rho1", "r0", "g", "l1", "l2"} <= ports
    assert "lifetime_spectrum" in ports  # what the transform reads

    decay = bff.GraphNodeRegistry.create("TCSPCDecay")
    decay.configure('{"number_of_lifetimes": 1, "spectrum_from_port": true}')
    assert "lifetime_spectrum" in decay.get_input_ports()


def test_a_polarization_is_named_not_numbered():
    """A description says VH; it must not have to know the enum's integer."""
    with pytest.raises((ValueError, RuntimeError)) as caught:
        bff.GraphNodeRegistry.create(
            "PhotophysicsAnisotropySpectrumNode"
        ).configure('{"polarization": "sideways"}')
    assert "sideways" in str(caught.value)


def test_caching_is_a_property_of_every_node():
    """memoize belongs to GraphNode, so no kernel has to re-declare it."""
    for name in ("TCSPCDecay", "GraphExpression", "FCSMdfCurve"):
        bff.GraphNodeRegistry.create(name).configure('{"memoize": true}')
