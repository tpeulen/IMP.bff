"""Static orientation factors in FRETSpectrumNode.

For dipoles that do not reorient during the donor's lifetime each molecule has
its own kappa^2, so a distance and an orientation factor make one transfer
rate together. Exact, every pair is a species; binned by apparent distance,
the rates are kept to within a bin at far fewer species.
"""

import numpy as np
import pytest

import IMP.bff as bff


def _node(orientation, bins=0, kappa2=None):
    node = bff.FRETSpectrumNode("fret")
    node.build_ports()
    node.configure('{"orientation": "%s", "kappa2_bins": %d}' % (orientation, bins))
    node.add_output_port("fret", bff.GraphPort([0.0], False, True))
    node.add_output_port("fret_rates", bff.GraphPort([0.0], False, True))
    node.get_input_port("donor_lifetime_spectrum").set_values_array(np.array([1.0, 4.0]))
    distances = np.ravel(np.column_stack([[0.3, 0.7], [40.0, 60.0]]))
    node.get_input_port("distance_distribution").set_values_array(np.ascontiguousarray(distances))
    if kappa2 is not None:
        node.get_input_port("kappa2_distribution").set_values_array(np.ascontiguousarray(kappa2))
    node.get_input_port("forster_radius").value = 52.0
    node.get_input_port("tau0").value = 4.0
    node.update()
    return node


def test_a_static_distribution_makes_one_rate_per_pair():
    kappa2 = np.array([0.25, 0.5, 0.75, 2.0])     # (w, k2) interleaved
    node = _node("static", kappa2=kappa2)
    rates = np.asarray(node.get_output_port("fret_rates").value)
    expected = []
    for p, r in ((0.3, 40.0), (0.7, 60.0)):
        for w, k2 in ((0.25, 0.5), (0.75, 2.0)):
            expected += [p * w, 1.5 * k2 / 4.0 * (52.0 / r) ** 6]
    np.testing.assert_allclose(rates, expected, rtol=1e-12)


def test_a_single_orientation_factor_is_the_dynamic_model():
    static = _node("static", kappa2=np.array([1.0, 0.8]))
    dynamic = _node("dynamic")
    dynamic.get_input_port("kappa2").value = 0.8
    dynamic.update()
    np.testing.assert_allclose(static.get_output_port("fret").value, dynamic.get_output_port("fret").value)


def test_binning_by_apparent_distance_keeps_the_mean_rate():
    exact = np.asarray(_node("static_isotropic").get_output_port("fret_rates").value)
    binned = np.asarray(_node("static_isotropic", bins=512).get_output_port("fret_rates").value)
    assert binned.size == 2 * 512
    assert exact[0::2].sum() == pytest.approx(1.0) and binned[0::2].sum() == pytest.approx(1.0)
    # The efficiency the two predict agrees to within the bin width.
    def mean_efficiency(rates):
        k = rates[1::2]
        return np.sum(rates[0::2] * k / (k + 0.25))
    assert mean_efficiency(binned) == pytest.approx(mean_efficiency(exact), rel=2e-3)


def test_an_unknown_orientation_is_refused():
    node = bff.FRETSpectrumNode("fret")
    node.build_ports()
    with pytest.raises(Exception):
        node.configure('{"orientation": "wobbling"}')
