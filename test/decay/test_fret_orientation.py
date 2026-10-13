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
    # 2 distances x 128 orientation factors = 256 pairs; binning applies below that.
    binned = np.asarray(_node("static_isotropic", bins=128).get_output_port("fret_rates").value)
    assert binned.size == 2 * 128
    assert exact[0::2].sum() == pytest.approx(1.0) and binned[0::2].sum() == pytest.approx(1.0)
    # The efficiency the two predict agrees to within the bin width.
    def mean_efficiency(rates):
        k = rates[1::2]
        return np.sum(rates[0::2] * k / (k + 0.25))
    assert mean_efficiency(binned) == pytest.approx(mean_efficiency(exact), rel=2e-3)


def test_no_more_pairs_than_bins_stays_exact():
    """Binning only cuts species; with 2 x 128 pairs and 512 bins it would only move rates."""
    exact = _rates(_node("static_isotropic"))
    assert np.array_equal(_rates(_node("static_isotropic", bins=512)), exact)
    assert np.array_equal(_rates(_node("static", bins=512, kappa2=np.array([1.0, 0.3]))),
                          _rates(_node("static", bins=0, kappa2=np.array([1.0, 0.3]))))


def test_an_unknown_orientation_is_refused():
    node = bff.FRETSpectrumNode("fret")
    node.build_ports()
    with pytest.raises(Exception):
        node.configure('{"orientation": "wobbling"}')


# --- an arbitrary kappa^2 distribution on the static path -------------------

def _rates(node):
    return np.asarray(node.get_output_port("fret_rates").value)


def test_static_with_no_distribution_is_the_isotropic_average_bit_for_bit():
    """The port's default is empty, and empty is isotropic: the same species, the same bits."""
    for bins in (0, 512):
        given = _node("static", bins=bins)
        isotropic = _node("static_isotropic", bins=bins)
        assert np.array_equal(_rates(given), _rates(isotropic))
        assert np.array_equal(np.asarray(given.get_output_port("fret").value),
                              np.asarray(isotropic.get_output_port("fret").value))


def test_a_delta_at_two_thirds_is_the_dynamic_average():
    static = _node("static", kappa2=np.array([1.0, 2.0 / 3.0]))
    dynamic = _node("dynamic")
    dynamic.get_input_port("kappa2").value = 2.0 / 3.0
    dynamic.update()
    np.testing.assert_allclose(static.get_output_port("fret").value,
                               dynamic.get_output_port("fret").value, rtol=1e-14)


def test_a_uniform_distribution_matches_its_closed_form_decay():
    """kappa^2 uniform on [0, a]: per molecule k = c kappa^2 with c = 3/2 (R0/R)^6 / tau0, so

        f(t) = exp(-t/tau_D) (1 - exp(-c a t)) / (c a t).

    The node's midpoint sum over n equal-weight orientation factors is that integral."""
    a, n, r, r0, tau0 = 1.5, 4000, 50.0, 52.0, 4.0
    k2 = (np.arange(n) + 0.5) * a / n
    node = bff.FRETSpectrumNode("fret")
    node.build_ports()
    node.configure('{"orientation": "static", "kappa2_bins": 0}')
    node.add_output_port("fret", bff.GraphPort([0.0], False, True))
    node.get_input_port("donor_lifetime_spectrum").set_values_array(np.array([1.0, tau0]))
    node.get_input_port("distance_distribution").set_values_array(np.array([1.0, r]))
    node.get_input_port("kappa2_distribution").set_values_array(
        np.ascontiguousarray(np.ravel(np.column_stack([np.ones(n), k2]))))
    node.get_input_port("forster_radius").value = r0
    node.get_input_port("tau0").value = tau0
    node.update()
    spectrum = np.asarray(node.get_output_port("fret").value)
    t = np.linspace(0.05, 20.0, 400)
    decay = np.sum(spectrum[0::2, None] * np.exp(-t[None, :] / spectrum[1::2, None]), axis=0)
    c = 1.5 * (r0 / r) ** 6 / tau0
    exact = np.exp(-t / tau0) * (1.0 - np.exp(-c * a * t)) / (c * a * t)
    np.testing.assert_allclose(decay, exact, rtol=1e-6)


def _family(static, kappa2=None, scalar=None):
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mcts"))
    import _fixtures
    decay, irf, dt, period = _fixtures._tcspc_dataset()
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    spec = bff.ModelSearchSpec.from_name("tcspc_fret_gaussian")
    spec.set_dataset("decay", decay)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    spec.set_scalar("static_orientation", float(static))
    # The family's default kappa2_bins (512): a delta is still exact, see above.
    if kappa2 is not None:
        spec.set_port("kappa2_distribution", bff.GraphPort(list(kappa2)))
    model = spec.get_model()
    if scalar is not None:
        port = model.get_parameter("fret.kappa2")
        held = port.fixed
        port.fixed = False
        port.value = scalar
        port.fixed = held
    return np.asarray(model.get_output("lifetime_spectrum"))


def test_a_family_reads_the_distribution_port_and_is_isotropic_without_one():
    isotropic = _family(True)
    narrow = _family(True, kappa2=[1.0, 0.09, 2.0, 0.1, 1.0, 0.11])
    assert isotropic.size != narrow.size or not np.allclose(isotropic, narrow)
    delta = _family(True, kappa2=[1.0, 2.0 / 3.0])
    dynamic = _family(False, scalar=2.0 / 3.0)
    np.testing.assert_allclose(delta, dynamic, rtol=1e-12)
    # ... and the dynamic one does follow its scalar, so the equality is not vacuous.
    assert not np.allclose(_family(False, scalar=0.2), dynamic)
