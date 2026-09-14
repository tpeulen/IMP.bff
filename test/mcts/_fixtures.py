"""The model-search problems the golden record pins.

Construction lives here, alone, because it is the only thing the declarative
port is allowed to change.  Each function returns a
``MultiStructureModelSearchProblem`` built the way the shipped factories build
it today; when a family becomes a description, only the body of its function
moves, and ``test_model_search_golden.py`` must keep passing untouched.
"""

from __future__ import annotations

import numpy as np

import IMP.bff as bff


def fcs_curve_2d(axis, n=2.5, baseline=1.0, td=0.7):
    """The closed form the FCS fixtures fit to, 2-D single component."""
    return baseline + (1.0 / n) / (1.0 + axis / td)


def _fcs_axis():
    return np.geomspace(1.0e-3, 20.0, 80)


def fcs_analytical():
    """The whole analytical family: 2-D and 3-D, one or two components."""
    axis = _fcs_axis()
    data = fcs_curve_2d(axis)
    return bff.FCSModelSearchFactory.create_analytical(
        list(axis), list(data), [0.01] * len(axis)
    )


def fcs_two_dimensional_single():
    """One topology only -- the configuration whose fit accuracy is asserted."""
    axis = np.geomspace(1.0e-3, 20.0, 100)
    data = fcs_curve_2d(axis, n=3.2, baseline=0.97, td=0.42)
    config = bff.FCSModelSearchConfig()
    config.set_include_3d(False)
    config.set_max_diffusion_components(1)
    config.set_max_relaxation_terms(0)
    return bff.FCSModelSearchFactory.create_analytical(
        list(axis), list(data), [0.002] * len(axis), config
    )


def _tcspc_dataset():
    n = 192
    dt = 0.05
    period = 12.5
    x = np.arange(n) * dt
    irf = np.exp(-0.5 * ((x - 0.8) / 0.08) ** 2)
    source = bff.TCSPCDecay("source")
    source.set_number_of_lifetimes(2)
    source.add_output_port("source", bff.GraphPort([0.0], False, True))
    source.set_response_array(np.ascontiguousarray(irf))
    source.set_timing(dt, period)
    source.set_convolution_range(n, n)
    source.get_input_port("a0").value = 0.65
    source.get_input_port("t0").value = 0.7
    source.get_input_port("a1").value = 0.35
    source.get_input_port("t1").value = 3.8
    source.get_input_port("n0").value = 20000.0
    source.set_normalize_amplitudes(True)
    source.update()
    clean = np.asarray(source.get_output_port("source").value)
    data = bff.FitDataset()
    data.set_values_array(np.ascontiguousarray(clean))
    data.set_noise_family(bff.FIT_NOISE_FAMILY_POISSON)
    return data, irf, dt, period


def tcspc_lifetime():
    """One to three lifetime components over a two-component decay."""
    data, irf, dt, period = _tcspc_dataset()
    factory = bff.TCSPCLifetimeSearchFactory()
    factory.set_dataset(data)
    factory.set_response(list(irf))
    factory.set_timing(dt, period)
    factory.set_component_range(1, 3)
    factory.set_parameter("instrument.n0", 15000.0, True, 0.0, 1e6)
    factory.set_parameter("instrument.background", 0.0, False, 0.0, 1e5)
    space = factory.build()
    problem = space.get_problem()
    # The space, not the problem, retains this family's decay and objective
    # nodes -- a port holds its node weakly -- so the space has to outlive it.
    problem._keepalive = space
    return problem


#: Golden-record name -> the function that builds its problem.
FIXTURES = {
    "fcs_analytical": fcs_analytical,
    "fcs_two_dimensional_single": fcs_two_dimensional_single,
    "tcspc_lifetime": tcspc_lifetime,
}
