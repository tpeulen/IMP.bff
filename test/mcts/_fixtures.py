"""The model-search problems the golden record pins.

Construction lives here, alone, because it is the only thing the declarative
port was allowed to change.  It has now changed: every family below is read
from a JSON description rather than assembled by a C++ factory, and
``test_model_search_golden.py`` still passes untouched.  That is the whole
argument for the port, so this file is the place to read it.

What a caller still does is bind the measurement and the few numbers that
belong to the instrument rather than the model.  A description holds no data.
"""

from __future__ import annotations

import pathlib

import numpy as np

import IMP.bff as bff

SPECS = pathlib.Path(__file__).resolve().parent / "specs"


def fcs_curve_2d(axis, n=2.5, baseline=1.0, td=0.7):
    """The closed form the FCS fixtures fit to, 2-D single component."""
    return baseline + (1.0 / n) / (1.0 + axis / td)


def fcs_correlation_dataset(axis, data, error):
    """One correlation curve as a measurement: lags, values and their errors.

    The errors travel as a stored variance rather than as a loose array,
    which is what lets one objective read the whole measurement -- mask and
    noise family included -- instead of three vectors that can disagree.
    """
    dataset = bff.FitDataset()
    dataset.set_values_array(np.ascontiguousarray(np.asarray(data, dtype=float)))
    dataset.set_coordinate_array(
        0, "lag", np.ascontiguousarray(np.asarray(axis, dtype=float))
    )
    dataset.set_noise_family(bff.FIT_NOISE_FAMILY_STORED)
    errors = np.full(len(data), float(error)) if np.isscalar(error) else np.asarray(error)
    dataset.set_stored_variance_array(np.ascontiguousarray(errors ** 2))
    return dataset


def fcs_analytical():
    """The whole analytical family: 2-D and 3-D, one or two components."""
    axis = np.geomspace(1.0e-3, 20.0, 80)
    spec = bff.ModelSearchSpec.from_name("fcs_analytical")
    spec.set_dataset("curve", fcs_correlation_dataset(axis, fcs_curve_2d(axis), 0.01))
    return spec.build()


def fcs_two_dimensional_single():
    """One topology only -- the configuration whose fit accuracy is asserted."""
    axis = np.geomspace(1.0e-3, 20.0, 100)
    data = fcs_curve_2d(axis, n=3.2, baseline=0.97, td=0.42)
    spec = bff.ModelSearchSpec.from_file(str(SPECS / "fcs_2d_single.json"))
    spec.set_dataset("curve", fcs_correlation_dataset(axis, data, 0.002))
    return spec.build()


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
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    spec = bff.ModelSearchSpec.from_name("tcspc_lifetime")
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    # The channel width and the excitation period are the instrument, not the
    # model, so the description names them and the caller supplies them.
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    spec.set_parameter("instrument.n0", 15000.0, True, 0.0, 1e6)
    spec.set_parameter("instrument.background", 0.0, False, 0.0, 1e5)
    return spec.build()


def kinetic_fcs_tcspc_spec():
    """One chain of states seen by a decay and a correlation curve at once.

    The data bound here only fixes the axes, the instrument and the noise;
    self-play and the tests replace the values with what a topology predicts.
    """
    decay, irf, dt, period = _tcspc_dataset()
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    axis = np.geomspace(1.0e-4, 10.0, 96)
    curve = fcs_correlation_dataset(axis, fcs_curve_2d(axis, n=2.0, td=0.5), 0.002)
    spec = bff.ModelSearchSpec.from_name("kinetic_fcs_tcspc")
    spec.set_dataset("decay", decay)
    spec.set_dataset("response", response)
    spec.set_dataset("curve", curve)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    return spec


def _decay(irf, dt, period, lifetimes, n0, background=0.0):
    """A noiseless decay of discrete lifetimes, as the bound measurement shape."""
    source = bff.TCSPCDecay("source")
    source.set_number_of_lifetimes(len(lifetimes))
    source.add_output_port("source", bff.GraphPort([0.0], False, True))
    source.set_response_array(np.ascontiguousarray(irf))
    source.set_timing(dt, period)
    source.set_convolution_range(len(irf), len(irf))
    for i, (amplitude, tau) in enumerate(lifetimes):
        source.get_input_port(f"a{i}").value = amplitude
        source.get_input_port(f"t{i}").value = tau
    source.get_input_port("n0").value = n0
    source.get_input_port("background").value = background
    source.set_normalize_amplitudes(True)
    source.update()
    data = bff.FitDataset()
    data.set_values_array(np.ascontiguousarray(np.asarray(source.get_output_port("source").value)))
    data.set_noise_family(bff.FIT_NOISE_FAMILY_POISSON)
    return data


def _fret_channels(dd, da, aa, *, acceptor=True):
    """Donor, acceptor-under-donor and acceptor-under-acceptor decays.

    Two FRET species (donor lifetimes 0.9 and 3.2 ns) seen by the donor and,
    as crosstalk, by the acceptor channel, which also carries the acceptor's
    own 2.6 ns decay. Self-play replaces the values; these fix shape, IRF and
    noise family.
    """
    _, irf, dt, period = _tcspc_dataset()
    species = [(0.6, 0.9), (0.4, 3.2)]
    channels = {
        dd: _decay(irf, dt, period, species, 30000.0, 2.0),
        da: _decay(irf, dt, period, species + [(0.5, 2.6)], 12000.0, 2.0),
    }
    if acceptor:
        channels[aa] = _decay(irf, dt, period, [(1.0, 2.6)], 15000.0, 1.0)
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    return channels, response, dt, period


def smfret_mfd_spec(acceptor=True):
    """smFRET/MFD bursts: DD and DA decays, and AA when the acceptor was excited.

    Without ``acceptor`` the AA decay is not bound -- an MFD measurement
    without acceptor excitation -- and the family withholds the move that
    needs it.
    """
    channels, response, dt, period = _fret_channels(
        "decay_dd", "decay_da", "decay_aa", acceptor=acceptor)
    spec = bff.ModelSearchSpec.from_name("smfret_mfd")
    for name, data in channels.items():
        spec.set_dataset(name, data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    return spec


def pie_alex_spec():
    """PIE/ALEX: prompt DD and DA windows and the delayed AA window."""
    channels, response, dt, period = _fret_channels("prompt_dd", "prompt_da", "delayed_aa")
    spec = bff.ModelSearchSpec.from_name("pie_alex")
    for name, data in channels.items():
        spec.set_dataset(name, data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    return spec


#: An evaluation budget the one-lifetime fit converges within and the
#: two-lifetime fit does not (measured: root needs > 20, two lifetimes > 45).
SHORT_BUDGET = 36


def tcspc_lifetime_fit_settings():
    """The lifetime family searched over structures and fit settings.

    Under a budget too short for two lifetimes to converge, a structure-only
    search cannot leave one lifetime; a second way of fitting -- a patient
    evaluation budget -- lets it reach the two the decay was built from.
    """
    problem = tcspc_lifetime()
    problem.set_minimizer_maxfev(SHORT_BUDGET)
    problem.add_fit_settings_candidate(bff.ModelSearchFitSettings(
        "declared", "leastsq", 0.0, 0.0, 0.0, 0, 0.0, "declared", True))
    problem.add_fit_settings_candidate(bff.ModelSearchFitSettings(
        "patient", "leastsq", 0.0, 0.0, 0.0, 2000, 0.0, "declared", True))
    problem.set_action_space(bff.ACTION_SPACE_FIT_SETTINGS)
    return problem


#: Golden-record name -> the function that builds its problem.
FIXTURES = {
    "fcs_analytical": fcs_analytical,
    "fcs_two_dimensional_single": fcs_two_dimensional_single,
    "tcspc_lifetime": tcspc_lifetime,
    "tcspc_lifetime_fit_settings": tcspc_lifetime_fit_settings,
}
