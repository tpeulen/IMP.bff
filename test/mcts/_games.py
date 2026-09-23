"""Every model family the action policy plays, each with a measurement bound.

A "game" is a family plus the data that fixes its axes, instrument and noise:
self-play replaces the values, never the shape, so what is bound here decides
what the policy's episodes look like. One entry per family bff ships a search
for; a family added to ``data/model_search`` is trained on once it is added
here.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402
import test_equations  # noqa: E402
import test_fret_families  # noqa: E402  (its family list)
import test_tcspc_anisotropy  # noqa: E402


def fcs_analytical():
    axis = np.geomspace(1.0e-3, 20.0, 120)
    curve = 1.0 + 0.5 * (0.4 / (1.0 + axis / 0.08) + 0.6 / (1.0 + axis / 0.8))
    spec = bff.ModelSearchSpec.from_name("fcs_analytical")
    spec.set_dataset("curve", _fixtures.fcs_correlation_dataset(axis, curve, 0.002))
    return spec


def _tcspc_family(family):
    data, irf, dt, period = _fixtures._tcspc_dataset()
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    spec = bff.ModelSearchSpec.from_name(family)
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    return spec


def tcspc_polarized():
    spec = _tcspc_family("tcspc_polarized")
    spec.set_scalar("polarization", 3)
    return spec


def tcspc_anisotropy():
    return test_tcspc_anisotropy._spec(test_tcspc_anisotropy._simulate())


def equations():
    x, y = test_equations._two_decay_data()
    dataset = test_equations._curve([("x", x)], y, np.full(x.size, 0.02))
    return test_equations._spec(test_equations.CATALOGUE, dataset)


GAMES = {
    "fcs_analytical": fcs_analytical,
    "tcspc_lifetime": lambda: _tcspc_family("tcspc_lifetime"),
    "tcspc_polarized": tcspc_polarized,
    "tcspc_anisotropy": tcspc_anisotropy,
    "equations": equations,
    "kinetic_fcs_tcspc": _fixtures.kinetic_fcs_tcspc_spec,
}
for _family in test_fret_families.FAMILIES:
    GAMES[_family] = (lambda family=_family: _tcspc_family(family))


def names():
    """Every game, in a fixed order."""
    return sorted(GAMES)


def spec(name):
    return GAMES[name]()


def describe():
    """The games as JSON: name and structure count, for a training report."""
    return json.dumps({name: len(spec(name).build().get_structure_keys())
                       for name in names()}, indent=1)
