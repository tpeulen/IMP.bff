"""At magic angle a decay carries no anisotropy, so a polarised family fits no rotation.

tcspc_polarized's rotation axis is bounded by the polarization: VM has none
(and r0, freed per rotation, stays put), VV, VH and VV/VH search one or two.
"""

import pathlib
import sys

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _fixtures  # noqa: E402


def _problem(polarization):
    data, irf, dt, period = _fixtures._tcspc_dataset()
    response = bff.FitDataset()
    response.set_values_array(np.ascontiguousarray(irf))
    spec = bff.ModelSearchSpec.from_name("tcspc_polarized")
    spec.set_dataset("decay", data)
    spec.set_dataset("response", response)
    spec.set_scalar("dt", dt)
    spec.set_scalar("period", period)
    spec.set_scalar("polarization", polarization)
    return spec.build()


def test_magic_angle_has_no_rotation_to_fit():
    problem = _problem(0)
    keys = list(problem.get_structure_keys())
    assert keys and all(key.endswith(".rotations.0") for key in keys)
    for key in keys:
        problem.activate_structure(key)
        for canonical in ("anisotropy.r0",):
            assert problem.get_parameter(canonical).fixed


@pytest.mark.parametrize("polarization", [1, 2, 3])
def test_polarised_decays_search_one_or_two_rotations(polarization):
    keys = list(_problem(polarization).get_structure_keys())
    assert {key.rsplit(".", 1)[-1] for key in keys} == {"1", "2"}
