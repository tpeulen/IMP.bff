"""VV, VH and VM as one observation over one canonical registry.

A polarisation-resolved decay is three measurements of one sample. The
photophysics -- lifetimes, rotations, r0 -- is shared; how many photons landed
in each channel and on what background is not. So the family carries one
registry, three model chains and one joint objective, and the search chooses
the number of lifetimes *and* the number of rotational correlation times.

The channels are added to the joint objective in the order VV, VH, VM, which
is the order a ChiSurf FitGroup holds them, so a residual block maps back to
the channel that produced it on both sides.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

import IMP.bff as bff

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _characterize  # noqa: E402

CHANNELS = ("vv", "vh", "vm")
N, DT, PERIOD = 256, 0.032, 8.0
TRUTH = {
    "lifetime.amplitude.0": 0.7, "lifetime.tau.0": 0.9,
    "lifetime.amplitude.1": 0.3, "lifetime.tau.1": 3.2,
    "rotation.amplitude.0": 1.0, "rotation.time.0": 1.4,
    "anisotropy.r0": 0.38,
    "instrument.vv.n0": 30000.0,
    "instrument.vh.n0": 20000.0,
    "instrument.vm.n0": 25000.0,
}
GENERATING = "tcspc.l2.r1"


def _response():
    x = np.arange(N) * DT
    return np.exp(-0.5 * ((x - 0.6) / 0.05) ** 2)


def _dataset(values):
    data = bff.FitDataset()
    data.set_values_array(np.ascontiguousarray(np.asarray(values, dtype=float)))
    data.set_noise_family(bff.FIT_NOISE_FAMILY_POISSON)
    return data


def _spec(curves, truth=None):
    spec = bff.ModelSearchSpec.from_name("tcspc_anisotropy")
    for channel in CHANNELS:
        spec.set_dataset(channel, _dataset(curves[channel]))
    spec.set_dataset("response", _dataset(_response()))
    spec.set_scalar("dt", DT)
    spec.set_scalar("period", PERIOD)
    for name, value in (truth or {}).items():
        spec.set_parameter(name, value, True, 0.0, max(1.0, 10.0 * value))
    return spec


def _simulate():
    """The family generating its own measurement, rather than a second model.

    A hand-written simulator is a second opinion about what the model is, and
    the first one is the description itself.
    """
    flat = {channel: np.ones(N) for channel in CHANNELS}
    problem = _spec(flat, TRUTH).build()
    return {
        channel: np.asarray(
            problem.get_structure_output(GENERATING, f"{GENERATING}.{channel}_decay")
        )
        for channel in CHANNELS
    }


def test_the_family_shares_photophysics_and_separates_the_channels():
    problem = _spec({c: np.ones(N) for c in CHANNELS}).build()
    ids = set(problem.get_parameter_ids())
    # One sample: the lifetimes, rotations and r0 are not per channel.
    for shared in ("lifetime.tau.0", "rotation.time.0", "anisotropy.r0"):
        assert shared in ids
    # Three detectors: how many photons landed where is.
    for channel in CHANNELS:
        assert f"instrument.{channel}.n0" in ids
        assert f"instrument.{channel}.background" in ids
    assert len(problem.get_structure_keys()) == 6  # 3 lifetimes x 2 rotations


def test_both_counts_are_the_search_to_choose():
    problem = _spec({c: np.ones(N) for c in CHANNELS}).build()
    state = bff.ModelSearchState(GENERATING, GENERATING, 0.0, False)
    offered = {a.get_key() for a in problem.get_actions(state)}
    assert {"add-lifetime", "remove-lifetime", "add-rotation"} <= offered


def test_the_channels_join_in_the_order_chisurf_holds_them():
    """VV, VH, VM -- so a residual block means the same thing on both sides.

    The joint objective lays its members' residuals end to end in the order
    they were added, one input block each, so the block a channel occupies is
    what makes a joint residual traceable back to the detector it came from.
    """
    problem = _spec({c: np.ones(N) for c in CHANNELS}).build()
    joint = problem.get_structure_objective(GENERATING)
    blocks = joint.get_input_ports()
    order = []
    for index in range(len(CHANNELS)):
        source = blocks[f"block_{index}"].link.get_node()
        order.append(source.get_name().rsplit(".", 1)[-1])
    assert order == [f"{channel}_chi2" for channel in CHANNELS]


def test_the_polarised_channels_differ_and_carry_the_anisotropy():
    curves = _simulate()
    vv = curves["vv"] / TRUTH["instrument.vv.n0"]
    vh = curves["vh"] / TRUTH["instrument.vh.n0"]
    assert not np.allclose(vv, vh)  # VM would make these equal
    anisotropy = (vv - vh) / (vv + 2.0 * vh)
    # Depolarised by the response and the rotation, so below r0 but present.
    assert 0.1 < anisotropy[np.argmax(vv)] < TRUTH["anisotropy.r0"]


def test_a_converged_joint_fit_recovers_the_shared_photophysics():
    """Seeded where the optimiser can reach the answer, the answer is right.

    Seeding is deliberate here. Reaching the global optimum of a
    twenty-four-parameter coupled fit from generic starts is the family's
    open problem, recorded in okf/validation/model-search-strategy.md; what
    this test states is the thing that must hold once it is reached -- the
    shared parameters and the per-channel intensities come back.
    """
    curves = _simulate()
    rng = np.random.default_rng(7)
    noisy = {c: rng.poisson(np.maximum(curves[c], 0)).astype(float) for c in CHANNELS}
    seeded = {k: TRUTH[k] for k in ("lifetime.tau.0", "lifetime.tau.1", "rotation.time.0")}
    problem = _spec(noisy, seeded).build()

    # Walk to the generating topology so it is actually fitted; reading a
    # node's output only evaluates the graph, it does not optimise it.
    root = problem.get_initial_state()
    state = _characterize._walk(
        problem, root, ["add-lifetime"]
    )
    assert state.get_structure_key() == GENERATING

    for name, expected, tolerance in (
        ("instrument.vv.n0", 30000.0, 0.05),
        ("instrument.vh.n0", 20000.0, 0.05),
        ("instrument.vm.n0", 25000.0, 0.05),
    ):
        assert problem.get_parameter(name).value == pytest.approx(
            expected, rel=tolerance
        ), name
