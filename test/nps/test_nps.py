"""Public Fast-NPS compatibility pins for the native IMP.bff API."""

import math
import sys

import IMP
import IMP.bff as bff
import pytest


def _direct_dye(x=0.0, y=0.0, z=0.0, m=0.0, phi=0.0, ravg=0.0):
    dye = bff.NPSDirectDye()
    dye.x = x
    dye.y = y
    dye.z = z
    dye.m = m
    dye.phi = phi
    dye.steady_state_anisotropy = ravg
    return dye


def test_fast_nps_model_indices_reproduce_the_reference_gui_flags():
    """Each of Fast-NPS's five UI models has one unambiguous flag tuple."""
    expected = {
        1: (False, False, False),
        2: (False, True, False),
        3: (True, True, True),
        4: (False, True, True),
        5: (False, False, True),
    }

    for index, flags in expected.items():
        model = bff.nps_dye_model(index)
        assert model.index == index
        assert (model.fixed_mean_position, model.isotropic,
                model.distance_convolved) == flags


@pytest.mark.parametrize("index", (-1, 0, 6))
def test_fast_nps_model_indices_reject_out_of_range_values(index):
    with pytest.raises(IMP.ValueException):
        bff.nps_dye_model(index)


def test_nps_direct_fret_efficiency_is_half_for_isotropic_dyes_at_riso():
    dye1 = _direct_dye()
    dye2 = _direct_dye(z=50.0)

    assert bff.nps_direct_fret_efficiency(dye1, dye2, 50.0) == pytest.approx(
        0.5, rel=1e-12, abs=1e-14)


def test_nps_direct_transfer_anisotropy_is_two_fifths_for_aligned_rigid_dyes():
    dye1 = _direct_dye(m=-1.0, ravg=0.4)
    dye2 = _direct_dye(m=-1.0, ravg=0.4)

    assert bff.nps_direct_transfer_anisotropy(dye1, dye2) == pytest.approx(
        0.4, rel=1e-12, abs=1e-14)


def test_nps_direct_transfer_anisotropy_is_negative_one_fifth_for_orthogonal_rigid_dyes():
    dye1 = _direct_dye(m=-1.0, ravg=0.4)
    dye2 = _direct_dye(m=0.0, phi=0.0, ravg=0.4)

    assert bff.nps_direct_transfer_anisotropy(dye1, dye2) == pytest.approx(
        -0.2, rel=1e-12, abs=1e-14)


def test_nps_direct_transfer_anisotropy_is_zero_at_the_magic_angle():
    dye1 = _direct_dye(m=-1.0, ravg=0.4)
    dye2 = _direct_dye(m=-1.0 / math.sqrt(3.0), ravg=0.4)

    assert bff.nps_direct_transfer_anisotropy(dye1, dye2) == pytest.approx(
        0.0, abs=1e-14)


def test_nps_direct_transfer_anisotropy_is_exactly_zero_for_a_depolarized_dye():
    dye1 = _direct_dye(m=-1.0, ravg=0.4)
    dye2 = _direct_dye(m=0.0, phi=math.pi / 2.0, ravg=0.0)

    assert bff.nps_direct_transfer_anisotropy(dye1, dye2) == 0.0


def test_nps_direct_transfer_anisotropy_matches_non_axis_partial_order_pin():
    dye1 = _direct_dye(m=-0.8, phi=0.0, ravg=0.1)
    dye2 = _direct_dye(m=-0.8, phi=math.pi / 2.0, ravg=0.225)

    assert bff.nps_direct_transfer_anisotropy(dye1, dye2) == pytest.approx(
        0.017160000000000043, rel=1e-12, abs=1e-14)


def test_nps_direct_transfer_anisotropy_is_invariant_to_axis_representations():
    dye1 = _direct_dye(m=-0.8, phi=0.0, ravg=0.1)
    dye2 = _direct_dye(m=-0.8, phi=math.pi / 2.0, ravg=0.225)
    anisotropy = bff.nps_direct_transfer_anisotropy(dye1, dye2)

    assert bff.nps_direct_transfer_anisotropy(dye2, dye1) == pytest.approx(
        anisotropy, rel=1e-12, abs=1e-14)
    assert bff.nps_direct_transfer_anisotropy(
        _direct_dye(m=0.8, phi=math.pi, ravg=0.1), dye2) == pytest.approx(
            anisotropy, rel=1e-12, abs=1e-14)
    assert bff.nps_direct_transfer_anisotropy(
        dye1, _direct_dye(m=-0.8, phi=5.0 * math.pi / 2.0, ravg=0.225)) == (
            pytest.approx(anisotropy, rel=1e-12, abs=1e-14))


def test_nps_direct_transfer_anisotropy_ignores_finite_dye_positions():
    dye1 = _direct_dye(m=-0.8, phi=0.0, ravg=0.1)
    coincident_dye2 = _direct_dye(m=-0.8, phi=math.pi / 2.0, ravg=0.225)
    far_dye2 = _direct_dye(
        z=1.0e6, m=-0.8, phi=math.pi / 2.0, ravg=0.225)

    coincident = bff.nps_direct_transfer_anisotropy(dye1, coincident_dye2)
    far = bff.nps_direct_transfer_anisotropy(dye1, far_dye2)

    assert math.isfinite(coincident)
    assert coincident == pytest.approx(far, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        pytest.param(f"{dye_name}.{field}", value,
                     id=f"{dye_name}-{field}-{value_name}")
        for dye_name in ("dye1", "dye2")
        for field in ("x", "y", "z", "m", "phi",
                      "steady_state_anisotropy")
        for value_name, value in (("nan", math.nan), ("infinite", math.inf))
    ] + [
        pytest.param(f"{dye_name}.m", value,
                     id=f"{dye_name}-m-{value_name}")
        for dye_name in ("dye1", "dye2")
        for value_name, value in (("below-minus-one", -1.000001),
                                  ("above-one", 1.000001))
    ] + [
        pytest.param(f"{dye_name}.steady_state_anisotropy", value,
                     id=f"{dye_name}-ravg-{value_name}")
        for dye_name in ("dye1", "dye2")
        for value_name, value in (("below-zero", -0.001),
                                  ("above-fundamental-limit", 0.400001))
    ],
)
def test_nps_direct_transfer_anisotropy_rejects_invalid_public_states(
        target, value):
    dye1 = _direct_dye()
    dye2 = _direct_dye()
    dye_name, attribute = target.split(".")
    setattr({"dye1": dye1, "dye2": dye2}[dye_name], attribute, value)

    with pytest.raises(IMP.ValueException,
                       match="nps_direct_transfer_anisotropy"):
        bff.nps_direct_transfer_anisotropy(dye1, dye2)


def test_nps_direct_fret_efficiency_is_six_sevenths_for_aligned_z_rigid_dyes():
    dye1 = _direct_dye(m=-1.0, ravg=0.4)
    dye2 = _direct_dye(z=50.0, m=-1.0, ravg=0.4)

    assert bff.nps_direct_fret_efficiency(dye1, dye2, 50.0) == pytest.approx(
        6.0 / 7.0, rel=1e-12, abs=1e-14)


def test_nps_direct_fret_efficiency_is_two_thirds_for_z_rigid_donor():
    donor = _direct_dye(m=-1.0, ravg=0.4)
    acceptor = _direct_dye(z=50.0)

    assert bff.nps_direct_fret_efficiency(donor, acceptor, 50.0) == pytest.approx(
        2.0 / 3.0, rel=1e-12, abs=1e-14)


def test_nps_direct_fret_efficiency_is_one_third_for_x_rigid_donor():
    donor = _direct_dye(m=0.0, phi=0.0, ravg=0.4)
    acceptor = _direct_dye(z=50.0)

    assert bff.nps_direct_fret_efficiency(donor, acceptor, 50.0) == pytest.approx(
        1.0 / 3.0, rel=1e-12, abs=1e-14)


def test_direct_fret_zero_orientation_factor_returns_exact_zero():
    dye1 = _direct_dye(m=0.0, phi=0.0, ravg=0.4)
    dye2 = _direct_dye(y=50.0, m=-1.0, phi=0.0, ravg=0.4)

    efficiency = bff.nps_direct_fret_efficiency(dye1, dye2, 50.0)

    assert math.isfinite(efficiency)
    assert efficiency == 0.0


def test_nps_direct_fret_efficiency_matches_non_axis_orientation_pin():
    dye1 = _direct_dye(m=-4.0 / 5.0, phi=0.0, ravg=0.1)
    dye2 = _direct_dye(
        x=3.0, y=4.0, z=12.0, m=-4.0 / 5.0, phi=math.pi / 2.0, ravg=0.225)

    assert bff.nps_direct_fret_efficiency(dye1, dye2, 13.0) == pytest.approx(
        0.7366734919903035, rel=1e-12, abs=1e-14)


def test_nps_direct_fret_efficiency_evaluates_the_inclusive_150_angstrom_cutoff():
    dye1 = _direct_dye()
    dye2 = _direct_dye(z=150.0)

    assert bff.nps_direct_fret_efficiency(dye1, dye2, 50.0) == pytest.approx(
        1.0 / 730.0, rel=1e-12, abs=1e-14)


def test_nps_direct_fret_efficiency_is_exactly_zero_beyond_150_angstroms():
    dye1 = _direct_dye()
    dye2 = _direct_dye(z=150.000001)

    assert bff.nps_direct_fret_efficiency(dye1, dye2, 50.0) == 0.0


def test_nps_direct_fret_efficiency_rejects_overflowed_derived_separation():
    dye1 = _direct_dye(x=-sys.float_info.max)
    dye2 = _direct_dye(x=sys.float_info.max)

    with pytest.raises(IMP.ValueException):
        bff.nps_direct_fret_efficiency(dye1, dye2, 50.0)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        pytest.param("dye1.m", -1.000001, id="m-below-minus-one"),
        pytest.param("dye2.m", 1.000001, id="m-above-one"),
        pytest.param("dye1.steady_state_anisotropy", -0.001,
                     id="negative-anisotropy"),
        pytest.param("dye2.steady_state_anisotropy", 0.400001,
                     id="anisotropy-above-fundamental-limit"),
        pytest.param("dye1.x", math.nan, id="nonfinite-coordinate"),
        pytest.param("dye2.phi", math.nan, id="nonfinite-phi"),
        pytest.param("r_iso", 0.0, id="zero-riso"),
        pytest.param("r_iso", -1.0, id="negative-riso"),
        pytest.param("r_iso", math.nan, id="nan-riso"),
        pytest.param("r_iso", math.inf, id="infinite-riso"),
        pytest.param("zero_separation", None, id="zero-separation"),
    ],
)
def test_nps_direct_fret_efficiency_rejects_invalid_public_inputs(target, value):
    dye1 = _direct_dye()
    dye2 = _direct_dye(z=50.0)
    r_iso = 50.0

    if target == "r_iso":
        r_iso = value
    elif target == "zero_separation":
        dye2 = _direct_dye()
    else:
        dye_name, attribute = target.split(".")
        setattr({"dye1": dye1, "dye2": dye2}[dye_name], attribute, value)

    with pytest.raises(IMP.ValueException):
        bff.nps_direct_fret_efficiency(dye1, dye2, r_iso)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        pytest.param(
            f"{dye_name}.{field}", value,
            id=f"{dye_name}-{field}-{value_name}")
        for dye_name in ("dye1", "dye2")
        for field in ("x", "y", "z", "m", "phi", "steady_state_anisotropy")
        for value_name, value in (("nan", math.nan), ("infinite", math.inf))
    ],
)
def test_nps_direct_fret_efficiency_rejects_nonfinite_every_dye_field(
        target, value):
    dye1 = _direct_dye()
    dye2 = _direct_dye(z=50.0)
    dye_name, attribute = target.split(".")
    setattr({"dye1": dye1, "dye2": dye2}[dye_name], attribute, value)

    with pytest.raises(IMP.ValueException):
        bff.nps_direct_fret_efficiency(dye1, dye2, 50.0)


def test_nps_direct_fret_efficiency_validates_before_far_distance_cutoff():
    dye1 = _direct_dye()
    dye2 = _direct_dye(z=151.0, phi=math.nan)

    with pytest.raises(IMP.ValueException):
        bff.nps_direct_fret_efficiency(dye1, dye2, 50.0)


if __name__ == "__main__":
    import sys
    try:
        import pytest
    except ImportError:
        print("pytest not installed; skipping", __file__)
        sys.exit(0)
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
