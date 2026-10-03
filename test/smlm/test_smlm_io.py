"""Native CSV preparation: measured 3D, explicit units and strict validation."""
import csv
from pathlib import Path

import numpy as np
import pytest

from IMP import bff


def write_csv(tmp_path, text):
    path = tmp_path / "localizations.csv"
    path.write_bytes(text.encode("utf-8"))
    return str(path)


def xyz(index):
    return np.asarray(index.get_coordinates()).reshape(-1, 3)


def sigmas(index):
    return np.asarray(index.get_sigmas()).reshape(-1, 3)


def test_source_xy_precision_groups_and_explicit_units(tmp_path):
    path = write_csv(tmp_path,
        "xnm,ynm,znm,locprecnm,locprecznm,phot,sitenumbers\r\n"
        "1,2,3,0.4,0.9,50000,7\r\n"
        "4,5,6,0.5,1.2,10,0\r\n"
        "7,8,9,0.6,1.3,20,-2\r\n")
    index = bff.read_smlm_csv(path, 10.0, [1, 2, 3])
    np.testing.assert_array_equal(xyz(index), [[0, 0, 0], [30, 30, 30], [60, 60, 60]])
    np.testing.assert_allclose(sigmas(index), [[4, 4, 9], [5, 5, 12], [6, 6, 13]])
    np.testing.assert_array_equal(index.get_particle_ids(), [7, -1, -1])
    np.testing.assert_array_equal(index.get_weights(), [1, 1, 1])
    options = bff.SMLMCSVOptions()
    options.weight_column = "PHOT"
    weighted = bff.read_smlm_csv(path, 1.0, [0, 0, 0], options)
    np.testing.assert_array_equal(weighted.get_weights(), [50000, 10, 20])


@pytest.mark.parametrize("headers", [
    '"X [nm]","Y [nm]","Z [nm]","uncertainty_x [nm]","uncertainty_y [nm]","uncertainty_z [nm]",note',
    "x_nm,y_nm,z_nm,xnmerr,ynmerr,locprecznm,note",
    "x,y,z,lpx,lpy,lpz,note",
])
def test_aliases_quoted_commas_and_multiline_fields(tmp_path, headers):
    path = write_csv(tmp_path, headers + '\r\n1,2,3,0.1,0.2,0.3,"note, with ""quotes""\nnext line"\r\n')
    index = bff.read_smlm_csv(path)
    np.testing.assert_array_equal(xyz(index), [[1, 2, 3]])
    np.testing.assert_allclose(sigmas(index), [[0.1, 0.2, 0.3]])
    np.testing.assert_array_equal(index.get_particle_ids(), [-1])


def test_per_axis_precision_takes_precedence_over_scalar_xy(tmp_path):
    path = write_csv(tmp_path, "x,y,z,xnmerr,ynmerr,locprecnm,locprecznm\n1,2,3,0.1,0.2,8,0.3\n")
    np.testing.assert_allclose(sigmas(bff.read_smlm_csv(path)), [[0.1, 0.2, 0.3]])


def test_empty_origin_is_the_explicit_zero_origin(tmp_path):
    path = write_csv(tmp_path, "x,y,z,locprecnm,locprecznm\n1,2,3,0.2,0.3\n")
    np.testing.assert_array_equal(xyz(bff.read_smlm_csv(path, 10.0, [])), [[10, 20, 30]])


def test_zero_based_particle_ids_require_explicit_source_convention(tmp_path):
    path = write_csv(tmp_path, "x,y,z,locprecnm,locprecznm,group\n1,2,3,0.2,0.3,0\n")
    assert list(bff.read_smlm_csv(path).get_particle_ids()) == [-1]
    options = bff.SMLMCSVOptions()
    assert options.zero_particle_id_is_unassigned is True
    options.zero_particle_id_is_unassigned = False
    index = bff.read_smlm_csv(path, 1.0, [], options)
    assert list(index.get_particle_ids()) == [0]
    assert bff.select_smlm_particles(index, [0]).get_number_of_localizations() == 1


@pytest.mark.parametrize("extra", ["xnm", "uncertainty", "zerr", "particleid"])
def test_conflicting_field_aliases_are_rejected(tmp_path, extra):
    path = write_csv(tmp_path,
        "x,y,z,locprecnm,locprecznm,group," + extra + "\n1,2,3,0.2,0.3,1,5\n")
    with pytest.raises((ValueError, RuntimeError), match="ambiguous CSV aliases"):
        bff.read_smlm_csv(path)


def test_large_source_origin_subtraction_is_scaled_without_intermediate_overflow(tmp_path):
    path = write_csv(tmp_path, "x,y,z,locprecnm,locprecznm\n1e308,0,0,1e308,1e308\n")
    index = bff.read_smlm_csv(path, 1e-308, [-1e308, 0, 0])
    np.testing.assert_allclose(xyz(index), [[2, 0, 0]], rtol=1e-15)
    np.testing.assert_allclose(sigmas(index), [[1, 1, 1]], rtol=1e-15)


@pytest.mark.parametrize("text, message", [
    ("x,y,locprecnm,locprecznm\n1,2,0.2,0.3\n", "missing z"),
    ("x,y,z,phot,sigma\n1,2,3,1000,150\n", "missing positional precision"),
    ("x,y,z,locprecnm\n1,2,3,0.2\n", "missing positional precision"),
    ("x,y,z,locprecnm,locprecznm\n1junk,2,3,0.2,0.3\n", "line 2"),
    ("x,y,z,locprecnm,locprecznm\nnan,2,3,0.2,0.3\n", "nonfinite"),
    ("x,y,z,locprecnm,locprecznm\n1,2,3,0,0.3\n", "positive"),
    ("x,y,z,locprecnm,locprecznm\n1,2,3,-1,0.3\n", "positive"),
    ("x,y,z,locprecnm,locprecznm\n1,2,3,0.2,inf\n", "nonfinite"),
    ("x,y,z,locprecnm,locprecznm\n1,2,3,0.2\n", "column count"),
    ("x,y,z,locprecnm,locprecznm\n1,2,3,0.2,\"0.3\n", "unterminated"),
    ("x,y,z,locprecnm,locprecznm\n1,2,3,0.2,\"0.3\"junk\n", "after CSV quote"),
    ("x_nm,x [nm],y,z,locprecnm,locprecznm\n1,1,2,3,0.2,0.3\n", "duplicate"),
    ("x,y,z,locprecnm,locprecznm,group\n1,2,3,0.2,0.3,1.5\n", "particle IDs"),
])
def test_invalid_input_is_rejected_without_silent_row_drops(tmp_path, text, message):
    path = write_csv(tmp_path, text)
    with pytest.raises((ValueError, RuntimeError), match=message):
        bff.read_smlm_csv(path)


@pytest.mark.parametrize("scale, origin", [(0, [0, 0, 0]), (-1, [0, 0, 0]),
                                           (float("inf"), [0, 0, 0]),
                                           (1, [0, 0]), (1, [0, 0, float("nan")])])
def test_invalid_unit_conversion_is_rejected(tmp_path, scale, origin):
    path = write_csv(tmp_path, "x,y,z,locprecnm,locprecznm\n1,2,3,0.2,0.3\n")
    with pytest.raises((ValueError, RuntimeError)):
        bff.read_smlm_csv(path, scale, origin)


def test_missing_values_require_explicit_caller_defaults(tmp_path):
    path = write_csv(tmp_path, "x,y\n1,2\n")
    options = bff.SMLMCSVOptions()
    options.allow_missing_z = True
    options.default_z = 3
    options.allow_missing_sigmas = True
    options.default_sigmas = [0.1, 0.2, 0.3]
    index = bff.read_smlm_csv(path, 10.0, [0, 0, 0], options)
    np.testing.assert_array_equal(xyz(index), [[10, 20, 30]])
    np.testing.assert_allclose(sigmas(index), [[1, 2, 3]])


def test_inclusive_region_preserves_source_measurements_and_ids():
    index = bff.SMLMIndex([0, 0, 0, 1, 2, 3, 2, 3, 4],
                         [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3], [9, -1, 12])
    selected = bff.select_smlm_region(index, [0, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(xyz(selected), xyz(index)[:2])
    np.testing.assert_array_equal(sigmas(selected), sigmas(index)[:2])
    np.testing.assert_array_equal(selected.get_weights(), [1, 2])
    np.testing.assert_array_equal(selected.get_particle_ids(), [9, -1])
    relabeled = bff.select_smlm_region(index, [0, 0, 0], [1, 2, 3], 42)
    np.testing.assert_array_equal(relabeled.get_particle_ids(), [42, 42])
    assert bff.select_smlm_region(index, [5, 5, 5], [6, 6, 6]).get_number_of_localizations() == 0
    with pytest.raises((ValueError, RuntimeError)):
        bff.select_smlm_region(index, [2, 0, 0], [1, 2, 3])


def test_saved_particle_selection_stays_native_and_preserves_source_order():
    index = bff.SMLMIndex([0, 0, 0, 1, 2, 3, 2, 3, 4, 5, 6, 7],
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4],
                         [0, 2, 3, 4], [9, -1, 12, 9])
    selected = bff.select_smlm_particles(index, [12, 9])
    np.testing.assert_array_equal(xyz(selected), xyz(index)[[0, 2, 3]])
    np.testing.assert_array_equal(sigmas(selected), sigmas(index)[[0, 2, 3]])
    np.testing.assert_array_equal(selected.get_weights(), [0, 3, 4])
    np.testing.assert_array_equal(selected.get_particle_ids(), [9, 12, 9])
    for ids in ([], [-1], [9, 9], [9, 99]):
        with pytest.raises((ValueError, RuntimeError)):
            bff.select_smlm_particles(index, ids)


@pytest.mark.parametrize("cell", range(1, 6))
def test_original_locmofit_csv_native_read_and_twenty_queries(cell):
    path = Path(__file__).resolve().parents[2] / ".omx/reference/locmofit/data" / (
        f"210121_U2OS_Nup96-SNAP-AF647_cell{cell}_sml.csv")
    if not path.is_file():
        pytest.skip("Original paper data absent; test performs no downloads")
    count, first = 0, []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if count < 20:
                first.append([float(row[name]) for name in ("xnm", "ynm", "znm")])
            count += 1
    index = bff.read_smlm_csv(str(path), 10.0)
    assert index.get_number_of_localizations() == count
    np.testing.assert_allclose(xyz(index)[:20], np.asarray(first) * 10, rtol=0, atol=0)
    result = index.evaluate_score((np.asarray(first) * 10).ravel().tolist())
    assert np.isfinite(result.score)
    assert len(result.densities) == 20
    assert np.isfinite(result.densities).all() and np.all(np.asarray(result.densities) > 0)
    if cell == 1:
        pore = bff.select_smlm_particles(index, [920])
        assert pore.get_number_of_localizations() > 0
        assert np.all(np.asarray(pore.get_particle_ids()) == 920)
