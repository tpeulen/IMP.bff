"""IMP's atom radii, readable without IMP: one atom-radius set for every AV door.

Owner, 2026-10-12: there is ONE atom-radius set for accessible volumes,
IMP's radii times ``AV_DEFAULT_RADIUS_SCALE``. The structure door reads them
off the particles ``IMP::atom::read_pdb`` builds; the file door
(``load_structure_with_vdw``, ``get_av_from_pdb``) cannot link IMP, so it reads
the same mapping from ``imp_atom_radius`` -- a table regenerated from IMP by
``prototypes/imp_atom_radii/generate.py``. These tests hold the two equal on
every atom of the shipped structures, and hold the file door to the table.
"""
import numpy as np
import pytest

import IMP
import IMP.bff

FIXTURES = (
    "structure/T4L/3GUN.pdb",
    "structure/HIV_RT/protein_1R0A.pdb",
    "structure/HIV_RT/dna.pdb",
    "structure/GBP/mGBP2A.pdb",
    "structure/MBP/AF-P0AEX9-F1.pdb",
)


def test_unknown_atoms_take_imps_default():
    assert IMP.bff.get_imp_atom_default_radius() == 1.7
    assert IMP.bff.imp_atom_radius("XYZ", "C1") == 1.7
    assert IMP.bff.imp_atom_radius("HOH", "O") == 1.7


def test_known_types_carry_charmm_radii():
    # CT1 (an alpha carbon), CT3 (alanine's methyl), O (carbonyl), NH1.
    assert IMP.bff.imp_atom_radius("ALA", "CA") == pytest.approx(2.275)
    assert IMP.bff.imp_atom_radius("ALA", "CB") == pytest.approx(2.06)
    assert IMP.bff.imp_atom_radius("ALA", "N") == pytest.approx(1.85)
    # carbonyl oxygen is 1.70, IMP's default, so it is not a row of the table
    assert IMP.bff.imp_atom_radius("ALA", "O") == pytest.approx(1.70)
    assert IMP.bff.get_number_of_imp_atom_radii() > 600


@pytest.mark.parametrize("fixture", FIXTURES)
def test_table_equals_read_pdb(fixture):
    """Every atom read_pdb types gets the table's radius, residue by residue."""
    import IMP.atom
    import IMP.core

    IMP.set_log_level(IMP.SILENT)
    m = IMP.Model()
    h = IMP.atom.read_pdb(IMP.bff.get_example_path(fixture), m,
                          IMP.atom.NonWaterPDBSelector())
    misses = []
    n = 0
    for a in IMP.atom.get_by_type(h, IMP.atom.ATOM_TYPE):
        parent = a.get_parent()
        if not IMP.atom.Residue.get_is_setup(parent):
            continue
        res = IMP.atom.Residue(parent).get_residue_type().get_string()
        name = IMP.atom.Atom(a).get_atom_type().get_string()
        r = IMP.core.XYZR(a).get_radius()
        n += 1
        if abs(IMP.bff.imp_atom_radius(res, name) - r) > 1e-4:
            misses.append((res, name, r, IMP.bff.imp_atom_radius(res, name)))
    assert n > 0
    assert misses == []


@pytest.mark.parametrize("fixture", FIXTURES[:3])
def test_file_door_is_imps_radii_times_the_scale(fixture):
    path = IMP.bff.get_example_path(fixture)
    records = IMP.bff.read_pdb_records(path)
    xyzr = np.asarray(IMP.bff.load_structure_with_vdw(path)).reshape(-1, 4)
    assert len(xyzr) == len(records)
    expected = np.array([IMP.bff.imp_atom_radius(r.res_name, r.atom_name)
                         for r in records]) * IMP.bff.AV_DEFAULT_RADIUS_SCALE
    np.testing.assert_allclose(xyzr[:, 3], expected, rtol=0, atol=1e-12)
    unscaled = np.asarray(IMP.bff.load_structure_with_vdw(path, 1.0)).reshape(-1, 4)
    np.testing.assert_allclose(unscaled[:, 3] * IMP.bff.AV_DEFAULT_RADIUS_SCALE,
                               xyzr[:, 3], rtol=0, atol=1e-12)


def test_file_door_refuses_a_non_positive_scale():
    path = IMP.bff.get_example_path(FIXTURES[0])
    with pytest.raises(Exception):
        IMP.bff.load_structure_with_vdw(path, 0.0)
