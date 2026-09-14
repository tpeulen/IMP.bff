"""`imp_bff potentials2pto`, run: four loose `.npy` tables in, one container out.

The program was `bin/imp_bff_potentials2pto`; it is a subcommand of the
compiled `imp_bff` now, with the conversions in ProbePotentialTables.cpp and
the `.npy` reader in internal/Npy.h. On ChiSurf's own database the compiled
command and the Python program wrote the same tables and the same manifest
(the container bytes differ run to run for both: ptolib is not
byte-deterministic).

These tests build a small synthetic database with numpy, so they need no
ChiSurf checkout, and hold the conversion to the four decisions the manifest
reports: NaNs and the bins below `min_dist` become the repulsion, the UNRES
upper triangle is what is written, the hydrogen-bond divergence is held flat,
and the proline sentinel becomes the least probable measured value.
"""

import json

import numpy as np
import pytest

import IMP.bff

ORDER = ["CYS", "MET", "PHE", "ILE", "LEU", "VAL", "TRP", "TYR", "ALA", "GLY",
         "THR", "SER", "GLN", "ASN", "GLU", "ASP", "HIS", "ARG", "LYS", "PRO"]


@pytest.fixture
def database(tmp_path):
    rng = np.random.default_rng(3)
    mj = rng.normal(-2.0, 1.0, (20, 20))
    np.save(tmp_path / "mj.npy", mj)

    unres = rng.normal(-0.1, 0.05, (20, 20, 120)).astype(np.float32)
    unres[0, 1, 0] = np.nan                    # a NaN in bin 0
    unres[1, 0, :] = 0.0                       # the empty lower triangle
    np.save(tmp_path / "unres.npy", unres)

    hb = rng.normal(0.0, 1.0, (4, 300))
    hb[:, :10] = 7e33                          # the divergence below 0.1 A
    np.save(tmp_path / "hb.npy", hb)

    n = 36
    maps = -rng.random((3, n * n)).astype(np.float32)
    maps[1, :5] = 1.0                          # proline's sentinel
    grids = np.zeros((2, n * n), dtype=np.float32)
    np.save(tmp_path / "rama_ala_pro_gly.npy", np.concatenate([grids, maps]))
    return tmp_path, mj, unres, hb, maps


def _run(capfd, database, out):
    capfd.readouterr()
    code = IMP.bff.command_line_main(
        ["potentials2pto", "--database", str(database), "--output", str(out)])
    return code, capfd.readouterr()


def test_the_ramachandran_resolution_is_the_shipped_one(database, tmp_path, capfd):
    """360 bins is the program's resolution: a 36-bin synthetic map is refused
    with the shapes named, not silently resampled."""
    db = database[0]
    code, captured = _run(capfd, db, tmp_path / "out.pto")
    assert code == 1
    assert "360" in captured.err


@pytest.fixture
def shipped_resolution_database(database):
    db, mj, unres, hb, _ = database
    rng = np.random.default_rng(5)
    maps = -rng.random((3, 360 * 360)).astype(np.float32)
    maps[1, :7] = 1.0
    grids = np.zeros((2, 360 * 360), dtype=np.float32)
    np.save(db / "rama_ala_pro_gly.npy", np.concatenate([grids, maps]))
    return db, mj, unres, hb, maps


def test_writes_every_table_and_says_so(shipped_resolution_database, tmp_path, capfd):
    db = shipped_resolution_database[0]
    out = tmp_path / "potentials.pto"
    code, captured = _run(capfd, db, out)
    assert code == 0, captured.err
    assert "wrote %s" % out in captured.out
    assert sorted(IMP.bff.potential_table_names(str(out))) == \
        ["hbond", "mj", "ramachandran", "unres"]


def test_the_conversions_are_the_ones_the_manifest_claims(
        shipped_resolution_database, tmp_path, capfd):
    db, mj, unres, hb, maps = shipped_resolution_database
    out = tmp_path / "potentials.pto"
    assert _run(capfd, db, out)[0] == 0
    manifest = json.loads(IMP.bff.read_potential_manifest(str(out)))
    assert manifest["made_by"] == "imp_bff potentials2pto"
    assert list(manifest["tables"]) == ["mj", "unres", "hbond", "ramachandran"]

    # mj: two bins of half the cutoff, both the matrix value, as repr() spells it
    text = IMP.bff.read_potential_table("mj", str(out)).text.splitlines()
    assert text[0] == "3.25 20"
    cys, met, v0, v1 = text[2].split()
    assert (cys, met) == ("CYS", "MET")
    assert v0 == v1 == repr(float(mj[0, 1]))

    # unres: the NaN counted, the low bins repulsive, the upper triangle written
    meta = manifest["tables"]["unres"]
    assert meta["note"].startswith("1 NaNs in bin 0")
    rows = IMP.bff.read_potential_table("unres", str(out)).text.splitlines()
    fields = rows[2].split()                   # CYS MET
    values = [float(x) for x in fields[2:]]
    below = int(round(3.5 / 0.05))
    assert all(v == 100.0 for v in values[:below])
    assert values[below:] == [float(x) for x in unres[0, 1, below:].astype(np.float64)]

    # hbond: the divergence held at the value at 1.3 A
    grid = IMP.bff.read_potential_table("hbond", str(out))
    hbond = np.asarray(grid.get_values()).reshape(list(grid.shape))
    first = int(round(1.3 / 0.01))
    assert np.all(hbond[:, :first] == hbond[:, first:first + 1])
    assert manifest["tables"]["hbond"]["note"].startswith("40 values above 1e5")

    # ramachandran: two coordinate grids dropped, the sentinel replaced
    rama = IMP.bff.read_potential_table("ramachandran", str(out))
    assert list(rama.shape) == [3, 360, 360]
    pro = np.asarray(rama.get_values()).reshape(3, -1)[1]
    assert pro[:7].tolist() == [float(maps[1, 7:].min())] * 7
    assert (pro <= 0.0).all()


def test_a_missing_table_is_named(tmp_path, capfd):
    code, captured = _run(capfd, tmp_path, tmp_path / "out.pto")
    assert code == 1
    assert "missing:" in captured.err and "mj.npy" in captured.err
