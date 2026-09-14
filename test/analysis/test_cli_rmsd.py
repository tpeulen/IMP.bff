"""`imp_bff rmsd`: RMSD columns added to an OLGA `.ol4` table.

The command was pandas + mdtraj inside the Python program `bin/imp_bff` (and
had not run since pandas 2 made `str.split`'s `n` keyword-only); it is C++
now (src/CommandLineRmsd.cpp). What it owes: one `RMSD_<reference>` column per
reference and one `best_<i>_frame_<k>` column per best-scoring row, in
Angstrom, over every atom after optimal superposition, written beside the
table as `<stem>.rmsd.ol4` with pandas' row index.
"""

import csv
import math

import numpy as np
import pytest

import IMP.bff


def _write(path, xyz):
    IMP.bff.write_pdb([float(v) for v in np.asarray(xyz).ravel()], str(path), "A", "ALA")


def _kabsch_rmsd(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    e = (a * a).sum() + (b * b).sum() - 2 * s.sum()
    return math.sqrt(max(e, 0.0) / len(a))


def test_rmsd_adds_reference_and_best_columns(tmp_path, capfd):
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 4, (12, 3))
    theta = 0.7
    rot = np.array([[math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    moved = ref @ rot.T + [3.0, -2.0, 5.0]
    bent = ref + rng.normal(0, 0.5, ref.shape)
    structures = tmp_path / "structures"
    structures.mkdir()
    _write(tmp_path / "ref.pdb", ref)
    _write(structures / "moved.pdb", moved)
    _write(structures / "bent.pdb", bent)
    table = tmp_path / "runs" / "run.ol4"
    table.parent.mkdir()
    table.write_text("structure\tall\tchi2\n"
                     "moved.pdb 0\t2.5\t1\n"
                     "bent.pdb 0\t1.5\t2\n")

    assert IMP.bff.command_line_main(
        ["rmsd", "-f", str(table), "-p", "../structures", "-r", str(tmp_path / "ref.pdb")]) == 0
    out = capfd.readouterr().out
    assert "best_file: bent.pdb" in out
    assert "Reference: RMSD_ref.pdb" in out

    with open(tmp_path / "runs" / "run.rmsd.ol4") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    header = rows[0]
    assert header[:4] == ["", "structure", "all", "chi2"]
    assert header[4:6] == ["file", "frame"]
    assert "RMSD_ref.pdb" in header
    assert "best_0_frame_0" in header and "best_1_frame_0" in header
    col = header.index("RMSD_ref.pdb")
    moved_rmsd = float(rows[1][col])
    bent_rmsd = float(rows[2][col])
    assert moved_rmsd == pytest.approx(0.0, abs=5e-3)
    assert bent_rmsd == pytest.approx(_kabsch_rmsd(bent, ref), abs=1e-3)
    # the best row (bent, score 1.5) against itself
    best = header.index("best_0_frame_0")
    assert float(rows[2][best]) == pytest.approx(0.0, abs=5e-3)


def test_rmsd_needs_a_reference(tmp_path, capfd):
    table = tmp_path / "t.ol4"
    table.write_text("structure\tall\nx.pdb 0\t1\n")
    assert IMP.bff.command_line_main(["rmsd", "-f", str(table)]) == 2
    capfd.readouterr()
