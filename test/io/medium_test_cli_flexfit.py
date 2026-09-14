"""`imp_bff flexfit`, compiled.

The Python command (`bin/imp_bff`) no longer ran by 2026-09-14 -- it read
`FlexFitSelection.residues`, an attribute the C++ value never had. With that one
accessor fixed, the port (`src/imp/FlexibleFitting.cpp`) was diffed against it
on `examples/structure/TG2` with IMP's generator seeded: every PDB frame and the
restraint log byte-identical, and the RMF trajectory (mode S) identical in
coordinates and stored restraint scores over all 44 frames. Here: the files a
run writes, and that a seed reproduces one. About half a minute -- the anneal.
"""

import os
from pathlib import Path

import pytest

import IMP.bff

REPO = Path(__file__).resolve().parents[2]
TG2 = REPO / "examples" / "structure" / "TG2"

pytestmark = pytest.mark.skipif(
    IMP.bff.get_build() == "core",
    reason="flexible fitting is the IMP connection layer, not compiled into this build")


def _flexfit(directory, output, seed):
    cwd = Path.cwd()
    try:
        os.chdir(directory)
        return IMP.bff.command_line_main(
            ["flexfit", "-i", str(TG2 / "topology.pdb"), "-l", str(TG2 / "flex.fps.json"),
             "-c", "S1_chi2", "-n", "10", "-p", "10", "-o", output, "--seed", str(seed)])
    finally:
        os.chdir(cwd)


def test_flexfit_pdb_frames_and_restraint_log(tmp_path, capfd):
    assert _flexfit(tmp_path, "out.pdb", 2) == 0
    capfd.readouterr()
    frames = sorted(tmp_path.glob("out_state_0_*.pdb"))
    log = (tmp_path / "out.rst.txt").read_text().splitlines()
    assert frames and len(frames) == len(log)
    # frame, the FRET network, the clash term
    first = log[0].split("\t")
    assert first[0] == "0" and len(first) == 4


def test_flexfit_rmf_is_numbered_and_a_seed_reproduces_it(tmp_path, capfd):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert _flexfit(a, "out.rmf3", 4) == 0
    assert _flexfit(b, "out.rmf3", 4) == 0
    capfd.readouterr()
    assert (a / "out.0.rmf3").is_file()
    pa = sorted(p.name for p in a.iterdir())
    assert pa == sorted(p.name for p in b.iterdir())


def test_flexfit_without_its_required_options_is_a_usage_error(capfd):
    assert IMP.bff.command_line_main(["flexfit", "-i", str(TG2 / "topology.pdb")]) == 2
    capfd.readouterr()
