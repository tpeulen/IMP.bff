"""`imp_bff dock` and `imp_bff dock-errors`, compiled.

Both were Python in `bin/imp_bff`: `dock` drove `IMP.pmi.macros.ReplicaExchange`
on one replica, `dock-errors` a process pool of `dock_minimize` trials. The C++
port (`src/imp/DockingReplicaExchange.cpp`) draws the same random numbers in the
same order, and was diffed against the Python on the shipped HIV-RT example with
IMP's generator seeded: every frame score, `scores.csv`, all best-scoring PDBs,
`pdbs/model.psf`, both RMF trajectories and the stat rows (stopwatch aside)
identical. These tests pin what does not need the Python to check: the layout
PMI wrote, that a seed reproduces a run, and the report.
"""

import ast
import os
from pathlib import Path

import pytest

import IMP.bff

REPO = Path(__file__).resolve().parents[2]
HIV = REPO / "examples" / "structure" / "HIV_RT"

pytestmark = pytest.mark.skipif(
    IMP.bff.get_build() == "core",
    reason="docking is the IMP connection layer, not compiled into this build")


def _dock(capfd, out, *extra):
    capfd.readouterr()
    code = IMP.bff.command_line_main(
        ["dock", "-p", str(HIV / "protein_1R0A.pdb"), "-p", str(HIV / "dna.pdb"),
         "-j", str(HIV / "hiv_rt.fps.json"), "-c", "resolved", "-o", str(out),
         "-n", "3", "--mc-steps", "2", "--n-best", "2"] + [str(e) for e in extra])
    captured = capfd.readouterr()
    return code, captured.out


def test_dock_writes_pmis_run_layout(tmp_path, capfd):
    out = tmp_path / "run"
    code, stdout = _dock(capfd, out, "--seed", 1)
    assert code == 0
    for name in ("initial.0.rmf3", "rmfs/0.rmf3", "pdbs/model.0.pdb",
                 "pdbs/model.1.pdb", "pdbs/model.psf", "best.scores.rex.py",
                 "stat.0.out", "stat_replica.0.out", "scores.csv"):
        assert (out / name).is_file(), name
    assert stdout.count("--- frame ") == 3
    assert "over 18 distances (10 accessible volumes)" in stdout
    assert f"trajectory {out}/initial.0.rmf3" in stdout


def test_dock_stat_files_read_as_pmi_stat2(tmp_path, capfd):
    """PMI's own readers take the header dict and the rows as Python literals."""
    out = tmp_path / "run"
    assert _dock(capfd, out, "--seed", 1)[0] == 0
    lines = (out / "stat.0.out").read_text().splitlines()
    header = ast.literal_eval(lines[0])
    assert header["STAT2HEADER"] == "STAT2HEADER"
    # the environment dump PMI wrote (credentials included) is not reproduced
    assert "STAT2HEADER_ENVIRON" not in header
    keys = {v: k for k, v in header.items() if isinstance(k, int)}
    assert {"Total_Score", "MonteCarlo_Nframe", "rmf_file"} <= set(keys)
    rows = [ast.literal_eval(line) for line in lines[1:]]
    assert [int(r[keys["MonteCarlo_Nframe"]]) for r in rows] == [0, 1, 2]
    # one replica runs at the ladder's minimum, whatever --temperature said
    assert all(r[keys["MonteCarlo_Temperature"]] == "1.0" for r in rows)
    best = ast.literal_eval((out / "best.scores.rex.py").read_text().split("=")[1])
    assert best == sorted(best) and len(best) == 2


def test_a_seed_reproduces_a_dock(tmp_path, capfd):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _dock(capfd, a, "--seed", 5)[0] == 0
    assert _dock(capfd, b, "--seed", 5)[0] == 0
    assert (a / "scores.csv").read_text() == (b / "scores.csv").read_text()
    assert (a / "pdbs" / "model.0.pdb").read_bytes() == (b / "pdbs" / "model.0.pdb").read_bytes()


def test_dock_errors_reports_the_spread_and_writes_the_uncertainty(tmp_path, capfd):
    out = tmp_path / "errors"
    capfd.readouterr()
    code = IMP.bff.command_line_main(
        ["dock-errors", "-p", str(HIV / "protein_1R0A.pdb"), "-p", str(HIV / "dna.pdb"),
         "-j", str(HIV / "hiv_rt.fps.json"), "-c", "resolved", "-o", str(out),
         "-n", "2", "--iterations", "5", "--n-workers", "1"])
    stdout = capfd.readouterr().out
    assert code == 0
    assert "2 trials, score " in stdout
    assert f"in {out}/trial_000" in stdout
    for t in ("trial_000", "trial_001"):
        assert (out / t / "scores.csv").is_file()
    # the Python program meant to write these and never did (a TypeError it
    # swallowed); the port writes them
    assert (out / "uncertainty.csv").is_file()
    assert (out / "uncertainty.pdb").is_file()


def test_dock_errors_trials_do_not_depend_on_the_schedule(tmp_path, capfd):
    """Trial i is seeded i + 1 wherever it runs: serial and forked agree."""
    serial, forked = tmp_path / "serial", tmp_path / "forked"
    for out, workers in ((serial, "1"), (forked, "2")):
        assert IMP.bff.command_line_main(
            ["dock-errors", "-p", str(HIV / "protein_1R0A.pdb"), "-p", str(HIV / "dna.pdb"),
             "-j", str(HIV / "hiv_rt.fps.json"), "-c", "resolved", "-o", str(out),
             "-n", "2", "--iterations", "5", "--n-workers", workers]) == 0
    capfd.readouterr()
    for t in ("trial_000", "trial_001"):
        assert (serial / t / "scores.csv").read_text() == (forked / t / "scores.csv").read_text()
