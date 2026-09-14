"""The compiled command line's groups, run as the program runs them.

`imp_bff probe-pdb2cif`, `imp_bff labelizer` and `imp_bff fps-distance` were
the Python programs `bin/imp_bff_probe_pdb2cif`, `bin/imp_bff_labelizer` and
`bin/imp_bff_fps_distance`; they are groups of the one C++ program now
(`bin/imp_bff.cpp`, `include/CommandLine.h`). Two bugs these tests were written
against stay pinned: `--probe-id`'s documented default path used to raise
before it read anything, and `labelizer --show` used to hand a structure to a
container reader, which complained about EBML headers.

`IMP.bff.command_line_main` is the dispatcher the executable's `main` calls,
so a list of words and an exit code is the whole interface. Output comes from
C++ at the file descriptor, hence `capfd`.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import IMP.bff

PDB = IMP.bff.get_example_path("structure/T4L/3GUN.pdb")
FPS_XYZ = Path(__file__).resolve().parent.parent / "input" / "fps" / "p_1bp_D.xyz"


def run(*words):
    return IMP.bff.command_line_main([str(w) for w in words])


@pytest.fixture
def probe_pdb(tmp_path):
    """A small probe PDB, written here: the shipped cgprobe structures are
    fetched on first use and may be absent."""
    coords = [c for i in range(12) for c in (i * 1.4, 0.3 * (i % 2), 0.0)]
    path = tmp_path / "probe.pdb"
    IMP.bff.write_pdb(coords, str(path), "A", "LYS")
    return path


def test_probe_pdb2cif_converts_with_its_default_id(tmp_path, probe_pdb):
    """`--probe-id` is empty by default, meaning "take it from the file"."""
    out = tmp_path / "dye.cif"
    assert run("probe-pdb2cif", probe_pdb, out) == 0
    text = out.read_text()
    assert "_atom_site" in text
    assert text.count("ATOM") + text.count("HETATM") >= 12


def test_probe_pdb2cif_takes_an_explicit_id(tmp_path, probe_pdb):
    out = tmp_path / "dye.cif"
    assert run("probe-pdb2cif", probe_pdb, out, "--probe-id", "A48") == 0
    assert "A48" in out.read_text()


def test_labelizer_scores_a_structure(tmp_path, capfd):
    out = tmp_path / "t4l.mmfdb.pto"
    assert run("labelizer", PDB, "--no-conservation", "-o", out) == 0
    assert len(IMP.bff.labelizer_read_pto_scores(str(out))) > 0
    assert "sites scored" in capfd.readouterr().out


def test_labelizer_default_output_sits_beside_the_structure(tmp_path):
    src = tmp_path / "t4l.pdb"
    shutil.copy(PDB, src)
    assert run("labelizer", src, "--no-conservation") == 0
    assert (tmp_path / "t4l.mmfdb.pto").is_file()


def test_labelizer_show_says_what_it_wants(capfd):
    """`--show` reads a container; given a structure it says so, exit 2."""
    assert run("labelizer", PDB, "--show") == 2
    assert "container" in capfd.readouterr().err


def test_labelizer_show_reads_what_it_wrote(tmp_path, capfd):
    out = tmp_path / "t4l.mmfdb.pto"
    assert run("labelizer", PDB, "--no-conservation", "-o", out) == 0
    capfd.readouterr()
    assert run("labelizer", out, "--show") == 0
    shown = capfd.readouterr().out
    assert "arithmetic" in shown
    assert "best sites" in shown


def test_labelizer_dyes_go_in_pairs(capfd):
    assert run("labelizer", PDB, "--no-conservation", "--donor", "Alexa488") == 1
    assert "go together" in capfd.readouterr().err


def test_fps_distance_reads_a_duplicate_expanded_fps_cloud(capfd):
    """FPS's AV3 export writes a voxel once per radius that fits: 5473 lines
    over 3187 voxels in this file. The duplicates are a weighting, kept; the
    unique count is reported beside them."""
    assert run("fps-distance", FPS_XYZ, FPS_XYZ, "--json") == 0
    report = json.loads(capfd.readouterr().out)
    assert report["points"] == [5473, 5473]
    assert report["unique_voxels"] == [3187, 3187]
    assert report["Rmp"] == pytest.approx(0.0, abs=1e-9)
    # efficiency weights close pairs as 1/r^6, so <RDA>_E sits above <RDA>
    # only when the cloud is far smaller than R0 -- here it is one cloud
    assert report["RDA_E"] > 0 and report["RDA_mean"] > 0


def test_fps_distance_report_names_the_duplicate_expansion(capfd):
    assert run("fps-distance", FPS_XYZ, FPS_XYZ, "-n", "1000") == 0
    out = capfd.readouterr().out
    assert "duplicate-expanded" in out
    assert "declared mean" in out
    assert "<RDA>_E" in out


def test_usage_errors_exit_2(capfd):
    assert run("fps-distance", FPS_XYZ) == 2
    assert run() == 2
    capfd.readouterr()


def _binary():
    found = shutil.which("imp_bff")
    if found and not open(found, "rb").read(2) == b"#!":
        return found
    build = os.environ.get("IMP_BIN_DIR")
    if build and os.path.isfile(os.path.join(build, "imp_bff")):
        return os.path.join(build, "imp_bff")
    return None


def test_the_executable_runs_the_same_dispatcher(tmp_path):
    """`bin/imp_bff.cpp` is a `main` around `command_line_main`; where the
    executable is on PATH, it answers the same words."""
    exe = _binary()
    if exe is None:
        pytest.skip("the compiled imp_bff is not on PATH")
    out = subprocess.run([exe, "fps-distance", str(FPS_XYZ), str(FPS_XYZ), "--json",
                          "-n", "1000"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["unique_voxels"] == [3187, 3187]
    assert subprocess.run([exe], capture_output=True).returncode == 2
