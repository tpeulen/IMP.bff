"""Every `imp_bff dye` command once, on the shipped cgprobe inputs.

The commands are compiled (`src/imp/CommandLineDye.cpp`); these drive them
through `IMP.bff.command_line_main`, the function the program's `main` calls.
"""
import os

import pytest

import IMP.bff
from IMP.bff import get_structure_dir

pytestmark = pytest.mark.skipif(
    IMP.bff.get_build() == "core",
    reason="the dye commands are in the IMP connection layer")


def dye(*words):
    return IMP.bff.command_line_main(["dye"] + [str(w) for w in words])


@pytest.fixture(scope="module")
def kinetic_lib(tmp_path_factory):
    """Generate a small kinetic library for testing other commands.

    build-lib reads every MOL2 in inputs/structures; very few steps keep it
    quick."""
    lib_dir = tmp_path_factory.mktemp("data") / "libs"
    dye("build-lib", "--n-steps", "50", "--cluster-threshold", "2.0",
        "--output-dir", lib_dir)
    for f in sorted(os.listdir(lib_dir)):
        if f.endswith(".rmf3"):
            return os.path.join(lib_dir, f)
    return None


def test_cli_help(capfd):
    assert dye("--help") == 0
    assert "label-fusion" in capfd.readouterr().out


def test_cli_label_pdb(tmp_path):
    out_pdb = tmp_path / "test_label.pdb"
    assert dye("label", get_structure_dir("1DG3.pdb"), "--residue", "481",
               "--dye", "Alexa488", "--linker", "C1R", "--output", out_pdb) == 0
    assert out_pdb.exists()


def test_cli_analyze_tc(kinetic_lib, capfd):
    if not kinetic_lib:
        pytest.skip("No kinetic library generated")
    capfd.readouterr()
    assert dye("analyze-tc", kinetic_lib) == 0
    assert "Slowest TC" in capfd.readouterr().out


def test_cli_reconstruct(kinetic_lib, tmp_path):
    if not kinetic_lib:
        pytest.skip("No kinetic library generated")
    out_rmf = tmp_path / "recon.rmf3"
    assert dye("reconstruct", "--lib-rmf", kinetic_lib, "--n-frames", "10",
               "--output-rmf", out_rmf) == 0
    assert out_rmf.exists()


def test_cli_sample_rotamer(tmp_path):
    out_rmf = tmp_path / "test_rot.rmf3"
    assert dye("sample-rotamer", "--protein-pdb", get_structure_dir("1DG3.pdb"),
               "--residue", "481", "--dye", "Alexa488", "--n-samples", "5",
               "--output-rmf", out_rmf) == 0
    assert out_rmf.exists()


def test_cli_label_fp_dual(tmp_path):
    out_pdb = tmp_path / "dual_fp.pdb"
    assert dye("label-fp", get_structure_dir("1DG3.pdb"), "--site", "6:eGFP",
               "--site", "583:mCherry", "--output", out_pdb) == 0
    assert out_pdb.exists()


# IMP runs every .py under test/ as a standalone script, and a file of bare
# pytest functions would import cleanly and exit 0 -- reporting success without
# running a single assertion. Hand the file to pytest explicitly so a failure
# here is a failure in ctest.
if __name__ == "__main__":
    import sys
    try:
        import pytest
    except ImportError:
        print("pytest not installed; skipping", __file__)
        sys.exit(0)
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
