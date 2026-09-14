"""The compiled program itself: `bin/imp_bff.cpp`, run as a process.

Every other command-line test drives `IMP.bff.command_line_main` in-process,
which is the dispatcher but not the program: it cannot see a `main` that was
not built, an executable that does not link, a library it cannot load at run
time, an exit code lost on the way out of the process, or output that never
reaches the descriptor. These tests run the executable the build produced, as a
user does.

Where it is:

* the IMP module build and the standalone build both put it at
  `<build>/bin/imp_bff`, beside `<build>/{lib,python}/IMP/bff`;
* an installed package puts it in `$PREFIX/bin` (IMP's `IMP_BIN_DIR` says so
  inside a build environment);
* the wheel installs no executable -- its `imp_bff` is the console script over
  the same dispatcher -- and that is the one case skipped. A build tree or an
  `IMP_BIN_DIR` that promises a binary and does not have one fails.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import IMP.bff

HERE = Path(__file__).resolve().parent
FPS_XYZ = HERE / "input" / "fps" / "p_1bp_D.xyz"
T4L_PDB = IMP.bff.get_example_path("structure/T4L/3GUN.pdb")

HAS_IMP_LAYER = IMP.bff.get_build() != "core"

#: The top-level subcommands and the nested ones, per build. A group that
#: silently stops being registered is exactly what this list catches.
CORE_GROUPS = {
    "help": [],
    "probe-pdb2cif": [],
    "traj2bcif": [],
    "traj2drot": [],
    "labelizer": [],
    "fps-distance": [],
    "potentials2pto": [],
    "rmsd": [],
}
LAYER_GROUPS = {
    "fps": ["score", "dock", "refine", "screen", "convert", "project"],
    "fps-av": [],
    "fps-export": ["errors", "table", "screen"],
    "av-export": [],
    "openmm": [],
    "build-system": [],
    "select-pairs": [],
    "rotamer": ["predict", "r0"],
    "dye": ["label", "build-lib", "analyze-tc", "reconstruct", "sample-rotamer",
            "sample-dof-walk", "sample-langevin", "label-fp", "label-fusion"],
    "simulate": [],
    "dock": [],
    "dock-errors": [],
    "flexfit": [],
    "analyze-trajectories": [],
    "av-vs-rotamer": [],
}
NESTED = {"fps project": ["init", "show", "convert"]}


def _is_native(path):
    """A compiled executable, not a script wrapper of the same name."""
    try:
        with open(path, "rb") as handle:
            magic = handle.read(4)
    except OSError:
        return False
    return magic[:2] == b"MZ" or magic == b"\x7fELF" or magic in (
        b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe")


def _find_executable():
    """(path or None, whether a build or environment promised one)."""
    name = "imp_bff.exe" if sys.platform == "win32" else "imp_bff"
    candidates, promised = [], False
    if os.environ.get("IMP_BIN_DIR"):
        candidates.append(Path(os.environ["IMP_BIN_DIR"]) / name)
        promised = True
    module_dir = Path(IMP.bff.__file__).resolve().parent          # .../IMP/bff
    build_bin = module_dir.parents[2] / "bin"                     # <build>/bin
    if (module_dir.parents[2] / "CMakeCache.txt").exists():
        candidates.append(build_bin / name)
        promised = True
    candidates.append(Path(sys.prefix) / "bin" / name)
    candidates.append(Path(sys.prefix) / "Library" / "bin" / name)
    on_path = shutil.which(name)
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file() and _is_native(candidate):
            return str(candidate), promised
    return None, promised


@pytest.fixture(scope="module")
def exe():
    path, promised = _find_executable()
    if path is None:
        if promised:
            pytest.fail("this build promises a compiled imp_bff (build tree or "
                        "IMP_BIN_DIR) and there is none: bin/imp_bff.cpp was not "
                        "built or not linked")
        pytest.skip("no compiled imp_bff here (the wheel ships the console "
                    "script instead)")
    return path


def run(exe, *words, cwd=None):
    return subprocess.run([exe] + [str(w) for w in words], capture_output=True,
                          text=True, cwd=cwd, timeout=600)


def expected_groups():
    groups = dict(CORE_GROUPS)
    if HAS_IMP_LAYER:
        groups.update(LAYER_GROUPS)
    return groups


# ---- the program is there and answers ---------------------------------------

def test_the_executable_is_a_compiled_program(exe):
    assert _is_native(exe)


def test_help_lists_exactly_the_registered_groups(exe):
    out = run(exe, "help")
    assert out.returncode == 0, out.stderr
    listed = set()
    for line in out.stdout.splitlines():
        if line.startswith("  ") and not line.startswith("   "):
            # "  probe-pdb2cif, pdb2cif   Convert ..." -> the name before the alias
            listed.add(line.split()[0].rstrip(","))
    assert listed == set(expected_groups())


@pytest.mark.parametrize("group", sorted(set(CORE_GROUPS) | set(LAYER_GROUPS)))
def test_every_group_answers_help(exe, group):
    if group not in expected_groups():
        pytest.skip("%s is in the IMP connection layer" % group)
    if group == "help":
        return
    out = run(exe, group, "--help")
    assert out.returncode == 0, out.stderr
    assert group in out.stdout
    for sub in expected_groups()[group]:
        assert sub in out.stdout, "%s --help does not list %s" % (group, sub)


def _nested():
    pairs = []
    for group, subs in sorted(LAYER_GROUPS.items()):
        pairs += [(group, sub) for sub in subs]
    for group, subs in sorted(NESTED.items()):
        pairs += [(group, sub) for sub in subs]
    return pairs


@pytest.mark.skipif(not HAS_IMP_LAYER, reason="nested groups are in the IMP layer")
@pytest.mark.parametrize("group,sub", _nested())
def test_every_nested_subcommand_answers_help(exe, group, sub):
    out = run(exe, *(group.split() + [sub, "--help"]))
    assert out.returncode == 0, out.stderr
    assert sub in out.stdout


# ---- exit codes survive the process boundary ---------------------------------

def test_no_subcommand_is_a_usage_error(exe):
    assert run(exe).returncode == 2


def test_an_unknown_subcommand_is_a_usage_error(exe):
    assert run(exe, "no-such-command").returncode == 2


def test_a_missing_argument_is_a_usage_error(exe):
    out = run(exe, "fps-distance", FPS_XYZ)
    assert out.returncode == 2
    assert "av2" in out.stderr + out.stdout


def test_a_failing_run_exits_1_and_says_which_command(exe, tmp_path):
    out = run(exe, "labelizer", tmp_path / "missing.pdb")
    assert out.returncode == 1
    assert out.stderr.startswith("imp_bff labelizer: ")


# ---- real runs, compared with the in-process dispatcher ----------------------

def test_fps_distance_through_the_program_matches_the_dispatcher(exe, capfd):
    out = run(exe, "fps-distance", FPS_XYZ, FPS_XYZ, "--json", "-n", "2000")
    assert out.returncode == 0, out.stderr
    capfd.readouterr()
    assert IMP.bff.command_line_main(
        ["fps-distance", str(FPS_XYZ), str(FPS_XYZ), "--json", "-n", "2000"]) == 0
    # in-process, the descriptor may also carry IMP log lines another test
    # left buffered; the report is the JSON object at the end
    captured = capfd.readouterr().out
    in_process = json.loads(captured[captured.index("{\n"):])
    assert json.loads(out.stdout) == in_process


def test_labelizer_writes_its_container_through_the_program(exe, tmp_path):
    src = tmp_path / "t4l.pdb"
    shutil.copy(T4L_PDB, src)
    out = run(exe, "labelizer", src, "--no-conservation", cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    container = tmp_path / "t4l.mmfdb.pto"
    assert container.is_file()
    assert len(IMP.bff.labelizer_read_pto_scores(str(container))) > 0
    shown = run(exe, "labelizer", container, "--show")
    assert shown.returncode == 0 and "best sites" in shown.stdout


def test_probe_pdb2cif_through_the_program(exe, tmp_path):
    pdb = tmp_path / "probe.pdb"
    IMP.bff.write_pdb([c for i in range(12) for c in (i * 1.4, 0.3 * (i % 2), 0.0)],
                      str(pdb), "A", "LYS")
    cif = tmp_path / "probe.cif"
    out = run(exe, "probe-pdb2cif", pdb, cif, "--probe-id", "LYP")
    assert out.returncode == 0, out.stderr
    assert "LYP" in cif.read_text()


def test_traj2drot_round_trips_through_the_program(exe, tmp_path):
    xyz = [c for frame in range(8) for i in range(6)
           for c in (i * 1.4 + 0.01 * frame, 0.02 * frame, 0.0)]
    src = tmp_path / "src.drot.pto"
    IMP.bff.write_probe_rotamer_drot(str(src), xyz, ["N", "CA", "C", "O", "CB", "SG"],
                                     ["N", "C", "C", "O", "C", "S"], ["LYS"] * 6,
                                     [1.0] * 8)
    dst = tmp_path / "dst.drot.pto"
    out = run(exe, "traj2drot", src, dst)
    assert out.returncode == 0, out.stderr
    assert "verified" in out.stdout
    assert IMP.bff.read_probe_rotamer_drot(str(dst)).n_rotamers == 8


@pytest.mark.skipif(not HAS_IMP_LAYER, reason="fps is in the IMP connection layer")
def test_fps_score_through_the_program_gives_the_pinned_score(exe):
    hiv = Path(IMP.bff.get_example_path("structure/HIV_RT"))
    out = run(exe, "fps", "score", "-p", hiv / "protein_1R0A.pdb", "-p", hiv / "dna.pdb",
              "-j", hiv / "hiv_rt.fps.json", "-c", "resolved", "--no-pairs")
    assert out.returncode == 0, out.stderr
    assert "score (chi2): 59.0404" in out.stdout


@pytest.mark.skipif(not HAS_IMP_LAYER, reason="fps-av is in the IMP connection layer")
def test_fps_av_through_the_program(exe, tmp_path):
    out = run(exe, "fps-av", "-p", T4L_PDB, "-c", "A", "-r", 44, "-d", "alexa488-long",
              "-o", tmp_path / "d44.xyz", "--json")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["n_voxels"] > 0
    bad = run(exe, "fps-av", "-p", T4L_PDB, "-r", 9999, "-o", tmp_path / "x.xyz")
    assert bad.returncode == 1
    assert bad.stderr.startswith("imp_bff fps-av: ")


@pytest.mark.skipif(not HAS_IMP_LAYER, reason="rotamer is in the IMP connection layer")
def test_rotamer_r0_through_the_program(exe):
    out = run(exe, "rotamer", "r0", "--donor", "AlexaFluor 488",
              "--acceptor", "AlexaFluor 594", "--k2", "0.684587")
    assert out.returncode == 0, out.stderr
    value, unit = out.stdout.split()[:2]
    assert float(value) == pytest.approx(57.12982, abs=1e-4)
    assert unit == "A"
