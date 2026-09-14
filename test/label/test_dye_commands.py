"""The `dye` commands, run.

Every one of these was dead, and none of them had a test -- which is the only
reason a command can be dead. What each was, when it was Python:

* `sample-dof-walk` imported `LinkerSampler`, a name from before the linker
  sampler became C++; then it read `resolve_probe_site`'s answer as a dict;
  then, once running, it rejected every proposal, because it counted the bond
  between the probe and its neighbouring residues as a clash.
* `sample-langevin` passed `timestep_fs=None` into a `double`, and asked
  `run()` to write an RMF -- which it stopped doing when the sampler became
  C++, because writing files is a program's business.
* `label-fusion` used `sys.exit` in a module that never imported `sys`.

They are compiled subcommands of `imp_bff` now (`src/imp/CommandLineDye.cpp`),
driven here through `IMP.bff.command_line_main`, the function the program's
`main` calls. Output is written from C++ at the file descriptor, hence capfd.

They are smoke tests: a few steps each, checking the command runs and writes
what it says it wrote. What the samplers *compute* is pinned elsewhere
(`test/cgprobe/test_langevin_sampler.py`).
"""

import pytest

import IMP.bff

PDB = IMP.bff.get_example_path("structure/T4L/3GUN.pdb")

needs_imp_layer = pytest.mark.skipif(
    IMP.bff.get_build() == "core",
    reason="the dye commands are in the IMP connection layer")


def dye(capfd, *words):
    """`imp_bff dye <words>`: the exit code and what it printed to stdout."""
    capfd.readouterr()
    code = IMP.bff.command_line_main(["dye"] + [str(w) for w in words])
    captured = capfd.readouterr()
    return code, captured.out + captured.err


@needs_imp_layer
def test_sample_dof_walk_accepts_something(capfd, tmp_path):
    """A walk that accepts nothing is not a sampler.

    The probe is placed *bonded into* the site, so its first atoms sit a bond
    length from residues i-1 and i+1; counting those as clashes rejected every
    proposal. They are excluded now, and a proposal that does not make the
    clash count worse is accepted -- otherwise a walk that starts inside the
    protein could never leave.
    """
    out = tmp_path / "walk.rmf3"
    code, output = dye(capfd,
        "sample-dof-walk", "--protein-pdb", PDB, "--chain", "A",
        "--residue", "132", "--dye", "alexa488", "--n-steps", "40",
        "--output-rmf", out)
    assert code == 0, output
    assert out.is_file()
    line = [l for l in output.splitlines() if "Finished" in l][-1]
    accepted = int(line.split()[1].split("/")[0])
    assert accepted > 0, line


@needs_imp_layer
def test_sample_dof_walk_repeats_the_python_run(capfd, tmp_path):
    """The walk draws CPython's numbers (`random.seed`, `random.gauss`), so the
    seeded run the Python program made is the run the compiled one makes:
    20 of 40 proposals accepted at T4L 132 with seed 42."""
    code, output = dye(capfd,
        "sample-dof-walk", "--protein-pdb", PDB, "--chain", "A",
        "--residue", "132", "--dye", "alexa488", "--n-steps", "40",
        "--output-rmf", tmp_path / "walk.rmf3")
    assert code == 0, output
    assert "Finished: 20/40 accepted, 0 clashing atoms left." in output


@needs_imp_layer
def test_sample_langevin_runs_and_writes_frames(capfd, tmp_path):
    out = tmp_path / "md.rmf3"
    code, output = dye(capfd,
        "sample-langevin", "--protein-pdb", PDB, "--chain", "A",
        "--residue", "132", "--dye", "alexa488", "--n-steps", "100",
        "--write-every", "50", "--minimize-steps", "10",
        "--output-rmf", out)
    assert code == 0, output
    assert out.is_file()

    import RMF
    fh = RMF.open_rmf_file_read_only(str(out))
    assert fh.get_number_of_frames() == 2


@needs_imp_layer
def test_the_timestep_is_chosen_when_it_is_not_given(capfd, tmp_path):
    """A negative timestep means "one per integrator": 2 fs for md, 0.5 for
    bd. It used to be `None`, which is not a `double`."""
    code, output = dye(capfd,
        "sample-langevin", "--protein-pdb", PDB, "--chain", "A",
        "--residue", "132", "--dye", "alexa488", "--integrator", "bd",
        "--n-steps", "50", "--write-every", "50", "--minimize-steps", "0",
        "--output-rmf", tmp_path / "bd.rmf3")
    assert code == 0, output
    assert "0.5 fs" in output, output


@needs_imp_layer
def test_label_fusion_runs(capfd, tmp_path):
    """`sys.exit` in a module that never imported `sys` was a `NameError` on
    the way out, not an exit."""
    code, output = dye(capfd,
        "label-fusion", PDB, "--chain", "A", "--output", tmp_path / "fusion.pdb")
    assert code == 0, output


@needs_imp_layer
def test_label_fp_attaches_a_fluorescent_protein(capfd, tmp_path):
    out = tmp_path / "fp.pdb"
    code, output = dye(capfd,
        "label-fp", PDB, "--site", "60:eGFP:A:C", "--output", out)
    assert code == 0, output
    assert "Attaching eGFP (anchor C) to A:60..." in output
    assert out.is_file()


@needs_imp_layer
def test_reconstruct_refuses_a_library_without_jump_counts(capfd, tmp_path):
    """The shipped rotamer templates carry conformers and no transitions, so
    there is no Markov walk to draw; the kernel says so and the command exits 1."""
    lib = IMP.bff.get_template_dir("rotamer/A48_C1R.rmf3")
    code, output = dye(capfd,
        "reconstruct", "--lib-rmf", lib, "--n-frames", "5",
        "--output-rmf", tmp_path / "recon.rmf3")
    assert code == 1
    assert "0 counts" in output


def test_rotamer_r0_reports_a_forster_radius(capfd):
    """`rotamer r0` is compiled now (`imp_bff rotamer r0`)."""
    capfd.readouterr()
    assert IMP.bff.command_line_main([
        "rotamer", "r0", "--donor", "AlexaFluor 488", "--acceptor", "AlexaFluor 594",
        "--k2", "0.6667"]) == 0
    out = capfd.readouterr().out
    value = float(out.split()[0])
    assert 40.0 < value < 70.0, out
