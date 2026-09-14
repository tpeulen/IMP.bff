"""`imp_bff simulate`, run as the program runs, on a force-field system built here.

The runner is C++ (`include/ProbeSystemSimulation.h`, the sub
`src/imp/CommandLineSimulate.cpp`); it used to be the Python `simulate` of
`bin/imp_bff`. Each mode runs in a subprocess, not through
`IMP.bff.command_line_main` in this process: the program sets IMP's check
level and seeds IMP's random number generator, and neither may leak into the
rest of the suite. The subprocess calls the same dispatcher the executable's
`main` calls, so the test does not depend on the executable being on PATH.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import IMP.bff
from IMP.bff import get_structure_dir

#: The compiled command line, in a fresh interpreter.
DISPATCH = "import sys, IMP.bff; sys.exit(IMP.bff.command_line_main(sys.argv[1:]))"


@unittest.skipIf(IMP.bff.get_build() == "core",
                 "simulate needs the IMP connection layer")
class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the CX4+atto655 force-field system into a temporary directory
        # instead of expecting a pre-built output/systems/*.system.cif: the
        # test used to skip everywhere the build-system task had not run.
        from IMP.bff import write_probe_forcefield_cif
        from IMP.bff import create_probe_protein_system
        from IMP.bff import get_template_dir
        cls._tmp = tempfile.TemporaryDirectory()
        system = create_probe_protein_system(
            str(get_structure_dir("cx4.mol2")),
            str(get_structure_dir("atto655.mol2")),
            "CX4",
            "atto655",
            protein_template=str(get_template_dir("cx4.template.cif")),
            probe_template=str(get_template_dir("atto655.template.cif")),
        )
        cls.system_cif = os.path.join(cls._tmp.name, "cx4_atto655.system.cif")
        write_probe_forcefield_cif(cls.system_cif, system)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_sim(self, args):
        cmd = [sys.executable, "-c", DISPATCH, "simulate"] + args
        return subprocess.run(cmd, env=os.environ.copy(), capture_output=True,
                              text=True, timeout=120)

    def assert_run_dir(self, tmpdir, initial=True):
        run_dir = os.path.join(tmpdir, "CX4_atto655_imp")
        self.assertTrue(os.path.isdir(run_dir))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "rmfs", "0.rmf3")))
        self.assertEqual(os.path.exists(os.path.join(run_dir, "initial.0.rmf3")), initial)
        self.assertTrue(os.path.exists(os.path.join(run_dir, "system.cif")))
        # the system copy beside the output root, since no `trajs` ancestor
        self.assertTrue(os.path.exists(
            os.path.join(tmpdir, "systems", "CX4_atto655.system.cif")))
        with open(os.path.join(run_dir, "stat.0.out")) as fh:
            lines = fh.read().splitlines()
        self.assertEqual(lines[0], "frame\tscore\tkinetic_energy")
        return lines

    def test_hybrid_md_mc_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_sim([
                "--system-cif", self.system_cif,
                "--output-root", tmpdir,
                "--sampling-mode", "hybrid_md_mc",
                "--md-steps", "10",
                "--write-every", "5",
                "--com-pull-k", "1.0",
            ])
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            lines = self.assert_run_dir(tmpdir)
            # init plus 10 // 5 frames
            self.assertEqual([l.split("\t")[0] for l in lines[1:]], ["init", "0", "1"])
            self.assertIn("Alternating RB-MC/MD: frames=2, mc_steps=8, md_steps=5",
                          result.stdout)
            self.assertIn("mcAcc=", result.stdout)
            # the pull adds one restraint to the set simple_md builds
            self.assertIn("Restraints: total=967 softsphere=1 go=269", result.stdout)

    def test_multi_restart_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_sim([
                "--system-cif", self.system_cif,
                "--output-root", tmpdir,
                "--sampling-mode", "multi_restart",
                "--n-restarts", "2",
                "--md-steps", "10",
                "--write-every", "5",
            ])
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            lines = self.assert_run_dir(tmpdir, initial=False)
            # one row per restart, no init row
            self.assertEqual([l.split("\t")[0] for l in lines[1:]], ["0", "1"])
            self.assertIn("Meta-sampling complete.", result.stdout)

    def test_analyze_trajectories_reads_what_simulate_wrote(self):
        """`imp_bff analyze-trajectories`, end to end on a `simulate` run: the
        region densities, profiles and the axis definition. The Python command
        this replaces failed on its first template (`.get` on a SWIG value)."""
        template = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input",
                                "cgprobe", "atto655_regions.template.cif")
        with tempfile.TemporaryDirectory() as tmpdir:
            traj = os.path.join(tmpdir, "traj")
            result = self.run_sim([
                "--system-cif", self.system_cif, "--output-root", traj,
                "--sampling-mode", "simple_md", "--md-steps", "10", "--write-every", "5",
            ])
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            out = os.path.join(tmpdir, "analysis")
            analysed = subprocess.run(
                [sys.executable, "-c", DISPATCH, "analyze-trajectories",
                 "--traj-root", traj, "--output-dir", out, "--mobile", "atto655",
                 "--mobile-template-cif", template, "--axis-element", "S"],
                env=os.environ.copy(), capture_output=True, text=True, timeout=300)
            self.assertEqual(analysed.returncode, 0, msg=analysed.stderr + analysed.stdout)
            written = os.path.join(out, "atto655")
            for name in ("axis_z_profile_regions.csv", "axis_xy_profile_regions.csv",
                         "axis_mobile_vs_fixed_orientation.csv",
                         "fixed_axis_definition.json", "occupancy_linker.mrc",
                         "occupancy_top.mrc", "occupancy_middle.mrc",
                         "occupancy_bottom.mrc"):
                self.assertTrue(os.path.isfile(os.path.join(written, name)), name)

    def test_simple_md_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_sim([
                "--system-cif", self.system_cif,
                "--output-root", tmpdir,
                "--sampling-mode", "simple_md",
                "--md-steps", "10",
                "--write-every", "5",
            ])
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            lines = self.assert_run_dir(tmpdir)
            self.assertEqual([l.split("\t")[0] for l in lines[1:]], ["init", "0", "1"])
            self.assertIn("Inferred mobile group: atto655_all", result.stdout)
            self.assertIn("DOF: movable=35 / 138 (freeze_mobile_rings=True, "
                          "fixed_flex_mode=static)", result.stdout)
            self.assertIn("Restraints: total=966 softsphere=1 go=269", result.stdout)

    def test_a_seed_repeats_a_run(self):
        """`--init-seed` seeds IMP's generator too, so the same seed gives the
        same trajectory -- the Python program drew its MD velocities from an
        unseeded generator and could not repeat itself."""
        stats = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = self.run_sim([
                    "--system-cif", self.system_cif, "--output-root", tmpdir,
                    "--sampling-mode", "hybrid_md_mc", "--md-steps", "10",
                    "--write-every", "5", "--init-seed", "3"])
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                with open(os.path.join(tmpdir, "CX4_atto655_imp", "stat.0.out")) as fh:
                    stats.append(fh.read())
        self.assertEqual(stats[0], stats[1])

    def test_flex_mode_runs(self):
        """`--fixed-flex-mode flex` crashed the Python runner (`.get` on a
        SWIG map); the fixed component here has no `_flex` group, so nothing
        is released and the run is the static one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_sim([
                "--system-cif", self.system_cif, "--output-root", tmpdir,
                "--sampling-mode", "simple_md", "--md-steps", "10",
                "--write-every", "5", "--fixed-flex-mode", "flex"])
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("fixed_flex_movable=0", result.stdout)

    def test_no_system_is_a_failure_not_a_crash(self):
        result = self.run_sim(["--sampling-mode", "simple_md"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("Provide --system-cif or --systems-dir", result.stderr)


if __name__ == "__main__":
    unittest.main()
