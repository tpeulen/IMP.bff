"""The trajectory density analysis: its axes and its tables.

`imp_bff analyze-trajectories` was Python inside the program `bin/imp_bff`;
its kernels are C++ now -- `IMP.bff.probe_fixed_axis` and
`IMP.bff.probe_long_axis` over particles (ProbeTrajectoryDensity.h), and the
array kernels `point_cloud_principal_axis`, `write_radial_histogram`,
`write_binned_profile` (TrajectoryAnalysis.h).
"""

import os
import tempfile
import unittest

import IMP
import IMP.algebra
import IMP.core

import IMP.bff


def _particles(model, coordinates):
    out = []
    for x, y, z in coordinates:
        p = IMP.Particle(model)
        IMP.core.XYZ.setup_particle(p).set_coordinates(IMP.algebra.Vector3D(x, y, z))
        out.append(p)
    return out


class TestAnalysis(unittest.TestCase):

    def test_compute_fixed_axis(self):
        """The minimal-variance axis of a flat ring in the XY plane is Z."""
        model = IMP.Model()
        ring = _particles(model, [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)])
        result = IMP.bff.probe_fixed_axis(ring)
        center, axis = result[:3], result[3:]
        for c in center:
            self.assertAlmostEqual(c, 0.0)
        self.assertAlmostEqual(abs(axis[2]), 1.0)
        self.assertAlmostEqual(axis[0], 0.0)
        self.assertAlmostEqual(axis[1], 0.0)

    def test_fixed_axis_follows_a_reference(self):
        model = IMP.Model()
        ring = _particles(model, [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)])
        up = IMP.bff.probe_fixed_axis(ring, ref_axis=[0, 0, 1])
        down = IMP.bff.probe_fixed_axis(ring, ref_axis=[0, 0, -1])
        self.assertAlmostEqual(up[5], 1.0)
        self.assertAlmostEqual(down[5], -1.0)

    def test_compute_long_axis(self):
        """The maximal-variance axis of two points on X is X."""
        model = IMP.Model()
        rod = _particles(model, [(-5, 0, 0), (5, 0, 0)])
        axis = IMP.bff.probe_long_axis(rod)
        self.assertAlmostEqual(abs(axis[0]), 1.0)
        self.assertAlmostEqual(axis[1], 0.0)
        self.assertAlmostEqual(axis[2], 0.0)
        self.assertEqual(len(IMP.bff.probe_long_axis(rod[:1])), 0)

    def test_principal_axis_of_an_array(self):
        result = IMP.bff.point_cloud_principal_axis(
            [1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0], True)
        self.assertAlmostEqual(abs(result[5]), 1.0)

    def test_write_radial_histogram(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "hist.csv")
            IMP.bff.write_radial_histogram([1.0, 2.0, 3.0, 1.5], csv_path, 1.0)
            with open(csv_path) as f:
                lines = f.read().splitlines()
            # bins from zero to the largest distance: [0,1) [1,2) [2,3]
            self.assertEqual(lines, ["bin_start_A,bin_end_A,count",
                                     "0.000,1.000,0",
                                     "1.000,2.000,2",
                                     "2.000,3.000,2"])

    def test_write_binned_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "profile.csv")
            IMP.bff.write_binned_profile(csv_path, ["a", "b"], [[0.1, 0.9], [0.4]], 0.5)
            with open(csv_path) as f:
                lines = f.read().splitlines()
            self.assertEqual(lines[0], "bin_start_A,bin_end_A,center_A,a_count,b_count,"
                                       "a_density,b_density")
            self.assertEqual(lines[1], "0.1000,0.6000,0.3500,1,1,1.00000000,2.00000000")
            self.assertEqual(lines[2], "0.6000,1.1000,0.8500,1,0,1.00000000,0.00000000")


if __name__ == "__main__":
    unittest.main()
