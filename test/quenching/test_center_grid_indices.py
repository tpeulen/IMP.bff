"""``center_grid_indices`` is callable from Python and returns its four outputs.

Its four ``std::vector<int>&`` outputs were exposed as *inputs* by SWIG, so the
function demanded nine arguments and refused any list for the sixth: no Python
caller could use it (found porting QuEst into ChiSurf, 2026-10-12). The voxel
map is the one ``slow_factor_grid`` stamps with, so these pins tie the two.
"""

import unittest

import numpy as np

import IMP
import IMP.test
import IMP.bff


class Tests(IMP.test.TestCase):

    def test_returns_four_int_arrays(self):
        rs = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, -1.0], [-0.26, 0.74, 0.5]])
        r0 = np.zeros(3)
        dg, ng = 0.5, 9
        radius = np.array([1.0, 1.4, 0.2])
        out = IMP.bff.center_grid_indices(rs.ravel(), r0, dg, ng, radius)
        self.assertEqual(len(out), 4)
        ix, iy, iz, ridx = (np.asarray(v) for v in out)
        centre = IMP.bff.grid_center_index(ng)
        expected = np.floor((rs - r0) / dg).astype(int) + centre
        np.testing.assert_array_equal(ix, expected[:, 0])
        np.testing.assert_array_equal(iy, expected[:, 1])
        np.testing.assert_array_equal(iz, expected[:, 2])
        np.testing.assert_array_equal(ridx, (radius / dg).astype(int))

    def test_floor_not_truncation(self):
        # A centre a quarter voxel below the anchor lands one voxel below it.
        ix, _, _, _ = IMP.bff.center_grid_indices(
            [-0.125, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5, 8, [1.0])
        self.assertEqual(int(ix[0]), IMP.bff.grid_center_index(8) - 1)

    def test_empty_input(self):
        out = IMP.bff.center_grid_indices([], [0.0, 0.0, 0.0], 0.5, 8, [])
        self.assertTrue(all(len(v) == 0 for v in out))


if __name__ == "__main__":
    IMP.test.main()
