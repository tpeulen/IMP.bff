"""`DensityGrid`: the lattice `PathMap` is built on, and the AV output it must not move.

`PathMap` derived from `IMP::em::SampledDensityMap` and is now built on
`IMP.bff.DensityGrid`, a lattice of this module's own. That swap was made
under one rule -- **the numbers do not change** -- and this file is where the
rule is enforced rather than remembered.

Two kinds of test, and they catch different things.

**The golden AV records** pin `get_av_from_structure` on three structures
at three grid steps, as SHA-256 of the density and of the point cloud, taken
from the `IMP.em`-backed build before the swap. They are what a change to the
sampling, the bounding box or the voxel/location arithmetic shows up in. They
did not, on their own, catch four regressions the swap introduced -- which is
what the second kind is for.

**The unit tests** each pin one of those four, at the level it broke:

* `set_origin` must rebuild the coordinate caches, not only invalidate them.
  They are `unique_ptr`s, so the failure is a null dereference, not a wrong
  number, and it surfaced only in the lattice fast path.
* `PathMap` must keep the methods it inherits. SWIG gives a class the methods
  of a base it has *seen*; when the base was not in the interface `PathMap`
  wrapped, imported, and had no `get_number_of_voxels`.
* The header is float-backed. `IMP::em::DensityHeader` stores spacing and
  origin as `float`; holding them as `double` computed voxel locations *more*
  accurately and moved AV mean positions in the eighth significant figure.
* Voxel-index arithmetic is IMP's, promotion for promotion. Two extra float
  roundings in `get_dim_index_by_location` pick a different voxel at a
  half-voxel boundary, and the legacy lattice search seeds itself there.
"""

import hashlib
import os
import unittest

import numpy as np

import IMP.algebra
import IMP.bff as bff

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha(a):
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


# (label, pdb, chain, resseq, atom, linker_length, linker_width, radius1)
_SITES = [
    ("T4L-132", "examples/structure/T4L/3GUN.pdb", "", 132, "CB", 22.0, 3.5, 3.5),
    ("T4L-150", "examples/structure/T4L/3GUN.pdb", "", 150, "CB", 22.0, 2.5, 3.5),
    ("GBP-400", "examples/structure/GBP/mGBP2A.pdb", "A", 400, "CB", 22.0, 4.5, 3.5),
]

# Recorded 2026-09-07 from the IMP.em-backed build, before PathMap moved onto
# DensityGrid; re-recorded 2026-10-11 twice: when the linker's first step began
# to reach one width plus one grid step out of the anchor (FRETraj's rule,
# T4L-132@0.5 331 156 -> 439 804 point values), and the same day when the
# owner settled the AV defaults on FPS's ("av defaults should be fps/olga
# defaults"): the derived clearance is FPS's seed, half the linker width (was
# max(1.5, W/2 + grid/2)), and the first step is FPS's -- a hop of
# LinkSearchNodes = 3 voxels (a 7x7x7 cube) from every voxel of that seed
# (PathMap::set_source_first_step). T4L-132@0.5 is now 371 548, +12 % over the
# stencil alone rather than +33 %; the explicit parameters here did not change.
# key -> (n_density, density_sha, n_points, points_sha, ng).
_GOLDEN = {
    "T4L-132@1.5": (29791, "37364ccd7f6dc91a", 15432, "af94c6ef17bd5374", 31),
    "T4L-132@1.0": (91125, "a9363ac57700ef0d", 55332, "5aee0d3d99199897", 45),
    "T4L-132@0.5": (704969, "66b5d1f6044217fe", 371548, "b04c129e437c3b86", 89),
    "T4L-150@1.5": (29791, "d772577956b42772", 10772, "6446b0d427beded8", 31),
    "T4L-150@1.0": (91125, "f465036cf24d5a1c", 31652, "324411a52b4ba2e5", 45),
    "T4L-150@0.5": (704969, "14fdaa1992ea4605", 260244, "2c6a18daeb677f81", 89),
    "GBP-400@1.5": (29791, "175f63dc9c965819", 14376, "45e8bc0c8b4c94a6", 31),
    "GBP-400@1.0": (91125, "6025d1f861ebc1e8", 53624, "016cc568fa8cddcb", 45),
    "GBP-400@0.5": (704969, "a8c9e5e677eb6400", 388276, "ac6a4dc685382727", 89),
}


class TestGoldenAV(unittest.TestCase):
    """The AV solver's output, bit for bit, across the structures shipped."""

    def test_av_records_are_unchanged(self):
        for label, rel, chain, resseq, atom, ll, lw, r1 in _SITES:
            pdb = os.path.join(_ROOT, rel)
            if not os.path.exists(pdb):
                self.skipTest("no %s in this checkout" % rel)
            for step in (1.5, 1.0, 0.5):
                key = "%s@%.1f" % (label, step)
                with self.subTest(key=key):
                    av = bff.get_av_from_structure(
                        pdb, chain, resseq, atom, ll, lw, r1, 0.0, 0.0, step)
                    d = np.asarray(av.get_density())
                    p = np.asarray(av.get_points())
                    n_d, sha_d, n_p, sha_p, ng = _GOLDEN[key]
                    self.assertEqual(int(av.get_ng()), ng)
                    self.assertEqual(d.size, n_d)
                    self.assertEqual(p.size, n_p)
                    self.assertEqual(_sha(d), sha_d, "density moved: " + key)
                    self.assertEqual(_sha(p), sha_p, "points moved: " + key)


def _grid(n=10, spacing=1.0, origin=(0.0, 0.0, 0.0)):
    g = bff.DensityGrid("g")
    h = g.get_header_writable()
    h.set_spacing(spacing)
    h.update_map_dimensions(n, n, n)
    g.resize(g.get_number_of_voxels())
    g.set_origin(IMP.algebra.Vector3D(*origin))
    return g


def _occupied(g):
    return sum(g.get_value(i) for i in range(g.get_number_of_voxels()))


class TestSampling(unittest.TestCase):
    """IMP's `BINARIZED_SPHERE`, reproduced: strict, accumulating."""

    def test_sphere_of_radius_two_marks_the_27_interior_points(self):
        # Lattice points strictly inside r=2 about a lattice point: every
        # (dx,dy,dz) in {-1,0,1}^3 has dx^2+dy^2+dz^2 <= 3 < 4, and any
        # coordinate of 2 gives >= 4. So 27, and nothing else.
        g = _grid()
        g.set_spheres([bff.GridSphere(IMP.algebra.Vector3D(5, 5, 5), 2.0)])
        g.resample()
        self.assertEqual(_occupied(g), 27)

    def test_the_boundary_is_strict(self):
        # r == 1 exactly: the six neighbours sit at distance 1, and 1 < 1 is
        # false, so only the centre is inside. IMP's kernel is `<`, not `<=`.
        g = _grid()
        g.set_spheres([bff.GridSphere(IMP.algebra.Vector3D(5, 5, 5), 1.0)])
        g.resample()
        self.assertEqual(_occupied(g), 1)

    def test_overlap_accumulates(self):
        # Two identical spheres: every voxel inside reads 2, not 1. IMP added
        # (`data[ivox] += value`) rather than assigned, and nothing reading the
        # map cares -- but the threshold that follows is `>`, so a change here
        # to assignment would still pass a naive occupancy count.
        g = _grid()
        s = bff.GridSphere(IMP.algebra.Vector3D(5, 5, 5), 2.0)
        g.set_spheres([s, s])
        g.resample()
        values = {g.get_value(i) for i in range(g.get_number_of_voxels())}
        self.assertEqual(values, {0.0, 2.0})

    def test_zero_radius_blocks_nothing(self):
        g = _grid()
        g.set_spheres([bff.GridSphere(IMP.algebra.Vector3D(5, 5, 5), 0.0)])
        g.resample()
        self.assertEqual(_occupied(g), 0)


class TestIndexing(unittest.TestCase):
    """Voxel <-> index <-> location, with IMP's rounding."""

    def test_index_roundtrip(self):
        g = _grid(n=7)
        for ix, iy, iz in ((0, 0, 0), (6, 6, 6), (3, 1, 5)):
            v = g.xyz_ind2voxel(ix, iy, iz)
            self.assertEqual([g.get_dim_index_by_voxel(v, d) for d in range(3)],
                             [ix, iy, iz])

    def test_location_roundtrip(self):
        g = _grid(n=7, spacing=1.5, origin=(-3.0, 2.0, 0.5))
        for v in (0, 17, 342):
            loc = g.get_location_by_voxel(v)
            self.assertEqual(g.get_voxel_by_location(loc), v)

    def test_half_voxel_rounds_up_as_imp_did(self):
        # IMP: floor(0.5 + (loc - origin) / spacing). Half a voxel past the
        # origin is index 1; just under half is index 0. The subtraction and
        # division are done in double after the float inputs, which is what
        # keeps a coordinate a hair under the boundary from flipping.
        g = _grid(n=7, spacing=2.0, origin=(0.0, 0.0, 0.0))
        self.assertEqual(g.get_dim_index_by_location(1.0, 0), 1)
        self.assertEqual(g.get_dim_index_by_location(0.98, 0), 0)
        self.assertEqual(g.get_dim_index_by_location(2.99, 0), 1)
        self.assertEqual(g.get_dim_index_by_location(3.0, 0), 2)


class TestTheFourRegressions(unittest.TestCase):
    """One test per bug the swap shipped past the golden records."""

    def test_set_origin_rebuilds_the_caches(self):
        # Invalidate-only left `x_loc_` null; reading a location afterwards
        # was a crash. The lattice fast path moves the origin and reads
        # without asking for a recompute in between.
        g = _grid(n=5)
        g.set_origin(IMP.algebra.Vector3D(10.0, 20.0, 30.0))
        loc = g.get_location_by_voxel(0)
        np.testing.assert_array_equal(np.asarray(list(loc)), [10.0, 20.0, 30.0])

    def test_pathmap_keeps_its_inherited_api(self):
        for name in ("get_number_of_voxels", "get_value", "get_spacing",
                     "get_location_by_voxel", "get_dim_index_by_voxel",
                     "get_voxel_by_location", "xyz_ind2voxel", "get_header",
                     "resample"):
            self.assertTrue(hasattr(bff.PathMap, name), name)

    def test_header_is_float_backed(self):
        # 0.1 is not representable; a float-backed header hands back the
        # float32 rounding, a double-backed one would hand back 0.1.
        h = _grid().get_header_writable()
        h.set_spacing(0.1)
        self.assertEqual(h.get_spacing(), float(np.float32(0.1)))
        self.assertNotEqual(h.get_spacing(), 0.1)

    def test_pathmapheader_carries_a_gridheader(self):
        # The em header it used to carry serialized ~40 EM fields; this one
        # serializes eleven. `get_density_header` is the by-pointer accessor
        # IMP's was, so the extent is readable from Python.
        pmh = bff.PathMapHeader()
        h = pmh.get_density_header()
        self.assertIsInstance(h, bff.GridHeader)


if __name__ == "__main__":
    unittest.main()
