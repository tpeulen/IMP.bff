"""The attachment atom is never an obstacle, at any door.

Owner, 2026-10-12: "the attachment site must be excluded". FPS drops the
attachment atom -- the atom alone, not its residue's side chain -- before it
rasterises anything (`av_routines.cpp:67`, `if (i == atom_i) continue;`), for
the linker and the dye alike. This module does the same by subtracting the
atom's own sphere from the shared occupancy count (`drop_source_obstruction`),
which is exact only if the subtraction removes precisely the voxels the raster
added. These tests hold that: a volume computed with the attachment atom among
the obstacles is **identical, voxel for voxel**, to one computed without it.

The legacy anchoring (`space_fixed=False`) is deliberately not covered: it is
frozen on the pre-PRD-105 numbers and keeps the atom.
"""
import IMP
import IMP.atom
import IMP.bff
import IMP.core
import IMP.test
import numpy as np

PDB = IMP.bff.get_example_path("structure/T4L/3GUN.pdb")
SITES = (132, 44, 69)


def _xyzr():
    return np.asarray(IMP.bff.load_structure_with_vdw(PDB), dtype=np.float64).reshape(-1, 4)


def _row(xyzr, residue):
    src = np.asarray(IMP.bff.get_attachment_point(PDB, "A", residue, "CB"))
    hit = np.where(np.all(np.abs(xyzr[:, :3] - src) < 1e-8, axis=1))[0]
    assert len(hit) == 1, hit
    return int(hit[0]), src


def _decorator_points(residue, zero_attachment_radius):
    mdl = IMP.Model()
    hier = IMP.atom.read_pdb(PDB, mdl, IMP.atom.NonWaterNonHydrogenPDBSelector())
    sel = IMP.atom.Selection(hier, residue_index=residue,
                             atom_type=IMP.atom.AtomType("CB"))
    source = sel.get_selected_particles()[0]
    if zero_attachment_radius:
        IMP.core.XYZR(source).set_radius(0.0)
    p = IMP.Particle(mdl)
    IMP.bff.ProbeAccessibleVolumeDecorator.do_setup_particle(mdl, p, source)
    av = IMP.bff.ProbeAccessibleVolumeDecorator(mdl, p)
    av.resample()
    pts = np.asarray([list(v) for v in av.get_map().get_xyz_density()])
    return pts[np.lexsort(pts[:, :3].T[::-1])] if len(pts) else pts


class Tests(IMP.test.TestCase):

    def setUp(self):
        super().setUp()
        IMP.set_log_level(IMP.SILENT)

    def test_array_door_attachment_row_changes_nothing(self):
        """get_av: the attachment atom in the array == the atom left out."""
        xyzr = _xyzr()
        for residue in SITES:
            i, src = _row(xyzr, residue)
            kept = IMP.bff.get_av(np.ascontiguousarray(xyzr), list(src))
            gone = IMP.bff.get_av(np.ascontiguousarray(np.delete(xyzr, i, axis=0)),
                                  list(src))
            self.assertGreater(int(kept.n_points), 0, residue)
            np.testing.assert_array_equal(np.asarray(kept.get_density()),
                                          np.asarray(gone.get_density()),
                                          err_msg=str(residue))

    def test_path_lengths_attachment_row_changes_nothing(self):
        """The path-length read-out of the same search agrees as well."""
        xyzr = _xyzr()
        i, src = _row(xyzr, 132)
        kept = IMP.bff.get_linker_path_lengths(np.ascontiguousarray(xyzr), list(src))
        gone = IMP.bff.get_linker_path_lengths(
            np.ascontiguousarray(np.delete(xyzr, i, axis=0)), list(src))
        def values(grid):
            hd = grid.get_header()
            n = hd.get_nx() * hd.get_ny() * hd.get_nz()
            return np.array([grid.get_value(k) for k in range(n)])

        a, b = values(kept), values(gone)
        self.assertGreater(int((a >= 0).sum()), 0)
        np.testing.assert_array_equal(a, b)

    def test_decorator_door_attachment_radius_changes_nothing(self):
        """Decorator (and the fps.json, docking and network doors behind it):
        the attachment atom at its own radius == the same atom at radius 0."""
        for residue in SITES:
            a = _decorator_points(residue, False)
            b = _decorator_points(residue, True)
            self.assertGreater(len(a), 0, residue)
            np.testing.assert_array_equal(a, b, err_msg=str(residue))

    def test_the_default_door_values_are_the_declared_ones(self):
        """At FPS's width the declared allowed sphere, an explicit 2.25 and the
        width-derived `0.5 W` are one volume."""
        xyzr = np.ascontiguousarray(_xyzr())
        _, src = _row(xyzr, 132)
        default = IMP.bff.get_av(xyzr, list(src))
        explicit = IMP.bff.get_av(xyzr, list(src), allowed_sphere_radius=2.25)
        derived = IMP.bff.get_av(xyzr, list(src), allowed_sphere_radius=-1.0)
        d = np.asarray(default.get_density())
        np.testing.assert_array_equal(d, np.asarray(explicit.get_density()))
        np.testing.assert_array_equal(d, np.asarray(derived.get_density()))


if __name__ == "__main__":
    IMP.test.main()
