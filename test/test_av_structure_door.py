"""The structure door computes the volume the IMP decorator computes.

`get_av_from_structure` is core: it strips the attachment residue, reads the
atoms as `IMP::atom::read_pdb` with a `NonWaterPDBSelector` reads them (first
model, alternate location blank or `A`, no water), inflates each by IMP's
radius times the scale and runs the lattice search. Until 2026-10-26 the IMP
module build computed it differently: read the stripped structure into an
`IMP::Model`, decorate the attachment atom and resample (`resample_av`), and
the standalone build had no structure door at all -- so contact volumes (a
weighting only this door applies) were unreachable there.

The reference below is that removed route, rebuilt from the decorator's own
pieces, on every position of the T4L network the chimol demo draws (3GUN,
seventeen accessible *contact* volumes, plus the plain volume of three). The
two must agree voxel for voxel, weight for weight.
"""

import json
import os
import unittest

import numpy as np

import IMP.bff

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "input", "av_structure_door")


def _decorator_route(pdb, position, step):
    """The IMP build's former structure door: IMP::Model + resample_av."""
    import IMP
    import IMP.atom

    chain = position.get("chain_identifier", "")
    resseq = int(position["residue_seq_number"])
    atom = position.get("atom_name", "CA")
    radii = IMP.bff.av_position_radii(json.dumps(position))
    stripped = IMP.bff.stripped_pdb_for(pdb, chain, resseq, atom, position.get("strip_mask", ""))
    model = IMP.Model()
    hierarchy = IMP.atom.read_pdb(stripped, model, IMP.atom.NonWaterPDBSelector())
    kwargs = dict(residue_index=resseq, atom_type=IMP.atom.AtomType(atom))
    if chain:
        kwargs["chain_id"] = chain
    (particle, *_rest) = IMP.atom.Selection(hierarchy, **kwargs).get_selected_particles()
    thickness = float(position.get("contact_volume_thickness", 0.0))
    trapped = float(position.get("contact_volume_trapped_fraction", -1.0))
    clearance = float(position["allowed_sphere_radius"]) if "allowed_sphere_radius" in position \
        else IMP.bff.av_derived_allowed_sphere_radius(float(position["linker_width"]))
    av = IMP.bff.resample_av(
        model, particle, float(position["linker_length"]), float(position["linker_width"]),
        radii[0], radii[1], radii[2], step, clearance, thickness, trapped, 0,
        float(position.get("radius_scale", IMP.bff.AV_DEFAULT_RADIUS_SCALE)))
    points = np.asarray(av.get_points(), dtype=float).reshape(-1, 4).copy()
    if not (thickness > 0.0 and 0.0 <= trapped < 1.0):
        points[:, 3] = 1.0      # the door's uniform-weight convention
    return av, points


class TestStructureDoor(unittest.TestCase):

    def setUp(self):
        if not hasattr(IMP.bff, "resample_av"):
            self.skipTest("the decorator route needs the IMP module build")
        with open(os.path.join(_DATA, "t4l_network.fps.json")) as fh:
            self.positions = json.load(fh)["Positions"]
        self.pdb = os.path.join(_DATA, "t4l_3gun.pdb")

    def _compare(self, position):
        step = float(position["simulation_grid_resolution"])
        reference, ref_points = _decorator_route(self.pdb, position, step)
        core = IMP.bff.get_av_from_structure(self.pdb, json.dumps(position), step)
        points = np.asarray(core.get_points(), dtype=float).reshape(-1, 4)
        self.assertEqual(points.shape, ref_points.shape)
        np.testing.assert_allclose(points, ref_points, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(core.get_density()),
                                   np.asarray(reference.get_density()), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(core.get_grid_origin()),
                                   np.asarray(reference.get_grid_origin()), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(core.get_attachment_point()),
                                   np.asarray(reference.get_attachment_point()), rtol=0, atol=1e-12)
        return len(points)

    def test_every_contact_volume_of_the_t4l_network(self):
        sizes = {}
        for name, position in sorted(self.positions.items()):
            with self.subTest(position=name):
                sizes[name] = self._compare(position)
        self.assertEqual(len(sizes), 17)
        self.assertTrue(all(n > 0 for n in sizes.values()), sizes)

    def test_plain_volumes_keep_uniform_weights(self):
        for name in sorted(self.positions)[:3]:
            position = dict(self.positions[name])
            position.pop("contact_volume_thickness")
            position.pop("contact_volume_trapped_fraction")
            with self.subTest(position=name):
                self._compare(position)


if __name__ == "__main__":
    unittest.main()
