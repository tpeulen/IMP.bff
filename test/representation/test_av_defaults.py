"""The accessible-volume defaults are FPS's and Olga's, and every door agrees.

Owner, 2026-10-11: "av defaults should be fps/olga defaults, take from there."
The values are read from the two programs' source (the table is in
`include/ProbeAccessibleVolumeDefaults.h`); this file holds the doors to them.
Before it, the same omitted parameter meant different things at different
doors: a linker width of 0.5 at the decorator, 1.0 at the fps.json position
reader, 1.5 on the command line; a dye radius of 3.0 at one json reader and 3.5
at the other; a grid of 1.5 everywhere, where FPS and Olga both use 0.9.
"""
import json

import IMP
import IMP.atom
import IMP.bff
import IMP.test

PDB = IMP.bff.get_example_path("structure/T4L/3GUN.pdb")


def _particle_av(mdl, hier, residue_index=132, atom_name="CB"):
    sel = IMP.atom.Selection(hier)
    sel.set_atom_type(IMP.atom.AtomType(atom_name))
    sel.set_residue_index(residue_index)
    source = sel.get_selected_particles()[0]
    p = IMP.Particle(mdl)
    IMP.bff.ProbeAccessibleVolumeDecorator.setup_particle(mdl, p, source)
    return IMP.bff.ProbeAccessibleVolumeDecorator(mdl, p)


class Tests(IMP.test.TestCase):

    def setUp(self):
        super().setUp()
        IMP.set_log_level(IMP.SILENT)
        self.mdl = IMP.Model()
        self.hier = IMP.atom.read_pdb(PDB, self.mdl)

    def test_the_constants_are_fps_first_preset_and_olgas_grid(self):
        """FPS `Fps/data/linker.txt` row 1 ("Alexa488 L linker") and Olga's 0.9 A."""
        self.assertEqual(IMP.bff.AV_DEFAULT_LINKER_LENGTH, 20.0)
        self.assertEqual(IMP.bff.AV_DEFAULT_LINKER_WIDTH, 4.5)
        self.assertEqual(IMP.bff.AV_DEFAULT_RADIUS1, 3.5)
        self.assertEqual((IMP.bff.AV_DEFAULT_AV3_RADIUS1, IMP.bff.AV_DEFAULT_AV3_RADIUS2,
                          IMP.bff.AV_DEFAULT_AV3_RADIUS3), (5.0, 4.5, 1.5))
        self.assertEqual(IMP.bff.AV_DEFAULT_SIMULATION_GRID_RESOLUTION, 0.9)
        self.assertEqual(IMP.bff.AV_DEFAULT_LINKER_INITIAL_SPHERE, 0.5)
        self.assertEqual(IMP.bff.AV_DEFAULT_LINK_SEARCH_NODES, 3)

    def test_the_decorator_door_takes_the_defaults(self):
        av = _particle_av(self.mdl, self.hier)
        self.assertEqual(av.get_linker_length(), 20.0)
        self.assertEqual(av.get_linker_width(), 4.5)
        self.assertEqual(av.get_radius1(), 3.5)
        self.assertEqual((av.get_radius2(), av.get_radius3()), (0.0, 0.0))
        self.assertAlmostEqual(av.get_simulation_grid_resolution(), 0.9)

    def test_the_search_seeds_and_steps_as_fps_does(self):
        """Seed `0.5 W` (LinkerInitialSphere) and a first step of `0.5 W + 3 dg`."""
        av = _particle_av(self.mdl, self.hier)
        self.assertAlmostEqual(av.get_effective_allowed_sphere_radius(), 2.25)
        self.assertAlmostEqual(av.get_effective_first_step_reach(), 2.25 + 3 * 0.9)
        av.set_linker_width(3.0)
        av.set_simulation_grid_resolution(0.5)
        self.assertAlmostEqual(av.get_effective_allowed_sphere_radius(), 1.5)
        self.assertAlmostEqual(av.get_effective_first_step_reach(), 1.5 + 1.5)

    def test_an_absent_json_field_means_the_same_at_every_door(self):
        """The fps.json reader of the decorator and of the position door agree."""
        av = _particle_av(self.mdl, self.hier)
        av.set_av_parameter(json.dumps({"simulation_type": "AV1"}))
        self.assertEqual(av.get_linker_width(), 4.5)
        self.assertEqual(av.get_radius1(), 3.5)
        self.assertAlmostEqual(av.get_simulation_grid_resolution(), 0.9)

    def test_an_av3_without_radii_gets_fpss_three(self):
        """Zero is the "unused" sentinel, so AV1 defaults would make it an AV1."""
        av = _particle_av(self.mdl, self.hier)
        av.set_av_parameter(json.dumps({"simulation_type": "AV3"}))
        self.assertEqual((av.get_radius1(), av.get_radius2(), av.get_radius3()),
                         (5.0, 4.5, 1.5))

    def test_the_schema_states_the_defaults(self):
        schema = json.loads(IMP.bff.fps_json_schema())
        text = json.dumps(schema)

        def default_of(field):
            # the position fields sit in one object; find the field anywhere
            def walk(node):
                if isinstance(node, dict):
                    if field in node and isinstance(node[field], dict) \
                            and "default" in node[field]:
                        return node[field]["default"]
                    for v in node.values():
                        found = walk(v)
                        if found is not None:
                            return found
                elif isinstance(node, list):
                    for v in node:
                        found = walk(v)
                        if found is not None:
                            return found
                return None
            return walk(schema)

        self.assertIn("linker_width", text)
        self.assertEqual(default_of("linker_length"), IMP.bff.AV_DEFAULT_LINKER_LENGTH)
        self.assertEqual(default_of("linker_width"), IMP.bff.AV_DEFAULT_LINKER_WIDTH)
        self.assertEqual(default_of("radius1"), IMP.bff.AV_DEFAULT_RADIUS1)
        self.assertEqual(default_of("simulation_grid_resolution"),
                         IMP.bff.AV_DEFAULT_SIMULATION_GRID_RESOLUTION)
        self.assertEqual(default_of("allowed_sphere_radius"), -1.0)


if __name__ == "__main__":
    IMP.test.main()
