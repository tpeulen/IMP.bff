"""The accessible-volume defaults are FPS's and Olga's, and every door agrees.

Owner, 2026-10-11: "av defaults should be fps/olga defaults, take from there."
The values are read from the two programs' source (the table is in
`include/probe/ProbeAccessibleVolumeDefaults.h`); this file holds the doors to them.
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
        # the declared allowed sphere is a radius, not a share of the width
        self.assertAlmostEqual(av.get_effective_allowed_sphere_radius(), 2.25)
        # a negative one derives FPS's `0.5 W` for this width
        av.set_allowed_sphere_radius(-1.0)
        self.assertAlmostEqual(av.get_effective_allowed_sphere_radius(), 1.5)
        self.assertAlmostEqual(av.get_effective_first_step_reach(), 1.5 + 1.5)

    def test_the_allowed_sphere_is_on_by_default(self):
        """Owner, 2026-10-12: "allowed sphere radius must be enabled by default".

        FPS's allowed sphere is `LinkerInitialSphere` (0.5, `ProjectData.cs`)
        times the linker width; every FPS linker preset has W 4.5, so 2.25 A.
        It was on before as well (a stored -1 derived the same 2.25 A at the
        default width), but the -1 read as "off" wherever it was shown."""
        self.assertEqual(IMP.bff.AV_DEFAULT_ALLOWED_SPHERE, 2.25)
        self.assertEqual(IMP.bff.AV_DEFAULT_ALLOWED_SPHERE,
                         IMP.bff.AV_DEFAULT_LINKER_INITIAL_SPHERE
                         * IMP.bff.AV_DEFAULT_LINKER_WIDTH)
        av = _particle_av(self.mdl, self.hier)
        self.assertEqual(av.get_allowed_sphere_radius(), 2.25)
        av.set_av_parameter(json.dumps({"simulation_type": "AV1"}))
        self.assertEqual(av.get_allowed_sphere_radius(), 2.25)

    def test_a_position_without_one_takes_fpss_rule_for_its_width(self):
        """FPS's allowed sphere is `0.5 W` per position. Its screening document
        has widths 2.5, 3.5 and 4.5 A; a flat 2.25 for all of them moved 3GUN
        away from FPS's own routine (fps_screening_ab.md, 2026-10-12)."""
        av = _particle_av(self.mdl, self.hier)
        av.set_av_parameter(json.dumps({"linker_width": 2.5}))
        self.assertEqual(av.get_allowed_sphere_radius(), 1.25)
        av.set_av_parameter(json.dumps({"linker_width": 2.5,
                                        "allowed_sphere_radius": 3.0}))
        self.assertEqual(av.get_allowed_sphere_radius(), 3.0)

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

    def test_an_av3_keeps_its_three_radii(self):
        """fps.json radius2/radius3 are read, not overwritten with radius1.

        The reader once did `set_radius2(r[0]); set_radius3(r[0])`, so every
        three-radius position became a one-radius AV (QuEst known issue,
        2026-08-10). Distinct values make a regression visible."""
        av = _particle_av(self.mdl, self.hier)
        av.set_av_parameter(json.dumps({"simulation_type": "AV3", "radius1": 6.0,
                                        "radius2": 4.0, "radius3": 2.0}))
        self.assertEqual((av.get_radius1(), av.get_radius2(), av.get_radius3()),
                         (6.0, 4.0, 2.0))

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
        self.assertEqual(default_of("allowed_sphere_radius"),
                         IMP.bff.AV_DEFAULT_ALLOWED_SPHERE)

    def test_the_position_template_states_the_defaults(self):
        """`data/template_av_position.fps.json` is data and cannot read the header.

        It carried a 0.5 linker width, a 1.5 grid and a 1.5 clearance until
        2026-10-11 (owner: "all use same defaults")."""
        path = IMP.bff.get_data_path("template_av_position.fps.json")
        with open(path) as fh:
            pos = json.load(fh)
        self.assertEqual(pos["linker_length"], IMP.bff.AV_DEFAULT_LINKER_LENGTH)
        self.assertEqual(pos["linker_width"], IMP.bff.AV_DEFAULT_LINKER_WIDTH)
        self.assertEqual(pos["radius1"], IMP.bff.AV_DEFAULT_RADIUS1)
        self.assertEqual(pos["simulation_grid_resolution"],
                         IMP.bff.AV_DEFAULT_SIMULATION_GRID_RESOLUTION)
        self.assertEqual(pos["allowed_sphere_radius"],
                         IMP.bff.AV_DEFAULT_ALLOWED_SPHERE)  # FPS's, on

    def test_the_labelizer_refines_with_the_av3_defaults(self):
        """`LabelizerFRETOptions` carried the reference's dye (9/4.5/1.5, 1.5 A grid)."""
        o = IMP.bff.LabelizerFRETOptions()
        self.assertEqual(o.linker_length, IMP.bff.AV_DEFAULT_LINKER_LENGTH)
        self.assertEqual(o.linker_width, IMP.bff.AV_DEFAULT_LINKER_WIDTH)
        self.assertEqual((o.r1, o.r2, o.r3),
                         (IMP.bff.AV_DEFAULT_AV3_RADIUS1, IMP.bff.AV_DEFAULT_AV3_RADIUS2,
                          IMP.bff.AV_DEFAULT_AV3_RADIUS3))
        self.assertEqual(o.grid_resolution, IMP.bff.AV_DEFAULT_SIMULATION_GRID_RESOLUTION)


if __name__ == "__main__":
    IMP.test.main()
