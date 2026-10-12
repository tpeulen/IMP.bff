"""The one obstacle-radius scale factor on IMP's radii (AV_DEFAULT_RADIUS_SCALE).

Owner, 2026-10-12: "i do not want a separate set of atom radii, if fps is
slightly smaller, use a single scale factor to match fps radii". So an AV
inflates its obstacles by the radius each particle carries -- IMP's united-atom
set after ``read_pdb`` -- times one declared factor, fitted to FPS's Bondi
radii (``okf/validation/fps_screening_ab.md``, "AV obstacle-radius scale").
What is pinned here is the mechanism: where the factor comes from, that it
reaches the obstacle set and nothing else, and that a shared raster cannot
silently mix two of them.
"""
import json

import pytest

import IMP
import IMP.atom
import IMP.bff
import IMP.core


def _t4l():
    IMP.set_log_level(IMP.SILENT)
    m = IMP.Model()
    h = IMP.atom.read_pdb(IMP.bff.get_example_path("structure/T4L/3GUN.pdb"), m,
                          IMP.atom.NonWaterNonHydrogenPDBSelector())
    return m, h


def _av(m, h, residue=132):
    src = IMP.atom.Selection(h, residue_index=residue,
                             atom_type=IMP.atom.AtomType("CB")
                             ).get_selected_particles()[0]
    p = IMP.Particle(m)
    IMP.bff.ProbeAccessibleVolumeDecorator.do_setup_particle(
        m, p, src, linker_length=20.0, radii=(3.5, 0.0, 0.0), linker_width=4.5,
        simulation_grid_resolution=0.9)
    return IMP.bff.ProbeAccessibleVolumeDecorator(m, p)


def test_the_default_is_declared_once():
    assert IMP.bff.AV_DEFAULT_RADIUS_SCALE == 0.84
    m, h = _t4l()
    assert _av(m, h).get_radius_scale() == IMP.bff.AV_DEFAULT_RADIUS_SCALE


def test_the_obstacles_are_imps_radii_times_the_factor():
    m, h = _t4l()
    av = _av(m, h)
    av.resample()
    leaves = IMP.atom.get_leaves(h)
    radii = list(av.get_map().get_obstacle_radii())
    assert len(radii) == len(leaves)
    for leaf, r in zip(leaves, radii):
        assert r == pytest.approx(
            IMP.core.XYZR(leaf).get_radius() * IMP.bff.AV_DEFAULT_RADIUS_SCALE)
    # the particles themselves are untouched: the clash term reads them
    assert IMP.core.XYZR(leaves[0]).get_radius() != pytest.approx(radii[0])


def test_a_smaller_factor_opens_the_volume_and_one_is_imps_radii():
    m, h = _t4l()
    counts = {}
    for scale in (1.0, IMP.bff.AV_DEFAULT_RADIUS_SCALE):
        av = _av(m, h)
        av.set_radius_scale(scale)
        av.resample()
        counts[scale] = len(av.get_map().get_xyz_density())
    # 7 475 at x1.0, 9 142 at x0.84; FPS's own routine 9 237 (Bondi radii)
    assert counts[1.0] == 7475
    assert counts[IMP.bff.AV_DEFAULT_RADIUS_SCALE] == 9142


def test_olgas_table_is_not_scaled():
    m, h = _t4l()
    av = _av(m, h)
    av.set_radii_source("olga")
    av.resample()
    radii = list(av.get_map().get_obstacle_radii())
    leaf_names = [IMP.atom.Atom(p).get_atom_type().get_string()
                  for p in IMP.atom.get_leaves(h)]
    i = leaf_names.index("CA")
    assert radii[i] == pytest.approx(IMP.bff.olga_vdw_radius("CA"))


def test_the_factor_must_be_positive():
    m, h = _t4l()
    av = _av(m, h)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(IMP.ValueException):
            av.set_radius_scale(bad)


def test_it_is_a_per_position_fps_json_field():
    """Schema 1.9 ``radius_scale``; absent means the declared default."""
    m, h = _t4l()
    av = _av(m, h)
    av.set_av_parameter(json.dumps({"linker_length": 20.0}))
    assert av.get_radius_scale() == IMP.bff.AV_DEFAULT_RADIUS_SCALE
    av.set_av_parameter(json.dumps({"linker_length": 20.0, "radius_scale": 1.0}))
    assert av.get_radius_scale() == 1.0
    schema = json.loads(IMP.bff.fps_json_schema())
    field = schema["$defs"]["position"]["properties"]["radius_scale"]
    assert field["default"] == IMP.bff.AV_DEFAULT_RADIUS_SCALE
    assert IMP.bff.fps_schema_version() >= "1.9"


def test_a_shared_raster_refuses_a_second_factor():
    m, h = _t4l()
    registry = IMP.bff.ProbeAccessibleVolumeOccupancyRegistry(IMP.atom.get_leaves(h))
    first, second = _av(m, h, 132), _av(m, h, 55)
    for av in (first, second):
        av.set_occupancy_registry(registry)
    first.resample()
    assert registry.get_radius_scale() == IMP.bff.AV_DEFAULT_RADIUS_SCALE
    second.set_radius_scale(1.0)
    with pytest.raises(IMP.ValueException):
        second.resample()


def test_the_structure_door_takes_it():
    pdb = IMP.bff.get_example_path("structure/T4L/3GUN.pdb")
    default = IMP.bff.get_av_from_structure(pdb, "A", 132, "CB", 20.0, 4.5,
                                            3.5, 0.0, 0.0, 0.9)
    unscaled = IMP.bff.get_av_from_structure(pdb, "A", 132, "CB", 20.0, 4.5,
                                             3.5, 0.0, 0.0, 0.9, "", 2.25,
                                             0.0, -1.0, 1.0)
    assert int(default.n_points) > int(unscaled.n_points) > 0
