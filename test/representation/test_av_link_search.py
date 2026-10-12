"""The link search: FPS's cube hop from every node, and the seed-only option.

FPS routes the linker by hopping up to ``LinkSearchNodes = 3`` voxels along
every axis at once from *every* node it reaches -- the 7x7x7 cube, landing on
any free voxel across whatever lies between (``av_routines.cpp:156-209``). That
is the default since 2026-10-12 (owner, reversing 2026-10-11's "av go fast by
default"); the seed-only search -- the hop out of the seed sphere, then the 74
stencil -- stays selectable as ``link_search = "fast"``.

What is pinned here:

* the names and the defaults, at every door (decorator, fps.json, docking);
* that the cube search *is* FPS's: the same reached set as FPS's own
  propagation (re-implemented below from ``calculate1R``, wavefront and all)
  and the same path lengths to float precision;
* that the lattice search's shortcuts (rings that cannot pass the bound are
  skipped, FPS's cube is relaxed a row at a time with SIMD) change no bit of
  the result: ``PathMap.set_plain_relaxation(True)`` turns them off.

Measured in ``okf/validation/fps_screening_ab.md``.
"""
import itertools
import json

import numpy as np
import pytest

import IMP
import IMP.algebra
import IMP.atom
import IMP.bff
import IMP.core

PDB = IMP.bff.get_example_path("structure/T4L/3GUN.pdb")


def test_the_default_is_fps_link_search():
    assert IMP.bff.AV_DEFAULT_LINK_SEARCH == "fps"
    assert IMP.bff.AV_SEARCH_STENCIL_FPS == 7 ** 3 - 1
    assert IMP.bff.AV_DEFAULT_SEARCH_STENCIL == IMP.bff.AV_SEARCH_STENCIL_FPS
    assert IMP.bff.AV_SEARCH_STENCIL_FAST == 74
    assert IMP.bff.av_link_search_stencil("fps") == 342
    assert IMP.bff.av_link_search_stencil("fast") == 74
    assert IMP.bff.av_link_search_name(342) == "fps"
    for stencil in (26, 30, 74):
        assert IMP.bff.av_link_search_name(stencil) == "fast"
    with pytest.raises(IMP.ValueException):
        IMP.bff.av_link_search_stencil("dijkstra")


class _Site:
    """One AV on T4L 3GUN through the decorator door; keeps its Model alive."""

    def __init__(self, residue, grid, link_search=None):
        IMP.set_log_level(IMP.SILENT)
        self.mdl = IMP.Model()
        self.hier = IMP.atom.read_pdb(PDB, self.mdl)
        src = IMP.atom.Selection(self.hier, residue_index=residue,
                                 atom_type=IMP.atom.AtomType("CB")).get_selected_particles()[0]
        p = IMP.Particle(self.mdl)
        IMP.bff.ProbeAccessibleVolumeDecorator.do_setup_particle(
            self.mdl, p, src, linker_length=20.0, radii=(3.5, 0.0, 0.0),
            linker_width=4.5, simulation_grid_resolution=grid)
        self.av = IMP.bff.ProbeAccessibleVolumeDecorator(self.mdl, p)
        if link_search is not None:
            self.av.set_link_search(link_search)
        # an atom inside the window, ~12 A out: nudging it makes the next
        # resample a real one (the volume does not move by a voxel)
        c = IMP.core.XYZ(src).get_coordinates()
        self.nudge = IMP.core.XYZ(min(
            IMP.atom.get_leaves(self.hier),
            key=lambda q: abs(IMP.algebra.get_distance(
                IMP.core.XYZ(q).get_coordinates(), c) - 12.0)))

    def points(self, plain):
        self.av.resample()
        self.av.get_map().set_plain_relaxation(plain)
        for step in (1e-3, -1e-3):        # two real searches, back where it was
            self.nudge.set_coordinates(
                self.nudge.get_coordinates() + IMP.algebra.Vector3D(step, 0, 0))
            self.av.resample()
        return np.array(self.av.get_map().get_xyz_density())


def test_the_decorator_defaults_to_fps_and_fast_is_selectable():
    site = _Site(44, 0.9)
    assert site.av.get_link_search() == "fps"
    assert site.av.get_search_stencil() == 342
    site.av.set_link_search("fast")
    assert site.av.get_search_stencil() == 74
    assert site.av.get_link_search() == "fast"
    site.av.set_search_stencil(26)
    assert site.av.get_link_search() == "fast"   # every radius stencil is seed-only
    with pytest.raises(IMP.ValueException):
        site.av.set_link_search("everywhere")


def test_fps_finds_more_volume_than_the_seed_only_search():
    """T4L 44 CB at FPS's width: the cube reaches what the 74 stencil walks around."""
    fps = len(_Site(44, 0.9).points(False))
    fast = len(_Site(44, 0.9, "fast").points(False))
    # 2026-10-12: IMP's radii x AV_DEFAULT_RADIUS_SCALE (0.84, fitted to FPS's
    # Bondi radii) are the obstacles; at IMP's radii as they are (x1.0) these
    # were 21739 and 20572.
    assert fps == 23078
    assert fast == 21882


@pytest.mark.parametrize("residue,grid", [(44, 0.9), (132, 0.9), (44, 0.5)])
@pytest.mark.parametrize("link_search", ["fps", "fast"])
def test_the_shortcuts_change_no_bit(residue, grid, link_search):
    """Ring pruning and the SIMD cube rows against one-offset-at-a-time relaxation."""
    plain = _Site(residue, grid, link_search).points(True)
    fast = _Site(residue, grid, link_search).points(False)
    assert len(plain) > 0
    assert np.array_equal(plain, fast)


def _fps_reference(spheres, h, L, W, nx, ny, nz, origin, linknodes=3, linkersphere=0.5):
    """FPS's ``calculate1R`` linker routing, on the given lattice, in numpy.

    The source is the lattice node at the origin of the coordinates. Clash: a
    node closer than ``vdW + W/2`` to an atom (``rmaxsq_linker``); seeds: every
    node with ``i^2 + j^2 + k^2 <= floor((ls W / dg)^2)`` at its straight-line
    distance, clash or not; then hops of up to ``linknodes`` voxels along every
    axis from every new node, at ``sqrt(i^2 + j^2 + k^2) dg``, onto any
    non-clashing node with a shorter path below ``L`` -- until nothing moves.
    """
    ix, iy, iz = (np.arange(n) for n in (nx, ny, nz))
    X, Y, Z = np.meshgrid(origin[0] + ix * h, origin[1] + iy * h, origin[2] + iz * h,
                          indexing="ij")
    clash = np.zeros(X.shape, bool)
    for x, y, z, r in spheres:
        clash |= (X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2 < (r + 0.5 * W) ** 2
    rlink = np.where(clash, -L, 2 * L)
    i2 = np.round(X / h) ** 2 + np.round(Y / h) ** 2 + np.round(Z / h) ** 2
    seed = i2 <= np.floor((linkersphere * W / h) ** 2)
    rlink[seed] = np.sqrt(i2[seed]) * h
    offsets = [o for o in itertools.product(range(-linknodes, linknodes + 1), repeat=3)
               if o != (0, 0, 0)]
    shape = (nx, ny, nz)
    changed = True
    while changed:
        changed = False
        for o in offsets:
            length = np.sqrt(sum(v * v for v in o)) * h
            src = tuple(slice(max(0, -v), n - max(0, v)) for v, n in zip(o, shape))
            dst = tuple(slice(max(0, v), n - max(0, -v)) for v, n in zip(o, shape))
            cand = rlink[src] + length
            cur = rlink[dst]
            ok = (rlink[src] >= 0) & (cand < cur) & (cand < L)
            if ok.any():
                cur[ok] = cand[ok]
                changed = True
    return rlink, clash, seed, np.sqrt(i2) * h


@pytest.mark.parametrize("seed", [3, 11])
def test_the_cube_search_is_fps_own_routing(seed):
    """Same reached set as FPS's propagation, same path lengths to float precision.

    Random spheres around a source on a lattice node (FPS centres its grid on
    the attachment atom), none inside the seed sphere -- FPS keeps a clashing
    seed out of the volume while this module opens it, a difference of the
    seed and not of the link search.
    """
    rng = np.random.default_rng(seed)
    h, L, W = 1.0, 12.0, 4.5
    spheres = []
    while len(spheres) < 60:
        p = rng.uniform(-10, 10, 3)
        r = rng.uniform(1.0, 2.0)
        if np.linalg.norm(p) - r - 0.5 * W < 0.5 * W + 0.5:
            continue
        spheres.append([*p, r])
    spheres = np.ascontiguousarray(spheres)
    grid = IMP.bff.get_linker_path_lengths(spheres, [0.0, 0.0, 0.0], L, W, 0.0, h, -1.0, 342)
    hd = grid.get_header()
    nx, ny, nz = hd.get_nx(), hd.get_ny(), hd.get_nz()
    origin = (hd.get_xorigin(), hd.get_yorigin(), hd.get_zorigin())
    ours = np.array([grid.get_value(i) for i in range(nx * ny * nz)]).reshape(nx, ny, nz)
    rlink, clash, seeds, d0 = _fps_reference(spheres, h, L, W, nx, ny, nz, origin)
    reached = (rlink >= 0) & (rlink <= L)
    mine = ours >= 0
    assert reached.sum() > 500
    assert np.array_equal(reached, mine)
    assert np.abs(rlink[mine] - ours[mine]).max() < 1e-5
    # and the seed-only search does not: it walks around what the cube hops over
    fast = IMP.bff.get_linker_path_lengths(spheres, [0.0, 0.0, 0.0], L, W, 0.0, h, -1.0, 74)
    fast = np.array([fast.get_value(i) for i in range(nx * ny * nz)]).reshape(nx, ny, nz)
    assert (fast >= 0).sum() < mine.sum()


def test_fps_json_position_selects_the_link_search():
    site = _Site(44, 0.9)
    site.av.set_av_parameter(json.dumps({"link_search": "fast"}))
    assert site.av.get_link_search() == "fast"
    site.av.set_av_parameter(json.dumps({}))
    assert site.av.get_link_search() == "fps"
    schema = json.loads(IMP.bff.fps_json_schema())
    field = schema["$defs"]["position"]["properties"]["link_search"]
    assert field["default"] == "fps"
    assert field["enum"] == ["fps", "fast"]


def test_docking_scores_the_positions_own_link_search():
    """The network restraint used to force stencil 26 on every AV."""
    mdl = IMP.Model()
    hier = IMP.atom.read_pdb(PDB, mdl)
    fps_json = IMP.bff.get_example_path("structure/T4L/fret.fps.json")
    restraint = IMP.bff.ProbeNetworkRestraint(hier, fps_json)
    assert restraint.get_search_stencil() == 0
    stencils = {av.get_search_stencil() for av in restraint.get_used_avs()}
    assert stencils == {342}
