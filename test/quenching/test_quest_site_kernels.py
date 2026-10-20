"""The QuEst site-table kernels against the MalE analysis they were ported from.

The MalE quenching analysis (``quest/prototype/male_quenching_analysis.ipynb``,
chapters 3b and 5) builds, for every labelled site, a per-voxel table: how many
residues of each type are near the dye, where every quencher's redox centre can
be over its side-chain rotamers, and how much of the dye water can reach. It did
that in Python with Bio.PDB and scipy; these are the C++ kernels that replace it:

* ``residue_type_neighbour_counts`` -- the notebook's "slow-factor map trick"
  (one QuenchedDonorDecay per residue type, factor 0.5, ``log s / log 0.5``);
* ``probe_solvent_exposure`` -- its Shrake-Rupley ``water_exposure``;
* ``quencher_rotamer_ensembles`` -- its ``quencher_positions``;
* ``ProbeAccessibleVolume.get_voxel_centres`` -- the voxel convention;
* ``stretched_linker_length`` -- its cell 21.

Each test runs the notebook's own code (verbatim, Bio.PDB/scipy where it used
them) on MalE apo (PDB 1OMP), site 72, as the reference.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

import IMP.bff as bff

Bio = pytest.importorskip("Bio.PDB")
cKDTree = pytest.importorskip("scipy.spatial").cKDTree
from Bio.PDB import MMCIFParser, PDBIO, Select  # noqa: E402
from Bio.PDB.Polypeptide import PPBuilder  # noqa: E402

HERE = Path(__file__).resolve().parent
CIF = HERE.parent.parent / "examples" / "structure" / "MBP" / "1OMP.cif"

SITE, CHAIN = 72, "A"
LINKER, WIDTH, DYE_RADIUS, GRID = 20.12, 0.5, 3.5, 0.5
SLOW_RADIUS, CLASH = 8.5, 2.8
QUENCHERS = ["TRP", "TYR", "MET", "HIS", "PRO"]
GROUPS = {
    "aromatic": ["TRP", "TYR", "HIS"],
    "hydrophobic": ["PHE", "LEU", "ILE", "VAL", "MET", "ALA", "PRO"],
    "basic": ["LYS", "ARG"],
    "acidic": ["ASP", "GLU"],
    "polar": ["SER", "THR", "ASN", "GLN", "GLY", "CYS"],
}
RESIDUES = [r for group in GROUPS.values() for r in group]
GRID_OPTIONS = dict(critical_distance=8.5, slow_radius=SLOW_RADIUS, probe_radius=DYE_RADIUS,
                    diffusion_coefficient=7.5, slow_fact=0.5, t_step=0.032, n_photons=1)
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}


def _library():
    path = Path(bff.get_data_path("rotamer_library")) / "sidechains.drot.pto"
    if not path.exists():
        pytest.skip("IMP.bff's side-chain library is not installed")
    return str(path)


# ---------------------------------------------------------------- the site --

def kept(atom, site):
    # notebook cell 23, verbatim
    residue = atom.get_parent()
    if residue.resname == "HOH" or atom.element == "H":
        return False
    at_site = residue.id[0] == " " and residue.id[1] == site and residue.get_parent().id == CHAIN
    return not (at_site and atom.get_id() not in {"N", "CA", "C", "O", "CB"})


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    if not CIF.exists():
        pytest.skip(f"{CIF} is not in this checkout")
    structure = MMCIFParser(QUIET=True).get_structure("APO", str(CIF))
    pdb = tmp_path_factory.mktemp("male") / f"1omp_site{SITE}.pdb"

    class Keep(Select):
        def accept_atom(self, atom):
            return kept(atom, SITE)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb), Keep())
    av = bff.get_av_from_pdb(str(pdb), CHAIN, SITE, "CB", LINKER, WIDTH, DYE_RADIUS, 0.0, 0.0,
                             GRID)
    atoms = [a for a in structure[0].get_atoms() if kept(a, SITE)]
    obstacles = bff.ObstacleAtoms()
    obstacles.chains = [a.get_parent().get_parent().id for a in atoms]
    obstacles.res_ids = [int(a.get_parent().id[1]) for a in atoms]
    obstacles.res_names = [a.get_parent().resname for a in atoms]
    obstacles.atom_names = [a.get_id() for a in atoms]
    obstacles.coords = [float(v) for a in atoms for v in a.coord]
    return dict(structure=structure, av=av, atoms=atoms, obstacles=obstacles)


# --------------------------------------------------------------- voxels --

def test_voxel_centres_are_the_cloud_and_the_walk_frame(site):
    av = site["av"]
    ng, dg = av.ng, av.grid_step
    density = np.asarray(av.get_density())
    points = np.flatnonzero(density > 0)
    assert np.array_equal(np.asarray(av.get_voxel_indices()), points)

    centres = np.asarray(av.get_voxel_centres())
    assert centres.shape == (points.size, 3)
    # the grid's own convention: the same set of positions as the cloud
    # get_points() reports (which lists them in its own order)
    cloud = np.asarray(av.get_points()).reshape(-1, 4)[:, :3]

    def rows(a):
        return a[np.lexsort(a.T[::-1])]

    assert np.abs(rows(centres) - rows(cloud)).max() < 1e-12

    # and the notebook's formula over the walk anchor (cell 25), now that the
    # quenching model anchors its frame on the grid rather than the attachment
    m = bff.QuenchedDonorDecay(av, site["obstacles"], tau0=4.0, n_trajectories=1, random_seed=1,
                               t_max=1.0, **GRID_OPTIONS)
    m.simulate_diffusion()
    walk = m.get_diffusion()
    ijk = np.stack(np.unravel_index(points, (walk.ng,) * 3), 1)
    notebook = np.asarray(walk.get_x0()) + (ijk - (walk.ng - 1) // 2 + 0.5) * walk.dg
    assert np.abs(notebook - centres).max() < 1e-9
    assert np.abs(np.asarray(m.get_voxel_centres()) - centres).max() == 0.0
    # every voxel centre lies within the linker of the attachment atom
    cb = np.asarray(av.get_attachment_point())
    assert np.linalg.norm(centres - cb, axis=1).max() <= LINKER + 1e-9


# ----------------------------------------------------- neighbour counts --

def test_neighbour_counts_equal_the_slow_factor_map_trick(site):
    av, obstacles = site["av"], site["obstacles"]
    started = time.time()
    counts = np.asarray(bff.residue_type_neighbour_counts(av, obstacles, RESIDUES, SLOW_RADIUS))
    elapsed = time.time() - started
    points = np.asarray(av.get_voxel_indices())
    assert counts.shape == (points.size, len(RESIDUES))

    # notebook cell 25, step 1, verbatim
    reference = []
    for residue in RESIDUES:
        table = bff.amino_acid_quenching_defaults(1.0, 1.0, DYE_RADIUS)
        entry = table[residue]
        entry.slow_factor = 0.5
        table[residue] = entry
        m = bff.QuenchedDonorDecay(av, obstacles, tau0=4.0, quenching_table=table,
                                   n_trajectories=1, random_seed=1, t_max=1.0, **GRID_OPTIONS)
        m.update_grids()
        s = np.asarray(m.get_slow_factor_map(), dtype=float).ravel()
        reference.append(np.rint(np.log(s[points]) / np.log(0.5)).astype(np.int8))
    reference = np.stack(reference, 1)
    assert np.array_equal(counts.astype(np.int8), reference)
    assert counts.sum() > 0
    assert elapsed < 2.0


# -------------------------------------------------------------- exposure --

def water_exposure(centres, atoms):
    # notebook cell 25, verbatim (DYE_RADIUS 3.5, water 1.4, 64 points)
    xyz = np.array([a.coord for a in atoms], dtype=float)
    reach = np.array([VDW.get(a.element, 1.7) for a in atoms]) + 1.4
    i = np.arange(64) + 0.5
    phi, theta = np.arccos(1 - 2 * i / 64), np.pi * (1 + 5 ** 0.5) * i
    shell = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1) \
        * (DYE_RADIUS + 1.4)
    tree, covered = cKDTree(xyz), np.zeros((len(centres), len(shell)), bool)
    for k, offset in enumerate(shell):
        points = centres + offset
        for n, near in enumerate(tree.query_ball_point(points, reach.max())):
            if near:
                covered[n, k] = bool((np.linalg.norm(xyz[near] - points[n], axis=1)
                                      < reach[near]).any())
    return 1.0 - covered.mean(1)


def test_fibonacci_points_are_the_notebooks_shell():
    i = np.arange(64) + 0.5
    phi, theta = np.arccos(1 - 2 * i / 64), np.pi * (1 + 5 ** 0.5) * i
    shell = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    ours = np.asarray(bff.fibonacci_sphere_points(64)).reshape(-1, 3)
    assert np.abs(ours - shell).max() < 1e-12
    assert np.allclose(np.linalg.norm(ours, axis=1), 1.0)


def test_exposure_equals_the_notebook(site):
    av, atoms = site["av"], site["atoms"]
    centres = np.asarray(av.get_voxel_centres())
    xyz = np.array([a.coord for a in atoms], dtype=float)
    radii = np.array([VDW.get(a.element, 1.7) for a in atoms])
    started = time.time()
    exposure = np.asarray(bff.probe_solvent_exposure(centres.ravel(), xyz.ravel(), radii,
                                                     DYE_RADIUS, 1.4, 64))
    elapsed = time.time() - started
    assert exposure.shape == (centres.shape[0],)
    # the reference on a deterministic sample (all ~1e5 voxels take a minute in Python)
    rows = np.random.default_rng(72).choice(centres.shape[0], 4000, replace=False)
    reference = water_exposure(centres[rows], atoms)
    assert np.abs(exposure[rows] - reference).max() < 1e-6
    assert np.array_equal(exposure[rows].astype(np.float32), reference.astype(np.float32))
    # e is a fraction of 64; most reachable positions are in open water
    assert np.allclose(exposure * 64, np.rint(exposure * 64))
    assert 0.0 <= exposure.min() and exposure.max() == 1.0
    assert exposure.mean() > 0.7
    assert elapsed < 5.0


def test_exposure_edge_cases():
    # no atoms: open water everywhere
    out = np.asarray(bff.probe_solvent_exposure([0.0, 0.0, 0.0, 5.0, 0.0, 0.0], [], []))
    assert np.array_equal(out, [1.0, 1.0])
    # a probe inside a big atom is buried
    out = np.asarray(bff.probe_solvent_exposure([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0]))
    assert out[0] == 0.0
    with pytest.raises(Exception):
        bff.probe_solvent_exposure([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0])


# -------------------------------------------------------------- rotamers --

CHI_ATOMS = {"TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
             "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
             "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
             "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"))}
ONE_LETTER = {"TRP": "W", "TYR": "Y", "HIS": "H", "MET": "M"}
BACKBONE = {"N", "CA", "C", "O"}


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1n) * b1n, b2 - (b2 @ b1n) * b1n
    return np.degrees(np.arctan2(np.cross(b1n, v) @ w, v @ w))


def rotate(points, origin, axis, angle):
    axis = axis / np.linalg.norm(axis)
    a, p = np.radians(angle), points - origin
    return origin + p * np.cos(a) + np.cross(axis, p) * np.sin(a) \
        + np.outer(p @ axis, axis) * (1 - np.cos(a))


def set_chi(xyz, names, chi_atoms, target):
    xyz = xyz.copy()
    current = dihedral(*xyz[[names.index(a) for a in chi_atoms]])
    pivot = chi_atoms[2]
    chain = ["N", "CA", "CB", "CG"]
    moving = [k for k, n in enumerate(names)
              if n not in BACKBONE and n not in chain[: chain.index(pivot) + 1]]
    origin = xyz[names.index(pivot)]
    xyz[moving] = rotate(xyz[moving], origin, origin - xyz[names.index(chi_atoms[1])],
                         target - current)
    return xyz


def quencher_positions(structure, site, library):
    # notebook cell 25, verbatim but for the arguments
    reference = bff.amino_acid_quenching_defaults(1.0, 1.0, 3.5)
    model = structure[0]
    backbone = {}
    for piece in PPBuilder().build_peptides(model[CHAIN]):
        for residue, (phi, psi) in zip(piece, piece.get_phi_psi_list()):
            backbone[residue.id[1]] = (np.degrees(phi) if phi else -60.0,
                                       np.degrees(psi) if psi else -45.0)
    heavy = [a for a in model.get_atoms() if kept(a, site)]
    owner = np.array([a.get_parent().id[1] if a.get_parent().id[0] == " " else -1 for a in heavy])
    everything = np.array([a.coord for a in heavy])
    out = []
    for residue in model[CHAIN]:
        name = residue.resname
        if name not in QUENCHERS or residue.id[1] == site:
            continue
        names = [a.get_id() for a in residue]
        xyz = np.array([a.coord for a in residue])
        centre_atoms = [names.index(a) for a in reference[name].quench_atoms if a in names]
        crystal, ensemble = xyz[centre_atoms].mean(0), []
        if name in CHI_ATOMS and residue.id[1] in backbone:
            phi, psi = backbone[residue.id[1]]
            rot = bff.read_protein_sidechain_dunbrack_rotamers(library, ONE_LETTER[name], phi, psi)
            chis = np.asarray(rot.get_chi()).reshape(rot.n_rotamers, rot.n_chi)
            others = everything[owner != residue.id[1]]
            side = [k for k, n in enumerate(names) if n not in BACKBONE | {"CB"}]
            chi1, chi2 = CHI_ATOMS[name]
            for (x1, x2, *_), prob in zip(chis, np.asarray(rot.get_probability())):
                built = set_chi(set_chi(xyz, names, chi1, x1), names, chi2, x2)
                if np.min(np.linalg.norm(built[side][:, None] - others[None], axis=2)) > CLASH:
                    ensemble.append((built[centre_atoms].mean(0), float(prob)))
        total = sum(w for _, w in ensemble)
        from_library = bool(ensemble)
        ensemble = [(c, w / total) for c, w in ensemble] if ensemble else [(crystal, 1.0)]
        out.append(dict(res_id=residue.id[1], name=name, crystal=crystal, ensemble=ensemble,
                        from_library=from_library))
    return out


def test_rotamer_ensembles_equal_the_notebook(site):
    library = _library()
    o = site["obstacles"]
    started = time.time()
    ours = bff.quencher_rotamer_ensembles(o.chains, o.res_ids, o.res_names, o.atom_names,
                                          o.coords, QUENCHERS, library)
    elapsed = time.time() - started
    reference = quencher_positions(site["structure"], SITE, library)
    assert [(q.res_id, q.res_name) for q in ours] == [(r["res_id"], r["name"]) for r in reference]
    n_rotamers = 0
    for q, r in zip(ours, reference):
        assert np.abs(np.asarray(q.get_crystal()) - r["crystal"]).max() < 1e-5
        centres = np.asarray(q.get_centres())
        probabilities = np.asarray(q.get_probabilities())
        ref_c = np.array([c for c, _ in r["ensemble"]])
        ref_w = np.array([w for _, w in r["ensemble"]])
        assert centres.shape == ref_c.shape, (q.res_name, q.res_id)
        # the notebook builds in float32 (Bio.PDB coordinates); this is float64
        assert np.abs(centres - ref_c).max() < 2e-3, (q.res_name, q.res_id)
        assert np.abs(probabilities - ref_w).max() < 1e-6
        assert abs(probabilities.sum() - 1.0) < 1e-12
        assert q.from_library == r["from_library"]
        n_rotamers += len(probabilities)
    assert any(q.from_library for q in ours)
    assert n_rotamers > len(ours)
    assert elapsed < 10.0


def test_one_residue_and_its_refusals(site):
    library = _library()
    o = site["obstacles"]
    trp = next(i for i, n in zip(o.res_ids, o.res_names) if n == "TRP")
    one = bff.quencher_rotamer_ensemble(o.chains, o.res_ids, o.res_names, o.atom_names, o.coords,
                                        CHAIN, trp, library, ["CD1", "CD2", "NE1", "CE2", "CE3",
                                                              "CZ2", "CZ3", "CH2", "CG"])
    assert one.res_name == "TRP" and one.res_id == trp
    assert -180.0 <= one.phi <= 180.0 and -180.0 <= one.psi <= 180.0
    with pytest.raises(Exception):
        bff.quencher_rotamer_ensemble(o.chains, o.res_ids, o.res_names, o.atom_names, o.coords,
                                      CHAIN, 99999, library, ["CG"])
    # the labelled site is cut back to CB: it has no quenching atoms left
    with pytest.raises(Exception):
        bff.quencher_rotamer_ensemble(o.chains, o.res_ids, o.res_names, o.atom_names, o.coords,
                                      CHAIN, SITE, library, ["OG", "SD", "NE2"])


# --------------------------------------------------------- linker length --

def stretched_linker_length_notebook(library_name, linker_residue, dye_residue):
    # notebook cell 21, verbatim
    library = bff.load_probe_rotamer_library(library_name)
    names, elements, residues = (list(library.atom_names), [e.upper() for e in library.elements],
                                 list(library.resnames))
    xyz = np.asarray(library.get_coords()).reshape(library.n_rotamers, library.n_atoms, 3)[0]
    n = len(names)
    covalent = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "H": 0.31}
    neighbours = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(xyz[i] - xyz[j]) < covalent.get(elements[i], .8) \
                    + covalent.get(elements[j], .8) + 0.45:
                neighbours[i].append(j)
                neighbours[j].append(i)

    def path(a, b):
        previous, queue = {a: None}, [a]
        while queue:
            u = queue.pop(0)
            for v in neighbours[u]:
                if v not in previous:
                    previous[v] = u
                    queue.append(v)
        out = [b]
        while previous[out[-1]] is not None:
            out.append(previous[out[-1]])
        return out[::-1]

    def in_ring(a, b):
        seen, queue = {a}, [a]
        while queue:
            u = queue.pop(0)
            for v in neighbours[u]:
                if (u, v) in ((a, b), (b, a)) or v in seen:
                    continue
                if v == b:
                    return True
                seen.add(v)
                queue.append(v)
        return False

    ca = [i for i, (a, r) in enumerate(zip(names, residues)) if a == "CA" and r == linker_residue][0]
    cb = [v for v in neighbours[ca]
          if elements[v] == "C" and any(elements[w] == "S" for w in neighbours[v])][0]
    ring_atoms = {i for i in range(n) for j in neighbours[i]
                  if residues[i] == dye_residue and in_ring(i, j)}
    systems, left = [], set(ring_atoms)
    while left:
        seed = left.pop()
        group, queue = {seed}, [seed]
        while queue:
            u = queue.pop(0)
            for v in neighbours[u]:
                if v in left and in_ring(u, v):
                    left.discard(v)
                    group.add(v)
                    queue.append(v)
        systems.append(group)
    chromophore = max(systems, key=len)
    centre = xyz[[i for i in chromophore if elements[i] != "H"]].mean(0)
    entry = min(chromophore, key=lambda i: len(path(cb, i)))
    route, length, i = path(cb, entry), 0.0, 0
    while i < len(route) - 1:
        u, v = route[i], route[i + 1]
        if in_ring(u, v):
            j = i
            while j + 1 < len(route) and in_ring(route[j], route[j + 1]):
                j += 1
            length += np.linalg.norm(xyz[route[j]] - xyz[u])
            i = j
        else:
            length += np.linalg.norm(xyz[v] - xyz[u]) * np.sin(np.radians(109.5) / 2)
            i += 1
    return length + np.linalg.norm(centre - xyz[entry])


def test_stretched_af488_linker_is_20_4():
    ours = bff.stretched_linker_length("AlexaFluor 488 C1R", "C1R", "A48")
    assert round(ours, 1) == 20.4
    assert abs(ours - stretched_linker_length_notebook("AlexaFluor 488 C1R", "C1R", "A48")) < 1e-9
    lib = bff.load_probe_rotamer_library("AlexaFluor 488 C1R")
    assert bff.stretched_linker_length(lib, "C1R", "A48") == ours
    with pytest.raises(Exception):
        bff.stretched_linker_length(lib, "XXX", "A48")
    with pytest.raises(Exception):
        bff.stretched_linker_length(lib, "C1R", "XXX")


# ------------------------------------------- the prototype's own site table --

PROTOTYPE = HERE.parent / "input" / "quenching" / "male_1omp72_L20.12_prototype.npz"


def test_the_prototype_site_table_is_reproduced(site):
    """Bit for bit, on the prototype's own volume and in its own frame.

    ``male_1omp72_L20.12_prototype.npz`` holds the notebook's cached site
    table for 1OMP / 72 / L 20.12 (``neighbours``, ``exposure``; chisurf
    ``quenching_estimator/test/data/male_site_1omp72_L20.12.npz``) and the
    volume it was built on: IMP.bff fceece9ed's ``get_av_from_pdb`` with the
    notebook's arguments (97 722 voxels; today's defaults give a different
    volume, see okf/log.md 2026-10-20). The prototype placed voxel ``i`` at
    ``attachment + (i - (ng - 1) // 2 + 0.5) * dg``, half a voxel or less
    from where the grid puts it; rebuilding the volume with its grid origin
    there reproduces that frame, so both kernels must give the table exactly.
    """
    data = np.load(PROTOTYPE)
    ng, dg = int(data["ng"]), float(data["dg"])
    attachment = np.asarray(data["attachment"], dtype=float)
    density = np.zeros(ng ** 3)
    density[data["voxel_indices"]] = 1.0
    origin = attachment - ((ng - 1) // 2 - 0.5) * dg
    av = bff.ProbeAccessibleVolume([], density, list(origin), dg, "prototype", list(attachment))
    assert np.abs(np.asarray(av.get_walk_anchor()) - attachment).max() < 1e-12
    assert np.array_equal(np.asarray(av.get_voxel_indices()), data["voxel_indices"])

    counts = np.asarray(bff.residue_type_neighbour_counts(av, site["obstacles"], RESIDUES,
                                                          SLOW_RADIUS))
    assert np.array_equal(counts.astype(np.int8), data["neighbours"])

    atoms = site["atoms"]
    xyz = np.array([a.coord for a in atoms], dtype=float)
    radii = np.array([VDW.get(a.element, 1.7) for a in atoms])
    exposure = np.asarray(bff.probe_solvent_exposure(
        np.asarray(av.get_voxel_centres()).ravel(), xyz.ravel(), radii, DYE_RADIUS, 1.4, 64))
    assert np.abs(exposure.astype(np.float32) - data["exposure"]).max() < 1e-6


def test_stamped_spheres_sit_on_the_nearest_voxel():
    """A residue's sphere is centred on the voxel nearest to it, in the grid's frame.

    The quenching model used to floor centres in a frame anchored at the
    attachment point, which is not where the grid's voxels are when the grid
    is snapped to its own lattice: every sphere sat up to a voxel off, in the
    same direction for every residue. Here the attachment point is deliberately
    off the lattice."""
    ng, dg = 41, 0.5
    origin = np.array([-10.0, -10.0, -10.0])
    attachment = origin + ((ng - 1) // 2) * dg + np.array([0.21, -0.17, 0.08])
    av = bff.ProbeAccessibleVolume([], np.ones(ng ** 3), list(origin), dg, "cube",
                                   list(attachment))
    centres = np.asarray(av.get_voxel_centres())
    for r in ([0.37, -0.12, 0.61], [-2.13, 1.07, 0.0], [3.3, 3.3, -3.3]):
        atoms = bff.ObstacleAtoms()
        atoms.chains, atoms.res_ids, atoms.res_names, atoms.atom_names = ["A"], [1], ["LYS"], ["CB"]
        atoms.coords = [float(v) for v in r]
        counts = np.asarray(bff.residue_type_neighbour_counts(av, atoms, ["LYS"], 3.0))[:, 0]
        inside = centres[counts > 0]
        nearest = centres[np.argmin(np.linalg.norm(centres - np.asarray(r), axis=1))]
        assert np.abs(inside.mean(0) - nearest).max() < 1e-9
        # the integer-index ball: |di|^2 + |dj|^2 + |dk|^2 < (3.0 / dg)^2
        assert counts.sum() == sum(1 for i in range(-6, 7) for j in range(-6, 7)
                                   for k in range(-6, 7) if i * i + j * j + k * k < 36)
