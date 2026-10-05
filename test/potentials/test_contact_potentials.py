"""One kernel, two doors: the coarse-grained contact terms as IMP restraints
and as array functions, on T4 lysozyme (148L).

Every term is evaluated three ways and they must agree:

* as an IMP object -- a `PairScore` over an `IMP.container.ClosePairContainer`
  in a `PairsRestraint`, or a `Restraint` -- inside
  `IMP.core.RestraintsScoringFunction`;
* through the array function of `ContactPotentials.h` on the same coordinates;
* against the numba kernel the term was ported from, kept verbatim in the
  sibling imp-tricks checkout (`tests/numba_oracles`); skipped where that
  checkout or numba is absent.

The first two share one C++ kernel (`internal/ContactKernels.h`), so a
difference between them can only come from the gate -- which pairs are
examined -- and each test sets the gates up to be the same set.
"""

import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest

import IMP
import IMP.algebra
import IMP.atom
import IMP.container
import IMP.core
import IMP.bff

RTOL = 1e-10

#: chisurf's `res2id` order, which is Miyazawa and Jernigan's.
MJ_ORDER = ["CYS", "MET", "PHE", "ILE", "LEU", "VAL", "TRP", "TYR", "ALA",
            "GLY", "THR", "SER", "GLN", "ASN", "GLU", "ASP", "HIS", "ARG",
            "LYS", "PRO"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _pdb_148l():
    here = pathlib.Path(__file__).resolve()
    for root in here.parents:
        p = root / "data" / "cgprobe" / "inputs" / "structures" / "148L.pdb"
        if p.exists():
            return str(p)
    pytest.skip("148L.pdb is not in this checkout")


def _oracle(name):
    """The numba original of a kernel, from the imp-tricks checkout."""
    pytest.importorskip("numba")
    here = pathlib.Path(__file__).resolve()
    for root in here.parents:
        d = root.parent / "imp-tricks" / "tests" / "numba_oracles"
        if (d / f"{name}.py").exists():
            spec = importlib.util.spec_from_file_location(
                f"_oracle_{name}", d / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("the imp-tricks numba oracles are not beside this checkout")


@pytest.fixture(scope="module")
def t4l():
    """148L chain E (T4 lysozyme), heavy atoms, with amide hydrogens placed."""
    m = IMP.Model()
    h = IMP.atom.read_pdb(_pdb_148l(), m,
                          IMP.atom.ChainPDBSelector(["E"])
                          & IMP.atom.NonWaterNonHydrogenPDBSelector())
    residues = IMP.atom.get_by_type(h, IMP.atom.RESIDUE_TYPE)
    prev_c = None
    for r in residues:
        atoms = {IMP.atom.Atom(a).get_atom_type().get_string(): a
                 for a in IMP.atom.Hierarchy(r).get_children()}
        n, ca = atoms.get("N"), atoms.get("CA")
        name = IMP.atom.Residue(r).get_residue_type().get_string()
        if prev_c is not None and n is not None and ca is not None and name != "PRO":
            vn = IMP.core.XYZ(n).get_coordinates()
            u = (vn - IMP.core.XYZ(prev_c).get_coordinates()).get_unit_vector() + \
                (vn - IMP.core.XYZ(ca).get_coordinates()).get_unit_vector()
            p = IMP.Particle(m, "H")
            IMP.atom.Atom.setup_particle(p, IMP.atom.AT_H)
            IMP.core.XYZ.setup_particle(p, vn + u.get_unit_vector() * 1.01)
            IMP.atom.Hierarchy(r).add_child(IMP.atom.Atom(p))
        prev_c = atoms.get("C")
    rng = np.random.default_rng(148)
    for a in IMP.atom.get_by_type(h, IMP.atom.ATOM_TYPE):
        p = a.get_particle()
        if not IMP.core.XYZR.get_is_setup(p):
            IMP.core.XYZR.setup_particle(p, float(rng.uniform(1.2, 2.0)))
        else:
            IMP.core.XYZR(p).set_radius(float(rng.uniform(1.2, 2.0)))
        q = 0.0 if rng.random() < 0.3 else float(rng.normal())
        if IMP.atom.Charged.get_is_setup(p):
            IMP.atom.Charged(p).set_charge(q)
        else:
            IMP.atom.Charged.setup_particle(p, q)
    return m, h


def _atoms(h):
    return [a.get_particle() for a in IMP.atom.get_by_type(h, IMP.atom.ATOM_TYPE)]


def _xyz(ps):
    return np.ascontiguousarray(
        [list(IMP.core.XYZ(p).get_coordinates()) for p in ps], dtype=np.float64)


def _sf(r):
    return IMP.core.RestraintsScoringFunction([r]).evaluate(False)


def _site_atoms(h, atom_type):
    """One atom of a given type per residue, in sequence; None where absent."""
    out = []
    for r in IMP.atom.get_by_type(h, IMP.atom.RESIDUE_TYPE):
        hit = None
        for a in IMP.atom.Hierarchy(r).get_children():
            if IMP.atom.Atom(a).get_atom_type() == atom_type:
                hit = a.get_particle()
        out.append(hit)
    return out


# ---------------------------------------------------------------------------
# Soft-sphere overlap
# ---------------------------------------------------------------------------

def test_soft_sphere_overlap_restraint_array_and_oracle_agree(t4l):
    m, h = t4l
    ps = _atoms(h)
    xyz = _xyz(ps)
    radii = np.array([IMP.core.XYZR(p).get_radius() for p in ps])
    array = IMP.bff.get_soft_sphere_overlap_energy(xyz, radii, 2.0, 1.5)
    assert array > 0.0

    # Sphere distance below 0 is exactly "the spheres overlap".
    cpc = IMP.container.ClosePairContainer(
        IMP.container.ListSingletonContainer(m, [p.get_index() for p in ps]),
        0.0, 1.0)
    r = IMP.container.PairsRestraint(
        IMP.bff.SoftSphereOverlapPairScore(2.0, 1.5), cpc)
    assert _sf(r) == pytest.approx(array, rel=RTOL)

    # The legacy array function is the same kernel.
    assert IMP.bff.clash_energy(xyz.ravel().tolist(), radii.tolist(), 2.0,
                                1.5) == pytest.approx(array, rel=RTOL)

    oracle = _oracle("cgmol_sterics")
    assert oracle._clash_kernel(xyz, radii, 2.0, 1.5) == pytest.approx(
        array, rel=RTOL)


def test_soft_sphere_overlap_derivatives_match_a_difference():
    m = IMP.Model()
    a = IMP.core.XYZR.setup_particle(
        IMP.Particle(m), IMP.algebra.Sphere3D(IMP.algebra.Vector3D(0, 0, 0), 1.8))
    b = IMP.core.XYZR.setup_particle(
        IMP.Particle(m), IMP.algebra.Sphere3D(IMP.algebra.Vector3D(2.1, 0.7, -0.4), 1.6))
    for d in (a, b):
        d.set_coordinates_are_optimized(True)
    score = IMP.bff.SoftSphereOverlapPairScore(2.0, 1.5)
    r = IMP.core.PairRestraint(m, score, (a.get_particle_index(), b.get_particle_index()))
    sf = IMP.core.RestraintsScoringFunction([r])
    sf.evaluate(True)
    analytic = a.get_derivatives()
    eps = 1e-6
    for k in range(3):
        v = a.get_coordinates()
        hi, lo = IMP.algebra.Vector3D(v), IMP.algebra.Vector3D(v)
        hi[k] += eps
        lo[k] -= eps
        a.set_coordinates(hi)
        e_hi = sf.evaluate(False)
        a.set_coordinates(lo)
        e_lo = sf.evaluate(False)
        a.set_coordinates(v)
        assert analytic[k] == pytest.approx((e_hi - e_lo) / (2 * eps), rel=1e-6)


# ---------------------------------------------------------------------------
# Typed contact tables
# ---------------------------------------------------------------------------

def _write_pmf(path, bin_width, values):
    """A (20, 20, n_bins) table, MJ order, as IMP's PMF format."""
    n_bins = values.shape[2]
    with open(path, "w") as f:
        f.write(f"{bin_width!r} {len(MJ_ORDER)}\n")
        for i, a in enumerate(MJ_ORDER):
            for j in range(i, len(MJ_ORDER)):
                f.write(a + " " + MJ_ORDER[j] + " " +
                        " ".join(repr(float(v)) for v in values[i, j]) + "\n")
    assert n_bins > 0


def _typed_sites(m, h, atom_type, table_path_reader):
    """The typed representatives, their coordinates and their type indices."""
    typed = IMP.bff.add_residue_type_score_data(h, atom_type)
    key = IMP.bff.get_residue_type_key()
    xyz = _xyz(typed)
    types = np.array([p.get_value(key) for p in typed], dtype=np.int32)
    return typed, xyz, types


def _key_order_matrix(mj_values):
    """Re-index a table from MJ order into the residue-type key's order."""
    idx = np.array([IMP.bff.ResidueContactType(n).get_index() for n in MJ_ORDER])
    n = int(idx.max()) + 1
    out = np.zeros((n, n) + mj_values.shape[2:])
    for a in range(20):
        for b in range(20):
            out[idx[a], idx[b]] = mj_values[a, b]
    return out


def test_mj_contact_restraint_array_and_oracle_agree(t4l, tmp_path):
    m, h = t4l
    a = np.array([[((i * 31 + j * 17) % 41 - 20) / 10.0 for j in range(20)]
                  for i in range(20)])
    mj = (a + a.T) / 2.0
    table = tmp_path / "mj.pmf"
    _write_pmf(table, 3.25, np.repeat(mj[:, :, None], 2, axis=2))

    typed, xyz, types = _typed_sites(m, h, IMP.atom.AT_CB, None)
    cpc = IMP.container.ClosePairContainer(
        IMP.container.ListSingletonContainer(m, [p.get_index() for p in typed]),
        6.5, 0.0)
    r = IMP.container.PairsRestraint(
        IMP.bff.MiyazawaJerniganPairScore(6.5, str(table)), cpc)
    restraint = _sf(r)

    n = len(typed)
    no_gate = np.zeros((n, n))
    n_c, e = IMP.bff.get_site_contact_energy(
        xyz, np.arange(n, dtype=np.int32), types, no_gate, 1.0,
        _key_order_matrix(mj), 6.5)
    assert n_c > 50
    assert e == pytest.approx(restraint, rel=RTOL)

    oracle = _oracle("cgmol_statpot")
    mj_types = np.array([MJ_ORDER.index(
        IMP.atom.Residue(IMP.atom.Atom(p).get_parent()).get_residue_type()
        .get_string()) for p in typed])
    look = np.full((n, 6), -1, dtype=np.int64)
    look[:, 4] = np.arange(n)
    big = np.zeros((n, n))   # the oracle's C-alpha gate, opened
    n_o, e_o = oracle._mj_kernel(look, mj_types, big, xyz, mj, 6.5, 4)
    assert n_o == n_c
    assert e_o == pytest.approx(e, rel=RTOL)


def test_unres_binned_restraint_array_and_oracle_agree(t4l, tmp_path):
    m, h = t4l
    rng = np.random.default_rng(7)
    bw, n_bins, d_min, d_max, rep = 0.05, 380, 3.5, 19.0, 100.0
    pot = rng.normal(size=(20, 20, n_bins))
    pot = (pot + pot.transpose(1, 0, 2)) / 2.0
    pot[:, :, : int(d_min / bw)] = rep      # the repulsion lives in the table
    table = tmp_path / "unres.pmf"
    _write_pmf(table, bw, pot)

    typed, xyz, types = _typed_sites(m, h, IMP.atom.AT_CB, None)
    n = len(typed)
    cpc = IMP.container.ClosePairContainer(
        IMP.container.ListSingletonContainer(m, [p.get_index() for p in typed]),
        d_max, 0.0)
    r = IMP.container.PairsRestraint(
        IMP.bff.UNRESCentroidPairScore(d_max, str(table)), cpc)
    restraint = _sf(r)

    # The restraint drops pairs at or past d_max; gate the array the same way.
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    keyed = _key_order_matrix(pot)
    n_p, e = IMP.bff.get_site_binned_pair_energy(
        xyz, np.arange(n, dtype=np.int32), types, np.ascontiguousarray(d),
        d_max - 1e-9, keyed.ravel(), keyed.shape[0], n_bins, d_min, d_max, bw,
        rep)
    assert n_p > 1000
    assert e == pytest.approx(restraint, rel=RTOL)

    oracle = _oracle("cgmol_statpot")
    mj_types = np.array([MJ_ORDER.index(
        IMP.atom.Residue(IMP.atom.Atom(p).get_parent()).get_residue_type()
        .get_string()) for p in typed])
    look = np.full((n, 6), -1, dtype=np.int64)
    look[:, 4] = np.arange(n)
    n_o, e_o = oracle.centroid2(look, mj_types, d, xyz, pot, d_max - 1e-9, 4,
                                d_min, d_max, bw, rep)
    assert n_o == n_p
    assert e_o == pytest.approx(e, rel=RTOL)


# ---------------------------------------------------------------------------
# Hydrogen bonds
# ---------------------------------------------------------------------------

def _hbond_table(n_bins=400):
    return np.array([[math.sin(0.02 * b + 0.5 * c) for b in range(n_bins)]
                     for c in range(4)])


def test_hbond_restraint_array_and_oracle_agree(t4l):
    m, h = t4l
    table = _hbond_table()
    r = IMP.bff.HydrogenBondRestraint(m, h, table.ravel().tolist(), 400, 8.0,
                                      3.0, 0.01)
    restraint = _sf(r)
    assert r.get_n_hbonds() > 50

    # The same atoms laid out as the restraint does: per residue with a CA,
    # the CA first, then whichever of N, C, O, H it has.
    sites = {t: _site_atoms(h, getattr(IMP.atom, "AT_" + t))
             for t in ("CA", "N", "C", "O", "H")}
    ps, idx = [], {t: [] for t in ("CA", "N", "C", "O", "H")}
    for k in range(len(sites["CA"])):
        if sites["CA"][k] is None:
            continue
        for t in ("CA", "N", "C", "O", "H"):
            p = sites[t][k]
            if p is None:
                idx[t].append(-1)
            else:
                idx[t].append(len(ps))
                ps.append(p)
    xyz = _xyz(ps)
    ca = xyz[idx["CA"]]
    ca_d2 = np.ascontiguousarray(((ca[:, None] - ca[None]) ** 2).sum(-1))
    a32 = {t: np.array(idx[t], dtype=np.int32) for t in idx}
    n_b, e = IMP.bff.get_backbone_hbond_energy(
        xyz, a32["N"], a32["C"], a32["O"], a32["H"], ca_d2, 64.0, table, 9.0,
        0.01)
    assert n_b == r.get_n_hbonds()
    assert e == pytest.approx(restraint, rel=RTOL)

    oracle = _oracle("cgmol_statpot")
    look = np.stack([a32["N"], a32["CA"], a32["C"], a32["O"],
                     np.full(len(ca), -1, dtype=np.int32), a32["H"]], axis=1)
    n_o, e_o = oracle._hbond_kernel(look.astype(np.int64), ca_d2, xyz, table,
                                    64.0, 9.0, 0.01)
    assert n_o == n_b
    assert e_o == pytest.approx(e, rel=RTOL)


# ---------------------------------------------------------------------------
# Go model
# ---------------------------------------------------------------------------

def test_go_restraint_array_and_oracle_agree(t4l):
    m, h = t4l
    cas = [p for p in _site_atoms(h, IMP.atom.AT_CA) if p is not None]
    native = _xyz(cas)
    r = IMP.bff.GoRestraint(m, [p.get_index() for p in cas], 1.0, 6.5, 0.1)

    rng = np.random.default_rng(3)
    moved = native + rng.normal(scale=0.8, size=native.shape)
    for p, v in zip(cas, moved):
        IMP.core.XYZ(p).set_coordinates(IMP.algebra.Vector3D(*v))
    try:
        restraint = _sf(r)
        oracle = _oracle("cgmol_sterics")
        d_native = np.sqrt(((native[:, None] - native[None]) ** 2).sum(-1))
        e_mat, rm_mat = oracle._go_init(d_native, 1.0, 0.1, 6.5)
        d_moved = np.ascontiguousarray(
            np.sqrt(((moved[:, None] - moved[None]) ** 2).sum(-1)))
        array = IMP.bff.get_matrix_lj_well_energy(d_moved, e_mat, rm_mat)
        assert array == pytest.approx(restraint, rel=RTOL)
        assert oracle._go_kernel(d_moved, e_mat, rm_mat) == pytest.approx(
            array, rel=RTOL)
    finally:
        for p, v in zip(cas, native):
            IMP.core.XYZ(p).set_coordinates(IMP.algebra.Vector3D(*v))


# ---------------------------------------------------------------------------
# Generalized Born
# ---------------------------------------------------------------------------

def test_generalized_born_restraint_array_and_oracle_agree(t4l):
    m, h = t4l
    ps = _atoms(h)
    r = IMP.bff.GeneralizedBornRestraint(m, [p.get_index() for p in ps], 4.0,
                                         80.1, 12.0)
    restraint = _sf(r)
    xyz = _xyz(ps)
    radii = np.array([IMP.core.XYZR(p).get_radius() for p in ps])
    charges = np.array([IMP.atom.Charged(p).get_charge() for p in ps])
    # The reference term at the *unsquared* cutoff, as the restraint keeps it.
    array = IMP.bff.get_generalized_born_energy(xyz, radii, charges, 4.0, 80.1,
                                                12.0, 12.0)
    assert array == pytest.approx(restraint, rel=RTOL)

    oracle = _oracle("cgmol_gb")
    assert oracle._gb_kernel(xyz, radii, charges, 4.0, 80.1, 12.0) == \
        pytest.approx(array, rel=RTOL)


# ---------------------------------------------------------------------------
# Accessible area
# ---------------------------------------------------------------------------

def test_site_area_restraint_array_and_oracle_agree(t4l):
    m, h = t4l
    cas = [p for p in _site_atoms(h, IMP.atom.AT_CA) if p is not None]
    r = IMP.bff.SiteAccessibleAreaRestraint(m, [p.get_index() for p in cas],
                                            92, 1.0, 2.5, 7.0)
    restraint = _sf(r)
    xyz = _xyz(cas)
    n = len(cas)
    d2 = np.ascontiguousarray(((xyz[:, None] - xyz[None]) ** 2).sum(-1))
    points = np.asarray(IMP.bff.sphere_points(92)).reshape(-1, 3)
    array = IMP.bff.get_site_accessible_area(
        xyz, np.arange(n, dtype=np.int32), d2, 49.0, points, 1.0, 2.5)
    assert array == pytest.approx(restraint, rel=RTOL)

    oracle = _oracle("cgmol_sterics")
    np.testing.assert_allclose(oracle._sphere_points(92), points, rtol=1e-14,
                               atol=1e-15)
    look = np.full((n, 7), -1, dtype=np.int64)
    look[:, 6] = np.arange(n)
    assert oracle._asa_kernel(xyz, look, d2, points, 1.0, 2.5) == \
        pytest.approx(array, rel=RTOL)
