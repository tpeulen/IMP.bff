"""The diffusion PDE with a potential: detailed balance, grid lifetimes, donor decay.

QuEst keeps three dye ensembles. The equilibrium one weighs each voxel by
``w = prod_t s_t^(-n_t)`` (a sticky residue group slows the dye *and* holds
it). The PDE ensemble has to reach that same equilibrium while the mobility
stays a separate input -- the plain Smoluchowski flux is uniform at
equilibrium, and the Ito flux reaches ``1/D`` only by tying mobility to it.
``flux_form="detailed_balance"`` is the Smoluchowski equation with a
potential:

    J_cm = 0.5 (D_c + D_m) sqrt(w_c w_m) (p_c / w_c - p_m / w_m)

What is checked, on a toy grid and on the MalE 1OMP site-72 accessible volume
(``test/input/quenching/male_1omp72_L20.12_prototype.npz``, the prototype's
97 722-voxel AV and its per-voxel residue-type neighbour counts):

* with no sink the weight field is the stationary state -- a fixed point of
  the explicit step, of the Krylov exponential, and the state an arbitrary
  start relaxes onto; ``equilibrium_occupancy`` and
  ``get_stationary_occupancy`` give it in closed form;
* population is conserved, and ``w = 1`` is the Smoluchowski form;
* the adjoint of the stepped scheme is its transpose (dot-product identity);
* the explicit, Krylov and linear-solve routes agree with each other;
* D -> infinity gives the equilibrium-averaged rate, D -> 0 the static
  (no-transport) average;
* a long Brownian walk started from p(x) gives the same donor lifetime at
  slow factors 1 and 0.3 (``expensive_test_detailed_balance_walk.py``, which
  uses ``walk_lifetime`` below).
"""

import pathlib

import numpy as np
import pytest

import IMP.bff as bff

HERE = pathlib.Path(__file__).resolve().parent
MALE = HERE.parent / "input" / "quenching" / "male_1omp72_L20.12_prototype.npz"
TAU0 = 4.0  # ns

# Column order of the fixture's neighbour counts (test_quest_site_kernels.py).
RESIDUES = ["TRP", "TYR", "HIS", "PHE", "LEU", "ILE", "VAL", "MET", "ALA", "PRO",
            "LYS", "ARG", "ASP", "GLU", "SER", "THR", "ASN", "GLN", "GLY", "CYS"]
# A sticky group (aromatic + charged) and QuEst's quenchers less HIS; rates
# chosen so the mean quenching rate is of the order of 1/tau0.
STICKY = [RESIDUES.index(r) for r in ("TRP", "TYR", "HIS", "LYS", "ARG", "ASP", "GLU")]
QUENCH = [RESIDUES.index(r) for r in ("TRP", "TYR", "MET", "PRO")]
K_CONTACT = 1.0  # 1/ns per quencher in range


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _toy(ng=17, slow=0.3, d0=1.0, seed=5):
    """A ball with three sticky centres: n_t counts centres within 3 voxels."""
    rng = np.random.default_rng(seed)
    i = np.arange(ng) - ng // 2
    X, Y, Z = np.meshgrid(i, i, i, indexing="ij")
    inside = (X ** 2 + Y ** 2 + Z ** 2) <= (ng // 2 - 2) ** 2
    centres = np.array([[3, 0, 0], [-2, 3, 1], [0, -3, -3]])
    n = np.zeros(X.shape)
    for c in centres:
        n += ((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) <= 9
    n *= inside
    w = slow ** (-n)
    d = d0 * (0.6 + 0.8 * rng.random(X.shape)) * inside
    k = (0.05 + 1.5 * (n > 0)) * inside
    flat = [a.ravel() for a in (d, inside.astype(float), w, k)]
    return ng, flat


def _simpson(f, t):
    """Composite Simpson on an even number of equal intervals."""
    h = t[1] - t[0]
    return h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())


def _solver(ng, d, b, w, k, p0=None, dg=1.0, t_step=0.02, form="detailed_balance"):
    p0 = w * b if p0 is None else p0
    return bff.GridDiffusionSolver(d, b, p0, k, dg, t_step, form, True, w)


@pytest.fixture(scope="module")
def male():
    data = np.load(MALE)
    ng0, dg = int(data["ng"]), float(data["dg"])
    # The prototype's volume reaches the edge of its 81^3 grid, where the
    # 7-point stencil has no neighbours; one empty voxel of padding each side.
    ng = ng0 + 2
    ijk = np.stack(np.unravel_index(data["voxel_indices"], (ng0,) * 3), axis=1) + 1
    idx = np.ravel_multi_index(ijk.T, (ng,) * 3)
    b = np.zeros(ng ** 3)
    b[idx] = 1.0
    counts = np.zeros((ng ** 3, len(RESIDUES)))
    counts[idx] = data["neighbours"]
    n_sticky = counts[:, STICKY].sum(1)
    n_quench = counts[:, QUENCH].sum(1)
    return dict(ng=ng, dg=dg, b=b, n=n_sticky * b, k=K_CONTACT * n_quench * b)


def _male_fields(m, slow, d0=10.0):
    """Mobility D0 s^n and equilibrium weight s^-n: the walk's own model."""
    w = np.where(m["b"] > 0, slow ** (-m["n"]), 1.0)
    d = d0 * slow ** m["n"] * m["b"]
    return d, w


# --------------------------------------------------------------------------
# 1. the stationary state is the weight field
# --------------------------------------------------------------------------
class TestStationary:
    def test_weight_is_a_fixed_point_of_the_explicit_step(self):
        ng, (d, b, w, k) = _toy()
        s = _solver(ng, d, b, w, np.zeros_like(k))
        r = s.run(2000, 500)
        p = w * b / (w * b).sum()
        rel = np.abs(np.asarray(r.get_density()) - p).max() / p.max()
        assert rel < 1e-13, rel
        assert np.abs(np.asarray(r.get_fluorescence()) - 1.0).max() < 1e-12

    def test_an_arbitrary_start_relaxes_onto_the_weight_field(self):
        ng, (d, b, w, k) = _toy()
        p = w * b / (w * b).sum()
        uniform = b / b.sum()
        # Krylov: one exponential at t = 2000 ns, far past the slowest mode.
        s = _solver(ng, d, b, w, np.zeros_like(k), p0=uniform, t_step=1.0)
        r = s.run_krylov(2000, 2000, 256)
        assert np.abs(np.asarray(r.get_density()) - p).max() / p.max() < 1e-8
        # The stepped scheme gets there too.
        s = _solver(ng, d, b, w, np.zeros_like(k), p0=uniform)
        eq = np.asarray(s.equilibrium(400000, 1e-13, 1000))
        assert np.abs(eq - p).max() / p.max() < 1e-9

    def test_closed_forms(self):
        ng, (d, b, w, k) = _toy()
        p = w * b / (w * b).sum()
        s = _solver(ng, d, b, w, k)
        np.testing.assert_allclose(np.asarray(s.get_stationary_occupancy()), p, rtol=1e-14, atol=0)
        eq = np.asarray(bff.equilibrium_occupancy(d, (b > 0).astype(int), "detailed_balance", w))
        np.testing.assert_allclose(eq, p, rtol=1e-14, atol=0)
        # No weight: Smoluchowski's uniform answer.
        eq = np.asarray(bff.equilibrium_occupancy(d, (b > 0).astype(int), "detailed_balance"))
        np.testing.assert_allclose(eq, b / b.sum(), rtol=1e-14, atol=0)

    def test_a_zero_weight_in_the_domain_is_refused(self):
        ng, (d, b, w, k) = _toy()
        bad = w.copy()
        bad[np.flatnonzero(b)[0]] = 0.0
        s = _solver(ng, d, b, bad, k, p0=b)
        with pytest.raises(ValueError):
            s.run(1, 1)
        with pytest.raises(ValueError):
            bff.equilibrium_occupancy(d, (b > 0).astype(int), "detailed_balance", bad)

    def test_male_weight_is_stationary(self, male):
        """On the real AV, at the sticky regime where the walk cannot go."""
        for slow in (1.0, 0.3, 0.1):
            d, w = _male_fields(male, slow)
            b = male["b"]
            p = w * b / (w * b).sum()
            # dg^2 / max sum D_cm sqrt(w_m/w_c): the solver's own bound.
            s = bff.GridDiffusionSolver(d, b, p, np.zeros_like(b), male["dg"], 5e-4,
                                        "detailed_balance", True, w)
            r = s.run(50, 50)
            dens = np.asarray(r.get_density())
            # The solver normalises the start with a serial sum over 97 722
            # voxels (1e-12 off numpy's pairwise one): compare shapes.
            dens = dens / dens.sum()
            nz = p > 0
            rel = np.abs(dens[nz] - p[nz]) / p[nz]
            assert rel.max() < 1e-12, (slow, rel.max())
            assert abs(np.asarray(r.get_fluorescence())[-1] - 1.0) < 1e-11


# --------------------------------------------------------------------------
# 2. conservation, the w = 1 limit, the adjoint
# --------------------------------------------------------------------------
class TestScheme:
    def test_mass_is_conserved_from_any_start(self):
        ng, (d, b, w, k) = _toy()
        rng = np.random.default_rng(1)
        p0 = b * rng.random(b.size)
        s = _solver(ng, d, b, w, np.zeros_like(k), p0=p0)
        f = np.asarray(s.run(3000, 300).get_fluorescence())
        assert np.abs(f - 1.0).max() < 1e-12
        f = np.asarray(s.run_krylov(3000, 300, 128).get_fluorescence())
        assert np.abs(f - 1.0).max() < 1e-10

    def test_uniform_weight_is_smoluchowski(self):
        ng, (d, b, w, k) = _toy()
        p0 = b / b.sum()
        a = _solver(ng, d, b, np.ones_like(w), k, p0=p0).run(500, 50)
        c = _solver(ng, d, b, np.ones_like(w), k, p0=p0, form="smoluchowski").run(500, 50)
        e = bff.GridDiffusionSolver(d, b, p0, k, 1.0, 0.02, "detailed_balance").run(500, 50)
        for x in (a, e):
            assert np.abs(np.asarray(x.get_density()) - np.asarray(c.get_density())).max() < 1e-16
            assert np.abs(np.asarray(x.get_fluorescence()) - np.asarray(c.get_fluorescence())).max() < 1e-14

    def test_the_stability_bound_reads_the_weights(self):
        """dt <= dg^2 / max_c sum_m D_cm sqrt(w_m / w_c), computed here."""
        ng, (d, b, w, k) = _toy()
        B, D, W = (a.reshape((ng,) * 3) for a in (b, d, w))
        total = np.zeros_like(D)
        for ax in range(3):
            for sh in (1, -1):        # the domain is clear of the edges, so roll is safe
                Dm, Wm, Bm = (np.roll(a, sh, axis=ax) for a in (D, W, B))
                total += 0.5 * (D + Dm) * np.sqrt(Wm / W) * Bm * B
        limit = 1.0 / total.max()                    # dg = 1
        _solver(ng, d, b, w, k, t_step=0.999 * limit).run(1, 1)
        with pytest.raises(ValueError):
            _solver(ng, d, b, w, k, t_step=1.001 * limit).run(1, 1)

    @pytest.mark.parametrize("which", ["D", "k", "p0"])
    def test_adjoint_is_the_transpose(self, which):
        """<dL/dtheta, v> == d/dh L(theta + h v), central difference."""
        ng, (d, b, w, k) = _toy(ng=13)
        rng = np.random.default_rng(7)
        p0 = b * (0.5 + rng.random(b.size))
        n_steps, n_out = 120, 10
        wf = rng.random(n_steps // n_out + 1) - 0.5

        def loss(dd, kk, pp):
            s = _solver(ng, dd, b, w, kk, p0=pp)
            return float(np.dot(wf, np.asarray(s.run(n_steps, n_out).get_fluorescence())))

        s = _solver(ng, d, b, w, k, p0=p0)
        g = s.gradient(list(wf), n_steps, n_out, p0)
        grad = {"D": g.get_d_diffusion(), "k": g.get_d_rate(), "p0": g.get_d_density()}[which]
        v = (rng.random(b.size) - 0.5) * b
        h = 1e-6
        args = {"D": (d, k, p0), "k": (d, k, p0), "p0": (d, k, p0)}[which]
        pos = list(args)
        neg = list(args)
        slot = {"D": 0, "k": 1, "p0": 2}[which]
        pos[slot] = args[slot] + h * v
        neg[slot] = args[slot] - h * v
        fd = (loss(*pos) - loss(*neg)) / (2 * h)
        adj = float(np.dot(np.asarray(grad), v))
        assert abs(adj - fd) / abs(fd) < 1e-6, (which, adj, fd)


# --------------------------------------------------------------------------
# 3. lifetimes: the three routes agree, and the two limits
# --------------------------------------------------------------------------
class TestLifetime:
    def test_linear_solve_matches_the_integrated_decay(self):
        ng, (d, b, w, k) = _toy()
        s = _solver(ng, d, b, w, k, t_step=0.01)
        lt = s.lifetime(TAU0, 1e-12)
        dec = s.donor_decay(TAU0, 12000, 1, 256)          # 120 ns, 30 tau0
        t, f = np.asarray(dec.get_time()), np.asarray(dec.get_fluorescence())
        assert abs(f[0] - 1.0) < 1e-12
        tau_x, tf = _simpson(f, t), _simpson(t * f, t) / _simpson(f, t)
        assert abs(lt.get_tau_x() - tau_x) / tau_x < 1e-5, (lt.get_tau_x(), tau_x)
        assert abs(lt.get_tau_f() - tf) / tf < 1e-5, (lt.get_tau_f(), tf)
        assert abs(lt.get_quantum_yield() - lt.get_tau_x() / TAU0) < 1e-15
        assert lt.get_residual() < 1e-12 and lt.get_iterations() > 0
        assert lt.get_method() == "dynamic"

    def test_krylov_matches_the_stepped_decay(self):
        """The stepped scheme converges onto the exponential at first order.

        Its error is the Lie splitting of transport and the sink (5e-4 at
        dt = 0.02 ns on this field) and halves with the step; Krylov solves the
        semi-discrete equation, so the distance must halve too."""
        ng, (d, b, w, k) = _toy()
        gaps = []
        for dt in (0.02, 0.01, 0.005):
            n = int(round(20.0 / dt))
            a = np.asarray(_solver(ng, d, b, w, k, t_step=dt).run_krylov(n, n // 20, 128)
                           .get_fluorescence())
            c = np.asarray(_solver(ng, d, b, w, k, t_step=dt).run(n, n // 20).get_fluorescence())
            gaps.append(np.abs(a - c).max())
        assert gaps[0] < 1e-3, gaps
        for g0, g1 in zip(gaps, gaps[1:]):
            assert 1.9 < g0 / g1 < 2.1, gaps

    def test_fast_transport_is_the_equilibrium_average(self):
        ng, (d, b, w, k) = _toy()
        s = _solver(ng, d * 1e4, b, w, k, t_step=1e-6)
        fast = s.lifetime_fast_exchange(TAU0)
        dyn = s.lifetime(TAU0, 1e-11, 100000)
        p = w * b / (w * b).sum()
        expected = 1.0 / (1.0 / TAU0 + (p * k).sum())
        assert abs(fast.get_tau_x() - expected) < 1e-14
        assert abs(dyn.get_tau_x() - expected) / expected < 1e-4, (dyn.get_tau_x(), expected)
        assert abs(dyn.get_tau_f() - expected) / expected < 1e-4

    def test_no_transport_is_the_static_average(self):
        ng, (d, b, w, k) = _toy()
        s = _solver(ng, d * 1e-8, b, w, k)
        stat = s.lifetime_static(TAU0)
        dyn = s.lifetime(TAU0, 1e-13)
        p = w * b / (w * b).sum()
        tau = 1.0 / (1.0 / TAU0 + k)
        assert abs(stat.get_tau_x() - (p * tau).sum()) < 1e-14
        assert abs(stat.get_tau_f() - (p * tau ** 2).sum() / (p * tau).sum()) < 1e-13
        assert abs(dyn.get_tau_x() - stat.get_tau_x()) / stat.get_tau_x() < 1e-6
        assert abs(dyn.get_tau_f() - stat.get_tau_f()) / stat.get_tau_f() < 1e-6

    def test_male_limits(self, male):
        """D -> 0 and D -> infinity on the MalE AV, slow factor 0.3."""
        d, w = _male_fields(male, 0.3)
        b, k = male["b"], male["k"]
        p = w * b / (w * b).sum()
        lo = bff.GridDiffusionSolver(d * 1e-8, b, p, k, male["dg"], 0.01,
                                     "detailed_balance", False, w)
        st, dy = lo.lifetime_static(TAU0), lo.lifetime(TAU0, 1e-12)
        assert abs(dy.get_tau_x() - st.get_tau_x()) / st.get_tau_x() < 1e-6
        # The approach to fast exchange is 1/D (the correction is the rate
        # variance times the relaxation time): 0.44 % at 1e4 D, 0.06 % at 1e5.
        gaps = []
        for f in (1e4, 1e5):
            hi = bff.GridDiffusionSolver(d * f, b, p, k, male["dg"], 0.01,
                                         "detailed_balance", False, w)
            fx, dy = hi.lifetime_fast_exchange(TAU0), hi.lifetime(TAU0, 1e-10, 400000)
            gaps.append((dy.get_tau_x() - fx.get_tau_x()) / fx.get_tau_x())
        assert 0.0 < gaps[1] < 1e-3 and gaps[0] / gaps[1] > 5.0, gaps
        # And the physical D sits between the two.
        mid = bff.GridDiffusionSolver(d, b, p, k, male["dg"], 0.01,
                                      "detailed_balance", False, w).lifetime(TAU0)
        assert st.get_tau_x() > mid.get_tau_x() > fx.get_tau_x()


# --------------------------------------------------------------------------
# 4. against the Brownian walk, started from p(x)
# --------------------------------------------------------------------------
def walk_lifetime(m, slow, d0=10.0, dt=0.002, t_max=8 * TAU0, n_walkers=20000, seed=11):
    """tau_x of walkers started from p ~ s^-n, each carrying its survival.

    The walk is the one ``internal/RandomWalk.h`` runs: positions in voxel
    units, a Gaussian step of variance ``2 D0 m(x) dt / dg^2`` per axis read
    at the current voxel (Ito), rejected when it leaves the volume. Per step
    the rate is constant, so each walker's integral is exact for its path.
    Returns (tau_x, standard error).
    """
    ng, dg, b = m["ng"], m["dg"], m["b"]
    mob = slow ** m["n"]
    w = np.where(b > 0, slow ** (-m["n"]), 0.0)
    rng = np.random.default_rng(seed)
    act = np.flatnonzero(b)
    start = rng.choice(act, size=n_walkers, p=w[act] / w[act].sum())
    pos = np.stack(np.unravel_index(start, (ng, ng, ng)), axis=1) + rng.random((n_walkers, 3))
    vox = start.copy()
    sigma = np.sqrt(2.0 * d0 * dt) / dg
    amp = np.ones(n_walkers)               # S(t) exp(-t/tau0)
    acc = np.zeros(n_walkers)
    for _ in range(int(round(t_max / dt))):
        lam = m["k"][vox] + 1.0 / TAU0
        decay = np.exp(-lam * dt)
        acc += amp * (1.0 - decay) / lam
        amp *= decay
        new = pos + rng.standard_normal((n_walkers, 3)) * (sigma * np.sqrt(mob[vox]))[:, None]
        ijk = np.floor(new).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < ng), axis=1)
        flat = np.where(ok, (ijk[:, 0] * ng + ijk[:, 1]) * ng + ijk[:, 2], 0)
        ok &= b[flat] > 0
        pos[ok] = new[ok]
        vox[ok] = flat[ok]
    # The tail past t_max, at each walker's current rate.
    acc += amp / (m["k"][vox] + 1.0 / TAU0)
    return acc.mean(), acc.std(ddof=1) / np.sqrt(n_walkers)
