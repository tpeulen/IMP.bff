---
okf_version: "0.2"
---

# QuEst site-table kernels: exposure, rotamer quenchers, neighbour counts, voxel frame

The MalE quenching analysis (`../quest/prototype/male_quenching_analysis.ipynb`, chapters 3b
and 5) builds one per-voxel table per labelled site. These kernels compute it in C++. ChiSurf's
`quenching_estimator` builds the table from them (its plan:
[`../chisurf/okf/plugins/quest-prototype-migration.md`](../chisurf/okf/plugins/quest-prototype-migration.md),
WP2/WP3).

| Python name | what it returns | notebook source it replaces |
|---|---|---|
| `fibonacci_sphere_points(n)` | flat unit-sphere points, z polar, `(k+½)` lattice | the shell in `water_exposure` |
| `probe_solvent_exposure(centres, atom_xyz, atom_radii, probe_radius=3.5, water_radius=1.4, n_points=64)` | uncovered fraction per centre (cell list) | `water_exposure` (cKDTree) |
| `residue_type_neighbour_counts(av, atoms, residue_types, radius=8.5)` | `(n_voxels, n_types)` int32 | the "slow-factor map trick", one QuenchedDonorDecay per type |
| `quencher_rotamer_ensembles(chains, res_ids, res_names, atom_names, coords, residue_types, library_path, table={}, clash=2.8)` / `quencher_rotamer_ensemble(..., chain, res_id, library_path, quench_atoms, clash)` | `QuencherRotamerEnsemble`: crystal centre, kept rotamer centres, renormalised probabilities | `quencher_positions` (Bio.PDB PPBuilder) |
| `ProbeAccessibleVolume.get_voxel_indices()` / `.get_voxel_centres()` / `.get_walk_anchor()`, `QuenchedDonorDecay.get_voxel_centres()` | the occupied voxels and where they are | `x0 + (ijk - (ng-1)//2 + 0.5) dg` |
| `stretched_linker_length(library or name, linker_residue, dye_residue)` | Å | cell 21 (AF488 A48_C1R: 20.4) |

All are tested against the notebook's own code in `test/quenching/test_quest_site_kernels.py`.
The test also reproduces the prototype's cached table for 1OMP/72/L 20.12 bit for bit
(`test/input/quenching/male_1omp72_L20.12_prototype.npz`: neighbours exact, exposure ≤ 1e-6),
using the prototype's own volume and frame.

## The voxel frame (fixed 2026-10-20)

An accessible volume's voxel `(i, j, k)` (flat C-order index) is centred at
`grid_origin + (i, j, k) dg`, which is exactly the cloud `get_points()` returns. The lattice is
snapped to the grid, so the attachment point is generally *not* on it. QuenchedDonorDecay,
DynamicAccessibleVolume and the walk used the attachment point as their anchor. That placed
every stamped sphere (quenchers, sticky residues) and every walker position up to one voxel off
the voxel the volume means, always in the same direction. For 1OMP/72 at dg 0.5 the offset is
(0.33, 0.01, 0.25) Å. They now anchor on the grid: `get_walk_anchor() = grid_origin + ((ng-1)//2 - 0.5) dg`
for the floor/walk kernels, plus 0.5 dg for the `grid_axis` maps. The notebook's centre formula
still holds with the new `get_x0()`.

**What it moves (measured):** MalE APO 233, slow 0.3, L 21.5, τ₀ 4.108, direct (p ∝ 1/s) τ:

| volume | old (attachment) frame | grid frame |
|---|---|---|
| notebook-era settings | 2.589 ns (the notebook's 2.59) | 2.869 ns |
| today's AV defaults | 2.523 ns | 2.832 ns |

The notebook-era settings are Bondi radii, seed-only search (stencil 74) and no allowed sphere.
The frame correction is about 0.3 ns. The AV-default change is about 0.07 ns.

## Why today's AV is not the notebook's (bisected 2026-10-20)

The notebook call `get_av_from_pdb(site_pdb, "A", 72, "CB", 20.12, 0.5, 3.5, 0, 0, 0.5)` returns
97 722 voxels at `fceece9ed` and `f2c272dc7`, and 106 992 today. The steps below are counts with
Bondi radii unless the step says otherwise:

1. `51a31f831` adds FPS's seed W/2 and FPS's first step (3-voxel cube hops); the derived
   clearance was `max(1.5, W/2 + dg/2)`. 97 898.
2. `e4a3e2b9d` makes FPS's link search (stencil 342) the default. 104 677.
3. `8ce0dcc28` turns on the allowed sphere at 2.25 Å. 107 309.
4. `3b1751b7e`/`3cd39fdc3` switch to IMP radii × 0.84. 106 992.

All four are owner rulings for FPS parity (log 2026-10-11/12). Step 1 cannot be undone through
any argument. `get_av(xyzr_bondi, cb, L, 0.5, 3.5, 0, 0, 0.5, -1.0, 74)` reaches 97 898, not
97 722.

## The diffusion PDE with detailed balance (the "pde" ensemble, 2026-10-20)

`GridDiffusionSolver(..., flux_form="detailed_balance", weight_map=w)` is the
Smoluchowski equation with a potential. The flux on the edge between voxel c and voxel m is

    J_cm = 0.5 (D_c + D_m) sqrt(w_c w_m) (p_c/w_c - p_m/w_m) b_m

so the equilibrium weight `w` and the mobility `D` are separate inputs. The flux is zero on
every edge at `p = w`, which makes `w` the discrete scheme's exact stationary state. With
`w = 1` it is the plain Smoluchowski form. The kernel is tttrlib's `LatticeDiffusion.h`
(`LATTICE_FLUX_DETAILED_BALANCE`, `lattice_sweep_weighted`, `lattice_adjoint_sweep_weighted`,
tttrlib `6097ec431`), vendored verbatim. It is implemented in four places, all consistent:

| route | entry | units |
|---|---|---|
| explicit stepper + adjoint | `run(n_steps, n_out)`, `gradient(dL_dF, n_steps, n_out, p0)` (dL/dD, dL/dk, dL/dp0; w held fixed) | D A^2/ns, k 1/ns, dg A, t_step ns |
| Lanczos exponential | `run_krylov(n_steps, n_out, krylov_dim=128)`; symmetric in `u = p/sqrt(w)` | report grid `n_out t_step` ns |
| donor decay | `donor_decay(tau0, n_steps, n_out, krylov_dim=128)`: `I(t)/I(0) = e^(-t/tau0) sum p(t)` | ns |
| lifetimes | `lifetime(tau0, tolerance=1e-10, max_iterations=0)`: two Jacobi-PCG solves of `(L + K + 1/tau0)`; `lifetime_static(tau0)` (D -> 0); `lifetime_fast_exchange(tau0)` (D -> inf) | `GridLifetime`: `get_tau_x()` = int F dt (ns), `get_tau_f()` = int tF / int F (ns), `get_quantum_yield()` = tau_x/tau0 (Phi/Phi0), `get_iterations()`, `get_residual()`, `get_method()` |
| GPU (wgpu plugin) | backend ABI 4: `ImpBffPropagateFn` gains `const double* weight`; f32 for this form, like Ito | same as run() |

Also: `get_stationary_occupancy()`, `get_weight_map()`/`set_weight_map()`,
`equilibrium_occupancy(D, bounds, "detailed_balance", w)`, `FLUX_DETAILED_BALANCE = 2`.
An empty or zero start density means "start from the stationary occupancy", which is an
ensemble excited out of its equilibrium. The explicit step's stability bound now reads the
weights: `dt <= dg^2 / max_c sum_m D_cm sqrt(w_m/w_c)`. Run_krylov, donor_decay and lifetime
have no step limit. Tests: `test/quenching/test_detailed_balance_pde.py` (16, 9 s) and
`expensive_test_detailed_balance_walk.py` (2, 70 s). tttrlib `test/cpp/test_lattice_diffusion.cpp`.

**Measured.** On the toy grid the weight field is a fixed point to 1e-13 after 2000 steps.
An arbitrary start relaxes onto it: Krylov to 1e-8, and `equilibrium()` to 1e-9. On the MalE
1OMP/72 AV (the prototype's 97 722 voxels padded to 83^3, since that volume touches the grid
edge) w is stationary to 1e-12 at slow 1, 0.3 and 0.1. The adjoint's dot-product identity
holds to 1e-6. The stepper minus Krylov is first order in dt (5e-4 at 0.02 ns, halving).
The linear solve matches the integrated decay to 1e-5. MalE, slow 0.3, tau0 4 ns, k = 1/ns
per Trp/Tyr/Met/Pro in range: static 3.232 ns, D0 = 10 A^2/ns 2.900 ns, fast exchange
1.676 ns. The approach to fast exchange goes as 1/D: 0.44 % at 1e4 D, 0.06 % at 1e5 D.
D -> 0 is static to 1e-6. PCG takes 750-1900 iterations, 1-3 s.

**Against the walk** (40 000 walkers started from p(x), dt 2 ps / 0.5 ps): slow 1.0 -0.02 % /
-0.04 %; slow 0.3 +0.75 % / +0.41 % (the walk's time-step bias: it shrinks with dt); slow 0.1
+0.75 % / +0.81 % (1.7-1.9 sigma, reported, not asserted). The static values are 3.75, 3.23
and 1.97 ns, so the PDE and the walk agree well inside the transport effect.

# Where to pick this up

1. **ChiSurf WP3 decides the re-pin.** Both effects are real code changes: the AV changes are
   owner-approved FPS parity, and the frame change is a correctness fix. The prototype's fitted
   parameters absorbed the frame bias. Its contact offset δ (AF488 −0.42 Å) is in the same range
   as the 0.5 Å registration error, so the δ fits have to be redone on the corrected geometry,
   not compared at 1e-9. To reproduce the prototype exactly, build the volume with
   `grid_origin = attachment - ((ng-1)//2 - 0.5) dg` from the prototype's voxel set. The
   reproduction test does this.
2. **Wired in ChiSurf WP9 (2026-10-20)**: `quenching_estimator/core/ensembles.py` builds
   `w = 1/m` from the plugin's own slow-factor map (`m = prod s^n`, the walk's mobility) and
   calls the solver below; MalE APO 95 (AF488 table, dg 0.9) PDE 3.637 / 2.494 / 0.890 ns at
   slow 1 / 0.3 / 0.1, and a p(x)-started walk agrees (3.638 +- 0.007, 2.487 +- 0.014). **New
   gap it found:** `simulate_probe_diffusion` and `quenched_donor_photons` take no start
   distribution (`internal/RandomWalk.h` draws a uniform voxel), so ChiSurf's p(x)-started
   walk is numpy until the walk takes a start weight (rebuild needed). The items below stay
   open. Original note: build
   `w = prod_t s_t^(-n_t)` from `residue_type_neighbour_counts`. For the walk's own physics,
   D is `D0 prod s^n`; D can also be set independently. Then call
   `GridDiffusionSolver(D, bounds, w*bounds, k, dg, t_step, "detailed_balance", False, w)`,
   followed by `.lifetime(tau0)` / `.donor_decay(tau0, ...)`. The open items:
   - The adjoint holds w fixed. There is no dL/dw, so fitting slow factors through the
     gradient needs either that term in `lattice_adjoint_sweep_weighted` (tttrlib first,
     then re-vendor) or finite differences through `lifetime()`.
   - `DynamicAccessibleVolume` and `QuenchedDonorDecay` still accept only "smoluchowski"
     and "ito". Their `flux_form` validation and `equilibrium_occupancy` call (which passes
     an empty weight) need the weight wired in if they should run this form.
   - The PDE's own lattice error at dg 0.5 A was not separated from the walk's dt bias.
     A dg 0.35 A run would show how much of slow 0.1's +0.8 % belongs to each.
   - Trap: an AV that touches its grid's outer shell is refused by every solver path. Pad
     it by one voxel (the MalE prototype volume needs this).
   - Unrelated and pre-existing: `test_vendored_headers.py::test_ptolib_sources_match_checkout_when_present`
     is red because the sibling ptolib checkout (98497f1, 2026-10-19) is newer than
     `include/ptolib`. Refresh the vendored copy.
3. Water radii in `probe_solvent_exposure` are the caller's. The notebook used C 1.70, N 1.55,
   O 1.52, S 1.80 (Bondi, `vdw_radius(element)` gives the same).
