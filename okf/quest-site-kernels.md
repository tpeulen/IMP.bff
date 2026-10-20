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

# Where to pick this up

1. **ChiSurf WP3 decides the re-pin.** Both effects are real code changes: the AV changes are
   owner-approved FPS parity, and the frame change is a correctness fix. The prototype's fitted
   parameters absorbed the frame bias. Its contact offset δ (AF488 −0.42 Å) is in the same range
   as the 0.5 Å registration error, so the δ fits have to be redone on the corrected geometry,
   not compared at 1e-9. To reproduce the prototype exactly, build the volume with
   `grid_origin = attachment - ((ng-1)//2 - 0.5) dg` from the prototype's voxel set. The
   reproduction test does this.
2. **Diffusion solver gaps for a Smoluchowski model with stickiness (WP9).**
   - There is no potential/weight field in the lattice flux. Smoluchowski is uniform at
     equilibrium and Itô gives p ∝ 1/D, so p ∝ ∏ s_t^(−n_t) is reachable only by tying the
     mobility to the equilibrium.
   - `equilibrium_occupancy` has no weights option.
   - The Brownian walk is Itô in mobility.
   - There is no grid ⟨τ⟩ or quantum yield, neither static nor via a solve.
   - `GridDiffusionSolver` only steps explicitly.

   A detailed-balance flux `D_cm √(w_c w_m)(p_c/w_c − p_m/w_m)` would touch the stepper, the
   adjoint, the Krylov symmetrisation and the GPU kernel together. It is not started.
3. Water radii in `probe_solvent_exposure` are the caller's. The notebook used C 1.70, N 1.55,
   O 1.52, S 1.80 (Bondi, `vdw_radius(element)` gives the same).
