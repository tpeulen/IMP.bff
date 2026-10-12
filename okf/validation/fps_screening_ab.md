---
okf_version: "0.2"
title: "IMP.bff against FPS's published screening numbers (T4L, 3GUN)"
description: "One obstacle-radius scale on IMP's radii, AV_DEFAULT_RADIUS_SCALE = 0.84 (2026-10-12, owner: no second radii table), fitted to FPS: FPS/IMP radius ratio 0.838 (median over 20 434 atoms of 148L/3GUN/1RTD), 148L CB volume vs FPS 0.91 -> 0.99, empties 48 -> 40 (FPS 33; the 8 extra are HETATM/off-node, not radii), 3GUN vs FPS +1.22 -> +0.90 A at equal rmsd, now equal to Olga's radii. The search keeps starting from the nearest grid point (owner, decided). Earlier: FPS's link search is the default, the attachment atom is excluded and the allowed sphere on."
tags: [validation, fps, screening, av, acv, radii, radius-scale, prd-121]
---

# IMP.bff against FPS's published screening numbers

**Date:** 2026-09-01 (first written 2026-08-31) · **Drivers:**
[`test/representation/test_fps_screening_ab.py`](../../test/representation/test_fps_screening_ab.py),
[`test/representation/test_av_contact_volume.py`](../../test/representation/test_av_contact_volume.py),
[`test/representation/test_olga_vdw_radii.py`](../../test/representation/test_olga_vdw_radii.py)
· **PRD:** [121](../prds/prd-121.md)

## Where to pick this up

00. **Radii: one scale factor on IMP's, 0.84 (2026-10-12, owner: "i do not
   want a separate set of atom radii, if fps is slightly smaller, use a single
   scale factor to match fps radii").** `AV_DEFAULT_RADIUS_SCALE`
   (`ProbeAccessibleVolumeDefaults.h`, position field `radius_scale`, schema
   1.9) multiplies the radius each particle carries as the AV's obstacles;
   Olga's table is not scaled; the docking clash term keeps its own
   `clash_radii_scale` (1.0). Fitted and chosen in the section "AV
   obstacle-radius scale" below. What is left of item 2 after it: 148L has 40
   empty CB sites against FPS's 33 (36 over ATOM records only); the 8 extra
   (3, 97, 104, 105, 106, 114, 117, 138) stay empty down to x0.78, so they are
   not the radii -- 3/97/104/106 open when HETATM and waters are dropped, which
   is what FPS reads; 105/114/117/138 are the off-node seed (item 1, decided).
   Open, not done: the **file door** (`get_av_from_pdb`, the standalone
   builder, and chimol's default AV path) still reads its own element-keyed
   Bondi table (`vdw_radii()` in `ProbeAccessibleVolumeBuilder.cpp`) -- a second
   radii set the owner does not want; it cannot read IMP's radii without IMP,
   so retiring it is a design decision (PRD-137), not a patch.
0. **The attachment atom and the allowed sphere were never what emptied the
   sites (2026-10-12, owner: "the attachment site must be excluded, also
   allowed sphere radius must be enabled by default").** Both were already in
   force on every lattice door and are now declared and shown: the atom alone
   is dropped (FPS's rule), the allowed sphere is on (`AV_DEFAULT_ALLOWED_SPHERE`
   2.25 A; a position without one takes FPS's `0.5 W` for its own width). The
   148L / T4L-132 / 3GUN numbers did not move (section "Since 2026-10-12
   (later)"). With the fixed lattice kept (owner) and the radii unchanged, the
   open front is still items 1-2: the off-node seed centre (candidate fix
   below keeps the lattice -- seeds = lattice nodes within `0.5 W` of the
   *anchor*, not of the source voxel) and the radii set. **Tried and reverted:**
   a flat 2.25 A for every position regardless of its width moved 3GUN away
   from FPS (its screening file has W 2.5, 3.5 and 4.5): +1.34 / 2.18 / 0.9747
   over 33, +0.67 / 1.18 / 0.9930 over 24 (IMP radii), against +1.22 / 1.99 /
   0.9792 and +0.57 / 1.01 / 0.9954 at FPS's per-width rule. Also measured and
   *not* a lever: removing every atom within 2.25 A of the anchor (Olga's
   `allowed_sphere_radius` semantics, `Position.cpp` `stripExpression`) leaves
   148L's empty count exactly where it is (48 IMP, 34 Bondi); stripping the
   site's side chain beyond CB (Olga's batch template) opens some (41 IMP, 28
   Bondi) but is not FPS's rule.
1. **Decided (owner, 2026-10-12): the search starts from the nearest grid
   point** ("of course the nearest grid point"); no anchor/seed change. T4L-132
   CB at a 0.5 A grid staying empty is a documented consequence, not an open
   item -- and the radius scale does not open it (0 voxels at x0.84 as at
   x1.0; 9 142 at 0.9 A, FPS 9 237). The record below stays as the reason.
   **The anchor is off-node; FPS centres its grid on it.** T4L-132 CB at W 4.5
   is still **empty at a 0.5 A grid** under FPS's link search, so the
   every-node hop was *not* what walled it in. Measured on the array door with
   FPS's radii (scratch scripts re-derive it: `IMP.bff.get_av(xyzr, src, 20,
   4.5, 3.5, 0, 0, 0.5, -1, stencil)`, xyzr from FPS's `vdW.txt` ladder over
   ATOM records): with the coordinates shifted so the CB sits on a lattice node
   the site opens under *either* search (41 646 voxels with stencil 342,
   38 085 with 74; FPS 42 151); off-node both are empty. The seed sphere and the
   hop cubes are taken from the *source voxel* (`PathMap::source_hop_targets`)
   while the open sphere is centred on the true anchor -- FPS has one centre
   for both. Candidate fix: seeds = lattice nodes within `0.5 W` of the anchor
   at their distance *from the anchor*. It changes every volume (the window is
   absolute so shared rasters can roll), so it is a decision, not a patch.
   Pinned in `test_av_source_clearance.py::test_fine_grid_at_the_fps_width_under_fps_first_step`.
2. **Empty sites at W 4.5 are the radii, not the search.** T4L 148L chain E,
   151 CB sites, grid 0.9, clearance derived (`chisurf/plugins/quenching_estimator/test/148l.pdb`):

   | obstacles | seed-only (74) | FPS search (342) | FPS's own routine |
   |---|---|---|---|
   | QuEst's radii (IMP united-atom, C up to 2.275 A), all atoms incl. HETATM | 50 empty | 48 empty | -- |
   | QuEst's radii, ATOM records only | 44 | 41 | -- |
   | FPS's Bondi radii (`vdW.txt`), ATOM only, as FPS reads | 33 | 33 | **33** |

   With FPS's radii both searches leave exactly FPS's count empty (31 shared;
   here-only 66, 105; FPS-only 24, 139). Of the classic sites, 99 and 129 are
   empty in FPS too; 96, 117, 132 open with FPS's radii and stay empty with
   IMP's. Volume against FPS where both are non-empty (median): 0.934 seed-only,
   **0.990** FPS search. Re-derive with the oracle (`prototypes/fps_oracle`,
   `FPS_DG=0.9`) over every chain-E CB serial.
3. **3GUN against FPS's own routine moved away under the true default.** The
   docking restraint forced stencil 26 on every AV until 2026-10-12, so the
   "+0.00 A / 0.84 A" below was the 26 stencil's, not the defaults'. Now every
   position's own link search is scored (table in the 2026-10-12 section);
   larger volumes sit *further* from FPS's <R_DA> under IMP's united-atom radii.
   Under Olga's (Bondi-like) radii the 24 pairs are +0.17 A / 1.03 A. Open:
   whether the residual is the radii, the anchoring of item 1, or FPS's
   exclusion of clashing seed voxels from the cloud (this module opens them).
4. **AV3 grid / radii** -- as before: FPS's relative rule gives 0.6 A for an
   AV3 with its 1.5 A radius (`fps_default_grid`); FPS's Bondi radii are not a
   selectable `radii_source` (owner, 2026-09-01: docking consistency).

## Since 2026-10-12 (latest): AV obstacle-radius scale

Owner, 2026-10-12: *"i do not want a separate set of atom radii, if fps is
slightly smaller, use a single scale factor to match fps radii."* FPS inflates
its obstacles by element-keyed Bondi radii (`Fps/data/vdW.txt`, read with FPS's
own name ladder, `StaticData.cs`); IMP's particles carry CHARMM united-atom
radii with implicit hydrogens. One declared factor,
`AV_DEFAULT_RADIUS_SCALE`, now multiplies IMP's radii wherever an AV reads
them: the decorator (`set_radius_scale`, legacy `space_fixed=False` frozen at
1.0), the fps.json position field `radius_scale` (schema 1.9), the structure
door (`get_av_from_structure`, `resample_av`), the docking restraint through
its positions. A shared occupancy raster adopts the scale beside the radii
source and refuses a second one. The array doors take the caller's radii as
given (QuEst scales its IMP radii itself). Search code unchanged, so timing is
unchanged (T4L-132 CB @0.9 1.19 ms at either scale, decorator door).
Test: `test/representation/test_av_radius_scale.py`.

**Per-atom ratio FPS/IMP** over the heavy atoms of 148L, 3GUN and 1RTD
(20 434 atoms, `NonWaterNonHydrogenPDBSelector`; FPS's radius from its ladder
on the atom name):

| element | n | median | IQR |
|---|---|---|---|
| C | 12 849 | 0.782 | 0.782-0.850 |
| O | 3 974 | 0.894 | 0.894-0.894 |
| N | 3 472 | 0.838 | 0.838-0.838 |
| P | 94 | 0.837 | -- |
| S | 45 | 0.900 | -- |
| **all** | 20 434 | **0.838** (mean 0.831) | least squares 0.821, ratio of sums 0.826, (r + W/2)^3-volume match 0.823 |

Per structure the median is 0.838 in each of 148L, 3GUN and 1RTD.

**Effect of the factor** (grid 0.9, L 20 / W 4.5 / R 3.5, FPS's link search,
allowed sphere 2.25 A; 148L chain E, 151 CB sites, QuEst's reader incl.
HETATM/waters -- FPS's own routine via `prototypes/fps_oracle` `FPS_DG=0.9`
over the 153 CB records, 33 empty; 3GUN against
`fps_screening_oracle_pins.json`, ACV off, bias / rmsd / r; docking scores
`chi2_C2_33p` T4L and HIV-RT `resolved`):

| scale | 148L empty (shared / here-only / FPS-only) | 148L ATOM only | vol / FPS median (IQR) | 3GUN 33 pairs | 3GUN 24 (no 132) | T4L dock | HIV-RT |
|---|---|---|---|---|---|---|---|
| 1.00 | 48 (33 / 15 / 0) | 42 | 0.906 (0.852-0.942) | +1.22 / 1.99 / 0.9792 | +0.57 / 1.01 / 0.9954 | 10.866 | 70.642 |
| 0.92 | -- | 39 | 0.953 (ATOM) | +1.11 / 2.06 / 0.9740 | +0.41 / 1.06 / 0.9929 | 10.696 | 70.202 |
| 0.90 | 41 (33 / 8 / 0) | 37 | 0.960 (0.916-0.992) | +1.05 / 2.03 / 0.9740 | +0.36 / 1.05 / 0.9928 | 10.562 | 69.612 |
| 0.88 | 41 (33 / 8 / 0) | 37 | 0.969 (0.930-1.004) | +0.98 / 2.01 / 0.9735 | +0.29 / 1.02 / 0.9927 | 10.436 | 69.448 |
| 0.86 | 41 (33 / 8 / 0) | 37 | 0.982 (0.941-1.016) | +0.93 / 2.00 / 0.9731 | +0.23 / 1.01 / 0.9926 | 10.456 | 69.119 |
| **0.84** | **40 (32 / 8 / 1)** | **36** | **0.990 (0.950-1.028)** | **+0.90 / 2.01 / 0.9719** | **+0.19 / 1.03 / 0.9921** | **10.504** | **68.546** |
| 0.82 | 40 (32 / 8 / 1) | 36 | 0.999 (0.963-1.041) | +0.86 / 2.01 / 0.9712 | +0.15 / 1.03 / 0.9919 | 10.364 | 68.037 |
| 0.80 | 39 (31 / 8 / 2) | 35 | 1.008 (0.971-1.052) | +0.81 / 2.01 / 0.9709 | +0.10 / 1.04 / 0.9917 | 10.353 | 68.609 |
| Olga (ref.) | -- | -- | -- | +0.88 / 2.02 / 0.9712 | +0.17 / 1.03 / 0.9920 | -- | -- |
| FPS Bondi, array door (ref.) | 34 (ATOM) | -- | 0.990 | -- | -- | -- | -- |

Here-only at every scale <= 0.90: 3, 97, 104, 105, 106, 114, 117, 138 (ATOM
only: 105, 114, 117, 138); FPS-only from 0.84: 139 (also FPS-only under FPS's
own Bondi radii -- the lattice, not the radii), 31 from 0.80. Classic sites:
99 and 129 stay empty (as in FPS), 96 and 132 open from 0.90, 117 stays empty.
T4L-132 CB (3GUN, decorator) at 0.9 A: 7 475 -> **9 142** voxels (FPS 9 237,
0.990 -- the same as FPS's Bondi radii on the array door, 9 143); at 0.5 A
still 0 (item 1). 3GUN against the published table with the ACV on:
+0.23 / 0.98 / 0.9912 (was +0.70 / 0.98 / 0.9954); Olga +0.23 / 1.01 / 0.9906.

**Chosen: 0.84.** It is the median per-atom FPS/IMP ratio (0.838, the same in
all three structures, and N's exact ratio); at it the median CB volume matches
FPS's own routine (0.99, the value FPS's Bondi radii themselves give through
this search), where the 3GUN rmsd is flat across the whole range (1.99-2.06)
and so cannot pick a value; 0.82 (least squares) overshoots the volume
(> 1.0) and starts losing FPS's empty sites (31). The 3GUN bias against FPS
falls from +1.22 to +0.90 A (+0.57 to +0.19 over the 24) at unchanged rmsd; r
drops 0.979 -> 0.972, all of it in the nine site-132 pairs (decided off-node
anchor). With the factor, IMP's radii and Olga's table give the same 3GUN
numbers to 0.02 A. Docking scores move little (T4L 10.87 -> 10.50, HIV-RT
70.64 -> 68.55), all 33 / 18 pairs finite.

Goldens re-pinned with the x1.0 values beside them: `test_density_grid.py`
(T4L-132@0.5 405 428 -> 512 452), T4L docking 10.8658 -> 10.5040 (restraint,
`score_structures`, screening), HIV-RT 70.6416 -> 68.5459, fixed-position chi2
13.7619 -> 14.1532, stencil-30 11.1970 -> 10.9134, stencil-26 11.1956 ->
10.9184, the restraint's 33 model distances (bias against the file's
experimental distances +0.45 -> +0.03 A), T4L-44 CB 21 739 -> 23 078 (fast
20 572 -> 21 882), the av_overlap edge 5.20 -> 4.93 A. Tests that put FPS's
Bondi radii on the particles (`test_fps_av_parity.py`), compare the radii
*sets* (`test_olga_vdw_radii.py`) or need a tie-free matrix
(`test_greedy_olga_ab.py`) ask for `radius_scale` 1.0 by name.

Re-derive: `IMP.bff.get_av(xyzr * [1, 1, 1, s], cb, 20, 4.5, 3.5, 0, 0, 0.9)`
over every chain-E CB of `chisurf/plugins/quenching_estimator/test/148l.pdb`
(xyzr from QuEst's reader); 3GUN through `score_structures` with
`"radius_scale": s` in every position of the screening file. Trap: 148L has
altloc CB records -- take a residue as open if any of its CB records is.

## Since 2026-10-12 (later): the attachment atom excluded, the allowed sphere on

Owner, 2026-10-12: *"the attachment site must be excluded, also allowed sphere
radius must be enabled by default"*; the fixed lattice stays, the radii stay.
Read from FPS (`av_routines.cpp:60-68`, `AVEngine.cs:112-121`): FPS drops the
**attachment atom alone** (`if (i == atom_i) continue;`) from both the linker
and the dye clash maps -- not its residue's side chain. Its allowed sphere is
the seed, `LinkerInitialSphere = 0.5` (`AVGlobalParameters`,
`ProjectData.cs:169`, `AVEngine.cs:78`) times the linker width: 2.25 A for every
FPS linker preset (`Fps/data/linker.txt`, all W 4.5).

What the code did before: every lattice door already subtracted the
attachment atom's own sphere from both occupancy rasters
(`drop_source_obstruction`; the array door when the source coincides with a
row) and already opened `0.5 W` around the anchor -- a stored -1 *derived* it,
which read as "off" in every editor. Now:

* `AV_DEFAULT_ALLOWED_SPHERE = 2.25` (`ProbeAccessibleVolumeDefaults.h`,
  `IMP.bff.AV_DEFAULT_ALLOWED_SPHERE`) is the default of every typed door
  (decorator setup, `get_av`, `get_av_lattice`, `get_av_from_pdb`,
  `get_linker_path_lengths`, `get_av_from_structure`), the schema default, and
  the template's value;
* a **position** without the key (decorator `set_av_parameter`, the fps.json
  structure door, `ProbeNetworkRestraint`, `imp_bff fps-av` without
  `--clearance`) takes FPS's rule for its own width, `0.5 W`, stored as that
  number; negative still means "derive";
* `test/representation/test_av_attachment_exclusion.py` holds the exclusion
  exact: the attachment row in the array == the row removed (density and path
  lengths, T4L 132/44/69), and the decorator volume == the same with the atom's
  radius 0. The legacy `space_fixed=False` anchoring keeps the atom (frozen).

Re-measured (no search code changed, so the volumes are the e4a3e2b9d ones):

| measurement | result |
|---|---|
| 148L chain E CB, IMP radii (QuEst's door, all atoms incl. HETATM, 151 sites), grid 0.9 | **48 empty** at 2.25 A (49 at QuEst's former 2.1 A: 161 opens) |
| same, FPS's Bondi radii over ATOM records (153 CB records incl. altlocs) | **34 empty** (FPS's own routine on the same 153: **33**; here-only 66, 105; FPS-only 139) |
| attachment row in the array vs removed | identical volume at every site, both radii sets |
| T4L-132 CB (3GUN), grid 0.9: decorator IMP / Olga / array Bondi / FPS | 7 475 / 8 960 / 9 261 / 9 237 |
| T4L-132 CB, grid 0.5, fixed lattice | **0** at every door and radii set (FPS 42 151): the off-node seed centre, item 1 |
| 3GUN vs FPS's routine, IMP radii, 33 / 24 pairs | +1.22 / 1.99 / 0.9792 and +0.57 / 1.01 / 0.9954 (unchanged) |
| same, Olga radii | +0.88 / 2.02 / 0.9712 and +0.17 / 1.03 / 0.9920 (unchanged) |
| search time (median of 9, decorator, IMP radii) | T4L-132@0.9 1.05 ms, T4L-44@0.9 2.87 ms, T4L-44@0.5 18.10 ms (unchanged) |

Re-derive: `IMP.bff.get_av(xyzr, cb, 20, 4.5, 3.5, 0, 0, 0.9)` over every
chain-E CB of `chisurf/plugins/quenching_estimator/test/148l.pdb` (xyzr from
QuEst's reader, or from FPS's `vdW.txt` ladder over ATOM lines); FPS via
`prototypes/fps_oracle` (`FPS_DG=0.9`, one call per CB serial). Trap: 148L has
altloc CB records, so "151" and "153" sites are both right for their reader.

## Since 2026-10-12: FPS's link search is the default

Owner, 2026-10-12, reversing 2026-10-11's "av go fast by default": the AV search
default is FPS's -- the linker hops `LinkSearchNodes = 3` voxels along every
axis at once from **every** node it reaches (`av_routines.cpp:156-209`, the
7x7x7 cube, landing on any free voxel across whatever lies between), not only
out of the seed sphere. Declared once (`AV_DEFAULT_LINK_SEARCH = "fps"`,
`AV_DEFAULT_SEARCH_STENCIL = AV_SEARCH_STENCIL_FPS = 342`,
`include/ProbeAccessibleVolumeDefaults.h`); the seed-only search stays
selectable as `link_search = "fast"` (stencil 74): decorator
`set_link_search`, fps.json position field `link_search` (schema 1.8), array
doors `search_stencil`. `ProbeNetworkRestraint`'s `search_stencil` default is
now 0 = each position's own (it forced 26). Implementation:
`PathMap::set_cube_stencil(3)` -- the cube as the stencil, Euclidean edge
lengths, target-only test -- after the unchanged seed-and-hop first step.

**It is FPS's search.** `test/representation/test_av_link_search.py` re-implements
`calculate1R`'s routing in numpy (clash map, seed sphere, wavefront of cube
hops until nothing moves) and compares on random obstacle sets: identical
reached set, path lengths to < 1e-5 voxel.

**Cost.** Search-only time (`PathMap::get_last_search_seconds`, median of 9 real
resamples, decorator door, IMP radii, L 20 / W 4.5 / R 3.5; load average 6.6-7.4
on 8 cores -- not a quiet machine):

| site | grid | seed-only (74) | FPS, plain loop | FPS, optimised | FPS opt / seed-only opt (search, whole resample) |
|---|---|---|---|---|---|
| T4L-132 CB | 0.9 | 0.96 ms | 4.05 ms | 1.03 ms | 1.07, 1.05 |
| T4L-44 CB | 0.9 | 2.69 ms | 11.68 ms | 2.88 ms | 1.07, 1.08 |
| T4L-44 CB | 0.5 | 16.25 ms | 64.09 ms | 18.22 ms | 1.12, 1.11 |
| HIV-RT A6 CB | 0.9 | 2.05 ms | 8.28 ms | 2.23 ms | 1.09, 1.06 |
| HIV-RT B287 CB | 0.9 | 1.17 ms | 4.69 ms | 1.23 ms | 1.05, 1.04 |

The naive cube is 3.5-4x the seed-only search; two exact shortcuts in
`PathMap::dijkstra_lattice` bring it to **1.05-1.12x**: (a) the stencil is
ordered by Chebyshev ring and a tile `c` short of the bound relaxes only the
rings `k` with `fl(c + k) < bound` (an offset k voxels out is at least k long,
so the rest fail the bound anyway); (b) interior tiles relax FPS's cube a
(dz, dy) row at a time with two NEON loads. Neither changes a bit:
`PathMap::set_plain_relaxation(True)` turns both off and the tests compare
(`test_the_shortcuts_change_no_bit`, both searches, three site/grid cases).
The docking restraint evaluation (T4L, 17 AVs, rigid shift each frame) is
15.2 ms at stencil 26, 16.3 ms at 74, 16.0 ms at 342. Array door end to end on
the same sites before the shortcuts: 1.04-1.60x.

**3GUN vs FPS's own routine** (`fps_screening_oracle_pins.json`, ACV off;
bias / rmsd / r):

| search | radii | all 33 | 24 without 132 | 9 with 132 |
|---|---|---|---|---|
| FPS (default) | IMP | +1.22 / 1.99 / 0.9792 | +0.57 / 1.01 / 0.9954 | +2.96 / 3.43 / 0.9669 |
| seed-only | IMP | +1.06 / 1.88 / 0.9798 | +0.42 / 0.93 / 0.9957 | +2.78 / 3.26 / 0.9676 |
| FPS (default) | Olga | +0.88 / 2.02 / 0.9712 | +0.17 / 1.03 / 0.9920 | +2.78 / 3.49 / 0.9518 |
| seed-only | Olga | +0.74 / 1.94 / 0.9721 | +0.04 / 1.01 / 0.9922 | +2.61 / 3.33 / 0.9539 |
| *stencil 26 (the restraint's former forced value)* | IMP | +0.63 / 1.65 / 0.9808 | +0.00 / 0.84 / 0.9959 | +2.30 / 2.83 / 0.9689 |

ACV on vs the published table: FPS search +0.70 / 0.98 / 0.9954 (IMP), +0.23 /
1.01 / 0.9906 (Olga); seed-only +0.66 / 0.97 and +0.22 / 0.99; stencil 26 +0.55 /
0.93 and +0.15 / 0.94. All 33 pairs resolve in every row.

**T4L-132 CB vs FPS** (W 4.5, R 3.5, L 20; voxels):

| door, radii | grid | seed-only | FPS search | FPS's routine (Bondi) |
|---|---|---|---|---|
| decorator, IMP | 0.9 | 6 899 | 7 475 | 9 237 |
| decorator, Olga | 0.9 | 8 312 | 8 960 | 9 237 |
| array, FPS's Bondi | 0.9 | 8 478 | **9 143** (0.990) | 9 237 |
| any, off-node anchor | 0.5 | 0 | 0 | 42 151 |
| array, Bondi, anchor on a node | 0.5 | 38 085 | **41 646** (0.988) | 42 151 |

**Found on the way: the quadrature diverged for overlapping clouds.**
`av_distance_quadrature`'s second-order correction grows as 1/d between block
centroids; T4L 44D-69A gave 36.93 A at K = 100 against an exact 28.82 A, and
the restraint's `get_quad_error_estimate` read 7.5 A under stencil 74 (0.01 A
under the forced 26, which is why it was never seen). Blocks within three
spreads are now one isotropic Gaussian (exact noncentral-chi mean): 28.823 A at
K = 100, estimate 0.0016 A (`test_av_lattice.py::TestQuadratureAccuracy::test_quadrature_of_overlapping_clouds`).

Goldens re-pinned with the previous values beside them: `test_density_grid.py`
(T4L-132@0.5 371 548 -> 405 428 points), HIV-RT `resolved` 67.3153 -> 70.6416,
T4L docking 11.1476 -> 10.8658, fixed-position chi2 13.2426 -> 13.7619, the
`test_AccessibleVolume.py` means and distances; T4L site 99 CB opens under IMP's
radii (1 014 -> 15 605 voxels). The greedy-selector parity test asks for the
26 stencil by name (its matrices have a near-tie under 342).

## Since 2026-10-11 (later): the AV defaults are FPS's and Olga's

Owner, 2026-10-11: *"av defaults should be fps/olga defaults, take from
there."* Read from FPS's and Olga's source (headers in `chisurf/junk/fps`,
`chisurf/junk/olga`) and declared once in
[`include/ProbeAccessibleVolumeDefaults.h`](../../include/ProbeAccessibleVolumeDefaults.h),
which every door reads (decorator, array/PDB builders, both fps.json readers,
schema, `av-export`, model comparison):

| parameter | old | FPS | Olga | new |
|---|---|---|---|---|
| linker length | 20 (CLI 20.5) | 20 (`linker.txt` row 1) | none (0) | 20 |
| linker width | 0.5 decorator / 1.0 position reader / 1.5 CLI | 4.5 | none (0) | 4.5 |
| AV1 radius | 3.5 (3.0 at one json reader) | 3.5 | none (0) | 3.5 |
| AV3 radii | (AV1 default, 0, 0) | 5, 4.5, 1.5 | none | 5, 4.5, 1.5 for an AV3 that omits them |
| grid | 1.5 | 0.2 min(L, W, 2R), >= 0.4: 0.9 (AV1), 0.6 (AV3) | 0.9 | 0.9 |
| clearance (derived) | max(1.5, W/2 + grid/2); array door flat 2.1 | seed `0.5 W` | `ignoreSphere` (lround(W/2dg)+1) dg | 0.5 W |
| first step | ball of W + dg (FRETraj) | 3-voxel cube hop from every seed voxel | none | FPS's (`PathMap::set_source_first_step`) |
| simulation type | AV1 | AV1 | AV1 | AV1 |
| contact volume | off | none | off (0, -1) | off |
| min. sphere fraction | 0 | none | 0 | 0 |

Where they disagree the search follows FPS. The legacy `space_fixed=False`
path keeps its frozen derivation and no first step. Measured on this file,
ACV off unless stated (FRETraj-first-step values in parentheses):

| comparison | radii | pairs | bias | rmsd | r |
|---|---|---|---|---|---|
| vs FPS's own routine | IMP (default) | 33 | +0.63 A (+0.70) | 1.65 A (1.70) | 0.9808 (0.9801) |
| *...the 24 without site 132* | IMP (default) | 24 | **+0.00 A** (+0.08) | **0.84 A** (0.87) | **0.9959** (0.9955) |
| *...the 9 with site 132* | IMP (default) | 9 | +2.30 A (+2.37) | 2.83 A (2.92) | 0.9689 (0.9667) |
| vs FPS's own routine, the same 24 | Olga | 24 | -0.33 A (-0.29) | 1.05 A (1.03) | 0.9926 (0.9928) |
| no ACV, vs the published table | IMP (default) | 33 | +2.66 A | 2.97 A | 0.9885 |
| no ACV, vs the published table | Olga | 33 | +2.38 A | 2.72 A | 0.9857 |
| ACV on, vs the published table | IMP (default) | 33 | +0.55 A (+0.58) | 0.93 A (0.94) | 0.9953 (0.9954) |
| ACV on, vs the published table | Olga | 33 | +0.15 A (+0.15) | 0.94 A (0.90) | 0.9915 (0.9923) |

FPS's defaults are the closest this module has come to FPS's own routine on
the 24 shared pairs, and the published table with Olga's radii does not
recover its stencil-only 0.71 A (0.94 A now); with the default radii the
ACV comparison is as before. Golden volumes (`test/test_density_grid.py`, explicit
parameters): T4L-132 CB at 0.5 A is 371 548 point values (331 156 stencil
only, 439 804 FRETraj). HIV-RT `resolved` score 67.3153 (59.0404 stencil only,
60.5172 FRETraj); T4L restraint `chi2_C2_33p` 11.1476 (11.3268, 11.2817).

## Since 2026-10-11: the linker's first step reaches out of the anchor

The search used to leave the attachment atom one stencil step at a time. A site
among bonded neighbours was then walled in at a fine grid and found at a coarse
one (T4L N132 CA at width 4.5: empty at 0.5 A, 5 400 A^3 at 0.7 A), and site 132
of this file had no volume under the default radii. FPS hops `linknodes = 3`
voxels from a seed of half the width; FRETraj joins the source to every free
voxel within `width + grid` (`maxR_source`). The search now does the latter
(`PathMap::set_source_hop_radius`, derived as
`ProbeAccessibleVolumeDecorator::get_effective_source_hop_radius`), on the
lattice doors only; `space_fixed=False` keeps the historical numbers. Measured
on this file, ACV off unless stated:

| comparison | radii | pairs | bias | rmsd | r |
|---|---|---|---|---|---|
| vs FPS's own routine | IMP (default) | 33 | +0.70 A | 1.70 A | 0.9801 |
| *...the 24 that resolved before* | IMP (default) | 24 | **+0.08 A** | **0.87 A** | **0.9955** (was -0.31 / 1.20 / 0.9926) |
| *...the 9 with site 132* | IMP (default) | 9 | +2.37 A | 2.92 A | 0.9667 |
| vs FPS's own routine, the same 24 | Olga | 24 | -0.29 A | 1.03 A | 0.9928 (was -0.62 / 1.13 / 0.9940) |
| ACV on, vs the published table | IMP (default) | 33 | +0.58 A | 0.94 A | 0.9954 (was 24 pairs, +0.22 / 0.91) |
| ACV on, vs the published table | Olga | 33 | +0.15 A | 0.90 A | 0.9923 (was -0.02 / 0.71 / 0.9954) |

It moved this module **towards FPS** -- on the pairs both resolved, and in
coverage (all 33 now; `test_fps_av_parity.py` p66 with explicit hydrogens went
from *empty* to 0.987 of FPS's volume, mean 0.49 A off) -- and **away from the
published table under Olga's radii** (0.71 -> 0.90 A rmsd). The authored
`allowed_sphere_radius: 2` no longer buries the structure under either radii set
(33 of 33 both ways). The tables below are the 2026-09-01 state and are kept as
the record of it. Setting the hop radius to 0 on the map restores the old search;
it is a physics default, so reverting it is a decision, not a fix.

## What is being compared

Zenodo 3376527 (Sanabria *et al.*, *Nat. Commun.* **11**, 1231, 2020) holds what
was published for **421 T4L structures**: 33 donor–acceptor pairs of ⟨R_DA⟩
each, plus χ² for three states, beside the `FRET_screening.fps.json` that names
the labelling sites. One of the 421 is **3GUN**, which this module already
ships — so the whole path can be measured against that answer without shipping a
new structure.

This is the complement to the single-volume pin in
[`test_fps_av_parity.py`](../../test/representation/test_fps_av_parity.py). That
one localises a disagreement; this one is where a **systematic** one cannot
hide. It is the one that found a bias, and then found what caused it.

## The radii, and why the default is not the one that fits best

Three radii sets are in circulation here and they are not interchangeable:
FPS's element-keyed **Bondi** (C 1.70, N 1.55, O 1.52, P 1.80, S 1.80 Å),
IMP's **CHARMM-derived united-atom** set from `read_pdb` (carbon 1.85–2.275 Å,
carrying implicit hydrogens), and **Olga's**, keyed by *atom name* rather than
element. An accessible volume must use exactly one of them.

**The default is IMP's** — the radii already on the particles
(`AV::set_radii_source`, spelled `"imp"`). The owner's decision, 2026-09-01:
*"use the IMP radii, charmm, as otherwise wont be consistent with IMP
docking."* The reason is the other half of a docking score. Excluded volume is
`clash_container`, which measures overlap through `IMP::core::XYZR` — the
**particles'** radii. Inflate the accessible volumes by Olga's table while the
clash term is built on IMP's, and the two halves of one score disagree about
how big an atom is, silently, in a quantity neither half reports. That is the
"one quantity, two places" failure this repository is organised against, and
consistency with IMP docking was judged worth more than agreement with Olga's
published numbers.

**The default is not free, and this page will not pretend otherwise.** With
IMP's radii:

* **nine of the 33 published ⟨R_DA⟩ have no model value at all** — the volumes
  at those sites (all the pairs involving site 132) come back empty, so every
  row below marked "24 pairs" is over a subset of the published table;
* the agreement with the Zenodo table is **worse**: +0.22 Å / 0.91 Å rmsd over
  24 pairs, where Olga's radii give −0.02 Å / 0.71 Å over all 33.

That is the price of internal consistency. **Anyone reproducing Olga-era
numbers should set `"radii_source": "olga"` on the positions** (schema 1.6,
per position); Olga's table stays vendored (`data/olga_vdw_radii.csv`,
`IMP::bff::olga_vdw_radius`) for exactly that.

Olga's radii were the default for part of 2026-09-01, before this reversal.
Every number below was re-measured under the current default rather than
carried over.

## Result

Two things changed on 2026-09-01 and the table has to be read with both of them
named: the accessible-contact volume the file asks for stopped being ignored
(PRD-121 **G9**), and Olga's radii became available as a per-position choice.
Every row is labelled with its radii set; the rows marked **default** are what
this module reports out of the box.

| | radii | pairs | bias | rmsd | r |
|---|---|---|---|---|---|
| **this module, no ACV, vs FPS's own routine** | **IMP (default)** | **24** | **−0.31 Å** | **1.20 Å** | **0.9926** |
| this module, no ACV, vs FPS's own routine | Olga | 33 | +0.02 Å | 1.62 Å | 0.9777 |
| *…the same 24 pairs as the default row* | Olga | 24 | −0.62 Å | 1.13 Å | 0.9940 |
| **this module, no ACV, vs the published table** | **IMP (default)** | **24** | **+2.05 Å** | **2.54 Å** | **0.9901** |
| this module, no ACV, vs the published table | Olga | 33 | +2.05 Å | 2.40 Å | 0.9890 |
| FPS's own routine vs the published table | FPS Bondi | 33 | +2.37 Å | 2.73 Å | 0.9735 |
| **this module, ACV on, vs the published table** | **IMP (default)** | **24** | **+0.22 Å** | **0.91 Å** | **0.9960** |
| this module, ACV on, vs the published table | Olga | 33 | −0.02 Å | **0.71 Å** | 0.9954 |
| *…the same 24 pairs* | Olga | 24 | −0.23 Å | 0.72 Å | **0.9966** |
| **this module, ACV on, vs FPS's own routine** | **IMP (default)** | **24** | **−2.15 Å** | **2.52 Å** | **0.9868** |

`prototypes/fps_oracle/` runs FPS's own routine on the same structure with the
same file, pinned in `test/references/fps_screening_oracle_pins.json`, so both
implementations can be measured against the `.dat` and against each other.

**Three things are established.** Without a contact volume this module and FPS
agree with each other and *neither* reproduces the table. With the contact
volume the file asks for, this module reproduces the table — 24 of its pairs to
0.91 Å rmsd by default, all 33 to 0.71 Å if Olga's radii are selected — with no
fitted parameter anywhere in the path. And the radii set is worth **more than
the search convention**: switching it moves the agreement with the published
table by more than the residual against FPS's own routine.

**The comparison against FPS deserves reading twice.** Olga's radii give a
*smaller* bias against the FPS oracle (+0.02 Å against −0.31 Å) and a *worse*
rmsd and correlation (1.62 Å / 0.9777 against 1.20 Å / 0.9926) — but that is a
different, larger set. Over the **same** 24 pairs Olga's radii are better on
both (−0.62 Å, 1.13 Å, r = 0.9940). The nine pairs they add are all pairs with
site **132**, which united-atom radii bury, and they are the hard ones. Both
figures are pinned in
`test_fps_screening_ab.py::test_rda_agrees_with_fps_itself` so neither can be
quoted without the other, and so that the default is not mistaken for the more
accurate choice. It is the more *consistent* one.

## The unit that had to be right

**Olga's numbers are nanometres upstream and Angstrom here, and the factor is
Olga's own, not an inference from the magnitudes.**
`Olga/src/AV/Position.cpp:118-124` (`coordsVdW`) scales the coordinate *and*
the radius by the same `10.0f` on one line, because pteros stores coordinates
in nm and `calculateAV()` works in Å:

```cpp
xyzw.emplace_back(frame.coord.at(i)[0] * 10.0f,
                  frame.coord.at(i)[1] * 10.0f,
                  frame.coord.at(i)[2] * 10.0f,
                  pterosVDW(system, i) * 10.0f);
```

So carbon 0.17 nm is **1.70 Å**, which is Bondi's carbon and FPS's. Pinned in
`test/representation/test_olga_vdw_radii.py::test_the_unit_is_nanometre_times_ten`.

**The fallback for an unknown atom name is flat.** `pterosVDW` is
`vdWRMap.value(name, 0.15)` (`Position.cpp:110`) — 0.15 nm = **1.50 Å** for any
name the table misses, with no element lookup behind it. It is not
near-transparent: at 1.50 Å it is within 0.01 Å of Olga's own oxygen. But it is
smaller than every heavy atom in the table and much smaller than the
united-atom carbon an unrecognised carbon would otherwise have taken, so it
matters where the name is a carbon. On the shipped fixtures: T4L 3GUN, **no**
misses; the HIV-RT DNA, 85 atoms of 1018 (`C1*`–`C5*`, `O1P`, `O2P`, `O3*`–`O5*`,
`C7` — the old spellings of names Olga writes `OP1`/`OP2`/`C1'`). That is
reproduced rather than corrected: Olga reading the same file would do the same.
It is also why the HIV-RT `resolved` score differs between the two sets
(59.0404 by default, 59.2547 under Olga's table).

**What the two sets do to the volumes.** T4L 3GUN, grid 0.5 Å, clearance 2.0:

| site | IMP united-atom (default) | Olga | mean moves |
|---|---|---|---|
| 132 CB | 82 418 voxels | 95 836 (+16.3 %) | 1.05 Å |
| 55 CB | 136 823 | 144 032 (+5.3 %) | 0.38 Å |
| 99 CB | 710 | 29 586 (×42) | 17.9 Å |

Site 99 is the one to read: at 710 voxels it is a numerical accident, not a
volume. Under the default it stays one.

## The authored clearance buries the structure — and that is the radii

Two statements, both true, and the second is the one that was nearly lost:

**1. Under the default radii, the authored clearance buries the structure.**
With `allowed_sphere_radius: 2` as `FRET_screening.fps.json` writes it, eight
of seventeen sites come back empty and only **9 of 33** pairs have a model
value, where deriving the clearance leaves one empty and gives **24**. FPS
seeds from `LinkerInitialSphere × linker_width` **unconditionally**, where this
module's clearance is a sphere carved out of an obstacle map already inflated
by half the linker width. Any fps.json carrying a small explicit clearance is
asking for something other than it appears to.

**2. What empties those volumes is the radii, not the clearance.** Select
Olga's table and the *same* authored clearance resolves **all 33** pairs, the
same as deriving it, and the two agree to 0.03 Å of bias against the published
table. A carbon of 2.275 Å walls a source in where one of 1.70 Å does not.

The two clearance conventions really are different quantities — that has not
been retracted. What was retracted is the claim that the difference is what
empties the volumes. Both behaviours are asserted, each against its own radii
source, in
`test_fps_screening_ab.py::test_the_authored_clearance_buries_the_structure_and_it_is_the_radii`,
so the attribution stays checkable rather than remembered. It survived the
default moving to Olga's radii and back on the same day, because it is a
statement about the two sets and not about which of them is default.

## The +2 Å was the accessible-contact volume after all

An earlier version of this page said the offset was "in the reference, not the
code", on the grounds that FPS has no ACV and so the ACV could not explain a
disagreement between FPS and the table. The first half is right — **FPS has no
contact volume at all** — and the second half is a non sequitur, because it
assumes the table was produced by FPS. It was not.

`FRET_screening.fps.json` asks for `contact_volume_thickness: 3` and a fitted
`contact_volume_trapped_fraction` at every one of its seventeen sites. Those
fields were accepted by `IMP::bff::AV` and did nothing (PRD-121 **G9**). Wiring
them up (2026-09-01) moves this module from +2.05 Å to **+0.22 Å** of the
published ⟨R_DA⟩ and cuts the rmsd from 2.54 Å to 0.91 Å. The conclusion is the
opposite of the earlier one: the file **is** the parameter set the table was
computed with, and the program that computed it was ACV-capable — Olga, not
FPS. Selecting Olga's radii on top of that closes the rest and widens it to all
33 pairs (−0.02 Å / 0.71 Å), which is corroboration of the same conclusion from
a second direction.

**The two references are from two eras, and the port matches each on its own
terms.** The accessible-contact volume did not exist yet when FPS was written
(owner, 2026-09-01); it is an Olga-era addition. So there is no single "correct"
answer to compare against here, and no contradiction between the two rows of the
table above:

* the **FPS oracle** is pre-ACV by construction, and the comparison against it
  drops `contact_volume_*` exactly as it drops `allowed_sphere_radius` — keys
  FPS has no concept of — giving −0.31 Å;
* the **Zenodo table** is Olga-era and asks for an ACV, and honouring that ask
  gives +0.22 Å.

A file gets an ACV if and only if it asks for one, which is what keeps both
comparisons honest. Reading the offset against one reference as evidence about
the other is the mistake this page made in its first version.

It holds under the other clearance convention too. With the authored
`allowed_sphere_radius: 2`, which leaves only nine pairs resolvable under the
default radii, the ACV moves those nine from +1.70 Å / 2.35 Å rmsd to
**+0.05 Å / 1.06 Å**.

The alternatives that were tried and do not fit are still worth recording,
because they are what makes the ACV explanation load-bearing rather than
merely available:

| statistic (FPS's own clouds, no ACV) | bias vs the table |
|---|---|
| ⟨R_DA⟩ | +2.03 Å |
| ⟨R_DA⟩_E | +3.11 Å |
| R_mp | −1.40 Å |

and recomputing with FPS's own AV3 dye radii from `Fps/data/linker.txt` (donor
5/4.5/1.5, acceptor 11/3/1.5) makes it **worse**, +6.36 Å.

## What the contact volume is, and one number that pins its discretisation

The rule is Olga's, taken from `Olga/src/AV/fretAV.cpp` (`path2points`): a cloud
voxel is *in contact* when excluded volume — the obstacle raster inflated by the
dye radius — lies within `contact_volume_thickness` of it, and the contact
voxels are then weighted so they carry `contact_volume_trapped_fraction` of the
cloud's total weight. Olga rounds the layer down to whole voxels first
(`deltaIlist` takes an `int`), which at `thickness = 3` and a 2 Å grid is a
one-voxel shell rather than a 3 Å sphere.

That truncation is reproduced rather than corrected, and the table says why: a
true 3 Å sphere gives **+0.49 Å bias, 1.04 Å rmsd** against the published
numbers where Olga's shell gives **+0.22 Å, 0.91 Å**. The trapped fraction is a
*fitted* number, and it is only meaningful against the region it was fitted
for. The consequence to know is that the contact layer is quantised by the
grid: a thickness below `simulation_grid_resolution` is no layer at all.

Note that the fractions were fitted against **Olga's** obstacle set, and the
default here is not that set. That is the sharpest form of the price above: a
fitted constant is being applied over a slightly different geometry. It is also
why `radii_source` lives on the *position* rather than on the run — the file
that carries the fitted fraction is the file that should name the radii it was
fitted with.

## What it does to the shipped T4L file

`examples/structure/T4L/fret.fps.json` asks for the same thing — thickness 3 at
every site, trapped fractions 0.33–0.72 — so every number this module reports
for it moved on 2026-09-01 when the ACV was honoured.

| | before | after (default radii) |
|---|---|---|
| mean ⟨R_DA⟩, 33 pairs of `chi2_C1_33p` | 46.91 Å | **43.88 Å** — −3.04 Å, all 33 shorter (−5.46 … −1.40) |
| model vs the file's experimental distances (`chi2_C2_33p`) | +3.93 Å bias, 5.78 Å rmsd | **+0.84 Å**, 3.93 Å |
| `ProbeNetworkRestraint` score, `chi2_C2_33p` | 22.5277 | **11.3268** |

Selecting Olga's radii moves the same three to 43.48 Å, +0.44 Å / 3.91 Å and
11.2866 — a further improvement, and not the default, for the reason at the top
of this page.

The mechanism is visible one site at a time: the fitted trapped fractions
(0.33–0.72) are all well above the *geometric* contact share of the cloud
(0.13–0.47, mean 0.20), so the surface layer is up-weighted. The mean position
moves 2.3 Å on average (0.04–4.6 Å) and **closer to the nearest protein atom at
sixteen of the seventeen sites**, by 1.81 Å.

## A second defect behind the first

Wiring the fields up immediately found that `AV::set_av_parameter()` read the
fraction as `j.value("contact_volume_trapped_fraction", -1)` — an **`int`**
default, so nlohmann deduced `int` and every fps.json fraction was truncated to
0 or 1. It could not be seen while the value was unused. The first measurements
made after the wiring were taken with it, and said the ACV *lengthened* ⟨R_DA⟩
by +1.47 Å; with the truncation fixed it shortens them by 3.04 Å. Both are in
this page's history for the reason such things usually are.

## What survives from the first pass

**1. The authored clearance buries the structure**, under the default radii —
see the section above, which also says what actually causes it.

**2. FPS is not the ACV.** The A/B against FPS's own routine must be run with
the contact volume **off**, because FPS has none. `_scored()` in the test drops
`contact_volume_*` for the same reason it drops `allowed_sphere_radius`: they
are keys FPS has no concept of, and leaving them in turns the A/B into a
comparison of two different models. With them off the −0.31 Å / 1.20 Å /
r = 0.9926 agreement with FPS is unchanged by any of this.

## The other half of the score: the clash term's own radii

Everything above is about the **accessible volume's** radii. The other half of
a docking score is the excluded volume, `clash_container`, and until 2026-09-01
it had no radii control at all — it read `IMP::core::XYZR`, the radii the
particles carry. That is why moving the *volume* to Olga's table left the clash
term exactly where it was (PRD-121 G2: 110.98 of 150.69 before, 110.85 of
140.58 after).

FPS's clash constant is `k_clash = 2/ClashTolerance²` and FPS applies it to
**Bondi** radii. Applied to IMP's united-atom radii, the same interface reads
as a far worse clash. Measured statically at the input pose over HIV-RT's
protein–DNA interface (8016 protein atoms against 1018 DNA atoms) at k = 8,
which is FPS's `ClashTolerance = 0.5` for refinement and error estimation:

| clash radii | overlapping pairs | total overlap | energy |
|---|---|---|---|
| `imp` (united-atom, the default) | **268** | **90.49 Å** | **198.40** |
| `olga` (Bondi scale) | **30** | **7.59 Å** | **11.11** |

A factor of **17.9** in energy, from the radii alone — the coordinates, the
constant and the pose are identical.

`DockingParameters::clash_radii_source` and `clash_radii_scale` (and
`--clash-radii-source` / `--clash-radii-scale` on `imp_bff_fps dock` and
`score`) are that control. The source picks the table, the scale multiplies it;
the defaults are `"imp"` and `1.0`, so nothing moves unless asked. The radii are
carried by **shadow spheres** — new particles at the atoms' coordinates, in the
same rigid bodies — rather than written onto the structure, so a volume built
afterwards still sees the radii its own `radii_source` asked for. FPS's D12
(`SpringEngine` mutating session-global `Molecule` objects and never reverting)
is not reproduced.

### One global scale is not defensible, and here is the measurement

The obvious cheap alternative is a single multiplier on IMP's radii. It does
not work, and the reason is that IMP's ratio to Bondi is **per element**:

| element | n | IMP min/mean/max (Å) | Bondi | Bondi / IMP mean |
|---|---|---|---|---|
| C | 5687 | 1.700 / 2.1033 / 2.275 | 1.70 | **0.8083** |
| O | 1764 | 1.700 / 1.7106 / 1.770 | 1.52 | 0.8886 |
| N | 1520 | 1.700 / 1.8495 / 1.850 | 1.55 | 0.8381 |
| P | 48 | 1.700 / 2.1406 / 2.150 | 1.80 | 0.8409 |
| S | 15 | 2.000 / 2.0000 / 2.000 | 1.80 | 0.9000 |

**The carbon answer is 0.8083** — 1.70 / 2.1033, the population mean over this
structure. It is not the answer for oxygen (0.889) or sulfur (0.900), and IMP's
carbon is not one number: it spans 1.70–2.275 Å where Bondi's is a single 1.70,
so no factor reproduces even carbon exactly.

Bisecting the scale against each of the three interface statistics gives three
different answers:

| statistic matched | scale | pairs | overlap | energy |
|---|---|---|---|---|
| Olga's 30 pairs | **0.8206** | 30 | 5.05 Å | 5.45 |
| Olga's 7.595 Å overlap | **0.8386** | 43 | 7.59 Å | 8.83 |
| Olga's energy 11.114 | **0.8476** | 50 | 9.27 Å | **11.11** |

At the pair-count match the energy is off by a factor of two; at the energy
match the pair count is off by 1.7×. A scale can reproduce **one** number of
the contact distribution and not the distribution. So: **use the scale to
loosen a clash term; use the source to reproduce FPS.** The two compose, and a
caller who wants the closest single-knob approximation of FPS's hard sphere on
this structure should use the source, not 0.83.

### What it does to the two doors

Scoring (`imp_bff_fps score`, k = 1, HIV-RT `resolved`, full AVs). The
restraint half is byte-identical in every row, which is the check that this
knob is the clash term's alone and does not reach the volumes:

| clash radii | score | of which clash | score − clash |
|---|---|---|---|
| `imp` ×1.0 (default) | **59.0404** | 24.8005 | 34.2399 |
| `imp` ×0.90 | 38.8090 | 4.5691 | 34.2399 |
| `imp` ×0.8476 | 35.6303 | 1.3904 | 34.2399 |
| `imp` ×0.80 | 34.6113 | 0.3714 | 34.2399 |
| `olga` ×1.0 | 35.6291 | 1.3893 | 34.2399 |

(`olga` at k = 1 gives 1.3893 = 11.1141/8, the static census divided by the
constant — the two measurements are the same number reached two ways.)

### The G2 bootstrap, before and after

This is the pass/fail the control was built for. PRD-121 G2 records that
HIV-RT's bootstrap collapses to **0.000 ± 0.000 Å** because the clash term
carries most of the parent score. Re-measured on the same cheap configuration
the test runs (`fps_error_estimation_parameters`, score set `resolved`,
`n_frames = 60`, `coarse_clash` off, 3 replicas, seed 3), varying only the
clash radii:

| clash radii | parent score | of which clash | clash / score | RMSD spread (Å) |
|---|---|---|---|---|
| `imp` ×1.00 (**default**) | 150.6884 | 110.9848 | **0.737** | **0.0000 ± 0.0000** |
| `olga` ×1.00 (FPS-like) | 35.8845 | 1.9295 | 0.054 | **0.0689 ± 0.0155** |
| `imp` ×0.85 | 43.2916 | 2.7068 | 0.063 | 0.0983 ± 0.0365 |
| `imp` ×0.80 | 39.9571 | 0.8630 | 0.022 | 0.4040 ± 0.0953 |
| `olga` ×0.90 | 39.7326 | 0.5171 | 0.013 | 0.3539 ± 0.2444 |
| `ev_weight = 0` | 24.3351 | 0.0000 | 0.000 | 2.1297 ± 1.4584 |

**Read this carefully; it is a partial result.** The collapse *is* resolved in
the strict sense — at FPS's own hard sphere the spread is no longer identically
zero, and the clash term stops dominating (73.7 % of the parent score → 5.4 %).
But 0.069 Å is still **thirty times** smaller than the 2.13 Å the same run
gives with the excluded volume switched off. Moving the clash term to Bondi
radii removes 98 % of the clash *energy* and does not remove the pinning. The
spread rises monotonically with the remaining clash fraction and only reaches
~0.4 Å once the clash term is down to 1–2 % of the score, which is well past
any radii set FPS would recognise.

So the honest statement is: **the radii artefact was real and is now
controllable, and it was not the whole story.** Whatever else holds this pose
is not the size of an atom. The `ev_weight = 0` row remains the number to
compare a bootstrap against before believing it, and `parent_e_clash` remains
the number to read first.

## Provenance, stated rather than inferred

The Zenodo data was produced **with Olga, using an accessible-contact volume**
(owner, 2026-09-01). This page originally reasoned the other way round — the fit
improved when the ACV was honoured, therefore the producer was ACV-capable — so
it is worth being explicit that the fit is now **corroboration of a known
provenance**, not the evidence for it. The two facts that looked contradictory,
"FPS has no ACV" and "the ACV closes the offset", are chronological: the ACV
postdates FPS.

## What this does *not* say

The 3GUN row is one of 421; the other structures are not in this checkout, so
"reproduces the published table" is one structure's pairs, not the study — and
by default it is 24 of 33 of them, not all. The −0.31 Å / 1.20 Å residual
against FPS without the ACV is not zero either: it is the two search
conventions (this module's tight stencil against FPS's three-voxel hop) and the
clearance mapping, both open in PRD-121. Nothing here validates the trapped
fractions themselves — they are fitted per site, and this measures that applying
them reproduces the numbers they were fitted to produce, not that they are
right. And nothing here says the default radii are the *better* ones for
reproducing Olga; the table says plainly that they are not. They are the ones
the rest of an IMP docking score is computed with.

## Reproducing

```bash
python -m pytest test/representation/test_fps_screening_ab.py \
                 test/representation/test_av_contact_volume.py \
                 test/representation/test_olga_vdw_radii.py -q
```
