---
type: Subsystem
title: Coarse-grained contact potentials — one kernel, two doors
description: The ProteinMC contact terms (MJ, UNRES centroid, backbone H-bond, Go, soft-sphere clash, Generalized Born, gated residue area) as IMP scores/restraints and as array functions over the same C++ kernels.
resource: include/internal/ContactKernels.h
tags: [potentials, coarse-grained, imp, restraints, kernels, imp-tricks, chisurf]
timestamp: '2026-10-06T00:00:00Z'
---

# Where to pick this up

1. **Landed and green 2026-10-06** (commit "Coarse-grained contact potentials:
   one C++ kernel each, as IMP objects and as arrays"): all seven terms run in
   C++, built and tested -- `test/potentials/test_contact_potentials.py` (148L,
   9 tests) + `test_potentials.py` (21 pre-existing pins, unchanged) = 30
   passed; imp-tricks' parity + cgmol suites 103 passed. Measured warm, best of
   five, imp-tricks `test_numba_parity._residues(n)` fixture, numba / NumPy / bff:

   | term | 164 res (984 atoms) | 600 res (3600 atoms) |
   |---|---|---|
   | MJ | 14.6 / 152 / 12.9 us | 254 / 1945 / 173 us |
   | H-bond | 13.3 / 212 / 22.4 us | 151 / 1929 / 243 us |
   | UNRES centroid | 14.2 / 129 / 13.2 us | 232 / 1631 / 136 us |
   | Go | 29.5 / 344 / 17.0 us | 764 / 5271 / 225 us |
   | clash | 788 / 4461 / 474 us | 11190 / 29383 / 2458 us |
   | ASA | 231 / 4618 / 224 us | 1478 / 23320 / 1309 us |
   | GB | 1555 / 5304 / 443 us | 20010 / 52558 / 4031 us |

   Every term is at or under numba **except H-bond, 1.6-1.7x numba**. Of its
   187 us at 600 residues, 3 us is the int32 column conversions; the rest is
   the kernel's scalar gate scan.
2. **Built? No. Tested? No. -- the H-bond fix in progress.** The working tree
   (uncommitted, *not* in the commit) has `for_gated_upper_pairs` in
   `include/internal/ContactKernels.h`: a `MatrixGate` overload that compacts
   each row's passing columns branch-free before visiting them, used by the
   contact, binned and H-bond sums (the `MatrixGate` struct moves from
   `src/ContactPotentials.cpp` into the header). Next: build under
   `/tmp/imp-bff-build.lock`, rerun the 30 bff tests + imp-tricks parity, rerun
   the bench (H-bond target <= numba); commit only if green. A copy of the
   edit is at the chisurf session scratchpad `keep_ContactKernels.branchless.h`
   in case the tree is reset.
3. **Open: derivatives.** Only `SoftSphereOverlapPairScore` returns them. GB
   and the Go well are differentiable and a minimiser would want gradients; the
   tabulated terms (MJ, UNRES, H-bond) and the area are step functions.
4. **Open: Ramachandran / phi-psi** stays in imp-tricks NumPy
   (`IMP.cgmol.protein._compute_phi_psi`); it was not in the slow set.
5. **Trap -- the gates differ between the doors.** The array functions take the
   caller's C-alpha matrix and threshold (the legacy convention: a *squared*
   matrix vs a squared cutoff for MJ/H-bond/ASA, the plain one for UNRES). The
   restraints measure their own particles: `HydrogenBondRestraint` applies its
   C-alpha cutoff as a plain distance (`validation/hbond_ca_cutoff.md`); the
   statistical scores take what their `ClosePairContainer` hands them. A
   parity test must set both gates to the same pair set -- the test file shows
   how for each term.
6. **Bounds checking is new.** The array functions reject atom indices past the
   coordinate array; the numba kernels read out of bounds silently. One
   imp-tricks fixture relied on that (fixed in the same change).

# Shape

* **`include/internal/ContactKernels.h`** -- the only implementation of each
  term: `soft_sphere_overlap`, `lj_well`, `generalized_born_sum`,
  `site_contact_sum`, `site_binned_sum`, `backbone_hbond_sum`, `site_area_sum`,
  plus the cell list (`for_close_pairs`) and the gated pair iterator
  (`for_gated_upper_pairs`, with a branchless row scan for a matrix gate --
  the scan, not the arithmetic, is the cost when a few percent of pairs pass).
* **`ContactPotentials.h`** -- the array door, numpy zero-copy (`IN_ARRAY`):
  `get_site_contact_energy`, `get_site_binned_pair_energy`,
  `get_backbone_hbond_energy`, `get_soft_sphere_overlap_energy`,
  `get_matrix_lj_well_energy`, `get_generalized_born_energy`,
  `get_site_accessible_area`. Standalone core (no IMP types).
* **`ProbePotentialRestraints.h`** -- the IMP door: `MiyazawaJerniganPairScore`
  and `UNRESCentroidPairScore` are `IMP::core::StatisticalPairScore` (IMP's own
  `score_functor::Statistical`, reused, not rewritten); `SoftSphereOverlapPairScore`
  (with derivatives), `HydrogenBondRestraint`, `GoRestraint`,
  `GeneralizedBornRestraint`, `SiteAccessibleAreaRestraint` gather their
  particles' coordinates and call the kernel. `clash_energy`,
  `generalized_born_energy` and `go_energy` are the same kernels.

## What IMP has, and why it is or is not used

| Term | IMP equivalent | Decision |
|---|---|---|
| MJ contact | `score_functor::Statistical` (one value per type pair) | reused as-is (`MiyazawaJerniganPairScore`) |
| UNRES centroid | `score_functor::Statistical`, binned, no interpolation | reused as-is; its bin is `floor(d / w)`, offset 0, zero at `>= w * n_bins` -- the legacy binning exactly |
| clash | `core::SoftSpherePairScore` (`k = 2/t^2`) | not reused for parity: it has no bonded-distance exclusion. `create_clash_restraint` (IMP's + a stereochemistry filter) stays as the exact-bonds form; `SoftSphereOverlapPairScore` is the legacy term as a PairScore |
| H-bond (4 channels) | none | bff restraint + kernel |
| Go truncated LJ well | none (`atom::LennardJones*` is a different form) | bff restraint + kernel |
| Generalized Born | none in IMP | bff restraint + kernel |
| residue area | `saxs::SolventAccessibleSurface` returns per-atom *fractions*; bff's own `solvent_accessible_surface_area` is exact | `SiteAccessibleAreaRestraint` keeps the legacy gate (occluders chosen by a C-alpha-type gate) |

## Parity (148L chain E, `test/potentials/test_contact_potentials.py`)

Restraint in `IMP.core.RestraintsScoringFunction` = array function = numba
oracle (imp-tricks `tests/numba_oracles`, verbatim) at `rtol=1e-10`, counts
exact, for all seven terms; the soft-sphere PairScore's gradient matches a
central difference at `1e-6`. The pre-existing pins in
`test/potentials/test_potentials.py` were unchanged by moving the restraints
onto the kernels.
