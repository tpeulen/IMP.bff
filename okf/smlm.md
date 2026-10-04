# SMLM localization scoring and selective particle averaging

The coordinate-level implementation is C++ in `SMLM.h/.cpp` and
`SMLMParticles.h/.cpp`, wrapped as the flat `IMP.bff` API. The owning index
copies coordinates, measured standard deviations, weights and source particle
IDs once. Iterative scoring, registration and accumulation remain native.

## Spatial index and scores

`read_smlm_csv` parses measured coordinates and per-axis positional precision
directly into the owning native index. Source units are explicit: use scale 10
for nm→Angstrom, with optional source-unit origin subtraction. Scalar xy
precision is used only when per-axis precision is absent; PSF widths and photon
errors are not substitutes. Missing z or precision is rejected unless the caller
explicitly supplies defaults. Photon counts do not silently become weights.
Ambiguous aliases for one field are rejected. SMAP's source ID zero is unassigned
by default; set `SMLMCSVOptions.zero_particle_id_is_unassigned=false` explicitly
for zero-based particle identities. Negative IDs remain unassigned.
`select_smlm_particles` and `select_smlm_region` perform native one-time
selection, retaining measurement uncertainties and source group IDs.

`SMLMIndex` uses a median-balanced KD-tree split along the widest coordinate
extent. Leaves default to 16 rows. Nodes retain bounding boxes and per-axis
maximum precision; KDE prunes against Gaussian support, not just point centers.
This prevents broad heterogeneous kernels from being discarded erroneously.
Nearest/radius searches retain original row identities. This is an exact
low-dimensional spatial index, not a claim of universal optimality for every
distribution or query workload.

`evaluate_density` sums normalized anisotropic 3D Gaussians inside the specified
Mahalanobis cutoff (default 4 standard deviations). Removed tails are not
redistributed. `evaluate_score` is weighted mean **query/model → measured
localization density** negative log likelihood and supplies analytic coordinate
gradients. Lower is better. The positive background is a density floor, not a
normalized uniform contamination model. Scores alone do not prove full data
coverage. Changing length units by a factor a changes densities by a^-3 and
scores by 3 log(a); the background must change consistently.

## Particle selection, frames and covariance

`SMLMGaussianOverlap` owns the localization index and scores arbitrary model
Gaussians with full covariance sums. Conservative bounding-box and covariance
bounds prune the persistent tree; remaining pairs are evaluated exactly inside
the Mahalanobis cutoff. This does not compress or replace measured points by
moment-matched mixture centers. The logarithmic score gradient remains stable
even when a raw overlap derivative would overflow in extreme length units.

`SMLMPointModel` implements the paper's forward per-observation likelihood,
explicit uniform background on a specified ROI and independent intrinsic blur.
It reuses its model tree under changing rigid poses. Actual summed log likelihood
is distinguished from weighted mean NLL for `smlm_aic` / `smlm_aicc`.

`select_smlm_precision` gates measured per-axis precision, without cropping
toward a desired model radius. Source row provenance belongs with the input
archive. `align_smlm_particles` independently centers each source particle and
rotates its least-variance axis to +z. Optional Huber weights reduce leverage of
distant points for frame estimation; deposition still uses retained measurements.
No scale, ring radius or rotational symmetry is imposed. In-plane angle is
retained: this initializer cannot resolve eightfold NPC pose ambiguity.

Provided row-major source→reference 3x4 rigid transforms are supported directly.
`refine_smlm_translation` preserves rotation; `refine_smlm_rigid` refines rotation
and translation by backtracking native likelihood descent. These are local
optimizers requiring initial overlap, not global registration guarantees.

`register_smlm_particles` performs a coarse azimuth search followed by full
six-degree-of-freedom likelihood refinement for each particle independently.
The default angular period is a full circle; a reduced period is valid only
with documented symmetry of the fixed model. Source positions are never snapped,
scaled or duplicated. The resulting maps are reference-assisted averages; the
source-to-reference transforms, fit scores and stopping flags must be retained.
Coarse starts are stably ranked by scalar indexes, not moved Eigen matrices:
this preserves tied-sample order without MSVC's over-aligned temporary-buffer
failure in the native wheel and IMP-module builds.

`average_smlm_particles` rotates FULL covariance R diag(sigma²) Rᵀ; it evaluates
rotated kernels in source coordinates. Particle count/score gates exclude whole
particles. Default normalization assigns equal mass to accepted particles,
regardless of localization count; each half map has its own denominator.
Particle half maps split by alternating accepted particle order, not localizations.
Independent PCA frames guarantee one half's data cannot alter the other's frames.
Reference-driven refinement/selection must be split before training to retain
that independence. Disjoint half maps alone do not establish gold-standard FSC.

## Density and display

Values use x-fast indexing x+nx*(y+ny*z); origin is voxel (0,0,0)'s center.
Integrate density with voxel volume. Grid-clipped mass is not renormalized.
`write_smlm_average_mrc` performs the existing MRC writer's C-order conversion
natively. MRC expects Angstrom and isotropic spacing; the computation otherwise
allows any single consistent length unit and anisotropic grid spacing.

ChiMOL's `plugins/smlm/average.py` is the IO/display adapter. Source archive rows
preserve measured xyz, sigma, source site/row/acquisition IDs. The NPC tutorial
contains 99 populated human Nup96 groups, no saved LocMoFit poses; the prior
53.7 nm file was a starting parameter setup, not a fitted result. ShareLoc
7182237 has x/y only and cannot provide measured 3D density. Human structural
context uses deposited 7R5J assembly operators, with a documented rigid frame;
Nup96 label density does not describe every NPC component.

## IMP structural modelling

The IMP-layer `SMLMRestraint` scores live XYZ emitter particles against an owned
measurement index and adds analytic derivatives through `DerivativeAccumulator`.
It can be combined with other IMP restraints and carried by native rigid bodies
and optimizers. No Python scoring callback or localization transfer occurs in
the objective loop. Mean NLL is the default; optional sum mode uses the total
observation weight exactly once. External IMP restraint weights remain separate.

The likelihood constructor takes explicit ROI/background options. A second
constructor takes model Gaussian covariances and uses the persistent data-tree
overlap score. Those covariances are fixed in the observation frame: anisotropic
rigid-member covariances do **not** automatically rotate. Use the point-emitter
likelihood for rigid structural fitting unless you explicitly manage covariance
frames. The small emitter-model tree is rebuilt after structural coordinates
change; the measured data remain owned natively.

Emitter sites should describe the localized label, not arbitrary protein atoms.
Measured uncertainty, intrinsic blur and model Gaussian width are distinct.
Neither atomic masses nor particle radii substitute for localization precision.
See `examples/imaging/smlm_structure.py` for CSV input and native IMP rigid-body
optimization; `okf/smlm-reference-data.json` records downloaded primary artifacts.

## Verification and resume

Tests: `test/smlm/test_smlm.py` (brute-force KDE, heterogeneous kernels, exact
search, finite-difference gradients, full covariance rotation, density mass,
selective gates, half maps, clipping and translation) and
`test_smlm_particles.py` (independent PCA frames, proper rigid transforms,
likelihood refinement and MRC axis/length contracts).

The remaining dedicated suites exercise forward likelihood, full model-Gaussian
overlap, particle registration, strict CSV IO, IMP restraint forces/weights and
the structural-modelling example. The final standalone core run passed 204
tests. The installed IMP run passed 264 tests plus nine subtests across SMLM and
the existing density/MRC regressions (example CLI integration verified separately).
ChiMOL's native density/IO/demo checks passed 45 tests with actual offscreen
rendering and histogram-marker events. Source artifacts and raw bulk data are
cached, not shipped as repository dependencies.

CI installs SWIG and the native compiler in its separate package-test prefix:
the isolated conda build prefix cannot supply tools to the GIL/director compile
fixtures. Those fixtures select environment/sysconfig compilers with native
fallbacks; Windows uses MSVC DLL flags and the running Python's import library.
They remain real runtime tests, not skips hiding a missing test toolchain.

Build in an isolated standalone directory while the shared IMP tree is claimed.
No new dependencies. The backend is CPU-only and remains usable without IMP.
Future work: class-specific iterative averaging, independent reference learning,
FSC/resolution estimates and explicit label/linker geometry. None are
implied by the current particle average.
