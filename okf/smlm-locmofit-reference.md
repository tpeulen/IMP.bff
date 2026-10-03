# LocMoFit as a reference for BFF SMLM

Primary reference: Wu et al., *Maximum-likelihood model fitting for quantitative
analysis of SMLM data*, Nature Methods 20, 139–148 (2023),
[DOI 10.1038/s41592-022-01676-z](https://www.nature.com/articles/s41592-022-01676-z),
[open full Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC9834062/).
No source-code port or claim of whole-framework equivalence is made.

| Paper method | BFF implementation / next work |
|---|---|
| Discrete expected-emitter PDF, each observation's own anisotropic precision | `SMLMPointModel.evaluate`, normalized per-emitter mixture and persistent model spatial tree |
| Contamination/background mixture and extra uncertainty | Explicit ROI-normalized uniform background plus intrinsic variance sigma²+epsilon² |
| Rigid position/orientation fitting | Native full 6D likelihood fit; particle-level angular search followed by damped information-matrix refinement |
| Gaussian point-cloud correlation | `SMLMGaussianOverlap`, covariance sums and exact support-aware pruning over an owning localization tree |
| Use expected label sites in structural modelling | IMP-layer `SMLMRestraint`, analytic XYZ forces and native rigid-body optimization; likelihood or persistent Gaussian-overlap mode |
| AIC and corrected AIC | `smlm_aic` / `smlm_aicc`, actual summed log likelihood and localization count |
| Reference-assisted averaging | Native selective particle transforms/deposition, measured rotated covariance, equal particle weights and disjoint particle half maps |
| Continuous parameterized geometries / composite multicolor models | Follow-up: sampled geometric model adapters, parameter links, channel-specific likelihood and shared poses |
| Algorithm 1 de novo particle fusion | Follow-up: all-against-all similarity ranking, independently learned half references, cumulative fusion and iterative registration |
| Chained fitting and pseudotime reconstructions | Follow-up: model/blur stages, fitted-parameter selection and sliding particle windows |
| Reported Hessian confidence intervals and robust simulation validation | Follow-up: exact curvature or validated uncertainty estimation, blinking/linkage/label-efficiency simulation |

The distinction between overlap and likelihood matters. A localization KDE
evaluated at model positions is a useful density score, but it is the reverse
direction of the paper's measured-localization likelihood. Gaussian overlap
convolves both covariances; the forward point-model PDF uses each observation's
own measurement covariance and models emitter positions as expected locations.

## Reference data and source audit

The paper's full experimental data are
[BioImage Archive S-BIAD563](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD563).
Its worked tutorial includes `U2OS_Nup96_BG-AF647_demo_sml.mat`: 37,129 measured
xyz/precision records, 99 populated positive site groups, 706 unassigned rows.
The two `NPC3D_step*_LocMoFit.mat` files and `dualRing_model1_fitPar.csv` are
parameter/model setup files; they do not contain fitted poses for those sites.
In the full cell1 `_sml.mat`, saved `allParsArg.value` are fitted values and
`parsInit.init` are starting values; complete ROI/membership and Euler-transform
semantics must be reconstructed before importing them as rigid poses.

All five single-color Nup96 experimental localization CSVs and the full cell1
MAT were downloaded into the ignored `.omx/reference/locmofit/data` directory.
The exact supplementary LocMoFit v1.1 source is extracted under
`junk/LocMoFit-v1.1`, with a separate sparse SMAP checkout for reference. These
are inert reference copies; MATLAB/MEX code is not run or linked into BFF.
Primary URLs, byte counts, SHA256s and source revisions are recorded in
[`smlm-reference-data.json`](smlm-reference-data.json). They are deliberately not
bulk-committed. The public source can be retrieved again using that manifest.

`read_smlm_csv` ingests those source tables directly into the native tree, with
explicit nm→Angstrom conversion and positional uncertainties. Source site IDs
can be selected natively. `examples/imaging/smlm_structure.py` connects a selected
measured site to arbitrary expected emitter positions through the IMP restraint;
it is not a claim that these data refine every protein in the NPC scaffold.

ChiMOL's source preparation records SHA256, source row/site identities, original
xyz and precisions, acquisition columns and nm→Angstrom conversion. There is no
annular selection. The default pipeline applies the paper's 5 nm lateral
precision threshold and an explicit 30 nm axial precision gate, then a minimum
80-localization particle gate. This tutorial reconstruction is a subset/example,
not a numerical reproduction of the paper's thousands-of-sites analysis.

The angular average uses fixed human
[7R5J assembly 1](https://www.rcsb.org/structure/7R5J) Nup96 terminal proxies:
32 expected points, deposited physical coordinates, coarse angular search over
the model's equivalent 45-degree interval followed by full 6D refinement. No
experimental point is duplicated, snapped, scaled or replaced by a reference
coordinate. The raw density is accumulated from the transformed measured points
and their full rotated uncertainty. This prior makes the reconstruction
reference-assisted; it does not establish de novo eightfold symmetry.

Measured default result: 94 accepted pores, 12,676 localizations. Mean per-pore
NLL improves from 22.8112 to 21.1670 (Angstrom units, fixed ROI policy/background);
70 fits satisfy the numerical stopping criterion and 24 hit the 80-iteration
budget. All poses and fit diagnostics are recorded. Reference terminal proxies
omit SNAP/linker coordinates and intrinsic blur is fixed to 5 nm, so the entire
624,240-Cα human scaffold is contextual display, not a structural refinement.

The 3D map and both particle half maps are standard MRC at 3 nm voxels. Density
controls remain ChiMOL's ordinary cryo-EM histogram, contour, color, opacity and
surface/mesh/solid display controls. Dataset-specific preparation is tooling;
scoring, registration, covariance rotation and deposition run in BFF C++.
