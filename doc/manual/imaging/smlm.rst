SMLM particle scoring and selective averaging
============================================

``IMP.bff.SMLMIndex`` stores three-dimensional measured localizations in a
balanced spatial tree. Coordinates and standard deviations must use one physical
unit. Keep measured localization precision separate from a visualization blur.

.. code-block:: python

   import numpy as np
   import IMP.bff as bff

   # xyz/sigma are N x 3, ids are source particle identities.
   index = bff.SMLMIndex(xyz.ravel(), sigma.ravel(), np.ones(len(xyz)), ids)
   selected = bff.select_smlm_precision(index, [15, 15, 30])  # example nm limits
   frames = bff.align_smlm_particles(selected, [], 70, 80)
   average = bff.average_smlm_particles(
       selected, frames.particle_ids, frames.transforms,
       [81, 81, 69], [-120, -120, -102], [3, 3, 3])
   score = index.evaluate_score(model_label_positions.ravel())
   print(score.score)  # model positions evaluated against measured density

The example's length unit is nm. For MRC export, supply all coordinates,
precision, spacing and origin in Angstrom from the outset, then call
``write_smlm_average_mrc(average, "density.mrc")``. Maps 1 and 2 write the
particle half maps. Density arrays use x-fast indexing and voxel-center origins.

Selection can name specific source particle IDs, set minimum/maximum particle
counts, or gate externally computed per-particle quality scores. Retained
measurements are accumulated using their full rotated covariance. Each particle
has equal integrated weight by default, and half maps normalize separately.
Finite grid extents and Gaussian truncation remove mass; removed mass is not
redistributed. Record your selection, transforms, uncertainties and source IDs.

The automatic frame initializer independently centers particles and aligns their
least-variance normals; it retains azimuth and imposes no symmetry or scaling.
Use supplied rigid transforms when available. Native ``refine_smlm_translation``
and ``refine_smlm_rigid`` are local likelihood optimizers requiring initial overlap.
Template-based alignment requires independently learned half references for
independent reconstruction; a disjoint split of points alone is insufficient.

The score is mean query-to-density negative log likelihood, lower being better.
Its gradient is analytic except at the finite-support boundary. Background is a
positive density floor in inverse volume units; it must be rescaled when changing
units. Low scores do not prove the full measured distribution is explained.

ChiMOL exposes ``smlm_average file.smlm.npz, name [, particle IDs]`` and its NPC
density demo. The average measures Nup96 labels; the human structural scaffold
is context, not a claim that every modeled protein is localized by that label.

Structural modelling with IMP
-----------------------------

Read source localizations once with ``read_smlm_csv(path, 10.0)`` for a nm CSV
used with Angstrom structural coordinates. The importer preserves measured
positional precision and source particle IDs; it rejects absent z/precision
unless explicit caller defaults are supplied. Select a saved group with
``select_smlm_particles(index, [particle_id])`` without transferring the full
table into Python. Weights are one by default, not photon counts.
Conflicting aliases for a field are rejected. Source ID zero is unassigned by
default, matching SMAP; set ``SMLMCSVOptions.zero_particle_id_is_unassigned`` to
``False`` explicitly for zero-based group identities.

``SMLMRestraint(model, emitter_particles, observations, likelihood_options)``
provides native IMP scores and analytic XYZ derivatives, including rigid-body
member forces. ``likelihood_options`` specifies a finite ROI containing the
observations, a normalized contamination fraction and optional intrinsic blur.
Emitter particles represent expected fluorophore sites. Atomic mass and radius
are not measurement noise. The default score is observation-weighted mean NLL;
sum mode uses weighted summed NLL, independently of external IMP restraint weights.

The Gaussian-overlap constructor instead accepts full model covariances and
reuses an owning measured-data tree. Covariances are fixed in the observation
frame and do not automatically rotate with rigid-body members. The point-model
likelihood is preferable for rigid fitting with anisotropic measurement noise.

``examples/imaging/smlm_structure.py`` demonstrates source CSV selection and
native IMP optimization of an arbitrary emitter model. It is a local fit needing
initial overlap, not a guarantee of a globally correct structural solution.
