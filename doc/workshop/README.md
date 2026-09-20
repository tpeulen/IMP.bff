# FRET workshop: from a structure to a measured, modelled ensemble

Seven steps. The question they answer together is the one a FRET experiment
has to answer — *which residues do I label, what will they tell me, and what do
the distances I measure say about the structure?* — and they answer it with the
library rather than with slides.

Two molecules, each where it belongs:

- **hGBP1**, human guanylate-binding protein 1, carries the single-chain
  steps. Its conformational transition is a recorded trajectory
  (`examples/structure/GBP/hgbp1_transition.dcd`, 58 frames on
  `hgbp1_cg.pdb`), so the ensemble the measurements have to resolve is real
  rather than interpolated, and the notebooks animate it.
- **BmrA**, a homodimeric ABC transporter, carries the two steps that need two
  protomers: choosing sites when one cysteine mutation labels every protomer,
  and docking one protomer against the other. Its two states are
  [6R72](https://www.rcsb.org/structure/6R72) (closed) and
  [8QOE](https://www.rcsb.org/structure/8QOE) (open).

Every notebook is executed against the build before it is published, and the
numbers printed in them come from those runs.

## Scientific provenance and further reading

The workshop uses the same ideas in a deliberately inspectable order: select
labelling sites, choose FRET pairs, model an ensemble, then make the least
committal population consistent with the data. The primary sources are:

- **Labelizer** — the [web application](https://labelizer.org/) and its
  [systematic label-site selection paper](https://doi.org/10.1038/s41467-025-58602-y).
  The score and pair-selection notebooks make the assumptions behind that
  workflow explicit rather than treating a ranking as an answer.
- **Software used by the workflow** — [Labelizer](https://labelizer.org/) for
  label-site and pair scoring, [Olga](https://github.com/Fluorescence-Tools/olga)
  for optimal FRET-assay design, and [FPS](https://www.mpc.hhu.de/software/fps)
  for accessible-volume and FRET-restrained structural modelling.
- **High-precision FRET modelling** — Kalinin *et al.*,
  [Nature Methods **9**, 1218–1225 (2012)](https://doi.org/10.1038/nmeth.2222).
  This is the accessible-volume and uncertainty-aware foundation for using
  flexible dyes as structural restraints.
- **FRET-assisted integrative modelling** — Dimura *et al.*,
  [Current Opinion in Structural Biology **40**, 163–185 (2016)](https://doi.org/10.1016/j.sbi.2016.11.012).
  It connects experiment design, distance networks and structural modelling.
- **hGBP1 conformational ensemble** — Peulen *et al.*,
  [eLife **12**, e79565 (2023)](https://doi.org/10.7554/eLife.79565).
  This is the primary integrative-dynamics study behind the hGBP1 system used
  through the single-chain workshop steps.
- **Maximum-entropy ensembles** — Dittrich *et al.*,
  [Journal of Chemical Theory and Computation **19**, 2389–2409 (2023)](https://doi.org/10.1021/acs.jctc.2c01090).
  The final notebook uses this framing: infer a population and retain the
  uncertainty the restraints cannot resolve.

The [Peulen publications page](https://www.peulen.xyz/) collects these and
related work, including an author-hosted copy of the maximum-entropy article.

## Before the room

```bash
conda install -c conda-forge imp.bff      # the IMP module: needed for the docking step
pip install --pre "imp-bff[notebooks]"    # or the core alone, for everything else
imp_bff_fetch_data                        # rotamer libraries, once, 62 MB
```

BmrA's two structures download themselves on first use into `_data/` beside
these notebooks; hGBP1's trajectory ships with the repository. `IMP.bff.get_build()`
says which package is in the interpreter: `"core"` (pip) or `"imp"` (conda).

## The steps

| notebook | the question |
|---|---|
| `01_accessible_volumes.ipynb` | Where can the dye be? Label models: accessible volumes (Part A) and explicit rotamers (Part B), with comparison graph. |
| `03_labelizer.ipynb` | Which residues can carry a dye at all — every term of the label score, not just the total. |
| `03b_labelizer_pairs.ipynb` | The Labelizer's own pair score, in one state and across two. A companion to step 3. |
| `04_pair_selection_olga.ipynb` | **hGBP1, one polypeptide:** choose individual FRET *pairs* and ask which measurement to make first (Olga's criterion). |
| `05_site_selection_homodimer.ipynb` | **BmrA, a homodimer:** choose labelling *sites*, because every mutation creates all of the pairs it implies. |
| `06_docking_homodimer.ipynb` | Given the distances, where does the second protomer sit? |
| `07_distance_networks.ipynb` | Reading a network: which measurements a model still disagrees with, and what the pattern means. |
| `08_ensemble_maxent.ipynb` | Not one structure but a population, with as little assumed as the data allow. |

## What is real and what is simulated

The structures, the trajectory, the accessible volumes, the rotamer ensembles,
the label scores, the selection, the docking and the maximum-entropy inversion
are all real computations by `IMP.bff`. The labelling positions and the Förster
radius in the hGBP1 steps are the published ones.

The *measurements* are not. `examples/structure/GBP/hGBP1.fps.json` holds 68
real measured distances, but they belong to the hGBP1 **dimer**: all 34
distinct values are inter-protomer, and 14 of the 20 whose residues exist in
the single-chain topology lie outside the whole range one molecule can produce
over the entire trajectory. Notebook 07 shows that rather than asserting it.
So where measured distances are needed, the notebooks generate them from a
known structure with stated noise, and say so where they do it — which also
makes every one of those steps a recovery test, where the answer can be
checked.

## Two things the material insists on

**A score is not a usable site.** A residue can pass the label score and have
nowhere for the dye to go. The pair screen then falls back to the attachment
point and ranks the pair anyway: on hGBP1, 123 of 1953 pairs rest on a site
with no accessible volume, the best of them at rank 73. Every notebook that
ranks pairs audits its shortlist by rebuilding the volumes.

**A better χ² is not a better model.** The networks step scores data from the
middle of a transition against the two end states: one of them wins at χ² 7.7,
and both are wrong — and they fail with opposite signs, which is what a
two-state model of a continuous motion cannot express.
