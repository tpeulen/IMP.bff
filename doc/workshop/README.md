# FRET workshop: from a structure to a measured, modelled ensemble

Seven steps, one molecule. **BmrA** is a homodimeric ABC transporter with two
experimentally determined arrangements of its nucleotide-binding domains —
[6R72](https://www.rcsb.org/structure/6R72) (closed, outward-facing, X-ray)
and [8QOE](https://www.rcsb.org/structure/8QOE) (open, inward-facing,
cryo-EM). The workshop asks the question a FRET experiment on BmrA has to
answer — *which residues do I mutate, what will they tell me, and what do the
distances I measure say about the structure?* — and answers it with the
library rather than with slides.

Every notebook runs. They are executed against the build as part of preparing
this material, and the numbers printed in them come from those runs.

## Before the room

```bash
conda install -c conda-forge imp.bff      # the IMP module: needed from step 5 on
pip install --pre "imp-bff[notebooks]"    # or the core alone, for steps 1-4
imp_bff_fetch_data                        # rotamer libraries, once, 62 MB
```

The structures download themselves on first use into `_data/` beside these
notebooks, so the room needs the network once, at the start, not per attendee
per notebook.

`IMP.bff.get_build()` says which package is in the interpreter: `"core"` (pip)
or `"imp"` (conda). Steps that need IMP's optimizers or hierarchies say so.

## The steps

| notebook | the question | needs |
|---|---|---|
| `01_label_sites.ipynb` | Which residues can carry a dye at all? | core |
| `03_volumes_and_rotamers.ipynb` | Where does the dye actually go — as a volume, and as a real dye? | core |
| `05_docking.ipynb` | Given the distances, where do the two protomers sit? | IMP |
| `06_network_circle_plot.ipynb` | Which measurements does my model still disagree with? | core |
| `07_maxent_ensemble.ipynb` | Which *population* of structures fits, assuming as little as possible? | core |

Two of the steps are older notebooks that already existed and were kept
because they are good:

- `../../ipynb/example/labelizer_greedy_homodimer.ipynb` — choosing *sites* for
  a homodimer, where one cysteine mutation labels both protomers and the pairs
  come for free. This one is on BmrA.
- `../../ipynb/example/labelizer_greedy_pipeline.ipynb` — the monomer version:
  which *pairs* to measure, ranked by what each one buys.

## What is real and what is simulated

The structures, the accessible volumes, the label scores, the docking and the
maximum-entropy inversion are all real computations by `IMP.bff`.

The *measurements* are not: no experimental BmrA FRET network ships with this
repository, so the notebooks that need measured distances generate them from
one structure and add noise, and say so where they do it. Everything
downstream of that point is therefore a recovery test — which is the honest
way to teach it, because you can check the answer.

One caution the material makes explicit, because it costs people days: BmrA
has no ConSurf conservation data (checked against UniProt O06967, ConSurf-DB
and labelizer.org). The published label score multiplies a conservation term
in, so scoring BmrA with the paper model unchanged returns exactly zero for
every residue. The notebooks drop that term and say what it costs.
