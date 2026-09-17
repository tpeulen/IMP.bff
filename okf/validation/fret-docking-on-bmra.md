# FRET docking on BmrA: what the workshop notebook measured

Written while building `doc/workshop/05_docking.ipynb` (2026-09-17). Two of
these are library behaviour that will cost the next person a day, and one is a
result about the system rather than about the code.

## 1. The fps.json route and `get_av_from_pdb` do not agree on defaults

The same site gives a volume through one path and nothing through the other.

| | `get_av_from_pdb` (what `workshop.av` calls) | `ProbeNetworkRestraint` from an fps.json |
|---|---|---|
| attachment residue | side chain stripped | `strip_mask: ""` -- strips nothing |
| clearance | `allowed_sphere_radius` defaults to 5 | derived, 3 |

On BmrA, **6 of 12 sites came back empty through the docking path** while being
non-empty through the helper. The fix inside the notebook is to write the
values explicitly into every position of the fps.json:

```python
position["strip_mask"] = IMP.bff.default_strip_mask(residue_seq_number)
position["allowed_sphere_radius"] = 7.0
```

Anyone building an fps.json by hand hits this. The defaults should be made to
agree, or the fps.json writer should fill both fields; until then, an empty
volume in a docking run is more likely to be this than a buried site.

## 2. The `imp` clash radii make FRET docking unusable on a tight interface

Scored at the **deposited** 6R72 dimer -- the answer, not a perturbation:

| clash radii | excluded-volume term | data term |
|---|---|---|
| `imp` (default) | 59.85 | 3.77 |
| `olga` | 3.35 | 3.77 |

A factor of 17.9 against the data at the correct pose: the optimiser spends
its effort pushing the two protomers apart. `coarse_clash=True` is refused
with non-default radii, so the working combination is
`coarse_clash=False` with `clash_radii_source="olga"`.

## 3. BmrA is the wrong system for a *blind* rigid-body dock, and that is measured

The protomers are domain-swapped and intertwined: no rigid motion takes the
assembled dimer to a separated one without passing through hundreds to
thousands of units of overlap. Tried, and all landing 16-30 A from the answer:

- clash-weight sweep, `ev_weight` 1.0 / 0.1 / 0.02 / 0.0;
- separated starts, 15-60 A along the centre-centre axis;
- a short `IMP.bff.dock` Monte-Carlo run (60 frames x 10 steps, annealed).

What *does* work, and is what the notebook shows: screening (closed 0.74
against open 6.55 reduced chi2 -- the network tells the states apart), a
scored landscape, and the **radius of convergence** of the local minimiser.
Starting 0.8-2.3 A away converges to 0.3 A; starting beyond about 3 A the
minimiser slides off the interface and ends 20-29 A out with a *low* clash
score. One start at 4.87 A ends 6.47 A away with chi2_r 2.19 -- as good as the
truth -- and clash 160: the data alone does not exclude it, the excluded
volume does.

Rigid-body FRET docking earns its keep on surface-associating complexes
(`examples/structure/fret_docking.py`), not on an intertwined homodimer.

## 4. Two AV paths, 1.4 A apart

`workshop.av` and the assembly's own AV path differ by 1.4 A mean, 6.3 A max
on the same sites. The notebook draws its synthetic measurements around the
*engine's* column rather than the helper's, and prints the comparison, so the
recovery test is not scored against a different forward model than the one it
optimises.

## 5. The pair screen ranks pairs whose dye has nowhere to go

Found while writing `doc/workshop/02_which_pair.ipynb`, confirmed separately.

`labelizer_fret_pair_scores` scores a pair from the two sites' volumes, and
when a volume comes back empty it falls back to the attachment point and
**still scores and ranks the row**. On BmrA's closed state, with the stock
`LabelizerFRETOptions`:

- residues A422 and B422 have **zero accessible voxels** -- the library says so
  on stderr, `AV P30014: no accessible voxel ... clearance below half the
  linker width walls the source in`;
- both clear the label-score threshold at 1.52, because the label score asks
  whether a cysteine belongs at that position, not whether a dye on a 20 A
  linker can reach anywhere from it;
- **247 of 7750 pairs involve residue 422**, and they carry ordinary-looking
  scores and distances: `A551-B422` sits at rank 96 with value 1.5702 and a
  quoted distance of 51.9 A.

A warning on stderr is not enough for a number that goes into a ranked table a
user reads. Either the row should carry a status the caller can filter on, as
`LabelizerScore.status` already does for unscored residues, or the pair should
be dropped from the screen. Until then, a screen's shortlist has to be audited
by rebuilding its sites' volumes -- which is what notebook 02 does, and how
this was found.
