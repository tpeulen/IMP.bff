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

## 5. The pair screen claimed a dye model it had not used -- fixed

`labelizer_fret_pair_scores` scores a pair from the two sites' accessible
volumes. When a volume comes back empty it measures between the two attachment
points instead -- which is a reasonable fallback -- and it used to label the
row `PROBE_MODEL_ACCESSIBLE_VOLUME` anyway, so nothing downstream could tell
the two apart.

Measured on BmrA's closed state with the screen's **own** default options
(linker 20/4.5, r1 9.0, grid 1.5):

- 125 sites clear the label-score threshold, and **12 of them have no
  accessible volume at all** (A125, A208, A219, A376, A421, A474 and their
  chain-B twins);
- those twelve sites carry **1422 of the 7750 pairs** -- 18 % of the screen --
  and every one of those rows was presented as a volume-based distance.

The fix marks the row instead of removing it: a row whose distance came from
the attachment points now carries `PROBE_MODEL_CBETA`, in the single-state
screen, the two-state screen and the refinement pass alike. A caller filters
on `probe_model`, the way the field was always meant to be read, and the
ranking is unchanged for anyone who does not.

**A correction to an earlier version of this note**, worth keeping because it
is the kind of mistake that is easy to repeat: the first write-up named
residue 422 as a dead site with 247 pairs resting on it. That was measured
with the *workshop's* dye geometry (r1 3.5 with a 5 A clearance), not the
screen's. Under the screen's own parameters A422 has 32 voxels -- thin, not
absent. The defect is real and larger than first reported; the example was
wrong because two different dye geometries were compared as though they were
one.

### Which builder drew the volume: the screen does not always use the core one

Worth knowing before comparing one build's screen against another's, and the
thing that made the first version of this fix's test fail. `IMP.bff` installs
an accessible-volume *door*, and an IMP build replaces the core builder with
IMP's own (`get_av_from_structure`: IMP::atom's reader and IMP's radii) at
load time. `labelizer_get_av_door_name()` says which one is in place --
`"imp"` in a conda build, the core one in a pip build.

They do not agree site for site. On `test/input/labelizer/1DDB-39.pdb`, of 195
residues the core builder finds no volume at 46 and the IMP builder at 42,
agreeing on 39: about ten sites change their answer with the build. A27 is one
of them -- 1085 voxels under the core reader, none under IMP's -- which is why
a check that rebuilds a site's volume has to rebuild it through the installed
door, or it ends up measuring the distance between the two readers instead of
the thing it meant to test. `test/label/test_labelizer_fret.py` picks the
builder from the door name for exactly that reason.

The site counts above were taken from the screen itself and hold for both: 12
of 125 sites empty either way on BmrA's closed state, which the pair
arithmetic confirms (7750 - 1422 = 6328 = C(113, 2)).
