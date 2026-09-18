"""One-off: switch the conservation term on in 03_labelizer.ipynb, execute, save.

Cell sources are matched by prefix and replaced; two cells (the what-changes
markdown and figure) are inserted after the model cell. Everything downstream
of the model -- components(), the per-term panels, the candidates -- picks the
term up through `model` and `used`.
"""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.read("03_labelizer.ipynb", as_version=4)


def replace_cell(prefix, source):
    hits = [i for i, c in enumerate(nb.cells)
            if c.cell_type in ("code", "markdown") and c.source.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    nb.cells[hits[0]].source = source
    return hits[0]


md_where = nbf.v4.new_markdown_cell(r"""## The conservation term, and where its grades come from

The published model carries a conservation term at weight 1. Conservation is
not computed from the structure -- it comes from a multiple alignment, as
ConSurf grades, and it has to exist for the protein in front of you.

For hGBP1 it does, from the second place you look. The paper's own webserver,
[labelizer.org](https://labelizer.org), is asked first over its REST surface
(the same one the `ipynb/example` notebooks drive): it lists the entries it
holds ConSurf grades for and serves the grades file directly. For 1F5N, the
hGBP1 crystal entry, it holds none -- its internal ConSurf job has never
delivered for that entry (an analysis job with the conservation toggle on,
submitted 2026-09-18, terminated with exposure, cysteine and
secondary-structure tables and no conservation). ConSurf-DB, the precomputed
database, has a finished run: 6252 homologues found, 300 in the final
alignment. `ws.hgbp1_consurf_grades()` asks the webserver first, falls
through to the database, downloads once into `_data/`, and writes one atom
per residue of the workshop topology with the normalised grade in the
B-factor column -- exactly the file `labelizer_read_consurf` reads.

One remap on the way in, checked rather than assumed. The grades arrive on
the crystal entry's numbering: 1F5N models residues 7-583 with a disordered
gap at 158-164 -- 570 residues, which the workshop topology numbers 1-570.
The k-th residue of the topology takes the grade of the k-th modelled
position, and the mapping is verified by residue identity before it is used:
all 570 match. That gap is also where the topology's `A`/`B` label boundary
at 151/152 falls -- the split the coarse-grained model inherited from what
the crystal resolved.""")

code_model = nbf.v4.new_code_cell(r"""model = paper                          # the full published model, conservation on
grades = ws.hgbp1_consurf_grades()     # fetched once into _data/, reused after

grade_map = bff.labelizer_read_consurf(str(grades))
residues = bff.labelizer_read_structure(str(structures["start"])).residues
covered = sum(1 for r in residues
              if bff.labelizer_residue_key(r.chain, r.seq_id) in grade_map)
values = np.array(sorted(grade_map.values()))
print(f"grades for {covered}/{len(residues)} residues of the topology, "
      f"{len(set(grade_map.values()))} distinct, "
      f"{values.min():.2f} conserved to {values.max():.2f} variable")


def label_scores(pdb_path):
    '''{(chain, residue): combined score} for one structure, conservation on.'''
    rows = bff.labelizer_score_structure(str(pdb_path), model,
                                         bff.LabelizerOptions(), str(grades))
    return {(r.asym_id, r.seq_id): r.value
            for r in rows if r.score_type == "combined"}


ls = {state: label_scores(path) for state, path in structures.items()}
for state, table in ls.items():
    finite = np.array([v for v in table.values() if np.isfinite(v)])
    print(f"{state:5s} {len(table):5d} residues, score {finite.min():.2f}-{finite.max():.2f}, "
          f"mean {finite.mean():.2f}")""")

md_change = nbf.v4.new_markdown_cell(r"""### What the conservation term changes

The same start structure scored with and without the term: x without, y with,
and the colour the ConSurf grade itself -- red conserved, green variable.
Conserved residues sit below the diagonal: the evolutionary record says a
position it is holding is a poor place to force a dye, and the term pushes
such sites down. On this protein the gradient is where the biology says it
should be -- the G-domain core is the conserved part, the stalk that swings
out is the variable one.""")

code_change = nbf.v4.new_code_cell(r"""no_cs = bff.LabelizerParameterList()
for parameter in paper:
    if parameter.tag != "cs":
        no_cs.append(parameter)

comb_cs = {bff.labelizer_residue_key(r.asym_id, r.seq_id): r.value
           for r in bff.labelizer_score_structure(
                   str(structures["start"]), model, bff.LabelizerOptions(),
                   str(grades))
           if r.score_type == "combined"}
comb_ncs = {bff.labelizer_residue_key(r.asym_id, r.seq_id): r.value
            for r in bff.labelizer_score_structure(
                    str(structures["start"]), no_cs, bff.LabelizerOptions(), "")
            if r.score_type == "combined"}

keys = sorted(set(comb_cs) & set(comb_ncs), key=lambda k: int(k[1:]))
x = np.array([comb_ncs[k] for k in keys])      # conservation off
y = np.array([comb_cs[k] for k in keys])       # the full published model
g = np.array([grade_map[k] for k in keys])     # the ConSurf grade itself

top_cs = set(sorted(comb_cs, key=comb_cs.get, reverse=True)[:10])
top_ncs = set(sorted(comb_ncs, key=comb_ncs.get, reverse=True)[:10])
print(f"top-10 sites with/without conservation: {len(top_cs & top_ncs)} of 10 shared")
demoted = sorted(top_ncs - top_cs, key=comb_ncs.get, reverse=True)
print("conservation demotes:", " ".join(f"{k}({comb_cs[k]:.2f})" for k in demoted))

fig, ax = plt.subplots(figsize=(5.6, 5.4))
lim = (min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05)
ax.plot(lim, lim, "k--", lw=0.8)
sc = ax.scatter(x, y, c=g, cmap="RdYlGn_r", s=22)
fig.colorbar(sc, ax=ax, label="ConSurf grade")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("LS, conservation off")
ax.set_ylabel("LS, full published model")
ax.set_title(f"hGBP1 frame {ENDS['start']}: what conservation changes")
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()""")

md_five = nbf.v4.new_markdown_cell(r"""## One number is five numbers

The combined score is the **weighted geometric mean** of the terms, so reading
it alone throws away the reason for it. Two sites can score the same because
one is beautifully exposed and barely cysteine-like, and the other the other
way round -- and those two sites behave differently when you put a dye on them.

`labelizer_score_structure` returns every term, not just the combination: one
row per (residue, score type). Here they are for the same structure.""")

code_components = nbf.v4.new_code_cell(r"""def components(pdb_path):
    '''{(chain, residue): {score type: value}} -- every term, not just the total.'''
    rows = bff.labelizer_score_structure(str(pdb_path), model,
                                         bff.LabelizerOptions(), str(grades))
    table = {}
    for row in rows:
        table.setdefault((row.asym_id, row.seq_id), {})[row.score_type] = row.value
    return table


parts = {state: components(path) for state, path in structures.items()}
TERMS = ["solvent_exposure", "cysteine_resemblance", "secondary_structure",
         "conservation", "charge_environment", "tryptophan_proximity"]
TAGS = {"solvent_exposure": "se", "cysteine_resemblance": "cr",
        "secondary_structure": "ss", "conservation": "cs",
        "charge_environment": "ce", "tryptophan_proximity": "tp"}
weights = {p.tag: p.weight for p in model}
print("term weights in this model:", {TAGS[t]: weights[TAGS[t]] for t in TERMS})

best = max(((values["combined"], key) for key, values in parts["start"].items()
            if np.isfinite(values["combined"])))[1]
print(f"\nthe best-scoring residue of the start structure, term by term: "
      f"{best[0]}{best[1]}")
site = parts["start"][best]
for term in TERMS + ["combined"]:
    print(f"  {term:24s} {site[term]:6.3f}")""")

md_combination = nbf.v4.new_markdown_cell(r"""### The combination, checked

Weight 1 on solvent exposure, cysteine resemblance, secondary structure and
conservation; weight 0 on charge environment and tryptophan proximity, so
those two are computed and reported but do not enter. The geometric mean of
the four that count reproduces the combined score exactly -- which is also
why a single term at zero takes the whole site to zero, as conservation did
above when its grades were missing.""")

code_used = nbf.v4.new_code_cell(r"""used = ["solvent_exposure", "cysteine_resemblance", "secondary_structure",
        "conservation"]
show_sites = [key for _, key in sorted(
    ((values["combined"], key) for key, values in parts["start"].items()
     if np.isfinite(values["combined"])), reverse=True)[:4]]

print(f"{'site':>6} {'se':>7} {'cr':>7} {'ss':>7} {'cs':>7} {'geo.mean':>9} {'combined':>9}")
for chain, resi in show_sites:
    values = parts["start"][(chain, resi)]
    geometric = np.prod([values[term] for term in used]) ** (1.0 / len(used))
    print(f"{chain + str(resi):>6} " + " ".join(f"{values[t]:7.3f}" for t in used)
          + f" {geometric:9.3f} {values['combined']:9.3f}")""")

md_terms = nbf.v4.new_markdown_cell(r"""### Every term along the sequence

The terms disagree, and where they disagree is where the choice is actually
made. Solvent exposure is the one that moves between the two ends of the
transition -- it is a property of the surface, and the surface is what the
stalk motion rearranges. Cysteine resemblance is a property of the sequence
alone and does not move at all; secondary structure moves only where the
trajectory melts or forms a turn.

Conservation does not move either, for the same reason: none of it comes from
the structure, so the two frames carry identical values. What varies is where
it sits along the chain -- low over the G-domain core the evolution is
holding, rising along the stalk out to the variable C-terminus, which is the
gradient the figure above showed as colour.

Tryptophan proximity is worth a word, because its panel is a flat line at 1.
That is the term declining to say anything, not the protein having no
tryptophans -- it has four in this model. In the published parameter list `tp`
carries weight 0 and no lookup table (printed as `-` above), so it returns the
neutral ratio everywhere. Charge environment, also at weight 0, does have a
table and does vary; it is reported and ignored.

The x axis runs over the whole polypeptide, 1 to 570, across the `A`/`B` label
boundary at residue 151, because the label boundary is bookkeeping and the
chain is one chain.""")

replace_cell("## The conservation term, and why it is switched off here", md_where.source)
i_model = replace_cell("model = bff.LabelizerParameterList()", code_model.source)
nb.cells[i_model + 1:i_model + 1] = [md_change, code_change]
replace_cell("## One number is four numbers", md_five.source)
replace_cell("def components(pdb_path):", code_components.source)
replace_cell("### The combination, checked", md_combination.source)
replace_cell('used = ["solvent_exposure"', code_used.source)
replace_cell("### Every term along the sequence", md_terms.source)

client = NotebookClient(nb, timeout=1800,
                        kernel_name=nb.metadata.get("kernelspec", {}).get("name", "python3"))
client.execute()
nbf.write(nb, "03_labelizer.ipynb")

errs = [o for c in nb.cells if c.cell_type == "code"
        for o in c.get("outputs", []) if o.output_type == "error"]
print("written; errors:", len(errs))
for e in errs:
    print(e.ename, e.evalue)
