---
title: Fast ConSurf -- conservation grades in seconds, natively
status: target met (20 chains: median 5.2 s on SSD, grades closer to the reference than the sampling spread); open: distribution, a larger benchmark
updated: 2026-10-01
---

# Fast ConSurf

`IMP.bff.compute_consurf` (`imp_bff consurf`) computes ConSurf's per-residue
conservation grades (Ashkenazy et al., NAR 44:W344, 2016) for a protein chain
in one process, with no external programs. The ConSurf server runs HMMER
against UniRef90, realigns with MAFFT and runs Rate4Site, which takes minutes
to hours per chain. Here every step is imp.bff C++, and the data sit in one
ptolib container. The target is **under 10 s per chain on UniRef**. The
reference chain throughout is 1lk2 chain A (274 residues).

This page is the methods record for a paper: what each stage does, the
parameters, why each choice was made, and the numbers measured so far.

## Pipeline

```
query sequence
  │
  ├─ Stage 1: find cluster representatives (UniRef50, 38.8 M sequences)
  │     (a) k-mer prefilter + Smith-Waterman over all representatives   [228 s]
  │     (b) ESM-2 embedding -> nearest representatives (PQ index)
  │           -> Smith-Waterman on those candidates only            [~2 s]
  │
  ├─ Stage 2: search the members (UniRef90) of the clusters found  [~11 s; early stop not yet measured]
  │
  ├─ Homologue selection by ConSurf's rules (E, coverage, identity,
  │     redundancy, cap 150 sampled)
  │
  ├─ Query-anchored alignment of the selected homologues
  │
  └─ Rate4Site's method (JTT + discrete gamma, NJ tree, empirical Bayes)
        -> normalised scores -> ConSurf's nine grades              [1.4 s]
```

## 1. The sequence database: UniRef as one clustered `.pto`

- **Content.** UniRef50 representatives at the root: 38,840,027 sequences,
  12.18 G residues. Below each one, as a ptolib *level*, are the UniRef90
  members of its cluster: 121,494,192 sequences, 45.55 G residues. The file
  `/Volumes/SD1TB/sequences/uniref.pto` is 23.4 GB. As plain FASTA the same
  data take 51 + 14.7 GB.
- **Membership** comes from `uniref50.xml.gz` (UniRef50 cluster → UniRef90
  members); no member is left unplaced. The route through UniProt's ID
  mapping (`idmapping_selected.tab.gz`) is also supported, with an `orphans`
  group for members the mapping does not place
  (`create_sequence_clusters`).
- **Storage** (ptolib format 6; the changes were made upstream in
  `../ptolib`):
  - residues are Huffman-coded, about 4.2 bits each (5-bit packing where
    Huffman does not help), and any single sequence decodes on its own;
  - members are stored as *edit scripts* (M/I/D operations plus Huffman
    literals) against their representative or an earlier sibling. The
    reference is chosen per cluster by 3-mer similarity, at most 16 siblings
    back, with chains at most 8 deep. The alignment behind a script is a
    banded local alignment capped at 16 M cells, which keeps the build under
    about 3 GB of memory;
  - headers are zstd-compressed in 16 KB segments with a trained dictionary;
  - compact ragged offsets and level extents mean a cluster's members are one
    contiguous row range.
- **Build:** `imp_bff sequence-db clustered`. It runs in two streaming passes
  in bounded memory (512 bucket files), with parallel alignment.

## 2. The search (MMseqs2's method, implemented independently)

Following Steinegger & Söding (Nat Biotechnol 35:1026, 2017). MMseqs2 was
read for behaviour only, never copied (it is GPL), and was A/B-tested as a
black box.

- **Prefilter.**
  - For each query position, its 5-mer and every 5-mer scoring at least
    T = 18 against it under BLOSUM62 (the neighbourhood).
  - The database streams past once. A target 5-mer found in a neighbourhood
    is a hit on diagonal `target position − query position`.
  - Two hits on one diagonal trigger an ungapped diagonal score (at least
    15 bits to pass).
  - The best `max_candidates` targets per query go on to alignment.
- **Alignment.** Smith–Waterman with affine gaps (Gotoh), BLOSUM62, gap open
  11 and extend 1 (a gap of length L costs 11 + L).
- **Significance.** Karlin–Altschul E-values with λ = 0.267 and K = 0.041
  (Altschul et al., NAR 25:3389, 1997), over the residues of the database
  searched.
- **Execution.** The database is read segment by segment from its memory map
  on all cores. Memory holds only the query tables, one diagonal array per
  thread and the candidates, whatever the database size.
- **Output.** A query-anchored alignment of each hit, the equivalent of an
  A3M row, used directly as the MSA.

## 3. Two stages through the cluster hierarchy

Every UniRef90 cluster lies in one UniRef50 cluster, and the UniRef50
representative is at least 50 % identical to its members. A homologue that
ConSurf keeps is at least 35 % identical to the query. Its representative is
therefore found too, provided stage 1 uses looser cut-offs:

- **Stage 1:** search the representatives, keeping up to 5,000 per query
  with E ≤ 10 (`SequenceClusterSearchOptions.max_representatives`,
  `max_representative_evalue`).
- **Stage 2:** search only the members of the clusters found, each cluster
  one contiguous row run (`search_sequence_database_rows`). E-values are
  against all member residues, so they equal those of a full UniRef90 search.
- **Early stop (optional, `min_homologues`):** clusters are taken best first
  by their representative's E-value, in growing batches, until every query
  has this many hits passing E ≤ 1e-4 and identity ≥ 35 %. The homologues are
  then the closest ones rather than a sample of all of them.

On a test fixture (a 60-sequence family plus 1,500 decoys), the two-stage
search returns exactly the hits of a single-stage search of all members
(`test_sequence_clusters.py`). On 1lk2 it has not yet been compared with a
full UniRef90 search at the homologue level; only timings are below.

## 4. The embedding prefilter for stage 1

Stage 1 over 38.8 M representatives costs 228 of 240 s (the k-mer scan). It
is replaced by nearest-neighbour search in a protein language model's
embedding space. The survivors are still aligned by Smith–Waterman, so the
E-values and the rest of the pipeline stay the same, and only recall can
change.

### 4.1 Model

- **ESM-2 t12 35M** (Lin et al., Science 379:1123, 2023): 12 layers, width
  480, 20 heads.
- **Per-sequence embedding:** the mean of the last layer over residues
  (without `<cls>`/`<eos>`). Sequences longer than 1,022 residues are
  embedded from their first 1,022, as the model was trained. In UniRef50
  1,378,753 representatives (3.5 %) are longer. Planned: extra window
  vectors (start 768, stride 768), which adds 2.0 M vectors and 1.37 G
  residues.
- **Model choice**, Swiss-Prot recall of true homologues by raw cosine at an
  estimated full-database rank:

  | model | @100 | @1000 | @5000 |
  |---|---|---|---|
  | 8M | 0.39 | 0.70 | 0.80 |
  | 35M | 0.445 | 0.80 | 0.878 |
  | 150M | 0.43 | 0.775 | 0.85 |
  | 650M | 0.446 | 0.79 | 0.84 |

  35M matches the largest model and is the cheapest that does. On an RTX 4000
  Ada it embeds 100k Swiss-Prot sequences in 189 s.

### 4.2 Contrastive projection head

Raw embeddings put too many true homologues outside the top thousands, so a
small head is trained so that cosine similarity ranks homologues first.

- **Architecture:** standardise each feature (mean and SD over all
  Swiss-Prot embeddings, a label-free statistic),
  then MLP 480 → 1024 → 1024 → 256 with exact GELU, plus a linear skip
  480 → 256, then L2-normalise.
- **Training pairs:** bff's own k-mer search (E ≤ 1e-3, 3,000 candidates) of
  2,000 random Swiss-Prot queries (60–1,000 residues) against all of
  Swiss-Prot (575,748 sequences). Positives are hits with E ≤ 1e-4 and
  identity ≥ 35 %, ConSurf's own homologue definition: 386,680 pairs.
- **Held-out test:** 200 other random queries (80–800 residues). Every pair
  that touches a test query or any of its true homologues (48,185 rows) is
  removed from training, so the test measures families never seen in
  training.
- **Loss:** symmetric InfoNCE, τ = 0.05, batch 4,096 pairs. Negatives are the
  in-batch pairs plus 4,096 random Swiss-Prot rows. Other pairs of the same
  query are masked out of the negatives.
- **Optimiser:** AdamW (lr 1e-3, weight decay 1e-4, one-cycle schedule),
  10 epochs. The best recall comes after 2–6 epochs; after that it
  overfits.
- **Result**, exact ranks over all of Swiss-Prot:

  | | @100 | @500 | @1000 | @5000 | within 10 × true count |
  |---|---|---|---|---|---|
  | raw ESM-2 35M | 0.454 | 0.742 | 0.827 | 0.908 | 0.843 |
  | + head (2,000 training queries) | 0.505 | 0.843 | 0.937 | 0.976 | 0.940 |
  | + head (10,000 training queries) | 0.503 | 0.841 | 0.942 | 0.977 | 0.933 |

  Five times more training pairs (1.95 M) change nothing. Recall is limited
  by the one mean-pooled vector per sequence, not by training data. The
  2,000-query head is the one in use (`head_35m.pt`).

### 4.3 Index

- **Product quantisation** (Jégou et al., IEEE TPAMI 33:117, 2011):
  - the 256-d unit vectors are cut into 32 subspaces of 8 dimensions;
  - each subspace has 256 centroids from k-means (200k-vector sample,
    15 iterations), so a vector is 32 bytes;
  - UniRef50 comes to about 1.2 GB (`create_embedding_index`,
    `EmbeddingIndex`).
- **Query:**
  - inner products of the query with every centroid (32 × 256 values);
  - each row's score is the sum of 32 table lookups;
  - codes are read in place from the memory-mapped `.pto`, in segments of
    512 Ki vectors on all cores, with a bounded heap per thread.
- An optional `rows` column maps several vectors (windows) to one row, and a
  row is reported once.
- **Loss from quantisation** on Swiss-Prot, same test: recall @100 0.473 vs
  0.501 exact, @1000 0.929 vs 0.936, @5000 0.974 vs 0.976.

### 4.4 Query path, all native

- `ProteinLanguageModel` reads the model from one GGUF file (Hugging Face
  tensor names, F16 matrices, head tensors `head.*`) and runs the ESM-2
  forward pass in C++:
  - pre-LayerNorm blocks;
  - the query scaled by `head_dim^-1/2` before rotary position embedding
    (θ = 10,000, rotate-half);
  - exact erf GELU;
  - embeddings scaled by 1 − 0.15·0.8 for the training's token dropout;
  - a final LayerNorm.
- Matrix products go through Accelerate's BLAS on macOS, Eigen elsewhere.
- **Validation:** matches Hugging Face transformers to 1e-4 on a random tiny
  model (the test fixture), and to F16 rounding (cosine > 0.999 after the
  head) on the real 35M model.
- **Speed:** 0.16 s for the 274-residue query, 0.63 s for 822 residues
  (M-series, both measured under load from other jobs). Plain Eigen took
  1.2 s under the same conditions.
- **Then:** the `embedding_candidates` (default 10,000) nearest
  representatives are aligned with `search_sequence_database_rows`. E-values
  stay against the whole representative database.
- **Interface:** arguments only (`SequenceClusterSearchOptions.embedding_model`,
  `embedding_index`, `embedding_candidates`; CLI `--embedding-model`,
  `--embedding-index`, `--embedding-candidates`). The model and the index are
  data; they will be distributed like the rotamer libraries (registry +
  download on first use), not with the core software and not through the
  settings file.

## 5. Homologue selection (ConSurf's rules)

`select_sequence_homologs`, with ConSurf's defaults:
- E ≤ 1e-4;
- aligned length at least 60 % of the query;
- identity to the query between 35 % and 95 % (identical columns over the
  alignment, gaps included);
- greedy redundancy removal, longest first (CD-HIT order), at 95 %;
- at most 150 homologues, sampled evenly over the E-value-ordered list;
- at least 5, else no result.

A target is aligned once here, so ConSurf's fragment rule (overlap ≤ 10 %)
does not arise.

## 6. Conservation (Rate4Site's method, implemented independently)

- **Model:** JTT with its own equilibrium frequencies. A discrete gamma with
  equally probable categories at their means.
- **Tree:** neighbour joining on pairwise ML distances.
- **Fit:** the gamma shape by maximum likelihood. ConSurf's setting does not
  refit branch lengths.
- **Rates:** per-site posterior mean rate (empirical Bayes), normalised to
  mean 0 and SD 1. ConSurf's nine grades follow from them.
- **Validation:** the defaults and conventions of Rate4Site 3.0 are
  reproduced, checked A/B against the program. The likelihood is scaled
  against underflow.
- The query-anchored alignment from the search is used directly; the server
  realigns with MAFFT.

## Measurements (1lk2 chain A, UniRef, Apple M-series, data on an SD card)

| configuration | time | notes |
|---|---|---|
| one full search of UniRef90 (FASTA-era store) | 1154 s | |
| two-stage, separate UniRef50/UniRef90 stores | 302 s → 267 s | |
| two-stage, one clustered `uniref.pto` | 239–241 s | 150 homologues, 321 MB |
| of which stage 1 (k-mer scan of 38.8 M representatives) | 228 s | 4,741 hits, 3,642 at E ≤ 1e-4 |
| of which stage 2 | ~11 s | |
| of which conservation | 1.4 s | |

**Retracted (2026-10-01):** an earlier version of this table listed early stop
at 150 / 300 / 600 homologues as 233 / 239 / 238 s with grades identical to
the full run. Those runs never stopped early. In Python,
`options.clusters.min_homologues = n` sets the field on a copy that SWIG
returns, and `options` is unchanged. The numbers are repeats of the full run,
and their spread (233–241 s) is run-to-run noise. Nested options must be
copied out, changed and assigned back
(`c = o.clusters; c.min_homologues = n; o.clusters = c`).

Stage 1 through candidate rows, simulated with the k-mer hits padded with
random representatives:

| candidates | cold (SD card) | warm | E ≤ 1e-4 hits kept |
|---|---|---|---|
| 5,000 | 1.7 s | 0.7 s | 3642/3642 |
| 10,000 | 4.9 s | 1.3 s | 3642/3642 |
| 20,000 | 9.2 s | 0.9 s | 3642/3642 |

"Cold" is bounded by the SD card's random reads; data on a fast SSD is
assumed for the target.

**Projected with the embedding prefilter:** 0.16 s embedding + ~0.3 s index
scan + ~1.3 s alignment + stage 2 (~11 s full; early stop unmeasured) + 1.4 s conservation
≈ **6 s**, provided the recall at UniRef50 scale holds.

## First reading at scale: the multi-domain limit (2026-09-30)

An index over the first 17 M representatives (44 %; built in 288 s, one
query scanned in 0.57 s) was tested against 1lk2's 2,451 strong stage-1 hits
among those rows. K is the full-database equivalent (the subset's top
K · 17/38.84):

| K | strong hits in the top K |
|---|---|
| 1,000 | 409 / 2451 (17 %) |
| 5,000 | 1350 / 2451 (55 %) |
| 10,000 | 1531 / 2451 (62 %) |
| 20,000 | 1592 / 2451 (65 %) |

This is well below the Swiss-Prot numbers. The missed hits are as similar to
the query as the found ones (median identity 0.35 vs 0.34), but the matching
region is a smaller part of them: median target coverage 0.46 vs 0.72, median
length 412 vs 347. Truncation at 1,022 residues explains only 2 %. The
missed hits are multi-domain proteins, where one mean-pooled vector dilutes
the shared domain. The first rows of UniRef50 hold the longest proteins, and
nearly all of their strong hits are missed (36 of 38 in the first million
rows). The Swiss-Prot test could not show this: its queries and matches are
mostly whole proteins.

**Window vectors** score a row by its best 256-residue window (stride 128).
They were tested fairly on cordeshub: 200k random background rows of the
same subset set the cut-off at each K, because windows lift unrelated rows
too. Scoring is exact cosine through the same model and head:

| K | whole sequence: cut-off, recall | windows: cut-off, recall |
|---|---|---|
| 1,000 | 0.633, 0.508 | 0.695, 0.570 (5 background rows above the cut-off) |
| 5,000 | 0.544, 0.635 | 0.627, 0.733 |
| 10,000 | 0.502, 0.685 | 0.581, 0.801 |
| 20,000 | 0.475, 0.713 | 0.538, 0.838 |

The exact whole-sequence recall (0.685) agrees with the PQ index's 0.625.
Windows add about 12 points and cost 3.8 windows per row in this long-protein
subset (2.7 per strong hit). This subset is the hardest part of UniRef50, so
full-database recall should be higher. For ConSurf, recall is not the
criterion: at most 150 homologues are kept, sampled over thousands of
candidates. Grade agreement with the full run decides, and it is measured
next.

## Full scale: 1lk2 end to end (2026-10-01)

Index over all 38,840,027 representatives: built in 505 s, 1.24 GB
(`/Volumes/SD1TB/sequences/uniref50_esm.pto`). Query embedding 0.11 s. Scan
20.5 s cold (1.24 GB at the SD card's ~60 MB/s) and **0.13 s warm** (page
cache).

Recall of the k-mer scan's stage-1 hits (3,642 strong, E ≤ 1e-4; 4,741 in
all):

| K | strong | all |
|---|---|---|
| 1,000 | 825 (23 %) | 865 |
| 5,000 | 1,724 (47 %) | 2,084 |
| 10,000 | 1,862 (51 %) | 2,359 |
| 20,000 | 1,969 (54 %) | 2,555 |

ConSurf end to end, against the k-mer two-stage run (284 s in this session;
the reference for scores and grades):

| configuration | time | score ρ | grades exact | within 1 |
|---|---|---|---|---|
| k-mer two-stage (reference) | 284 s | 1 | 1 | 1 |
| embedding, 10k candidates | 46.7 s | 0.951 | 0.642 | 0.942 |
| embedding, 20k candidates | 28.0 s | 0.951 | 0.642 | 0.942 |
| **embedding, 10k candidates, early stop at 600** | **6.4 s** | 0.952 | 0.650 | 0.927 |

The 46.7 s → 28.0 s difference between the first two rows is caching: the
second run found the member rows already read. With 20k candidates the
selected homologues and grades are identical to 10k, so the added candidates
are too distant to be chosen. **The time target is met with early stop**
(6.4 s, index warm). The grades move: 64 % agree exactly with the reference
and 93 % within one grade. Is that more than the sensitivity of grades to which 150
homologues are sampled? Baselines keep the k-mer search fixed and change only
the choice of homologues:

| same k-mer search, homologue choice varied | time | score ρ | grades exact | within 1 |
|---|---|---|---|---|
| the 150 closest instead of 150 sampled (`sampling="best"`) | 328 s | 0.886 | 0.515 | 0.792 |
| early stop at 600 homologues | 351 s | 0.942 | 0.591 | 0.916 |
| *for comparison: embedding, 10k, early stop at 600* | *6.4 s* | *0.952* | *0.650* | *0.927* |

Changing only which homologues are kept moves 40–50 % of the grades. The fast
path agrees with the reference better than either variant of the exact
search. **The grade differences of the embedding path are inside the spread
that ConSurf's own homologue sampling produces.** They are not a loss
specific to the prefilter. For a paper, the honest statement is: same
scores to ρ ≈ 0.95, grades within one class at 93 %, 44× faster (284 → 6.4 s),
on one chain. More chains are needed before this generalises.

The k-mer runs with early stop were slower than without (351 vs 284 s): the
early stop saves only stage-2 work, and run-to-run time varies by tens of
seconds on the SD card. Timings in this table are single runs.

## Benchmark: 20 Swiss-Prot chains (2026-10-01)

20 random Swiss-Prot sequences (seed 2026, 80–800 residues;
`prototypes/esm_retrieval/consurf_e2e/bench.py`, results in `bench.tsv`).
One of them (CT62_HUMAN, 136 residues) has 3 homologues in UniRef and gets
no result in either method (5 are needed), leaving 19:

| configuration | score ρ, median (min) | grades exact, median | within 1, median (min) | time |
|---|---|---|---|---|
| k-mer two-stage (reference), all 20 queries in one pass | 1 | 1 | 1 | 551 s, 27.6 s per query |
| same search, the 150 closest homologues (sampling spread) | 0.899 (0.795) | 0.517 | 0.771 (0.616) | 492 s, 24.6 s per query |
| **embedding, 10k candidates, early stop at 600**, one query at a time | **0.982 (0.902)** | 0.684 | **0.960 (0.771)** | median 22.1 s, max 40.3 s (SD card) |

- **Quality:** on every chain the fast path is closer to the reference than
  the sampling spread of the exact search. Four chains match exactly
  (ρ = 1.000). The weakest is ZN350_HUMAN (ρ 0.902), a zinc-finger protein:
  repetitive multi-domain, the case the whole-sequence embedding handles
  worst.
- **Time:** 22 s on the SD card, not the 6.4 s of the warm 1lk2 run. Cold
  runs read thousands of scattered candidate and member rows, and the SD
  card's random reads dominate. The SSD measurement below separates storage
  from computation.
- **Batching:** the k-mer search shares one pass of the database among all
  queries of a call, so a batch of 20 costs 27.6 s per query. The fast path
  is per query. The honest comparison depends on the use: one chain at a
  time (the Labelizer, an interactive run) 284 s → seconds; a batch of many
  chains, about 28 s per chain against the fast path's per-chain cost.

**On a fast SSD** (the Mac's internal SSD, database and index copied there;
`bench_ssd.py`), the same 20 queries one at a time, first pass after the
copy and second pass:

| storage | median | max | 20 queries |
|---|---|---|---|
| SD card | 22.1 s | 40.3 s | ~460 s |
| internal SSD, first pass | **5.2 s** | **9.2 s** | 105 s |
| internal SSD, second pass | 5.1 s | 9.9 s | 104 s |

Every query finishes under 10 s. The homologues are the same as from the SD
card on all 20. First and second pass agree, so the run is bound by
computation, not storage. (The first pass is only partly cold: the copy had
just been written and part of it was in the page cache. The 23 GB database
does not fit in the 16 GB of RAM, so most member rows came from the SSD.)
Time grows with the query: 1.4 s at 136 residues, 9.2 s at 752.

**The < 10 s target is met** on 20 random chains with data on an SSD:
median 5.2 s, at quality closer to the reference than ConSurf's own homologue
sampling spread.

## One file for consurfer (2026-10-01)

`/Volumes/SD1TB/sequences/uniref_consurf.pto` (24.7 GB) holds everything a
ConSurf run needs:
- the clustered UniRef store (a copy of `uniref.pto`);
- the embedding index (`embedding_index`);
- the ESM-2 35M + head weights (`protein_language_model`, the GGUF as an
  attachment).

It was made with `add_embedding_prefilter(database, model, index)` (CLI
`imp_bff sequence-db add-prefilter`), which took 60 s. When no model or index
is given, the clustered search uses the database's own prefilter
(`SequenceClusterSearchOptions.use_database_prefilter`, default on). So a run
needs only:

```python
o = IMP.bff.ConsurfOptions()
o.database = "/Volumes/SD1TB/sequences/uniref_consurf.pto"
o.clusters.min_homologues = 600        # optional early stop
r = IMP.bff.compute_consurf([sequence], o)[0]
```

On 1lk2 this gives the same result as the separate files (ρ 0.952, grades
exact 0.650, within one 0.927). Loading the model from the container takes
0.9 s, and the scan is the same as from the separate index (21 s cold,
0.13 s warm). Repeated runs on the SD card took 29.3 → 11.4 → **3.5 s** as
the page cache filled. Copying 47 GB just before had emptied it; that is why
the first runs were slow, not the layout.

`uniref.pto` and `uniref50_esm.pto` remain next to it for now (a duplicate
of 23.4 + 1.2 GB on the card).

## Benchmark: 100 Swiss-Prot chains (2026-10-03)

100 further random Swiss-Prot chains (seed 7, 60–1000 residues, disjoint from
the 20; `consurf_e2e/bench2.py`, `bench2.tsv`). Two have too few homologues
in either method, leaving 98:

| configuration | score ρ, median (10th pct, min) | grades exact, median | within 1, median (10th pct) |
|---|---|---|---|
| same k-mer search, 150 closest (sampling spread) | 0.920 (0.837, 0.697) | 0.530 | 0.782 (0.677) |
| **embedding, 10k candidates, early stop 600** | **0.977 (0.944, 0.891)** | 0.657 | **0.944 (0.856)** |

The fast path is closer to the reference than the sampling spread on 82 of 98
chains (on the 20-chain set: all of them). The k-mer reference took 4,080 s
for the batch of 100 (41 s per chain). The per-chain times of this run
(median 46 s, SD card) are not usable: a Parallels VM (6.3 GB) on the 16 GB
Mac drove the machine into swap during the run, and single chains took up to
4,935 s. The SSD timing of the 20 chains (median 5.2 s) stands.

## Pending

- **UniRef50 embedding** of all 38.84 M representatives on cordeshub
  (RTX 4000 Ada; streamed in 1 M-row chunks, FASTA order = `uniref.pto` root
  order, verified on sampled rows). The file starts with the longest
  proteins: 5,028 s for the first chunk, falling to 2,694 s by chunk 6.
  11.32 G residues in all.
- Then: project through the head, build the index, and measure:
  - recall of the 3,642 strong stage-1 hits of 1lk2 in the top
    1k/5k/10k/20k;
  - end-to-end time;
  - grade agreement with the 240 s run (`scratchpad/ab/e2e.py`).
- Window vectors for long representatives (+12 recall points measured on the
  first 17 M rows) if a benchmark shows grade drift beyond the sampling
  spread.

## Reproduction

- **Code:** imp.bff `744201978` (ProteinLanguageModel, EmbeddingIndex,
  prefilter), `78b4292a9` (row search from Python), `9c92a462a` (arguments
  only). Earlier: the search, ConSurf and the clustered store,
  `128fa147a..7168b73a0`. ptolib format 6 is upstream in `../ptolib`.
- **Tests:** `test/sequence/test_protein_language_model.py`,
  `test_embedding_index.py`, `test_sequence_clusters.py` (the prefilter with
  every representative as a candidate returns exactly the k-mer search's
  hits and E-values).
- **Training and evaluation scripts** (prototypes, not shipped) in
  `prototypes/esm_retrieval/`:
  - `ground_truth.py`, `train_truth.py`, `train_truth2.py`: test and training
    pairs;
  - `embed.py`, `embed_stream.py`: GPU embedding;
  - `train_head.py`, `project.py`;
  - `to_gguf.py`, `make_tiny.py`: GGUF and the test fixture;
  - `pq_recall.py`, `evaluate3.py`.
- **Data:**
  - `/Volumes/SD1TB/sequences/uniref.pto`;
  - `uniref50_35m_proj/` (projected chunks + `.ids`);
  - on cordeshub `/mnt/data/tpeulen/esm/`: raw embeddings, `head_35m.pt`,
    `esm2_35m_head.gguf`.

## References

- Ashkenazy H. et al. ConSurf 2016. NAR 44:W344.
- Steinegger M., Söding J. MMseqs2. Nat Biotechnol 35:1026 (2017).
- Altschul S.F. et al. Gapped BLAST and PSI-BLAST. NAR 25:3389 (1997).
- Pupko T. et al. Rate4Site. Bioinformatics 18:S71 (2002); Mayrose I. et al.
  MBE 21:1781 (2004).
- Jones D.T., Taylor W.R., Thornton J.M. JTT. CABIOS 8:275 (1992).
- Yang Z. Discrete gamma. J Mol Evol 39:306 (1994).
- Saitou N., Nei M. Neighbour joining. MBE 4:406 (1987).
- Suzek B.E. et al. UniRef clusters. Bioinformatics 31:926 (2015).
- Lin Z. et al. ESM-2. Science 379:1123 (2023).
- van den Oord A. et al. InfoNCE / CPC. arXiv:1807.03748 (2018).
- Jégou H., Douze M., Schmid C. Product quantization. IEEE TPAMI 33:117 (2011).
