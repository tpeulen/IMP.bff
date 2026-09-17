.. _workshop:

=====================================================
Workshop: from a structure to a measured ensemble
=====================================================

Seven steps on one molecule. **BmrA** is a homodimeric ABC transporter with two
experimentally determined arrangements of its nucleotide-binding domains,
`6R72 <https://www.rcsb.org/structure/6R72>`_ (closed, outward-facing, X-ray)
and `8QOE <https://www.rcsb.org/structure/8QOE>`_ (open, inward-facing,
cryo-EM). The workshop asks what a FRET experiment on BmrA has to answer --
which residues to mutate, what they will report, and what the measured
distances say about the structure -- and answers it with the library.

Every notebook here is executed against the build before it is published, and
the numbers in them come from those runs. What the measurements are, and are
not, is stated in :doc:`README <README>`: the structures, volumes, scores,
docking and inversion are real computations; the *measured* distances are
simulated, because no experimental BmrA network ships with this repository.

.. toctree::
   :maxdepth: 1

   README
   01_label_sites
   02_which_pair
   03_volumes_and_rotamers
   04_greedy_olga
   05_docking
   06_network_circle_plot
   07_maxent_ensemble

Two further steps are older notebooks that were kept because they are good:
choosing *sites* for a homodimer, where one cysteine mutation labels both
protomers (``ipynb/example/labelizer_greedy_homodimer.ipynb``), and the
monomer's question of which *pairs* to measure
(``ipynb/example/labelizer_greedy_pipeline.ipynb``).
