.. _overview:

IMP.bff
=======

Bayesian Fluorescence Framework for Integrative Structural Modeling
-------------------------------------------------------------------

**IMP.bff** is an open-source framework and `IMP <https://integrativemodeling.org>`_ module
for fluorescence-guided integrative structural biology. It provides coordinate-level modeling
of fluorescent probes, accessible volume simulations, rotamer libraries, fluorophore path maps,
optimal label site selection, and experiment-neutral fluorescence observables.

Core algorithms are implemented in high-performance C++ and exposed seamlessly to Python and
NumPy.

Quick links
-----------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - Getting Started
     - :doc:`getting_started`
   * - Interactive Workshop
     - :doc:`workshop/index`
   * - User Manual
     - :doc:`manual/index`
   * - Example Gallery
     - :doc:`auto_examples/index`
   * - Labelizer Guide
     - :doc:`labelizer`
   * - C++ API Reference
     - `Doxygen C++ API <api/index.html>`__
   * - Source Code
     - `GitHub Repository <https://github.com/fluorescence-tools/imp.bff>`__

Key Features
------------

* **Accessible Volumes & Probe Sampling**:
  Fast grid-based accessible volume (AV) computation, contact volumes, and sterically allowed
  fluorophore conformations around attachment residues.

* **Rotamer Libraries & Dunbrack Tables**:
  Off-lattice rotamer sampling using backbone-dependent Dunbrack libraries for realistic dye
  linker dynamics.

* **Fluorophore Path Maps**:
  Dynamic path exploration of fluorophore mobility across macromolecular surfaces.

* **Site & Pair Selection (Labelizer)**:
  Automated multi-objective screening to select optimal labeling sites and FRET pairs for
  resolving conformational changes.

* **Distance Networks & Ensemble Modeling**:
  FRET distance restraints, maximum-entropy ensemble reweighting, and integrative docking.

* **Fast C++ Execution**:
  Zero-copy NumPy array interop, parallelized evaluation, and clean boundaries separating
  photon data (`tttrlib`) from coordinate-level structural models (`imp.bff`).

Documentation
-------------

.. toctree::
   :maxdepth: 2

   getting_started
   install
   workshop/index
   manual/index
   auto_examples/index
   labelizer
   about
   faq
   whats_new
   roadmap
   related_projects

.. toctree::
   :hidden:

   preface
   contents
   zreferences

External resources
------------------

* `IMP.bff on GitHub <https://github.com/tpeulen/IMP.bff>`__
* `tttrlib Documentation <https://fluorescence-tools.github.io/tttrlib>`__
* `Integrative Modeling Platform (IMP) <https://integrativemodeling.org>`__
