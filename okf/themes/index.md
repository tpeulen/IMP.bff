# Source themes

src/ and include/ are laid out by the same themes: sources in `src/<theme>/`, public headers in `include/<theme>/` (`<IMP/bff/<theme>/X.h>`), one umbrella `include/<theme>.h` per theme. Each theme has a `README.md` beside its sources and a concept here. `src/imp/` and `include/imp/` are the connection layer to IMP; `include/internal/` and `src/internal/` hold private helpers. IMP's module tooling links only `include/*.h` and `include/internal/*.h` into its build tree, so `cmake/ThemeHeaders.cmake` links the themed directories (both the IMP-module and the in-tree build call it; the standalone build copies the tree as is), and `IMP/bff.h` -- what the SWIG wrapper includes -- reaches every header through the umbrellas. `tools/dev/theme_layout.py` regenerates `src/ThemeSources.cpp` and the umbrellas; `test/test_theme_headers.py` checks the layout, that every include names the themed path, and that lower themes do not reach upward.

* [util](util.md) - Utilities: settings, command line, numerics, registry
* [graph](graph.md) - Expression graph: nodes, ports, sessions, evaluation
* [fit](fit.md) - Curve fitting: objectives, minimizers, factor-graph inference, MaxEnt
* [bayesian](bayesian.md) - Bayesian inference: decay posteriors, samplers, causal models
* [spectroscopy](spectroscopy.md) - Photophysics and decays: lifetimes, quenching, kinetics, FCS
* [fret](fret.md) - FRET: transfer, networks, landscapes, labelizer, polymer distances
* [search](search.md) - Model search: MCTS policy, self-play, search specs
* [learn](learn.md) - Learned models: neural nets, embeddings, protein language model
* [probe](probe.md) - Dye probes: accessible volumes, rotamers, diffusion, FPS
* [structure](structure.md) - Structure and trajectory I/O, selections, clustering, surfaces
* [sequence](sequence.md) - Sequences: alignment, MSA, conservation, coevolution
* [smlm](smlm.md) - Single-molecule localisation: particles, likelihoods, registration
