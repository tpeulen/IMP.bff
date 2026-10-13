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

## Where to pick this up

1. **Standalone wheel build not compiled after the header split.** Measured: `cmake -S standalone` configures and stages `IMP/bff/<theme>/` (165 headers in the wrapper aggregate = 186 public − 20 `imp/` − `IMPCompatibility.h`), and `test/expensive_test_standalone_core_compiles.py` syntax-checks the whole core against the themed tree; a full `pip wheel .` was not run. It is what chisurf installs (`imp-bff` from git, branch `dev`), so the split reaches chisurf only after a push and a pin bump -- run the wheel build before pushing.
2. **Existing build trees on case-insensitive disks.** The first configure after the split lost `fret.h`/`smlm.h` to IMP's stale-link cleanup (old `FRET.h`/`SMLM.h`); `cmake/ThemeHeaders.cmake` re-links missing umbrellas. Also: `git add` of the removed `include/FRET.h` on macOS staged the new `fret.h` content under the old name -- check `git ls-tree HEAD include/` after any case-only rename.
3. **Out-of-repo C++ that includes flat paths** still compiles only against an old install: `ucfret/investigation/pinn_pR_anisotropy/s89_cpp/*` names `<IMP/bff/X.h>`. Rewrite with the theme of `include/<theme>/X.h` when that investigation is next built.
