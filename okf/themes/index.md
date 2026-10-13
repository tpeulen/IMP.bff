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

Nothing open from the header split (2026-10-13). What was checked, to re-run after a layout change:

1. **Standalone wheel**: `CMAKE_BUILD_PARALLEL_LEVEL=4 python -m pip wheel . --no-deps -w <dir>` (with build isolation: the arm64 env has no `setuptools_scm`). Installed into a scratch venv, `IMP.bff` imports as `core`, the pyproject `test-command` passes, and the module's top-level names equal those of the pre-split SWIG output (1373; the one difference, `_swig_add_metaclass`, is SWIG 4.5.1 vs 4.4.1, not the API). Baseline trick: a `git worktree` at the previous commit, `cmake -S standalone`, then only the ninja target `swig/IMP_bff_standalonePYTHON_wrap.cxx` -- SWIG, no compile. `test/test_public_api_names.py` fails 14 cases on the wheel, all IMP-layer or RMF names that the core build never had (same set before the split); that suite targets the IMP build.
2. **Case-insensitive disks**: an existing build tree loses `fret.h`/`smlm.h` to IMP's cleanup of the old `FRET.h`/`SMLM.h` links; `cmake/ThemeHeaders.cmake` re-links missing umbrellas. `git add` of a removed `FRET.h` on macOS records the new `fret.h` content under the old name -- check `git ls-tree HEAD include/` after any case-only rename.
3. **Out-of-repo C++**: ucfret's `investigation/pinn_pR_anisotropy/s89_cpp` drivers include the themed paths (ucfret d1e7ccd) and syntax-check with their build scripts' flags. Those scripts symlink `IMP/bff -> include`, which works unchanged.
