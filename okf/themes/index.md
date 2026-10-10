# Source themes

src/ is laid out by theme; each theme has a `README.md` beside its sources and a concept here. Headers stay flat in `include/`; `src/<theme>/Headers.cmake` assigns them. `src/imp/` is the connection layer to IMP and `src/internal/` holds private helpers.

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
