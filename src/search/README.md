# search: Model search: MCTS policy, self-play, search specs

Model search: the MCTS policy, self-play, search specification and the photon experiment it scores. Public API is stable; the MCTS work lands here.

- **Inputs:** a model space and an experiment.
- **Relations:** Uses `fit` and `bayesian`.
- **Layout:** sources in `src/search/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/search/Headers.cmake`.

Public headers: `ModelSearch.h`, `ModelSearchPolicy.h`, `ModelSearchSelfPlay.h`, `ModelSearchSpec.h`, `PhotonExperiment.h`.
