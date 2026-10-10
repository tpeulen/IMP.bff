# search: Model search: MCTS policy, self-play, search specs

Model search: the MCTS policy, self-play, search specification and the photon experiment it scores. Public API is stable; the MCTS work lands here.

- **Inputs:** a model space and an experiment.
- **Relations:** Uses `fit` and `bayesian`.
- **Layout:** sources in `src/search/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/search/Headers.cmake`.

Public headers: `ModelSearch.h`, `ModelSearchPolicy.h`, `ModelSearchSelfPlay.h`, `ModelSearchSpec.h`, `PhotonExperiment.h`.

## What a search state is (2026-10-10)

- **Modality token.** The policy state ends with a one-hot of the worst residual block's `ModelSearchBlockKind` (`generic`, `tcspc`, `polarized`, `fcs`, `burst_mfd`, `pie_alex`) and log of the number of distinct kinds; a description declares `"dataset_kinds"`. Width 78 + 7 action features; a policy trained at another width is refused.
- **`request_information`.** A move may `"requires": [<dataset>]`; unbound, it is withheld (`get_withheld_actions`), and a search whose best state is not acceptable and has a withheld move ends with `ModelSearchResult::get_outcome() == "request_information"` and `get_missing_observables()`.
- **`when_bound`.** A node, joint member or free entry written with `"when_bound": "<dataset>"` exists only when that dataset is bound, in every topology alike -- topologies are only comparable on the same data. An unbound optional dataset has `<slot>_size == 0` in rules.
- **Fit-settings action space.** `set_action_space(ACTION_SPACE_FIT_SETTINGS)` makes a state a structure fitted with one `ModelSearchFitSettings` candidate (algorithm, tolerances, budget, step factor, start `declared|primary|parent`, bounds). Structural moves keep the candidate, `fit-settings:<key>` moves refit the structure with another; the tree keys states by `get_search_key()` = `<structure>|<settings>`. Default candidates never turn bounds off. Proof: `test/mcts/test_model_search_fit_settings.py` (a patient budget rescues the two-lifetime fit a short budget loses) and `golden/tcspc_lifetime_fit_settings.json`.
- **Games.** `smfret_mfd` and `pie_alex` (`_fixtures.py`, `_games.py`) are self-play games whose channels are recorded as photons through TTTRLib; near their declared parameters every topology is recovered (`test_self_play.py`).
