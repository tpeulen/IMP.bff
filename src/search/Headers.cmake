# Public headers of the `search` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_search_headers "ModelSearch.h;ModelSearchPolicy.h;ModelSearchSelfPlay.h;ModelSearchSpec.h;PhotonExperiment.h")
# Sources without a public header of their own.
set(imp_bff_search_private_sources "")
