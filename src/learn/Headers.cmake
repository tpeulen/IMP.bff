# Public headers of the `learn` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_learn_headers "EmbeddingIndex.h;HMMSurrogate.h;NeuralNet.h;NeuralNetTraining.h;ProteinLanguageModel.h")
# Sources without a public header of their own.
set(imp_bff_learn_private_sources "")
