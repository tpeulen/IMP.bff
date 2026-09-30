/*
 * A protein language model (ESM-2 from GGUF), run natively: embeddings for
 * the embedding prefilter of the sequence search.
 */

IMP_SWIG_VALUE(IMP::bff, ProteinLanguageModel, ProteinLanguageModels);

%include "IMP/bff/ProteinLanguageModel.h"
