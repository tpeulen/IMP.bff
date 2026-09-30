/*
 * Nearest neighbours of protein embeddings over a whole sequence database
 * (product-quantised codes): the embedding prefilter of the search.
 */

IMP_SWIG_VALUE(IMP::bff, EmbeddingIndex, EmbeddingIndexes);

%feature("compactdefaultargs") IMP::bff::create_embedding_index;

%include "IMP/bff/EmbeddingIndex.h"
