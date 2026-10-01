/**
 *  \file IMP/bff/EmbeddingIndex.h
 *  \brief Nearest neighbours of protein embeddings over a whole sequence
 *         database: product-quantised codes in a `.pto` container.
 *
 * A sequence database's rows (e.g. UniRef50's 39 M cluster representatives)
 * embedded by a #IMP::bff::ProteinLanguageModel's projection (unit vectors,
 * homologues near each other) are stored as product-quantisation codes
 * (Jégou et al., IEEE TPAMI 33:117, 2011): the vector is split into
 * `subspaces` pieces, each replaced by the nearest of 256 centroids learnt by
 * k-means on a sample, one byte a piece. 256 dimensions in 32 subspaces are
 * 32 bytes a row, 1.2 GB for UniRef50.
 *
 * A query is scored against every row by table lookup: its inner product with
 * each subspace's centroids is computed once (`subspaces x 256` values), and a
 * row's score is the sum of its pieces' entries. The codes are read in place
 * from the memory-mapped container, segment by segment on all cores.
 *
 * Several vectors may stand for one database row (windows of a long
 * sequence): the optional `rows` column maps vector to row, and a row is
 * reported once, by its best vector.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_EMBEDDINGINDEX_H
#define IMPBFF_EMBEDDINGINDEX_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pto { class StoreReader; }

IMPBFF_BEGIN_NAMESPACE

//! Build an embedding index from vectors in NumPy `.npy` files.
/*!
    \param[in] vectors `.npy` files of `n x d` float16 or float32 rows (unit
               length), concatenated in this order
    \param[in] out the `.pto` container to create (replaced if present)
    \param[in] model the name of the model that made the vectors, recorded
    \param[in] subspaces pieces a vector is cut into; must divide `d`
    \param[in] sample vectors k-means learns the centroids from, spread
               evenly over all
    \param[in] iterations k-means iterations
    \param[in] rows for each vector its database row, or empty for vector i =
               row i (a file of one row number per line)
    \return the number of vectors
    \throw IOException when a file cannot be read or written
*/
IMPBFFEXPORT std::size_t create_embedding_index(const Strings& vectors, const std::string& out,
                                                const std::string& model = "",
                                                int subspaces = 32, int sample = 200000,
                                                int iterations = 15, const std::string& rows = "");

//! An embedding index, mapped read-only; cheap to copy, safe from many threads.
class IMPBFFEXPORT EmbeddingIndex {
public:
    EmbeddingIndex() {}
    //! Open the index in \p path: the file #IMP::bff::create_embedding_index
    //! made, or any `.pto` container it was copied into
    //! (#IMP::bff::add_embedding_prefilter).
    explicit EmbeddingIndex(const std::string& path);

    //! The name of the index's object in a `.pto` container.
    static const char* get_object_name() { return "embedding_index"; }

    std::string get_path() const { return path_; }
    std::size_t get_number_of_vectors() const { return n_; }
    int get_dimension() const { return d_; }
    int get_number_of_subspaces() const { return m_; }
    //! The model that made the vectors.
    std::string get_model() const { return model_; }

    //! The database rows of the \p k vectors nearest \p query (largest inner
    //! product), best first, each row once.
    Ints get_nearest(const Floats& query, int k) const;
    //! Their approximate inner products with \p query, in the same order.
    Floats get_nearest_scores(const Floats& query, int k) const;

#ifndef SWIG
    //! (score, row) of the \p k nearest rows to each of \p queries (`q x d`
    //! floats, unit length), best first; one pass over the codes for all.
    std::vector<std::vector<std::pair<float, std::size_t>>> nearest(
            const std::vector<float>& queries, std::size_t k) const;
#endif

    IMP_SHOWABLE_INLINE(EmbeddingIndex, out << "EmbeddingIndex(" << path_ << ", " << n_
                                            << " vectors, " << d_ << " dimensions)");

private:
    std::string path_, model_;
    std::size_t n_ = 0;
    int d_ = 0, m_ = 0;
    std::vector<float> centroids_;          // m x 256 x (d / m)
    std::vector<std::uint32_t> rows_;       // empty: vector i is row i
    std::shared_ptr<pto::StoreReader> reader_;
};
IMP_VALUES(EmbeddingIndex, EmbeddingIndexes);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_EMBEDDINGINDEX_H */
