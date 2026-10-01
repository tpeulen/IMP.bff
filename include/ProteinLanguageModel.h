/**
 *  \file IMP/bff/ProteinLanguageModel.h
 *  \brief A protein language model (ESM-2) read from a GGUF file and run
 *         natively on the CPU: per-residue and per-sequence embeddings.
 *
 * ESM-2 (Lin et al., Science 379:1123, 2023) is a BERT-style transformer
 * trained on UniRef50. Its mean-pooled last layer places homologous proteins
 * near each other, so a search can find a query's cluster representatives
 * by nearest neighbours instead of a k-mer scan of the whole database.
 *
 * The weights are data: one GGUF file (the `ggml` container, version 3) with
 * the Hugging Face tensor names (`encoder.layer.0.attention.self.query.weight`
 * ...), matrices in F16 or F32, and the hyper-parameters under `esm2.*`
 * (`block_count`, `embedding_length`, `head_count`, `feed_forward_length`,
 * `layer_norm_epsilon`, `rope_theta`, `token_dropout`, `max_length`) and the
 * vocabulary under `tokenizer.ggml.tokens`. The forward pass is the published
 * one: pre-LayerNorm blocks, rotary positions on the query (scaled by
 * `head_dim^-1/2` first) and the key, exact (erf) GELU, embeddings scaled
 * by 0.88 for the training's token dropout, and a final LayerNorm.
 *
 * A file may also carry a projection head (`head.*`), trained so that cosine
 * similarity ranks homologues first: standardisation (`head.mean`,
 * `head.std`), an MLP (`head.mlp.{0,2,4}`, GELU between) plus a linear skip
 * (`head.skip`), normalised to unit length (#get_projected_embedding).
 *
 * Sequences longer than #get_max_length are embedded from their first
 * #get_max_length residues, as the model was trained.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_PROTEINLANGUAGEMODEL_H
#define IMPBFF_PROTEINLANGUAGEMODEL_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <memory>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! An ESM-2 protein language model, loaded from GGUF; cheap to copy, safe
//! from many threads.
class IMPBFFEXPORT ProteinLanguageModel {
public:
    ProteinLanguageModel() {}
    //! Load \p path: a GGUF file, or a `.pto` container holding one as its
    //! #get_object_name object (#IMP::bff::add_embedding_prefilter).
    /*! \throw IOException when the file cannot be read or is not an ESM-2 GGUF */
    explicit ProteinLanguageModel(const std::string& path);

    //! The name of the model's object in a `.pto` container.
    static const char* get_object_name() { return "protein_language_model"; }

    std::string get_path() const { return path_; }
    //! The model's name (`general.name`).
    std::string get_name() const;
    //! Width of the per-residue and pooled embeddings.
    int get_embedding_length() const;
    int get_number_of_layers() const;
    //! Width of #get_projected_embedding; 0 when the file has no head.
    int get_projection_length() const;
    //! Residues embedded at most; longer sequences are truncated.
    int get_max_length() const;

    //! The last layer for each residue: `n x` #get_embedding_length, row by row.
    Floats get_residue_embeddings(const std::string& sequence) const;
    //! The last layer averaged over the residues (the tokens `<cls>` and `<eos>`
    //! excluded).
    Floats get_embedding(const std::string& sequence) const;
    //! Whether the file carries ESM's contact head (#get_contacts).
    bool get_has_contact_head() const;
    //! Residue contact probabilities, `n x n` (`n` residues, at most
    //! #get_max_length): ESM's contact head, a logistic regression over the
    //! symmetrised, average-product-corrected attention maps of every layer
    //! and head (Rao et al., ICLR 2021). From the single sequence, no MSA.
    /*! \throw ValueException when the file has no contact head */
    void get_contacts(const std::string& sequence, double** out_matrix, int* n_out_rows,
                      int* n_out_cols) const;

    //! #get_embedding through the projection head, unit length.
    /*! \throw ValueException when the file has no head */
    Floats get_projected_embedding(const std::string& sequence) const;

#ifndef SWIG
    //! #get_residue_embeddings into \p out (`n x d` floats); returns `n`.
    int compute(const std::string& sequence, std::vector<float>& out) const;
    //! #get_contacts row by row; \p n gets the number of residues.
    std::vector<double> contacts(const std::string& sequence, int* n) const;
    //! #get_projected_embedding (or the pooled embedding when \p project is
    //! false) of each of \p sequences, in parallel: `sequences.size() x width`.
    std::vector<float> embed(const std::vector<std::string>& sequences, bool project) const;
#endif

    IMP_SHOWABLE_INLINE(ProteinLanguageModel, out << "ProteinLanguageModel(" << path_ << ")");

private:
    struct Weights;
    std::string path_;
    std::shared_ptr<const Weights> w_;
};
IMP_VALUES(ProteinLanguageModel, ProteinLanguageModels);

//! Contact probabilities for pairs of structure residues: an alternative to
//! #IMP::bff::probe_pair_coevolution as the per-pair cost of a
//! #IMP::bff::ProbePairCostTerm, from the single sequence and already in
//! `[0, 1]`.
/*!
    Residue `r` of the structure is position `r - residue_offset` (1-based)
    of \p sequence; NaN when either residue is outside it (ineligible to the
    cost term).

    \param[in] model a protein language model with a contact head
    \param[in] sequence the structure's sequence
    \param[in] pair_residues,n_pair_rows,n_pair_cols two structure residue numbers per pair
    \param[in] residue_offset structure numbering minus sequence numbering
*/
IMPBFFEXPORT std::vector<double> probe_pair_contacts(
        const ProteinLanguageModel& model, const std::string& sequence, int* pair_residues,
        int n_pair_rows, int n_pair_cols, int residue_offset = 0);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_PROTEINLANGUAGEMODEL_H */
