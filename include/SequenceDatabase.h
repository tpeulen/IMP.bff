/**
 *  \file IMP/bff/SequenceDatabase.h
 *  \brief A protein sequence database as binary FASTA in a `.pto` container,
 *         built and read in bounded memory.
 *
 * #create_sequence_database streams a FASTA file (gzip-compressed if imp.bff
 * was built with zlib) into one `dstore` object of a `.pto` container
 * (ptolib's streamed store, format 5):
 *
 * - `residues`: one byte per residue, variable-length rows, stored raw so a
 *   sequence is read in place from the memory map. The codes are those of
 *   #IMP::bff::SequenceMSA: 1..20 are `ACDEFGHIKLMNPQRSTVWY`, 0 is anything
 *   else (X, B, Z, J, U, O).
 * - `headers`: the FASTA header line without `>`, compressed (zstd when
 *   available): headers are read rarely and compress well.
 *
 * Neither conversion nor reading ever holds the database: the writer keeps one
 * segment per column, the reader maps the file. UniRef90 (about 45 G
 * residues) converts in a few hundred MB.
 *
 * A database is opened by path or by the name the settings give it (see
 * #IMP::bff::get_sequence_search_settings).
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_SEQUENCEDATABASE_H
#define IMPBFF_SEQUENCEDATABASE_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pto { class StoreReader; }

IMPBFF_BEGIN_NAMESPACE

//! Convert a FASTA file into a sequence database.
/*!
    \param[in] fasta a FASTA file; `.gz` needs zlib at build time
    \param[in] out the `.pto` container to create (replaced if present)
    \param[in] name the store object inside the container
    \param[in] segment_mb segment size of the streamed store, in MB
    \return the number of sequences written
    \throw IOException when a file cannot be read or written
*/
IMPBFFEXPORT std::size_t create_sequence_database(const std::string& fasta,
                                                  const std::string& out,
                                                  const std::string& name = "sequences",
                                                  int segment_mb = 16);

//! Whether this build reads gzip-compressed FASTA.
IMPBFFEXPORT bool get_sequence_database_reads_gzip();

//! A sequence database, mapped read-only; cheap to copy, safe from many threads.
class IMPBFFEXPORT SequenceDatabase {
public:
    SequenceDatabase() {}
    //! Open \p path_or_name: a `.pto` path, or a database name from the settings.
    /*! \param[in] name the store object inside the container */
    explicit SequenceDatabase(const std::string& path_or_name,
                              const std::string& name = "sequences");

    std::string get_path() const { return path_; }
    std::size_t get_number_of_sequences() const { return n_sequences_; }
    std::size_t get_number_of_residues() const { return static_cast<std::size_t>(n_residues_); }
    //! Residues of sequence \p i, as letters (`X` for code 0).
    std::string get_sequence(std::size_t i) const;
    int get_length(std::size_t i) const;
    //! The FASTA header of sequence \p i, without `>`.
    std::string get_header(std::size_t i) const;
    //! The first word of the header: the accession.
    std::string get_identifier(std::size_t i) const;

#ifndef SWIG
    //! Sequence \p i's codes in place (1..20 amino acids, 0 other); \p n its length.
    const unsigned char* get_codes(std::size_t i, std::uint64_t* n) const;
    //! Streaming: the database in segments of whole sequences.
    std::size_t get_number_of_segments() const;
    //! Segment \p k: its first sequence and how many it holds, and its codes
    //! (back to back, in place). Sequence boundaries come from #get_codes.
    const unsigned char* get_segment(std::size_t k, std::size_t* first,
                                     std::size_t* count, std::uint64_t* n_codes) const;
    //! Offset of sequence \p i's first code in the concatenation of all.
    std::uint64_t get_offset(std::size_t i) const;
    //! The offsets of sequences \p first .. \p first + \p count, inclusive:
    //! `count + 1` values into \p out.
    void get_offsets(std::size_t first, std::size_t count, std::uint64_t* out) const;
    //! Tell the system the next reads are one pass in order.
    void advise_sequential() const;
#endif

    IMP_SHOWABLE_INLINE(SequenceDatabase, out << "SequenceDatabase(" << path_ << ", "
                                              << n_sequences_ << " sequences)");

private:
    std::string path_;
    std::shared_ptr<pto::StoreReader> reader_;
    std::size_t n_sequences_ = 0;
    std::uint64_t n_residues_ = 0;
};
IMP_VALUES(SequenceDatabase, SequenceDatabases);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_SEQUENCEDATABASE_H */
