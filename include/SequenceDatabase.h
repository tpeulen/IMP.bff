/**
 *  \file IMP/bff/SequenceDatabase.h
 *  \brief A protein sequence database as binary FASTA in a `.pto` container,
 *         built and read in bounded memory.
 *
 * #create_sequence_database streams a FASTA file (gzip-compressed if imp.bff
 * was built with zlib) into one `dstore` object of a `.pto` container
 * (ptolib's streamed store, format 5):
 *
 * - `residues`: variable-length rows of residue codes, Huffman-coded per
 *   segment (~4.2 bits a residue on UniRef; 5 bits where Huffman does not
 *   shrink) with any sequence still decoding alone, or optionally one byte
 *   each, read in place from the memory map. The codes are those of
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
#include <functional>
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
    \param[in] packed residues Huffman-coded (~4.2 bits each; any sequence
               still reads alone), else one byte each (read in place)
    \return the number of sequences written
    \throw IOException when a file cannot be read or written
*/
IMPBFFEXPORT std::size_t create_sequence_database(const std::string& fasta,
                                                  const std::string& out,
                                                  const std::string& name = "sequences",
                                                  int segment_mb = 16, bool packed = true);

//! Rewrite a sequence database, packed (~4.2 bits a residue) or as bytes, in one
//! streaming pass; rows keep their order, so a cluster membership stays valid,
//! and the container's other objects (such as that membership) are copied.
IMPBFFEXPORT std::size_t repack_sequence_database(const std::string& in, const std::string& out,
                                                  bool packed = true,
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
    //! The residues' description (JSON): alphabet and source.
    std::string get_metadata() const;
    //! Whether residues are stored packed (Huffman / 5 bits) rather than a byte each.
    bool get_is_packed() const;

#ifndef SWIG
    //! Sequence \p i's codes (1..20 amino acids, 0 other); \p n its length.
    //! In place from the map when stored as bytes; decoded into \p scratch
    //! when packed. Valid until \p scratch changes.
    const unsigned char* get_codes(std::size_t i, std::uint64_t* n,
                                   std::vector<unsigned char>& scratch) const;
    //! Streaming: the database in segments of whole sequences.
    std::size_t get_number_of_segments() const;
    //! Segment \p k: its first sequence and how many it holds, and its codes
    //! back to back (in place, or decoded into \p scratch when packed).
    //! Sequence boundaries come from #get_offsets.
    const unsigned char* get_segment(std::size_t k, std::size_t* first, std::size_t* count,
                                     std::uint64_t* n_codes,
                                     std::vector<unsigned char>& scratch) const;
    //! Every sequence in row order with its header, one segment of each
    //! column decoded at a time: `visit(row, codes, n, header)`.
    void scan_records(const std::function<void(std::size_t, const unsigned char*, std::uint64_t,
                                               const std::string&)>& visit) const;
    //! Offset of sequence \p i's first code in the concatenation of all.
    std::uint64_t get_offset(std::size_t i) const;
    //! The offsets of sequences \p first .. \p first + \p count, inclusive:
    //! `count + 1` values into \p out.
    void get_offsets(std::size_t first, std::size_t count, std::uint64_t* out) const;
    //! Tell the system the next reads are one pass in order.
    void advise_sequential() const;
    //! Every sequence's identifier (the header's first word) in row order,
    //! decoding each header segment once: `visit(row, identifier)`.
    void scan_identifiers(
            const std::function<void(std::size_t, const std::string&)>& visit) const;
#endif

    IMP_SHOWABLE_INLINE(SequenceDatabase, out << "SequenceDatabase(" << path_ << ", "
                                              << n_sequences_ << " sequences)");

    //! Whether each sequence has members below it: a clustered database
    //! (#IMP::bff::create_clustered_sequence_database), searched in two stages.
    bool get_has_members() const;
    //! The members' level as a database of its own (the same mapped file).
    SequenceDatabase get_members_database() const;
    //! The members of sequence \p i: rows [first, end) of #get_members_database.
    Ints get_member_range(std::size_t i) const;

private:
    std::string path_;
    std::string level_;                        // "" or "members/" ...
    std::string residues_ = "residues", headers_ = "headers";
    std::shared_ptr<pto::StoreReader> reader_;
    std::size_t n_sequences_ = 0;
    std::uint64_t n_residues_ = 0;
};
IMP_VALUES(SequenceDatabase, SequenceDatabases);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_SEQUENCEDATABASE_H */
