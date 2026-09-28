/**
 * \file SequenceDatabase.cpp
 * \brief A protein sequence database as binary FASTA in a `.pto` container.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SequenceDatabase.h>
#include <IMP/bff/BffSettings.h>
#include <IMP/bff/SequenceMSA.h>
#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/ptolib.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>

#ifdef IMP_BFF_HAS_ZLIB
#include <zlib.h>
#endif

IMPBFF_BEGIN_NAMESPACE

namespace {

const char kColumnResidues[] = "residues";
const char kColumnHeaders[] = "headers";
//! Code -> letter: the SequenceMSA states, with X for 0.
const char kLetters[] = "XACDEFGHIKLMNPQRSTVWY";
const unsigned char kSkip = 255;

//! Byte -> residue code, or kSkip for what is not a residue (line ends,
//! blanks, `*` stop codons, gaps).
std::array<unsigned char, 256> make_codes() {
    std::array<unsigned char, 256> t;
    t.fill(kSkip);
    const std::string alphabet = get_sequence_alphabet();
    for (int c = 'A'; c <= 'Z'; ++c) {
        const std::string::size_type k = alphabet.find(static_cast<char>(c));
        const unsigned char code = k == std::string::npos ? 0 : static_cast<unsigned char>(k + 1);
        t[static_cast<unsigned char>(c)] = code;
        t[static_cast<unsigned char>(c - 'A' + 'a')] = code;
    }
    return t;
}

//! A FASTA file, plain or gzip-compressed, read in blocks.
class FastaSource {
public:
    explicit FastaSource(const std::string& path) : path_(path) {
#ifdef IMP_BFF_HAS_ZLIB
        // gzread reads an uncompressed file unchanged, so one path serves both.
        gz_ = gzopen(path.c_str(), "rb");
        if (gz_ == nullptr) IMP_THROW("create_sequence_database: cannot read " << path, IOException);
        gzbuffer(gz_, 1u << 20);
#else
        f_ = std::fopen(path.c_str(), "rb");
        if (f_ == nullptr) IMP_THROW("create_sequence_database: cannot read " << path, IOException);
        unsigned char magic[2] = {0, 0};
        const std::size_t got = std::fread(magic, 1, 2, f_);
        if (got == 2 && magic[0] == 0x1f && magic[1] == 0x8b) {
            std::fclose(f_);
            IMP_THROW("create_sequence_database: " << path << " is gzip-compressed and "
                      "this imp.bff was built without zlib; decompress it first",
                      IOException);
        }
        std::rewind(f_);
#endif
    }
    ~FastaSource() {
#ifdef IMP_BFF_HAS_ZLIB
        if (gz_ != nullptr) gzclose(gz_);
#else
        if (f_ != nullptr) std::fclose(f_);
#endif
    }
    FastaSource(const FastaSource&) = delete;
    FastaSource& operator=(const FastaSource&) = delete;

    //! Up to \p n bytes; 0 at the end.
    std::size_t read(char* into, std::size_t n) {
#ifdef IMP_BFF_HAS_ZLIB
        const int got = gzread(gz_, into, static_cast<unsigned>(n));
        if (got < 0) {
            int code = 0;
            IMP_THROW("create_sequence_database: " << path_ << ": " << gzerror(gz_, &code),
                      IOException);
        }
        return static_cast<std::size_t>(got);
#else
        const std::size_t got = std::fread(into, 1, n, f_);
        if (got < n && std::ferror(f_))
            IMP_THROW("create_sequence_database: error reading " << path_, IOException);
        return got;
#endif
    }

private:
    std::string path_;
#ifdef IMP_BFF_HAS_ZLIB
    gzFile gz_ = nullptr;
#else
    std::FILE* f_ = nullptr;
#endif
};

std::string resolve_database(const std::string& path_or_name) {
    std::ifstream probe(path_or_name, std::ios::binary);
    if (probe) return path_or_name;
    const SequenceSearchSettings settings = get_sequence_search_settings();
    if (settings.get_has_database(path_or_name)) return settings.get_database(path_or_name);
    IMP_THROW("SequenceDatabase: " << path_or_name << " is neither a file nor a database "
              "named in the settings (" << get_settings_path() << ")", IOException);
}

}  // namespace

bool get_sequence_database_reads_gzip() {
#ifdef IMP_BFF_HAS_ZLIB
    return true;
#else
    return false;
#endif
}

std::size_t create_sequence_database(const std::string& fasta, const std::string& out,
                                     const std::string& name, int segment_mb) {
    if (segment_mb < 1) IMP_THROW("create_sequence_database: segment_mb must be >= 1",
                                  ValueException);
    static const std::array<unsigned char, 256> codes = make_codes();
    FastaSource source(fasta);

    pto::File container;
    if (!container.create(out, "sequence database"))
        IMP_THROW("create_sequence_database: " << container.error(), IOException);
    container.set_writing_app("imp.bff create_sequence_database");
    std::size_t n = 0;
    {
        pto::StoreWriter store(container, "table", name, pto::StoreOptions(),
                               static_cast<std::size_t>(segment_mb) << 20);
        // Residues raw, to be read in place; headers compressed in small
        // segments, since a hit's header is looked up alone.
        const int residues = store.add_column(kColumnResidues, pto::ColumnType::UInt8, true, "none");
        const std::string header_codec = pto::can_compress("zstd") ? "zstd" : "none";
        const int headers = store.add_column(kColumnHeaders, pto::ColumnType::UInt8, true,
                                             header_codec, std::size_t(64) << 10);
        nlohmann::json meta;
        meta["alphabet"] = kLetters;
        meta["source"] = fasta;
        store.set_metadata(residues, meta.dump());

        std::string header;
        std::vector<unsigned char> sequence;
        bool have = false, in_header = false, line_start = true;
        auto emit = [&]() {
            while (!header.empty() && (header.back() == '\r' || header.back() == ' '))
                header.pop_back();
            store.append_row(residues, sequence.data(), sequence.size());
            store.append_row(headers, header.data(), header.size());
            ++n;
        };
        std::vector<char> block(std::size_t(1) << 20);
        for (std::size_t got; (got = source.read(block.data(), block.size())) > 0;) {
            for (std::size_t i = 0; i < got; ++i) {
                const char c = block[i];
                if (in_header) {
                    if (c == '\n') { in_header = false; line_start = true; }
                    else header.push_back(c);
                    continue;
                }
                if (c == '>' && line_start) {
                    if (have) emit();
                    header.clear();
                    sequence.clear();
                    have = true;
                    in_header = true;
                    continue;
                }
                line_start = c == '\n';
                const unsigned char code = codes[static_cast<unsigned char>(c)];
                if (code == kSkip) continue;
                if (!have)
                    IMP_THROW("create_sequence_database: " << fasta
                              << " is not FASTA: residues before the first header",
                              IOException);
                sequence.push_back(code);
            }
        }
        if (have) emit();
        if (!store.close())
            IMP_THROW("create_sequence_database: could not write " << out, IOException);
    }
    if (!container.commit())
        IMP_THROW("create_sequence_database: " << container.error(), IOException);
    container.close();
    return n;
}

SequenceDatabase::SequenceDatabase(const std::string& path_or_name, const std::string& name)
    : path_(resolve_database(path_or_name)) {
    pto::File container;
    if (!container.open(path_))
        IMP_THROW("SequenceDatabase: " << path_ << ": " << container.error(), IOException);
    const std::uint64_t uid = container.find(name);
    if (uid == 0)
        IMP_THROW("SequenceDatabase: " << path_ << " holds no store '" << name << "'",
                  IOException);
    try {
        reader_ = std::make_shared<pto::StoreReader>(container, uid);
    } catch (const std::exception& error) {
        IMP_THROW("SequenceDatabase: " << error.what(), IOException);
    }
    if (!reader_->has_column(kColumnResidues) || !reader_->has_column(kColumnHeaders) ||
        !reader_->info(kColumnResidues).ragged)
        IMP_THROW("SequenceDatabase: " << path_ << " is not a sequence database",
                  IOException);
    const pto::ColumnInfo info = reader_->info(kColumnResidues);
    n_sequences_ = static_cast<std::size_t>(info.n_rows);
    n_residues_ = info.n_values;
}

const unsigned char* SequenceDatabase::get_codes(std::size_t i, std::uint64_t* n) const {
    if (!reader_ || i >= n_sequences_)
        IMP_THROW("SequenceDatabase: no sequence " << i, IndexException);
    const void* p = reader_->row_view(kColumnResidues, i, n);
    return static_cast<const unsigned char*>(p);
}

std::string SequenceDatabase::get_sequence(std::size_t i) const {
    std::uint64_t n = 0;
    const unsigned char* codes = get_codes(i, &n);
    std::string out(static_cast<std::size_t>(n), 'X');
    for (std::uint64_t k = 0; k < n; ++k)
        out[static_cast<std::size_t>(k)] = codes[k] <= 20 ? kLetters[codes[k]] : 'X';
    return out;
}

int SequenceDatabase::get_length(std::size_t i) const {
    if (!reader_ || i >= n_sequences_)
        IMP_THROW("SequenceDatabase: no sequence " << i, IndexException);
    return static_cast<int>(reader_->row_size(kColumnResidues, i));
}

std::string SequenceDatabase::get_header(std::size_t i) const {
    if (!reader_ || i >= n_sequences_)
        IMP_THROW("SequenceDatabase: no sequence " << i, IndexException);
    std::vector<unsigned char> buf;
    reader_->row(kColumnHeaders, i, buf);
    return std::string(buf.begin(), buf.end());
}

std::string SequenceDatabase::get_identifier(std::size_t i) const {
    const std::string header = get_header(i);
    const std::string::size_type end = header.find_first_of(" \t");
    return end == std::string::npos ? header : header.substr(0, end);
}

std::size_t SequenceDatabase::get_number_of_segments() const {
    return reader_ ? reader_->n_segments(kColumnResidues) : 0;
}

const unsigned char* SequenceDatabase::get_segment(std::size_t k, std::size_t* first,
                                                   std::size_t* count,
                                                   std::uint64_t* n_codes) const {
    if (!reader_ || k >= get_number_of_segments())
        IMP_THROW("SequenceDatabase: no segment " << k, IndexException);
    const pto::SegmentInfo seg = reader_->segment(kColumnResidues, k);
    if (first) *first = static_cast<std::size_t>(seg.first_row);
    if (count) *count = static_cast<std::size_t>(seg.n_rows);
    if (n_codes) *n_codes = seg.n_values;
    // Stored raw, so always in place.
    return static_cast<const unsigned char*>(reader_->view(kColumnResidues, seg.first_value,
                                                           seg.n_values));
}

std::uint64_t SequenceDatabase::get_offset(std::size_t i) const {
    if (!reader_ || i > n_sequences_)
        IMP_THROW("SequenceDatabase: no sequence " << i, IndexException);
    std::uint64_t o = 0;
    if (i == n_sequences_) return n_residues_;
    reader_->offsets(kColumnResidues, i, 0, &o);
    return o;
}

void SequenceDatabase::get_offsets(std::size_t first, std::size_t count,
                                   std::uint64_t* out) const {
    if (!reader_ || first + count > n_sequences_)
        IMP_THROW("SequenceDatabase: no sequences " << first << "+" << count, IndexException);
    reader_->offsets(kColumnResidues, first, count, out);
}

void SequenceDatabase::scan_identifiers(
        const std::function<void(std::size_t, const std::string&)>& visit) const {
    if (!reader_) return;
    std::vector<unsigned char> scratch;
    std::vector<std::uint64_t> offsets;
    std::string id;
    for (std::size_t k = 0; k < reader_->n_segments(kColumnHeaders); ++k) {
        const pto::SegmentInfo seg = reader_->segment(kColumnHeaders, k);
        const char* text = static_cast<const char*>(
                reader_->segment_data(kColumnHeaders, k, scratch));
        offsets.resize(static_cast<std::size_t>(seg.n_rows) + 1);
        reader_->offsets(kColumnHeaders, seg.first_row, seg.n_rows, offsets.data());
        for (std::uint64_t r = 0; r < seg.n_rows; ++r) {
            const char* h = text + (offsets[r] - offsets[0]);
            const std::size_t n = static_cast<std::size_t>(offsets[r + 1] - offsets[r]);
            std::size_t end = 0;
            while (end < n && h[end] != ' ' && h[end] != '\t') ++end;
            id.assign(h, end);
            visit(static_cast<std::size_t>(seg.first_row + r), id);
        }
    }
}

void SequenceDatabase::advise_sequential() const {
    if (reader_) reader_->advise_sequential(kColumnResidues);
}

IMPBFF_END_NAMESPACE
