/**
 * \file SequenceMSA.cpp
 * \brief An encoded protein multiple sequence alignment.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SequenceMSA.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace {

const char kAlphabet[] = "ACDEFGHIKLMNPQRSTVWY";

//! 0 for a gap or anything that is not one of the 20 amino acids.
signed char encode(char c) {
    const char u = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    for (int k = 0; k < 20; ++k) {
        if (kAlphabet[k] == u) return static_cast<signed char>(k + 1);
    }
    return 0;
}

bool is_match_state(char c) {
    return c != '.' && !std::islower(static_cast<unsigned char>(c));
}

}  // namespace

std::string get_sequence_alphabet() { return std::string(kAlphabet); }

SequenceMSA::SequenceMSA(const std::vector<std::string>& names,
                         const std::vector<std::string>& sequences, int reference,
                         bool match_columns_only)
    : n_seq_(static_cast<int>(sequences.size())), reference_(reference), names_(names) {
    if (sequences.empty()) IMP_THROW("SequenceMSA: no sequences", ValueException);
    if (names.size() != sequences.size()) {
        IMP_THROW("SequenceMSA: one name per sequence", ValueException);
    }
    if (reference < 0 || reference >= n_seq_) {
        IMP_THROW("SequenceMSA: reference " << reference << " is not a sequence index",
                  ValueException);
    }
    const std::size_t length = sequences[0].size();
    for (std::size_t m = 0; m < sequences.size(); ++m) {
        if (sequences[m].size() != length) {
            IMP_THROW("SequenceMSA: sequence " << m << " (" << names[m] << ") has length "
                      << sequences[m].size() << ", the first " << length
                      << "; the input is not aligned", ValueException);
        }
    }
    const std::string& ref = sequences[static_cast<std::size_t>(reference)];
    std::vector<std::size_t> keep;
    int residue = 0;
    for (std::size_t j = 0; j < length; ++j) {
        const char c = ref[j];
        const bool is_residue = c != '-' && c != '.';
        if (is_residue) ++residue;
        if (match_columns_only && !is_match_state(c)) continue;
        keep.push_back(j);
        ref_pos_.push_back(is_residue ? residue : -1);
    }
    n_col_ = static_cast<int>(keep.size());
    data_.resize(static_cast<std::size_t>(n_seq_) * n_col_);
    for (int m = 0; m < n_seq_; ++m) {
        const std::string& s = sequences[static_cast<std::size_t>(m)];
        signed char* row = &data_[static_cast<std::size_t>(m) * n_col_];
        for (int k = 0; k < n_col_; ++k) row[k] = encode(s[keep[static_cast<std::size_t>(k)]]);
    }
}

std::string SequenceMSA::get_sequence(int sequence) const {
    std::string out(static_cast<std::size_t>(n_col_), '-');
    for (int k = 0; k < n_col_; ++k) {
        const int s = get_state(sequence, k);
        if (s > 0) out[static_cast<std::size_t>(k)] = kAlphabet[s - 1];
    }
    return out;
}

SequenceMSA read_sequence_msa(const std::string& path, int reference,
                              bool match_columns_only) {
    std::ifstream in(path.c_str());
    if (!in) IMP_THROW("read_sequence_msa: cannot read " << path, IOException);
    std::vector<std::string> names, sequences;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
        if (line.empty()) continue;
        if (line[0] == '>') {
            names.push_back(line.substr(1));
            sequences.push_back(std::string());
            continue;
        }
        if (sequences.empty()) {
            IMP_THROW("read_sequence_msa: " << path << " does not start with a '>' header",
                      ValueException);
        }
        for (char c : line) {
            if (!std::isspace(static_cast<unsigned char>(c))) sequences.back().push_back(c);
        }
    }
    return SequenceMSA(names, sequences, reference, match_columns_only);
}

std::vector<double> get_sequence_weights(const SequenceMSA& msa, double theta) {
    const int m_seq = msa.get_n_sequences();
    const int n_col = msa.get_n_columns();
    std::vector<double> weights(static_cast<std::size_t>(m_seq), 1.0);
    if (!(theta > 0.0) || n_col == 0) return weights;
    const signed char* data = msa.get_data().data();
    // d(m, n) < theta  <=>  mismatches < theta * n_col; counting stops at the bound.
    const double bound = theta * n_col;
    std::vector<int> neighbours(static_cast<std::size_t>(m_seq), 0);
#pragma omp parallel for schedule(dynamic, 16)
    for (int m = 0; m < m_seq; ++m) {
        const signed char* a = data + static_cast<std::size_t>(m) * n_col;
        int count = 0;
        for (int n = 0; n < m_seq; ++n) {
            if (n == m) continue;
            const signed char* b = data + static_cast<std::size_t>(n) * n_col;
            int diff = 0;
            bool close = true;
            for (int k = 0; k < n_col; ++k) {
                diff += a[k] != b[k];
                if (diff >= bound) { close = false; break; }
            }
            if (close) ++count;
        }
        neighbours[static_cast<std::size_t>(m)] = count;
    }
    for (int m = 0; m < m_seq; ++m) {
        weights[static_cast<std::size_t>(m)] = 1.0 / (1.0 + neighbours[static_cast<std::size_t>(m)]);
    }
    return weights;
}

std::string get_residue_variety(const SequenceMSA& msa, int column) {
    if (column < 0 || column >= msa.get_n_columns()) {
        IMP_THROW("get_residue_variety: column " << column << " out of range", ValueException);
    }
    bool seen[21] = {false};
    for (int m = 0; m < msa.get_n_sequences(); ++m) seen[msa.get_state(m, column)] = true;
    std::string out;
    for (int k = 1; k <= 20; ++k) {
        if (seen[k]) out.push_back(kAlphabet[k - 1]);
    }
    return out;
}

int get_n_with_data(const SequenceMSA& msa, int column) {
    if (column < 0 || column >= msa.get_n_columns()) {
        IMP_THROW("get_n_with_data: column " << column << " out of range", ValueException);
    }
    int n = 0;
    for (int m = 0; m < msa.get_n_sequences(); ++m) n += msa.get_state(m, column) > 0;
    return n;
}

IMPBFF_END_NAMESPACE
