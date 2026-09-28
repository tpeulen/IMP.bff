/**
 * \file SequenceCoevolution.cpp
 * \brief Mean-field direct coupling analysis of an alignment.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SequenceCoevolution.h>
#include <IMP/bff/internal/OutputView.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

IMPBFF_BEGIN_NAMESPACE

namespace {

const int kQ = 21;       // gap + 20 amino acids
const int kQm = kQ - 1;  // states kept in the correlation matrix; the last is the gauge

void publish(const std::vector<double>& table, int n, double** out_matrix, int* n_out_rows,
             int* n_out_cols) {
    if (out_matrix == nullptr || n_out_rows == nullptr || n_out_cols == nullptr) return;
    int n_flat = 0;
    double* buffer = internal::new_double_view(table.size(), out_matrix, &n_flat);
    *n_out_rows = 0;
    *n_out_cols = n;
    if (buffer == nullptr) return;
    if (!table.empty()) std::memcpy(buffer, table.data(), table.size() * sizeof(double));
    *n_out_rows = n;
}

//! Direct information of one pair from its coupling block and the two marginals.
/*! The two-site model `P(a, b) ~ exp(e(a, b)) h_i(a) h_j(b)` with fields
    fixed so that its marginals equal \p pi and \p pj; solved by alternating
    the two marginal conditions until the fields stop moving (1e-4). */
double direct_information(const double* w, const double* pi, const double* pj) {
    double h_i[kQ], h_j[kQ], next_i[kQ], next_j[kQ];
    for (int a = 0; a < kQ; ++a) h_i[a] = h_j[a] = 1.0 / kQ;
    for (int iteration = 0; iteration < 10000; ++iteration) {
        double sum_i = 0.0, sum_j = 0.0;
        for (int a = 0; a < kQ; ++a) {
            double row = 0.0;                    // sum_b w(a, b) h_j(b)
            double col = 0.0;                    // sum_b h_i(b) w(b, a)
            for (int b = 0; b < kQ; ++b) {
                row += w[a * kQ + b] * h_j[b];
                col += h_i[b] * w[b * kQ + a];
            }
            next_i[a] = pi[a] / row;
            next_j[a] = pj[a] / col;
            sum_i += next_i[a];
            sum_j += next_j[a];
        }
        double change = 0.0;
        for (int a = 0; a < kQ; ++a) {
            next_i[a] /= sum_i;
            next_j[a] /= sum_j;
            change = std::max(change, std::max(std::fabs(next_i[a] - h_i[a]),
                                               std::fabs(next_j[a] - h_j[a])));
            h_i[a] = next_i[a];
            h_j[a] = next_j[a];
        }
        if (change <= 1e-4) break;
    }
    double p[kQ * kQ];
    double z = 0.0;
    for (int a = 0; a < kQ; ++a) {
        for (int b = 0; b < kQ; ++b) {
            p[a * kQ + b] = w[a * kQ + b] * h_i[a] * h_j[b];
            z += p[a * kQ + b];
        }
    }
    const double tiny = 1e-100;
    double di = 0.0;
    for (int a = 0; a < kQ; ++a) {
        for (int b = 0; b < kQ; ++b) {
            const double pd = p[a * kQ + b] / z;
            di += pd * std::log((pd + tiny) / (pi[a] * pj[b] + tiny));
        }
    }
    return di;
}

}  // namespace

SequenceCoevolution::SequenceCoevolution(int n_columns, double n_effective,
                                         const std::vector<int>& reference_positions,
                                         const std::vector<double>& direct_information,
                                         const std::vector<double>& mutual_information)
    : n_(n_columns), m_eff_(n_effective), ref_pos_(reference_positions),
      di_(direct_information), mi_(mutual_information) {
    const std::size_t n2 = static_cast<std::size_t>(n_columns) * n_columns;
    if (di_.size() != n2 || mi_.size() != n2 ||
        ref_pos_.size() != static_cast<std::size_t>(n_columns)) {
        IMP_THROW("SequenceCoevolution: tables must be n_columns x n_columns", ValueException);
    }
}

void SequenceCoevolution::get_direct_information_matrix(double** out_matrix, int* n_out_rows,
                                                        int* n_out_cols) const {
    publish(di_, n_, out_matrix, n_out_rows, n_out_cols);
}

void SequenceCoevolution::get_mutual_information_matrix(double** out_matrix, int* n_out_rows,
                                                        int* n_out_cols) const {
    publish(mi_, n_, out_matrix, n_out_rows, n_out_cols);
}

SequenceCoevolution compute_sequence_coevolution(const SequenceMSA& msa, double theta,
                                                 double pseudocount) {
    if (!(pseudocount >= 0.0 && pseudocount < 1.0)) {
        IMP_THROW("compute_sequence_coevolution: pseudocount must be in [0, 1)", ValueException);
    }
    const int n = msa.get_n_columns();
    const int m_seq = msa.get_n_sequences();
    if (n < 2) IMP_THROW("compute_sequence_coevolution: needs at least two columns", ValueException);
    const signed char* x = msa.get_data().data();
    const std::vector<double> w = get_sequence_weights(msa, theta);
    double m_eff = 0.0;
    for (double v : w) m_eff += v;

    // Single-site frequencies, then with the pseudocount.
    std::vector<double> f1(static_cast<std::size_t>(n) * kQ, 0.0);
    for (int m = 0; m < m_seq; ++m) {
        for (int i = 0; i < n; ++i) f1[static_cast<std::size_t>(i) * kQ + x[static_cast<std::size_t>(m) * n + i]] += w[m];
    }
    for (double& v : f1) v /= m_eff;
    const double lambda = pseudocount;
    std::vector<double> p1(f1.size());
    for (std::size_t k = 0; k < f1.size(); ++k) p1[k] = (1.0 - lambda) * f1[k] + lambda / kQ;

    // Connected correlations over the first q-1 states, filled straight from
    // the sequences: pair counts of kept states, one row block per thread.
    const int d = n * kQm;
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(d, d);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        for (int m = 0; m < m_seq; ++m) {
            const signed char* row = x + static_cast<std::size_t>(m) * n;
            const int a = row[i];
            if (a >= kQm) continue;
            const double wm = w[m];
            for (int j = i + 1; j < n; ++j) {
                const int b = row[j];
                if (b < kQm) c(i * kQm + a, j * kQm + b) += wm;
            }
        }
    }
    const double pair_pc = lambda / (kQ * kQ);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < kQm; ++a) {
            const int r = i * kQm + a;
            const double pia = p1[static_cast<std::size_t>(i) * kQ + a];
            // diagonal block: P_ii(a, b) = P_i(a) delta_ab
            for (int b = 0; b < kQm; ++b) {
                const double pib = p1[static_cast<std::size_t>(i) * kQ + b];
                c(r, i * kQm + b) = (a == b ? pia : 0.0) - pia * pib;
            }
            for (int j = i + 1; j < n; ++j) {
                for (int b = 0; b < kQm; ++b) {
                    const int s = j * kQm + b;
                    const double pij = (1.0 - lambda) * c(r, s) / m_eff + pair_pc;
                    c(r, s) = pij - pia * p1[static_cast<std::size_t>(j) * kQ + b];
                }
            }
        }
    }
    // Upper triangle is complete; the factorisation reads the lower.
    c.triangularView<Eigen::StrictlyLower>() = c.transpose();

    Eigen::MatrixXd inv = Eigen::MatrixXd::Identity(d, d);
    {
        Eigen::LLT<Eigen::MatrixXd> llt(c);
        if (llt.info() == Eigen::Success) {
            llt.solveInPlace(inv);
        } else {
            Eigen::LDLT<Eigen::MatrixXd> ldlt(c);
            if (ldlt.info() != Eigen::Success) {
                IMP_THROW("compute_sequence_coevolution: the correlation matrix is singular; "
                          "raise the pseudocount", ValueException);
            }
            inv = ldlt.solve(inv);
        }
    }
    c.resize(0, 0);

    std::vector<double> di(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> mi(static_cast<std::size_t>(n) * n, 0.0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n - 1; ++i) {
        double wmat[kQ * kQ];
        double joint[kQ * kQ];
        for (int j = i + 1; j < n; ++j) {
            // couplings e(a, b) = -inv; the gauge row and column are exp(0) = 1
            for (int a = 0; a < kQ; ++a) {
                for (int b = 0; b < kQ; ++b) {
                    wmat[a * kQ + b] = (a < kQm && b < kQm)
                            ? std::exp(-inv(i * kQm + a, j * kQm + b)) : 1.0;
                }
            }
            const double value = direct_information(wmat, &p1[static_cast<std::size_t>(i) * kQ],
                                                    &p1[static_cast<std::size_t>(j) * kQ]);
            // mutual information of the reweighted frequencies, no pseudocount
            std::fill(joint, joint + kQ * kQ, 0.0);
            for (int m = 0; m < m_seq; ++m) {
                const signed char* row = x + static_cast<std::size_t>(m) * n;
                joint[row[i] * kQ + row[j]] += w[m];
            }
            double info = 0.0;
            for (int a = 0; a < kQ; ++a) {
                for (int b = 0; b < kQ; ++b) {
                    const double pab = joint[a * kQ + b] / m_eff;
                    if (pab > 0.0) {
                        info += pab * std::log(pab / (f1[static_cast<std::size_t>(i) * kQ + a] *
                                                      f1[static_cast<std::size_t>(j) * kQ + b]));
                    }
                }
            }
            di[static_cast<std::size_t>(i) * n + j] = di[static_cast<std::size_t>(j) * n + i] = value;
            mi[static_cast<std::size_t>(i) * n + j] = mi[static_cast<std::size_t>(j) * n + i] = info;
        }
    }
    return SequenceCoevolution(n, m_eff, msa.get_reference_positions(), di, mi);
}

std::vector<double> probe_pair_coevolution(const SequenceCoevolution& coevolution,
                                           int* pair_residues, int n_pair_rows,
                                           int n_pair_cols, int residue_offset) {
    if (n_pair_cols != 2) {
        IMP_THROW("probe_pair_coevolution: two residues per pair", ValueException);
    }
    const std::vector<int>& ref = coevolution.get_reference_positions();
    auto column_of = [&](int residue) {
        const int wanted = residue - residue_offset;
        for (std::size_t k = 0; k < ref.size(); ++k) {
            if (ref[k] == wanted) return static_cast<int>(k);
        }
        return -1;
    };
    std::vector<double> out(static_cast<std::size_t>(n_pair_rows),
                            std::numeric_limits<double>::quiet_NaN());
    for (int p = 0; p < n_pair_rows; ++p) {
        const int a = column_of(pair_residues[2 * p]);
        const int b = column_of(pair_residues[2 * p + 1]);
        if (a < 0 || b < 0 || a == b) continue;
        out[static_cast<std::size_t>(p)] = coevolution.get_direct_information(a, b);
    }
    return out;
}

IMPBFF_END_NAMESPACE
