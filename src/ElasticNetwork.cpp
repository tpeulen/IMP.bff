/**
 *  \file ElasticNetwork.cpp
 *  \brief Anisotropic network model: Hessian, modes, pair fluctuations.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/ElasticNetwork.h>
#include <IMP/bff/internal/OutputView.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

IMPBFF_BEGIN_NAMESPACE

namespace elastic_network {

//! Fluctuation of pair (i, j) in the first \p k modes.
double fluctuation(const ElasticNetworkModes& m, int i, int j, int k) {
    const std::vector<double>& x = m.coordinates();
    double e[3], norm = 0;
    for (int c = 0; c < 3; ++c) {
        e[c] = x[3 * j + c] - x[3 * i + c];
        norm += e[c] * e[c];
    }
    norm = std::sqrt(norm);
    if (norm == 0) return 0;
    double var = 0;
    for (int mode = 0; mode < k; ++mode) {
        const double* u = m.mode(mode);
        double p = 0;
        for (int c = 0; c < 3; ++c) p += e[c] / norm * (u[3 * j + c] - u[3 * i + c]);
        var += p * p / m.eigenvalues()[mode];
    }
    return var;
}

int checked_modes(const ElasticNetworkModes& m, int n_modes) {
    if (n_modes < 1 || n_modes > m.get_number_of_modes())
        IMP_THROW("elastic network: " << n_modes << " modes asked, " << m.get_number_of_modes()
                                      << " available", ValueException);
    return n_modes;
}

}  // namespace elastic_network

ElasticNetworkModes::ElasticNetworkModes(double* coordinates, int n_points, int n_dim,
                                         double cutoff)
    : n_all_(n_points), cutoff_(cutoff) {
    if (n_dim != 3) IMP_THROW("ElasticNetworkModes: coordinates must be n x 3", ValueException);
    network_index_.resize(static_cast<std::size_t>(n_points));
    for (int i = 0; i < n_points; ++i) network_index_[static_cast<std::size_t>(i)] = i;
    build(std::vector<double>(coordinates, coordinates + 3 * n_points));
}

ElasticNetworkModes::ElasticNetworkModes(double* coordinates, int n_points, int n_dim,
                                         double* confidence, int n_confidence,
                                         double min_confidence, double cutoff)
    : n_all_(n_points), cutoff_(cutoff) {
    if (n_dim != 3) IMP_THROW("ElasticNetworkModes: coordinates must be n x 3", ValueException);
    if (n_confidence != n_points)
        IMP_THROW("ElasticNetworkModes: " << n_confidence << " confidences for " << n_points
                  << " points", ValueException);
    network_index_.assign(static_cast<std::size_t>(n_points), -1);
    std::vector<double> kept;
    for (int i = 0; i < n_points; ++i) {
        if (!(confidence[i] >= min_confidence)) continue;            // NaN is left out too
        network_index_[static_cast<std::size_t>(i)] = static_cast<int>(kept.size() / 3);
        kept.insert(kept.end(), coordinates + 3 * i, coordinates + 3 * i + 3);
    }
    build(kept);
}

void ElasticNetworkModes::build(const std::vector<double>& xyz) {
    n_ = static_cast<int>(xyz.size() / 3);
    if (n_ < 3) IMP_THROW("ElasticNetworkModes: at least three points in the network", ValueException);
    xyz_ = xyz;
    const int n_points = n_;
    const int N = 3 * n_points;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N);
    const double c2 = cutoff_ * cutoff_;
    for (int i = 0; i < n_points; ++i)
        for (int j = i + 1; j < n_points; ++j) {
            double d[3], r2 = 0;
            for (int c = 0; c < 3; ++c) {
                d[c] = xyz_[3 * j + c] - xyz_[3 * i + c];
                r2 += d[c] * d[c];
            }
            if (r2 >= c2 || r2 == 0) continue;
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    const double v = -d[p] * d[q] / r2;      // off-diagonal super-element
                    H(3 * i + p, 3 * j + q) = v;
                    H(3 * j + q, 3 * i + p) = v;
                    H(3 * i + p, 3 * i + q) -= v;
                    H(3 * j + p, 3 * j + q) -= v;
                }
        }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
    if (solver.info() != Eigen::Success)
        IMP_THROW("ElasticNetworkModes: the eigensolver failed", ValueException);
    // the six smallest are the rigid-body motions of a connected network
    const int first = 6;
    for (int k = first; k < N; ++k) {
        eigenvalues_.push_back(solver.eigenvalues()[k]);
        for (int r = 0; r < N; ++r) vectors_.push_back(solver.eigenvectors()(r, k));
    }
    if (!eigenvalues_.empty() && eigenvalues_.front() <= 1e-8 * std::max(1.0, eigenvalues_.back()))
        IMP_THROW("ElasticNetworkModes: more than six zero modes; the network falls apart at "
                  << cutoff_ << " A (raise the cut-off)", ValueException);
}

void ElasticNetworkModes::get_mode(int k, double** out_matrix, int* n_out_rows,
                                   int* n_out_cols) const {
    if (k < 0 || k >= get_number_of_modes())
        IMP_THROW("ElasticNetworkModes: no mode " << k, IndexException);
    if (out_matrix == nullptr || n_out_rows == nullptr || n_out_cols == nullptr) return;
    int n_flat = 0;
    double* buffer = internal::new_double_view(std::size_t(3) * n_, out_matrix, &n_flat);
    *n_out_rows = 0;
    *n_out_cols = 3;
    if (buffer == nullptr) return;
    std::memcpy(buffer, mode(k), std::size_t(3) * n_ * sizeof(double));
    *n_out_rows = n_;
}

std::vector<double> get_pair_distance_fluctuations(const ElasticNetworkModes& modes, int* pairs,
                                                   int n_pair_rows, int n_pair_cols, int n_modes) {
    if (n_pair_cols != 2) IMP_THROW("get_pair_distance_fluctuations: two points per pair", ValueException);
    const int k = elastic_network::checked_modes(modes, n_modes);
    std::vector<double> out(static_cast<std::size_t>(n_pair_rows),
                            std::numeric_limits<double>::quiet_NaN());
    for (int p = 0; p < n_pair_rows; ++p) {
        const int a = pairs[2 * p], b = pairs[2 * p + 1];
        if (a == b || !modes.get_is_in_network(a) || !modes.get_is_in_network(b)) continue;
        out[static_cast<std::size_t>(p)] =
                elastic_network::fluctuation(modes, modes.network_index(a), modes.network_index(b), k);
    }
    return out;
}

std::vector<double> get_pair_change_probabilities(const ElasticNetworkModes& modes, int* pairs,
                                                  int n_pair_rows, int n_pair_cols, int n_modes,
                                                  double a, double b) {
    const std::vector<double> var =
            get_pair_distance_fluctuations(modes, pairs, n_pair_rows, n_pair_cols, n_modes);
    // the structure's own scale: the 95th percentile over pairs |i - j| >= 6
    const int n = modes.get_number_of_network_points();
    std::vector<double> all;
    for (int i = 0; i < n; ++i)
        for (int j = i + 6; j < n; ++j) all.push_back(elastic_network::fluctuation(modes, i, j, n_modes));
    if (all.empty())
        IMP_THROW("get_pair_change_probabilities: too few points for a reference scale", ValueException);
    const std::size_t q = static_cast<std::size_t>(0.95 * double(all.size() - 1));
    std::nth_element(all.begin(), all.begin() + q, all.end());
    const double q95 = all[q];
    std::vector<double> out(var.size(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t p = 0; p < var.size(); ++p) {
        if (std::isnan(var[p])) continue;
        const double x = std::min(4.0, std::max(-8.0, std::log(var[p] / q95 + 1e-12)));
        out[p] = 1.0 / (1.0 + std::exp(-(a * x + b)));
    }
    return out;
}

IMPBFF_END_NAMESPACE
