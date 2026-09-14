/**
 * \file PolymerChain.cpp
 * \brief End-to-end distance distributions of ideal and worm-like chains.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/PolymerChain.h>
#include <IMP/bff/SpecialFunctions.h>
#include <IMP/bff/Distributions.h>
#include <IMP/bff/internal/Normalize.h>
#include <IMP/bff/internal/OutputView.h>
#include <IMP/bff/internal/DistanceKernels.h>

#include <algorithm>
#include <cmath>
#include <cstring>

IMPBFF_BEGIN_NAMESPACE

//! The ideal-chain kernel as a buffer, for the view-publishing wrapper.
std::vector<double> gaussian_chain_impl(const std::vector<double>& distances,
                                        double segment_length,
                                        int number_of_segments) {
    const double ree = gaussian_chain_ree(segment_length, number_of_segments);
    const double r2_mean = ree * ree;
    std::vector<double> out(distances.size(), 0.0);
    if (r2_mean == 0.0) return out;
    const double denom = std::pow(2.0 / 3.0 * M_PI * r2_mean, 1.5);
    for (std::size_t i = 0; i < distances.size(); ++i) {
        const double r = distances[i];
        out[i] = 4.0 * M_PI * r * r / denom * std::exp(-1.5 * r * r / r2_mean);
    }
    return out;
}

//! The worm-like-chain kernel as a buffer (the wrapper publishes a view).
/*! The arithmetic is internal/DistanceKernels.h's, shared with the
    derivative PolymerDistances reports. */
std::vector<double> worm_like_chain_impl(const std::vector<double>& distances,
                                         double kappa, double chain_length,
                                         bool normalize, bool distance) {
    return internal::worm_like_chain_t<double>(distances, kappa, chain_length,
                                               normalize, distance);
}

double gaussian_chain_ree(double segment_length, int number_of_segments) {
    return segment_length * std::sqrt(static_cast<double>(number_of_segments));
}

void gaussian_chain(const std::vector<double>& distances,
                    double segment_length, int number_of_segments,
                    double** out_view, int* n_out_view) {
    internal::copy_to_view(
            gaussian_chain_impl(distances, segment_length, number_of_segments),
            out_view, n_out_view);
}

void worm_like_chain(const std::vector<double>& distances, double kappa,
                     double chain_length, bool normalize, bool distance,
                     double** out_view, int* n_out_view) {
    internal::copy_to_view(
            worm_like_chain_impl(distances, kappa, chain_length, normalize, distance),
            out_view, n_out_view);
}

void worm_like_chain_linker(const std::vector<double>& distances, double kappa,
                            double chain_length, double sigma, bool normalize,
                            double** out_view, int* n_out_view) {
    // The broadening kernel is the distance between two Gaussian clouds, not
    // a normal density; see internal/DistanceKernels.h.
    internal::copy_to_view(
            internal::worm_like_chain_linker_t<double>(distances, kappa, chain_length,
                                                       sigma, normalize),
            out_view, n_out_view);
}


void ising_chain(const std::vector<double>& distances, int number_of_residues,
                 double b_structured, double b_unstructured, double coupling,
                 double field, int n_k, double** out_view, int* n_out_view) {
    internal::copy_to_view(
            internal::ising_chain_t<double>(distances, number_of_residues, b_structured,
                                            b_unstructured, coupling, field, n_k),
            out_view, n_out_view);
}


void saw_nu(const std::vector<double>& distances, double r_rms, double nu,
            double gamma_exp, double** out_view, int* n_out_view) {
    internal::copy_to_view(internal::saw_nu_t<double>(distances, r_rms, nu, gamma_exp),
                           out_view, n_out_view);
}

IMPBFF_END_NAMESPACE
