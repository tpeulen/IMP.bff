/**
 *  \file IMP/bff/ElasticNetwork.h
 *  \brief Anisotropic elastic network modes of a structure, and from them
 *         which residue pairs are likely to change distance: a dynamics
 *         signal for choosing FRET pairs.
 *
 * The anisotropic network model (ANM; Atilgan et al., Biophys J 80:505,
 * 2001) joins every pair of points closer than a cut-off by a spring of
 * equal stiffness. The Hessian of that network has six zero modes (rigid
 * motion) and, above them, the normal modes. The softest ones are known to
 * resemble functional motions such as hinge closure (Tama & Sanejouand,
 * Protein Eng 14:1, 2001).
 *
 * For a pair (i, j), the distance fluctuation carried by the k softest modes
 * is
 *
 * \f[ \sigma^2_{ij} = \sum_{m \le k} \frac{\big(\hat e_{ij}\cdot(u^m_j - u^m_i)\big)^2}{\lambda_m}, \f]
 *
 * with \f$\hat e_{ij}\f$ the unit vector between the two points and
 * \f$u^m\f$, \f$\lambda_m\f$ mode m and its eigenvalue.
 *
 * #get_pair_change_probabilities turns the fluctuation into the probability
 * that the pair's distance changes by more than 5 Å between two states. It
 * uses a logistic in \f$x = \log(\sigma^2/\sigma^2_{95})\f$, the fluctuation
 * relative to the 95th percentile over all residue pairs at least 6 apart in
 * sequence, clipped to [−8, 4]. Its default parameters (a = 0.666,
 * b = 0.295, k = 2) were fitted on seven open/closed proteins, each counted
 * once: adenylate kinase, maltose-, ribose-, glutamine- and LAO-binding
 * protein, lactoferrin, guanylate kinase. Leaving one protein out, the AUC
 * for pairs changing by more than 5 Å was 0.86–0.96 for six of them and 0.70
 * for lactoferrin, whose softest modes move its two lobes while the change
 * is within one (okf/esm-contacts.md, "Elastic network").
 *
 * The Hessian is diagonalised in full: 3n × 3n, seconds at a few hundred
 * residues.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_ELASTICNETWORK_H
#define IMPBFF_ELASTICNETWORK_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The normal modes of an anisotropic elastic network.
class IMPBFFEXPORT ElasticNetworkModes {
public:
    ElasticNetworkModes() {}
    //! The network of \p coordinates (`n x 3`, e.g. the Cα of one chain),
    //! springs between points closer than \p cutoff (Å).
    /*! \throw ValueException when there are fewer than three points or the
               coordinates are not `n x 3` */
    ElasticNetworkModes(double* coordinates, int n_points, int n_dim, double cutoff = 15.0);
    //! The same, leaving out every point whose \p confidence is below
    //! \p min_confidence: e.g. AlphaFold's pLDDT (the B-factor column of its
    //! models) with 70. Disordered tails and signal peptides otherwise take
    //! the softest modes; on the AlphaFold models of seven open/closed
    //! proteins, leaving out pLDDT < 70 raised the top-20 precision for pairs
    //! changing by > 5 A from 0.49 to 0.74 (AUC 0.795 -> 0.820), while
    //! pLDDT-weighted springs did not help. Pairs with a left-out point are
    //! NaN in #get_pair_distance_fluctuations / #get_pair_change_probabilities.
    ElasticNetworkModes(double* coordinates, int n_points, int n_dim, double* confidence,
                        int n_confidence, double min_confidence = 70.0, double cutoff = 15.0);

    //! All points given, including any left out of the network.
    int get_number_of_points() const { return n_all_; }
    //! Whether point \p i is part of the network.
    bool get_is_in_network(int i) const {
        return i >= 0 && i < n_all_ && network_index_[static_cast<std::size_t>(i)] >= 0;
    }
    //! The points in the network.
    int get_number_of_network_points() const { return n_; }
    double get_cutoff() const { return cutoff_; }
    //! Modes above the six rigid-body ones, softest first.
    int get_number_of_modes() const { return static_cast<int>(eigenvalues_.size()); }
    //! Their eigenvalues, ascending.
    std::vector<double> get_eigenvalues() const { return eigenvalues_; }
    //! Mode \p k as displacements of the network's points, `n x 3` (unit norm).
    void get_mode(int k, double** out_matrix, int* n_out_rows, int* n_out_cols) const;

#ifndef SWIG
    const std::vector<double>& eigenvalues() const { return eigenvalues_; }
    //! Mode \p k: `3n` values, point by point.
    const double* mode(int k) const { return &vectors_[static_cast<std::size_t>(k) * 3 * n_]; }
    const std::vector<double>& coordinates() const { return xyz_; }
    //! Index of point \p i in the network, -1 when left out.
    int network_index(int i) const { return network_index_[static_cast<std::size_t>(i)]; }
#endif

    IMP_SHOWABLE_INLINE(ElasticNetworkModes, out << "ElasticNetworkModes(" << n_ << " points, "
                                                 << get_number_of_modes() << " modes)");

private:
    void build(const std::vector<double>& xyz);
    int n_ = 0, n_all_ = 0;
    double cutoff_ = 15.0;
    std::vector<int> network_index_;     // per point given: its index in the network, or -1
    std::vector<double> xyz_;            // n x 3
    std::vector<double> eigenvalues_;    // non-rigid, ascending
    std::vector<double> vectors_;        // per mode, 3n
};
IMP_VALUES(ElasticNetworkModes, ElasticNetworkModesList);

//! Distance fluctuation of each pair in the \p n_modes softest modes.
/*!
    \param[in] modes the network
    \param[in] pairs,n_pair_rows,n_pair_cols two point indices (0-based) per pair
    \param[in] n_modes how many of the softest modes
    \return \f$\sigma^2_{ij}\f$ per pair, in the network's arbitrary units;
            NaN for a pair with an index out of range or i = j
*/
IMPBFFEXPORT std::vector<double> get_pair_distance_fluctuations(
        const ElasticNetworkModes& modes, int* pairs, int n_pair_rows, int n_pair_cols,
        int n_modes = 2);

//! Probability that each pair's distance changes by more than 5 Å between
//! states: the calibrated logistic in the pair's fluctuation relative to the
//! structure's own (see the file description).
/*!
    Usable as the per-pair probabilities of a #IMP::bff::ProbePairBenefitTerm.
    \param[in] modes the network
    \param[in] pairs,n_pair_rows,n_pair_cols two point indices (0-based) per pair
    \param[in] n_modes,a,b the softest modes used, and the logistic's slope
               and offset (defaults fitted on seven proteins)
    \return a probability per pair; NaN for an invalid pair
*/
IMPBFFEXPORT std::vector<double> get_pair_change_probabilities(
        const ElasticNetworkModes& modes, int* pairs, int n_pair_rows, int n_pair_cols,
        int n_modes = 2, double a = 0.666, double b = 0.295);

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_ELASTICNETWORK_H */
