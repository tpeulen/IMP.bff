/**
 * \file IMP/bff/NPS.h
 * \brief Fast-NPS direct dye and dye-model metadata.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_NPS_H
#define IMPBFF_NPS_H

#include <IMP/bff/bff_config.h>

#include <IMP/bff/IMPCompatibility.h>

#include <IMP/Restraint.h>
#include <IMP/core/XYZ.h>
#include <IMP/isd/Nuisance.h>
#include <IMP/particle_index.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! One direct Fast-NPS dye state.
struct IMPBFFEXPORT NPSDirectDye {
    //! Cartesian dye mean position, Å.
    double x, y, z;
    //! NPS convention: \f$m=-\cos(\theta)\f$.
    double m;
    //! Azimuth in radians.
    double phi;
    //! Steady-state anisotropy \f$r_{\rm avg}\f$.
    double steady_state_anisotropy;

    NPSDirectDye()
        : x(0.0), y(0.0), z(0.0), m(0.0), phi(0.0),
          steady_state_anisotropy(0.0) {}

    IMP_SHOWABLE_INLINE(NPSDirectDye,
                        out << "NPSDirectDye(" << x << ", " << y << ", "
                            << z << ", " << m << ", " << phi << ", "
                            << steady_state_anisotropy << ")");
};
IMP_VALUES(NPSDirectDye, NPSDirectDyes);

//! The metadata selected by one Fast-NPS dye-model index.
struct IMPBFFEXPORT NPSDyeModel {
    int index;
    //! Whether the model fixes the dye mean position.
    bool fixed_mean_position;
    //! Whether the model assumes isotropic dye orientation.
    bool isotropic;
    //! Whether the model evaluates a distance-convolved FRET observable.
    bool distance_convolved;

    NPSDyeModel()
        : index(0), fixed_mean_position(false), isotropic(false),
          distance_convolved(false) {}

    IMP_SHOWABLE_INLINE(NPSDyeModel, out << "NPSDyeModel(" << index << ")");
};
IMP_VALUES(NPSDyeModel, NPSDyeModels);

//! Resolve one of Fast-NPS's five dye-model selections.
IMPBFFEXPORT NPSDyeModel nps_dye_model(int index);

//! Predict direct non-distance-convolved FRET efficiency for one dye pair.
//! \param r_iso Isotropic Förster radius, in Å, matching coordinate units.
IMPBFFEXPORT double nps_direct_fret_efficiency(
        const NPSDirectDye& dye1, const NPSDirectDye& dye2, double r_iso);

//! Predict direct transfer anisotropy for one dye pair.
//! Dye coordinates are validated as finite but do not affect this observable.
IMPBFFEXPORT double nps_direct_transfer_anisotropy(
        const NPSDirectDye& dye1, const NPSDirectDye& dye2);

//! Per-dye network metadata, Fast-NPS's ``Dye`` anisotropy/model flags.
struct IMPBFFEXPORT NPSNetworkDye {
    //! Absolute average axial depolarization factor
    //! \f$\sqrt{r_{\rm avg}/0.4}\f$; Fast-NPS derives it from the
    //! steady-state anisotropy exactly this way (makeSetting.cpp:278).
    double dep;
    //! Whether the dye is isotropically averaged.
    bool iso;
    //! Whether the dye is averaged over several positions, selecting the
    //! distance-convolved (polynomial) efficiency branch.
    bool dist_conv;

    NPSNetworkDye() : dep(0.0), iso(false), dist_conv(false) {}

    IMP_SHOWABLE_INLINE(NPSNetworkDye,
                        out << "NPSNetworkDye(" << dep << ", " << iso
                            << ", " << dist_conv << ")");
};
IMP_VALUES(NPSNetworkDye, NPSNetworkDyes);

//! One Fast-NPS measurement: dye pair, active observables, and data.
struct IMPBFFEXPORT NPSMeasurement {
    int dye1;
    int dye2;
    //! Whether the FRET efficiency was measured for this pair.
    bool fret_active;
    //! Average observed FRET efficiency.
    double e_avg;
    //! Standard error of the observed FRET efficiency.
    double e_err;
    //! Isotropic Förster radius to the sixth power, in coordinate
    //! units to the sixth.
    double r_iso6;
    //! Whether the transfer anisotropy was measured for this pair.
    bool ta_active;
    //! Average observed transfer anisotropy.
    double r_t_avg;
    //! Standard error of the observed transfer anisotropy.
    double r_t_err;
    //! Distance-convolution coefficient rows (12 coefficients each). Which
    //! row a measurement uses follows the dye models: row 0 for an
    //! iso/iso pair, the 625-entry pair orientation grid for a
    //! non-iso/non-iso pair, the 25-entry single-dye grid for a mixed
    //! pair, indexed by nps_orientation_row_index() and
    //! nps_single_orientation_row_index().
    std::vector<std::vector<double>> eff_conv_coeff;

    NPSMeasurement()
        : dye1(0), dye2(0), fret_active(false), e_avg(0.0), e_err(1.0),
          r_iso6(0.0), ta_active(false), r_t_avg(0.0), r_t_err(1.0) {}

    //! Set the coefficient rows from a sequence of numeric rows.
    /*! A method, not the raw member, so Python lists and numpy arrays
        convert through the shared row typemap. */
    void set_eff_conv_coeff(
            const std::vector<std::vector<double>>& rows) {
        eff_conv_coeff = rows;
    }

    IMP_SHOWABLE_INLINE(NPSMeasurement,
                        out << "NPSMeasurement(" << dye1 << ", " << dye2
                            << ")");
};
IMP_VALUES(NPSMeasurement, NPSMeasurements);

//! One Fast-NPS configuration row is \f$[x, y, z, m, \varphi]\f$ with the
//! NPS convention \f$m=-\cos\theta\f$; dye indices address these rows.

//! Fast-NPS's two-dye orientation-grid row for distance-convolved
//! non-iso pairs (logLikelihood.cpp line 175): a 625-entry table indexed
//! by two 5x5 grids over azimuth and polar angle. Slots that would leave
//! the table (azimuth at \f$\pi\f$ with the reference's 3.142 constant,
//! polar angle at the poles) clamp to the edge, where the reference
//! indexes out of bounds.
//! \throws IMP::ValueException when an angle is non-finite or \f$m\f$
//!     leaves [-1, 1].
IMPBFFEXPORT int nps_orientation_row_index(
        double m1, double phi1, double m2, double phi2);

//! Fast-NPS's single-dye orientation-grid row for mixed iso/non-iso
//! pairs (logLikelihood.cpp line 186): a 25-entry table, clamped as in
//! nps_orientation_row_index().
IMPBFFEXPORT int nps_single_orientation_row_index(double m, double phi);

//! Build a distance-convolved efficiency table from two label clouds.
/*!
    The concept is Fast-NPS's modelMCSimulation: for each orientation slot of
    the table layout below, average the transfer efficiency over the two
    labels' position distributions at each mean separation 1..150 A, then
    summarize each 150-point curve by 12 power-basis coefficients — the rows
    nps_network_fret_efficiency() consumes. The implementation differs from
    the original on purpose:

    - the position average is taken over the caller's clouds (an accessible
      volume, a rotamer library, anything that is a States cloud), not a
      rejection-sampled sphere of hardcoded 20 A radius;
    - pair draws are shared across all distances and slots (sample once,
      reuse), instead of re-sampling per bin;
    - the fit solves the least-squares system in the scaled variable
      u = dist / 150 (condition number ~1e3, not the raw-power Vandermonde's
      ~1e20) and rescales the solved coefficients back to the power basis,
      so the table layout and the evaluator's Horner evaluation stay
      exactly as committed;
    - the orientation cosine uses the sampled separation, not the mean one
      (the reference divides by the mean distance, biasing cTh).

    dep arguments are the dyes' depolarization factors q; the non-iso slots
    integrate the wobbling-in-cone orientation factor (2 - q)/3 + q cTh^2
    with cTh from the sampled geometry, matching the committed direct branch.

    \param[in] cloud1, cloud2 flat (n, 4) clouds — x, y, z, weight per point
    \param[in] dep1, dep2 depolarization factors; dep = 1 marks an isotropic
               dye (one table row), dep < 1 puts the dye on the orientation
               grid — iso/iso tables pass (1, 1), mixed (q, 1) or (1, q),
               and the pair grid (q1, q2).
    \param[in] r_iso6 the isotropic Forster distance to the sixth power, A^6
    \param[in] n_samples pair draws per slot evaluation (shared across the
               table); 0 selects the default (60000)
    \param[in] seed RNG seed for the shared pair draws; a fixed seed makes
               tables reproducible
    \return the coefficient table: 1, 25 or 625 rows of 12 coefficients,
            indexed by the same layout functions the evaluator uses
*/
IMPBFFEXPORT std::vector<std::vector<double> > nps_convolved_efficiency_table(
        const std::vector<double>& cloud1,
        const std::vector<double>& cloud2,
        double dep1, double dep2, double r_iso6,
        int n_samples = 0, int seed = 0);

//! Evaluate one Fast-NPS distance-convolution polynomial at a separation.
/*! Horner evaluation of the 12 coefficients
    \f$E(d)=\sum_{k=0}^{11} c_k d^k\f$, exactly the polynomial Fast-NPS
    repeats five times across its regimes. Separations beyond the 150
    \f$\text{\AA}\f$ compatibility cutoff return exactly 0.
    \throws IMP::ValueException when the distance is non-finite, or the
        row does not hold exactly 12 finite coefficients. */
IMPBFFEXPORT double nps_convolved_efficiency(
        double distance, const std::vector<double>& coefficients);

//! Network-oriented FRET efficiency for one dye pair of a configuration.
/*! Dynamic-regime forward model of Fast-NPS's logLikelihood.cpp: the
    non-distance-convolved branch evaluates the committed
    wobbling_kappa2() orientation factor on the dye basis vectors (pinned
    algebraically identical to Fast-NPS's dep-factor closed form), the
    distance-convolved branches evaluate nps_convolved_efficiency() on
    the row the dye models select. Separations beyond the 150
    \f$\text{\AA}\f$ cutoff return exactly 0.
    \param[in] config rows \f$[x, y, z, m, \varphi]\f$, one per dye
    \param[in] dyes per-dye metadata, addressed by the same indices
    \param[in] dye1,dye2 indices into both arrays
    \param[in] r_iso6 isotropic Förster radius to the sixth power
    \param[in] eff_conv_coeff coefficient table for the convolved
        branches; unused (may be empty) for the direct branch
    \throws IMP::ValueException on any invalid dye state, index, radius,
        or a convolved regime without a usable coefficient row. */
IMPBFFEXPORT double nps_network_fret_efficiency(
        const std::vector<std::vector<double>>& config,
        const NPSNetworkDyes& dyes,
        int dye1, int dye2, double r_iso6,
        const std::vector<std::vector<double>>& eff_conv_coeff
            = std::vector<std::vector<double>>());

//! Network transfer anisotropy for one dye pair of a configuration.
/*! \f$\hat r_T=(\delta_1\delta_2/5)(3\cos^2\theta_{12}-1)\f$ via the
    committed nps_direct_transfer_anisotropy() leaf (dep equals its
    \f$q=\sqrt{r_{\rm avg}/0.4}\f$ by definition), so no formula is
    duplicated.
    \throws IMP::ValueException on any invalid dye state or index. */
IMPBFFEXPORT double nps_network_transfer_anisotropy(
        const std::vector<std::vector<double>>& config,
        const NPSNetworkDyes& dyes, int dye1, int dye2);

//! Fast-NPS multi-measurement log likelihood of one dye configuration.
/*! Accumulates the normalized Gaussian log density of every active
    observable over all measurements (logLikelihood.cpp lines 197-228)
    through the committed normal_log_density(); a measurement may carry
    FRET, transfer anisotropy, or both. This is the likelihood core the
    Fast-NPS sampler maximizes; sampling itself stays IMP's.
    \throws IMP::ValueException on any invalid state, index, or error. */
IMPBFFEXPORT double nps_network_log_likelihood(
        const std::vector<std::vector<double>>& config,
        const NPSNetworkDyes& dyes,
        const NPSMeasurements& measurements);

//! Thin Bayesian direct-FRET likelihood on live existing IMP label sites.
/*!
    The `bff_structural_direct` tracer's likelihood bridge. It reads the two
    endpoint `IMP::core::XYZ` coordinates and the two `IMP::isd::Nuisance`
    values (additive efficiency bias \f$b_E\f$ and
    \f$\eta_R=\log(R_{\rm iso}/1\,\text{\AA})\f$) at every score, and owns no
    coordinates, hierarchy, probe/AV state, graph port, prior, or sampler:
    one `IMP::core::MonteCarlo` owns the whole posterior transaction.

    The isotropic direct efficiency is evaluated in log space,
    \f$z=6(\log r-\eta_R)\f$, so it stays finite for any finite \f$\eta_R\f$.
    Finite separations beyond the 150 Å compatibility cutoff give exactly 0;
    a finite coincident pair gives the isotropic limit 1. Unscorable live
    state (non-finite coordinates, nuisances, separation, mean, or score
    arithmetic) returns +infinity rather than throwing, so every
    proposal-reachable model state has a score and a rejected move rolls back
    cleanly.

    \throw IMP::ValueException at construction when an endpoint lacks `XYZ`,
    a nuisance particle lacks `IMP::isd::Nuisance`, the observation is not
    finite, or the scale is not finite and strictly positive.
    \throw IMP::UsageException when derivatives are requested: this restraint
    is Monte-Carlo-only in the first slice.
*/
class IMPBFFEXPORT NPSIsotropicFRETEfficiencyRestraint : public IMP::Restraint {
    IMP::ParticleIndex donor_, acceptor_, bias_, log_r_iso_;
    double observed_efficiency_;
    double sigma_;

public:
    //! \param[in] m the model the four particles belong to
    /*! \param[in] donor,acceptor the two existing label-site particles whose
                   current `XYZ` coordinates are scored
        \param[in] bias,log_r_iso existing `IMP::isd::Nuisance` particles for
                   \f$b_E\f$ and \f$\eta_R=\log(R_{\rm iso}/1\,\text{\AA})\f$
        \param[in] observed_efficiency the calibrated raw \f$E_{\rm obs}\f$
        \param[in] sigma the fixed finite positive \f$\sigma_E\f$
        \param[in] name the restraint's name */
    NPSIsotropicFRETEfficiencyRestraint(
            IMP::Model* m, IMP::ParticleIndexAdaptor donor,
            IMP::ParticleIndexAdaptor acceptor,
            IMP::ParticleIndexAdaptor bias,
            IMP::ParticleIndexAdaptor log_r_iso,
            double observed_efficiency, double sigma,
            std::string name = "NPSIsotropicFRETEfficiencyRestraint%1%");

    //! The current isotropic direct efficiency \f$\hat E_{\rm iso}\f$.
    /*! Returns +infinity when the current live state is unscorable. */
    double get_model_efficiency() const;
    //! \f$\hat E_{\rm iso}+b_E\f$, the current Gaussian observation mean.
    /*! Returns +infinity when the current live state is unscorable. */
    double get_observation_mean() const;

    virtual double unprotected_evaluate(
            IMP::DerivativeAccumulator* accum) const override;
    virtual IMP::ModelObjectsTemp do_get_inputs() const override;

    IMP_OBJECT_METHODS(NPSIsotropicFRETEfficiencyRestraint);
};

IMP_OBJECTS(NPSIsotropicFRETEfficiencyRestraint,
            NPSIsotropicFRETEfficiencyRestraints);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_NPS_H
