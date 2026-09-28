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
