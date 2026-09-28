/**
 * \file NPS.cpp
 * \brief Fast-NPS direct dye functions and dye-model metadata.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/NPS.h>

#include <IMP/bff/Distributions.h>
#include <IMP/bff/FRETOrientationFactor.h>

#include <IMP/algebra/Vector3D.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

IMPBFF_BEGIN_NAMESPACE

namespace {

struct UnitVector {
    double x, y, z;
};

void validate_direct_dye(const NPSDirectDye& dye, const char* api_name,
                         const char* dye_name) {
    if (!std::isfinite(dye.x) || !std::isfinite(dye.y) ||
        !std::isfinite(dye.z) || !std::isfinite(dye.m) ||
        !std::isfinite(dye.phi) ||
        !std::isfinite(dye.steady_state_anisotropy)) {
        IMP_THROW(api_name << ": " << dye_name << " fields must be finite",
                  ValueException);
    }
    if (dye.m < -1.0 || dye.m > 1.0) {
        IMP_THROW(api_name << ": " << dye_name << ".m must be in [-1, 1]",
                  ValueException);
    }
    if (dye.steady_state_anisotropy < 0.0 ||
        dye.steady_state_anisotropy > 0.4) {
        IMP_THROW(api_name << ": " << dye_name << ".steady_state_anisotropy"
                           << " must be in [0, 0.4]",
                  ValueException);
    }
}

UnitVector get_direction(const NPSDirectDye& dye) {
    const double c = -dye.m;
    const double s = std::sqrt(1.0 - c * c);
    UnitVector direction = {
            s * std::cos(dye.phi), s * std::sin(dye.phi), c};
    return direction;
}

double clamp_dot_product(double value) {
    return std::max(-1.0, std::min(1.0, value));
}

}  // namespace

NPSDyeModel nps_dye_model(int index) {
    NPSDyeModel model;
    model.index = index;
    switch (index) {
        case 1:
            return model;
        case 2:
            model.isotropic = true;
            return model;
        case 3:
            model.fixed_mean_position = true;
            model.isotropic = true;
            model.distance_convolved = true;
            return model;
        case 4:
            model.isotropic = true;
            model.distance_convolved = true;
            return model;
        case 5:
            model.distance_convolved = true;
            return model;
        default:
            IMP_THROW("nps_dye_model: index must be between 1 and 5, not "
                              << index,
                      ValueException);
    }
}

double nps_direct_fret_efficiency(
        const NPSDirectDye& dye1, const NPSDirectDye& dye2, double r_iso) {
    validate_direct_dye(dye1, "nps_direct_fret_efficiency", "dye1");
    validate_direct_dye(dye2, "nps_direct_fret_efficiency", "dye2");
    if (!std::isfinite(r_iso) || r_iso <= 0.0) {
        IMP_THROW("nps_direct_fret_efficiency: r_iso must be finite and "
                          "strictly positive",
                  ValueException);
    }

    const double dx = dye2.x - dye1.x;
    const double dy = dye2.y - dye1.y;
    const double dz = dye2.z - dye1.z;
    const double distance = std::hypot(std::hypot(dx, dy), dz);
    if (!std::isfinite(distance) || distance <= 0.0) {
        IMP_THROW("nps_direct_fret_efficiency: dye separation must be finite "
                          "and strictly positive",
                  ValueException);
    }
    if (distance > 150.0) return 0.0;

    const double q1 = std::sqrt(dye1.steady_state_anisotropy / 0.4);
    const double q2 = std::sqrt(dye2.steady_state_anisotropy / 0.4);
    const UnitVector u1 = get_direction(dye1);
    const UnitVector u2 = get_direction(dye2);
    const double rhat_x = dx / distance;
    const double rhat_y = dy / distance;
    const double rhat_z = dz / distance;
    const double c_t = u1.x * u2.x + u1.y * u2.y + u1.z * u2.z;
    const double c1 = rhat_x * u1.x + rhat_y * u1.y + rhat_z * u1.z;
    const double c2 = rhat_x * u2.x + rhat_y * u2.y + rhat_z * u2.z;
    const double kappa2 = wobbling_kappa2(
            std::acos(clamp_dot_product(c_t)), q1, q2,
            std::acos(clamp_dot_product(c1)),
            std::acos(clamp_dot_product(c2)));
    if (!std::isfinite(kappa2)) {
        IMP_THROW("nps_direct_fret_efficiency: orientation factor must be finite",
                  ValueException);
    }
    if (kappa2 <= 0.0) return 0.0;

    // Calculate the ratio first to avoid separately evaluating distance^6 and
    // r_iso^6; this does not prevent every possible overflow.
    const double distance_ratio = distance / r_iso;
    const double distance_ratio6 = std::pow(distance_ratio, 6.0);
    const double kappa_term = 1.5 * kappa2;
    return kappa_term / (kappa_term + distance_ratio6);
}

double nps_direct_transfer_anisotropy(const NPSDirectDye& dye1,
                                      const NPSDirectDye& dye2) {
    validate_direct_dye(dye1, "nps_direct_transfer_anisotropy", "dye1");
    validate_direct_dye(dye2, "nps_direct_transfer_anisotropy", "dye2");

    const double q1 = std::sqrt(dye1.steady_state_anisotropy / 0.4);
    const double q2 = std::sqrt(dye2.steady_state_anisotropy / 0.4);
    const UnitVector u1 = get_direction(dye1);
    const UnitVector u2 = get_direction(dye2);
    const double c_t = clamp_dot_product(
            u1.x * u2.x + u1.y * u2.y + u1.z * u2.z);
    return q1 * q2 * (3.0 * c_t * c_t - 1.0) / 5.0;
}

namespace {

//! Isotropic direct efficiency in log space for finite 0 < r <= 150 Angstrom.
/*!
    Equivalent to the isotropic branch of nps_direct_fret_efficiency() for
    ordinary finite inputs -- there q1 = q2 = 0, so the orientation factor is
    the isotropic 2/3 and E = 1/(1 + (r/R_iso)^6) -- but never materializes
    exp(eta_R), so it stays finite for any finite eta_R. Both logistic
    branches evaluate exp() on a nonpositive argument only.
*/
double logspace_isotropic_direct_efficiency(double log_r, double eta_r) {
    const double z = 6.0 * (log_r - eta_r);
    if (z >= 0.0) {
        // r well beyond R_iso: exp(-z) can only underflow to 0, never
        // overflow, so the efficiency degrades gracefully toward 0.
        const double e = std::exp(-z);
        return e / (1.0 + e);
    }
    // r well inside R_iso: writing 1/(1+exp(-z)) in this equivalent form
    // keeps the exponent negative on this branch too, so it approaches 1
    // without overflow for any finite eta_R.
    const double e = std::exp(z);
    return 1.0 / (1.0 + e);
}

bool are_finite(const IMP::algebra::Vector3D& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

}  // namespace

NPSIsotropicFRETEfficiencyRestraint::NPSIsotropicFRETEfficiencyRestraint(
        IMP::Model* m, IMP::ParticleIndexAdaptor donor,
        IMP::ParticleIndexAdaptor acceptor, IMP::ParticleIndexAdaptor bias,
        IMP::ParticleIndexAdaptor log_r_iso, double observed_efficiency,
        double sigma, std::string name)
    : IMP::Restraint(m, name),
      donor_(static_cast<IMP::ParticleIndex>(donor)),
      acceptor_(static_cast<IMP::ParticleIndex>(acceptor)),
      bias_(static_cast<IMP::ParticleIndex>(bias)),
      log_r_iso_(static_cast<IMP::ParticleIndex>(log_r_iso)),
      observed_efficiency_(observed_efficiency),
      sigma_(sigma) {
    if (!IMP::core::XYZ::get_is_setup(m, donor_)) {
        IMP_THROW("NPSIsotropicFRETEfficiencyRestraint: donor particle "
                  "must have IMP::core::XYZ",
                  ValueException);
    }
    if (!IMP::core::XYZ::get_is_setup(m, acceptor_)) {
        IMP_THROW("NPSIsotropicFRETEfficiencyRestraint: acceptor particle "
                  "must have IMP::core::XYZ",
                  ValueException);
    }
    if (!IMP::isd::Nuisance::get_is_setup(m, bias_)) {
        IMP_THROW("NPSIsotropicFRETEfficiencyRestraint: bias particle "
                  "must have IMP::isd::Nuisance",
                  ValueException);
    }
    if (!IMP::isd::Nuisance::get_is_setup(m, log_r_iso_)) {
        IMP_THROW("NPSIsotropicFRETEfficiencyRestraint: log_r_iso particle "
                  "must have IMP::isd::Nuisance",
                  ValueException);
    }
    if (!std::isfinite(observed_efficiency)) {
        IMP_THROW("NPSIsotropicFRETEfficiencyRestraint: observed efficiency "
                  "must be finite",
                  ValueException);
    }
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        IMP_THROW("NPSIsotropicFRETEfficiencyRestraint: sigma must be "
                  "finite and strictly positive",
                  ValueException);
    }
}

double NPSIsotropicFRETEfficiencyRestraint::get_model_efficiency() const {
    IMP::Model* m = get_model();
    const IMP::algebra::Vector3D c1 =
            IMP::core::XYZ(m, donor_).get_coordinates();
    const IMP::algebra::Vector3D c2 =
            IMP::core::XYZ(m, acceptor_).get_coordinates();
    if (!are_finite(c1) || !are_finite(c2)) {
        return std::numeric_limits<double>::infinity();
    }
    const double dx = c2[0] - c1[0];
    const double dy = c2[1] - c1[1];
    const double dz = c2[2] - c1[2];
    const double distance = std::hypot(std::hypot(dx, dy), dz);
    // isfinite() rejects NaN and +/-inf alike; hypot overflows to +inf for
    // finite endpoints too far apart to score, which maps to +inf below.
    if (!std::isfinite(distance)) {
        return std::numeric_limits<double>::infinity();
    }
    if (distance == 0.0) {
        // Isotropic r -> 0 limit; the oriented public leaf rejects this state.
        return 1.0;
    }
    if (distance > 150.0) return 0.0;
    const double eta_r = IMP::isd::Nuisance(m, log_r_iso_).get_nuisance();
    if (!std::isfinite(eta_r)) {
        return std::numeric_limits<double>::infinity();
    }
    return logspace_isotropic_direct_efficiency(std::log(distance), eta_r);
}

double NPSIsotropicFRETEfficiencyRestraint::get_observation_mean() const {
    const double efficiency = get_model_efficiency();
    if (!std::isfinite(efficiency)) {
        return std::numeric_limits<double>::infinity();
    }
    const double bias = IMP::isd::Nuisance(get_model(), bias_).get_nuisance();
    if (!std::isfinite(bias)) {
        return std::numeric_limits<double>::infinity();
    }
    const double mean = efficiency + bias;
    if (!std::isfinite(mean)) {
        return std::numeric_limits<double>::infinity();
    }
    return mean;
}

double NPSIsotropicFRETEfficiencyRestraint::unprotected_evaluate(
        IMP::DerivativeAccumulator* accum) const {
    IMP_USAGE_CHECK(accum == nullptr,
                    "NPSIsotropicFRETEfficiencyRestraint does not provide "
                    "derivatives; it is Monte-Carlo-only");
    // Totality boundary: every proposal-reachable model state must have a
    // score, so an IMP MonteCarlo proposal can always be scored, rejected,
    // and rolled back. Unscorable live state maps to +inf, never a throw.
    const double mean = get_observation_mean();
    if (!std::isfinite(mean)) {
        return std::numeric_limits<double>::infinity();
    }
    const double score =
            -normal_log_density(observed_efficiency_, mean, sigma_);
    // A finite extreme residual underflows the density to a literal -inf
    // log score (defined behavior of normal_log_density); only NaN --
    // which would indicate an arithmetic contract break -- maps to +inf.
    if (std::isnan(score)) {
        return std::numeric_limits<double>::infinity();
    }
    return score;
}

IMP::ModelObjectsTemp NPSIsotropicFRETEfficiencyRestraint::do_get_inputs()
        const {
    IMP::Model* m = get_model();
    IMP::ModelObjectsTemp out;
    out.push_back(m->get_particle(donor_));
    out.push_back(m->get_particle(acceptor_));
    out.push_back(m->get_particle(bias_));
    out.push_back(m->get_particle(log_r_iso_));
    return out;
}

IMPBFF_END_NAMESPACE
