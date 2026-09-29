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

namespace {

//! One Fast-NPS configuration row: [x, y, z, m, phi].
struct ConfigRow {
    double x, y, z, m, phi;
};

ConfigRow get_config_row(const std::vector<std::vector<double>>& config,
                         int index, const char* api_name) {
    if (index < 0 || static_cast<std::size_t>(index) >= config.size()) {
        IMP_THROW(api_name << ": dye index " << index << " leaves the "
                          << config.size() << "-row configuration",
                  ValueException);
    }
    const std::vector<double>& row = config[index];
    if (row.size() != 5) {
        IMP_THROW(api_name << ": configuration row " << index
                          << " must hold [x, y, z, m, phi], not "
                          << row.size() << " values",
                  ValueException);
    }
    for (int k = 0; k < 5; ++k) {
        if (!std::isfinite(row[k])) {
            IMP_THROW(api_name << ": configuration row " << index
                              << " value " << k << " must be finite",
                      ValueException);
        }
    }
    if (row[3] < -1.0 || row[3] > 1.0) {
        IMP_THROW(api_name << ": configuration row " << index
                          << " m must be in [-1, 1]",
                  ValueException);
    }
    ConfigRow out;
    out.x = row[0];
    out.y = row[1];
    out.z = row[2];
    out.m = row[3];
    out.phi = row[4];
    return out;
}

void validate_network_dye(const NPSNetworkDye& dye, int index,
                          const char* api_name) {
    if (!std::isfinite(dye.dep) || dye.dep < 0.0 || dye.dep > 1.0) {
        IMP_THROW(api_name << ": dye " << index
                          << " dep must be finite in [0, 1]",
                  ValueException);
    }
}

//! NPS transition-dipole unit vector from one configuration row, the
//! same get_direction() construction the direct leaves use.
UnitVector get_row_direction(const ConfigRow& row) {
    const double c = -row.m;
    const double s = std::sqrt(std::max(0.0, 1.0 - c * c));
    UnitVector direction = {
            s * std::cos(row.phi), s * std::sin(row.phi), c};
    return direction;
}

//! Azimuth slot of the Fast-NPS orientation grids: floor(scale * phi)
//! truncated toward zero as the reference's int cast does, clamped into
//! [0, 5) so azimuths outside [0, pi) cannot leave the table.
int azimuth_slot(double phi, double scale) {
    const double scaled = scale * phi;
    if (scaled >= 5.0) return 4;
    if (scaled <= -1.0) return 0;
    return static_cast<int>(scaled);  // (-1, 5) truncates into [0, 4]
}

//! Polar slot of the Fast-NPS orientation grids; m in [-1, 1] maps to
//! [0, 5) with the same clamp the azimuth slot applies at the edge.
int polar_slot_of(double m, double scale) {
    const double scaled = scale * (-m + 1.0);
    if (scaled >= 5.0) return 4;
    return static_cast<int>(scaled);
}

}  // namespace

int nps_orientation_row_index(double m1, double phi1,
                              double m2, double phi2) {
    if (!std::isfinite(m1) || !std::isfinite(m2) || m1 < -1.0 || m1 > 1.0
            || m2 < -1.0 || m2 > 1.0) {
        IMP_THROW("nps_orientation_row_index: m must be finite in [-1, 1]",
                  ValueException);
    }
    if (!std::isfinite(phi1) || !std::isfinite(phi2)) {
        IMP_THROW("nps_orientation_row_index: phi must be finite",
                  ValueException);
    }
    // Fast-NPS uses the literal 3.142, not pi (logLikelihood.cpp line 175).
    const double azimuth_scale = 5.0 / 3.142;
    const double polar_scale = 2.5;
    // Clamp into [0, 5): phi = pi lands at 4.9993 in the reference (int
    // 4), larger or negative azimuths would leave the table, where the
    // reference indexes out of bounds.
    const int phi_slot1 = azimuth_slot(phi1, azimuth_scale);
    const int polar_slot1 = polar_slot_of(m1, polar_scale);
    const int phi_slot2 = azimuth_slot(phi2, azimuth_scale);
    const int polar_slot2 = polar_slot_of(m2, polar_scale);
    return 125 * phi_slot1 + 25 * polar_slot1 + 5 * phi_slot2 + polar_slot2;
}

int nps_single_orientation_row_index(double m, double phi) {
    if (!std::isfinite(m) || m < -1.0 || m > 1.0) {
        IMP_THROW("nps_single_orientation_row_index: m must be finite "
                          "in [-1, 1]",
                  ValueException);
    }
    if (!std::isfinite(phi)) {
        IMP_THROW("nps_single_orientation_row_index: phi must be finite",
                  ValueException);
    }
    const double azimuth_scale = 5.0 / 3.142;
    const int phi_slot = azimuth_slot(phi, azimuth_scale);
    const int polar_slot = polar_slot_of(m, 2.5);
    return 5 * phi_slot + polar_slot;
}

double nps_convolved_efficiency(
        double distance, const std::vector<double>& coefficients) {
    if (!std::isfinite(distance)) {
        IMP_THROW("nps_convolved_efficiency: distance must be finite",
                  ValueException);
    }
    if (coefficients.size() != 12) {
        IMP_THROW("nps_convolved_efficiency: expected 12 coefficients, got "
                          << coefficients.size(),
                  ValueException);
    }
    for (int k = 0; k < 12; ++k) {
        if (!std::isfinite(coefficients[k])) {
            IMP_THROW("nps_convolved_efficiency: coefficient " << k
                              << " must be finite",
                      ValueException);
        }
    }
    if (distance > 150.0) return 0.0;
    // Horner: the single evaluation of the polynomial Fast-NPS repeats
    // five times across its regimes.
    double value = 0.0;
    for (int k = 11; k >= 0; --k) {
        value = value * distance + coefficients[k];
    }
    return value;
}

double nps_network_fret_efficiency(
        const std::vector<std::vector<double>>& config,
        const NPSNetworkDyes& dyes,
        int dye1, int dye2, double r_iso6,
        const std::vector<std::vector<double>>& eff_conv_coeff) {
    const char* api_name = "nps_network_fret_efficiency";
    if (dye1 < 0 || static_cast<std::size_t>(dye1) >= dyes.size()
            || dye2 < 0 || static_cast<std::size_t>(dye2) >= dyes.size()) {
        IMP_THROW(api_name << ": dye index leaves the " << dyes.size()
                          << "-dye array",
                  ValueException);
    }
    if (!std::isfinite(r_iso6) || r_iso6 <= 0.0) {
        IMP_THROW(api_name << ": r_iso6 must be finite and strictly positive",
                  ValueException);
    }
    validate_network_dye(dyes[dye1], dye1, api_name);
    validate_network_dye(dyes[dye2], dye2, api_name);
    const ConfigRow row1 =
            get_config_row(config, dye1, api_name);
    const ConfigRow row2 =
            get_config_row(config, dye2, api_name);

    const double dx = row2.x - row1.x;
    const double dy = row2.y - row1.y;
    const double dz = row2.z - row1.z;
    const double distance = std::hypot(std::hypot(dx, dy), dz);
    if (!std::isfinite(distance)) {
        IMP_THROW(api_name << ": dye separation must be finite",
                  ValueException);
    }
    if (distance > 150.0) return 0.0;

    const NPSNetworkDye& meta1 = dyes[dye1];
    const NPSNetworkDye& meta2 = dyes[dye2];
    if (!meta1.dist_conv && !meta2.dist_conv) {
        // Direct branch: the committed Dale-Eisinger/wobbling route,
        // pinned equal to Fast-NPS's dep-factor closed form. Rebuilt as
        // NPSDirectDye so no orientation formula is written twice.
        if (distance <= 0.0) {
            IMP_THROW(api_name << ": coincident dyes have no defined "
                              "direct separation",
                      ValueException);
        }
        NPSDirectDye direct1;
        direct1.x = row1.x;
        direct1.y = row1.y;
        direct1.z = row1.z;
        direct1.m = row1.m;
        direct1.phi = row1.phi;
        // dep = sqrt(ravg / 0.4) inverts to ravg = 0.4 dep^2 exactly the
        // relation makeSetting.cpp:278 derives dep from.
        direct1.steady_state_anisotropy = 0.4 * meta1.dep * meta1.dep;
        NPSDirectDye direct2;
        direct2.x = row2.x;
        direct2.y = row2.y;
        direct2.z = row2.z;
        direct2.m = row2.m;
        direct2.phi = row2.phi;
        direct2.steady_state_anisotropy = 0.4 * meta2.dep * meta2.dep;
        const double r_iso = std::pow(r_iso6, 1.0 / 6.0);
        return nps_direct_fret_efficiency(direct1, direct2, r_iso);
    }

    // Distance-convolved branches: pick the coefficient row by the same
    // regime rules as logLikelihood.cpp lines 168-190.
    std::size_t row_index = 0;
    if (meta1.iso && meta2.iso) {
        // iso/iso pair: single shared row.
        if (eff_conv_coeff.empty()) {
            IMP_THROW(api_name << ": iso/iso pair needs coefficient "
                              << "row 0, table is empty",
                      ValueException);
        }
    } else if (!meta1.iso && !meta2.iso) {
        // Non-iso pair: the 625-row orientation grid.
        row_index = static_cast<std::size_t>(nps_orientation_row_index(
                row1.m, row1.phi, row2.m, row2.phi));
        if (row_index >= eff_conv_coeff.size()) {
            IMP_THROW(api_name << ": coefficient table holds "
                              << eff_conv_coeff.size() << " rows, pair "
                                 "grid needs row " << row_index,
                      ValueException);
        }
    } else {
        // Mixed pair: the 25-row single-dye grid over the non-iso dye.
        const ConfigRow& non_iso_row = meta1.iso ? row2 : row1;
        row_index = static_cast<std::size_t>(
                nps_single_orientation_row_index(non_iso_row.m,
                                                 non_iso_row.phi));
        if (row_index >= eff_conv_coeff.size()) {
            IMP_THROW(api_name << ": coefficient table holds "
                              << eff_conv_coeff.size() << " rows, single "
                                 "grid needs row " << row_index,
                      ValueException);
        }
    }
    return nps_convolved_efficiency(distance, eff_conv_coeff[row_index]);
}

double nps_network_transfer_anisotropy(
        const std::vector<std::vector<double>>& config,
        const NPSNetworkDyes& dyes, int dye1, int dye2) {
    const char* api_name = "nps_network_transfer_anisotropy";
    if (dye1 < 0 || static_cast<std::size_t>(dye1) >= dyes.size()
            || dye2 < 0 || static_cast<std::size_t>(dye2) >= dyes.size()) {
        IMP_THROW(api_name << ": dye index leaves the " << dyes.size()
                          << "-dye array",
                  ValueException);
    }
    validate_network_dye(dyes[dye1], dye1, api_name);
    validate_network_dye(dyes[dye2], dye2, api_name);
    const ConfigRow row1 = get_config_row(config, dye1, api_name);
    const ConfigRow row2 = get_config_row(config, dye2, api_name);
    NPSDirectDye direct1;
    direct1.x = row1.x;
    direct1.y = row1.y;
    direct1.z = row1.z;
    direct1.m = row1.m;
    direct1.phi = row1.phi;
    direct1.steady_state_anisotropy = 0.4 * dyes[dye1].dep * dyes[dye1].dep;
    NPSDirectDye direct2;
    direct2.x = row2.x;
    direct2.y = row2.y;
    direct2.z = row2.z;
    direct2.m = row2.m;
    direct2.phi = row2.phi;
    direct2.steady_state_anisotropy = 0.4 * dyes[dye2].dep * dyes[dye2].dep;
    return nps_direct_transfer_anisotropy(direct1, direct2);
}

double nps_network_log_likelihood(
        const std::vector<std::vector<double>>& config,
        const NPSNetworkDyes& dyes,
        const NPSMeasurements& measurements) {
    const char* api_name = "nps_network_log_likelihood";
    double log_likelihood = 0.0;
    for (std::size_t m_index = 0; m_index < measurements.size();
            ++m_index) {
        const NPSMeasurement& meas = measurements[m_index];
        const int index1 = meas.dye1;
        const int index2 = meas.dye2;
        if (meas.fret_active) {
            if (!std::isfinite(meas.e_avg) || !std::isfinite(meas.e_err)
                    || meas.e_err <= 0.0) {
                IMP_THROW(api_name << ": measurement " << m_index
                                  << " FRET data must be finite with a "
                                     "positive error",
                          ValueException);
            }
            if (!std::isfinite(meas.r_iso6) || meas.r_iso6 <= 0.0) {
                IMP_THROW(api_name << ": measurement " << m_index
                                  << " r_iso6 must be finite and "
                                     "strictly positive",
                          ValueException);
            }
            const double efficiency = nps_network_fret_efficiency(
                    config, dyes, index1, index2, meas.r_iso6,
                    meas.eff_conv_coeff);
            log_likelihood += normal_log_density(
                    meas.e_avg, efficiency, meas.e_err);
        }
        if (meas.ta_active) {
            if (!std::isfinite(meas.r_t_avg)
                    || !std::isfinite(meas.r_t_err)
                    || meas.r_t_err <= 0.0) {
                IMP_THROW(api_name << ": measurement " << m_index
                                  << " TA data must be finite with a "
                                     "positive error",
                          ValueException);
            }
            const double ta = nps_network_transfer_anisotropy(
                    config, dyes, index1, index2);
            log_likelihood += normal_log_density(
                    meas.r_t_avg, ta, meas.r_t_err);
        }
    }
    return log_likelihood;
}

IMPBFF_END_NAMESPACE
