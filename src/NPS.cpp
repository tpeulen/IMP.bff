/**
 * \file NPS.cpp
 * \brief Fast-NPS direct dye functions and dye-model metadata.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/NPS.h>

#include <IMP/bff/Distributions.h>
#include <IMP/bff/FRETOrientationFactor.h>
#include <IMP/bff/SamplerDiagnostics.h>
#include <IMP/bff/States.h>
#include <IMP/algebra/Vector3D.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

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

const double kNpsInfinity = std::numeric_limits<double>::infinity();

void check_xyz(const std::vector<double>& v, const char* what) {
    if (v.size() != 3) {
        IMP_THROW(what << " must have three coordinates, got " << v.size(),
                  ValueException);
    }
}

}  // namespace

double nps_isotropic_direct_efficiency(const std::vector<double>& donor,
                                       const std::vector<double>& acceptor,
                                       double log_r_iso) {
    check_xyz(donor, "nps_isotropic_direct_efficiency: donor");
    check_xyz(acceptor, "nps_isotropic_direct_efficiency: acceptor");
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(donor[i]) || !std::isfinite(acceptor[i])) return kNpsInfinity;
    }
    const double distance = std::hypot(std::hypot(acceptor[0] - donor[0],
                                                  acceptor[1] - donor[1]),
                                       acceptor[2] - donor[2]);
    // hypot overflows to +inf for finite positions too far apart to score.
    if (!std::isfinite(distance)) return kNpsInfinity;
    // The isotropic r -> 0 limit; the oriented leaf rejects this state.
    if (distance == 0.0) return 1.0;
    if (distance > 150.0) return 0.0;
    if (!std::isfinite(log_r_iso)) return kNpsInfinity;
    return logspace_isotropic_direct_efficiency(std::log(distance), log_r_iso);
}

double nps_isotropic_direct_score(const std::vector<double>& donor,
                                  const std::vector<double>& acceptor,
                                  double bias, double log_r_iso,
                                  double observed_efficiency, double sigma) {
    if (!std::isfinite(observed_efficiency)) {
        IMP_THROW("nps_isotropic_direct_score: observed efficiency must be finite",
                  ValueException);
    }
    if (!std::isfinite(sigma) || sigma <= 0.0) {
        IMP_THROW("nps_isotropic_direct_score: sigma must be finite and strictly "
                  "positive", ValueException);
    }
    const double efficiency = nps_isotropic_direct_efficiency(donor, acceptor, log_r_iso);
    if (!std::isfinite(efficiency) || !std::isfinite(bias)) return kNpsInfinity;
    const double mean = efficiency + bias;
    if (!std::isfinite(mean)) return kNpsInfinity;
    const double score = -normal_log_density(observed_efficiency, mean, sigma);
    // A finite extreme residual underflows the density to a -inf log, a defined
    // result; only NaN -- an arithmetic contract break -- maps to +inf.
    return std::isnan(score) ? kNpsInfinity : score;
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

double nps_network_chi2(
        const std::vector<std::vector<double>>& config,
        const NPSNetworkDyes& dyes,
        const NPSMeasurements& measurements) {
    const char* api_name = "nps_network_chi2";
    double chi2 = 0.0;
    bool any_active = false;
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
            const double z = (meas.e_avg - efficiency) / meas.e_err;
            chi2 += z * z;
            any_active = true;
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
            const double z = (meas.r_t_avg - ta) / meas.r_t_err;
            chi2 += z * z;
            any_active = true;
        }
    }
    if (!any_active) {
        IMP_THROW(api_name << ": at least one measurement must be active",
                  ValueException);
    }
    return chi2;
}

// ---------------------------------------------------------------------------
// Table generation: the Fast-NPS modelMCSimulation concept on bff states.
// ---------------------------------------------------------------------------

namespace {

//! The estimator's separations: 1..150 A, the grid the tables summarize.
constexpr int kTableBinCount = 150;
//! Power-basis order of every table row (degree-11 polynomial).
constexpr int kTableCoefficients = 12;

//! Least squares for the scaled fit: min |V c - y| with V[i][j] = u_i^j,
//! u = dist / 150. Solved by Householder QR on V itself — the scaled
//! Vandermonde still has cond ~1e8 (12 powers of u), whose square kills
//! the normal equations (cond(Gram) ~1e16, half of double's digits), which
//! is exactly how the reference's raw-power fit degenerates. QR on V loses
//! only cond(V), not its square.
std::vector<double> solve_scaled(const std::vector<double>& u,
                                 const std::vector<double>& y,
                                 int n_coeff) {
    const int n_pts = static_cast<int>(u.size());
    // V: n_pts x n_coeff, augmented with y in the last column.
    std::vector<double> V(
            static_cast<std::size_t>(n_pts) * (n_coeff + 1), 0.0);
    for (int i = 0; i < n_pts; ++i) {
        V[static_cast<std::size_t>(i) * (n_coeff + 1)] =
                y[static_cast<std::size_t>(i)];
        V[static_cast<std::size_t>(i) * (n_coeff + 1) + 1] = 1.0;
        for (int j = 2; j <= n_coeff; ++j) {
            V[static_cast<std::size_t>(i) * (n_coeff + 1) + j] =
                    V[static_cast<std::size_t>(i) * (n_coeff + 1) + j - 1] *
                    u[static_cast<std::size_t>(i)];
        }
    }
    // Householder QR in place over the coefficient columns; y rides along.
    for (int k = 0; k < n_coeff; ++k) {
        double norm = 0.0;
        for (int i = k; i < n_pts; ++i) {
            const double v = V[static_cast<std::size_t>(i) *
                                    (n_coeff + 1) + k + 1];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm == 0.0) {
            IMP_THROW("nps_convolved_efficiency_table: rank-deficient "
                              "design (degenerate sampled curve)",
                      ValueException);
        }
        const double a0 =
                V[static_cast<std::size_t>(k) * (n_coeff + 1) + k + 1];
        const double alpha = a0 >= 0.0 ? -norm : norm;
        // v = x - alpha e1; v[0] = a0 - alpha
        std::vector<double> v(n_pts, 0.0);
        v[static_cast<std::size_t>(k)] = a0 - alpha;
        double vtv = v[static_cast<std::size_t>(k)] *
                     v[static_cast<std::size_t>(k)];
        for (int i = k + 1; i < n_pts; ++i) {
            v[static_cast<std::size_t>(i)] =
                    V[static_cast<std::size_t>(i) * (n_coeff + 1) + k + 1];
            vtv += v[static_cast<std::size_t>(i)] *
                   v[static_cast<std::size_t>(i)];
        }
        if (vtv == 0.0) continue;
        const double beta = 2.0 / vtv;
        for (int c = 0; c <= n_coeff; ++c) {
            double dot = 0.0;
            for (int i = k; i < n_pts; ++i) {
                dot += v[static_cast<std::size_t>(i)] *
                       V[static_cast<std::size_t>(i) * (n_coeff + 1) + c];
            }
            dot *= beta;
            for (int i = k; i < n_pts; ++i) {
                V[static_cast<std::size_t>(i) * (n_coeff + 1) + c] -=
                        dot * v[static_cast<std::size_t>(i)];
            }
        }
    }
    // Back-substitution on the leading n_coeff x n_coeff of R.
    std::vector<double> scaled(static_cast<std::size_t>(n_coeff), 0.0);
    for (int k = n_coeff - 1; k >= 0; --k) {
        double acc = V[static_cast<std::size_t>(k) * (n_coeff + 1)];
        for (int c = k + 1; c < n_coeff; ++c) {
            acc -= V[static_cast<std::size_t>(k) * (n_coeff + 1) + c + 1] *
                   scaled[static_cast<std::size_t>(c)];
        }
        const double pivot =
                V[static_cast<std::size_t>(k) * (n_coeff + 1) + k + 1];
        if (pivot == 0.0) {
            IMP_THROW("nps_convolved_efficiency_table: rank-deficient "
                              "design (degenerate sampled curve)",
                      ValueException);
        }
        scaled[static_cast<std::size_t>(k)] = acc / pivot;
    }
    return scaled;
}

//! Rewrite coefficients of p(u), u = dist / scale, as coefficients of
//! p(dist) in the same power basis. (d/s)^j = d^j / s^j, so the transform
//! is exactly c_j = a_j / s^j — no binomial expansion, no cancellation.
std::vector<double> rescale_power_coefficients(
        const std::vector<double>& scaled, double scale) {
    std::vector<double> out(scaled.size(), 0.0);
    double scale_j = 1.0;  // scale^j
    for (std::size_t j = 0; j < scaled.size(); ++j) {
        out[j] = scaled[j] / scale_j;
        scale_j *= scale;
    }
    return out;
}

}  // namespace

std::vector<std::vector<double> > nps_convolved_efficiency_table(
        const std::vector<double>& cloud1,
        const std::vector<double>& cloud2,
        double dep1, double dep2, double r_iso6,
        int n_samples, int seed) {
    const char* api_name = "nps_convolved_efficiency_table";
    if (cloud1.empty() || cloud2.empty() ||
            cloud1.size() % 4 != 0 || cloud2.size() % 4 != 0) {
        IMP_THROW(api_name << ": clouds must be flat (n, 4) arrays — "
                          "x, y, z, weight per point",
                  ValueException);
    }
    for (double dep : {dep1, dep2}) {
        if (!std::isfinite(dep) || dep < 0.0 || dep > 1.0) {
            IMP_THROW(api_name << ": dep must be finite in [0, 1]",
                      ValueException);
        }
    }
    if (!std::isfinite(r_iso6) || r_iso6 <= 0.0) {
        IMP_THROW(api_name << ": r_iso6 must be finite and strictly positive",
                  ValueException);
    }
    const std::size_t n1 = cloud1.size() / 4;
    const std::size_t n2 = cloud2.size() / 4;

    // Generator-specific layout convention: dep = 1 selects isotropic
    // averaging (no orientation slot). In the *direct* evaluator, physical
    // NPSNetworkDye.dep = sqrt(r_avg/0.4) = 1 is instead the rigid limit;
    // the iso flag is separate. Do not infer direct-dye dynamics from this
    // row-layout sentinel. Both iso -> 1 row, mixed -> 25, pair -> 625.
    const bool grid1 = dep1 < 1.0;
    const bool grid2 = dep2 < 1.0;
    const int grid_slots = (grid1 ? 25 : 1) * (grid2 ? 25 : 1);
    if (n_samples <= 0) {
        // Estimator noise scales 1/sqrt(n); the 625-row table shares its
        // draws across slots so per-slot counts can stay moderate without
        // the table getting noisy, and one-row tables get the most.
        n_samples = grid_slots == 1 ? 200000 : (grid_slots == 25 ? 40000
                                                                 : 8000);
    }

    // Shared pair draws: the inter-cloud difference vectors v_s are drawn
    // once and reused by every slot and every bin. The reference instead
    // re-draws (with rejection from a hardcoded sphere) per bin.
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed));
    std::uniform_int_distribution<std::size_t> draw1(0, n1 - 1);
    std::uniform_int_distribution<std::size_t> draw2(0, n2 - 1);
    const std::size_t ns = static_cast<std::size_t>(n_samples);
    std::vector<double> vx(ns), vy(ns), vz(ns), vw(ns);
    for (std::size_t s = 0; s < ns; ++s) {
        const std::size_t i1 = draw1(rng);
        const std::size_t i2 = draw2(rng);
        vx[s] = cloud1[4 * i1 + 0] - cloud2[4 * i2 + 0];
        vy[s] = cloud1[4 * i1 + 1] - cloud2[4 * i2 + 1];
        vz[s] = cloud1[4 * i1 + 2] - cloud2[4 * i2 + 2];
        vw[s] = cloud1[4 * i1 + 3] * cloud2[4 * i2 + 3];
    }

    // Slot representative dipoles: the mid-slot angles of the committed
    // grid layout (polar mid of the 0.4-wide m bins, azimuth mid of the
    // 3.142-scaled bins), matching how the evaluator later picks rows
    // from actual dye angles.
    auto slot_direction = [](int polar_slot, int phi_slot) {
        const double m_mid = -1.0 + (2.0 * (polar_slot + 0.5)) / 5.0;
        const double phi_mid = (phi_slot + 0.5) * 3.142 / 5.0;
        const double c = m_mid;
        const double s = std::sqrt(std::max(0.0, 1.0 - c * c));
        return UnitVector{s * std::cos(phi_mid), s * std::sin(phi_mid), c};
    };

    // The bin axis offsets cloud 2 along +x (the reference's dD shift), so
    // a sample's separation at bin d is ov = v_s - d * xhat and the dipole
    // projections are cThK = (dirK . v_s - dirK_x * d) / |ov|. Per slot the
    // dot products dirK . v_s are constant — precompute them and the bin
    // loop is a handful of flops per sample instead of a re-sampling pass.
    std::vector<std::vector<double> > table(
            static_cast<std::size_t>(grid_slots),
            std::vector<double>(kTableCoefficients, 0.0));
    std::vector<double> u(kTableBinCount);
    std::vector<double> y(kTableBinCount);
    std::vector<double> a1(ns), a2(ns);
    for (int slot = 0; slot < grid_slots; ++slot) {
        // Row layout matches the evaluator: a mixed table lists dye 1's
        // slots as 5 * phi_slot + polar_slot; the pair grid interleaves
        // both dyes as 125 * phi1 + 25 * polar1 + 5 * phi2 + polar2.
        const int phi_slot1 = grid1 ? (grid2 ? slot / 125 : slot / 5) : 0;
        const int polar_slot1 =
                grid1 ? (grid2 ? (slot % 125) / 25 : slot % 5) : 0;
        const int phi_slot2 = grid2 ? (slot % 25) / 5 : 0;
        const int polar_slot2 = grid2 ? slot % 5 : 0;
        const UnitVector dir1 =
                grid1 ? slot_direction(polar_slot1, phi_slot1)
                      : UnitVector{0.0, 0.0, 1.0};
        const UnitVector dir2 =
                grid2 ? slot_direction(polar_slot2, phi_slot2)
                      : UnitVector{0.0, 0.0, 1.0};
        // Mutual angle of the two slot dipoles — constant within the slot.
        const double c_th_t = dir1.x * dir2.x + dir1.y * dir2.y +
                              dir1.z * dir2.z;
        if (grid1 || grid2) {
            for (std::size_t s = 0; s < ns; ++s) {
                a1[s] = dir1.x * vx[s] + dir1.y * vy[s] +
                        dir1.z * vz[s];
                a2[s] = dir2.x * vx[s] + dir2.y * vy[s] +
                        dir2.z * vz[s];
            }
        }
        double w_sum = 0.0;
        for (std::size_t s = 0; s < ns; ++s) w_sum += vw[s];
        if (w_sum <= 0.0) {
            IMP_THROW(api_name << ": clouds carry no positive weight",
                      ValueException);
        }
        for (int d = 0; d < kTableBinCount; ++d) {
            const double dist = static_cast<double>(d + 1);
            const double dist6 = dist * dist * dist * dist * dist * dist;
            double acc = 0.0;
            for (std::size_t s = 0; s < ns; ++s) {
                const double ovx = vx[s] - dist;
                const double r2 = ovx * ovx + vy[s] * vy[s] +
                                  vz[s] * vz[s];
                const double r = std::sqrt(r2);
                double avg_r6 = r_iso6;  // iso/iso: k2 = 2/3, x1.5 -> Riso6
                if (grid1 && grid2) {
                    // Both dyes on the orientation grid: the full two-dye
                    // form of the reference nonIsoToNonIso generator, with
                    // cosines normalized by the sampled separation (the
                    // reference divides by the mean bin distance here,
                    // which biases its cosine when |ov| deviates from d).
                    const double c1 = (a1[s] - dir1.x * dist) / r;
                    const double c2 = (a2[s] - dir2.x * dist) / r;
                    const double kappa_x2 =
                            (c_th_t - 3.0 * c1 * c2) *
                            (c_th_t - 3.0 * c1 * c2);
                    const double avg_k2 =
                            kappa_x2 * dep1 * dep2 +
                            (2.0 - dep1 - dep2) / 3.0 +
                            c1 * c1 * dep1 * (1.0 - dep2) +
                            c2 * c2 * dep2 * (1.0 - dep1);
                    avg_r6 = avg_k2 * 1.5 * r_iso6;
                } else if (grid1 || grid2) {
                    // Exactly one orientation-grid dye: the reference
                    // nonIsoToIso form — the isotropic dye averages out
                    // analytically, leaving k2 = (2 - dep)/3 + c^2 dep
                    // with only the grid dye's cosine.
                    const double dep = grid1 ? dep1 : dep2;
                    const double c1 =
                            ((grid1 ? a1[s] : a2[s]) -
                             (grid1 ? dir1.x : dir2.x) * dist) / r;
                    const double avg_k2 = (2.0 - dep) / 3.0 +
                                          c1 * c1 * dep;
                    avg_r6 = avg_k2 * 1.5 * r_iso6;
                }
                const double r6 = r2 * r2 * r2;
                acc += vw[s] * (avg_r6 / (avg_r6 + r6));
            }
            y[static_cast<std::size_t>(d)] = acc / w_sum;
            u[static_cast<std::size_t>(d)] = dist / 150.0;
        }
        table[static_cast<std::size_t>(slot)] =
                rescale_power_coefficients(solve_scaled(u, y,
                                                        kTableCoefficients),
                                           150.0);
    }
    return table;
}

namespace {

//! Validate a flat (n, 4) cloud; return its point count.
/*! Throws exactly the cloud-prior validation contract; shared by the
    cloud-prior API so the validation exists once. */
std::size_t validate_cloud(const std::vector<double>& cloud,
                           const char* api_name) {
    if (cloud.empty() || cloud.size() % 4 != 0) {
        IMP_THROW(api_name << ": cloud must be a flat (n, 4) array — "
                          "x, y, z, weight per point",
                  ValueException);
    }
    const std::size_t n = cloud.size() / 4;
    double w_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(cloud[4 * i + static_cast<std::size_t>(c)])) {
                IMP_THROW(api_name << ": cloud coordinates must be finite",
                          ValueException);
            }
        }
        const double w = cloud[4 * i + 3];
        if (!std::isfinite(w) || w < 0.0) {
            IMP_THROW(api_name << ": cloud weights must be finite and "
                          "non-negative",
                  ValueException);
        }
        w_sum += w;
    }
    if (!(w_sum > 0.0)) {
        IMP_THROW(api_name << ": cloud must carry a positive weight",
                  ValueException);
    }
    return n;
}

//! Scott's-rule bandwidth of a weighted cloud (d = 3).
/*! The weighted RMS spread about the weighted mean (population form —
    weights are frequencies, the KDE kernel itself regularises the scale)
    times Scott's factor n^(-1/(d+4)) (Scott 1992). Weights are used raw
    and divided by their sum. */
double bandwidth_of(const std::vector<double>& cloud, std::size_t n,
                    const char* api_name) {
    double w_sum = 0.0, mx = 0.0, my = 0.0, mz = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double w = cloud[4 * i + 3];
        w_sum += w;
        mx += w * cloud[4 * i + 0];
        my += w * cloud[4 * i + 1];
        mz += w * cloud[4 * i + 2];
    }
    const double inv_w = 1.0 / w_sum;
    mx *= inv_w;
    my *= inv_w;
    mz *= inv_w;
    double v = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double w = cloud[4 * i + 3];
        const double dx = cloud[4 * i + 0] - mx;
        const double dy = cloud[4 * i + 1] - my;
        const double dz = cloud[4 * i + 2] - mz;
        v += w * (dx * dx + dy * dy + dz * dz);
    }
    if (!(v > 0.0)) {
        IMP_THROW(api_name << ": cloud positions carry no spread",
                  ValueException);
    }
    return std::sqrt(v * inv_w) *
           std::pow(static_cast<double>(n), -0.2);
}

//! Weighted Gaussian-KDE log density — log( sum_i w_i phi_h(x - x_i) / W ).
/*! Two allocation-free passes: the running maximum of the kernel terms,
    then the log-sum-exp accumulation. Weights raw, W their sum. */
double kde_log_density(const std::vector<double>& cloud, std::size_t n,
                       double h, double x, double y, double z) {
    const double inv_two_h2 = 1.0 / (2.0 * h * h);
    const double log_kernel_norm =
            -3.0 * std::log(h) - 1.5 * std::log(2.0 * M_PI);
    double w_sum = 0.0;
    double max_term = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        const double w = cloud[4 * i + 3];
        if (w == 0.0) continue;
        w_sum += w;
        const double dx = x - cloud[4 * i + 0];
        const double dy = y - cloud[4 * i + 1];
        const double dz = z - cloud[4 * i + 2];
        const double t = std::log(w) + log_kernel_norm
                                 - inv_two_h2 * (dx * dx + dy * dy + dz * dz);
        if (t > max_term) max_term = t;
    }
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double w = cloud[4 * i + 3];
        if (w == 0.0) continue;
        const double dx = x - cloud[4 * i + 0];
        const double dy = y - cloud[4 * i + 1];
        const double dz = z - cloud[4 * i + 2];
        const double t = std::log(w) + log_kernel_norm
                                 - inv_two_h2 * (dx * dx + dy * dy + dz * dz);
        acc += std::exp(t - max_term);
    }
    return max_term + std::log(acc) - std::log(w_sum);
}

}  // namespace

double nps_cloud_bandwidth(const std::vector<double>& cloud) {
    const std::size_t n = validate_cloud(cloud, "nps_cloud_bandwidth");
    return bandwidth_of(cloud, n, "nps_cloud_bandwidth");
}

double nps_cloud_log_prior(const std::vector<double>& cloud,
                           double x, double y, double z) {
    const char* api_name = "nps_cloud_log_prior";
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        IMP_THROW(api_name << ": query position must be finite",
                  ValueException);
    }
    const std::size_t n = validate_cloud(cloud, api_name);
    const double h = bandwidth_of(cloud, n, api_name);
    return kde_log_density(cloud, n, h, x, y, z);
}

std::vector<double> nps_config_from_cloud(
        const std::vector<double>& cloud) {
    validate_cloud(cloud, "nps_config_from_cloud");
    const std::vector<double> mean = points_weighted_mean(cloud);
    return std::vector<double>{mean[0], mean[1], mean[2], 0.0, 0.0};
}

double nps_cloud_prior_score(const std::vector<double>& cloud,
                             double x, double y, double z, double scale) {
    if (!std::isfinite(scale) || scale <= 0.0) {
        IMP_THROW("nps_cloud_prior_score: scale must be finite and strictly positive",
                  ValueException);
    }
    const std::size_t n = validate_cloud(cloud, "nps_cloud_prior_score");
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return kNpsInfinity;
    const double bandwidth = bandwidth_of(cloud, n, "nps_cloud_prior_score");
    const double log_density = kde_log_density(cloud, cloud.size() / 4, bandwidth, x, y, z);
    return std::isfinite(log_density) ? -scale * log_density : kNpsInfinity;
}

// ---------------------------------------------------------------------------
// The sampler objective and its diagnostics: the Fast-NPS sampling loop
// on bff's MCMCSampler, without a per-move crossing into Python.
// ---------------------------------------------------------------------------

NPSNetworkObjective::NPSNetworkObjective(const NPSNetworkDyes& dyes,
                                         const NPSMeasurements& measurements,
                                         int n_dyes, std::string name)
    : GraphNode(name),
      dyes_(dyes),
      measurements_(measurements),
      n_dyes_(n_dyes) {
    if (n_dyes <= 0) {
        IMP_THROW("NPSNetworkObjective: n_dyes must be positive",
                  ValueException);
    }
    // The node seeds m = 0, phi = 0 and links no angle ports, so it can
    // only evaluate a position-only posterior: every active observable
    // must be provably independent of the dye orientations it freezes.
    // Angle-independence of each likelihood term (avg_kappa_2 / row
    // selection / TA), from the committed leaves:
    // - direct FRET: avg_kappa_2 loses its cThT/cTh1/cTh2 dependence
    //   exactly when dep1 = 0 AND dep2 = 0 (kappa^2 = 2/3);
    // - convolved FRET: iso/iso reads only row 0; a mixed pair reads the
    //   25-row single-dye grid and a non-iso pair the 625-row grid, both
    //   keyed by the frozen m/phi -- an arbitrary row would be pinned;
    // - TA: dep1*dep2*(3 cThT^2 - 1)/5 is constant exactly when
    //   dep1*dep2 = 0.
    const char* because = "NPSNetworkObjective: the objective fixes the "
                          "dye angles (m = 0, phi = 0) and has no angle "
                          "ports, so an orientation-dependent likelihood "
                          "would be silently conditioned on that "
                          "orientation";
    for (std::size_t m_index = 0; m_index < measurements_.size();
            ++m_index) {
        const NPSMeasurement& meas = measurements_[m_index];
        if (meas.dye1 < 0 || static_cast<std::size_t>(meas.dye1) >= dyes_.size()
                || meas.dye2 < 0
                || static_cast<std::size_t>(meas.dye2) >= dyes_.size()) {
            IMP_THROW("NPSNetworkObjective: measurement " << m_index
                          << " references a dye outside the " << dyes_.size()
                          << "-dye array",
                      ValueException);
        }
        const NPSNetworkDye& meta1 = dyes_[meas.dye1];
        const NPSNetworkDye& meta2 = dyes_[meas.dye2];
        if (meas.fret_active) {
            if (!meta1.dist_conv && !meta2.dist_conv) {
                if (meta1.dep != 0.0 || meta2.dep != 0.0) {
                    IMP_THROW(because
                              << " (direct FRET needs dep1 = dep2 = 0, got "
                              << meta1.dep << ", " << meta2.dep << ")",
                              ValueException);
                }
            } else if (!(meta1.iso && meta2.iso)) {
                IMP_THROW(because
                          << " (distance-convolved FRET needs an iso/iso "
                             "pair -- row 0 -- not an orientation-grid "
                             "row)",
                          ValueException);
            }
        }
        if (meas.ta_active && meta1.dep * meta2.dep != 0.0) {
            IMP_THROW(because << " (transfer anisotropy needs dep1*dep2 = "
                              << "0, got " << meta1.dep * meta2.dep << ")",
                      ValueException);
        }
    }
}

void NPSNetworkObjective::evaluate() {
    std::vector<std::vector<double>> config;
    config.reserve(static_cast<std::size_t>(n_dyes_));
    for (int i = 0; i < n_dyes_; ++i) {
        // The parameter ports carry one scalar each (MCMCSampler writes
        // scalars), so read them as scalars.
        config.push_back(std::vector<double>{
                get_input_port("x" + std::to_string(3 * i))->get_value(),
                get_input_port("x" + std::to_string(3 * i + 1))->get_value(),
                get_input_port("x" + std::to_string(3 * i + 2))->get_value(),
                0.0, 0.0});
    }
    double value;
    if (output_is_log_likelihood_) {
        value = nps_network_log_likelihood(config, dyes_, measurements_);
    } else {
        value = nps_network_chi2(config, dyes_, measurements_);
    }
    get_output_port("chi2")->set_value(value);
}

double nps_cross_entropy(const std::vector<double>& chi2,
                         const std::vector<double>& ln_prior) {
    const char* api_name = "nps_cross_entropy";
    if (chi2.size() != ln_prior.size() || chi2.empty()) {
        IMP_THROW(api_name << ": chi2 and ln_prior must be equal-length, "
                          "non-empty records",
                  ValueException);
    }
    double log_post_sum = 0.0;
    std::size_t kept = 0;
    for (std::size_t i = 0; i < chi2.size(); ++i) {
        // unscorable states skip: chi2 = inf (the objective rejected the
        // move) is symmetric with a non-finite recorded prior -- a NaN
        // prior must not be averaged in as a NaN cross-entropy.
        if (!std::isfinite(chi2[i]) || !std::isfinite(ln_prior[i])) continue;
        log_post_sum += -0.5 * chi2[i] + ln_prior[i];
        ++kept;
    }
    if (kept == 0) {
        IMP_THROW(api_name << ": the records hold no finite state",
                  ValueException);
    }
    return -log_post_sum / static_cast<double>(kept);
}

std::vector<double> nps_mean_mcse(
        const std::vector<std::vector<double>>& chain) {
    const char* api_name = "nps_mean_mcse";
    if (chain.empty() || chain[0].empty()) {
        IMP_THROW(api_name << ": chain must be non-empty",
                  ValueException);
    }
    const std::size_t ndim = chain[0].size();
    std::vector<double> out(ndim, 0.0);
    for (std::size_t d = 0; d < ndim; ++d) {
        McmcChains single(1, std::vector<double>());
        single[0].reserve(chain.size());
        for (const auto& row : chain) {
            if (row.size() != ndim) {
                IMP_THROW(api_name << ": chain rows must share one width",
                          ValueException);
            }
            single[0].push_back(row[d]);
        }
        out[d] = mcse_mean(single);
    }
    return out;
}

IMPBFF_END_NAMESPACE
