/** \file SMLMGaussianOverlap.cpp
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SMLMGaussianOverlap.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

IMPBFF_BEGIN_NAMESPACE
namespace {
constexpr double covariance_tolerance = 1e-12;
constexpr long double smlm_overlap_log_gaussian_coefficient = -2.75681559961401805267L;

void overlap_require(bool condition, const char* message) {
  if (!condition) IMP_THROW(message, IMP::ValueException);
}

double overlap_double(long double value, const char* message) {
  overlap_require(std::isfinite(value) &&
                      std::abs(value) <= std::numeric_limits<double>::max(),
                  message);
  return static_cast<double>(value);
}

// Scale before summing: normalization never overflows on finite weights.
std::vector<double> normalized_overlap_weights(
    const std::vector<double>& weights, std::size_t count) {
  overlap_require(weights.empty() || weights.size() == count,
                  "Gaussian overlap weights must match component count");
  if (weights.empty())
    return std::vector<double>(count, count ? 1.0 / count : 0.0);
  double largest = 0.0;
  for (double weight : weights) {
    overlap_require(std::isfinite(weight) && weight >= 0,
                    "Gaussian overlap weights must be finite and nonnegative");
    largest = std::max(largest, weight);
  }
  std::vector<double> normalized(count, 0.0);
  if (largest == 0) return normalized;
  long double sum = 0.0;
  for (double weight : weights)
    sum += static_cast<long double>(weight) / largest;
  for (std::size_t i = 0; i < count; ++i)
    normalized[i] = static_cast<double>(
        (static_cast<long double>(weights[i]) / largest) / sum);
  return normalized;
}

Eigen::Matrix3d overlap_covariance(const double* entries) {
  double scale = 0.0;
  for (int k = 0; k < 9; ++k) {
    overlap_require(std::isfinite(entries[k]),
                    "Gaussian model covariances must be finite");
    scale = std::max(scale, std::abs(entries[k]));
  }
  if (scale == 0) return Eigen::Matrix3d::Zero();
  Eigen::Matrix3d covariance;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const double a = entries[3 * row + column];
      const double b = entries[3 * column + row];
      overlap_require(std::abs(a / scale - b / scale) <= covariance_tolerance,
                      "Gaussian model covariance must be symmetric");
      covariance(row, column) = 0.5 * a + 0.5 * b;
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance / scale);
  overlap_require(solver.info() == Eigen::Success &&
                      solver.eigenvalues().allFinite(),
                  "Gaussian model covariance eigendecomposition failed");
  overlap_require(solver.eigenvalues().minCoeff() >= -covariance_tolerance,
                  "Gaussian model covariance must be positive semidefinite");
  if (solver.eigenvalues().minCoeff() < 0) {
    covariance = (solver.eigenvectors() *
                  solver.eigenvalues().cwiseMax(0.0).asDiagonal() *
                  solver.eigenvectors().transpose()).eval() * scale;
    // Make the numerical representation explicitly symmetric.
    covariance = (0.5 * covariance + 0.5 * covariance.transpose()).eval();
  }
  overlap_require(covariance.allFinite(),
                  "Gaussian model covariance is not representable");
  return covariance;
}
}  // namespace

SMLMGaussianOverlap::SMLMGaussianOverlap(const SMLMIndex& index)
    : index_(index),
      normalized_data_weights_(normalized_overlap_weights(
          index_.weights_, index_.coordinates_.size() / 3)) {
  for (double weight : normalized_data_weights_)
    if (weight > 0) has_data_weight_ = true;
}

int SMLMGaussianOverlap::get_number_of_localizations() const {
  return index_.get_number_of_localizations();
}

SMLMGaussianOverlapResult SMLMGaussianOverlap::evaluate(
    const std::vector<double>& model_coordinates,
    const std::vector<double>& model_covariances,
    const std::vector<double>& model_weights,
    double cutoff_sigma, double background) const {
  overlap_require(model_coordinates.size() % 3 == 0,
                  "Gaussian model coordinates must be flat M x 3");
  const std::size_t count = model_coordinates.size() / 3;
  overlap_require(count <= std::numeric_limits<std::size_t>::max() / 9 &&
                      model_covariances.size() == 9 * count,
                  "Gaussian model covariances must be flat M x 9");
  for (double coordinate : model_coordinates)
    overlap_require(std::isfinite(coordinate),
                    "Gaussian model coordinates must be finite");
  overlap_require(std::isfinite(cutoff_sigma) && cutoff_sigma > 0,
                  "Gaussian overlap cutoff_sigma must be finite and positive");
  overlap_require(std::isfinite(background) && background > 0,
                  "Gaussian overlap background must be finite and positive");
  const std::vector<double> weights = normalized_overlap_weights(model_weights, count);
  SMLMGaussianOverlapResult result;
  result.component_overlaps.assign(count, 0.0);
  result.gradient.assign(model_coordinates.size(), 0.0);
  std::vector<long double> component_log_gradient(model_coordinates.size(), 0.0L);
  std::vector<long double> weighted_overlaps(count, 0.0L);
  long double overlap = 0.0L;
  // Roundoff in an enclosing support bound must never reject a boundary pair.
  const long double pruning_cutoff = static_cast<long double>(cutoff_sigma) *
      (1.0L + 32.0L * std::numeric_limits<double>::epsilon());
  std::vector<int> pending;
  pending.reserve(64);

  for (std::size_t component = 0; component < count; ++component) {
    // Validate every covariance, including zero-weight/background-only inputs.
    const Eigen::Matrix3d covariance = overlap_covariance(
        model_covariances.data() + 9 * component);
    if (!has_data_weight_) continue;
    const double* centre = model_coordinates.data() + 3 * component;
    std::array<double, 3> axis_sigma, bounding_sigma;
    for (int axis = 0; axis < 3; ++axis) {
      axis_sigma[axis] = std::sqrt(std::max(0.0, covariance(axis, axis)));
      bounding_sigma[axis] = axis_sigma[axis];
      for (int other = 0; other < 3; ++other)
        if (axis != other)
          bounding_sigma[axis] = std::hypot(bounding_sigma[axis],
              std::sqrt(std::abs(covariance(axis, other))));
    }
    long double component_overlap = 0.0L;
    std::array<long double, 3> component_gradient = {{0.0L, 0.0L, 0.0L}};
    pending.clear();
    if (!index_.nodes_.empty()) pending.push_back(0);
    while (!pending.empty()) {
      const int node_index = pending.back();
      pending.pop_back();
      const SMLMIndex::Node& node = index_.nodes_[node_index];
      bool outside = false;
      long double ellipsoid_distance = 0.0L;
      for (int axis = 0; axis < 3; ++axis) {
        const long double distance = centre[axis] < node.lower[axis]
            ? static_cast<long double>(node.lower[axis]) - centre[axis]
            : (centre[axis] > node.upper[axis]
               ? static_cast<long double>(centre[axis]) - node.upper[axis]
               : 0.0L);
        const long double support = std::hypot(
            static_cast<long double>(node.max_sigma[axis]),
            static_cast<long double>(axis_sigma[axis]));
        if (distance / support > pruning_cutoff) {
          outside = true;
          break;
        }
        // Gershgorin: C <= diag(C_aa + sum_b!=a |C_ab|). Adding the
        // node's maximum measured variances gives a Loewner upper bound
        // for EVERY pair covariance below the node. Its diagonal ellipsoid
        // is a conservative superset even with model cross-covariances.
        const long double bound = std::hypot(
            static_cast<long double>(node.max_sigma[axis]),
            static_cast<long double>(bounding_sigma[axis]));
        ellipsoid_distance = std::hypot(ellipsoid_distance, distance / bound);
      }
      if (outside || ellipsoid_distance > pruning_cutoff) continue;
      if (node.left >= 0) {
        pending.push_back(node.right);
        pending.push_back(node.left);
        continue;
      }
      for (int position = node.begin; position < node.end; ++position) {
        const int observation = index_.order_[position];
        const double data_weight = normalized_data_weights_[observation];
        if (data_weight == 0) continue;
        const double* sigma = index_.sigmas_.data() + 3 * observation;
        const double* data = index_.coordinates_.data() + 3 * observation;
        double scale = 0.0;
        for (int axis = 0; axis < 3; ++axis)
          scale = std::max(scale, std::max(sigma[axis], axis_sigma[axis]));
        // Factor the dimensionless covariance instead of squaring a possibly
        // enormous/small physical scale, and obtain log(det) from Cholesky.
        Eigen::Matrix3d sum = (covariance / scale) / scale;
        Eigen::Vector3d delta;
        bool pair_outside = false;
        for (int axis = 0; axis < 3; ++axis) {
          const double relative_sigma = sigma[axis] / scale;
          sum(axis, axis) += relative_sigma * relative_sigma;
          const long double difference =
              (static_cast<long double>(centre[axis]) - data[axis]) / scale;
          if (std::abs(difference) / std::sqrt(sum(axis, axis)) > pruning_cutoff) {
            pair_outside = true;
            break;
          }
          delta[axis] = overlap_double(difference,
              "Gaussian pair separation is not representable");
        }
        if (pair_outside) continue;
        Eigen::LLT<Eigen::Matrix3d> factor(sum);
        overlap_require(factor.info() == Eigen::Success,
                        "Gaussian pair covariance is not representable as positive definite");
        const Eigen::Vector3d whitened = factor.matrixL().solve(delta);
        const double distance = whitened.stableNorm();
        if (distance > cutoff_sigma) continue;
        const Eigen::Matrix3d lower = factor.matrixL();
        long double log_kernel = smlm_overlap_log_gaussian_coefficient -
            3.0L * std::log(static_cast<long double>(scale));
        for (int axis = 0; axis < 3; ++axis)
          log_kernel -= std::log(static_cast<long double>(lower(axis, axis)));
        log_kernel -= 0.5L * distance * distance;
        const long double contribution = std::exp(
            log_kernel + std::log(static_cast<long double>(data_weight)));
        overlap_require(std::isfinite(contribution),
                        "Gaussian pair overlap is not representable");
        const long double combined_overlap = component_overlap + contribution;
        if (weights[component] > 0 && contribution > 0) {
          const Eigen::Vector3d inverse_delta = factor.matrixU().solve(whitened);
          // Accumulate the component's LOG derivative as a weighted mean.
          // A raw overlap derivative can overflow even when both overlap and
          // the final score gradient are representable (e.g. tiny units).
          for (int axis = 0; axis < 3; ++axis)
            component_gradient[axis] =
                (component_overlap / combined_overlap) * component_gradient[axis] -
                ((contribution / combined_overlap) * inverse_delta[axis]) / scale;
        }
        component_overlap = combined_overlap;
      }
    }
    result.component_overlaps[component] = overlap_double(component_overlap,
        "Gaussian component overlap is not representable");
    weighted_overlaps[component] = weights[component] * component_overlap;
    overlap += weighted_overlaps[component];
    for (int axis = 0; axis < 3; ++axis)
      component_log_gradient[3 * component + axis] = component_gradient[axis];
  }
  result.overlap = overlap_double(overlap, "Gaussian mixture overlap is not representable");
  const long double floor_scale = std::max(overlap, static_cast<long double>(background));
  const long double scaled_total = overlap / floor_scale + background / floor_scale;
  result.score = overlap_double(-std::log(floor_scale) - std::log(scaled_total),
                               "Gaussian overlap score is not representable");
  for (std::size_t component = 0; component < count; ++component) {
    const long double fraction = (weighted_overlaps[component] / floor_scale) / scaled_total;
    for (int axis = 0; axis < 3; ++axis)
      result.gradient[3 * component + axis] = overlap_double(
          -fraction * component_log_gradient[3 * component + axis],
          "Gaussian overlap score gradient is not representable");
  }
  return result;
}

IMPBFF_END_NAMESPACE
