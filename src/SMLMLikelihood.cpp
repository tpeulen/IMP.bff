/** \file SMLMLikelihood.cpp
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SMLMLikelihood.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

IMPBFF_BEGIN_NAMESPACE
namespace {
constexpr double smlm_likelihood_log_gaussian_coefficient = -2.756815599614018;
constexpr double smlm_likelihood_negative_infinity = -std::numeric_limits<double>::infinity();
using smlm_likelihood_Point = std::array<double, 3>;

void smlm_likelihood_require(bool condition, const char* message) {
  if (!condition) IMP_THROW(message, IMP::ValueException);
}

std::vector<double> smlm_likelihood_normalized_weights(const std::vector<double>& coordinates,
                                       const std::vector<double>& weights) {
  smlm_likelihood_require(!coordinates.empty() && coordinates.size() % 3 == 0,
          "SMLM point model needs nonempty flattened J x 3 coordinates");
  const std::size_t count = coordinates.size()/3;
  smlm_likelihood_require(weights.empty() || weights.size() == count,
          "SMLM model weight length must equal emitter count");
  if (weights.empty()) return std::vector<double>(count, 1.0/count);
  double maximum = 0;
  for (double weight : weights) {
    smlm_likelihood_require(std::isfinite(weight) && weight >= 0,
            "SMLM model weights must be finite and nonnegative");
    maximum = std::max(maximum, weight);
  }
  smlm_likelihood_require(maximum > 0, "SMLM model must have positive total emitter weight");
  std::vector<double> result(weights);
  // Normalize by the maximum first: valid finite model weights can sum above
  // double's maximum without making their normalized distribution invalid.
  long double total = 0;
  for (double& weight : result) { weight /= maximum; total += weight; }
  for (double& weight : result) weight = static_cast<double>(weight/total);
  return result;
}

std::vector<double> smlm_likelihood_checked_pose(const std::vector<double>& pose) {
  if (pose.empty()) return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  smlm_likelihood_require(pose.size() == 12, "SMLM likelihood pose must have twelve 3x4 entries");
  for (double value : pose)
    smlm_likelihood_require(std::isfinite(value), "SMLM likelihood pose must be finite");
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      double dot = 0;
      for (int k = 0; k < 3; ++k) dot += pose[4*a+k]*pose[4*b+k];
      smlm_likelihood_require(std::isfinite(dot) && std::abs(dot-(a == b ? 1.0 : 0.0)) <= 1e-6,
              "SMLM likelihood pose rotation must be orthonormal");
    }
  const double determinant = pose[0]*(pose[5]*pose[10]-pose[6]*pose[9]) -
      pose[1]*(pose[4]*pose[10]-pose[6]*pose[8]) +
      pose[2]*(pose[4]*pose[9]-pose[5]*pose[8]);
  smlm_likelihood_require(std::abs(determinant-1) <= 1e-6,
          "SMLM likelihood pose must be a proper rotation");
  return pose;
}

double smlm_likelihood_log_add(double a, double b) {
  if (a == smlm_likelihood_negative_infinity) return b;
  if (b == smlm_likelihood_negative_infinity) return a;
  const double maximum = std::max(a, b);
  return maximum + std::log1p(std::exp(std::min(a, b)-maximum));
}

double smlm_likelihood_finite_cast(long double value, const char* message) {
  smlm_likelihood_require(std::isfinite(value) && std::abs(value) <= std::numeric_limits<double>::max(),
          message);
  return static_cast<double>(value);
}

double smlm_likelihood_density_from_log(double value) {
  if (value == smlm_likelihood_negative_infinity) return 0;
  const double density = std::exp(value);
  smlm_likelihood_require(std::isfinite(density), "SMLM likelihood PDF overflow");
  return density;  // Underflow to zero does not discard the finite log PDF.
}

struct smlm_likelihood_Contribution {
  int emitter;
  double log_kernel;
  smlm_likelihood_Point rotated_position, displacement;
};

// Damped 6x6 Gauss-Newton solve, partial pivoting. This fixed-size problem
// needs no separate dense-matrix allocation or external math dependency.
std::array<double, 6> smlm_likelihood_descent_step(const SMLMLikelihoodResult& result) {
  double matrix[6][7];
  for (int i = 0; i < 6; ++i) {
    // Translation and angular coordinates have different units. Damping each
    // diagonal relative to itself avoids making Angstrom translation steps
    // artificially small because the rotational diagonal is numerically large.
    const double damping = std::max(1e-12, result.pose_information[6*i+i]*1e-6);
    for (int j = 0; j < 6; ++j)
      matrix[i][j] = result.pose_information[6*i+j] + (i == j ? damping : 0);
    matrix[i][6] = -result.pose_gradient[i];
  }
  for (int column = 0; column < 6; ++column) {
    int pivot = column;
    for (int row = column+1; row < 6; ++row)
      if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) pivot = row;
    for (int j = column; j < 7; ++j) std::swap(matrix[pivot][j], matrix[column][j]);
    smlm_likelihood_require(std::isfinite(matrix[column][column]) && std::abs(matrix[column][column]) > 0,
            "SMLM likelihood pose information is numerically singular");
    for (int row = column+1; row < 6; ++row) {
      const double factor = matrix[row][column]/matrix[column][column];
      for (int j = column+1; j < 7; ++j) matrix[row][j] -= factor*matrix[column][j];
    }
  }
  std::array<double, 6> step{};
  for (int row = 5; row >= 0; --row) {
    double residual = matrix[row][6];
    for (int j = row+1; j < 6; ++j) residual -= matrix[row][j]*step[j];
    step[row] = residual/matrix[row][row];
    smlm_likelihood_require(std::isfinite(step[row]), "SMLM likelihood rigid step overflow");
  }
  return step;
}

std::vector<double> updated_pose(const std::vector<double>& pose,
                                  const std::array<double, 6>& step, double alpha) {
  const double x = alpha*step[3], y = alpha*step[4], z = alpha*step[5];
  const double angle = std::hypot(std::hypot(x, y), z);
  const double a = angle > 1e-8 ? std::sin(angle)/angle : 1-angle*angle/6;
  const double b = angle > 1e-8 ? (1-std::cos(angle))/(angle*angle)
                               : 0.5-angle*angle/24;
  const double delta[9] = {
    1-b*(y*y+z*z), b*x*y-a*z, b*x*z+a*y,
    b*x*y+a*z, 1-b*(x*x+z*z), b*y*z-a*x,
    b*x*z-a*y, b*y*z+a*x, 1-b*(x*x+y*y)};
  std::vector<double> result(pose);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      result[4*row+column] = 0;
      for (int k = 0; k < 3; ++k)
        result[4*row+column] += delta[3*row+k]*pose[4*k+column];
    }
    result[4*row+3] += alpha*step[row];
  }
  return result;
}
}  // namespace

double SMLMLikelihoodOptions::get_roi_volume() const {
  smlm_likelihood_require(roi_min.size() == 3 && roi_max.size() == 3,
          "SMLM likelihood needs explicit three-dimensional ROI bounds");
  long double volume = 1;
  for (int k = 0; k < 3; ++k) {
    smlm_likelihood_require(std::isfinite(roi_min[k]) && std::isfinite(roi_max[k]) && roi_max[k] > roi_min[k],
            "SMLM likelihood ROI bounds must be finite with positive extents");
    volume *= static_cast<long double>(roi_max[k])-roi_min[k];
  }
  const double result = smlm_likelihood_finite_cast(volume, "SMLM likelihood ROI volume overflow");
  smlm_likelihood_require(result > 0, "SMLM likelihood ROI volume underflow");
  return result;
}

SMLMPointModel::SMLMPointModel(const std::vector<double>& coordinates,
                               const std::vector<double>& weights, int leaf_size)
    : model_index_(coordinates, std::vector<double>(coordinates.size()/3, 1.0),
                   smlm_likelihood_normalized_weights(coordinates, weights), std::vector<int>(), leaf_size) {}

SMLMLikelihoodResult SMLMPointModel::evaluate(
    const SMLMIndex& observations, const SMLMLikelihoodOptions& options,
    const std::vector<double>& pose) const {
  const double volume = options.get_roi_volume();
  smlm_likelihood_require(std::isfinite(options.background_fraction) && options.background_fraction >= 0 &&
          options.background_fraction < 1, "SMLM background_fraction must be in [0,1)");
  smlm_likelihood_require(std::isfinite(options.intrinsic_sigma) && options.intrinsic_sigma >= 0,
          "SMLM intrinsic_sigma must be finite and nonnegative");
  smlm_likelihood_require(std::isfinite(options.cutoff_sigma) && options.cutoff_sigma > 0,
          "SMLM likelihood cutoff_sigma must be finite and positive");
  const auto transform = smlm_likelihood_checked_pose(pose);
  const auto& positions = observations.get_coordinates();
  const auto& sigmas = observations.get_sigmas();
  const auto& observation_weights = observations.get_weights();
  const auto& model_positions = get_coordinates();
  const auto& model_weights = get_weights();
  const std::size_t count = observation_weights.size();
  SMLMLikelihoodResult result;
  result.localization_count = observations.get_number_of_localizations();
  result.total_observation_weight = std::accumulate(observation_weights.begin(),
                                                   observation_weights.end(), 0.0);
  smlm_likelihood_require(std::isfinite(result.total_observation_weight), "SMLM observation weight sum overflow");
  result.pdf.resize(count);
  result.log_pdf.resize(count);
  result.model_pdf.resize(count);
  result.signal_fraction.resize(count);
  result.pose_gradient.assign(6, 0);
  result.pose_information.assign(36, 0);
  if (options.compute_model_gradient) result.model_gradient.assign(model_positions.size(), 0);
  const double log_signal_weight = std::log1p(-options.background_fraction);
  const double log_background = options.background_fraction > 0
      ? std::log(options.background_fraction)-std::log(volume) : smlm_likelihood_negative_infinity;
  long double log_likelihood = 0, weighted_log_likelihood = 0;
  bool undefined_gradient = false;
  std::vector<smlm_likelihood_Contribution> contributions;
  for (std::size_t i = 0; i < count; ++i) {
    const double* position = positions.data()+3*i;
    smlm_likelihood_Point sigma, inverse_point{};
    double maximum_sigma = 0, log_normalizer = smlm_likelihood_log_gaussian_coefficient;
    for (int k = 0; k < 3; ++k) {
      smlm_likelihood_require(position[k] >= options.roi_min[k] && position[k] <= options.roi_max[k],
              "SMLM observation is outside the explicit likelihood ROI");
      sigma[k] = std::hypot(sigmas[3*i+k], options.intrinsic_sigma);
      smlm_likelihood_require(std::isfinite(sigma[k]), "SMLM effective observation sigma overflow");
      maximum_sigma = std::max(maximum_sigma, sigma[k]);
      log_normalizer -= std::log(sigma[k]);
      long double coordinate = 0;
      for (int axis = 0; axis < 3; ++axis)
        coordinate += transform[4*axis+k]*(static_cast<long double>(position[axis])-
                                          transform[4*axis+3]);
      inverse_point[k] = smlm_likelihood_finite_cast(coordinate, "SMLM inverse pose coordinate overflow");
    }
    const double radius = options.cutoff_sigma*maximum_sigma;
    smlm_likelihood_require(std::isfinite(radius), "SMLM likelihood search support radius overflow");
    const auto candidates = model_index_.radius_search(
        std::vector<double>(inverse_point.begin(), inverse_point.end()), radius);
    contributions.clear();
    double log_model = smlm_likelihood_negative_infinity;
    for (int emitter : candidates) {
      if (model_weights[emitter] == 0) continue;
      smlm_likelihood_Contribution term;
      term.emitter = emitter;
      smlm_likelihood_Point scaled;
      for (int axis = 0; axis < 3; ++axis) {
        long double rotated = 0;
        for (int k = 0; k < 3; ++k)
          rotated += static_cast<long double>(transform[4*axis+k]) *
                     model_positions[3*static_cast<std::size_t>(emitter)+k];
        term.rotated_position[axis] = smlm_likelihood_finite_cast(rotated, "SMLM model pose coordinate overflow");
        term.displacement[axis] = smlm_likelihood_finite_cast(rotated + transform[4*axis+3] - position[axis],
                                               "SMLM model-observation displacement overflow");
        scaled[axis] = term.displacement[axis]/sigma[axis];
      }
      const double distance = std::hypot(std::hypot(scaled[0], scaled[1]), scaled[2]);
      if (distance > options.cutoff_sigma) continue;
      term.log_kernel = std::log(model_weights[emitter]) + log_normalizer - 0.5*distance*distance;
      if (term.log_kernel == smlm_likelihood_negative_infinity) continue;
      log_model = smlm_likelihood_log_add(log_model, term.log_kernel);
      contributions.push_back(term);
    }
    const double log_signal = log_model + log_signal_weight;
    const double log_pdf = smlm_likelihood_log_add(log_signal, log_background);
    result.log_pdf[i] = log_pdf;
    result.pdf[i] = smlm_likelihood_density_from_log(log_pdf);
    result.model_pdf[i] = smlm_likelihood_density_from_log(log_model);
    result.signal_fraction[i] = log_signal == smlm_likelihood_negative_infinity
        ? 0 : std::exp(log_signal-log_pdf);
    log_likelihood += log_pdf;
    if (observation_weights[i] == 0) continue;
    weighted_log_likelihood += static_cast<long double>(observation_weights[i])*log_pdf;
    if (log_pdf == smlm_likelihood_negative_infinity) { undefined_gradient = true; continue; }
    if (contributions.empty()) continue;
    ++result.supported_localizations;
    const double observation_fraction = observation_weights[i]/result.total_observation_weight;
    for (const auto& term : contributions) {
      const double responsibility = std::exp(log_signal_weight + term.log_kernel-log_pdf);
      const double fraction = observation_fraction*responsibility;
      if (fraction == 0) continue;
      smlm_likelihood_Point gradient;
      for (int axis = 0; axis < 3; ++axis) {
        gradient[axis] = fraction*(term.displacement[axis]/sigma[axis])/sigma[axis];
        result.pose_gradient[axis] += gradient[axis];
      }
      const auto& r = term.rotated_position;
      result.pose_gradient[3] += r[1]*gradient[2]-r[2]*gradient[1];
      result.pose_gradient[4] += r[2]*gradient[0]-r[0]*gradient[2];
      result.pose_gradient[5] += r[0]*gradient[1]-r[1]*gradient[0];
      if (options.compute_model_gradient)
        for (int k = 0; k < 3; ++k)
          for (int axis = 0; axis < 3; ++axis)
            result.model_gradient[3*static_cast<std::size_t>(term.emitter)+k] +=
                transform[4*axis+k]*gradient[axis];
      const double jacobian[3][6] = {
          {1, 0, 0, 0, r[2], -r[1]},
          {0, 1, 0, -r[2], 0, r[0]},
          {0, 0, 1, r[1], -r[0], 0}};
      for (int row = 0; row < 6; ++row)
        for (int column = 0; column <= row; ++column) {
          double information = 0;
          for (int axis = 0; axis < 3; ++axis)
            information += fraction*(jacobian[axis][row]/sigma[axis]) *
                                     (jacobian[axis][column]/sigma[axis]);
          result.pose_information[6*row+column] += information;
          if (row != column) result.pose_information[6*column+row] += information;
        }
    }
  }
  result.log_likelihood = log_likelihood == smlm_likelihood_negative_infinity
      ? smlm_likelihood_negative_infinity : smlm_likelihood_finite_cast(log_likelihood, "SMLM summed log likelihood overflow");
  result.weighted_log_likelihood = weighted_log_likelihood == smlm_likelihood_negative_infinity
      ? smlm_likelihood_negative_infinity : smlm_likelihood_finite_cast(weighted_log_likelihood, "SMLM weighted log likelihood overflow");
  result.mean_nll = result.total_observation_weight > 0
      ? static_cast<double>(-weighted_log_likelihood/result.total_observation_weight) : 0;
  if (undefined_gradient) {
    std::fill(result.pose_gradient.begin(), result.pose_gradient.end(), 0);
    std::fill(result.model_gradient.begin(), result.model_gradient.end(), 0);
    std::fill(result.pose_information.begin(), result.pose_information.end(), 0);
  }
  for (const auto* array : {&result.pose_gradient, &result.model_gradient, &result.pose_information})
    for (double value : *array) smlm_likelihood_require(std::isfinite(value), "SMLM likelihood derivative overflow");
  return result;
}

SMLMLikelihoodFitResult fit_smlm_likelihood_rigid(
    const SMLMIndex& observations, const SMLMPointModel& model,
    const SMLMLikelihoodOptions& likelihood_options,
    const std::vector<double>& initial_pose,
    const SMLMLikelihoodFitOptions& fit_options) {
  smlm_likelihood_require(observations.get_number_of_localizations() > 0,
          "SMLM likelihood rigid fitting needs observations");
  smlm_likelihood_require(fit_options.max_iterations >= 0 && fit_options.max_backtracks > 0,
          "Invalid SMLM likelihood fit iteration limits");
  for (double limit : {fit_options.max_translation_step, fit_options.max_rotation_step,
                       fit_options.tolerance})
    smlm_likelihood_require(std::isfinite(limit) && limit > 0, "SMLM likelihood fit limits must be finite and positive");
  auto iteration_options = likelihood_options;
  iteration_options.compute_model_gradient = false;
  SMLMLikelihoodFitResult result;
  result.transform = smlm_likelihood_checked_pose(initial_pose);
  auto current = model.evaluate(observations, iteration_options, result.transform);
  smlm_likelihood_require(current.total_observation_weight > 0,
          "SMLM likelihood rigid fitting needs positive observation weight");
  result.initial_mean_nll = current.mean_nll;
  for (int iteration = 0; iteration < fit_options.max_iterations; ++iteration) {
    if (current.supported_localizations == 0 || !std::isfinite(current.mean_nll)) break;
    double gradient_norm = 0;
    for (double value : current.pose_gradient) gradient_norm = std::hypot(gradient_norm, value);
    if (gradient_norm <= fit_options.tolerance) { result.converged = true; break; }
    auto step = smlm_likelihood_descent_step(current);
    const double translation_norm = std::hypot(std::hypot(step[0], step[1]), step[2]);
    const double rotation_norm = std::hypot(std::hypot(step[3], step[4]), step[5]);
    // Scale the WHOLE descent vector together: independently clipping the two
    // blocks could destroy descent in coupled translation/rotation directions.
    double scale = 1;
    if (translation_norm > fit_options.max_translation_step)
      scale = std::min(scale, fit_options.max_translation_step/translation_norm);
    if (rotation_norm > fit_options.max_rotation_step)
      scale = std::min(scale, fit_options.max_rotation_step/rotation_norm);
    for (double& value : step) value *= scale;
    double directional_derivative = 0;
    for (int k = 0; k < 6; ++k) directional_derivative += step[k]*current.pose_gradient[k];
    if (!(directional_derivative < 0)) break;
    bool accepted = false;
    double alpha = 1;
    for (int backtrack = 0; backtrack < fit_options.max_backtracks; ++backtrack) {
      auto candidate_pose = updated_pose(result.transform, step, alpha);
      auto candidate = model.evaluate(observations, iteration_options, candidate_pose);
      if (std::isfinite(candidate.mean_nll) &&
          candidate.mean_nll <= current.mean_nll + 1e-4*alpha*directional_derivative) {
        result.transform = std::move(candidate_pose);
        current = std::move(candidate);
        ++result.iterations;
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
    if (alpha*translation_norm*scale <= fit_options.tolerance &&
        alpha*rotation_norm*scale <= fit_options.tolerance) {
      result.converged = current.supported_localizations > 0;
      break;
    }
  }
  result.likelihood = model.evaluate(observations, likelihood_options, result.transform);
  return result;
}

double smlm_aic(double log_likelihood, int free_parameters) {
  smlm_likelihood_require(free_parameters >= 0, "SMLM AIC free parameter count must be nonnegative");
  smlm_likelihood_require(std::isfinite(log_likelihood) || log_likelihood == smlm_likelihood_negative_infinity,
          "SMLM AIC log likelihood must be finite or negative infinity");
  return 2.0*free_parameters-2.0*log_likelihood;
}

double smlm_aicc(double log_likelihood, int localization_count, int free_parameters) {
  const double aic = smlm_aic(log_likelihood, free_parameters);
  smlm_likelihood_require(localization_count >= 0, "SMLM AICc localization count must be nonnegative");
  if (static_cast<long long>(localization_count) <= static_cast<long long>(free_parameters)+1)
    return std::numeric_limits<double>::infinity();
  const double parameters = free_parameters;
  return aic + 2*parameters*(parameters+1)/(localization_count-parameters-1);
}

IMPBFF_END_NAMESPACE
