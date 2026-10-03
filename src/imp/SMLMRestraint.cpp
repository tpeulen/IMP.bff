/** \file SMLMRestraint.cpp
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SMLMRestraint.h>
#include <IMP/core/XYZ.h>
#include <IMP/algebra/Vector3D.h>
#include <cmath>
#include <numeric>
#include <set>

IMPBFF_BEGIN_NAMESPACE

SMLMRestraint::SMLMRestraint(
    IMP::Model* model, const IMP::ParticlesTemp& emitter_particles,
    const SMLMIndex& observations, const SMLMLikelihoodOptions& options,
    const std::vector<double>& model_weights, bool sum_negative_log_likelihood,
    int leaf_size, std::string name)
    : IMP::Restraint(model, name), observations_(new SMLMIndex(observations)),
      options_(options), sum_negative_log_likelihood_(sum_negative_log_likelihood),
      leaf_size_(leaf_size) {
  initialize_emitters(emitter_particles);
  if (observations.get_number_of_localizations() == 0 ||
      !(std::accumulate(observations.get_weights().begin(),
                        observations.get_weights().end(), 0.0) > 0)) {
    IMP_THROW("SMLMRestraint needs observations with positive total weight", IMP::ValueException);
  }
  SMLMPointModel initial_model(get_emitter_coordinates(), model_weights, leaf_size_);
  model_weights_ = initial_model.get_weights();
  // Validate ROI, precision/background options and their ownership boundary
  // immediately, even when a score is not evaluated until after optimization.
  auto validation_options = options_;
  validation_options.compute_model_gradient = false;
  initial_model.evaluate(*observations_, validation_options);
}

SMLMRestraint::SMLMRestraint(
    IMP::Model* model, const IMP::ParticlesTemp& emitter_particles,
    const SMLMIndex& observations, const std::vector<double>& model_covariances,
    const std::vector<double>& model_weights, double cutoff_sigma,
    double background, std::string name)
    : IMP::Restraint(model, name), overlap_(new SMLMGaussianOverlap(observations)),
      model_covariances_(model_covariances), cutoff_sigma_(cutoff_sigma),
      background_(background) {
  initialize_emitters(emitter_particles);
  if (observations.get_number_of_localizations() == 0 ||
      !(std::accumulate(observations.get_weights().begin(),
                        observations.get_weights().end(), 0.0) > 0)) {
    IMP_THROW("SMLMRestraint needs observations with positive total weight", IMP::ValueException);
  }
  const auto coordinates = get_emitter_coordinates();
  // Reuse the existing native model-weight validator and normalization.
  SMLMPointModel initial_model(coordinates, model_weights);
  model_weights_ = initial_model.get_weights();
  // The native overlap implementation owns full covariance validation.
  overlap_->evaluate(coordinates, model_covariances_, model_weights_,
                     cutoff_sigma_, background_);
}

void SMLMRestraint::initialize_emitters(const IMP::ParticlesTemp& particles) {
  if (particles.empty()) {
    IMP_THROW("SMLMRestraint needs at least one emitter particle", IMP::ValueException);
  }
  std::set<IMP::ParticleIndex> unique;
  for (IMP::Particle* particle : particles) {
    if (!particle || particle->get_model() != get_model()) {
      IMP_THROW("SMLM emitter particles must belong to the restraint's Model", IMP::ValueException);
    }
    if (!IMP::core::XYZ::get_is_setup(particle)) {
      IMP_THROW("SMLM emitter particles must have XYZ coordinates", IMP::ValueException);
    }
    if (!unique.insert(particle->get_index()).second) {
      IMP_THROW("SMLM emitter particles must be unique", IMP::ValueException);
    }
    emitter_indexes_.push_back(particle->get_index());
  }
}

std::vector<double> SMLMRestraint::get_emitter_coordinates() const {
  std::vector<double> coordinates(3*emitter_indexes_.size());
  for (std::size_t i = 0; i < emitter_indexes_.size(); ++i) {
    const auto point = IMP::core::XYZ(get_model(), emitter_indexes_[i]).get_coordinates();
    for (int axis = 0; axis < 3; ++axis) coordinates[3*i+axis] = point[axis];
  }
  return coordinates;
}

int SMLMRestraint::get_number_of_localizations() const {
  return observations_ ? observations_->get_number_of_localizations()
                       : overlap_->get_number_of_localizations();
}

std::string SMLMRestraint::get_score_mode() const {
  return observations_ ? "forward_likelihood" : "gaussian_overlap";
}

double SMLMRestraint::unprotected_evaluate(IMP::DerivativeAccumulator* accumulator) const {
  const auto coordinates = get_emitter_coordinates();
  std::vector<double> gradient;
  double score, derivative_scale = 1;
  if (observations_) {
    SMLMPointModel model(coordinates, model_weights_, leaf_size_);
    auto evaluation_options = options_;
    // An IMP derivative request always receives analytic particle gradients,
    // regardless of the caller's diagnostic-only core option at construction.
    evaluation_options.compute_model_gradient = accumulator != nullptr;
    auto result = model.evaluate(*observations_, evaluation_options);
    score = sum_negative_log_likelihood_ ? -result.weighted_log_likelihood : result.mean_nll;
    if (sum_negative_log_likelihood_) derivative_scale = result.total_observation_weight;
    gradient = std::move(result.model_gradient);
  } else {
    auto result = overlap_->evaluate(coordinates, model_covariances_, model_weights_,
                                     cutoff_sigma_, background_);
    score = result.score;
    gradient = std::move(result.gradient);
  }
  if (accumulator) {
    for (std::size_t i = 0; i < emitter_indexes_.size(); ++i) {
      const IMP::algebra::Vector3D derivative(
          derivative_scale*gradient[3*i], derivative_scale*gradient[3*i+1],
          derivative_scale*gradient[3*i+2]);
      for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(derivative[axis])) {
          IMP_THROW("SMLM restraint particle derivative overflow", IMP::ValueException);
        }
      // IMP's accumulator includes restraint and enclosing RestraintSet weights.
      // Applying any of those again here would double-scale the gradient.
      IMP::core::XYZ(get_model(), emitter_indexes_[i]).add_to_derivatives(derivative, *accumulator);
    }
  }
  return score;
}

IMP::ModelObjectsTemp SMLMRestraint::do_get_inputs() const {
  IMP::ModelObjectsTemp inputs;
  for (IMP::ParticleIndex index : emitter_indexes_)
    inputs.push_back(get_model()->get_particle(index));
  return inputs;
}

IMPBFF_END_NAMESPACE
