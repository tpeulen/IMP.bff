/**
 * \file MaxEntSpectrum.cpp
 * \brief Maximum-entropy amplitudes over a decay basis, as a graph node.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/MaxEntSpectrum.h>

#include <IMP/bff/GraphPort.h>
#include <IMP/bff/internal/MaxEntQp.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {

const double kMinProb = 1e-12;

//! The reduced chi-square of `model` against the data, weighted by the model
//! (floored at one count, as the data weights are): the second score.
double pearson_reduced(const std::vector<double>& model, const std::vector<double>& data,
                       const std::vector<bool>& active) {
  double sum = 0.0;
  int m = 0;
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (!active[i]) continue;
    const double r = model[i] - data[i];
    sum += r * r / std::max(model[i], 1.0);
    ++m;
  }
  return m > 0 ? sum / m : 0.0;
}

}  // namespace

MaxEntSpectrum::MaxEntSpectrum(const std::string& name) : GraphNode(name) {}

void MaxEntSpectrum::set_grouping(const std::string& grouping, int donor_components) {
  if (grouping != "lifetime" && grouping != "fret") {
    throw std::domain_error("MaxEntSpectrum '" + get_name() + "': grouping is 'lifetime' or 'fret', not '" +
                            grouping + "'");
  }
  if (donor_components < 1) {
    throw std::domain_error("MaxEntSpectrum '" + get_name() + "': at least one donor component");
  }
  fret_ = grouping == "fret";
  donor_components_ = donor_components;
  set_valid(false);
}

void MaxEntSpectrum::bind_dataset(const std::string& role, const FitDataset& dataset) {
  if (role != "data") {
    throw std::domain_error("MaxEntSpectrum '" + get_name() + "' binds 'data', not '" + role + "'");
  }
  data_ = dataset.get_values();
  // The weights at the data, floored at one count: an inversion is solved
  // once per evaluation against fixed weights.
  std::vector<double> floor(data_.size());
  for (std::size_t i = 0; i < data_.size(); ++i) floor[i] = std::max(data_[i], 1.0);
  double* view = nullptr;
  int n = 0;
  dataset.variance(floor, &view, &n);
  sigma_.assign(static_cast<std::size_t>(std::max(n, 0)), 1.0);
  for (int i = 0; i < n; ++i) sigma_[static_cast<std::size_t>(i)] = std::sqrt(std::max(view[i], 1e-300));
  std::free(view);
  const std::vector<double>& mask = dataset.get_mask();
  active_.assign(data_.size(), true);
  if (mask.size() == data_.size()) {
    for (std::size_t i = 0; i < mask.size(); ++i) active_[i] = mask[i] != 0.0;
  }
  set_valid(false);
}

void MaxEntSpectrum::evaluate() {
  const std::string where = "MaxEntSpectrum '" + get_name() + "'";
  if (data_.empty()) throw std::domain_error(where + " needs its data bound");
  const auto vector_in = [&](const char* key) -> const std::vector<double>& {
    const std::shared_ptr<GraphPort> port = get_input_port(key);
    if (!port) throw std::domain_error(where + " has no input '" + key + "'");
    return port->get_values_ref();
  };
  const auto scalar_in = [&](const char* key) {
    const std::shared_ptr<GraphPort> port = get_input_port(key);
    if (!port) throw std::domain_error(where + " has no input '" + key + "'");
    return port->get_value();
  };
  const std::vector<double>& basis = vector_in("basis");
  const std::vector<double>& spectrum = vector_in("spectrum");
  const std::vector<double>& grid = vector_in("grid");
  const std::vector<double>& response = vector_in("response");
  const std::size_t n = data_.size();
  const std::size_t species = spectrum.size() / 2;
  if (species == 0 || basis.size() != n * species) {
    throw std::domain_error(where + ": the basis is not the data's channels by the spectrum's species");
  }
  if (response.size() != n) throw std::domain_error(where + ": the response is not as long as the data");

  // The columns: one per species, or, for FRET, one per distance.
  std::size_t columns = species;
  std::size_t per = 1, distances = species;
  if (fret_) {
    per = static_cast<std::size_t>(donor_components_);
    if (species <= per || (species - per) % per != 0) {
      throw std::domain_error(where + ": the FRET spectrum is not distances x donor components plus the donor");
    }
    distances = (species - per) / per;
    columns = distances;
  }
  std::vector<double> design(n * columns, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double* row = &basis[i * species];
    double* out = &design[i * columns];
    if (!fret_) {
      std::copy(row, row + species, out);
      continue;
    }
    double donor_only = 0.0;
    for (std::size_t j = 0; j < per; ++j) donor_only += spectrum[2 * (distances * per + j)] * row[distances * per + j];
    for (std::size_t d = 0; d < distances; ++d) {
      double quenched = donor_only;
      for (std::size_t j = 0; j < per; ++j) quenched += spectrum[2 * (d * per + j)] * row[d * per + j];
      out[d] = quenched;
    }
  }

  const double background = scalar_in("background");
  const double scatter = scalar_in("scatter");
  std::vector<double> additive(n);
  for (std::size_t i = 0; i < n; ++i) additive[i] = background + scatter * response[i];

  // One scale for every column, so the amplitudes are fractions: the entropy
  // is measured against a normalised prior, and it only regularises
  // amplitudes of that order. With it, a uniform distribution already carries
  // the fluorescence counts the data leave over the additive part.
  double counts = 0.0, column_total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!active_[i]) continue;
    counts += std::max(data_[i] - additive[i], 0.0);
    for (std::size_t c = 0; c < columns; ++c) column_total += design[i * columns + c];
  }
  const double scale = column_total > 0.0 && counts > 0.0
                           ? counts * static_cast<double>(columns) / column_total : 1.0;
  for (double& v : design) v *= scale;

  // The weighted normal equations over the active channels, scaled as
  // tttrlib's MEM TCSPC analysis scales them (chi-square per point).
  std::vector<double> weighted, target;
  weighted.reserve(n * columns);
  for (std::size_t i = 0; i < n; ++i) {
    if (!active_[i]) continue;
    for (std::size_t c = 0; c < columns; ++c) weighted.push_back(design[i * columns + c] / sigma_[i]);
    target.push_back((data_[i] - additive[i]) / sigma_[i]);
  }
  const int m = static_cast<int>(target.size());
  if (m == 0) throw std::domain_error(where + ": no active channel");
  const int k = static_cast<int>(columns);
  std::vector<double> H, g0;
  double constant = 0.0;
  tttrlib::build_normal_equations(weighted, target, {}, m, k, H, g0, constant);
  for (double& v : H) v /= m;
  for (double& v : g0) v /= m;
  constant /= m;
  // The prior: uniform, or the `prior` input where one is given (one weight per
  // column, floored at the positivity bound, normalised).
  std::vector<double> prior(columns, 1.0 / static_cast<double>(columns));
  if (const std::shared_ptr<GraphPort> given = get_input_port("prior")) {
    const std::vector<double>& weights = given->get_values_ref();
    if (weights.size() == columns) {
      double total = 0.0;
      for (std::size_t c = 0; c < columns; ++c) {
        prior[c] = weights[c] > 0.0 ? weights[c] : kMinProb;
        total += prior[c];
      }
      for (double& v : prior) v /= total;
    } else if (!(weights.size() <= 1)) {
      throw std::domain_error(where + ": the prior has " + std::to_string(weights.size()) +
                              " weights for " + std::to_string(columns) + " grid points");
    }
  }

  double nu = std::pow(10.0, scalar_in("log10_nu"));
  tttrlib::MaxEntResult result;
  if (target_chisq_ > 0.0) {
    const tttrlib::MemTargetChisqResult searched = tttrlib::run_mem_target_chisq(
        H, g0, prior, constant, target_chisq_, nu, std::max(max_iter_, 1000), 1e-2, tol_, kMinProb);
    result = searched.result;
    nu = searched.nu;
  } else {
    result = tttrlib::run_mem(H, g0, prior, constant, nu, max_iter_, tol_, kMinProb);
  }
  if (!result.success || result.p.size() != columns) {
    throw std::domain_error(where + ": the maximum-entropy solve failed");
  }
  amplitudes_ = result.p;

  std::vector<double> out(spectrum);
  if (!fret_) {
    for (std::size_t s = 0; s < species; ++s) out[2 * s] = scale * amplitudes_[s] * spectrum[2 * s];
  } else {
    double total = 0.0;
    for (std::size_t d = 0; d < distances; ++d) {
      total += amplitudes_[d];
      for (std::size_t j = 0; j < per; ++j)
        out[2 * (d * per + j)] = scale * amplitudes_[d] * spectrum[2 * (d * per + j)];
    }
    for (std::size_t j = 0; j < per; ++j)
      out[2 * (distances * per + j)] = scale * total * spectrum[2 * (distances * per + j)];
  }
  const std::shared_ptr<GraphPort> spectrum_out = get_output_port(get_name());
  if (!spectrum_out) throw std::domain_error(where + " writes to the output keyed by its own name");
  spectrum_out->set_sanitize(false);
  spectrum_out->set_value_vector(out);

  if (const std::shared_ptr<GraphPort> port = get_output_port("amplitudes")) {
    std::vector<double> counts(amplitudes_);
    for (double& v : counts) v *= scale;
    port->set_value_vector(counts);
  }
  if (const std::shared_ptr<GraphPort> port = get_output_port("distribution")) {
    std::vector<double> pairs(2 * columns);
    for (std::size_t c = 0; c < columns; ++c) {
      pairs[2 * c] = amplitudes_[c];
      pairs[2 * c + 1] = 2 * c + 1 < grid.size() ? grid[2 * c + 1] : 0.0;
    }
    port->set_value_vector(pairs);
  }
  // The reduced chi-square the programme minimised, at the solution.
  if (const std::shared_ptr<GraphPort> port = get_output_port("chisq")) port->set_value(result.chisq);
  if (const std::shared_ptr<GraphPort> port = get_output_port("entropy")) port->set_value(result.S);
  if (const std::shared_ptr<GraphPort> port = get_output_port("nu")) port->set_value(nu);
  if (const std::shared_ptr<GraphPort> port = get_output_port("converged")) {
    port->set_value(result.converged ? 1.0 : 0.0);
  }
  if (const std::shared_ptr<GraphPort> port = get_output_port("chisq_pearson")) {
    std::vector<double> model(additive);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t c = 0; c < columns; ++c) model[i] += design[i * columns + c] * amplitudes_[c];
    port->set_value(pearson_reduced(model, data_, active_));
  }
  set_valid(true);
}

std::string MaxEntSpectrum::get_node_type() const { return "MaxEntSpectrum"; }

void MaxEntSpectrum::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  const std::string grouping = config.has("grouping") ? config.get_string("grouping") : get_grouping();
  const int donor = config.has("donor_components") ? config.get_int("donor_components") : donor_components_;
  set_grouping(grouping, donor);
  if (config.has("max_iter")) max_iter_ = config.get_int("max_iter");
  if (config.has("tol")) tol_ = config.get_double("tol");
  if (config.has("target_chisq")) target_chisq_ = config.get_double("target_chisq");
  config.apply_common(*this);
  config.require_all_used();
}

IMPBFF_END_NAMESPACE
