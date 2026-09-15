/**
 *  \file IMP/bff/MCMCSampler.cpp
 *  \brief ChiSurf's MCMC samplers over a bff GraphPort/GraphNode model, in C++.
 *
 *  The implementation notes below name the chisurf function each piece is
 *  ported from; the algorithms are those functions, not re-derivations of
 *  them. Numeric helpers (Cholesky, a Jacobi eigensolver for the walker
 *  degeneracy test, a JSON object reader for the port prior specs) live in
 *  the sampler_detail namespace because this file is compiled into IMP's
 *  unity build, where two anonymous namespaces are the same namespace and
 *  same-named helpers would collide.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 */
#include <IMP/bff/MCMCSampler.h>

#include <IMP/bff/InferenceFactorGraph.h>
#include <IMP/bff/NutsKernel.h>
#include <IMP/bff/Registry.h>
#include <IMP/bff/SamplerKernels.h>
#include <IMP/bff/Sampling.h>
#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/ordered_map.h>
#include <IMP/bff/GraphNode.h>
#include <IMP/bff/GraphPort.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <utility>

IMPBFF_BEGIN_NAMESPACE

namespace sampler_detail {

//! Skip whitespace; \return the character at the position (0 at the end).
char json_peek(const std::string& s, std::size_t& i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' ||
                          s[i] == '\r'))
    ++i;
  return i < s.size() ? s[i] : '\0';
}

//! Read a JSON string (the opening quote is at i); \return the contents.
std::string json_string(const std::string& s, std::size_t& i) {
  std::string out;
  ++i;  // the opening quote
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\' && i + 1 < s.size()) {
      ++i;
      if (s[i] == 'n') {
        out += '\n';
      } else if (s[i] == 't') {
        out += '\t';
      } else {
        out += s[i];
      }
    } else {
      out += s[i];
    }
    ++i;
  }
  ++i;  // the closing quote
  return out;
}

}  // namespace sampler_detail

// ---------------------------------------------------------------- lifecycle

MCMCSampler::MCMCSampler(const std::string& algorithm, unsigned int seed)
    : algorithm_("stretch"), seed_(seed), rng_(seed) {
  set_algorithm(algorithm);
}

MCMCSampler::~MCMCSampler() {}

void MCMCSampler::set_algorithm(const std::string& algorithm) {
  // Every name and alias the registry holds for category `sampler` (PRD-147): no list here.
  typedef nlohmann::basic_json<nlohmann::ordered_map> FacadeJson;
  const FacadeJson entries = FacadeJson::parse(registry_category_json("sampler"));
  std::string canonical;
  if (entries.contains(algorithm)) canonical = algorithm;
  for (auto it = entries.begin(); it != entries.end() && canonical.empty(); ++it)
    if (it.value().contains("aliases"))
      for (const auto& a : it.value()["aliases"])
        if (a.is_string() && a.get<std::string>() == algorithm) canonical = it.key();
  if (canonical.empty()) {
    std::string known;
    for (auto it = entries.begin(); it != entries.end(); ++it) known += (known.empty() ? "'" : ", '") + it.key() + "'";
    throw MCMCSamplerConfigurationError("unknown sampler algorithm '" + algorithm + "'; registered: " + known);
  }
  if (canonical != algorithm_ || !kernel_) {
    algorithm_ = canonical;
    initialized_ = false;  // the ensemble belongs to the old algorithm
    kernel_.reset();
  }
  read_sampler_entry();
}

void MCMCSampler::read_sampler_entry() {
  typedef nlohmann::basic_json<nlohmann::ordered_map> FacadeJson;
  const FacadeJson entry = FacadeJson::parse(registry_category_json("sampler"))[algorithm_];
  population_ = entry.value("kind", std::string("chain")) == "chain" ? std::string("single") : entry.value("population", std::string("walkers"));
  acceptance_mode_ = entry.value("acceptance_rate", std::string("proposals"));
  restores_parameters_ = entry.value("restores_parameters", false);
  const FacadeJson w = entry.contains("default_warmup") ? entry["default_warmup"] : FacadeJson::object();
  warmup_rule_ = w.value("rule", std::string("fixed"));
  warmup_value_ = w.value("value", 0);
  warmup_divisor_ = w.value("divisor", 1);
  warmup_min_ = w.value("min", 0);
  warmup_max_ = w.value("max", 0);
  option_names_.clear();
  if (entry.contains("params_schema") && entry["params_schema"].contains("properties"))
    for (auto it = entry["params_schema"]["properties"].begin(); it != entry["params_schema"]["properties"].end(); ++it)
      option_names_.push_back(it.key());
}

const std::string& MCMCSampler::get_algorithm() const { return algorithm_; }

void MCMCSampler::set_seed(unsigned int seed) {
  seed_ = seed;
  rng_.seed(seed);
}

unsigned int MCMCSampler::get_seed() const { return seed_; }

// --------------------------------------------------------------- parameters

void MCMCSampler::set_parameter_ports(
    const std::vector<std::shared_ptr<GraphPort> >& parameters) {
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    if (!parameters[i])
      throw MCMCSamplerConfigurationError("parameter " + std::to_string(i) +
                               " is a null port");
    if (parameters[i]->get_fixed())
      throw MCMCSamplerConfigurationError(
          "parameter '" + parameters[i]->get_name() +
          "' is fixed; a fixed port cannot be sampled (chisurf samples "
          "the free parameters)");
  }
  parameters_ = parameters;
  ndim_ = static_cast<unsigned int>(parameters.size());
  initialized_ = false;
}

std::vector<std::shared_ptr<GraphPort> > MCMCSampler::get_parameter_ports() const {
  return parameters_;
}

std::vector<std::string> MCMCSampler::get_parameter_names() const {
  if (!names_.empty()) return names_;
  std::vector<std::string> out;
  for (std::size_t i = 0; i < parameters_.size(); ++i)
    out.push_back(parameters_[i]->get_name().empty()
                      ? "x" + std::to_string(i)
                      : parameters_[i]->get_name());
  return out;
}

void MCMCSampler::set_initial_values(const std::vector<double>& values) {
  initial_values_ = values;
  if (ndim_ == 0) ndim_ = static_cast<unsigned int>(values.size());
  initialized_ = false;
}

std::vector<double> MCMCSampler::get_initial_values() const { return initial_values_; }

void MCMCSampler::set_bounds(const std::vector<double>& lower,
                         const std::vector<double>& upper) {
  if (lower.size() != upper.size())
    throw MCMCSamplerConfigurationError(
        "set_bounds: lower and upper must have the same length");
  if (ndim_ != 0 && lower.size() != ndim_)
    throw MCMCSamplerConfigurationError("set_bounds: expected " +
                             std::to_string(ndim_) + " bounds, got " +
                             std::to_string(lower.size()));
  lower_ = lower;
  upper_ = upper;
  bounds_explicit_ = true;
  initialized_ = false;
}

// ---------------------------------------------------------------- objective

void MCMCSampler::set_objective(std::shared_ptr<GraphNode> node,
                            const std::string& output_port) {
  if (!node)
    throw MCMCSamplerConfigurationError("set_objective: the node is a null pointer");
  if (!node->get_output_port(output_port))
    throw MCMCSamplerConfigurationError(
        "set_objective: node '" + node->get_name() +
        "' has no output port '" + output_port + "'");
  objective_node_ = node;
  output_port_ = node->get_output_port(output_port);
  objective_function_ = nullptr;
  initialized_ = false;
}

std::shared_ptr<GraphNode> MCMCSampler::get_objective() const { return objective_node_; }

void MCMCSampler::set_output_is_log_likelihood(bool v) {
  output_is_log_likelihood_ = v;
}

bool MCMCSampler::get_output_is_log_likelihood() const {
  return output_is_log_likelihood_;
}

void MCMCSampler::set_objective_function(
    std::function<double(const std::vector<double>&)> objective) {
  if (!objective)
    throw MCMCSamplerConfigurationError("set_objective_function: a null function");
  objective_function_ = objective;
  objective_node_.reset();
  output_port_.reset();
  initialized_ = false;
}

void MCMCSampler::set_objective_function_with_gradient(
    std::function<double(const std::vector<double>&, std::vector<double>&)> objective) {
  if (!objective) throw MCMCSamplerConfigurationError("set_objective_function_with_gradient: a null function");
  objective_gradient_function_ = objective;
  objective_function_ = [objective](const std::vector<double>& x) {
    std::vector<double> g(x.size());
    return objective(x, g);
  };
  initialized_ = false;
}

bool MCMCSampler::has_objective() const {
  return objective_node_ != nullptr || objective_function_ != nullptr;
}

// ----------------------------------------------------------------- blocking

void MCMCSampler::set_factor_graph(InferenceFactorGraph* graph) {
  factor_graph_ = graph;
  initialized_ = false;
}

InferenceFactorGraph* MCMCSampler::get_factor_graph() const {
  return factor_graph_;
}

void MCMCSampler::set_blocks(const std::vector<int>& flat_indices,
                         const std::vector<int>& block_sizes) {
  explicit_blocks_.clear();
  std::size_t offset = 0;
  for (std::size_t b = 0; b < block_sizes.size(); ++b) {
    const int size = block_sizes[b];
    if (size <= 0)
      throw MCMCSamplerConfigurationError("set_blocks: block sizes must be positive");
    if (offset + static_cast<std::size_t>(size) > flat_indices.size())
      throw MCMCSamplerConfigurationError("set_blocks: more block entries than indices");
    std::vector<int> block(
        flat_indices.begin() + offset,
        flat_indices.begin() + offset + static_cast<std::size_t>(size));
    for (std::size_t i = 0; i < block.size(); ++i) {
      if (ndim_ != 0 &&
          (block[i] < 0 ||
           block[i] >= static_cast<int>(ndim_)))
        throw MCMCSamplerConfigurationError("set_blocks: index out of range");
    }
    explicit_blocks_.push_back(block);
    offset += static_cast<std::size_t>(size);
  }
  initialized_ = false;
}

std::vector<std::vector<int> > MCMCSampler::get_blocks() const {
  if (const BlockedMetropolisKernel* k = dynamic_cast<const BlockedMetropolisKernel*>(kernel_.get())) return k->blocks();
  return std::vector<std::vector<int> >();
}

std::vector<int> MCMCSampler::get_block_sizes() const {
  std::vector<int> out;
  for (const auto& b : get_blocks()) out.push_back(static_cast<int>(b.size()));
  return out;
}

// ---------------------------------------------------------------- tunables

void MCMCSampler::set_number_of_walkers(int n) {
  n_walkers_setting_ = n;
  initialized_ = false;
}

int MCMCSampler::get_number_of_walkers() const {
  if (population_ != "walkers") return 1;
  if (n_walkers_setting_ > 0) return n_walkers_setting_;
  const int d = static_cast<int>(ndim_);
  return std::max(2 * d + 2, 10);
}

void MCMCSampler::set_stretch_scale(double a) {
  if (!(a > 1.0))
    throw MCMCSamplerConfigurationError("the stretch scale must exceed one");
  stretch_scale_ = a;
}

double MCMCSampler::get_stretch_scale() const { return stretch_scale_; }

double MCMCSampler::get_slice_mu() const {
  if (const EnsembleSliceKernel* k = dynamic_cast<const EnsembleSliceKernel*>(kernel_.get())) return k->mu();
  return slice_mu_;
}

void MCMCSampler::set_slice_mu(double mu) {
  if (!(mu > 0.0))
    throw MCMCSamplerConfigurationError(
        "the slice direction scale must be positive");
  slice_mu_ = mu;
  // An explicit scale is a decision; tuning would overwrite it.
  slice_tuning_ = false;
}

void MCMCSampler::set_slice_max_steps(int n) {
  slice_max_steps_ = n > 0 ? n : 10000;
}

int MCMCSampler::get_slice_max_steps() const { return slice_max_steps_; }

long MCMCSampler::get_slice_truncations() const {
  if (const EnsembleSliceKernel* k = dynamic_cast<const EnsembleSliceKernel*>(kernel_.get())) return k->truncations();
  return 0;
}

void MCMCSampler::set_live_dangerously(bool v) { live_dangerously_ = v; }

bool MCMCSampler::get_live_dangerously() const { return live_dangerously_; }

void MCMCSampler::set_number_of_chains(int n) {
  n_chains_setting_ = n;
  initialized_ = false;
}

int MCMCSampler::get_number_of_chains() const {
  if (population_ != "chains") return 1;
  int n = n_chains_setting_ > 0
              ? n_chains_setting_
              : std::max(8, 2 * static_cast<int>(ndim_));
  return std::max(4, n);
}

void MCMCSampler::set_jitter(double jitter) {
  if (!(jitter >= 0.0))
    throw MCMCSamplerConfigurationError("the jitter must not be negative");
  jitter_ = jitter;
}

double MCMCSampler::get_jitter() const { return jitter_; }

void MCMCSampler::set_snooker(double fraction) {
  if (!(fraction >= 0.0 && fraction <= 1.0))
    throw MCMCSamplerConfigurationError("the snooker fraction must lie in [0, 1]");
  snooker_ = fraction;
}

double MCMCSampler::get_snooker() const { return snooker_; }

void MCMCSampler::set_step_size(double step_size) {
  if (!(step_size > 0.0))
    throw MCMCSamplerConfigurationError("the step size must be positive");
  step_size_ = step_size;
}

double MCMCSampler::get_step_size() const { return step_size_; }

void MCMCSampler::set_proposal_covariance(
    const std::vector<std::vector<double> >& cov) {
  if (!cov.empty()) {
    const std::size_t n = cov.size();
    for (const std::vector<double>& row : cov) {
      if (row.size() != n) {
        throw MCMCSamplerConfigurationError(
            "set_proposal_covariance: the covariance must be square");
      }
    }
  }
  proposal_covariance_ = cov;
}

std::vector<std::vector<double> > MCMCSampler::get_proposal_covariance() const {
  return proposal_covariance_;
}

void MCMCSampler::set_n_adapt(int n) { n_adapt_setting_ = n; }

int MCMCSampler::get_n_adapt() const { return n_adapt_setting_; }

void MCMCSampler::set_temp(double temp) {
  if (!(temp > 0.0))
    throw MCMCSamplerConfigurationError("the temperature must be positive");
  temp_ = temp;
}

double MCMCSampler::get_temp() const { return temp_; }

void MCMCSampler::set_chi2max(double chi2max) { chi2max_ = chi2max; }

double MCMCSampler::get_chi2max() const { return chi2max_; }

void MCMCSampler::set_walker_start_std(double std) {
  if (!(std > 0.0))
    throw MCMCSamplerConfigurationError("the walker start spread must be positive");
  walker_start_std_ = std;
}

double MCMCSampler::get_walker_start_std() const { return walker_start_std_; }

void MCMCSampler::set_walker_start(
    const std::vector<std::vector<double> >& start) {
  walker_start_override_ = start;
  initialized_ = false;
}

std::vector<std::vector<double> > MCMCSampler::get_walker_start() const {
  return walker_start_override_;
}

void MCMCSampler::set_observer(std::function<void(int, int)> observer) {
  observer_ = observer;
}

// --------------------------------------------------------------- internals

double MCMCSampler::PriorSpec::get(const std::string& name,
                               double fallback) const {
  for (std::size_t i = 0; i < numbers.size(); ++i)
    if (numbers[i].first == name) return numbers[i].second;
  return fallback;
}

//! chisurf priors.py, kind for kind; unknown kinds contribute nothing.
double MCMCSampler::prior_lnpdf(const PriorSpec& spec, double x) {
  const double inf = std::numeric_limits<double>::infinity();
  const double k2pi = 6.283185307179586476925286766559;
  const double half_ln_2pi = 0.918938533204672741780329736406;
  if (spec.kind == "uniform") {
    const double lb = spec.get("lb", -inf), ub = spec.get("ub", inf);
    if (x < lb || x > ub) return -inf;
    const double width = ub - lb;
    if (std::isfinite(width) && width > 0.0) return -std::log(width);
    return 0.0;
  }
  if (spec.kind == "normal") {
    const double mu = spec.get("mu", 0.0), sigma = spec.get("sigma", 1.0);
    const double z = (x - mu) / sigma;
    return -0.5 * z * z - std::log(sigma) - half_ln_2pi;
  }
  if (spec.kind == "truncated_normal") {
    const double lb = spec.get("lb", -inf), ub = spec.get("ub", inf);
    if (x < lb || x > ub) return -inf;
    const double mu = spec.get("mu", 0.0), sigma = spec.get("sigma", 1.0);
    const double z = (x - mu) / sigma;
    return -0.5 * z * z - std::log(sigma) - half_ln_2pi;
  }
  if (spec.kind == "half_normal") {
    const double sigma = spec.get("sigma", 1.0), loc = spec.get("loc", 0.0);
    if (x < loc) return -inf;
    const double z = (x - loc) / sigma;
    return -0.5 * z * z - std::log(sigma) + 0.5 * std::log(2.0 / M_PI);
  }
  if (spec.kind == "lognormal") {
    const double mu = spec.get("mu", 0.0), sigma = spec.get("sigma", 1.0);
    if (x <= 0.0) return -inf;
    const double lx = std::log(x);
    const double z = (lx - mu) / sigma;
    return -0.5 * z * z - lx - std::log(sigma) - half_ln_2pi;
  }
  if (spec.kind == "exponential") {
    const double scale = spec.get("scale", 1.0), loc = spec.get("loc", 0.0);
    if (x < loc) return -inf;
    return -(x - loc) / scale - std::log(scale);
  }
  if (spec.kind == "gamma") {
    const double alpha = spec.get("alpha", 1.0), beta = spec.get("beta", 1.0),
                 loc = spec.get("loc", 0.0);
    const double t = x - loc;
    if (t <= 0.0) return -inf;
    return (alpha - 1.0) * std::log(t) - beta * t + alpha * std::log(beta) -
           std::lgamma(alpha);
  }
  if (spec.kind == "beta") {
    const double alpha = spec.get("alpha", 1.0), beta = spec.get("beta", 1.0);
    if (x <= 0.0 || x >= 1.0) return -inf;
    const double log_b =
        std::lgamma(alpha) + std::lgamma(beta) - std::lgamma(alpha + beta);
    return (alpha - 1.0) * std::log(x) + (beta - 1.0) * std::log1p(-x) - log_b;
  }
  return 0.0;  // an unrecognised kind is no prior (prior_from_state -> None)
}

MCMCSampler::PriorSpec MCMCSampler::parse_prior(const std::string& json) {
  PriorSpec spec;
  if (json.empty()) return spec;
  std::size_t i = 0;
  if (sampler_detail::json_peek(json, i) != '{') return spec;
  ++i;
  while (true) {
    char c = sampler_detail::json_peek(json, i);
    if (c == '\0') return PriorSpec();  // malformed: no prior
    if (c == '}') return spec;
    if (c == ',') {
      ++i;
      continue;
    }
    if (c != '"') return PriorSpec();
    const std::string key = sampler_detail::json_string(json, i);
    if (sampler_detail::json_peek(json, i) != ':') return PriorSpec();
    ++i;
    c = sampler_detail::json_peek(json, i);
    if (c == '"') {
      const std::string value = sampler_detail::json_string(json, i);
      if (key == "kind") spec.kind = value;
    } else if (c == 'n' || c == 't' || c == 'f') {
      // null / true / false: skipped, as nothing numeric is keyed there
      while (i < json.size() && json[i] != ',' && json[i] != '}') ++i;
    } else {
      const char* begin = json.c_str() + i;
      char* end = nullptr;
      const double value = std::strtod(begin, &end);
      if (end == begin) return PriorSpec();
      i += static_cast<std::size_t>(end - begin);
      spec.numbers.push_back(std::make_pair(key, value));
    }
  }
}

void MCMCSampler::configure_from_ports() {
  if (parameters_.empty()) {
    // Plain-vector mode: no ports, no priors, bounds only when explicit.
    if (!bounds_explicit_) {
      lower_.assign(ndim_, -std::numeric_limits<double>::infinity());
      upper_.assign(ndim_, std::numeric_limits<double>::infinity());
    }
    priors_.assign(ndim_, PriorSpec());
    names_.clear();
    for (unsigned int i = 0; i < ndim_; ++i)
      names_.push_back("x" + std::to_string(i));
    if (initial_values_.empty()) initial_values_.assign(ndim_, 0.0);
    return;
  }
  if (initial_values_.empty()) {
    initial_values_.resize(ndim_);
    for (unsigned int i = 0; i < ndim_; ++i)
      initial_values_[i] = parameters_[i]->get_value();
  }
  if (!bounds_explicit_) {
    // A port that enforces bounds gives its box; anything else is
    // unbounded (NaN stored on a port stands for no bound, as (nan, nan)
    // is what the Python surface reports for enforcement off).
    lower_.resize(ndim_);
    upper_.resize(ndim_);
    for (unsigned int i = 0; i < ndim_; ++i) {
      const std::shared_ptr<GraphPort>& p = parameters_[i];
      double lb = -std::numeric_limits<double>::infinity();
      double ub = std::numeric_limits<double>::infinity();
      if (p->get_is_bounded()) {
        if (std::isnan(p->get_lower_bound()))
          lb = -std::numeric_limits<double>::infinity();
        else
          lb = p->get_lower_bound();
        if (std::isnan(p->get_upper_bound()))
          ub = std::numeric_limits<double>::infinity();
        else
          ub = p->get_upper_bound();
      }
      lower_[i] = lb;
      upper_[i] = ub;
    }
  }
  priors_.resize(ndim_);
  for (unsigned int i = 0; i < ndim_; ++i)
    priors_[i] = parse_prior(parameters_[i]->get_prior());
  names_.clear();
  for (unsigned int i = 0; i < ndim_; ++i) {
    const std::string& n = parameters_[i]->get_name();
    names_.push_back(n.empty() ? "x" + std::to_string(i) : n);
  }
}

//! chisurf fit.lnprior with explicit bounds: the box first, then the priors.
double MCMCSampler::log_prior(const std::vector<double>& x) const {
  const double inf = std::numeric_limits<double>::infinity();
  for (unsigned int i = 0; i < ndim_; ++i) {
    if (x[i] < lower_[i] || x[i] > upper_[i]) return -inf;
  }
  double lp = 0.0;
  for (unsigned int i = 0; i < ndim_; ++i) {
    if (priors_[i].empty()) continue;
    lp += prior_lnpdf(priors_[i], x[i]);
    if (!std::isfinite(lp)) return -inf;
  }
  return lp;
}

//! chisurf fit.lnprob_parts: the prior short-circuits the evaluation.
MCMCSampler::Parts MCMCSampler::evaluate(const std::vector<double>& x) {
  ++n_evaluations_;
  const double inf = std::numeric_limits<double>::infinity();
  Parts parts;
  parts.lnprior = log_prior(x);
  if (!std::isfinite(parts.lnprior)) {
    parts.lnpost = -inf;
    parts.chi2 = inf;
    return parts;
  }
  double lnlike;
  if (objective_function_) {
    lnlike = objective_function_(x);
    if (std::isnan(lnlike))
      throw MCMCSamplerConfigurationError("the log-probability returned NaN");
    parts.chi2 = -2.0 * lnlike;
  } else {
    for (unsigned int i = 0; i < ndim_; ++i)
      parameters_[i]->set_value(x[i]);
    objective_node_->update();
    const double value = output_port_->get_value();
    if (std::isnan(value))
      throw MCMCSamplerConfigurationError("the log-probability returned NaN");
    if (output_is_log_likelihood_) {
      lnlike = value;
      parts.chi2 = -2.0 * value;
    } else {
      parts.chi2 = value;
      lnlike = (value < chi2max_) ? -0.5 * value : -inf;
    }
  }
  parts.lnpost = lnlike + parts.lnprior;
  if (std::isnan(parts.lnpost)) {
    // -inf likelihood plus a -inf prior is a rejection; NaN is chisurf's
    // hard error. A finite prior cannot make -inf finite again, so this
    // branch is only ever the function objective's doing.
    parts.lnpost = -inf;
  }
  return parts;
}

std::vector<std::vector<double> > MCMCSampler::spread_walkers(int n) const {
  // chisurf's _ensemble_walker_start: the bounded range where there is
  // one, the value (floored by the absolute std) where there is not, so
  // no direction is ever left without spread.
  std::vector<double> spread(ndim_);
  for (unsigned int i = 0; i < ndim_; ++i) {
    const bool lo = std::isfinite(lower_[i]), hi = std::isfinite(upper_[i]);
    if (lo && hi) {
      spread[i] = (upper_[i] - lower_[i]) * 1e-4;
    } else {
      spread[i] = std::fabs(initial_values_[i]) > 1e-15
                      ? std::fabs(initial_values_[i]) * walker_start_std_
                      : walker_start_std_;
    }
    if (!(spread[i] > 0.0)) spread[i] = walker_start_std_;
  }
  std::vector<std::vector<double> > start(
      static_cast<std::size_t>(n), std::vector<double>(ndim_));
  std::normal_distribution<double> normal(0.0, 1.0);
  for (int w = 0; w < n; ++w)
    for (unsigned int i = 0; i < ndim_; ++i) {
      double v = initial_values_[i] + spread[i] * normal(rng_);
      start[static_cast<std::size_t>(w)][i] =
          std::min(std::max(v, lower_[i]), upper_[i]);
    }
  return start;
}

void MCMCSampler::validate() const {
  if (ndim_ == 0)
    throw MCMCSamplerConfigurationError(
        "no parameters: call set_parameter_ports() or set_initial_values()");
  if (!has_objective())
    throw MCMCSamplerConfigurationError(
        "no objective: call set_objective() or set_objective_function()");
  if (!initial_values_.empty() && initial_values_.size() != ndim_)
    throw MCMCSamplerConfigurationError("the initial values do not match the parameters");
  if (bounds_explicit_ &&
      (lower_.size() != ndim_ || upper_.size() != ndim_))
    throw MCMCSamplerConfigurationError("the bounds do not match the parameters");
}

std::vector<std::vector<int> > MCMCSampler::block_partition() const {
  std::vector<std::vector<int> > partition;
  if (!explicit_blocks_.empty()) {
    partition = explicit_blocks_;
  } else if (factor_graph_) {
    // chisurf's _default_blocks: the graph's partition, a parameter the graph cannot place dropped,
    // and one block over everything when the cover misses anything.
    const std::vector<std::vector<std::string> > graph_blocks = factor_graph_->get_sampling_blocks();
    std::vector<int> covered;
    for (std::size_t b = 0; b < graph_blocks.size(); ++b) {
      std::vector<int> block;
      for (std::size_t k = 0; k < graph_blocks[b].size(); ++k) {
        const int idx = factor_graph_->index_of(graph_blocks[b][k]);
        if (idx >= 0 && idx < static_cast<int>(ndim_)) block.push_back(idx);
      }
      if (block.empty()) continue;
      std::sort(block.begin(), block.end());
      partition.push_back(block);
      covered.insert(covered.end(), block.begin(), block.end());
    }
    std::sort(covered.begin(), covered.end());
    bool full = covered.size() == ndim_;
    for (unsigned int i = 0; full && i < ndim_; ++i) full = covered[i] == static_cast<int>(i);
    if (!full) partition.clear();
  }
  return partition;  // empty: the kernel uses one block over everything
}

void MCMCSampler::initialize_ensemble() {
  configure_from_ports();
  chain_.clear();
  log_prob_.clear();
  ln_prior_.clear();
  chi2_.clear();
  iteration_ = 0;
  acceptance_fractions_.clear();
  read_sampler_entry();

  // The start, by chisurf's policy for the population the entry declares.
  std::vector<std::vector<double> > walkers;
  if (population_ == "walkers") {
    const int n = get_number_of_walkers();
    if (n < 4)
      throw MCMCSamplerConfigurationError(
          "an ensemble of " + std::to_string(n) +
          " walkers is too small to be split into two halves that propose from each other; use at least 4");
    if (n < 2 * static_cast<int>(ndim_) && !live_dangerously_)
      throw MCMCSamplerConfigurationError(
          "an ensemble of " + std::to_string(n) + " walkers cannot span " + std::to_string(ndim_) +
          " dimensions; use at least " + std::to_string(2 * ndim_) + " walkers");
    walkers = walker_start_override_.empty() ? spread_walkers(n) : walker_start_override_;
  } else if (population_ == "chains") {
    // chisurf's seeding: a population spread around the start, big enough that the first difference
    // vectors mean something, clipped to the bounds, with member 0 left exactly at the start.
    const int n = get_number_of_chains();
    walkers.assign(static_cast<std::size_t>(n), std::vector<double>(ndim_, 0.0));
    const double floor_scale = std::max(jitter_, 1e-3);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int c = 0; c < n; ++c)
      for (unsigned int i = 0; i < ndim_; ++i) {
        double scale = std::fabs(initial_values_[i]) * std::max(jitter_, 1e-3) * 10.0;
        if (scale < 1e-12) scale = floor_scale;
        double v = initial_values_[i] + normal(rng_) * scale;
        walkers[static_cast<std::size_t>(c)][i] = std::min(std::max(v, lower_[i]), upper_[i]);
      }
    for (unsigned int i = 0; i < ndim_; ++i) walkers[0][i] = initial_values_[i];
  } else {
    walkers.assign(1, initial_values_);
  }

  // The kernel's options: every setting of this sampler whose name the entry's schema declares.
  typedef nlohmann::basic_json<nlohmann::ordered_map> FacadeJson;
  FacadeJson available = FacadeJson::object();
  available["stretch_scale"] = stretch_scale_;
  available["live_dangerously"] = live_dangerously_;
  if (!slice_tuning_) available["mu"] = slice_mu_;
  available["max_steps"] = slice_max_steps_;
  available["jitter"] = jitter_;
  available["snooker"] = snooker_;
  available["temp"] = temp_;
  available["step_size"] = step_size_;
  available["blocks"] = block_partition();
  available["proposal_covariance"] = proposal_covariance_;
  FacadeJson options = FacadeJson::object();
  for (const std::string& name : option_names_)
    if (available.contains(name)) options[name] = available[name];
  try {
    kernel_ = std::shared_ptr<SamplerKernel>(create_sampler_kernel(algorithm_, options.dump()).release());
  } catch (const SamplerConfigurationError& e) {
    throw MCMCSamplerConfigurationError(e.what());
  }

  SamplingTarget target;
  target.dim = ndim_;
  target.names = names_;
  target.log_density = [this](const std::vector<double>& x) { return evaluate(x).lnpost; };
  target.log_density_blobs = [this](const std::vector<double>& x, std::vector<double>& blobs) {
    const Parts p = evaluate(x);
    blobs.assign({p.lnprior, p.chi2});
    return p.lnpost;
  };
  if (objective_gradient_function_) {
    const auto g = objective_gradient_function_;
    target.log_density_gradient = [this, g](const std::vector<double>& x, std::vector<double>& grad) {
      ++n_evaluations_;
      const double lp = log_prior(x);
      if (!std::isfinite(lp)) return -std::numeric_limits<double>::infinity();
      return g(x, grad) + lp;
    };
  }
  try {
    kernel_->initialize(target, walkers, rng_);
  } catch (const SamplerConfigurationError& e) {
    kernel_.reset();
    throw MCMCSamplerConfigurationError(e.what());
  }
  accepted_.assign(kernel_->walkers().size(), 0);
  initialized_ = true;
}

void MCMCSampler::record_state() {
  const auto& walkers = kernel_->walkers();
  const auto& lps = kernel_->log_density();
  const auto& blobs = kernel_->blobs();
  for (std::size_t w = 0; w < walkers.size(); ++w) {
    chain_.push_back(walkers[w]);
    log_prob_.push_back(lps[w]);
    const bool have = w < blobs.size() && blobs[w].size() == 2;
    ln_prior_.push_back(have ? blobs[w][0] : 0.0);
    chi2_.push_back(have ? blobs[w][1] : -2.0 * lps[w]);
  }
  ++iteration_;
  acceptance_fractions_.assign(walkers.size(), 0.0);
  for (std::size_t w = 0; w < walkers.size(); ++w)
    acceptance_fractions_[w] = static_cast<double>(accepted_[w]) / static_cast<double>(iteration_);
}

// -------------------------------------------------------------------- run

void MCMCSampler::run(int n_steps, int thin) {
  thin_ = std::max(1, thin);
  validate();
  if (!initialized_) {
    initialize_ensemble();
    // The warm-up the algorithm's registry entry declares (chisurf's per-algorithm defaults: none for
    // the stretch move, DE's min(500, max(50, n/4)), the slice sampler's min(200, max(20, n/20)), the
    // blocked walk's min(500, max(100, n/20))), unless set_n_adapt() said otherwise.
    const int n_samples = std::max(1, n_steps / thin_);
    int n_adapt = n_adapt_setting_;
    if (n_adapt < 0) {
      const int total = n_samples * thin_;
      n_adapt = warmup_rule_ == "clip" ? std::min(warmup_max_, std::max(warmup_min_, total / std::max(1, warmup_divisor_)))
                                       : warmup_value_;
    }
    try {
      kernel_->begin_warmup(n_adapt);
      for (int i = 0; i < n_adapt; ++i) kernel_->transition(rng_);
      kernel_->end_warmup();
    } catch (const SamplerConfigurationError& e) {
      throw MCMCSamplerConfigurationError(e.what());
    }
  }

  const int n_stored = std::max(1, n_steps / thin_);
  const int total = n_stored * thin_;
  int done = 0;
  std::vector<double> stats;
  const std::vector<std::string> names = kernel_->stat_names();
  const std::size_t n_stats = names.size();
  const std::size_t accepted_column =
      static_cast<std::size_t>(std::find(names.begin(), names.end(), "accepted") - names.begin());
  for (int s = 0; s < n_stored; ++s) {
    for (int t = 0; t < thin_; ++t) {
      kernel_->transition(rng_);
      // chisurf's EnsembleSampler.sample() carries the per-walker acceptance flags of the LAST
      // substep of a stored step into its accepted tally.
      if (t == thin_ - 1 && accepted_column < n_stats) {
        kernel_->stats(stats);
        for (std::size_t w = 0; w < accepted_.size() && (w + 1) * n_stats <= stats.size(); ++w)
          accepted_[w] += stats[w * n_stats + accepted_column] > 0.0 ? 1 : 0;
      }
      ++done;
      if (observer_) observer_(done, total);
    }
    record_state();
  }

  if (restores_parameters_ && !parameters_.empty()) {
    // chisurf's DE sampler leaves the model where it started it.
    for (unsigned int i = 0; i < ndim_; ++i) parameters_[i]->set_value(initial_values_[i]);
    objective_node_->update();
  }
}

void MCMCSampler::step() { run(1, 1); }

void MCMCSampler::reset() {
  kernel_.reset();
  chain_.clear();
  log_prob_.clear();
  ln_prior_.clear();
  chi2_.clear();
  acceptance_fractions_.clear();
  accepted_.clear();
  iteration_ = 0;
  initialized_ = false;
  rng_.seed(seed_);
}

// ----------------------------------------------------------------- results

std::vector<std::vector<double> > MCMCSampler::get_chain() const {
  return chain_;
}

std::vector<std::vector<double> > MCMCSampler::get_chain_of_walker(
    int walker) const {
  std::vector<std::vector<double> > out;
  const std::size_t n = kernel_ ? kernel_->walkers().size() : 0;
  if (n == 0) return out;
  if (walker < 0 || static_cast<std::size_t>(walker) >= n)
    throw MCMCSamplerConfigurationError("no walker " + std::to_string(walker));
  for (unsigned int s = 0; s < iteration_; ++s)
    out.push_back(chain_[static_cast<std::size_t>(s) * n + static_cast<std::size_t>(walker)]);
  return out;
}

std::vector<std::vector<double> > MCMCSampler::get_walkers() const {
  return kernel_ ? kernel_->walkers() : std::vector<std::vector<double> >();
}

std::vector<double> MCMCSampler::get_log_prob() const { return log_prob_; }

std::vector<double> MCMCSampler::get_lnprior() const { return ln_prior_; }

std::vector<double> MCMCSampler::get_chi2() const { return chi2_; }

double MCMCSampler::get_acceptance_rate() const {
  if (iteration_ == 0 || !kernel_) return std::nan("");
  if (acceptance_mode_ == "walker_fractions") {
    double sum = 0.0;
    for (std::size_t w = 0; w < acceptance_fractions_.size(); ++w) sum += acceptance_fractions_[w];
    return acceptance_fractions_.empty() ? std::nan("") : sum / static_cast<double>(acceptance_fractions_.size());
  }
  return static_cast<double>(kernel_->accepted()) / static_cast<double>(std::max<long>(1, kernel_->proposed()));
}

std::vector<double> MCMCSampler::get_acceptance_fractions() const {
  if (acceptance_mode_ == "walker_fractions") return acceptance_fractions_;
  return std::vector<double>(1, get_acceptance_rate());
}

std::vector<double> MCMCSampler::get_block_acceptance_rates() const {
  if (const BlockedMetropolisKernel* k = dynamic_cast<const BlockedMetropolisKernel*>(kernel_.get()))
    return k->block_acceptance_rates();
  return std::vector<double>();
}

unsigned int MCMCSampler::get_number_of_evaluations() const {
  return n_evaluations_;
}

unsigned int MCMCSampler::get_iteration() const { return iteration_; }

unsigned int MCMCSampler::get_number_of_parameters() const { return ndim_; }

std::string MCMCSampler::describe() const {
  std::ostringstream out;
  out << "MCMCSampler(algorithm='" << algorithm_ << "', n_parameters=" << ndim_
      << ", n_walkers=" << (kernel_ ? kernel_->walkers().size() : 0) << ", iteration=" << iteration_
      << ", acceptance_rate=" << get_acceptance_rate() << ")";
  return out.str();
}

IMPBFF_END_NAMESPACE
