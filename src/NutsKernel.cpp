/**
 * \file NutsKernel.cpp
 * \brief The NUTS kernel's registry entry (category `sampler`) and factory, beside the kernel.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/NutsKernel.h>

#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/ordered_map.h>

IMPBFF_BEGIN_NAMESPACE

namespace {

const char* const kNutsEntry = R"JSON({
  "label": "No-U-Turn (NUTS)",
  "summary": "Hamiltonian Monte Carlo that chooses its own trajectory length; needs the gradient of the log density.",
  "description": "Stan's NUTS (base_nuts): leapfrog trajectories doubled forwards or backwards until they turn back on themselves (generalised no-U-turn criterion), the next state drawn multinomially from the whole trajectory. The step size is tuned during warm-up by dual averaging to the target acceptance statistic. A dense inverse metric (for example the Laplace covariance at a mode) makes it affine to that shape. Mixes on high-dimensional, correlated posteriors where random-walk and ensemble samplers need orders of magnitude more evaluations; it needs an unconstrained target and an exact gradient. Divergences after warm-up mean the posterior has regions the step size cannot resolve: the run is not valid. Benchmarked against CmdStan 2.39.0 (tree depth, acceptance, divergences, moments, ESS per gradient).",
  "references": [
    {"type": "article", "authors": "Hoffman MD, Gelman A", "title": "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo", "year": 2014, "journal": "J Mach Learn Res", "volume": "15", "pages": "1593-1623"},
    {"type": "article", "authors": "Betancourt M", "title": "A Conceptual Introduction to Hamiltonian Monte Carlo", "year": 2017, "url": "https://arxiv.org/abs/1701.02434"},
    {"type": "software", "authors": "Stan Development Team", "title": "Stan: base_nuts.hpp", "year": 2025, "url": "https://github.com/stan-dev/stan"}
  ],
  "params_schema": {
    "type": "object",
    "properties": {
      "max_depth": {"type": "integer", "title": "Maximum tree depth", "description": "at most 2^max_depth leapfrog steps per transition", "default": 10, "minimum": 1, "maximum": 20},
      "target_accept": {"type": "number", "title": "Target acceptance", "description": "dual-averaging target of the mean acceptance statistic during warm-up", "default": 0.8, "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0},
      "step_size": {"type": "number", "title": "Initial step size", "description": "start value; refined by the initial search and adapted during warm-up", "default": 1.0, "exclusiveMinimum": 0.0, "advanced": true},
      "find_step_size": {"type": "boolean", "title": "Initial step-size search", "description": "Stan's doubling/halving search before warm-up", "default": true, "advanced": true},
      "adapt_step_size": {"type": "boolean", "title": "Adapt the step size", "description": "dual averaging during warm-up; off keeps step_size fixed (Stan's adapt engaged = 0)", "default": true, "advanced": true},
      "max_delta_h": {"type": "number", "title": "Divergence threshold", "description": "energy error that counts as a divergence", "default": 1000.0, "exclusiveMinimum": 0.0, "advanced": true},
      "inverse_metric": {"type": "array", "title": "Inverse metric", "description": "dim x dim covariance, row-major; empty for the identity", "items": {"type": "number"}, "default": [], "advanced": true}
    }
  },
  "kind": "chain",
  "requires_gradient": true,
  "supports_bounds": false,
  "uses_covariance_seed": true,
  "uses_blocks": false,
  "aliases": [],
  "default_warmup": {"rule": "fixed", "value": 1000},
  "statistics": {
    "accept_stat": "mean Metropolis acceptance over the trajectory (what warm-up targets)",
    "step_size": "leapfrog step used by the transition",
    "tree_depth": "doublings made",
    "n_leapfrog": "leapfrog steps (gradient evaluations)",
    "divergent": "1 when the energy error exceeded max_delta_h",
    "energy": "Hamiltonian at the new state (for E-BFMI)"
  },
  "required_checks": [
    "no divergent transitions after warm-up",
    "tree_depth below max_depth (saturation means the trajectory was cut short)",
    "rank R-hat and bulk/tail ESS over independent chains against thresholds declared before the run"
  ]
})JSON";

using NutsJson = nlohmann::basic_json<nlohmann::ordered_map>;

std::unique_ptr<SamplerKernel> make_nuts(const std::string& options_json) {
  const NutsJson o = NutsJson::parse(options_json);
  NutsOptions opt;
  if (o.contains("max_depth")) opt.max_depth = o["max_depth"].get<int>();
  if (o.contains("target_accept")) opt.target_accept = o["target_accept"].get<double>();
  if (o.contains("step_size")) opt.step_size = o["step_size"].get<double>();
  if (o.contains("find_step_size")) opt.find_step_size = o["find_step_size"].get<bool>();
  if (o.contains("adapt_step_size")) opt.adapt_step_size = o["adapt_step_size"].get<bool>();
  if (o.contains("max_delta_h")) opt.max_delta_h = o["max_delta_h"].get<double>();
  if (o.contains("inverse_metric")) opt.inverse_metric = o["inverse_metric"].get<std::vector<double>>();
  return std::unique_ptr<SamplerKernel>(new NutsKernel(opt));
}

const bool registered_nuts = register_sampler_kernel("nuts", kNutsEntry, make_nuts);

}  // namespace

IMPBFF_END_NAMESPACE
