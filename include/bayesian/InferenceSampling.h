/**
 *  \file IMP/bff/bayesian/InferenceSampling.h
 *  \brief Any evaluable factor graph as a SamplingTarget (PRD-156 step 4).
 *
 *  Written 2026-11-11 by claude-code/Opus 5.5 for tpeulen; replaces the decay-specific
 *  `bayesian_decay_sampling_target` (BayesianDecaySampling.h): every kernel registered in the
 *  sampler interface (Sampling.h; NUTS needs the gradient) runs on a graph's log posterior.
 *  Variables in `fixed` are held at their values in `theta`; the target's coordinates are the
 *  free ones, in theta order. A hyperparameter may be left free here -- a sampler integrates it,
 *  it does not maximise it.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INFERENCESAMPLING_H
#define IMPBFF_INFERENCESAMPLING_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/bayesian/Sampling.h>
#include <IMP/bff/fit/InferenceFactorGraph.h>
#include <IMP/bff/fit/InferenceLaplace.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

#ifndef SWIG
//! The graph's log posterior over its free coordinates; the graph must outlive every run on the target.
/*! The gradient comes from the factors' scores (InferenceFactorGraph::eval). */
inline SamplingTarget inference_sampling_target(const InferenceFactorGraph& graph, const std::vector<double>& theta,
                                                const std::vector<std::string>& fixed = std::vector<std::string>()) {
  const std::vector<int> F = inference_free_coordinates(graph, fixed);
  //: each call fills its own copy of theta, so kernels that evaluate walkers in parallel do not race
  SamplingTarget t = sampling_target(
      F.size(), std::function<double(const std::vector<double>&, std::vector<double>&)>(
                    [&graph, F, theta](const std::vector<double>& x, std::vector<double>& g) {
                      std::vector<double> full = theta;
                      for (std::size_t a = 0; a < F.size(); ++a) full[std::size_t(F[a])] = x[a];
                      const InferenceScore s = graph.eval(full, true);
                      g.resize(F.size());
                      for (std::size_t a = 0; a < F.size(); ++a) g[a] = s.gradient[std::size_t(F[a])];
                      return s.log_density;
                    }));
  for (int i : F) t.names.push_back(graph.key_at(i));   //: "" inside a vector-valued variable
  return t;
}
#endif

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_INFERENCESAMPLING_H */
