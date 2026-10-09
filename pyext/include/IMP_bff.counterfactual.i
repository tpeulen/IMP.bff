/*
 * Native counterfactuals (okf/counterfactuals.md): a linear-Gaussian structural
 * causal model with closed-form abduction, interventions and natural effects;
 * the site-bias test of a distance network (abduction, do(b = 0), hinge,
 * counterfactual replay); Gumbel-max counterfactual trajectories of a Markov
 * chain. Direction on the factor graph itself (factor children, intervene,
 * twin) is in InferenceFactorGraph.h.
 *
 * Matrices cross as flat row-major tuples; `numpy.reshape` gives them shape.
 */
%include "IMP/bff/CausalLinearGaussian.h"
%include "IMP/bff/CounterfactualDistanceNetwork.h"
%include "IMP/bff/CounterfactualMarkovChain.h"
