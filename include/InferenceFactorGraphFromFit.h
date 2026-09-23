/**
 * \file IMP/bff/InferenceFactorGraphFromFit.h
 * \brief The factor graph of a running fit, read off its node graph.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_INFERENCEFACTORGRAPHFROMFIT_H
#define IMPBFF_INFERENCEFACTORGRAPHFROMFIT_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/FitObjective.h>
#include <IMP/bff/GraphPort.h>
#include <IMP/bff/InferenceFactorGraph.h>

#include <memory>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The factor graph of a fit as it runs: derived, not declared.
/*!
    Every port in `ports` is a variable keyed by `keys`: role `free` (size 1)
    unless it is fixed (`fixed`, size 0) or follows another port of the list
    (`follower`, size 0, joined to its master by a INFERENCE_FACTOR_LINK).
    Each member of `objective` -- a FitJointChiSquared's members, or the
    objective itself -- is a likelihood factor (fit_index = member position)
    whose scope is the free variables its graph reads upstream, followers
    resolved to their masters; the fixed ones it reads become its evidence.
    Nothing is declared twice: the graph that is optimised is the graph that
    is described.
*/
IMPBFFEXPORT InferenceFactorGraph get_fit_factor_graph(
    std::shared_ptr<FitObjective> objective, const std::vector<std::string>& keys,
    const std::vector<std::shared_ptr<GraphPort> >& ports);

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INFERENCEFACTORGRAPHFROMFIT_H
