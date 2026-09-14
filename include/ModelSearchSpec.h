/**
 *  \file IMP/bff/ModelSearchSpec.h
 *  \brief A model family, read rather than compiled.
 *
 *  A model-search family is four things: a canonical parameter registry, a
 *  set of complete objective graphs over it, which parameters each graph
 *  frees, and the moves the search may make between them. None of that is a
 *  numerical kernel, so none of it needs to be C++ -- and while it was, every
 *  new family cost a factory that open-coded its own registry, its own mask
 *  arithmetic and its own complexity constant, with nothing shared but the
 *  problem it built.
 *
 *  This reads the same four things from a JSON document. Nodes come from
 *  GraphNodeRegistry and are told what they are by GraphNode::configure(), so
 *  this class never learns what a kernel can do. What it adds is the wiring a
 *  description implies: canonical owners, links between nodes, and the
 *  measurements bound at run time.
 *
 *  **A description holds no data.** Values, errors, masks, response functions
 *  and lag axes arrive through #set_dataset, because a model outlives any one
 *  experiment. Seeds and bounds that depend on the data -- an initial photon
 *  count from the measured sum, a lifetime bound from the excitation period --
 *  are written as *expressions over named statistics* of whatever was bound,
 *  evaluated once at #build. The expression language is the one bff already
 *  has: GraphExpression, the same evaluator the models themselves use.
 *
 *  **Complexity is never written down.** A structure lists which canonical
 *  parameters it frees, and the count is the complexity. Writing both invites
 *  them to disagree, which is how a selection score goes quietly wrong.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_MODELSEARCHSPEC_H
#define IMPBFF_MODELSEARCHSPEC_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/FitDataset.h>
#include <IMP/bff/ModelSearch.h>

#include <memory>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! One model family, described in data and built on demand.
class IMPBFFEXPORT ModelSearchSpec {
 public:
  ModelSearchSpec();
  ~ModelSearchSpec();
  ModelSearchSpec(const ModelSearchSpec& other);
  ModelSearchSpec& operator=(const ModelSearchSpec& other);

  //! Read a description from a file.
  static ModelSearchSpec from_file(const std::string& path);
  //! Read a description from JSON text.
  static ModelSearchSpec from_json(const std::string& text);
  //! A shipped family, by name, from `data/model_search/<name>.json`.
  static ModelSearchSpec from_name(const std::string& name);
  //! The shipped family names.
  static std::vector<std::string> get_available_names();

  //! The description's own identity, for results and refusals.
  const std::string& get_family() const;
  //! The dataset slots this description needs bound before it can build.
  std::vector<std::string> get_dataset_names() const;
  //! The scalars this description expects from the caller.
  std::vector<std::string> get_scalar_names() const;
  //! The canonical parameter ids, in registry order.
  std::vector<std::string> get_parameter_ids() const;
  //! The structure keys this description declares.
  std::vector<std::string> get_structure_keys() const;

  //! Bind one measurement to a slot the description names.
  void set_dataset(const std::string& name, const FitDataset& dataset);
  //! Supply one scalar the description names (a channel width, a period).
  void set_scalar(const std::string& name, double value);

  //! Override one canonical parameter, as a caller rather than a description.
  /*! Applied after the description's own rules, so a caller always wins. */
  void set_parameter(const std::string& canonical_id, double initial,
                     bool free, double lower, double upper);

  //! Build the problem this description and the bound data describe.
  /*! \throws std::domain_error naming what is missing, unknown or
      inconsistent. A description that cannot be built completely builds
      nothing: a partially wired graph would fit, and fit the wrong thing. */
  std::shared_ptr<MultiStructureModelSearchProblem> build() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_MODELSEARCHSPEC_H
