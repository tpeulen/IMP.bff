/**
 * \file IMP/bff/FitJointChiSquared.h
 * \brief One misfit over several datasets: the grouping, as a node.
 *
 * A fit of several datasets at once is two things, and bff already had one
 * of them. **Sharing a parameter** between datasets is `GraphPort::set_link` --
 * the member models' ports follow one master port, and the link graph is
 * enforced acyclic -- so the coupling that makes a joint fit *joint* has
 * been on this side of the boundary since the GraphPort runtime landed. What was
 * missing is the other half: **one objective over all of them.**
 *
 * That is all this class is. Its residual is the members' residuals laid end
 * to end, and its chi-square is their sum, which is exactly what minimising
 * a joint fit means: one Levenberg-Marquardt step moves the shared
 * parameters using the curvature of every dataset at once, rather than each
 * dataset in turn hoping they agree.
 *
 * It is a `GraphNode`, so the members are reached the ordinary way: each member's
 * residual output port is *linked* to one of this node's input ports, and
 * `GraphNode::update()` therefore evaluates the whole tree -- every member's
 * model, every member's misfit, then this -- from one call, with nothing
 * crossing into Python. A `FitMinimizer` pointed at this node optimises the
 * group.
 *
 * **Not called a fit group**, and not shaped like one. ChiSurf's `FitGroup`
 * is a container of `Fit` objects that also owns a selection, a result
 * history, plots and a run policy; this is the arithmetic underneath that
 * and nothing else. The layering rule in `AGENTS.md` puts the objective here
 * and the bookkeeping in the application, so the name says what the object
 * *is* -- the joint chi-square -- rather than which application assembles
 * it.
 *
 * The members do not have to be `FitChiSquared` nodes. Any `FitObjective`
 * qualifies, including a Python subclass wrapping a model this library cannot
 * represent, so a group may mix representable and unrepresentable members and
 * still take one step.
 *
 * \see FitChiSquared, FitMinimizer, GraphExpression, GraphPort
 *
 * \authors Thomas-Otavio Peulen
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_FITJOINTCHISQUARED_H
#define IMPBFF_FITJOINTCHISQUARED_H

#include <IMP/bff/bff_config.h>

#include <memory>
#include <string>
#include <vector>

#include <IMP/bff/FitObjective.h>
#include <IMP/bff/GraphNode.h>
#include <IMP/bff/GraphPort.h>

IMPBFF_BEGIN_NAMESPACE

//! The misfit of several datasets at once: their residuals, end to end.
class IMPBFFEXPORT FitJointChiSquared : public FitObjective {
 public:
  explicit FitJointChiSquared(const std::string& name = "joint");

  //! Add a member, linking its residuals to a new input of this node.
  /**
      \param member the objective of one dataset
      \return the index of the block this member occupies

      The member keeps its own data, fit range, mask and noise model -- a
      group whose members share none of those is the normal case, not an
      awkward one. Only the *parameters* are shared, and they are shared by
      linking ports, which happens outside this class.

      Members are evaluated in the order they were added, and the joint
      residual is in that order, so a caller can map a block back to the
      dataset that produced it.
   */
  int add_member(std::shared_ptr<FitObjective> member);

  //! How many members the group holds.
  unsigned int get_number_of_members() const {
    return static_cast<unsigned int>(members_.size());
  }

  //! One member, by the index `add_member` returned.
  /** Reached one at a time because `std::vector<std::shared_ptr<GraphNode> >` is
      not a template this module may name (see `IMP_bff.types.i`), and
      wrapping it anyway leaks -- SWIG finds no destructor for it. */
  std::shared_ptr<FitObjective> get_member(int index) const;

  //! The members' names, in the order they were added.
  std::vector<std::string> get_member_names() const;

  //! How long each member's residual block was in the last evaluation.
  /** Empty until the node has been evaluated. A block length can change
      between evaluations -- a member is free to truncate against a short
      model curve, which `FitChiSquared` does -- so this describes the last
      evaluation rather than a promise about the next. */
  std::vector<int> get_block_sizes() const { return block_sizes_; }

  //! Where each member's block starts in the joint residual.
  std::vector<int> get_block_offsets() const;

  //! Read every member's residuals, concatenate, write the sum out.
  void evaluate() override;

  std::string describe() const;

  std::string get_node_type() const override;
  void configure(const std::string& json_text) override;
  void add_member_node(std::shared_ptr<GraphNode> member) override;

 private:
  std::vector<std::shared_ptr<FitObjective> > members_;
  //! This node's input ports, one per member, in member order.
  std::vector<std::shared_ptr<GraphPort> > blocks_;
  std::vector<int> block_sizes_;
};

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_FITJOINTCHISQUARED_H
