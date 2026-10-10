/**
 * \file FitJointChiSquared.cpp
 * \brief One misfit over several datasets: the grouping, as a node.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/FitJointChiSquared.h>
#include <IMP/bff/internal/NodeConfig.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

FitJointChiSquared::FitJointChiSquared(const std::string& name) : FitObjective(name) {}

int FitJointChiSquared::add_member(std::shared_ptr<FitObjective> member) {
  if (!member) {
    throw std::domain_error(
        "FitJointChiSquared::add_member: the member is a null pointer");
  }
  const std::shared_ptr<GraphPort> source = member->get_residuals_port();

  // The block is an ordinary linked input, so `GraphNode::update()` is what walks
  // the group: it evaluates each member whose node is out of date, copies the
  // residuals in, and only then evaluates this node. Nothing here has to know
  // how deep a member's own graph goes.
  std::ostringstream key;
  key << "block_" << members_.size();
  std::shared_ptr<GraphPort> block(new GraphPort(std::vector<double>(1, 0.0)));
  // A member's residuals are fit transport: a NaN must survive the copy.
  block->set_sanitize(false);
  block->set_link(source);
  add_input_port(key.str(), block);

  members_.push_back(member);
  blocks_.push_back(block);
  return static_cast<int>(members_.size()) - 1;
}

std::shared_ptr<FitObjective> FitJointChiSquared::get_member(int index) const {
  if (index < 0 || static_cast<std::size_t>(index) >= members_.size()) {
    std::ostringstream m;
    m << "FitJointChiSquared::get_member: index " << index << " of "
      << members_.size() << " members";
    throw std::domain_error(m.str());
  }
  return members_[static_cast<std::size_t>(index)];
}

std::vector<std::string> FitJointChiSquared::get_member_names() const {
  std::vector<std::string> names;
  names.reserve(members_.size());
  for (const std::shared_ptr<FitObjective>& m : members_) names.push_back(m->get_name());
  return names;
}

std::vector<int> FitJointChiSquared::get_block_offsets() const {
  std::vector<int> offsets;
  offsets.reserve(block_sizes_.size());
  int at = 0;
  for (int n : block_sizes_) {
    offsets.push_back(at);
    at += n;
  }
  return offsets;
}

void FitJointChiSquared::evaluate() {
  block_sizes_.clear();
  block_sizes_.reserve(blocks_.size());
  std::size_t total = 0;
  for (const std::shared_ptr<GraphPort>& block : blocks_) {
    const std::size_t n = block->get_values_ref().size();
    block_sizes_.push_back(static_cast<int>(n));
    total += n;
  }

  std::vector<double> joined;
  joined.reserve(total);
  for (const std::shared_ptr<GraphPort>& block : blocks_) {
    const std::vector<double>& v = block->get_values_ref();
    joined.insert(joined.end(), v.begin(), v.end());
  }
  // A NaN anywhere makes the whole group infinitely bad (get_chi2), which is
  // what makes a sampler reject rather than propagate it into the posterior.
  set_residuals(joined);

  const std::shared_ptr<GraphPort> out = get_output_port(get_name());
  if (!out) {
    throw std::domain_error(
        "FitJointChiSquared '" + get_name() +
        "' writes chi-square to the output port keyed by its own name, "
        "which this node does not have");
  }
  out->set_value(get_chi2());

  set_valid(true);
}

std::string FitJointChiSquared::describe() const {
  std::ostringstream out;
  out << "members        : " << members_.size() << "\n";
  const std::vector<int> offsets = get_block_offsets();
  for (std::size_t i = 0; i < members_.size(); ++i) {
    out << "  [" << i << "] " << members_[i]->get_name();
    if (i < block_sizes_.size()) {
      out << " -- " << block_sizes_[i] << " residuals at " << offsets[i];
    }
    out << "\n";
  }
  out << "residuals      : " << get_number_of_residuals() << "\n"
      << "chi2           : " << get_chi2() << "\n";
  return out.str();
}

std::string FitJointChiSquared::get_node_type() const {
  return "FitJointChiSquared";
}

void FitJointChiSquared::configure(const std::string& json_text) {
  internal::NodeConfig config(get_node_type(), json_text);
  // Members are not settings: a member is another node, and a description
  // adds it by naming the graph edge, not by naming a value here.
  config.apply_common(*this);
  config.require_all_used();
}

void FitJointChiSquared::add_member_node(std::shared_ptr<GraphNode> member) {
  // Members are evaluated, and their residuals concatenated, in the order
  // they are added -- so the order a description lists them in is the order
  // a caller can map a residual block back to the dataset that produced it.
  const std::shared_ptr<FitObjective> objective =
      std::dynamic_pointer_cast<FitObjective>(member);
  if (!objective) {
    throw std::domain_error("member '" + (member ? member->get_name() : "") +
                            "' is not a fit objective");
  }
  add_member(objective);
}

IMPBFF_END_NAMESPACE
