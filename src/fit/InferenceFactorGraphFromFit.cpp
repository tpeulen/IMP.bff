/**
 * \file InferenceFactorGraphFromFit.cpp
 * \brief The factor graph of a running fit, read off its node graph.
 *
 * Kept apart from InferenceFactorGraph.cpp, whose structural machinery needs
 * nothing but the standard library; this is the one place it meets nodes.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/InferenceFactorGraphFromFit.h>
#include <IMP/bff/FitJointChiSquared.h>
#include <IMP/bff/FitObjective.h>
#include <IMP/bff/GraphNode.h>
#include <IMP/bff/GraphPort.h>

#include <deque>
#include <map>
#include <set>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {

//! The members a fit's likelihood splits into: a joint's, recursively, else itself.
void members_of(const std::shared_ptr<FitObjective>& objective,
                std::vector<std::shared_ptr<FitObjective> >& out) {
  const std::shared_ptr<FitJointChiSquared> joint =
      std::dynamic_pointer_cast<FitJointChiSquared>(objective);
  if (!joint) {
    out.push_back(objective);
    return;
  }
  for (unsigned int i = 0; i < joint->get_number_of_members(); ++i) {
    members_of(joint->get_member(static_cast<int>(i)), out);
  }
}

//! Every port a node's value depends on: its inputs and what they follow.
std::set<const GraphPort*> read_upstream(const std::shared_ptr<GraphNode>& start) {
  std::set<const GraphPort*> read;
  std::set<const GraphNode*> seen;
  std::deque<std::shared_ptr<GraphNode> > queue(1, start);
  seen.insert(start.get());
  while (!queue.empty()) {
    const std::shared_ptr<GraphNode> node = queue.front();
    queue.pop_front();
    const GraphNode::GraphPortMap inputs = node->get_input_ports();
    for (GraphNode::GraphPortMap::const_iterator it = inputs.begin(); it != inputs.end(); ++it) {
      read.insert(it->second.get());
      for (std::shared_ptr<GraphPort> link = it->second->get_link(); link;
           link = link->get_link()) {
        read.insert(link.get());
        // Following another node's *output* is data flowing from that node;
        // following an input only shares its value, not its model.
        if (!link->get_is_output()) continue;
        const std::shared_ptr<GraphNode> owner = link->get_node();
        if (owner && seen.insert(owner.get()).second) queue.push_back(owner);
      }
    }
  }
  return read;
}

}  // namespace

InferenceFactorGraph get_fit_factor_graph(
    std::shared_ptr<FitObjective> objective, const std::vector<std::string>& keys,
    const std::vector<std::shared_ptr<GraphPort> >& ports) {
  if (!objective) throw std::invalid_argument("get_fit_factor_graph: the objective is null");
  if (keys.size() != ports.size()) {
    throw std::invalid_argument("get_fit_factor_graph: one key per parameter port");
  }
  std::map<const GraphPort*, int> position;
  for (std::size_t i = 0; i < ports.size(); ++i) {
    if (!ports[i]) throw std::invalid_argument("get_fit_factor_graph: a parameter port is null");
    position[ports[i].get()] = static_cast<int>(i);
  }
  // A port whose link chain reaches another listed port follows it; the last
  // listed port on the chain is the master everything along it reads.
  std::vector<int> master(ports.size(), -1);
  for (std::size_t i = 0; i < ports.size(); ++i) {
    for (std::shared_ptr<GraphPort> link = ports[i]->get_link(); link; link = link->get_link()) {
      const auto found = position.find(link.get());
      if (found != position.end()) master[i] = found->second;
    }
  }
  std::vector<std::string> role(ports.size());
  for (std::size_t i = 0; i < ports.size(); ++i) {
    role[i] = master[i] >= 0 ? "follower" : (ports[i]->get_fixed() ? "fixed" : "free");
  }

  InferenceFactorGraph graph;
  for (std::size_t i = 0; i < ports.size(); ++i) {
    graph.add_variable(keys[i], ports[i]->get_name().empty() ? keys[i] : ports[i]->get_name(),
                       static_cast<int>(i), -1, role[i] == "free" ? 1 : 0, role[i]);
  }
  std::vector<std::shared_ptr<FitObjective> > members;
  members_of(objective, members);
  for (std::size_t m = 0; m < members.size(); ++m) {
    const std::set<const GraphPort*> read = read_upstream(members[m]);
    std::set<int> scope, evidence;
    for (const GraphPort* port : read) {
      const auto found = position.find(port);
      if (found == position.end()) continue;
      const int reads = master[static_cast<std::size_t>(found->second)] >= 0
                            ? master[static_cast<std::size_t>(found->second)]
                            : found->second;
      (role[static_cast<std::size_t>(reads)] == "free" ? scope : evidence).insert(reads);
    }
    std::vector<std::string> scope_keys, evidence_keys;
    for (int i : scope) scope_keys.push_back(keys[static_cast<std::size_t>(i)]);
    for (int i : evidence) evidence_keys.push_back(keys[static_cast<std::size_t>(i)]);
    const std::string key = "likelihood:" + members[m]->get_name();
    graph.add_factor(key, INFERENCE_FACTOR_LIKELIHOOD, scope_keys, static_cast<int>(m),
                     static_cast<int>(members[m]->get_number_of_residuals()));
    if (!evidence_keys.empty()) graph.set_factor_evidence(key, evidence_keys);
  }
  for (std::size_t i = 0; i < ports.size(); ++i) {
    if (master[i] < 0) continue;
    graph.add_factor("link:" + keys[i], INFERENCE_FACTOR_LINK,
                     {keys[i], keys[static_cast<std::size_t>(master[i])]});
  }
  return graph;
}

IMPBFF_END_NAMESPACE
