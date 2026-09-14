/**
 *  \file IMP/bff/internal/NodeConfig.h
 *  \brief Reading the settings a description carries for one node.
 *
 *  GraphNode::configure() takes JSON *text* rather than a parsed document so
 *  that nlohmann stays out of every public header. This is the other side of
 *  that: the small reader each node uses to pull its own settings out again.
 *
 *  It tracks which keys were read and refuses the ones that were not. A
 *  description is a contract with a model, and a setting nobody consumed is
 *  almost always a misspelling or a key that belongs to a different node
 *  type -- either way a silently ignored one produces a graph that evaluates
 *  happily and fits the wrong thing.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_NODECONFIG_H
#define IMPBFF_INTERNAL_NODECONFIG_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/GraphNode.h>
#include <IMP/bff/internal/json.h>

#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE
namespace internal {

//! One node's settings, read by name, with the leftovers refused.
class NodeConfig {
 public:
  NodeConfig(const std::string& node_type, const std::string& json_text)
      : node_type_(node_type) {
    const std::string text = json_text.empty() ? std::string("{}") : json_text;
    try {
      document_ = nlohmann::json::parse(text);
    } catch (const std::exception& error) {
      throw std::domain_error(where("") + "settings are not valid JSON: " +
                              error.what());
    }
    if (!document_.is_object()) {
      throw std::domain_error(where("") + "settings must be a JSON object");
    }
  }

  //! Whether a setting was supplied. Does not count as reading it.
  bool has(const std::string& key) const {
    return document_.find(key) != document_.end();
  }

  bool get_bool(const std::string& key) {
    const nlohmann::json& value = read(key);
    // A switch may be literal, or the 0/1 a rule over caller-supplied values
    // evaluates to -- which is how a per-fit choice such as autoscaling
    // reaches a node without the description itself having to branch.
    if (value.is_boolean()) return value.get<bool>();
    if (value.is_number_integer()) {
      const long long n = value.get<long long>();
      if (n == 0 || n == 1) return n == 1;
    }
    throw wrong_type(key, "a boolean, or 0 or 1");
  }

  int get_int(const std::string& key) {
    const nlohmann::json& value = read(key);
    if (!value.is_number_integer()) throw wrong_type(key, "an integer");
    return value.get<int>();
  }

  double get_double(const std::string& key) {
    const nlohmann::json& value = read(key);
    if (!value.is_number()) throw wrong_type(key, "a number");
    return value.get<double>();
  }

  std::string get_string(const std::string& key) {
    const nlohmann::json& value = read(key);
    if (!value.is_string()) throw wrong_type(key, "a string");
    return value.get<std::string>();
  }

  std::vector<double> get_doubles(const std::string& key) {
    const nlohmann::json& value = read(key);
    if (!value.is_array()) throw wrong_type(key, "an array of numbers");
    std::vector<double> out;
    for (nlohmann::json::const_iterator it = value.begin(); it != value.end();
         ++it) {
      if (!it->is_number()) throw wrong_type(key, "an array of numbers");
      out.push_back(it->get<double>());
    }
    return out;
  }

  std::vector<std::string> get_strings(const std::string& key) {
    const nlohmann::json& value = read(key);
    if (!value.is_array()) throw wrong_type(key, "an array of strings");
    std::vector<std::string> out;
    for (nlohmann::json::const_iterator it = value.begin(); it != value.end();
         ++it) {
      if (!it->is_string()) throw wrong_type(key, "an array of strings");
      out.push_back(it->get<std::string>());
    }
    return out;
  }

  std::vector<int> get_ints(const std::string& key) {
    const nlohmann::json& value = read(key);
    if (!value.is_array()) throw wrong_type(key, "an array of integers");
    std::vector<int> out;
    for (nlohmann::json::const_iterator it = value.begin(); it != value.end();
         ++it) {
      if (!it->is_number_integer()) {
        throw wrong_type(key, "an array of integers");
      }
      out.push_back(it->get<int>());
    }
    return out;
  }

  //! Apply the settings every node has, whatever kernel it is.
  /*! Caching is a property of GraphNode rather than of any one kernel, so it
      is read here and no subclass has to re-declare it. A subclass calls
      this once, before #require_all_used. */
  void apply_common(GraphNode& node) {
    if (has("memoize")) node.set_memoize(get_bool("memoize"));
  }

  //! Refuse every setting no getter consumed.
  void require_all_used() const {
    std::vector<std::string> extra;
    for (nlohmann::json::const_iterator it = document_.begin();
         it != document_.end(); ++it) {
      if (used_.find(it.key()) == used_.end()) extra.push_back(it.key());
    }
    if (extra.empty()) return;
    std::ostringstream message;
    message << where("") << "does not understand ";
    for (std::size_t i = 0; i < extra.size(); ++i) {
      message << (i == 0 ? "'" : ", '") << extra[i] << "'";
    }
    throw std::domain_error(message.str());
  }

 private:
  const nlohmann::json& read(const std::string& key) {
    nlohmann::json::const_iterator found = document_.find(key);
    if (found == document_.end()) {
      throw std::domain_error(where(key) + "is missing");
    }
    used_.insert(key);
    return *found;
  }

  std::domain_error wrong_type(const std::string& key,
                               const std::string& expected) const {
    return std::domain_error(where(key) + "must be " + expected);
  }

  std::string where(const std::string& key) const {
    std::ostringstream out;
    out << "node type '" << node_type_ << "': ";
    if (!key.empty()) out << "setting '" << key << "' ";
    return out.str();
  }

  std::string node_type_;
  nlohmann::json document_;
  std::set<std::string> used_;
};

}  // namespace internal
IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INTERNAL_NODECONFIG_H
