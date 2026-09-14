/**
 * \file ModelSearchSpec.cpp
 * \brief A model family, read rather than compiled (see ModelSearchSpec.h).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/ModelSearchSpec.h>

#include <IMP/bff/GraphExpression.h>
#include <IMP/bff/GraphNodeRegistry.h>
#include <IMP/bff/ProbeDataPaths.h>
#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/ordered_map.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

IMPBFF_BEGIN_NAMESPACE

namespace {

//! Document order is the registry order, so the map has to keep it.
typedef nlohmann::basic_json<nlohmann::ordered_map> SpecJson;

const char* kSchema = "bff.model_search.v1";

[[noreturn]] void refuse(const std::string& what) {
  throw std::domain_error("model search description: " + what);
}

const SpecJson& require_object(const SpecJson& parent, const std::string& key,
                               const std::string& where) {
  SpecJson::const_iterator found = parent.find(key);
  if (found == parent.end() || !found->is_object()) {
    refuse(where + " needs an object '" + key + "'");
  }
  return *found;
}

std::string require_string(const SpecJson& parent, const std::string& key,
                           const std::string& where) {
  SpecJson::const_iterator found = parent.find(key);
  if (found == parent.end() || !found->is_string()) {
    refuse(where + " needs a string '" + key + "'");
  }
  return found->get<std::string>();
}

//! Every statistic of one bound measurement that a rule may refer to.
/*!
    These are seeds and bounds, not science: a starting photon count taken
    from the measured sum, a lag taken from where the curve has half
    decayed. They live here rather than in a description because they read
    the data, and a description holds no data.
*/
void add_dataset_scalars(const std::string& slot, const FitDataset& dataset,
                         std::map<std::string, double>& scope) {
  const std::vector<double>& values = dataset.get_values();
  const std::vector<double>& mask = dataset.get_mask();
  const std::size_t n = values.size();

  std::size_t first = 0;
  std::size_t last = n;
  for (std::size_t i = 0; i < n; ++i) {
    const bool in = mask.empty() || mask[i] != 0.0;
    if (in) { first = i; break; }
  }
  for (std::size_t i = n; i > 0; --i) {
    const bool in = mask.empty() || mask[i - 1] != 0.0;
    if (in) { last = i; break; }
  }

  double maximum = 1.0;
  double minimum = std::numeric_limits<double>::infinity();
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    maximum = std::max(maximum, values[i]);
    minimum = std::min(minimum, values[i]);
    total += values[i];
  }
  total = std::max(maximum, total);
  if (!std::isfinite(minimum)) minimum = 0.0;

  const double first_value = n ? values[first] : 0.0;
  const double last_value = (last > 0 && last <= n) ? values[last - 1] : 0.0;
  const double amplitude = first_value - last_value;

  scope[slot + "_size"] = static_cast<double>(n);
  scope[slot + "_max"] = maximum;
  // An additive background cannot exceed the smallest thing measured, so the
  // minimum is the largest value it can plausibly take -- and, unlike zero,
  // it is off the parameter's own lower bound.
  scope[slot + "_min"] = minimum;
  scope[slot + "_sum"] = total;
  scope[slot + "_first"] = first_value;
  scope[slot + "_last"] = last_value;
  scope[slot + "_amplitude"] = amplitude;
  // Where the curve first rises above a tenth of its maximum: the channel a
  // modelled response peak starts from, which ChiSurf takes as its position.
  {
    double peak = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) peak = std::max(peak, values[i]);
    std::size_t rise = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (values[i] > 0.1 * peak) { rise = i; break; }
    }
    scope[slot + "_rise_index"] = static_cast<double>(rise);
  }
  scope[slot + "_coordinates"] =
      static_cast<double>(dataset.get_number_of_coordinates());
  // The sampling step of an evenly sampled axis -- a TCSPC channel width, read
  // from the measurement instead of typed in beside it.
  if (dataset.get_number_of_coordinates() > 0) {
    const std::vector<double>& sampled = dataset.get_coordinate(0);
    if (sampled.size() >= 2) {
      scope[slot + "_dx"] = (sampled.back() - sampled.front()) /
                            static_cast<double>(sampled.size() - 1);
    }
  }

  // The characteristic lag of a decaying curve: the sampling position where
  // it is nearest half way down. Without an amplitude there is nothing to be
  // half of, so the median positive position stands in -- the same fallback,
  // and the same order of magnitude, a closed-form seed needs.
  const std::vector<double>& axis =
      dataset.get_number_of_coordinates() > 0 ? dataset.get_coordinate(0)
                                              : values;
  std::vector<double> positive;
  for (std::size_t i = first; i < last && i < axis.size(); ++i) {
    if (axis[i] > 0.0) positive.push_back(axis[i]);
  }
  double characteristic = 1e-12;
  if (!positive.empty()) {
    std::vector<double> sorted = positive;
    const std::size_t middle = sorted.size() / 2;
    std::nth_element(sorted.begin(), sorted.begin() + middle, sorted.end());
    double median = sorted[middle];
    if (sorted.size() % 2 == 0) {
      std::nth_element(sorted.begin(), sorted.begin() + middle - 1,
                       sorted.end());
      median = 0.5 * (sorted[middle - 1] + median);
    }
    characteristic = std::max(median, 1e-12);
  }
  if (amplitude > 1e-12) {
    const double half = last_value + 0.5 * amplitude;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = first; i < last && i < axis.size(); ++i) {
      const double distance = std::fabs(values[i] - half);
      if (distance < best && axis[i] > 0.0) {
        best = distance;
        characteristic = axis[i];
      }
    }
  }
  scope[slot + "_characteristic"] = characteristic;
}

//! A number, or an expression over the scope that produces one.
double evaluate_rule(const SpecJson& rule, const std::string& where,
                     const std::map<std::string, double>& scope) {
  if (rule.is_number()) return rule.get<double>();
  if (!rule.is_string()) {
    refuse(where + " must be a number or an expression");
  }
  const std::string text = rule.get<std::string>();
  std::shared_ptr<GraphExpression> expression =
      std::make_shared<GraphExpression>("rule");
  try {
    expression->set_expression(text);
  } catch (const std::exception& error) {
    refuse(where + ": '" + text + "' is not an expression (" + error.what() +
           ")");
  }
  const std::vector<std::string> variables = expression->get_variable_names();
  for (std::size_t i = 0; i < variables.size(); ++i) {
    std::map<std::string, double>::const_iterator found =
        scope.find(variables[i]);
    if (found == scope.end()) {
      std::ostringstream known;
      for (std::map<std::string, double>::const_iterator it = scope.begin();
           it != scope.end(); ++it) {
        known << (it == scope.begin() ? " " : ", ") << it->first;
      }
      refuse(where + ": '" + text + "' uses '" + variables[i] +
             "', which nothing supplies; available are" + known.str());
    }
    expression->add_input_port(
        variables[i], std::make_shared<GraphPort>(found->second));
  }
  std::shared_ptr<GraphPort> out = std::make_shared<GraphPort>(0.0, false, true);
  expression->add_output_port(expression->get_name(), out);
  expression->update();
  const double value = out->get_value();
  if (!std::isfinite(value)) {
    refuse(where + ": '" + text + "' did not produce a finite number");
  }
  return value;
}

//! A rule result, kept integral when it is, so an integer setting stays one.
SpecJson as_setting(double value) {
  if (value == std::floor(value) && std::fabs(value) < 9.0e15) {
    return SpecJson(static_cast<long long>(value));
  }
  return SpecJson(value);
}

double optional_rule(const SpecJson& parent, const std::string& key,
                     double fallback, const std::string& where,
                     const std::map<std::string, double>& scope) {
  SpecJson::const_iterator found = parent.find(key);
  if (found == parent.end()) return fallback;
  return evaluate_rule(*found, where + " '" + key + "'", scope);
}

int port_type_from_name(const std::string& name, const std::string& where) {
  if (name == "float") return GRAPH_PORT_FLOAT;
  if (name == "float_vector") return GRAPH_PORT_FLOAT_VECTOR;
  if (name == "int") return GRAPH_PORT_INT;
  if (name == "int_vector") return GRAPH_PORT_INT_VECTOR;
  if (name == "bool") return GRAPH_PORT_BOOL;
  if (name == "bool_vector") return GRAPH_PORT_BOOL_VECTOR;
  refuse(where + ": '" + name +
         "' is not a port type; use float, float_vector, int, int_vector, "
         "bool or bool_vector");
  return GRAPH_PORT_FLOAT;
}


//! Replace every {name} in a string with the binding of that name.
std::string substitute_text(const std::string& text,
                            const std::map<std::string, int>& bindings) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    if (text[i] != '{') {
      out.push_back(text[i++]);
      continue;
    }
    const std::size_t close = text.find('}', i);
    if (close == std::string::npos) {
      out.push_back(text[i++]);
      continue;
    }
    const std::string name = text.substr(i + 1, close - i - 1);
    std::map<std::string, int>::const_iterator found = bindings.find(name);
    if (found == bindings.end()) {
      // Not a binding: leave it alone, so an expression keeping braces for
      // its own reasons survives untouched.
      out.append(text, i, close - i + 1);
    } else {
      out += std::to_string(found->second);
    }
    i = close + 1;
  }
  return out;
}

//! How many times a repeat runs: an integer, or the value of an axis.
int repeat_count(const SpecJson& value, const std::map<std::string, int>& bindings,
                 const std::string& where) {
  if (value.is_number_integer()) return value.get<int>();
  if (value.is_string()) {
    const std::string name = value.get<std::string>();
    std::map<std::string, int>::const_iterator found = bindings.find(name);
    if (found == bindings.end()) {
      refuse(where + ": '" + name + "' is not an axis in scope");
    }
    return found->second;
  }
  refuse(where + ": a repeat count must be an integer or an axis name");
  return 0;
}

SpecJson expand(const SpecJson& node, const std::map<std::string, int>& bindings);

//! One `repeat` block, yielding the expansions of its body in order.
std::vector<SpecJson> expand_repeat(const SpecJson& spec,
                                    const std::map<std::string, int>& bindings,
                                    const std::string& where,
                                    std::vector<std::string>* keys) {
  if (!spec.is_object()) refuse(where + ": 'repeat' must be an object");
  const std::string index = require_string(spec, "index", where + " repeat");
  const SpecJson::const_iterator count_it = spec.find("count");
  if (count_it == spec.end()) refuse(where + " repeat needs a 'count'");
  const int count = repeat_count(*count_it, bindings, where + " repeat count");
  int from = 0;
  if (spec.contains("from")) from = spec["from"].get<int>();
  const SpecJson::const_iterator body = spec.find("body");
  if (body == spec.end()) refuse(where + " repeat needs a 'body'");

  std::vector<SpecJson> out;
  for (int i = from; i < count; ++i) {
    std::map<std::string, int> inner = bindings;
    inner[index] = i;
    if (body->is_object() && keys != nullptr) {
      // Merged into the surrounding object: remember the keys in order.
      for (SpecJson::const_iterator it = body->begin(); it != body->end();
           ++it) {
        keys->push_back(substitute_text(it.key(), inner));
        out.push_back(expand(*it, inner));
      }
    } else {
      out.push_back(expand(*body, inner));
    }
  }
  return out;
}

//! Substitute bindings and run every repeat, leaving ordinary JSON alone.
SpecJson expand(const SpecJson& node,
                const std::map<std::string, int>& bindings) {
  if (node.is_string()) {
    const std::string text = node.get<std::string>();
    // A value that is nothing but one binding becomes the number, so a
    // setting declared as an integer stays an integer.
    if (text.size() > 2 && text[0] == '{' && text[text.size() - 1] == '}' &&
        text.find('}') == text.size() - 1) {
      std::map<std::string, int>::const_iterator found =
          bindings.find(text.substr(1, text.size() - 2));
      if (found != bindings.end()) return SpecJson(found->second);
    }
    return SpecJson(substitute_text(text, bindings));
  }
  if (node.is_array()) {
    SpecJson out = SpecJson::array();
    for (SpecJson::const_iterator it = node.begin(); it != node.end(); ++it) {
      // `each` expands into several elements; `repeat` inside an object
      // merges keys into that one object. An array can hold either, so they
      // cannot share a name -- a list of seed maps wants the second.
      if (it->is_object() && it->contains("each") && it->size() == 1) {
        const std::vector<SpecJson> many =
            expand_repeat((*it)["each"], bindings, "array", nullptr);
        for (std::size_t k = 0; k < many.size(); ++k) out.push_back(many[k]);
      } else {
        out.push_back(expand(*it, bindings));
      }
    }
    return out;
  }
  if (node.is_object()) {
    SpecJson out = SpecJson::object();
    for (SpecJson::const_iterator it = node.begin(); it != node.end(); ++it) {
      if (it.key() == "repeat") {
        // One object often repeats over more than one thing -- a registry
        // repeats over lifetimes *and* rotations -- and a JSON object holds
        // one key of a name, so a list of blocks is the only way to say it.
        std::vector<SpecJson> blocks;
        if (it->is_array()) {
          for (SpecJson::const_iterator b = it->begin(); b != it->end(); ++b) {
            blocks.push_back(*b);
          }
        } else {
          blocks.push_back(*it);
        }
        for (std::size_t b = 0; b < blocks.size(); ++b) {
          std::vector<std::string> keys;
          const std::vector<SpecJson> many =
              expand_repeat(blocks[b], bindings, "object", &keys);
          for (std::size_t k = 0; k < many.size() && k < keys.size(); ++k) {
            out[keys[k]] = many[k];
          }
        }
      } else {
        out[substitute_text(it.key(), bindings)] = expand(*it, bindings);
      }
    }
    return out;
  }
  return node;
}


//! Whether a measurement slot is one a description lets a caller omit.
bool is_optional_dataset(const SpecJson& document, const std::string& slot) {
  SpecJson::const_iterator optional = document.find("optional_datasets");
  if (optional == document.end() || !optional->is_array()) return false;
  for (SpecJson::const_iterator it = optional->begin(); it != optional->end();
       ++it) {
    if (it->is_string() && it->get<std::string>() == slot) return true;
  }
  return false;
}

//! Turn `axes` + `template` + `moves` into explicit structures and actions.
/*!
    A general model family is a cross-product -- lifetimes by rotations by
    whether a term is present -- and writing it out is neither reviewable nor
    maintainable past a handful of topologies. This is the `for` loop the C++
    factories had, moved into the description: one construct, not a language.

    Everything downstream sees only the expanded document, so a family may be
    written either way and the rest of the loader neither knows nor cares.
*/
void expand_template(SpecJson& document) {
  SpecJson::const_iterator axes_it = document.find("axes");
  SpecJson::const_iterator template_it = document.find("template");
  if (axes_it == document.end() && template_it == document.end()) return;
  if (axes_it == document.end() || template_it == document.end()) {
    refuse("'axes' and 'template' are only meaningful together");
  }
  if (document.contains("structures")) {
    refuse("a family is written with 'structures' or with 'axes' and "
           "'template', not both");
  }

  std::vector<std::string> names;
  std::vector<int> lows;
  std::vector<int> highs;
  for (SpecJson::const_iterator it = axes_it->begin(); it != axes_it->end();
       ++it) {
    if (!it->is_object() || !it->contains("from") || !it->contains("to")) {
      refuse("axis '" + it.key() + "' needs 'from' and 'to'");
    }
    const int low = (*it)["from"].get<int>();
    const int high = (*it)["to"].get<int>();
    if (high < low) refuse("axis '" + it.key() + "' is empty");
    names.push_back(it.key());
    lows.push_back(low);
    highs.push_back(high);
  }

  // The registry has to cover the widest topology any axis value asks for.
  // Alongside each axis, its bounds: a registry has to cover the widest
  // topology the family admits, not just the one being expanded.
  std::map<std::string, int> bounds;
  for (std::size_t a = 0; a < names.size(); ++a) {
    bounds[names[a] + "_min"] = lows[a];
    bounds[names[a] + "_max"] = highs[a];
  }
  std::map<std::string, int> widest = bounds;
  for (std::size_t a = 0; a < names.size(); ++a) widest[names[a]] = highs[a];
  if (document.contains("parameters")) {
    document["parameters"] = expand(document["parameters"], widest);
  }

  std::vector<std::vector<int> > combinations(1, std::vector<int>());
  for (std::size_t a = 0; a < names.size(); ++a) {
    std::vector<std::vector<int> > grown;
    for (std::size_t c = 0; c < combinations.size(); ++c) {
      for (int v = lows[a]; v <= highs[a]; ++v) {
        std::vector<int> one = combinations[c];
        one.push_back(v);
        grown.push_back(one);
      }
    }
    combinations.swap(grown);
  }

  std::map<std::vector<int>, std::string> keys;
  SpecJson structures = SpecJson::object();
  for (std::size_t c = 0; c < combinations.size(); ++c) {
    std::map<std::string, int> bindings = bounds;
    for (std::size_t a = 0; a < names.size(); ++a) {
      bindings[names[a]] = combinations[c][a];
    }
    SpecJson one = expand(*template_it, bindings);
    const SpecJson::const_iterator key_it = one.find("key");
    if (key_it == one.end() || !key_it->is_string()) {
      refuse("the template needs a string 'key'");
    }
    const std::string key = key_it->get<std::string>();
    one.erase("key");
    if (structures.contains(key)) {
      refuse("the template produces the key '" + key + "' more than once");
    }
    keys[combinations[c]] = key;
    structures[key] = one;
  }
  document["structures"] = structures;

  SpecJson actions = document.contains("actions") ? document["actions"]
                                                  : SpecJson::array();
  SpecJson::const_iterator moves_it = document.find("moves");
  if (moves_it != document.end()) {
    if (!moves_it->is_array()) refuse("'moves' must be an array");
    for (std::size_t c = 0; c < combinations.size(); ++c) {
      for (SpecJson::const_iterator m = moves_it->begin();
           m != moves_it->end(); ++m) {
        const std::string action = require_string(*m, "action", "a move");
        const double prior = m->contains("prior")
                                 ? (*m)["prior"].get<double>() : 1.0;
        if (!m->contains("axis")) {
          // A move with no axis stays where it is: the stop.
          SpecJson one = SpecJson::object();
          one["from"] = keys[combinations[c]];
          one["action"] = action;
          one["to"] = keys[combinations[c]];
          one["prior"] = prior;
          one["terminal"] = m->contains("terminal")
                                ? (*m)["terminal"].get<bool>() : true;
          actions.push_back(one);
          continue;
        }
        const std::string axis = (*m)["axis"].get<std::string>();
        std::size_t index = names.size();
        for (std::size_t a = 0; a < names.size(); ++a) {
          if (names[a] == axis) index = a;
        }
        if (index == names.size()) {
          refuse("move '" + action + "' names the axis '" + axis +
                 "', which the family does not declare");
        }
        const int delta = m->contains("delta") ? (*m)["delta"].get<int>() : 1;
        std::vector<int> target = combinations[c];
        target[index] += delta;
        if (target[index] < lows[index] || target[index] > highs[index]) {
          continue;  // the move would leave the family
        }
        SpecJson one = SpecJson::object();
        one["from"] = keys[combinations[c]];
        one["action"] = action;
        one["to"] = keys[target];
        one["prior"] = prior;
        one["terminal"] = false;
        actions.push_back(one);
      }
    }
  }
  document["actions"] = actions;

  SpecJson::const_iterator start_it = document.find("initial_axes");
  if (start_it != document.end()) {
    std::vector<int> start;
    for (std::size_t a = 0; a < names.size(); ++a) {
      if (!start_it->contains(names[a])) {
        refuse("'initial_axes' does not say where '" + names[a] + "' starts");
      }
      start.push_back((*start_it)[names[a]].get<int>());
    }
    if (!keys.count(start)) refuse("'initial_axes' is outside the family");
    document["initial_structure"] = keys[start];
  }
}

//! Expand a catalogue of equations into ordinary structures and parameters.
/*!
    The frame -- the description's `equations` block -- says which measurement
    the equations are fitted to and what one structure looks like around its
    expression. Nothing here knows what an equation is *of*: a variable that
    names a coordinate of the measurement is an axis, and every other variable
    is a parameter, shared by name across the catalogue. So a correlation curve
    and an image-correlation carpet are the same case with a different number
    of coordinates.
*/
void expand_equations(SpecJson& document, const SpecJson& equations,
                      const std::map<std::string, FitDataset>& datasets) {
  // A copy: the document gains parameters, structures and actions below, and
  // an ordered map that grows moves what a reference into it pointed at.
  const SpecJson frame = require_object(document, "equations", "the description");
  const std::string slot = frame.contains("dataset")
                               ? frame["dataset"].get<std::string>()
                               : std::string("curve");
  const std::string node_key = frame.contains("expression_node")
                                   ? frame["expression_node"].get<std::string>()
                                   : std::string("model");
  const SpecJson& body = require_object(frame, "structure", "'equations'");
  if (!equations.is_object() || equations.empty()) {
    refuse("a catalogue of equations must name at least one equation");
  }
  std::map<std::string, FitDataset>::const_iterator data = datasets.find(slot);
  if (data == datasets.end()) {
    refuse("the equations are fitted to the measurement '" + slot +
           "', which nothing bound");
  }
  std::set<std::string> coordinates;
  for (int k = 0; k < data->second.get_number_of_coordinates(); ++k) {
    coordinates.insert(data->second.get_coordinate_name(k));
  }

  // What the frame declares itself -- an instrument around the equation --
  // is kept, and an equation variable may not reuse one of its ids.
  SpecJson parameters = document.contains("parameters") && document["parameters"].is_object()
                            ? document["parameters"]
                            : SpecJson::object();
  const SpecJson frame_parameters = parameters;
  SpecJson structures = SpecJson::object();
  std::vector<std::string> keys;
  for (SpecJson::const_iterator it = equations.begin(); it != equations.end();
       ++it) {
    const std::string key = it.key();
    const std::string where = "equation '" + key + "'";
    if (!it->is_object()) refuse(where + " must be an object");
    const std::string text = require_string(*it, "equation", where);
    GraphExpression probe("probe");
    try {
      probe.set_expression(text);
    } catch (const std::exception& error) {
      refuse(where + ": '" + text + "' does not compile (" + error.what() + ")");
    }
    const std::vector<std::string> variables = probe.get_variable_names();

    SpecJson initial = SpecJson::object();
    if (it->contains("initial")) initial = (*it)["initial"];
    SpecJson bounds = SpecJson::object();
    if (it->contains("bounds")) bounds = (*it)["bounds"];
    std::set<std::string> held;
    if (it->contains("fixed")) {
      for (SpecJson::const_iterator f = (*it)["fixed"].begin();
           f != (*it)["fixed"].end(); ++f) {
        held.insert(f->get<std::string>());
      }
    }
    SpecJson groups = SpecJson::object();
    if (it->contains("groups")) groups = (*it)["groups"];

    SpecJson structure = body;
    SpecJson& nodes = structure["nodes"];
    if (!nodes.contains(node_key)) {
      refuse("the equations frame has no expression node '" + node_key + "'");
    }
    SpecJson& node = nodes[node_key];
    node["config"]["expression"] = text;
    // The frame's own free parameters (an instrument's background, say)
    // come first; the equation's are added after them.
    SpecJson free = body.contains("free") && body["free"].is_array()
                        ? body["free"]
                        : SpecJson::array();
    bool has_axis = false;
    for (std::size_t v = 0; v < variables.size(); ++v) {
      const std::string& name = variables[v];
      if (coordinates.count(name)) {
        if (initial.contains(name)) {
          refuse(where + ": '" + name + "' is a coordinate of '" + slot +
                 "' and cannot also be given a starting value");
        }
        node["inputs"][name] = "@" + slot + "." + name;
        has_axis = true;
        continue;
      }
      if (frame_parameters.contains(name)) {
        refuse(where + ": '" + name +
               "' names a parameter the frame declares itself");
      }
      node["inputs"][name] = "#" + name;
      if (!parameters.contains(name)) {
        SpecJson entry = SpecJson::object();
        entry["name"] = name;
        entry["initial"] = initial.contains(name) ? initial[name] : SpecJson(1.0);
        if (bounds.contains(name)) {
          entry["lower"] = bounds[name][0];
          entry["upper"] = bounds[name][1];
        } else {
          entry["lower"] = nullptr;
          entry["upper"] = nullptr;
        }
        if (groups.contains(name)) entry["group"] = groups[name];
        parameters[name] = entry;
      }
      if (!held.count(name)) free.push_back(name);
    }
    if (!has_axis) {
      std::ostringstream offered;
      for (std::set<std::string>::const_iterator k = coordinates.begin();
           k != coordinates.end(); ++k) {
        offered << (k == coordinates.begin() ? " " : ", ") << *k;
      }
      refuse(where + " uses none of the coordinates of '" + slot +
             "'; it has" + (coordinates.empty() ? std::string(" none") : offered.str()));
    }
    structure["free"] = free;
    structure["label"] = it->contains("label") ? (*it)["label"] : SpecJson(key);
    structures[key] = structure;
    keys.push_back(key);
  }

  document["parameters"] = parameters;
  document["structures"] = structures;
  if (!document.contains("initial_structure")) {
    document["initial_structure"] = keys.front();
  }
  if (frame.contains("moves") && frame["moves"] == "all") {
    SpecJson actions = SpecJson::array();
    for (std::size_t i = 0; i < keys.size(); ++i) {
      SpecJson stop = SpecJson::object();
      stop["from"] = keys[i];
      stop["action"] = "stop";
      stop["to"] = keys[i];
      stop["terminal"] = true;
      actions.push_back(stop);
      for (std::size_t j = 0; j < keys.size(); ++j) {
        if (i == j) continue;
        SpecJson use = SpecJson::object();
        use["from"] = keys[i];
        use["action"] = "use:" + keys[j];
        use["to"] = keys[j];
        actions.push_back(use);
      }
    }
    document["actions"] = actions;
  }
}

}  // namespace

struct ModelSearchSpec::Impl {
  SpecJson document;
  std::string family;
  std::map<std::string, FitDataset> datasets;
  std::map<std::string, double> scalars;
  struct Override {
    double initial;
    bool free;
    double lower;
    double upper;
    //! False when the caller supplied a value but not bounds; the
    //! description's own bounds then stand.
    bool has_bounds = true;
  };
  std::map<std::string, Override> overrides;
  //! The live model: built on first request, rebuilt over the same parameter
  //! ports when what it was built from changes.
  std::shared_ptr<MultiStructureModelSearchProblem> model;
  bool dirty = true;
  //! A catalogue of equations to expand into structures; see set_equations.
  SpecJson equations;
  //! The document after expanding what depends on bound data.
  SpecJson expanded() const {
    SpecJson result = document;
    if (!equations.is_null()) expand_equations(result, equations, datasets);
    return result;
  }
};

ModelSearchSpec::ModelSearchSpec() : impl_(new Impl) {}
ModelSearchSpec::~ModelSearchSpec() {}
ModelSearchSpec::ModelSearchSpec(const ModelSearchSpec& other)
    : impl_(new Impl(*other.impl_)) {}
ModelSearchSpec& ModelSearchSpec::operator=(const ModelSearchSpec& other) {
  if (this != &other) impl_.reset(new Impl(*other.impl_));
  return *this;
}

ModelSearchSpec ModelSearchSpec::from_json(const std::string& text) {
  ModelSearchSpec spec;
  try {
    spec.impl_->document = SpecJson::parse(text);
  } catch (const std::exception& error) {
    refuse(std::string("not valid JSON: ") + error.what());
  }
  if (!spec.impl_->document.is_object()) refuse("must be a JSON object");
  const std::string schema =
      require_string(spec.impl_->document, "schema", "the description");
  if (schema != kSchema) {
    refuse("schema '" + schema + "' is not '" + std::string(kSchema) + "'");
  }
  spec.impl_->family =
      require_string(spec.impl_->document, "family", "the description");
  expand_template(spec.impl_->document);
  require_object(spec.impl_->document, "parameters", "the description");
  require_object(spec.impl_->document, "structures", "the description");
  return spec;
}

ModelSearchSpec ModelSearchSpec::from_file(const std::string& path) {
  std::ifstream in(path.c_str());
  if (!in) refuse("cannot read '" + path + "'");
  std::ostringstream text;
  text << in.rdbuf();
  return from_json(text.str());
}

ModelSearchSpec ModelSearchSpec::from_name(const std::string& name) {
  return from_file(get_data_path("model_search/" + name + ".json"));
}

const std::string& ModelSearchSpec::get_family() const {
  return impl_->family;
}

std::vector<std::string> ModelSearchSpec::get_dataset_names() const {
  std::vector<std::string> names;
  SpecJson::const_iterator found = impl_->document.find("datasets");
  if (found != impl_->document.end() && found->is_array()) {
    for (SpecJson::const_iterator it = found->begin(); it != found->end();
         ++it) {
      names.push_back(it->get<std::string>());
    }
  }
  return names;
}

std::vector<std::string> ModelSearchSpec::get_scalar_names() const {
  std::vector<std::string> names;
  SpecJson::const_iterator found = impl_->document.find("scalars");
  if (found != impl_->document.end() && found->is_array()) {
    for (SpecJson::const_iterator it = found->begin(); it != found->end();
         ++it) {
      names.push_back(it->get<std::string>());
    }
  }
  return names;
}

std::vector<std::string> ModelSearchSpec::get_parameter_ids() const {
  std::vector<std::string> ids;
  // Expanded when a catalogue and its measurement allow it, so a caller
  // sees the structures a catalogue will build before building it.
  SpecJson document = impl_->document;
  if (!impl_->equations.is_null()) {
    try {
      document = impl_->expanded();
    } catch (const std::exception&) {
    }
  }
  const SpecJson& parameters = document["parameters"];
  for (SpecJson::const_iterator it = parameters.begin();
       it != parameters.end(); ++it) {
    ids.push_back(it.key());
  }
  return ids;
}

std::vector<std::string> ModelSearchSpec::get_structure_keys() const {
  std::vector<std::string> keys;
  // Expanded when a catalogue and its measurement allow it, so a caller
  // sees the structures a catalogue will build before building it.
  SpecJson document = impl_->document;
  if (!impl_->equations.is_null()) {
    try {
      document = impl_->expanded();
    } catch (const std::exception&) {
    }
  }
  const SpecJson& structures = document["structures"];
  for (SpecJson::const_iterator it = structures.begin();
       it != structures.end(); ++it) {
    keys.push_back(it.key());
  }
  return keys;
}

void ModelSearchSpec::set_dataset(const std::string& name,
                                  const FitDataset& dataset) {
  impl_->dirty = true;
  impl_->datasets[name] = dataset;
}

void ModelSearchSpec::unset_dataset(const std::string& name) {
  if (impl_->datasets.erase(name)) impl_->dirty = true;
}

const std::vector<double>& ModelSearchSpec::get_dataset_values(
    const std::string& name) const {
  std::map<std::string, FitDataset>::const_iterator found =
      impl_->datasets.find(name);
  if (found == impl_->datasets.end()) {
    refuse("nothing is bound to the measurement '" + name + "'");
  }
  return found->second.get_values();
}

void ModelSearchSpec::set_scalar(const std::string& name, double value) {
  std::map<std::string, double>::const_iterator found = impl_->scalars.find(name);
  // Supplying what is already there changes nothing, and must not cost the
  // live model a rebuild.
  if (found != impl_->scalars.end() && found->second == value) return;
  impl_->dirty = true;
  impl_->scalars[name] = value;
}

void ModelSearchSpec::set_parameter(const std::string& canonical_id,
                                    double initial, bool free, double lower,
                                    double upper) {
  Impl::Override override_value;
  override_value.initial = initial;
  override_value.free = free;
  override_value.lower = lower;
  override_value.upper = upper;
  impl_->overrides[canonical_id] = override_value;
  impl_->dirty = true;
}

void ModelSearchSpec::set_parameter_value(const std::string& canonical_id,
                                          double initial, bool free) {
  Impl::Override override_value;
  override_value.initial = initial;
  override_value.free = free;
  override_value.lower = 0.0;
  override_value.upper = 0.0;
  override_value.has_bounds = false;
  impl_->overrides[canonical_id] = override_value;
  impl_->dirty = true;
}

std::vector<std::string> ModelSearchSpec::get_available_names() {
  // The families are files, so the list belongs beside them rather than in
  // this function -- it was hardcoded here and had already fallen a family
  // behind. Reading a manifest rather than the directory because portable
  // directory iteration is not available in this build's standard, and a
  // test asserts the manifest matches what is shipped, so it cannot drift
  // again quietly.
  std::vector<std::string> names;
  try {
    std::ifstream in(get_data_path("model_search/index.json").c_str());
    if (!in) return names;
    std::ostringstream text;
    text << in.rdbuf();
    const SpecJson listed = SpecJson::parse(text.str());
    if (!listed.is_array()) return names;
    for (SpecJson::const_iterator it = listed.begin(); it != listed.end();
         ++it) {
      if (it->is_string()) names.push_back(it->get<std::string>());
    }
  } catch (const std::exception&) {
    return std::vector<std::string>();  // no data installed: nothing to offer
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::shared_ptr<MultiStructureModelSearchProblem> ModelSearchSpec::build()
    const {
  return build_over(std::shared_ptr<MultiStructureModelSearchProblem>());
}

std::shared_ptr<MultiStructureModelSearchProblem> ModelSearchSpec::get_model() {
  if (impl_->model && !impl_->dirty) return impl_->model;
  const std::shared_ptr<MultiStructureModelSearchProblem> previous =
      impl_->model;
  std::shared_ptr<MultiStructureModelSearchProblem> model = build_over(previous);
  if (previous) {
    const std::vector<std::string> ids = model->get_parameter_ids();
    const std::vector<std::string> old_ids = previous->get_parameter_ids();
    const std::set<std::string> known(old_ids.begin(), old_ids.end());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (known.count(ids[i]) && previous->get_parameter_locked(ids[i])) {
        model->set_parameter_locked(ids[i], true);
      }
      if (known.count(ids[i]) && previous->get_parameter_released(ids[i])) {
        model->set_parameter_released(ids[i], true);
      }
    }
    const std::string active = previous->get_active_structure();
    const std::vector<std::string> keys = model->get_structure_keys();
    if (!active.empty() &&
        std::find(keys.begin(), keys.end(), active) != keys.end()) {
      model->select_structure(active);
    } else {
      model->select_structure(model->get_initial_structure());
    }
  } else {
    // A new model stands where its description starts: the initial topology
    // at its declared, data-derived seeds.
    model->activate_structure(model->get_initial_structure());
  }
  impl_->model = model;
  impl_->dirty = false;
  return model;
}

bool ModelSearchSpec::get_model_is_current() const {
  return impl_->model && !impl_->dirty;
}

double ModelSearchSpec::evaluate(const std::string& expression) const {
  std::map<std::string, double> scope;
  SpecJson::const_iterator optional_scalars =
      impl_->document.find("optional_scalars");
  if (optional_scalars != impl_->document.end() && optional_scalars->is_object()) {
    for (SpecJson::const_iterator it = optional_scalars->begin();
         it != optional_scalars->end(); ++it) {
      if (it->is_number()) scope[it.key()] = it->get<double>();
    }
  }
  for (std::map<std::string, double>::const_iterator it =
           impl_->scalars.begin();
       it != impl_->scalars.end(); ++it) {
    scope[it->first] = it->second;
  }
  for (std::map<std::string, FitDataset>::const_iterator it =
           impl_->datasets.begin();
       it != impl_->datasets.end(); ++it) {
    add_dataset_scalars(it->first, it->second, scope);
  }
  return evaluate_rule(SpecJson(expression), "the expression", scope);
}

void ModelSearchSpec::set_equations(const std::string& catalogue_json) {
  if (!impl_->document.contains("equations")) {
    refuse("family '" + impl_->family +
           "' takes no catalogue of equations; its structures are declared");
  }
  SpecJson catalogue;
  try {
    catalogue = SpecJson::parse(catalogue_json);
  } catch (const std::exception& error) {
    refuse(std::string("the catalogue of equations is not valid JSON: ") +
           error.what());
  }
  if (!catalogue.is_object() || catalogue.empty()) {
    refuse("a catalogue of equations must name at least one equation");
  }
  impl_->equations = catalogue;
  impl_->dirty = true;
}

std::string ModelSearchSpec::get_description_json() const {
  // With a catalogue and its measurement bound, the structures exist only
  // after expansion; without the measurement, the frame is what there is.
  if (!impl_->equations.is_null()) {
    try {
      return impl_->expanded().dump();
    } catch (const std::exception&) {
    }
  }
  return impl_->document.dump();
}

std::shared_ptr<MultiStructureModelSearchProblem> ModelSearchSpec::build_over(
    const std::shared_ptr<MultiStructureModelSearchProblem>& previous) const {
  const SpecJson document = impl_->expanded();
  // Everything the description says it needs has to be here before anything
  // is built. A half-wired graph still evaluates, and fits the wrong thing.
  // Optional scalars are per-fit switches and instrument numbers a caller
  // may leave alone -- autoscaling, pile-up -- so the description supplies a
  // default and a caller overrides it. A required scalar still has none.
  std::map<std::string, double> scope;
  SpecJson::const_iterator optional_scalars =
      document.find("optional_scalars");
  if (optional_scalars != document.end()) {
    if (!optional_scalars->is_object()) {
      refuse("'optional_scalars' must map a name to its default");
    }
    for (SpecJson::const_iterator it = optional_scalars->begin();
         it != optional_scalars->end(); ++it) {
      if (!it->is_number()) {
        refuse("optional scalar '" + it.key() + "' needs a numeric default");
      }
      scope[it.key()] = it->get<double>();
    }
  }
  for (std::map<std::string, double>::const_iterator it =
           impl_->scalars.begin();
       it != impl_->scalars.end(); ++it) {
    scope[it->first] = it->second;
  }
  const std::vector<std::string> wanted_datasets = get_dataset_names();
  for (std::size_t i = 0; i < wanted_datasets.size(); ++i) {
    std::map<std::string, FitDataset>::const_iterator found =
        impl_->datasets.find(wanted_datasets[i]);
    if (found == impl_->datasets.end()) {
      refuse("family '" + impl_->family + "' needs the measurement '" +
             wanted_datasets[i] + "', which nothing bound");
    }
    add_dataset_scalars(wanted_datasets[i], found->second, scope);
  }
  const std::vector<std::string> wanted_scalars = get_scalar_names();
  for (std::size_t i = 0; i < wanted_scalars.size(); ++i) {
    if (!impl_->scalars.count(wanted_scalars[i])) {
      refuse("family '" + impl_->family + "' needs the value '" +
             wanted_scalars[i] + "', which nothing supplied");
    }
  }

  std::shared_ptr<MultiStructureModelSearchProblem> problem =
      std::make_shared<MultiStructureModelSearchProblem>();

  // --- the canonical registry, in document order -------------------------
  const SpecJson& parameters = document["parameters"];
  std::vector<std::string> ids;
  std::vector<std::shared_ptr<GraphPort> > owners;
  std::map<std::string, std::shared_ptr<GraphPort> > owner_by_id;
  std::map<std::string, double> registry_initial;
  std::map<std::string, bool> registry_free;
  for (SpecJson::const_iterator it = parameters.begin();
       it != parameters.end(); ++it) {
    const std::string id = it.key();
    const SpecJson& entry = *it;
    if (!entry.is_object()) refuse("parameter '" + id + "' must be an object");
    const std::string where = "parameter '" + id + "'";
    double initial = optional_rule(entry, "initial", 0.0, where, scope);
    // A bound written as null is no bound: a ratio to a fixed reference has a
    // floor and no ceiling, and inventing one makes the answer depend on
    // which component happens to be the reference.
    const bool no_lower = entry.contains("lower") && entry["lower"].is_null();
    const bool no_upper = entry.contains("upper") && entry["upper"].is_null();
    double lower = no_lower ? -std::numeric_limits<double>::infinity()
                            : optional_rule(entry, "lower", 0.0, where, scope);
    double upper = no_upper ? std::numeric_limits<double>::infinity()
                            : optional_rule(entry, "upper", 0.0, where, scope);
    bool free = true;
    SpecJson::const_iterator free_it = entry.find("free");
    if (free_it != entry.end()) {
      if (!free_it->is_boolean()) refuse(where + " 'free' must be a boolean");
      free = free_it->get<bool>();
    }
    std::string name = id;
    SpecJson::const_iterator name_it = entry.find("name");
    if (name_it != entry.end()) name = name_it->get<std::string>();

    // A caller always outranks the description: an experimenter who fixed a
    // parameter or knows a better starting value is not overruled by a file.
    std::map<std::string, Impl::Override>::const_iterator over =
        impl_->overrides.find(id);
    if (over != impl_->overrides.end()) {
      initial = over->second.initial;
      free = over->second.free;
      if (over->second.has_bounds) {
        lower = over->second.lower;
        upper = over->second.upper;
      }
      // A starting value outside the description's own bounds is the
      // caller's value clipped, not the bound silently widened.
      initial = std::min(upper, std::max(lower, initial));
    }
    if (!(upper > lower)) {
      refuse(where + " has an empty range [" + std::to_string(lower) + ", " +
             std::to_string(upper) + "]");
    }
    std::shared_ptr<GraphPort> owner;
    std::vector<std::string> previous_ids;
    if (previous) previous_ids = previous->get_parameter_ids();
    if (previous && std::find(previous_ids.begin(), previous_ids.end(), id) !=
                        previous_ids.end()) {
      // The same port, so everything that holds it -- an application's view
      // of the parameter above all -- keeps holding the live one. Its value
      // is the user's and stays; only bounds that follow the data move, and
      // the value is clipped into them rather than re-seeded.
      owner = previous->get_parameter(id);
      owner->set_fixed(false);
      owner->set_bounds(lower, upper);
      owner->set_is_bounded(true);
      // A parameter that follows another model keeps following it; its
      // value is that model's, not one to clip here.
      if (!owner->get_link()) {
        owner->set_value(std::min(upper, std::max(lower, owner->get_value())));
      }
    } else {
      owner = std::make_shared<GraphPort>(initial, false, false, false, true,
                                          lower, upper, GRAPH_PORT_FLOAT, name);
    }
    ids.push_back(id);
    owners.push_back(owner);
    owner_by_id[id] = owner;
    registry_initial[id] = initial;
    registry_free[id] = free;
    problem->add_parameter(id, owner);
  }
  if (ids.empty()) refuse("family '" + impl_->family + "' has no parameters");

  // A caller override that matches nothing is a caller describing a
  // different model. Building anyway would fit without it and say nothing,
  // which is the silent fallback this library refuses to have.
  for (std::map<std::string, Impl::Override>::const_iterator it =
           impl_->overrides.begin();
       it != impl_->overrides.end(); ++it) {
    if (!owner_by_id.count(it->first)) {
      std::ostringstream known;
      for (std::size_t i = 0; i < ids.size(); ++i) {
        known << (i == 0 ? " " : ", ") << ids[i];
      }
      refuse("family '" + impl_->family + "' has no parameter '" + it->first +
             "'; it has" + known.str());
    }
  }

  // --- every structure: a complete graph over that one registry ----------
  const SpecJson& structures = document["structures"];
  for (SpecJson::const_iterator sit = structures.begin();
       sit != structures.end(); ++sit) {
    const std::string key = sit.key();
    const SpecJson& structure = *sit;
    const std::string where = "structure '" + key + "'";
    if (!structure.is_object()) refuse(where + " must be an object");
    const SpecJson& nodes = require_object(structure, "nodes", where);

    // Pass one: every node exists, is told what it is, and is handed its
    // measurements. Wiring waits until they all exist, because an input may
    // name a node the document has not reached yet.
    std::map<std::string, std::shared_ptr<GraphNode> > built;
    std::vector<std::string> node_order;
    std::set<std::string> used_ids;
    for (SpecJson::const_iterator nit = nodes.begin(); nit != nodes.end();
         ++nit) {
      const std::string node_key = nit.key();
      const SpecJson& node_spec = *nit;
      const std::string node_where = where + " node '" + node_key + "'";
      if (!node_spec.is_object()) refuse(node_where + " must be an object");
      const std::string type = require_string(node_spec, "type", node_where);
      std::shared_ptr<GraphNode> node;
      try {
        node = GraphNodeRegistry::create(type, key + "." + node_key);
      } catch (const std::exception& error) {
        refuse(node_where + ": " + error.what());
      }
      // Settings are literal; settings that depend on the measurement are
      // arithmetic over it. Keeping them in separate blocks means an
      // expression can never be mistaken for a model's own expression text.
      SpecJson settings = SpecJson::object();
      SpecJson::const_iterator config = node_spec.find("config");
      if (config != node_spec.end()) {
        if (!config->is_object()) refuse(node_where + " 'config' must be an object");
        settings = *config;
      }
      SpecJson::const_iterator rules = node_spec.find("config_rules");
      if (rules != node_spec.end()) {
        if (!rules->is_object()) {
          refuse(node_where + " 'config_rules' must be an object");
        }
        for (SpecJson::const_iterator rit = rules->begin(); rit != rules->end();
             ++rit) {
          const std::string rule_where =
              node_where + " setting '" + rit.key() + "'";
          if (settings.contains(rit.key())) {
            refuse(rule_where + " is given both literally and as a rule");
          }
          if (rit->is_object()) {
            // A choice among named settings, picked by a rule: a numeric
            // switch the caller sets selects a node's named mode.
            if (!rit->contains("options") || !(*rit)["options"].is_array() ||
                (*rit)["options"].empty() || !rit->contains("index")) {
              refuse(rule_where +
                     " as an object must give 'options' and an 'index' rule");
            }
            const SpecJson& options = (*rit)["options"];
            const double index =
                evaluate_rule((*rit)["index"], rule_where + " index", scope);
            const double picked = std::floor(index + 0.5);
            if (picked < 0.0 || picked >= static_cast<double>(options.size())) {
              std::ostringstream m;
              m << rule_where << " index " << index << " picks none of the "
                << options.size() << " options";
              refuse(m.str());
            }
            settings[rit.key()] = options[static_cast<std::size_t>(picked)];
          } else if (rit->is_array()) {
            SpecJson values = SpecJson::array();
            for (SpecJson::const_iterator eit = rit->begin();
                 eit != rit->end(); ++eit) {
              values.push_back(as_setting(evaluate_rule(*eit, rule_where, scope)));
            }
            settings[rit.key()] = values;
          } else {
            settings[rit.key()] = as_setting(evaluate_rule(*rit, rule_where, scope));
          }
        }
      }
      if (!settings.empty()) {
        try {
          node->configure(settings.dump());
        } catch (const std::exception& error) {
          refuse(node_where + ": " + error.what());
        }
      }
      SpecJson::const_iterator bind = node_spec.find("bind");
      if (bind != node_spec.end()) {
        if (!bind->is_object()) refuse(node_where + " 'bind' must be an object");
        for (SpecJson::const_iterator bit = bind->begin(); bit != bind->end();
             ++bit) {
          const std::string slot = bit->get<std::string>();
          std::map<std::string, FitDataset>::const_iterator data =
              impl_->datasets.find(slot);
          if (data == impl_->datasets.end()) {
            // An optional measurement nobody supplied -- a linearisation
            // table on an instrument without one -- leaves that stage off.
            if (is_optional_dataset(document, slot)) continue;
            refuse(node_where + " binds '" + bit.key() + "' to '" + slot +
                   "', which nothing supplied");
          }
          try {
            node->bind_dataset(bit.key(), data->second);
          } catch (const std::exception& error) {
            refuse(node_where + ": " + error.what());
          }
        }
      }
      built[node_key] = node;
      node_order.push_back(node_key);
    }

    // Pass two: outputs first, so an input may link to one.
    for (std::size_t n = 0; n < node_order.size(); ++n) {
      const SpecJson& node_spec = nodes[node_order[n]];
      const std::string node_where = where + " node '" + node_order[n] + "'";
      SpecJson::const_iterator outputs = node_spec.find("outputs");
      if (outputs == node_spec.end()) continue;
      if (!outputs->is_object()) refuse(node_where + " 'outputs' must be an object");
      for (SpecJson::const_iterator oit = outputs->begin();
           oit != outputs->end(); ++oit) {
        const int type = port_type_from_name(oit->get<std::string>(),
                                             node_where + " output '" +
                                                 oit.key() + "'");
        std::shared_ptr<GraphPort> port;
        if (type == GRAPH_PORT_FLOAT_VECTOR || type == GRAPH_PORT_INT_VECTOR ||
            type == GRAPH_PORT_BOOL_VECTOR) {
          port = std::make_shared<GraphPort>(std::vector<double>(1, 0.0), false,
                                             true, false, false, 0.0, 0.0,
                                             type, oit.key());
        } else {
          port = std::make_shared<GraphPort>(0.0, false, true, false, false,
                                             0.0, 0.0, type, oit.key());
        }
        const std::string name =
            oit.key() == "@name" ? built[node_order[n]]->get_name() : oit.key();
        built[node_order[n]]->add_output_port(name, port);
      }
    }

    // Members, between outputs and inputs: a group reads its members'
    // residual outputs, which have to exist, and the order a description
    // lists them in is the order their residual blocks appear.
    for (std::size_t n = 0; n < node_order.size(); ++n) {
      const SpecJson& node_spec = nodes[node_order[n]];
      const std::string node_where = where + " node '" + node_order[n] + "'";
      SpecJson::const_iterator members_it = node_spec.find("members");
      if (members_it == node_spec.end()) continue;
      if (!members_it->is_array()) refuse(node_where + " 'members' must be an array");
      for (SpecJson::const_iterator mit = members_it->begin();
           mit != members_it->end(); ++mit) {
        std::string member_key;
        std::string residual_key = "residuals";
        if (mit->is_string()) {
          member_key = mit->get<std::string>();
        } else if (mit->is_object()) {
          member_key = require_string(*mit, "node", node_where + " member");
          if (mit->contains("residuals")) {
            residual_key = (*mit)["residuals"].get<std::string>();
          }
        } else {
          refuse(node_where + " each member is a node name or an object");
        }
        std::map<std::string, std::shared_ptr<GraphNode> >::const_iterator
            member = built.find(member_key);
        if (member == built.end()) {
          refuse(node_where + " groups '" + member_key +
                 "', which this structure does not declare");
        }
        try {
          built[node_order[n]]->add_member_node(member->second, residual_key);
        } catch (const std::exception& error) {
          refuse(node_where + ": " + error.what());
        }
      }
    }

    // Pass three: inputs. A port that the node made itself is linked where
    // it stands; one the description introduces is created first. Either way
    // a candidate graph reads the canonical owner rather than a copy of it.
    for (std::size_t n = 0; n < node_order.size(); ++n) {
      const std::string node_key = node_order[n];
      const SpecJson& node_spec = nodes[node_key];
      const std::string node_where = where + " node '" + node_key + "'";
      SpecJson::const_iterator inputs = node_spec.find("inputs");
      if (inputs == node_spec.end()) continue;
      if (!inputs->is_object()) refuse(node_where + " 'inputs' must be an object");
      for (SpecJson::const_iterator iit = inputs->begin();
           iit != inputs->end(); ++iit) {
        const std::string port_name = iit.key();
        const std::string input_where =
            node_where + " input '" + port_name + "'";
        if (!iit->is_string()) refuse(input_where + " must be a reference");
        const std::string reference = iit->get<std::string>();
        std::shared_ptr<GraphNode> node = built[node_key];

        if (!reference.empty() && reference[0] == '#') {
          const std::string id = reference.substr(1);
          std::map<std::string, std::shared_ptr<GraphPort> >::const_iterator
              owner = owner_by_id.find(id);
          if (owner == owner_by_id.end()) {
            refuse(input_where + " follows '" + id +
                   "', which is not a canonical parameter");
          }
          std::shared_ptr<GraphPort> port = node->get_input_port(port_name);
          if (!port) {
            port = std::make_shared<GraphPort>(owner->second->get_value());
            node->add_input_port(port_name, port);
          }
          port->set_link(owner->second);
          used_ids.insert(id);
          continue;
        }

        if (!reference.empty() && reference[0] == '@') {
          const std::string body = reference.substr(1);
          const std::size_t dot = body.find('.');
          if (dot == std::string::npos) {
            refuse(input_where + " must name a measurement field, as "
                                 "'@<measurement>.<field>'");
          }
          const std::string slot = body.substr(0, dot);
          const std::string field = body.substr(dot + 1);
          std::map<std::string, FitDataset>::const_iterator data =
              impl_->datasets.find(slot);
          if (data == impl_->datasets.end()) {
            refuse(input_where + " reads '" + slot +
                   "', which nothing supplied");
          }
          std::vector<double> values;
          if (field == "values") {
            values = data->second.get_values();
          } else if (field == "axis") {
            values = data->second.get_number_of_coordinates() > 0
                         ? data->second.get_coordinate(0)
                         : std::vector<double>();
          } else if (field == "mask") {
            values = data->second.get_mask();
          } else {
            // Any coordinate the measurement names: a curve's lag, a carpet's
            // two spatial lags and its lag time -- as many as the data has.
            bool found_coordinate = false;
            for (int k = 0; k < data->second.get_number_of_coordinates(); ++k) {
              if (data->second.get_coordinate_name(k) == field) {
                values = data->second.get_coordinate(k);
                found_coordinate = true;
                break;
              }
            }
            if (!found_coordinate) {
              std::ostringstream offered;
              offered << "values, axis, mask";
              for (int k = 0; k < data->second.get_number_of_coordinates(); ++k) {
                offered << ", " << data->second.get_coordinate_name(k);
              }
              refuse(input_where + " reads '" + field +
                     "'; the measurement '" + slot + "' offers " + offered.str());
            }
          }
          if (values.empty()) {
            refuse(input_where + " reads '" + slot + "." + field +
                   "', which is empty");
          }
          std::shared_ptr<GraphPort> port = node->get_input_port(port_name);
          if (!port) {
            port = std::make_shared<GraphPort>(values, true, false, false,
                                               false, 0.0, 0.0,
                                               GRAPH_PORT_FLOAT_VECTOR,
                                               port_name);
            node->add_input_port(port_name, port);
          } else {
            port->set_value_vector(values);
          }
          continue;
        }

        // A bare node name means its primary output -- the port keyed by
        // the node's own name, which is where a single-valued node writes.
        const std::size_t dot = reference.find('.');
        std::string source_key = reference;
        std::string source_port;
        if (dot == std::string::npos) {
          if (!built.count(reference)) {
            refuse(input_where + " must be '#parameter', "
                                 "'@measurement.field', '<node>' or "
                                 "'<node>.<port>'");
          }
        } else {
          source_key = reference.substr(0, dot);
          source_port = reference.substr(dot + 1);
        }
        std::map<std::string, std::shared_ptr<GraphNode> >::const_iterator
            source = built.find(source_key);
        if (source == built.end()) {
          refuse(input_where + " follows node '" + source_key +
                 "', which this structure does not declare");
        }
        std::shared_ptr<GraphPort> from =
            source_port.empty()
                ? source->second->get_output_port(source->second->get_name())
                : source->second->get_output_port(source_port);
        if (!from) {
          refuse(input_where + " follows '" + reference +
                 "', which is not an output of that node");
        }
        std::shared_ptr<GraphPort> port = node->get_input_port(port_name);
        if (!port) {
          port = std::make_shared<GraphPort>(std::vector<double>(1, 0.0));
          node->add_input_port(port_name, port);
        }
        port->set_link(from);
      }
    }

    // --- which parameters this structure frees, and therefore its cost ---
    SpecJson::const_iterator free_it = structure.find("free");
    if (free_it == structure.end() || !free_it->is_array()) {
      refuse(where + " needs an array 'free'");
    }
    std::set<std::string> freed;
    for (SpecJson::const_iterator fit = free_it->begin();
         fit != free_it->end(); ++fit) {
      const std::string id = fit->get<std::string>();
      if (!owner_by_id.count(id)) {
        refuse(where + " frees '" + id + "', which is not a parameter");
      }
      freed.insert(id);
    }

    std::map<std::string, double> structure_initial;
    SpecJson::const_iterator init_it = structure.find("initial");
    if (init_it != structure.end()) {
      if (!init_it->is_object()) refuse(where + " 'initial' must be an object");
      for (SpecJson::const_iterator xit = init_it->begin();
           xit != init_it->end(); ++xit) {
        if (!owner_by_id.count(xit.key())) {
          refuse(where + " seeds '" + xit.key() + "', which is not a parameter");
        }
        structure_initial[xit.key()] =
            evaluate_rule(*xit, where + " seed '" + xit.key() + "'", scope);
      }
    }

    std::vector<double> initial_values;
    std::vector<int> fixed_mask;
    int complexity = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      const std::string& id = ids[i];
      double value = registry_initial[id];
      std::map<std::string, double>::const_iterator seeded =
          structure_initial.find(id);
      if (seeded != structure_initial.end()) value = seeded->second;
      if (impl_->overrides.count(id)) value = impl_->overrides[id].initial;
      // Free here *and* allowed to move at all: a description proposes, the
      // registry and the caller dispose.
      const bool active = freed.count(id) != 0 && registry_free[id];
      initial_values.push_back(value);
      fixed_mask.push_back(active ? 0 : 1);
      if (active) ++complexity;
    }

    const std::string objective_key =
        require_string(structure, "objective", where);
    std::map<std::string, std::shared_ptr<GraphNode> >::const_iterator
        objective = built.find(objective_key);
    if (objective == built.end()) {
      refuse(where + " names the objective '" + objective_key +
             "', which it does not declare");
    }
    std::string residual_key = "residuals";
    SpecJson::const_iterator res_it = structure.find("residual_key");
    if (res_it != structure.end()) residual_key = res_it->get<std::string>();

    problem->add_structure(key, objective->second, ids, owners, initial_values,
                           fixed_mask, residual_key);
    problem->set_structure_parameter_uses(
        key, std::vector<std::string>(used_ids.begin(), used_ids.end()));

    // Which node's curve is compared against which measurement. The objective
    // bound to a measurement reads its model from one node, so this is not a
    // guess -- and recording it lets the family be run backwards, generating
    // the measurement its parameters imply.
    for (std::size_t n = 0; n < node_order.size(); ++n) {
      const SpecJson& node_spec = nodes[node_order[n]];
      SpecJson::const_iterator bind = node_spec.find("bind");
      if (bind == node_spec.end()) continue;
      SpecJson::const_iterator data = bind->find("data");
      if (data == bind->end()) continue;
      const std::shared_ptr<GraphPort> model_port =
          built[node_order[n]]->get_input_port("model");
      if (!model_port || !model_port->get_link()) continue;
      const std::shared_ptr<GraphNode> source = model_port->get_link()->get_node();
      if (!source) continue;
      problem->set_structure_curve(key, data->get<std::string>(),
                                   source->get_name());
    }

    // Further declared starting points. One seed cannot be trusted to reach
    // the best fit a topology admits, and a topology judged on a bad basin
    // loses a model comparison it should win. Declared, so the best of them
    // still depends on the model and the data alone.
    SpecJson::const_iterator starts_it = structure.find("starts");
    if (starts_it != structure.end()) {
      if (!starts_it->is_array()) refuse(where + " 'starts' must be an array");
      for (SpecJson::const_iterator sit2 = starts_it->begin();
           sit2 != starts_it->end(); ++sit2) {
        if (!sit2->is_object()) {
          refuse(where + " each entry of 'starts' must be an object");
        }
        std::map<std::string, double> overrides;
        for (SpecJson::const_iterator oit = sit2->begin(); oit != sit2->end();
             ++oit) {
          if (!owner_by_id.count(oit.key())) {
            refuse(where + " start seeds '" + oit.key() +
                   "', which is not a parameter");
          }
          overrides[oit.key()] =
              evaluate_rule(*oit, where + " start '" + oit.key() + "'", scope);
        }
        std::vector<double> start_values = initial_values;
        for (std::size_t i = 0; i < ids.size(); ++i) {
          std::map<std::string, double>::const_iterator over =
              overrides.find(ids[i]);
          if (over == overrides.end()) continue;
          // A caller who pinned a value meant it; a description's alternative
          // start does not get to argue with them.
          if (impl_->overrides.count(ids[i])) continue;
          start_values[i] = over->second;
        }
        problem->add_structure_start(key, start_values);
      }
    }
    for (std::size_t n = 0; n < node_order.size(); ++n) {
      if (node_order[n] == objective_key) continue;
      problem->add_structure_node(key, built[node_order[n]]);
    }

    // How this topology is to be compared with the others. A family that
    // says nothing here is refused rather than ranked on misfit alone, which
    // is not a comparison between models but a preference for the larger.
    SpecJson::const_iterator ess_it = structure.find("ess");
    SpecJson::const_iterator selection_it = structure.find("selection");
    const bool has_score = structure.contains("score_output");
    if (ess_it == structure.end() && !has_score) {
      refuse(where + " declares no 'ess', so nothing says how many "
                     "observations its score is judged against; a family "
                     "without a selection criterion always prefers its "
                     "richest topology");
    }
    if (ess_it != structure.end()) {
      const double ess = evaluate_rule(*ess_it, where + " 'ess'", scope);
      if (!(ess > 0.0)) {
        refuse(where + " has a non-positive effective sample size");
      }
      ModelSelectionCriterion criterion = MODEL_SELECTION_BIC;
      if (selection_it != structure.end()) {
        const std::string name = selection_it->get<std::string>();
        if (name == "aic") {
          criterion = MODEL_SELECTION_AIC;
        } else if (name != "bic") {
          refuse(where + " selects on '" + name +
                 "', which is neither 'bic' nor 'aic'");
        }
      }
      // Complexity is counted, never declared: the two cannot disagree.
      problem->set_structure_selection(key, criterion, ess,
                                       static_cast<double>(complexity));
      // Optional: the chi-square test a structure must pass to be reported
      // as describing the data, which is not the same as winning.
      SpecJson::const_iterator accept_it = structure.find("acceptable_above");
      if (accept_it != structure.end()) {
        problem->set_structure_acceptance(
            key, evaluate_rule(*accept_it, where + " 'acceptable_above'",
                               scope));
      }
    }
    SpecJson::const_iterator score_it = structure.find("score_output");
    if (score_it != structure.end()) {
      problem->set_structure_score_output(key, score_it->get<std::string>());
    }
    SpecJson::const_iterator ok_it = structure.find("acceptable_output");
    if (ok_it != structure.end()) {
      problem->set_structure_acceptable_output(key,
                                               ok_it->get<std::string>());
    }
  }

  // How much work one candidate's fit may do. MINPACK's 200 * (n + 1) is a
  // convention for fitting n parameters once; a joint problem whose members
  // share strongly coupled parameters converges more slowly than that allows,
  // and a family knows its own difficulty better than the default does.
  SpecJson::const_iterator budget_it = document.find("maxfev");
  if (budget_it != document.end()) {
    const double budget =
        evaluate_rule(*budget_it, "the description 'maxfev'", scope);
    if (!(budget > 0.0)) refuse("'maxfev' must be positive");
    problem->set_minimizer_maxfev(static_cast<int>(budget));
  }

  problem->set_initial_structure(
      require_string(document, "initial_structure", "the description"));

  SpecJson::const_iterator actions = document.find("actions");
  if (actions != document.end()) {
    if (!actions->is_array()) refuse("'actions' must be an array");
    for (SpecJson::const_iterator ait = actions->begin();
         ait != actions->end(); ++ait) {
      const SpecJson& action = *ait;
      const std::string where = "action '" +
                                (action.contains("action")
                                     ? action["action"].get<std::string>()
                                     : std::string("?")) +
                                "'";
      const std::string from = require_string(action, "from", where);
      const std::string name = require_string(action, "action", where);
      const std::string to = require_string(action, "to", where);
      double prior = 1.0;
      if (action.contains("prior")) prior = action["prior"].get<double>();
      bool terminal = false;
      if (action.contains("terminal")) {
        terminal = action["terminal"].get<bool>();
      }
      problem->add_action(from, name, to, prior, terminal);
    }
  }
  return problem;
}

IMPBFF_END_NAMESPACE
