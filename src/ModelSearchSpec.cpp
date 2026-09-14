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
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    maximum = std::max(maximum, values[i]);
    total += values[i];
  }
  total = std::max(maximum, total);

  const double first_value = n ? values[first] : 0.0;
  const double last_value = (last > 0 && last <= n) ? values[last - 1] : 0.0;
  const double amplitude = first_value - last_value;

  scope[slot + "_size"] = static_cast<double>(n);
  scope[slot + "_max"] = maximum;
  scope[slot + "_sum"] = total;
  scope[slot + "_first"] = first_value;
  scope[slot + "_last"] = last_value;
  scope[slot + "_amplitude"] = amplitude;

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
  };
  std::map<std::string, Override> overrides;
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
  const SpecJson& parameters = impl_->document["parameters"];
  for (SpecJson::const_iterator it = parameters.begin();
       it != parameters.end(); ++it) {
    ids.push_back(it.key());
  }
  return ids;
}

std::vector<std::string> ModelSearchSpec::get_structure_keys() const {
  std::vector<std::string> keys;
  const SpecJson& structures = impl_->document["structures"];
  for (SpecJson::const_iterator it = structures.begin();
       it != structures.end(); ++it) {
    keys.push_back(it.key());
  }
  return keys;
}

void ModelSearchSpec::set_dataset(const std::string& name,
                                  const FitDataset& dataset) {
  impl_->datasets[name] = dataset;
}

void ModelSearchSpec::set_scalar(const std::string& name, double value) {
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
}

std::vector<std::string> ModelSearchSpec::get_available_names() {
  // The shipped families are files; listing them is the capability answer.
  static const char* kShipped[] = {"fcs_analytical", "tcspc_lifetime"};
  return std::vector<std::string>(kShipped, kShipped + 2);
}

std::shared_ptr<MultiStructureModelSearchProblem> ModelSearchSpec::build()
    const {
  // Everything the description says it needs has to be here before anything
  // is built. A half-wired graph still evaluates, and fits the wrong thing.
  std::map<std::string, double> scope = impl_->scalars;
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
  const SpecJson& parameters = impl_->document["parameters"];
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
    double lower = optional_rule(entry, "lower", 0.0, where, scope);
    double upper = optional_rule(entry, "upper", 0.0, where, scope);
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
      lower = over->second.lower;
      upper = over->second.upper;
      free = over->second.free;
    }
    if (!(upper > lower)) {
      refuse(where + " has an empty range [" + std::to_string(lower) + ", " +
             std::to_string(upper) + "]");
    }
    std::shared_ptr<GraphPort> owner = std::make_shared<GraphPort>(
        initial, false, false, false, true, lower, upper, GRAPH_PORT_FLOAT,
        name);
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
  const SpecJson& structures = impl_->document["structures"];
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
          if (rit->is_array()) {
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
            refuse(input_where + " reads '" + field +
                   "'; a measurement offers values, axis or mask");
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

    SpecJson::const_iterator ess_it = structure.find("ess");
    if (ess_it != structure.end()) {
      const double ess =
          evaluate_rule(*ess_it, where + " 'ess'", scope);
      if (!(ess > 0.0)) {
        refuse(where + " has a non-positive effective sample size");
      }
      // Complexity is counted, never declared: the two cannot disagree.
      problem->set_structure_bic_metadata(key, ess,
                                          static_cast<double>(complexity));
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

  problem->set_initial_structure(
      require_string(impl_->document, "initial_structure", "the description"));

  SpecJson::const_iterator actions = impl_->document.find("actions");
  if (actions != impl_->document.end()) {
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
