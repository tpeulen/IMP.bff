/**
 * \file Sampling.cpp
 * \brief Kernels by registry name: registration with a factory, option checking against the entry's
 *        JSON Schema, construction.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 * PRD-147 amendment A2 (R2): a sampler exists by being registered; nothing here lists samplers.
 */

#include <IMP/bff/Sampling.h>
#include <IMP/bff/Registry.h>

#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/ordered_map.h>

#include <cmath>
#include <deque>
#include <mutex>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace {

using SamplingJson = nlohmann::basic_json<nlohmann::ordered_map>;

//! Factories live as long as the process; the registry keeps their addresses as impl handles.
std::deque<SamplerKernelFactory>& sampler_factories() {
  static std::deque<SamplerKernelFactory> f;
  return f;
}
std::mutex& sampler_factories_mutex() {
  static std::mutex m;
  return m;
}

std::string option_doc(const SamplingJson& spec) {
  std::string d = spec.contains("description") && spec["description"].is_string() ? spec["description"].get<std::string>() : "";
  std::string t = spec.contains("title") && spec["title"].is_string() ? spec["title"].get<std::string>() : "";
  return t.empty() ? d : (d.empty() ? t : t + ": " + d);
}

//! Check options against a JSON Schema object's properties: declared keys, types, bounds, enums.
void check_options(const std::string& name, const SamplingJson& schema, const SamplingJson& options) {
  if (!options.is_object()) throw SamplerConfigurationError("sampler '" + name + "': options must be a JSON object");
  const SamplingJson props = schema.contains("properties") ? schema["properties"] : SamplingJson::object();
  for (auto it = options.begin(); it != options.end(); ++it) {
    if (!props.contains(it.key())) {
      std::ostringstream m;
      m << "sampler '" << name << "' has no option '" << it.key() << "'; declared:";
      for (auto p = props.begin(); p != props.end(); ++p) m << " " << p.key();
      throw SamplerConfigurationError(m.str());
    }
    const SamplingJson& spec = props[it.key()];
    const SamplingJson& v = it.value();
    const std::string type = spec.contains("type") && spec["type"].is_string() ? spec["type"].get<std::string>() : "";
    const std::string where = "sampler '" + name + "' option '" + it.key() + "' (" + option_doc(spec) + ")";
    bool ok = true;
    if (type == "integer") ok = v.is_number_integer() || (v.is_number() && std::floor(v.get<double>()) == v.get<double>());
    else if (type == "number") ok = v.is_number();
    else if (type == "boolean") ok = v.is_boolean();
    else if (type == "string") ok = v.is_string();
    else if (type == "array") ok = v.is_array();
    if (!ok) throw SamplerConfigurationError(where + ": expected " + type);
    if (v.is_number()) {
      const double x = v.get<double>();
      if (spec.contains("minimum") && x < spec["minimum"].get<double>()) throw SamplerConfigurationError(where + ": below the minimum");
      if (spec.contains("maximum") && x > spec["maximum"].get<double>()) throw SamplerConfigurationError(where + ": above the maximum");
      if (spec.contains("exclusiveMinimum") && !(x > spec["exclusiveMinimum"].get<double>())) throw SamplerConfigurationError(where + ": must exceed the minimum");
      if (spec.contains("exclusiveMaximum") && !(x < spec["exclusiveMaximum"].get<double>())) throw SamplerConfigurationError(where + ": must stay below the maximum");
    }
    if (spec.contains("enum")) {
      bool found = false;
      for (const auto& e : spec["enum"]) found = found || e == v;
      if (!found) throw SamplerConfigurationError(where + ": not one of the allowed values");
    }
  }
}

}  // namespace

bool register_sampler_kernel(const std::string& key, const std::string& entry_json, SamplerKernelFactory factory) {
  SamplerKernelFactory* handle;
  {
    std::lock_guard<std::mutex> lock(sampler_factories_mutex());
    sampler_factories().push_back(std::move(factory));
    handle = &sampler_factories().back();
  }
  return register_algorithm_json("sampler", key, entry_json, handle);
}

std::unique_ptr<SamplerKernel> create_sampler_kernel(const std::string& name_or_alias, const std::string& options_json) {
  const SamplingJson entries = SamplingJson::parse(registry_category_json("sampler"));
  std::string key;
  if (entries.contains(name_or_alias)) {
    key = name_or_alias;
  } else {
    for (auto it = entries.begin(); it != entries.end() && key.empty(); ++it)
      if (it.value().contains("aliases"))
        for (const auto& a : it.value()["aliases"])
          if (a.is_string() && a.get<std::string>() == name_or_alias) key = it.key();
  }
  if (key.empty()) {
    std::ostringstream m;
    m << "unknown sampler '" << name_or_alias << "'; registered:";
    for (auto it = entries.begin(); it != entries.end(); ++it) m << " " << it.key();
    throw SamplerConfigurationError(m.str());
  }
  const SamplingJson options = SamplingJson::parse(options_json.empty() ? std::string("{}") : options_json, nullptr, false);
  if (options.is_discarded()) throw SamplerConfigurationError("sampler '" + key + "': options are not valid JSON");
  const SamplingJson& entry = entries[key];
  check_options(key, entry.contains("params_schema") ? entry["params_schema"] : SamplingJson::object(), options);
  auto* factory = static_cast<SamplerKernelFactory*>(registry_impl(key));
  if (!factory || !*factory) throw SamplerConfigurationError("sampler '" + key + "' is registered without a factory");
  return (*factory)(options.dump());
}

IMPBFF_END_NAMESPACE
