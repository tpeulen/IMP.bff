/**
 * \file Registry.cpp
 * \brief bff's one registry table, on tttrlib's mechanism (internal/RegistryCore.h).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/Registry.h>

#include <IMP/bff/internal/RegistryCore.h>
#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/ordered_map.h>

#include <IMP/bff/ModelSearchSpec.h>
#include <IMP/bff/ProbeDataPaths.h>

#include <algorithm>
#include <fstream>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace {

//! Key order is part of the surface (a schema's properties render in declared order).
using RegistryJson = nlohmann::basic_json<nlohmann::ordered_map>;

//! Function-local, so a static initialiser in any translation unit registers into a constructed table.
::tttrlib::AlgorithmTable<RegistryJson>& table() {
  static ::tttrlib::AlgorithmTable<RegistryJson> t;
  return t;
}

//! The shipped model-search families, one entry each, from the family files themselves.
RegistryJson model_search_entries() {
  RegistryJson out = RegistryJson::object();
  for (const std::string& name : ModelSearchSpec::get_available_names()) {
    RegistryJson e = RegistryJson::object();
    std::string title, family = name;
    // The family file declares its own `family` and `title`; read them rather than restate them.
    std::ifstream in(get_data_path("model_search/" + name + ".json").c_str());
    std::ostringstream text;
    if (in) text << in.rdbuf();
    const RegistryJson doc = RegistryJson::parse(text.str(), nullptr, false);
    if (doc.is_object()) {
      if (doc.contains("family") && doc["family"].is_string()) family = doc["family"].get<std::string>();
      if (doc.contains("title") && doc["title"].is_string()) title = doc["title"].get<std::string>();
    } else {
      // A family file that does not parse is reported, not dropped: the entry says so.
      title = "(the family file does not parse)";
    }
    e["name"] = name;
    e["label"] = title.empty() ? family : title;
    e["summary"] = title;
    e["description"] = "";
    e["params_schema"] = RegistryJson::object();
    e["capability"] = "model_search";
    e["provider"] = "imp.bff";
    e["api"] = RegistryJson::array({"ModelSearchSpec.from_name"});
    e["family"] = family;
    out[name] = e;
  }
  return out;
}

RegistryJson build() {
  RegistryJson root = table().categories();
  RegistryJson ms = model_search_entries();
  if (!ms.empty()) root["model_search"] = ms;
  // categories() orders capabilities alphabetically; keep that for the assembled one too
  RegistryJson sorted = RegistryJson::object();
  std::vector<std::string> keys;
  for (auto it = root.begin(); it != root.end(); ++it) keys.push_back(it.key());
  std::sort(keys.begin(), keys.end());
  for (const std::string& k : keys) sorted[k] = root[k];
  return sorted;
}

std::string with_provider(const std::string& entry_json) {
  RegistryJson e = RegistryJson::parse(entry_json, nullptr, false);
  if (e.is_discarded() || !e.is_object()) return entry_json;
  if (!e.contains("provider")) e["provider"] = "imp.bff";
  return e.dump();
}

}  // namespace

bool register_algorithm_json(const std::string& capability, const std::string& key,
                             const std::string& entry_json) {
  return table().add_json(capability, key, with_provider(entry_json));
}

bool register_algorithm_json(const std::string& capability, const std::string& key,
                             const std::string& entry_json, void* impl) {
  return table().add_json(capability, key, with_provider(entry_json), impl);
}

std::string registry_json() { return build().dump(2); }

std::string registry_category_json(const std::string& category) {
  const RegistryJson root = build();
  if (!root.contains(category)) return "{}";
  return root[category].dump(2);
}

std::vector<std::string> registry_categories() {
  const RegistryJson root = build();
  std::vector<std::string> out;
  for (auto it = root.begin(); it != root.end(); ++it) out.push_back(it.key());
  return out;
}

std::vector<std::string> registry_keys(const std::string& capability) {
  return table().keys(capability);
}

void* registry_impl(const std::string& key) {
  const ::tttrlib::AlgorithmDescriptor* d = table().find(key);
  return d ? d->impl : nullptr;
}

IMPBFF_END_NAMESPACE
