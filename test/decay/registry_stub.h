// registry_stub.h -- TEST ONLY (PRD-156 step 3, 2026-11-11, claude-code/Opus 5.5 for tpeulen).
// The standalone C++ drivers of test/decay compile bff sources directly; the real registry
// (src/util/Registry.cpp) pulls in model search and data paths they do not build. This stands in
// for the three registry functions the factor graph's save/load calls, with the same contract
// (a taken key is refused; registry_category_json lists one capability's keys). The real registry
// is exercised through the wheel: test/decay/test_inference_factor_graph_save_data_python.py.
#include <IMP/bff/util/Registry.h>
#include <map>
#include <utility>
IMPBFF_BEGIN_NAMESPACE
namespace {
std::map<std::string, std::pair<std::string, void*> >& stub_table() {
  static std::map<std::string, std::pair<std::string, void*> > t;
  return t;
}
}  // namespace
bool register_algorithm_json(const std::string& capability, const std::string& key, const std::string&, void* impl) {
  return stub_table().emplace(key, std::make_pair(capability, impl)).second;
}
void* registry_impl(const std::string& key) {
  auto it = stub_table().find(key);
  return it == stub_table().end() ? nullptr : it->second.second;
}
std::string registry_category_json(const std::string& category) {
  std::string out = "{";
  for (const auto& kv : stub_table())
    if (kv.second.first == category) out += std::string(out.size() > 1 ? "," : "") + "\"" + kv.first + "\": {}";
  return out + "}";
}
IMPBFF_END_NAMESPACE
