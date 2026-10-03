/**
 * \file BffSettings.cpp
 * \brief The user's settings file.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/BffSettings.h>
#include <IMP/bff/internal/json.h>

#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

namespace {

std::string parent_directory(const std::string& path) {
    const std::string::size_type slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

bool is_absolute(const std::string& path) {
    if (path.empty()) return false;
    if (path[0] == '/' || path[0] == '\\' || path[0] == '~') return true;
    return path.size() > 1 && path[1] == ':';   // C:\...
}

std::string expand_home(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    const char* home = std::getenv("HOME");
    if (home == nullptr) home = std::getenv("USERPROFILE");
    return home == nullptr ? path : std::string(home) + path.substr(1);
}

#ifdef _WIN32
constexpr char kDirSep = '\\';   // kPathSep is Config.cpp's list separator
#else
constexpr char kDirSep = '/';
#endif

//! A path from the settings: `~` expanded, relative ones against the file.
//! `/` inside a configured path becomes the platform's separator: a settings
//! file written on one OS is read on another, and callers compare against
//! paths the OS built (Python's pathlib normalises both ways).
std::string resolve(const std::string& path, const std::string& settings_dir) {
    std::string p = expand_home(path);
    if (p.empty()) return p;
#ifdef _WIN32
    for (char& c : p)
        if (c == '/') c = '\\';
#endif
    if (is_absolute(p) || settings_dir.empty()) return p;
    return settings_dir + kDirSep + p;
}

std::string string_map_entry(const nlohmann::json& value, const std::string& where) {
    if (!value.is_string())
        IMP_THROW("settings: " << where << " must be a path (a string)", ValueException);
    return value.get<std::string>();
}

double number(const nlohmann::json& value, const std::string& where) {
    if (!value.is_number()) IMP_THROW("settings: " << where << " must be a number", ValueException);
    return value.get<double>();
}

void refuse_unknown(const nlohmann::json& object, const std::set<std::string>& known,
                    const std::string& where) {
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (known.count(it.key()) == 0)
            IMP_THROW("settings: unknown key '" << it.key() << "' in " << where,
                      ValueException);
    }
}

}  // namespace

std::string get_settings_path() {
    const char* explicit_path = std::getenv("IMP_BFF_SETTINGS");
    if (explicit_path != nullptr && *explicit_path != '\0') return expand_home(explicit_path);
    const char* config = std::getenv("XDG_CONFIG_HOME");
    if (config != nullptr && *config != '\0')
        return expand_home(std::string(config) + kDirSep + std::string("imp.bff") +
                           kDirSep + "settings.json");
#ifdef _WIN32
    const char* appdata = std::getenv("APPDATA");
    if (appdata != nullptr) return std::string(appdata) + "\\imp.bff\\settings.json";
#endif
    return expand_home("~/.config/imp.bff/settings.json");
}

std::vector<std::string> SequenceSearchSettings::get_database_names() const {
    std::vector<std::string> out;
    for (const auto& kv : databases_) out.push_back(kv.first);
    return out;
}

bool SequenceSearchSettings::get_has_database(const std::string& name) const {
    return databases_.count(name) != 0;
}

std::string SequenceSearchSettings::get_database(const std::string& name) const {
    const auto it = databases_.find(name);
    if (it == databases_.end())
        IMP_THROW("settings: no sequence database '" << name << "' (settings file: "
                  << get_settings_path() << ")", ValueException);
    return it->second;
}

void SequenceSearchSettings::set_database(const std::string& name, const std::string& path) {
    databases_[name] = path;
}

std::string SequenceSearchSettings::get_index(const std::string& name) const {
    const auto it = indexes_.find(name);
    return it == indexes_.end() ? std::string() : it->second;
}

void SequenceSearchSettings::set_index(const std::string& name, const std::string& path) {
    indexes_[name] = path;
}

std::string SequenceSearchSettings::get_representatives(const std::string& name) const {
    const auto it = representatives_.find(name);
    return it == representatives_.end() ? std::string() : it->second;
}

void SequenceSearchSettings::set_representatives(const std::string& name, const std::string& path) {
    representatives_[name] = path;
}

std::vector<std::string> SequenceSearchSettings::get_server_names() const {
    std::vector<std::string> out;
    for (const auto& kv : servers_) out.push_back(kv.first);
    return out;
}

bool SequenceSearchSettings::get_has_server(const std::string& name) const {
    return servers_.count(name) != 0;
}

SequenceSearchServer SequenceSearchSettings::get_server(const std::string& name) const {
    const auto it = servers_.find(name);
    if (it == servers_.end())
        IMP_THROW("settings: no sequence search server '" << name << "'", ValueException);
    return it->second;
}

void SequenceSearchSettings::set_server(const SequenceSearchServer& server) {
    if (server.name.empty()) IMP_THROW("settings: a server needs a name", ValueException);
    servers_[server.name] = server;
}

SequenceSearchSettings get_sequence_search_settings(const std::string& path) {
    const std::string file = path.empty() ? get_settings_path() : expand_home(path);
    SequenceSearchSettings out;
    std::ifstream in(file);
    if (!in) return out;
    std::stringstream text;
    text << in.rdbuf();
    nlohmann::json doc;
    try {
        doc = nlohmann::json::parse(text.str());
    } catch (const std::exception& error) {
        IMP_THROW("settings: " << file << " is not valid JSON: " << error.what(), ValueException);
    }
    if (!doc.is_object()) IMP_THROW("settings: " << file << " must hold a JSON object", ValueException);
    const auto section = doc.find("sequence_search");
    if (section == doc.end()) return out;
    const nlohmann::json& s = *section;
    if (!s.is_object()) IMP_THROW("settings: sequence_search must be an object", ValueException);
    refuse_unknown(s, {"databases", "default_database", "indexes", "clusters", "servers",
                       "fallback_server", "threads", "memory_budget_mb"}, "sequence_search");
    const std::string dir = parent_directory(file);

    for (const char* key : {"databases", "indexes", "clusters"}) {
        const auto it = s.find(key);
        if (it == s.end()) continue;
        if (!it->is_object())
            IMP_THROW("settings: sequence_search." << key << " must map names to paths",
                      ValueException);
        for (auto e = it->begin(); e != it->end(); ++e) {
            const std::string where = std::string("sequence_search.") + key + "." + e.key();
            const std::string p = resolve(string_map_entry(e.value(), where), dir);
            if (std::string(key) == "databases") out.set_database(e.key(), p);
            else if (std::string(key) == "indexes") out.set_index(e.key(), p);
            else out.set_representatives(e.key(), p);
        }
    }
    const auto servers = s.find("servers");
    if (servers != s.end()) {
        if (!servers->is_object())
            IMP_THROW("settings: sequence_search.servers must map names to servers",
                      ValueException);
        for (auto e = servers->begin(); e != servers->end(); ++e) {
            const std::string where = "sequence_search.servers." + e.key();
            if (!e.value().is_object()) IMP_THROW("settings: " << where << " must be an object",
                                                  ValueException);
            refuse_unknown(e.value(), {"url", "protocol", "poll_seconds", "timeout_seconds"}, where);
            SequenceSearchServer server;
            server.name = e.key();
            const auto url = e.value().find("url");
            if (url == e.value().end() || !url->is_string())
                IMP_THROW("settings: " << where << " needs a url", ValueException);
            server.url = url->get<std::string>();
            const auto protocol = e.value().find("protocol");
            if (protocol != e.value().end()) server.protocol = string_map_entry(*protocol, where + ".protocol");
            const auto poll = e.value().find("poll_seconds");
            if (poll != e.value().end()) server.poll_seconds = number(*poll, where + ".poll_seconds");
            const auto timeout = e.value().find("timeout_seconds");
            if (timeout != e.value().end())
                server.timeout_seconds = number(*timeout, where + ".timeout_seconds");
            out.set_server(server);
        }
    }
    const auto default_database = s.find("default_database");
    if (default_database != s.end())
        out.default_database = string_map_entry(*default_database, "sequence_search.default_database");
    const auto fallback = s.find("fallback_server");
    if (fallback != s.end())
        out.fallback_server = string_map_entry(*fallback, "sequence_search.fallback_server");
    const auto threads = s.find("threads");
    if (threads != s.end()) out.threads = static_cast<int>(number(*threads, "sequence_search.threads"));
    const auto budget = s.find("memory_budget_mb");
    if (budget != s.end()) out.memory_budget_mb = number(*budget, "sequence_search.memory_budget_mb");

    if (!out.default_database.empty() && !out.get_has_database(out.default_database))
        IMP_THROW("settings: default_database '" << out.default_database
                  << "' is not among sequence_search.databases", ValueException);
    if (!out.fallback_server.empty() && !out.get_has_server(out.fallback_server))
        IMP_THROW("settings: fallback_server '" << out.fallback_server
                  << "' is not among sequence_search.servers", ValueException);
    return out;
}

IMPBFF_END_NAMESPACE
