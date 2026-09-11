/**
 * \file PotentialTables.cpp
 * \brief The parameter tables the coarse-grained potentials read, in one file.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/PotentialTables.h>

#include <IMP/bff/DataPaths.h>
#include <IMP/bff/Pto.h>
#include <IMP/bff/internal/OutputView.h>
#include <IMP/bff/internal/Text.h>
#include <IMP/bff/internal/json.h>

#include <IMP/bff/Base.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>

// The vendored codec, the same one `.drot` uses. The paths are relative to
// `src/`, which is where this file also sits.

IMPBFF_BEGIN_NAMESPACE

namespace {

//! Decode the pre-PTO-codec framing used by the shipped v1 container.
std::vector<unsigned char> pot_brotli_decompress(const unsigned char* data,
                                                 std::size_t size) {
    if (size < 8) {
        IMP_THROW("the potential container's payload is truncated",
                  IOException);
    }
    unsigned long long raw = 0;
    for (int i = 0; i < 8; ++i) {
        raw |= static_cast<unsigned long long>(data[i]) << (8 * i);
    }
    if (raw > (unsigned long long)1 << 32) {
        IMP_THROW("the potential container claims a " << raw
                                                      << " octet payload",
                  IOException);
    }
    std::vector<unsigned char> out;
    if (!pto::decompress_bytes("brotli", data + 8, size - 8,
                               static_cast<std::size_t>(raw), out)) {
        IMP_THROW("the potential container's payload does not decompress",
                  IOException);
    }
    return out;
}

//! Read a payload, accepting the v1 manually framed container during migration.
std::vector<unsigned char> read_potential_payload(const PtoReader& reader,
                                                   const PtoObject& object) {
    try {
        return reader.data(object);
    } catch (const std::exception&) {
        const std::vector<unsigned char> stored =
                reader.file().read_stored(object.uid);
        return pot_brotli_decompress(stored.empty() ? NULL : &stored[0],
                                     stored.size());
    }
}

const char* const kGridKind = "pot.grid";
const char* const kPmfKind = "pot.pmf";
const char* const kManifestName = "manifest.json";

//! The container to read, given what the caller asked for.
std::string container_path(const std::string& path) {
    const std::string p = path.empty() ? get_potential_container_path() : path;
    if (!internal::file_exists(p)) {
        IMP_THROW("no potential container at " << p, IOException);
    }
    return p;
}

//! `name` with the suffix the container stores it under.
std::string stored_name(const std::string& name, const std::string& kind) {
    if (kind == kPmfKind) return name + ".pmf";
    return name;
}

//! Little-endian float64 in, doubles out.
std::vector<double> to_doubles(const std::vector<unsigned char>& bytes) {
    std::vector<double> out(bytes.size() / sizeof(double));
    if (!out.empty()) std::memcpy(&out[0], &bytes[0], out.size() * sizeof(double));
    return out;
}

std::vector<unsigned char> to_bytes(const std::vector<double>& values) {
    std::vector<unsigned char> out(values.size() * sizeof(double));
    if (!out.empty()) std::memcpy(&out[0], &values[0], out.size());
    return out;
}

//! The shape the manifest records for this table, or empty.
std::vector<int> shape_from_manifest(const std::string& manifest,
                                     const std::string& name) {
    std::vector<int> shape;
    nlohmann::json j = nlohmann::json::parse(manifest, NULL, false);
    if (j.is_discarded() || !j.is_object()) return shape;
    if (!j.contains("tables") || !j["tables"].is_object()) return shape;
    const nlohmann::json& tables = j["tables"];
    if (!tables.contains(name)) return shape;
    const nlohmann::json& entry = tables[name];
    if (!entry.is_object() || !entry.contains("shape")) return shape;
    const nlohmann::json& s = entry["shape"];
    if (!s.is_array()) return shape;
    for (nlohmann::json::const_iterator it = s.begin(); it != s.end(); ++it) {
        if (it->is_number()) shape.push_back(it->get<int>());
    }
    return shape;
}

}  // namespace

// ---------------------------------------------------------------------------

void PotentialTable::get_values(double** out_view, int* n_out_view) const {
    internal::copy_to_view(values, out_view, n_out_view);
}

int PotentialTable::get_size() const {
    if (shape.empty()) return 0;
    int n = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) n *= shape[i];
    return n;
}

std::string get_potential_container_path() {
    return IMP::bff::get_data_path("potentials.pto");
}

std::vector<std::string> potential_table_names(std::string path) {
    PtoReader reader(container_path(path));
    const std::vector<PtoObject>& objects = reader.objects();
    std::vector<std::string> out;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        if (objects[i].name == kManifestName) continue;
        std::string name = objects[i].name;
        if (objects[i].kind == kPmfKind && internal::ends_with(name, ".pmf")) {
            name = name.substr(0, name.size() - 4);
        }
        out.push_back(name);
    }
    return out;
}

std::string read_potential_manifest(std::string path) {
    PtoReader reader(container_path(path));
    const int i = reader.find(kManifestName);
    if (i < 0) return std::string("{}");
    // ptolib's reader decodes by the object's encoding -- no second pass.
    const std::vector<unsigned char> raw =
            read_potential_payload(reader, reader.objects()[i]);
    return std::string(raw.begin(), raw.end());
}

PotentialTable read_potential_table(std::string name, std::string path) {
    const std::string file = container_path(path);
    PtoReader reader(file);
    // A table is stored either as text (`name.pmf`) or as a grid (`name`).
    int i = reader.find(stored_name(name, kPmfKind));
    if (i < 0) i = reader.find(name);
    if (i < 0) {
        IMP_THROW("the container at " << file << " has no table called `"
                                      << name << "`",
                  IOException);
    }
    const PtoObject& object = reader.objects()[i];
    // ptolib's reader decodes by the object's encoding -- no second pass.
    const std::vector<unsigned char> raw = read_potential_payload(reader, object);

    PotentialTable table(name, object.kind);
    if (object.kind == kPmfKind) {
        table.text.assign(raw.begin(), raw.end());
    } else {
        table.values = to_doubles(raw);
        table.shape = shape_from_manifest(read_potential_manifest(file), name);
        if (table.shape.empty()) {
            table.shape.push_back(static_cast<int>(table.values.size()));
        }
    }
    return table;
}

void write_potential_tables(const std::string& path,
                            const std::vector<PotentialTable>& tables,
                            const std::string& manifest_json) {
    PtoWriter writer(path);
    const std::vector<unsigned char> manifest(manifest_json.begin(),
                                              manifest_json.end());
    writer.add_coded(kManifestName, "pot.manifest", "utf8", "brotli",
                     manifest.empty() ? NULL : &manifest[0], manifest.size());

    for (std::size_t i = 0; i < tables.size(); ++i) {
        const PotentialTable& t = tables[i];
        const bool is_text = t.kind == kPmfKind || !t.text.empty();
        const std::vector<unsigned char> raw =
                is_text ? std::vector<unsigned char>(t.text.begin(),
                                                     t.text.end())
                        : to_bytes(t.values);
        writer.add_coded(stored_name(t.name, is_text ? kPmfKind : kGridKind),
                         is_text ? kPmfKind : kGridKind,
                         is_text ? "utf8" : "f64", "brotli",
                         raw.empty() ? NULL : &raw[0], raw.size(), 11);
    }
    writer.close();
}

IMPBFF_END_NAMESPACE
