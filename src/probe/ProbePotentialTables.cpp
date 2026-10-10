/**
 * \file ProbePotentialTables.cpp
 * \brief The parameter tables the coarse-grained potentials read, in one file.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/ProbePotentialTables.h>

#include <IMP/bff/ProbeDataPaths.h>
#include <IMP/bff/internal/ptolib.h>
#include <IMP/bff/internal/OutputView.h>
#include <IMP/bff/internal/Text.h>
#include <IMP/bff/internal/json.h>

#include <IMP/bff/IMPCompatibility.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <memory>

IMPBFF_BEGIN_NAMESPACE

namespace {

//! Decode the legacy eight-byte-size-prefixed Brotli payload.
/*! Older potential containers label this private envelope +brotli. New
    containers use standard Zstd frames and ptolib's generic reader. */
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
    pto::File reader;
    if (!reader.open(container_path(path), false)) {
        IMP_THROW("PTO: cannot open container " << container_path(path) << ": " << reader.error(), IOException);
    }
    const std::vector<pto::PtoObject>& objects = reader.objects();
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
    pto::File reader;
    if (!reader.open(container_path(path), false)) {
        IMP_THROW("PTO: cannot open container " << container_path(path) << ": " << reader.error(), IOException);
    }
    const std::vector<std::uint64_t> matches = reader.find_all(kManifestName);
    if (matches.empty()) return std::string("{}");
    const pto::PtoObject object = reader.object(matches.front());
    std::vector<unsigned char> raw;
    if (pto::split_encoding(object.encoding).codec == "brotli") {
        const std::vector<unsigned char> stored = reader.read_stored(object.uid);
        raw = pot_brotli_decompress(stored.data(), stored.size());
    } else {
        raw = reader.read(object.uid);
    }
    return std::string(raw.begin(), raw.end());
}

PotentialTable read_potential_table(std::string name, std::string path) {
    const std::string file = container_path(path);
    pto::File reader;
    if (!reader.open(file, false)) {
        IMP_THROW("PTO: cannot open container " << file << ": " << reader.error(), IOException);
    }
    // A table is stored either as text (`name.pmf`) or as a grid (`name`).
    std::vector<std::uint64_t> matches = reader.find_all(stored_name(name, kPmfKind));
    if (matches.empty()) matches = reader.find_all(name);
    if (matches.empty()) {
        IMP_THROW("the container at " << file << " has no table called `"
                                      << name << "`",
                  IOException);
    }
    const pto::PtoObject object = reader.object(matches.front());
    std::vector<unsigned char> raw;
    if (pto::split_encoding(object.encoding).codec == "brotli") {
        const std::vector<unsigned char> stored = reader.read_stored(object.uid);
        raw = pot_brotli_decompress(stored.data(), stored.size());
    } else {
        raw = reader.read(object.uid);
    }

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
    if (!pto::can_compress("zstd")) {
        IMP_THROW("write_potential_tables: unavailable output codec 'zstd'",
                  IOException);
    }
    pto::File writer;
    if (!writer.create(path, "", std::string(pto::kDefaultBanner) +
            "\nThis container was written by IMP.bff.\nhttps://github.com/tpeulen/IMP.bff\n")) {
        IMP_THROW("PTO: cannot open " << path << " for writing: " << writer.error(), IOException);
    }
    writer.set_writing_app("IMP.bff");
    const std::vector<unsigned char> manifest(manifest_json.begin(),
                                              manifest_json.end());
    if (!writer.add_coded("pot.manifest", "utf8", "zstd", kManifestName,
                          manifest.data(), manifest.size(), 3)) {
        IMP_THROW("PTO: writing manifest failed: " << writer.error(), IOException);
    }

    for (std::size_t i = 0; i < tables.size(); ++i) {
        const PotentialTable& t = tables[i];
        const bool is_text = t.kind == kPmfKind || !t.text.empty();
        const std::vector<unsigned char> raw =
                is_text ? std::vector<unsigned char>(t.text.begin(),
                                                     t.text.end())
                        : to_bytes(t.values);
        if (!writer.add_coded(is_text ? kPmfKind : kGridKind,
                   is_text ? "utf8" : "f64", "zstd",
                   stored_name(t.name, is_text ? kPmfKind : kGridKind),
                   raw.data(), raw.size(), 3)) {
            IMP_THROW("PTO: writing " << t.name << " failed: " << writer.error(), IOException);
        }
    }
    if (!writer.commit()) {
        IMP_THROW("PTO: writing " << path << " failed: " << writer.error(), IOException);
    }
    writer.close();
}

// ---- the converters of ChiSurf's loose .npy tables ---------------------------

// named, not anonymous: the module build is one translation unit
namespace potential_conversion {

//! Python's `repr(float)`: shortest digits that read back, positional for
//! decimal exponents -4..15, scientific otherwise.
std::string python_repr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    const std::string sign = std::signbit(v) ? "-" : "";
    const double a = std::fabs(v);
    if (a == 0.0) return sign + "0.0";
    char buf[48];
    for (int p = 1; p <= 17; ++p) {
        std::snprintf(buf, sizeof(buf), "%.*e", p - 1, a);
        if (std::strtod(buf, nullptr) == a) break;
    }
    const std::string sci = buf;
    const std::size_t e = sci.find('e');
    std::string digits = sci.substr(0, e);
    digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
    while (digits.size() > 1 && digits[digits.size() - 1] == '0') {
        digits.erase(digits.size() - 1);
    }
    const int exponent = std::atoi(sci.c_str() + e + 1);
    const int n = static_cast<int>(digits.size());
    if (exponent >= -4 && exponent < 16) {
        const int point = exponent + 1;
        if (point <= 0) {
            return sign + "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
        }
        if (point >= n) {
            return sign + digits + std::string(static_cast<std::size_t>(point - n), '0') + ".0";
        }
        return sign + digits.substr(0, point) + "." + digits.substr(point);
    }
    const std::string mantissa =
            n == 1 ? digits : digits.substr(0, 1) + "." + digits.substr(1);
    char exp[16];
    std::snprintf(exp, sizeof(exp), "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
    return sign + mantissa + exp;
}

//! numpy's round-half-to-even `int(round(x))`.
int round_half_even(double x) { return static_cast<int>(std::nearbyint(x)); }

const std::size_t kResidueTypes = 20;

}  // namespace potential_conversion

using potential_conversion::python_repr;
using potential_conversion::round_half_even;
using potential_conversion::kResidueTypes;

std::vector<std::string> potential_residue_order() {
    static const char* const names[] = {
        "CYS", "MET", "PHE", "ILE", "LEU", "VAL", "TRP", "TYR", "ALA", "GLY",
        "THR", "SER", "GLN", "ASN", "GLU", "ASP", "HIS", "ARG", "LYS", "PRO"};
    return std::vector<std::string>(names, names + kResidueTypes);
}

std::string potential_pmf_text(double bin_width, const std::vector<double>& values,
                               int n_bins) {
    const std::vector<std::string> names = potential_residue_order();
    const std::size_t n = names.size();
    const std::size_t bins = static_cast<std::size_t>(n_bins);
    if (n_bins <= 0 || values.size() != n * n * bins) {
        IMP_THROW("potential_pmf_text: " << values.size() << " values are not "
                  << n << " x " << n << " x " << n_bins, ValueException);
    }
    std::string out = python_repr(bin_width) + " " + std::to_string(n) + "\n";
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            out += names[i] + " " + names[j];
            const std::size_t base = (i * n + j) * bins;
            for (std::size_t k = 0; k < bins; ++k) {
                out += " " + python_repr(values[base + k]);
            }
            out += "\n";
        }
    }
    return out;
}

PotentialConversion convert_mj_potential(const std::vector<double>& matrix,
                                         double cutoff) {
    const std::size_t n = kResidueTypes;
    if (matrix.size() != n * n) {
        IMP_THROW("convert_mj_potential: " << matrix.size()
                  << " values are not a 20 x 20 matrix", ValueException);
    }
    std::vector<double> values(n * n * 2);
    for (std::size_t k = 0; k < n * n; ++k) {
        values[2 * k] = matrix[k];
        values[2 * k + 1] = matrix[k];
    }
    PotentialConversion out;
    out.table = PotentialTable("mj", "pot.pmf");
    out.table.text = potential_pmf_text(cutoff / 2.0, values, 2);
    return out;
}

PotentialConversion convert_unres_potential(const std::vector<double>& grid,
                                            int n_bins, double bin_width,
                                            double min_dist, double repulsion) {
    const std::size_t n = kResidueTypes;
    const std::size_t bins = static_cast<std::size_t>(n_bins);
    if (n_bins <= 0 || grid.size() != n * n * bins) {
        IMP_THROW("convert_unres_potential: " << grid.size() << " values are not 20 x 20 x "
                  << n_bins, ValueException);
    }
    PotentialConversion out;
    std::vector<double> clean(grid);
    for (std::size_t k = 0; k < clean.size(); ++k) {
        if (std::isnan(clean[k])) {
            clean[k] = repulsion;
            ++out.n_changed;
        } else if (std::isinf(clean[k])) {
            clean[k] = clean[k] > 0 ? std::numeric_limits<double>::max()
                                    : -std::numeric_limits<double>::max();
        }
    }
    const std::size_t below =
            std::min<std::size_t>(bins, static_cast<std::size_t>(
                    std::max(0, round_half_even(min_dist / bin_width))));
    for (std::size_t p = 0; p < n * n; ++p) {
        for (std::size_t k = 0; k < below; ++k) clean[p * bins + k] = repulsion;
    }
    // the upper triangle carries the potential: (a, b) reads [min][max]
    std::vector<double> values(n * n * bins);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t src = (std::min(i, j) * n + std::max(i, j)) * bins;
            std::copy(clean.begin() + src, clean.begin() + src + bins,
                      values.begin() + (i * n + j) * bins);
        }
    }
    out.table = PotentialTable("unres", "pot.pmf");
    out.table.text = potential_pmf_text(bin_width, values, n_bins);
    return out;
}

PotentialConversion convert_hbond_potential(const std::vector<double>& table,
                                            int n_channels, int n_bins,
                                            double bin_width, double physical_A) {
    const std::size_t channels = static_cast<std::size_t>(n_channels);
    const std::size_t bins = static_cast<std::size_t>(n_bins);
    if (n_channels <= 0 || n_bins <= 0 || table.size() != channels * bins) {
        IMP_THROW("convert_hbond_potential: " << table.size() << " values are not "
                  << n_channels << " x " << n_bins, ValueException);
    }
    const int first = round_half_even(physical_A / bin_width);
    if (first < 0 || static_cast<std::size_t>(first) >= bins) {
        IMP_THROW("convert_hbond_potential: " << physical_A << " A is outside the table",
                  ValueException);
    }
    PotentialConversion out;
    std::vector<double> clean(table);
    for (std::size_t c = 0; c < channels; ++c) {
        const double held = clean[c * bins + static_cast<std::size_t>(first)];
        for (std::size_t k = 0; k < static_cast<std::size_t>(first); ++k) {
            if (std::fabs(clean[c * bins + k]) > 1e5) ++out.n_changed;
            clean[c * bins + k] = held;
        }
    }
    out.table = PotentialTable("hbond", "pot.grid");
    out.table.shape.push_back(n_channels);
    out.table.shape.push_back(n_bins);
    out.table.values.swap(clean);
    return out;
}

PotentialConversion convert_ramachandran_potential(const std::vector<double>& stack,
                                                   int n_channels, int n_bins) {
    const std::size_t cells = static_cast<std::size_t>(n_bins) * static_cast<std::size_t>(n_bins);
    if (n_channels < 2 || n_bins <= 0 || stack.size() % static_cast<std::size_t>(n_channels) ||
        (stack.size() / static_cast<std::size_t>(n_channels)) *
                        (static_cast<std::size_t>(n_channels) - 2) != 3 * cells) {
        IMP_THROW("convert_ramachandran_potential: " << stack.size() << " values in "
                  << n_channels << " channels do not leave 3 x " << n_bins << " x "
                  << n_bins << " maps after the two coordinate grids", ValueException);
    }
    const std::size_t offset = 2 * (stack.size() / static_cast<std::size_t>(n_channels));
    PotentialConversion out;
    std::vector<double> maps(stack.begin() + offset, stack.end());
    for (std::size_t c = 0; c < 3; ++c) {
        const std::vector<double>::iterator begin = maps.begin() + c * cells;
        bool any = false, have_least = false, nan_seen = false;
        double least = 0.0;
        for (std::size_t k = 0; k < cells; ++k) {
            const double v = *(begin + k);
            if (v > 0.0) {
                any = true;
            } else if (std::isnan(v)) {
                // numpy's min propagates a NaN
                least = v;
                have_least = nan_seen = true;
            } else if (!nan_seen && (!have_least || v < least)) {
                least = v;
                have_least = true;
            }
        }
        if (!any) continue;
        if (!have_least) {
            IMP_THROW("convert_ramachandran_potential: channel " << c
                      << " holds only sentinels", ValueException);
        }
        for (std::size_t k = 0; k < cells; ++k) {
            if (*(begin + k) > 0.0) {
                *(begin + k) = least;
                ++out.n_changed;
            }
        }
    }
    out.table = PotentialTable("ramachandran", "pot.grid");
    out.table.shape.push_back(3);
    out.table.shape.push_back(n_bins);
    out.table.shape.push_back(n_bins);
    out.table.values.swap(maps);
    return out;
}

IMPBFF_END_NAMESPACE
