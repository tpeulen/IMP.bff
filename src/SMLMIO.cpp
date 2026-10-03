/** \file SMLMIO.cpp
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SMLMIO.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_set>

IMPBFF_BEGIN_NAMESPACE
namespace {

void smlm_io_require(bool condition, const std::string& message) {
  if (!condition) IMP_THROW(message, IMP::ValueException);
}

std::string smlm_io_trim(const std::string& text) {
  std::size_t begin = 0, end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])))
    ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end-1])))
    --end;
  return text.substr(begin, end-begin);
}

std::string smlm_io_canonical(const std::string& text) {
  std::string result;
  for (unsigned char c : text) {
    if (c >= 'A' && c <= 'Z') result += static_cast<char>(c-'A'+'a');
    else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))
      result += static_cast<char>(c);
  }
  return result;
}

std::string smlm_io_context(const std::string& path, std::size_t line) {
  return path + ": line " + std::to_string(line) + ": ";
}

// CSV quotes, escaped quotes, CRLF and embedded newlines are handled once at IO.
bool smlm_io_record(std::istream& stream, std::vector<std::string>& fields,
                    std::size_t& line_number, const std::string& path) {
  enum SMLMIOCSVState { smlm_io_start, smlm_io_unquoted,
                       smlm_io_quoted, smlm_io_after_quote };
  SMLMIOCSVState state = smlm_io_start;
  fields.clear();
  std::string field, line;
  bool started = false;
  while (std::getline(stream, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!started && smlm_io_trim(line).empty()) continue;
    started = true;
    for (char c : line) {
      if (state == smlm_io_quoted) {
        if (c == '"') state = smlm_io_after_quote;
        else field += c;
      } else if (state == smlm_io_after_quote) {
        if (c == '"') {
          field += c;
          state = smlm_io_quoted;
        } else if (c == ',') {
          fields.push_back(field);
          field.clear();
          state = smlm_io_start;
        } else {
          smlm_io_require(std::isspace(static_cast<unsigned char>(c)),
              smlm_io_context(path, line_number) + "unexpected text after CSV quote");
        }
      } else if (c == ',') {
        fields.push_back(field);
        field.clear();
        state = smlm_io_start;
      } else if (c == '"') {
        smlm_io_require(state == smlm_io_start || smlm_io_trim(field).empty(),
            smlm_io_context(path, line_number) + "unexpected quote in CSV field");
        field.clear();
        state = smlm_io_quoted;
      } else {
        field += c;
        if (!std::isspace(static_cast<unsigned char>(c))) state = smlm_io_unquoted;
      }
    }
    if (state != smlm_io_quoted) {
      fields.push_back(field);
      return true;
    }
    field += '\n';
  }
  smlm_io_require(!stream.bad(), path + ": failed while reading CSV");
  smlm_io_require(!started, smlm_io_context(path, line_number) + "unterminated CSV quote");
  return false;
}

int smlm_io_column(const std::map<std::string, int>& columns,
                   std::initializer_list<const char*> names,
                   const std::string& path) {
  int selected = -1;
  std::string selected_name;
  for (const char* name : names) {
    auto found = columns.find(name);
    if (found != columns.end()) {
      smlm_io_require(selected < 0, path + ": ambiguous CSV aliases: " +
                      selected_name + " and " + name);
      selected = found->second;
      selected_name = name;
    }
  }
  return selected;
}

double smlm_io_number(const std::vector<std::string>& fields, int column,
                      const std::string& name, const std::string& context) {
  const std::string text = smlm_io_trim(fields[static_cast<std::size_t>(column)]);
  double value = 0.0;
  std::size_t consumed = 0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception&) {
    IMP_THROW(context << "invalid numeric " << name << ": " << text,
              IMP::ValueException);
  }
  smlm_io_require(consumed == text.size() && std::isfinite(value),
      context + "invalid or nonfinite numeric " + name + ": " + text);
  return value;
}

void smlm_io_vector3(const std::vector<double>& values, const char* name) {
  smlm_io_require(values.size() == 3, std::string(name) + " must contain three values");
  for (double value : values)
    smlm_io_require(std::isfinite(value), std::string(name) + " must be finite");
}

}  // namespace

SMLMIndex read_smlm_csv(const std::string& path, double coordinate_scale,
                        const std::vector<double>& origin,
                        const SMLMCSVOptions& options) {
  smlm_io_require(std::isfinite(coordinate_scale) && coordinate_scale > 0,
                  "SMLM CSV coordinate_scale must be finite and positive");
  const std::vector<double> offset = origin.empty()
      ? std::vector<double>(3, 0.0) : origin;
  smlm_io_vector3(offset, "SMLM CSV origin");
  smlm_io_require(options.leaf_size > 0, "SMLM CSV leaf_size must be positive");
  if (options.allow_missing_z)
    smlm_io_require(std::isfinite(options.default_z), "SMLM CSV default_z must be finite");
  if (options.allow_missing_sigmas) {
    smlm_io_vector3(options.default_sigmas, "SMLM CSV default_sigmas");
    for (double sigma : options.default_sigmas)
      smlm_io_require(sigma > 0, "SMLM CSV default_sigmas must be positive");
  }
  std::ifstream stream(path, std::ios::binary);
  smlm_io_require(stream.is_open(), path + ": unable to open SMLM CSV");
  std::vector<std::string> headers, fields;
  std::size_t line_number = 0;
  smlm_io_require(smlm_io_record(stream, headers, line_number, path),
                  path + ": missing CSV header");
  std::map<std::string, int> columns;
  for (std::size_t i = 0; i < headers.size(); ++i) {
    const std::string name = smlm_io_canonical(headers[i]);
    smlm_io_require(!name.empty(), path + ": empty CSV column name");
    smlm_io_require(columns.emplace(name, static_cast<int>(i)).second,
                    path + ": duplicate CSV column: " + name);
  }
  const int coordinate_columns[3] = {
      smlm_io_column(columns, {"xnm", "x"}, path),
      smlm_io_column(columns, {"ynm", "y"}, path),
      smlm_io_column(columns, {"znm", "z"}, path)};
  smlm_io_require(coordinate_columns[0] >= 0 && coordinate_columns[1] >= 0,
                  path + ": missing x/y coordinate columns");
  smlm_io_require(coordinate_columns[2] >= 0 || options.allow_missing_z,
                  path + ": missing z column (measured 3D required)");
  const int xy_sigma = smlm_io_column(columns,
      {"locprecnm", "uncertaintynm", "uncertainty", "precisionxy"}, path);
  const int sigma_columns[3] = {
      smlm_io_column(columns, {"xnmerr", "uncertaintyxnm", "uncertaintyx", "lpx"}, path),
      smlm_io_column(columns, {"ynmerr", "uncertaintyynm", "uncertaintyy", "lpy"}, path),
      smlm_io_column(columns, {"locprecznm", "znmerr", "uncertaintyznm",
                               "uncertaintyz", "lpz", "zerr"}, path)};
  int resolved_sigmas[3] = {sigma_columns[0], sigma_columns[1], sigma_columns[2]};
  for (int axis = 0; axis < 3; ++axis) {
    if (axis < 2 && resolved_sigmas[axis] < 0) resolved_sigmas[axis] = xy_sigma;
    smlm_io_require(resolved_sigmas[axis] >= 0 || options.allow_missing_sigmas,
                    path + ": missing positional precision for axis " + std::to_string(axis));
  }
  const int particle_column = smlm_io_column(columns,
      {"sitenumbers", "sitenumber", "siteid", "particleids", "particleid", "groupid", "group"}, path);
  int weight_column = -1;
  if (!options.weight_column.empty()) {
    const auto found = columns.find(smlm_io_canonical(options.weight_column));
    smlm_io_require(found != columns.end(), path + ": missing requested weight column");
    weight_column = found->second;
  }
  std::vector<double> coordinates, sigmas, weights;
  std::vector<int> particle_ids;
  while (smlm_io_record(stream, fields, line_number, path)) {
    const std::string context = smlm_io_context(path, line_number);
    smlm_io_require(fields.size() == headers.size(), context + "CSV column count differs from header");
    smlm_io_require(weights.size() < static_cast<std::size_t>(std::numeric_limits<int>::max()),
                    context + "too many localizations");
    for (int axis = 0; axis < 3; ++axis) {
      const int column = coordinate_columns[axis];
      const double source = column >= 0
          ? smlm_io_number(fields, column, headers[column], context) : options.default_z;
      const double difference = source - offset[axis];
      // A large opposite-sign source/origin can overflow subtraction even
      // though a small scale makes the final coordinate representable.
      const double coordinate = std::isfinite(difference)
          ? difference * coordinate_scale
          : source * coordinate_scale - offset[axis] * coordinate_scale;
      smlm_io_require(std::isfinite(coordinate), context + "scaled coordinate is not finite");
      coordinates.push_back(coordinate);
      const int sigma_column = resolved_sigmas[axis];
      const double sigma = (sigma_column >= 0
          ? smlm_io_number(fields, sigma_column, headers[sigma_column], context)
          : options.default_sigmas[axis]) * coordinate_scale;
      smlm_io_require(std::isfinite(sigma) && sigma > 0,
                      context + "positional sigmas must be finite and positive");
      sigmas.push_back(sigma);
    }
    const double weight = weight_column < 0 ? 1.0
        : smlm_io_number(fields, weight_column, headers[weight_column], context);
    smlm_io_require(weight >= 0, context + "weights must be nonnegative");
    weights.push_back(weight);
    int particle_id = -1;
    if (particle_column >= 0) {
      const double id = smlm_io_number(fields, particle_column, headers[particle_column], context);
      smlm_io_require(id == std::floor(id) && id <= std::numeric_limits<int>::max(),
                      context + "particle IDs must be integers within int range");
      if (id > 0 || (id == 0 && !options.zero_particle_id_is_unassigned))
        particle_id = static_cast<int>(id);
    }
    particle_ids.push_back(particle_id);
  }
  return SMLMIndex(coordinates, sigmas, weights, particle_ids, options.leaf_size);
}

SMLMIndex select_smlm_region(const SMLMIndex& index,
                             const std::vector<double>& lower,
                             const std::vector<double>& upper,
                             int new_particle_id) {
  smlm_io_vector3(lower, "SMLM region lower");
  smlm_io_vector3(upper, "SMLM region upper");
  for (int axis = 0; axis < 3; ++axis)
    smlm_io_require(lower[axis] <= upper[axis], "SMLM region bounds are reversed");
  const auto& source = index.get_coordinates();
  const auto& source_sigmas = index.get_sigmas();
  const auto& source_weights = index.get_weights();
  const auto& source_ids = index.get_particle_ids();
  std::vector<double> coordinates, sigmas, weights;
  std::vector<int> particle_ids;
  for (int row = 0; row < index.get_number_of_localizations(); ++row) {
    bool inside = true;
    for (int axis = 0; axis < 3; ++axis)
      inside = inside && source[3*row+axis] >= lower[axis] && source[3*row+axis] <= upper[axis];
    if (!inside) continue;
    coordinates.insert(coordinates.end(), source.begin()+3*row, source.begin()+3*row+3);
    sigmas.insert(sigmas.end(), source_sigmas.begin()+3*row, source_sigmas.begin()+3*row+3);
    weights.push_back(source_weights[row]);
    particle_ids.push_back(new_particle_id >= 0 ? new_particle_id : source_ids[row]);
  }
  return SMLMIndex(coordinates, sigmas, weights, particle_ids);
}

SMLMIndex select_smlm_particles(const SMLMIndex& index,
                                const std::vector<int>& particle_ids) {
  smlm_io_require(!particle_ids.empty(), "SMLM particle selection must be nonempty");
  std::unordered_set<int> selected;
  for (int id : particle_ids) {
    smlm_io_require(id >= 0, "SMLM particle selection IDs must be nonnegative");
    smlm_io_require(selected.insert(id).second, "SMLM particle selection IDs must be unique");
    smlm_io_require(!index.get_particle_localizations(id).empty(),
                    "SMLM particle selection ID is absent: " + std::to_string(id));
  }
  const auto& source = index.get_coordinates();
  const auto& source_sigmas = index.get_sigmas();
  const auto& source_weights = index.get_weights();
  const auto& source_ids = index.get_particle_ids();
  std::vector<double> coordinates, sigmas, weights;
  std::vector<int> ids;
  for (std::size_t row = 0; row < source_ids.size(); ++row) {
    if (selected.count(source_ids[row]) == 0) continue;
    coordinates.insert(coordinates.end(), source.begin()+3*row, source.begin()+3*row+3);
    sigmas.insert(sigmas.end(), source_sigmas.begin()+3*row, source_sigmas.begin()+3*row+3);
    weights.push_back(source_weights[row]);
    ids.push_back(source_ids[row]);
  }
  return SMLMIndex(coordinates, sigmas, weights, ids);
}

IMPBFF_END_NAMESPACE
