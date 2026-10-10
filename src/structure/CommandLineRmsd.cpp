/** \file CommandLineRmsd.cpp
 *  \brief `imp_bff rmsd`: RMSDs of the structures an OLGA `.ol4` table names,
 *         against reference structures and against the best-scoring rows.
 *
 *  Port of the `rmsd` command of the Python program `bin/imp_bff`, which read
 *  the table with pandas and the structures with mdtraj. The table round trip
 *  follows pandas (`read_csv` type inference, `to_csv` with the row index),
 *  and the RMSD is mdtraj's -- every atom, after optimal superposition -- in
 *  Angstrom. Two differences, both deliberate:
 *
 *  - the Python split the structure column with `str.split(' ', 1,
 *    expand=True)`, which pandas 2 rejects (`n` is keyword-only), so the
 *    command had not run since; this splits on the first space, which is what
 *    it meant;
 *  - mdtraj computes in float32, so its RMSDs are float32 values; this
 *    computes in double and rounds to float32 the same way, so the printed and
 *    written numbers agree with mdtraj's to about 1e-6 A, not bit for bit.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <IMP/bff/HierarchyFrame.h>
#include <IMP/bff/StructureIO.h>
#include <IMP/bff/TrajectoryIO.h>
#include <IMP/bff/internal/NumpyCompat.h>
#include <IMP/bff/internal/PdbFrames.h>
#include <IMP/bff/internal/Text.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace rmsd {

namespace nc = numpy_compat;

struct Args {
  std::string ol4, rel_path = "..", key = "structure";
  std::vector<std::string> references;
};

//! A float32 as numpy's `str(np.float32)` spells it: shortest round trip.
std::string float32_str(float v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  const std::string sign = std::signbit(v) ? "-" : "";
  const float a = std::fabs(v);
  if (a == 0.0f) return sign + "0.0";
  char buf[48];
  for (int p = 1; p <= 9; ++p) {
    std::snprintf(buf, sizeof(buf), "%.*e", p - 1, static_cast<double>(a));
    if (static_cast<float>(std::strtod(buf, nullptr)) == a) break;
  }
  const std::string sci = buf;
  const std::size_t e = sci.find('e');
  std::string digits = sci.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  while (digits.size() > 1 && digits[digits.size() - 1] == '0') digits.erase(digits.size() - 1);
  const int exponent = std::atoi(sci.c_str() + e + 1);
  const int n = static_cast<int>(digits.size());
  if (exponent >= -4 && exponent < 16) {
    const int point = exponent + 1;
    if (point <= 0) return sign + "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
    if (point >= n) return sign + digits + std::string(static_cast<std::size_t>(point - n), '0') + ".0";
    return sign + digits.substr(0, point) + "." + digits.substr(point);
  }
  const std::string mantissa = n == 1 ? digits : digits.substr(0, 1) + "." + digits.substr(1);
  return sign + mantissa + format("e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
}

//! Python's `repr` of a string, as a tuple of paths prints it.
std::string py_str_repr(const std::string& s) {
  const bool has_single = s.find('\'') != std::string::npos;
  const bool has_double = s.find('"') != std::string::npos;
  const char q = (has_single && !has_double) ? '"' : '\'';
  std::string out(1, q);
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '\\') out += "\\\\";
    else if (c == q) out += std::string("\\") + q;
    else if (c == '\n') out += "\\n";
    else if (c == '\t') out += "\\t";
    else if (c == '\r') out += "\\r";
    else out += c;
  }
  return out + q;
}

// ---- the table, as pandas reads and writes it -----------------------------

//! FLOAT32: a column pandas created from a float32 scalar (`df.at` on a new
//! name), which it writes as numpy spells a float32.
enum Kind { INT, FLOAT, FLOAT32, BOOL, OBJECT };

struct Column {
  std::string name;
  Kind kind;
  std::vector<std::string> text;   // OBJECT; empty string with `missing`
  std::vector<double> number;      // INT, FLOAT, BOOL (0/1)
  std::vector<bool> missing;
};

std::vector<std::string> split_tsv(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (quoted) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          cur += '"';
          ++i;
        } else {
          quoted = false;
        }
      } else {
        cur += c;
      }
    } else if (c == '"' && cur.empty()) {
      quoted = true;
    } else if (c == '\t') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  out.push_back(cur);
  return out;
}

bool is_na_token(const std::string& s) {
  static const char* const tokens[] = {"", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
                                       "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A",
                                       "NA", "NULL", "NaN", "None", "n/a", "nan", "null"};
  for (std::size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); ++i) {
    if (s == tokens[i]) return true;
  }
  return false;
}

bool parse_int(const std::string& s, double* out) {
  if (s.empty()) return false;
  std::size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
  if (i == s.size()) return false;
  for (std::size_t k = i; k < s.size(); ++k) {
    if (!std::isdigit(static_cast<unsigned char>(s[k]))) return false;
  }
  *out = std::strtod(s.c_str(), nullptr);
  return true;
}

bool parse_float(const std::string& s, double* out) {
  if (s.empty()) return false;
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0') return false;
  *out = v;
  return true;
}

std::vector<Column> read_table(const std::string& path) {
  std::ifstream in(path.c_str());
  if (!in) throw SubError(path + ": cannot read");
  std::string line;
  if (!std::getline(in, line)) throw SubError(path + ": empty table");
  if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
  const std::vector<std::string> header = split_tsv(line);
  std::vector<std::vector<std::string> > rows;
  while (std::getline(in, line)) {
    if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
    if (line.empty()) continue;
    std::vector<std::string> fields = split_tsv(line);
    fields.resize(header.size());
    rows.push_back(fields);
  }
  std::vector<Column> cols(header.size());
  for (std::size_t c = 0; c < header.size(); ++c) {
    Column& col = cols[c];
    col.name = header[c];
    bool all_int = true, all_float = true, all_bool = true, any_missing = false;
    for (std::size_t r = 0; r < rows.size(); ++r) {
      const std::string& s = rows[r][c];
      double v;
      if (is_na_token(s)) {
        any_missing = true;
        all_bool = false;
        continue;
      }
      if (!parse_int(s, &v)) all_int = false;
      if (!parse_float(s, &v)) all_float = false;
      if (!(s == "True" || s == "False" || s == "TRUE" || s == "FALSE" || s == "true" ||
            s == "false")) {
        all_bool = false;
      }
    }
    if (all_bool && !rows.empty()) col.kind = BOOL;
    else if (all_int && !any_missing && !rows.empty()) col.kind = INT;
    else if (all_float && !rows.empty()) col.kind = FLOAT;
    else col.kind = OBJECT;
    if (rows.empty()) col.kind = OBJECT;
    for (std::size_t r = 0; r < rows.size(); ++r) {
      const std::string& s = rows[r][c];
      const bool na = is_na_token(s);
      col.missing.push_back(na);
      double v = std::numeric_limits<double>::quiet_NaN();
      if (col.kind == BOOL) v = (s[0] == 'T' || s[0] == 't') ? 1.0 : 0.0;
      else if ((col.kind == INT || col.kind == FLOAT) && !na) parse_float(s, &v);
      col.number.push_back(v);
      col.text.push_back(na ? std::string() : s);
    }
  }
  return cols;
}

std::string csv_field(const std::string& s) {
  if (s.find_first_of("\t\"\n\r") == std::string::npos) return s;
  std::string out = "\"";
  for (std::size_t i = 0; i < s.size(); ++i) out += s[i] == '"' ? std::string("\"\"") : std::string(1, s[i]);
  return out + "\"";
}

void write_table(const std::string& path, const std::vector<Column>& cols, std::size_t n_rows) {
  std::ofstream out(path.c_str());
  if (!out) throw SubError(path + ": cannot write");
  for (std::size_t c = 0; c < cols.size(); ++c) out << "\t" << csv_field(cols[c].name);
  out << "\n";
  for (std::size_t r = 0; r < n_rows; ++r) {
    out << r;
    for (std::size_t c = 0; c < cols.size(); ++c) {
      const Column& col = cols[c];
      out << "\t";
      switch (col.kind) {
        case INT:
          out << format("%lld", static_cast<long long>(col.number[r]));
          break;
        case BOOL:
          out << (col.number[r] != 0.0 ? "True" : "False");
          break;
        case FLOAT:
          if (!std::isnan(col.number[r])) out << nc::python_repr(col.number[r]);
          break;
        case FLOAT32:
          if (!std::isnan(col.number[r])) out << float32_str(static_cast<float>(col.number[r]));
          break;
        case OBJECT:
          if (!col.missing[r]) out << csv_field(col.text[r]);
          break;
      }
    }
    out << "\n";
  }
}

// ---- structures, as mdtraj loads them ---------------------------------------

//! Every frame of a PDB or a DCD (with a topology PDB), flat per frame.
std::vector<std::vector<double> > load_frames(const std::string& path, const std::string& top) {
  std::vector<std::vector<double> > frames;
  if (ends_with(path, ".dcd")) {
    if (top.empty()) throw SubError(path + ": a DCD needs a topology");
    const std::vector<ProteinFrame> topology = read_pdb_frames(top, PDB_ALL, true, 1);
    const std::size_t n_atoms = topology[0].coords.size() / 3;
    double* view = nullptr;
    int n = 0;
    read_dcd(path, -1, &view, &n);
    std::vector<double> xyz(view, view + n);
    std::free(view);
    if (n_atoms == 0 || xyz.size() % (3 * n_atoms) != 0) {
      throw SubError(path + ": the DCD does not match the " + format("%lu", (unsigned long)n_atoms) +
                     " atoms of " + top);
    }
    for (std::size_t f = 0; f < xyz.size() / (3 * n_atoms); ++f) {
      frames.push_back(std::vector<double>(xyz.begin() + f * 3 * n_atoms,
                                           xyz.begin() + (f + 1) * 3 * n_atoms));
    }
    return frames;
  }
  const std::vector<ProteinFrame> pdb = read_pdb_frames(path, PDB_ALL, false, -1);
  for (std::size_t f = 0; f < pdb.size(); ++f) frames.push_back(pdb[f].coords);
  return frames;
}

//! mdtraj's `rmsd(frame, reference) * 10.0`: nm in float32, then scaled in float32.
float mdtraj_rmsd_angstrom(const std::vector<double>& frame, const std::vector<double>& reference) {
  if (frame.size() != reference.size()) {
    throw SubError(format("the structures have %lu and %lu atoms", (unsigned long)(frame.size() / 3),
                          (unsigned long)(reference.size() / 3)));
  }
  // identical coordinates are an exact zero in mdtraj's QCP; a superposition
  // in double leaves ~1e-15 of rotation noise instead
  if (frame == reference) return 0.0f;
  const double rmsd_a = get_rmsd(frame, reference, std::vector<int>(), true);
  const float nm = static_cast<float>(rmsd_a / 10.0);
  return nm * 10.0f;
}

void run(const Args& a) {
  std::cout << "Adding RMSDs to OL4 file\n";
  std::cout << "========================\n";
  std::cout << "OL4 File: " << a.ol4 << "\n";
  std::string refs = "(";
  for (std::size_t i = 0; i < a.references.size(); ++i) {
    refs += (i ? ", " : "") + py_str_repr(a.references[i]);
  }
  refs += a.references.size() == 1 ? ",)" : ")";
  std::cout << "Reference structure: " << refs << "\n";
  std::cout << "Structure key: " << a.key << "\n";

  std::vector<Column> cols = read_table(a.ol4);
  std::size_t n_rows = cols.empty() ? 0 : cols[0].text.size();
  const std::string path = path_join(path_dirname(a.ol4), a.rel_path);
  const std::string& top_file = a.references[0];

  int all_col = -1, key_col = -1;
  for (std::size_t c = 0; c < cols.size(); ++c) {
    if (cols[c].name == "all") all_col = static_cast<int>(c);
    if (cols[c].name == a.key) key_col = static_cast<int>(c);
  }
  if (all_col < 0) throw SubError(a.ol4 + ": no column 'all'");
  if (key_col < 0) throw SubError(a.ol4 + ": no column '" + a.key + "'");
  if (n_rows == 0) throw SubError(a.ol4 + ": no rows");
  const std::vector<double> all = cols[all_col].number;

  std::size_t best_idx = 0;
  for (std::size_t r = 1; r < n_rows; ++r) {
    if (all[r] < all[best_idx]) best_idx = r;
  }
  std::vector<std::size_t> sorted_idx(n_rows);
  for (std::size_t r = 0; r < n_rows; ++r) sorted_idx[r] = r;
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&all](std::size_t x, std::size_t y) { return all[x] < all[y]; });

  const std::vector<std::string> structure = cols[key_col].text;
  const bool use_traj = structure[0].find(".dcd") != std::string::npos;
  Column file_col, frame_col;
  file_col.name = "file";
  frame_col.name = "frame";
  file_col.kind = frame_col.kind = OBJECT;
  for (std::size_t r = 0; r < n_rows; ++r) {
    const std::string& s = structure[r];
    const std::size_t sp = s.find(' ');
    file_col.text.push_back(sp == std::string::npos ? s : s.substr(0, sp));
    frame_col.text.push_back(sp == std::string::npos ? std::string() : s.substr(sp + 1));
    file_col.missing.push_back(false);
    frame_col.missing.push_back(sp == std::string::npos);
    file_col.number.push_back(0);
    frame_col.number.push_back(0);
  }
  cols.push_back(file_col);
  cols.push_back(frame_col);
  // copies: the columns vector grows below, and references into it would dangle
  const std::vector<std::string> files = cols[cols.size() - 2].text;
  const std::vector<std::string> frame_text = cols[cols.size() - 1].text;

  const std::string best_file = files[best_idx];
  const int best_frame = std::atoi(frame_text[best_idx].c_str());
  std::cout << "best_score: " << nc::python_repr(all[best_idx]) << "\n";
  std::cout << "best_file: " << best_file << "\n";
  std::cout << "best_frame: " << best_frame << "\n";
  std::cout << "Loading trajectory..." << std::endl;
  const std::vector<std::vector<double> > traj =
          load_frames(path_join(path, best_file), top_file);

  // the references, in insertion order; a repeated name replaces its value
  std::vector<std::string> names;
  std::map<std::string, std::vector<double> > references;
  for (std::size_t i = 0; i < a.references.size(); ++i) {
    const std::string name = "RMSD_" + basename_of(a.references[i]);
    Column c;
    c.name = name;
    c.kind = FLOAT;
    c.number.assign(n_rows, std::numeric_limits<double>::quiet_NaN());
    c.missing.assign(n_rows, true);
    c.text.assign(n_rows, std::string());
    bool exists = false;
    for (std::size_t k = 0; k < cols.size(); ++k) {
      if (cols[k].name == name) {
        cols[k] = c;
        exists = true;
      }
    }
    if (!exists) cols.push_back(c);
    if (!references.count(name)) names.push_back(name);
    references[name] = load_frames(a.references[i], "")[0];
  }
  const std::size_t n_best = 10;
  for (std::size_t i = 0; i < sorted_idx.size() && i < n_best; ++i) {
    const int frame = std::atoi(frame_text[sorted_idx[i]].c_str());
    if (frame < 0 || static_cast<std::size_t>(frame) >= traj.size()) {
      throw SubError(format("index %d is out of bounds for the %lu frames of ", frame,
                            (unsigned long)traj.size()) + best_file);
    }
    const std::string name = format("best_%lu", (unsigned long)i) + format("_frame_%d", frame);
    if (!references.count(name)) names.push_back(name);
    references[name] = traj[frame];
  }

  std::cout << "Computing RMSD:\n";
  std::cout << "---------------\n";
  for (std::size_t k = 0; k < names.size(); ++k) {
    const std::string& name = names[k];
    std::cout << "Reference: " << name << "\n";
    const std::vector<double>& reference = references[name];
    int col = -1;
    for (std::size_t c = 0; c < cols.size(); ++c) {
      if (cols[c].name == name) col = static_cast<int>(c);
    }
    if (col < 0) {
      // created by the first `df.at` assignment, of a float32
      Column c;
      c.name = name;
      c.kind = FLOAT32;
      c.number.assign(n_rows, std::numeric_limits<double>::quiet_NaN());
      c.missing.assign(n_rows, true);
      c.text.assign(n_rows, std::string());
      cols.push_back(c);
      col = static_cast<int>(cols.size()) - 1;
    }
    if (use_traj) {
      std::vector<float> rmsds(traj.size());
      for (std::size_t f = 0; f < traj.size(); ++f) rmsds[f] = mdtraj_rmsd_angstrom(traj[f], reference);
      for (std::size_t r = 0; r < n_rows; ++r) {
        const int frame = std::atoi(frame_text[r].c_str());
        if (frame < 0 || static_cast<std::size_t>(frame) >= rmsds.size()) {
          throw SubError(format("frame %d is past the end of the trajectory", frame));
        }
        cols[col].number[r] = static_cast<double>(rmsds[frame]);
      }
    } else {
      for (std::size_t r = 0; r < n_rows; ++r) {
        const std::string pdb_fn = path_join(path, files[r]);
        const float value = mdtraj_rmsd_angstrom(load_frames(pdb_fn, "")[0], reference);
        std::cout << "PDB/RMSD: " << pdb_fn << " " << float32_str(value) << "\n";
        cols[col].number[r] = static_cast<double>(value);
      }
    }
  }
  const std::string suffix = path_suffix(a.ol4);
  const std::string nf = a.ol4.substr(0, a.ol4.size() - suffix.size()) + ".rmsd.ol4";
  write_table(nf, cols, n_rows);
}

}  // namespace rmsd

void add_rmsd_subs(CLI::App& app) {
  std::shared_ptr<rmsd::Args> a = std::make_shared<rmsd::Args>();
  CLI::App* sub = app.add_subcommand("rmsd", "Compute RMSDs of the structures an OLGA .ol4 table names against reference structures.");
  sub->footer(
      "Adds one column per reference structure, and one per best-scoring row, to an OLGA\n"
      ".ol4 table and writes <table>.rmsd.ol4 beside it. The structure column names a file\n"
      "and a frame, separated by a space: a PDB per row, or a DCD for every row (read with\n"
      "the first reference as its topology). RMSDs are in Angstrom, over every atom, after\n"
      "optimal superposition.\n\n"
      "Examples:\n"
      "  imp_bff rmsd -f run_5_6_7.ol4 -p . -r nodes1_001.pdb -k structure\n"
      "  imp_bff rmsd -f species_sampling.sort.ol4 -p ../rrt/long/ -r nodes1_001.pdb -k structure1");
  sub->add_option("-f,--olga-ol4-file", a->ol4, "Input OL4 file.")->required();
  sub->add_option("-p,--rel-path", a->rel_path,
                  "Relative path to PDB files / trajectory (default: ..).");
  sub->add_option("-r,--reference-structure-file", a->references,
                  "Path to reference PDB file (multiple files possible).")
      ->required();
  sub->add_option("-k,--structure-key", a->key,
                  "Column name containing the structure information (file name, frame nbr)");
  sub->callback([a] {
    set_current_sub("rmsd");
    rmsd::run(*a);
  });
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE
