/**
 *  \file IMP/bff/internal/CommandLineSubs.h
 *  \brief What the compiled command line's groups share.
 *
 *  One group per former `bin/` program, one source file per group:
 *  `src/CommandLine<Group>.cpp` for the core, `src/imp/CommandLine<Group>.cpp`
 *  for the groups that need IMP. Each file exposes one `add_*` function that
 *  hangs its subcommands on the root app; `src/CommandLine.cpp` calls them and
 *  owns the exit codes.
 *
 *  The module build is a unity build (every `src/*.cpp` is one translation
 *  unit), so a group's private helpers live in a *named* namespace of its
 *  own, never in an anonymous one -- two anonymous `read_weights` in two
 *  files are one redefinition there.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_COMMANDLINESUBS_H
#define IMPBFF_INTERNAL_COMMANDLINESUBS_H

#include <IMP/bff/bff_config.h>

#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <IMP/bff/internal/CLI11.h>

//! True when the IMP connection layer (src/imp/) is compiled in.
/*! The module build always has it; the standalone build only with
    IMPBFF_WITH_IMP. */
#if defined(IMPBFF_WITH_IMP) || !defined(IMPBFF_STANDALONE)
#  define IMPBFF_CLI_HAS_IMP_LAYER 1
#else
#  define IMPBFF_CLI_HAS_IMP_LAYER 0
#endif

IMPBFF_BEGIN_INTERNAL_NAMESPACE

namespace cli {

//! A sub's own failure: printed as `imp_bff <sub>: <message>`, exit 1.
struct SubError : std::runtime_error {
  explicit SubError(const std::string& what) : std::runtime_error(what) {}
};

//! End the sub with a given exit code; a non-empty message goes to stderr
//! with the usual `imp_bff <sub>: ` prefix.
struct SubExit : SubError {
  int code;
  SubExit(int c, const std::string& what = std::string()) : SubError(what), code(c) {}
};

//! The `imp_bff <sub>: ` prefix of the running sub, for notes on stderr.
IMPBFFEXPORT std::string sub_prefix();

//! pathlib's `suffix`: the last `.xxx` of the file name, or empty.
inline std::string path_suffix(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  const std::size_t start = slash == std::string::npos ? 0 : slash + 1;
  const std::size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || dot <= start || dot + 1 == path.size()) return std::string();
  return path.substr(dot);
}

//! pathlib's `with_suffix(s)`: replace the last suffix (or append).
inline std::string path_with_suffix(const std::string& path, const std::string& s) {
  const std::string old = path_suffix(path);
  return path.substr(0, path.size() - old.size()) + s;
}

//! Name the running sub, for the error prefix. Groups call it first thing.
IMPBFFEXPORT void set_current_sub(const std::string& name);

//! The last path component.
inline std::string basename_of(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

//! A whole file as a string; SubError when it cannot be read.
inline std::string read_text_file(const std::string& path) {
  std::ifstream in(path.c_str(), std::ios::binary);
  if (!in) throw SubError(path + ": cannot read");
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

//! printf into a std::string.
IMPBFFEXPORT std::string format(const char* fmt, ...);

//! One JSON value, as Python's `json.dumps` spells it: shortest round-trip
//! floats with a `.0` on integral values, `NaN`/`Infinity` for the
//! non-finite, strings escaped.
IMPBFFEXPORT std::string json_float(double v);
IMPBFFEXPORT std::string json_string(const std::string& s);

//! A JSON object printed as `json.dumps(obj, indent=2)` prints a dict built in
//! insertion order -- the programs' `--json` output, key for key.
class IMPBFFEXPORT OrderedJson {
 public:
  OrderedJson& raw(const std::string& key, const std::string& json_text);
  OrderedJson& str(const std::string& key, const std::string& v) {
    return raw(key, json_string(v));
  }
  OrderedJson& num(const std::string& key, double v) { return raw(key, json_float(v)); }
  OrderedJson& integer(const std::string& key, long v) { return raw(key, format("%ld", v)); }
  OrderedJson& boolean(const std::string& key, bool v) { return raw(key, v ? "true" : "false"); }
  OrderedJson& null(const std::string& key) { return raw(key, "null"); }
  //! A list of already-spelled JSON values.
  OrderedJson& list(const std::string& key, const std::vector<std::string>& items);
  //! The text, at nesting depth \p depth (for an object inside an object).
  std::string dump(int depth = 0) const;

 private:
  std::vector<std::pair<std::string, std::string> > items_;
};

// ---- the groups -----------------------------------------------------------
// Each adds its subcommand(s) to `app`. The values a parse fills live in a
// std::shared_ptr the group's callback captures, so they live exactly as
// long as the app that holds the callback.

//! probe-pdb2cif, traj2bcif, traj2drot
IMPBFFEXPORT void add_trajectory_subs(CLI::App& app);
//! labelizer
IMPBFFEXPORT void add_labelizer_subs(CLI::App& app);
//! fps-distance
IMPBFFEXPORT void add_fps_distance_subs(CLI::App& app);

#if IMPBFF_CLI_HAS_IMP_LAYER
#endif

}  // namespace cli

IMPBFF_END_INTERNAL_NAMESPACE

#endif  // IMPBFF_INTERNAL_COMMANDLINESUBS_H
