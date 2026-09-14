/** \file CommandLine.cpp
 *  \brief The command line, compiled: the dispatcher over the groups.
 *
 *  Every program that was a Python script in `bin/` is a group of
 *  subcommands of the one executable `bin/imp_bff.cpp` (owner rulings
 *  2026-09-09 and 2026-09-14). The groups live one file each,
 *  `src/CommandLine<Group>.cpp` and `src/imp/CommandLine<Group>.cpp`; this
 *  file builds the root app from them, owns the exit codes, and forwards the
 *  commands not compiled yet to the Python program `imp_bff_py`.
 *
 *  The grammar is CLI11 (include/internal/CLI11.h, vendored verbatim from
 *  github.com/CLIUtils/CLI11 v2.7.2, BSD-3): subcommands first-class, the
 *  help text generated, usage errors answered with exit code 2.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/CommandLine.h>
#include <IMP/bff/internal/CommandLineSubs.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <process.h>
#else
#  include <spawn.h>
#  include <sys/wait.h>
extern char** environ;
#endif

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {

namespace dispatch {
std::string g_sub = "imp_bff";
}

void set_current_sub(const std::string& name) { dispatch::g_sub = name; }

std::string sub_prefix() { return "imp_bff " + dispatch::g_sub + ": "; }

std::string format(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list copy;
  va_copy(copy, args);
  const int n = std::vsnprintf(nullptr, 0, fmt, copy);
  va_end(copy);
  std::string out;
  if (n > 0) {
    std::vector<char> buf(static_cast<std::size_t>(n) + 1);
    std::vsnprintf(&buf[0], buf.size(), fmt, args);
    out.assign(&buf[0], static_cast<std::size_t>(n));
  }
  va_end(args);
  return out;
}

std::string json_float(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  // Python's repr: the shortest digit string that reads back to the same
  // double, positional for decimal exponents -4..15, scientific otherwise.
  std::string sign = std::signbit(v) ? "-" : "";
  const double a = std::fabs(v);
  char buf[48];
  for (int p = 1; p <= 17; ++p) {
    std::snprintf(buf, sizeof(buf), "%.*e", p - 1, a);
    if (std::strtod(buf, nullptr) == a) break;
  }
  const std::string sci = buf;                       // d.ddde+XX
  const std::size_t e = sci.find('e');
  std::string digits = sci.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  while (digits.size() > 1 && digits[digits.size() - 1] == '0') digits.erase(digits.size() - 1);
  const int exponent = std::atoi(sci.c_str() + e + 1);
  const int n = static_cast<int>(digits.size());
  if (a == 0.0) return sign + "0.0";
  if (exponent >= -4 && exponent < 16) {
    const int point = exponent + 1;
    if (point <= 0) return sign + "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
    if (point >= n) return sign + digits + std::string(static_cast<std::size_t>(point - n), '0') + ".0";
    return sign + digits.substr(0, point) + "." + digits.substr(point);
  }
  const std::string mantissa = n == 1 ? digits : digits.substr(0, 1) + "." + digits.substr(1);
  return sign + mantissa + format("e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
}

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (std::size_t i = 0; i < s.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) out += format("\\u%04x", c);
        else out += static_cast<char>(c);
    }
  }
  return out + "\"";
}

OrderedJson& OrderedJson::raw(const std::string& key, const std::string& json_text) {
  items_.push_back(std::make_pair(key, json_text));
  return *this;
}

OrderedJson& OrderedJson::list(const std::string& key, const std::vector<std::string>& items) {
  // indent=2 puts every element on its own line; the depth is fixed at dump
  std::string text = "\x01[";
  for (std::size_t i = 0; i < items.size(); ++i) {
    text += (i ? "\x02" : "") + items[i];
  }
  text += "]";
  if (items.empty()) text = "[]";
  return raw(key, text);
}

std::string OrderedJson::dump(int depth) const {
  if (items_.empty()) return "{}";
  const std::string pad(static_cast<std::size_t>(2 * (depth + 1)), ' ');
  const std::string close_pad(static_cast<std::size_t>(2 * depth), ' ');
  std::string out = "{\n";
  for (std::size_t i = 0; i < items_.size(); ++i) {
    std::string value = items_[i].second;
    if (!value.empty() && value[0] == '\x01') {
      // a list: one element per line, one level deeper
      std::string body = value.substr(2, value.size() - 3);
      std::string lines = "[\n" + pad + "  ";
      for (std::size_t k = 0; k < body.size(); ++k) {
        if (body[k] == '\x02') lines += ",\n" + pad + "  ";
        else lines += body[k];
      }
      value = lines + "\n" + pad + "]";
    }
    out += pad + json_string(items_[i].first) + ": " + value;
    out += i + 1 < items_.size() ? ",\n" : "\n";
  }
  return out + close_pad + "}";
}

namespace dispatch {

//! The Python program the not-yet-compiled commands still run in.
const char* PY_PROGRAM = "imp_bff_py";

//! The commands of the former `bin/imp_bff` that are not compiled yet.
/*! Each is forwarded, words untouched, to `imp_bff_py`. An entry leaves this
    list in the commit that compiles it. */
const char* const FORWARDED[][2] = {
    {"flexfit", "Flexible fitting against FRET distances (runs imp_bff_py)."},
    {"rmsd", "RMSDs of an OLGA ol4 table to reference structures (runs imp_bff_py)."},
    {"select-pairs", "Rank labelling pairs by information gain (runs imp_bff_py)."},
    {"openmm", "Write OpenMM flat-bottom FRET restraints (runs imp_bff_py)."},
    {"av-export", "Compute and write one accessible volume (runs imp_bff_py)."},
    {"dye", "Dye labelling, rotamer libraries and sampling (runs imp_bff_py)."},
    {"rotamer", "Rotamer FRET prediction and Forster radii (runs imp_bff_py)."},
    {"av-vs-rotamer", "Compare AV and rotamer label models (runs imp_bff_py)."},
    {"analyze-trajectories", "Dye density analysis of trajectories (runs imp_bff_py)."},
    {"simulate", "Molecular probe simulation (runs imp_bff_py)."},
    {"build-system", "Build a probe force-field system (runs imp_bff_py)."},
    {"dock", "Replica-exchange rigid-body docking (runs imp_bff_py)."},
    {"dock-errors", "Docking spread over independent starts (runs imp_bff_py)."},
};

//! Run `imp_bff_py <words...>` and hand back its exit code.
int run_python_program(const std::vector<std::string>& words) {
  std::vector<char*> argv;
  argv.push_back(const_cast<char*>(PY_PROGRAM));
  for (std::size_t i = 0; i < words.size(); ++i) {
    argv.push_back(const_cast<char*>(words[i].c_str()));
  }
  argv.push_back(nullptr);
  std::cout << std::flush;
  std::cerr << std::flush;
#ifdef _WIN32
  const intptr_t rc = _spawnvp(_P_WAIT, PY_PROGRAM,
                               const_cast<const char* const*>(&argv[0]));
  if (rc == -1) {
    throw SubError(std::string("cannot run ") + PY_PROGRAM + ": " + std::strerror(errno));
  }
  return static_cast<int>(rc);
#else
  pid_t pid = 0;
  const int err = posix_spawnp(&pid, PY_PROGRAM, nullptr, nullptr, &argv[0], environ);
  if (err != 0) {
    throw SubError(std::string("cannot run ") + PY_PROGRAM + " (the Python half of the "
                   "command line, not compiled yet): " + std::strerror(err));
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) throw SubError("waitpid failed");
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return 1;
#endif
}

//! The root app, every group attached. \p rc receives a forwarded exit code.
std::unique_ptr<CLI::App> build_app(std::shared_ptr<int> rc) {
  std::unique_ptr<CLI::App> app(new CLI::App(
      "imp_bff -- fluorescence forward models on structures. One program; "
      "each group below was a bin/ script.",
      "imp_bff"));
  app->require_subcommand(1);
  // a word that is not a subcommand is a mistyped one, not a positional
  app->allow_extras(false);
  app->set_help_all_flag("--help-all", "print help for every subcommand");

  // `imp_bff help`: the list, for callers that expect a word rather than a flag
  CLI::App* help = app->add_subcommand("help", "Print the subcommands and exit.");
  CLI::App* root = app.get();
  help->callback([root] {
    // App::help() follows the selected subcommand, which is this one; list
    // the root's own instead
    std::cout << root->get_description() << "\n\nusage: imp_bff <subcommand> [options...]\n\n";
    const std::vector<CLI::App*> subs = root->get_subcommands([](CLI::App*) { return true; });
    for (std::size_t i = 0; i < subs.size(); ++i) {
      std::string name = subs[i]->get_name();
      const std::vector<std::string> aliases = subs[i]->get_aliases();
      for (std::size_t k = 0; k < aliases.size(); ++k) name += ", " + aliases[k];
      std::cout << format("  %-24s %s", name.c_str(), subs[i]->get_description().c_str()) << "\n";
    }
    std::cout << "\n`imp_bff <subcommand> --help` explains one.\n" << std::flush;
  });

  add_trajectory_subs(*app);
  add_labelizer_subs(*app);
  add_fps_distance_subs(*app);

  for (std::size_t i = 0; i < sizeof(FORWARDED) / sizeof(FORWARDED[0]); ++i) {
    const std::string name = FORWARDED[i][0];
    CLI::App* sub = app->add_subcommand(name, FORWARDED[i][1]);
    // every word after the name, --help included, belongs to the Python program
    sub->prefix_command();
    sub->allow_extras();
    sub->set_help_flag();
    sub->callback([sub, name, rc] {
      set_current_sub(name);
      std::vector<std::string> words(1, name);
      const std::vector<std::string> rest = sub->remaining();
      words.insert(words.end(), rest.begin(), rest.end());
      *rc = run_python_program(words);
    });
  }
  return app;
}

}  // namespace dispatch
}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE

IMPBFF_BEGIN_NAMESPACE

int command_line_main(int argc, char** argv) {
  namespace cli = internal::cli;
  std::shared_ptr<int> rc = std::make_shared<int>(0);
  std::unique_ptr<CLI::App> app = cli::dispatch::build_app(rc);
  cli::set_current_sub("imp_bff");
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    // CLI11 prints the message; the codes it returns are its own (106 for a
    // missing subcommand), and a caller only needs "usage error"
    const int code = app->exit(e);
    return code == 0 ? 0 : 2;
  } catch (const cli::SubExit& e) {
    std::cout << std::flush;
    if (e.what()[0] != '\0') std::cerr << cli::sub_prefix() << e.what() << "\n";
    return e.code;
  } catch (const std::exception& e) {
    std::cout << std::flush;
    std::cerr << "imp_bff " << cli::dispatch::g_sub << ": " << e.what() << "\n";
    return 1;
  }
  std::cout << std::flush;
  return *rc;
}

int command_line_main(const std::vector<std::string>& args) {
  // The (argc, argv) overload skips the program name and reads the words in
  // their natural order; CLI11's vector overload expects them reversed.
  std::vector<char*> line;
  line.push_back(const_cast<char*>("imp_bff"));
  for (std::size_t i = 0; i < args.size(); ++i) {
    line.push_back(const_cast<char*>(args[i].c_str()));
  }
  line.push_back(nullptr);
  return command_line_main(static_cast<int>(args.size() + 1), &line[0]);
}

std::vector<std::pair<std::string, std::string> > command_line_subcommands() {
  std::unique_ptr<CLI::App> app =
      internal::cli::dispatch::build_app(std::make_shared<int>(0));
  std::vector<std::pair<std::string, std::string> > out;
  const std::vector<CLI::App*> subs = app->get_subcommands([](CLI::App*) { return true; });
  for (std::size_t i = 0; i < subs.size(); ++i) {
    out.push_back(std::make_pair(subs[i]->get_name(), subs[i]->get_description()));
  }
  return out;
}

IMPBFF_END_NAMESPACE
