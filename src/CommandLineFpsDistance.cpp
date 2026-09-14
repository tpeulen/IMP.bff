/** \file CommandLineFpsDistance.cpp
 *  \brief `imp_bff fps-distance`: the distances between two dye clouds --
 *         FPS's distance calculator.
 *
 *  Port of `bin/imp_bff_fps_distance`, itself a port of the
 *  `DistanceCalculator` dialog of FPS (Kalinin *et al.*, *Nat. Methods* **9**,
 *  1218, 2012). Given two accessible volumes it reports the numbers a FRET
 *  measurement is compared against:
 *
 *      Rmp        the distance between the clouds' mean positions
 *      <RDA>      the mean donor-acceptor distance
 *      sigma_DA   the width of that distribution
 *      <E>        the mean transfer efficiency
 *      <RDA>_E    the distance a measured <E> would be read as
 *
 *  The three distances are **not** interchangeable and an experiment measures
 *  one of them. `<RDA>_E` is always the shortest: efficiency weights close
 *  pairs as `1/r^6`, so averaging E and converting back is not averaging r.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <IMP/bff/FPS.h>
#include <IMP/bff/States.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace fps_distance {

struct Args {
  std::string av1, av2;
  double forster_radius = 52.0;
  int n_samples = 200000;
  bool as_json = false;
};

void run(const Args& a) {
  const XyzPointCloud c1 = read_points_xyz(a.av1);
  const XyzPointCloud c2 = read_points_xyz(a.av2);
  const States s1(c1.points), s2(c2.points);
  const std::vector<double> stats = av_pair_statistics(s1, s2, a.forster_radius, a.n_samples);
  const double rmp = stats[0], rda = stats[1], rda_e = stats[2], sigma = stats[3];
  const double efficiency = (rda_e != 0.0 && std::isfinite(rda_e))
                                ? 1.0 / (1.0 + std::pow(rda_e / a.forster_radius, 6))
                                : std::nan("");
  const long n1 = static_cast<long>(c1.points.size() / 4);
  const long n2 = static_cast<long>(c2.points.size() / 4);

  if (a.as_json) {
    OrderedJson out;
    out.str("av1", a.av1).str("av2", a.av2);
    out.list("points", {format("%ld", n1), format("%ld", n2)});
    out.list("unique_voxels", {format("%d", c1.n_unique_voxels), format("%d", c2.n_unique_voxels)});
    out.num("forster_radius", a.forster_radius).integer("n_samples", a.n_samples);
    out.num("Rmp", rmp).num("RDA_mean", rda).num("sigma_DA", sigma);
    out.num("E_mean", efficiency).num("RDA_E", rda_e);
    std::cout << out.dump() << "\n";
    return;
  }

  const std::string paths[2] = {a.av1, a.av2};
  const XyzPointCloud* clouds[2] = {&c1, &c2};
  for (int i = 0; i < 2; ++i) {
    const long n = static_cast<long>(clouds[i]->points.size() / 4);
    std::string note;
    if (clouds[i]->n_unique_voxels < n) {
      note = format("   (%d unique voxels -- the file is duplicate-expanded, which is a weighting)",
                    clouds[i]->n_unique_voxels);
    }
    std::cout << format("%-28s %7ld points%s", basename_of(paths[i]).c_str(), n, note.c_str())
              << "\n";
    if (clouds[i]->declared_mean.size() == 3) {
      const std::vector<double>& m = clouds[i]->declared_mean;
      std::cout << format("%-28s declared mean (%.3f, %.3f, %.3f)", "", m[0], m[1], m[2]) << "\n";
    }
  }
  std::cout << "\n";
  std::cout << format("Rmp       = %8.2f A     distance between the mean positions", rmp) << "\n";
  std::cout << format("<RDA>     = %8.2f A     mean donor-acceptor distance", rda) << "\n";
  std::cout << format("sigma_DA  = %8.2f A     width of that distribution", sigma) << "\n";
  std::cout << format("<E>       = %8.3f       mean transfer efficiency (R0 = %g A)", efficiency,
                      a.forster_radius)
            << "\n";
  std::cout << format("<RDA>_E   = %8.2f A     what a measured <E> reads as", rda_e) << "\n";
}

}  // namespace fps_distance

void add_fps_distance_subs(CLI::App& app) {
  std::shared_ptr<fps_distance::Args> a = std::make_shared<fps_distance::Args>();
  CLI::App* sub = app.add_subcommand(
      "fps-distance", "Report Rmp, <RDA>, sigma_DA, <E> and <RDA>_E for two dye clouds.");
  sub->footer(
      "Reads the .xyz clouds FPS's AV interface writes, and the .xyz this module writes\n"
      "(imp_bff fps-av, imp_bff av-export). An FPS .xyz is duplicate-expanded: its AV3\n"
      "export writes a voxel once per dye radius that fits. That is a weighting, so the\n"
      "duplicates are kept and the unique voxel count is reported separately.\n\n"
      "Examples:\n"
      "  imp_bff fps-av -p 3GUN.pdb -r 132 -o d.xyz\n"
      "  imp_bff fps-av -p 3GUN.pdb -r 55  -o a.xyz\n"
      "  imp_bff fps-distance d.xyz a.xyz\n"
      "  imp_bff fps-distance \"p66(D).xyz\" \"p_1bp(D).xyz\" -r 60\n\n"
      "Compare a TCSPC species fraction against <RDA>, a measured efficiency against\n"
      "<RDA>_E, and only a model's own geometry against Rmp.");
  sub->add_option("av1", a->av1, "first dye cloud (.xyz)")->required()->check(CLI::ExistingFile);
  sub->add_option("av2", a->av2, "second dye cloud (.xyz)")->required()->check(CLI::ExistingFile);
  sub->add_option("-r,--forster-radius", a->forster_radius,
                  "R0 in Angstrom; only <E> and <RDA>_E depend on it.")
      ->capture_default_str();
  sub->add_option("-n,--n-samples", a->n_samples, "Monte-Carlo pairs drawn from the two clouds.")
      ->capture_default_str();
  sub->add_flag("--json", a->as_json, "Emit one JSON object instead of a report.");
  sub->callback([a] {
    set_current_sub("fps-distance");
    fps_distance::run(*a);
  });
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE
