"""PRD-147 step 3: the shared warm-up (SamplerWarmup.h) against Stan's own code. Written 2026-09-15.

The references below were produced by compiling CmdStan 2.39.0's headers (`stan/mcmc/windowed_adaptation.hpp`,
`stan/mcmc/stepsize_adaptation.hpp`) with the two generator programs kept at the end of this file:

- window ends for warm-ups of 150, 200, 500, 1000, 5000 and 20000 must equal Stan's exactly;
- the documented differences must hold: below 150 Stan closes no window (its reduced-stage branch does
  not restart its counters) where bff closes one; for 151-199 Stan closes an extra one-iteration window;
- dual averaging on a fixed stream of acceptance statistics (with one value above 1, which both clamp)
  must reproduce Stan's step sizes and the final averaged step to 1e-15 relative.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "decay"))
from bayesian_cxx import run_driver  # noqa: E402

STAN_WINDOWS = {"150": [100], "200": [100, 150], "500": [100, 150, 250, 450], "1000": [100, 150, 250, 450, 950], "5000": [100, 150, 250, 450, 850, 1650, 4950], "20000": [100, 150, 250, 450, 850, 1650, 3250, 6450, 19950]}
STAN_DUAL_AVERAGING = [1.7387348363544286, 1.4444010958421083, 1.9242610204337864, 2.399278707521136, 1.5226546135767833, 0.3679815870511263, 0.0412076786839739, 0.047426594477135604, 0.007349968389932014, 0.002890980045567734, 0.0028740281123325094, 0.004628555713053722, 0.006209075475378924, 0.003944273119639401, 0.0009739535372232892, 0.00012101228931439461, 1.3582406074961478e-05, 2.5912695262266383e-06, 1.231016471206754e-06, 1.39463733232955e-06, 2.4242173257417697e-06, 3.4567475907481553e-06, 2.4137437932418688e-06, 6.983601627200127e-07, 1.0786334335883252e-07, 1.538594149959229e-08, 3.6272632146839435e-09, 1.9930126761632653e-09, 2.4255290960966068e-09, 4.316164048884171e-09, 6.2862482267553485e-09, 4.673262441826019e-09, 1.5323347535354095e-09, 2.824632760193041e-10, 4.883090554782216e-11, 1.3556475457398845e-11, 8.262799869682294e-12, 1.0466988799976476e-11, 1.8634135160958413e-11, 2.7162240145794758e-11, 9.974956548343202e-09]

DRIVER = r"""
#include <IMP/bff/SamplerWarmup.h>
#include <cmath>
#include <cstdio>
using namespace IMP::bff;
int main() {
  std::printf("{\"windows\": {");
  for (int n = 0; n <= 20000; ++n) {
    const WarmupWindows w = warmup_windows(n);
    std::printf("%s\"%d\": [", n ? "," : "", n);
    for (std::size_t i = 0; i < w.ends.size(); ++i) std::printf("%s%d", i ? "," : "", w.ends[i]);
    std::printf("]");
  }
  std::printf("}, \"dual_averaging\": [");
  DualAveragingStepSize a(0.8);
  a.restart(0.3);
  double eps = 0.3;
  for (int k = 0; k < 40; ++k) { eps = a.learn(0.5 + 0.45 * std::sin(0.7 * k) + (k == 7 ? 0.8 : 0.0)); std::printf("%s%.17g", k ? "," : "", eps); }
  std::printf(",%.17g]}\n", a.final_step(eps));
  return 0;
}
"""


def test_windows_and_dual_averaging_match_stan():
    out = run_driver(DRIVER)
    ours = out["windows"]
    for n, ends in STAN_WINDOWS.items():
        assert ours[n] == ends, (n, ours[n], ends)
    # the documented differences, where Stan's own code does not adapt as its schedule describes
    for n in range(20, 150):
        assert len(ours[str(n)]) == 1, n
    assert ours["151"] == [101] and ours["199"] == [149]
    for n in range(0, 20):
        assert ours[str(n)] == [], n
    da = out["dual_averaging"]
    assert len(da) == len(STAN_DUAL_AVERAGING)
    for a, b in zip(da, STAN_DUAL_AVERAGING):
        assert abs(a - b) <= 1e-15 * abs(b), (a, b)


GENERATOR_WINDOWS = r"""#include <stan/mcmc/windowed_adaptation.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <cstdio>
#include <sstream>
// Stan's windowed_adaptation, driven as covar_adaptation drives it: per warm-up iteration, add a sample when
// adaptation_window(), and at end_adaptation_window() compute_next_window() and record the iteration count.
struct Probe : stan::mcmc::windowed_adaptation {
  Probe() : stan::mcmc::windowed_adaptation("probe") {}
  void run(unsigned n) {
    std::printf("{\"n\": %u, \"ends\": [", n);
    bool first = true; unsigned in_window = 0, first_in = 0;
    for (unsigned it = 0; it < n; ++it) {
      if (adaptation_window()) { if (!in_window) first_in = it; ++in_window; }
      if (end_adaptation_window()) { compute_next_window(); std::printf("%s%u", first ? "" : ",", it + 1); first = false; }
      ++adapt_window_counter_;
    }
    std::printf("], \"first_adapt_iteration\": %u, \"adapt_iterations\": %u}\n", first_in, in_window);
  }
};
int main(int argc, char** argv) {
  std::stringstream a, b, c, d, e;
  stan::callbacks::stream_logger logger(a, b, c, d, e);
  for (int i = 1; i < argc; ++i) {
    unsigned n = std::stoul(argv[i]);
    Probe p; p.set_window_params(n, 75, 50, 25, logger); p.run(n);
  }
}
"""

GENERATOR_DUAL_AVERAGING = r"""#include <stan/mcmc/stepsize_adaptation.hpp>
#include <cmath>
#include <cstdio>
// Stan's dual averaging on a fixed stream of acceptance statistics: adapt_stat_k = 0.5 + 0.45 sin(0.7 k), step 0.3, delta 0.8.
int main() {
  stan::mcmc::stepsize_adaptation a;
  a.set_mu(std::log(10 * 0.3)); a.set_delta(0.8); a.set_gamma(0.05); a.set_kappa(0.75); a.set_t0(10); a.restart();
  double eps = 0.3;
  std::printf("[");
  for (int k = 0; k < 40; ++k) { a.learn_stepsize(eps, 0.5 + 0.45 * std::sin(0.7 * k) + (k == 7 ? 0.8 : 0.0)); std::printf("%s%.17g", k ? "," : "", eps); }
  a.complete_adaptation(eps);
  std::printf(",%.17g]\n", eps);
}
"""
