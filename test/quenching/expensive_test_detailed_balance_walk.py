"""The PDE ensemble against a Brownian walk started from p(x), on the MalE AV.

The walk is the particle picture QuEst's FRET ensemble uses (Ito in mobility:
jump variance 2 D0 s^n dt, rejected at the volume's wall); its stationary
state is p ~ s^-n, the weight the detailed-balance PDE is given, and the two
share a continuum limit. Started from p(x), the walk needs no equilibration,
so the comparison is of the lifetimes alone.

Measured 2026-10-20 (40 000 walkers, tau0 = 4 ns, D0 = 10 A^2/ns, dg 0.5 A):

    slow  PDE tau_x  walk dt 2 ps        walk dt 0.5 ps
    1.0   3.3950     -0.02 % (-0.1 s)    -0.04 % (-0.3 s)
    0.3   2.9001     +0.75 % (+3.3 s)    +0.41 % (+1.8 s)
    0.1   1.9089     +0.75 % (+1.7 s)    +0.81 % (+1.9 s)

The walk's excess at 0.3 shrinks with its time step (a sticky boundary
crossed by a finite jump); the static (no-transport) values are 3.75, 3.23
and 1.97 ns, so the agreement is well inside the transport effect. 0.1 is
reported, not asserted: there the walk's own bias and its error are the same
size as each other and as the residual.
"""

import numpy as np
import pytest

import IMP.bff as bff

from test_detailed_balance_pde import TAU0, _male_fields, male, walk_lifetime  # noqa: F401

WALK_BIAS = 0.01   # the walk's time-step bias at dt = 2 ps, measured above


@pytest.mark.parametrize("slow", [1.0, 0.3])
def test_the_walk_agrees_with_the_pde(male, slow):
    d, w = _male_fields(male, slow)
    b, k = male["b"], male["k"]
    p = w * b / (w * b).sum()
    pde = bff.GridDiffusionSolver(d, b, p, k, male["dg"], 0.01, "detailed_balance",
                                  False, w).lifetime(TAU0)
    tau, se = walk_lifetime(male, slow)
    print("slow %.2f: PDE tau_x %.4f ns, walk %.4f +- %.4f ns (%.2f sigma)"
          % (slow, pde.get_tau_x(), tau, se, (tau - pde.get_tau_x()) / se))
    assert abs(tau - pde.get_tau_x()) < 3.0 * se + WALK_BIAS * pde.get_tau_x()
