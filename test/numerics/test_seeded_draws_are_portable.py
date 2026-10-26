"""A seeded sample draws the same numbers on every platform.

The engines (mt19937, mt19937_64) are fixed by the C++ standard; the
distributions on top of them are not, and libstdc++ (Linux) and libc++
(macOS) implement them differently -- the same seed once gave a different
50 000-pair AV distance histogram on each. IMP.bff's seeded draws go through
``IMP/bff/internal/PortableRandom.h``, which specifies them bit for bit.

The golden counts below were recorded on macOS arm64 (libc++); the same test
must pass unchanged on Linux x86_64 (libstdc++). Integer counts with unit
weights keep floating-point summation out of the comparison.
"""

import IMP.bff

GOLDEN = [0] * 13 + [97, 1636, 1998, 269] + [0] * 23


def _cloud(n, offset):
    """A deterministic cloud of ``n`` unit-weight points (no random numbers)."""
    out = []
    for i in range(n):
        out += [
            offset + 0.37 * (i % 7) + 0.011 * i,
            1.13 * (i % 5) - 0.017 * i,
            0.71 * (i % 3) + 0.005 * i,
            1.0,
        ]
    return out


def test_the_pair_sample_of_a_seed_is_the_same_everywhere():
    axis = [float(x) for x in range(0, 81, 2)]
    counts = IMP.bff.cloud_distance_distribution(
        _cloud(97, 0.0), _cloud(113, 30.0), axis, 4000, 7, False
    )
    assert [int(round(v)) for v in counts] == GOLDEN


def test_another_seed_draws_another_sample():
    axis = [float(x) for x in range(0, 81, 2)]
    counts = IMP.bff.cloud_distance_distribution(
        _cloud(97, 0.0), _cloud(113, 30.0), axis, 4000, 8, False
    )
    assert sum(counts) == 4000
    assert [int(round(v)) for v in counts] != GOLDEN
