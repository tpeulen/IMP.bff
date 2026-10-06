---
type: reference
title: Portable numerical driver builds
description: Numerical driver headers must use the existing standalone compatibility seam; source and installed array exports are verified as one provider.
resource: src/Minimize.cpp, src/Odeint.cpp, src/SparseLinearAlgebra.cpp
tags: [numerics, standalone, build, validation]
timestamp: '2026-10-06T00:00:00Z'
---

# Where to pick this up

The three numerical drivers now include `IMP/bff/IMPCompatibility.h`, the
existing owner of portable exception helpers. Direct `IMP/exception.h`
includes had escaped the broader standalone test because its unrelated RMF
headers were absent and caused a skip. The dedicated
`test/test_numerics_standalone_compiles.py` instead compiles each actual driver
with standalone headers and the existing Eigen headers only:3failures before,
3passes after, with inherited include paths removed so a full IMP install
cannot hide the packaging defect.

Next: build the exact committed core provider and verify exported F
distribution arrays against the numerical oracle. ChiSurf's former pinned
provider lacks the newer arrays even though the owning source already has
them. Rebuild wrapper and binary together; never substitute a client-side
statistics fallback. Cross-stack progress belongs to ChiSurf PRD-154 W7.
