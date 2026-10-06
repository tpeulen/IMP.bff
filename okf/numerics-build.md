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

## Selected-model confidence in core builds (2026-10-07)

The owned core link had an undefined read_structure_table reference because
LabelizerScore read pLDDT through an IMP-only adapter. The existing portable
PDB record now retains the temperature factor and LabelizerStructure carries
a parallel selected-atom vector. Confidence comes from that selected model's
CA index, preserving the arithmetic and documented zero default for absent
or invalid factors. It cannot be overwritten by a later MODEL block.
Seven portable scientific cases passed, including the actual148L fixture,
model selection, factor text variants and unchanged record inventory.
The candidate also passed scalar/array F-tail oracles; its Python residual
callback still needs the owning SWIG director-input conversion repaired.
No candidate wheel has been installed as a shipping provider.
