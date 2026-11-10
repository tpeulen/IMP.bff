# pocketfft (vendored)

Vendored 2026-11-10 by claude-code/Opus 5.5 for tpeulen (PRD-155 step 1).

- `include/pocketfft/pocketfft_hdronly.h`: byte-identical copy of
  `tttrlib/thirdparty/pocketfft/pocketfft_hdronly.h` (tttrlib commit
  8b3099794257fccaf713ce1bfa3f6ae3542db9d8, the last change to that file).
  Upstream: https://gitlab.mpcdf.mpg.de/mtr/pocketfft (C++ header-only branch),
  Martin Reinecke and Peter Bell.
- `LICENSE.md`: the BSD-3-Clause licence shipped with it.

Why: the Bayesian decay headers (`include/bayesian/`) compile only where
`pocketfft/pocketfft_hdronly.h` is on the include path. Until now that was only
true for drivers given tttrlib's `thirdparty/`, so neither the native nor the
Pyodide wheel contained the Bayesian decay fit. Private to the library build:
no installed public header needs it.

Update: copy the header from tttrlib again (keep the two identical) and record
the tttrlib commit here.
