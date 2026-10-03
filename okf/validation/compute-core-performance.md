---
okf_version: "0.2"
---

# Compute-core performance campaign — verified 2026-10-03

## Acceptance and source identity

The parent completed a **normal CMake configure and `IMP.bff-python` build**, then independently exercised the actual generated bindings and native library. The accepted work is portable CPU value inference, allocation-free scalar graph writes, contiguous chain export, and ChiSurf's progress/result boundary. No new numerical dependency, precision change, posterior target, search budget, or policy weight was introduced.

The preserved BFF baseline is `bac0b26691b7cdd67f927d7c710b6d16c9a09208`; the original ChiSurf sampler source was captured at `667b3e3c899d4226cf62ea0ae5ca37f9d0971a34`. Runtime was Apple ARM64, Release `-O3 -DNDEBUG`, `IMP_BFF_GPU=off`, the `arm64` conda interpreter, and offscreen Qt with an isolated HOME.

**Durable raw evidence:** [compute-core-performance-results.json](compute-core-performance-results.json). It contains all eight fresh-process runs, loaded library paths, native/shim/ChiSurf source SHA-256 identities, work counts, output hashes, and the additional boundary microbenchmarks. Native, shim and ChiSurf source identities were constant within each mode across all four pairs and differed between original and candidate.

The baseline facade was generated from archived **original BFF headers and SWIG interfaces**, not from the candidate with a few methods deleted. This matters because unrelated new bindings in the shared tree can make a current facade fail against an older extension. The original dylib and extension were loaded non-destructively from an isolated directory. The baseline ChiSurf sampler module was loaded from its preserved source. No live artifact was swapped.

Every sampling repetition builds a fresh graph/fit, and every real search repetition rebuilds its prepared problem. Recorded values match exactly: Gaussian sampling retains **19,216 evaluations / 19,200 rows** and identical complete-chain hashes; ChiSurf retains **601 evaluations / 600 rows**, callback counts and identical result hashes for each seed; real FRET retains **20 simulations / 3 evaluations**, winner and reward. Scaled and unscaled NN output hashes also match.

## What changed

- `include/internal/MlpCore.h`: a separate value-only forward path borrows unscaled input. Derivatives, backward and training retain their owning workspace; scaled inference retains the original path. No GEMM arithmetic or activation implementation changed.
- `GraphPort`: direct one-slot scalar storage avoids the temporary one-element vector while preserving coercion, bounds, sanitisation, invalidation and link propagation.
- `MCMCSampler`: `get_chain_flat()` exports a contiguous row-major chain. The legacy nested API and empty `(0,)` public `chain` shape remain unchanged. `chain_flat` is a separate property.
- ChiSurf: signatures are inspected once per run. Inspectable progress-only callbacks no longer trigger partial-result materialisation. Result-aware callbacks retain their schema. Opaque two-argument callbacks retain their result-first/binding-error fallback; Python callback-body `TypeError` is propagated without a duplicate invocation.
- SWIG: thread support is enabled, ordinary wrapper releases remain opt-in, and **all Python directors acquire the GIL**. Native MCMC and ModelSearch runs release it. ModelSearch exceptions are translated after GIL restoration using the existing IMP/standalone translator; pending Python director errors retain their original exception.
- Tracked CMake helpers enable BFF-only SWIG thread generation despite IMP adding pyext as a sibling directory. Normal regeneration uses the configured Python/real SWIG through a generated launcher; the parent/cache and other modules' SWIG remain unchanged. Standalone UseSWIG also enables director thread support. Ignored generated `pyext/CMakeLists.txt` edits are not the deliverable.

## Measured performance

Four interleaved original/candidate pairs, three fresh fixtures per process, reported as the median of per-process medians:

| End-to-end workload | Original | Candidate | Interpretation |
|---|---:|---:|---|
| Native correlated-Gaussian sampler | 76.07 ms | 65.47 ms | **1.16x**, 13.9% less elapsed time; all four candidate runs faster |
| ChiSurf, no callback | 64.18 ms | 63.47 ms | Essentially neutral; no stable speedup claimed |
| ChiSurf, progress-only callback | 112.64 ms | 61.32 ms | **1.84x**, 45.6% less elapsed time; no unnecessary partial export |
| ChiSurf, full result callback | 109.78 ms | 120.63 ms | **About 10% slower in this run**; not an end-to-end win |
| Real FRET native MCTS | 39.72 ms | 40.25 ms | Essentially neutral; no search-algorithm speedup claimed |
| Tabular native MCTS | 0.275 ms | 0.272 ms | Essentially neutral |

The full-result case includes fit construction, sampling, conversion, and callback capture. The contiguous export is retained because its isolated conversion benefit is independently measured, not because every whole callback workload improved. Display/result construction can still dominate. The raw per-process medians are retained rather than hiding the slower case.

### Input-copy-dominated NN microbenchmarks

A deterministic, **unscaled, single identity layer with one output** isolates the copies removed by value inference. Inputs are pre-created contiguous NumPy arrays. Two additional interleaved pairs, seven timings of forty calls per profile, with identical output hashes:

| Inputs / batch rows | Original / call | Candidate / call | Speedup |
|---|---:|---:|---:|
| 64 / 128 | 8.87 us | 5.53 us | **1.60x** |
| 1,024 / 256 | 841.51 us | 480.75 us | **1.75x** |
| 4,096 / 128 | 1,799.73 us | 1,077.97 us | **1.67x** |

These are copy-sensitive fixtures, **not a universal dense-network improvement**. The tiny two-layer NN cases in the end-to-end runner were neutral/noisy, including small scaled-input slowdowns. Scaled inference is not changed by the new value path. Earlier unmatched 12.1% surrogate and 1.46x sampling observations are superseded by the verified comparisons above.

### Chain export boundary

On the same final native sampler, nested conversion and flat conversion return identical arrays. Two independent candidate processes measured:

| Two-parameter chain rows | Nested export + NumPy | Flat export + reshape | Speedup range |
|---|---:|---:|---:|
| 512 | 185–192 us | 2.33–2.37 us | **79–81x** |
| 4,096 | 1,564–1,565 us | 13.60–14.18 us | **110–115x** |
| 16,384 | 6.07–6.09 ms | 54.11–63.60 us | **96–112x** |

This is strictly the export boundary, not sampler throughput or end-to-end fitting. The native chain representation and ordering are unchanged.

### Final frozen-candidate recheck under load

A concurrent SMLM build changed the shared native library/facade after the
quiet measurements. The parent froze a matching extension, dylib and generated
facade in `candidate-pinned/`, and repeated all four pairs with explicit
facade/extension loading for **both** modes. SHA-256 identities and complete
results remain invariant; the ChiSurf production source is unchanged.

This later machine was heavily loaded (other compiler/test processes and the
desktop renderer were active). The Gaussian medians were 157.22 -> 154.47 ms
and real FRET 84.65 -> 88.13 ms: the modest earlier sampler gain was not
resolved through the noise, so **1.16x is a quiet-fixture observation, not a
universal throughput guarantee**. Progress-only ChiSurf retained the clear
benefit, 277.24 -> 138.02 ms (**2.01x**), with identical work/results. Full
results were 259.23 -> 272.80 ms, about 5% slower. Large copy-heavy NN cases
remained faster (1.30x / 1.85x), while the 64-input case was neutral. All
figures and loaded paths from both runs are versioned in the evidence JSON.

The frozen parent and **verification subprocesses** both use the preserved
facade, avoiding an older extension paired with a newly regenerated SMLM
facade. A first frozen run exposed that harness mismatch in the isolated
exception test (570 passed / 65 skipped, one import mismatch); the harness
was fixed without changing production code or weakening that regression.

## Independent verification

- **BFF: 571 passed, 65 skipped**, independently repeated against the final frozen candidate with both parent and subprocess facade/native identities pinned. The earlier normal-build run also passed 571/65. Scope: all `test_neural_net*.py`, sampler, graph, port/director, minimizer, all MCTS, exception family, and the new SWIG thread/packaging tests. Optional-dependency/platform skips remain visible.
- **HmmSurrogate: 19 passed** on the exact frozen candidate, CPU-only (`test/test_hmm_surrogate.py`), in addition to the 571-test BFF scope.
- **ChiSurf: 58 passed, 1 unrelated existing failure** across sampler/results, native MCTS/execution/fixed-structure/controller, real responsiveness and backend architecture. The failure is `test_the_session_format_is_bffs` in the independently dirty `project.py`. Parent ran that assertion against the current source and preserved HEAD source: current failed, HEAD passed. No project lifecycle code or test was changed to disguise it.
- Focused callback fix: observed RED (`np.add` rejected the `result` keyword), then **10 passed** with the opaque callback and callback-body error regressions.
- Independent read-only review found that one callback compatibility issue; a separate final review accepted its fix with empty security/logic concerns. The rest of the scoped diff had no reported blocking issue.
- Ruff passes all owned production/helper/new test files; scoped whitespace checks pass.
- Real native tests demonstrate MCTS timer-driven cancellation before the 10-million-simulation budget, MCMC background heartbeat, Python director success/error propagation, and ordinary Python exceptions for an unconfigured ModelSearch run rather than a process abort.
- Tiny compiled shared-core director fixtures exercise both IMP and standalone thread/exception semantics; generated real director bodies have `SWIG_PYTHON_THREAD_BEGIN_BLOCK`.
- No GUI source was changed. Intel/AMD execution, real Windows launcher runtime, GPU runtime, and a full standalone numerical build were **not** performed in this campaign. Architecture dispatches and generic CPU paths remain intact.

## Reproduction

Normal build, using the existing local cached configuration:

```sh
/Users/tpeulen/mambaforge/envs/arm64/bin/cmake -S /Users/tpeulen/dev/imp -B /Users/tpeulen/dev/imp/cmake-build-arm64
/Users/tpeulen/mambaforge/envs/arm64/bin/cmake --build /Users/tpeulen/dev/imp/cmake-build-arm64 --target IMP.bff-python -j 2
```

The BFF generator command must use `modules/bff/pyext/swig-launcher/swig-threads`; CMakeCache retains the original configured SWIG executable. This was checked after normal regeneration.

Parent test-driver and benchmark scripts, exact commands, RED/GREEN logs and archived baseline sources are in `/Users/tpeulen/.hermes/cache/scratch/bff-core-performance/`: `parent_verify.py`, `generate_original_shim.py`, `paired_runner.py`, `check_paired.py`, and `boundary_microbench.py`. Scratch may be pruned; the versioned JSON above is the permanent acceptance evidence. Rebuild the archived baseline revision rather than using the candidate Python facade with an old library. Never time these workloads while another agent rebuilds the shared native library; verify the native/shim hashes remain constant across pairs.

## Rejected experiments and where to pick this up

1. No MCTS algorithm change survived: path-vector reuse and no-op snapshot restore did not produce a stable real-family win. Fixed-budget search winners/rewards remain unchanged. Profile objective/minimizer work before another search optimization.
2. Full-result progress remains an open performance front: the final paired run was slower, despite the large isolated export gain. Profile fit/result/display work and segment scheduling without dropping arrays, callbacks, or statistical work.
3. Keep value borrowing separate from the derivative workspace. The earlier shared-workspace borrowing attempt segfaulted and remains rejected. Do not credit unchanged scaled inference with unscaled-copy gains.
4. Run actual x86/Windows/full standalone lanes before claiming those platforms verified. Thread support must survive normal regeneration; a manual wrapper build alone is insufficient.
