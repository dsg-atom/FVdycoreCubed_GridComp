# Prompt 55 — PT/DELZ CUDA Kernel Out-of-Bounds Fix, and a New Blocking Finding

Date: 2026-09-22. Branch: `optimization/cuda-fv3-performance`. Model: Sonnet
(no Opus escalation — the Prompt 54 OOB was already localized to a concrete
indexing/shape mismatch by static analysis before this prompt began).

## 1. Scope

Prompt 54 left `pt_delz_remap_kernel` failing with a compute-sanitizer-
confirmed out-of-bounds `__global__` read at a device-buffer boundary,
blocking every invocation. Prompt 55's mandate: fix that one indexing defect
(single-fix budget), revalidate through standalone fixtures, production-shape
smoke, and a first real C32 integrated run, then stop before any C512 work or
further kernel changes.

## 2. Stages 0-5 — root cause (static analysis, no new GPU runs required)

`FV3_FID_PE`/`FV3_FID_PELN` are registered in
`fv3_gpu_field_registry.F90` from `Atm%pe(is-1:ie+1,npz+1,js-1:je+1)` /
`Atm%peln(is:ie,npz+1,js:je)` — physical Fortran dimension order `(i,k,j)`,
with `k` as the **middle** axis. Every other field the kernel touches
(`pt`/`delz`/`delp`/`pk`/`pkz`) has `k` **last**. `RiemViewC3::at(i,j,k)` is a
plain positional accessor with no per-field axis awareness — calling it with
`(i,j,k)` for `pe`/`peln` silently walks into the wrong (adjacent) device
buffer instead of overflowing its own, exactly matching Prompt 54's
sanitizer-observed boundary-straddling addresses (a 101,728-byte allocation
and an adjacent 90,112-byte one).

## 3. Stage 6 — fix applied (the one authorized substantial change)

`cuda/dyn_core/vertical/fv3_pt_delz_remap.cu`: 4 call sites changed from
`.at(i,j,k)` to `.at(i,k,j)` for `a.pe`/`a.peln` only (read-loop and
closing-update loop). No other field's accessor pattern changed; no formula
changed. `cuda/dyn_core/vertical/fv3_pt_delz_remap_smoke.cu` (single-column
view builder) updated to match: the stride-1 ("k axis") slot moved from `s2`
to `s1` for `pe`/`peln`'s `RiemViewC3` construction.

## 4. Stage 7 — standalone regression

`fv3_wp_c_pt_delz_remap_gpu_smoke` (existing single-column test, updated for
the new view convention) and the existing `map_scalar` fixtures: rebuilt in
`build_cuda_r8_mpi`, run via `srun` on one A100 — **6/6 PASS at roundoff**
(rel errors ~1e-16). Rerun under `compute-sanitizer --tool memcheck
--leak-check full`: **0 errors, 0 bytes leaked**, identical numerical output.

## 5. Stage 8 — new production-shape multi-column smoke test

`fv3_pt_delz_remap_prod_shape_smoke.cu` (new file, added this prompt): builds
device views with the **same** lower-bound/stride layout the real registry
produces for a production-shaped C32 tile (1-cell halo on `pe`/`peln`,
3-cell halo on `pt`/`delz`/`delp`, no halo on `pk`/`pkz`), then launches the
real production launcher (`launch_pt_delz_remap`) over the full interior 2D
grid — the same launch shape that triggered Prompt 54's finding. Every
column is broadcast from one of the 6 existing fixture cases, so per-column
output must reproduce that fixture's single-column result. Result: **6/6
PASS at roundoff**, rebuilt and rerun under compute-sanitizer memcheck: **0
errors, 0 bytes leaked**. This closed the gap between "hand-built degenerate
single-column view" (Prompt 52C, cannot reproduce Prompt 54's finding at
all) and a real multi-dimensional registry-shaped launch.

## 6. Production binary rebuild

The production `StandAlone_FV3_Dycore.x` is built in a **separate repo**,
`GEOSfvdycore_fortran_perf/build`, linking against 9 prebuilt static
archives in `FVdycoreCubed_GridComp/build_cuda_r8_intelmpi` (confirmed via
`CMakeCache.txt`'s `FV3_CUDA_LIBRARY_DIR`). `fv3_pt_delz_remap.cu` belongs to
the `fv3_wp_c_vertical` CMake target specifically — **not**
`fv3_dyn_core_gpu`, which shares the same archive group but does not contain
this source file (confirmed by mtime: rebuilding `fv3_dyn_core_gpu` alone
left `fv3_pt_delz_remap.cu.o` untouched). Rebuilt `fv3_wp_c_vertical`
explicitly under the Intel-MPI production stack, then relinked
`StandAlone_FV3_Dycore.x`:

- Executable: `GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x`
- mtime: `2026-09-22 16:12:10.684743290`
- sha256: `06e8939b0203028e4164dcec098b5e693e5164e9c8a5b98d9868bf95ff684da8`
  (distinct from Prompt 54's `74d7f7c6...`)
- `strings` confirms 51 `fv3_gpu_bridge_init|RUN_CUDA_FV3|pt_delz_remap` hits
  and 7 `fv3_gpu_pt_delz_remap|launch_pt_delz_remap` hits.
- Per-job preflight (sha256 + symbol-count check, fail-fast on mismatch)
  retained in `p55_stage9_c32.sbatch`, following Prompt 54's convention.

## 7. Stage 9 — first real integrated C32 run: mixed result

Job **58512259** (`p55_stage9_c32.sbatch`), same-node matched pair, reused
`p52c_stage11_{fortran,cuda}` job dirs unchanged (C32, L10,
`kord_tm=kord_wz=9`, Held-Suarez, `RUN_CUDA_FV3=1` only in the cuda dir),
CUDA arm wrapped in `compute-sanitizer --tool memcheck` (no source changes;
supplementary memory-safety evidence only). Completed, `sacct` ExitCode
`0:0`, 39s.

**The Prompt 54 OOB is confirmed fixed**: `compute-sanitizer` reports **0
errors** on all 6 ranks; the prior failure strings
(`"unspecified launch failure"`, `"fv3_gpu_sync_to_host failed"`) are absent
(0 hits) from the log. This is the first time `pt_delz_remap_kernel` has
ever executed to completion in the integrated harness.

**A new, severe, CUDA-arm-only numerical corruption was found**, unrelated
to the OOB just fixed:

- `Range_Warn` (MAPL/FMS physical-range check) fires **1,901,524** times in
  the CUDA arm's log, starting at the very first diagnostic opportunity —
  *before* the first `DYN Date` progress line is even printed (log line 323
  vs. 58771) — with grossly nonphysical temperatures, e.g.
  `Range_Warn: TA_dyn(1,1,1)= -813.3985 at LON:306.1773 LAT:-34.6999`.
- `NaN` appears **1,843,383** times.
- `[RIEM3-BAD i=<i> j=<j>] dz_nonneg=... dz_worst=nan pe_nonpos=...
  pe_nonmono=... ws=nan ...` fires at many scattered `(i,j)` points
  throughout the run. This string is a **pre-existing** diagnostic in
  `cuda/dyn_core/vertical/fv3_vertical_kernels.cu` (WP-C Riemann solver, the
  already Gate-C2-validated nonhydrostatic vertical kernel) — it is not new
  instrumentation added by this prompt, and it is downstream of, not inside,
  `pt_delz_remap_kernel`.
- The matched **Fortran arm** (same job, same node): completely clean — 0
  `Range_Warn`, 0 `NaN`, 0 abort hits.
- The run did not crash (no `MPI_Abort`/`FATAL`) and reached its configured
  end with a normal-looking MAPL Finalize/timing report — i.e. this is *not*
  a repeat of Prompt 54's crash; the dispatch is memory-safe but producing
  wrong values.

**Interpretation**: `pt_delz_remap_kernel`'s output (`pt`/`delz`/`delp`/
`pkz`) feeds directly into the next acoustic substep's Riemann solver
(`fv3_vertical_kernels.cu`). The corruption's timing (present from the very
first call, before any progress is logged) and its complete absence from
every standalone/fixture/production-shape test to date (Stages 7-8, all
6/6 PASS at roundoff) means the real, live registry-computed views differ
from every view this prompt's tests constructed by hand — i.e. this is a
**second, distinct defect** from the OOB just fixed, most likely a further
registry-view stride/extent/halo mismatch specific to the live
`fv3_gpu_field_registry.F90` registration path, not reproduced by any
synthetic view built so far. It has not been localized further.

**This finding was not root-caused.** Per Prompt 55's explicit stop-on-
mismatch rule and its Forbidden list (no further kernel-math changes without
fresh authorization, and Prompt 55's one-substantial-fix budget was already
spent in Stage 6), no further source change was attempted. Stage 10 (C32
correctness comparison) was not attempted since the CUDA arm's raw output is
already known to be grossly invalid — a field-by-field diff would not add
information beyond what §7 above already establishes. Stages 11-12 (C512,
milestone restoration) were not attempted.

## 8. Summary

- Prompt 54's out-of-bounds `pt_delz_remap_kernel` read: **FIXED**, confirmed
  by standalone regression (Stage 7), a new production-shape multi-column
  smoke test (Stage 8), and 0 compute-sanitizer errors in the first real
  integrated invocation (Stage 9) — all direct evidence of memory safety.
- pt/delz CUDA dispatch integrated-timestep correctness: **NOT NUMERICALLY
  VALID**. GPU EXECUTED: yes (no crash, memory-safe). NUMERICALLY VALIDATED:
  **no** — a new, severe, CUDA-arm-only corruption (immediate, ~1.9M
  Range_Warn / ~1.8M NaN hits, RIEM3-BAD Riemann-solver diagnostics) was
  found on first real use, distinct from and not explained by the OOB fix
  above. Root cause not yet identified.
- Job ID of record: **58512259**. Executable sha256: `06e8939b...`.
- Gate C2 (54-stage production first-run inventory) is **unaffected** — this
  dispatch was never part of that closed inventory.
- **Update (Prompt 56)**: the recommended comparison above was performed —
  live registry-view metadata AND live pre-kernel input values were dumped
  directly from a real C32 invocation and compared against the Stages 7-8
  assumptions. Both match exactly; the registry-view/stale-input hypothesis
  is **refuted**. The corruption is therefore Category H — inside
  `pt_delz_remap_kernel`'s own arithmetic (or its downstream Riemann-solver
  consumption, unmodified), not a registry/view/layout/sync defect. See
  `docs/performance/p56_pt_delz_live_registry_corruption_2026-09-22.md` for
  full detail. **Next step still requires fresh, explicit authorization**:
  a kernel-math-adjacent investigation (savepoint comparison of this
  kernel's own outputs, or line-by-line arithmetic review against
  `fv_mapz.F90`) is needed to localize the specific defective expression —
  out of scope for a registry-diagnostics-only prompt.
