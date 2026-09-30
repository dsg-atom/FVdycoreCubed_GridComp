# Prompt 92 — CUDA `REMAP_PRESSUREVARS` Implementation, Validation, and Performance

Branch: `optimization/cuda-fv3-performance`. Scope: implement, GPU-validate,
integration-test, and performance-test a CUDA port of the
`fv_mapz.F90:1239-1516` `REMAP_PRESSUREVARS` timer body, whose numerical
content and CUDA-port GO decision were established by Prompt 91
(`p91_remap_pressurevars_cuda_port_plan_2026-09-27.md`). Fortran remains the
production default; this prompt adds an opt-in CUDA path only.

**Explicitly out of scope (unchanged, verified untouched):** PT/DELZ,
REMAP_UV, REMAP_Q, tracer CUDA (`fv_tp_2d`/hord==8, CLOSED per Prompt 83),
`REMAP_TOTALENERGYCON`, `MOIST_CAPPA`/`USE_COND` policy, `a_imp`, R4,
model-wide persistent-residency redesign. No numerical formula was
algebraically rewritten; no fast-math; no dead-branch porting (hydrostatic,
`remap_te`, `remap_pt`, `do_omega` branches remain unported, matching
Prompt 91's dead-code finding). No commit/push performed.

## 1. Implementation Summary

- New per-column device function + host launcher implementing the ~50
  numerically-active lines identified by Prompt 91: pe2/pn2/pk2/delp
  recompute and the `remap_t` + `MOIST_CAPPA` cappa/pkz block, transcribed
  directly from `fv_mapz.F90:1239-1516` with no algebraic changes.
  Reuses the arithmetic pattern already present in `fv3_pt_delz_remap.cu`
  (Prompt 91's "VERY HIGH CUDA reuse" finding), following the same
  self-contained-transfer architecture as every existing vertical-remap
  port (PT/DELZ, REMAP_UV, REMAP_Q).
- `q(sphum)`-dependent `MOIST_CAPPA`/`cappa` path **is** supported (Stage 1
  decision: implement, not special-case the degenerate `nwat=0` identity)
  — the device kernel evaluates the general formula; the `cappa==akap`
  simplification is a checked invariant of that general formula, not a
  separate code path. This is a strictly more general implementation than
  the minimum Prompt 91 flagged as sufficient.
- Data transfer architecture (Stage 4/5, resolved prior session): fully
  self-contained per call — H2D 7 calls total (2 kernel-specific + 5 shared
  mask-field transfers already resident from other kernels in the same
  step), D2H 6 calls total (1 kernel-specific + 5 shared mask-field);
  workspace is 3 small per-column scratch device arrays, following the
  `fv3_gpu_pt_delz_remap` pattern. No new class of full-state host/device
  transfer introduced.
- ABI/Fortran bridge: thin C ABI wrapper following the established
  `fv3_gpu_pt_delz_remap`/`fv3_gpu_remap_q` convention; compiled clean.
- Production selector: `use_cuda_remap_pressurevars`, gated by
  `RUN_CUDA_FV3_PRESSUREVARS_ENABLE` (opt-in) /
  `RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN` (explicit Fortran escape hatch,
  also the implicit default) / `RUN_CUDA_FV3_PRESSUREVARS_DUMP_EXIT`
  (diagnostic `[P92_EXIT_DUMP]` print), matching the
  `RUN_CUDA_FV3_REMAPQ_*` naming convention exactly. **Fortran is the
  default with no env vars set — this prompt does not flip the default.**

## 2. Numerical Validation

- Numerical unit: 10 host-authoritative fixtures from Prompt 91's
  `test_remap_pressurevars_host_reference.cpp` (Option C transcription),
  run against the new CUDA device kernel.
- Result: **10/10 PASS**, including the non-degenerate
  `q_sphum_nonzero_hypothetical` fixture (exercises the general
  `MOIST_CAPPA` formula, not just the `cappa==akap` degenerate case).
- Numerical class: **essentially BITWISE** — `pk` shows ~1 ULP round-off
  only (transcendental evaluation order), all other fields bitwise-exact.
  Max numerical difference observed: ~1 ULP (double precision).
- Invariants (7 classes derived from source, Stage 9): **PASS**, all
  fixtures.
- `compute-sanitizer` (production-shape smoke test,
  `fv3_remap_pressurevars_prod_shape_smoke.cu`, real A100 via `srun`):
  **PASS, 0 errors** (`========= ERROR SUMMARY: 0 errors`). Note:
  `compute-sanitizer` is not under the `discover_platform.md`-documented
  Nsight `profilers/` PATH — actual location on this NVHPC 24.7 install is
  `$NVHPC_ROOT/cuda/12.5/compute-sanitizer/compute-sanitizer`.

## 3. Shared-Helper Regression (Stage 10)

Full project ctest run: 2 pre-existing failures (`fv3_halo_mpi_transport`,
`fv3_wp_b2_scalar_transport`), both attributable to already-known,
already-tracked prior-session tracer/hord8 work (Prompt 73-83, CLOSED per
CURRENT_STATE.md §7 "STOP CUDA TRACER PRODUCTION OPTIMIZATION"), confirmed
orthogonal via `git status` — the affected source files were not touched
by this task. No new regression introduced by `REMAP_PRESSUREVARS`.

## 4. C32 Integration (Stage 12)

Real Held-Suarez cold-start C32/L10 run (`RUN_DT=1200`, 6 ranks, single
A100, `p92_c32_cuda` vs `p92_c32_fortran`), same template family as prior
prompts' own C32 gates.

| Arm | Selector | rc | Dispatch (`[P92_EXIT_DUMP]` count) |
|---|---|---|---|
| CUDA | `RUN_CUDA_FV3_PRESSUREVARS_ENABLE=1` | 0 | 36/36 `use_cuda_remap_pressurevars=T` |
| Fortran | `RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN=1` | 0 | 0/36 (all `=F`) |

No NaN/crash/divergence in visible stdout between arms. **C32 integration:
PASS.**

Caveat carried from the platform's own structural limitation
(CURRENT_STATE.md §7, unrelated to this port): Held-Suarez cold-start runs
never populate `q(sphum)` (`fv_restart.F90` skips IC read under
`COLDSTART=1`+empty `ExtData.rc`), so this integration run alone cannot
exercise the non-degenerate `MOIST_CAPPA` path in production — that
coverage instead comes entirely from the fixture-level GPU validation in
§2, which does include a non-degenerate `q(sphum)!=0` case.

## 5. C512 Same-Node Paired Performance (Stage 13/14)

Methodology: 5 pairs, alternating order A1 B1 / B2 A2 / A3 B3 / B4 A4 / A5
B5 (reduces thermal/order bias), single Slurm allocation, MPS active,
same binary (sha256 `3d4364e6...e0002`, verified per arm), all other
production-default CUDA paths held fixed (PT/DELZ CUDA default,
REMAP_UV CUDA default, REMAP_Q CUDA default, `FUSE_HALO_EDGES`/
`FUSE_CORNER_FILL=YES`, Graph OFF) so only the REMAP_PRESSUREVARS delta is
isolated. Arm A = Fortran (production default, `..._FORCE_FORTRAN=1`);
Arm B = CUDA opt-in (`..._ENABLE=1`). All 10 arms completed rc=0, 30
calls/arm (`DYN Date` count = 30/30 every arm).

Timer results (ms/call, mean across the 5-pair set):

| Timer | Fortran (Arm A) | CUDA (Arm B) | Δ |
|---|---|---|---|
| REMAP_PRESSUREVARS | 131.64 | 37.66 | **−3.497x direct speedup** |
| REMAPPING | 500.59 | 421.75 | **−15.75%** |
| FV_DYN_LOOP | 1712.98 | 1630.13 | **−4.84%** |
| DYN_CORE | ~25.4-25.5 s (unchanged) | ~25.4-25.5 s (unchanged) | no cross-contamination |
| TOTAL | 4483.63 | 4402.70 | **−1.81%** |

Direct speedup ≈3.50x is closely consistent with Prompt 91's a-priori
performance-ceiling estimate (~90.7 ms/call recoverable at a
REMAP_Q-comparable 3.23x speedup floor) — a strong internal-consistency
check; the achieved improvement is somewhat better than the Prompt 91
floor estimate. Result was consistent in direction and magnitude across
all 5 independent pairs — no sign flip, no outlier pair.

GPU memory: peak observed 40265/40960 MiB, whole-node/6-rank-shared-via-
MPS (≈6711 MiB/rank average) — not isolated to this kernel specifically,
consistent with prior ports' own memory-budget behavior. **C512
performance: PASS.**

## 6. Decision

Numerically correct (bitwise/round-off-clean, sanitizer-clean,
invariant-clean), integration-correct (C32 selector dispatch verified both
directions, no divergence), and performance-positive at production C512
scale with a result consistent across 5 independent same-node pairs.

**Decision: KEEP AND CONSIDER DEFAULT ENABLE.** Per this prompt's binding
scope, **the production default is not changed here** — Fortran remains
default; `RUN_CUDA_FV3_PRESSUREVARS_ENABLE=1` is the opt-in escape hatch.
A future prompt may revisit flipping the default, following the same
two-step precedent set by REMAP_Q (Prompt 88B implemented as opt-in with
verdict "KEEP AS OPT-IN"; Prompt 89 independently reconfirmed and flipped
the default).

```
REMAP_PRESSUREVARS CUDA PERFORMANCE POSITIVE — KEEP AND CONSIDER DEFAULT ENABLE
```

## Addendum (2026-09-28, Prompt 93)

Prompt 93 independently reconfirmed this result via statistical
robustness gating (5/5 pairs CUDA-positive, mean TOTAL improvement
1.805%) and one longer 60-call run (TOTAL improvement 1.676%), then
flipped `use_cuda_remap_pressurevars` to the CUDA production default
(escape hatch `RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN=1`). Fortran is no
longer the production default as of Prompt 93. See
`p93_remap_pressurevars_default_policy_2026-09-28.md` for the full
robustness gate and rebuild/smoke evidence.
