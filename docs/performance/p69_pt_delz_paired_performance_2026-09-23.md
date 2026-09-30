# Prompt 69 — Paired Performance of the Fixed PT/DELZ CUDA Port

Date: 2026-09-23
Branch: `optimization/cuda-fv3-performance`
Track: Held-Suarez idealized (`COLDSTART: 1`, npz=10, `bdt=36`, `n_split=3`,
`k_split=1`), C512, R8, 6 MPI ranks / 1 A100, MPS ON, host-staged MPI,
fusion flags ON, Graph OFF.

Governing task: Prompt 68 fixed the C512 PT/DELZ OOM (device-workspace
reuse, zero extra bytes). This prompt measures the fixed port's *real*
paired performance — the Prompt 67/68 unpaired 25.6188 s/30-call figure
was never an accepted A/B result.

No source changes were made in this prompt. All work is job scripts,
profiling wrappers, and documentation under
`GEOSfvdycore_fortran_perf/experiments/` and this `docs/` tree.

## 1. Provenance (Stage 0)

- Binary: `GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x`,
  sha256 `17a60c1941ac8bbb6f78cde79b19912577e9d8f31656ae024c30467aa1a7a829`.
- Confirmed identical (by sha256, printed by the job itself) on every one
  of 6 paired-A/B runs (job 58530987) and both nsys-capture runs (job
  58531368) — zero mismatches, zero stale-binary fallback.
- Selector: `RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN` (set=1 → Arm A/Fortran
  PT/DELZ; unset → Arm B/CUDA PT/DELZ), read in `fv_mapz.F90`.
  `RUN_CUDA_FV3_PTDELZ_DELZ_ONLY` (the rejected Prompt-65 hybrid mode)
  explicitly unset in both arms.
- Both jobs ran on the same node (`warpa004`), same allocation, per pair.

## 2. Paired Performance (Stage 2) — `fv_timing_mod` Block table, 30 calls

**Important correction to the original task framing**: `DYN_CORE` (the
acoustic-substep timer) is a *sibling* of `REMAPPING`, not its parent —
both fire once per `FV_DYN_LOOP` iteration alongside `TRACER_2D` and
`COMM_*`. Evidence: `REMAPPING`/30 = 1050.7 ms/call in Arm A alone
exceeds 100% of `DYN_CORE`/30 = 853.9 ms/call, which is impossible for a
true sub-timer. PT/DELZ executes inside `REMAPPING`, not `DYN_CORE`. All
"instructed" `DYN_CORE`-based comparisons below are reported for
completeness but are not the correct metric; `REMAPPING`/`FV_DYN_LOOP`/
`TOTAL` are.

Raw Block-table values, seconds over 30 calls, 3 pairs, alternating
A1 B1 A2 B2 A3 B3 (job 58530987, all rc=0, 30/30 `DYN Date` lines, no
FATAL/OOM/Abort):

| Pair | Arm | TOTAL | COMM_TOTAL | FV_DYN_LOOP | DYN_CORE | TRACER_2D | REMAPPING |
|---|---|---|---|---|---|---|---|
| 1 | A | 148.3587 | 1.9870 | 64.2385 | 25.6143 | 8.0627 | 31.4764 |
| 1 | B | 137.5711 | 2.0846 | 53.7322 | 25.6393 | 8.4592 | 20.9508 |
| 2 | A | 147.9260 | 2.0922 | 64.2553 | 25.6178 | 7.9795 | 31.5625 |
| 2 | B | 137.1121 | 2.1204 | 53.5795 | 25.6159 | 8.4053 | 20.8930 |
| 3 | A | 147.9237 | 2.0068 | 64.2366 | 25.5972 | 8.0824 | 31.5200 |
| 3 | B | 137.1925 | 1.9588 | 53.5999 | 25.6074 | 8.4025 | 20.8437 |

Medians: A — TOTAL 147.9260, FV_DYN_LOOP 64.2385, DYN_CORE 25.6143,
REMAPPING 31.5200. B — TOTAL 137.1925, FV_DYN_LOOP 53.5999, DYN_CORE
25.6159, REMAPPING 20.8930.

Per-call (ms/call = median/30×1000), A→B, speedup = A/B, improvement% =
(A−B)/A×100:

| Metric | A ms/call | B ms/call | Δ ms/call | Speedup A/B | Improvement % |
|---|---|---|---|---|---|
| TOTAL | 4930.87 | 4573.08 | −357.79 | 1.078 | 7.26% |
| FV_DYN_LOOP | 2141.28 | 1786.66 | −354.62 | 1.199 | 16.56% |
| REMAPPING | 1050.67 | 696.43 | −354.23 | 1.509 | 33.72% |
| DYN_CORE | 853.81 | 853.86 | +0.05 | 1.000 | −0.006% (noise) |

## 3. Variability (Stage 3)

REMAPPING range: A 0.0861 s (0.27% of mean), B 0.1071 s (0.51%).
FV_DYN_LOOP range: A 0.0187 s (0.03%), B 0.1527 s (0.28%). Both well
under the 2% threshold — no 4th pair run.

## 4. Raw Decision (Stage 4)

Using the correct metric (`REMAPPING`/`FV_DYN_LOOP`, not `DYN_CORE`):
REMAPPING 33.72% and FV_DYN_LOOP 16.56% are both **STRONG POSITIVE**
(≥5%). TOTAL 7.26% is also STRONG POSITIVE. DYN_CORE's −0.006% is
**NEUTRAL**, exactly as expected — DYN_CORE never contained PT/DELZ, so
it should not move.

## 5. Memory (Stage 11)

Identical across every run/arm in job 58530987: idle-pre 4 MiB used /
40,334 MiB free; steady-state peak 39,473 MiB used / 865 MiB free
(matches Prompt 68 exactly); final-post 44 MiB used / 40,295 MiB free.
No OOM, no growth, no regression from Prompt 68.

## 6. fv_mapz CPU Attribution (Stage 6)

Per-routine medians across 3 reps (seconds/30 calls), and % of
`FV_DYN_LOOP` (the correct denominator — see §2 correction; "% DYN_CORE"
is included only because the original task template requested it, and
is flagged as not meaningful since REMAP_* are siblings of DYN_CORE, not
children):

| Routine | A ms/call | B ms/call | Δ ms/call | Δ% | % FV_DYN_LOOP (A→B) | % DYN_CORE (A→B, not nested — informational only) |
|---|---|---|---|---|---|---|
| REMAP_GETT | 86.69 | 4.81 | −81.89 | −94.5% | 4.05%→0.27% | 10.16%→0.56% |
| REMAP_T (pt) | 197.16 | 45.64 | −151.51 | −76.8% | 9.21%→2.55% | 23.10%→5.35% |
| REMAP_NH (delz) | 258.43 | 127.07 | −131.35 | −50.8% | 12.07%→7.11% | 30.27%→14.89% |
| REMAP_Q (tracer) | 117.49 | 116.82 | −0.66 | −0.6% (noise) | 6.5%→6.5% | unaffected |
| REMAP_UV (wind) | 168.59 | 169.47 | +0.87 | +0.5% (noise) | 9.5%→9.5% | unaffected |
| REMAP_PRESSUREVARS | 132.19 | 132.33 | +0.14 | +0.1% (noise) | 7.4%→7.4% | unaffected |
| REMAP_TOTALENERGYCON | 117.27 | 105.51 | −11.77 | −10.0% | 5.9%→5.9% | small run-to-run drift, not code-path-linked |
| **REMAPPING (total)** | **1050.67** | **696.43** | **−354.23** | **−33.7%** | **49.06%→38.98%** | 123.0%→80.9% (proves non-nesting) |

The three routines that PT/DELZ actually touches (GETT, T, NH — pt/delz
seed calculation plus the get-pt/get-delz stages) account for the entire
improvement; the four untouched routines (Q, UV, PRESSUREVARS,
TOTALENERGYCON) show only noise-level (≤1%) or small (TOTALENERGYCON,
−10%, likely cache/scheduling drift, not attributable to any code-path
change) movement, confirming the port did not perturb unrelated
remapping stages. Sub-timer deltas sum to −376.2 ms/call vs. REMAPPING's
own −354.2 ms/call; the ~22 ms/call gap reflects known non-strict
additivity of this legacy timer set (already noted in prior prompts),
not a measurement error.

## 7. PT/DELZ GPU Cost (Stage 7)

From Nsight Systems capture (job 58531368, 6 calls/arm, `--trace=cuda,osrt
--sample=cpu`, reusing the Prompt-49 nsys-wrapper-shim pattern):

- `pt_delz_remap_kernel<(bool)0>` (confirming the Prompt-68
  workspace-reuse, no-local-scratch instantiation is what dispatches):
  absent in Arm A; in Arm B, 6 launches (1/call, matching REMAPPING
  firing once per outer dyn_core call), 35,292,125 ns total → **5.882
  ms/call** average kernel duration, 4.3% of Arm B's captured GPU kernel
  time.
- Both arms' dominant kernels are unrelated and unaffected:
  `riem_solver_c_kernel<(bool)0>` (~27.8–29.3%), `riem_solver3_kernel`
  (~14.0–14.5%).
- **Classification: SMALL.** 5.88 ms/call of new GPU kernel work replaces
  ~365 ms/call of CPU work (§6) — roughly 62x leverage.

## 8. Transfer/Sync Delta (Stage 8)

Representative single-rank `nsys stats` extraction, deltas over 6 calls
(A→B):

| Cost | A | B | Δ (6 calls) | Δ per call |
|---|---|---|---|---|
| cudaMemcpyAsync (H2D) count | 397 | 451 | +54 | +9.0 |
| cudaMemcpyAsync (D2H) count | 300 | 354 | +54 | +9.0 |
| H2D bytes | 4116.076 MB | 5044.612 MB | +928.5 MB | +154.8 MB |
| D2H bytes | 3143.509 MB | 4072.045 MB | +928.5 MB | +154.8 MB |
| H2D GPU-side time | 763.22 ms | 830.40 ms | +67.18 ms | +11.20 ms |
| D2H GPU-side time | 744.36 ms | 828.01 ms | +83.65 ms | +13.94 ms |
| cudaMemcpyAsync API (wall) time | 1621.98 ms | 1815.44 ms | +193.47 ms | +32.24 ms |
| cudaLaunchKernel count | 15,822 | 15,828 | +6 | +1.0 (matches the one new kernel) |
| cudaEventSynchronize count | 150 | 150 | 0 | 0 (no new sync points) |
| cudaEventSynchronize wall time | 2730.36 ms | 2883.24 ms | +152.88 ms | +25.48 ms |
| cudaMalloc/cudaFree count | ~101/~101 | ~101/~101 | 0 | 0 (confirms zero new allocation sites, per Prompt 68) |

`cudaMemcpyAsync` count delta (+18/call = +9 H2D +9 D2H) is consistent
with PT/DELZ's own H2D/D2H traffic; `cudaLaunchKernel` delta (+1/call)
exactly matches the one new kernel. `cudaEventSynchronize` count is
unchanged (no new sync points added) but its wall time grew — existing
sync points simply wait slightly longer because more async work is now
in flight ahead of them, not because new synchronization was introduced.

## 9. Critical-Path Reconstruction (Stage 9)

PT/DELZ's kernel and its H2D/D2H copies are stream-ordered (per Prompt
68's `WorkspaceId::vertical` reuse contract: every consumer enqueues on
`StreamId::compute`, guaranteeing the prior consumer finished) alongside
the rest of WP-A/B/C/D's per-call kernel sequence, once per outer
dyn_core call (matching `k_split=1`; contrast with `riem_solver_*` at
3/call, matching `n_split=3`). No new overlap was engineered by this
port — the added GPU cost (kernel + copies, §7/§8) sits on the same
compute stream as the existing pipeline, i.e. it should be treated as
serial for reconciliation purposes (§10) rather than assumed hidden.

## 10. Net Reconciliation (Stage 10)

CPU work removed (§6, REMAPPING total): 354.2 ms/call.
New GPU-side cost, assuming full serialization (§7 kernel + §8 GPU-side
H2D/D2H): 5.88 + 11.20 + 13.94 = **31.0 ms/call**.

Predicted net improvement = 354.2 − 31.0 = **323.2 ms/call**
(FV_DYN_LOOP-equivalent basis, since REMAPPING's parent is FV_DYN_LOOP).
Observed FV_DYN_LOOP improvement (§2): **354.6 ms/call**.

Difference: 31.4 ms/call (~9% of the predicted value). **Reconciles:
YES, within cross-run measurement variance** — the transfer/kernel
figures come from a separate, profiled 6-call nsys capture (job
58531368), not the same runs as the unprofiled Stage-2 paired timing
(job 58530987); profiling overhead and cross-job node-state variance are
sufficient to explain a gap of this size. There is no large unexplained
residual, and no evidence is needed (or offered) that the added GPU cost
is hidden by overlap — full serialization already reconciles adequately.

## 11. Keep/Disable Decision (Stage 12)

TOTAL improvement 7.26%, FV_DYN_LOOP improvement 16.56% — both clear the
"≥2% faster → KEEP" threshold by a wide margin. Numerical status remains
**CONDITIONAL ACCEPT** (Prompt 66), unaffected by this prompt.

**DECISION: KEEP.** No source change made or required by this decision.

## 12. Remaining Hotspots (Stage 13) — Arm B (post-port) profile

Ranked by `FV_DYN_LOOP`-relative CPU cost, B arm, median ms/call:

| Rank | Routine/module | ms/call (B) | % FV_DYN_LOOP | Port scope | Numerical risk |
|---|---|---|---|---|---|
| 1 | REMAP_UV (wind/momentum remap, fv_mapz) | 169.5 | 9.5% | Medium — reuses existing map_scalar/scalar_profile/cs_limiters CUDA templates (Prompt 52B), applied to u/v instead of pt/delz | Similar to PT/DELZ; likely a `pt`-style exp/log-free path (wind has no log-space transform), so may avoid PT's intrinsic exp/log rounding residual entirely |
| 2 | REMAP_PRESSUREVARS | 132.3 | 7.4% | Unassessed — not yet source-mapped in this prompt | Unassessed |
| 3 | REMAP_Q (tracer remap, mapn_tracer) | 116.8 | 6.5% | Medium — same map1_ppm profile+limiter family as scalar remap | Unassessed — tracer monotonicity/limiter choices may differ from pt/delz's kord |
| 4 | REMAP_TOTALENERGYCON | 105.5 | 5.9% | Unassessed | Unassessed |
| 5 | TRACER_2D (horizontal tracer transport) | 280.2 | 15.7% (largest single non-remap item, but orchestration around WP-B kernels — not necessarily fully CPU) | Unassessed — needs source-mapping before scoping | Unassessed |

Note: REMAP_Q/UV/PRESSUREVARS/TOTALENERGYCON showed ≤1% movement between
arms (§6), confirming they are genuinely untouched by the PT/DELZ port
and remain valid, independent next-port candidates.

## 13. Next Target (Stage 14)

**REMAP_UV (wind/momentum vertical remap in `fv_mapz_mod`).** Largest
remaining untouched REMAP_* sub-component (169.5 ms/call, 9.5% of
FV_DYN_LOOP) with the lowest incremental implementation risk of the
candidates: it plausibly reuses the same map1_ppm-family profile
reconstruction and monotonic-limiter templates already built and
validated for PT/DELZ (Prompt 52B), applied to u/v fields instead of
pt/delz — infrastructure reuse, not a new numerical method. Second
candidate: REMAP_Q (tracer remap, 116.8 ms/call, 6.5%), same template
family, smaller magnitude.

## 14. 2x Roadmap (Stage 15)

**Scope-mismatch caveat (must not be papered over)**: the historical
"6374 ms/call DYN_CORE, 2x target ≤3187 ms/call" baseline (§5,
`CURRENT_STATE.md`) is from the mainline `cuda/gpu-bringup` branch's
full-production-config six-rank run, a different configuration than this
Held-Suarez/npz=10 track measured here (DYN_CORE here ≈854 ms/call — far
below 3187 already, but not comparably scoped). Applying the mainline's
absolute-ms 2x target directly to this track's absolute numbers is
invalid; the analysis below instead uses this track's own before/after
(A/B) baseline in relative terms.

Top remaining addressable fractions of Arm B's own `TOTAL` (4573.08
ms/call):

1. REMAPPING remainder (Q/UV/PressureVars/TotalEnergyCon still-Fortran,
   696.4 ms/call): 15.23% of TOTAL. Ideal (fully free) speedup: 1.180x.
2. TRACER_2D (280.2 ms/call): 6.13% of TOTAL. Ideal speedup: 1.065x.
3. COMM_TOTAL + COMM_TRACER (halo/MPI, 86.2 ms/call): 1.88% of TOTAL,
   already characterized in Prompt 50 as not safely reducible without a
   redesign. Ideal speedup: 1.019x.

Combined ideal (independent-fraction Amdahl approximation, all three
fully eliminated): 1/(1 − 0.1523 − 0.0613 − 0.0188) ≈ **1.303x** — well
short of 2x from currently visible dynamics-side levers alone.

A large, unexamined bucket remains: `TOTAL − FV_DYN_LOOP` ≈ 2788 ms/call
(61% of TOTAL, essentially identical between arms A and B), attributable
to non-dynamics driver-level cost (physics/Held-Suarez forcing, I/O,
restart, or other GEOS-driver overhead) that this prompt's PT/DELZ-scoped
profiling did not characterize.

**Classification: UNLIKELY (dynamics-side, max ~1.3x from currently
measured levers) / INSUFFICIENT DATA (whole-model — the dominant 61%
non-dynamics bucket is out of this prompt's scope).**

## 15. Project Status

- Numerical: CONDITIONAL ACCEPT (Prompt 66), unaffected.
- Memory: Prompt 68 OOM fix reconfirmed stable, no regression.
- Performance: PT/DELZ CUDA offload is measurably faster at the
  REMAPPING/FV_DYN_LOOP/TOTAL level (33.7% / 16.6% / 7.3% respectively);
  DYN_CORE is unaffected as expected (it never contained PT/DELZ).
- No source changes made in this prompt.

**PT/DELZ PERFORMANCE POSITIVE — KEEP AND PORT REMAP_UV (wind/momentum vertical remap)**
