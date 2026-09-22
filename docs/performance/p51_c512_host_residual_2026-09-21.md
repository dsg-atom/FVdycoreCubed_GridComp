# C512 Host-Side Residual Decomposition via Nsight Systems CPU Sampling (Prompt 51)

Date: 2026-09-21
Status: **MEASUREMENT/CHARACTERIZATION ONLY.** No source changed. No
optimization implemented. One GPU allocation used (job 58494926; budget: 1
preferred, max 1 rerun — no rerun needed, first capture was healthy). `ncu`
not used. No hardware performance counters used. No CUDA-aware MPI or
multi-GPU work. No new timing matrix (reused Prompt 48/49's validated
same-node config verbatim).

## 0. Run configuration (identical to Prompt 49/50's validated baseline)

| Field | Value |
|---|---|
| Job | 58494926, `gpu_a100`/`rome`, 1 A100, MPS ON, RC=0 |
| Grid | C512, R8, npz=10, `k_split=1`, `n_split=3`, `RUN_DT=36` |
| Ranks | 6 MPI ranks (`FV_NX=1`/`FV_NY=6`), 1 tile/rank |
| Backend | `RUN_CUDA_FV3=1`, `RUN_HELD_SUAREZ=1`, `COLDSTART=1` |
| Halo transport | host-staged (default) |
| Fusion flags | `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES` |
| Graph mode | OFF |
| Capture | `nsys profile --trace=cuda,osrt --sample=process-tree`, all 6 ranks, PID-keyed output |
| Duration | 6 `DYN_CORE` cycles (1 warmup + 5 steady-state), same as Prompt 49 |

Reproducibility check (Stage 0): run health RC=0, no NaN/Inf/fatal in
`run.log`; all 6 `.sqlite` exports present and queryable. CPU sampling did
not visibly destabilize the run (same cycle count, same RC as the
untraced/CUDA-only-traced Prompt 49 run).

## 1. Method

Nsight Systems CPU-sampling export tables used: `COMPOSITE_EVENTS` (one row
per periodic on-CPU sample: `id`, `start`, `globalTid`) joined to
`SAMPLING_CALLCHAINS` (`stackDepth=0` = innermost/leaf frame; higher depths
= callers) joined to `StringIds` for symbol/module text. This only captures
"Running" (on-CPU) samples — no separate off-CPU/blocked accounting was
attempted beyond a coarse on-CPU-fraction sanity check (§2).

The residual is defined identically to Prompt 49: for each rank, build the
interval union ("covered") of all `CUPTI_ACTIVITY_KIND_KERNEL` ∪
`CUPTI_ACTIVITY_KIND_RUNTIME` ∪ `CUPTI_ACTIVITY_KIND_MEMCPY` intervals inside
that rank's own kernel-activity window (first kernel start → last kernel
end); every main-thread sample not inside a covered interval is a "gap"
(residual) sample. Each sample is independently classified into one category
by its leaf symbol/module (`CASE`/`LIKE` on `StringIds`), then split into
covered vs. gap. This gives both (a) a category's share of the whole window
and (b) a category's share specifically of the untraced residual.

Representative rank: PID 218001 (deep, gap/covered-split analysis).
Cross-checked against 3 other ranks (218002/218003/218004) for outlier
detection — same precedent as Prompt 49 (4 of 6 ranks checked).

## 2. Stage 2 — On-CPU sanity check (rank 218001)

| Metric | Value |
|---|---|
| Window span | 19,815.7 ms |
| On-CPU samples in window | 23,031 (99.44% of window, by sample-quantum accounting) |
| Sample quantum | 855,652 ns/sample (empirically: on-CPU time ÷ sample count) |
| Covered (traced CUDA activity) | 4,544.2 ms (22.93%) |
| **Residual (gap)** | **15,271.6 ms (77.07%)** |

This closely reproduces Prompt 49's 22.3%/77.7% split (different rank, same
run) — cross-validates that the residual is not a rank- or run-specific
artifact and that the CPU-sampling capture is measuring the same phenomenon
as Prompt 49's CUDA-trace-only capture.

## 3. Stage 3 — Cross-rank outlier check

Total in-window leaf-sample category counts, all 4 checked ranks (raw
sample counts, no covered/gap split — that split was only computed in depth
for the representative rank):

| Category | 218001 | 218002 | 218003 | 218004 |
|---|---|---|---|---|
| B_FV3_FORTRAN | 11,397 (49.5%) | 13,383 (58.1%) | 13,611 (59.0%) | 11,305 (48.9%) |
| E_MAPL_griddedio (math, see §7) | 3,579 (15.5%) | 3,648 (15.8%) | 3,656 (15.9%) | 3,646 (15.8%) |
| C_CUDA_DRIVER (libcuda.so) | 4,125 (17.9%) | 4,108 (17.8%) | 4,089 (17.7%) | 4,135 (17.9%) |
| A_MPI | 2,351 (10.2%) | 475 (2.1%) | 265 (1.2%) | 2,402 (10.4%) |
| F_THREAD_WAIT | 1,330 (5.8%) | 1,165 (5.1%) | 1,175 (5.1%) | 1,333 (5.8%) |
| D_CPP_DISPATCH | 77 (0.3%) | 122 (0.5%) | 124 (0.5%) | 80 (0.3%) |
| H_UNKNOWN | 156 (0.7%) | 112 (0.5%) | 118 (0.5%) | 196 (0.8%) |
| P/C2 (negligible) | 16 | 12 | 11 | 11 |
| **Total samples** | 23,031 | 23,025 | 23,049 | 23,108 |

**No outlier rank.** B_FV3_FORTRAN dominant in all 4 (49–59%);
E_MAPL_griddedio and C_CUDA_DRIVER are remarkably flat across ranks
(15.5–15.9% and 17.7–17.9% respectively — the latter almost entirely inside
covered/traced intervals, see §8); F_THREAD_WAIT flat (5.1–5.8%). A_MPI is
the one category with real rank-to-rank spread (1.2–10.4%), consistent with
cubed-sphere edge/corner ranks having different neighbor-halo timing —
expected topology-driven variance, not measurement noise. Total per-rank
sample counts agree within 0.3%, confirming all 4 ranks measured
comparable-length windows. Representative rank 218001 is confirmed
representative; no second GPU allocation was needed.

## 4. Stage 5 — Top leaf functions (rank 218001, top 30 of full capture)

| Symbol | Module | Samples | Category |
|---|---|---|---|
| `fv_mapz_mod_mp_map_scalar_` | StandAlone_FV3_Dycore.x | 2724 | B (Fortran) |
| `__bwr_log` | libMAPL.griddedio.so | 1445 | B (math, see §7) |
| `__bwr_exp` | libMAPL.griddedio.so | 1397 | B (math, see §7) |
| `__intel_avx_rep_memcpy` | StandAlone_FV3_Dycore.x | 963 | B (Fortran) |
| `fvdycorecubed_gridcomp_mp_run_` | StandAlone_FV3_Dycore.x | 728 | B (Fortran) |
| `fv_mapz_mod_mp_scalar_profile_` | StandAlone_FV3_Dycore.x | 663 | B (Fortran) |
| `__intel_avx_rep_memset` | StandAlone_FV3_Dycore.x | 639 | B (Fortran) |
| `[vdso]` (gettimeofday-class) | [vdso] | 591 | F (thread-wait) |
| `fv_mapz_mod_mp_lagrangian_to_eulerian_` | StandAlone_FV3_Dycore.x | 523 | B (Fortran) |
| `sw_core_mod_mp_d2a2c_vect_` | StandAlone_FV3_Dycore.x | 513 | B (Fortran) |
| `__bwr_pow` | libMAPL.griddedio.so | 497 | B (math, see §7) |
| `fv_mapz_mod_mp_mapn_tracer_` | StandAlone_FV3_Dycore.x | 413 | B (Fortran) |
| `libcuda.so` internals (×4, unresolved addrs) | libcuda.so.535.104.12 | 399/361/325/243/198 | C (driver, see §8) |
| `psm3_verbs_ips_ptl_poll` | libpsm3-fi.so | 315 | A (MPI) |
| `fv_dynamics_mod_mp_fv_dynamics_` | StandAlone_FV3_Dycore.x | 297 | B (Fortran) |
| `fv_statemod_mp_fv_to_state_` | StandAlone_FV3_Dycore.x | 297 | B (Fortran) |
| `fv_grid_utils_mod_mp_c2l_ord4_` | StandAlone_FV3_Dycore.x | 291 | B (Fortran) |
| `tp_core_mod_mp_fv_tp_2d_` | StandAlone_FV3_Dycore.x | 291 | B (Fortran) |
| `tp_core_mod_mp_yppm_` | StandAlone_FV3_Dycore.x | 277 | B (Fortran) |
| `fv_statemod_mp_state_to_fv_` | StandAlone_FV3_Dycore.x | 265 | B (Fortran) |
| `fv_tracer2d_mod_mp_tracer_2d_1l_` | StandAlone_FV3_Dycore.x | 256 | B (Fortran) |
| `libmpi.so` internal (unresolved) | libmpi.so.12.0.0 | 237 | A (MPI) |
| `tp_core_mod_mp_xppm_` | StandAlone_FV3_Dycore.x | 226 | B (Fortran) |
| `fv_statemod_mp_ppme_` | StandAlone_FV3_Dycore.x | 203 | B (Fortran) |
| `MPIDI_Progress_test` | libmpi.so.12.0.0 | 198 | A (MPI) |
| `__bwr_cos` | libMAPL.griddedio.so | 193 | B (math, see §7) |

Single largest individual symbol in the entire capture: `fv_mapz_mod_mp_map_scalar_`
(2724 samples ≈ 2331 ms ≈ 11.8% of the whole window) — part of the vertical
Lagrangian-to-Eulerian remapping subsystem (see §7).

## 5. Stage 6 — MPI wait/progress decomposition (Category A)

| Metric | Value |
|---|---|
| ms/`DYN_CORE` window | 2010.8 ms |
| % of window | 10.15% |
| % of residual | 13.17% |

Top symbols: `psm3_verbs_ips_ptl_poll` (315), `MPIDI_Progress_test` (198),
plus `MPIR_Progress_hook_exec_on_vci`, `MPIDI_OFI_progress`,
`psm3_mq_ipeek_dequeue_multi`, `psm3_verbs_recvhdrq_progress` (smaller). All
are Intel MPI 2021.13 / PSM3 **busy-poll** progress-engine functions —
confirmed CPU-spinning, not OS-sleep, consistent with the near-100% on-CPU
finding (§2). 99.96% of these samples are gap (untraced) samples, i.e. this
category is almost entirely part of the residual, not overlapping with
traced CUDA activity — this main thread is not doing GPU-side work while it
spins in MPI progress. Rank-to-rank variance (1.2–10.4% of samples, §3) is
consistent with topology-dependent neighbor-wait asymmetry, not measurement
noise.

## 6. Stage 7 — FV3 Fortran CPU compute classification (Category B)

**Classification: DOMINANT.** 9751.0 ms gap / 9751.9 ms total = 49.21% of
the window, 63.85% of the residual (before folding in §7's `libMAPL.griddedio.so`-attributed
math calls — see §7 for the combined figure).

Top untranslated routines, by module:

- **`fv_mapz_mod`** (vertical Lagrangian-to-Eulerian remapping):
  `map_scalar_` (2724), `scalar_profile_` (663), `lagrangian_to_eulerian_`
  (523), `mapn_tracer_` (413) — **4323 samples ≈ 3699 ms ≈ 18.67% of the
  entire window**, the single largest coherent cluster in the capture.
- **`tp_core_mod`/transport**: `fv_tp_2d_` (291), `yppm_` (277), `xppm_`
  (226) — 794 samples ≈ 679 ms.
- **`fv_tracer2d_mod_mp_tracer_2d_1l_`** (large-timestep tracer transport):
  256 samples ≈ 219 ms.
- **`sw_core_mod_mp_d2a2c_vect_`**: 513 samples ≈ 439 ms.
- **`fv_statemod`** (state↔FV conversion / diagnostics): `fv_to_state_`
  (297), `state_to_fv_` (265), `ppme_` (203), `fv_getpkz_`/`fv_getepv_`
  (smaller) — driven partly by `__bwr_*` math calls (§7).
- **`fvdycorecubed_gridcomp_mp_run_`** (top-level GridComp driver, glue
  code): 728 samples ≈ 623 ms.
- **`__intel_avx_rep_memcpy`/`memset`** (compiler-inserted array
  copy/init): 1602 samples ≈ 1371 ms.

**Important architectural observation (not a correctness claim — Stage 6
already ruled the CUDA halo-transport sync region correct, and this task
does not re-audit correctness):** the routine names here — `fv_mapz_mod`
(vertical remap), `fv_tracer2d_mod` (tracer transport), `fv_statemod`
(state conversion) — match the authoritative FV3 algorithm's
**large-timestep companions to `dyn_core`**, not `dyn_core`'s own internal
acoustic-substep stages. Per `docs/agent_instructions/project_architecture.md`,
the CUDA-translated 54-stage inventory (WP-A/B/C/D) covers `dyn_core`'s
internal C-grid/D-grid/vertical-Riemann/pressure-operator stages; it does
not claim coverage of `fv_dynamics`'s `n_map`-loop siblings (vertical remap,
tracer transport, state conversion) that run once per `k_split` iteration
alongside — not inside — the CUDA-replaced `dyn_core` call. This is
consistent with (not contradicted by) Gate C2's closure: those 54 stages
remain correctly, numerically validated; this residual is a **coverage gap**
(work never targeted for CUDA translation), not a correctness gap in
already-translated work.

## 7. `__bwr_log`/`__bwr_exp`/`__bwr_pow`/`__bwr_cos` disambiguation (Stage 10's open question, resolved)

Call-stack context above the leaf frame (`SAMPLING_CALLCHAINS.stackDepth`
1–4) for all 3579 `__bwr_*` samples shows every immediate caller is a core
FV3 dynamics/physics routine, not MAPL diagnostic/I-O code:

| Caller (stackDepth=1) | Samples |
|---|---|
| `fv_mapz_mod_mp_lagrangian_to_eulerian_` | 259 |
| `held_suarez_forcingmod_mp_held_suarez_forcing_` | 199 |
| `fvdycorecubed_gridcomp_mp_run_` | 195 |
| `fv_dynamics_mod_mp_fv_dynamics_` | 96 |
| `fv_statemod_mp_state_to_fv_` | 64 |
| `fv_statemod_mp_fv_getpkz_` | 60 |
| `fv_statemod_mp_fv_getepv_` | 18 |

The leaf frame's *module* is `libMAPL.griddedio.so` because that shared
object exports the `__bwr_*` symbol (Intel compiler runtime's range-reduced
vectorized `log`/`exp`/`pow`/`cos`/`sin` implementation) that the dynamic
linker resolved these calls to — **the work itself is not MAPL griddedio
I/O**, it is genuine transcendental math executed directly by FV3 dynamics/
physics code: pressure→`p**kappa` (`fv_getpkz_`), Ertel PV (`fv_getepv_`),
vertical coordinate remapping (`lagrangian_to_eulerian_`), and
latitude/pressure-dependent Newtonian relaxation (`held_suarez_forcing_`).

**Correction to the original category label**: this 3579-sample /
3062.4 ms / 15.45%-of-window cluster is reclassified from
"E. MAPL/FMS/ESMF overhead" to **Category B (FV3 Fortran compute)** — it is
the same kind of untranslated dynamics/physics CPU work as §6, just routed
through a shared library's exported math symbol rather than the main
executable's own text segment. No genuine MAPL/FMS/ESMF I/O or diagnostic
overhead was found in the residual at a material level (the true
`libMAPL`/`libFMS`/`libESMF`-*owned* control-flow code, as opposed to its
exported math routines, does not appear as a separate material cluster in
the top-30 list, §4).

**Combined B (Fortran compute, including griddedio-resolved math):**

| Metric | Value |
|---|---|
| ms/`DYN_CORE` window | 12,813.4 ms |
| % of window | **64.66%** |
| % of residual | **83.90%** |

## 8. Stage 8 — CUDA runtime/driver blocking beyond the known halo sync (Category C/C2)

| Metric | Value |
|---|---|
| `libcuda.so` (driver) samples | 4125 total, only **2 in the gap** |
| `libcudart` (runtime API) samples | 8 total, only **1 in the gap** |
| Combined contribution to residual | 2.6 ms ≈ **0.02% of residual** |

Exhaustive per-address check (§3's prior segment, ~180+ distinct unresolved
`libcuda.so` addresses) confirmed essentially all driver-internal CPU
samples occur *inside* already-traced/covered CUDA API call intervals
(`cudaEventSynchronize`, `cudaMemcpyAsync`, `cudaLaunchKernel`, etc. —
already measured by Prompt 49). **Conclusion: no material additional
CUDA-driver-level host blocking exists beyond Prompt 49's own
`cudaEventSynchronize`/`cudaMemcpyAsync` figures.** The residual is
essentially CUDA-driver-free.

## 9. Stage 9 — Thread/scheduler waiting (Category F)

On-CPU fraction for the main CUDA-API-issuing thread: 99.44% in-window
(§2) — consistent with Prompt 49's finding that this thread is not
blocked/sleeping for any material fraction of the window (caveat: sampled
"Running" states only measure on-CPU presence at each sample tick, not true
continuous scheduler state — do not over-interpret this as a certified
zero-blocking guarantee, only as consistent with one).

Category F (libc/pthread/vDSO thread-wait): 1330 total samples, but only
**300 in the gap** (256.7 ms ≈ 1.68% of residual) — the remaining 1030 are
inside already-covered CUDA-API intervals (e.g., pthread mutex/futex calls
made internally by the CUDA runtime/driver while servicing a traced API
call, or `NSYS_OSRT_*` profiler-injection wrapper functions around traced
`poll`/`sched_yield` calls). No OpenMP runtime activity (`libomp`) detected
on the main thread in-window — Category G is confirmed zero, not a
classification artifact (verified by isolating the covered-join from the
category `CASE` logic in separate temp tables, per the debugging note
below).

## 10. Stage 11 — Host-residual decomposition table

Representative rank 218001, residual = 15,271.6 ms (77.07% of the 19,815.7 ms
window):

| Category | ms/`DYN_CORE` window | % of window | % of residual |
|---|---|---|---|
| **B. FV3 Fortran compute** (incl. griddedio-resolved math, §7) | 12,813.4 | 64.66% | **83.90%** |
| **A. Intel MPI progress engine** | 2,010.8 | 10.15% | 13.17% |
| **F. Thread/scheduler wait** (gap only) | 256.7 | 1.30% | 1.68% |
| **H. Unknown/unresolved** | 88.1 | 0.44% | 0.58% |
| **D. C++ CUDA dispatch bridge** (`fv3_gpu_*`) | 65.9 | 0.33% | 0.43% |
| **C/C2/P.** CUDA driver, runtime API, nsys-injection (negligible) | 5.2 | 0.03% | 0.03% |
| **Total** | 15,240.1 | 76.94%¹ | 99.79%¹ |

¹ Cross-check: independently summing per-category gap-time (15,240.1 ms)
vs. the interval-union-derived residual (15,271.6 ms) agrees within
**0.21%** — internal validation that the sampling-based decomposition and
the CUDA-trace-based residual are measuring the same underlying phenomenon.
**Combined accounting requirement (≥90% of the residual): met at 99.79%,
and 100% by construction of the category partition (H_UNKNOWN itself is a
named, bounded 0.58%-of-residual "genuinely unresolved" bucket, not missing
accounting).**

## 11. Stage 12 — Combined full critical-path table (Prompt 49 + Prompt 51)

| Bucket | % of `DYN_CORE` window | Source |
|---|---|---|
| GPU kernel compute (171 kernels) | 3.9% | Prompt 49 |
| `cudaEventSynchronize` (halo-transport blocking sync) | 14.4% | Prompt 49 |
| `cudaMemcpyAsync` (host-staged transfer) | 7.9% | Prompt 49 |
| `cudaLaunchKernel` | 1.7% | Prompt 49 |
| Other CUDA API | 0.4% | Prompt 49 |
| *(Traced/covered subtotal — this rank: 22.93%, cross-validating Prompt 49's 22.3%)* | | |
| **FV3 Fortran compute** (vertical remap, tracer transport, state conversion, math) | **64.66%** | **Prompt 51** |
| Intel MPI progress engine | 10.15% | Prompt 51 |
| Thread/scheduler wait | 1.30% | Prompt 51 |
| Unknown/unresolved | 0.44% | Prompt 51 |
| C++ CUDA dispatch bridge | 0.33% | Prompt 51 |
| CUDA driver/runtime API/nsys overhead (residual-side) | 0.03% | Prompt 51 |
| **Total** | **100.0%** (this rank: 22.93% + 77.07%) | |

**100% of the window is now named and measured** — no honestly-unattributed
bucket remains larger than 0.44% (Category H).

## 12. Stage 13 — Next optimization target (decision rule applied)

Decision rule (unchanged from Prompt 49): select the largest bucket that is
both (a) fully attributable to a specific, bounded subsystem and (b)
actionable without new profiling instrumentation.

- Rank 2 from Prompt 49 (halo-transport blocking sync, 14.4%) was
  investigated in Prompt 50 and found **not safely reducible** without a
  redesign that relocates the same cost — closed, not revisited (per this
  task's own explicit instruction).
- **New #1: Category B, FV3 Fortran compute — 64.66% of the window, 4.5x
  larger than the halo sync.** Fully attributable to named routines
  (`fv_mapz_mod`'s vertical Lagrangian-to-Eulerian remap is the single
  largest sub-cluster at 18.67% of the window alone; `fv_tracer2d_mod`
  tracer transport, `fv_statemod` state conversion, and Held-Suarez forcing
  math make up the rest). Actionable without new profiling: the specific
  routines and their approximate cost share are now known; the next step
  is a scoping decision (does this fall under an existing WP or does it
  need a new one), not more measurement.

**Selected target: CUDA translation of `dyn_core`'s large-timestep
companions — principally `fv_mapz_mod`'s vertical remapping (highest
single-routine cost, 18.67% of window) and `fv_tracer2d_mod`'s tracer
transport.** This is a **new finding**, not previously listed in
`docs/agent_handoff/CURRENT_STATE.md` §7/§9 (which lists `MOIST_CAPPA`/
`USE_COND` and `a_imp<=0.999` as the only known correctness/coverage gaps).
This is a **coverage gap, not a correctness gap** — these routines run once
per `k_split` iteration alongside (not inside) the already-validated,
CUDA-replaced `dyn_core` acoustic loop; Gate C2's closure is unaffected.
Scoping the actual CUDA port (WP ownership, algorithmic structure,
correctness-validation plan) is a dedicated future prompt, not performed
here.

## 13. Stage 14 — Amdahl 2x-feasibility estimate

If Category B (64.66% of the window) were fully eliminated (moved to GPU at
zero incremental cost — an optimistic ceiling, not a promise):

```
remaining fraction = 1 - 0.6466 = 0.3534
theoretical max speedup = 1 / 0.3534 ≈ 2.83x
```

This **exceeds** the project's ≥2x C512 target (current baseline: CUDA at
parity with Fortran, 6373.86 ms/call; 2x target ≤3187 ms/call) under the
idealized full-elimination assumption. This is a substantially more
promising path than the halo-sync bucket alone (Prompt 49/50: ≤1.17x
ceiling, and shown not achievable). Caveats: (1) real GPU implementation
cost for `fv_mapz_mod`/`fv_tracer2d_mod` would not be zero — some fraction
of the eliminated CPU time would reappear as new GPU kernel time and
possibly new halo/sync overhead; (2) this is a single-rank estimate,
consistent across the 4 cross-checked ranks (§3) but not re-verified with a
live GPU-ported prototype; (3) this estimate should be treated as
"establishes feasibility is plausible," not as a committed performance
number for a not-yet-designed implementation.

## 14. Debugging note (SQLite query-construction bug, fixed during this analysis)

A single-statement version of the classification query (embedding the
`CASE`/`LIKE` category logic together with a correlated `EXISTS(SELECT 1
FROM covered ...)` subquery in one `CREATE TEMP TABLE ... AS SELECT`)
produced a spurious 780-sample "G_OPENMP" bucket attributed to two
unambiguously-Fortran symbols (`fvdycorecubed_gridcomp_mp_run_`,
`fvdycorecubed_gridcomp_mp_fillout3_`) that should have matched the
`B_FV3_FORTRAN` `WHEN` clause. Isolated by testing each `LIKE` predicate
individually against the literal symbol string (all correctly returned
false) and by reproducing correct classification with the same `CASE` logic
in a standalone query without the correlated join. Root cause not further
isolated (deliberately — out of scope, and a working alternative was found).
Fix: split into three separate temp tables (`catonly`: category only, no
covered-join in the same statement; `iscov`: covered-join only, no `CASE`
logic; a final join of the two) — reproduced clean, self-consistent,
verified-correct results (Category G confirmed genuinely zero, §9). This
is a data-integrity note for anyone reusing this SQL pattern, not a project
source-code finding.

## 15. Verification

- No source file modified. No numerical/algorithmic change. No optimization
  implemented.
- One GPU allocation used (job 58494926); no rerun needed.
- `ncu` not used. No hardware performance counters used. No CUDA-aware MPI,
  no multi-GPU work, no long science run, no new timing matrix.
- `docs/agent_handoff/CURRENT_STATE.md` not touched.
- No commit, no push.
- New untracked artifacts: this file; `prompt51_c512_cpusample.sbatch`
  (repo root); the experiment directory
  `GEOSfvdycore_fortran_perf/experiments/p51_c512_cpusample_cuda/` (separate
  repo, outside this repo's git tracking); scratch SQL/analysis files under
  `/tmp/` (not part of either repo, not committed).

---

## Final Report

- **Job**: 58494926, `gpu_a100`, RC=0. GPU allocations used: 1/1 budgeted
  (no rerun needed — first CPU-sampling capture was healthy).
- **Config**: C512 R8, 6 MPI ranks/1 A100, MPS ON, host-staged transport,
  fusion ON, Graph OFF, `RUN_HELD_SUAREZ=YES` — identical to Prompt 48/49's
  validated baseline.
- **Residual decomposed**: yes. 77.07% of the `DYN_CORE`-scoped window
  (representative rank; 77.7% per Prompt 49 on a different rank/run —
  cross-validated) is now broken into named categories summing to 99.79% of
  the residual (100% by partition construction), cross-validated two
  independent ways (gap-sample-fraction vs. residual-ns-fraction; per-
  category gap-time sum vs. interval-union residual, within 0.21%).
- **Dominant category**: **B. FV3 Fortran compute — 64.66% of the window,
  83.90% of the residual** — led by `fv_mapz_mod`'s vertical
  Lagrangian-to-Eulerian remapping (18.67% of the window alone, the single
  largest routine in the capture), plus `fv_tracer2d_mod` tracer transport,
  `fv_statemod` state conversion, and Held-Suarez forcing math (resolved
  via `libMAPL.griddedio.so`-exported `__bwr_*` symbols but confirmed by
  call-stack context to be genuine FV3 dynamics/physics compute, not MAPL
  I/O — §7). Secondary: Intel MPI progress-engine busy-poll (13.17% of
  residual, topology-dependent rank variance). CUDA driver/runtime blocking
  beyond the already-known halo sync: confirmed negligible (0.02–0.03% of
  residual). No OpenMP activity found. Unknown/unresolved: 0.58% of
  residual.
- **No outlier rank**: cross-checked 4 of 6 ranks; category proportions
  agree closely except MPI-wait share (topology-dependent, expected).
- **New finding vs. `CURRENT_STATE.md`**: this residual is dominated by a
  previously-unrecognized **CUDA coverage gap** — `dyn_core`'s
  large-timestep companions (vertical remap, tracer transport, state
  conversion), which run once per `k_split` iteration alongside the
  already-translated/validated `dyn_core` acoustic loop, were never in
  scope for the WP-A/B/C/D 54-stage inventory. This does not affect Gate
  C2's closure or numerical validation status.
- **2x feasibility**: Amdahl ceiling ≈2.83x if Category B were fully
  eliminated — feasible in principle, exceeds the ≥2x target, but requires
  a new, currently unscoped CUDA-porting effort (not evaluated for
  implementation cost/risk here).
- **Selected next target**: CUDA translation of `fv_mapz_mod`'s vertical
  remapping (highest single-routine cost) and `fv_tracer2d_mod`'s tracer
  transport — a scoping/design task for a future prompt, not implemented
  here.
- **Source changed**: NO. **Jobs submitted**: 1. **`ncu` used**: NO.
  **CUDA-aware MPI / multi-GPU work**: NO. **Opus escalation**: NO (SQL/
  measurement work, no ambiguous CUDA/MPI correctness semantics requiring
  escalation).

HOST-SIDE RESIDUAL DECOMPOSED — READY TO OPTIMIZE FV3 FORTRAN VERTICAL-REMAP/TRACER-TRANSPORT CUDA COVERAGE GAP (fv_mapz_mod, fv_tracer2d_mod)
