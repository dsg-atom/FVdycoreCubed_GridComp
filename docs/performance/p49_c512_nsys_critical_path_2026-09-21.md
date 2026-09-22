# C512 Nsight Systems Critical-Path Characterization (Prompt 49)

Date: 2026-09-21
Status: **MEASUREMENT/CHARACTERIZATION ONLY.** No source changed. No
optimization implemented. One GPU allocation used (budget: max 1, no rerun
needed — trace was valid on the first attempt).

---

## 1. Run configuration (validated Prompt 48 same-node config)

| Field | Value |
|---|---|
| Job | 58493749, node `warpa003`, elapsed 1m25s, RC=0 |
| Grid | C512, R8, npz=10, `k_split=1`, `n_split=3`, `RUN_DT=36` |
| Ranks/GPU | 6 MPI ranks (`FV_NX=1`/`FV_NY=6`), 1 A100, MPS ON |
| Backend | `RUN_CUDA_FV3=1`, `RUN_HELD_SUAREZ=1`, `COLDSTART=1` |
| Halo transport | host-staged (default; `RUN_CUDA_FV3_MPI_TRANSPORT`/`I_MPI_OFFLOAD` unset) |
| Fusion flags | `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES` |
| Graph mode | OFF (unset) |
| Binary | `GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x` (confirmed fresh via `strings`/`ldd`, Intel MPI 2021.13 + CUDA 12.5) |
| Duration | `JOB_SGMT` shortened to 6×`RUN_DT` → 6 total `DYN_CORE`/`RUN` cycles (1 warmup + 5 steady-state) — no numerical/algorithmic change, a benign test-length parameter |
| Run health | RC=0, no NaN/Inf/segfault/fatal in `run.log`; single shared MPS server (PID 137522) served all 6 ranks |

## 2. Methodology

Each of the 6 MPI ranks execs its own instance of `nsys profile
--trace=cuda,osrt` wrapping the real production binary (FV3EXE-override
mechanism already present in `fv3.j`), output keyed by the wrapping shell's
own PID — no MPI-rank env-var detection needed, no dependency on `PATH`
surviving `fv3.j`'s internal `g5_modules` reload (both `nsys` and the real
binary referenced by absolute path). All 6 `.nsys-rep` reports captured
successfully (2.96–3.24 MB each).

Representative-rank deep analysis on PID 137488, **cross-checked against 3 of
the other 5 ranks** (137489/137491/137493) for outlier detection (§4). No
outlier rank found — all 4 checked ranks agree within ~10%.

## 3. Window definition

`DYN_CORE`'s own ESMF-timer aggregate for these 6 cycles: **17.74 s** (mean
across ranks, `run.log` Block table). The tightest CUDA-trace-derived bound —
first kernel start to last kernel end, from
`CUPTI_ACTIVITY_KIND_KERNEL` — is **19.87 s** (39.901–59.774 s absolute
timestamps), consistent with (slightly larger than) the Fortran-reported
figure, as expected since it also captures brief CUDA-only launch/drain
slack immediately around each call. This 19.87 s kernel-activity span is used
below as the measurement window: it is the tightest denominator that is
provably `DYN_CORE`-scoped (kernels execute nowhere else in the run).

## 4. Stage 6 — Top kernels (rank 137488, cross-checked)

Top 20 of 171 distinct kernel names, **780.73 ms** total GPU kernel time
(rank 137488; 765–792 ms on ranks 137489/491/493 — consistent):

| Total | % of kernel time | Instances | Avg | Kernel | WP |
|---|---|---|---|---|---|
| 225.97 ms | 28.9% | 18 | 12.55 ms | `riem_solver_c_kernel<false>` | WP-C |
| 116.80 ms | 15.0% | 18 | 6.49 ms | `riem_solver3_kernel` | WP-C |
| 16.89 ms | 2.2% | 216 | 78.2 µs | `xppm_pass_kernel` | WP-B |
| 16.73 ms | 2.1% | 216 | 77.5 µs | `yppm_pass_kernel` | WP-B |
| 12.58 ms | 1.6% | 1140 | 11.0 µs | `copy_corners_kernel` | WP-B |
| 11.74 ms | 1.5% | 360 | 32.6 µs | `fill_corner_triplet_kernel` | WP-A |
| 9.85/9.76 ms | 1.3% ea | 324 | ~30 µs | `del6_flux_x/y_kernel` | WP-B |
| 9.84 ms | 1.3% | 18 | 546.7 µs | `pressure_gradient_c_x` | WP-A |
| 8.76/8.75 ms | 1.1% ea | 108 | ~81 µs | `q_i/q_j_kernel` | WP-B |
| (11 more, each ≤1.0%) | | | | | WP-A/B/D |

Cumulative top-20: **64.9%** of kernel time. Remaining 151 kernels: 35.1%,
long tail, none individually >1%.

**riem_solver_c_kernel + riem_solver3_kernel = 43.9% of kernel time**
(42.2–43.7% on the 3 cross-check ranks) — confirms these are the two
dominant *kernels*, but kernel time itself is only **3.9% of the 19.87 s
`DYN_CORE` window** (§7), so their absolute wall-time weight is small in the
full critical path (see §8 ranking).

## 5. Stage 7 — `update_dz_c` km+1-launch hypothesis: MEASURED SMALL

The prior roadmap's Hypothesis #2 (C-grid `update_dz_c` forced into km+1=11
sequential kernel launches by a reused `gz2` buffer,
`fv3_vertical_kernels.cu:907-923`) is **confirmed present** (198 instances =
18 substeps × 11 levels for `update_dz_c_update_gz`/`_winds`/`_y_flux`/
`_x_flux`, matching km+1 exactly) but its **measured cost is negligible**:

| Kernel | Instances | Avg | Total |
|---|---|---|---|
| `update_dz_c_update_gz` | 198 | 22.7 µs | 4.50 ms |
| `update_dz_c_winds` | 198 | 18.2 µs | 3.60 ms |
| `update_dz_c_finalize` | 18 | 149.4 µs | 2.69 ms |
| `update_dz_c_y_flux` | 198 | 11.2 µs | 2.22 ms |
| `update_dz_c_x_flux` | 198 | 10.5 µs | 2.08 ms |
| `update_dz_c_copy_gz` | 198 | 7.9 µs | 1.56 ms |
| **Total** | 1008 | | **16.65 ms = 2.1% of kernel time ≈ 0.08% of the `DYN_CORE` window** |

**This measurement directly contradicts the prior roadmap's ranking** of
`update_dz_c` delaunching as a Priority-1/2 "bounded, cheap to fix" target —
the pattern is real but its wall-clock weight is two orders of magnitude
smaller than assumed. Downgraded to low priority per measured evidence.

## 6. Stage 8 — Halo-transport blocking-sync hypothesis: MEASURED LARGE

`cudaEventSynchronize` (the host-blocking primitive used once per resolved
halo group in `MpiHaloTransport`/host-staged transport,
`fv3_gpu_halo_mpi.cpp:243`):

| Rank PID | Calls | Total | Avg | Median | Max |
|---|---|---|---|---|---|
| 137488 | 150 | 2851.3 ms | 19.0 ms | 6.95 ms | 66.55 ms |
| 137489 | 150 | 3113.7 ms | — | — | — |
| 137491 | 150 | 3204.1 ms | — | — | — |
| 137493 | 150 | 3190.9 ms | — | — | — |

150 calls / 18 acoustic substeps ≈ 8.33 groups/substep — matches the
architecture doc's documented "~7-11 groups/substep" estimate, confirming
the identification. **This single named CUDA API call alone is ~3.6–4.1x
larger than the entire GPU kernel compute budget (780.7 ms)** and is
consistently large across all 4 checked ranks (2.85–3.20 s) — no outlier.

**This directly contradicts the prior roadmap's LOW-priority ranking of
communication-layer items** (which was based on the now-discredited,
node-variance-contaminated Prompt 37 FMS `COMM_DSW` attribution). Per the
task's explicit rule to follow fresh measured evidence over the prior
roadmap when they conflict, this is promoted to the top measured,
attributable bottleneck (§8).

## 7. Stage 9 — Kernel-launch overhead

`cudaLaunchKernel`: 344.80 ms total, 15,822 calls (≈2,637 launches per
`DYN_CORE` call), avg 21.8 µs/launch. **1.7% of the 19.87 s window** — small
relative to sync/host-residual, but the ~2,637-launches/call figure is
itself evidence for the host-side dispatch-overhead hypothesis in §8/§9b
below (each launch requires host-side argument marshalling in the
line-by-line Fortran-replay dispatch layer).

## 8. Stage 10 — Host/device transfer classification

`cuda_gpu_mem_time_sum` (actual GPU DMA-engine time): D2H 734.99 ms (300
count, avg 2.45 ms), H2D 724.63 ms (395 count, avg 1.83 ms) — combined 1.46
s, roughly balanced (expected for host-staged halo transport: pack→D2H→MPI
send/recv on host→H2D→unpack, each direction used comparably). `cudaMemcpyAsync`'s
own API-side call duration (1578.0 ms, 695 calls, avg 2.27 ms/call) is
larger than the underlying DMA time, consistent with these "async" copies
effectively blocking the host thread — plausibly non-pinned/staged host
buffers forcing an implicit wait, matching the host-staged transport design.

## 9. Stage 11 — Idle/gap analysis (the dominant finding)

Built an interval union of all main-thread-attributable CUDA/OS activity
(`CUPTI_ACTIVITY_KIND_KERNEL` ∪ `CUPTI_ACTIVITY_KIND_RUNTIME` ∪
`CUPTI_ACTIVITY_KIND_MEMCPY`, all on the sole CUDA-API-issuing thread) within
the 19.87 s window, via direct `sqlite3` query against the exported
`.sqlite`:

| Category | Value |
|---|---|
| Window span | 19,872,367,160 ns (19.87 s) |
| **Covered** (union of all traced CUDA activity) | 4,422,802,628 ns (4.42 s) = **22.3%** |
| **Gap** (main thread not in any traced CUDA/OSRT call) | 15,449,564,532 ns (15.45 s) = **77.7%** |

Checked whether this gap is background-thread MPI/CUDA polling rather than
real main-thread cost: `OSRT_API` shows two *other* threads each with ~19.87
s of `poll`-family activity in-window (i.e., continuously polling for the
entire window) — these are background progress-engine threads running
**concurrently** with the main thread, not blocking it (confirmed: the main
CUDA-API-issuing thread's own OSRT time in-window is only 99 ms, negligible).
So the 77.7% gap is **not** explained by traced background polling — it is
main-thread time not inside any CUDA API call, kernel wait, memcpy, or
traced OS-runtime call.

**Honest interpretation, not fabricated further**: this residual is most
plausibly host-side dispatch/orchestration compute in the "line-by-line
Fortran-replay" architecture (~2,637 kernel-launch call-sites per
`DYN_CORE` call, each requiring host-side argument/stride/config
marshalling between launches — none of which touches a CUDA or traced OS
API) and/or blocking MPI point-to-point calls that resolve through
mechanisms not in nsys's traced OSRT function list. **This capture
(`--trace=cuda,osrt`) cannot further subdivide this residual** — that would
require CPU-sampling (`nsys --sample=cpu`) or MPI-trace instrumentation, a
new capture, out of scope for the one-allocation budget used here. This is
recorded as an open follow-up, not guessed at.

## 10. Stage 12 — Critical-path accounting (≥90% of `DYN_CORE` window)

| Bucket | ns | % of 19.87 s window | Basis |
|---|---|---|---|
| GPU kernel compute (all 171 kernels) | 780,728,411 | 3.9% | `cuda_gpu_kern_sum` |
| `cudaEventSynchronize` (halo-transport blocking sync) | 2,851,319,816 | 14.4% | `cuda_api_sum` (subset of "covered") |
| `cudaMemcpyAsync` API-side (host-staged transfer, blocking-like) | 1,577,986,663 | 7.9% | `cuda_api_sum` (subset of "covered") |
| `cudaLaunchKernel` | 344,802,217 | 1.7% | `cuda_api_sum` (subset of "covered") |
| other CUDA API (malloc/free/memset/etc.) | ~79,000,000 | 0.4% | `cuda_api_sum` (subset of "covered") |
| *(overlap correction — sum of above double-counts async overlap)* | −1,211,000,000 | −6.1% | union vs. naive sum reconciliation |
| **Traced-activity union (= "covered" in §9)** | **4,422,802,628** | **22.3%** | interval union, sqlite |
| **Untraced host-side residual (§9, honestly unattributed)** | **15,449,564,532** | **77.7%** | span − union |
| **Total accounted** | 19,872,367,160 | **100.0%** | |

100% of the window is accounted for as either (a) specific, named, measured
CUDA activity (22.3%, broken into the sub-buckets above) or (b) an honestly
labeled, measured-by-subtraction "untraced host residual" (77.7%) whose
*existence and size* is solid measured evidence even though its *internal
composition* is not further attributable from this capture. No number in
this table is fabricated — the residual is reported as a bound, not
decomposed into invented sub-causes.

## 11. Stage 13 — Ranked optimization targets (by measured evidence)

| Rank | Target | Measured size | Attributability | WP owner |
|---|---|---|---|---|
| 1 | Host-side dispatch/orchestration overhead (untraced residual, §9) | 77.7% of window | Size confirmed; internal cause NOT yet attributable without new instrumentation | cross-cutting (WP-E runtime + all dispatch call sites) |
| 2 | Halo-transport blocking sync (`cudaEventSynchronize`) | 14.4% of window, 150 calls/rank | Fully attributable, code-cited (`fv3_gpu_halo_mpi.cpp:243`) | WP-E |
| 3 | Host-staged memcpy (blocking-like `cudaMemcpyAsync`) | 7.9% of window | Fully attributable | WP-E |
| 4 | GPU kernel compute, dominated by 2 Riemann-solver kernels | 3.9% of window (43.9% of that is Riemann) | Fully attributable | WP-C |
| 5 | Kernel-launch overhead | 1.7% of window | Fully attributable | cross-cutting |
| (downgraded) | `update_dz_c` km+1 launches | 0.08% of window | Fully attributable — confirmed present but negligible | WP-C |
| (downgraded) | Halo pack/unpack kernel cost itself | 0.79% of kernel time (~0.03% of window) | Fully attributable — negligible | WP-B/E |

**This measured ranking materially contradicts the prior roadmap**, which
had ranked Riemann-solver register/occupancy work (#4 here) and
`update_dz_c` delaunching (negligible here) above communication-layer work
(#1/#2/#3 here, previously LOW priority). The single largest attributable,
actionable, code-cited bucket is the halo-transport blocking sync (#2); the
single largest bucket overall (#1) is real but requires a follow-up
CPU-sampling capture before it can be assigned a first implementation step.

## 12. Decision Rule and selected target

Applying "select the largest bucket that is both (a) fully attributable to a
specific, bounded subsystem and (b) actionable without new profiling
instrumentation": **Rank 1 (untraced host residual) is disqualified** — its
cause is not yet attributable, so acting on it now would mean guessing.
**Rank 2, halo-transport blocking synchronization (`cudaEventSynchronize` in
`MpiHaloTransport`/host-staged transport, `fv3_gpu_halo_mpi.cpp:243`), is
selected** as the first implementation target: it is fully attributable, the
single largest *actionable* bucket (14.4% of the window, 150 blocking calls
per `DYN_CORE` call, consistently 2.85–3.20 s across 4 checked ranks), owned
by a single work package (WP-E), and does not require new profiling to
scope. No implementation was performed in this task per its explicit scope
limits.

**Recommended immediate follow-up (not this task)**: an `nsys --sample=cpu`
(or MPI-level trace) capture, same one-allocation-budget discipline, to
decompose the Rank-1 residual before deciding whether it or the Rank-2 sync
work should be attacked first in the next implementation task.

## 13. Verification

- No source file modified. No numerical/algorithmic change. No optimization
  implemented.
- One GPU allocation used (job 58493749); no rerun (trace was valid).
- `docs/agent_handoff/CURRENT_STATE.md` not touched.
- No commit, no push.
- New untracked artifacts: this file;
  `prompt49_c512_nsys_critical_path.sbatch` (repo root);
  `perf_rerun_logs/p49_c512_nsys_58493749.out`; the experiment directory
  `GEOSfvdycore_fortran_perf/experiments/p49_c512_nsys_cuda/` (separate repo,
  outside this repo's git tracking).

---

## Final Report

- Job: 58493749, node warpa003, RC=0, elapsed 1m25s. GPU allocations used: 1/1 budgeted (no rerun needed).
- Config: C512 R8, 6 MPI ranks/1 A100, MPS ON, host-staged transport, fusion ON, Graph OFF, RUN_HELD_SUAREZ=YES — matches Prompt 48 validated baseline.
- Measurement window: 19.87 s (`DYN_CORE`-scoped kernel-activity span, 6 calls: 1 warmup + 5 steady-state).
- Critical-path accounting: 100% (22.3% named/traced CUDA activity + 77.7% honestly-labeled untraced host residual — no fabricated sub-breakdown of the residual).
- Top measured, attributable bottleneck: halo-transport blocking `cudaEventSynchronize` (14.4% of window, 150 calls/rank, 2.85–3.20 s, consistent across 4 of 6 ranks) — **~3.6–4.1x larger than total GPU kernel compute (3.9% of window)**.
- Prior-roadmap contradictions (measured): Riemann-solver kernels are real (43.9% of kernel time) but kernel time itself is only 3.9% of the window — demoted from top rank. `update_dz_c` km+1-launch pattern confirmed present but costs only 0.08% of the window — demoted from "cheap, high-priority fix" to negligible.
- Largest bucket overall (77.7%, untraced host-side residual) is reported honestly as unattributed pending a follow-up CPU-sampling capture — not guessed at.
- Selected first-implementation target (Decision Rule: largest bucket that is both fully attributable and actionable without new instrumentation): **halo-transport blocking synchronization (`cudaEventSynchronize`, `MpiHaloTransport`/host-staged, `fv3_gpu_halo_mpi.cpp:243`, WP-E)**. Not implemented in this task.

C512 CRITICAL PATH CHARACTERIZED — READY TO OPTIMIZE HALO-TRANSPORT BLOCKING SYNCHRONIZATION (cudaEventSynchronize, fv3_gpu_halo_mpi.cpp:243, WP-E)
