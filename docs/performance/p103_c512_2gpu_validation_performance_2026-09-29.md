# Prompt 103 — C512 2-GPU Validation and Performance (2026-09-29)

## 0. Status

**C512 2-GPU VALIDATION COMPLETE — MULTI-GPU PERFORMANCE POSITIVE.**

Session was interrupted by a connection loss mid-setup and resumed under an
explicit recovery protocol (git/Slurm/filesystem evidence only, no reliance
on prior-turn memory). Recovery found 9 rundirs already staged (binaries
copied, `fv3.j` templates written) but **no job had been submitted yet** —
i.e. Stage-0 rundir setup was done, job execution was not. A stale-`EXPDIR`
defect (all 9 `fv3.j` pointed at an unrelated prior rundir, `p101a_pair1_B`)
was found and fixed via `sed -i` before submission — this exact pitfall is
documented in `CURRENT_STATE.md` §8 ("grep any copied `fv3.j` for `setenv
EXPDIR` before reuse"). Job `58642794` was then submitted and completed
successfully; this document reports that job's results.

## 1. Configuration

- C512, R8, 6 MPI ranks (`FV_NX=1`/`FV_NY=6`), Intel MPI 2021.13, Held-Suarez
  forcing, `COLDSTART=1`.
- Device mapping: existing, unmodified `dev = mod(comm_rank, device_count)`
  (`fv3_gpu_dyncore_bridge.F90:164`) — **no source change**.
- 1-GPU arm: `--gres=gpu:2` allocation, `CUDA_VISIBLE_DEVICES=0` (restricts
  visibility to device 0, giving 6 ranks/1 GPU inside the same allocation as
  the 2-GPU arm).
- 2-GPU arm: same allocation, `CUDA_VISIBLE_DEVICES` unset → 3 ranks/GPU
  automatic split.
- Both arms: current production CUDA defaults (PT/DELZ, REMAP_UV, REMAP_Q,
  REMAP_PRESSUREVARS = CUDA; TRACER_2D, State_To_FV, REMAP_TOTALENERGYCON =
  Fortran), `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`,
  `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES` (established best config), Graph OFF,
  MPS ON.
- Binary provenance: all 9 runs verified sha256
  `2339366b10885c4532f08efa089d4e3b0a062403397f0c64c6a396b186e2064d`
  (matches the P101A production-default binary) before execution.
- Job `58642794`, host `warpa008`, `2026-09-29T13:55:21`→`14:13:37`, exit
  `0:0`. All 9 runs: rc=0, 30/30 `DYN Date`, **zero** FATAL/OOM/Abort/NaN/Inf
  matches.

## 2. Correctness (1-GPU vs 2-GPU device split)

Compared via `tools/analysis/hs_netcdf_inspect.py compare-backends` between
`p103_correctness_1gpu` and `p103_correctness_2gpu` (identical IC/config,
GPU count/placement only difference), at the first HS HISTORY.rc checkpoint
(`18910301_0010z.nc4`):

| Field | Level | max_abs | rms | max_rel (guarded) | NaN either side |
|---|---|---|---|---|---|
| T | 500 hPa | 0.0 | 0.0 | 0.0 | 0/131584 |
| U | 250 hPa | 0.0 | 0.0 | 0.0 | 0/131584 |
| V | 250 hPa | 0.0 | 0.0 | 0.0 | 0/131584 |
| PS | surface | 0.0 | 0.0 | 0.0 | 0/131584 |

**Classification: BITWISE EXACT.** GPU count/placement (1-GPU/6-ranks vs.
2-GPU/3-ranks-per-GPU) produces zero difference in any compared field — as
expected, since device topology is a resource-mapping decision, not a
numerical-algorithm change. This also rules out MPS-sharing or
device-index-dependent nondeterminism as a source of divergence at this
density.

## 3. Memory

Per-GPU memory measured via a 5s-interval background `nvidia-smi` poller
during the 2-GPU correctness run (`P103_CORR_poll.log`, 90 samples):

| Phase | GPU0 used/free | GPU1 used/free |
|---|---|---|
| Idle (t=1-7) | 44 MiB / 40295 MiB | 43 MiB / 40296 MiB |
| Ramp (t=8) | 11003 MiB / 29335 MiB | 11003 MiB / 29335 MiB |
| **Converged (t=9-90, 82/90 samples)** | **20153 MiB / 20185 MiB** | **20153 MiB / 20185 MiB** |

Converged density: **19.68 GiB used/GPU** (20153/1024), identical across
both physical GPUs. Compare to Prompt 102's projected 20.3-21.3 GiB/GPU for
3-ranks-per-GPU: **measured value is close to, slightly below, the
projection** — consistent, not flagged. **Neither GPU exceeded 30 GiB** (the
mandated flag threshold). Free headroom at converged state: **19.7 GiB/GPU**.

## 4. Performance (1-GPU vs 2-GPU, same allocation)

3 alternating-order pairs (A=1gpu, B=2gpu: A1 B1 / B2 A2 / A3 B3), `ms/call`
from `fv_timing_mod`, 30 calls each:

| Timer | 1gpu (avg of 3) | 2gpu (avg of 3) | Speedup S=T1/T2 | % improvement |
|---|---|---|---|---|
| TOTAL | 122.90 | 109.75 | 1.120 | 10.70% |
| FV_DYN_LOOP | 46.47 | 33.26 | 1.397 | 28.42% |
| DYN_CORE | 25.52 | 14.40 | 1.773 | 43.58% |
| TRACER_2D (Fortran) | 11.14 | 9.43 | 1.182 | 15.38% |
| REMAPPING | 11.78 | 10.77 | 1.094 | 8.60% |
| REMAP_TOTALENERGYCON | 4.22 | 3.41 | 1.240 | 19.32% |
| COMM_TOTAL | 1.012 | 1.016 | 0.996 | -0.34% (flat, within noise) |

Per-pair TOTAL: pair1 122.59→110.09 (10.20%), pair2 123.29→109.60 (11.10%),
pair3 122.80→109.54 (10.80%) — **consistent 3/3, low pair-to-pair variance
(±0.5pp)**.

**Interpretation**: DYN_CORE and FV_DYN_LOOP show the largest gains —
expected, since halving ranks-per-GPU (6→3) directly reduces per-GPU
device-side contention for the GPU-resident work. TRACER_2D and
REMAP_TOTALENERGYCON, both host/Fortran-resident, also improved
measurably — plausibly reduced host-side/MPS-scheduling contention
side-effects rather than a direct GPU-compute effect. COMM_TOTAL is flat as
expected: host-staged MPI across the same 6 ranks is unaffected by GPU
device-count.

**Indicative GPU-scaling efficiency** E = S/2 = 1.120/2 = **0.560 (56.0%)**
for TOTAL — indicative only, since rank count (6) and MPI/host work are
fixed; this is not a clean strong-scaling efficiency metric.

### Combination with Prompt 101A (DERIVED CONTEXT)

Prompt 101A measured current-CUDA-defaults-vs-all-Fortran at **1-GPU**:
TOTAL 1.153x. This Prompt 103 result measures current-CUDA-defaults at
**1-GPU vs 2-GPU**: TOTAL 1.120x. These are orthogonal comparisons (different
independent variable) — **not directly multiplicative or verified in
combination**. As **DERIVED CONTEXT ONLY** (not directly measured): a naive
compositional estimate of CUDA-2GPU vs. all-Fortran-1GPU would be
≈1.153×1.120 ≈ **1.292x**. This combined figure has not been independently
measured and should not be cited as a validated result without a direct A/B.

## 5. State_To_FV CUDA Opt-In Smoke (2-GPU)

`p103_state_to_fv_smoke`: `RUN_CUDA_FV3_STATE_TO_FV_ENABLE=1`, 2-GPU (3
ranks/GPU), same binary/provenance. Result: rc=0, 30/30 `DYN Date`, zero
FATAL/OOM/NaN/Inf.

`State_To_FV` timer for this run: **1.62-2.40 ms/call** (30 calls) — notably
*lower* than the Fortran-default `State_To_FV` timing observed in all 8
other runs in this same job (which ranged 2.30-3.31 ms/call, consistent with
the Fortran path). This timing differential is **suggestive that the CUDA
State_To_FV path was actually dispatched and is faster**, but no explicit
"CUDA path taken" log line was captured — this is **timing-based evidence,
not a direct dispatch-confirmation log message**. No performance claim is
made from this single smoke run (opt-in correctness/stability check only,
per the original task scope).

**Memory feasibility**: converged 2-GPU free headroom is **19.7 GiB/GPU**
(§3), vastly exceeding both the required ≥512 MiB and preferred ≥1 GiB
safety margins, and the ~440 MiB shortfall identified at 6-ranks/1-GPU
density (P98-100). **State_To_FV CUDA is CONFIRMED memory-feasible at
2-GPU/3-ranks-per-GPU density** — this resolves Prompt 102's "feasible at
≥2 GPUs, not yet measured" projection with a direct measurement.

## 6. Classification Summary

- Correctness: **BITWISE EXACT** (T/U/V/PS, 0 diff)
- Memory: **19.68 GiB/GPU converged, not flagged** (<30 GiB; close to P102's
  20.3-21.3 GiB projection)
- Performance: **POSITIVE, consistent 3/3 pairs** (TOTAL 1.120x/10.70%;
  FV_DYN_LOOP 1.397x; DYN_CORE 1.773x)
- Dispatch: no unexpected Fortran fallback observed in any of the 9 runs
  (all production-default selectors behaved as configured)
- State_To_FV CUDA: **memory-feasible at this density (confirmed)**;
  dispatch plausible from timing, not log-confirmed
- **Recommended C512 mode: 2-GPU/3-ranks-per-GPU** — strictly better than
  1-GPU/6-ranks on both memory (19.68 vs. saturated ~40 GiB/GPU) and
  performance (+10.70% TOTAL), zero correctness cost, zero source change
  required.

**Final status line: C512 2-GPU VALIDATION COMPLETE — MULTI-GPU PERFORMANCE
POSITIVE.**

## 7. Next Task

**Selected: Option A — enable/validate State_To_FV CUDA on 2 GPUs.**
Rationale: the P98-100 port was correctness-validated (GPU EXECUTED,
C32-correct) but memory-blocked at 6-ranks/1-GPU; this prompt directly
confirms 19.7 GiB/GPU free headroom at 2-GPU/3-ranks-per-GPU — more than 40x
the ~440 MiB shortfall. The smoke run here executed cleanly with plausible
CUDA dispatch (faster State_To_FV timing than the Fortran baseline in the
same job). A dedicated follow-up should: (a) confirm CUDA dispatch
explicitly (not just via timing) — e.g. a debug log line or `nsys` trace at
this density; (b) run a correctness pair (State_To_FV CUDA vs Fortran,
2-GPU, same HISTORY.rc comparison as §2); (c) run a proper A/B performance
study (paired, alternating order, ≥3 pairs) to quantify the
State_To_FV-specific speedup at 2-GPU density, analogous to P89/P93's
methodology for REMAP_Q/PRESSUREVARS.

No CUDA/Fortran source changed, no production defaults changed, no MPI
changed, no commit/push — this prompt was test-scripts/logging/docs only.

**Update 2026-09-29 (Prompt 104):** executed Option A. CUDA dispatch of
`state_to_fv_phase_a`/`_c` CONFIRMED at this density (via branch-selection +
ABI-success + device-memory-footprint + timing-signature evidence, not a
raw counter — the planned counter instrumentation was never implemented).
Direct `State_To_FV` speedup **1.633x mean (38.75% reduction, 3/3 pairs,
stddev 0.006x)** — highly reproducible. End-to-end TOTAL **+1.01% mean**
(MEANINGFUL, boundary), FV_DYN_LOOP +0.60% (SMALL POSITIVE), DYN_CORE flat
as architecturally expected (`State_To_FV` is outside the `k_split` loop).
Memory: **+440.4 MiB/GPU** measured (vs. P100's +452.97 MiB/GPU
prediction — close match), 19.3 GiB/GPU free at converged state — SAFE.
**Correctness gap found**: no field-level HISTORY/`.nc4` comparison was
possible (the short harness wrote no HISTORY output) — only clean-run
(no FATAL/NaN) evidence exists at this density; the C32 bitwise validation
from P98-99 is unchanged but does not cover C512/2-GPU. **Decision: KEEP
OPT-IN** pending a follow-up correctness pair with `HISTORY.rc` configured.
No source/production-default/MPI change; no commit/push. See
`docs/performance/p104_state_to_fv_2gpu_validation_performance_2026-09-29.md`.
