# Prompt 93 — REMAP_PRESSUREVARS Performance Robustness and Default Policy

Branch: `optimization/cuda-fv3-performance`. Scope: bounded
performance-confirmation + production-policy task. Determine whether the
Prompt 92 CUDA `REMAP_PRESSUREVARS` result is robust enough to enable by
default in production. **No REMAP_PRESSUREVARS numerics, CUDA kernel, or
transfer architecture were modified.** PT/DELZ, REMAP_UV, REMAP_Q, tracer
CUDA, `REMAP_TOTALENERGYCON`, and R4 were not touched. No Nsight Compute
run. No next-CUDA-port work started.

## 1. Stage 0 — Provenance

- `git status --short` on this repo (`FVdycoreCubed_GridComp`,
  `optimization/cuda-fv3-performance`, HEAD `df17dba3`): unchanged from
  session start — no unexpected drift.
- Prompt-92 executable
  (`GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x`):
  sha256 `3d4364e6f64839918802ceb1f6085d72217faf8660519da39dd953b716e00002`
  — **exact match** to the value recorded at the end of Prompt 92. Stage
  0's STOP condition (unexpected provenance drift) did **not** trigger.
- Confirmed Intel MPI 2021.13 linkage on both the pre-existing binary
  (`BASEDIR` = `ifort_2021.13.0-intelmpi_2021.13.0`) and, after rebuild
  (§5), directly via `ldd` (`libmpi.so.12`/`libmpifort.so.12` from
  `/usr/local/intel/oneapi/2024/mpi/2021.13/`).
- Confirmed current runtime policy at task start: PT/DELZ=CUDA default,
  REMAP_UV=CUDA default, REMAP_Q=CUDA default, REMAP_PRESSUREVARS=Fortran
  default (opt-in via `RUN_CUDA_FV3_PRESSUREVARS_ENABLE`), TRACER=Fortran
  default — matches `CURRENT_STATE.md` exactly (`fv_mapz.F90:567`,
  pre-edit: `use_cuda_remap_pressurevars = .false.`).

## 2. Stage 1 — Existing 5-Pair Data (Prompt 92, not re-run)

Recovered from `p92_pair{1..5}_{A,B}/p92_result.log` (raw accumulated
seconds over 30 calls/arm; ms/call = value×1000/30) and cross-checked
against Prompt 92's own reported aggregate (exact match).

| Pair | Order | Fortran TOTAL (ms/call) | CUDA TOTAL (ms/call) | TOTAL improvement | REMAP_PRESSUREVARS speedup | REMAPPING improvement | FV_DYN_LOOP improvement |
|---|---|---|---|---|---|---|---|
| 1 | A1 B1 | 4472.42 | 4403.97 | 1.531% | 3.452x | 10.363% | 3.663% |
| 2 | B2 A2 | 4490.54 | 4403.97 | 1.928% | 3.633x | 17.117% | 5.222% |
| 3 | A3 B3 | 4461.85 | 4390.94 | 1.589% | 3.535x | 16.809% | 4.777% |
| 4 | B4 A4 | 4501.14 | 4427.66 | 1.633% | 3.470x | 16.229% | 4.435% |
| 5 | A5 B5 | 4492.19 | 4386.94 | 2.343% | 3.399x | 17.994% | 6.089% |

Summary statistics (n=5):

- CUDA TOTAL wins: **5/5**
- TOTAL improvement: mean **1.805%**, median **1.633%**, sample stddev
  **0.337%** (population stddev 0.302%), min **1.531%**, max **2.343%**
- REMAP_PRESSUREVARS direct speedup: mean **3.498x**, median 3.470x, min
  3.399x, max 3.633x
- Provenance: all 10 arms rc=0, all 10 arms `count=30/30` `DYN Date`
  completions, all 10 provenance lines sha256-matched
  (`3d4364e6...e0002`), zero `FATAL`/`out of memory`/`Abort`/
  `BINARY_SHA_MISMATCH`/`MPS_START_FAILED` matches anywhere in the
  420KB sbatch stdout (`p92_pressurevars_ab5_58617749.out`).
- Order effect: alternating order (A1 B1 / B2 A2 / A3 B3 / B4 A4 / A5 B5)
  shows no correlation between "which arm ran first" and improvement
  magnitude — the highest (pair 5, A-first) and lowest (pair 1, A-first)
  improvements both occur under the same ordering; pair 2 (B-first) is
  second-highest. No dominant order effect.

## 3. Stage 2 — Default Candidate Gate

All 7 required conditions evaluated against §2's data:

1. Correctness PASS — yes (Prompt 92 §2: 10/10 fixtures, sanitizer 0
   errors, C32 dispatch verified both directions).
2. CUDA wins TOTAL in ≥4/5 pairs — **5/5**. PASS.
3. Median TOTAL improvement ≥1.0% — **1.633%**. PASS.
4. Mean TOTAL improvement ≥1.0% — **1.805%**. PASS.
5. REMAP_PRESSUREVARS wins clearly in every valid pair — all 5 pairs
   ≥3.3x speedup. PASS.
6. No memory instability/OOM/runtime errors — confirmed zero matches.
   PASS.
7. No abnormal node-order effect dominates — confirmed above. PASS.

**Stage 2 result: DEFAULT CANDIDATE.**

## 4. Stage 3 — Independent Longer-Run Confirmation

One longer same-node paired A/B (60 calls, `JOB_SGMT` doubled to
`00:36:00`, following the exact Prompt 89 Stage 6 longer-run precedent),
single Slurm allocation, job **58618735**, host `warpa009`. Arm A =
Fortran REMAP_PRESSUREVARS (`RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN=1`,
the then-current production default); Arm B = CUDA REMAP_PRESSUREVARS
(`RUN_CUDA_FV3_PRESSUREVARS_ENABLE=1`). All other paths at production
defaults: PT/DELZ CUDA, REMAP_UV CUDA, REMAP_Q CUDA,
`FUSE_HALO_EDGES`/`FUSE_CORNER_FILL=YES`, Graph OFF, TRACER Fortran.
Same binary as §2 (sha256 `3d4364e6...e0002`, verified per arm).

Both arms: rc=0, 60/60 `DYN Date` completions, sha256-verified, zero
`FATAL`/`out of memory`/`Abort`/`NaN`/`Infinity` matches.

| Timer | Fortran (Arm A, ms/call) | CUDA (Arm B, ms/call) | Δ |
|---|---|---|---|
| REMAP_PRESSUREVARS | 131.95 | 41.31 | **3.194x speedup / 68.70% reduction** |
| REMAPPING | 624.50 | 556.51 | **+10.887%** |
| FV_DYN_LOOP | 1825.52 | 1750.94 | **+4.086%** |
| DYN_CORE | 848.02 | 848.72 | unchanged (+0.08%, no cross-contamination) |
| TOTAL | 4327.04 | 4254.50 | **+1.676%** |

Result is directionally and magnitude-consistent with the 5-pair set in
§2 (TOTAL improvement 1.676% falls within the 5-pair range 1.531–2.343%;
direct speedup 3.194x is close to, slightly below, the 5-pair mean
3.498x — same order, same sign, no reversal).

## 5. Stage 4 — Longer-Run Gate

REMAP_PRESSUREVARS clearly CUDA-positive (3.194x): PASS. REMAPPING
positive (+10.887%): PASS. FV_DYN_LOOP positive (+4.086%): PASS. TOTAL
improvement 1.676% ≥ 1.0% threshold.

**Stage 4 result: DEFAULT CONFIRMED.** (No threshold tuning performed
after seeing this result — the ≥1.0%/0.5–1.0%/<0.5% bands were fixed
before Stage 3 was run.)

## 6. Stage 5 — Policy Change

Since Stage 2 = DEFAULT CANDIDATE and Stage 4 = DEFAULT CONFIRMED, the
smallest production-policy change was made in
`GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/@fvdycore/model/fv_mapz.F90`
(lines ~555–573), mirroring the exact REMAP_Q P88B→P89 selector-flip
precedent (`use_cuda_remap_q`, lines ~538–553 of the same file):

- `use_cuda_remap_pressurevars` is now assigned `p92_base_gate` directly
  (was `.false.` unconditionally).
- `RUN_CUDA_FV3_PRESSUREVARS_ENABLE` is retained but is now a no-op when
  the base gate already holds (kept for backward compatibility with
  existing harness scripts).
- `RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN` is preserved unchanged as the
  explicit escape hatch back to Fortran.
- The base gate itself (`p92_base_gate`, restricting to `.not.
  hydrostatic .and. remap_t .and. .not. remap_pt .and. .not. remap_te
  .and. .not. do_omega .and. nwat == 0 .and. kind(q) == kind(1.0d0) .and.
  fv3_gpu_backend_enabled()`) is **unchanged** — unsupported
  configurations still fall back to Fortran automatically.
- **No numerical code was modified.** Only the selector's default-value
  assignment changed.

## 7. Stage 6 — Rebuild

Rebuilt under the canonical Discover production module stack:

```
module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv
module load comp/gcc/12.3.0
module load comp/intel/2024.2.0
module load mpi/impi/2021.13
module load nvidia/nvhpc-byo-compiler/24.7
export I_MPI_CXX=g++
```

`make -j8 fvdycore` recompiled `fv_mapz.F90` and relinked
`libfvdycore.a` clean. `make -j8 install` (whole-target) hit an
unrelated, pre-existing, out-of-scope failure building
`MAPL_cfio_r4`'s `ShaveMantissa_` Python extension module (a Python
version/build-environment issue, orthogonal to `fv_mapz.F90`) — this
target is not a dependency of `StandAlone_FV3_Dycore.x`. Building the
exact executable target directly instead
(`make -j8 StandAlone_FV3_Dycore.x`) succeeded cleanly (only benign
pre-existing `libcrypto.so.1.1`/`libcrypto.so.3` linker warnings,
identical class of warning seen in prior prompts' rebuilds).

- Executable:
  `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x`
- mtime: `Mon 28 Sep 2026 11:01:07 AM EDT`
- sha256: `b482b64703f0db7652b41310131815bef48f610d4b8ed208e7efef5d89b3ec96`
  (changed from Prompt 92's `3d4364e6...e0002`, as expected — one source
  file recompiled).
- MPI linkage verified via `ldd`: `libmpi.so.12`, `libmpifort.so.12`,
  `libmpicxx.so.12` all resolve to
  `/usr/local/intel/oneapi/2024/mpi/2021.13/lib/` — Intel MPI 2021.13
  confirmed, no OpenMPI substitution.

**Rebuild: PASS.**

## 8. Stage 7 — C32 Selector Smoke

Real Held-Suarez cold-start C32/L10 run (`RUN_DT=1200`, 6 ranks, single
A100), same template family as Prompt 92's own C32 gate
(`p92_c32_cuda`/`p92_c32_fortran`), rebuilt binary copied into two new
run directories (`p93_c32_default`, `p93_c32_forcefortran`).

| Arm | Env override | Job | Status | CUDA dispatch | Fortran dispatch | `DYN Date` count | NaN/Abort/FATAL |
|---|---|---|---|---|---|---|---|
| A — DEFAULT | none (no `_ENABLE`, no `_FORCE_FORTRAN`) | 58619031 | 0 | 36/36 | 0/36 | 6/6 | 0 |
| B — escape hatch | `RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN=1` | 58619032 | 0 | 0/36 | 36/36 | 6/6 | 0 |

Both arms `rc=0` (`Status = 0`), 0 NaN/Inf/Abort/FATAL, no divergence in
visible stdout. Dispatch counts (36 = 6 calls × 6 ranks, matching Prompt
92's own C32 convention exactly) confirm: with **no environment override
at all**, the rebuilt binary now dispatches to CUDA by default, and the
explicit `FORCE_FORTRAN` escape hatch still correctly forces 100% Fortran
dispatch.

**Stage 7: PASS** (both arms). No C512 rerun performed — per Stage 7's
own instruction, none is required after a selector-only default flip
when the rebuilt binary's CUDA numerical objects are unchanged (confirmed
by Stage 6: only `fv_mapz.F90`'s Fortran object recompiled; no `.cu`/
`.cuh`/`.cpp` under `cuda/` touched this prompt) and this smoke passes.

## 9. Stage 8 — Final Production Policy

All required gates passed (Stage 2 = DEFAULT CANDIDATE, Stage 4 =
DEFAULT CONFIRMED, Stage 6 = rebuild PASS, Stage 7 = both smoke arms
PASS). Per Stage 8's "no partial policy changes" rule, the full policy
flip stands:

**REMAP_PRESSUREVARS is now a CUDA production default** whenever its
existing base-configuration gate holds (non-hydrostatic, `remap_t`, not
`remap_pt`/`remap_te`, not `do_omega`, `nwat==0`, R8, CUDA backend
enabled) — the same production first-run configuration already assumed
by every other CUDA-default remap path (PT/DELZ, REMAP_UV, REMAP_Q).
Escape hatch: `RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN=1`.
`RUN_CUDA_FV3_PRESSUREVARS_ENABLE` remains accepted but is now a no-op
for backward compatibility with existing harness scripts. Unsupported
configurations (hydrostatic, `remap_pt`/`remap_te`/`do_omega` true,
non-zero `nwat`, R4) automatically fall back to Fortran via the
unchanged base gate — no explicit env var is required to get correct
behavior in those cases.

## 10. Stage 9 — Profile Impact Note

Based on the validated §4 longer run (the more conservative of the two
confirmed data points), the expected default-on production impact at
C512 scale is: TOTAL **≈+1.7%** faster, FV_DYN_LOOP **≈+4.1%** faster,
REMAPPING **≈+10.9%** faster, REMAP_PRESSUREVARS itself **≈3.2x**
faster, DYN_CORE unaffected. The 5-pair set in §2 suggests this may be a
conservative floor (mean TOTAL improvement there was 1.805%, direct
speedup 3.498x) — true production impact likely falls in the 1.7–1.9%
TOTAL-improvement band already observed across both datasets.

Per Stage 9's explicit instruction, **no new full Prompt-90-style
production hotspot profile was run this prompt** — that refresh is
deferred to a future task, to be run only after this policy decision is
closed and before selecting the next CUDA optimization target
(`REMAP_TOTALENERGYCON` remains the not-yet-selected, higher-risk
alternative per Prompt 90/92).

## 11. Summary

```
Existing paired runs: 5
CUDA TOTAL wins: 5/5
TOTAL mean improvement: 1.805%
TOTAL median improvement: 1.633%
TOTAL stddev: 0.337%
Direct REMAP_PRESSUREVARS speedup: 3.498x
Stage 2: DEFAULT CANDIDATE
Longer-run job: 58618735
Longer-run REMAP_PRESSUREVARS: Fortran 131.95 ms/call CUDA 41.31 ms/call Improvement 68.70% Speedup 3.194x
Longer-run REMAPPING improvement: 10.887%
Longer-run FV_DYN_LOOP improvement: 4.086%
Longer-run TOTAL improvement: 1.676%
Stage 4: DEFAULT CONFIRMED
Default-policy source change: YES
Rebuild: PASS
Executable: /discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x
sha256: b482b64703f0db7652b41310131815bef48f610d4b8ed208e7efef5d89b3ec96
Default C32 arm: CUDA dispatch 36, Fallback 0, PASS
FORCE_FORTRAN C32 arm: CUDA dispatch 0, Fortran count 36, PASS
Final production default: CUDA
No numerical code changed: YES
No commit/push: YES
```

```
REMAP_PRESSUREVARS PERFORMANCE ROBUST — CUDA ENABLED AS PRODUCTION DEFAULT
```
