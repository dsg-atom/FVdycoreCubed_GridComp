# Prompt 106 — State_To_FV CUDA Default-Enable Policy (2026-09-29)

## 0. Task

Prompts 104-105 closed all three validation-hierarchy legs for the
`State_To_FV` CUDA port (GPU EXECUTED, NUMERICALLY VALIDATED — T/PS
bitwise, U/V round-off clean, PERFORMANCE VALIDATED — 1.633x direct,
+1.01% TOTAL) at exactly one density: C512, single node, 2 A100 GPUs,
3 ranks/GPU, MPS ON, R8. The default was left at Fortran pending a
dedicated enablement task. This is that task.

Goal: make CUDA the *default* for `State_To_FV` only at the validated
density, while (a) preserving `RUN_CUDA_FV3_STATE_TO_FV_FORCE_FORTRAN=1`
as an unconditional escape hatch, (b) preserving
`RUN_CUDA_FV3_STATE_TO_FV_ENABLE=1` as an explicit dev/testing opt-in
usable at any density including the known-unsafe 1-GPU/6-ranks case,
and (c) never silently defaulting to CUDA at that known-unsafe density
(baseline ~70 MiB/GPU free at 6-ranks/1-GPU; this port adds ~440
MiB/GPU — see P100).

Change budget: State_To_FV selector/default policy only. No numerical,
CUDA-kernel, ABI, MPI-semantics, or other-selector change. No
commit/push.

## 1. Selector Precedence (implemented)

```
1. RUN_CUDA_FV3_STATE_TO_FV_FORCE_FORTRAN set  -> Fortran   (always wins)
2. p99_base_gate false (unsupported config)    -> Fortran
3. RUN_CUDA_FV3_STATE_TO_FV_ENABLE set          -> CUDA      (dev opt-in,
                                                    any device_count,
                                                    unless FORCE_FORTRAN
                                                    also set)
4. p99_base_gate true and device_count>=2       -> CUDA      (new default,
                                                    no env needed)
5. otherwise (device_count<2, no override)      -> Fortran  (safe default)
```

`p99_base_gate` is the pre-existing Prompt 99 numerical/runtime
eligibility gate (`grid_type<4`, `.not. SW_DYNAMICS`, `.not.
hydrostatic`, `.not. ADJUST_SPLITS`, R8, backend enabled) — unchanged
by this task.

## 2. Implementation

**`cuda/interface/fv3_gpu_dyncore_bridge.F90`** (this repo, sole source
of truth — confirmed via build-artifact absolute-path evidence, no
duplicate copy exists): added a module-scope cache
`cached_device_count`, populated once inside the existing
`fv3_gpu_bridge_init` capabilities query (`caps%device_count`, itself
backed by `cudaGetDeviceCount` in `fv3_gpu_runtime.cpp` — process-local,
respects `CUDA_VISIBLE_DEVICES`, zero new ABI/CUDA code needed), and
exposed via a new pure query function `fv3_gpu_device_count()`,
mirroring the existing `fv3_gpu_backend_enabled()` idiom. 4 lines added
net (1 declaration, 1 cache-write, 1 function, 1 export).

**`FV_StateMod.F90`** (in the separate `GEOSfvdycore_fortran_perf`
checkout — this is the file production builds actually use; the copy
inside this git repo is a stale, untracked-by-production duplicate, see
§5 caveat below): rewrote the existing 3-branch selector block
(`p99_base_gate` -> `use_cuda_state_to_fv` -> `ENABLE` override ->
`FORCE_FORTRAN` override) to insert the `fv3_gpu_device_count()>=2`
condition into the base default assignment, ahead of the existing
`ENABLE`/`FORCE_FORTRAN` overrides, which are otherwise unchanged.

No CUDA kernel, ABI, buffer, or MPI-semantics code touched. No other
selector (PT/DELZ, REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS, TRACER,
REMAP_TOTALENERGYCON, FV_To_State) touched.

## 3. Stage 4/5 — Selector Matrix (C32, job 58650663)

All 6 required cases run in one `--gres=gpu:2` allocation (2-GPU cases:
`CUDA_VISIBLE_DEVICES` unset; 1-GPU cases: `CUDA_VISIBLE_DEVICES=0` —
same technique as `p103_c512_2gpu_validation.sbatch`). Evidence: exit
rc, `[P99_EXIT_DUMP]` per-call `use_cuda_state_to_fv=T|F` counts (T/F
split within a case reflects the pre-existing `ADJUST_SPLITS`-true
cold-start call(s), unrelated to this change — same split appears in
every CUDA-selected case).

| Case | CUDA_VISIBLE_DEVICES | ENABLE | FORCE_FORTRAN | Expected | rc | T/F | Result |
|---|---|---|---|---|---|---|---|
| 2gpu_default | unset | unset | unset | CUDA | 0 | 36/6 | PASS |
| 2gpu_forcefortran | unset | unset | 1 | Fortran | 0 | 0/42 | PASS |
| 2gpu_enable | unset | 1 | unset | CUDA | 0 | 36/6 | PASS |
| 1gpu_default | 0 | unset | unset | Fortran | 0 | 0/42 | PASS |
| 1gpu_enable | 0 | 1 | unset | CUDA (dev opt-in) | 0 | 36/6 | PASS |
| 1gpu_enable_force | 0 | 1 | 1 | Fortran (FORCE wins) | 0 | 0/42 | PASS |

**All 6/6 PASS.** FORCE_FORTRAN precedence over both the new default
and ENABLE confirmed (`1gpu_enable_force`). Device-count safety at
1-GPU confirmed (`1gpu_default`). Dev opt-in at 1-GPU confirmed
(`1gpu_enable`).

An initial run (job 58650610) hit the documented stale-`EXPDIR`
"Do Not Repeat" lesson (CURRENT_STATE.md §8) — a copied `fv3.j` for the
new `p106_c32_selector` rundir still pointed `EXPDIR` at the source
rundir (`p99_c32_cuda`), silently overwriting that rundir's original
Prompt 99 scratch/log evidence with this job's last case. Fixed by
editing the one `setenv EXPDIR` line and resubmitting (job 58650663,
clean). The corrupted `p99_c32_cuda` scratch artifacts were removed;
the P99 *finding* itself is unaffected (already recorded in
`p99_state_to_fv_cuda_implementation_2026-09-28.md`, independent of the
raw rundir). No production doc content was lost — only raw rundir
scratch/log files, which are not committed artifacts.

## 4. Stage 6/7 — C512/2-GPU Smoke (job 58651017)

Reused P104's `p104_correctness_cuda`/`p104_correctness_fortran`
rundirs (already at the validated topology: 6 ranks, 2 A100 GPUs,
3-ranks/GPU, MPS ON, R8), binaries refreshed to the new build (sha256
`f6e30af3...`), single `--gres=gpu:2` allocation, MPS lifecycle
started via the existing `fv3_mps_lifecycle.sh`.

**Stage 6 — bare new default, no env override** (`p104_correctness_cuda`,
both `RUN_CUDA_FV3_STATE_TO_FV_ENABLE` and `_FORCE_FORTRAN` left unset):

- Dispatch: CUDA selected (`use_cuda_state_to_fv` T=180/F=6 across 30
  `State_To_FV` calls x 6 ranks — same `ADJUST_SPLITS`-explained split
  ratio as the Stage-4 matrix, scaled up)
- rc=0, `DYN_DATE_COUNT=30` (matches the established P104/P105 call
  count for this config)
- 0 `MPI_Abort`/`FATAL`/`This Job Failed`/OOM hits; the only
  case-insensitive `NaN|Inf` grep hits are `INFO:`-logging-line false
  positives (substring match on "Inf" inside "INFO"), not real
  divergence — same benign pattern as prior prompts' identical grep
  usage
- Peak per-GPU memory during the run (concurrent poller, not a
  post-hoc idle checkpoint — see CURRENT_STATE.md §8 on why a
  post-hoc check would be meaningless): **20585 MiB/GPU both GPUs** —
  matches the ~20.6 GiB/GPU prediction (P103's 19.68 GiB/GPU converged
  baseline + P100's predicted +452.97 MiB, P104's confirmed +440.4
  MiB/GPU) to within measurement granularity
- Timing block: TOTAL 109.83 ms/call, FV_DYN_LOOP 33.19, DYN_CORE
  14.89, COMM_TOTAL 1.92 (single-pair smoke, not a performance A/B —
  P103/P104 remain the authoritative performance figures)

**Result: PASS.** The new default correctly selects CUDA at the
validated density with no environment variable set.

**Stage 7 — escape hatch** (`p104_correctness_fortran`,
`RUN_CUDA_FV3_STATE_TO_FV_FORCE_FORTRAN=1`):

- Dispatch: Fortran selected (T=0/F=186, all calls)
- rc=0, `DYN_DATE_COUNT=30`, 0 Abort/FATAL/OOM, same benign
  `INFO`-substring-only NaN/Inf grep pattern
- Timing block: TOTAL 107.53 ms/call, FV_DYN_LOOP 31.91, DYN_CORE
  15.03, COMM_TOTAL 2.77

**Result: PASS.** The escape hatch is unaffected by this change and
still deterministically forces Fortran at the validated 2-GPU density.

## 5. Stage 8 — 1-GPU Safety

Satisfied directly by the Stage-4 matrix's `1gpu_default` case
(rc=0, T=0/F=42 — all Fortran) per the task's own allowance to use
cheap selector evidence rather than re-running the known-unsafe
C512/1-GPU/`State_To_FV`-CUDA path. **The known-unsafe default
(CUDA at 6-ranks/1-GPU, ~70 MiB/GPU free baseline before this port's
~440 MiB/GPU addition) is correctly avoided: the new default only
activates at `device_count>=2`.**

## 6. Stage 9 — Execution/Stability Regression

Both Stage 6/7 C512/2-GPU runs: rc=0, 0 Abort/FATAL/OOM,
`DYN_DATE_COUNT=30` matching the established call count, no numerically
alarming log content. No full correctness rerun performed — Prompts
104-105 remain the authoritative numerical-agreement evidence (T/PS
bitwise, U/V round-off clean) for this exact density and code path; this
task changed only the selector's default-assignment condition, not the
CUDA dispatch code path itself, so P105's field-level comparison result
carries forward unchanged. No new HISTORY-output diff was run (the
Stage 6/7 rundirs' scratch output was overwritten during the smoke
test, and a fresh diff was judged unnecessary given the above).

## 7. Production CUDA-Default Table (updated)

| Port | Default | Notes |
|---|---|---|
| PT/DELZ | CUDA | Unconditional, all densities (P49-69) |
| REMAP_UV | CUDA | Unconditional, all densities (P71-84A) |
| REMAP_Q | CUDA | Unconditional, all densities (P84B-89) |
| REMAP_PRESSUREVARS | CUDA | Unconditional, all densities (P90-93) |
| **State_To_FV** | **CONDITIONAL CUDA** (this task) | CUDA iff `p99_base_gate` and process-local `device_count>=2`; Fortran otherwise; `FORCE_FORTRAN`/`ENABLE` override in both directions |
| TRACER_2D (hord==8) | Fortran | STOP CUDA PRODUCTION OPTIMIZATION (P73-83); moot in production (`hord_mt/vt/tm=6`) |
| REMAP_TOTALENERGYCON | Fortran | BLOCKED BY COMMUNICATION, `mpp_global_sum`-bound (P95) |
| FV_To_State | Fortran | KEEP ON CPU — already host-resident both ends (P97) |

## 8. Validated Scope (explicit)

This enablement is validated **only** for:

- Grid: C512
- Precision: **R8 only** (R4 not tested by this task or by P103-105)
- Topology: **single node**, 6 MPI ranks, 2 A100 GPUs, 3 ranks/GPU
- MPS: ON
- Config: production HS first-run (`grid_type=0`, `hydrostatic=.F.`,
  `a_imp=1.0`, non-`SW_DYNAMICS`, non-`ADJUST_SPLITS` steps)

**Not validated by this task or its inputs, do not claim:**

- R4 (untested at any density for this port)
- C1024 or any grid beyond C512
- Multi-node topologies (see §9 below — the device-mapping formula is
  single-node-safe only)
- Any density other than 2-GPU/3-ranks-per-GPU (1-GPU is Fortran by
  design; 3-4 GPU spread was only P102's zero-source-change memory
  diagnostic, never performance- or correctness-validated for this
  specific port at that density)
- A default flip to CUDA at 1-GPU (explicitly and intentionally NOT
  done — the dev opt-in `ENABLE=1` remains the only path to CUDA at
  that density)

## 9. Multi-Node Device-Mapping Limitation (documented, not implemented)

`fv3_gpu_bridge_init`'s device assignment,
`dev = mod(comm_rank, int(caps%device_count))`, and this task's new
`fv3_gpu_device_count()>=2` default gate both assume **all 6 ranks and
all visible GPUs are on the same node** (`comm_rank` is the global
communicator rank, not a node-local rank). At single-node C512 (the
only topology validated through P105 and this task) this is correct by
construction. It would be **incorrect** at a genuine multi-node
topology (e.g. ranks spread across 2 nodes with 1 GPU visible per
node): `comm_rank` would not correspond to a node-local GPU index, and
`device_count` as queried per-process would reflect that node's visible
device count, not the whole job's. Fixing this correctly requires a
node-local communicator (e.g. `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED,
...)`) to derive a node-local rank and node-local device count,
replacing the raw global-`comm_rank` modulo. **Out of scope for this
task** (single-node only, per task scope) — flagged here as a known
limitation for whoever attempts multi-node scaling next (§11 Next
Task, option A).

## 10. Modernization Context (derived, NOT a new validated result)

For context only — **not a new measurement, not to be promoted as a
validated production figure**: combining P103's 2-GPU TOTAL 1.120x with
this task's now-default `State_To_FV` (P104's isolated 1.633x direct /
+1.01% TOTAL contribution) gives an illustrative composite of roughly
1.13x TOTAL at this exact density — this is **DERIVED CONTEXT ONLY**,
not independently re-measured by this task's Stage 6/7 smoke (which
ran a single unpaired sample, not a same-node A/B). The authoritative
per-port numbers remain P103 (2-GPU scaling) and P104-105
(`State_To_FV` port), each already correctly labeled in CURRENT_STATE.md
§5.

## 11. Next Recommended Task

**DONE — see `p107_node_local_gpu_binding_2026-09-30.md`.** Node-local
`MPI_Comm_split_type`-based binding implemented, GPU-executed, and
validated single-node (bit-identical to this doc's mapping) and genuine
multi-node (divergence proof vs. the old global-rank rule). No
performance regression; State_To_FV's `device_count>=2` policy from this
doc unaffected.

**Recommend: Option A — implement node-local GPU binding
(`MPI_Comm_split_type`-based), replacing the current global-`comm_rank`
modulo.** Rationale: this is the binding architectural constraint
blocking any multi-node scaling attempt (§9), already flagged twice
before (P102 §9, CURRENT_STATE.md §7 "CUDA dyn_core bridge hardcodes
`comm_size==6`"). It is also lower-risk and more foundational than
Option B (C1024 Option-A `State_To_FV` test), which would inherit the
same single-node assumption and cannot itself proceed past a single
node's GPU count without this fix first. Fixing it does not require
resolution of the C1024 stability search (its own separate open item,
CURRENT_STATE.md §7) and is independently testable at the existing
2-GPU/C512 density before any larger-scale attempt.

## 12. Final Status

```
No numerical changes: YES
No commit/push: YES
```

**STATE_TO_FV CUDA CONDITIONAL DEFAULT ENABLED — 2-GPU VALIDATED SCOPE
ACTIVE**
