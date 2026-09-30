# Prompt 53: pt/delz CUDA Remap — Performance-Neutrality Root Cause

Date: 2026-09-22
Branch: `optimization/cuda-fv3-performance`
Scope: profiling/attribution only — no numerical code changes, no new CUDA
coverage, no MPI/runtime topology changes. This document reports what was
found; it does not fix it.

## 0. Task and outcome

Prompt 53 asked why the C512 CUDA pt/delz `fv_mapz` remap integration
(previously reported as GPU EXECUTED / NUMERICALLY VALIDATED at "Stage 13",
job 58506276) produced no measurable whole-model speedup, via a same-node
Nsight Systems A/B isolating the dispatch with `kord_tm`/`kord_wz` 8-vs-9.

**Outcome: the intended profiling could not be performed, because the CUDA
pt/delz dispatch does not execute at all — it crashes with an immediate
`MPI_Abort` on every rank the first time it is invoked.** Root-causing that
crash led to a second, more severe finding: the previously-recorded
"Stage 13"/"Stage 11" GPU-execution and numerical-validation results for
this feature are invalid. Both arms of both prior validation jobs ran the
same stale, pre-CUDA-integration Fortran-only binary — the CUDA pt/delz path
was never actually exercised in either. See §3.

## 1. Reproduction: immediate MPI_Abort on every real invocation

Job 58510550 (same-node, same-allocation, two-arm nsys capture, both arms
`RUN_CUDA_FV3=1`, `RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`,
`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`, MPS ON, C512, `JOB_SGMT` shortened to
6 `DYN_CORE` calls, correct `build/bin/StandAlone_FV3_Dycore.x` confirmed via
an explicit `FV3EXE`-override nsys wrapper — `REAL=.../build/bin/
StandAlone_FV3_Dycore.x`, mtime 2026-09-22 08:09, 51 CUDA-symbol hits via
`strings`):

- **baseline arm** (`kord_tm=8`/`kord_wz=8` — `use_cuda_pt_delz_remap` gate
  cannot engage): completed cleanly, RC=0, `CUDA dyn_core backend ready`
  printed on all ranks, 6 `DYN_CORE` cycles recorded.
- **integrated arm** (`kord_tm=9`/`kord_wz=9` — gate engaged): `CUDA dyn_core
  backend ready` printed on all ranks, then within ~150 log lines **all 6
  ranks** fail identically:

  ```
  FATAL from PE     5: CUDA dyn_core backend: fv3_gpu_pt_delz_remap failed:
  pt/delz remap requires pe, peln, pt, delz, delp, pk, pkz, ak, bk
  ```

  followed by `MPI_Abort` stack traces through
  `fv3_gpu_dyncore_bridge.F90:749` → `:427` (the `check(crc, err,
  'fv3_gpu_pt_delz_remap')` call site) → `fv_mapz.F90:504` →
  `fv_dynamics.F90:952` → `FV_StateMod.F90:2050` →
  `DynCore_GridCompMod.F90:4464`.

The job then hung post-crash (Slurm kept reporting `RUNNING` for 30+ minutes
with a completely static log/mtime after all 6 ranks had already printed
their abort traces — nsys-wrapped `mpirun` not propagating the abort
cleanly) and was cancelled (`scancel 58510550`).

## 2. Root cause: `ak`/`bk` are never registered on the path actually used

`launch_pt_delz_remap_ctx` (`cuda/dyn_core/vertical/fv3_pt_delz_remap.cu:227-
240`) looks up `FieldId::ak`/`FieldId::bk` via the field registry and rejects
the launch if either lookup returns null:

```cpp
const auto *v_ak = x.fields->find(FieldId::ak);
const auto *v_bk = x.fields->find(FieldId::bk);
...
if (!v_pe || !v_peln || !v_pt || !v_delz || !v_delp || !v_pk || !v_pkz ||
    !v_ak || !v_bk)
  return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
              "pt/delz remap requires pe, peln, pt, delz, delp, pk, pkz, "
              "ak, bk");
```

`FieldId::ak`/`FieldId::bk` are declared in the runtime's field enum
(`fv3_gpu_internal.hpp`) — the enum is complete. The gap is in
**registration**. Two independent, non-interoperating Fortran-side
registration mechanisms exist in this repo:

- **`cuda/interface/fv3_gpu_field_registry.F90`** (`fv3_gpu_registry_build`)
  — the mechanism `fv3_gpu_bridge_init` actually calls
  (`fv3_gpu_dyncore_bridge.F90:213`, the only registry call in that
  subroutine). This is the real, single init path used by the production
  call site and the StandAlone HS harness. `Atm%ak`/`Atm%bk` appear in this
  file only as inputs to derive host-side scratch (`r_dp_ref`, `pfull`) —
  `grep -ni "field_ak\|field_bk\|FieldId::ak\|FieldId::bk"` across the whole
  932-line file returns **zero hits**. `ak`/`bk` are never registered as
  standalone device fields here.
- **`cuda/interface/fv3_gpu_interface_mod.F90`** (`register_immutable_grid_
  fields`, using `add_field_1d`/`FIELD_AK`/`FIELD_BK`) — DOES register
  `ak`/`bk` as device fields, at lines 722-754. This subroutine is built and
  linked (referenced from `fv3_gpu_runtime.cpp` and listed in
  `cuda/interface/CMakeLists.txt`) but is **never called from
  `fv3_gpu_bridge_init`** — confirmed by reading that subroutine's full body
  (lines 112-295), whose only registry call is `fv3_gpu_registry_build`.
  This module is dead code with respect to the real execution path, most
  likely a holdover from an earlier (Prompt 5-era) GEOS-interface design
  that was superseded by `fv3_gpu_field_registry.F90` without this second
  module being removed or reconciled.

**Consequence: the CUDA pt/delz dispatch cannot succeed in any real run.**
This is not specific to the shortened `JOB_SGMT` test in job 58510550 — the
same `find(FieldId::ak)`/`find(FieldId::bk)` null-lookup rejection fires on
the very first invocation of `fv3_gpu_pt_delz_remap` in any configuration
that reaches `use_cuda_pt_delz_remap`. It is a deterministic, permanent
implementation gap, not a flaky or resolution-dependent failure.

This is a source-level defect requiring a fix (wiring `ak`/`bk` into
`fv3_gpu_registry_build`, or invoking `register_immutable_grid_fields`
from `fv3_gpu_bridge_init`) that this task's scope explicitly excludes
making. Per the project's core rule, this is flagged for explicit
authorization rather than fixed here.

## 3. Why the crash contradicts the previously-recorded "Stage 11/13 PASS"

`docs/performance/p52c_fv_mapz_cuda_integration_2026-09-21.md` records, under
"Update (Prompt 52C-IMPLEMENT, Stages 11 & 13, 2026-09-22)":

- Stage 11 (c32, jobs 58506027/58506114/58506170): "both backends RC=0."
- Stage 13 (C512, job 58506276, node `warpa005`): "Both backends RC=0, 0
  range_warn/NaN/error... Bitwise-identical at every field/checkpoint"
  (T/U/V/PS, max_abs=0.0, max_rel=0.0 at all 3 checkpoints), and a final
  status block claiming "GPU EXECUTED: yes, including the first real
  integrated-path execution (Stage 11 c32, Stage 13 C512). NUMERICALLY
  VALIDATED: yes — Stage 13 C512 production correctness gate PASSED
  bitwise vs. Fortran."

Given §1/§2's finding that the CUDA dispatch crashes immediately and
deterministically whenever it actually engages, an `RC=0` "cuda" arm for
either stage is inconsistent with the dispatch having actually run. Direct
inspection of both stages' own job logs resolves the contradiction:

```
$ grep -n "mpirun.*StandAlone" p52c_stage11_same_node_fc_58506170.out
.../install/bin/StandAlone_FV3_Dycore.x --logging_config logging.yaml   (fortran arm)
.../install/bin/StandAlone_FV3_Dycore.x --logging_config logging.yaml   (cuda arm)

$ grep -n "mpirun.*StandAlone" p52c_stage13_same_node_fc_58506276.out
.../install/bin/StandAlone_FV3_Dycore.x --logging_config logging.yaml   (fortran arm)
.../install/bin/StandAlone_FV3_Dycore.x --logging_config logging.yaml   (cuda arm)
```

**Both arms of both jobs launched via `install/bin/StandAlone_FV3_Dycore.x`,
not `build/bin/StandAlone_FV3_Dycore.x`.** Neither job directory
(`p52c_stage11_cuda`, `p52c_stage13_cuda`) contained an `FV3EXE`-override
executable file, so `fv3.j` fell back to the default GEOSBIN-resolved path,
which resolves to `install/bin`. Binary inspection:

| Binary | mtime | `strings \| grep -c "fv3_gpu_bridge_init\|RUN_CUDA_FV3\|pt_delz_remap"` |
|---|---|---|
| `install/bin/StandAlone_FV3_Dycore.x` | 2026-08-31 11:26:37 (3 weeks stale, predates all CUDA integration) | **0** |
| `build/bin/StandAlone_FV3_Dycore.x` | 2026-09-22 08:09:41 (current, correct) | **51** |

`install/bin` is a pure pre-CUDA-integration Fortran-only build — it
contains no `RUN_CUDA_FV3` handling and cannot dispatch to CUDA regardless
of `kord_tm`/`kord_wz`/`AGCM.rc` settings, which fully explains the
"bitwise-identical, exactly 0.0" comparison result: **both the "fortran" and
"cuda" arms of Stage 11 and Stage 13 ran the identical non-CUDA binary.**
This is the exact stale-binary risk `CURRENT_STATE.md` §7 already documents
in the abstract ("`install/bin`... not auto-synced... always refresh and
verify via `strings`") — materializing concretely here.

By contrast, this task's own job 58510550 used an explicit `FV3EXE`-override
nsys wrapper hardcoding `REAL=.../build/bin/StandAlone_FV3_Dycore.x`
(confirmed via the job log's own `***** USING ****`/`mpirun` lines pointing
at the wrapper script inside each arm's scratch directory, which in turn
execs the correct `build/bin` binary) — which is why this task is the first
job to have genuinely exercised the CUDA pt/delz dispatch, and the first to
observe it fail.

## 4. Conclusion

- The CUDA pt/delz dispatch (`use_cuda_pt_delz_remap`, `kord_tm=9`/
  `kord_wz=9`) has **never been GPU-executed successfully in any real
  multi-rank run.** Every prior claim of GPU execution or numerical
  validation for this specific dispatch path (Stage 11 c32, Stage 13 C512,
  in `p52c_fv_mapz_cuda_integration_2026-09-21.md`) is **invalid** — both
  arms of both jobs ran the same stale non-CUDA binary.
- The dispatch currently crashes deterministically on its first invocation
  due to a genuine source-level gap: `ak`/`bk` are required by
  `launch_pt_delz_remap_ctx` but never registered by
  `fv3_gpu_field_registry.F90`'s `fv3_gpu_registry_build`, the only
  registration subroutine actually called by `fv3_gpu_bridge_init`.
- "Performance neutrality" as originally framed by Prompt 53 does not
  apply — there is no performance data for this dispatch to be neutral,
  because it has never completed a run. The prior "no speedup" observation
  was against a binary that also never ran the feature.
- This is a correctness/validation-status finding, not merely a performance
  one. It is reported here rather than fixed, per this task's explicit
  scope restriction against numerical/source changes.

## 5. What is NOT affected

- Gate C2 (the 54-stage production first-run dyn_core inventory) is
  unaffected — it does not depend on the pt/delz remap dispatch.
- The standalone kernel validation from Prompt 52B (`map_scalar`/
  `scalar_profile`/`cs_limiters`, job 58499304) and the fixture-level
  pt/delz arithmetic validation from Prompt 52C-PREP are unaffected — both
  are standalone/fixture-level, not integrated-dispatch, validation, and
  neither claimed integrated GPU execution.
- The C512 Fortran-vs-CUDA baseline performance parity result (job 58468244,
  Prompt 48) is unaffected — it predates and does not involve the pt/delz
  dispatch.

## 6. Recommended next step (requires explicit authorization — not done here)

Fix the `ak`/`bk` registration gap in `fv3_gpu_field_registry.F90`'s
`fv3_gpu_registry_build` (the actually-used registry path), then re-run
Stage 11 (c32) and Stage 13 (C512) correctness gates using a verified
`build/bin` binary (explicit `FV3EXE` override or a fresh `install/bin`
sync, confirmed via `strings` before trusting the run) before any pt/delz
performance profiling is attempted again.

---

PROFILE BLOCKED — CUDA pt/delz remap dispatch crashes with an immediate
`MPI_Abort` on every rank due to an `ak`/`bk` field-registration gap
(`fv3_gpu_field_registry.F90` never registers `ak`/`bk`, which
`launch_pt_delz_remap_ctx` requires); no steady-state `DYN_CORE` cycle
can be captured for this dispatch until that source-level gap is fixed
under explicit authorization, and the prior "Stage 11/13 GPU EXECUTED /
NUMERICALLY VALIDATED" claims for this feature are invalidated (both
arms of both jobs ran a stale non-CUDA `install/bin` binary).
