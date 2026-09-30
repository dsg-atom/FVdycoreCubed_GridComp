# Prompt 54 — ak/bk Field Registration Fix and PT/DELZ CUDA Revalidation

Date: 2026-09-22. Branch: `optimization/cuda-fv3-performance`. Model: Sonnet
(no Opus escalation — ak/bk ownership/lifetime was unambiguous from source).

## 1. Scope

Prompt 54 followed Prompt 53's finding that the CUDA pt/delz remap dispatch
(`fv3_gpu_pt_delz_remap` → `launch_pt_delz_remap_ctx`) crashed with an
immediate `MPI_Abort` on every rank because `FieldId::ak`/`FieldId::bk` were
never registered by the only registration path the live build actually
calls, `fv3_gpu_field_registry.F90::fv3_gpu_registry_build` (invoked from
`fv3_gpu_bridge_init`). Prompt 54's mandate: fix that one registration gap,
revalidate, and stop before any performance work.

## 2. Stage 0 — Confirmed init path (no Opus escalation needed)

- `fv3_gpu_bridge_init` → `fv3_gpu_registry_build` is the **only** live CUDA
  registration path. A grep of the authoritative
  `GEOSfvdycore_fortran_perf/.../FV_StateMod.F90` for
  `fv3_gpu_interface_init|fv3_gpu_bridge_init|fv3_dyncore_backend|USE_FV3_CUDA`
  returned **zero** hits for `fv3_gpu_interface_init` — only
  `fv3_gpu_bridge_init`/`fv3_gpu_bridge_final` are referenced. The disconnected
  `fv3_gpu_interface_mod.F90::register_immutable_grid_fields` helper (which
  *does* register ak/bk) is dead code on this execution path; it is called
  only from this repo's own unused root-level `FV_StateMod.F90`, not the
  authoritative one in `GEOSfvdycore_fortran_perf`.
- Confirmed `ak`, `bk` absent from `fv3_gpu_registry_build`'s existing
  `add`/`add1`/`add2`/`add3` call list before this fix; confirmed
  `launch_pt_delz_remap_ctx` (`cuda/dyn_core/vertical/fv3_pt_delz_remap.cu`)
  requires both via `a.ak`/`a.bk` lookups; confirmed no later registration
  call occurs anywhere before first dispatch.

## 3. Stage 1/2 — Fix choice and semantics

**Option A** (add `ak`/`bk` registration directly to `fv3_gpu_registry_build`)
chosen over **Option B** (call the dead-code
`register_immutable_grid_fields` from `fv3_gpu_bridge_init`) — no source
evidence that B is the intended canonical path; A is the minimal,
convention-consistent change.

Semantics confirmed by direct source inspection (not guessed):
- `Atm%ak`/`Atm%bk` are genuine `Atm` components (not module-owned scratch),
  declared deferred-shape allocatable in
  `GEOSfvdycore_fortran_perf/.../model/fv_arrays.F90:1260-1261`, allocated
  1-based, size `npz+1` at lines 1503-1504
  (`allocate(Atm%ak(npz_2d+1))`/`allocate(Atm%bk(npz_2d+1))`).
- CUDA kernel usage (`fv3_pt_delz_remap.cu`, lines ~60-100) confirms
  `ak`/`bk` are accessed 0-based over `[0, km]` (`km+1` values) — matching
  the existing `add1(FV3_FID_DP_REF, ..., 0, npz)` convention already in use
  for `r_dp_ref` in the same file (0-based lower index, extent = element
  count, not upper-bound index).

## 4. Stage 3 — Fix applied

Single substantial source change, `cuda/interface/fv3_gpu_field_registry.F90`
(`fv3_gpu_registry_build`), immediately after the existing `r_dp_ref`
registration:

```fortran
call add1(FV3_FID_AK, c_loc(Atm%ak(1)), 0, npz+1)
call add1(FV3_FID_BK, c_loc(Atm%bk(1)), 0, npz+1)
```

No other file changed. `pt_delz` kernel math, `map_scalar`, `fv_mapz`
numerical formulas, dispatch conditions, MPI topology, and FMA policy were
not touched.

## 5. Stage 4/5 — Build and binary provenance

Rebuilt via `make -j 12 StandAlone_FV3_Dycore.x` in
`GEOSfvdycore_fortran_perf/build` (production Intel-MPI stack). Only
`fv3_gpu_field_registry.F90.o` and `fv3_gpu_dyncore_bridge.F90.o` (dependent,
not edited) recompiled; clean relink, `MAKE_RC=0`.

- Executable: `.../GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x`
- mtime: `2026-09-22 15:05:21.035074791 -0400`
- sha256: `74d7f7c6aaffadb374b38e4b31312233dd56d48531e51a4d3cca4dc2a2e1488e`
- `strings` confirms `fv3_gpu_bridge_init`/`RUN_CUDA_FV3`/`pt_delz_remap`
  (51 hits) and `fv3_gpu_pt_delz_remap`/`launch_pt_delz_remap` (7 hits)
  present; `ldd` confirms Intel MPI 2021.13 linkage.
- A per-job preflight (hash + symbol-count check, fail-fast on mismatch) was
  added to the sbatch script itself
  (`p54_stage6_7_c32.sbatch`) to prevent any repeat of the Prompt 53
  stale-`install/bin` failure. Both arms were pointed at this exact binary
  via `FV3EXE`-override wrapper scripts placed in each job directory
  (`p54_stage6_7_c32_wrapper_{fortran,cuda}.sh`), following the same
  override mechanism Prompt 53 validated for its nsys wrapper.

## 6. Stage 6 result — registration fix confirmed, NEW blocking bug found

Job **58511314** (`p54_stage6_7_c32.sbatch`), same-node matched pair, reused
`p52c_stage11_{fortran,cuda}` job dirs (C32, L10, `kord_tm=kord_wz=9`,
Held-Suarez, `RUN_CUDA_FV3=1` only in the cuda dir), CUDA arm wrapped in
`compute-sanitizer --tool memcheck` (zero source changes; supplementary
GPU-execution/memory-safety evidence only, per Prompt 54's ban on Nsight
tools before correctness is proven).

**Registration fix confirmed working**: the specific old FATAL string
(`"requires pe, peln, pt, delz, delp, pk, pkz, ak, bk"`) is **absent** (0
hits) from the cuda arm's log — `FieldId::ak`/`FieldId::bk` lookups inside
`launch_pt_delz_remap_ctx` now succeed. `fv3_gpu_bridge_init` itself
completes normally on all ranks (`"CUDA dyn_core backend ready"`, PE 0).

**New, unrelated, deterministic failure**: the CUDA arm still aborts — every
rank hits `FATAL from PE <n>: CUDA dyn_core backend: fv3_gpu_sync_to_host
failed: unspecified launch failure`, followed by `MPI_Abort`. The
compute-sanitizer memcheck logs (identical across all 6 ranks, 20 error
blocks each, no `ERROR SUMMARY` line — i.e. the process was aborted before
sanitizer could print its final summary) pinpoint the exact cause:

```
Invalid __global__ read of size 8 bytes
    at fv3::gpu::vertical::<unnamed>::pt_delz_remap_kernel(PtDelzRemapArgs)+0x13c0
    by thread (0..15,1,0) in block (0,1,0) and (1,1,0)
    Address ... is out of bounds, 1/9/17/25/33/41/49/57/65/73 bytes AFTER
      the nearest allocation (base 0x14bc4eb75200, size 101,728 bytes)
    [the same faulting addresses are simultaneously 8/16/24/.../80 bytes
      BEFORE the next adjacent allocation (base 0x14bc4eb8e000, size
      90,112 bytes) — i.e. one boundary between two adjacent device buffers]
```

This is a **first-ever real invocation** finding: `pt_delz_remap_kernel` had
never actually executed in a live run before this fix (every prior attempt
died at the ak/bk lookup before reaching the kernel). The over-read is by up
to 10 elements (8-byte stride) past a buffer's end, reproducible identically
on all 6 ranks — a genuine indexing bug inside the kernel itself, unrelated
to ak/bk: the newly-registered `ak`/`bk` arrays are only 11 elements
(~44-88 bytes), far smaller than either buffer involved (101,728 / 90,112
bytes), so this is not a side effect of the registration fix — it is a
pre-existing kernel defect that the registration fix has, for the first
time, made reachable.

**Exit-code caveat (harness artifact, not a correctness signal)**: the raw
mpirun output shows `status = 1` (real abort, rank-reported), but the
sbatch script's own captured `rc=$?` after `csh -f ./fv3.j` read `0` in both
arms. This is `csh`/`esma_mpirun` exit-status propagation not surfacing
`mpirun`'s nonzero code to the calling shell — a pre-existing harness
quirk, not evidence that the run succeeded. The log content and
compute-sanitizer output are unambiguous and are the authoritative record;
`rc`/`RC=` in any p52c/p54 job script should not be trusted as a pass/fail
signal without corroborating log inspection.

Fortran arm (same job, matched pair): clean pass, `RC=0`, zero
`Range_Warn`/`NaN`/`Abort` hits.

## 7. Stages 7-12 — not reached

Blocked by the Stage 6 kernel bug above. Per Prompt 54's own stop condition
("if significant mismatch: STOP, do not profile") and its explicit
Forbidden list (no pt/delz kernel math changes), this bug cannot be fixed
within Prompt 54's authorized scope. Small-res correctness (Stage 7),
multi-step validation (Stage 8), the C512 gate (Stage 9), milestone
restoration (Stage 10), and the Nsight A/B (Stage 11/12) all remain
unreached and must wait for a dedicated, separately-authorized kernel fix.

**Resolution (Prompt 55)**: the out-of-bounds read above is fixed (root
cause: `pe`/`peln`'s real `(i,k,j)` registration axis order vs. every other
field's `(i,j,k)`). Confirmed by standalone regression and a new
production-shape smoke test, both 6/6 PASS at roundoff with 0
compute-sanitizer errors, and by 0 sanitizer errors in the first real
integrated invocation. However, that same first integrated invocation
(job 58512259) exposed a **second, distinct, still-unresolved** numerical
corruption (immediate, CUDA-arm-only, ~1.9M `Range_Warn`/~1.8M `NaN` hits) —
see `docs/performance/p55_pt_delz_oob_fix_2026-09-22.md` §7-8 for full
detail. Stages 7-12 above remain unreached; a fresh, separately-authorized
prompt is required before any of them can be attempted.

## 8. Summary

- ak/bk registration gap (Prompt 53's finding): **FIXED**, confirmed by
  direct evidence (old FATAL string gone, `fv3_gpu_bridge_init` completes,
  `ak`/`bk` lookups inside `launch_pt_delz_remap_ctx` succeed).
- pt/delz CUDA dispatch: still **NOT GPU EXECUTED successfully** — a
  different, newly-exposed, deterministic out-of-bounds read inside
  `pt_delz_remap_kernel` itself now blocks every invocation. This is a
  kernel-internal indexing defect, out of scope for Prompt 54 to fix.
- Job ID of record: **58511314** (supersedes nothing — no prior job for
  this specific test; the invalidated 58506114/58506170/58506276 remain
  invalidated and unused).
