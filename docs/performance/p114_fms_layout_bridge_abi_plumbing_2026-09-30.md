# Prompt 114 — FMS layout_x/layout_y CUDA Bridge ABI Plumbing

Date: 2026-09-30
Branch: `optimization/cuda-fv3-performance`
Scope: WIRES ONLY. No >6-rank enablement, no gate relaxation, no topology
behavior change, no numerical change, no commit/push.

## 1. Goal

Add authoritative FMS decomposition layout metadata (`layout_x`, `layout_y`,
from `Atm%layout(1:2)`) to the existing Fortran -> C ABI -> CUDA runtime
initialization path. The runtime stores these values and exposes them for
diagnostics/tests, but takes no new production branch based on them yet.
This is the implementation of the bounded next task specified by Prompt 113
(`p113_fms_bridge_subtile_metadata_design_2026-09-30.md` §15).

## 2. Change Summary

Additive-only, trailing-field/trailing-argument ABI extension. Four files
edited, each purely additive:

| File | Change |
|---|---|
| `cuda/interface/fv3_gpu_abi_mod.F90` | `fv3_gpu_create_desc` gains trailing `layout_x, layout_y` (`c_int32_t`); `fv3_gpu_set_cubed_sphere_topology` interface gains trailing `layout_x, layout_y` value args (before `err`) |
| `cuda/interface/fv3_gpu_dyncore_bridge.F90` | derives `layout_x/layout_y` from `Atm%layout(1:2)` in `fv3_gpu_bridge_init`; FATALs if either is `<1`; stores into `cd%layout_x/layout_y`; passes both to `fv3_gpu_set_cubed_sphere_topology`; extends `build_rank_of_tile` with a second `MPI_Allgather` cross-checking every rank reports the same layout; extends the existing `RUN_CUDA_FV3_LOG_GPU_BINDING`-gated debug block with an `FV3_GPU_LAYOUT layout=<x>x<y> tile=<t> rank=<r>` print |
| `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` | `fv3_gpu_set_cubed_sphere_topology` definition gains trailing `int32_t layout_x, layout_y`; stored into `TopologyView.layout_x/layout_y` with the same `<=0`-defaults-to-`1` normalization as `fv3_gpu_create`'s existing fields; storage happens *after* the unchanged `mpi_size != halo::cubed_sphere_tiles` gate and the unchanged `make_cubed_sphere_topology` call — does not affect which topology builder ran or any neighbour/seam computation |
| `cuda/interface/fv3_gpu_interface_mod.F90` | mirrored the same two edits in the legacy/parallel wrapper module (`fv3_gpu_create_desc_t`, its own `fv3_gpu_set_cubed_sphere_topology` interface, derivation, call site) for byte-layout parity with the ABI struct — **NOT compile-verified this session** (see §5) |

`fv3_gpu_internal.hpp`'s `TopologyView` already carried `layout_x=1,
layout_y=1` default-valued fields (added in a prior session, per
`internal_contract_version`'s documented purely-additive-default-valued-field
convention — no version bump needed); this task's `fv3_gpu_runtime.cpp` edit
is what actually populates them from the new ABI arguments instead of leaving
them at their compiled-in default.

No file in the explicit Do-NOT list was touched:
- Fortran bridge `comm_size==6` gate (`fv3_gpu_dyncore_bridge.F90:168`) — unchanged.
- `build_rank_of_tile`'s one-rank-per-tile FATAL (`fv3_gpu_dyncore_bridge.F90:1516`) — unchanged (a second, independent FATAL was *added* by this task for layout-consistency, not a relaxation of the existing one).
- `fv3_gpu_runtime.cpp`'s `mpi_size != halo::cubed_sphere_tiles` check (line 1585) — unchanged, still the sole topology-entry gate.
- `create_mpi_transport()`'s production `comm_size==6` gate (`fv3_gpu_halo_mpi.cpp`) — untouched.
- No topology builder switch (`make_cubed_sphere_topology` still the only one invoked from the production call site).
- No corner-transport work, no seam/vector-math edit, no CUDA-aware MPI change, no production default change, no R4 touch, no commit/push.

## 3. Compile Verification

### 3a. C++/CUDA side (`build_cuda_r8_mpi`, OpenMPI dev stack)
- `cmake --build . --target fv3_gpu_api_test -j 8` -> EXIT=0.
- Full rebuild, all targets: `cmake --build . -j 8` -> EXIT=0, zero errors.
- `./fv3_gpu_api_test` -> PASS (confirms a zero-initialized
  `fv3_gpu_create_desc d{}`, i.e. `layout_x=layout_y=0`, still normalizes to
  1 and does not regress this pre-existing caller).
- `ctest --output-on-failure`: 37/39 PASS. The 2 failures
  (`fv3_wp_b2_hord8_transport`, `fv3_pert_ppm_iv1`) are the well-documented
  no-GPU-on-login-node kernel-launch failure
  (`CUDA driver version is insufficient for CUDA runtime version`) — not a
  regression from this change (both are unrelated to the ABI struct/topology
  path).

### 3b. Fortran/ABI side, against the REAL production call site (Intel MPI
production stack, `GEOSfvdycore_fortran_perf/build`)
- `FV3_CUDA_SOURCE_DIR` in that fixture's `CMakeCache.txt` points directly at
  this repo's live `cuda/` tree — no sync/copy step; edits are picked up
  automatically.
- `cmake --build . --target fvdycore -j 8` -> EXIT=0. Freshly compiled
  `fv3_gpu_abi_mod.F90.o`, `fv3_gpu_field_registry.F90.o`,
  `fv3_gpu_dyncore_bridge.F90.o`; linked `libfvdycore.a`; zero errors.
- `cmake --build . --target StandAlone_FV3_Dycore.x -j 8` -> initially failed
  only at the *final link* step (missing prebuilt CUDA archive directory,
  see §4) — all Fortran compilation, including `FV_StateMod.F90.o` (the real
  `fv3_gpu_bridge_init(FV_Atm(1), ...)` call site), succeeded with only 6
  pre-existing, unrelated warnings (`#6843`, unset `INTENT(OUT)` dummy args
  `u_inc/v_inc/t_inc/dp_inc/q_inc/o3_inc`).
- `fv3_gpu_interface_mod.F90` (legacy module) is confirmed via the fixture's
  own `@fvdycore/CMakeLists.txt` source list to NOT be part of this build —
  it is referenced only by this repo's own root `FV_StateMod.F90`, not built
  this session. Its edits are structurally mirrored from the
  compile-verified active-bridge pattern but are **NOT independently
  compile-verified**.

## 4. Build-Infrastructure Reconstruction (unrelated to the ABI change)

`build_cuda_r8_intelmpi/` (the 9 prebuilt CUDA static archives
`StandAlone_FV3_Dycore.x` links against) did not exist in this session's
checkout. Reconstructed it:
- Configure required `-DFV3_DYN_CORE_ENABLE_CUDA=ON` (default `OFF` in
  `cuda/dyn_core/CMakeLists.txt`, gates `enable_language(CUDA)`), the exact
  `nvcc` path `/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/bin/nvcc`
  (the bare `nvcc` resolved by the loaded module is a different, incomplete
  install missing `libcudadevrt.a`/`libcudart_static.a`), and the full Intel
  MPI production module stack (`comp/intel/2024.2.0` +
  `mpi/impi/2021.13`) loaded so `mpicxx`'s `I_MPI_SUBSTITUTE_INSTALLDIR`
  token resolves — using only `GEOSenv` left it unexpanded, breaking
  `FindMPI`.
- Built the exact 9 archives named by
  `GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/@fvdycore/CMakeLists.txt`
  (`fv3_production_dispatch`, `fv3_dyn_core_gpu`, `fv3_wp_a_cgrid`,
  `fv3_wp_b_dgrid`, `fv3_wp_b_cuda`, `fv3_wp_c_vertical`,
  `fv3_wp_d_operators`, `fv3_wp_d_cuda`, `fv3_cube_geometry`) — all 9
  targets exist verbatim in `cuda/dyn_core/CMakeLists.txt`; build EXIT=0.
- Re-linked `StandAlone_FV3_Dycore.x` -> EXIT=0 (only pre-existing
  `libcrypto.so.1.1`/`.so.3` linker warnings, unrelated).
- New binary sha256: `f3aefa5e06bf346c5feaeb51f99fbd085ecdf2b399189bbcf212b1eabf0c9270`.

This reconstruction added no code; it only restored a missing build
directory so the ABI change could be exercised at production scale.

## 5. Stage 9 — C32 Production Smoke (R8, 6 ranks, layout=(1,1))

Rundir `GEOSfvdycore_fortran_perf/experiments/p114_c32_smoke/` (fresh copy of
the validated `p106_c32_selector` job script; `EXPDIR` updated;
`RUN_CUDA_FV3_STATE_TO_FV_DUMP_EXIT` debug var removed in favor of
`RUN_CUDA_FV3_LOG_GPU_BINDING=1`). Slurm job 58665460, `warpa005`,
`gpu_a100`/`rome`, 6 ranks/1 GPU, 2h segment at `RUN_DT=1200`.

- `Status = 0`.
- All 6 `FV3_GPU_LAYOUT layout=1x1 tile=<1..6> rank=<0..5>` lines present,
  each rank's `tile`/`rank` self-consistent (tile = rank+1, one-rank-per-tile
  as expected for `FV_NX=1/FV_NY=6`).
- Zero FATAL/error/NaN/Inf matches.
- 6 `DYN Date` lines (20-min steps to a 2h segment) — normal completion, no
  truncation.

**Stage 9: GPU EXECUTED, clean.**

## 6. Stage 10 — C512 Regression Smoke (6 ranks, 2 GPUs, MPS ON, single node)

Rundir `GEOSfvdycore_fortran_perf/experiments/p114_c512_smoke/` (fresh copy
of the validated `p104_correctness_cuda` C512/2-GPU config; new sbatch
wrapper `p114_c512_smoke.sbatch` reusing the `fv3_mps_lifecycle.sh`
mechanism verbatim from `p106_c512_2gpu_smoke.sbatch`). Slurm job 58665576,
`warpa005`, 2 GPUs, MPS, 18-min segment at `RUN_DT=36` (30 steps).

- `rc=0`.
- `DYN_DATE_COUNT=30` (full, untruncated segment).
- `ABORT_COUNT=0`; no FATAL/Abort/NaN/Inf/OOM matches (only benign unrelated
  `INFO:` substring hits from the broad case-insensitive grep).
- All 6 `FV3_GPU_LAYOUT layout=1x1 tile=<1..6> rank=<0..5>` lines present and
  self-consistent, alongside the pre-existing `FV3_GPU_BIND` lines showing
  the expected 3-ranks/GPU MPS split (`devices=2`, `selected=0` or `1`).

**Stage 10: GPU EXECUTED, clean.**

## 7. Stage 11 — Behavioral-Neutrality Check vs Pre-P114

Compared against `p104_correctness_cuda/P104_CORR_result.log` (pre-P114
binary, same C512/6-rank/2-GPU/MPS/18-min-segment configuration):

| Metric | Pre-P114 | P114 (this task) |
|---|---|---|
| `DYN Date` line count | 30 | 30 |
| `DYN_CORE` timer (ms/call, mean) | 14.3733 | 14.3752 |
| `TOTAL` timer (ms/call) | 109.3909 | 110.2908 |

`DYN_CORE` differs by ~0.01% (round-off/measurement noise); `TOTAL` differs
by ~0.8%, within the documented node-to-node/run-to-run variance for this
platform (CURRENT_STATE.md §8: "Node-to-node A100 variance is large (>2x at
C512)" — this is a same-node rerun, so the observed 0.8% is unremarkable).
Step count is identical. No new numerical validation infrastructure was
built for this (out of scope for a WIRES-ONLY task); this is a
production-shape smoke-level neutrality check, not a bitwise/savepoint
comparison.

**Stage 11: PASS (behavior-neutral within measurement noise).**

## 8. Stage 12 — Re-Verification of All Four >6-Rank Gates

Re-read (not just diffed) each gate after the edits:

1. `cuda/interface/fv3_gpu_dyncore_bridge.F90:168`:
   `if (comm_size /= 6) then` -> FATAL. **Unchanged.**
2. `build_rank_of_tile`'s existing one-rank-per-tile FATAL at line ~1516
   (allgathered tile map cross-check). **Unchanged** — a second, new,
   *independent* FATAL was added a few lines earlier in the same subroutine
   for layout-consistency (not tile-consistency); it does not touch or
   weaken the pre-existing tile-map FATAL.
3. `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp:1585`:
   `if(c->data.create.mpi_size!=halo::cubed_sphere_tiles)` -> reject.
   **Unchanged**, and the new `t.layout_x=...`/`t.layout_y=...` assignments
   execute strictly *after* this check and after the unchanged
   `make_cubed_sphere_topology` call.
4. `cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp`'s production `comm_size==6`
   gate in `create_mpi_transport()`. **Untouched** (not referenced by any
   edit in this task).

**Stage 12: PASS — all four gates confirmed intact.**

## 9. Stage 13 — No Sub-Tile Derivation in Production

Confirmed by construction: no `local_rank_within_face`, `sub_i`, `sub_j`, or
any derived within-tile index was added anywhere in this task's edits. The
only new runtime state is the two flat integers `layout_x`/`layout_y`,
stored and cross-checked, never used to compute a neighbor, an offset, or a
transport peer.

## 10. Stage 14 — Future Contract Note

The eventual sub-tile decomposition contract (not implemented here) is
expected to take the form, for a rank `r` within a face under a
`layout_x * layout_y` grid of sub-tiles per face:

```
local_rank_within_face = <FMS within-tile rank ordering, UNVERIFIED>
sub_j = local_rank_within_face / layout_x
sub_i = local_rank_within_face mod layout_x
```

**RESOLVED by Prompt 115 (2026-10-01):** the assumption below is now
VERIFIED, not just assumed. A Fortran-only `layout=(2,2)`/24-rank diagnostic
(`RUN_FV3_LOG_FMS_LAYOUT_DIAG`-gated print in `domain_decomp`) confirmed
`local_rank_within_tile = sub_j*layout_x + sub_i` exactly, with zero
exceptions across all 6 tiles, plus tile-major-contiguous grouping (4
ranks/tile) and `mpp_pe()==MPI_Comm_rank()` identity. See
`docs/performance/p115_fms_within_tile_rank_ordering_2026-10-01.md` for full
detail. The paragraph below is retained for historical context only.

~~Unverified assumption, flagged again (unchanged from Prompt 113):~~ FMS's
actual within-tile rank ordering (row-major/x-fastest vs. column-major, and
whether it is contiguous per-face at all under the real `mpp_domains`
decomposition) has not been confirmed from FMS source — FMS is not vendored
in either this repo or `GEOSfvdycore_fortran_perf`. Prompt 113 recommended a
runtime cross-check via an additional `MPI_Allgather` of `comm_rank`
alongside grid coordinates; that check has NOT been implemented in this
task (would require deriving actual `(sub_i, sub_j)` from FMS's own
decomposition metadata, which is a separate, larger task — see §11 Option A
below).

## 11. Stage 15 — Next Task Selection

**DONE (Prompt 115, 2026-10-01): Option A was executed and VERIFIED** — see
`docs/performance/p115_fms_within_tile_rank_ordering_2026-10-01.md`.
**DONE (Prompt 116, 2026-10-01): the resulting design task was executed** —
see `docs/performance/p116_subtile_id_derivation_design_2026-10-01.md`
(canonical derivation formula, C++-runtime location decision, NO new ABI
field, exact 1×1 reduction proof, 24-rank oracle, recommended bounded next
slice: derivation helper + 1×1 shadow-topology check + test-only oracle
unit test — no gate relaxation). The recommendation text below is retained
for historical context.

**Recommended: Option A** — verify the real FMS within-tile rank ordering
empirically via a Fortran-only test at `layout=(2,2)` (8 sub-tiles/face x 6
faces = not a real constraint since this is Fortran-only, no CUDA bridge
involvement) by printing `Atm%layout`, each rank's `(is,ie,js,je)` window,
and `mpp_pe()`-relative ordering, and comparing across ranks to determine
whether within-face rank assignment is contiguous/row-major. This resolves
the §10 unverified assumption without touching the CUDA bridge, the
`comm_size==6` gate, or any topology builder — it is strictly a
read-only Fortran diagnostic, matching the spirit of "WIRES ONLY" carried
into the next task.

**NOT Option C** (implementing `make_subtile_cubed_sphere_topology` as the
production topology builder) — explicitly deferred per Prompt 114's Do-NOT
list and per Prompt 113's own recommendation; the within-tile ordering
assumption must be verified (Option A) before any topology-builder switch
is safe to attempt.

Lower priority, deferred (unchanged from Prompt 113): (a) Phase B
corner-transport design; (b) folding Prompt 109's duplicated vector-sign
formula into one shared helper.

## 12. Production Safety Statement

- Default production behavior (`layout=(1,1)`, `comm_size==6`,
  one-rank-per-tile) is **unchanged** — GPU-executed and confirmed identical
  in shape (step count) and comparable in timing (within documented
  node-variance) to the pre-P114 binary at both C32 and C512 production
  scale.
- No production topology, transport, or numerical code path was altered.
- All four >6-rank production gates remain intact and were independently
  re-read after the edits (§8).
- No commit, no push.

## 13. Unresolved / Honest Caveats

- `fv3_gpu_interface_mod.F90`'s mirrored edits are structurally consistent
  with the compile-verified active-bridge pattern but were **not
  independently compiled** this session (not part of the production
  fixture's build). Flag before relying on this file for any future task.
- Stage 11's neutrality check is a smoke-level (step-count + timer-magnitude)
  comparison, not a savepoint/bitwise numerical comparison — appropriate for
  a WIRES-ONLY task with no numerical-behavior change expected, but should
  not be cited as "NUMERICALLY VALIDATED" in the project's four-level
  terminology; it is GPU EXECUTED with a behavioral-neutrality smoke check.
- **RESOLVED by Prompt 115** (2026-10-01): the FMS within-tile rank-ordering
  assumption (§10) is now VERIFIED, not open — see
  `docs/performance/p115_fms_within_tile_rank_ordering_2026-10-01.md`.
