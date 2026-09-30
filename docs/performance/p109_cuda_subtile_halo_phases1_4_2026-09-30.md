# Prompt 109 — CUDA Sub-Tile Halo Topology/Transport Generalization, Phases 1-4

Date: 2026-09-30
Branch: `optimization/cuda-fv3-performance`
Status: **CLOSED — Phases 1-4 VALIDATED**. Not committed/pushed.

## 1. Scope

Prompt 108 (design/mapping only) identified that the CUDA dyn_core bridge's
`comm_size==6` assumption (one MPI rank per cubed-sphere tile) is the binding
constraint for a future C1024 24-rank (2x2-ranks-per-face) decomposition, and
derived an 8-phase implementation plan. This prompt implements and validates
**Phases 1-4 only**: the CUDA-side topology/donor-identification layer for a
synthetic `layout_x x layout_y` (target 2x2, 24-rank) sub-tile decomposition,
at small grid size, on a single node.

Explicitly out of scope (per the standing "Do NOT" list, unchanged from
Prompt 108): relaxing the Fortran bridge `comm_size==6` gate (Phase 6),
Fortran/FMS layout plumbing, halo-fill kernel or `HaloTransport` field-data
generalization (Phase 5+), CUDA-aware MPI, any real 24-rank FV3 model run
(Phase 8), and any change to seam-rotation mathematics, numerical algorithms,
or production defaults. PT/DELZ, REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS,
`State_To_FV`, tracer CUDA, and `FV_To_State` were not touched. R4 was not
touched.

## 2. What Was Built

New additive header, reusing the legacy seam-rotation/mosaic algebra
unchanged:

- `cuda/dyn_core/include/fv3_gpu_halo_subtile_topology.hpp` — `SubTileId`
  (face, sub_i, sub_j); rank<->`SubTileId` bijection
  (`subtile_rank_within_face`/`subtile_global_rank`/
  `subtile_from_global_rank`, face-major numbering matching FMS's
  `pe_start` convention); `subtile_edge_is_face_boundary`;
  `subtile_edge_neighbor` (same-face cardinal step, or cross-face
  partial-segment donor via the unchanged `cubed_sphere_neighbor` mosaic
  lookup); corner taxonomy (`classify_subtile_corner`:
  `intra_face_diagonal` / `mixed` / `true_cube_corner`) and donor
  resolution (`subtile_corner_donor`, canonical precedence: intra-face
  step first, then cross-face); `subtile_corner_reciprocal_index`; a
  collision-free MPI tag scheme (`subtile_edge_tag`/`subtile_corner_tag`).
  Every function takes `layout_x`/`layout_y` as ordinary parameters — no
  hardcoded layout==2 assumption anywhere in this file.

Three new permanent tests:

- `tests/runtime/test_halo_subtile_topology.cpp` — host-only unit tests
  T1-T11: rank bijection, layout=1x1 exact regression against the
  UNCHANGED legacy `make_cubed_sphere_topology` (edges + corners),
  layout=2x2 same-face/cross-face/corner-taxonomy/reciprocity checks.
- `tests/integration/test_halo_subtile_mpi_routing.cpp` — 24-rank real-MPI
  routing test (`mpirun -np 24`). All-nonblocking Irecv/Isend-then-Waitall
  by design (an earlier blocking-`Sendrecv` version deadlocked — see §4).
  Each receiver independently re-derives, from the sender's own identity
  carried in the message, what the exchange must be, rather than trusting
  the wire.
- `tests/runtime/test_halo_subtile_topology_cuda.cu` — CUDA GPU-executed
  smoke test: one thread per of 24 synthetic ranks computes all 4 edge
  neighbors + 4 corner donors on the device, compared bit-for-bit against
  the same computation on the host.

## 3. The Device-Callability Fix

**File**: `cuda/dyn_core/include/fv3_gpu_halo_topology.hpp` and
`cuda/dyn_core/runtime/fv3_gpu_halo_topology.cpp`.

**Original error**: nvcc warning `#20011-D: calling a __host__ function
("cubed_sphere_neighbor") from a __host__ __device__ function
("subtile_edge_neighbor") is not allowed`.

**Root cause**: `cubed_sphere_neighbor` was declared as a plain (non-
`FV3_HALO_HD`) `bool` function in the header, defined in a separately-
compiled, host-only `.cpp`. `subtile_edge_neighbor` (the new header) is
`FV3_HALO_HD` and calls it for every cross-face edge — a majority of edges
at layout=2x2 (every sub-rank has exactly 2 face-boundary edges out of 4).
A `__host__ __device__` function's body must be visible to nvcc in every
translation unit that instantiates it; a separately-compiled host-only
`.cpp` cannot supply device object code for a function declared in a
shared header.

**Change**: moved the mosaic table, its `static_assert` consistency proofs,
and `cubed_sphere_neighbor`'s definition into the header, as
`inline FV3_HALO_HD`. Two intermediate build errors were hit and resolved
while converging on the final form:

1. `error: identifier "...::mosaic" is undefined in device code` — a
   namespace-scope `constexpr` array has no device-side storage unless
   explicitly `__device__`-qualified.
2. `error: An inline __device__ variable must have internal linkage when
   the program is compiled in whole program mode (-rdc=false)` — an
   `inline __device__` array has external-linkage semantics unsupported
   under whole-program (non-`-rdc`) nvcc compilation.

**Final fix**: one `#define`d initializer-list macro generates both a
host-side `mosaic` (`constexpr`, used by `static_assert`s and host code)
and a device-side `mosaic_device` (`__device__ static constexpr`, internal
linkage), so there is exactly one textual source of truth for the table.
`cubed_sphere_neighbor`'s body branches on `#if defined(__CUDA_ARCH__)` to
pick the correct array.

**Numerical semantics changed: NO.** The mosaic table's data and the
seam-rotation/vector-sign algebra are byte-for-byte unchanged; this is
purely a device-storage/linkage mechanical fix, independently re-verified
by rerunning every pre-existing test (see §4).

## 4. Validation Results

| Check | Result |
|---|---|
| `fv3_halo_topology_test` (legacy, Stage 16 regression oracle) | PASS, RC=0, no regression |
| `fv3_halo_subtile_topology_test` (host, T1-T11) | PASS, RC=0 |
| `fv3_halo_subtile_mpi_routing_test` (`mpirun -np 24`) | PASS, RC=0 |
| `fv3_halo_subtile_topology_cuda_test` (real A100, Slurm) | PASS — host/device bit-identical, 24 ranks x 4 edges x 4 corners |
| `compute-sanitizer --tool memcheck` on the CUDA test | PASS — "ERROR SUMMARY: 0 errors" |
| Bridge `comm_size==6` gate | Confirmed untouched (`fv3_gpu_dyncore_bridge.F90:163`, `if (comm_size /= 6)`) |
| Transport-layer 6-rank gate | Confirmed untouched (`fv3_gpu_halo_mpi.cpp:416`, `if (size != cubed_sphere_tiles)`) |

Platform note: `compute-sanitizer` is not on the standard validated-stack
`PATH`; found at
`/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/compute-sanitizer`
(also under a `cuda/11.8/` sibling and `compilers/bin/`) — add its
directory to `PATH` explicitly in the Slurm job shell.

Validation hierarchy achieved for Phases 1-4: **IMPLEMENTED, COMPILED, GPU
EXECUTED, NUMERICALLY VALIDATED** (host-vs-device bit-identical; this is a
topology/routing validation, not a savepoint-vs-Fortran numerical
comparison, since Phases 1-4 move no CUDA field data — see §6).

## 5. Implementation Review (Stage 19)

All six files (`fv3_gpu_halo_subtile_topology.hpp`,
`fv3_gpu_halo_topology.hpp`, `fv3_gpu_halo_topology.cpp`,
`test_halo_subtile_topology.cpp`, `test_halo_subtile_mpi_routing.cpp`,
`test_halo_subtile_topology_cuda.cu`) were read in full and reviewed
against seven defect classes:

- **Hardcoded layout==2 assumptions**: CLEAN. Every production function
  takes `layout_x`/`layout_y` as parameters; only test files hardcode
  `LX=LY=2`, which is in-scope test setup, not a production defect.
- **Scattered rank arithmetic**: CLEAN. All rank math is funneled through
  `subtile_rank_within_face`/`subtile_global_rank`/
  `subtile_from_global_rank`; the MPI test calls these rather than
  reimplementing them.
- **Duplicated seam-rotation formulas**: **FINDING (minor, hygiene
  only)**. `subtile_vector_signs` (`fv3_gpu_halo_subtile_topology.hpp:139-144`)
  duplicates, verbatim, the sign-derivation formula in
  `make_cubed_sphere_topology` (`fv3_gpu_halo_topology.cpp:96-99`) instead
  of factoring it into one shared helper. Both call
  `seam_vector_transform`, and T3 cross-checks the result byte-for-byte
  against the legacy path at layout 1x1, so there is no live correctness
  risk today — but if `seam_vector_transform`'s sign convention ever
  changes, both copies must be updated by hand. Recommended future
  cleanup, not blocking.
- **Fragile message ordering**: CLEAN. The MPI routing test deliberately
  uses all-nonblocking Irecv/Isend-then-Waitall specifically because an
  earlier blocking-`Sendrecv` version (fixed order W/E/S/N on every rank)
  deadlocked — every one of 24 ranks stuck on its first `Sendrecv`, a
  genuine circular wait, not a topology bug (independently confirmed by
  exhaustive offline reciprocity checks before the fix).
- **Corner double-writes**: CLEAN. `classify_subtile_corner` partitions
  the 4 corner indices exhaustively into exactly one of
  `{true_cube_corner (self, no message), mixed, intra_face_diagonal}`
  (proven by T7: exactly 1/2/1 per sub-rank at layout=2x2), and
  `subtile_corner_reciprocal_index` is a proven bijection (T11), so no
  rank ever double-owns a corner write target.
- **Tag collisions**: CLEAN. Edge tags (100-103) and corner tags
  (110-113) are disjoint ranges; each rank sends at most one message per
  own edge/corner index, so it never reuses a tag to the same
  destination; every `Irecv` specifies an explicit source rank, so
  MPI's `(source, tag)` matching cannot cross ordered pairs.
- **Unnecessary allocations**: N/A/CLEAN. The only heap use
  (`std::vector`, <=14 elements/rank) is in the test-only MPI routing
  harness, not in the reviewed production headers; immaterial to the
  topology-only scope of Phases 1-4.

**Verdict: no correctness-affecting defects found.** One minor,
non-blocking hygiene finding (documented above).

## 6. What Phases 1-4 Do NOT Cover

- No CUDA field data movement — no halo-fill kernel
  (`fv3_gpu_halo_exchange.hpp`/`fv3_cgrid_corners.hpp`) was generalized.
  That is Phase 5+.
- No `HaloTransport`/`MpiHaloTransport` production pack/unpack change.
- No bridge `comm_size==6` gate change, no Fortran/FMS layout plumbing
  (Phase 6).
- No real 24-rank/C1024 FV3 model run (Phase 8) — this remains a
  synthetic-rank, single-node, small-grid validation only.

## 7. Recommended Next Task

**Superseded by Prompt 110 (CLOSED, design/scoping only) — see
`docs/performance/p110_subtile_halo_transport_phase5_design_2026-09-30.md`.**
Prompt 110 mapped exactly where production kernels touch the one-rank-
per-face model, found pack needs zero change and unpack needs only
correctly-populated donor metadata, and selected the smallest Phase-5
implementation slice: an additive `SubTileId`+layout to `TopologyView`
builder plus a synthetic-rank field-data test harness (edges only,
layout=2x2, no bridge/gate change). Corners remain a wholly new,
deferred mechanism. That document also corrects one of its own inputs:
the transport-side 6-rank gate lives in `fv3_gpu_halo_mpi.cpp:416`
(`create_mpi_transport()`), not `fv3_gpu_halo_mpi.hpp`. Lower-priority
hygiene, still open: fold the §5 duplicated vector-sign formula into one
shared helper.

## 8. Final Status

```
CUDA SUB-TILE HALO PHASES 1-4 VALIDATED — READY FOR BRIDGE METADATA INTEGRATION
```

(i.e., ready for Phase 5 design work; Phase 6 bridge-gate relaxation and
Phase 8 real multi-rank runs remain separately gated, future tasks.)

**Update (Prompt 111, CLOSED):** the Phase-5 field-data harness built on
this layer's topology math found that `launch_halo_transfer` (not this
file's topology/donor-identification code, which remains fully valid)
hardcodes a 6-tile `peer_send` index bound, blocking real pack/transfer/
unpack verification at layout>1x1. See
`p111_subtile_topology_fielddata_validation_2026-09-30.md` and
`p110_...md` §13.
