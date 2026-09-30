# Prompt 111 — Sub-Tile Topology Builder + Field-Data Harness (Edges Only, Layout=2x2)

Branch `optimization/cuda-fv3-performance`. No commit/push. No production
source touched. All work is additive (new header, two new/extended test
files, one CMake test-registration block).

## 1. Goal (as scoped)

Implement Prompt 110's selected Phase-5 slice: (1) an additive
`SubTileId`+layout → `TopologyView` builder; (2) exact layout=(1,1)
equivalence with the legacy one-rank-per-face builder; (3) real
field-data correctness at layout=(2,2) using the EXISTING, UNMODIFIED
production pack/transfer/unpack kernel chain. Edges only — no corner
message. No bridge/gate change, no production dispatch change.

## 2. New builder

`cuda/dyn_core/include/fv3_gpu_halo_subtile_view.hpp` —
`make_subtile_cubed_sphere_topology(SubTileId me, int layout_x, int
layout_y, int64_t npx_local, int64_t npy_local, TopologyView &out)`.
Host-only (not `FV3_HALO_HD`, matching the legacy builder's own
convention — `TopologyView` is one-time-per-run host metadata, never
built per-thread on device). Populates `EdgeNeighbor.{rank,tile,edge,
orientation,vector_sign_x,vector_sign_y}` for all 4 edges and
`{sw,se,ne,nw}_corner`/`corner_owner_rank` for all 4 corners, using only
Prompt 109's `subtile_edge_neighbor`/`subtile_corner_donor`/
`subtile_global_rank` — no new seam or rank algebra. Output type is
exactly `TopologyView`, the same production-consumed struct the legacy
`make_cubed_sphere_topology` produces.

`.tile` is deliberately reinterpreted as `donor_global_rank + 1` (a
1-based slot id), not a cube face id — documented in the header's field-
mapping comment. This choice is exact and lossless at layout=(1,1) (see
§4) and is what §6 below shows to be the actual blocker at layout=(2,2).

## 3. 1x1 METADATA REGRESSION

`cuda/dyn_core/tests/runtime/test_halo_subtile_topology.cpp`, new tests
T12/T13 (host-only, no GPU). T12 checks the new builder's output at
layout=(1,1) against the legacy `make_cubed_sphere_topology` field-by-
field for all 6 faces (edges + corners, exact match required). T13
checks the new builder's output at layout=(2,2) against Prompt 109's
oracle functions directly (not by calling the builder a second time) for
all 24 synthetic ranks. Result:

```
PASS: all sub-tile halo topology tests passed (T1-T13)
```

RC=0, host-only, no GPU required. Ran clean on the login node and again
inside the GPU allocation (identical output, as expected — no CUDA
device code in this file).

## 4. 2x2 TOPOLOGY

Same test file, same T13 pass — 24-rank layout=(2,2) `TopologyView`
construction verified correct against the oracle for every edge and
corner, including the corner-index-vs-struct-field-order mapping
(`subtile_corner_edges` convention 0=SW,1=SE,2=NW,3=NE vs `TopologyView`
declared field order sw,se,**ne**,**nw** — the builder maps
`corner_flag[2]→nw_corner`, `corner_flag[3]→ne_corner` explicitly, not by
naive positional copy). No topology-construction defect found at any
stage of this task.

## 5. FIELD-DATA HARNESS

`cuda/dyn_core/tests/integration/test_halo_subtile_exchange.cu` — new
file, generalizing the existing `test_halo_exchange_multitile.cu`
(Milestone 3A.10) to N = 6·layout_x·layout_y synthetic ranks. Calls the
real, unmodified `launch_halo_pack` → `launch_halo_transfer` →
`launch_halo_unpack` chain for all 6 production halo groups
(`H_DWIND_IN`, `H_CWIND`, `H_W_IN`, `H_DIVGD`, `H_PKC2`, `H_DSCALAR`), at
layout=(1,1) (N=6, Stage-12 regression) and layout=(2,2) (N=24, Stages
5-8). Preserves the independently-transcribed `expected_sign()` oracle
verbatim (anti-circularity precedent from the legacy file — never calls
`seam_component_sign()`/`seam_rotation()` itself).

## 6. CUDA

GPU EXECUTED on a real A100 (`warpa009`, `gpu_a100`/`rome`, job
58661258/58661327) via `srun`, using the validated dev module stack
(GCC 12.3.0 + OpenMPI 4.1.6 + `nvidia/nvhpc-byo-compiler/24.7`).

- **layout=1x1 (N=6), Stage 12 regression: PASS, 0 failures.** All 6
  halo groups, 20736 checks (3744+3744+1728+2016+2160+6912), bitwise
  match against the legacy-builder multitile test's already-proven
  behavior. The new builder's exact 1x1 equivalence (§3) carries through
  the real kernel chain unchanged.
- **layout=2x2 (N=24), Stages 5-8: FAIL, 1245/9021 checks failed, 100%
  attributable to one cause (see below) — NOT a per-group defect.**
  Per-group failure counts (isolated via temporary instrumentation, one
  `srun` rerun): H_DWIND_IN 222, H_CWIND 222, H_W_IN 135, H_DIVGD 135,
  H_PKC2 135, H_DSCALAR 396 — **all six** groups fail, not just the
  vector groups printed first in the un-instrumented run (the original
  run's console output showed FAIL lines only for H_DWIND_IN because the
  test's 25-line print cap, inherited unchanged from the legacy file,
  was exhausted by that group's failures before later groups printed
  theirs — confirmed by delta-instrumenting `failures` around each
  `exchange_group()` call).

### Root cause

`launch_halo_transfer` (`cuda/dyn_core/runtime/fv3_gpu_halo_exchange.cu:
262-279`) contains:

```cpp
const EdgeNeighbor &nb = topo.neighbors[q];
if (nb.tile < 1 || nb.tile > cubed_sphere_tiles)
  return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
              "edge has no neighbour tile; a connected topology is "
              "required before any halo group may be exchanged");
const fv_real *src = peer_send[nb.tile];
```

`cubed_sphere_tiles == 6`. `peer_send` is a flat, 1-indexed table sized
for exactly 6 slots (production's `PeerHaloTransport::member_
[cubed_sphere_tiles+1]`, `fv3_gpu_halo_peer.hpp:191`) — this is not a
one-off check; the identical `cubed_sphere_tiles` bound recurs at
`fv3_gpu_halo_peer.hpp:95,105,123,125` and again, independently, inside
`MpiHaloTransport` at `fv3_gpu_halo_mpi.cpp:262,404,416,440,444-446`. The
whole halo-transport subsystem, at every layer, is built on a fixed
"exactly 6 slots" model.

The new builder's `.tile = donor_rank + 1` is correct and matches this
bound exactly at layout=(1,1) (max value 6 — see §4 of
`fv3_gpu_halo_subtile_view.hpp`'s own header comment and T12/T13). At
layout=(2,2), `.tile` legitimately ranges 1..24 for the 24 synthetic
sub-ranks; any cross-face donor with global rank ≥ 6 produces `.tile` >
6, which `launch_halo_transfer` rejects before touching any field data.
Most sub-ranks have at least one cross-face edge (a corner-position
sub-tile within a face has exactly 2 same-face edges and 2 cross-face
edges), which is why the large majority of ranks fail at every group,
not a specific field or vector/scalar distinction.

This was **not visible from the topology math alone** (§3/§4 both pass):
Prompt 110 §4/§8 correctly found that `launch_halo_pack` needs zero
change, but overstated `launch_halo_unpack`'s independence from
`.tile` — **correction (Prompt 112, Stage 15 re-audit)**:
`launch_halo_unpack` also reads `nb.tile` directly, at an identical
`nb.tile < 1 || nb.tile > cubed_sphere_tiles` bound check
(`fv3_gpu_halo_exchange.cu:351,377`, both the fused- and unfused-edge
paths). The Prompt 110/111 "neither reads `.tile`" claim was wrong;
the accurate statement is narrower: unpack never uses `.tile` **as a
peer-buffer array index** (only `.edge` drives its rotation/sign/donor-
slot math), so the bound check is a pure sanity gate rather than an
indexing operation, and it is always satisfied post-Prompt-112 because
`.tile` is now kept a physical face (1..6) by construction at every
layout. `launch_halo_transfer` — the middle stage, doing the actual
peer-buffer lookup — is the stage that genuinely could not be
exercised unmodified beyond 6 slots, because it indexed `peer_send`
directly by `.tile` (Prompt 112 fixed this by indexing `peer_send` by
`.rank` instead; see the new p112 doc).

## 7. SANITIZER

`compute-sanitizer --tool memcheck` (found at
`.../hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/` — not under
the documented `profilers/` PATH addition; that addition covers `nsys`/
`ncu` only, not `compute-sanitizer`, which ships with the CUDA toolkit
component instead) against all three CUDA test binaries touched by this
task, on the same GPU allocation:

- `fv3_halo_subtile_exchange_test`: **0 errors.** The binary's own exit
  code is a logical FAIL (§6's layout=2x2 result); sanitizer confirms
  this is a clean `FV3_GPU_INVALID_ARGUMENT` return path with no memory
  violation — the gate check happens before any kernel launch for the
  rejected edges.
- `fv3_halo_subtile_topology_cuda_test` (pre-existing Prompt 109 test,
  rebuilt clean, rerun as part of this task's baseline): **0 errors.**
- `fv3_halo_exchange_multitile_test` (pre-existing legacy test,
  rebuilt clean, rerun as baseline): **0 errors.**

## 8. 1x1 REAL PACK/UNPACK REGRESSION

Satisfied by §6's layout=1x1 result: 20736/20736 checks passed, 0
failures, through the real unmodified `launch_halo_pack`/
`launch_halo_transfer`/`launch_halo_unpack` chain, using the new
builder's `TopologyView` output in place of the legacy builder's. This
is the mandatory bitwise proof that the new builder is a correct,
lossless substitute for the legacy one at the one layout where
production semantics are byte-for-byte preserved.

## 9. PRODUCTION SAFETY

Confirmed by direct grep, not assumption:

- `grep -rl "fv3_gpu_halo_subtile_view.hpp\|make_subtile_cubed_sphere_topology"` across
  `cuda/` returns only the header itself and the two new/extended test
  files — **zero production references.**
- Bridge gate (`cuda/interface/fv3_gpu_dyncore_bridge.F90:163`,
  `if (comm_size /= 6) then`) — **textually unchanged.**
- `fv3_gpu_set_cubed_sphere_topology` production call sites
  (`fv3_gpu_dyncore_bridge.F90:304`, `fv3_gpu_interface_mod.F90:399`,
  `fv3_gpu_runtime.cpp:1557`) — **textually unchanged**, no new call
  sites added.
- `MpiHaloTransport::create_mpi_transport`'s gate
  (`fv3_gpu_halo_mpi.cpp:416`, `if (size != cubed_sphere_tiles)`) —
  **textually unchanged.**
- `launch_halo_pack`/`launch_halo_transfer`/`launch_halo_unpack`
  (`fv3_gpu_halo_exchange.cu`) — **textually unchanged.** The §6 root
  cause is a discovered pre-existing constraint of this unmodified code,
  not a change made by this task.

No production dispatch, gate, or numerical-algorithm source was modified
by this task. R4 untouched (all new/edited files are precision-templated
via the existing `fv_real`/`FV3_GPU_FV_REAL_IS_DOUBLE` machinery, exactly
as the legacy files already were).

## 10. Source hygiene (Stage 14)

- No hardcoded `layout_x==2`/`layout_y==2` anywhere in the builder header
  (`fv3_gpu_halo_subtile_view.hpp`) — fully parameterized; the literal
  `2,2` appears only in the test file's `main()` call sites, as intended.
- No duplicated seam math — the builder calls Prompt 109's
  `subtile_edge_neighbor`/`subtile_corner_donor`/`subtile_global_rank`/
  `classify_subtile_corner` exclusively; no re-derivation.
- No dynamic or device allocation in the builder (confirmed by grep — no
  `new`/`malloc`/`cudaMalloc`).
- New builder lives under `cuda/dyn_core/include/`, matching Prompt 109's
  own placement of `fv3_gpu_halo_subtile_topology.hpp` in the same
  directory (established project convention for additive, non-runtime,
  host-side topology-construction headers) — not leaked into
  `runtime/`.

## 11. PHASE RESULT

| Stage | Result |
|---|---|
| 0 Baseline (Prompt-109 regression oracle) | PASS, unchanged |
| 1 Additive builder | IMPLEMENTED, COMPILED |
| 2 1x1 metadata equivalence (T12) | PASS |
| 3 2x2 TopologyView vs oracle (T13) | PASS |
| 4 Field-data harness (real kernels) | IMPLEMENTED, COMPILED |
| 5 Same-face scalar edge | GPU EXECUTED, **FAIL** (gate, §6) |
| 6 Cross-face segmented scalar | GPU EXECUTED, **FAIL** (gate, §6) |
| 7 U/V vector transform | GPU EXECUTED, **FAIL** (gate, §6) |
| 8 All-edge coverage (96 cases/group) | GPU EXECUTED, **FAIL** (gate, §6) |
| 9 No corner claim | Honored — no corner message packed/transferred/unpacked |
| 10 Real A100 GPU execution | PASS — no CPU fallback, `warpa009` |
| 11 compute-sanitizer memcheck | PASS — 0 errors, all 3 binaries |
| 12 1x1 real pack/unpack regression | **PASS — 20736/20736, 0 failures** |
| 13 Production safety | PASS — zero production references/changes |
| 14 Source hygiene | PASS |
| 15 Overall PASS/FAIL | **FAIL at layout=2x2**; **PASS at layout=1x1** |

## 12. NEXT TASK

Phase 6 bridge metadata plumbing (extracting `layout_x/layout_y` from
the FMS domain and passing them across the ABI, per Prompt 110 §11) must
now also account for generalizing `launch_halo_transfer`'s `peer_send`
indexing/gate beyond 6 slots — not a new, separate item, but a larger
scope for the already-planned Phase 6 than Prompt 110 assumed. This is a
production-dispatch change and requires fresh authorization before any
code is written; not attempted here. Until then, sub-tile field-data
correctness beyond layout=(1,1) cannot be demonstrated through the real
kernel chain without either violating the "no production dispatch
changes" constraint or duplicating transfer logic (itself against the
"do not duplicate numerical functionality" core rule) — recommend
treating this as the explicit scope of a future Prompt 112, not
attempting a workaround now.

```
SUB-TILE HALO FIELD-DATA VALIDATION BLOCKED — launch_halo_transfer's
peer_send index is gated to nb.tile in [1, cubed_sphere_tiles=6]
(fv3_gpu_halo_exchange.cu:273), which the layout=2x2 sub-tile builder's
donor-rank-based .tile slot id (correctly, up to 24) exceeds for any
cross-face donor; layout=1x1 (N=6) real pack/transfer/unpack regression
is bitwise-clean (0/20736 failures) and topology-only correctness at
layout=2x2 is fully validated (T12/T13), but real field-data movement at
layout=2x2 is blocked by this one unmodified production gate, not by any
defect in the new builder or test harness.
```
