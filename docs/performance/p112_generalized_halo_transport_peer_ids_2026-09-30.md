# Prompt 112 — Generalize Halo Transport Peer Identifiers Beyond 6, Bridge/Gates Closed

Status: **CLOSED.** Edge-transport peer-identity generalization IMPLEMENTED,
COMPILED, GPU EXECUTED, NUMERICALLY VALIDATED at a real 24-rank MPI scale.
Bridge/production `comm_size==6` gates untouched by construction. No commit/
push. No numerical/seam-math change. No R4 change. No corner production
transport.

## 1. Goal

Prompt 111 found the real blocker for sub-tile (>6-rank) halo topologies:
`launch_halo_transfer`'s `peer_send` table was indexed directly by
`EdgeNeighbor.tile` (physical cube face, 1..6), and `PeerHaloTransport`/
`MpiHaloTransport`'s peer storage was fixed-size-6. This conflated two
distinct identities that happen to coincide only in the current 6-rank/
one-rank-per-face production configuration:

- **Physical face identity** (`EdgeNeighbor.tile`, 1..6) — a seam-math fact,
  permanent, never generalizes past 6 (a cube has 6 faces).
- **Transport peer identity** (`EdgeNeighbor.rank`, 0..comm_size-1) — which
  MPI rank/peer owns the donor data, which *does* need to generalize once
  more than one rank can share a face (sub-tile decomposition, C1024+).

Prompt 112's goal: separate these two identities consistently through the
transport data path, so the transport machinery is *capable* of representing
peer counts beyond 6 — validated with a real synthetic >6-rank MPI run —
while every real production gate (Fortran bridge `comm_size==6`, MPI
factory's `create_mpi_transport` `comm_size==6`) remains fully in force and
completely unreachable-past-6, exactly as before this prompt.

## 2. Semantic change

`EdgeNeighbor.tile` now means, everywhere in the touched code paths, the
donor's **physical cube face** (1..6), and `EdgeNeighbor.rank` means the
donor's **transport peer identity** (0..peer_count-1). Before this prompt,
`launch_halo_transfer` used `.tile` for both purposes at once (peer lookup
*and* implicit face identity), which is what capped it at 6 slots. No seam
rotation, sign, rotation-detection, or cubed-sphere geometry math changed —
only which field of `EdgeNeighbor` the transport layer's buffer-lookup code
reads.

## 3. BUILDER

`fv3_gpu_halo_subtile_view.hpp`'s `make_subtile_cubed_sphere_topology`
already set `nb.rank = donor_rank; nb.tile = sn.donor.face;` (done in a
prior session/window as Stages 3-8 of this same prompt's plan). Re-verified
this window via full-file read: the assignment is unconditional for every
edge, at every layout, and corner-field population remains metadata-only
(§ "corner scope" below) — no change made this window.

## 4. TRANSFER

`launch_halo_transfer` (`fv3_gpu_halo_exchange.cu:269-312`) now:

- indexes `peer_send` by `nb.rank` (0-based, bound-checked against the
  caller-supplied `peer_count`), not by `nb.tile`;
- accepts `peer_count` as an explicit parameter instead of assuming
  `cubed_sphere_tiles` (6);
- is otherwise byte-for-byte the same slot arithmetic, pack/unpack
  boundary, and D2D `cudaMemcpyAsync` call it always was.

Production call sites (`fv3_gpu_runtime.cpp:864-874`) still build a
6-entry `send_by_rank` table and pass `peer_count = halo::cubed_sphere_tiles`
— unchanged behavior, verified byte-identical via the mandatory layout=1x1
no-regression check (§8).

## 5. PEER TRANSPORT

- **`PeerHaloTransport`** (`fv3_gpu_halo_peer.hpp`): untouched this prompt.
  Its member table remains `PeerMember member_[cubed_sphere_tiles+1]`,
  indexed by physical face — intentional, in scope for the in-process
  backend which still assumes exactly one member per face. Not part of
  Prompt 112's change budget (no design doc or test claims >6 support for
  this backend).
- **`MpiHaloTransport`** (`fv3_gpu_halo_mpi.cpp`, generalized in a prior
  session/window, re-verified this window via full-file read):
  `int size_` (measured from the communicator, not hardcoded 6) and
  `std::vector<int32_t> tile_of_rank_` (dynamically sized, not
  `std::array<int32_t,6>`). `attach()`/`check_topology_once()` validate
  against `tile_`/`tile_of_rank_` generically, independent of peer count.

## 6. FIELD-DATA VALIDATION (Stages 9-12, prior window)

- layout=1x1 and layout=2x2 in-process field-data harness
  (`test_halo_subtile_exchange.cu`): **0/102,402 checks failed** combined.
- Mandatory 1x1 bitwise-identical no-regression check against the legacy
  6-tile path: **0/20,484 failures**.
- 96-edge-case coverage and vector U/V sign/rotation regression: clean.

## 7. 24-RANK SYNTHETIC REAL MPI (Stage 13, this window)

New test `cuda/dyn_core/tests/integration/test_halo_subtile_mpi_transport.cu`
+ CMake target `fv3_halo_subtile_mpi_transport_test` (registered inside the
existing `if(FV3_DYN_CORE_ENABLE_MPI_HALO)` block in `CMakeLists.txt`).

Combines `test_halo_mpi_transport.cu`'s real cross-process MPI mechanics
with `test_halo_subtile_exchange.cu`'s sub-tile topology and rank-keyed
provenance oracle (`value(id,i,j,k)` encoding/decoding, keyed on global
rank because at layout=2x2 four ranks can share one face — a face-keyed
oracle cannot disambiguate them). Uses **`create_mpi_transport_for_testing`
exclusively** — the Stage-13 test-only factory that bypasses
`create_mpi_transport`'s production `comm_size==6` gate entirely; the
production factory and its gate were re-confirmed via source read to be a
completely separate, untouched code path (`fv3_gpu_halo_mpi.cpp:426-427`
vs. `:507`).

- 24 = `cubed_sphere_tiles * layout_x(2) * layout_y(2)` real MPI ranks,
  oversubscribed onto one A100 (`mpirun --oversubscribe -np 24`).
- Deterministic rank↔face map via `subtile_from_global_rank` (no MPI
  communication needed to build `tile_of_rank`).
- Two halo groups exercised per rank: `H_W_IN` (scalar) and `H_DWIND_IN`
  (vector, tests seam sign/rotation).
- Result: **GPU EXECUTED, PASS — checks: 22224, failures: 0.**
- One self-caught, self-fixed bug during development: the verify loop
  initially read `TileArray::at()` (host-side data) without first calling
  `TileArray::fetch()` (D2H copy-back) — caught by the compiler's own
  "declared but never referenced" warning on `fetch()` before ever running
  the test on a GPU; fixed by inserting the missing `fetch()` calls before
  the verify loop. Rebuild was then 100% warning-clean.

## 8. SANITIZER (Stage 14, this window)

`compute-sanitizer --tool memcheck --error-exitcode 1`, one combined Slurm
GPU allocation, three binaries:

| Binary | Result |
|---|---|
| `fv3_halo_subtile_exchange_test` (1x1+2x2 in-process field-data) | checks 102402, failures 0, **0 errors** |
| `fv3_halo_exchange_multitile_test` (legacy 6-tile regression) | checks 20484, failures 0, **0 errors** |
| `fv3_halo_subtile_mpi_transport_test` (new, 24-rank real MPI) | checks 22224, failures 0, **0 errors on all 24 ranks** (24 separate `ERROR SUMMARY: 0 errors` lines) |

## 9. REMAINING FIXED SIXES (Stage 15)

Full grep sweep of every file touched by this prompt's change budget, plus
the two production gates, classified:

**INTENTIONAL PRODUCTION GATE (untouched, must remain, re-verified by
source read):**
- `cuda/interface/fv3_gpu_dyncore_bridge.F90:163` — `if (comm_size /= 6)`.
- `cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp:426-427` —
  `create_mpi_transport`'s `size != cubed_sphere_tiles` refusal.
- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp:1579` —
  `c->data.create.mpi_size != halo::cubed_sphere_tiles`.

**PHYSICAL 6-FACE CONSTANT (correct, permanent — a cube has 6 faces; these
are face-ID range checks, not peer-count limits):**
- `fv3_gpu_halo_exchange.cu:351,377` — `launch_halo_unpack`'s
  `nb.tile < 1 || nb.tile > cubed_sphere_tiles` sanity bound (see §"doc
  corrections" below — this reads `.tile` but never indexes with it).
- `fv3_gpu_halo_mpi.cpp:270,414-415,456-457,493-494` — face-ID bound checks
  in `MpiHaloTransport`.
- `fv3_gpu_halo_subtile_view.hpp:92` — `me.face` bound check.
- `fv3_gpu_halo_peer.hpp:95,105,123,125,191` — `PeerHaloTransport`'s
  intentionally-still-fixed-6 member table (§5, out of scope).
- `fv3_gpu_runtime.cpp:1240` — public-API `tile` bound check.
- `test_halo_subtile_mpi_transport.cu:62` — `kNumRanks = cubed_sphere_tiles
  * kLayoutX * kLayoutY`, the deliberate multiplicand proving
  generalization, not a residual limit.

**BUG-UNGENERALIZED TRANSPORT: none found.** Every per-rank storage
structure that changed scope under this prompt (`MpiHaloTransport::
tile_of_rank_`, `launch_halo_transfer`'s peer-lookup indexing) was exercised
at 24 real MPI ranks with 0/22,224 failures and 0 sanitizer errors on every
rank; no remaining literal `6`/`cubed_sphere_tiles` reference silently
truncates a >6-peer exchange.

## 10. Documentation corrections made this window (Stage 15 fallout)

`p110_*.md` §"Where the transport touches TopologyView" and `p111_*.md` §6
both previously claimed `launch_halo_unpack` "never reads `.tile`." Re-audit
found this inaccurate: `launch_halo_unpack` reads `nb.tile` at an identical
`1..cubed_sphere_tiles` sanity-bound check (`fv3_gpu_halo_exchange.cu:
351,377`, both fused- and unfused-edge paths) — it just never uses `.tile`
as a peer-buffer array index (only `.edge` drives rotation/sign/donor-slot
math), which is the property that actually mattered for Prompt 111's
diagnosis. Both docs corrected in place this window; no behavior changed by
the correction, only the written claim.

## 11. Corner scope (Stage 16)

No corner production transport logic was touched by this prompt or its
prior-window predecessor. `fv3_gpu_halo_subtile_view.hpp`'s own header
comment (lines 54-64, re-read this window) explicitly documents that
corner-field population is metadata-only ("NOT a corner MESSAGE
mechanism... makes no corner-transport claim," Prompt 111 Stage 9 scope
boundary) — neither `launch_halo_pack` nor `launch_halo_unpack` reads any
corner field. The new Stage-13 test's own printed output explicitly states
"NOT covered here: corner halo blocks." Corner transport remains
not implemented by design, unchanged from Prompt 111.

## 12. Production safety statement

- Fortran bridge `comm_size==6` gate: **unchanged, verified via source
  read**, still the only path from GEOS/FV3 into the CUDA dyn_core.
- `create_mpi_transport`'s production `comm_size==6` gate: **unchanged,
  verified via source read**, and its code path never calls or is called by
  `create_mpi_transport_for_testing`.
- No production default changed. No production call site passes
  `peer_count != cubed_sphere_tiles` or a rank-count other than 6.
- No CUDA-aware MPI enabled (`TransportMode::host_staged` used throughout
  the new test, matching every other project default).
- R4 untouched — all Stage 9-14 work is R8-only, matching every other
  sub-tile-halo prompt to date.
- No commit, no push. Working tree still holds only the uncommitted diffs
  listed in `CURRENT_STATE.md` §10, plus this prompt's new test file and
  one `CMakeLists.txt` addition.

## 13. Bridge metadata readiness (Stage 17)

**NO — not ready to implement, by design and by explicit instruction.**
The transport *machinery* (builder, transfer, peer storage) is now proven
capable of representing peer identity beyond 6, validated at real 24-rank
MPI scale. What remains entirely undesigned is the actual metadata
plumbing: extracting real `layout_x`/`layout_y` from the FMS domain
decomposition and threading them across the Fortran/C ABI into the bridge
(`p110_*.md` §11 flagged the *data already exists upstream* in FMS's own
domain object, but the bridge today only reads/passes `comm_size` +
`comm_rank` + the fixed 6-entry `rank_of_tile`). That extraction, its ABI
surface, and the bridge's own `comm_size==6` gate relaxation are all
explicitly out of this prompt's change budget and require dedicated
fresh-authorization design work before any code is written.

## 14. Next task (Stage 18)

Two remaining candidates, exactly one to select:

- **Phase A — bridge metadata plumbing design.** Design (not implement)
  extraction of `layout_x/layout_y` from the FMS domain and its ABI
  surface into the bridge, plus the generalization of the bridge's own
  `comm_size==6` gate. This is the natural continuation now that the
  transport layer beneath it is proven generalized — `CURRENT_STATE.md`
  §9 already flags this as the recommended next step (there called "Phase
  6"). **Recommended.** It is a production-dispatch change and must not be
  attempted without explicit fresh sign-off, per this prompt's own
  constraints and the standing project rule against modifying production
  gates without authorization.
- **Phase B — corner transport design.** Design (not implement) a real
  corner message-delivery mechanism, distinct from the existing
  metadata-only corner-flag population. Lower priority: C1024+ readiness
  depends on the edge path (now generalized) more than the corner path,
  which is a smaller fraction of total halo traffic and was already
  deferred twice (Prompt 110 §9, Prompt 111 §7).

Recommendation: **Phase A**, scoped strictly as a design/mapping exercise
(mirroring Prompt 108's approach to the sub-tile topology itself) — do not
begin any bridge or gate code without a fresh, explicit authorization
request first.

```
GENERALIZED HALO EDGE TRANSPORT VALIDATED — READY FOR PHASE A (BRIDGE
METADATA PLUMBING DESIGN, FRESH AUTHORIZATION REQUIRED BEFORE
IMPLEMENTATION)
```

## 15. Update (Prompt 113, CLOSED) — Phase A design delivered

Phase A (bridge metadata plumbing design) was completed as Prompt 113:
`Atm%layout(1:2)` identified as the authoritative, already-resident
runtime source for `layout_x/layout_y`; the bridge call site
(`fv3_gpu_bridge_init`, already receiving the full `Atm`) needs zero new
plumbing to reach it; a second, previously undocumented one-rank-per-tile
gate found inside `build_rank_of_tile` (`fv3_gpu_dyncore_bridge.F90:
1451`) alongside the known top-level `comm_size==6` check; minimal
additive ABI extension and a bounded, SMALL-complexity next task
(wire-only, no gate relaxation) specified. See
`p113_fms_bridge_subtile_metadata_design_2026-09-30.md`.
