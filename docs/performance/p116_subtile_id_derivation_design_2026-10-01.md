# Prompt 116 — Production Sub-Tile ID Derivation Design

Date: 2026-10-01
Branch: `optimization/cuda-fv3-performance`
Scope: DESIGN ONLY. No production source changes, no ABI changes, no gate
relaxation, no topology-builder switch, no real >6-rank CUDA run, no corner
transport, no seam/vector-math change, no CUDA-aware MPI, no production
default change, no R4 touch, no commit/push.

## 0. Context Read

- `CLAUDE.md`, `docs/agent_instructions/discover_platform.md`,
  `docs/agent_instructions/project_architecture.md`: present, read in prior
  sessions, re-applied here.
- `docs/agent_instructions/cost_efficiency.md`: **does not exist** in this
  repo (`ls` confirmed `No such file or directory`). Non-blocking — Stage 0
  said "if present"; cost-efficiency guidance is instead embedded directly in
  `CLAUDE.md`'s own "Cost-Efficient Agent Operation" section, already applied
  (targeted `grep`/`Read`, no broad scans).
- `docs/agent_handoff/CURRENT_STATE.md`, `p113_*`, `p114_*`, `p115_*`: all
  read. §12 of each already anticipated this exact task (P114 §11, P115 §15).

## 1. Stage 1 — Available Inputs Table

Inspected at the two real production call sites: `fv3_gpu_bridge_init`
(`cuda/interface/fv3_gpu_dyncore_bridge.F90`) and
`fv3_gpu_set_cubed_sphere_topology` (`cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`,
signature in `cuda/dyn_core/include/fv3_gpu_api.h:308`).

| Value | Available? | Source |
|---|---|---|
| `comm_rank` | YES, Fortran | `MPI_Comm_rank(comm, comm_rank, ierr)` in `fv3_gpu_bridge_init` |
| `comm_rank` (as `mpi_rank`) | YES, C++ | `fv3_gpu_create_desc.mpi_rank` (pre-existing ABI field, `fv3_gpu_api.h:103`) — crosses the ABI **unchanged**, same integer `MPI_Comm_rank` produced Fortran-side |
| `comm_size` | YES, both | `fv3_gpu_create_desc.mpi_size` (`fv3_gpu_api.h:104`); gate source for all 4 >6-rank checks |
| `Atm%tile` (face, 1-based) | YES, Fortran | `Atm%tile`, set at `fv_mp_mod.F90:662`; already read in `fv3_gpu_bridge_init` |
| `tile` | YES, C++ | passed as a **direct, pre-existing argument** to `fv3_gpu_set_cubed_sphere_topology(ctx, int32_t tile, ...)` (`fv3_gpu_api.h:308`) — predates Prompt 109, not a new field |
| `layout_x`/`layout_y` | YES, both | `Atm%layout(1:2)` -> `fv3_gpu_create_desc.layout_x/layout_y` (Prompt 114 ABI fields, `fv3_gpu_api.h:117-118`), already stored into `TopologyView` |
| `pelist(1)` | **NO**, neither side | Not read anywhere in `fv3_gpu_bridge_init`; not an ABI field; not needed — see Stage 3 |
| `rank_within_tile`/`sub_i`/`sub_j` | **NO** | Not computed anywhere today — this is the gap this task designs |

**Conclusion: every input the canonical formula needs (`mpi_rank`, `tile`,
`layout_x`, `layout_y`) already exists on both sides of the ABI, unchanged
since Prompt 114.** No new metadata needs to be read from FMS and no new ABI
field is required (Stage 11 confirms this from the struct definition itself).

## 2. Stage 2 — Derivation Location: C++ RUNTIME

**Decision: C++ runtime (`fv3_gpu_runtime.cpp` / a new pure function in
`fv3_gpu_halo_subtile_topology.hpp`), NOT the Fortran bridge.**

Reasons, from source:

1. All four required inputs cross the ABI unchanged (Stage 1) — there is
   nothing the Fortran side can compute that the C++ side cannot compute
   identically from the same integers it already receives.
2. The C++ runtime already owns topology construction. The existing
   `make_subtile_cubed_sphere_topology(SubTileId me, layout_x, layout_y,
   npx_local, npy_local, out)` (`fv3_gpu_halo_subtile_view.hpp:89`) takes an
   **already-derived** `SubTileId` — it does not derive one. The missing
   piece is strictly "given `(mpi_rank, tile, layout_x, layout_y)`, produce a
   `SubTileId`" — a pure function that belongs beside the other pure
   `SubTileId` algebra (`subtile_rank_within_face`, `subtile_global_rank`,
   `subtile_from_global_rank`) already in
   `fv3_gpu_halo_subtile_topology.hpp` (built Prompt 109).
3. Keeping the arithmetic in one language avoids a second duplicated-formula
   risk. `docs/performance/p113_*.md`/`p114_*.md` already flag one open
   duplicated-formula item (Prompt 109's vector-sign formula, §9/§11 "lower
   priority, deferred"); mirroring the rank->SubTileId formula into Fortran
   as well would create a second one for no benefit, since Fortran has no
   use for the derived value today (the gate it would feed,
   `fv3_gpu_set_cubed_sphere_topology`'s topology construction, lives in
   C++).
4. The Fortran-side `build_rank_of_tile` already has a clear, narrower role
   (cluster-wide **validation** via `MPI_Allgather`, not per-rank
   derivation) — see Stage 13. Derivation is a local, no-communication
   computation; validation is a cluster-wide, communication-requiring one.
   Conflating them in one subroutine would blur that separation.

## 3. Stage 3 — Is `pelist(1)` Required? NO.

**Decision: NO. Zero-based `tile_start = (tile-1)*layout_x*layout_y` is safe
by construction in this architecture.** Documented, not assumed:

- **Structural argument**: `GEOSfvdycore_fortran_perf`'s standalone driver
  initializes MPI exactly once; the same world communicator is threaded
  through ESMF's VM, FMS's `pelist`, and the CUDA bridge's `comm`
  (`fv3_gpu_dyncore_bridge.F90`'s `comm` argument, passed through unmodified
  via `MPI_Comm_f2c`). No sub-communicator split exists anywhere in the
  current architecture between "the communicator FMS numbers `pelist` over"
  and "the communicator the CUDA bridge's `mpi_rank` is drawn from" — they
  are the same communicator. For a single, unsplit world communicator,
  `pelist(1)` is 0 by MPI/FMS construction (the first rank in the full
  pelist is rank 0). This was first argued in `p113_*.md` §6 ("moderate
  confidence... translation required: NO") and is now empirically closed.
- **Empirical argument**: Prompt 115 directly measured `rank == pe ==
  mpirank` identically for every one of 30 collected lines (6 control ranks
  + 24 test ranks, layout=(1,1) and layout=(2,2)) — zero offset, zero
  reordering (`p115_*.md` §7, Stage 9). This is the same `comm_rank` the ABI
  passes as `mpi_rank` today (both are `MPI_Comm_rank` on the one shared
  communicator).
- **Caveat, not a blocker**: if a future refactor ever introduces a
  CUDA-bridge-specific sub-communicator (none exists today, no plan to add
  one), this assumption would need re-verification. Stage 5's generalized
  validation (below) gives a cheap, always-on runtime tripwire for exactly
  this failure mode — it does not require `pelist(1)` as an input to detect
  a violation.

**No new ABI field for `pelist(1)` is proposed.**

## 4. Stage 4 — Canonical Formula

Directly from Prompt 115 §12, restated with explicit integer types/bounds for
the C++ implementation:

```cpp
// All inputs int32_t. tile is 1-based (face, matches TopologyView::tile).
// Returns false (SubTileId left default) on any out-of-range input.
bool subtile_id_from_rank(int32_t mpi_rank, int32_t tile,
                           int32_t layout_x, int32_t layout_y,
                           SubTileId &out) noexcept {
  if (tile < 1 || tile > cubed_sphere_tiles) return false;      // face in [1,6]
  if (layout_x < 1 || layout_y < 1) return false;
  const int64_t ranks_per_tile = int64_t(layout_x) * int64_t(layout_y);
  const int64_t tile_start = (int64_t(tile) - 1) * ranks_per_tile;   // pelist(1)=0
  const int64_t rank_within_tile = int64_t(mpi_rank) - tile_start;
  if (rank_within_tile < 0 || rank_within_tile >= ranks_per_tile) return false;
  out.face  = tile;
  out.sub_i = static_cast<int32_t>(rank_within_tile % layout_x);
  out.sub_j = static_cast<int32_t>(rank_within_tile / layout_x);
  return true;
}
```

`int64_t` intermediate arithmetic avoids any overflow concern at realistic
`layout_x*layout_y` values; final `sub_i`/`sub_j` are bounds-safe once the
`rank_within_tile` range check passes. This is a **pure function**: no MPI
call, no global state, matching `make_subtile_cubed_sphere_topology`'s own
contract (`cuda/dyn_core/include/fv3_gpu_halo_subtile_view.hpp:89`, "Returns
false... on any malformed input").

## 5. Stage 5 — Validation Contract (Design Only)

Local (C++ runtime, no communication — enforced inside
`subtile_id_from_rank` itself, shown above):

- `tile` in `[1, cubed_sphere_tiles]`.
- `layout_x >= 1`, `layout_y >= 1`.
- `0 <= rank_within_tile < layout_x*layout_y` — this is the load-bearing
  check: it is what would catch a `pelist(1) != 0` or non-contiguous-block
  violation of Stage 3's assumption, with no `pelist(1)` input needed to
  detect it.

Cluster-wide (Fortran, future generalization of the existing
`build_rank_of_tile` allgather — NOT implemented this task): extend the
existing `(tile, layout_x, layout_y)` `MPI_Allgather` to also collect
`comm_rank` (already trivially known per rank, no new communication
pattern), and assert that for every tile `t`, the set of `comm_rank` values
claiming it is **exactly** the contiguous block
`{(t-1)*ranks_per_tile, ..., t*ranks_per_tile - 1}`. This generalizes the
existing one-rank-per-tile FATAL (today's degenerate `ranks_per_tile=1`
case) and gives defense-in-depth validation of the same zero-based
contiguity assumption Stage 3 argues for, without ever reading `pelist(1)`.
A future `mpi_size == ranks_per_tile * cubed_sphere_tiles` assertion
(generalizing today's `fv3_gpu_runtime.cpp:1585` check) is the natural
companion gate — **not implemented**, consistent with the Do-NOT list.

## 6. Stage 6 — 1×1 Exact Reduction: YES

At `layout_x = layout_y = 1`: `ranks_per_tile = 1`, `tile_start = tile-1`,
`rank_within_tile = mpi_rank - (tile-1)`. Today's existing
`build_rank_of_tile` one-rank-per-tile FATAL already guarantees exactly one
rank per tile with `rank_of_tile[t-1] = t-1` (the convention cited directly
in `fv3_gpu_halo_subtile_view.hpp`'s own header comment, "LAYOUT=1x1 EXACT
REDUCTION"), so for the owning rank `mpi_rank == tile-1`, giving
`rank_within_tile = 0` unconditionally, hence `sub_i = sub_j = 0` for every
one of the 6 faces — i.e. `SubTileId{tile, 0, 0}` exactly. That same header
comment already proves (and Prompt 111's Stage 2 regression tests already
exercise) that `make_subtile_cubed_sphere_topology(SubTileId{face,0,0}, 1, 1,
...)` reduces bit-for-bit to legacy `make_cubed_sphere_topology`'s output.
**No new proof is required — the derivation formula's 1×1 output feeds
directly into an already-proven-exact reduction.**

## 7. Stage 7 — 24-Rank Oracle Table: COMPLETE

Generated directly from the Stage 4 formula at `layout_x=layout_y=2`
(`ranks_per_tile=4`), matching Prompt 115 §4's empirically-observed grouping
exactly:

| mpi_rank | tile (face) | rank_within_tile | sub_i | sub_j |
|---|---|---|---|---|
| 0 | 1 | 0 | 0 | 0 |
| 1 | 1 | 1 | 1 | 0 |
| 2 | 1 | 2 | 0 | 1 |
| 3 | 1 | 3 | 1 | 1 |
| 4 | 2 | 0 | 0 | 0 |
| 5 | 2 | 1 | 1 | 0 |
| 6 | 2 | 2 | 0 | 1 |
| 7 | 2 | 3 | 1 | 1 |
| 8 | 3 | 0 | 0 | 0 |
| 9 | 3 | 1 | 1 | 0 |
| 10 | 3 | 2 | 0 | 1 |
| 11 | 3 | 3 | 1 | 1 |
| 12 | 4 | 0 | 0 | 0 |
| 13 | 4 | 1 | 1 | 0 |
| 14 | 4 | 2 | 0 | 1 |
| 15 | 4 | 3 | 1 | 1 |
| 16 | 5 | 0 | 0 | 0 |
| 17 | 5 | 1 | 1 | 0 |
| 18 | 5 | 2 | 0 | 1 |
| 19 | 5 | 3 | 1 | 1 |
| 20 | 6 | 0 | 0 | 0 |
| 21 | 6 | 1 | 1 | 0 |
| 22 | 6 | 2 | 0 | 1 |
| 23 | 6 | 3 | 1 | 1 |

Cross-checked against Prompt 115 §4/§5's measured `(is:ie, js:je)` compute
windows and derived `(sub_i, sub_j)` table — identical row-for-row. This is
the exact table the Stage 10 test-only oracle unit test should assert
against.

## 8. Stage 8 — Builder Migration Strategy: **Option C (Shadow First)**

Three options considered:

- **Option A (immediate replace)**: swap `fv3_gpu_set_cubed_sphere_topology`
  to call `make_subtile_cubed_sphere_topology` unconditionally. Rejected —
  no regression evidence would exist before the switch; violates this
  project's established design-then-implement pattern for anything touching
  the bridge's >6-rank gates (CURRENT_STATE.md §9).
- **Option B (branch on layout)**: call the legacy builder at
  `layout_x=layout_y=1`, the generalized builder otherwise. Rejected for
  now — still requires relaxing the `comm_size==6` gate to ever exercise the
  non-1×1 branch in production, which is explicitly out of scope.
- **Option C (generalized builder, shadow-mode first)**: **RECOMMENDED.**
  Production keeps calling only the legacy builder (gate, builder call, and
  output all unchanged). An env-gated, default-off block additionally
  derives `SubTileId` via Stage 4's helper and runs
  `make_subtile_cubed_sphere_topology` in **shadow mode** — its output is
  compared field-by-field against the legacy `TopologyView` already
  computed, and discarded (not used for any real transport). Preferred
  because it proves the new derivation + generalized builder are exactly
  equivalent on every real production run, at zero behavioral risk, before
  any gate is ever touched.

## 9. Stage 9 — Shadow-Mode Design

Inside `fv3_gpu_set_cubed_sphere_topology` (`fv3_gpu_runtime.cpp`), **after**
the existing `mpi_size != halo::cubed_sphere_tiles` gate and **after** the
existing `make_cubed_sphere_topology` call (both unchanged, matching
Prompt 114's precedent of appending `layout_x`/`layout_y` storage strictly
after these two points):

```
if (env_flag_enabled("RUN_CUDA_FV3_SHADOW_SUBTILE_TOPOLOGY")) {   // default OFF
  SubTileId me;
  if (halo::subtile_id_from_rank(c->data.create.mpi_rank, tile,
                                  t.layout_x, t.layout_y, me)) {
    TopologyView shadow{};
    if (halo::make_subtile_cubed_sphere_topology(me, t.layout_x, t.layout_y,
                                                  npx, npy, shadow)) {
      // compare every TopologyView field (tile, npx, npy, neighbors[4]
      // {rank,tile,edge,orientation,vector_sign_x,vector_sign_y}, *_corner,
      // corner_owner_rank) against `t`; FATAL on any mismatch, debug/test
      // builds only.
    }
  }
}
```

At today's only reachable production configuration (`comm_size==6`,
`layout_x=layout_y=1`), Stage 6 already proves this comparison is exact —
so the shadow check, once implemented, should PASS on every single existing
and future production run with the flag enabled, with zero expected
divergence. This is the regression proof Option C is chosen for.

## 10. Stage 10 — Test-Only 24-Rank Injection (No Real MPI Run)

`subtile_id_from_rank` (Stage 4) is a **pure function** — no MPI call, no
global/static state, four plain integer arguments. A new, test-only unit
test (e.g. `cuda/dyn_core/tests/runtime/test_subtile_id_derivation.cpp`,
host-only, no CUDA/MPI runtime needed) can call it directly with all 24
`(mpi_rank, tile=…, layout_x=2, layout_y=2)` tuples from Stage 7's oracle
table and assert the returned `SubTileId` matches, exactly as
`test_halo_subtile_topology.cpp` already does for `subtile_edge_neighbor`/
`subtile_corner_donor` at synthetic ranks (Prompt 109). This tests the
derivation at full 24-rank diversity on a login node, in a CPU-only unit
test binary, without a real multi-rank MPI job and without touching any
gate — satisfying Stage 10 exactly as specified.

## 11. Stage 11 — ABI Impact: **NO NEW ABI FIELDS**

Confirmed directly from `cuda/dyn_core/include/fv3_gpu_api.h`:

- `fv3_gpu_create_desc` already has `mpi_rank` (line 103), `mpi_size` (line
  104), `layout_x`/`layout_y` (lines 117-118, Prompt 114).
- `fv3_gpu_set_cubed_sphere_topology`'s declared signature (line 308) already
  takes `tile` as a direct `int32_t` argument — this predates Prompt 109 and
  is not part of any prior ABI extension.

Every input Stage 4's formula needs is already present on both sides of the
ABI boundary, unchanged. **This confirms Prompt 113 §9's original
recommendation ("do NOT widen `rank_of_tile`, derive `sub_i`/`sub_j`
CUDA-side instead") and closes the question definitively: zero new ABI
surface is required for this derivation.**

## 12. Stage 12 — Failure/Error-Handling Design

- `subtile_id_from_rank` returns `bool`, matching
  `make_subtile_cubed_sphere_topology`'s own existing contract — `false` on
  any invalid `tile`/`layout_x`/`layout_y`/out-of-range `rank_within_tile`,
  with the output struct left untouched (no partial/garbage `SubTileId`).
- For the Stage 9 shadow-mode slice specifically: shadow derivation only
  ever runs **after** the real `comm_size==6` gate has already accepted the
  call, so by construction `tile`/`layout_x`/`layout_y` are already
  known-valid in every reachable production configuration today. The only
  way `subtile_id_from_rank` could return `false` in that context is a
  latent bug in the helper itself or a (currently unobserved,
  structurally-excluded per Stage 3) `pelist(1) != 0` violation — both
  should FATAL in debug/test builds under the shadow-mode flag, not fail
  silently or fall back to legacy output (legacy output is unaffected
  either way, since shadow mode never feeds real transport).
- No silent fallback-to-legacy path is proposed for a *future* generalized
  gate (out of scope here) — Stage 5/12's design intent for that later work
  is fail-fast at init, consistent with every existing gate's FATAL
  convention in this codebase.

## 13. Stage 13 — `build_rank_of_tile` Future Role

Confirmed, matching Prompt 113's original recommendation (§9/§16): its role
**narrows**, not widens. It remains a **cluster-wide validation** mechanism
(via `MPI_Allgather`, cross-checking that every rank's locally-visible
`(tile, layout_x, layout_y)` — and, in a future generalization, `comm_rank`
contiguity-per-tile, Stage 5 — agrees with every other rank's). It is
explicitly **not**:
- the source of truth for `SubTileId` derivation (that is Stage 4's local,
  no-communication pure function), and
- a transport-peer-rank lookup for sub-tiles (that role already belongs to
  `subtile_global_rank`, existing CUDA-side since Prompt 109).

## 14. Stage 14 — Corner-Transport Dependency: **NO**

SubTileId derivation (Stage 4) and the shadow-topology equivalence check
(Stage 9) depend only on `tile`/`layout_x`/`layout_y`/`mpi_rank` arithmetic
and on `make_subtile_cubed_sphere_topology`'s existing EDGE-field population
(`subtile_edge_neighbor`-derived, GPU-validated at real 24-rank MPI scale per
Prompt 112). The CORNER fields that same builder populates are already
metadata-only (no message mechanism, per `fv3_gpu_halo_subtile_view.hpp`'s
own header comment, "makes no corner-transport claim") — the shadow
comparison would simply compare these already-existing (if
incomplete-by-design) corner fields the same way legacy's own layout=1×1
output already does today. **Corner-transport completion is not a
prerequisite for this task's recommended next slice.**

## 15. Stage 15 — Recommended Next Implementation Slice

**IMPLEMENT PRODUCTION SUB-TILE ID DERIVATION HELPER + 1×1 SHADOW TOPOLOGY
EQUIVALENCE CHECK + 24-RANK TEST-ONLY ORACLE** — exactly the task's own
suggested candidate; source evidence supports it without modification:

1. Add `subtile_id_from_rank(mpi_rank, tile, layout_x, layout_y, out)`
   (Stage 4) to `fv3_gpu_halo_subtile_topology.hpp`, beside the existing
   `SubTileId` algebra.
2. Add a new test-only unit test exercising it against the full 24-rank
   oracle table (Stage 7/10).
3. Add an env-gated (`RUN_CUDA_FV3_SHADOW_SUBTILE_TOPOLOGY`, default OFF),
   debug/test-only FATAL-on-mismatch shadow-topology comparison inside
   `fv3_gpu_set_cubed_sphere_topology`, strictly after the existing gate and
   legacy builder call (Stage 9).

No new ABI field, no gate relaxation, no topology-builder switch for real
traffic, no real >6-rank model run.

**Complexity: SMALL.** One new ~15-line pure function; one new self-contained
unit test file (~24 assertions, following the existing
`test_halo_subtile_topology.cpp` pattern); one small, strictly-additive,
default-off block in `fv3_gpu_runtime.cpp`.

## 16. Stage 16 — File / Change / Risk Table

| File | Change | Risk |
|---|---|---|
| `cuda/dyn_core/include/fv3_gpu_halo_subtile_topology.hpp` | add `subtile_id_from_rank(...)` pure function | SMALL |
| `cuda/dyn_core/tests/runtime/test_subtile_id_derivation.cpp` (new) | 24-rank oracle unit test, host-only, no MPI | SMALL |
| `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` | env-gated (default off) shadow-mode block in `fv3_gpu_set_cubed_sphere_topology`, after existing gate + legacy build | SMALL-MEDIUM (production runtime file touched, but strictly additive and off by default) |
| `cuda/dyn_core/CMakeLists.txt` | register new test target | SMALL |

No file on the explicit Do-NOT list (`fv3_gpu_dyncore_bridge.F90`'s
`comm_size/=6` FATAL, `build_rank_of_tile`'s tile-map FATAL,
`fv3_gpu_runtime.cpp`'s `mpi_size!=cubed_sphere_tiles` gate,
`fv3_gpu_halo_mpi.cpp`'s `create_mpi_transport()` gate) needs to change for
this slice.

## 17. Stage 17 — Stop Conditions: NONE TRIGGERED

- `pelist(1)` ambiguity: **RESOLVED** (Stage 3, structural + empirical).
- Runtime communicator numbering guarantee: **CONFIRMED** (Stage 3, Prompt
  115 §7, ABI already passes `mpi_rank` unchanged).
- Required metadata availability: **CONFIRMED present without further ABI
  expansion** (Stage 1/11, read directly from `fv3_gpu_api.h`).
- 1×1 shadow comparison exactness: **PROVEN** (Stage 6, via the existing
  Prompt 111 header comment + Stage 2 regression tests already in-repo).
- Dependency on relaxing a production gate: **NONE** (Stage 15's slice is
  reachable entirely within the existing `comm_size==6`/`layout_x=layout_y=1`
  production envelope).

**Design is unblocked. Proceed to the Stage 15 slice when authorized.**

## 18. Production Safety Statement

- CUDA production code changed: **NO** (design/documentation only).
- ABI changed: **NO**.
- Bridge/runtime/transport gates changed: **NO**.
- Topology builder switched: **NO**.
- Real >6-rank CUDA model run: **NO**.
- Corner transport implemented: **NO**.
- Seam/vector math changed: **NO**.
- CUDA-aware MPI changed: **NO**.
- Production defaults changed: **NO**.
- R4 touched: **NO**.
- Commit/push: **NO**.
