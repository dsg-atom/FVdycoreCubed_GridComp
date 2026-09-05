# Prompt 15 — Cross-Process MPI Halo Transport for the CUDA dyn_core

**Date:** 2026-09-04
**Branch:** `cuda/gpu-bringup`
**Git HEAD at time of work:** `eb6d689` ("Optimize CUDA dyn_core launch
orchestration" — committed by the user between sessions; it consolidates the
Prompt 12 `a2b4` restructuring and the Prompt 13 CUDA-Graph launch repair).
**Status:** COMPLETE — the MPI halo transport is IMPLEMENTED, COMPILED, GPU
EXECUTED across six separate MPI processes, and NUMERICALLY VALIDATED against
the in-process `HaloPeerGroup` path (BITWISE IDENTICAL at the dyn_core exit).
**Numerical algorithm changed: NO.**

This pass closes the blocker that stopped Prompt 14
(`prompt_14_standalone_cuda_integration.md`): the only implemented CUDA halo
transport was single-process/six-host-thread ("Model B"), and real production
is six separate MPI ranks (`FV_NX=1`/`FV_NY=6`, one tile per rank).

---

## 1. Per-group logical halo contract

Source of truth: the group table at
`cuda/dyn_core/include/fv3_gpu_halo_exchange.hpp:200-300`. Every group's
`halo_width` is `fv3_domain_halo_width` = 3 except `H_DWIND_SEAM`. `ring_begin`
is 1 (halo rings) except `H_DWIND_SEAM`, which is 0 (owned cells).

| group | fields | stagger | vector role / partner | dest_edges | authoritative call site |
|-------|--------|---------|----------------------|------------|-------------------------|
| `H_DWIND_IN` | `u`, `v` | `u` north-staggered, `v` east-staggered | `u` = slot_x (partner `v`), `v` = slot_y (partner `u`) | `0xF` | `dyn_core.F90:402`, DGRID_NE |
| `H_DWIND_NEXT` | `u`, `v` | same as above | same as above | `0xF` | `dyn_core.F90:1109`, DGRID_NE |
| `H_CWIND` | `uc`, `vc` | **`uc` east-staggered, `vc` north-staggered** | `uc` = slot_x (partner `vc`), `vc` = slot_y (partner `uc`) | `0xF` | `dyn_core.F90:618`, CGRID_NE |
| `H_W_IN` | `w` | center | scalar | `0xF` | `dyn_core.F90:427` |
| `H_GZ_FIRST` | `gz` | center | scalar | `0xF` | `dyn_core.F90:459` (`it==1` only) |
| `H_DIVGD` | `divgd` | corner | scalar | `0xF` | `dyn_core.F90:528` (`nord>0`) |
| `H_DSCALAR` | `q_con`, `delp`, `pt`, `cappa` | center ×4 | scalar ×4 | `0xF` | `dyn_core.F90:844-847` |
| `H_ZH` | `zh` | center | scalar | `0xF` | `dyn_core.F90:931`/`:934` |
| `H_PKC2` | `pkc` | center | scalar | `0xF` | width is **3**, not 2 — the "2" in the name is a documented misnomer |
| `H_HEAT` | `heat_source` | center | scalar | `0xF` | post-loop, live only when `d_con > 0` |
| `H_DWIND_SEAM` (E11) | `u`, `v` | `u` north, `v` east | `u` = slot_x (partner `v`), `v` = slot_y (partner `u`) | `u`: `1<<north`; `v`: `1<<east` | `dyn_core.F90:1178` call, `:1181-1188` write-back; `ring_begin = 0`, `halo_width = 0` |

**`H_CWIND` is the reason the transport must key the vector sign on the
argument slot (`VectorRole`), not on staggering.** `H_DWIND_*` and `H_CWIND`
carry the *same two staggerings in the opposite slots*. A formulation keyed on
"north-staggered ⇒ FIELD_X" would silently apply the wrong rotation sign to
one of the two groups. The existing pack/unpack code already keys on
`VectorRole`; the MPI backend inherits that and adds nothing of its own (§3).

FMS sign table, transcribed locally and never called: FIELD_X gives `+fieldy`
at NINETY and `-fieldy` at MINUS_NINETY; FIELD_Y gives `-fieldx` at NINETY and
`+fieldx` at MINUS_NINETY; FIELD_S gives `+field` always. Rotation is NINETY on
west/east halo edges and MINUS_NINETY on south/north.

---

## 2. MPI rank ↔ tile mapping — CONFIRMED, no semantic mismatch

Production is `FV_NX=1`/`FV_NY=6` ⇒ six ranks, one tile each, and the mapping
is the identity **rank `r` ↔ tile `r+1`** (tiles are 1-based throughout this
codebase; `run_small_perf_bench.cpp:2279` derives its on-disk directory as
`"tile" + (tile-1)`, which is how the base is read off an actual caller rather
than inferred). All 12 cubed-sphere contact lines match the topology the
in-process path already uses; nothing about the neighbour relation changes when
the six tiles move from threads to processes.

**The transport does not trust that mapping.** `create_mpi_host_staged_transport`
`MPI_Allgather`s each rank's tile and refuses a duplicate or an out-of-range
value, and `check_topology_once()` verifies for every neighbour that
`nb.rank` is in range and `tile_of_rank_[nb.rank] == nb.tile`, and that
`topo.tile == tile_`. A mis-wired launcher fails loudly at attach instead of
exchanging wrong data. This was verified at runtime: job 58240534 prints
`rank 0 owns tile 1` … `rank 5 owns tile 6` (12 banner lines, 6 ranks × 2 MPI
cases).

---

## 3. The abstraction (§3) — one interface, two transports, zero duplicated logic

New header `cuda/dyn_core/include/fv3_gpu_halo_transport.hpp` defines

```
Halo topology/descriptor  ->  pack CUDA kernels  ->  HaloTransport  ->  unpack CUDA kernels
                                                    { PeerHaloTransport | MpiHaloTransport }
```

`class HaloTransport` has exactly six virtuals, of which only two are
transport-dependent:

- `name()`, `attach(tile, HaloMemberResources)`, `detach(tile)`,
  `attach_failure_reason()`, `agree_on_layout(group, field_count)`,
  `abort_transport()` — lifecycle/agreement;
- **`transfer(eff, geom, group, ...)`** — make the donors' packed bytes readable
  and fill my recv buffer;
- **`release_send(stream, ...)`** — nobody may repack a send buffer somebody is
  still reading.

That is the whole transport-dependent surface, because the proven exchange
ordering is:

```
compute -> pack -> record pack_done -> [transfer] -> unpack -> record xfer_done
        -> order consumer behind xfer_done -> [release_send]
```

Only the two bracketed steps depend on how bytes cross the boundary.
**Rotation, sign, slot addressing, ring iteration, `dest_edges` filtering and
the pack/unpack kernels are untouched and shared** — the MPI backend contains
no edge-rotation formula of its own, satisfying the task's explicit
prohibition. The in-process path was adapted, not replaced: `PeerHaloTransport`
is a thin adapter over the existing `HaloPeerGroup`, and **no functionality was
removed from `HaloPeerGroup`** (§5). Both transports remain available and the
default is unchanged.

Buffer addressing is unchanged and is what makes a *single* bulk transfer per
group correct: `slot_base(geom, width, field, edge) = (field*4 + edge) * slot`
and `group_buffer_elements = field_count * 4 * slot`, so a group's used region
is contiguous from the buffer base.

---

## 4–7. The host-staged MPI backend

`cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp` (~400 lines),
`cuda/dyn_core/include/fv3_gpu_halo_mpi.hpp` (factories that expose **no MPI
type**: `create_mpi_host_staged_transport(int comm_fortran_handle, int32_t
tile, std::string *why)` / `destroy_mpi_host_staged_transport`).

`class MpiHaloTransport final : public HaloTransport`, `name() ==
"mpi_host_staged"`.

- **Pinned host staging, sized once at attach.** Two `cudaHostAlloc(...,
  cudaHostAllocDefault)` buffers (send, recv) sized from
  `HaloMemberResources::buffer_elements`, i.e. the max over all live groups —
  no per-exchange allocation. Both are `std::memset` to 0 at attach: the
  unused tail of a group's slot region is literally put on the wire, so leaving
  it undefined would send uninitialized host memory. One
  `cudaEventCreateWithFlags(&ev_d2h_, cudaEventDisableTiming)`.
- **`transfer()` sequence:** one `cudaMemcpyAsync` D2H of the whole group send
  region → `cudaEventRecord(ev_d2h_)` + `cudaEventSynchronize(ev_d2h_)` →
  `MPI_Irecv` × (4 edges × fields) → `MPI_Isend` × (4 edges × fields) → one
  `MPI_Waitall` → one `cudaMemcpyAsync` H2D of the recv region.
- **Sync discipline (§7).** The only host wait is
  `cudaEventSynchronize(ev_d2h_)`, which is strictly narrower than a stream
  synchronize and is **not** a device synchronize. There is no
  `cudaDeviceSynchronize` anywhere in the backend. Persistent device residency
  is preserved: the staging traffic is halo slots only, never model state, and
  the H2D lands back in the same device recv buffer the existing unpack kernels
  already read.
- **Tag construction:** `tag = ((group*4) + edge) * max_halo_group_fields +
  field`, max value 207 with 12 groups × 4 edges × 4 fields — comfortably
  inside the guaranteed `MPI_TAG_UB ≥ 32767`. Distinct per (group, edge, field)
  so no two in-flight messages can be confused even with several groups
  outstanding.
- **Rotated seams.** When the contact line is reversed and the role is not
  scalar, the donor slot is remapped as `sn = field_slot(eff, fs.partner)`.
  This is the *addressing* consequence of the cross-component pairing already
  in the group table; the sign/rotation arithmetic itself stays in the shared
  unpack kernels.
- **`release_send()` is a justified no-op:** `MPI_Waitall` has already
  completed before `transfer()` returns, so the send buffer is free by
  construction. `abort_transport()` is a documented no-op — MPI has no
  library-level rendezvous to poison the way `HaloPeerGroup::abort_group()`
  poisons the in-process one; callers must `MPI_Abort` instead (and the
  benchmark does, §12).
- **Cross-rank layout agreement (§4's `agree_on_layout` equivalent).**
  Positional field compaction is configuration-dependent (e.g. `H_DIVGD` only
  when `nord>0`, `H_HEAT` only when `d_con>0`), so two ranks that compacted a
  group differently would read each other's buffers at the wrong offset.
  `agree_on_layout` does a cached per-group MIN/MAX `MPI_Allreduce` on the
  field count and refuses on disagreement. This is the cross-process analogue
  of the in-process `layout_` check, not a new invention.
- `MPI_Comm` is passed as the **Fortran INTEGER handle** and converted with
  `MPI_Comm_f2c`. This is deliberate: `MPI_Comm` is `int` in MPICH and an
  opaque pointer in Open MPI, so it cannot appear in a stable C ABI.
- `create_mpi_host_staged_transport` checks `MPI_Initialized`, the tile range,
  that the communicator size is exactly 6, and the allgathered tile map, before
  `new (std::nothrow) MpiHaloTransport`.

---

## 8. Corner classification — nothing corner-specific is sent

There is **no corner-specific message** in either transport, and this is a
property of the FV3 halo contract, not a simplification:

1. The four cubed-sphere **diagonal corner halo cells have no donor tile** at
   all. This is the same fact already established during Gate C2, where those
   cells are excluded from every comparator box (`CURRENT_STATE.md` §4). There
   is nothing to receive, so there is nothing to send.
2. The **corner *rotations*** that FV3 does perform (`copy_corners`,
   `fill_4corners`, `fill2_4corners`, and the `corner_edge_service`) are
   **intra-tile** device-side operations on a tile's own arrays. They are
   launched by the numerical stages, are already validated (Gate C2, Prompts
   9–11), and never cross a tile boundary — so they are invisible to the
   transport layer by construction.
3. A **corner-staggered field** on a rotated seam is still an ordinary edge
   exchange. `H_DIVGD` is CORNER-staggered and is exchanged across rotated
   seams; it takes the FIELD_S branch (scalar, `+field`, no sign flip), which
   is legal precisely because `divgd` is a scalar in the FMS sense despite
   living on corners.

Consequence for the MPI backend: the per-group message set is exactly
4 edges × field_count, with no fifth "corner" class. That is what the tag
formula in §4 encodes and what the test in §11 exercises.

---

## 9. E11 (`H_DWIND_SEAM`) on the MPI backend — explicit mapping

E11 is `mpp_get_boundary(u, v, domain, ebuffery, nbufferx, gridtype=DGRID_NE)`
at `dyn_core.F90:1178` with write-back at `:1181-1188`, gated
`it==n_split .and. grid_type<4 .and. .not. nested`. It is a **ring-0
reconciliation of the one shared panel seam line that two tiles compute
redundantly** — not a halo exchange. Its mapping onto the transport is:

| aspect | E11 | ordinary group |
|--------|-----|----------------|
| `ring_begin` | **0** (owned cells) | 1 |
| `halo_width` | **0** | 3 |
| ring count | `halo_width - ring_begin + 1` = 1 | 3 |
| `dest_edges` | `u`: north only; `v`: east only | `0xF` |
| source component | **cross-component**: `u` ← `v`, `v` ← `u` (that is what DGRID_NE means) | same-component |
| destination cells | `u(is:ie, je+1, :)` and `v(ie+1, js:je, :)` — OWNED | halo rings |

**Nothing in the MPI backend is E11-specific.** It rides the identical path
because:

- `ring_begin`/`halo_width` enter only through `group_slot_elements(geom,
  width)` and the shared ring iteration, so a 1-ring group produces a smaller
  contiguous slot region and one correspondingly smaller D2H/H2D;
- `dest_edges` is applied **on the unpack side only** — the pack side and the
  wire carry all four edges, and the unpack kernel drops the three edges whose
  bit is clear. The transport therefore does not need to know that E11's
  destinations are asymmetric;
- the cross-component pairing is expressed by the group table's `partner`
  field, which the shared pack/unpack already honours and which the MPI
  backend consults only for the donor-slot remap on a reversed seam (§4).

E11 was exercised explicitly and separately in the six-rank test (§11) — 48
received values decoded and verified on rank 0 — precisely because it is the
one group whose destinations are owned cells, where a mistake would corrupt
state rather than a halo.

---

## 10. Public transport-selection API — no C++ or MPI types in the ABI

`cuda/dyn_core/include/fv3_gpu_communication.h` (existing header, extended):

```c
FV3_GPU_HALO_TRANSPORT_NONE           = 0
FV3_GPU_HALO_TRANSPORT_IN_PROCESS_PEER = 1
FV3_GPU_HALO_TRANSPORT_MPI_HOST_STAGED = 2

int fv3_gpu_attach_mpi_halo_transport(fv3_gpu_context *context,
                                      int communicator_fortran_handle,
                                      int32_t tile,
                                      fv3_gpu_error *error);
int fv3_gpu_detach_halo_transport(fv3_gpu_context *context,
                                  fv3_gpu_error *error);
int fv3_gpu_halo_transport_kind_of(const fv3_gpu_context *context);
```

Only `int`, `int32_t` and opaque pointers appear. The communicator crosses as
the **Fortran INTEGER handle**, so an `ISO_C_BINDING` wrapper can pass
`MPI_COMM_WORLD` (or a GEOS sub-communicator) directly with no C++/MPI type in
the interface. Built without the CMake option,
`fv3_gpu_attach_mpi_halo_transport` returns
`FV3_GPU_UNSUPPORTED_CONFIGURATION` rather than failing to link — so a
Fortran caller compiled against the header always links.

`fv3_gpu_halo_transport_kind_of` exists so a caller can **verify** which
transport it got rather than assume; the benchmark uses it (§12), because a
silent fallback to the in-process transport would make a six-process run report
success while exchanging nothing.

---

## CMake

```cmake
option(FV3_DYN_CORE_ENABLE_MPI_HALO "..." OFF)
```

When ON (inside `if(FV3_DYN_CORE_ENABLE_CUDA)`): `find_package(MPI REQUIRED
COMPONENTS CXX)`, `runtime/fv3_gpu_halo_mpi.cpp` added to the library,
`FV3_DYN_CORE_ENABLE_MPI_HALO=1` defined, `MPI::MPI_CXX` linked; plus the
`fv3_halo_mpi_transport_test` target and its six-rank ctest
(`LABELS "gpu;mpi"`, `SKIP_RETURN_CODE 77`).

**Recorded hazard, handled explicitly.** The compile definition is PRIVATE on
`fv3_dyn_core_gpu`, so it does **not** reach the benchmark's translation unit.
A bench built without it would preprocess every `--mpi-halo` line away and then
report "requires a build configured with …ON" in a build that has it on. It is
therefore repeated on the `fv3_gpu_perf_bench` target with a comment citing the
same class of silent-preprocess-away hazard already recorded at
`CMakeLists.txt:164-168`.

**Proven inert when off / additive when on:** 22/22 ctests with MPI OFF, 23/23
with MPI ON (§19).

---

## 11–13. Six-rank MPI transport test

`cuda/dyn_core/tests/integration/test_halo_mpi_transport.cu`, run as
`mpirun -np 6`, one tile per rank. Modelled on
`test_halo_exchange_multitile.cu`'s **encode/decode provenance oracle** (each
sent value encodes its origin tile/field/ring/position, so a received value
proves *which donor cell it came from* — a wrong-neighbour or wrong-offset bug
cannot pass) and its **LOCAL, non-circular sign oracle** (the expected sign is
derived from the locally transcribed FMS table, not from the implementation
under test).

Coverage, per the task's four required classes:

| class | group | result (rank 0 decoded/verified) |
|-------|-------|----------------------------------|
| scalar | `H_W_IN` (center) | 288 values |
| vector/staggered | `H_DWIND_IN` (DGRID_NE, slot_x north) | 624 values |
| vector/staggered, transposed slots | `H_CWIND` (CGRID_NE, slot_x east) | 624 values |
| multi-ring, multi-field | `H_DSCALAR` (4 fields, width 3) | 1152 values |
| E11 / ring 0 / owned destinations | `H_DWIND_SEAM` | 48 values |

Result on real hardware (`warpa003`, job 58240280, and re-run in the §14 build
as job 58240534 section 0):

```
transport: mpi_host_staged
=== Summary (all 6 ranks) ===
  checks: 16692   failures: 0
Six-rank MPI halo transport: PASS
```

Both `H_DWIND_IN` and `H_CWIND` are included deliberately: they are the pair
that distinguishes a slot-keyed from a staggering-keyed sign, so passing both
is the evidence that the transposition in §1 is handled correctly.

---

## 12. Six-MPI-process CUDA dyn_core mode

`cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` gained a `--mpi-halo` mode
(six guarded edits, all `#if defined(FV3_DYN_CORE_ENABLE_MPI_HALO)`,
default off). It is the only integrated six-tile CUDA dyn_core driver, and
previously had no MPI mode at all: `run_one_repeat` spawned six `std::thread`s
in one process and `phase_e::setup` hard-called `attach_halo_peer_group`.

**The only functional difference between the two configurations is the setup
branch** — `fv3_gpu_attach_mpi_halo_transport(m.ctx,
MPI_Comm_c2f(MPI_COMM_WORLD), tile, &err)` followed by a check that
`fv3_gpu_halo_transport_kind_of(m.ctx) ==
FV3_GPU_HALO_TRANSPORT_MPI_HOST_STAGED`, versus `attach_halo_peer_group(...)`.
Everything downstream (state, geometry, config, dispatch, dumps) is shared
code. That is what makes §14 a test of the transport rather than of two
different configurations.

Collective-safety hazards handled by design rather than discovered at runtime:

- a rank returning SKIP 77 alone would leave the other five blocked forever in
  the first halo `MPI_Waitall` — a hang, not a skip ⇒ device usability is
  reduced with `MPI_Allreduce(MPI_MIN)` so any rank's "no device" is
  everybody's skip;
- a rank failing mid-run has no rendezvous to poison ⇒ failure is `MPI_Abort`,
  not `return false`;
- a rank detaching while another is mid-exchange ⇒ `phase_e::teardown` runs
  **before** the timing `MPI_Allreduce`, so the collective cannot be reached by
  a rank still exchanging;
- `--sp0-verify-only` is an in-process decomposition and is explicitly refused
  with `--mpi-halo` rather than silently reinterpreted;
- a non-6 rank count is refused.

`MPI_Allreduce(MPI_MAX)` on elapsed time preserves the existing
MAX-across-tiles statistic; rank 0 prints `CSV,cuda_gpu_mpi,...`.

---

## 14. Six-rank dyn_core exit vs. the single-process `HaloPeerGroup` path

Job **58240534**, node `warpa003`. Both paths run **from the same binary**
(`build_cuda_r8_mpibench/fv3_gpu_perf_bench`), with the same flags, the same
authoritative Gate-C2 SP0 state/geometry/config (c8 / npz=10 / n_split=3 / R8 /
six tiles, `--bdt 3600`, authoritative `--grid-metrics`, `--sp0-state`,
`--sp1-boundary`, `--damp-n-sponge -1`, `--akap-exner`), on the same node, in
the same job. **The only difference is `--mpi-halo`.** Any difference in the
dumped exit state is therefore attributable to the transport and nothing else.

Resourcing: 6 ranks × 1 tile = 6 tile contexts on one A100 — the same device
load the six-thread path carries. Deliberately **not** Prompt 8's
oversubscribed 36-context arrangement.

The comparison needs no new tooling: the debug dump is keyed on
`topology.tile`, so six processes write the same tree layout six threads do,
and a plain `cmp` suffices. Dumped stage `dyn_core_exit`, all 13 exit fields
(`u,v,w,delz,delp,pt,pe,peln,pk,pkz,ua,va,omga`) × 6 tiles = 78 files.

| section | result |
|---------|--------|
| 0 — six-rank transport test in this build | `PASS`, RC 0 |
| 1 — `SMOKE_PEER_1` (1 dyn_core call, threads) | RC 0, bench PASS |
| 2 — `SMOKE_MPI_1` (1 dyn_core call, 6 processes) | RC 0, bench PASS on all 6 ranks |
| **3 — `SS14_1CALL`** | **`IDENTICAL=78 DIFFERS=0 MISSING=0` ⇒ BITWISE IDENTICAL** |
| 4 — `SMOKE_PEER_5` (5 calls = 15 substeps) | RC 0 |
| 5 — `SMOKE_MPI_5` (5 calls = 15 substeps) | RC 0 |
| **6 — `SS14_5CALL`** | **`IDENTICAL=78 DIFFERS=0 MISSING=0` ⇒ BITWISE IDENTICAL** |

Zero NaN/Inf on every `[FV3DBG]` line across the whole job
(`nan/inf=0/2250` and `nan/inf=0/2475` only). The comparison was run in both
directions, so a file present on only one side would have counted as a
mismatch.

**Interpretation.** BITWISE identity at 5 calls / 15 substeps means every
in-loop resolve point — including E11 at `it = n_split` and the post-loop
`H_HEAT` — delivered byte-identical halo data across six OS processes as it
does across six threads. Because the in-process path is the one Gate C2
validated against authoritative Fortran, **the Gate C2 numerical result carries
over to the MPI transport unchanged for this configuration**, without needing a
new Fortran comparison run.

Recorded honestly: the wall-clock numbers in the CSV lines
(`cuda_gpu` 445.6 ms vs `cuda_gpu_mpi` 214.2 ms for 1 call; 171.3 vs 95.1
ms/step for 5 calls) are **not a performance claim**. The task forbids
performance matrices, these are 1- and 5-iteration smokes with no warmup and no
repeats, and the two configurations differ in host threading model as well as
transport. Do not quote them.

---

## 16. Fix budget

**Zero of the three permitted substantial transport fixes were consumed.** The
transport passed its six-rank provenance test and the bitwise dyn_core
comparison on the first hardware attempt. The two build failures encountered
were configuration mistakes on my side, not transport defects:

1. missing `-DFV3_DYN_CORE_ENABLE_CUDA=ON` (defaults OFF) plus a non-existent
   `-DFV3_GPU_PRECISION=R8` (the real name is `FV3_DYN_CORE_PRECISION`, already
   defaulting to R8) ⇒ no CUDA binaries were produced;
2. CMake auto-detected nvcc at `.../24.7/cuda/bin/nvcc`, whose implicit link
   dirs lack `libcudadevrt`/`libcudart_static` ⇒ pin
   `-DCMAKE_CUDA_COMPILER=.../24.7/cuda/12.5/bin/nvcc`, as the existing working
   builds do.

Both are recorded because the next person configuring a fresh build directory
here will hit them.

---

## 19. Regression evidence

Job **58240280** (`warpa003`):

| section | result |
|---------|--------|
| A — R8 CTest, MPI **OFF** (`build_cuda_r8`) | `100% tests passed, 0 tests failed out of 22`, RC 0 |
| B — six-rank MPI halo transport test | `checks: 16692  failures: 0`, `PASS`, RC 0 |
| C — single-process transport smoke (`PeerHaloTransport` adapter) | `=== 3A.11 CUDA perf benchmark: PASS ===`, RC 0 |
| D — R8 CTest, MPI **ON** (`build_cuda_r8_mpi`) | `100% tests passed, 0 tests failed out of 23`; `#17 fv3_halo_mpi_transport … Passed 1.87 sec`; labels `gpu = 6.64 sec*proc (10 tests)`, `mpi = 1.87 sec*proc (1 test)`, RC 0 |

Plus job **58240534** sections 1/2/4/5 (four dyn_core smokes, all RC 0) and
§14's two bitwise comparisons.

Section C matters specifically because the default in-process path was
refactored to go through the same `HaloTransport` interface via
`PeerHaloTransport`; it confirms the refactor left that path working end to
end. Section A vs D together show the CMake option is inert when off and
purely additive when on.

**No performance matrix was run**, per the task's instruction.

---

## Files changed / added

Modified (uncommitted):

| file | ± |
|------|---|
| `cuda/dyn_core/CMakeLists.txt` | +75 |
| `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` | +208 |
| `cuda/dyn_core/include/fv3_gpu_communication.h` | +45 |
| `cuda/dyn_core/include/fv3_gpu_context.hpp` | +23 |
| `cuda/dyn_core/include/fv3_gpu_halo_exchange.hpp` | +15 |
| `cuda/dyn_core/include/fv3_gpu_internal.hpp` | +20 |
| `cuda/dyn_core/runtime/fv3_gpu_halo_exchange.cu` | +11 |
| `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` | +295 |

Added (untracked):

- `cuda/dyn_core/include/fv3_gpu_halo_transport.hpp` — the interface (§3)
- `cuda/dyn_core/include/fv3_gpu_halo_mpi.hpp` — MPI-type-free factories
- `cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp` — the backend (§4–§7)
- `cuda/dyn_core/tests/integration/test_halo_mpi_transport.cu` — the six-rank
  test (§11)
- `docs/agent_handoff/prompt_15_mpi_halo_transport.md` — this document

Artifacts that must **not** be committed: `build_cuda_r8_mpi/`,
`build_cuda_r8_mpibench/`, `scratch_p15/`,
`perf_rerun_logs/prompt15_mpi_halo.sbatch`,
`perf_rerun_logs/prompt15_mpi_dyncore.sbatch` and their `*.out` logs.

Nothing was committed or pushed.

---

## Validation level, stated precisely

- **IMPLEMENTED** — the host-staged MPI transport, the transport abstraction,
  the public selection ABI, the six-rank test, the benchmark's MPI mode.
- **COMPILED** — MPI OFF (22 ctests) and MPI ON (23 ctests) both build clean;
  the bench builds in both.
- **GPU EXECUTED** — six separate MPI processes, one A100, real halo traffic:
  16,692 provenance checks / 0 failures, plus four dyn_core smokes.
- **NUMERICALLY VALIDATED** — against the in-process `HaloPeerGroup` path,
  BITWISE IDENTICAL on all 13 exit fields × 6 tiles at both 1 and 5 dyn_core
  calls. Scope is that comparison and the Gate C2 configuration it inherits
  (c8 / npz=10 / n_split=3 / k_split=1 / R8 / six tiles / production first-run
  numerics). **Not** claimed for R4, other grids, or other configurations.

---

## What this unblocks, and what is still open

Prompt 14's blocker is closed: a real cross-process halo transport now exists
and is validated. Standalone CUDA production integration can resume with the
§2 interface table from
`prompt_14_standalone_cuda_integration.md` (complete and reusable as-is).

Still open, and now the next real work:

1. **Public ABI extension for per-run setup.** Prompt 14's secondary finding
   stands: `NumericalConfig` installation, real (`TopologyView`-based) topology
   installation and transport attachment are only partly reachable from the
   stable C ABI (`install_config`/`set_topology` are still internal C++). The
   transport half is now public (§10); the config/topology half is not.
2. **The Fortran `ISO_C_BINDING` wrapper and the `fv_dynamics.F90:645-655`
   call-site branch** — now safe to write, since the transport direction is
   settled.
3. **Stage 2/3 communication** (CUDA-aware MPI / GPU-direct, then
   comm/compute overlap) remains deferred. The `HaloTransport` interface was
   designed so those are new backends, not a redesign: they replace
   `transfer()`/`release_send()` and nothing else.
4. Everything in `CURRENT_STATE.md` §10 is unchanged by this pass — in
   particular R4 is still not savepoint-compared, `ncu` is still
   platform-blocked, and the workload is still GPU-execution-bound.
