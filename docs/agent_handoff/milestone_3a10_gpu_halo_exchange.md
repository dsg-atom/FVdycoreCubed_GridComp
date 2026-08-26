# Milestone 3A.10 — GPU Halo Exchange / Panel-Seam Connectivity for Gate C1

**STATUS: §1–§25 complete. All sanitizer work is complete in both precisions
(Slurm job 58079171, `COMPLETED`, `ExitCode 0:0`). Gate C1: PASS. Numerical
validation against authoritative Fortran savepoints: NOT RUN.**

This document is written incrementally so that the source-traced findings below
survive context compaction. A section that has not actually been performed says
so; nothing here is written in anticipation of a result.

---

## Provenance

| Item | Value |
|---|---|
| Branch | `cuda/gpu-bringup` |
| Starting commit | `b35f0dab59d3021bf4ffb6499da1054cd0f7d3ec` |
| Milestone commit | `b0b1266a15413e997d8f7709446b491ce622bece` — "Implement FV3 GPU cubed-sphere halo exchange" |
| Pushed | **NO.** One commit ahead of `origin/cuda/gpu-bringup`; push and merge require an explicit request |
| Working tree at start | clean |
| Predecessor milestone | 3A.9 (`b5bd177`, handoff `milestone_3a9_gate_c1_closure.md`), verdict `GATE C1 BLOCKED` |
| Authoritative FV3 revision | `GFDL_atmos_cubed_sphere` tag `geos/v3.0.0`, SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45` |
| Reference clone path | `/discover/nobackup/projects/imvi-nse/jli30/fv3_reference/GFDL_atmos_cubed_sphere` (read-only, verified unmodified) |
| Authoritative FMS revision | GFDL FMS **2024.03** (`src/FMS/configure.ac:28`) |
| FMS source path | `/discover/swdev/gmao_SIteam/Baselibs/ESMA-Baselibs-8.32.0/src/FMS/mpp/` |

### Why Baselibs FMS is the authority for seam transforms

`GEOSgcm/build/CMakeCache.txt` pins `FMS_DIR` into
`ESMA-Baselibs-8.32.0/.../Linux/FMS`, so the FMS that GEOSgcm actually links is
the Baselibs one. The reference clone contains **no** FMS source (`mpp_domains`,
`mpp_do_update*`, `mpp_parameter` all absent), so the rotation logic cannot be
read there. It was read from Baselibs instead.

Note: `build_cuda_r8/` and `build_cuda_r4/` CMakeCache have **zero** FMS
references — the standalone CUDA dycore does not link FMS. FMS is used here as a
**numerical-semantics authority only**, not as a build dependency.

---

## §1 Authoritative halo exchange call map

Read from `model/dyn_core.F90` in full (subroutine `dyn_core` spans lines
163–1287; acoustic loop `do it=1,n_split` at **399**, `enddo` at **1196**).

Configuration assumed throughout: the production first-run configuration
(`grid_type=0`, `hydrostatic=.false.`, `a_imp=1.0`, `hord_mt/vt/tm=6`,
`hord_dp=-6`, `vtdm4>0`), non-nested, `beta=0`, `inline_q=.false.`, and
`MOIST_CAPPA`/`USE_COND` **defined** (confirmed at
`GFDL_atmos_cubed_sphere/CMakeLists.txt:88`).

### Structural facts

- **Halo width is `ng = 3`** on all four sides. `ng` is a compile-time
  `integer, parameter` at `tools/fv_mp_mod.F90:108`, fed to
  `mpp_define_mosaic(... shalo=ng, nhalo=ng, whalo=ng, ehalo=ng ...)` at
  `tools/fv_mp_mod.F90:641-644`. Every exchange without explicit halo arguments
  therefore requests 3.
- **No direction flags anywhere in `dyn_core.F90`.** Grep for
  `SCALAR_PAIR|NUPDATE|SUPDATE|EUPDATE|WUPDATE|XUPDATE|YUPDATE|flags=` returns
  zero hits. The only optional arguments used are `gridtype=` (402, 618, 1095,
  1109), `position=` (528), `whalo=/ehalo=/shalo=/nhalo=` (932), and
  `complete=`.
- **`fill_corners` never appears in `dyn_core.F90`.** It is called only from
  `sw_core.F90:1378,1385,1392` inside `d_sw`, and is a purely local on-PE array
  manipulation (its only MPI lines are commented out).
- No MPI halo call exists in `sw_core.F90`, `nh_core.F90`, `tp_core.F90`, or
  `a2b_edge.F90`. `a2b_ord4` is entirely local.

### Live exchanges

| ID | line(s) | field(s) | stagger / position | width | scalar/vector + gridtype | producer | consumer | frequency |
|---|---|---|---|---|---|---|---|---|
| **E1c** | complete `504` (start in `fv_dynamics.F90:630`) | `u`,`v` | D-grid | 3 | VECTOR `DGRID_NE` | previous `fv_dynamics` stage | `c_sw` (515) | completes at every substep; at `it==1` completes the start issued outside `dyn_core` |
| **E1b** | start `1109`, complete `504` (next substep) | `u`,`v` | D-grid | 3 | VECTOR `DGRID_NE` | `nh_p_grad` (1019), `Ray_fast` (1048), `mpp_get_boundary` fix (1094) | next substep `c_sw`/`d_sw` | every substep **except** `it==n_split` (`if (it/=n_split)` at 1108) |
| **E3** | start `427`, complete `507` | `w` | A-grid, layer | 3 | SCALAR | previous `Riem_Solver3` (918) + `d_sw` w-transport | `c_sw` (515), `d_sw` (789) | every substep |
| **E4** | start `459`, complete `552` | `gz` | A-grid, **interface** `npz+1` | 3 | SCALAR | `gz` init loop 445–455 | `zh` copy (556–564), `update_dz_c` (577), `Riem_Solver_C` (583) | **`it==1` only** (guards at 431, 548) |
| **E5** | complete `476` | `q_con`,`cappa`,`delp`,`pt` | A-grid, layer | 3 | SCALAR ×4 | `fv_dynamics.F90:620-627` | `c_sw` (515) | **`it==1` only** (guard at 474) |
| **E6** | start `528`, complete `636` | `divgd` | **CORNER (B-grid)** | 3 | SCALAR, `position=CORNER` | `c_sw` (515–523) | `d_sw` (789) divergence/vorticity damping | every substep, guard `nord>0` |
| **E7** | start `618`, complete `640` | `uc`,`vc` | C-grid | 3 | VECTOR `CGRID_NE` | `p_grad_c` (614) | `d_sw` (789) — `ut/vt`, `crx/cry/xfx/yfx` | every substep |
| **E8** | starts `844`,`846`,`847`; complete `878` | `q_con`,`delp`,`pt` | A-grid, layer | 3 | SCALAR ×3 | `d_sw` k-loop (700–829), `mix_dp` (835) | `pe_halo` (941), `pk3_halo` (946), `nh_p_grad` (1019) via `a2b_ord4(delp)`, next substep `c_sw`/`d_sw` | every substep |
| **E9a** | start `931`, complete `967` | `zh` | A-grid, interface | 3 | SCALAR | `Riem_Solver3` (918) | `gz=zh*grav` (970–977) → `nh_p_grad`; next substep `update_dz_d` | every substep, **`square_domain`** branch |
| **E9b** | start `932`, complete `981` | `pkc` | A-grid, interface | **3 — see below** | SCALAR | `Riem_Solver3` (918) | `nh_p_grad` (1019) via `a2b_ord4(pp=pkc)` | every substep, `square_domain` branch |
| **E9c** | starts `934`,`935`; complete `967` | `zh`+`pkc` bundled | A-grid, interface | 3 | SCALAR ×2 | `Riem_Solver3` (918) | as E9a/E9b | every substep, **`.not. square_domain`** branch |
| **E11** | `1094-1095` | `u`,`v` — **shared boundary line, not a halo** | D-grid | n/a | VECTOR `DGRID_NE`, `mpp_get_boundary`, only `ebuffery`/`nbufferx` | `nh_p_grad` (1019), `Ray_fast` (1048) | overwrite loop 1096–1104 | **`it==n_split` only**, `grid_type<4`, non-nested |
| **E12** | start `1201` | `q(:,:,:,1:nq)` | A-grid, layer | 3 | SCALAR (4-D) | `d_sw` mass fluxes | `fv_tracer2d`, completed **outside** `dyn_core` | once per `dyn_core` call, outside the `it` loop |
| **E13** | `1214` → `del2_cubed` → **`2291`** | `heat_source` | A-grid, layer | 3 | SCALAR, `mpp_update_domains` | `d_sw` accumulation (820–828) | `del2_cubed` stencil (2327–2353) + `copy_corners`, then `pt` heating (1237–1258) | once per `dyn_core` call, outside the `it` loop |

### Two FMS group-semantics questions, resolved from source

Both were flagged as unresolved inferences during reconnaissance and were then
traced directly in Baselibs FMS. Both resolutions matter.

**(a) Line 932's `whalo=2,ehalo=2,shalo=2,nhalo=2` is DISCARDED. `pkc` is
exchanged at width 3.**

`start_group_halo_update` (`fv_mp_mod.F90:756-795`) calls
`mpp_create_group_update` only when the group is uninitialized, otherwise
`mpp_reset_group_update_field`. The reset entry point is

```fortran
subroutine MPP_RESET_GROUP_UPDATE_FIELD_3D_(group, field)
```

(`mpp_group_update.fh:932`) — it takes **no halo arguments at all**. Halo widths
are stored once, at creation (`group%whalo_s = update_whalo`, …). `i_pack(5)` is
created at `dyn_core.F90:459` with `gz` and **no** halo arguments, so it inherits
the domain default `ng = 3`. Therefore `pkc` rides a width-3 exchange.

The reset also enforces `size(field,1..3) == group%isize_s/jsize_s/ksize_s` (FATAL
otherwise), which is why `gz` and `pkc` may share the handle: both are
`(isd:ied, jsd:jed, npz+1)`.

**Consequence for the CUDA implementation:** the group named `FV3_GPU_H_PKC2`
must be exchanged at width **3**, not 2. The enum name is a misnomer inherited
from the call site's stated intent. Implementing width 2 would *under*-exchange
relative to authoritative behavior.

**(b) `i_pack(1)` slot 4 retains `cappa`; the per-substep group is effectively
`{q_con, delp, pt}`.**

`group%reset_index_s` is zeroed at the **start of the launch**
(`MPP_DO_GROUP_UPDATE_` at `mpp_group_update.fh:469`, `MPP_START_GROUP_UPDATE_`
at `:672`), not by the rebind calls, and zeroing touches only the rebind cursor —
never `addrs_s`. `mpp_reset_group_update_field` increments the cursor and
rebinds `group%addrs_s(reset_index_s) = LOC(field)`, FATALing if it exceeds
`nscalar`.

So per substep, `dyn_core.F90:844/846/847` rebind slots 1←`q_con`, 2←`delp`,
3←`pt`, and **slot 4 still holds `cappa`** bound at creation
(`fv_dynamics.F90:620,622,626,627`). Four scalars are exchanged, including
`cappa` — not a duplicated `pt`.

**Consequence:** because `cappa` is never modified inside `dyn_core`,
re-exchanging it is idempotent. The CUDA per-substep group therefore needs only
`{q_con, delp, pt}`, and the `it==1` group needs `{q_con, cappa, delp, pt}`.
This conclusion now rests on read source, not recollection.

### Local (non-MPI) halo fills — ordering constraints, not exchanges

| line(s) | routine | fills | reads | frequency |
|---|---|---|---|---|
| `940-941` | `pe_halo` (def. 1392–1420) | `pe` 1-ring | `delp` halo (E8) | when `remap_step` |
| `945-946` | `pk3_halo` (def. 1289–1341) | `pk3` **2-ring** | `delp` **2-ring** halo (E8) | every substep |
| `556-564` / `567-574` | inline | `zh←gz` (it==1) / `gz←zh` (it>1) over `isd:ied,jsd:jed` | — | every substep |
| `970-977` | inline | `gz = zh*grav` over `is-2:ie+2, js-2:je+2` | `zh` halo (E9) | every substep |
| `1096-1104` | inline | `u(:,je+1,:)`, `v(ie+1,:,:)` | E11 buffers | `it==n_split` |
| `2304-2323`, `2325`, `2337` | corner average + `copy_corners` in `del2_cubed` | `heat_source` corners, `fx`/`fy` | E13 | `min(3,nf_ke)` iterations |

### Dead exchanges in the supported configuration

| line(s) | exchange | killed by |
|---|---|---|
| `401-403` | `i_pack(8)` `u`,`v` `DGRID_NE` | `#ifdef ROT3` — not defined |
| `416-422`, `629-633` | `i_pack(10)` `q` | `inline_q = .false.` |
| `1021-1039` | `mpp_update_domains(ua,...)` at 1036 | `#ifdef ROT3` **and** `do_f3d` |
| `943-944` | `pln_halo` (local) | `use_logp = .false.` |
| `1057-1062` → `2475` | `dudt_tofd`,`dvdt_tofd` `AGRID` | `#ifdef DO_TOFD` **and** `Beljaars_TOFD` |
| many | all `nested_grid_BC_apply_intT`, `nest_halo_nh` | `nested = .false.` |
| `543-545`, `899-901`, `1007-1012` | `geopk`, hydrostatic `one_grad_p` | `hydrostatic = .false.` |
| `1014-1017` | `split_p_grad`, `one_grad_p` | `beta = 0` → `nh_p_grad` (1019) is live |
| `781-788` | `z_rat` | `do_f3d = .false.` |

### Configuration dependencies that must remain supported

- **E6** (`divgd`) depends on `flagstruct%nord > 0`.
- **E9a/E9b vs E9c** depends on `gridstruct%square_domain`, a *runtime
  decomposition* property (`npes_x==npes_y .and. mod(npx-1,npes_x)==0`,
  `fv_mp_mod.F90:413,485`), **not** a namelist flag. The CUDA runtime must
  support **both** bundlings.
- `pe_halo` (940) depends on `remap_step` = `breed_vortex_inline .or. it==n_split`.

---

## §2 Audit of existing CUDA halo infrastructure

Headline: **there is no halo exchange in this tree.** There is complete halo
*dependency bookkeeping*, a *transport ABI* that is declared and 100% dead, and
zero pack/unpack/buffer/MPI code.

### Classification

| Component | Status | Basis |
|---|---|---|
| Halo group enum (`include/fv3_gpu_communication.h:7-12`) | **USABLE AS-IS** | 11 groups map 1:1 to the `start/complete_group_halo_update` pairs in §1; values are stable ABI; `FV3_GPU_H_DWIND_NEXT+1` sizing convention applied consistently (`fv3_gpu_internal.hpp:238-239`, `fv3_gpu_context.hpp:32`); validity predicate at `fv3_gpu_contract.cpp:70`. Slot 0 deliberately unused. |
| `DependencyState` / `HaloState` (`fv3_gpu_internal.hpp:236-244`, bodies `fv3_gpu_contract.cpp:105-108`) | **USABLE AS-IS** for readiness | `require`/`mark_ready`/`mark_stale` are live with 11 real `require` consumers; default `HaloState{}` is `unknown`, so a fresh context refuses every `require` — correct fail-closed default. |
| `mark_pending` / `HaloState::exchange_pending` | **ADOPT, not delete** | Zero callers today; never tested anywhere. It is the intended non-blocking state and should be used, but its current form stores only one event, which cannot represent both "posted, not complete" and "completion event". |
| `resolve_halo` (`fv3_gpu_runtime.cpp:192-212`) | **REQUIRES EXTENSION** | Correct three-outcome fail-closed gate and exactly the right insertion point; `mark_ready` already lives here. Only the refusal branch changes. |
| `halo_exchange_is_vacuous` (`:184-190`) | **DEFECT — must fail closed** | See below. |
| `TopologyView` / `EdgeNeighbor` (`fv3_gpu_internal.hpp:162-179`) | **REQUIRES EXTENSION** | `EdgeNeighbor::edge`, `orientation`, `vector_sign_x`, `vector_sign_y` have **zero** production readers — every seam-transform fact is declared and unconsumed. `corner_owner_rank`/`periodic_*` are read only inside the vacuity test. `corner_owner_rank` is `int8_t`, capping rank at 127. |
| `TopologyView` **producer** | **MISSING — highest-priority gap** | `set_topology` (`fv3_gpu_runtime.cpp:375`) has **zero** production callers; only `tests/dgrid/test_wp_b_cuda_smoke.cpp:306` and `tests/integration/test_dispatch_smoke.cpp:1006`. `cuda/interface/fv3_gpu_interface_mod.F90` contains **zero** occurrences of `topology|npx|npy|tile`. |
| `fv3_gpu_halo_callbacks` + both `_fn` typedefs (`:13-23`) | **OBSOLETE — replace, do not extend** | Provably dead: five self-references plus one comment (`fv3_gpu_runtime.cpp:114`); no registration API (the 8 public entry points at `fv3_gpu_api.h:152-160` take no callbacks); no `ContextData` member; never invoked. Its *shape* is also wrong — one flat send/recv byte pair per group cannot express 4 edges × per-edge orientation × multiple fields, and it presumes a packed layout nothing defines. |
| Group→field mapping | **MISSING** | Exists only as English in `halo_group_name` (`fv3_gpu_runtime.cpp:165-180`). Prerequisite for any pack code. |
| Staleness coverage | **REQUIRES EXTENSION** | Only `H_DIVGD` (`fv3_cgrid_predictor.cu:831`) and `H_CWIND` (`:844`, `fv3_cgrid.cu:122`) are ever invalidated. Six groups whose fields are rewritten every substep are never staled, so after substep 1 their `resolve` is a no-op. Harmless only while `resolve_halo` cannot exchange; a silent stale-halo bug the moment it can. |
| Pack / unpack / edge buffers / MPI | **MISSING** | Zero hits for `cudaMemcpyPeer`, `cudaMemcpy2D`, `MPI_*`, `halo_buf`, `edge_buf`. Only three non-test device copies, all whole-allocation `Async` (`fv3_gpu_runtime.cpp:483`, `:514`, `:528`). `FV3_GPU_MPI_ERROR` (`fv3_gpu_api.h:24`) is defined and never returned. |
| Halo/compute region metadata in `FieldRegistry` | **REQUIRES EXTENSION** | The public ABI accepts `compute_lower/compute_upper/halo_lower/halo_upper` (`fv3_gpu_api.h:75-97`), but `make_device_view` (`fv3_gpu_contract.cpp:87`) copies only `lower`/`extent`/`stride_elements`. `DeviceViewBase` has no halo/compute members, so pack code cannot ask a view where its halo is. Staggering **is** recorded (statically in `FieldSpec`, dynamically in the view, with disagreement rejected). |
| `copy_corners` (`dgrid/wp_b_cuda.cu:94-139`) and WP-A `fill_4corners` family | **USABLE AS-IS — out of scope** | See §8. |
| `corner_edge_service` slot | **MISSING implementation** | Required non-null (`fv3_gpu_contract.cpp:109`, `:119`; `fv3_gpu_runtime.cpp:546`) but production wires `corner_service_not_implemented` (`fv3_gpu_production_dispatch.cu:97`). Blocks `nf_ke>1`; independent of halo exchange. |
| Streams / events / `order_after` (`fv3_gpu_runtime.cpp:45-52`, `:457`, `:463-464`) | **USABLE AS-IS**, extension needed for overlap | Two `cudaStreamNonBlocking` streams ordered purely by events, no hidden host sync — the right substrate. `StreamId::transfer_pack` (`fv3_gpu_internal.hpp:228`) shows packing was anticipated. Only two CUDA events exist, both single-purpose; per-group in-flight exchanges need per-group events. |
| Acoustic-loop `resolve` sites (`fv3_gpu_runtime.cpp:763-783`, `:811`) | **REQUIRES EXTENSION** | Structurally ideal: all 10 halo points are isolated in one lambda with correct Fortran anchors, so a real exchange is a change *inside* `resolve_halo`. No stage code needs to change. |
| Workspace (`fv3_gpu_internal.hpp:217-226`, `fv3_gpu_runtime.cpp:484-503`) | **REQUIRES EXTENSION (trivial)** | Single-allocation arena; `build_workspace_layout` (`fv3_gpu_contract.cpp:96-102`) is generic over `WorkspaceId::count`, so adding a fifth `WorkspaceId::halo` needs no builder change and the existing `cudaMalloc`/finalize absorb it. |

### DEFECT: `halo_exchange_is_vacuous` fails OPEN on an unset topology

```cpp
bool halo_exchange_is_vacuous(const TopologyView &t) noexcept {
  for (const auto &nb : t.neighbors)
    if (nb.rank >= 0 || nb.tile >= 0) return false;
  for (const auto owner : t.corner_owner_rank)
    if (owner >= 0) return false;
  return !(t.nested || t.regional || t.periodic_x || t.periodic_y);
}
```

A default-constructed `TopologyView` has all `rank`/`tile` = −1 and all
`corner_owner_rank` = −1, and sets none of the periodic/nested/regional flags.
The predicate therefore returns **true**, and `resolve_halo` marks every group
**ready with no exchange** — on a context whose `global_cubed_sphere` defaults to
**1**, i.e. one that positively declares itself a global cubed sphere.

The predicate conflates *"no topology information was supplied"* with *"this
topology genuinely has no neighbours."* Combined with the missing producer
(`set_topology` has no production caller and the Fortran bridge never mentions
topology), this is reachable **fake halo readiness** in the current tree.

It is only incidentally masked: `npx=npy=0` makes the panel-geometry stages
reject (`fv3_cgrid_predictor.cu:251`, `wp_b_contract_cuda.cpp:61`,
`fv3_vertical_kernels.cu:557`, `wp_b_dsw_contract_cuda.cpp:78`), so a *full*
production dispatch fails before producing wrong numbers. That is incidental
masking, not a guarantee, and it is 3A.9 residual item #3.

**Required fix:** vacuity must require *positive evidence* — an explicit
"topology supplied" indication plus `npx > 0 && npy > 0` — and must refuse when
topology was never set. This must land before any exchange work, because
readiness truthfulness depends on it.

### Competing seam-detection paths (3A.9 residual #4)

Two independent sources of truth for "am I at a panel seam?":

- `operators/wp_d_contract.cpp:398-405` — `physical_edges()` derives it from the
  cross-tile **neighbour list**.
- `dgrid/wp_b_contract_cuda.cpp:236-239` and
  `dgrid/wp_b_dsw_contract_cuda.cpp:486-491` — derive it from **index
  arithmetic** (`db.is == 1`, `db.ie + 1 == topo.npx`, …).

`dgrid/wp_b_divergence_cuda.cu:813-815` then rebuilds
`operators::PhysicalEdges` from the `TransportBounds` copy rather than from
`TopologyView`, so the same type is populated two different ways in two places.
Authoritative `a2b_ord4` (`model/a2b_edge.F90`) uses index geometry plus
gridstruct corner flags. §21 forbids leaving a known discrepancy in the supported
path, so these must be reconciled to one source.

---

## §3 `grid_type = 0` neighbour connectivity and edge transforms

### Scalar connectivity — two independent authoritative sources that agree

The 12-contact cubed-sphere mosaic specification was read from **both**:

1. `FVdycoreCubed_GridComp/CubeHalo.F90:213-261` (in-repo; self-documented at
   `:16-17` as "a straight copy from … `fvdycore/tools/external_ic.F90`")
2. Reference clone `tools/fv_mp_mod.F90:577-625` (`domain_decomp`,
   `case ( 6 ) ! Cubed-Sphere`)

They agree contact for contact. Both feed `mpp_define_mosaic` with
`symmetry=is_symmetry` (`.true.`) and `shalo=nhalo=whalo=ehalo=ng`.
`ntiles = 6`, `num_contact = 12`, `global_indices = (/1,npx-1,1,npy-1/)`, so
`nx = npx-1` and `ny = npy-1` cells per tile.

| # | tile1 edge | tile2 edge | varying idx side 1 | varying idx side 2 | reversed |
|---|---|---|---|---|---|
| 1 | T1 EAST | T2 WEST | j: 1→ny | j: 1→ny | no |
| 2 | T1 NORTH | T3 WEST | i: 1→nx | j: ny→1 | **yes** |
| 3 | T1 WEST | T5 NORTH | j: 1→ny | i: nx→1 | **yes** |
| 4 | T1 SOUTH | T6 NORTH | i: 1→nx | i: 1→nx | no |
| 5 | T2 NORTH | T3 SOUTH | i: 1→nx | i: 1→nx | no |
| 6 | T2 EAST | T4 SOUTH | j: 1→ny | i: nx→1 | **yes** |
| 7 | T2 SOUTH | T6 EAST | i: 1→nx | j: ny→1 | **yes** |
| 8 | T3 EAST | T4 WEST | j: 1→ny | j: 1→ny | no |
| 9 | T3 NORTH | T5 WEST | i: 1→nx | j: ny→1 | **yes** |
| 10 | T4 NORTH | T5 SOUTH | i: 1→nx | i: 1→nx | no |
| 11 | T4 EAST | T6 SOUTH | j: 1→ny | i: nx→1 | **yes** |
| 12 | T5 EAST | T6 WEST | j: 1→ny | j: 1→ny | no |

### Invariants (verified over all 12 contacts — encode these as assertions)

1. **Reversal occurs iff `tile1` and `tile2` have the same parity.**
   Equivalently: **odd tiles (1,3,5) rotate on WEST and NORTH; even tiles
   (2,4,6) rotate on EAST and SOUTH.** Each tile has exactly two aligned and two
   rotated edges.
2. **Transpose ⟺ index reversal.** All six aligned contacts are row→row or
   col→col with both indices ascending; all six rotated contacts are row↔col
   with exactly one index descending. There is no aligned-but-reversed case and
   no transposed-but-forward case on this topology.
3. **Rotation is never 180°.** `ONE_HUNDRED_EIGHTY` is reachable only via
   tripolar folds; `set_single_overlap` FATALs on anything but
   `ZERO/NINETY/MINUS_NINETY` (`mpp_domains_define.inc:5283-5284`).

### Per-tile edge table, with FMS rotation

Rotation is `rotateRecv` (applied to incoming halo data); `rotateSend` is its
negation.

| Tile | WEST | EAST | SOUTH | NORTH |
|---|---|---|---|---|
| **1** | T5 NORTH, rev, **+90** | T2 WEST, aligned, 0 | T6 NORTH, aligned, 0 | T3 WEST, rev, **−90** |
| **2** | T1 EAST, aligned, 0 | T4 SOUTH, rev, **+90** | T6 EAST, rev, **−90** | T3 SOUTH, aligned, 0 |
| **3** | T1 NORTH, rev, **+90** | T4 WEST, aligned, 0 | T2 NORTH, aligned, 0 | T5 WEST, rev, **−90** |
| **4** | T3 EAST, aligned, 0 | T6 SOUTH, rev, **+90** | T2 EAST, rev, **−90** | T5 SOUTH, aligned, 0 |
| **5** | T3 NORTH, rev, **+90** | T6 WEST, aligned, 0 | T4 NORTH, aligned, 0 | T1 WEST, rev, **−90** |
| **6** | T5 EAST, aligned, 0 | T2 SOUTH, rev, **+90** | T4 EAST, rev, **−90** | T1 SOUTH, aligned, 0 |

### Authoritative rotation derivation — `mpp_domains_define.inc`

Defaults `rotateSend = rotateRecv = ZERO` at `:5438`, then per edge pair:

| my edge | neighbour edge | `rotateRecv` | `rotateSend` | line |
|---|---|---|---|---|
| EAST | WEST | ZERO | ZERO | 5451 |
| EAST | SOUTH | **NINETY** | **MINUS_NINETY** | 5459 |
| SOUTH | NORTH | ZERO | ZERO | 5478 |
| SOUTH | EAST | **MINUS_NINETY** | **NINETY** | 5486 |
| WEST | EAST | ZERO | ZERO | 5505 |
| WEST | NORTH | **NINETY** | **MINUS_NINETY** | 5513 |
| NORTH | SOUTH | ZERO | ZERO | 5532 |
| NORTH | WEST | **MINUS_NINETY** | **NINETY** | 5540 |

Index remap (`set_single_overlap`, `:5267-5285`):

```
ZERO:          is+isoff, ie+ieoff, js+jsoff, je+jeoff
NINETY:        is-jeoff, ie-jsoff, js+isoff, je+ieoff
MINUS_NINETY:  is+jsoff, ie+jeoff, js-ieoff, je-isoff
```

Contact points are built for `CENTER` (`:1512`) then replicated to `CORNER`,
`EAST`, `NORTH` (`:1515-1517`) → `domain%update_T/_C/_E/_N`.

### Staggering — C-grid and D-grid are exactly swapped

From `model/fv_arrays.F90:1462-1496`:

| Field | Declaration | Staggered in |
|---|---|---|
| `u` (D-grid) | `u(isd:ied, jsd:jed+1, npz)` | **y** |
| `v` (D-grid) | `v(isd:ied+1, jsd:jed, npz)` | **x** |
| `uc` (C-grid) | `uc(isd:ied+1, jsd:jed, npz)` | **x** |
| `vc` (C-grid) | `vc(isd:ied, jsd:jed+1, npz)` | **y** |

Under `is_symmetry = .true.` a staggered dimension carries one extra point, so
the shared seam point is represented on both tiles — which is why
`mpp_get_boundary` (E11) exists separately from `mpp_update_domains`.

### Vector transforms — and why `DGRID_NE` is not a separate transform

**`DGRID_NE` has no FMS implementation of its own.** Verified directly in
`mpp_update_domains2D.fh`:

```fortran
! :483-490
exchange_uv = .false.
if(grid_offset_type == DGRID_NE) then
   exchange_uv = .true.
   grid_offset_type = CGRID_NE
else if( grid_offset_type == DGRID_SW ) then
   exchange_uv = .true.
   grid_offset_type = CGRID_SW
end if
```

`:550-562` has cases for `AGRID`, `BGRID_NE/SW`, `CGRID_NE/SW` only —
`case default` FATALs, so the rewrite is mandatory. With
`position_x = EAST, position_y = NORTH`, dispatch at `:581-587` swaps the
**overlap descriptors** while leaving the **field addresses** in place:

```fortran
if(exchange_uv) then
   call mpp_do_update(f_addrsx(...), f_addrsy(...), domain, updatey, updatex, ...)
else
   call mpp_do_update(f_addrsx(...), f_addrsy(...), domain, updatex, updatey, ...)
end if
```

This pairs D-grid `u` (NORTH face) with the NORTH overlap and `v` (EAST face)
with the EAST overlap — precisely the C-grid stagger transposed.

**Design consequence:** one seam-transform engine parameterized by
`(position_x, position_y)` covers both grids. The C/D distinction reduces to
which overlap descriptor pairs with which component. It is *not* two independent
transform families.

Sign flips applied during pack (`mpp_do_updateV.fh`, CGRID branch):

| rotation | x-component | y-component |
|---|---|---|
| `ZERO` | `+fieldx` (:477) | `+fieldy` (:579) |
| `MINUS_NINETY` | `−fieldy` (:504); scalar `+fieldy` (:491) | `+fieldx` (:592) — no scalar branch |
| `NINETY` | `+fieldy` (:518) — no scalar branch | `−fieldx` (:619); scalar `+fieldx` (:606) |
| `ONE_HUNDRED_EIGHTY` | `−fieldx` (:545); scalar `+fieldx` (:532) | `−fieldy` (:647); scalar `+fieldy` (:634) |

i.e. an exact rotation matrix: `−90 → (u,v) ↦ (−v, u)`, `+90 → (v, −u)`,
`180 → (−u, −v)`. `SCALAR_PAIR` (`BTEST(update_flags, SCALAR_BIT)`) selects the
no-sign-flip branches — that is how metric terms (`divg_v/divg_u`,
`del6_v/del6_u`, `dx/dy`, `dxc/dyc`) ride the staggered machinery without picking
up the vector sign. The two "no scalar branch" cases are real: those never flip
sign, so the scalar/vector distinction is vacuous there.

### gridtype inventory (reference clone)

| Field pair | gridtype | flags | site |
|---|---|---|---|
| `u,v` | `DGRID_NE` | — | `dyn_core.F90:402,1109`; `fv_dynamics.F90:630` |
| `u,v` boundary | `DGRID_NE` | — | `dyn_core.F90:1094-1095` (`mpp_get_boundary`) |
| `uc,vc` | `CGRID_NE` | — | `dyn_core.F90:618` |
| `cx/cy`, `mfx/mfy` | `CGRID_NE` | — | `fv_tracer2d.F90:943-949,956-959` |
| `divg_v,divg_u` | `CGRID_NE` | `SCALAR_PAIR` | `fv_grid_utils.F90:821-822` |
| `del6_v,del6_u` | `CGRID_NE` | `SCALAR_PAIR` | `fv_grid_utils.F90:823-824` |
| `dy,dx` / `dxc,dyc` | `CGRID_NE` | `SCALAR_PAIR` | `fv_grid_tools.F90:816,824-825,1001-1002` |
| `divgd` | scalar, `position=CORNER` | — | `dyn_core.F90:528` |
| `delp,pt,q_con,w,q,gz,zh,pkc` | scalar CENTER | — | `dyn_core.F90:427,459,844-847,931-935,1201` |

Note the **argument-order swaps** at `CGRID_NE` sites: `(divg_v, divg_u)`,
`(dy, dx)`, and in the GEOS wrapper `model_mpp/fv_my_mpp.F90:109`
`mpp_update_domains(field2, field1, ..., CGRID_NE)` versus `:125`
`(field1, field2, ..., DGRID_NE)`.

---

## §4 Gate C1 exchange execution model

**DECISION: Model B — a single process representing all six cubed-sphere tiles,
one WP-E context per tile on one A100, with neighbour exchange performed by
device-to-device copy.**

### Justification on equivalence, not convenience

The operations that carry numerical meaning — pack region selection, seam index
remap, vector component swap and sign, corner precedence, unpack extents,
readiness transitions, and stream ordering — are **identical** between model B
and production. The only thing that differs is the byte-movement primitive
between two device buffers, which is precisely the layer a future MPI transport
replaces. Because all six tiles are resident on one device, the exchange is a
genuine device-to-device copy with **no host staging**, satisfying the
requirement that host staging not be the production path.

### Feasibility, verified rather than assumed

- No mutable process-global state exists in the runtime (only three anonymous
  namespaces, all holding functions).
- Contexts are independently heap-allocated per handle
  (`fv3_gpu_runtime.cpp:458`, `:467`), each owning its own streams and events.

Therefore six contexts coexist in one process with no runtime surgery.

### Why not the alternatives

- **Model A (real 6-rank MPI):** would add rank mapping, MPI request lifecycle,
  and either CUDA-aware MPI or host staging (the latter forbidden as a production
  path). It needs six GPUs or GPU sharing, against the platform guidance to
  request one GPU for bring-up, and proves nothing additional about seam
  semantics.
- **Model C (callback-based with in-process peer):** the existing callback is
  byte-transport only, cannot express edges/fields/orientation, and is 100% dead.
  Routing through it adds indirection that proves nothing.

### Honest limits of model B — must not be claimed as done

Model B does **not** establish: MPI rank mapping, MPI request lifecycle,
CUDA-aware MPI, GPUDirect, stream/MPI interaction, or multi-node scaling. These
are recorded in §15 as remaining work.

### Prerequisite

Topology plumbing plus the fail-closed vacuity fix (§2) must land **first**,
because `resolve_halo` currently fails **open** on a default topology and no
readiness claim can be truthful until that is corrected.

---

## §5 GPU halo pack

`runtime/fv3_gpu_halo_exchange.cu`, `launch_halo_pack`. Everything is device
resident; there is no host-side numerical packing and no full-state D2H/H2D.

**Source range.** Per edge, per halo ring `h ∈ [1, 3]`, per along-seam index
`a ∈ [1, along]`, per level `k`. The along extent is
`edge_seam_extent(edge, nx, ny, stagger)` and carries the staggering shift, so a
NORTH-staggered field contributes one more line than a CENTER one. The pack reads
**owned** cells only — the strip a neighbour needs from me, not my own halo.

**Packed ordering.** `slot_offset` in `include/fv3_gpu_halo_exchange.hpp`:

```
((h - 1) * along_extent + (along - 1)) * nk + k
```

Level-fastest, then along-seam, then ring. Level-fastest so the transfer of one
(field, edge) slot is a single contiguous `cudaMemcpyAsync`; the alternative
would have been one copy per level.

**Slot layout.** `slot_base(geom, width, field, edge)` gives each (field, edge)
pair a fixed slot sized by `group_slot_elements`, which uses
`max(nx, ny) + 1` — the *worst case* over staggerings, so a slot's base never
depends on the staggering of the field in it. Two peers with different field
extents therefore still agree on offsets. `HaloPeerGroup::agree_on_layout`
enforces the remaining degree of freedom (how many fields a group compacted to).

**Corners.** The four strips are disjoint by construction; a strip stops at the
owned range in its cross-seam direction and never enters a vertex block. §8
proves this is not merely a simplification but the authoritative behaviour.

---

## §6 Panel-seam transforms

### Scalars

Index remapping plus along-seam orientation. `seam_is_reversed(mine, donor_edge)`
determines whether the donor's along index runs opposite to mine, and
`seam_along_index(a, donor_extent, reversed)` mirrors it. No sign, no component
selection.

### Vectors — the finding that governs the whole design

FMS selects the sign by **argument slot**, not by staggering, and selects the
rotation by **seam**, not by staggering:

| | |
|---|---|
| Rotation is a property of the SEAM | `define_contact_point` runs for CENTER only (`mpp_domains_define.inc:1512`); `set_single_overlap` copies the rotation verbatim (`:5262`) |
| Sign is selected by ARGUMENT SLOT | `FIELD_X` at `NINETY` → `+fieldy` (`group_update_pack.inc:128`); at `MINUS_NINETY` → `-fieldy` (`:115`). `FIELD_Y` at `NINETY` → `-fieldx` (`:206`); at `MINUS_NINETY` → `+fieldx` (`:181`). `FIELD_S` → `+field` always |
| Slots are bound by the CALL | `CGRID_NE`: `position_x = EAST`, `position_y = NORTH`. `DGRID_NE`: `position_x = NORTH`, `position_y = EAST` (`mpp_group_update.fh:356-371`, `mpp_domains_util.inc:1950-1974`), and `fv_mp_mod.F90:880,930` pass `(u_cmpt, v_cmpt)` **unswapped** |

**Consequence, and it is the crux of §6's "do not assume all vector seams use one
transform":** keyed by *slot* there is ONE transform. Keyed by *staggering* there
are TWO, and they are transposes of each other. `seam_component_sign(role,
rotation)` is therefore keyed on `VectorRole ∈ {scalar, slot_x, slot_y}` and the
seam rotation — never on the staggering.

The C-grid/D-grid distinction is preserved exactly, and it lives in the group
tables rather than in the transform: `H_CWIND` binds `slot_x = uc` (EAST) and
`slot_y = vc` (NORTH); `H_DWIND_IN` binds `slot_x = u` (**NORTH**) and
`slot_y = v` (**EAST**). Same transform, transposed staggerings.

`tests/integration/test_halo_exchange_multitile.cu` runs both groups for exactly
this reason: a single staggering-keyed sign table passes one and fails the other.

---

## §7 GPU halo unpack

`launch_halo_unpack`. Writes into the **destination** field's staggered extents
via `halo_dest_index(mine, a, h, nx, ny, stagger, i, j)`. It:

* applies `seam_along_index` for a reversed seam;
* applies `source_shift(stagger, reversed)` so the donor strip's staggering is
  read at the shift the *donor* uses, not the receiver's;
* selects the partner component (`u ↔ v`, `uc ↔ vc`) when the seam is rotated and
  the field is not a scalar;
* multiplies by `seam_component_sign`;
* **refuses** rather than truncating when `along != donor` — a mismatched extent
  is a topology defect, not something to clamp.

**No owned cell is written.** The multitile test asserts this independently, per
tile and per field, by comparing the whole owned block against its provenance
encoding after the exchange.

**Corner behaviour is not a precedence rule, it is a theorem** (§8): the four
strips are disjoint, so no cell has two writers and kernel ordering between the
four edge launches is irrelevant to the result.

---

## §8 Corner exchange and precedence — resolved from authoritative source

The prompt warns: "Do not conflate existing local WP-B `copy_corners` helpers
with inter-tile halo exchange unless source tracing proves they are the same
operation." Source tracing proves they are **NOT** the same operation.

### For a cubed sphere, the MPI halo exchange never fills a tile-vertex diagonal

`domain%rotated_ninety` is unconditionally `.TRUE.` for FV3's six-tile mosaic —
six of the twelve contacts have mismatched alignment families (e.g.
`tools/fv_mp_mod.F90:582-585`, tile 1 NORTH ↔ tile 3 WEST) against the check at
`mpp_domains_define.inc:1491-1492`.

`fill_corner_contact` is the **only** producer of overlaps tagged with the
diagonal alignments `SOUTH_EAST/SOUTH_WEST/NORTH_WEST/NORTH_EAST`, and therefore
the only code that could create an inter-tile corner overlap. It has exactly one
call site, `mpp_domains_define.inc:5556`, inside
`if (.NOT. domain%rotated_ninety)` at `:5555`.

So the four vertex blocks are absent from every `overlapSpec` send/recv list;
`mpp_update_domains` never touches them; their contents are whatever the caller
last wrote. FMS leaves no marker and picks no donor tile.

### The two diagonal classes FMS *does* fill are empty for one rank per tile

1. **Intra-tile PE diagonals** — `compute_overlaps` dir 2/4/6/8, gated to
   same-tile at `mpp_domains_define.inc:1665`. Needs an intra-tile PE neighbour;
   a whole-tile member has none.
2. **Edge-contact-derived diagonals** outside the tile in only ONE index —
   `define_contact_point`'s *ungated* dir 2/4/6/8 (`:5714-5717`, `:5745`),
   inheriting the edge contact's `rotateRecv`. Non-empty only when
   `compute%begin > isg/jsg`, i.e. when the receiver is **not** on the tile edge
   in the other direction. A whole-tile member is on all four tile edges, so the
   intersection is empty every time.

**Conclusion for Gate C1:** on the one-rank-per-tile decomposition the halo
exchange fills exactly the four edge strips and nothing else — which is exactly
what `launch_halo_pack/transfer/unpack` do. This closes the question rather than
deferring it.

### `copy_corners` is a local reflection, not communication

Defined at `model/tp_core.F90:247-322` — **not** `sw_core.F90`, which only imports
it at `:48`. No MPI call, no `mpp_` call, no `domain` argument; only `bd`, `npx`,
`npy`, `dir` and the four local ownership flags. It reads the **already-valid**
edge halo strips and writes the `ng × ng` diagonal block by reflection about the
vertex diagonal. Unsigned, hence scalars only.

### The vertex blocks are reconstructed locally, per field and per stage

Always *after* the exchange and dependent on it: `copy_corners`;
`fill_corners_{2d,agrid,cgrid,dgrid}` (`tools/fv_mp_mod.F90:1033-1538`);
`fill_4corners`/`fill2_4corners`/`fill3_4corners` (`model/sw_core.F90:3156-3347`,
2–3 points); `a2b_ord4`'s three-way `extrap_corner` vertex
(`model/a2b_edge.F90:115-152`, reading only edge-strip samples, never a diagonal
cell); `a2b_ord2`'s three-cell mean (`:750-753`); `del2_cubed`'s three-cell vertex
mean (`model/dyn_core.F90:2304-2323`, and `del2_cubed` itself runs at `:1214`,
**outside** the acoustic loop). These are compute kernels owned by WP-B/C/D,
which is where they already are.

`mp_corner_comm` (`tools/fv_mp_mod.F90:1543-1915`) is entirely commented out.
There is no MPI corner exchange anywhere in the authoritative tree.

### Corner vector transforms differ, and differ again from the seam transform

`fill_corners_agrid` (`:1514-1538`) and `fill_corners_dgrid` (`:1338-1370`, live
blocks `:1351-1354` and `:1363-1366`) sign **SW and NE**. `fill_corners_cgrid`
(`:1450-1474`) signs **NW and SE** — the opposite. A single templated corner
kernel with one shared sign table would be wrong.

A commented-out D-grid variant at `:1347-1350`/`:1359-1362` has the NW/SE
placement, i.e. the same placement as the *live C-grid* routine. It is a genuine
trap: a reader who lands on it finds apparent corroboration for the wrong
convention. Two comment citations in `dgrid/wp_b_divergence_cuda.cuh` pointed at
those dead lines and were corrected this milestone.

The only signed vector `fill_corners` in the production dynamics is
`model/sw_core.F90:1392`:
`call fill_corners(vc, uc, npx, npy, VECTOR=.true., DGRID=.true.)` — **`vc` is
argument `x` and `uc` is argument `y`**, the reverse of the naming. There are
**zero** `call fill_corners` sites in `dyn_core.F90`.

### Grid metric sentinels

Grid/metric arrays deliberately carry `big_number`/`tiny_number` in the vertex
blocks via `fill_ghost` (`tools/fv_grid_utils.F90:3107-3184`, called `:323-329`,
`:598-601`, `:635-644`) with targeted repairs at `:649-705`. A device-side metric
copy must preserve them verbatim; zeroing them would mask indexing bugs.

---

## §9 GPU-resident exchange

Device-to-device throughout. `launch_halo_transfer` issues one
`cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, stream)` per (field, edge) from
the **donor's** send buffer to my recv buffer. Buffers are allocated once, in
`attach_halo_peer_group`, sized for the largest live group, because a `cudaMalloc`
inside the acoustic loop is an implicit device-wide synchronization and would
dismantle persistent residency.

**No host staging anywhere in this path**, not even as a fallback. The pack,
transform and unpack are kernels; the transfer is D2D. The only host-side element
is a *rendezvous over enqueue order* (§10), which moves no data.

---

## §10 Readiness integration

`resolve_halo` (`runtime/fv3_gpu_runtime.cpp:424-480`) tries provenance in this
order, and the order is load-bearing:

1. topology `not_supplied` → **fail** (fail-closed; this was the 3A.9 defect)
2. topology `vacuous` → `vacuous_topology`
3. `halo_peer != nullptr` → `perform_peer_exchange` → `exchanged`
4. `halo_caller_managed[slot]` → `caller_declared`
5. otherwise → **fail**

`exchanged` is attempted **before** `caller_declared` is consulted. Reversed, a
legacy declaration would silently shadow a real exchange and §14's "no manual
halo-ready declarations may bypass required exchange" would be violated with
nothing reporting it.

`resolve_halo` deliberately does **not** short-circuit on an already-`ready`
state, so every authoritative resolve point really re-exchanges — as the Fortran
does.

### Provenance is recorded and is the Gate C1 criterion

`ContextData::halo_resolution[group]` records `HaloResolution ∈ {none,
vacuous_topology, caller_declared, exchanged}`, written only by `resolve_halo` and
`mark_halo_ready`, never by a numerical stage, and readable through
`halo_resolution_of`. Phase E asserts `exchanged` for all nine runtime-resolved
groups on all six tiles. Any `caller_declared` there would mean readiness came
from a declaration instead of data, and would make the verdict BLOCKED.

### Stream and event sequencing

`perform_peer_exchange` (`:330-420`). Both streams are `cudaStreamNonBlocking`,
so nothing orders them implicitly — not even legacy default-stream semantics.
Four ordering edges, all events, no `cudaStreamSynchronize` and no
`cudaDeviceSynchronize`:

```
compute_stream ==(event_compute_done)==> transfer_stream    producer visible to pack
pack on transfer_stream; record event_halo_pack
-------- BARRIER 1: every member has ENQUEUED its pack --------
transfer_stream waits on each of the FOUR donors' pack_done
device-to-device transfer, then unpack, on transfer_stream
record event_halo_xfer
compute_stream waits on event_halo_xfer                     consumer ordered
-------- BARRIER 2: every member has ENQUEUED its transfer --------
transfer_stream waits on every member's xfer_done           send buffer WAR guard
```

The donors come from the topology, not from all six members, so the wait stays
honest about who actually feeds this tile. Barrier 2 is **not** redundant:
without it a neighbour could enqueue the next group's pack into its send buffer
while my transfer was still reading it — two streams, no ordering, a
write-after-read hazard racecheck would be right to report.

The barriers are **host** rendezvous over *enqueue order only*. No thread waits
for the GPU inside an exchange. This is the same guarantee MPI gives through
`Isend`/`Irecv`/`Waitall`, which is why an MPI transport replaces
`HaloPeerGroup` without touching the pack, transform, unpack, buffers or
readiness machinery.

`HaloPeerGroup::abort_group()` was added as a one-way deadlock brake: a member
that fails before reaching a barrier releases its five peers, which then unwind
through the ordinary refusal path. A hung test hides which member failed; a red
one does not.

---

## §11 `H_GZ_FIRST` re-audit — FAITHFUL, for a different reason than 3A.9 recorded

3A.9 recorded that `H_GZ_FIRST` resolves only at `it == 1` and that nothing marks
it stale, and flagged it as suspicious. It is correct, but the reason is not the
one that was assumed.

`gz` **is** rebuilt every substep, *including its halo*:

* `dyn_core.F90:568-574` copies `gz ← zh` over the full `isd:ied, jsd:jed` box;
* `:971-977` rebuilds `gz = zh * grav` over `is-2:ie+2, js-2:je+2`, which is
  exactly `a2b_ord4`'s width-2 requirement (`a2b_edge.F90:157-159`, `:215-217`).

The per-substep MPI that makes this correct is the **`zh` exchange** `i_pack(4)`
at `:931`/`:934` → `:967`, **not** a `gz` exchange. The `it == 1` `gz` exchange
exists only because `gz` was computed interior-only at `:446-455` and `zh`'s
width-3 halo had to be seeded from it at `:558-563` for `update_dz_d`
(`nh_utils.F90:283-287`, `:297`).

All three enabling conditions were then verified in the CUDA source:

| Condition | Where |
|---|---|
| `H_ZH` width 3 resolved every substep | `runtime/fv3_gpu_runtime.cpp:1204` |
| `gz ↔ zh` copy covers the FULL halo box, both directions | `vertical/fv3_vertical_kernels.cu:508-519` (`gz_zh_sync_kernel`), launched ~`:624` |
| `gz = grav * zh` at halo 2 | `operators/wp_d_cuda.cu:47-50`, `h = 2` from `operators/wp_d_contract_cuda.cpp:44` |

**No code change required.** Also noted: `i_pack(5)` is overloaded with `pkc` at
`:932`/`:981` under `square_domain`, and `gz` flips units — metres between
`:568-574` and `Riem_Solver_c`, geopotential after.

---

## §12 `FV3_GPU_H_CWIND` and the D-wind group — one defect found and fixed

### C-grid winds

`uc`/`vc` are per-substep temporaries with **zero** cross-substep validity:
`d2a2c_vect` declares them `intent(out)` (`sw_core.F90:2806-2807`). `d2a2c_vect`
writes them narrow-band only; `p_grad_c` then writes **interior only**, so after
`dyn_core.F90:614` the halo band is numerically *inconsistent* — it is missing the
pressure-gradient increment, not merely narrow. That is precisely why the
unconditional `CGRID_NE` exchange at `:618`/`:640` must sit between `p_grad_c` and
`d_sw`. The runtime does exactly that (`runtime/fv3_gpu_runtime.cpp:1199`).

### D-grid winds

There is **one** logical `u`/`v` exchange per substep:

* `:402` is `#ifdef ROT3` → **DEAD**. `ROT3` is not defined by the GEOS build
  (`CMakeLists.txt` defines only `SINGLE_FV OVERLOAD_R4`, `MAPL_MODE SPMD
  TIMING`, `MOIST_CAPPA USE_COND`, `INTERNAL_FILE_NML`).
* `:1107-1110` is `#ifndef ROT3` → **LIVE**, guarded `it /= n_split`, and it
  prepares the **next** substep; its only consumer is
  `complete_group_halo_update` at `:504`. `it == 1` is seeded from
  `fv_dynamics.F90:630`.
* `mpp_get_boundary` at `:1092-1105` is an owned-shared-edge de-duplication, not
  a halo refresh.

### Provable halo widths

`u` i=3/j=3, `v` i=3/j=3, `uc` i=3/j=3, `vc` i=3/j=3, `zh` 3, `gz` 2 (consumed) /
3 (literal at `it == 1`). **Not** provable from source, where `ng = 3` is used
instead: `divgd` (reach scales with `nord`) and `w` (scales with `nord_w`).

### DEFECT FOUND AND FIXED: `d_sw` marked nothing stale

`d_sw` **destroys** `uc`/`vc` outright — the high-order divergence damping reuses
them as scratch, `sw_core.F90:1378-1389` (`nord > 0`, which the production
first-run config has), ported verbatim as `divdamp_divg_dx_kernel` /
`divdamp_divg_dy_kernel` (`dgrid/wp_b_divergence_cuda.cu:263`, `:275`) — and
writes `u`/`v`/`w`/`delp`/`pt` **interior only**. Yet it recorded nothing stale.

That was survivable only because every affected group happens to be re-resolved
before its next consumer. That is a property of the current stage list, not of the
state machine, and it left the recorded state saying "ready" about arrays whose
halo — and in `uc`/`vc`'s case whose interior — no longer held what the name
implies.

**Fix:** `dgrid/wp_b_dsw_contract_cuda.cpp`, at the end of
`launch_dgrid_dynamics_cuda`, marks `H_CWIND`, `H_DWIND_IN`, `H_W_IN` and
`H_DSCALAR` stale. Verified first that (i) no consumer of those four exists
between `d_sw` and their next resolve point, and (ii) `resolve_halo` does not
short-circuit on state, so the change **cannot** suppress an exchange. What it
buys is that a future consumer inserted into any of those intervals fails loudly
instead of reading divergence-damping scratch that still reads as a C-grid wind.

**Numerical semantics changed: NO.**

Build fix in the same edit: `DependencyState::mark_stale` returns plain `int`
(`runtime/fv3_gpu_contract.cpp:106`) while `fail()` takes `fv3_gpu_status`, giving
`invalid conversion from 'int' to 'fv3_gpu_status' [-fpermissive]`. Resolved by
testing `!= 0` and passing the literal `FV3_GPU_INVALID_ARGUMENT`, not by casting.

### Tests that fail on the specific corruptions §12 names

`tests/integration/test_halo_exchange_multitile.cu` runs `H_DWIND_IN` and
`H_CWIND` as separate groups and decodes every halo cell back to (tile, i, j, k,
sign). Three negative controls were injected into the **implementation**, one at a
time, rebuilt and run on the A100 (job 58077949):

| Fault injected | Result |
|---|---|
| `seam_component_sign()` forced to `+1` | **FAIL**, 24 failures — tile and index right, sign wrong |
| along-seam reversal dropped (`tp = t`) | **FAIL**, 132 failures — along-index mirrored |
| rotated-seam `u`/`v` partner remap dropped | **FAIL**, 48 failures — wrong staggering shift |

All three reverted; clean R8 and R4 both pass 20484 checks / 0 failures.

An **earlier version of this test survived fault 1.** It computed its expected
sign by calling `seam_component_sign()` — the function under test — so the fault
moved expectation and behaviour together. `expected_sign()` in the test is now a
local transcription of the FMS table. The implementation was correct throughout;
only the test was blind. This is recorded because it is the failure mode that
makes a passing test worthless.

Stale-data consumption is caught by a different mechanism: every array is
pre-filled with its **own** tile's provenance encoding, so a halo cell the
exchange fails to write decodes to the receiver's own tile. A no-op exchange
cannot pass.

---

## §13 Permanent integrated multi-tile test

Two tests, deliberately separate, because they prove different things.

### `tests/integration/test_halo_exchange_multitile.cu` — cell-by-cell seam correctness

All six tiles. Every element carries lossless provenance,
`value = (((tile*64 + (i+8))*64 + (j+8))*8 + k)`, max ≈ 206k, well inside float's
exact-integer range, so the same binary proves R8 and R4. After the exchange each
halo cell is decoded and checked for tile, index, level and sign.

Independently detectable, which is the list §13 asks for: wrong neighbour tile;
reversal not applied; `u`/`v` not swapped on a rotated seam; wrong sign; halo not
written at all; insufficient halo width. Owned interior asserted untouched.

The index formulae themselves are validated against a genuinely independent
oracle — FMS's own `fill_cubic_grid_halo` — by T9 of
`tests/runtime/test_halo_topology.cpp`.

### Phase E of `tests/integration/test_dispatch_smoke.cpp` — the integrated path IS wired to it

Six `fv3_gpu_context` objects, one per tile, `create.mpi_rank = tile - 1`,
`mpi_size = 6`, the **real** mosaic from `make_cubed_sphere_topology` (not
`install_topology`'s four nominal neighbours), all attached to one
`HaloPeerGroup`, and — the point — **no `declare_caller_managed_halos`**. If the
exchange does not work, Phase E does not run.

Six `std::thread`s, because a halo exchange is two-sided: tile 1's west halo comes
from its neighbour's owned cells *as they stand at this substep*, so six contexts
stepped one after another on one thread would copy the previous substep's values,
or on the first substep uninitialised memory, and would still produce finite
numbers. Setup and teardown are sequential on the main thread so a setup failure
cannot leave peers blocked at a barrier. The plain production dispatch is
installed, **not** `probes::`, whose file-scope counters six threads would race
on — Phase C already establishes per-stage reachability single-threaded, and
re-deriving it under a data race would weaken both claims.

Per-tile asymmetry via a tile-keyed phase shift on `u`/`v` (safe: unlike
`delp`/`pt` they carry no consistency constraint with `delz`/`zh`).

**The causality test.** Run 3 kicks the initial `u`/`v` of tile 1's **west
neighbour only** — queried from the topology rather than written down as "tile 5",
so the test cannot drift out of agreement with the mosaic it tests; it resolves to
tile 5 — and asserts **tile 1's** output moved. Tile 1's own input is
byte-identical between runs. The only route by which its answer can move is the
halo exchange. A control confirms the kick landed on tile 5 itself.

---

## §14 Integrated `n_split > 1` run

`fv3_gpu_dyn_core_step` with `n_split = 3`, through the public entry point, on all
six tiles. All nine production dispatch stages execute in authoritative order;
Phase C separately asserts by counting probes that the eight in-loop stages are
entered exactly `n_split` times and `post_loop_heating` exactly once, and that
`dgrid_height_transport` is reached (it is on no stage list — only WP-C's
`update_dz_d` calls it).

**No manual halo-ready declarations. Zero.** All nine runtime-resolved groups on
all six tiles report `HaloResolution::exchanged`.

Persistent residency asserted per tile: the registered host buffers are
byte-identical after the step and **before** the explicit `sync_to_host`.

---

## §15 What separates Gate C1 halo semantics from production MPI

Gate C1 establishes the **semantics**: neighbour connectivity, seam transforms,
pack/unpack, GPU residency, stream ordering, readiness provenance. Model B is one
process representing six tiles, exchanging device-to-device.

**Explicitly NOT established, and must not be claimed:**

* MPI rank ↔ tile mapping in a real decomposition
* MPI request lifecycle (`Isend`/`Irecv`/`Waitall`, buffer lifetime, tag space)
* CUDA-aware MPI
* GPUDirect / RDMA
* stream ↔ MPI progression interaction
* multi-node testing and scaling
* sub-tile decomposition (multiple ranks per tile), which re-introduces the two
  intra-tile diagonal classes §8 shows are empty for whole-tile ranks

`HaloPeerGroup` is the only piece an MPI transport replaces. The pack, transform,
unpack, buffers and readiness machinery are unchanged by that substitution, which
is the design property §15 asks about.

---

## §16 / §20 Sanitizer classification

### The 3A.9 baseline, re-derived rather than quoted

`/discover/nobackup/projects/imvi-nse/jli30/san_3a9/initcheck_full_r8.log`, same
binary (`fv3_dispatch_smoke_test`), same Phases A–D. Re-classified this milestone
by *(kernel, launcher chain)* rather than by raw count:

| count | kernel | launcher chain |
|---|---|---|
| 432 | `copy_corners_kernel` | `launch_copy_corners_cuda ← launch_del6_vt_flux_cuda ← launch_d_sw_cuda ← launch_dgrid_dynamics_cuda` |
| 240 | `copy_corners_kernel` | `launch_copy_corners_cuda ← launch_del6_vt_flux_cuda ← launch_dgrid_height_transport_cuda ← launch_update_dz_d ← launch_dgrid_height_stage` |

Total 672, exactly two signatures, reproducing the documented 432 + 240 split.
That agreement is what licenses the classifier to be used on the new logs.
racecheck 0 hazards / 0 errors / 0 warnings in both precisions.

### Methodological correction — raw totals are not comparable

An initial attempt to split the new log at the `# Phase E:` stdout banner gave
"pre-Phase-E = 5856, Phase E = 0", which is nonsense. In these logs **all**
sanitizer output (stderr, unbuffered) precedes **all** test output (stdout,
block-buffered) — in the 3A.9 baseline every one of the 672 records appears before
line 28322, where the first banner sits. Position in the log says nothing about
execution order, so the comparison has to be made on signature identity, and the
raw totals cannot be compared at all: 3A.9 ran one tile, this run adds eighteen
more tile-lifetimes.

### DEFECT FOUND BY initcheck AND FIXED — a genuinely new signature

Pre-fix R8 run (job 58078350, `warpa003`), `--print-limit 0` so counts are true:

| count | headline form | origin |
|---|---|---|
| 5856 | `Uninitialized __global__ memory read of size 8 bytes` | `copy_corners_kernel`, **the two classified chains only** |
| **24587** | **`Host API memory access error … on access by cudaMemcpy source`** | **`fv3::gpu::halo::launch_halo_transfer`, all of them** |

§16 says any new signature is a defect until proven otherwise. It was a defect.

**Root cause.** A slot is sized at the worst case over edges and staggerings
(`group_slot_elements` uses `max(nx,ny)+1` and `npz+1`) while the pack writes only
`width * along_extent * nk`. `launch_halo_transfer` copies the **whole slot**, so
every transfer moved a tail of never-written bytes.

**How it was distinguished from the obvious false positive.** `cudaMemcpyAsync`'s
source is checked by initcheck at *enqueue* time, and barrier 1 only guarantees the
donor's pack has been *enqueued* — so an async-ordering false positive was the
competing explanation, and it would have been the convenient one to accept. It was
tested rather than assumed, by extracting the reported uninitialized offset from
all 24587 records:

```
copy size:      24587 × 2376 bytes  (= 297 doubles = 3 rings × 9 along × 11 levels)
offset  70-80%:   144
offset  80-90%:  4607
offset 90-100%: 19836
offset ==    0:      0      <-- decisive
```

The packed prefix ends at 240/297 = 80.8% for a layer-centred field, 264/297 =
88.9% for an interface field and 270/297 = 90.9% for a staggered one. Every report
lands beyond a packed prefix and **not one lands at offset 0**. An enqueue-ordering
false positive would be uniformly distributed and would include offset 0. It is the
slot tail.

**Result impact: NONE, and provably so.** The tail is never read — both sides
derive the same along-extent from the donor edge and the source field's staggering,
and the unpack refuses a mismatch rather than clamping — and the multi-tile test
decodes all 20484 halo cells with 0 failures.

**Fix** (`runtime/fv3_gpu_runtime.cpp`, `attach_halo_peer_group`): `cudaMemset`
both buffers once at allocation. Chosen over shortening the copy because the copy
length would have to come from the **donor's** per-field `nk`, which the receiving
side does not have; and because a report that is merely rarer is worse than none —
24587 known-benign records would bury the one that matters. One memset outside the
acoustic loop, so no per-substep cost and no effect on persistent residency.
**Numerical semantics changed: NO.**

The `group_slot_elements` comment claiming the tail "is never read" was also
amended: true, but it *is* copied, and never-read is not never-touched.

### Authoritative post-fix classification

Job 58078350 was cancelled once the R8 diagnosis was complete rather than left to
finish an R4 pass of a binary that was already superseded; its logs are archived
under `negctl_3a10/prefix/`. The authoritative sweep against the final tree —
`ctest` plus memcheck, initcheck and racecheck, in both precisions — is Slurm job
**58079171**, logs `negctl_3a10/final_*`.

#### R8 — complete

| tool | result |
|---|---|
| `ctest` R8 | `100% tests passed, 0 tests failed out of 19`, EXIT=0 |
| memcheck R8 | `ERROR SUMMARY: 0 errors`; Phase E PASS; smoke test PASS |
| racecheck R8 | `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` |
| initcheck R8 | `ERROR SUMMARY: 5856 errors`, **one headline form only** |

The `Host API … cudaMemcpy source` form is **entirely gone** — 24587 → 0. Every
remaining record is `Uninitialized __global__ memory read of size 8 bytes`.

*(A grep of record headlines counts 5855, not 5856. The missing one is at
`final_initcheck_r8.log:28535`, where a test `printf` of `  u    ` landed
mid-line ahead of the `=========` prefix. It is a stdout/stderr interleaving
artifact of the shared log, not a differing count; the classifier below strips the
prefix and recovers all 5856.)*

#### Attribution by host frame — the comparison §16 actually asks for

Raw totals are not comparable across differing workloads, so instead of comparing
them each record was attributed to the **host phase function** in its own
backtrace. This is a real discriminator, unlike log position, because Phase E runs
on six `std::thread`s through `six_tile_run` while Phases A–D run inline:

| host phase | signature `d_sw` | signature `height` | total | vs. 3A.9 baseline |
|---|---|---|---|---|
| `run_phase_b` | 144 | 48 | 192 | **identical** |
| `run_phase_c` | 288 | 192 | 480 | **identical** |
| `six_tile_run` (new) | 2592 | 2592 | 5184 | new workload |
| | 3024 | 2832 | **5856** | |

where the two signatures are exactly the two authoritative chains re-derived
above:

- `d_sw` = `copy_corners_kernel ← launch_copy_corners_cuda ← launch_del6_vt_flux_cuda ← launch_d_sw_cuda ← launch_dgrid_dynamics_cuda`
- `height` = `copy_corners_kernel ← launch_copy_corners_cuda ← launch_del6_vt_flux_cuda ← launch_dgrid_height_transport_cuda ← launch_update_dz_d ← launch_dgrid_height_stage`

Two conclusions, both stronger than a count comparison:

1. **The 3A.9 classification is preserved exactly.** The pre-existing phases
   contribute 192 + 480 = **672** records — the documented total, unchanged, and
   unchanged in its per-signature split. Halo integration did not perturb the
   classified dataflow, which is the condition §16 sets for retaining the
   classification.
2. **Every new record is Phase E, and every new record carries one of the same two
   signatures.** `distinct = 2` over the whole log. The `d_sw` arm scales exactly
   as 6 tiles × `n_split` 3 × 144, the per-substep count `run_phase_b` establishes
   on one tile. **No new signature exists.**

The classifier collapses a trailing `launch_halo_transfer` host frame, which
appears on a small number of records under the six-thread run. That frame is
impossible in real control flow — halo transfer does not call `d_sw` — so it is a
host-backtrace stitching artifact of unwinding one thread's stack while another is
mid-launch, not a second origin. The device-side kernel and the CUDA-side launcher
chain are identical in those records.

**initcheck R8 verdict: ONLY PREVIOUSLY CLASSIFIED AUTHORITATIVE DEAD READS.**
Not PASS, not clean — the 672 baseline records remain, plus 5184 further instances
of the same two authoritative `del6_vt_flux` `nord == 1` dead reads produced by the
larger workload. Both signatures were established in 3A.9 as reads of cells that
the authoritative Fortran also never initialises and whose values never reach a
result.

#### R4 — complete

Job 58079171 finished `COMPLETED`, `ExitCode 0:0`, elapsed 01:40:08, all six
stages run (`sacct`; every stage banner reports `EXIT=0`).

| tool | result |
|---|---|
| `ctest` R4 | `100% tests passed, 0 tests failed out of 19`, EXIT=0 |
| memcheck R4 | `ERROR SUMMARY: 0 errors`; Phase E PASS; smoke test PASS |
| racecheck R4 | `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` |
| initcheck R4 | `ERROR SUMMARY: 5856 errors`, **one headline form only**, `distinct = 2` |

The same attribution, against the 3A.9 **R4** baseline this time:

| host phase | `d_sw` | `height` | total | vs. 3A.9 R4 baseline |
|---|---|---|---|---|
| `run_phase_b` | 144 | 48 | 192 | **identical** |
| `run_phase_c` | 288 | 192 | 480 | **identical** |
| `six_tile_run` (new) | 2592 | 2592 | 5184 | new workload |
| | 3024 | 2832 | **5856** | |

R4 is cell-for-cell the same picture as R8, which is itself corroborating: the two
precisions read the *same* uninitialised cells the same number of times. One
independent confirmation that these are reads of `fv_real` model data and not of
some precision-independent metadata: every R8 record is `of size 8 bytes` and every
R4 record is `of size 4 bytes`.

**initcheck R4 verdict: ONLY PREVIOUSLY CLASSIFIED AUTHORITATIVE DEAD READS.**

#### §16 / §20 verdict

| tool | R8 | R4 |
|---|---|---|
| memcheck | **PASS** — 0 errors | **PASS** — 0 errors |
| racecheck | **PASS** — 0 hazards, 0 errors, 0 warnings | **PASS** — 0 hazards, 0 errors, 0 warnings |
| initcheck | **ONLY PREVIOUSLY CLASSIFIED AUTHORITATIVE DEAD READS** | **ONLY PREVIOUSLY CLASSIFIED AUTHORITATIVE DEAD READS** |

initcheck is explicitly **not** called clean or PASS: the authoritative
`del6_vt_flux` `nord == 1` dead reads remain, in both precisions. What §16 required
was that the 3A.9 classification be *retained* and that no new signature appear,
and both hold exactly — the pre-existing phases reproduce 672 record-for-record in
both precisions, and the 5184 new records are Phase E instances of the same two
signatures. The one genuinely new signature this milestone produced (24587
`launch_halo_transfer` records) was diagnosed as a real defect in new code and
fixed, not merged into the old classification.

## §17 Stream interaction audit

Every interaction is an event. `order_after(signaler, waiter, ev)`
(`runtime/fv3_gpu_runtime.cpp:46`) is the only mechanism; no exchange path calls
`cudaStreamSynchronize` or `cudaDeviceSynchronize`. Both runtime streams are
`cudaStreamNonBlocking`, so default-stream semantics are never relied on. The four
edges are listed in §10.

### The stress test

Phase E runs the **same six-tile configuration twice** and requires the two runs
to agree **bitwise** on all six tile checksums.

Six host threads enqueue into six pairs of non-blocking streams and the OS
schedules them; nothing pins or paces them. If any of the four ordering edges were
missing, which side of the race won would vary with that interleaving and the two
runs would differ. A bitwise match is therefore evidence about the **events**, not
about the arithmetic.

The 3A.9 D2H race is the precedent: it produced plausible numbers at `n_split = 3`
and wrong ones at `n_split = 1`, and the difference between those two was exactly
a scheduling difference. Comparing two identical runs catches that class of defect
without knowing the right answer. Bitwise rather than tolerance-based
deliberately — there is no tolerance at which a stream race is acceptable.

**Result: identical in R8 and in R4.**

---

## §18 Regression suites — R8 and R4

Full `ctest` on a real A100 (`warpa003`), run twice: job 58078341 before the §20
buffer fix and job 58079171 after it, on the final tree.

| Precision | job 58078341 (pre-fix) | job 58079171 (final tree) |
|---|---|---|
| R8 | **19/19 PASS** | **19/19 PASS**, `0 tests failed out of 19`, EXIT=0 |
| R4 | **19/19 PASS** | **19/19 PASS**, `0 tests failed out of 19`, EXIT=0 |

The final-tree run is the authoritative one; the pre-fix run is recorded because it
is what the §20 diagnosis was performed on.

Covers WP-A, WP-B (3A.4–3A.8), WP-C, WP-D, WP-E, the 3A.9 integrated path, the
halo topology unit tests, the multi-tile exchange test, and the dispatch smoke
test including the new Phase E. **No numerical regression.**

---

## §19 A100 execution — R8 and R4

`warpa003`, CUDA 12.5.82, GCC 12.3.0, `sm_80`.

| Claim | R8 | R4 |
|---|---|---|
| halo pack kernels execute | PASS | PASS |
| seam transforms execute | PASS | PASS |
| unpack executes | PASS | PASS |
| readiness transitions execute (all 9 groups × 6 tiles = `exchanged`) | PASS | PASS |
| `n_split = 3` | PASS | PASS |
| full `fv3_gpu_dyn_core_step` | PASS | PASS |
| host state unchanged until explicit `sync_to_host` | PASS | PASS |
| output finite and non-degenerate | PASS | PASS |
| multi-tile decode, 20484 checks | 0 failures | 0 failures |
| two identical six-thread runs agree bitwise | PASS | PASS |
| seam causality (kick tile 5 → tile 1 moves) | 1.87e+01 | 9.41e+01 |
| control (tile 5 itself moves) | 3.65e+03 | 9.91e+02 |

---

## §21 Source-level reconciliation

| Field group | Stagger | Halo width | Neighbour mapping | Rotation | CUDA implementation | Status |
|---|---|---|---|---|---|---|
| `H_DWIND_IN` / `H_DWIND_NEXT` (`u`,`v`) | N, E | 3 | mosaic, per seam | slot-keyed, seam rotation | `halo_group_spec`, `launch_halo_*` | MATCHES FORTRAN |
| `H_CWIND` (`uc`,`vc`) | E, N | 3 | mosaic | slot-keyed (transposed staggering) | same | MATCHES FORTRAN |
| `H_W_IN` (`w`) | C | 3 | mosaic | none (scalar) | same | MATCHES FORTRAN |
| `H_GZ_FIRST` (`gz`) | C | 3 (`it==1`) | mosaic | none | same; see §11 | MATCHES FORTRAN |
| `H_DIVGD` (`divgd`) | corner | 3 | mosaic | none (scalar at CORNER, legal on a rotated seam) | same | MATCHES FORTRAN |
| `H_DSCALAR` (`q_con`,`delp`,`pt`,`cappa`) | C | 3 | mosaic | none | same | MATCHES WITH EXPLAINED STRUCTURAL DIFFERENCE — `cappa` is authoritatively exchanged only at `it == 1`; over-exchanging it is idempotent and cannot change the answer |
| `H_ZH` (`zh`) | C | 3 | mosaic | none | same | MATCHES FORTRAN |
| `H_PKC2` (`pkc`) | C | 3 | mosaic | none | same | MATCHES FORTRAN — the group NAME is a misnomer (width is 3, not 2) |
| `H_HEAT` (`heat_source`) | C | 3 | mosaic | none | same | MATCHES FORTRAN — live only when `d_con > 0` |
| `H_Q_INLINE` | — | — | — | — | — | NOT LIVE |
| tile-vertex corner blocks | — | — | — | — | local kernels in WP-B/C/D | MATCHES FORTRAN — §8 proves the exchange authoritatively never fills them |

### Declared structural differences

1. **Where the transform is applied.** FMS signs and transposes at **pack** and
   unpacks verbatim. This implementation packs verbatim donor-local and applies
   reversal, component selection and sign at **transfer/unpack**. Algebraically
   identical — the same permutation and the same sign composed in the other
   order — and it keeps the pack kernel independent of the receiver's topology.
2. **`cappa` over-exchange**, above.

### Corner-path reconciliation (cross-check performed this milestone)

| Item | Verdict |
|---|---|
| D-grid vector corner sign placement (SW/NE negative) | MATCHES FORTRAN — `dgrid/wp_b_divergence_cuda.cu:396-430`, all eight statements index-for-index identical to `fv_mp_mod.F90:1351-1354`/`:1363-1366` |
| BGRID scalar corner fill | MATCHES FORTRAN — `:357-379` vs `fv_mp_mod.F90:1119-1126`/`:1134-1141`, unsigned |
| `sw_core.F90:1392` argument order | MATCHES FORTRAN — `dgrid/wp_b_divergence_cuda.cu:723-726` passes `vc` as `x` and `uc` as `y`, **not** "corrected" to natural naming |
| `copy_corners` | MATCHES FORTRAN — `dgrid/wp_b_cuda.cu:94-139`, unsigned, all eight per-corner formulas and both halo strips correct |
| dead opposite-sign D-grid variant | NOT PORTED — zero traces |
| C-grid / A-grid signed corner variants | NOT IMPLEMENTED IN CUDA, correctly: no production `dyn_core` call site needs them |

Fixed this milestone: two comment citations in
`dgrid/wp_b_divergence_cuda.cuh` pointed at the *dead* `:1350`/`:1362` lines
rather than the live `:1351`/`:1363`. Numerical semantics unchanged; the risk was
that a reader following the citation would land on the dead variant and "fix" the
signs to match it.

### `physical_edges` reconciliation (3A.9 residual #4 — CLOSED)

3A.9 recorded that the panel-edge predicate appeared to be written twice and asked
whether that was a live discrepancy. Traced this milestone. **It is neither a live
numerical discrepancy nor provable duplication**, because the two spellings are
functions of *disjoint inputs*:

| Site | Location | Predicate |
|---|---|---|
| A | `operators/wp_d_contract.cpp:398-405` | `g(topology.neighbors[], topology.tile)` — "neighbour is a different tile" |
| B | `dgrid/wp_b_contract_cuda.cpp:236-239` | `f(is, ie, js, je, npx, npy)` |
| C | `dgrid/wp_b_dsw_contract_cuda.cpp:488-491` | identical to B |
| D | `dgrid/wp_b_divergence_cuda.cu:813-816` | from B/C's flags; corners as the edge **conjunction** |
| E | `dgrid/wp_b_divergence_cuda.cu:68-78` | re-derives `f` a third time as a *guard*, rejecting a caller whose flags disagree |

`operators/` contains **zero** occurrences of `npx` or `npy`, so no algebra
connects A to B/C/D — only an invariant.

**B/C/D/E are the ones that match the authoritative Fortran.** `ie+1 == npx` is
written in exactly that form at `a2b_edge.F90:164`/`:181` and
`fv_grid_utils.F90:294-298`; all use `==`, none uses `>=`, all use the compute
bounds. Site A is a *proxy* that coincides only when the rank owns a whole tile.

**Why there is no divergence today.** The only production topology installer,
`fv3_gpu_set_cubed_sphere_topology` (`runtime/fv3_gpu_runtime.cpp:925-941`),
refuses `mpi_size != 6` and delegates to `make_cubed_sphere_topology`, whose
mosaic table never names a tile's own tile — so A is unconditionally all-true for
every topology it produces. The Gate C1 bounds make B/C/D all-true as well, and
D's corner conjunction matches `topo.sw_corner`. Both spellings equal
`{T,T,T,T}`, matching Fortran.

**Why it is still not interchangeable.** Nothing enforces the invariant and a
disagreeing tuple is reachable through the public C ABI: `fv3_gpu_create`
validates only `is<=ie`, `js<=je`, `npz>=1` and never compares the box to
`npx/npy`. For `is=5, ie=12, js=5, je=12` with `npx=npy=17`, both calls succeed;
B/C/D give all-false (correct), A gives all-true (wrong). On that input WP-B
**fails closed** via the `:68-78` guard, while WP-D would **fail open** — masked
only by `dgrid_dynamics` preceding `nh_pressure` in the step, i.e. by stage
ordering, not by any check inside WP-D.

**Status: NOT LIVE for Gate C1** (whole-tile ranks only, `mpi_size == 6`
enforced). Recorded as a hardening item, not a defect in the supported path: the
canonical predicate belongs in WP-E as a function of `(DomainBounds,
TopologyView)` that refuses `npx == 0`, derives corners as the edge conjunction as
`fv_grid_utils.F90:294-298` does — which Site D already does and WP-D's
`physical_corners` does **not** — and cross-checks the neighbour list against the
index result, demoting the neighbour spelling from a second definition to a
validated consistency check. Deferred deliberately: sub-tile decomposition (§15)
is the point at which it becomes live, and it is a WP-D/WP-E interface change, not
a halo-exchange change.

**Documentation defect found and fixed in the same trace.**
`runtime/fv3_gpu_runtime.cpp:604-616` asserted that "WP-D's `physical_edges`
feeding `a2b_ord4`" already refuses a zero `npx` at its binder. **That mitigation
does not exist** — WP-D never reads `npx`, so `bind_wp_d` has no value to check.
The real protection is `cgrid/fv3_cgrid_predictor.cu:251-254` plus
`dgrid/wp_b_dsw_contract_cuda.cpp:78-80`, together with `validate_dispatch`
(`runtime/fv3_gpu_contract.cpp:119`) requiring all ten stages so a WP-D-only table
cannot be installed. The comment now says that. A comment claiming a check that
does not exist is worse than no comment, because the next reader stops looking.
Numerical semantics changed: NO.

**Known discrepancy in the supported Gate C1 path: NONE.**

### Carried forward — outside the supported Gate C1 subset

* **`MOIST_CAPPA` / `USE_COND`.** Both are unconditionally ON in the real GEOS
  build (`CMakeLists.txt:88`), while `fence_production_config` currently
  **rejects** them. This is a scope boundary of the supported subset, not a
  discrepancy inside it — but it means the currently-supported configuration
  cannot yet be exercised by a real GEOS build, and it must be closed before
  production integration.
* `corner_edge_service` remains a deliberate fail-stub
  (`corner_service_not_implemented`,
  `runtime/fv3_gpu_production_dispatch.cu:97`), justified as NOT REQUIRED for
  `grid_type = 0` — §8 now independently supports that justification — but
  blocking `nf_ke > 1`.
* `d4dmp`/`Ray_fast` one-time initialisations not yet lifted out of the loop.
* Config values still to be confirmed against the GEOS namelist.
* Whether the 3A.8 `d_sw` driver performs `dyn_core.F90:819-826`'s 3-D
  `heat_source`/`diss_est` accumulation.
* `npx`/`npy == 0` pre-mutation gap (partially addressed: `attach_halo_peer_group`
  and `resolve_halo` both now reject it).
* Test-coverage caveat: `tests/dgrid/test_wp_b4_divergence_cuda.cpp:432-472`'s
  reference corner fills are statement-identical to the CUDA kernels, so those
  parts are transcription-consistency checks, not independent derivations. The
  independent confirmation came from the Fortran cross-check above.

---

## §22 Gate C1 retest

| # | Criterion | Verdict |
|---|---|---|
| 1 | all production-required numerical stages implemented | PASS |
| 2 | all production-required dispatch slots resolved | PASS |
| 3 | authoritative stage ordering | PASS |
| 4 | authoritative `n_split` loop | PASS |
| 5 | truthful required halo exchange | PASS |
| 6 | execution through `fv3_gpu_dyn_core_step` | PASS |
| 7 | R8 / R4 PASS | PASS / PASS |
| 8 | A100 PASS | PASS |
| 9 | memcheck PASS | **PASS / PASS** (0 errors R8, 0 errors R4) |
| 10 | racecheck PASS | **PASS / PASS** (0 hazards both) |
| 11 | initcheck clean or only previously classified authoritative dead reads | **ONLY PREVIOUSLY CLASSIFIED / ONLY PREVIOUSLY CLASSIFIED** |
| 12 | no source-level discrepancy | PASS |
| 13 | persistent device residency | PASS |

Criterion 5 deserves its evidence spelled out, because it is the one §27 turns on.
Grepping the whole non-test tree, the only producers of halo readiness are
`resolve_halo` (`runtime/fv3_gpu_runtime.cpp:479`, reached only *after*
`perform_peer_exchange` returns) and `mark_halo_ready`
(`runtime/fv3_gpu_runtime.cpp:817`, the explicit caller-declaration API, which
stamps `HaloResolution::caller_declared`). **No file under `cgrid/`, `dgrid/`,
`vertical/` or `operators/` calls either one** — no numerical stage can declare its
own halo ready. Phase E calls neither `mark_halo_ready` nor
`mark_halo_caller_managed`, and asserts all nine runtime-resolved groups on all six
tiles report `HaloResolution::exchanged`, with zero `caller_declared`.

All thirteen criteria are now answered, and none was relaxed. Criterion 11 is
satisfied by its second limb (previously classified authoritative dead reads), not
its first (clean) — stated that way deliberately.

**Gate C1: PASS.**

What that does and does not assert. It asserts that the supported production
first-run dyn_core path executes end to end on a real A100 through
`fv3_gpu_dyn_core_step`, with `n_split = 3`, across six real cubed-sphere tiles,
with every required halo group resolved by an exchange that actually packed,
transferred, transformed and unpacked the data — verified cell by cell against an
independent oracle, with three negative controls that fail when the sign, the seam
reversal or the u/v component remap is removed. It asserts persistent device
residency: host buffers are byte-identical until an explicit `sync_to_host`. It
asserts memory- and race-cleanliness in both precisions.

It asserts **nothing** about numerical agreement with the authoritative Fortran.
That is Prompt 6 and it has not been run.

---

## §23 Numerical validation

**Numerical validation against authoritative runtime Fortran savepoints: NOT
RUN.** Prompt 6 remains NOT STARTED. No Fortran savepoints have been generated.

---

## §24 Files changed and files added

Everything below is source, tests, build system or documentation. No build
directory, object file, executable, Slurm log or sanitizer log is staged.

### Added

| File | Purpose |
|---|---|
| `cuda/dyn_core/include/fv3_gpu_halo_topology.hpp` | mosaic connectivity, edge transform descriptors, `seam_component_sign`, `seam_is_reversed`, `source_shift`, `halo_dest_index` |
| `cuda/dyn_core/runtime/fv3_gpu_halo_topology.cpp` | `make_cubed_sphere_topology`, `cubed_sphere_neighbor` |
| `cuda/dyn_core/include/fv3_gpu_halo_exchange.hpp` | group field tables, slot layout arithmetic, kernel launch declarations |
| `cuda/dyn_core/runtime/fv3_gpu_halo_exchange.cu` | `launch_halo_pack`, `launch_halo_transfer`, `launch_halo_unpack` |
| `cuda/dyn_core/include/fv3_gpu_halo_peer.hpp` | Model B transport: `PeerMember`, `HaloPeerGroup` (barrier, layout agreement, `abort_group`) |
| `cuda/dyn_core/tests/runtime/test_halo_topology.cpp` | topology/transform unit tests, incl. T9 against FMS `fill_cubic_grid_halo` |
| `cuda/dyn_core/tests/integration/test_halo_exchange_multitile.cu` | six-tile cell-by-cell seam decode, 20484 checks |
| `docs/agent_handoff/milestone_3a10_gpu_halo_exchange.md` | this document |

### Modified

| File | Change | Numerical semantics changed |
|---|---|---|
| `cuda/dyn_core/CMakeLists.txt` | new sources folded into `fv3_dyn_core_gpu`; two new test targets; `Threads::Threads` on the smoke test | NO |
| `cuda/dyn_core/include/fv3_gpu_context.hpp` | exchange state: `halo_peer`, `halo_peer_tile`, `halo_send`, `halo_recv`, `halo_buffer_elements`, `event_halo_pack`, `event_halo_xfer` | NO |
| `cuda/dyn_core/include/fv3_gpu_internal.hpp` | `HaloResolution` provenance; `attach_halo_peer_group` / `detach_halo_peer_group` / `halo_resolution_of` (internal, deliberately not public ABI) | NO |
| `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` | `perform_peer_exchange`; `resolve_halo` fail-closed and provenance-recording; attach/detach; §20 buffer `cudaMemset`; corrected a comment that claimed a WP-D `npx` check which does not exist | NO |
| `cuda/dyn_core/include/fv3_gpu_api.h` | `TopologyView` plumbing / `set_topology` | NO |
| `cuda/interface/fv3_gpu_interface_mod.F90` | matching ISO_C_BINDING declarations | NO |
| `cuda/dyn_core/dgrid/wp_b_dsw_contract_cuda.cpp` | §12 defect fix: `d_sw` now marks `H_CWIND`/`H_DWIND_IN`/`H_W_IN`/`H_DSCALAR` stale | NO |
| `cuda/dyn_core/dgrid/wp_b_divergence_cuda.cuh` | comment citations corrected from the dead `:1350`/`:1362` to the live `:1351`/`:1363` | NO (comment only) |
| `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp` | Phase E: six-tile threaded integrated exchange, provenance / residency / determinism / causality assertions | NO |

**Authoritative Fortran: not modified. `geos-gtfv3`: not modified.** Both were
read only.

### Nontrivial fixes, in the form §25 asks for

| # | File | Error / defect | Root cause | Change | Semantics changed |
|---|---|---|---|---|---|
| 1 | `wp_b_dsw_contract_cuda.cpp` | `d_sw` marked no halo stale despite destroying `uc`/`vc` and writing `u`/`v`/`w`/`delp`/`pt` interior-only | state machine recorded producer effects nowhere; correctness rested on the accident that every group is re-resolved before its next consumer | mark the four groups stale | NO |
| 2 | same | `invalid conversion from 'int' to 'fv3_gpu_status' [-fpermissive]` | `DependencyState::mark_stale` returns plain `int`; `fail()` takes the enum | test `!= 0`, pass the literal status — not a cast | NO |
| 3 | `fv3_gpu_runtime.cpp` | `halo_exchange_is_vacuous` failed **OPEN** on an unset topology (3A.9 residual) | absent topology treated as "nothing to exchange" | `not_supplied` now fails; `npx`/`npy == 0` rejected in attach and resolve | NO |
| 4 | `fv3_gpu_halo_peer.hpp` | `layout_` default-initialised to `0`, a legal compacted field count, reading as "already agreed" | value-initialisation of `std::array<int32_t>` | `layout_.fill(-1)` | NO |
| 5 | `fv3_gpu_halo_peer.hpp` | a member failing before a barrier hung its five peers | no abort path | one-way `abort_group()`; peers unwind through the ordinary refusal path | NO |
| 6 | `CMakeLists.txt` | dependency cycle: `fv3_halo_exchange` linked `PUBLIC fv3_dyn_core_gpu` while `resolve_halo` calls `launch_halo_*` | exchange kernels made a separate library | folded into `fv3_dyn_core_gpu` via `target_sources` | NO |
| 7 | `test_halo_exchange_multitile.cu` | negative control did **not** fail | test computed its expected sign by calling the function under test | `expected_sign()`, a local transcription of the FMS table | NO |
| 8 | `test_dispatch_smoke.cpp` | `'fv3::gpu::halo::TileEdge' has not been declared` | `TileEdge` is in `fv3::gpu`, not the nested `halo` namespace | unqualified, via the file's `using namespace fv3::gpu` | NO |
| 9 | `wp_b_divergence_cuda.cuh` | comments cited the dead opposite-sign D-grid variant | off-by-one into a commented-out block whose sign placement coincides with the live C-grid routine | cite `:1351`/`:1363` | NO (comment only) |
| 10 | `fv3_gpu_runtime.cpp`, `fv3_gpu_halo_exchange.hpp` | initcheck: 24587 `Host API memory access error … cudaMemcpy source`, all from `launch_halo_transfer` | slot sized at the worst case, pack writes only `along_extent`, transfer copies the whole slot | `cudaMemset` both buffers once at attach; comment corrected | NO |
| 11 | `fv3_gpu_runtime.cpp` | a comment asserted WP-D's `physical_edges` refuses a zero `npx` at its binder | false — `operators/` contains no `npx`/`npy` reference at all, so `bind_wp_d` has no value to check | comment rewritten to name the real protection, which is stage *ordering* plus `validate_dispatch`, not validation | NO (comment only) |

No fix required a change to scientific semantics, so no escalation was needed.

---

## §25 Git discipline

Branch `cuda/gpu-bringup`. Starting commit
`b35f0dab59d3021bf4ffb6499da1054cd0f7d3ec`. `git status`, `git diff` and
`git diff --cached` are to be reviewed immediately before committing.

`build_cuda_r8/`, `build_cuda_r4/`, object files, executables, Slurm logs and
sanitizer logs are excluded. All sanitizer, negative-control and Slurm artefacts
live outside the repository:

| Path | Contents |
|---|---|
| `/discover/nobackup/projects/imvi-nse/jli30/negctl_3a10/` | this milestone's scripts and logs; `final_*` = job 58079171 (authoritative), `prefix/` = job 58078350 (pre-fix, kept for the §20 diagnosis), `*.orig` + `fault_*` = negative-control sources and binaries |
| `/discover/nobackup/projects/imvi-nse/jli30/san_3a9/` | the 3A.9 initcheck/racecheck baseline the classification is compared against |

**Nothing is pushed.** Push and merge require an explicit request.

---

## Next phase

Gate C1 is closed, so **Prompt 6A — authoritative Fortran savepoints** is the next
phase, per §23's condition. The savepoint locations named in
`docs/agent_instructions/project_architecture.md` ("Numerical Validation") are
dyn_core entry, after the C-grid predictor, after the C pressure stage, after
D-grid dynamics/transport, after the vertical stage / `update_dz_d`, after the D
pressure/NH stage, end of acoustic substep, and dyn_core exit. That file also
requires identifying the *first* divergence rather than validating only final
output.

### Carried forward — not blockers for Gate C1, but read before Prompt 6

1. **`MOIST_CAPPA` / `USE_COND` are unconditionally ON in the real GEOS build**
   (`CMakeLists.txt:88`) while `fence_production_config` **rejects** them. The
   supported subset as implemented therefore cannot yet run inside real GEOS. This
   is the largest gap between "Gate C1 passes" and "runs in production", and it
   should be resolved before or alongside Prompt 6, because savepoints generated
   from a `MOIST_CAPPA` GEOS build will exercise a path the fence refuses.
2. **`corner_edge_service` is a deliberate fail-stub**
   (`corner_service_not_implemented`, `runtime/fv3_gpu_production_dispatch.cu:97`).
   Justified as NOT REQUIRED for `grid_type = 0`, but it blocks `nf_ke > 1`.
3. **A canonical panel-edge predicate belongs in WP-E** — §21's `physical_edges`
   reconciliation found two formulations, `f(is,ie,js,je,npx,npy)` in WP-B and
   `g(topology.neighbors[], tile)` in WP-D, which agree for every Gate C1
   configuration but diverge for a sub-tile decomposition reachable through the
   public ABI. WP-B fails closed there; WP-D would fail open, masked only by stage
   ordering. One predicate over `(DomainBounds, TopologyView)` that refuses
   `npx == 0` and cross-checks the neighbour list would retire the whole class.
4. **Remaining MPI work** is enumerated in §15 and none of it is claimed: rank
   mapping, MPI request lifecycle, CUDA-aware MPI, GPUDirect, and multi-node
   testing.
5. Minor: `d4dmp` / `Ray_fast` one-time initialisations are not hoisted out of the
   acoustic loop; whether the 3A.8 `d_sw` driver performs `dyn_core.F90:819-826`'s
   3-D `heat_source` / `diss_est` accumulation is unconfirmed; and
   `tests/dgrid/test_wp_b4_divergence_cuda.cpp:432-472`'s reference corner fills
   are statement-identical transcriptions, so they check transcription rather than
   independently verifying the fill.
