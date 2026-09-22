# Prompt 52C: FV_MAPZ CUDA Production Integration — Architecture Resolved, Implementation Not Started

Branch: `optimization/cuda-fv3-performance`, HEAD `33e10c5` (tag
`cuda-perf-baseline-c512`), unchanged (no source edits made in this
prompt). Continues Prompt 52 (port plan), 52A (Fortran baseline fixtures),
52B (standalone CUDA `map_scalar`/`scalar_profile`/`cs_limiters`,
correctness-validated).

This prompt's mandate was to integrate 52B's standalone CUDA kernel into
the real FV3 production timestep, through build and multi-stage GPU
validation. Stages 0-1 (git check, production-sequence mapping) are
complete and resolve the central structural question this prompt was
launched to answer. That resolution surfaces a second, previously
unknown scope item (§4) that must be closed, with its own fixture-level
validation, before any kernel/dispatch code is written. No implementation,
build, or GPU job was attempted.

## 1. Stage 0 — Git/worktree check

`pwd`/branch/HEAD/status/log all as expected: branch
`optimization/cuda-fv3-performance`, HEAD `33e10c5`, three pre-existing
modified tracked files and the 52B untracked CUDA files present, no
unexpected state. No cleanup needed.

## 2. Stage 1 — Production sequence, resolved

`fv_dynamics.F90`'s `do n_map=1,k_split` loop (lines 651-1019) calls, in
this order, every iteration: `fv3_gpu_interface_step_dyn` (748, CUDA
acoustic substep) → `fv3_gpu_interface_pe_halo` (763) → tracer halo start
(773) → `tracer_2d`/`tracer_2d_1L` (893-907, host, Fortran, untouched) →
`Lagrangian_to_Eulerian` (952, host, Fortran — contains the `map_scalar`
calls). CUDA step and L2E are interleaved once per `n_map` iteration, not
"CUDA runs all k_split substeps, then L2E runs once" as originally framed.

**Central finding — contradicts Prompt 52's "no new transfer needed"
prediction.** `cuda/interface/fv3_gpu_dyncore_bridge.F90:17-27` states the
architecture explicitly: "What does cross per call is one H2D of the
mutable prognostic state before the step and one D2H after it. That is
required, not convenient: between two successive dyn_core calls
fv_dynamics runs tracer advection and the Lagrangian-to-Eulerian remap,
both of which rewrite delp/pt/u/v/w/delz/pe/pk/pkz **on the host**.
Remapping is one of the synchronization boundaries project_architecture.md
names explicitly." Confirmed in code: `fv3_gpu_sync_from_host` (H2D, line
337) at call entry, `fv3_gpu_sync_to_host` (D2H, line 361) at call exit,
of the full 26-field mutable-state mask (`FV3_FID_U`..`FV3_FID_WS`,
`fv3_gpu_abi_mod.F90:200-225`) — an explicit, load-bearing sync boundary,
not a convenience/partial one.

**Consequence for Stage 2 design:** by the time `Lagrangian_to_Eulerian`
(and therefore `map_scalar`) runs, all its inputs are already
host-resident by design. Any CUDA `map_scalar` integration at this program
point is **not free** — it requires a genuinely new H2D immediately before
the ported kernel(s) and a new D2H immediately after, because (a)
`tracer_2d` already ran host-side before L2E in the same iteration (fixed
ordering, not to be changed — Stage 7), and (b) subsequent host code in
the same iteration (mfx/mfy/cx/cy accumulation, `fv_debug` prints,
`del2_cubed` at `last_step`) and the **next** `n_map` iteration's own H2D
still need host-resident results. This is a resolved architectural fact,
not a structural ambiguity — no Opus escalation is warranted (Case E does
not apply) — but it means Prompt 52's original zero-transfer assumption is
wrong and must be replaced with an explicit, minimized-subset transfer
design (§3).

## 3. Stage 1 continued — `iv` audit (`fv_mapz.F90`)

Exactly three `map_scalar` call sites are non-tracer and `iv∈{0,1}`
(52B's supported range):

| Field | Line | `iv` | `kord` param | Notes |
|---|---|---|---|---|
| `te` (total energy) | 522 | 1 | `kord(ikord_tm)` | gated by `remap_te`; conditional, not always active |
| `pt` (temperature) | 529 | 1 | `kord(ikord_tm)` | always active |
| `delz` | 604 | 1 | `kord(ikord_wz)` | active whenever `hydrostatic=.false.` — always true for the mandated production config |

Everything else is **ineligible** for 52B's current CUDA path: `w` (616)
uses `iv=-2`; all of `u`/`v`/`mfx`/`mfy`/`cx`/`cy` (704-767) use `iv=-1`;
`mapn_tracer`'s internal `scalar_profile` calls (1709) are tracer data,
out of scope regardless of `iv`. This is narrower than this prompt's
originating framing ("T + winds") — winds are not portable without
extending 52B's `iv` coverage, which is out of this prompt's scope.

**Recommended Stage 2 boundary:** `pt` and `delz` only. `te` excluded —
conditional (`remap_te`), and `te` is a Fortran-local diagnostic array,
not a registered `FieldId` in the persistent GPU state (§5), so porting it
would require a new field-registry entry for a variable that is not
always live; not worth the added scope for a conditional path.

## 4. New finding — `map_scalar`'s surrounding per-call setup is unported and unvalidated

52B's port scope (`fv3_map_scalar_column.hpp`) is exactly and only the
naked `map_scalar`/`scalar_profile`/`cs_limiters` primitives, taking
caller-supplied `pe1`/`dpe1`/`pe2`/`dpe2`. Reading the actual call sites
(`fv_mapz.F90:481-620`) shows both eligible fields depend on **per-call
Fortran-local temporaries that are not registered device state and have
no existing fixture coverage**:

- `peO` (target/Eulerian edge pressure, lines 482-490): `peO(:,1,:)=ptop`,
  `peO(:,km+1,:)=pe(:,km+1,:)`, `peO(i,2:km,j)=ak(2:km)+bk(2:km)*pe(i,km+1,j)`
  for the interior. Simple, but new — not in any 52A/52B fixture.
- `pt` path (lines 513-520): operates in **log-pressure space**, not raw
  pressure — `pn1=peln(:,1:km+1,j)` (registered `FID_PELN`), `pn2` derived
  from `log(peO)` with copied endpoints, `dpn1`/`dpn2` are differences of
  those logs. Also gated by `if (gmao_remap > 0)`: when true, the
  authoritative path calls `map1_gmao` instead of `map_scalar` entirely
  (lines 502-511) — a CUDA dispatch must check this and fall back to
  Fortran when `gmao_remap > 0`, not attempt to port `map1_gmao`.
- `delz` path (lines 576-613): operates in **raw pressure space** via
  `dpe`/`dpeO` (plain differences of `pe`/`peO`, lines 578-585), *and*
  applies a specific-volume transform immediately before the call
  (`delz(i,j,k) = -delz(i,j,k)/dpe(i,k,j)`, line 601) and inverts it
  immediately after (`delz(i,j,k) = -q2(i,k)*dpeO(i,k,j)`, line 611).

None of `peO`, `dpe`, `dpeO`, `pn1`, `pn2`, `dpn1`, `dpn2`, the
`gmao_remap` gate, or `delz`'s pre/post transform exist in
`fv3_map_scalar_column.hpp` or in any of the 8 Prompt 52A fixtures. This
is genuinely new, correctness-critical numerical-translation work — not
large or algorithmically hard (all closed-form arithmetic, no iterative
solves, directly transcribable from the lines above), but **untested**,
and a translation error here (e.g. an off-by-one in the `ak`/`bk`
indexing, or a sign error in the specific-volume transform) would not be
caught by 52B's existing regression — it would only surface as a silent
numerical divergence at Stage 11+ (first integrated GPU step), consuming
GPU job budget to discover.

This is not a Case E (ordering/data-residency) ambiguity — it is a
resolved, well-defined, boundable piece of additional work. But per this
project's bitwise-correctness bar and "for every nontrivial CUDA bring-up
fix, document... whether numerical semantics changed" doctrine, and given
this arithmetic currently has **zero** fixture coverage, writing the
production kernel/dispatch directly against it (skipping straight to
Stage 8) would be exactly the kind of ungated numerical change the
project's own methodology (52A-then-52B: baseline fixtures before any
CUDA code) was built to avoid.

## 5. Stage 3-5 precedent — resolved, favorable, reusable as-is

- **FMA (Stage 3):** `cuda/dyn_core/CMakeLists.txt:315` already compiles
  the entire CUDA dyn_core target with `-fmad=false` globally. A new
  `map_scalar` launcher added to the same target inherits this
  automatically — 52B's GPU-bitwise-with-`-fmad=false` result is already
  the production-consistent outcome. No new FMA decision needed.
- **Workspace (Stage 4):** existing precedent is
  `riem_solver_c_max_local_levels = 128`
  (`fv3_riem_solver_c.cuh:93`) — a bounded-local-storage-with-launcher-
  level-rejection convention. 52B's smoke kernel's `kMaxLevels=32`
  (`fv3_map_scalar_kernel.cu:38`) was explicitly flagged there as
  "illustrative only." A production launcher should use `128` to match
  established convention and remain viable for any future higher-`npz`
  config, not `32`.
- **Launcher/dispatch convention (Stage 5):** confirmed precedent —
  `launch_riem_solver_c(Args, stream, error)` (low-level, validates,
  launches, checks `cudaGetLastError`) plus
  `launch_riem_solver_c_ctx(const ExecutionContext&, error)` (high-level,
  looks up device pointers via `x.fields->find(FieldId::...)`). A new
  `map_scalar` launcher should follow this exact two-layer pattern.
  `corner_edge_service_cuda` (`fv3_gpu_production_dispatch.cu`) is the
  precedent for documenting a new adapter's reuse rationale and "Numerical
  semantics changed: NO" statement.
- **H2D/D2H mechanism (Stage 6):** `fv3_gpu_field_mask` is a genuine
  bitmask (`mask%words`, per-field-id `ibset`,
  `fv3_gpu_dyncore_bridge.F90:697-706`), not a fixed contiguous range —
  `fv3_gpu_sync_from_host`/`fv3_gpu_sync_to_host` already accept an
  arbitrary subset mask. A new, narrow mask covering only
  `FID_PT`/`FID_DELZ`/`FID_PE`/`FID_PELN`/`FID_AK`/`FID_BK` (skipping the
  other ~20 mask fields) is directly constructible with the existing
  mechanism — no new sync infrastructure needed, only a new
  `map_scalar_state_mask(mask)` subroutine mirroring the existing
  `mutable_state_mask`.

## 6. Recommendation

Architecture is fully resolved and the integration is buildable with
existing project mechanisms end to end (mask-subset sync, ExecutionContext/
FieldRegistry launcher convention, existing FMA policy, existing workspace
convention). But Stage 8 (implementation) should not proceed directly
against `pt`/`delz` production dispatch until the §4 gap is closed the
same way 52A closed it for the primitives: a small, dedicated fixture-level
host-only validation of the `peO`/`pn1`/`pn2`/`dpn1`/`dpn2`/`dpe`/`dpeO`
derivation and the `delz` pre/post transform, against authoritative
Fortran, before any of it is wired into a CUDA kernel or a production call
site. This is bounded (closed-form arithmetic, no new algorithm), but it
is new, untested, and correctness-critical, and doing it first is
directly consistent with this project's own 52A→52B methodology and its
"do not modify scientific algorithms... without validation" doctrine.

No source files were changed in this prompt. No build or GPU job was run.
`docs/performance/cuda_fv3_optimization_roadmap_2026-09-21.md` and
`docs/agent_handoff/CURRENT_STATE.md` are intentionally not updated here —
no validated integration/performance result exists yet to record.

## Final status

**FV_MAPZ CUDA INTEGRATION BLOCKED — architecture and integration design
fully resolved (Stages 0-6); implementation (Stage 8) requires a
fixture-validated port of `map_scalar`'s surrounding pressure-derivation
and `delz` transform arithmetic first (§4), which is new scope beyond
52B's standalone primitives and was not attempted in this prompt.**

## Update (Prompt 52C-PREP, 2026-09-21)

The §4 blocker above is now closed at the fixture level. A host-only,
CPU-only baseline harness (`model/pt_delz-driver/`) reproduces the exact
production pt/delz surrounding arithmetic (`peO`/`pn1`/`pn2`/`dpn1`/
`dpn2`/`dpe`/`dpeO` derivation, pt/delz pre/post-transforms, joint `pkz`)
directly against the unmodified authoritative `map_scalar`, with 6
deterministic fixtures, all self-checks passing (finite, valid pressure
ordering, positive thickness, bitwise repeatability). Full detail:
`docs/performance/p52c_prep_pt_delz_fixture_2026-09-21.md`. This clears
the way to resume Stage 8 (CUDA implementation) of this integration —
no further fixture-level prep is outstanding for pt/delz.

## Update (Prompt 52C-IMPLEMENT, Stages 6/8/9, 2026-09-21)

Kernel/launcher (`cuda/dyn_core/vertical/fv3_pt_delz_remap.{cuh,cu}`,
`launch_pt_delz_remap`/`launch_pt_delz_remap_ctx`) were already written and
COMPILED before this update. This update closes Stage 6 (the one
deferred loose end) and executes Stage 8/9 (GPU smoke test against the
Prompt 52C-prep fixtures) for the first time.

**Stage 6 — resolved, no ordering ambiguity.** Direct citation against
`GFDL_atmos_cubed_sphere` (tag `geos/v3.0.0`, `ca30e26`) `model/fv_mapz.F90`:

- `remap_option=0` sets `remap_t=.true.` exclusively (`remap_pt=.false.`,
  `remap_te=.false.`, lines 244-257) — the production config activates
  exactly one control path, not a superposition.
- The pt pre-transform the kernel implements
  (`fv3_pt_delz_remap.cu:96-101`) is the `if (remap_t)`/non-hydrostatic/
  non-`MOIST_CAPPA` branch, `fv_mapz.F90:357,371-392`, formula at line 388
  — bit-for-bit the same formula, confirmed by direct read.
- `do 1000`'s `pt` branch (`fv_mapz.F90:512-536`, taken because
  `gmao_remap=0` and `remap_te=.false.`) matches the kernel's `map_scalar`
  call on `pt_pre` with `q_min=t_min`.
- `do 1002`'s delz pre/remap/post-transform (`fv_mapz.F90:594-611`,
  specifically `delz(i,j,k)=-q2(i,k)*dpeO(i,k,j)` at line 611) matches the
  kernel's `delz_out = -q2_delz[k]*new_delp` exactly, where
  `new_delp = dpeO[k]` in both.
- The closing pkz/delp/pe/peln/pk block (`do 2000`, `fv_mapz.F90:780-943`)
  is **field-independent and idempotent** with respect to the GPU path:
  its `pe2`/`pn2`/`pk2`/`delp` recomputation (lines 793-825) is a pure
  function of `ak`/`bk`/old-`pe(km+1,j)` only, and its `remap_t` pkz branch
  (non-hydrostatic, non-`MOIST_CAPPA`, lines 917-930:
  `pkz(i,j,k)=exp(akap*log(rrg*delp(i,j,k)/delz(i,j,k)*pt(i,j,k)))`) uses
  exactly the already-updated `delp`/`delz`/`pt` — bitwise identical to the
  kernel's own closing formula, given the same inputs. **Conclusion: a
  future Stage-7 dispatch only needs to gate/skip `do 1000`'s pt branch and
  `do 1002`'s delz sub-block (not the w-remap statements that follow it in
  the same loop); `do 2000`'s closing block can be left untouched — it
  redundantly but harmlessly reproduces the GPU's own values bit-for-bit
  once GPU output is D2H-synced back into the host `pt`/`delz` arrays
  first.** None of the four Opus-escalation conditions in this prompt's
  mandate fire.

**Sync-boundary finding (relevant to Stage 7, not yet implemented).**
`fv3_gpu_interface_step_dyn` (`cuda/interface/fv3_gpu_dyncore_bridge.F90`)
performs its own internal `fv3_gpu_sync_from_host` (H2D, line 337) and
`fv3_gpu_sync_to_host` (D2H, line 361) before returning to
`fv_dynamics.F90`. By the time `Lagrangian_to_Eulerian` runs (line 952),
`pe`/`peln`/`pt`/`delz`/`delp`/`pk`/`pkz` are host-authoritative — there is
no live persistent-device-residency state to reconcile with a new pt/delz
GPU call at this call site. A future Stage-7 dispatch is architecturally a
clean, standalone H2D→kernel→D2H round trip at this boundary (consistent
with "remapping" being one of `project_architecture.md`'s named
synchronization boundaries), not a host/device ownership conflict.

**Stage 8/9 — GPU EXECUTED, NUMERICALLY VALIDATED.** New smoke-test
harness `cuda/dyn_core/vertical/fv3_pt_delz_remap_smoke.cu` (single
column, `is=ie=js=je=0`, `km=10`), built as CMake target
`fv3_wp_c_pt_delz_remap_gpu_smoke` (linked against `fv3_dyn_core_gpu` +
`fv3_wp_c_vertical`). First run (job 58499219) failed all 6 fixture cases
with a uniform `pt`/`pkz` relative error of `1.715e-05` (while
`delz`/`delp` matched to round-off) — root-caused to a wrong guessed
`grav` constant in the smoke test's own host-side setup (not a kernel bug:
`delz`/`delp`'s formulas never touch `rrg`/`akap`/`rdgas`/`grav`, so their
round-off match while `pt`/`pkz` alone diverged is the exact signature of
a wrong multiplicative constant reachable only through `rrg`). Fixed by
parsing `rrg`/`akap` directly from each fixture's own header line (the
exact `constants_mod` values the Fortran `pt_delz-driver` used) instead of
guessing a `(rdgas, cp_air, grav)` triple, and deriving `rdgas=-rrg`,
`grav=1` (the kernel only ever uses the ratio `rrg=-rdgas/grav`). Rerun on
one A100 (`srun --partition=gpu_a100 --constraint=rome --gres=gpu:1`, job
**58499304**, exit code 0):

```
[case1_smooth_monotonic]   pt rel=2.703e-16  delz rel=1.364e-15  pkz rel=1.559e-16  delp/dpeO rel=1.111e-15  OK=1
[case2_strong_gradient]    pt rel=3.793e-16  delz rel=1.010e-15  pkz rel=1.378e-16  delp/dpeO rel=1.111e-15  OK=1
[case3_local_extrema]      pt rel=2.124e-16  delz rel=7.855e-16  pkz rel=9.134e-16  delp/dpeO rel=1.111e-15  OK=1
[case4_near_uniform]       pt rel=4.694e-16  delz rel=1.205e-15  pkz rel=8.968e-16  delp/dpeO rel=1.111e-15  OK=1
[case5_limiter_activating] pt rel=2.192e-16  delz rel=1.198e-15  pkz rel=8.126e-16  delp/dpeO rel=1.111e-15  OK=1
[case6_realistic_c512]     pt rel=3.734e-16  delz rel=1.098e-15  pkz rel=9.226e-16  delp/dpeO rel=1.111e-15  OK=1
PT_DELZ_GPU_SMOKE: 0/6 cases failed
```

All 6 cases PASS at round-off (worst relative error `1.364e-15`, far
inside the `1e-9` tolerance). This is the first real-GPU execution of
`fv3_pt_delz_remap.cu`'s production kernel, validated against the Prompt
52C-prep fixtures. **This is standalone smoke-test validation only — the
kernel is not yet wired into `Lagrangian_to_Eulerian`/production dispatch
(Stage 7, not started).**

## Status after this update

IMPLEMENTED: yes (kernel + launcher + ExecutionContext entry).
COMPILED: yes. GPU EXECUTED: yes (job 58499304). NUMERICALLY VALIDATED:
yes, standalone smoke test only, round-off vs. the Prompt 52C-prep
fixtures. NOT YET DONE: Stage 7 (Fortran dispatch wiring into
`Lagrangian_to_Eulerian` + a new ISO_C_BINDING bridge entry + new C ABI
function — the architecture for this is now fully resolved per above, but
no code for it has been written), and therefore Stages 10-17 (regression,
small-res, C512 correctness, paired performance A/B, docs/roadmap
closure) have not been attempted. No source files other than the new
smoke test and its `CMakeLists.txt` entry were touched. No commit was
made.

## Update (Prompt 52C-IMPLEMENT, Stage 7, 2026-09-22)

**In-repo half of Stage 7 (C ABI + Fortran bridge wiring, this repo):
IMPLEMENTED, COMPILED. GPU EXECUTED / NUMERICALLY VALIDATED: not yet —
no live call site exists (see below).**

`fv3_gpu_context` is opaque outside `fv3_gpu_runtime.cpp` (only
forward-declared in `fv3_gpu_api.h`), so a new ExecutionContext-building
C ABI wrapper cannot live in `fv3_gpu_production_dispatch.cu` alongside
`fv3_build_production_dispatch` (that TU cannot see `c->data`'s members).
Resolved by mirroring the existing `corner_edge_service`
function-pointer-indirection pattern exactly:

- `NumericalDispatch::pt_delz_remap` (new additive-tail `LaunchFn` slot,
  `fv3_gpu_internal.hpp`), populated by `fv3_build_production_dispatch`
  (`fv3_gpu_production_dispatch.cu`, the one TU allowed to reference
  `fv3::gpu::vertical::launch_pt_delz_remap_ctx` by name).
- New public C ABI function `fv3_gpu_pt_delz_remap` (`fv3_gpu_runtime.cpp`,
  declared in `fv3_gpu_api.h`) — builds the `ExecutionContext` using the
  same field-by-field pattern as `fv3_gpu_dyn_core_step`, orders the
  compute stream after the caller's H2D via the existing `order_after`
  helper, and invokes the launcher only through
  `dispatch.pt_delz_remap(...)` — never a direct symbol reference — which
  is what breaks the `fv3_dyn_core_gpu` / `fv3_wp_c_vertical` link cycle.
- Fortran bridge: `fv3_gpu_abi_mod.F90` gets the `bind(C)` interface;
  `fv3_gpu_dyncore_bridge.F90` gets `pt_delz_remap_state_mask` (narrow
  9-field mask: PE/PELN/PT/DELZ/DELP/PK/PKZ/AK/BK) and public wrapper
  `fv3_gpu_interface_pt_delz_remap` (H2D -> kernel -> D2H -> one host wait,
  same shape as `fv3_gpu_interface_step_dyn`).

Build-verified, Intel-MPI production stack (`gcc/12.3.0`,
`intel/2024.2.0`, `impi/2021.13`, `nvhpc-byo-compiler/24.7`,
`I_MPI_CXX=g++`):
- `build_cuda_r8_intelmpi`, target `fv3_dyn_core_gpu`: SUCCESS.
- `build_cuda_r8_intelmpi`, target `fv3_production_dispatch`: SUCCESS
  (includes `fv3_pt_delz_remap.cu.o` recompiling clean against the new
  include).
- Cross-repo Fortran verification: `fv_mapz.F90`/`Lagrangian_to_Eulerian`
  does not exist in this repo -- its only real, buildable copy is
  `GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/@fvdycore/model/fv_mapz.F90`
  (a separate GEOS-ESM/GEOSfvdycore checkout; this repo's own
  `cuda/interface/CMakeLists.txt` is not reached by this repo's build --
  the other repo's `CMakeLists.txt` points `FV3_CUDA_SOURCE_DIR` back at
  this repo's `cuda/` and compiles the bridge `.F90` files directly from
  here). Rebuilt target `fvdycore` in `GEOSfvdycore_fortran_perf/build`:
  SUCCESS -- `fv3_gpu_abi_mod.F90.o` and `fv3_gpu_dyncore_bridge.F90.o`
  (both from this repo's absolute paths) compiled with zero errors;
  `libfvdycore.a` linked clean.

**Call-site edit in `fv_mapz.F90` itself: NOT MADE, and not authorized.**
Per project/user decision, the actual gated skip in `Lagrangian_to_Eulerian`
(`do 1000`'s pt branch, `do 1002`'s delz sub-block) was produced as a
reviewable unified diff only, not applied to `GEOSfvdycore_fortran_perf`
or any file in this repo -- see the Stage 7 report to the coordinator,
2026-09-22, for the literal diff text. Without that edit applied and
rebuilt, `fv3_gpu_pt_delz_remap`/`fv3_gpu_interface_pt_delz_remap` are
unreachable from any real run, so Stages 10-17 (fixture rerun, small-res
correctness, C512 correctness, paired performance A/B, Nsight,
KEEP/REVERT decision) remain BLOCKED pending diff review/application --
not attempted in this update.

## Status after this update (supersedes the block above)

IMPLEMENTED: yes, both halves (kernel/launcher from the prior update;
C ABI + Fortran bridge dispatch plumbing from this update). COMPILED:
yes, both halves, including the separate-repo Fortran bridge rebuild.
GPU EXECUTED: yes for the standalone kernel (job 58499304, prior update);
NOT YET for the new `fv3_gpu_pt_delz_remap` C ABI entry point itself (no
call site exists yet to exercise it). NUMERICALLY VALIDATED: yes for the
standalone kernel only; the integrated dispatch path has no numerical
result yet because it has no caller. NOT YET DONE: the `fv_mapz.F90`
call-site edit itself (diff produced, not applied -- awaiting review), and
therefore Stages 10-17. No commit was made.

## Update (Prompt 52C-IMPLEMENT, Stages 11 & 13, 2026-09-22)

The `fv_mapz.F90` call-site diff described in the Stage 7 update above was
applied to `GEOSfvdycore_fortran_perf/src/Components/
@FVdycoreCubed_GridComp/@fvdycore/model/fv_mapz.F90` (file mtime 08:05)
and the `fvdycore` library rebuilt, making `use_cuda_pt_delz_remap`
(gated on `remap_option==0 .and. gmao_remap==0`, engaged via
`input.nml +kord_tm=9,+kord_wz=9`) reachable for the first time from a
real run.

**Stage 11 — small-res correctness, c32/L10/R8, HS, same-node
Fortran-vs-CUDA (`p52c_stage11_same_node_fc`).** Three submissions
(58506027, 58506114, 58506170); the final one completed in 21s, both
backends RC=0.

**Stage 13 — C512/L10/R8 production correctness gate, same-node
Fortran-vs-CUDA (`p52c_stage13_same_node_fc`, job 58506276, node
`warpa005`, 3h52m).** Both backends RC=0, 0 range_warn/NaN/error, 6/6
`.nc4` checkpoints each. `tools/analysis/hs_netcdf_inspect.py
compare-backends`, fields T/U/V (`inst3_3d_diag`) and PS
(`inst1_2d_diag`), all 3 checkpoints (0200z/0400z/0600z):

```
T @ ~500hPa  time_idx=0,1,2: max_abs=0.0  max_rel=0.0  n_nan_either=0/131584
U @ ~500hPa  time_idx=0,1,2: max_abs=0.0  max_rel=0.0  n_nan_either=0/131584
V @ ~500hPa  time_idx=0,1,2: max_abs=0.0  max_rel=0.0  n_nan_either=0/131584
PS           time_idx=0,1,2: max_abs=0.0  max_rel=0.0  n_nan_either=0/131584
```

Bitwise-identical at every field/checkpoint. Diagnostic `.nc4` output is
float32 (confirmed via `netCDF4` dtype inspection), consistent with the
same bitwise-at-float32-output acceptance criterion already used for the
prior C512 gates in CURRENT_STATE.md Sec 4 — not a weaker standard than
precedent.

**Paired same-node performance (`fv_timing_mod` `DYN_CORE`, same job,
MPS ON + halo/corner fusion ON, Graph OFF):** Fortran mean 6387.92 ms/call,
CUDA mean 6429.66 ms/call, ratio 1.0065 — comparable, well inside the
documented >2x node-to-node noise floor (CURRENT_STATE.md Sec 5/8). New
dispatch shows no performance regression at C512.

## Status after this update (supersedes all blocks above)

IMPLEMENTED: yes (kernel/launcher, C ABI, Fortran bridge, and now the
live `fv_mapz.F90` call-site gate). COMPILED: yes, both repos. GPU
EXECUTED: yes, including the first real integrated-path execution (Stage
11 c32, Stage 13 C512). NUMERICALLY VALIDATED: yes — Stage 13 C512
production correctness gate PASSED bitwise vs. Fortran (T/U/V/PS, 3
checkpoints). PERFORMANCE VALIDATED: yes, comparable, no regression.
NOT YET DONE: R4 validation, `mapn_tracer` coverage, Nsight capture of
the new dispatch path (Stage 15/16 of the original plan), formal
KEEP/REVERT decision + roadmap doc closure (Stage 17), and committing the
applied `fv_mapz.F90` diff (currently live only in the separate
`GEOSfvdycore_fortran_perf` checkout, not in this repo, and not yet
committed anywhere).
