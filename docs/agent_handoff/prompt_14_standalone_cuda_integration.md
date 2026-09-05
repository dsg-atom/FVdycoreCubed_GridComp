# Prompt 14 — Standalone CUDA Production Integration: BLOCKED, ESCALATED

## 0. Scope and outcome

Task: design/implement the minimum production-facing integration so the real
`StandAlone_FV3_Dycore.x` Fortran executable (6 MPI ranks, one per
cubed-sphere tile — `FV_NX=1`/`FV_NY=6`) can execute the Gate-C2-validated
CUDA `dyn_core` backend, opt-in, alongside the unmodified Fortran backend.

**Outcome: BLOCKED at the architecture-verification stage (§1/§2 of the task),
before any source or build change was made.** A real, source-confirmed
contradiction was found between the real production MPI decomposition (six
separate OS processes, one per tile) and the only implemented CUDA halo
exchange transport (a single-process, six-host-thread, mutex/condition-variable
rendezvous that requires every tile's device buffers to be directly
dereferenceable pointers in one address space). This is not a mapping,
build, or interface defect fixable within the task's 3-substantial-fix
integration budget — it requires implementing a new cross-process transport,
which the task's own hard-stop conditions (3) and (6) identify as
out-of-scope for this pass. **No source files were modified. No build was
attempted.** Per the task's own escalation instruction, this is reported for
Opus-level architectural decision-making rather than attempted on Sonnet.

## 1. Working trees confirmed (§1 — COMPLETE)

Two distinct repositories, not one:

- **Repo A — CUDA implementation.** `FVdycoreCubed_GridComp`
  (`/discover/nobackup/projects/imvi-nse/jli30/FVdycoreCubed_GridComp`, also
  mounted at `/gpfsm/dnb06/projects/p295/jli30/FVdycoreCubed_GridComp`).
  Branch `cuda/gpu-bringup`, HEAD `eb6d689` (3 commits ahead of
  `origin/cuda/gpu-bringup` at `191157f`; not pushed). Working tree clean
  except this session's own `CURRENT_STATE.md` edit and the long-standing
  untracked scratch/tooling files already recorded there. Contains the
  Gate-C2-validated `cuda/dyn_core/` implementation and its public C ABI
  (`cuda/dyn_core/include/fv3_gpu_api.h`).
- **Repo B — standalone Fortran fixture / authoritative integration tree.**
  `GEOSfvdycore_fortran_perf`
  (`/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf`).
  Branch `main`, HEAD `8ba7bdd` (tag `v3.2.0`, matches `origin/main` exactly).
  Working tree clean except four untracked log/experiment artifacts
  (`build.log`, `install.log`, `rebuild_6c.log`, `experiments/`) — no stash
  needed. Nested checkouts:
  `src/Components/@FVdycoreCubed_GridComp/StandAlone_FV3_Dycore.F90` (the
  standalone driver) and
  `src/Components/@FVdycoreCubed_GridComp/@fvdycore/model/{fv_dynamics.F90,dyn_core.F90}`
  (the authoritative caller and callee).

Had this proceeded: CUDA runtime/ABI changes would go in Repo A
(`cuda/dyn_core/`); the Fortran `ISO_C_BINDING` wrapper and backend-switch
plumbing would go in Repo B (inside `@fvdycore/model/` next to
`dyn_core.F90`, or a new sibling module); build-system changes would span
both (Repo A exporting a linkable CUDA library; Repo B's CMake linking
against it).

## 2. Production call boundary (§2 — COMPLETE)

Real, single call site, `fv_dynamics.F90:645-655`:

```fortran
call dyn_core(npx, npy, npz, ng, sphum, nq, mdt, k_split, n_split, zvir, cp_air, akap, cappa, grav, hydrostatic, &
              u, v, w, delz, pt, q, delp, pe, pk, phis, varflt, ws, omga, ptop, pfull, ua, va, &
              dudt_rf, dvdt_rf, dwdt_rf, uc, vc, &
              mfxL, mfyL, cxL, cyL, &                 ! (or the R8 aliases under SINGLE_FV)
              pkz, peln, q_con, ak, bk, dpx, ks, &
              gridstruct, flagstruct, neststruct, idiag, bd, &
              domain, n_map==1, i_pack, last_step, diss_est, time_total)
```

inside the `do n_map=1, k_split` loop (`fv_dynamics.F90:616`), preceded by
Fortran-side halo starts for `delp`/`pt`/`(u,v)` and a `dp1` snapshot, and
followed (outside the shown excerpt) by mass-flux/Courant aliasing and
diagnostics prints. `dyn_core`'s own signature is
`dyn_core.F90:166-170`, declarations `:172-263`. Condensed interface table
(only fields dyn_core's argument list actually carries; static
config/geometry it reads through `gridstruct`/`flagstruct`/`bd` is listed
separately below):

| Fortran arg | type | shape | intent | CUDA equivalent | host/device sync |
|---|---|---|---|---|---|
| `npx,npy,npz,ng,sphum,nq,k_split,n_split,ks` | integer scalar | — | in | `NumericalConfig`/`fv3_gpu_create_desc` scalars | static, init only |
| `mdt`(→`bdt`) | real scalar | — | in | `fv3_gpu_step_desc.bdt` | per k_split iter (dynamic, but Gate C2's fixed-`bdt` constraint applies to `--cuda-graph`) |
| `zvir,cp_air,akap,grav` | real scalar | — | in | `NumericalConfig::{grav,rdgas,akap_exner,cv_air}` (derived) | static |
| `cappa` | real 3D `(isd:ied,jsd:jed,npz)` | inout | dynamic | `views.cappa`-equivalent (not yet a registered field; see §4 gap) | per-call H2D if Fortran can mutate it between calls |
| `hydrostatic` | logical | — | in | `NumericalConfig::hydrostatic` | static |
| `u,v,w,delz,pt,delp` | real 3D, staggered/A-grid, `isd:ied(+1)` × `jsd:jed(+1)` × `npz` | inout | dynamic prognostic | registered `fv3_gpu_field_desc` fields (already exercised end-to-end by Gate C2) | H2D before, D2H after (persistent residency across repeated calls is the whole point — see §8 intent) |
| `q` | real 4D `(...,npz,nq)` | inout | dynamic tracer | **not implemented in CUDA dyn_core at all** — Gate C2 scope has no tracer transport in `dyn_core` itself (tracers are `fv_tracer2d`, outside `dyn_core`) | N/A for this integration — `q` passes through `dyn_core` untouched in Fortran too (only touched if `inline_q`, `fv_dynamics.F90:436`, a separate call) |
| `pe,pk,peln,pkz` | real 3D, various boxes incl. `is-1:ie+1`/`js-1:je+1` | inout | dynamic pressure state | registered fields; `pe`'s halo ring is the known Gate-C2 coverage gap (§7/§10 of `CURRENT_STATE.md`) | D2H after |
| `phis,varflt` | real 2D | inout(phis)/inout(varflt) | **static across the whole run** (topography) | STATIC/INIT ONCE | H2D once at context creation |
| `ws,omga,ua,va,uc,vc` | real 2/3D | out/inout | dynamic | registered fields (`omga` is the Gate-C2-validated finalization path) | D2H after |
| `dudt_rf,dvdt_rf,dwdt_rf` | real 3D | inout | dynamic (Rayleigh friction tendencies) | **no CUDA equivalent found** — needs a targeted check whether Gate-C2's dyn_core computes/consumes these, or whether they are a hydrostatic/Rayleigh-friction path outside Gate C2's covered production first-run config | flagged, not resolved (blocked before reaching this) |
| `mfx,mfy,cx,cy` (`mfxL/mfyL/cxL/cyL` at the call site) | real(8) 3D flux-capacitor arrays | inout, accumulate over `k_split` | dynamic, persistent across `n_split` substeps within one `dyn_core` call, RESET by Fortran outside it | registered fields | per-`dyn_core`-call in/out |
| `q_con` | real 3D | inout | dynamic (`MOIST_CAPPA`/`USE_COND` path) | **known Gate-C2 gap** — `q_con ≡ 0` was proven for the validated fixture only; do not assume general validity | flagged |
| `ak,bk` | real 1D `(npz+1)` | in | static | `NumericalConfig::dp_ref` is derived from these once | STATIC/INIT ONCE |
| `dpx` | real(8) 3D | inout | dynamic diagnostic (`DryMassRoundoffControl`) | no CUDA equivalent found; likely diagnostic-only, needs targeted check | flagged |
| `gridstruct,flagstruct,neststruct,idiag,bd,domain` | opaque Fortran derived types | in/inout | STATIC geometry/config + the MPI `domain2d` | `NumericalConfig` fields (§10 of `CURRENT_STATE.md`) derived from `flagstruct`; `gridstruct` carries `grid_lonlat`-equivalent geometry; `domain` is the FMS `mpp_domains` handle — **this is exactly what the halo-transport blocker below is about** | STATIC/INIT ONCE for geometry/config; `domain` is the real MPI decomposition object |
| `i_pack` | `group_halo_update_type(*)` | inout | the **live FMS halo-exchange handles** for the fields dyn_core itself starts/completes mid-call (`u,v` DGRID_NE at `:422`, `w`/`gz` NH updates, etc.) | CUDA's own GPU-side halo exchange (`HaloPeerGroup`) is a **separate, parallel mechanism** that does not touch `i_pack` at all | **this is the blocker's second half** — see §3 below |
| `init_step,end_step` (`n_map==1`,`last_step`) | logical | in | per-call flags | `fv3_gpu_step_desc.{init_step,end_step}` | matches existing ABI exactly |
| `diss_est` | real 3D | inout | dynamic (SKEB) | registered field (`heat_source`/`diss_est` accumulator, Gate-C2-validated) | per-call |

## 3. THE BLOCKER — real cross-process MPI halo transport does not exist (source-verified)

This is the finding that stopped the task, discovered while verifying §12's
explicit warning ("do not assume rank==tile without verifying the standalone
decomposition... do not add communication stubs") before writing any ABI or
wrapper code.

**Fact 1 — the real standalone configuration is genuinely multi-process, not
multi-threaded.** Per the task's own §12 and `CURRENT_STATE.md` throughout:
`FV_NX=1`, `FV_NY=6`, **six separate MPI ranks**, one tile per rank. Each
rank is a distinct OS process with its own address space; `domain` in the
`dyn_core` argument list above is the real FMS `mpp_domains` decomposition
spanning those six processes, and the real halo exchange for `u,v,w,gz,...`
at the Fortran level goes through `i_pack`/`start_group_halo_update`, which
is genuine inter-process MPI.

**Fact 2 — the only implemented CUDA halo transport is single-process,
intra-thread, and cannot cross a process boundary.** Read directly from
`cuda/dyn_core/include/fv3_gpu_halo_peer.hpp:1-160`
(`fv3::gpu::halo::HaloPeerGroup`), whose own header comment names this
"Model B — one process represents all six cubed-sphere tiles, and neighbours
exchange by device-to-device copy" (confirmed again verbatim at
`docs/agent_handoff/milestone_3a10_gpu_halo_exchange.md:434`). Mechanics,
read directly from the source, not inferred:

- Each tile's halo readiness is tracked in a `PeerMember` holding raw
  `fv_real*` `send`/`recv` device pointers and raw `cudaEvent_t`/
  `cudaStream_t` handles (`:74-81`) — these are only meaningful within the
  process that allocated them; a pointer or CUDA handle from process A is
  not valid in process B without CUDA IPC, which nothing here uses.
- The two rendezvous barriers (`HaloPeerGroup::barrier()`, `:134-147`) are
  implemented with a plain `std::mutex`/`std::condition_variable`
  (`:66-67`) — **process-local kernel primitives**. Six separate OS
  processes cannot synchronize on the same `std::condition_variable`
  instance at all; it does not exist in more than one address space.
- `HaloPeerGroup::attach()` (`:94-102`) requires the caller to hand it a
  `PeerMember` by value, meaning the group and every member must be
  constructed inside one process's memory.
- The device-to-device transfer itself (`send_table()`, `:121-126`,
  consumed by `launch_halo_transfer`) copies directly between two tiles'
  device buffers — this requires either both tiles' contexts to share one
  GPU and one CUDA (primary) context, exactly as the current 6-thread
  benchmark and `test_dispatch_smoke` do, or CUDA peer-to-peer/IPC
  plumbing that is not present anywhere in this codebase.

**Fact 3 — there is no real MPI call anywhere in the CUDA halo path.**
`grep -rn "MPI_" cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` finds exactly two
hits, both inside comments describing conceptual analogies to
`MPI_Waitall`/`MPI_Finalize` (`:829`, `:1259`) — no actual MPI API call.
`fv3_gpu_create_desc::communicator_token` (`fv3_gpu_api.h:109`) — the one
public-ABI field that looks purpose-built for a future real communicator
handle — is set nowhere and read nowhere (`grep -rn communicator_token
cuda/dyn_core --include=*.cpp --include=*.hpp --include=*.cu` outside the
header itself returns zero hits). `HaloPeerGroup` is the sole halo-readiness
path (`attach_halo_peer_group`, `fv3_gpu_internal.hpp:444`); there is no
alternate MPI-based branch to fall back to.

**Fact 4 — this is a known, previously self-documented limitation, not a new
defect.** Milestone 3A.10's own handoff states this explicitly at design
time: *"Model B does **not** establish: MPI rank mapping, MPI request
lifecycle..."* (`milestone_3a10_gpu_halo_exchange.md:471`) and *"HaloPeerGroup
is the only piece an MPI transport replaces. The pack, transform, [...]"*
(`:955`) — i.e. the 3A.10 design always intended Model B to be a
placeholder for a real MPI transport, and explicitly deferred building that
transport. `project_architecture.md`'s own "Communication Evolution"
section anticipates this staging (*"Stage 1: Host-staged MPI is
acceptable"*) but Stage 1 itself has never been implemented for the CUDA
side — only Model B (in-process) exists.

**Conclusion.** Running the validated CUDA `dyn_core` under the real
production decomposition (six MPI processes) requires a **new cross-process
halo transport** — at minimum, real MPI `Isend`/`Irecv`/`Waitall` (or
equivalent) moving the same pack/unpack buffers `HaloPeerGroup` already
stages, but crossing actual process boundaries instead of thread boundaries.
This is squarely the task's own **hard-stop condition (6)** ("production
integration requires redesigning the general halo runtime") and **hard-stop
condition (3)** ("MPI/tile decomposition contradicts the current CUDA
runtime architecture"). It is not fixable within the 3-substantial-fix
integration budget (§19's allowed fix categories — ABI signatures, shape/
stride mapping, context lifecycle, build/link, backend selection, config/
geometry plumbing, host/device sync, MPI/tile *association* — presuppose a
transport that can carry data between the real ranks; none of them can
manufacture one). Per the task's own instruction, this is reported for
escalation rather than attempted.

## 4. Secondary, non-blocking finding: the public C ABI already anticipates
   a Fortran caller, but has no `NumericalConfig` surface yet

Not the reason for the stop, but material to record so the eventual
redesign does not have to rediscover it. `cuda/dyn_core/include/fv3_gpu_api.h`
already exposes a clean, ABI-safe, Fortran-callable lifecycle —
`fv3_gpu_create` / `fv3_gpu_initialize_state` / `fv3_gpu_set_cubed_sphere_topology`
/ `fv3_gpu_dyn_core_step` / `fv3_gpu_sync_to_host` / `fv3_gpu_finalize` — with
no C++ types leaked (opaque `fv3_gpu_context*`, plain structs, explicit
status codes). `fv3_gpu_set_cubed_sphere_topology`'s own doc comment
(`fv3_gpu_api.h:156-179`) was written explicitly because "the Fortran bridge
never mentioned tile, npx, npy" — i.e. this entry point was already built
*for* this exact integration. This confirms §4/§9's "reuse the exact Gate-C2
implementation" instruction is achievable in principle for lifecycle/topology.

However, the per-run numerical configuration (`NumericalConfig` — the ~30
knobs enumerated in the task's §10: `n_sponge`, `nf_ke`, `n_con`, `akap_exner`,
`edge_*`, `grid_lonlat`/`agrid_lonlat`, `en1/en2/ec1/ec2`, etc.) is currently
only installed by a **benchmark-local, non-ABI helper**,
`install_config()` (`cuda/dyn_core/benchmarks/run_small_perf_bench.cpp:2234`),
which calls internal C++ setters declared in
`cuda/dyn_core/include/fv3_gpu_internal.hpp` — not the public C ABI. The same
is true of `set_topology()` (full `TopologyView`, not the ABI's
tile/npx/npy/rank_of_tile form) and `attach_halo_peer_group()`. A real
production ABI needs a plain-C `fv3_gpu_numerical_config` struct and an
`fv3_gpu_set_numerical_config()` entry point mirroring `install_config`'s
logic — straightforward, and explicitly NOT started in this pass, since it
is downstream of the halo-transport blocker (there is no point wiring
configuration plumbing for a call path that cannot execute under six real
MPI ranks yet).

## 5. What was and was not done

- Read only: this repo's `CLAUDE.md`, `CURRENT_STATE.md`,
  `prompt_13_architectural_optimization.md` (already in context from the
  prior task), and targeted greps/reads of `fv_dynamics.F90`, `dyn_core.F90`,
  `fv3_gpu_api.h`, `fv3_gpu_halo_peer.hpp`, `fv3_gpu_internal.hpp`,
  `fv3_gpu_runtime.cpp` (MPI-call grep only), `run_small_perf_bench.cpp`
  (targeted `install_config`/`set_topology` call sites only), and
  `milestone_3a10_gpu_halo_exchange.md` (three targeted greps, not a full
  read). Did **not** broadly scan physics, diagnostics, or I/O, per §2's
  instruction.
- **No source file was modified in either repository.** No CMake/build
  change. No Slurm job submitted. Zero of the 3 permitted integration fixes
  were used, because none of the allowed fix categories address a transport
  redesign.
- Fortran backend: unmodified, therefore unaffected (still builds/runs
  exactly as before — not independently re-verified this pass since nothing
  touched it).
- The interface table in §2 is the concrete, source-read deliverable of
  this pass and should be reused directly once the transport question is
  resolved — it does not need to be redone.

## 6. Recommended next step (for whoever picks this up, Opus or otherwise)

Decide, at the architecture level, how the CUDA `dyn_core` will exchange
halos across real MPI ranks before any further Fortran/ABI/build work is
useful. Two directions worth weighing (not evaluated in depth here — that
evaluation is itself the escalated decision):

1. **Host-staged real MPI (matches `project_architecture.md`'s own "Stage 1:
   host-staged MPI is acceptable").** D2H the pack buffers, exchange via
   ordinary `MPI_Isend`/`Irecv`/`Waitall` over the real Fortran/FMS
   communicator (or a duplicated one), H2D the unpacked result. Reuses
   `HaloPeerGroup`'s pack/transform/unpack kernels untouched (per the 3A.10
   handoff's own note that only the rendezvous class is replaced); adds a
   real per-exchange D2H/H2D round trip not present in Model B.
2. **CUDA-aware MPI / GPU-direct (Stage 2/3 in the same roadmap).** Avoids
   the extra copy but is a larger lift and was explicitly deferred by
   `project_architecture.md` ("Do not prematurely optimize MPI during early
   correctness validation").

Either direction is a genuine multi-file runtime change to
`cuda/dyn_core/include/fv3_gpu_halo_peer.hpp` and its callers, sized well
above this task's 3-fix integration budget, and involves reasoning about
correctness across a real distributed-process boundary — exactly the kind
of decision the task instructed to escalate rather than attempt on Sonnet.

## 7. Resumption (Prompt 16) — §1/§2 ABI completion DONE; a NEW, distinct
##    escalation trigger found at the real production call site

The §6 blocker above is **CLOSED by Prompt 15**
(`docs/agent_handoff/prompt_15_mpi_halo_transport.md`): a real cross-process
`MpiHaloTransport` backend now exists behind the `HaloTransport` interface,
validated bitwise against the in-process path (six-rank `dyn_core_exit`
78/78 identical). This pass resumed integration from that point and reused
this file's §2 interface table as instructed (not re-derived).

### 7.1 §2 public C ABI — completed and verified

Two ABI functions declared in `fv3_gpu_api.h` (`fv3_gpu_set_numerical_config`,
`fv3_gpu_install_production_dispatch`) needed real implementations; neither
had one before this pass. §3/§4 (geometry/topology installation) needed
**no new ABI** — `fv3_gpu_set_cubed_sphere_topology` (Milestone 3A.10) and the
existing state/field-descriptor path already cover it.

A real CMake constraint, not previously encountered by any earlier prompt,
governed where the second function could be defined:
`cuda/dyn_core/CMakeLists.txt` builds `fv3_dyn_core_gpu` (the main static
library) and a separate `fv3_production_dispatch` static library that links
**PUBLIC** against `fv3_dyn_core_gpu` — never the reverse, on pain of a
static-library dependency cycle the CMakeLists.txt's own comment already
warns about. `fv3_build_production_dispatch()` (the function
`fv3_gpu_install_production_dispatch` must call) lives only in
`fv3_production_dispatch`. Defining the ABI function inside
`fv3_gpu_runtime.cpp` (part of `fv3_dyn_core_gpu`) therefore failed to link
(`undefined reference to fv3::gpu::fv3_build_production_dispatch`) — confirmed,
not guessed, by a full rebuild. Fixed by moving the implementation into
`cuda/dyn_core/runtime/fv3_gpu_production_dispatch.cu`, right after
`fv3_build_production_dispatch`'s own closing brace, as a global-scope
`extern "C"` function; `fv3_gpu_set_numerical_config` stayed in
`fv3_gpu_runtime.cpp` (no cycle there — `set_numerical_config` is native to
that same library). **This consumed 1 of the 3 permitted integration fixes**
(build/link-integration category, §17(11)) — it is a correction of this
pass's own not-yet-finalized ABI addition, not a fix against pre-existing
code, but is tracked conservatively as one fix.

Verified: full clean rebuild (`build_cuda_r8_mpibench`, 100%, no errors);
login-node CTest 13/13 host-only tests PASS (10 GPU-labeled tests correctly
`***Skipped`, no device on the login node); one `gpu_a100` Slurm CTest run,
**22/23 PASS**. The sole failure, `fv3_halo_mpi_transport`, is attributable to
an ad hoc `srun` resourcing/topology artifact of this investigation (Slurm
spread the requested 6 tasks across 4 separate single-GPU nodes instead of
packing them onto the one node the test's 6-ranks-on-one-A100 design
requires, and the run idled to its wall-clock limit) — **not a code
regression**: neither new ABI function touches `fv3_gpu_halo_mpi.cpp` or any
MPI-transport code path, and this exact test already passed at 16,692
checks/0 failures under correct single-node resourcing in Prompt 15. Per the
project's bounded-debugging rule, this was not chased further; reproducing
Prompt 15's exact `sbatch` (not an ad hoc `srun`) is the correct way to
re-confirm it if ever needed, and is recorded here rather than treated as an
open regression.

### 7.2 NEW finding — a second, distinct halo-ownership ambiguity at the real
###     `fv_dynamics.F90` call site (ESCALATION TRIGGER)

Reading the real production call boundary in Repo B to begin §5 (the Fortran
`ISO_C_BINDING` wrapper) surfaced a halo-management layer that no prior
prompt (6A–15) had reason to look at, because all of them worked strictly
inside `dyn_core.F90`'s own savepoint boundaries. `fv_dynamics.F90:616-655`
(the `do n_map=1,k_split` loop, immediately before the `dyn_core` call):

```fortran
616  do n_map=1, k_split
...
620    call start_group_halo_update(i_pack(1), q_con, domain, complete=.false.)   ! (nonhydro/USE_COND path)
626    call start_group_halo_update(i_pack(1), delp,  domain, complete=.false.)
627    call start_group_halo_update(i_pack(1), pt,    domain, complete=.true.)
630    call start_group_halo_update(i_pack(8), u, v,  domain, gridtype=DGRID_NE)   ! (#ifndef ROT3)
...
645    call dyn_core(..., domain, n_map==1, i_pack, last_step, diss_est, time_total)
```

`fv_dynamics.F90` itself **starts** two asynchronous FMS group-halo updates
(`i_pack(1)` for `delp`/`pt`/`q_con`, `i_pack(8)` for `u`/`v` with
`gridtype=DGRID_NE`) and then passes the same `i_pack(*)` array into
`dyn_core` as `intent(inout)`, for `dyn_core` to **complete** internally at
specific points. This is consistent with — and now explains — a detail
already on record from Gate C2 (`CURRENT_STATE.md` §4, `sp2c`): *"the `u`/`v`
halos are in flight (`i_pack(8)` started, not completed)"*. It is a real,
load-bearing FMS halo-pipelining pattern, layered **above** `dyn_core`'s own
internal six-tile GPU halo exchange (`PeerHaloTransport`/`MpiHaloTransport`),
not the same thing as it.

This is squarely the ambiguity the task instructed to escalate on rather than
guess at (§10, "careful FMS-vs-CUDA halo ownership — STOP and escalate if
ambiguous"), for a concrete, source-grounded reason: when the CUDA backend is
selected, `dyn_core`'s real work (all halo exchange, six-tile, GPU-side) is
done entirely inside the CUDA runtime — it has no use for, and does not
complete, `i_pack(1)`/`i_pack(8)`. Two unresolved, correctness-relevant
questions follow directly from that, neither answerable from `dyn_core.F90`
or the CUDA runtime alone:

1. **If the CUDA branch skips issuing `start_group_halo_update(i_pack(1)/(8), ...)`
   entirely** (the seemingly obvious choice, since CUDA does not consume the
   handles), **does anything else in `fv_dynamics.F90` downstream of the
   `n_map` loop, or in a later `n_map` iteration, assume those specific
   `i_pack` slots were started/completed** — e.g. a diagnostic, a later reuse
   of the same slot index for an unrelated field, or FMS-internal bookkeeping
   tied to `i_pack` slot lifetime? This was not checked; it requires reading
   beyond the `dyn_core` call boundary this whole project has scoped to so
   far.
2. **If instead the calls are left in place but simply never completed** (no
   corresponding `complete=.true.`/consumption inside a CUDA `dyn_core`
   substitute), does FMS's group-halo-update machinery tolerate a
   started-but-never-completed handle being reused on the next `n_map`
   iteration, or does it deadlock/corrupt state? This is exactly the kind of
   "non-obvious Fortran/CUDA state-ownership semantics" question the task
   named as its own escalation trigger, and guessing wrong here would risk
   silent state corruption in the *Fortran* side of a mixed-backend run —
   the opposite failure mode of everything Gate C2 validated, and outside
   this task's disallowed-changes list only by accident (it disallows
   "numerical kernel changes" and "new halo transport design," neither of
   which describes this — but resolving it safely still requires reading
   FMS's `mpp_domains` halo-update-handle contract, which is out of the
   3-fix integration budget and this pass's Sonnet-appropriate scope per the
   task's own model-policy).

No source in either repository was modified to investigate this section; it
was found by reading the real call site (`fv_dynamics.F90:606-670`) and the
real `dyn_core` signature (`dyn_core.F90:166-274`) directly, both reproduced
above for reuse without re-deriving them.

### 7.3 What is and is not yet done

Done and verified this pass: §1 (working trees — reused from Prompt 14,
unchanged), §2 (public C ABI — complete, both functions implemented in the
correct target, full rebuild clean), §3/§4 (confirmed no new ABI needed).
§20 regression check performed: R8 ctest 22/23 GPU + 13/13 host, one
resourcing artifact explained above, not a regression.

Not started: §5 (Fortran `ISO_C_BINDING` wrapper — blocked by 7.2 above,
since the wrapper's own halo-related code cannot be written correctly until
the `i_pack` question is resolved), §6 (backend switch), §7 (communicator
handoff), §8 (context lifecycle), §9 (state-sync taxonomy per field — the
raw field list is now available verbatim from the real signature in this
section, ready for whoever resolves 7.2), §10 (this section *is* that
analysis, escalated rather than resolved), §11 (build/link), §12–§16 (no run
attempted), §21 partial (this section + the `CURRENT_STATE.md` update
constitute the required documentation for this pass; nothing else to add
until 7.2 is resolved).

**1 of 3 permitted integration fixes consumed** (§7.1, the ABI
placement/build-link correction). No numerical kernel was changed. No new
halo transport was designed. No source in Repo B was modified.

---

## 8. Resumption (Prompt 17) — the `i_pack` / FMS group-halo ownership question,
##    RESOLVED, and the Fortran-side integration built on it

§7.2 stopped on one question: `fv_dynamics.F90` starts FMS group halo updates
`i_pack(1)` and `i_pack(8)` *before* calling `dyn_core` and passes the handle
array in for completion inside it, while the CUDA `dyn_core` performs its own
six-tile device-side exchange. May the CUDA branch skip issuing those starts,
must it complete them some other way, or are both mechanisms required?

**Answer: the CUDA branch must NOT ISSUE `i_pack(1)`/`i_pack(8)` at all — and
must still issue `i_pack(10)`.** Not one of those three is a judgement call;
each follows from a balance count in the authoritative source plus the FMS
non-blocking group-update contract.

### 8.1 The Fortran pipeline, counted (§1)

For one `n_map` iteration, with `n_split` acoustic substeps:

| handle | fields | started in `fv_dynamics` | started in `dyn_core` | completed in `dyn_core` | net outstanding at `dyn_core` exit |
|--------|--------|--------------------------|-----------------------|-------------------------|-------------------------------------|
| `i_pack(1)` | `delp`, `pt` (+`q_con`/`cappa` when `USE_COND`/`MOIST_CAPPA`) | 1 (`:663-664`, and `:656-660` for the conditional pair) | `n_split` (`:899-902`) | 1 at `it==1` (`:496`) + `n_split` (`:933`) | **0** |
| `i_pack(8)` | `u`, `v`, `gridtype=DGRID_NE` | 1 (`:667`) | `n_split−1` (`:1193`, `it<n_split`) | `n_split` (`:524`) | **0** |
| `i_pack(10)` | `q` tracers | 0 | 1 (`:1295-1301`, after the acoustic loop) | 0 | **1 — deliberately** |

`i_pack(10)` is the only handle that crosses `dyn_core`'s exit outstanding, and
`fv_dynamics` completes it downstream (`:727/:733/:737`). Both other slots are
exactly balanced across the `dyn_core` call: the outer start is not a separate
logical operation, it is the *first iteration* of a rotation whose remaining
iterations `dyn_core` performs itself.

### 8.2 The FMS contract (§2)

Established from the `mpp_domains` non-blocking group-update implementation
(reached through `fv_mp_mod`'s `start_group_halo_update` /
`complete_group_halo_update` wrappers), reported as facts only:

- **Start is eager, not lazy.** `mpp_start_group_update` packs the field data
  into the shared non-blocking buffer, pre-posts the receives, posts the sends,
  increments the module-level `num_nonblock_group_update`, and advances the
  shared `nonblock_group_buffer_pos` cursor. Data is captured at start; the
  field arrays may be modified afterwards, which is exactly why FV3 can start
  `i_pack(1)` on `delp`/`pt` and then overwrite `dp1` from `delp` immediately.
- **Complete is mandatory once started.** `mpp_complete_group_update` FATALs if
  the counter is already below 1, decrements it, waits on the outstanding
  requests, and rewinds the shared cursor **only when the counter reaches zero**.
- **Therefore a started-but-never-completed handle is a leak, not a no-op**: one
  buffer slot and its MPI requests per `k_split` iteration, with the shared
  cursor never rewound. Design B ("leave FMS active, let CUDA complete the
  handles") would also require CUDA to consume data it does not use, and its
  single `COMM_TAG_1` makes the group-update sequence order-sensitive across
  ranks — so an asymmetric completion pattern is a hang risk, not merely waste.
- **Never starting is legal.** FMS keeps no per-handle state that a later
  operation requires; the handles are `save`d module data in `fv_dynamics`
  (`:282`, `type(group_halo_update_type), save :: i_pack(12)`) and are
  reinitialized by their next `start`. Verified by grep that no other
  `fv_dynamics` code path reads `i_pack(1)` or `i_pack(8)`.

### 8.3 Mapping to CUDA logical groups (§3)

| FMS handle | fields / staggering | CUDA group | same consumer, same point? | verdict |
|---|---|---|---|---|
| `i_pack(1)` | `q_con`, `cappa`, `delp`, `pt`; A-grid scalars | `FV3_GPU_H_DSCALAR` | yes — resolved before `c_sw` (`dyn_core.F90:535`), matching the Fortran completion at `:496` | **DUPLICATE** |
| `i_pack(8)` | `u`, `v`; DGRID_NE | `FV3_GPU_H_DWIND_IN` (and `H_DWIND_NEXT` for the intra-loop restarts) | yes — resolved before the same substep consumers, matching `:524` | **DUPLICATE** |
| `i_pack(10)` | `q` tracers | *none* | n/a — `q` is a host-side array the CUDA `dyn_core` neither reads nor writes | **DISTINCT REQUIRED** |

### 8.4 Design classification (§4)

| design | verdict | reason |
|---|---|---|
| **A** — CUDA owns the exchange; the corresponding FMS starts are not issued | **CORRECT** — adopted | the two handles are balanced across the call, so not issuing them leaves FMS in exactly the state a completed rotation would; the exchanges are proven duplicates |
| **B** — FMS stays active; CUDA completes/consumes the handles | **UNSAFE** | requires CUDA to consume data it does not use, and an asymmetric completion pattern against a single shared tag/cursor is a hang risk |
| **C** — start *and* immediately complete in the CUDA branch | CORRECT but **REDUNDANT** | balances the counter, but pays a full six-rank host MPI exchange per `k_split` whose result is discarded |
| **D** — leave the starts, never complete | **INCORRECT** | leaks one shared-stack slot and its MPI requests per `k_split`; cursor never rewound |
| **E** — fabricate placeholder handles for the CUDA path | **INCORRECT/unnecessary** | no handle is passed at all under A, so there is nothing to fabricate (§7 of the task) |

### 8.5 What was implemented (§6, §8)

Repo B (`GEOSfvdycore_fortran_perf`), production Fortran — the *only* changes,
all inside `#ifdef USE_FV3_CUDA` or guarded by a runtime flag that is a
compile-time `.false.` when that macro is absent, so the Fortran backend is
source-semantically unchanged:

- `@fvdycore/model/fv_dynamics.F90` — `use fv3_gpu_dyncore_bridge`; a
  `use_cuda_dyncore` flag set from `fv3_gpu_backend_enabled()`; the
  `if (.not. use_cuda_dyncore)` guard around the `i_pack(1)`/`i_pack(8)` start
  block (`:654-669`); and the backend branch at the `dyn_core` call
  (`:682-706`), which calls `fv3_gpu_interface_step_dyn` and then issues
  `i_pack(10)` under the same `nq > 0 .and. .not. inline_q` guard the Fortran
  backend uses. The reasoning above is recorded in-source at the two decision
  points so it cannot drift from the code.
- `FV_StateMod.F90` — a `RUN_CUDA_FV3:` MAPL resource (default 0 ⇒ Fortran
  backend), read alongside the existing `RUN_GTFV3:`; `fv3_gpu_bridge_init` at
  the end of `FV_InitState`, where `FV_Atm(1)%gridstruct` is fully built and the
  ESMF VM communicator is in scope (`ESMF_VMGet(VM, mpiCommunicator=...)`); and
  `fv3_gpu_bridge_final` in `FV_Finalize` **before** `fv_end`, because the
  registered descriptors point at `FV_Atm` host memory and the transport detach
  is collective. `FV_Run` needs no change at all — unlike gtfv3, this backend
  branches *inside* `fv_dynamics`.
- `CMakeLists.txt` / `@fvdycore/CMakeLists.txt` — `BUILD_FV3_CUDA_INTERFACE`
  (OFF by default) with `FV3_CUDA_SOURCE_DIR` / `FV3_CUDA_LIBRARY_DIR` /
  `FV3_CUDA_TOOLKIT_LIBDIR`. The three bridge modules are compiled into the
  **`fvdycore` subcomponent**, not the parent: `fv_dynamics.F90` is their
  consumer, and putting them in the parent would make the module dependency
  circular. The nine prebuilt archives are linked as a `--start-group` set
  because the production dispatch table and the work packages reference each
  other mutually.

Repo A (`FVdycoreCubed_GridComp`), new `cuda/interface/`:

- `fv3_gpu_abi_mod.F90` — member-for-member `bind(C)` mirror of the public ABI
  (8 structs, ~30 constants, the `FieldId` identity map, 14 interface blocks).
  Every entry point validates `abi_version`/`struct_bytes` against `sizeof` on
  the C side (`header_ok`, `fv3_gpu_runtime.cpp:40`), so a layout mismatch fails
  loudly at the boundary instead of corrupting silently. Compiles clean under
  `ifort` with `-r8`.
- `fv3_gpu_field_registry.F90` — builds the field descriptors from `FV_Atm`.
- `fv3_gpu_dyncore_bridge.F90` — owns the persistent per-rank GPU context and
  runs the verified lifecycle
  `create → initialize_state → set_numerical_config → install_production_dispatch
  → set_cubed_sphere_topology → attach_mpi_halo_transport`, then one
  `sync_from_host → dyn_core_step → sync_to_host → event_wait` per `dyn_core`
  call. The rank↔tile map is allgathered and cross-checked rather than assumed,
  so a mis-wired launcher fails at init instead of exchanging plausible-looking
  wrong halo data.

The per-call H2D/D2H is required, not a residency regression: between two
successive `dyn_core` calls `fv_dynamics` runs tracer advection and the
Lagrangian-to-Eulerian remap, both of which rewrite `delp/pt/u/v/w/delz/pe/pk/pkz`
on the host. Remapping is one of the synchronization boundaries
`project_architecture.md` names explicitly. Nothing crosses *between* work
packages inside a step; the acoustic loop stays entirely on the device.

### 8.6 Build environment reconciliation

The fixture is Intel oneAPI 2024.2 `ifort` + Intel MPI 2021.13, and its own
`@env/g5_modules` already loads `comp/gcc/12.3.0`. The CUDA archives were
therefore rebuilt with GCC 12.3 / `nvcc` 12.5 against **Intel MPI 2021.13** —
`find_package(MPI)` resolved to `/usr/local/intel/oneapi/2024/mpi/2021.13`,
matching the fixture exactly, so the MPI halo transport and the `ifort`
executable share one MPI ABI. Build dir `build_cuda_r8_intelmpi`
(`FV3_DYN_CORE_ENABLE_MPI_HALO=ON`, `FV3_DYN_CORE_PRECISION=R8`,
`CMAKE_CUDA_ARCHITECTURES=80`): all 9 archives built, 0 errors. `nvcc` must be
given as the explicit `.../24.7/cuda/12.5/bin/nvcc` path — the generic
`.../24.7/cuda/bin/nvcc` on `PATH` has a `targets` layout without
`libcudart_static.a`/`libcudadevrt.a` and fails CMake's CUDA language probe.

### 8.7 State-transfer corrections found by comparing the two backends (§10, §11)

Nine dyn_core-boundary state items needed attention before the two backends
agreed. All are *boundary completion* — state that crosses `dyn_core`'s
interface and that the CUDA path was not handing back — not numerical changes.
Localization order was the one that found all 13 Gate C2 defects: **audit what
crosses the boundary before touching arithmetic.**

Two savepoint stages were added to `fv_dynamics.F90` for this, both
instrumentation only: **`dce`** (dyn_core ENTRY) and **`dcx`** (dyn_core EXIT),
17 fields each, dumped per rank per call. `dce` is what proves an *entry*-state
mismatch is an integration-fidelity defect rather than a kernel defect; `dcx` is
the accepted comparison boundary. One comparator serves both:
`tools/savepoints/compare_dcx_backends.py --stage {dce,dcx} --call N <fortran_dir> <cuda_dir>`
(the two directories are **positional**).

1. **Flux capacitors `mfx`/`mfy`/`cx`/`cy` were never copied out.** They are
   accumulated inside `dyn_core` and consumed by tracer advection and the
   remap. Added `fv3_gpu_registry_export_flux_capacitors(mfx,mfy,cx,cy,crc)`,
   called in `fv3_gpu_interface_step_dyn` after the D2H event wait. Binding
   `Atm%mfx` at registry-build time would have been **wrong**: the arrays
   `fv_dynamics` passes to `dyn_core` are its own locals, not `Atm` members.
2. **`ws` was not copied out.** Added `fv3_gpu_registry_export_ws`. Worth
   recording honestly: this was implemented on contract grounds and **measurably
   changed nothing** in the comparison — `ws` is not read after `dyn_core` in
   this configuration.
3. **Three field descriptors had wrong bounds** (`z_rat`, `ut`/`vt`, `peln`);
   corrected and the full descriptor set re-audited.
4. **`pe`'s one halo ring — the root cause of the repeated-call divergence.**
   This was the last and by far the most consequential item, so it is written
   out in full below.

#### The `pe` halo ring (the repeated-call defect)

**Symptom.** Call 1 was BITWISE identical at the entry boundary on all 13 state
fields, 6/6 ranks. Call 2 was not: `pt` differed by 1.032e-04 **with the same
maximum at every vertical level**, `pkz` by 1.5e-05→1.28e-04 growing with `k`,
and `u` by 1.845e+00 — but `u` at only 960 of 12600 cells, i.e. 16 per level per
rank, while `w`/`delz`/`delp` had **zero** cells above threshold.

**Localization.** The footprint was the discriminator, not the magnitude: `u`
differed at **exactly `j=js` and `j=je+1`**, and `v` at **exactly `i=is` and
`i=ie+1`**, on all six ranks. That is a seam-line signature, and it points at
something read one cell outside the owned box.

**The consumer, from source.** `Lagrangian_to_Eulerian`'s D-grid wind/flux
vertical remap reads `pe`'s halo ring directly:

```
fv_mapz.F90:686   pe1(i,k) = 0.5*(pe(i,k,j-1)+pe(i,k,j))     j = js..je+1
fv_mapz.F90:693   pe2 likewise from pe(i,km+1,j-1)
fv_mapz.F90:729   pe0(i,k) = 0.5*(pe(i-1,k,j)+pe(i,k,j))     i = is..ie+1
fv_mapz.F90:740   pe3 likewise from pe(i-1,km+1,j)
```

`pe1`/`pe2` drive `map_scalar` for `u`/`mfy`/`cy`; `pe0`/`pe3` for `v`/`mfx`/`cx`.
At `j=js` that reads halo row `js-1`; at `j=je+1`, row `je+1`; at `i=is`, column
`is-1`; at `i=ie+1`, column `ie+1`. **Exactly the measured footprint and no
other** — and it explains why `cy` shared `u`'s error lines while `cx`/`mfx`
shared `v`'s.

**The authoritative writer.** `pe` is declared
`pe(is-1:ie+1, 1:npz+1, js-1:je+1)` — owned box plus exactly one ring — and the
Fortran fills that ring *inside* `dyn_core` with `pe_halo`
(`dyn_core.F90:1504-1532`), called at `:1011` under `if (remap_step)`.
`remap_step` (`:424-428`) is `breed_vortex_inline .or. it==n_split`, i.e. true on
the last acoustic substep. The CUDA `dyn_core` never writes that ring — a
coverage gap already on record — so the D2H handed the ring back holding the
**previous call's** uploaded values. That is why call 1 was clean and call 2 was
not.

**Measured, before the fix:** at the call-1 exit the ring differed by
**≈1.0016e+05–1.0018e+05** on all six ranks while the owned box agreed at
**2.9e-11–4.4e-11**. The pre-existing `CURRENT_STATE.md` §10 note that this ring
is "proven INERT downstream of dyn_core" is therefore **incomplete and is
corrected**: the wind/flux remap is a real consumer.

**Why a host-side recompute is bit-for-bit equivalent, not an approximation.**
`pe_halo` runs only at `it==n_split`, and `awk` over
`dyn_core.F90` confirms **no `delp(...)=` assignment exists between line 1011 and
line 1375**. So the exit ring is exactly `cumsum(ptop, exit delp ring)`. `delp`'s
own halo ring is already host-resident after the D2H (`delp` is registered over
`isd:ied,jsd:jed`), so reproducing `pe_halo`'s two loops on the host in the same
order and precision gives the same bits — with **no added host/device transfer
and no added kernel**.

**Implementation.** `fv3_gpu_interface_pe_halo` in
`cuda/interface/fv3_gpu_dyncore_bridge.F90` (public, guarded on
`bridge_ready`, reproducing `dyn_core.F90:1512-1530` verbatim), called from
`fv_dynamics.F90` immediately after the CUDA step's error check, inside the
`use_cuda_dyncore` arm only. `pe_halo` itself is `private`
(`dyn_core.F90:142`), which is why the loop is reproduced rather than called.
Same category as the `ws` and flux-capacitor copy-outs above.

**The secondary signature is explained, not merely correlated.** `consv_te = 1.0`
in this fixture, so `compute_total_energy` runs inside
`Lagrangian_to_Eulerian`. Corrupted seam-line winds corrupt total energy, and
the correction redistributes a **uniform** `pt` offset over the whole owned box —
which is why `pt`'s difference was identical at every `k`, with `pkz` following
it. This supersedes the earlier leading `r_cappa = kappa_dry` / `-DMOIST_CAPPA`
hypothesis as the explanation for the repeated-call divergence; that item
remains a separately recorded coverage gap, but is no longer a candidate here.

### 8.8 Result — matched Fortran vs CUDA standalone runs (§9, §10)

Both backends ran in **one** Slurm allocation (job **58244118**, one node, 6
tasks, `gpu_a100`/`rome`/`gpu:1`), from the **same executable**, differing only
in `AGCM.rc`'s `RUN_CUDA_FV3` (0 vs 1). Configuration: c8, npz=10, n_split=3,
k_split=1, R8, 6 MPI ranks, one tile per rank. Both arms `rc=0`; elapsed 25 s.

**6 dyn_core calls** were executed (clears the ≥5 requirement). §9 checks:

| check | result | evidence |
|---|---|---|
| six ranks participate | **PASS** | 1296 CUDA savepoint files = 6 ranks × 2 stages × 6 calls × 18 fields, exactly; all 6 ranks present at every call |
| one context per rank, created once | **PASS** | `fv3_gpu_create` (`bridge:181`) is inside the `save`d `if (bridge_ready) return` guard (`:130`) |
| transport attached once | **PASS** | `fv3_gpu_attach_mpi_halo_transport` (`:251`) inside the same guard |
| context reused, not recreated | **PASS** | the init NOTE (`:263`) appears **exactly once** in a 6-call run |
| no FMS-handle leak / deadlock | **PASS** | run completed in 25 s, `rc=0`, no FATAL; §8.1's balance count predicts zero outstanding |
| no CUDA halo deadlock | **PASS** | six-rank host-staged `MPI_Waitall` exchange completed on all 6 calls |
| no NaN/Inf | **PASS** | 0 nonfinite over 6 calls × 6 ranks × 13 fields |
| `delp` positive | **PASS** | owned-box min 9967.18 → 9957.52 Pa across calls 1→6, monotone and physical |

**dyn_core EXIT (`dcx`) — the accepted comparison boundary.** All 17 fields,
all 6 calls, worst over 6 ranks, grade **ROUND-OFF SCALE or SMALL NUMERICAL
DIFFERENCE**:

| call | worst max_rel | field | `u` max_abs | `v` max_abs | `pt` max_abs |
|------|---------------|-------|-------------|-------------|--------------|
| 1 | 1.716e-11 | `w` | 3.517e-13 | 4.012e-13 | 1.776e-14 |
| 2 | 1.463e-11 | `w` | 4.192e-13 | 4.339e-13 | 3.197e-14 |
| 3 | 1.889e-11 | `w` | 5.730e-13 | 5.698e-13 | 3.553e-14 |
| 6 | 2.008e-11 | `w` | 8.562e-13 | 6.999e-13 | 4.441e-14 |

Growth from call 1 to call 6 is slow and roughly linear (`u` 3.5e-13 → 8.6e-13),
which is round-off accumulation across six 6-rank remap cycles, **not**
divergence. `w`'s max_rel is its own cancellation floor, as at Gate C2. `mfx`/`mfy`
show max_abs ≈ 4–17 against a dynamic range ~1e14, i.e. max_rel ≤ 3.3e-14 —
ROUND-OFF, and exactly the reading rule already recorded in `CURRENT_STATE.md`
§9 ("a large `max_abs` is not by itself a failure").

**dyn_core ENTRY (`dce`) — first-call integration fidelity.** Call 1 is
**BITWISE identical, 6/6 ranks**, on all 13 state fields (`u`, `v`, `w`, `delz`,
`delp`, `pt`, `omga`, `ua`, `va`, `pe`, `peln`, `pk`, `pkz`) — the strongest
available statement that the integration hands `dyn_core` the same state the
Fortran backend does. Call 2 entry grades ROUND-OFF/SMALL, matching the exit.

One reading caveat, established rather than assumed: at the **entry** boundary
`mfxL`/`mfyL`/`cxL`/`cyL` are the flux/Courant accumulators *before* `dyn_core`
writes them, and they demonstrably hold uninitialized memory in **both**
backends — values of 6.6e+203, 5.8e-310, 2.3e-312, 1.0e-115, with the NONFINITE
counts **identical** Fortran-side and CUDA-side. The entry comparison for those
four fields is therefore vacuous; their defined boundary is the exit, where they
grade ROUND-OFF.

### 8.9 Reproduction

- `experiments/p17_both.sbatch` — runs **both** backends in one allocation. The
  freshly built binary must be staged into **both** `p17_fortran_ref/` and
  `p17_cuda/`, because `fv3.j:426-431` copies `$EXPDIR/StandAlone_FV3_Dycore.x`
  into the scratch dir when present.
- `p17_rebuild_exe.sh` — Fortran-only changes (sources `@env/g5_modules.sh`,
  `make -j 12 StandAlone_FV3_Dycore.x`).
- `p17_rebuild_all.sh` — only when a CUDA C++/`.cu` source changes; rebuilds the
  nine archives in the GCC/nvcc/Intel-MPI environment first, then relinks.
  `FV3_CUDA_SOURCE_DIR` points at this repo's own `cuda/` tree, so there is no
  second copy of the CUDA sources to keep in sync.
- `p17_regress.sbatch` — the R8 CTest regression, **one node / 6 tasks / 1 GPU**,
  which is what `fv3_halo_mpi_transport`'s `mpiexec -n 6` requires. Do not let
  the six tasks spread across nodes; a node-spread failure of that test is a
  scheduler artifact, not a code regression.
- Comparator: `tools/savepoints/compare_dcx_backends.py --stage {dce,dcx}
  --call N <fortran_dir> <cuda_dir>` (positional dirs). Needs
  `source .../GEOSfvdycore_fortran_perf/@env/g5_modules.sh` for a numpy-bearing
  Python, and `PYTHONPATH=$PWD` from `tools/savepoints`.

### 8.10 Regression (§14) — and one NEW OPEN DEFECT found by it

| item | result | evidence |
|---|---|---|
| existing Fortran backend | **PASS** | `p17_fortran_ref` arm of job 58244118, `rc=0`, savepoints written by all 6 ranks |
| R8 CUDA CTest (MPI halo ON) | **22/23** | job 58244300, single node / 6 tasks / 1 GPU |
| single-process `HaloPeerGroup` path | **PASS** | covered by the 22 passing tests, incl. the multitile in-process halo test and `fv3_dispatch_smoke` |
| Prompt-15 six-rank MPI transport | **FAIL, intermittent** | jobs 58244300 / 58244365 / 58244385 — see below |
| new standalone CUDA short run | **PASS** | §8.8 |

**The one CTest failure is a genuine, pre-existing, intermittent defect — not
this pass's doing, and not a resourcing artifact.** Correct single-node
six-task resourcing was used throughout, so §14's node-spread exemption does
not apply and was not invoked.

Characterization, in the order the evidence was gathered:

1. **Intel MPI 2021.13 build** (`build_cuda_r8_intelmpi`, the build the
   standalone executable links; job 58244365, 5 repeats): **5/5 FAIL**, 1–2
   failed checks per run out of ~16.6 k.
2. **Prompt 15's own OpenMPI 4.1.6 build** (`build_cuda_r8_mpi`, job 58244385,
   5 repeats): **3/5 PASS, 2/5 FAIL.** This is decisive on attribution — that
   build directory and all of its objects predate this pass, and every
   transport source (`fv3_gpu_halo_exchange.{cu,hpp}`,
   `fv3_gpu_halo_transport.hpp`, `fv3_gpu_halo_mpi.cpp`) plus the test itself
   is byte-unchanged since Prompt 15 (mtimes 11:15–11:44 vs this pass's
   12:37/13:55/16:29 edits, which are confined to the ABI/registry/bridge).
   **Prompt 15's recorded "16692 checks / 0 failures" was therefore a passing
   sample, not a determinism guarantee**, and that claim should be read as
   retracted in its strong form.
3. **Signature, stable across every occurrence:** the offending destination
   cell is always at **`h=1 a=1 k=1`** of some `(edge, field)` block; it always
   holds **exactly 0.0**, which the provenance oracle decodes as
   `tile 0 (-8,-8,0)` — i.e. **never written**; and the afflicted
   `(rank, tile, edge, field)` **moves between runs** (observed: tiles 1/3/5,
   edges N and E, fields 0 and 1). Only wind groups were observed failing;
   scalar groups never were, in 10 runs.
4. **The per-occurrence blast radius is UNKNOWN and may exceed one cell.** The
   test emits one report per `(edge, field)` and then
   `goto next_field` (`test_halo_mpi_transport.cu:328`), so the remaining cells
   of that block are never checked — and the check-count drop
   (16692 → 16621 / 16609 / 16538) is exactly those skipped cells. An entire
   ~71-cell edge block may be missing. **Do not assume "one cell".**
5. **Already ruled out by reading, so do not redo it:** the send-side ordering
   is correct. `fv3_gpu_halo_mpi.cpp:206-216` issues the D2H as
   `cudaMemcpyAsync` on the **caller's** stream — hence stream-ordered behind
   the pack the runtime enqueued — and waits with
   `cudaEventRecord`/`cudaEventSynchronize` on that same stream before any
   `MPI_Isend`. A pack→D2H race is not the lead.

Recommended first diagnostics for the next pass, both pure instrumentation:
(a) remove the `goto next_field` suppression and re-run, to learn whether one
cell or a whole block is lost — those imply very different mechanisms;
(b) log `slot_base`/`used_bytes` per `(group, edge, field)` on donor and
receiver to test a slot-offset/size disagreement (`agree_on_layout`'s
positional field compaction is the obvious place for a donor/receiver
mismatch to hide).

**What this does and does not invalidate.** It does **not** retract Prompt
15's dyn_core-exit equivalence (78/78 bitwise at 1 and 5 calls) or §8.8's
standalone comparison (all 17 exit fields ROUND-OFF/SMALL over 6 calls) —
those are measured outcomes on real runs and stand as recorded. It **does**
retract any claim that the six-rank transport is *deterministically* correct,
and it must be closed before a longer standalone validation run, since that
transport is exactly what the standalone CUDA backend uses.

### 8.11 Budget and stop condition

**3 of 3 permitted substantial fixes consumed:** (1) the `fv_dynamics.F90`
Design A start-skip branch; (2) the bridge/registry/ABI plumbing, including the
flux-capacitor and `ws` copy-outs and the `z_rat`/`ut`/`vt`/`peln` descriptor
corrections; (3) `fv3_gpu_interface_pe_halo`. Savepoints, the comparator's
`--stage` option, the `/tmp` analysis probes and the three `p17_*.sbatch` jobs
are instrumentation, not fixes. The intermittent transport defect in §8.10 was
therefore **characterized and reported rather than fixed** — attempting it
would have been a fourth substantial fix and a §12(4) stop. Nothing was
committed or pushed.
