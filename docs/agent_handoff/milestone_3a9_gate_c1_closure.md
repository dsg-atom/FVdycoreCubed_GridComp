# Milestone 3A.9 — Full dyn_core Gate C1 Closure

## Status line

**GATE C1 BLOCKED** — one blocker, named in §20 and §25 below. Everything else
in the gate definition is closed and demonstrated on an A100.

The blocker is not a defect introduced by this milestone and not a missing
numerical stage. It is that **no halo transport exists**, so honest halo
readiness cannot be established on a production `grid_type = 0` topology, and
§8 of the milestone prompt explicitly forbids faking it:

> If a genuine inter-rank exchange is required even for the target path, report
> Gate C1 blocked rather than fake readiness.

It is required, so this reports blocked.

## Provenance

| item | value |
| --- | --- |
| branch | `cuda/gpu-bringup` |
| parent commit | `7ddb6f9` (Record milestone 3A.8 commit SHA in handoff) |
| this milestone's commit | recorded in a follow-up commit, as in 3A.6/3A.7/3A.8 |
| authoritative Fortran | `GEOS-ESM/GFDL_atmos_cubed_sphere`, tag `geos/v3.0.0`, SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45` |
| reference clone used | `/discover/nobackup/projects/imvi-nse/jli30/fv3_reference/GFDL_atmos_cubed_sphere` |
| platform | NASA NCCS Discover, `gpu_a100`, `--constraint=rome`, node `warpa003` |
| toolchain | GCC 12.3.0, OpenMPI 4.1.6, CUDA 12.5.82, CMake 3.30.3 |
| builds | `build_cuda_r8`, `build_cuda_r4` |

All `.F90` line numbers below refer to that SHA.

---

## §1 Authoritative dyn_core sequence

Reconstructed from `model/dyn_core.F90` at `ca30e26`, for the production
first-run configuration (`grid_type = 0`, non-nested, nonhydrostatic,
`a_imp = 1.0`, `hord_* = ±6`, `beta = 0`, `n_con ≠ 0`, `d_con > 1e-5`).

`it` is the acoustic substep index; the loop is `do it=1,n_split` at `:399`,
closing at `:1196`. Everything before `:399` and after `:1196` is once per
dyn_core call.

| # | Fortran | operation | CUDA service | dispatch slot | halo required on entry | writes | next consumer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0a | `:296-298` | `dt = bdt/n_split`, `dt2`, `rdt` | runtime, once per call | — | — | `DispatchParams` | every stage |
| 0b | `:300` | `ms = max(1, m_split/2)` for `Riem_Solver_c` | derived at the C-grid call site | — | — | — | 4 |
| 0c | `:372-375` | zero `mfx, mfy, cx, cy` | `zero_accumulators` | — | — | 4 capacitors | 7, 8 |
| 0d | `:377-379` | zero `heat_source` when `d_con > 1e-5` | `zero_accumulators` | — | — | `heat_source` | 12 |
| 1 | `:402/:504` | `u, v` halo (`H_DWIND_IN`) | `resolve_halo` | — | — | — | 2 |
| 2 | `:427/:507` | `w` halo (`H_W_IN`) | `resolve_halo` | — | — | — | 2 |
| 3 | `:515-523` | `d2a2c_vect` + `divergence_corner` + `c_sw`, `k = 1..npz` | `launch_cgrid_predictor_v3` | `cgrid_predictor` | `H_DWIND_IN`, `H_W_IN` | `ua, va, ut, vt, uc, vc, delpc, ptc, divgd` | 4, 5, 6 |
| 4 | `:528/:636` | `divgd` halo (`H_DIVGD`), only when `nord > 0` | `resolve_halo` | — | — | — | 7 |
| 5 | `:459/:552` | `gz` halo (`H_GZ_FIRST`), **`it == 1` only** | `resolve_halo` | — | — | — | 6 |
| 6 | `:543-583` | `update_dz_c` then `Riem_Solver_c` | `launch_cgrid_vertical_solver` | `cgrid_vertical_solver` | `H_GZ_FIRST` (first substep) | `gz, ws3, pkc, pk3` (+ `zh` C-grid) | 7 |
| 7 | `:614` | `p_grad_c` | `launch_cgrid_pressure_v3` | `cgrid_pressure` | — | `uc, vc` | 8 |
| 8 | `:618/:640` | `uc, vc` halo (`H_CWIND`) | `resolve_halo` | — | — | — | 9 |
| 9 | `:700-829` | per-level damping controls + `d_sw` | `launch_dgrid_dynamics_cuda` | `dgrid_dynamics` | `H_CWIND`, `H_DIVGD` | `u, v, w, delp, pt, mfx, mfy, cx, cy, heat_source, diss_est` | 10, 12 |
| 10 | `:844/:878` | `delp, pt, w` halo (`H_DSCALAR`) | `resolve_halo` | — | — | — | 11 |
| 11 | `:905-906` | `update_dz_d` — WP-C sequencing around WP-B transport | `launch_dgrid_height_stage` → `launch_update_dz_d` → `dispatch.dgrid_height_transport` | `dgrid_height` (+ `dgrid_height_transport` **called from inside it**) | `H_DSCALAR` | `zh, ws` | 12 |
| 12 | `:918-923` | `Riem_Solver3` (`ms = m_split`) | `launch_vertical_solver` | `vertical_solver` | — | `w, delz, pe, pkc, pk3` | 13 |
| 13 | `:931/:967` | `zh` and `pkc` halos (`H_ZH`, `H_PKC2`) | `resolve_halo` | — | — | — | 14 |
| 14 | `:940-1019` | `nh_p_grad` (`a2b_ord4`, `pk3_halo`, `pln_halo`, `pe_halo`) | `launch_nh_pressure_cuda` | `nh_pressure` | `H_ZH`, `H_PKC2` | `u, v, du, dv` | next substep / 15 |
| — | `:1196` | end acoustic loop | — | — | — | — | — |
| 15 | `:1231-1240` | `heat_source` halo then `del2_cubed` + diffusive heating, `nf_ke` passes, bounded by `bdt*delt_max` | `launch_post_loop_heating_cuda` | `post_loop_heating` | `H_HEAT` | `pt, heat_source` | dyn_core exit |

Dead in this configuration and deliberately absent: `nest_halo_nh`,
`Riem_Solver3test`, `imp_diff_w`, `d3_damp`.

### The four capacitors, precisely

`:372-375` zeroes **`mfx`, `mfy`, `cx`, `cy` and nothing else**, unconditionally,
once per `k_split`, *before* `do it=1,n_split`. `heat_source` is also zeroed but
only under `d_con > 1.0e-5` (`:377-379`). `diss_est` and `dpx` are **not** zeroed
in `dyn_core` — they belong to `fv_dynamics`. `zero_accumulators`
(`runtime/fv3_gpu_runtime.cpp`) matches this exactly, including the conditional.

---

## §2 Dispatch slot audit

`NumericalDispatch` has nine `LaunchFn` slots plus `corner_edge_service`.

| slot | status entering 3A.9 | action | status now |
| --- | --- | --- | --- |
| `cgrid_predictor` | NULL | **A** — wire existing WP-A service | WIRED + GPU EXECUTED |
| `cgrid_vertical_solver` | NULL | **A** — genuinely distinct (`update_dz_c` + `Riem_Solver_c`), not an alias | WIRED + GPU EXECUTED |
| `cgrid_pressure` | WIRED + GPU EXECUTED | — | WIRED + GPU EXECUTED |
| `dgrid_dynamics` | WIRED + GPU EXECUTED | — | WIRED + GPU EXECUTED |
| `dgrid_height_transport` | WIRED but **UNREACHABLE** through the public entry | **A** — made reachable via `ExecutionContext::dispatch` | WIRED + GPU EXECUTED **through `fv3_gpu_dyn_core_step`** |
| `dgrid_height` | NULL | **A** — resolved as the WP-C `update_dz_d` sequencing stage; ownership split from `dgrid_height_transport` documented below | WIRED + GPU EXECUTED |
| `vertical_solver` | WIRED + GPU EXECUTED | — | WIRED + GPU EXECUTED |
| `nh_pressure` | WIRED + GPU EXECUTED | — | WIRED + GPU EXECUTED |
| `post_loop_heating` | WIRED + GPU EXECUTED | — | WIRED + GPU EXECUTED |
| `corner_edge_service` | deliberate fail-stub | **D** — unnecessary for this configuration; see §19 | fail-stub, never requested |

No production-required slot is NULL or a fail-stub. See §19 for the evidence
behind each "GPU EXECUTED" claim, which is per-slot probe counts rather than an
overall success code.

---

## §3 `cgrid_predictor` — closed

Wired to the existing `cgrid::launch_cgrid_predictor_v3`. No WP-A kernel was
duplicated; the stage is the existing `d2a2c_vect` + `divergence_corner` + `c_sw`
composition over `k = 1..npz`, which is `dyn_core.F90:515-523`.

One structural difference from the Fortran, recorded in §18: the CUDA
`csw_fluxes_level` service is split into separate x-flux and y-flux phases
(milestone task #12). The Fortran computes both inside one `k` body; splitting
them changes launch structure, not arithmetic or ordering, because the y phase
reads no value the x phase writes.

## §4 `cgrid_vertical_solver` — closed, and it is genuinely distinct

The §4 question was whether this slot is a real stage, an alias, or a historical
abstraction to collapse. It is **real**. `dyn_core.F90:543-583` performs two
operations that nothing else performs:

1. `update_dz_c` (`nh_utils.F90`) — the C-grid height update, which uses the
   `dp_ref` (dp0) ratio weights and writes `ws3`;
2. `Riem_Solver_c` with `ms = max(1, m_split/2)` and `dt = dt2`.

Neither is the D-grid `Riem_Solver3` and neither is `update_dz_d`. Wiring it was
not "making a nullptr disappear": before this milestone the entire C-grid
vertical half of every acoustic substep was absent from the integrated path.

Two real defects were found and repaired while closing it:

- **`gama` was hardcoded to 1.4.** `Riem_Solver_c`/`Riem_Solver3` derive
  `gama = 1/(1 - akap)` (`nh_core.F90:347`). 1.4 is only correct for the
  particular `akap` that happens to give it. Now derived from
  `NumericalConfig::akap`, with an `akap ∈ (0,1) && m_split ≥ 1` guard.
  *Numerical semantics changed:* YES, and deliberately — the previous value was
  wrong for any `akap ≠ 2/7`.
- **`m_split` could not express both solvers.** `dyn_core.F90:918` hands
  `Riem_Solver3` `flagstruct%m_split`; `:300`/`:583` hand `Riem_Solver_c`
  `max(1, m_split/2)`. The runtime previously hardcoded `vertical_ms = 1`, which
  silently forced a single vertical substep on the D-grid solver regardless of
  the namelist. `NumericalConfig::m_split` is now the single source, `0` is a
  rejected "not supplied" sentinel, and each call site derives its own value.
  *Numerical semantics changed:* YES, and deliberately.

## §5 `dgrid_height` vs `dgrid_height_transport` — ownership resolved

These are **not** two slots for one operation, which is what the §5 warning was
about. `nh_utils.F90`'s `update_dz_d` body is WP-C *sequencing* wrapped around
WP-B *horizontal transport*:

- `dgrid_height` → `vertical::launch_dgrid_height_stage` → `launch_update_dz_d`:
  requires `H_DSCALAR`, calls the transport service, then applies the `ws`
  computation (`nh_utils.F90:313`) and the top-down monotonicity clamp
  (`:315-319`).
- `dgrid_height_transport` → `dgrid::launch_dgrid_height_transport_cuda`: the
  WP-B `fv_tp_2d`/`ra_x`/`ra_y` transport body, including the authoritative
  area-weighted height update at `nh_utils.F90:300-301`.

A dead `update_dz_d_post_transport_kernel` in
`vertical/fv3_vertical_kernels.cu` was **removed**. It had zero callers, applied
its own `rdt`-scaled flux-divergence increment where the authoritative update is
an area-weighted *ratio* already applied inside the transport body (so any caller
would have double-counted), and wrote `delz`, which on the D-grid belongs to
`Riem_Solver3` (`nh_core.F90:181`) — `update_dz_d` writes only `zh` and `ws`
(`nh_utils.F90:310-320`). Its `ws`-then-clamp ordering and its `k = km-1..0`
clamp direction were correct, which is what made it look plausible enough to be
dangerous. *Numerical semantics changed:* NO (zero callers).

`launch_dgrid_height_v2` in the same file also has zero callers and must never be
wired; it is left in place with that note rather than removed, because unlike the
kernel above it is not actively misleading about ownership.

## §6 `dgrid_height_transport` is now reachable

`LaunchFn` takes no `NumericalDispatch`, so WP-C had no way to reach a WP-B
service. That is the whole reason this slot was IMPLEMENTED, WIRED, GPU EXECUTED
in isolation, and **unreachable** through the public entry point.

Fix: `ExecutionContext` gained a borrowed `const NumericalDispatch *dispatch`,
which `fv3_gpu_dyn_core_step` points at the table it is walking. `launch_update_dz_d`
calls `x.dispatch->dgrid_height_transport` and fails with
`FV3_GPU_INVALID_ARGUMENT` when `x.dispatch` is null (a launcher driven directly
by a focused test). No transport code was duplicated.

The production chain now runs end to end:

```
WP-B d_sw  ->  H_DSCALAR  ->  WP-C update_dz_d  ->  WP-B dgrid_height_transport
           ->  WP-C Riem_Solver3
```

`ExecutionContext::dispatch` and `DispatchParams::bdt` are additive tail members
on an internal C++ contract compiled into every consumer from one header, not a
cross-ABI record, so `internal_contract_version` remains 4.

---

## §7 The authoritative `n_split` loop

The prompt forbids looping the whole `stages[]` array `n_split` times unless
that matches the Fortran exactly. It does not: halo resolution is **interleaved**
between stages, one group is resolved only on the first substep, two are
re-resolved every substep, and the diffusive-heating tail is outside the loop
entirely.

So the `substep_stages[]` array was **deleted** and replaced with an explicit
script that mirrors `dyn_core.F90:399-1196` statement by statement:

```cpp
for (int32_t it = 1; it <= n_split; ++it) {
  if(!resolve(FV3_GPU_H_DWIND_IN)) return rc;               // :402  -> :504
  if(!resolve(FV3_GPU_H_W_IN)) return rc;                   // :427  -> :507
  if(!run(dispatch.cgrid_predictor)) return rc;             // :515-523
  if(config.nord>0 && !resolve(FV3_GPU_H_DIVGD)) return rc; // :528  -> :636
  if(it==1 && !resolve(FV3_GPU_H_GZ_FIRST)) return rc;      // :459  -> :552
  if(!run(dispatch.cgrid_vertical_solver)) return rc;       // :543-583
  if(!run(dispatch.cgrid_pressure)) return rc;              // :614
  if(!resolve(FV3_GPU_H_CWIND)) return rc;                  // :618  -> :640
  if(!run(dispatch.dgrid_dynamics)) return rc;              // :700-829
  if(!resolve(FV3_GPU_H_DSCALAR)) return rc;                // :844  -> :878
  if(!run(dispatch.dgrid_height)) return rc;                // :905-906
  if(!run(dispatch.vertical_solver)) return rc;             // :918-923
  if(!resolve(FV3_GPU_H_ZH)) return rc;                     // :931  -> :967
  if(!resolve(FV3_GPU_H_PKC2)) return rc;                   // :931  -> :967
  if(!run(dispatch.nh_pressure)) return rc;                 // :940-1019
}
// then, outside the loop: H_HEAT + post_loop_heating, :1231-1240
```

`dt = bdt/n_split` is computed **here and nowhere else** (`:296`). Milestone 3A.8
had a stopgap in `dgrid/wp_b_dsw_contract_cuda.cpp` that divided by `n_split`
itself, because this runtime was handing it the big timestep and walking the list
once; that stopgap is **removed**. Leaving both would have shortened the
integration by a factor of `n_split`.

### Why `it == 1`-only `H_GZ_FIRST` survives `n_split > 1`

This was the one question that could have invalidated the whole script. Answer,
established by exhaustive grep of every `mark_stale` call site in the tree:

| group | marked stale by | re-resolved by the script |
| --- | --- | --- |
| `H_CWIND` | `cgrid/fv3_cgrid.cu:122`, `cgrid/fv3_cgrid_predictor.cu:844` | yes, every substep, after `cgrid_pressure` |
| `H_DIVGD` | `cgrid/fv3_cgrid_predictor.cu:831` | yes, every substep, when `nord > 0` |
| `H_GZ_FIRST` | **nothing** | no — and correctly so |

`vertical/fv3_vertical_kernels.cu:632` documents `H_GZ_FIRST` as deliberately
never staled ("unchanged by this phase, which is why there is no mark_stale for
it"), matching `dyn_core.F90:459`'s own `if (it==1)` guard. Those are the *only*
three `mark_stale` calls in the tree, so no other group can silently go stale
mid-loop.

`mark_stale` semantics are fully preserved. Nothing was suppressed.

Verified on device with `n_split = 3`; see §12.

---

## §8 / §9 Single-rank halo readiness — and its honest limit

No MPI was implemented. `fv3_gpu_halo_callbacks` was examined and deliberately
not honoured: it is a **host-staged pack/unpack** design with no registration
API, no `ContextData` storage, and no consumer anywhere in the tree.
Implementing per-group packing — which fields, which strips, which rotation at a
panel seam — *is* a halo-exchange implementation, which this milestone defers.

`resolve_halo` (`runtime/fv3_gpu_runtime.cpp:174-194`) has exactly three
outcomes, and **only** three:

1. the topology makes the exchange **provably vacuous**
   (`halo_exchange_is_vacuous`: all four `neighbors[].rank < 0 && .tile < 0`,
   all four `corner_owner_rank < 0`, and not nested/regional/periodic) → resolve;
2. the **caller has declared** the group caller-managed
   (`mark_halo_caller_managed`) → resolve, and the caller owns the contents;
3. otherwise → `FV3_GPU_UNSUPPORTED_CONFIGURATION` with a message naming the
   group.

Nothing marks halos globally ready. `mark_halo_caller_managed` is a **declaration
of ownership, not a readiness flag**; it is sticky and lives in `ContextData`
rather than `DependencyState` precisely because a stage may mark a group stale
*mid*-substep (`cgrid_predictor` → `H_CWIND`/`H_DIVGD`) and the caller has no
opportunity to re-declare from outside. Staleness stays fully live; what is
delegated is only *who may clear it*.

### §9 Halo state transition table

`n_split = 3` shown; `it` is the substep index.

| stage | group required | state on entry | who makes it ready | made stale by | next consumer |
| --- | --- | --- | --- | --- | --- |
| `cgrid_predictor` | `H_DWIND_IN` | ready | `resolve` at loop top, every `it` | never | `cgrid_predictor` |
| `cgrid_predictor` | `H_W_IN` | ready | `resolve` at loop top, every `it` | never | `cgrid_predictor` |
| `cgrid_vertical_solver` | `H_GZ_FIRST` | ready | `resolve`, **`it == 1` only** | **never** — so it stays ready for `it = 2, 3` | `cgrid_vertical_solver` each `it` |
| `dgrid_dynamics` | `H_DIVGD` | ready | `resolve` after `cgrid_predictor`, when `nord > 0` | `cgrid_predictor` (`predictor.cu:831`) | `d_sw` |
| `dgrid_dynamics` | `H_CWIND` | ready | `resolve` after `cgrid_pressure` | `cgrid_predictor:844` **and** `cgrid_pressure` (`fv3_cgrid.cu:122`) | `d_sw` |
| `dgrid_height` | `H_DSCALAR` | ready | `resolve` after `d_sw` | never | `update_dz_d` |
| `nh_pressure` | `H_ZH` | ready | `resolve` after `vertical_solver` | never | `nh_p_grad` |
| `nh_pressure` | `H_PKC2` | ready | `resolve` after `vertical_solver` | never | `nh_p_grad` |
| `post_loop_heating` | `H_HEAT` | ready | `resolve` once, after the loop | never | `del2_cubed` |
| — | `H_Q_INLINE` | n/a | — | — | not live (`inline_q` fenced off) |
| — | `H_DWIND_NEXT` | n/a | — | — | not live in this subset |

Ordering is enforced structurally: each `resolve()` sits between the stage that
stales the group and the stage that requires it. No stage executes with a falsely
*ready* flag.

### The blocker, stated plainly

A whole-tile `grid_type = 0` rank genuinely has four cross-tile neighbours. There
is no single-rank topology that is simultaneously

- internally consistent for the stages that need panel geometry — WP-B `d_sw`
  tests index geometry (`b.is == 1`, `b.ie + 1 == npx`, …) exactly as
  `sw_core.F90` does, and WP-D's `a2b_ord4` guard requires
  `c.sw == (e.west && e.south)` from the neighbour-derived `physical_edges`
  (`operators/wp_d_contract.cpp:398-405`) — and
- provably vacuous for the halo exchange, which requires *no* neighbours.

Making the block interior instead requires same-tile neighbours with `rank ≥ 0`,
which is not vacuous either. So for the production configuration, outcome (1) is
unreachable and outcome (2) with a no-op implementation **is a falsely-ready halo
in substance**, however carefully it is labelled. §20's "no false halo-readiness
assumption" criterion is therefore not met, and this reports **GATE C1 BLOCKED**.

What *was* done instead of faking it: the integrated path was run with the
declaration made explicitly and loudly, the result is reported at the **GPU
EXECUTED** level only, and the harness prints, on every run:

```
NOTE: this harness performs NO halo exchange. The declaration
lets the integrated path execute; it does NOT make panel-seam
halo values correct. Gate C1 halo criterion: NOT met.
```

A **negative** test proves the refusal is real rather than decorative: Phase D's
last case installs the production config and the connected topology but does
*not* declare the groups, and the step is refused at the very first resolve
point with `FV3_GPU_UNSUPPORTED_CONFIGURATION` naming `H_DWIND_IN`.

---

## §10 initcheck classification — `AUTHORITATIVE DEAD READ — CLASSIFIED AND ACCEPTED`

**Both** signatures reduce to one mechanism, in the one shared subroutine body.
There are two *call sites*, not two defects — see §17 for how the second one was
found and why sampling had hidden it.

### The mechanism, in the authoritative Fortran

`del6_vt_flux` (`sw_core.F90:1544`) fills its `d2` **out**-intent scratch only over

```
i1 = is-1-nord .. i2 = ie+1+nord      (sw_core.F90:1585-1592)
j1 = js-1-nord .. j2 = je+1+nord
```

and then, when `nord > 0`, calls `copy_corners(d2, npx, npy, 1, …)`, which
unconditionally rotates all `ng = 3` halo rings (`tp_core.F90:257-286` for
`dir == 1`; `ng` is a parameter at `tools/fv_mp_mod.F90:108`). At `nord == 1` the
fill covers only `[is-2, ie+2] × [js-2, je+2]`, so the **Fortran itself** reads
the undefined outermost `is-3`/`ie+3`/`js-3`/`je+3` ring — and in both call sites
the actual argument is an unassigned local automatic, so the values are genuinely
indeterminate rather than stale-but-bounded.

At `nord == 2` the fill is `[is-3, ie+3]`, which covers the whole `ng = 3` halo, so
nothing fires. At `nord == 0` `copy_corners` is not called at all. Only `nord == 1`
levels are affected, which is exactly what the per-level counts show.

### Why the contamination is dead — verified index-by-index

Taking the SW corner, `dir == 1` (`tp_core.F90:259-265`), with `is = js = 1` and
`ng = 3`:

```
do j = 1-ng, 0        ! j = -2 .. 0   = js-3 .. js-1
   do i = 1-ng, 0     ! i = -2 .. 0   = is-3 .. is-1
      q(i,j) = q(j, 1-i)
```

- **Reads** are at first index `j ∈ [-2, 0]`. The fill region starts at `-1`, so a
  read is undefined **iff `j == -2`**, i.e. only on the outermost ring.
- **Writes** for `j == -2` land at `(i, -2)` — that is, back on the outermost ring,
  outside the fill region.
- Every write that lands *inside* the fill region needs `i ≥ -1` **and** `j ≥ -1`,
  and those writes read `q(j, 1-i)` with `j ∈ {-1, 0}`, which is **defined**.

So undefined reads occur only on the outermost ring and propagate only back onto
that same ring. The union of every subsequent `d2` read in the body is
`[is-2, ie+2] × [js-2, je+2]` (the `fx2`/`fy2` stencils at `:1594-1600` and the
`n = 1..nord` passes, which narrow rather than widen), provably disjoint from it.
Nothing reaches `fx2`, `fy2`, `zh`, `dw` or `heat_source`. The se/ne/nw corners
follow by the same construction.

### The two call sites

| # | site | `d2` actual | `nord` / `damp` source | which `k` fire |
| --- | --- | --- | --- | --- |
| 1 | `sw_core.F90:974`, inside `d_sw` **w-damping** | `wk`, unassigned local at `sw_core.F90:571` | `nord_w`, `damp4` | `k ∈ {1, 5, 9}` — `nord_w[4] = {0,1,0,2}` with `s = k % 4`, and `damp_w[1] = 0.06 > 1e-5` |
| 2 | `nh_utils.F90:288`, inside `update_dz_d` **height damping** | `wk2`, unassigned local at `nh_utils.F90:238`, never written anywhere in the subroutine | `ndif = nord_v`, `damp = damp_vt` (`dyn_core.F90:905-906`) | levels with `nord_v(k) == 1` and `damp_vt(k) > 1e-5` (`nh_utils.F90:280`) |

Site 2 is guarded by `if ( damp(k) > 1.E-5 )` at `nh_utils.F90:280`; the `else`
branch transports `zh` directly with no `del6_vt_flux` and no scratch, so it
cannot fire.

The sibling **vorticity** `del6_vt_flux` at `sw_core.F90:1478` does **not** fire in
either precision, because its `d2` actual is `vort`, fully written at
`:1453-1457`. That asymmetry — same body, same `nord`, no finding — is independent
confirmation that the mechanism is the unassigned actual and not the kernel.

### Verdict

`AUTHORITATIVE DEAD READ — CLASSIFIED AND ACCEPTED`, for both call sites.

Two value-neutral hardenings exist (widen the `d2` fill to `ng` rings; pass a ring
count to `copy_corners`) but both depart from authoritative loop bounds, so
neither converts this to ELIMINATED WITHOUT NUMERICAL CHANGE, and the prompt
forbids altering validated 3A.4 semantics to obtain a zero count.

**Caveat to carry forward:** re-check this ring if any future change extends an
`fx2`/`fy2` stencil to `is-3`/`ie+3`/`js-3`/`je+3`, or raises `nord` handling in
either call site.

---

## §11 The real production stage graph

```
fv3_gpu_dyn_core_step
  |
  +-- fence_production_config            (§13, before any mutation)
  +-- order transfer_stream -> compute_stream   (H2D visible to kernels)
  +-- zero_accumulators                  mfx,mfy,cx,cy [+heat_source if d_con>1e-5]
  |
  +-- do it = 1, n_split
  |     resolve H_DWIND_IN, H_W_IN
  |     cgrid_predictor          d2a2c_vect + divergence_corner + c_sw
  |     resolve H_DIVGD          (nord>0)      [staled by cgrid_predictor]
  |     resolve H_GZ_FIRST       (it==1 only)  [never staled]
  |     cgrid_vertical_solver    update_dz_c + Riem_Solver_c (ms=max(1,m_split/2))
  |     cgrid_pressure           p_grad_c                    [stales H_CWIND]
  |     resolve H_CWIND
  |     dgrid_dynamics           per-level damping + d_sw
  |     resolve H_DSCALAR
  |     dgrid_height ------------> update_dz_d (WP-C)
  |     |                            \--> dgrid_height_transport (WP-B)
  |     vertical_solver          Riem_Solver3 (ms=m_split)
  |     resolve H_ZH, H_PKC2
  |     nh_pressure              nh_p_grad (a2b_ord4, pk3_halo, pln_halo, pe_halo)
  |   enddo
  |
  +-- resolve H_HEAT             (only when n_con != 0 && d_con > 1e-5)
  +-- post_loop_heating          nf_ke x del2_cubed + heating, bounded by bdt*delt_max
```

`corner_edge_service` appears nowhere in this graph. That is the §19 claim, and
it is load-bearing rather than an omission: see §19.

---

## §12 Integrated execution through the public entry point

Everything below ran through `fv3_gpu_dyn_core_step`. No claim here comes from a
focused launcher test.

Evidence mechanism: each of the nine slots is wrapped in a **counting probe**
that calls straight through to the real production launcher. The probes are
installed by *copying* the production table, so a slot the production builder
leaves null stays null and its expected count is zero rather than being silently
satisfied. This matters because the runtime **skips** a null slot by design —
"the step returned SUCCESS" is compatible with a stage never having been entered.

R8, A100 `warpa003`, `n_split = 3`:

```
  Per-stage reachability (n_split=3):
    cgrid_predictor (WP-A c_sw/d2a2c_vect)                   3/3 REACHED
    cgrid_vertical_solver (WP-C update_dz_c + Riem_Solver_C) 3/3 REACHED
    cgrid_pressure (WP-A p_grad_c)                           3/3 REACHED
    dgrid_dynamics (WP-B d_sw)                               3/3 REACHED
    dgrid_height (WP-C update_dz_d sequencing)               3/3 REACHED
    vertical_solver (WP-C Riem_Solver3)                      3/3 REACHED
    nh_pressure (WP-D nh_p_grad)                             3/3 REACHED
    post_loop_heating (WP-D del2_cubed + heating)            1/1 REACHED
    dgrid_height_transport (WP-B, called BY WP-C)            3/3 REACHED
```

Expected counts follow the Fortran's structure, not convenience: the eight
in-loop stages are entered exactly `n_split` times; `post_loop_heating` is the
tail after `:1196`, so exactly once regardless of `n_split`.

`dgrid_height_transport` at 3/3 is the direct §6 proof — it is on no stage list
and is entered *only* because WP-C's `update_dz_d` calls it through
`ExecutionContext::dispatch`.

Output: all 23 checked fields finite; `delp` and `u` both changed from their H2D
values; the four capacitors were zeroed (the deliberately nonzero 11.0/13.0/
0.17/0.19 seeds are provably gone) and then accumulated to nonzero.

Acoustic-loop probe (§7 evidence — if the runtime walked its stage list once
regardless of `n_split`, these would be identical, since nothing else differs
between the two runs):

```
    sum|mfx| at n_split=1: 2.284317559e+14
    sum|mfx| at n_split=3: 1.810186358e+14
    differ (so the loop is a real loop, not one pass): YES
```

---

## §13 Production configuration fence

A new `fence_production_config` in `runtime/fv3_gpu_runtime.cpp` runs **before**
`zero_accumulators` and before the first launch.

This was not bookkeeping. Every launcher already fences its own configuration,
but a per-stage fence fires at the stage that *cares*, and the stages that do not
care have already run. Concretely: with `grid_type = 4` the C-grid predictor takes
its `grid_type >= 3` **branch** rather than refusing
(`cgrid/fv3_cgrid_csw.cu:151`), so on the pre-3A.9 path the whole C-grid half of
the substep executed and wrote `uc, vc, delpc, ptc, divgd` to device memory
before WP-B returned `UNSUPPORTED_CONFIGURATION`. The call failed, but the state
was already torn — the caller could neither retry nor trust a D2H. That violates
`project_architecture.md`'s "unsupported configurations should fail before
mutating model state".

Fenced: `grid_type ≠ 0`; `hydrostatic`; `a_imp ≤ 0.999`; any `|hord_*| ≠ 6`;
`hord_dp > 0` (the sign selects a different `fv_tp_2d` variant, so `+6` is a
different scheme); `use_cond`/`moist_cappa`; `inline_q`; `do_f3d`/`do_skeb`/
`rot3`; `beta ≠ 0`; `m_split < 1`; `dz_min ≤ 0`; nested/regional; not
`global_cubed_sphere`.

Deliberately **not** fenced, with reasons in the source comment:

- **`vtdm4 > 0`** — §13 lists it, but `NumericalConfig` has no `vtdm4` member and
  should not gain one. `vtdm4` reaches the kernels already resolved into the
  per-level `damp_vt` (`FieldId::damping_v`) and `nord_v`
  (`FieldId::damping_ndif`) arrays, which is how `dyn_core.F90:699-765` itself
  hands it to `d_sw`. A scalar check would either duplicate that or contradict
  the sponge levels, where the authoritative code forces the vorticity-damping
  controls to zero.
- **`npx`/`npy == 0`** — a missing *input*, not an unsupported configuration. The
  two consumers that need panel geometry already refuse zero at their binders
  with `FV3_GPU_INVALID_ARGUMENT`, and minimal-dispatch drivers that need no
  panel semantics are legitimately drivable without it. Their refusals are not
  pre-mutation, which is a **residual gap** (see §24) rather than something to
  paper over by imposing a false requirement on every driver.
- tuning parameters (`nord`, `dddmp`, `da_min`, `p_fac`, `delt_max`) — these do
  not change which code path runs and are range-checked by their consumers.

The fence does **not** set `ContextPhase::failed`. A configuration refusal
mutated nothing, so the context remains usable; marking it failed would conflate
"you asked for something unimplemented" with "device state is now
untrustworthy", and the entire point of fencing before mutation is that those
differ.

Verified on device — all six Phase D cases refused with
`FV3_GPU_UNSUPPORTED_CONFIGURATION`, and in the five config cases an explicit D2H
after the refusal confirmed `delp` and `u` byte-identical to their H2D values.

---

## §14 Persistent device residency

No stage performs a host/device transfer. The only transfers in the tree are
`fv3_gpu_initialize_state`, `fv3_gpu_sync_to_device` and `fv3_gpu_sync_to_host` —
all caller-invoked, all at explicit boundaries.

Asserted on device rather than assumed: the harness snapshots the registered
**host** buffers after H2D, runs the step, and verifies they are byte-identical
*before* the explicit `sync_to_host`, then verifies they differ after it.

```
  Host buffers unchanged before explicit sync_to_host: YES (no implicit D2H)
```

### A real synchronization defect found and fixed here

Writing that assertion exposed a genuine runtime bug. Both streams are created
`cudaStreamNonBlocking`, so **nothing** ordered them against each other:
`fv3_gpu_sync_to_host` enqueued its D2H on the transfer stream with no dependency
on the acoustic-loop kernels on the compute stream, and `fv3_gpu_event_wait`
synchronized only the transfer stream. A caller doing
`step -> sync_to_host -> event_wait` — the exact sequence
`project_architecture.md` prescribes for an explicit D2H boundary — could read
pre-step values.

Not theoretical. With `n_split = 3` the host spent long enough enqueueing kernels
that the copies landed after them and the data looked right; with `n_split = 1`
the D2H won the race and every field came back byte-identical to the H2D input.
The `n_split = 3` result was luck: its `sum|mfx|` was `1.139e14` before the fix
and `1.810e14` after, i.e. the pre-fix run was reading partially-computed data.

Fix (`runtime/fv3_gpu_runtime.cpp`): two context-owned `cudaEvent_t`s
(`cudaEventDisableTiming`) and an `order_after(signaler, waiter, ev)` helper.

- `fv3_gpu_sync_to_host` orders the transfer stream after the compute stream;
- `fv3_gpu_dyn_core_step` orders the compute stream after the transfer stream, so
  the first substep cannot read un-copied device memory (today's harnesses
  host-block on the H2D event first, which hid it; a caller following the
  residency model has no reason to);
- `fv3_gpu_event_wait` now synchronizes **both** streams — waiting on a step
  event previously returned immediately while the acoustic loop was still
  running.

These are `cudaStreamWaitEvent` dependencies, not host synchronization. Using
`cudaStreamSynchronize` there instead would have been exactly the hidden host sync
`project_architecture.md` forbids. *Numerical semantics changed:* NO — the
arithmetic is unchanged; what changed is that the host now observes it correctly.

---

## §15 Regression suites

Run on A100 `warpa003` through Slurm.

| build | result |
| --- | --- |
| `build_cuda_r8` | **17/17 PASS**, 0 failed |
| `build_cuda_r4` | **17/17 PASS**, 0 failed |

No regression. Both precisions built clean with no new warnings. Re-run after the
final rebuild, so these numbers refer to the sources actually committed rather
than to an intermediate binary.

## §16 The permanent integrated test

`cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp`, registered as the
`fv3_dispatch_smoke` CTest target in both precisions. Four phases:

| phase | what it establishes |
| --- | --- |
| A | minimal dispatch on a **vacuous** topology — the one place halo readiness is established with no caller declaration at all |
| B | the full production table succeeds |
| C | **the Gate C1 evidence run** — nine probed slots, `n_split = 3`, reachability counts, residency, finiteness, capacitor zero-then-accumulate, plus an `n_split = 1` control for the substep-loop probe |
| D | the §13 fence and the §8 halo refusal — six cases, each of which must be *refused* before mutation |

Harness repairs required to get there, each a real contract gap rather than test
scaffolding:

- `m_split = 2` (was unset; `0` is now correctly rejected, and `2` gives
  `Riem_Solver3 ms = 2` and `Riem_Solver_C ms = 1`, exercising the
  two-different-values path rather than the degenerate case);
- `dz_min = 2.0` (nothing rejected zero, and the SIM1 column solve divides by
  `dz` → silent Inf/NaN from a valid-looking config);
- six unregistered fields: `omga` (zero would remove `w3` from the
  `Riem_Solver_C` RHS entirely), `f_c` (deliberately offset from `f0` so a mix-up
  shows), `cos_sg`, `dp_ref` (the `update_dz_c` dp0-ratio weights), `ws3` (seeded
  zero **because it is an output** — a nonzero seed would mask a missing write),
  `phis` (`= grav*zs`, required for the gz-in-metres / hs-in-geopotential
  closure);
- `production_config()` split out of `install_config()` so the fence phase
  perturbs exactly one field of the real production config.

## §17 compute-sanitizer

Tool: `/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer`,
run on the full four-phase integrated harness (not a focused launcher).

| tool | R8 | R4 |
| --- | --- | --- |
| memcheck | **0 errors** | **0 errors** |
| racecheck | **0 hazards** (0 errors, 0 warnings) | **0 hazards** (0 errors, 0 warnings) |
| initcheck | 672 errors — **classified, not clean** | 672 errors — **classified, not clean** |

**initcheck is NOT clean and is not reported as clean.**

### The census was run exhaustively, and it corrected a wrong inference

The first initcheck pass truncated at the default print limit — `572 errors were
not printed` — and all 100 printed findings shared the `d_sw` signature. That is
*not* sufficient to characterise 672 findings, so the run was repeated with
`--print-limit 5000` and every finding enumerated. The full census shows **two**
call paths, not one:

| findings | device kernel | host frames |
| --- | --- | --- |
| 432 | `copy_corners_kernel` | `launch_copy_corners_cuda` ← `launch_del6_vt_flux_cuda` ← `launch_d_sw_cuda` ← `launch_dgrid_dynamics_cuda` |
| 240 | `copy_corners_kernel` | `launch_copy_corners_cuda` ← `launch_del6_vt_flux_cuda` ← `launch_dgrid_height_transport_cuda` ← `launch_update_dz_d` ← `launch_dgrid_height_stage` |

Identical in R8 and R4: 672 total, 432/240 split, all `Uninitialized __global__
memory read of size 8 bytes`, one device kernel, zero other error kinds. That the
counts are **bit-identical across precisions** is itself corroboration — a purely
geometric ring read has a precision-independent count, which a genuine
data-dependent uninitialized read would not.

Had this been left at the 100-sample, the handoff would have claimed a single
signature and been wrong. The second path is `nh_utils.F90:288` and it appears
**for the first time in this milestone**, because §6 is what made
`dgrid_height_transport` reachable at all. Its classification is derived in §10
from the Fortran, not assumed from resemblance to the first.

Both paths are the same `del6_vt_flux` body at `nord == 1` with an unassigned `d2`
actual, so both are `AUTHORITATIVE DEAD READ — CLASSIFIED AND ACCEPTED`. The
total is higher than 3A.8's because the harness now performs several full
integrated runs (Phases B, C ×2, D) rather than one.

Logs: `initcheck_full_{r8,r4}.log`, `racecheck_{r8,r4}.log`,
`memcheck_{r8,r4}.log` under `san_3a9/` (not committed, per §23).

---

## §18 Source-level sequence reconciliation

| # | Fortran | status | note |
| --- | --- | --- | --- |
| 0a–0d | `:296-379` | MATCHES FORTRAN | including the four-capacitor set and the `d_con > 1e-5` conditional on `heat_source` |
| 3 | `:515-523` | MATCHES WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE | `csw_fluxes_level` split into x-flux and y-flux phases; the y phase reads nothing the x phase writes, so ordering and arithmetic are unchanged |
| 3 | `:515-523` | MATCHES FORTRAN | `csw_circulation` coverage repaired in a prior 3A.9 session |
| 6 | `:543-583` | MATCHES FORTRAN | after the `gama = 1/(1-akap)` repair and the `ms = max(1, m_split/2)` split; `update_dz_c` finalize triple-inversion repaired |
| 6/12 | `:300` / `:918` | MATCHES FORTRAN | the two solvers deliberately receive different `ms`; `DispatchParams::vertical_ms` carries only the `:918` value |
| 9 | `:700-829` | MATCHES FORTRAN | after removing the 3A.8 `dt/n_split` stopgap; per-level `nord_k`/`nord_w`/`d_con_k` kept per level, not collapsed to scalars |
| 11 | `:905-906` / `nh_utils.F90:300-321` | MATCHES FORTRAN | ownership split resolved; the double-counting dead kernel removed |
| 14 | `:940-1019` | MATCHES FORTRAN | `beta = 0` fenced, so `beta_d` is 0 on every substep and the `it == 1` special case is a no-op |
| 14 | `a2b_edge.F90` panel-seam detection | **DISCREPANCY** (recorded, not repaired) | WP-D's `physical_edges` (`operators/wp_d_contract.cpp:398-405`) derives seams from the cross-tile **neighbour list**; authoritative `a2b_ord4` (`model/a2b_edge.F90`) uses **index geometry** (`is == 1`, `ie+1 == npx`, `js == 1`) plus the gridstruct corner flags. They agree for any consistently-described topology, so changing it was judged risk without gate benefit. It is a real divergence in *what is consulted*, and it is why an all-corners/no-edges topology is rejected rather than silently mis-stenciled. |
| 15 | `:1231-1240` | MATCHES FORTRAN | after the `del2_cubed` `cd = 0.20*da_min` repair and taking `bdt` from `DispatchParams::bdt` rather than reconstructing `dt*n_split` |
| — | `nest_halo_nh`, `Riem_Solver3test`, `imp_diff_w`, `d3_damp` | NOT LIVE IN SUPPORTED CONFIG | dead code; deliberately absent |
| — | `MOIST_CAPPA` / `USE_COND` | **NOT LIVE IN SUPPORTED CONFIG — documented scope divergence** | both are unconditionally **on** in the real GEOS FV3 build, but the 3A.8 `d_sw` driver rejects `use_cond` and §13's fence does not list them as required. This subset is narrower than production GEOS and must be widened before production integration. |

## §19 Production dispatch final audit

| slot | wired to | reached via public entry | probe count at `n_split=3` | production-required |
| --- | --- | --- | --- | --- |
| `cgrid_predictor` | `cgrid::launch_cgrid_predictor_v3` | yes | 3 | yes |
| `cgrid_vertical_solver` | `vertical::launch_cgrid_vertical_solver` | yes | 3 | yes |
| `cgrid_pressure` | `cgrid::launch_cgrid_pressure_v3` | yes | 3 | yes |
| `dgrid_dynamics` | `dgrid::launch_dgrid_dynamics_cuda` | yes | 3 | yes |
| `dgrid_height` | `vertical::launch_dgrid_height_stage` | yes | 3 | yes |
| `dgrid_height_transport` | `dgrid::launch_dgrid_height_transport_cuda` | yes — **from inside WP-C**, not from a stage list | 3 | yes |
| `vertical_solver` | `vertical::launch_vertical_solver` | yes | 3 | yes |
| `nh_pressure` | `operators::launch_nh_pressure_cuda` | yes | 3 | yes |
| `post_loop_heating` | `operators::launch_post_loop_heating_cuda` | yes | 1 (post-loop) | yes |
| `corner_edge_service` | `corner_service_not_implemented` (fail-stub) | **never requested** | 0 | **no** |

**No production-required slot is NULL or a fail-stub.**

`corner_edge_service` justification, since a fail-stub in the table would
otherwise block the gate: the `grid_type = 0` production path implements its
corner semantics *locally* and never requests the generic service.
`launch_update_dz_c_interface`'s `launch_fill_4corners` uses `CornerFillArgs`
directly and does **not** go through `ExecutionContext::corner_service`; `d_sw`
likewise calls `launch_copy_corners_cuda` directly. `post_loop_heating` is the
one stage that *would* request it, and only when `nt > 0`; with `nf_ke = 1`,
`passes = 1` and `nt = passes - 1 = 0`. The evidence is that the whole
integrated path succeeds while the slot remains a fail-stub — if any stage
requested it, the step would fail with `FV3_GPU_UNSUPPORTED_CONFIGURATION`.

This is status **D (unnecessary)** in §2's taxonomy, not "wired to make a
nullptr disappear".

---

## §20 Gate C1 assessment

| # | requirement | status |
| --- | --- | --- |
| 1 | authoritative sequence reconstructed from source | PASS (§1) |
| 2 | every dispatch slot audited | PASS (§2) |
| 3 | `cgrid_predictor` implemented + wired + GPU executed | PASS |
| 4 | `cgrid_vertical_solver` resolved and closed | PASS |
| 5 | `dgrid_height` ownership resolved | PASS |
| 6 | `dgrid_height_transport` reachable through the production path | PASS |
| 7 | authoritative `n_split` loop, tested with `n_split > 1` | PASS (§7, §12) |
| 8 | single-rank halo readiness without false readiness | **FAIL — the blocker** (§8) |
| 9 | halo state transition table, no falsely-ready halo | PASS as a model (§9); vacated in practice by #8 |
| 10 | initcheck finding classified | PASS — `AUTHORITATIVE DEAD READ — CLASSIFIED AND ACCEPTED` (§10) |
| 11 | real production stage graph | PASS (§11) |
| 12 | full execution through `fv3_gpu_dyn_core_step` | PASS (§12) |
| 13 | production configuration fence | PASS (§13) |
| 14 | persistent device residency | PASS (§14) |
| 15 | full R8 and R4 suites, no regression | PASS — 17/17 both (§15) |
| 16 | permanent integrated Gate C1 test | PASS (§16) |
| 17 | memcheck / initcheck / racecheck | memcheck PASS (0/0); racecheck PASS (0/0); initcheck **classified, not clean** — 672 findings both precisions, two call sites, both `AUTHORITATIVE DEAD READ` (§10, §17) |

**Verdict: GATE C1 BLOCKED**, on requirement 8 alone.

## §21 Prompt 6 has not been started

Numerical validation against authoritative runtime Fortran savepoints:
**NOT RUN**. Nothing in this milestone compares CUDA output to Fortran output. No
host transcription comparison anywhere in this work is a numerical validation,
and none is described as one. Every claim above is at the IMPLEMENTED, COMPILED
or GPU EXECUTED level, explicitly labelled.

## §24 Structured completion report

| field | value |
| --- | --- |
| milestone | 3A.9 — Full dyn_core Gate C1 closure |
| branch | `cuda/gpu-bringup` |
| parent commit | `7ddb6f9` |
| authoritative reference | `GFDL_atmos_cubed_sphere` `geos/v3.0.0` / `ca30e26` |
| authoritative sequence reconstructed | YES — from `dyn_core.F90`, not from the existing `stages[]` |
| dispatch slots total | 9 `LaunchFn` + `corner_edge_service` |
| slots NULL entering the milestone | 3 (`cgrid_predictor`, `cgrid_vertical_solver`, `dgrid_height`) |
| slots NULL now | 0 |
| production-required fail-stubs now | 0 |
| `corner_edge_service` | fail-stub, classified **NOT REQUIRED** for `grid_type = 0`, never requested |
| `cgrid_predictor` | IMPLEMENTED + WIRED + GPU EXECUTED |
| `cgrid_vertical_solver` | IMPLEMENTED + WIRED + GPU EXECUTED; genuinely distinct, not an alias |
| `dgrid_height` / `dgrid_height_transport` | ownership resolved and documented; both GPU EXECUTED |
| `dgrid_height_transport` reachable via public entry | YES — 3/3 at `n_split = 3`, entered only from WP-C |
| `n_split` loop | authoritative interleaved script; `stages[]` array deleted |
| `n_split > 1` executed | YES (`n_split = 3`), with an `n_split = 1` control that differs |
| stages entered per substep | 8; `post_loop_heating` once, outside the loop |
| public entry used | `fv3_gpu_dyn_core_step` only |
| persistent device residency | PASS — asserted, no implicit D2H |
| cross-stream ordering defect | FOUND and FIXED (two events; `event_wait` now covers both streams) |
| production config fence | PASS — 6/6 refusals, pre-mutation, `UNSUPPORTED_CONFIGURATION` |
| halo model | 3-outcome `resolve_halo`; `mark_stale` fully preserved; nothing marked globally ready |
| halo transport implemented | **NO** — deferred per §8 |
| R8 CTest | 17/17 PASS |
| R4 CTest | 17/17 PASS |
| memcheck R8 / R4 | 0 / 0 errors |
| racecheck R8 / R4 | 0 / 0 hazards |
| initcheck R8 / R4 | 672 / 672 — two call sites, both `AUTHORITATIVE DEAD READ — CLASSIFIED AND ACCEPTED` |
| initcheck reported clean | **NO** |
| IMPLEMENTED | yes, all production-required stages |
| COMPILED | yes, R8 and R4, no new warnings |
| GPU EXECUTED | yes, A100 `warpa003`, via the public entry, per-stage probe counts |
| NUMERICALLY VALIDATED | **NO — NOT RUN** |
| Prompt 6 started | NO |
| pushed / merged | NO |
| verdict | **GATE C1 BLOCKED** — one blocker: no halo transport (§8) |

## Residual items for the next milestone

Ordered by how much they matter.

1. **Halo transport.** The blocker. Needs a real design decision: implement the
   `fv3_gpu_halo_callbacks` host-staged pack/unpack (per-group field lists,
   strips, and panel-seam rotation), or move straight to a device-side exchange.
   Until then no `grid_type = 0` rank can have honest halo readiness.
2. **Widen the subset to `MOIST_CAPPA` / `USE_COND`.** Both are unconditionally
   on in the real GEOS FV3 build; the current subset rejects them. This must
   close before production GEOS integration.
3. **`npx`/`npy == 0` pre-mutation gap** (§13). A dispatch table containing a
   panel-geometry consumer plus an `npx == 0` topology can still tear device
   state, because those binders refuse after earlier stages have run.
4. **WP-D `physical_edges` neighbour-vs-index-geometry divergence** (§18).
5. **One-time inits still inside the loop:** the `d4dmp` block
   (`dyn_core.F90:678-690`, `save` flag at `:150`) and Ray_fast
   (`:2517-2545`, flag at `:154`) are guarded by `save`d first-call flags in the
   Fortran and are not yet lifted out of the CUDA loop. Idempotent, so not a
   correctness defect today.
6. **Config values to confirm against the GEOS namelist:** `do_vort_damp`
   default `.false.`; `d_con` default `0.`; that `n_sponge` per-level damping is
   not hoisted out of the `k` loop; the `d_ext` guard inconsistency at `:777`/
   `:813` vs `:851`; `z_rat` uninitialized when `do_f3d = false`.
7. **Whether the 3A.8 `d_sw` driver performs dyn_core's `:819-826` 3-D
   `heat_source`/`diss_est` accumulation** or only `d_sw`'s own `:965` 2-D reset.
8. Minor contract cleanups: `DivergenceCornerArgs` carries no `nested` flag;
   `CSwArgs::nord` is a dead member; `Bounds` has no halo member.
9. **Methodological note, worth keeping.** The initcheck signature census in §17
   was initially drawn from the default 100-error print limit and was **wrong** —
   it missed the entire `update_dz_d` call path, which is 240 of 672 findings and
   which this milestone itself newly exposed. Always raise `--print-limit` above
   the reported error count and enumerate, rather than characterising a
   distribution from its truncated head.

## §23 Git discipline

Modified in this milestone:

```
cuda/dyn_core/include/fv3_gpu_context.hpp        two ordering events
cuda/dyn_core/include/fv3_gpu_internal.hpp       ExecutionContext::dispatch, DispatchParams::bdt,
                                                 NumericalConfig::m_split/scale_m,
                                                 mark_halo_caller_managed
cuda/dyn_core/runtime/fv3_gpu_runtime.cpp        n_split script, resolve_halo, fence_production_config,
                                                 order_after + the three stream boundaries
cuda/dyn_core/runtime/fv3_gpu_production_dispatch.cu   nine slots wired
cuda/dyn_core/vertical/fv3_vertical_kernels.cu   gama repair, dgrid_height_stage adapter,
                                                 dead kernel removed
cuda/dyn_core/vertical/fv3_riem_solver_c.cuh     stale vertical_ms comments corrected
cuda/dyn_core/cgrid/…, dgrid/…                   prior-session repairs in this milestone
cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp   Phases C and D
docs/agent_handoff/milestone_3a9_gate_c1_closure.md       this file
```

No authoritative Fortran was modified. `geos-gtfv3` was not touched. No build
directory, object file, executable, Slurm log or sanitizer log is staged. Nothing
was pushed.

---

**GATE C1 BLOCKED**
