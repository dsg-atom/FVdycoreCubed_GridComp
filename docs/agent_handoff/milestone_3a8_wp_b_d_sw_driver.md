# Milestone 3A.8 — WP-B.5 Full Authoritative `d_sw` Driver and Production Dispatch Activation

## 0. Identification

| Item | Value |
|---|---|
| Branch | `cuda/gpu-bringup` |
| Starting commit | `295fb029d231073d5a9b51d8b44c3ac8f03b1d9c` (*Record milestone 3A.7 commit SHA in handoff*) |
| 3A.7 implementation commit | `1fe108c` (*Implement authoritative WP-B divergence damping / corner slice*) |
| Final commit | `f5a2f18adea38c64b750b9cac2a598ca5f6c0081` (*Integrate authoritative WP-B d_sw CUDA driver*) |
| Authoritative Fortran revision | GEOS-ESM/GFDL_atmos_cubed_sphere, tag `geos/v3.0.0`, SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45` |
| Local authoritative clone | `/discover/nobackup/projects/imvi-nse/jli30/fv3_reference/GFDL_atmos_cubed_sphere` |
| Primary Fortran sources | `model/sw_core.F90` (`d_sw` :520–1540, `del6_vt_flux` :1544–…), `model/dyn_core.F90` (:699–802), `model/tp_core.F90` (`fv_tp_2d` :109–243, `copy_corners` :247–322) |
| Platform | NASA NCCS Discover, `gpu_a100` / `--constraint=rome`, NVIDIA A100-SXM4-40GB (`warpa003`), driver 535.104.12, CUDA 12.5.82, GCC 12.3.0, OpenMPI 4.1.6 |

Authoritative Fortran: **not modified**. `geos-gtfv3`: **not modified**.

---

## 1. Files

New:

| File | Lines | Role |
|---|---:|---|
| `cuda/dyn_core/dgrid/wp_b_dsw_cuda.cuh` | 633 | The authoritative decomposition (§1 stage table S1–S38, §2 driver kernels K1–K12, §3 anti-dependence table, §4 per-level control, §5 flux capacitors, §6 fencing, §7 `corner_edge_service` disposition) plus the `DSwOptions` / `DSwLevels` / `DSwGeometry` / `DSwState` / `WpBDSwArena` contracts and the K1–K12 + driver declarations |
| `cuda/dyn_core/dgrid/wp_b_dsw_cuda.cu` | 1272 | K1–K12 kernels, the per-level coefficient derivation kernel, and `launch_d_sw_cuda` |
| `cuda/dyn_core/dgrid/wp_b_dsw_contract_cuda.cpp` | 621 | `bind_wp_b_d_sw` (≈70 registered views + arena carving) and `launch_dgrid_dynamics_cuda` |
| `cuda/dyn_core/dgrid/wp_b_contract_views.hpp` | 110 | View-wrapping / coverage-checking helpers shared by both WP-B binders |
| `cuda/dyn_core/tests/dgrid/test_wp_b5_dsw_cuda.cpp` | 2547 | Permanent focused test, Parts A–M |

Modified: `CMakeLists.txt`, `dgrid/wp_b_contract_cuda.{hpp,cpp}`, `dgrid/wp_b_cuda.{cuh,cu}`, `include/fv3_gpu_internal.hpp`, `runtime/fv3_gpu_contract.cpp`, `runtime/fv3_gpu_production_dispatch.cu`, `runtime/fv3_gpu_runtime.cpp`, `tests/integration/test_dispatch_smoke.cpp`.

---

## 2. Exact authoritative `d_sw` operation sequence

Reconstructed line-by-line from `sw_core.F90` at the SHA above for the production
first-run configuration (`grid_type == 0`, non-nested, `hydrostatic == false`,
`inline_q == false`, `do_f3d == false`, `do_skeb == false`, `SW_DYNAMICS` and
`USE_COND` undefined, `a_imp == 1.0`, `hord_mt = hord_vt = hord_tm = 6`,
`hord_dp = -6`). The full table with inputs, outputs, in-place mutations,
workspace and per-level controls is `§1` of `dgrid/wp_b_dsw_cuda.cuh`; the
ordered summary is:

| S | `sw_core.F90` | Operation | CUDA owner | Classification |
|---:|---|---|---|---|
| 1 | 672–707 | contravariant `ut`/`vt` interior | K1 | driver-owned state update |
| 2 | 709–779 | `ut`/`vt` tile-edge overrides (8 ordered) | K2 (8 launches) | driver-owned state update |
| 3 | 791–863 | `ut`/`vt` 2×2 corner solves (4 ordered) | K3 (4 launches) | driver-owned state update |
| 4 | 882–892 | `xfx_adv = dt*ut`, `yfx_adv = dt*vt` | K4a | driver-owned state update |
| 5 | 898–921 | upwind `crx_adv`/`cry_adv` finalize + `xfx`/`yfx` rescale | K4b | driver-owned state update |
| 6 | 927–936 | `ra_x` / `ra_y` | 3A.4 `launch_ra_x/y_cuda` | already-implemented service |
| 7 | 939–940 | `fv_tp_2d(delp, hord_dp, nord_v, damp_v)` → `fx,fy` | 3A.5 scalar transport | already-implemented service |
| 8 | 943–960 | flux capacitors `cx,cy,xflux,yflux` | K5 | driver-owned state update |
| 9 | 963–968 | `heat_source = 0`, `diss_est = 0` | K6 | driver-owned state update |
| 10 | 970–985 | `damp_w`: `del6_vt_flux(w)` → `dw`, heat-source seed | 3A.4 `del6_vt_flux` + K7 | service + driver-owned |
| 11 | 986–987 | `fv_tp_2d(w, hord_vt, mfx=fx, mfy=fy)` → `gx,gy` | 3A.5 scalar transport | already-implemented service |
| 12 | 988–992 | `w = OLD delp * w + div(gx,gy)` | K8 | driver-owned state update |
| 13 | 993–1001 | `USE_COND` `q_con` | — | **deferred / not live** |
| 14 | 1004–1006 | `fv_tp_2d(pt, hord_tm, mfx=fx, mfy=fy, mass=delp, nord_t, damp_t)` → `gx,gy` | 3A.5 scalar transport | already-implemented service |
| 15 | 1034–1047 | `pt = pt*delp + div(gx,gy)`; `delp += div(fx,fy)`; `pt /= delp` | K9 | driver-owned state update |
| 16 | 1050–1054 | `dpx += div(fx,fy)` | K10 | driver-owned state update |
| 17 | 1065–1074 | `dt5`/`dt4`, `is2/ie1/js2/je1` | host arithmetic | driver-owned control flow |
| 18 | 1077–1120 | `vb` construction | 3A.6 `bgrid_vb` | already-implemented service |
| 19 | 1122–1123 | `ytp_v(vb)` → `ub` | 3A.6 `ytp_v` | already-implemented service |
| 20 | 1125–1129 | `ke = vb*ub` | 3A.6 `ke_product` | already-implemented service |
| 21 | 1131–1177 | `ub` construction | 3A.6 `bgrid_ub` | already-implemented service |
| 22 | 1179–1180 | `xtp_u(ub)` → `vb` | 3A.6 `xtp_u` | already-implemented service |
| 23 | 1182–1186 | `ke = 0.5*(ke + ub*vb)` | 3A.6 `ke_average` | already-implemented service |
| 24 | 1191–1216 | KE tile-corner fix (`dt6 = dt/6`) | 3A.6 `ke_corner_fix` | already-implemented service |
| 25 | 1219–1223 | `vt = u*dx` (**overwrites** S1–S3 `vt`) | 3A.6 vorticity precursor | already-implemented service |
| 26 | 1224–1228 | `ut = v*dy` (**overwrites** S1–S3 `ut`) | 3A.6 vorticity precursor | already-implemented service |
| 27 | 1231–1235 | `wk = rarea*curl(vt,ut)` | 3A.6 vorticity precursor | already-implemented service |
| 28 | 1249–1255 | `w = w / NEW delp` | K11 | driver-owned state update |
| 29 | 1256–1262 | `damp_w`: `w += dw` | K12 | driver-owned state update |
| 30 | 1263–1269 | `USE_COND` `q_con/delp` | — | **deferred / not live** |
| 31 | 1277–1423 | divergence damping (both `nord_k` arms), `ke += vort` | 3A.7 | already-implemented service |
| 32 | 1425–1436 | `d_con`: `ub`/`vb` = `vort` differences | 3A.7 `dcon_vort_diff` | already-implemented service |
| 33 | 1439–1459 | `vort = wk + f0` | 3A.6 `vorticity_abs` | already-implemented service |
| 34 | 1461–1462 | `fv_tp_2d(vort, hord_vt)` → `fx,fy` | 3A.5 scalar transport | already-implemented service |
| 35 | 1463–1472 | `u`/`v` D-wind update from `vt,ut,ke,fx,fy` | 3A.6 `dwind_update` | already-implemented service |
| 36 | 1476–1479 | `damp_v`: `del6_vt_flux(wk)` → `ut,vt` | 3A.7 vorticity damping | already-implemented service |
| 37 | 1481–1520 | `d_con` heat-source chain | 3A.7 `dcon_scale` + `dcon_heat_source` | already-implemented service |
| 38 | 1523–1534 | `damp_v`: `u += vt`, `v -= ut` | 3A.7 `dwind_damping_add` | already-implemented service |

Nothing is classified BLOCKED.

Not-live arms, all **fenced** rather than silently skipped: `#ifdef SW_DYNAMICS`
(:560–670, :922–925, :1057–1062, :1536), `#ifdef USE_COND` (:993–1001,
:1263–1269), `inline_q` (:1009–1032), `do_f3d` / `#ifdef ROT3` (:1238–1248 and
`vort = wk + f0*z_rat` at :1445), `do_skeb` (:981–983, :1508–1513),
`grid_type >= 3` (:867–880), `nested` (:675–687, :1069–1070, :1191, :1280–1296,
:1372), `hydrostatic` (skips S10–S12 and S28–S29).

---

## 3. Driver-owned state updates

### `pt` — S15, `sw_core.F90:1034–1047`

Kernel K9. The authoritative form is strictly three ordered statements per cell:

```
pt(i,j) = pt(i,j)*delp(i,j) + (gx(i,j)-gx(i+1,j)+gy(i,j)-gy(i,j+1))*rarea(i,j)
delp(i,j) = delp(i,j) + (fx(i,j)-fx(i+1,j)+fy(i,j)-fy(i,j+1))*rarea(i,j)
pt(i,j) = pt(i,j)/delp(i,j)
```

- mass-weight with the **old** `delp`;
- divide by the **new** `delp`;
- the heating contribution is *not* here — it enters `pt` only through WP-D's
  post-loop `heat_source` application, and `d_sw`'s own contribution to
  `heat_source` is S10 (seed) and S37 (add);
- the damping contribution reaches `pt` through the `nord_t`/`damp_t`
  `deln_flux` folded into `gx`/`gy` by S14's `fv_tp_2d`, not as a separate term.

`pt` is **not** updated earlier even though `gx`/`gy` are available after S14:
S15 must follow S14 and must be a single fused per-cell statement group so that
`delp`'s old value is still live for the mass weighting.

### `delp` — S15, same kernel

Only the `div(fx,fy)*rarea` mass-flux term; `fx`/`fy` come from S7's
`fv_tp_2d(delp, hord_dp = -6)` and already carry the `nord_v`/`damp_v`
`deln_flux` contribution. No additional metric factor beyond `rarea`. The old
`delp` is required by S14 (`mass=delp`) and S12 (`delp*w`), both of which
precede S15, and by S15 itself — so no double buffer is needed: K9 loads `delp`
into a register once and writes `pt` and `delp` from the same thread.

### `w` — S10 → S12 → S28 → S29

`w`'s treatment is deliberately **not** the generic mass-weighted-scalar
treatment:

- S10 applies `del6_vt_flux` (**not** `deln_flux`) to `w`, producing separate
  `fx2`/`fy2`, from which `dw = div(fx2,fy2)*rarea`. Confirmed against
  `sw_core.F90:970–985`.
- S11's `fv_tp_2d(w, hord_vt, mfx=fx, mfy=fy)` passes mass fluxes but **no**
  `nord`/`damp_c`, so `w` gets **no** `deln_flux` at all. This is the trap
  recorded in the 3A.5 handoff ("mass fluxes imply mass-weighted damping" is
  false for `w`) and it is honoured.
- S12 mass-weights with the **old** `delp`; S28 divides by the **new** `delp`,
  sixteen source lines later. Fusing S12 and S28 would divide by the wrong
  `delp`; the separation is preserved as four distinct kernels.
- S29 adds `dw` **after** the division, on the `damp_w > 1.E-5` levels only.

### `u` / `v` — S35 then S38

S35 (`3A.6 dwind_update`) consumes `vt`/`ut` in their **second** meaning
(`u*dx`, `v*dy` from S25/S26), the divergence-damping-augmented `ke` from S31,
and the vorticity fluxes `fx`/`fy` from S34. S38 adds the `damp_v` diffusive
increment (`u += vt`, `v -= ut`) where `vt`/`ut` now hold S36's `del6_vt_flux`
output — a **third** meaning of the same two arrays.

`u` and `v` are written in the same S35 launch because their right-hand sides
read only `ke`, `fx`, `fy`, `vt`, `ut` — never each other. They are **not**
fused with S36/S38, because S36 overwrites `ut`/`vt`, which S35 still reads: an
anti-dependence boundary.

### Flux capacitors — S8 and S16

Five `real(kind=8)` `intent(INOUT)` accumulators (`sw_core.F90:548–554`), state
that persists **across acoustic substeps**:

| accumulator | Fortran box | `+=` | line |
|---|---|---|---|
| `cx` | `(is:ie+1, jsd:jed)` | `crx_adv` | 944–948 |
| `xflux` (dyn_core `mfx`) | `(is:ie+1, js:je)` | `fx` | 949–953 |
| `cy` | `(isd:ied, js:je+1)` | `cry_adv` | 955–957 |
| `yflux` (dyn_core `mfy`) | `(is:ie, js:je+1)` | `fy` | 958–960 |
| `dpx` | `(is:ie, js:je)` | `div(fx,fy)*rarea` | 1050–1054 |

No coefficient, no sign flip, no scaling: every one is a plain `+=`.
Accumulation is performed in `accumulator_real` (kind=8) **even in an R4 build**,
as the Fortran declaration demands; K5 and K10 promote the `fv_real` addend
before adding. `d_sw` **never initializes** them (`dyn_core.F90:379–392` zeroes
them once per `k_split`), so the driver must never zero them — Part J
accumulates across **two** `launch_d_sw_cuda` invocations from a deliberately
nonzero initial state to prove this, and the integrated Phase B test seeds
`mfx=11.0, mfy=13.0, cx=0.17, cy=0.19, dpx=23.0` so accumulation is
distinguishable from overwrite.

`heat_source` and `diss_est` are the opposite case: `intent(OUT)`, zeroed by S9
on every call, accumulated only *within* one call. They are not flux
capacitors.

---

## 4. Per-level control

`dyn_core.F90:699–765` recomputes eleven controls inside its `k` loop. Which
genuinely vary with `k`, and the carrier field each uses:

| control | per level? | why | carrier |
|---|---|---|---|
| `nord_k` | YES | 0 on all three sponge branches | `damping_nord_k` |
| `nord_v(k)` | YES | 0 on sponge 1–2 if `do_vort_damp` | `damping_ndif` |
| `nord_w` | YES | 0 on all three sponge branches | `damping_nord_w` |
| `nord_t` | NO | :715 assignment never reset | `config.nord_t` |
| `damp_vt(k)` | YES | `0.5*d2_divg` on sponge 1–2 | `damping_v` |
| `damp_w` | YES | `d2_divg` on all three sponge branches | `damping_w` |
| `damp_t` | NO | :718 copies the *pre*-sponge `damp_vt` | `damping_t` (array, equal entries) |
| `d2_divg` | YES | three distinct sponge values | `damping_d2_bg` |
| `d4dmp(k)` | YES | `dyn_core.F90:684` | `damping_d4_bg` |
| `d_con_k` | YES | 0 on all three sponge branches | `damping_d_con` |
| `kgb` | YES | `ke_bg` at `k == npz`, 0 elsewhere | `damping_kgb` |

These are **not** reduced to global scalars. `DivergenceLevels` and
`LevelFilter` remain per-level and are indexed at `min(k, npz-1)`;
`LevelFilter::gate` continues to separate *activation* from *coefficient*.

**The required production behaviour** — with `n_sponge > 0` the top levels take
the `nord == 0` divergence-damping arm while the interior levels take the
higher-order arm **in the same `d_sw` invocation** — is preserved: both branch
families are launched on every call, each filtered to the levels that own it.
Part I asserts exactly this, and the integrated Phase B fixture cycles the
validated four-level mixed pattern (`nord_k = {0,2,0,1}`,
`nord_v = {0,2,1,2}`, `nord_w = {0,1,0,2}`, `d_con = {0,1.0,0.5,0}`,
`kgb = {0,0,1.0e-4,5.0e-5}`) over all `npz` levels.

`nord_cap` values are bounded from the **host** without reading any device
array, because the Fortran expressions bound them structurally
(`nord_k ∈ {0,nord}`, `nord_v`/`nord_w ∈ {0,min(2,nord)}`). **No D2H is
performed anywhere in this driver.**

### Derived per-level coefficients

`launch_dsw_level_coefficients_cuda` computes, in one small device kernel over
`npz`:

```
deln_damp_delp(k) = (damp_v(k)*da_min)**(nord_v(k)+1)     gate: damp_v(k) > 1.e-4
deln_damp_pt(k)   = (damp_t(k)*da_min)**(nord_t+1)        gate: damp_t(k) > 1.e-4
del6_damp_w(k)    = (damp_w(k)*da_min_c)**(nord_w(k)+1)   gate: damp_w(k) > 1.E-5
del6_damp_v(k)    = (damp_v(k)*da_min_c)**(nord_v(k)+1)   gate: damp_v(k) > 1.E-5
```

The two operator families are kept strictly distinct, which the 3A.3b handoff
flagged as easy to get wrong:

- `deln_flux` (`tp_core.F90:1263`) — coefficient `(damp_c*da_min)**(nord+1)`,
  threshold `1.e-4`, flux **folded into** `fx`/`fy`;
- `del6_vt_flux` (`sw_core.F90:1544`) — the subroutine uses `damp` **raw**
  because `d_sw`'s call sites pre-scale it: `sw_core.F90:973` passes
  `damp4 = (damp_w*da_min_c)**(nord_w+1)` and `:1477` passes
  `damp4 = (damp_v*da_min_c)**(nord_v+1)` — note **`da_min_c`**, not `da_min` —
  while gating on the **raw** `damp_w`/`damp_v` against `1.E-5`. Emits separate
  `fx2`/`fy2`.
- The `update_dz_d` height path (3A.4) passes `damp(k)` **un**scaled. That
  difference is real and is preserved.

Verified directly against `sw_core.F90:960–990` and `:1465–1485` during this
milestone.

---

## 5. Anti-dependence audit

| field | old value's last consumer | new value's first producer | overwrite point | workspace needed |
|---|---|---|---|---|
| `delp` | S14 (`mass=delp`), S12 (`delp*w`) | S15 | S15 | **NO** — K9 reads `delp` once and writes `pt`/`delp` from the same thread; the old value lives in a register |
| `pt` | S15 itself (`pt*delp`) | S15 | S15 | NO (same) |
| `w` | S12 (`delp*w`) | S12 | S12 | NO (same thread) |
| `w` | S28 divides by the **new** `delp`, 16 lines after S12 weighted by the **old** one | S28 | S28 | NO — but S12 and S28 **must not** be fused |
| `ut`/`vt` (1st meaning: contravariant C winds) | S18 / S21 / S24 | S25 / S26 (`u*dx`, `v*dy`) | S25/S26 | NO — but never fuse across the boundary |
| `ut`/`vt` (2nd meaning) | S35 | S36 (`del6` diffusive flux) | S36 | NO — same rule |
| `ke` | S35 | S20 | S20 | NO |
| `vort` (damping) | S32 | S33 (`wk + f0`) | S33 | NO — S32 **must** precede S33 |
| `vort` (absolute) | S34 | S36 (used as `del6` `d2` scratch) | S36 | NO |
| `wk` | S33 and S36 | S27 | S27 | NO — S10 uses `wk` as `del6` scratch while it is dead |
| `fx`/`fy` | S16 (delp fluxes) | S34 (vorticity fluxes) | S34 | NO |
| `fx`/`fy` | S35 (vorticity fluxes) | S37 (`v*rdy`, `u*rdx`) | S37 | NO |
| `uc`/`vc` | S21 / S31 low-order arm | S31 high-order arm (**destroys both**) | S31 | NO — nothing in `d_sw` reads `uc`/`vc` after S31 |
| `dw` | S29 | S10 | S10 | **YES** — dedicated arena slot; lifetime spans S10…S29 |
| `mfx`/`mfy`/`cx`/`cy`/`dpx` | — (accumulators) | — | never overwritten | NO — read-modify-write in place |

Every ordering constraint is enforced by launch order on **one** stream
(`StreamId::compute`); no stage is fused across an anti-dependence boundary and
no additional double buffering was required beyond `dw`. Part K asserts the
invariants that would break if any of these orderings changed.

---

## 6. `corner_edge_service` disposition

`NumericalDispatch::corner_edge_service` is **not** used by this driver, and
remains bound to the deliberate fail-stub `corner_service_not_implemented`
(`FV3_GPU_UNSUPPORTED_CONFIGURATION`).

Every corner/edge operation the production `grid_type == 0` `d_sw` path needs is
a **local, on-rank, ng×ng device operation with FV3-specific algebra**, and each
is already implemented as a focused local kernel by the work package that owns
the surrounding numerics:

| operation | owner |
|---|---|
| K2/K3 tile-edge and 2×2 corner wind solves | this milestone (specific algebra, not a generic fill) |
| `copy_corners` inside `fv_tp_2d` / `deln_flux` / `del6_vt_flux` | 3A.4 / 3A.5 |
| `fill_corners` BGRID scalar + DGRID vector inside the S31 high-order arm | 3A.7 |
| KE corner fix (S24) | 3A.6 |
| `delpc` corner extra-term removals (S31 `nord == 0` arm) | 3A.7 |

Routing these through the generic `CornerServiceRequest` abstraction would
require encoding FV3-specific algebra into a generic operation enum, which the
3A.7 review explicitly rejected, and the 3A.3b handoff independently recorded
that "the current `LaunchFn` slot is the wrong shape" and that the operation
"cannot be hoisted or cached" (dir=1/dir=2 write the same memory with different
formulas).

**Therefore:** the generic slot is formally *not the mechanism* for this path.
Corner **support** is complete for the production path. `corner_edge_service`
stays reserved for genuine halo/communication corner work at an MPI boundary and
was deliberately **not** populated merely to eliminate the stub.

---

## 7. Unsupported-configuration fencing

`launch_d_sw_cuda` validates **before its first launch**, so a rejected
configuration can never leave prognostic state half-updated.

`FV3_GPU_UNSUPPORTED_CONFIGURATION`:

| fence | why |
|---|---|
| `grid_type != 0` | the `>= 3` arm replaces S1–S3 with `ut=uc`, `vt=vc` |
| `nested != 0` | changes S1–S3, S17, S24 and S31 simultaneously |
| `hydrostatic != 0` | S10–S12 and S28–S29 vanish; `w` is not prognostic |
| `inline_q != 0` | replaces S15 with the `wk`-buffered form, adds an `nq` loop |
| `use_cond != 0` | adds S13 and S30 |
| `do_f3d != 0` | changes S28 and S33 |
| `do_skeb != 0` | adds `diss_est` production at S10 and S37 |
| `rot3 != 0` | the `ROT3` arm of S28 |
| `abs(hord_mt/vt/tm/dp) != 6` | only order-6 PPM is implemented |
| `nord < 0` or `nord > 3` | `nt = nord-1` would index past `ng = 3` |
| `npx == 0` or `npy == 0` | panel-seam stencils would silently behave as interior |
| `a_imp != 1.0` | not read by `d_sw`, but a different value means the caller is not running the configuration this driver was validated against |
| `stream == nullptr` | S31's `smag_corner` reuses WP-D's `a2b_ord4`, which rejects a null stream; without this fence it would surface thirty stages in, after `pt`, `delp`, `w` and `dpx` had already been mutated |

`FV3_GPU_INVALID_ARGUMENT` (wiring defect, not a configuration choice), also
before the first launch: `nk != npz`, any missing or under-covering state /
geometry / level / arena view, a non-`compute` submission stream, and a
`damping_*` view whose element width does not match the active `fv_real`.

On the **nested** question raised by 3A.7 finding (8): `TransportBounds::nested`
is the only nested signal the contract carries, and `DivergenceCornerArgs` still
has no nested flag. Rather than expose a nested path silently, `nested != 0` is
rejected outright at the driver boundary — the minimal contract information
needed to reject it safely. Nothing here is a numerical approximation of an
unsupported mode.

---

## 8. `launch_d_sw_cuda` and `launch_dgrid_dynamics_cuda`

`launch_d_sw_cuda(b, opt, lv, g, st, ws, stream)` — the internal WP-B service.
It resolves nothing itself: it receives validated typed views, enforces §7,
constructs the per-level selectors (`DivergenceLevels`, `LevelFilter`) and the
derived coefficient arrays on device, then enqueues S1…S38 in authoritative
order on the caller's stream. No host numerical fallback, no per-stage H2D/D2H,
no device synchronization — persistent device residency is preserved and the
service only enqueues work.

`launch_dgrid_dynamics_cuda(ExecutionContext, fv3_gpu_error*)` — the
`LaunchFn`-shaped wrapper: `bind_wp_b_d_sw` resolves ≈70 registered views and
carves `WpBDSwArena` (34 whole `ni × nj × (npz+1)` `fv_real` slots from
`WorkspaceId::transport`, the last subdivided into the seven derived per-level
columns), then calls `launch_d_sw_cuda`. `dt` is divided by `n_split` here
because `dyn_core.F90:297` defines `dt = bdt/n_split` while
`runtime/fv3_gpu_runtime.cpp:203` passes `p.dt = s->bdt` verbatim.

### Contract defect found and fixed during sanitizer verification

`bind_wp_b_d_sw` originally required **one uniform** `[isd,ied] × [jsd,jed]` box
for all seventeen 3-D state fields, justified by a comment claiming that "with
halo >= 3 the extended box already contains the +1 stagger". That reasoning is
**false** — `isd`/`ied` *are* `is-halo`/`ie+halo`, so `jed+1` lies one row
outside the box no matter how large the halo is. The requirement was therefore
too **weak** and accepted a registration the kernels over-read.

`compute-sanitizer memcheck` caught it on the integrated
`fv3_gpu_dyn_core_step` path: "Invalid `__global__` read of size 8 bytes" in
`dsw_ut_interior_kernel`, exactly 8 bytes past the end of a 14×14×10 field —
element `(isd, jed+1, npz-1)`. The read is legitimate;
`sw_core.F90:690–696` evaluates
`ut(i,j) = uc(i,j) - 0.25*cosa_u(i,j)*(vc(i-1,j)+vc(i,j)+vc(i-1,j+1)+vc(i,j+1))`
over `j = jsd..jed`, so at `j == jed` it reads `vc(i, jed+1)`; the `vt` loop
(:702–706) reads `uc(i+1, j)` at `i == ied`. This matches the declarations at
`sw_core.F90:548–556`.

The binder now requires the **exact** Fortran box per stagger group:

| group | fields | required box |
|---|---|---|
| A-grid | `w, delp, pt, ua, va, delpc, ptc, heat_source, diss_est, z_rat` | `[isd,ied] × [jsd,jed]` |
| y-face | `u, vc, vt` | `[isd,ied] × [jsd,jed+1]` |
| x-face | `v, uc, ut` | `[isd,ied+1] × [jsd,jed]` |
| B-grid | `divgd` | `[isd,ied+1] × [jsd,jed+1]` |

(`heat_source`/`diss_est` are a deliberate over-requirement: Fortran declares
them `(is:ie, js:je)` but the S9 reset kernel covers the extended box.)

The focused test had missed this because its hand-built fixture pads its arrays;
the integrated harness now registers every field over the union box
`[isd,ied+1] × [jsd,jed+1]`.

Numerical semantics changed: **NO** (a coverage requirement and a test
registration).

---

## 9. Production dispatch activation

`runtime/fv3_gpu_production_dispatch.cu:49`:

```cpp
d.dgrid_dynamics = &fv3::gpu::dgrid::launch_dgrid_dynamics_cuda;
```

This is the first milestone in which populating `dgrid_dynamics` is permitted,
and it was populated only after the complete supported S1–S38 sequence existed
and the focused Parts A–M passed. Full audit:

| slot | implementation | wired? | GPU executed through `fv3_gpu_dyn_core_step`? | production required? |
|---|---|:--:|:--:|:--:|
| `cgrid_predictor` | — | **NO** (`nullptr`) | NO | **YES** |
| `cgrid_vertical_solver` | — | **NO** (`nullptr`) | NO | **YES** |
| `cgrid_pressure` | `cgrid::launch_cgrid_pressure_v3` | YES | YES | YES |
| `dgrid_dynamics` | `dgrid::launch_dgrid_dynamics_cuda` | **YES (3A.8)** | **YES** | YES |
| `dgrid_height` | — | **NO** (`nullptr`) | NO | **YES** |
| `dgrid_height_transport` | `dgrid::launch_dgrid_height_transport_cuda` | YES | **NO** — its only caller is `launch_update_dz_d`, which `launch_vertical_solver` does not call | YES |
| `vertical_solver` | `vertical::launch_vertical_solver` | YES | YES | YES |
| `nh_pressure` | `operators::launch_nh_pressure_cuda` | YES | YES | YES |
| `post_loop_heating` | `operators::launch_post_loop_heating_cuda` | YES | YES | YES |
| `corner_edge_service` | deliberate fail-stub | stub | NO | **NO** for this path (§6) |

Production-required slots still `nullptr`: **`cgrid_predictor`,
`cgrid_vertical_solver`, `dgrid_height`.** Gate C1 is therefore **not** passed.

`runtime/fv3_gpu_runtime.cpp` now walks seven stages, `dgrid_dynamics` inserted
after `cgrid_pressure`:

```cpp
const LaunchFn stages[]={dispatch.cgrid_predictor,dispatch.cgrid_vertical_solver,
                         dispatch.cgrid_pressure,dispatch.dgrid_dynamics,
                         dispatch.vertical_solver,dispatch.nh_pressure,
                         dispatch.post_loop_heating};
```

`WorkspaceId::transport` is sized at 34 `value_bytes` (= `WpBDSwArena::array_count`);
the binder re-checks the slice against that constant so a stale literal fails
loudly instead of corrupting neighbouring scratch.

### Halo readiness — a deliberate omission, recorded as a blocker

`bind_wp_b_d_sw` makes **no** `dependencies->require(...)` call, so no halo
group is declared ready for `dgrid_dynamics`. This is deliberate:
(a) this milestone must not introduce MPI work; (b) `cgrid/fv3_cgrid.cu:122`
calls `mark_stale(FV3_GPU_H_CWIND)`, so adding
`require(FV3_GPU_H_CWIND)` would make the integrated stage list fail with
`FV3_GPU_STATE_CONFLICT` until a halo-exchange stage exists. It is listed as a
Gate C1 blocker in §14, not silently ignored.

---

## 10. Tests

Permanent focused test `cuda/dyn_core/tests/dgrid/test_wp_b5_dsw_cuda.cpp`
(CTest `fv3_wp_b5_dsw`, label `gpu`, `SKIP_RETURN_CODE 77`) and the extended
integration test `tests/integration/test_dispatch_smoke.cpp` (CTest
`fv3_dispatch_smoke`). Mapping to the §16 requirement letters:

| §16 requirement | where |
|---|---|
| A exact `d_sw` service ordering | focused Part A (K1/K2/K3 ordered wind construction) + Part K (end-to-end ordering invariants) |
| B mixed per-level `nord_k` branches in one run | focused **Part I** |
| C `pt` update | focused **Part C** (joint K9) |
| D `delp` update | focused **Part C** (joint K9) + Part D (`dpx`) |
| E `w` update | focused **Part E** (K7/K8/K11/K12 chain) |
| F `u`/`v` update | focused Part K + Part M; the S35/S38 kernels themselves are covered by `fv3_wp_b3_momentum` and `fv3_wp_b4_divergence` |
| G KE / vorticity ordering | focused **Part K** |
| H divergence-damping ordering | focused **Part K** + Part I |
| I heat-source / `d_con` ordering | focused **Part G** (S9 reset) + Part E + Part K |
| J flux-capacitor accumulation | focused **Part J** (two invocations, nonzero start) + Part F (K5) |
| K anti-dependence preservation | focused **Part K** |
| L unsupported-configuration rejection | focused **Part L** (asserts no state mutation) |
| M direct `launch_d_sw_cuda` GPU execution | focused **Part M** (seam and interior) |
| N production `NumericalDispatch.dgrid_dynamics` execution | `fv3_dispatch_smoke` Phase B — hard-fails if `dgrid_dynamics` is `nullptr`, so N and O are separately falsifiable |
| O execution through `fv3_gpu_dyn_core_step` | `fv3_dispatch_smoke` Phase B |
| P R8 | both, `build_cuda_r8` |
| Q R4 | both, `build_cuda_r4` |

Supporting parts: Part B (K4a/K4b + `ra_x`/`ra_y`), Part F (K5), Part H (derived
per-level coefficient arrays).

Fields are deliberately non-degenerate. Two fixture defects found and fixed
during A100 bring-up, both in the integrated harness (test input only,
numerical semantics changed: **NO**):

1. **`rarea` inversion.** `rarea` was seeded as `1.0e10` rather than
   `1.0/area ≈ 1.0e-10`. Every `rarea`-weighted increment was inflated by
   ~1e20, which overflowed `fv_real` in R4 once `d_sw` multiplied a mass-flux
   divergence by it. Fixed the input, not the check; the absolute `1e-12`
   change threshold was replaced by a reported `max|delta|` comparison against
   a pre-step snapshot.
2. **Degenerate `uc`/`vc`.** `uc`/`vc` were seeded to exactly zero. `d_sw`
   builds its entire transport half from them (S1 contravariant → `ut`/`vt` →
   `crx`/`cry` → `xfx`/`yfx` → `delp`/`pt`/`w` and the capacitors), so the
   transport half was degenerate — precisely what §16 forbids. R8 resolved a
   `delp` change of 2.45e-05 out of 9990 Pa (2.5e-09 relative); R4 rounded it to
   exactly zero and Phase B correctly failed. Fixed by seeding nonzero analytic
   C-grid winds sized so the per-cell variation (~1 m/s over `dx = 1e5` m) gives
   divergence ~1e-5 s⁻¹ and a ~1 Pa `delp` increment over `dt = 10` s. Both
   precisions then report `delp max|delta| = 1.501035` (R8) / `1.500977` (R4).

A third defect was in the harness topology: all four `*_corner` flags were set
while `neighbors[]` stayed at its `{rank=-1}` default, so `physical_edges()`
reported no seams while `physical_corners()` reported all four.
`consistent_corners` in `launch_a2b_ord4_global_cuda` correctly rejected the
contradiction (`A2B (pkc) CUDA launch failed`). Fixed by populating four
cross-tile `EdgeNeighbor`s. The two independent "is this a panel edge"
mechanisms — WP-B `d_sw` tests `b.is==1 / b.ie+1==npx / b.js==1 / b.je+1==npy`,
WP-D uses `physical_edges(topology)` on `neighbors[]` — are now documented in
the harness.

---

## 11. Build and test results

### Build

| | result |
|---|---|
| R8 (`build_cuda_r8`) CUDA compilation | **PASS**, zero warnings |
| R4 (`build_cuda_r4`) CUDA compilation | **PASS**, zero warnings |

Toolchain: GCC/G++/GFortran 12.3.0, nvcc 12.5.82 (`sm_80`), OpenMPI 4.1.6,
CMake 3.30.3, via `module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15`
→ `GEOSenv`, `comp/gcc/12.3.0`, `mpi/openmpi/4.1.6/gcc-12.3.0`,
`nvidia/nvhpc-byo-compiler/24.7`.

### Full CTest on A100 (Slurm job `58069277`, `warpa003`)

| precision | result |
|---|---|
| R8 | **17/17 PASS** |
| R4 | **17/17 PASS** |

Previous baseline was 16/16; the seventeenth test is `fv3_wp_b5_dsw`. Full list:
`fv3_gpu_descriptor`, `fv3_gpu_api`, `fv3_gpu_c_header`, `fv3_gpu_contract`,
`fv3_wp_a_view`, `fv3_wp_b_kernels`, `fv3_wp_b_contract`, `fv3_wp_c_column`,
`fv3_wp_d_operators`, `fv3_wp_d_contract`, `fv3_integrated_smoke`,
`fv3_dispatch_smoke`, `fv3_wp_b_cuda_smoke`, `fv3_wp_b2_scalar_transport`,
`fv3_wp_b3_momentum`, `fv3_wp_b4_divergence`, `fv3_wp_b5_dsw`.

**Prior-milestone regressions: NONE.** 3A.4 height transport
(`fv3_wp_b_cuda_smoke`), 3A.5 scalar transport (`fv3_wp_b2_scalar_transport`),
3A.6 momentum (`fv3_wp_b3_momentum`) and 3A.7 divergence/corner
(`fv3_wp_b4_divergence`) all pass unchanged in both precisions. No regression
was accepted to make the integrated driver pass.

### A100 GPU execution — §18 evidence

Device: NVIDIA A100-SXM4-40GB, driver 535.104.12, `sm_80`. Executed via Slurm
(`gpu_a100`, `--constraint=rome`, `--gres=gpu:1`, `--mem=60G`,
`--cpus-per-task=12`). No CUDA executable was run on a login node.

| §18 item | R8 | R4 |
|---|---|---|
| direct `launch_d_sw_cuda` execution (Part M) | PASS | PASS |
| `dgrid_dynamics` dispatch execution (Phase B, hard-asserted non-null) | PASS | PASS |
| `fv3_gpu_dyn_core_step` execution (Phase B) | PASS | PASS |
| mixed damping branches in one call (Part I) | PASS | PASS |
| flux-capacitor state changes across invocations (Part J) | PASS | PASS |
| finite outputs (no NaN/Inf in any probed field) | PASS | PASS |
| non-degenerate prognostic updates | PASS | PASS |
| no CUDA runtime errors | PASS | PASS |

Phase B `max|delta|` (integrated, one `fv3_gpu_dyn_core_step` call):

| field | R8 | R4 |
|---|---|---|
| `u` | 7.4248e+02 | 7.4248e+02 |
| `v` | 7.4353e+02 | 7.4353e+02 |
| `delp` | 1.501035 | 1.500977 |
| `w` | 6.5275e+02 | 6.5275e+02 |
| `mfx` | 1.307123e+11 | 1.307123e+11 |
| `mfy` | 8.884476e+10 | 8.884476e+10 |
| `cx` | 1.428968e-03 | 1.428968e-03 |
| `cy` | 9.067089e-04 | 9.067089e-04 |
| `pt` | 8.0 | 8.0 |

All CHANGED. Phase A `uc` delta: 3.567063e-05 (R8) / 3.528595e-05 (R4).

---

## 12. compute-sanitizer

Run on the A100 via
`/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/compute-sanitizer`
(absolute path; the tool is not on `PATH`). Output written under
`/discover/nobackup/projects/imvi-nse/jli30/slurm_logs/` because `/tmp` on a
compute node is node-local and invisible from the login node.

### `fv3_wp_b5_dsw_test` (focused driver test)

| tool | R8 | R4 |
|---|---|---|
| memcheck | **0 errors** | **0 errors** |
| initcheck | **0 errors** | **0 errors** |
| racecheck | **0 hazards** | **0 hazards** |

### `fv3_dispatch_smoke_test` (integrated `fv3_gpu_dyn_core_step` path)

| tool | R8 | R4 |
|---|---|---|
| memcheck | **0 errors** | **0 errors** |
| initcheck | 144 errors | 144 errors |
| racecheck | **0 hazards** | **0 hazards** |

**Classification of the memcheck finding (FIXED).** The first run reported 107
memcheck errors: 11 "Invalid `__global__` read of size 8 bytes" plus 96
consequent `cudaErrorLaunchFailure` reports (the first invalid read
sticky-fails the CUDA context, so everything after it is a cascade). Root cause
was the under-specified staggered-field coverage requirement in
`bind_wp_b_d_sw` described in §8. After the fix, memcheck is **0 errors in both
precisions**, and because the sticky failure is gone the whole `d_sw` sequence
now executes under memcheck — so this is a clean verdict, not an early abort.

**Classification of the initcheck finding (NOT fixed; benign, documented).**
All 100 printed findings (of 144) are the *same* kernel on the *same* call path
and occur at exactly three levels:

```
Uninitialized __global__ memory read of size 8 bytes
  at   fv3::gpu::dgrid::copy_corners_kernel
  from launch_copy_corners_cuda
   <-  launch_del6_vt_flux_cuda
   <-  launch_d_sw_cuda  <-  launch_dgrid_dynamics_cuda  <-  fv3_gpu_dyn_core_step
  blocks (0,0,1), (0,0,5), (0,0,9)
```

Blocks 1/5/9 are exactly the levels where the harness's four-level cycling
pattern sets `nord_w = 1` (`nord_w = {0,1,0,2}`, `s = k % 4`). That pins the
cause precisely:

- `del6_vt_flux` fills `d2` only over `[is-1-nord, ie+1+nord]²`
  (`sw_core.F90:1588–1592`). For `nord == 1` that is `[is-2, ie+2]`.
- `copy_corners(d2, …, dir)` (`tp_core.F90:257–320`) loops the fixed
  `ng = 3` corner blocks `[1-ng,0]` and reads e.g. `q(j, 1-i)` for the SW/dir=1
  block, whose source index reaches `i = -2` — one cell **outside** the region
  `del6_vt_flux` just filled. For `nord == 2` the filled region is the whole
  `ng = 3` box and nothing is read undefined; for `nord == 0` `copy_corners` is
  not called at all. So only `nord == 1` exposes it.
- The affected destination cells are those with `i == -2` or `j == -2`. The
  subsequent `fx2`/`fy2` loops read `d2` only over
  `[is-nord-1, ie+nord+1]` = `[-1, ie+2]`, so **no cell whose value derives from
  an undefined read is ever consumed.** Destination cells that *are* consumed
  have sources at `i, j ∈ [-1, 0]`, which are inside the filled region.
- The vorticity call at `sw_core.F90:1477` does not trigger it even at
  `nord_v == 1`, because there `d2` is `vort`, which an earlier stage has
  already written over the whole box. Only the `w` call at `:973`, whose `d2` is
  the still-dead `wk`, is exposed — matching the observed call path exactly.

**Verdict: this is a faithful reproduction of the authoritative Fortran, in
which the same cells are equally undefined** (`wk` is an unassigned local
automatic at `:973`), **and the undefined values are provably never consumed.**
The values are non-propagating: the test's finiteness assertions pass in both
precisions and no NaN/Inf appears in any probed field.

It was deliberately **not** "fixed" by zeroing the arena — §19 forbids that —
and deliberately **not** fixed by widening the `d2` fill in
`launch_del6_vt_flux_cuda`, because that service is shared with the validated
3A.4 height path and changing it to silence a benign finding risks a regression
this milestone is not authorised to accept. Recorded as a Gate C1 follow-up
hardening item (§14). The focused `fv3_wp_b5_dsw_test` is initcheck-clean, so a
clean sanitizer surface for the driver itself does exist.

Specific §19 attention items, all clean: in-place prognostic updates
(memcheck/racecheck 0), workspace reuse between stages (0), flux-capacitor
accumulation (0), per-level selector arrays (0), edge/corner local kernels (0),
state lifetimes across service calls (0). Racecheck reports **0 hazards, 0
errors, 0 warnings** in both precisions despite S1–S38 running on a single
stream with heavy in-place mutation.

---

## 13. Source-level reconciliation

Ordered side-by-side, `sw_core.F90` sequence vs. CUDA sequence. Status key:
**M** = matches Fortran; **M\*** = matches with explained CUDA structural
difference; **D** = discrepancy; **NL** = not live.

| S | Fortran | CUDA | status |
|---:|---|---|:--:|
| 1 | :672–707 interior `ut`/`vt` | K1 `dsw_ut_interior_kernel` / `dsw_vt_interior_kernel` | M |
| 2 | :709–779 8 edge overrides | K2, 8 ordered launches | M\* — Fortran's `if (j/=…)` guards inside one loop nest become 8 separately-bounded launches; the write set and order are identical |
| 3 | :791–863 4 corner solves | K3, 4 ordered launches | M\* — same reason; the 2×2 solve statement order within each corner block is preserved (statement 4 reads `uc(2,1)`, not `ut(2,1)`) |
| 4 | :882–892 | K4a | M |
| 5 | :898–921 | K4b | M |
| 6 | :927–936 | 3A.4 `launch_ra_x/y_cuda` | M |
| 7 | :939–940 | 3A.5 `launch_scalar_transport_cuda(hord_dp, nord_v, damp_v)` | M |
| 8 | :943–960 | K5 | M\* — `accumulator_real` promotion is explicit in CUDA; Fortran gets it from the `kind=8` declaration |
| 9 | :963–968 | K6 | M\* — covers the extended box rather than `(is:ie, js:je)`; strictly larger, and the extra cells are never read |
| 10 | :970–985 | 3A.4 `launch_del6_vt_flux_cuda` (pre-scaled `damp4`, raw-`damp_w` gate) + K7 | M |
| 11 | :986–987 | 3A.5 scalar transport, `mfx`/`mfy` present, **no** `nord`/`damp_c` | M |
| 12 | :988–992 | K8 | M |
| 13 | :993–1001 | — | NL (`USE_COND` undefined; fenced) |
| 14 | :1004–1006 | 3A.5 scalar transport (`mass=delp`, `nord_t`, `damp_t`) | M |
| 15 | :1034–1047 | K9, three ordered statements in one thread | M |
| 16 | :1050–1054 | K10 | M |
| 17 | :1065–1074 | host arithmetic in `launch_d_sw_cuda` | M\* — computed once on the host instead of per-`k`; the values are `k`-independent |
| 18 | :1077–1120 | 3A.6 `bgrid_vb` | M |
| 19 | :1122–1123 | 3A.6 `ytp_v` | M |
| 20 | :1125–1129 | 3A.6 `ke_product` | M |
| 21 | :1131–1177 | 3A.6 `bgrid_ub` | M |
| 22 | :1179–1180 | 3A.6 `xtp_u` | M |
| 23 | :1182–1186 | 3A.6 `ke_average` | M |
| 24 | :1191–1216 | 3A.6 `ke_corner_fix` | M |
| 25–27 | :1219–1235 | 3A.6 vorticity precursor (one launcher) | M\* — Fortran's three loop nests are one launcher with internal ordering; `vt`/`ut` overwrite semantics preserved |
| 28 | :1249–1255 | K11 | M |
| 29 | :1256–1262 | K12 | M |
| 30 | :1263–1269 | — | NL (`USE_COND`; fenced) |
| 31 | :1277–1423 | 3A.7 both `nord_k` arms, per-level filtered | M |
| 32 | :1425–1436 | 3A.7 `dcon_vort_diff` | M |
| 33 | :1439–1459 | 3A.6 `vorticity_abs` (`wk + f0`; the `f0*z_rat` `do_f3d` form is fenced) | M |
| 34 | :1461–1462 | 3A.5 scalar transport (`hord_vt`, no mass, no damping) | M |
| 35 | :1463–1472 | 3A.6 `dwind_update`, `u` and `v` in one launch | M\* — Fortran's two loop nests are one launch; their right-hand sides are mutually independent |
| 36 | :1476–1479 | 3A.7 vorticity damping, `d2 = vort`, fluxes → `ut`/`vt` | M |
| 37 | :1481–1520 | 3A.7 `dcon_scale` + `dcon_heat_source` | M |
| 38 | :1523–1534 | 3A.7 `dwind_damping_add` | M |

**Known discrepancies in the supported production path: NONE.** Every row is
M or M\*; the only NL rows are `#ifdef USE_COND` blocks that are not compiled in
the production configuration and are additionally fenced at the driver
boundary.

Corrections carried in from earlier milestones and verified this milestone:

- `del6_vt_flux` call-site pre-scaling with **`da_min_c`** (not `da_min`) and
  the raw-`damp` `1.E-5` gate — verified directly against `sw_core.F90:960–990`
  and `:1465–1485`. The 3A.3b handoff's warning is honoured.
- `w` gets **no** `deln_flux` (3A.5's trap) — honoured.
- The K3 corner-block comment was re-verified against `sw_core.F90:790–864`;
  statement 4 reads `uc(2,1)`, not `ut(2,1)`. An earlier working note claiming
  this comment was wrong is itself incorrect and no edit was applied.

---

## 14. Status, blockers, next phase

### Completion level

```
WP-B d_sw DRIVER:            COMPLETE for the production first-run configuration
  IMPLEMENTED                YES  (S1..S38, K1..K12)
  COMPILED                   YES  (R8 and R4, zero warnings)
  GPU EXECUTED               YES  (A100, direct + dispatch + fv3_gpu_dyn_core_step, R8 and R4)
  MEMORY-SANITIZED           memcheck 0 / racecheck 0 both precisions;
                             initcheck: one classified benign non-propagating finding
  NUMERICALLY VALIDATED      NO
```

**Numerical validation against authoritative runtime Fortran savepoints: NOT
RUN.**

`WP-B D_SW DRIVER COMPLETE` is **not** `FULL DYN_CORE GATE C1 COMPLETE`. Gate C1
requires every production-required dispatch slot to be implemented, wired and
GPU-executed through the integrated path; three are still `nullptr` and one is
wired but unreachable.

### Remaining Gate C1 blockers

1. **`cgrid_predictor` is `nullptr`** — production-required, not wired.
2. **`cgrid_vertical_solver` is `nullptr`** — production-required, not wired.
3. **`dgrid_height` is `nullptr`** — production-required, not wired.
4. **`dgrid_height_transport` is wired and separately GPU-executed but reached
   by nothing on the stage list** — its only caller is `launch_update_dz_d`,
   which `launch_vertical_solver` does not call.
5. **No `n_split` acoustic loop in the runtime.** The runtime walks its stage
   list once per `fv3_gpu_dyn_core_step` call;
   `dyn_core.F90:239–…` iterates `n_split` acoustic substeps. `dt` is currently
   corrected inside `launch_dgrid_dynamics_cuda` (`bdt/n_split`) as a stopgap.
6. **No halo-readiness declaration for `dgrid_dynamics`** (§9). Adding
   `require(FV3_GPU_H_CWIND)` needs a halo-exchange stage to exist first.
7. **`corner_edge_service` remains a fail-stub.** Not required by this path
   (§6), but required before any configuration that needs genuine
   halo/communication corner work.
8. **Prompt 6 authoritative Fortran savepoint comparison has not been run** for
   any WP-B stage.

### Follow-up hardening (not blockers)

- Widen or explicitly define the `d2` region `copy_corners` reads inside
  `launch_del6_vt_flux_cuda`, to clear the classified initcheck finding (§12)
  without changing any consumed value. Must be re-validated against the 3A.4
  height path.
- Public C ABI integer element type (`FV3_GPU_ELEMENT_I32`) so per-level
  integral orders stop travelling in `fv_real` storage.
- Retire or repair the legacy host `deln_flux_transport`
  (`wp_b_kernels.hpp:1183`), which performs no `copy_corners` rotation of `d2`
  and so does not reproduce `deln_flux` on a cubed sphere.

### Next recommended phase

Wire the remaining production-required WP-A slots (`cgrid_predictor`,
`cgrid_vertical_solver`) and `dgrid_height`, then add the `n_split` acoustic
loop, then close Gate C1 — before beginning Prompt 6 numerical validation.

---

## 15. Commands used

```bash
# environment (login node)
module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv comp/gcc/12.3.0 mpi/openmpi/4.1.6/gcc-12.3.0 \
            nvidia/nvhpc-byo-compiler/24.7

# reconfigure is required after adding a CMake test target
cmake build_cuda_r8 -B build_cuda_r8
cmake build_cuda_r4 -B build_cuda_r4
cmake --build build_cuda_r8 -j 12
cmake --build build_cuda_r4 -j 12

# A100 CTest + sanitizers (Slurm; job 58069277 on warpa003)
sbatch slurm_logs/3a8_fix_verify.sh     # ctest R8/R4, then
                                        # compute-sanitizer memcheck/initcheck/racecheck
                                        # on fv3_dispatch_smoke_test in both precisions
```

The sanitizer binary must be invoked by absolute path:
`/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/compute-sanitizer`.
