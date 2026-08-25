# Milestone 3A.5 — WP-B.2 Arbitrary Scalar Transport and D3 Correction

**Date:** 2026-08-25
**Branch:** `cuda/gpu-bringup`
**Starting HEAD:** `4548d72b7162d4e7deaffacfe7ceafd22c68ab0d`
(Complete authoritative WP-B height transport repair)
**Working-tree state at start:** clean.
**Final commit:** "Implement authoritative WP-B scalar transport CUDA" — the
single milestone commit whose parent is `4548d72`. This document is part of that
commit, so it cannot name its own SHA; resolve it with
`git log --oneline 4548d72..cuda/gpu-bringup`. **Not pushed.**

**Authoritative Fortran revision used:** `GEOS-ESM/GFDL_atmos_cubed_sphere`,
tag `geos/v3.0.0`, SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45`, read-only
clone at
`/discover/nobackup/projects/imvi-nse/jli30/fv3_reference/GFDL_atmos_cubed_sphere`.

**Numerical validation against runtime Fortran savepoints: NOT RUN.** Nothing in
this document is authoritative numerical validation. No Fortran-generated
savepoint was produced or compared. That is Prompt 6. What this milestone
establishes is source-level agreement with the authoritative Fortran, GPU
execution on a real A100 in both precisions, sanitizer cleanliness, and targeted
agreement between each CUDA stage and an independent host transcription of the
corresponding Fortran.

---

## Objective

Milestone 3A.4 delivered the authoritative `update_dz_d` height-transport path,
whose damping operator is `sw_core.F90`'s `del6_vt_flux`. It left one recorded
defect, **D3**: the WP-B `deln_flux` entry point still used a raw damping
coefficient instead of `tp_core.F90`'s `(damp_c * da_min) ** (nord + 1)`, and
omitted the `copy_corners` rotations of the working array `d2`.

WP-B.2 does two things:

1. Repairs `deln_flux` to the authoritative `tp_core.F90:1263` semantics (D3).
2. Adds a reusable **arbitrary scalar transport** service — the `fv_tp_2d`
   + `deln_flux` composition that `d_sw` invokes four times per level — so that
   the later `dgrid_dynamics` slice can consume it without a redesign.

Explicitly **out of scope** and not started: `xtp_u` / `ytp_v`, momentum and
vorticity transport, the kinetic-energy / heat-source terms, divergence damping,
the D-wind update, the full `d_sw` driver, a generalised `corner_edge_service`,
production `dgrid_dynamics` activation, and Prompt 6 validation.

---

## 1. Authoritative scalar transport call flow

Traced at the SHA above.

```
sw_core.F90:d_sw
  |
  |-- 939  fv_tp_2d(delp, crx_adv, cry_adv, hord_dp, fx, fy,
  |          xfx_adv, yfx_adv, ra_x, ra_y, nord=nord_v, damp_c=damp_v)
  |            ^ no mfx/mfy  -> flux weighting by xfx_adv/yfx_adv
  |            ^ no mass     -> deln_flux WITHOUT mass weighting
  |
  |-- 986  fv_tp_2d(w, ..., hord_vt, gx, gy, xfx_adv, yfx_adv, ra_x, ra_y,
  |          mfx=fx, mfy=fy)
  |            ^ mfx/mfy present, nord/damp_c ABSENT -> no deln_flux at all
  |
  |-- 993  fv_tp_2d(q_con, ..., hord_dp, ..., mfx=fx, mfy=fy, mass=delp,
  |          nord=nord_t, damp_c=damp_t)
  |
  |-- 1004 fv_tp_2d(pt, ..., hord_tm, ..., mfx=fx, mfy=fy, mass=delp,
             nord=nord_t, damp_c=damp_t)
                   ^ mfx/mfy + mass -> mass-weighted deln_flux

tp_core.F90:fv_tp_2d (110-241)
  |
  |-- copy_corners(q, dir=2);  yppm(fy2, q, cry, jord)          [Y first]
  |-- ra_y-weighted q_i update                                  (166-171)
  |-- xppm(fx,  q_i, crx, ord_ou)                               (176)
  |
  |-- copy_corners(q, dir=1);  xppm(fx2, q, crx, iord)          [X first]
  |-- ra_x-weighted q_j update                                  (188-193)
  |-- yppm(fy,  q_j, cry, ord_ou)                               (195)
  |
  |-- flux averaging: fx = 0.5*(fx + fx2) * {mfx | xfx}         (201-234)
  |                   fy = 0.5*(fy + fy2) * {mfy | yfx}
  |
  `-- if present(nord) .and. present(damp_c) .and. damp_c > 1.e-4:
          damp = (damp_c * gridstruct%da_min) ** (nord + 1)
          deln_flux(nord, ..., damp, q, fx, fy, [mass])          (215-240)

tp_core.F90:xppm / yppm
  |-- interior order-6 al: p1*(q(i-1)+q(i)) + p2*(q(i-2)+q(i+1)),
  |     p1 = 7/12, p2 = -1/12                                    (446-450)
  |-- panel-edge one-sided al with c1=-2/14, c2=11/14, c3=5/14,
  |     gated on (.not.nested .and. grid_type < 3)               (452-...)
  |-- mord == 6 limiter: smt5 = 3*|b0| < |bl - br|               (523-529)
  |-- WMP sign-test override at the four seam cells:
  |     smt5 = bl*br < 0                                         (532-541)
  |-- iord < 0: al = max(0., al) over is-1..ie+2, AFTER the edge
  |     overrides                                                (390-394)
  `-- flux loop                                                  (546-555)

tp_core.F90:deln_flux (1263-1411)   <-- the D3 subject
```

`q` is `intent(inout)` in `fv_tp_2d` and is mutated by the two `copy_corners`
calls (dir=2 then dir=1). The **twice-mutated** `q` is what `deln_flux`
receives. This is preserved in the CUDA path.

`edge_profile` is *not* on the scalar transport path — it belongs to
`update_dz_d` (`nh_utils.F90`), which 3A.4 owns. See §9.

---

## 2. The WP-B.2 contract

Declared in `cuda/dyn_core/dgrid/wp_b_cuda.cuh`.

`HeightTransportBounds` was renamed `TransportBounds` and given an explicit
`nk`, because the two contracts differ in vertical extent:

| | height transport (3A.4) | scalar transport (3A.5) |
|---|---|---|
| `TransportBounds::nk` | `npz + 1` (interface levels) | `npz` (cell levels) |
| damping operator | `del6_vt_flux` (`sw_core.F90:1544`) | `deln_flux` (`tp_core.F90:1263`) |
| damping coefficient | raw per-level `damp(k)` | `(damp_c*da_min)**(nord+1)`, level-independent |
| diffusive flux delivery | separate `fx2`/`fy2` outputs | folded into `fx`/`fy` |
| `LevelFilter` | per-level damped/undamped split | mode 0, all levels |
| `edge_profile` | yes | no |
| state update | area-weighted `zh` renormalisation | none — fluxes are the output |
| workspace arena | yes | caller-supplied views |

```c++
struct ScalarTransportOptions {
  int32_t  hord               = 0;  // signed; hord<0 also applies the al>=0 clamp
  int32_t  mass_flux_weighting = 0; // 0 -> xfx_adv/yfx_adv, 1 -> mfx/mfy
  int32_t  deln_flux_enabled  = 0;  // mirrors present(nord).and.present(damp_c)
  int32_t  nord               = 0;
  fv_real  damp_c             = 0;
  fv_real  da_min             = 0;
};

fv3_gpu_status launch_scalar_transport_cuda(
    const TransportBounds &b, const ScalarTransportOptions &opt,
    const LevelFilter &lf,
    View3<fv_real> q,                      // intent(inout), mutated by copy_corners
    View3<const fv_real> mass,             // delp, or empty when unweighted
    View3<const fv_real> crx_adv, cry_adv, // Courant numbers
    View3<const fv_real> xfx_adv, yfx_adv, // advective mass fluxes
    View3<const fv_real> mfx, mfy,         // mass fluxes (pt/tracer/q_con path)
    View2<const fv_real> area, rarea, dxa, dya, del6_u, del6_v,
    View3<const fv_real> ra_x, ra_y,
    View3<fv_real> q_i, q_j, fx2_tp, fy2_tp,   // fv_tp_2d workspaces
    View3<fv_real> d2, fx2_dn, fy2_dn,         // deln_flux workspaces
    View3<fv_real> fx, fy,                     // outputs
    void *stream) noexcept;
```

Design decisions:

- **The scalar field is a view parameter, never a hard-coded field.** The four
  authoritative `d_sw` modes differ only in `(hord, weighting, mass, nord,
  damp_c)`, all of which live in `ScalarTransportOptions`. Nothing in the
  kernels knows whether it is transporting `delp`, `pt`, `w`, or `q_con`.
- **Fluxes are the output; state updates are the caller's.** The authoritative
  update differs per scalar (`delp` is an additive flux divergence, `pt`/`w`/
  `q_con` are mass-weighted, `zh` is renormalised by the swept area), so folding
  a state update into this service would force the later `d_sw` driver to fight
  it.
- **Halo requirement:** `q` must be valid on `[isd, ied] x [jsd, jed]` with
  `ng >= nord + 1`; `deln_flux` reads `d2` on `[is-1-nord, ie+1+nord]²`.
- **Unsupported configurations are rejected before any launch,** so a rejected
  configuration can never leave model state partially mutated. Rejected:
  `abs(hord) != 6` (only the order-6 PPM path is ported) and, with damping
  enabled, `nord` outside `[0, ng-1]`.

The two contracts were deliberately kept distinct rather than merged; the height
path continues to call `launch_dgrid_height_transport_cuda` unchanged.

---

## 3. D3 resolved — authoritative `deln_flux`

`launch_deln_flux_transport_cuda` in `cuda/dyn_core/dgrid/wp_b_cuda.cu` was
rewritten against `tp_core.F90:1263-1411`.

| aspect | authoritative Fortran | before 3A.5 | now |
|---|---|---|---|
| coefficient | `damp = (damp_c*da_min)**(nord+1)`, computed in `fv_tp_2d:217/237` | raw `damp` | `std::pow(damp_c*da_min, nord+1)` |
| enable test | `damp_c > 1.e-4` in `fv_tp_2d` | absent | host-side gate before launch |
| `nord` range | 0,1,2 (`nord=3` "requires more ghosting") | unfenced | `nord` in `[0, ng-1]`, else `FV3_GPU_UNSUPPORTED_CONFIGURATION` |
| `d2` init | `damp*q` (no mass) or `q` (mass) over `[is-1-nord, ie+1+nord]²` | partial | full box, both branches |
| corner rotations | `nord>0`: 2 pre-loop + 2 per iteration = `2 + 2*nord`; `nord==0`: none | none | exact |
| first flux signs | `fx2 = del6_v*(d2(i-1)-d2(i))`, `fy2 = del6_u*(d2(j-1)-d2(j))` | matched | matched |
| iterated flux signs | sign flips to `d2(i)-d2(i-1)` / `d2(j)-d2(j-1)` | matched | matched |
| add, no mass | `fx += fx2` | matched | matched |
| add, mass | `damp2 = 0.5*damp`; `fx += damp2*(mass(i-1)+mass(i))*fx2` | matched | matched |

Two additional points worth recording:

- **`nord == 0` performs no corner rotation at all.** The Fortran guards the
  two pre-loop `copy_corners` calls with `if (nord > 0)` and the iteration loop
  does not execute. A port that unconditionally rotates would be wrong at
  `nord = 0`; test F pins this.
- **`da_min` is a cell *area*, and that is why the coefficient has the exponent
  it does.** Each del-2 iteration applies `rarea * del6_{u,v}`, i.e. roughly
  `1/area`, so `nord + 1` factors of `damp_c * da_min` are exactly what restore
  dimensional balance. This matters for test fixtures: a fixture using an
  unphysical `da_min` (say `1.0`) against `area ~ 1e8` drives the `nord=1`
  diffusive flux to `~1e-8` of the transport flux and `nord=2` to `~1e-16`,
  i.e. below round-off, and any liveness probe then silently measures nothing.
  The tests therefore use `GRID_DA_MIN = 1.0e8`, matching the fixture's cell
  area. This was a real fixture bug found and fixed during this milestone.

`del6_vt_flux` (height path) and `deln_flux` (scalar path) remain two separate
device entry points. Neither calls the other. The Fortran keeps them distinct
and so do we.

---

## 4. PPM reuse

**Reused unchanged from the corrected 3A.4 implementation** — no second copy was
created:

- `al_x_at` / `al_y_at`: interior order-6 `al`, the panel-edge one-sided `al`
  (`c1=-2/14`, `c2=11/14`, `c3=5/14`), the seam gating on
  `cubed_sphere && !nested`, and the `hord < 0` `al = max(0, al)` clamp applied
  *after* the edge overrides.
- `smt5_from`: the `mord == 6` limiter `3*|b0| < |bl - br|` and the WMP
  sign-test override `bl*br < 0`.
- `seam_cell_x` / `seam_cell_y`: which four cells take the WMP override.
- `xppm_face_flux` / `yppm_face_flux`: the flux loop.
- `copy_corners` device kernels (dir = 1 and dir = 2).
- The corrected `ra_x` / `ra_y` extents.

**Generalised, not duplicated:** `launch_fv_tp_2d_cuda` previously hard-wired the
flux-averaging weights to `xfx_adv`/`yfx_adv`. It now takes `View3<const fv_real>
wx, wy` after `yfx_adv`, and the caller chooses. `wp_b_contract_cuda.cpp` (the
height path) passes `xfx_adv`/`yfx_adv` at both call sites, so its behaviour is
byte-identical to 3A.4; the scalar path passes `mfx`/`mfy` when
`mass_flux_weighting == 1`. That single parameter is the whole difference between
`fv_tp_2d`'s two flux-averaging branches.

**Scalar-specific and therefore new:** the `deln_flux` kernel family
(`deln_init_kernel`, `deln_flux_x_kernel` / `deln_flux_y_kernel` — merged behind
a `bool flipped` for the iterated sign change — `deln_iter_div_kernel`,
`deln_add_no_mass_kernel`, `deln_add_mass_kernel`) and the
`launch_scalar_transport_cuda` composition.

`edge_profile` is not reused because it is not on this call path (§9).

---

## 5. Mass and positivity semantics

Audited against the four `d_sw` call sites and implemented exactly as they
require — no more:

- **Flux weighting.** `mfx`/`mfy` when the Fortran passes them, else
  `xfx_adv`/`yfx_adv`. `delp` uses the advective pair; `w`, `pt`, `q_con` use
  the mass pair.
- **Mass-weighted `deln_flux`.** Applies only when the Fortran has *both*
  `mfx/mfy` and `mass` present. `w` has `mfx/mfy` but no `nord`/`damp_c`, so `w`
  gets no `deln_flux` at all — a case that is easy to get wrong by assuming
  "mass fluxes imply mass-weighted damping".
- **`hord_dp = -6`.** The negative sign does not select a different PPM order; it
  adds `al = max(0, al)` over `is-1 .. ie+2` *after* the panel-edge one-sided
  values are written. Test H confirms this branch is live: 36 unclamped `al`
  values are negative in the fixture and 5 flux cells change.
- **Limiter.** `mord == 6` uses `3*|b0| < |bl - br|`, overridden by the WMP sign
  test `bl*br < 0` at the four seam cells. The monotone `iord >= 7` /
  `pert_ppm` paths are *not* ported and are rejected by the `abs(hord) != 6`
  fence, because the production configuration does not use them.

---

## 6. Edge and corner semantics

Following 3A.3B's conclusion that corner/edge handling is not one monolithic
service, this milestone reuses the device-level corner behaviour directly and
does **not** route anything through the generic `corner_edge_service` dispatch
slot (which remains `corner_service_not_implemented`).

What scalar transport actually needs, and nothing else:

- `copy_corners(q, dir=2)` then `copy_corners(q, dir=1)` inside `fv_tp_2d`,
  mutating `q` in place.
- `2 + 2*nord` `copy_corners` rotations of `d2` inside `deln_flux` when
  `nord > 0`; none when `nord == 0`.
- The panel-edge one-sided PPM stencil, gated on `cubed_sphere && !nested`.
- Corner-flag suppression under `nested` (test C: 0 cells rotated).

Broader corner/edge unification stays WP-B.4 work.

---

## 7. Tests

New permanent test: `cuda/dyn_core/tests/dgrid/test_wp_b2_scalar_transport_cuda.cpp`,
registered as CTest `fv3_wp_b2_scalar_transport` (label `gpu`,
`SKIP_RETURN_CODE 77`). It is a separate executable from
`fv3_wp_b_cuda_smoke_test` because the contracts are distinct (§2).

The test carries an **independent host transcription** of the Fortran
(`h_al_x`, `h_al_y`, `h_smt5`, `h_seam_cell_x/y`, `h_xppm`, `h_yppm`,
`h_copy_corners`, `h_fv_tp_2d`, `h_deln_flux`) written from `tp_core.F90`, not
from the CUDA source, and compares device output against it. Workspaces are
NaN-poisoned outside the Fortran's own loop bounds so an out-of-bounds write is
caught rather than tolerated. Fields are non-degenerate (tilted, wavy, strictly
positive where the physics requires it).

| part | what it pins |
|---|---|
| A | interior order-6 PPM transport, both sweeps, both flux components |
| B | panel-edge one-sided PPM; `fx2 = xppm(q)`, `fy2 = yppm(q)`; seam liveness (159 of 468 cells) |
| C | `copy_corners` rotation liveness (108 halo cells) and `nested` suppression (0) |
| D | `deln_flux` enabled at `nord = 0, 1, 2`, each verified live |
| E | `deln_flux` disabled: `damp_c = 0`, `1e-5`, `1e-4` (== threshold, must be a no-op), `2e-4` (> threshold, must damp); plus `nord` inert when damping is off |
| F | `nord`-dependent behaviour: orders mutually distinct; `nord=0` performs no rotation; `nord=1` rotates corners |
| G | the `(damp_c*da_min)**(nord+1)` coefficient law, and the `del6_vt_flux` discriminator (below) |
| H | mass-flux weighting, mass-weighted `deln_flux`, all four `d_sw` modes, `hord<0` positivity clamp, limiter-branch admissibility |
| I | R8 full path |
| J | R4 full path |
| K | unsupported-configuration rejection: `hord = 8, 10, 0`, `nord = 3, -1`, each with `q` verified untouched |

**The `del6_vt_flux` discriminator (§8 of the task).** Test G exploits the fact
that the two damping operators depend on their coefficient differently. Under
`deln_flux`, the diffusive flux scales as `(damp_c * da_min)**(nord+1)`, so
doubling `damp_c` multiplies it by `2**(nord+1)`: `x2` at `nord=0`, `x4` at
`nord=1`, `x8` at `nord=2`. Under raw `del6_vt_flux` semantics it would scale
linearly — `x2` always. G asserts the exponential law directly (worst deviation
`0.00e+00` over 312 cells for `damp_c` and for `da_min`, at `nord = 0, 1, 2`)
and separately asserts "not raw-damp linear (x4, not x2)" and "(x8, not x2)".
If `del6_vt_flux` semantics were substituted, G fails.

### The PPM limiter branch indeterminacy

Investigating an initial Part H mismatch (7/702 cells, worst rel `1.0e-3` in R4;
5/702, worst rel `4.3e-4` in R8 — a magnitude that does *not* shrink with
precision, so not round-off accumulation) produced a finding worth recording.

Where the transported field is locally constant along the sweep direction,
`al(i)` is mathematically *exactly* `q(i)`, so `bl = al(i) - q(i)` and
`br = al(i+1) - q(i)` are pure round-off and their computed signs depend on fma
contraction order. Adjacent to a rotated panel corner one of the pair can be
O(1) while the other is round-off, which makes the `mord == 6` seam override
`smt5 = bl*br < 0` a round-off-decided branch while the flux difference between
the two branches, `fx1 ~ -(1-c)*c*bl`, is O(1e-4).

This is an inherent conditional instability of the FV3 `mord = 6` limiter, not a
port defect: two Fortran builds with different fma contraction disagree at the
same cells in the same way. The CUDA `al_x_at` / `xppm_face_flux` and the host
`h_al_x` / `h_xppm` were read side by side and are line-for-line identical.

Rather than hide it, the test confronts it. `h_xppm_alt` returns *both*
admissible flux values plus a `decided` flag (a branch is "decided" only when
`bl`, `br` are resolved to more than `32 * eps` relative to `q`). Subtest
**"H limiter branch admissible"** runs an exactly-plateau field through both
`copy_corners` rotations and requires every device flux cell to equal one of the
two admissible branch values; **"H tie precondition"** requires that the
round-off-undecided population is non-empty, so the test cannot silently stop
exercising the tie. Observed: 702 cells all admissible, 648 undecided, 208 (R8)
/ 204 (R4) on the suppressed branch. The well-posed fixtures additionally carry
a `0.004*i` tilt so that every limiter decision there *is* well posed.

---

## 8. Build, GPU execution, sanitizers

Environment: the validated Discover stack — GCC/G++/GFortran 12.3.0, OpenMPI
4.1.6, CUDA 12.5 (`nvidia/nvhpc-byo-compiler/24.7`), CMake 3.30.3.

**Build.** `build_cuda_r8/` and `build_cuda_r4/`, out of source, both clean with
zero errors and zero warnings.

**GPU execution.** Discover `gpu_a100`, `--constraint=rome`, 1 GPU, 60G, 12 CPUs,
node `warpa003`, A100-SXM4-40GB (sm_80).

Full CTest suite, run on the GPU node so the four `gpu`-labelled tests actually
execute rather than skip:

```
build_cuda_r8:  100% tests passed, 0 tests failed out of 14
build_cuda_r4:  100% tests passed, 0 tests failed out of 14
```

`fv3_wp_b2_scalar_transport_test` standalone: **ALL PARTS PASS (11 parts)** in
both precisions. Representative worst relative errors against the host
transcription:

| | R8 | R4 |
|---|---|---|
| interior order-6 `fx` (A) | `4.35e-16` | `2.85e-07` |
| panel-edge seam `fx` (B) | — | `1.98e-07` |
| all four `d_sw` modes (H) | `<= 5.85e-16` | `<= 2.18e-07` |
| full path (I / J) | `4.35e-16` | — |

i.e. at round-off for the respective precision. Outputs were finite and
non-degenerate throughout (flux ranges printed per subtest, e.g.
`[4.97e+08, 7.52e+08]`), and every liveness probe reported a non-zero changed-cell
count. No CUDA runtime error was reported in any configuration.

**compute-sanitizer** (`.../24.7/cuda/12.5/compute-sanitizer/compute-sanitizer`,
`--error-exitcode 9`), on `fv3_wp_b2_scalar_transport_test`:

| tool | R8 | R4 |
|---|---|---|
| `memcheck` | 0 errors | 0 errors |
| `initcheck` | 0 errors | 0 errors |
| `racecheck` | 0 hazards, 0 errors, 0 warnings | 0 hazards, 0 errors, 0 warnings |

`racecheck` was run because the `copy_corners` rotations and the panel-edge
stencil write into overlapping halo regions. No unresolved sanitizer defect
remains.

---

## 9. Source-level Fortran reconciliation

Classification of every element in WP-B.2 scope. This is source-level
reconciliation, **not** numerical validation.

| element | authoritative source | status | note |
|---|---|---|---|
| PPM interior order-6 `al` | `tp_core.F90:446-450` | **MATCH** | `p1 = 7/12`, `p2 = -1/12`; clipped bounds `is1 = max(3,is-1)`, `ie3 = min(npx-2,ie+2)`, `ie1 = min(npx-3,ie+1)` reproduced |
| PPM panel-edge `al` | `tp_core.F90:452-...` | **MATCH** | `c1 = -2/14`, `c2 = 11/14`, `c3 = 5/14`; `dxa`/`dya`-weighted one-sided pair at the `al(1)` cell; gated on `.not.nested .and. grid_type < 3` |
| `mord == 6` limiter | `tp_core.F90:523-529` | **MATCH** | `smt5 = 3*\|b0\| < \|bl - br\|` |
| WMP seam override | `tp_core.F90:532-541` | **MATCH** | `smt5 = bl*br < 0` at the four seam cells. Branch selection is round-off-decided where the field is locally constant; inherent to the algorithm, see §7 |
| `iord < 0` positivity clamp | `tp_core.F90:390-394` | **MATCH** | `al = max(0, al)` over `is-1 .. ie+2`, applied after the edge overrides |
| `copy_corners` | `tp_core.F90` | **MATCH** | dir = 1 and dir = 2; `nested` suppression |
| `fv_tp_2d` two-pass composition | `tp_core.F90:110-241` | **MATCH** | Y-X then X-Y, `q_i`/`q_j` intermediates, `ra_x`/`ra_y` weighting, `0.5`-averaged flux, `q` mutated in place |
| flux-averaging weight selection | `tp_core.F90:201-234` | **MATCH** | `mfx`/`mfy` vs `xfx`/`yfx`, chosen by `mass_flux_weighting` |
| `deln_flux` | `tp_core.F90:1263-1411` | **MATCH** | coefficient, threshold, `d2` box, rotation count, both flux sign conventions, both add branches — see the §3 table |
| mass weighting | `tp_core.F90:1385-1397` | **MATCH** | `damp2 = 0.5*damp`; applies only when `mfx/mfy` *and* `mass` are present |
| `edge_profile` | `nh_utils.F90` | **N/A** | not on the scalar transport path; owned by 3A.4's `update_dz_d` |
| `pert_ppm`, `iord >= 7` monotone paths | `tp_core.F90:560+`, `1180+` | **N/A** | not required by the production configuration; rejected by the `abs(hord) != 6` fence rather than silently mis-executed |
| `nord = 3` (del-8) | `tp_core.F90:1269` | **STRUCTURAL DIFFERENCE** | the Fortran comment says it "requires more ghosting than current"; CUDA returns `FV3_GPU_UNSUPPORTED_CONFIGURATION` for `nord` outside `[0, ng-1]` instead of reading past the halo |
| `USE_SG` `sin_sg` flux form | `tp_core.F90:1319-1332` | **STRUCTURAL DIFFERENCE** | the `#ifdef USE_SG` branch is not compiled in the GEOS configuration; CUDA implements the `del6_u`/`del6_v` form only |
| per-level workspace layout | — | **STRUCTURAL DIFFERENCE** | Fortran `deln_flux` uses 2-D `k`-private locals; CUDA uses 3-D `d2`/`fx2_dn`/`fy2_dn` views so all levels run concurrently. Same arithmetic per level |

**Known discrepancies inside WP-B.2 scope: NONE.**

### Legacy host `deln_flux_transport`

`cuda/dyn_core/dgrid/wp_b_kernels.hpp:1183` carries a host-side
`deln_flux_transport`. Its coefficient is correct
(`std::pow(damp * da_min, nord + 1)`) and its flux/add structure matches, but it
performs **no `copy_corners` rotation of `d2`** and so does not reproduce
`deln_flux` on a cubed sphere. It is reachable only from
`tests/dgrid/test_wp_b_kernels.cpp` — nothing in the production dispatch or in
any CUDA translation unit includes that header. It is left in place rather than
edited opportunistically; the device path
(`launch_deln_flux_transport_cuda`) is authoritative and is what
`launch_scalar_transport_cuda` calls. Retiring or fixing the host copy belongs
with the WP-B.4 corner/edge slice.

---

## 10. Production dispatch boundary

Unchanged where the task requires it:

- `d.dgrid_height_transport = &launch_dgrid_height_transport_cuda;` — still the
  corrected 3A.4 path.
- `d.dgrid_dynamics = nullptr;` — **not** populated. `launch_scalar_transport_cuda`
  is a reusable *internal* WP-B service, not a dispatch entry. `d_sw` still needs
  momentum/vorticity transport, divergence damping, the kinetic-energy /
  heat-source terms and the order-critical `pt`/`delp`/`w` state updates; wiring
  a partial `dgrid_dynamics` would let the production path silently skip them.
  The comment in `fv3_gpu_production_dispatch.cu` records this.
- `d.corner_edge_service = &corner_service_not_implemented;` — unchanged.

The stale 3A.4 note claiming `launch_deln_flux_transport_cuda` is merely
"retained for later stages" was updated: it is now repaired and consumed, and is
still never invoked from the height path.

---

## 11. Status summary

| level | status |
|---|---|
| IMPLEMENTED | arbitrary scalar transport + authoritative `deln_flux`: **YES** |
| COMPILED | R8: **PASS**. R4: **PASS**. Zero warnings |
| GPU EXECUTED | A100 `warpa003`, R8: **PASS** (14/14 CTest, 11/11 parts). R4: **PASS** (14/14, 11/11) |
| MEMORY-SANITIZED | `memcheck` / `initcheck` / `racecheck`, both precisions: **0 errors** |
| NUMERICALLY VALIDATED | **NOT RUN** — no Fortran runtime savepoint comparison. Prompt 6 |

---

## 12. Remaining WP-B work

1. **WP-B.3 — momentum transport.** `xtp_u`, `ytp_v`, the `ub`/`vb` D-grid wind
   update, and the kinetic-energy terms. Not started; no helper from this
   milestone leaks into it.
2. **WP-B.4 — corner/edge unification.** Decide the real shape of
   `corner_edge_service`, and retire or repair the legacy host
   `deln_flux_transport` in `wp_b_kernels.hpp` (§9).
3. **WP-B.5 — the `d_sw` driver.** Divergence damping, the heat source, the
   order-critical `pt`/`delp`/`w` updates, and the `cx`/`cy`/`xflux`/`yflux`
   flux-capacitor accumulation. Only then may `d.dgrid_dynamics` be populated.
4. **Prompt 6.** Fortran savepoint generation and first-divergence comparison at
   the documented dyn_core savepoints.

---

## Next recommended phase

WP-B.3 — the momentum transport slice (`xtp_u` / `ytp_v`), consuming the PPM
infrastructure this milestone generalised.
