# Milestone 3A.3B — Authoritative WP-B Numerical Reconciliation

**Date:** 2026-08-24
**Branch:** `cuda/gpu-bringup`
**HEAD at start:** `ad1d97f` (working tree clean; 3A.1/3A.2/3A.3 committed as
`14f85ae` + `ad1d97f` by the user before this milestone began)
**Nature of this milestone:** AUDIT / NUMERICAL MAPPING. Read-only with respect
to all numerical source.

**Numerical validation: NOT RUN.** Nothing in this document is a numerical
validation. Every finding below is derived from *source inspection* of the
authoritative Fortran and of the existing CUDA. No Fortran reference or
savepoint data was generated or compared. That is Prompt 6.

This document does not overwrite `milestone_3a3_wp_b_cuda.md`. Where 3A.3's
findings are superseded, that is stated explicitly here and annotated there.

---

## Section 1 — Authoritative FV3 source: RESOLVED

`@fvdycore` is **absent** from this repository and remains absent. It is a
`mepo` mount point, not a git submodule:

- no `.gitmodules`, no `git submodule status` output;
- `.gitignore` explicitly lists `/@fvdycore`, `/fvdycore`, `/fvdycore@`;
- `CMakeLists.txt` uses `esma_add_library(${this} SRCS ${srcs} SUBCOMPONENTS fvdycore ...)`;
- there is no `components.yaml` in this repository — the revision pin lives in
  the **parent** fixture (`GEOSfvdycore` / `GEOSgcm`).

### Revision pin — three independent sources agree

| Source | `fvdycore` pin |
|---|---|
| `/discover/nobackup/projects/imvi-nse/jli30/GEOSgcm/components.yaml` | `remote: ../GFDL_atmos_cubed_sphere.git`, `tag: geos/v3.0.0` |
| `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore/components.yaml` | `remote: ../GFDL_atmos_cubed_sphere.git`, `tag: geos/v3.0.0` |
| `/discover/nobackup/projects/imvi-nse/jli30/GEOSgcm/.mepo/state.0.json` | `"remote": "git@github.com:GEOS-ESM/GFDL_atmos_cubed_sphere.git", "version": ["geos/v3.0.0", "t", true]` |

`GEOSfvdycore`'s `components.yaml` carries the identical `fvdycore` pin at tags
`v3.0.0`, `v3.1.0`, and `v3.2.0`, so the pin is stable across our branch's
upstream base date.

### Resolved revision

```text
repository: GEOS-ESM/GFDL_atmos_cubed_sphere
tag:        geos/v3.0.0
commit:     ca30e26305fda43831095334c02a3f5e0b8c0c45
            ("Merge pull request #127 from GEOS-ESM/geos/develop", 2026-05-27)
```

The tag resolves to the **same SHA** in both `GEOS-ESM` and the `dsg-atom`
fork, so there is no fork ambiguity. Cross-check from the other direction: this
repository's upstream merge-base `ac43844` is an ancestor of local tag `v3.0.1`,
which is exactly the `FVdycoreCubed_GridComp` version the parent fixtures pair
with `fvdycore: geos/v3.0.0`.

**No ambiguity. No arbitrary revision was used. No upstream default branch was
cloned.**

### Where the source was read from

A pristine, clean, detached checkout at exactly `geos/v3.0.0` **already existed
on Discover** inside the user's full GEOSgcm fixture:

```text
/discover/nobackup/projects/imvi-nse/jli30/GEOSgcm/src/Components/@GEOSgcm_GridComp/
  GEOSagcm_GridComp/GEOSsuperdyn_GridComp/@FVdycoreCubed_GridComp/@fvdycore
```
(`git status --porcelain` → 0 lines; `git tag --points-at HEAD` → `geos/v3.0.0`.)

Before that checkout was located, a separate read-only reference clone of the
same tag was made at
`/discover/nobackup/projects/imvi-nse/jli30/fv3_reference/GFDL_atmos_cubed_sphere`
(HEAD `ca30e26`, depth-1). Both trees are at the identical SHA and were used
interchangeably; findings below cite `model/<file>.F90:<line>` which is valid in
either.

**`@fvdycore` inside this working tree was deliberately NOT populated.** The
audit needs read access only, and populating it would change build behaviour
through `esma_add_library(... SUBCOMPONENTS fvdycore ...)`. `git status`
therefore remains clean and the decision is fully reversible.

---

## Section 2 — Authoritative WP-B Fortran routine map

Traced by following actual call flow, not names. All paths relative to `model/`.

### Transport core (`tp_core.F90`)

| Routine | Definition | Called from | Calls |
|---|---|---|---|
| `fv_tp_2d(q, crx, cry, npx, npy, hord, fx, fy, xfx, yfx, gridstruct, bd, ra_x, ra_y, lim_fac, mfx, mfy, mass, nord, damp_c)` | `tp_core.F90:109-243` | `d_sw` ×6 (`sw_core.F90:939, 986, 994, 1004, 1023, 1461`); `update_dz_d` ×2 (`nh_utils.F90:286, 296`) | `copy_corners` ×2, `yppm` ×2, `xppm` ×2, `deln_flux` (conditional) |
| `copy_corners(q, npx, npy, dir, nested, bd, sw/se/nw/ne_corner)` | `tp_core.F90:247-322` | `fv_tp_2d:163` (`dir=2`), `fv_tp_2d:181` (`dir=1`); `del6_vt_flux` ×4; `deln_flux` ×3 | — |
| `xppm(flux, q, c, iord, is,ie,isd,ied, jfirst,jlast,jsd,jed, npx,npy, dxa, nested, grid_type, lim_fac)` | `tp_core.F90:324-709` | `fv_tp_2d:179` (`ord_ou`, on `q_i`), `fv_tp_2d:184` (`ord_in`, on `q`) | `pert_ppm` (high-order paths only) |
| `yppm(flux, q, c, jord, ifirst,ilast, isd,ied, js,je,jsd,jed, npx,npy, dya, nested, grid_type, lim_fac)` | `tp_core.F90:712-1147` | `fv_tp_2d:166` (`ord_in`, on `q`), `fv_tp_2d:195` (`ord_ou`, on `q_j`) | `pert_ppm` |
| `pert_ppm(im, a0, al, ar, iv)` | `tp_core.F90:1201-1260` | `xppm`/`yppm`, `iord>=8` paths | — |
| `deln_flux(nord, is,ie,js,je, npx,npy, damp, q, fx, fy, gridstruct, bd, mass)` | `tp_core.F90:1263-1411` | **only** `fv_tp_2d:218` (mass branch) and `fv_tp_2d:238` (no-mass branch) | `copy_corners` |

**`deln_flux` has exactly two call sites in the entire repository, both inside
`fv_tp_2d`.** It is unreachable except through `fv_tp_2d`'s *optional*
`nord`/`damp_c`(`/mass`) arguments. Verified by repo-wide grep.

### Damping / D-grid (`sw_core.F90`)

| Routine | Definition | Called from |
|---|---|---|
| `d_sw(...)` | `sw_core.F90:~520` | `dyn_core.F90:789` (inside the per-`k` OpenMP loop) |
| `del6_vt_flux(nord, npx, npy, damp, q, d2, fx2, fy2, gridstruct, bd)` | `sw_core.F90:1544-1655` | `sw_core.F90:974` (`w`), `sw_core.F90:1478` (vorticity), **`nh_utils.F90:288` (`zh`)** |
| `fill_4corners`, `divergence_corner`, `divergence_corner_nest` | `sw_core.F90` (public, `:97`) | `d_sw` / `dyn_core` |

### Height transport (`nh_utils.F90`)

| Routine | Definition | Called from |
|---|---|---|
| `update_dz_c(...)` | `nh_utils.F90:70-207` | `dyn_core.F90:577` (C-grid stage) |
| `update_dz_d(ndif, damp, hord, is,ie,js,je, km, ng, npx,npy, area, rarea, dp0, zs, zh, crx, cry, xfx, yfx, delz, ws, rdt, dz_min, gridstruct, bd, lim_fac)` | `nh_utils.F90:210-323` | `dyn_core.F90:905` (**one** call site) |
| `edge_profile(q1, q2, q1e, q2e, i1, i2, j1, j2, j, km, dp0, uniform_grid, limiter)` | `nh_utils.F90:1547-1653` | `update_dz_d:256` (`crx`/`xfx`), `update_dz_d:259` (`cry`/`yfx`) |

### Authoritative `update_dz_d` body — the routine 3A.3 actually implements

`nh_utils.F90:210-323`, in execution order:

1. `damp(km+1) = damp(km)`; `ndif(km+1) = ndif(km)` (`:247-248`).
2. **`edge_profile`** over `j=jsd..jed`: `crx,xfx (km) -> crx_adv,xfx_adv (km+1)`;
   and for `js<=j<=je+1`: `cry,yfx (km) -> cry_adv,yfx_adv (km+1)` (`:255-261`).
3. `do k=1,km+1` (`:267`):
   - `ra_x(i,j) = area(i,j) + (xfx_adv(i,j,k) - xfx_adv(i+1,j,k))` over
     `j=jsd..jed, i=is..ie` (`:269-273`);
   - `ra_y(i,j) = area(i,j) + (yfx_adv(i,j,k) - yfx_adv(i,j+1,k))` over
     `j=js..je, i=isd..ied` (`:274-278`);
   - **branch on `damp(k) > 1.E-5`** (`:280`):
     - **damped:** copy `z2 = zh(:,:,k)` over the full `isd:ied × jsd:jed` box;
       `fv_tp_2d(z2, crx_adv, cry_adv, ..., hord, fx, fy, xfx_adv, yfx_adv, ..., ra_x, ra_y, lim_fac)`
       — **no `nord`, no `damp_c`, no `mass`**; then
       `del6_vt_flux(ndif(k), npx, npy, damp(k), z2, wk2, fx2, fy2, ...)`; then
       ```fortran
       zh(i,j,k) = (z2(i,j)*area(i,j)+(fx(i,j)-fx(i+1,j))+(fy(i,j)-fy(i,j+1)))  &
                 / ((ra_x(i,j)+ra_y(i,j))-area(i,j))                            &
                 + ((fx2(i,j)-fx2(i+1,j))+(fy2(i,j)-fy2(i,j+1)))*rarea(i,j)
       ```
     - **undamped:** `fv_tp_2d(zh(isd,jsd,k), ...)` in place; then
       ```fortran
       zh(i,j,k) = (zh(i,j,k)*area(i,j)+(fx(i,j)-fx(i+1,j))+(fy(i,j)-fy(i,j+1))) &
                 / ((ra_x(i,j) + ra_y(i,j)) - area(i,j))
       ```
4. `ws(i,j) = (zs(i,j) - zh(i,j,km+1)) * rdt`; then **descending** `k=km..1`:
   `zh(i,j,k) = max(zh(i,j,k), zh(i,j,k+1) + dz_min)` (`:310-321`).

`delz` is declared `intent(out)` but **is never assigned** in the body. It is
produced downstream by `Riem_Solver3`.

---

## Section 3 — Reconciliation of the 3A.3 CUDA against authoritative Fortran

Scope: `cuda/dyn_core/dgrid/wp_b_cuda.cu`, `dgrid/wp_b_contract_cuda.cpp`, and
`vertical/fv3_vertical_kernels.cu:110-174` (`launch_update_dz_d`).

**No source was changed. Findings only.**

### Verdict table

| # | Stage / aspect | Verdict |
|---|---|---|
| 1 | PPM order-6 interior arithmetic (`ppm6_face_flux`) | **MATCHES FORTRAN** |
| 2 | `fv_tp_2d` internal Y-X / X-Y sequencing and pass structure | **MATCHES FORTRAN** |
| 3 | `fv_tp_2d` flux averaging (`0.5*(fx+fx2)*xfx`) | **MATCHES FORTRAN** |
| 4 | `q_i` / `q_j` intermediate formulas | **MATCHES FORTRAN** |
| 5 | Per-pass loop bounds / halo extents inside `fv_tp_2d` | **MATCHES FORTRAN** |
| 6 | Kernel-boundary decomposition, all-`k`-at-once launches, shared `q_work`/`fx2` arena | **MATCHES WITH EXPLAINED STRUCTURAL DIFFERENCE** |
| 7 | `hord < 0` positivity clamp on `al` | **MATCHES WITH EXPLAINED STRUCTURAL DIFFERENCE** |
| 8 | `lim_fac` unused for `mord==6` | **MATCHES FORTRAN** |
| 9 | **`edge_profile` (crx→crx_adv) entirely absent** | **DISCREPANCY** |
| 10 | **Flux-divergence update formula** | **DISCREPANCY** |
| 11 | **Damping routine is `deln_flux`, not `del6_vt_flux`** | **DISCREPANCY** |
| 12 | **`damp(k) > 1.E-5` two-branch structure absent** | **DISCREPANCY** |
| 13 | **`copy_corners` calls inside `fv_tp_2d` absent** | **DISCREPANCY** |
| 14 | **Panel-edge `al` formulas + `is1`/`ie3` clamping absent** | **DISCREPANCY** |
| 15 | **`smt5` panel-edge override absent** | **DISCREPANCY** |
| 16 | `ra_x`/`ra_y` computed over a superset box | **DISCREPANCY (extent)** |
| 17 | `delz` written by `height_boundary_v2_kernel` | **DISCREPANCY** |
| 18 | Descending monotonicity recurrence / `ws` | MATCHES FORTRAN |
| 19 | FP ordering / associativity | **MATCHES WITH EXPLAINED STRUCTURAL DIFFERENCE** |

### MATCHES — detail

**(1) PPM order-6 interior arithmetic.** Production `hord_tm = 6` enters `xppm`
with `iord=6`, so `iord < 7` (`tp_core.F90:365`) and `mord = 6`, falling to the
`else` block at `:523-529`:

```fortran
        do i=is-1,ie+1
           bl(i) = al(i)   - q1(i)
           br(i) = al(i+1) - q1(i)
           b0(i) = bl(i) + br(i)
           smt5(i) = 3.*abs(b0(i)) < abs(bl(i)-br(i))
        enddo
```
with `al(i) = p1*(q1(i-1)+q1(i)) + p2*(q1(i-2)+q1(i+1))` (`:370`,
`p1=7./12.`, `p2=-1./12.`) and the flux loop at `:546-555`:
```fortran
         if ( c(i,j) > 0. ) then
              fx1(i) = (1.-c(i,j))*(br(i-1) - c(i,j)*b0(i-1))
              flux(i,j) = q1(i-1)
         else
              fx1(i) = (1.+c(i,j))*(bl(i) + c(i,j)*b0(i))
              flux(i,j) = q1(i)
         endif
         if (smt5(i-1).or.smt5(i)) flux(i,j) = flux(i,j) + fx1(i)
```

`wp_b_cuda.cu:20-51` (`ppm6_face_flux`) reproduces this term for term. With the
substitution `q0=q1(i)`, `qm1=q1(i-1)`, `qm2=q1(i-2)`, `qm3=q1(i-3)`,
`qp1=q1(i+1)`, `qp2=q1(i+2)`: `al_m1≡al(i-1)`, `al_0≡al(i)`, `al_p1≡al(i+1)`,
`bl_m1≡bl(i-1)`, `br_m1≡br(i-1)`, `bl_0≡bl(i)`, `br_0≡br(i)`,
`smt5_m1≡smt5(i-1)`, `smt5_0≡smt5(i)`, and the donor/`fx1`/`smt5`-gate structure
is identical including the `c > 0` vs `c <= 0` split. Constants `7/12` and
`-1/12` match `tp_core.F90:95` (`p1`, `p2`). **The core numerics of the 3A.3
port are correct.**

**(2)-(5) `fv_tp_2d` structure and bounds.** `tp_core.F90:163-196` sequences:
`copy_corners(dir=2)` → `yppm(fy2, q, cry, ord_in, isd..ied)` → `fyy`/`q_i` →
`xppm(fx, q_i, crx, ord_ou, is..ie+1 × js..je)` → `copy_corners(dir=1)` →
`xppm(fx2, q, crx, ord_in, is..ie+1 × jsd..jed)` → `fx1`/`q_j` →
`yppm(fy, q_j, cry, ord_ou, is..ie × js..je+1)`. The six CUDA passes
(`wp_b_cuda.cu:89-175`, launched in order at `:324-353`) reproduce this
sequence and every loop bound:

| Fortran | CUDA kernel | Bounds |
|---|---|---|
| `yppm` `ord_in` (`:166`) | `fy2_from_qin_kernel` | `i=isd..ied`, `j=js..je+1` ✓ |
| `fyy`+`q_i` (`:168-177`) | `qwork_from_ray_kernel` | `i=isd..ied`, `j=js..je` ✓ |
| `xppm` `ord_ou` (`:179`) | `fx_from_qwork_kernel` | `i=is..ie+1`, `j=js..je` ✓ |
| `xppm` `ord_in` (`:184`) | `fx2_from_qin_kernel` | `i=is..ie+1`, `j=jsd..jed` ✓ |
| `fx1`+`q_j` (`:186-193`) | `qwork_from_rax_kernel` | `i=is..ie`, `j=jsd..jed` ✓ |
| `yppm` `ord_ou` (`:195`) | `fy_from_qwork_kernel` | `i=is..ie`, `j=js..je+1` ✓ |
| `fx`/`fy` averaging (`:225-233`) | `flux_weight_{x,y}_kernel` | ✓ |

Because `update_dz_d` passes **no** `mfx`/`mfy`, the authoritative flux
averaging takes the `else` branch (`:221-234`): `fx = 0.5*(fx+fx2)*xfx`,
`fy = 0.5*(fy+fy2)*yfx`. `wp_b_cuda.cu:185/195` matches. `ord_in = ord_ou =
hord` for `hord=6` (`:156-161`), so using one `hord` for all four PPM passes is
correct.

**(6) Structural difference — kernel decomposition.** Fortran runs a `k`-loop
with `private(z2, fx2, fy2, ra_x, ra_y, fx, fy, wk2)` 2D temporaries; the CUDA
launches each pass once over all `npz+1` levels with 3D workspace slices. The
`q_work` arena is reused between pass 2 (`q_i`) and pass 5 (`q_j`), which is
safe only because same-stream kernels execute in issue order and pass 3 has
already consumed pass 2's values. Semantics preserved; the reuse is correct but
fragile and should be commented at the arena definition.

**(7) `hord < 0` clamp.** Fortran clamps `al(i) = max(0., al(i))` over
`i=is-1..ie+2` as a separate pass (`:388-392`) *after* the panel-edge formulas
and *before* `bl`/`br`. The CUDA clamps `al_m1`/`al_0`/`al_p1` inside
`ppm6_face_flux` (`:27-31`). For interior points each `al` value is clamped
exactly once either way, so the result is identical; the difference is only that
the CUDA form recomputes rather than caches. Note `hord_tm = 6 > 0` in
production, so this path is not taken by `update_dz_d` at all — consistent with
3A.3's report that the negative-hord branch is NOT RUN.

**(19) FP ordering.** Fortran writes `(fx(i,j)-fx(i+1,j))+(fy(i,j)-fy(i,j+1))`
with explicit parentheses; the CUDA divergence expression drops them
(`wp_b_cuda.cu:294`). Under IEEE non-associativity these can differ in the last
bits. Not scientifically significant on its own, but it will matter for a
bitwise-comparison criterion in Prompt 6 and should be parenthesised to match.

### DISCREPANCIES — detail

**(9) `edge_profile` is entirely absent — the most consequential finding.**

Authoritative `update_dz_d` does **not** transport with `crx`/`cry`/`xfx`/`yfx`.
It first reconstructs interface-level Courant/flux arrays (`nh_utils.F90:255-261`):

```fortran
  do j=jsd,jed
     call edge_profile(crx, xfx, crx_adv, xfx_adv, is,  ie+1, jsd, jed, j, km, &
                            dp0, uniform_grid, 0)
     if ( j<=je+1 .and. j>=js )      &
     call edge_profile(cry, yfx, cry_adv, yfx_adv, isd, ied,  js, je+1, j, km, &
                            dp0, uniform_grid, 0)
  enddo
```

`crx`/`xfx` have `km` levels; `crx_adv`/`xfx_adv` have `km+1`
(`nh_utils.F90:225-226` vs `:232-233`). `edge_profile`
(`nh_utils.F90:1547-1653`) is a vertical reconstruction using the reference
pressure thickness `dp0(km)`, and `fv_tp_2d` is then called with
`crx_adv(is,jsd,k)` / `xfx_adv(is,jsd,k)` (`:286-287`, `:296-297`).

The CUDA has no equivalent. `wp_b_contract_cuda.cpp:110-113` requires
`covers3(out.crx, ..., 0, b.npz)`, i.e. `k ∈ [0, npz]` = `npz+1` levels, and the
kernels index `crx(i,j,k)` for `k` up to `npz` (`wp_b_cuda.cu:96, 125, 139`).
So the CUDA contract **silently expects `crx_adv`-shaped arrays registered under
the `FieldId::crx` name** — and `tests/dgrid/test_wp_b_cuda_smoke.cpp:220`
registers `crx`/`cry`/`xfx`/`yfx` with `NPZ1 = NPZ+1` levels, which is why the
smoke test passes and the sanitizer is clean.

Consequences:
- a required numerical stage (`edge_profile`, plus the `dp0` reference-pressure
  input and the `uniform_grid`/`limiter=0` options) is missing;
- the field contract conflates `crx` (D-grid, `npz`) with `crx_adv`
  (interface, `npz+1`). Wired against a real `d_sw` producing `npz`-level
  `crx`, the current kernels would read one level out of bounds;
- `NumericalConfig` carries no `dp0`/`dp_ref` array (it is built at
  `dyn_core.F90:310-313`), so the input is not plumbed either.

A host C++ translation **does already exist** —
`vertical/fv3_vertical_column.hpp:40`, `edge_profile_column`, documented as an
"Exact translation of `nh_utils.F90:edge_profile`". Its **only** caller is
`tests/vertical/test_vertical_column.cpp:15`. It has no CUDA kernel and is not
reachable from any dispatch path. Note also that
`vertical/fv3_vertical_kernels.cuh:24` claims step 1 of `launch_update_dz_d` is
"Pre-transport height preparation (edge_profile on zh)" — this comment is wrong
twice over: no `edge_profile` is called, and `edge_profile` operates on the
Courant/flux arrays, never on `zh`.

**(10) Flux-divergence update formula is not the authoritative one.**

Authoritative (`nh_utils.F90:300-301`, undamped branch):
```fortran
zh(i,j,k) = (zh(i,j,k)*area(i,j)+(fx(i,j)-fx(i+1,j))+(fy(i,j)-fy(i,j+1)))   &
          / ((ra_x(i,j) + ra_y(i,j)) - area(i,j))
```
CUDA (`wp_b_cuda.cu:294-296`):
```cpp
const fv_real div = (fx(i,j,k) - fx(i+1,j,k) + fy(i,j,k) - fy(i,j+1,k)) * rarea(i,j);
zh(i,j,k) = zh(i,j,k) + div;
```

These are different operators. The authoritative form is a **finite-volume
remap**: numerator `q*area + flux divergence`, denominator the *effective
advected area* `(ra_x + ra_y) - area`. The CUDA form is an **additive increment
scaled by `rarea`**. They agree only in the degenerate limit
`ra_x + ra_y - area → area` with small fluxes. This is exactly the risk 3A.3
flagged: `apply_height_flux_divergence_kernel` was "modeled on
`apply_mass_and_scalar_flux`'s divergence pattern" rather than translated from
the authoritative height update, and that model was the wrong one. The
`rarea` factor belongs only to the *damping* term (see below), not to the
transport term.

**(11) The damping routine is the wrong one — and this resolves the 3A.3
`damp_c` blocker.** See Section 4.

**(12) The `damp(k) > 1.E-5` branch structure is absent.** Authoritative has two
distinct code paths (`nh_utils.F90:280` / `:295`) that differ in more than a
coefficient: the damped path copies `zh` into `z2` over the full halo box first
and transports `z2`, so `fv_tp_2d`'s `copy_corners` side effects land on `z2`
and **not** on `zh`'s halo corners; the undamped path passes `zh` directly and
therefore *does* mutate `zh`'s halo corners. The CUDA implements neither: it
transports `zh` in place with no `copy_corners` at all, and has no branch.

**(13) `copy_corners` is absent.** `fv_tp_2d` calls
`copy_corners(q, npx, npy, 2, ...)` before the first `yppm` (`tp_core.F90:163`)
and `copy_corners(q, npx, npy, 1, ...)` before the second `xppm` (`:181`), both
guarded only by `.not. gridstruct%nested` — i.e. **active for our production
`grid_type = 0`**. `del6_vt_flux` adds four more `copy_corners` calls
(`sw_core.F90:1594, 1606, 1627, 1640`) under `nord > 0`, and production
`nord_v = min(2, nord) = 2`. None of these exist in the CUDA. This is the
concrete numerical content that `corner_edge_service` must supply (Section 8).

**(14) Panel-edge `al` formulas and `is1`/`ie3` clamping are absent.**
`tp_core.F90:349-355` restricts the interior `al` loop for `grid_type < 3`:
```fortran
 if ( .not. nested .and. grid_type<3 ) then
    is1 = max(3,is-1);  ie3 = min(npx-2,ie+2)
                        ie1 = min(npx-3,ie+1)
```
and `:373-386` replaces `al` at the tile edges with one-sided formulas using the
`c1 = -2./14.`, `c2 = 11./14.`, `c3 = 5./14.` constants (`tp_core.F90:~90`) and
a `dxa`-weighted expression:
```fortran
     if ( is==1 ) then
       al(0) = c1*q1(-2) + c2*q1(-1) + c3*q1(0)
       al(1) = 0.5*(((2.*dxa(0,j)+dxa(-1,j))*q1(0)-dxa(0,j)*q1(-1))/(dxa(-1,j)+dxa(0,j)) &
             +      ((2.*dxa(1,j)+dxa( 2,j))*q1(1)-dxa(1,j)*q1( 2))/(dxa(1, j)+dxa(2,j)))
       al(2) = c3*q1(1) + c2*q1(2) +c1*q1(3)
     endif
```
with the mirror image at `(ie+1)==npx`, and symmetrically in `yppm` using `dya`.
The CUDA computes the interior 6-point stencil everywhere and never
special-cases tile edges. It also does not resolve `dxa`/`dya` at all —
`bind_wp_b_height_transport` resolves only `area`, `rarea`, `del6_u`, `del6_v`
from `x.immutable->grid`, so the required metrics are not even available.

**(15) `smt5` panel-edge override is absent** (`tp_core.F90:532-541`):
```fortran
   if ( (.not.nested) .and. grid_type < 3) then
     if( is==1 ) then
       smt5(0) = bl(0)*br(0) < 0.
       smt5(1) = bl(1)*br(1) < 0.
     endif
     if( (ie+1)==npx ) then
        smt5(npx-1) = bl(npx-1)*br(npx-1) < 0.
        smt5(npx  ) = bl(npx  )*br(npx  ) < 0.
     endif
   endif
```
At tile edges the monotonicity switch changes from `3*|b0| < |bl-br|` to the
sign test `bl*br < 0`. The CUDA always uses the interior test.

Items (14) and (15) mean the CUDA is currently correct **only in the tile
interior**. Since the smoke test uses a single synthetic patch with no tile-edge
semantics, this was not detectable by any test run in 3A.3.

**(16) `ra_x`/`ra_y` extents.** Authoritative: `ra_x(is:ie, jsd:jed)`,
`ra_y(isd:ied, js:je)` — deliberately different boxes
(`nh_utils.F90:239-240, 269-278`). `launch_ra_xy_cuda`
(`wp_b_cuda.cu:301-312`) computes **both** over the full
`[is-h, ie+h] × [js-h, je+h]` box. The values consumed are a correct superset,
so numerics in the used region are unaffected; but with authoritative field
extents (`xfx(is:ie+1, jsd:jed)`, `yfx(isd:ied, js:je+1)`) the extra points read
`xfx(i+1,...)` / `yfx(...,j+1)` outside the declared array. The smoke test masks
this by registering oversized `(NI_H+1) × (NJ_H+1)` face boxes.

**(17) `delz`.** `height_boundary_v2_kernel` writes
`delz[k] = zh[k+1] - zh[k]` (per `fv3_vertical_kernels.cu:163-166` comments and
the kernel's own `delz` output). Authoritative `update_dz_d` **never assigns
`delz`** despite its `intent(out)` declaration; `delz` is produced by
`Riem_Solver3` (`dyn_core.F90:918`). Writing it in the CUDA `update_dz_d` is an
extra side effect not present in the reference. Whether it is harmless depends
on whether the CUDA `Riem_Solver3` overwrites `delz` unconditionally — not
determined in this audit.

**(18) MATCHES FORTRAN — monotonicity recurrence and `ws`.**
Initially classified UNRESOLVED, then resolved by reading
`vertical/fv3_vertical_kernels.cu:32-37` (`height_boundary_v2_kernel`).
Authoritative requires `ws = (zs - zh(km+1))*rdt` computed **before** the
clamp, then a strictly **descending** `k = km..1` recurrence
`zh(k) = max(zh(k), zh(k+1) + dz_min)` — sequential in `k`, so a naive
parallel-over-`k` kernel would be wrong. The CUDA kernel is correct on both
counts: it assigns `ws.at(i,j) = (zs.at(i,j) - zh.at(i,j,km))*rdt` first, then
runs `for (int k = km-1; k >= 0; --k)` with one thread per `(i,j)` and the
recurrence serial within the thread. The `max` is written as an explicit
`if (zh < floor) zh = floor` — identical semantics.

The same descending pattern appears in the host reference
`vertical/fv3_vertical_column.hpp::update_height_boundary_column`, which agrees
with the kernel.

The *only* deviation in this kernel is the extra `delz` write, already recorded
separately as discrepancy 17.

---

## Section 4 — `damp_c` / `vtdm4`: RESOLVED

**Verdict: DIFFERENT VARIABLES** (with, additionally, a missing per-level
contract element).

The 3A.3 handoff recorded this as an open ambiguity — "no config plumbing exists
to carry a `damp_c` value" — and deferred wiring `deln_flux_transport`. The
authoritative source shows the premise was wrong: **the height/`zh` path never
calls `deln_flux` at all.** There is no missing `damp_c`; `deln_flux` is simply
the wrong routine for this call site.

### The two damping routines are genuinely different

| | `deln_flux` (`tp_core.F90:1263`) | `del6_vt_flux` (`sw_core.F90:1544`) |
|---|---|---|
| Reached from | only `fv_tp_2d:218/238`, via optional `nord`+`damp_c` | direct calls |
| Coefficient | `damp = (damp_c * gridstruct%da_min)**(nord+1)`, formed at `tp_core.F90:217/237` | `damp` used **raw** as `d2 = damp*q` |
| Output | **adds into** `fx`/`fy` (`:1385-1410`) | **emits separate** `fx2`/`fy2`; caller folds them in |
| Used for `zh`? | **NO** | **YES** (`nh_utils.F90:288`) |

`update_dz_d`'s `fv_tp_2d` calls (`nh_utils.F90:286, 296`) pass no `nord`, no
`damp_c`, no `mass`, so the `present(nord) .and. present(damp_c)` guard at
`tp_core.F90:235` is false and `deln_flux` is unreachable. Height damping is
instead the separate additive `del6_vt_flux` term, folded in via `*rarea(i,j)`,
guarded by `damp(k) > 1.E-5` (note: `1.E-5`, not `deln_flux`'s `1.e-4`).

### The coefficient chain that does apply

`update_dz_d`'s `damp` dummy is the **whole array** `damp_vt(npz+1)`, and `ndif`
is `nord_v(npz+1)` (`dyn_core.F90:905-906`, both `intent(inout)`). Their origin
(`dyn_core.F90:713-727`):

```fortran
       nord_v(k) = min(2, flagstruct%nord)
...
       if ( flagstruct%do_vort_damp ) then
            damp_vt(k) = max(0.01,min(flagstruct%vtdm4,flagstruct%d4_bg_top/4.0))     ! for delp, delz, and vorticity
       else
            damp_vt(k) = 0.
       endif
```

so, per level, for the non-sponge case:

```text
damp(k) = max(0.01, min(vtdm4, d4_bg_top/4.0))      if do_vort_damp
        = 0.                                         otherwise
ndif(k) = min(2, nord)
```
with sponge-layer overrides at `k=1` and `k<=max(2,n_sponge-1)` replacing the
`vtdm4` origin with `0.5*d2_divg` and forcing `nord_v(k)=0`
(`dyn_core.F90:741-758`). `vtdm4` is declared in `fv_arrays.F90:371` (default
`0.0`, nondimensional) and read from `fv_core_nml` at `fv_control.F90:671`.

**Relationship to `vtdm4`: TRANSFORMED** — clamped from below by the literal
`0.01` and from above by `d4_bg_top/4.0`, gated by `do_vort_damp`, and
overridden by `d2_bg`/`d2_bg_k1`/`d2_bg_k2` in sponge levels. It is **not** used
raw, and it is **not** exponentiated by `da_min` on this path (unlike
`d_sw`'s own `del6_vt_flux` calls at `sw_core.F90:973/1477`, which do prescale
with `(damp*da_min_c)**(nord+1)`).

For the GEOS production configuration read from this repository's
`FV_StateMod.F90` (`nord=2`, `d4_bg_top=0.12`, `do_vort_damp=.true.`,
`vtdm4=0.01`): `damp_vt(k) = max(0.01, min(0.01, 0.03)) = 0.01` and
`nord_v(k) = 2` at all non-sponge levels. Since `0.01 > 1.E-5`, **the damped
branch is the production path** — the branch the CUDA does not implement.

### What the CUDA has, and what the contract is missing

`launch_deln_flux_transport_cuda` (`wp_b_cuda.cu:375-432`) is a hybrid:

- kernels `deln_init` / `deln_first_flux_{x,y}` / `deln_iter_div` /
  `deln_iter_flux_{x,y}` reproduce **`del6_vt_flux`'s body exactly** —
  `d2 = damp*q` over `[is-1-nord, ie+1+nord]`, `fx2 = del6_v*(d2(i-1)-d2(i))`,
  `fy2 = del6_u*(d2(i,j-1)-d2(i,j))`, the `nord` cascade with
  `d2 = (fx2-fx2+fy2-fy2)*rarea` and the sign-flipped flux recomputation — all
  bounds matching `sw_core.F90:1585-1652`;
- but it prescales with `damp_coef = pow(damp * da_min, nord+1)`
  (`wp_b_cuda.cu:382`), which is `deln_flux`'s coefficient rule, wrong here;
- and it ends with `deln_add_no_mass_kernel` adding `fx2`/`fy2` **into**
  `fx`/`fy` (`:425-429`), which is `deln_flux`'s output convention. Authoritative
  `del6_vt_flux` leaves `fx2`/`fy2` separate so the caller can fold them in with
  `*rarea` in the `zh` update;
- and it omits all four `copy_corners` calls.

So the numerical core is closer to correct than 3A.3 believed, but it is wired
with the wrong coefficient rule and the wrong output convention.

**Missing contract element — and it is not a scalar.** `NumericalConfig`
(`include/fv3_gpu_internal.hpp:78-91`) has `nord`, `nord_v`, `nord_w`,
`nord_t`, `da_min`, `da_min_c`, `lim_fac`, `dz_min` — all **scalars** — and no
`vtdm4`, `d4_bg_top`, `do_vort_damp`, `d2_bg`, or `dp0`. Authoritative
`update_dz_d` needs `damp(k)` and `ndif(k)` as **per-level arrays of length
`npz+1`**, because the sponge overrides make them genuinely `k`-dependent, and
because `update_dz_d` itself mutates index `km+1`
(`damp(km+1)=damp(km)`, `ndif(km+1)=ndif(km)`, `nh_utils.F90:247-248`).

Therefore the correct fix is **not** to add a scalar `damp_c`/`vtdm4` field to
`NumericalConfig`. It is to plumb two per-level arrays (`damp_vt[npz+1]`,
`nord_v[npz+1]`) — computed host-side by the existing
`dyn_core.F90:713-762` logic and made device-resident, alongside `dp0[npz]`
for `edge_profile`. Per the milestone constraint, **no configuration field was
invented and no source was changed.** The authoritative call chain does
demonstrate the need, so this is a justified design input for the next
implementation slice, not a speculative addition.

---

## Section 5 — `update_dz_d` production placement: RESOLVED

```text
update_dz_d executes AFTER d_sw — strictly after the entire OpenMP k-loop over
d_sw closes (dyn_core.F90:829), NOT inside it and NOT before it.

  guarded by:  the `else` branch of `if (hydrostatic)` at dyn_core.F90:899
               (i.e. .not. hydrostatic), inside #ifndef SW_DYNAMICS
               (opens :903, closes :985). Satisfied by production config.
  preceded by: complete_group_halo_update(i_pack(1)) at :878
               (delp, pt, q_con — needed by Riem_Solver3, not by update_dz_d),
               then the nested-BC block :883-898.
  call site:   dyn_core.F90:905  (the single call site in the file)
  followed by: fv_debug prt_mxm (:910) and id_ws send_data (:913), both
               diagnostic-only, then Riem_Solver3 at :918.
```

Verbatim (`dyn_core.F90:899-907`):
```fortran
     if ( hydrostatic ) then
        call geopk(ptop, pe, peln, delp, pkc, gz, phis, pt, q_con, pkz, npz, akap, .false., &
                   gridstruct%nested, .true., npx, npy, flagstruct%a2b_ord, bd)
     else
#ifndef SW_DYNAMICS
                                            call timing_on('UPDATE_DZ')
        call update_dz_d(nord_v, damp_vt, flagstruct%hord_tm, is, ie, js, je, npz, ng, npx, npy, gridstruct%area,  &
                         gridstruct%rarea, dp_ref, zs, zh, crx, cry, xfx, yfx, delz, ws, rdt, flagstruct%dz_min, gridstruct, bd, flagstruct%lim_fac)
```

### Ownership and architectural consequences

- **Orchestration owner: WP-C** (it is the `update_dz_d`/nonhydrostatic height
  path), consuming WP-B's transport as a *service* — which is exactly how
  `launch_update_dz_d` already calls `dispatch.dgrid_height_transport`. The
  existing division of labour is correct.
- **`update_dz_d` is once-per-substep over all `npz+1` levels, whereas `d_sw` is
  per-`k`.** A CUDA orchestration must not fuse or hoist it into the D-grid
  level loop.
- **Hard data dependency on the full 3D `crx`/`cry`/`xfx`/`yfx` stacks.** These
  are `intent(out)` dummies of `d_sw` (`sw_core.F90:557-558`), computed inside
  it from `uc`/`vc` via the contravariant `ut`/`vt` transform
  (`sw_core.F90:880-921`). `edge_profile` reconstructs level `km+1` from
  vertically adjacent levels, so `update_dz_d` is inherently cross-`k` and
  requires every level of `d_sw` to have completed.
- **Control-parameter dependency**: `damp_vt(k)`/`nord_v(k)` are populated
  *inside* the `d_sw` k-loop (`dyn_core.F90:713-757`), reinforcing the ordering
  constraint.
- **`zh` must be persistently device-resident with a valid 3-wide halo across
  substeps.** There is no `zh` halo exchange immediately before line 905; the
  halo comes from the previous substep's `i_pack(4)` exchange
  (started `:931`, completed `:967`), or on `it==1` from the `gz` `i_pack(5)`
  exchange (`:459` → `:552`) followed by `zh = gz` over the full halo box
  (`:556-564`).
- **`crx`/`cry`/`xfx`/`yfx` need no halo exchange** — their allocations
  (`dyn_core.F90:333-336`) exactly match what `d_sw` writes.
- **`update_dz_d` + `Riem_Solver3` should be one fused device region**: the only
  halo boundary is *after* `Riem_Solver3` (`:928-938`, exchanging `zh` and
  `pkc` jointly). Splitting them would force a spurious `zh` round-trip.

### Intended CUDA insertion point

Inside `fv3_gpu_dyn_core_step`'s per-substep sequence, in the
`hydrostatic == false` branch: after the WP-B `dgrid_dynamics` stage has
completed for all levels and after the scalar-halo (`H_DSCALAR`) completion that
corresponds to `i_pack(1)`, immediately before the WP-C `vertical_solver`
(`Riem_Solver3`) stage — i.e. `launch_update_dz_d` then `launch_vertical_solver`
back-to-back on the compute stream with no intervening host sync.

**NOT WIRED in this milestone.** The placement is unambiguous, but the change is
*not* purely orchestration: `dgrid_dynamics` is still `nullptr`
(`fv3_gpu_production_dispatch.cu:25`), so the `crx`/`cry`/`xfx`/`yfx` inputs are
not produced by any GPU stage yet, and the Section 3 discrepancies (9)-(15) mean
wiring it now would place a knowingly-incorrect operator into the production
path. Per the milestone constraint ("wire it only if the placement is
unambiguous **and** the change is purely orchestration"), it was left unwired.

---

## Section 6 — Remaining WP-B D-grid dynamics: dependency-ordered slice map

### Structural facts established first

**`d_sw` is strictly per-`k`. There are no cross-level dependencies inside it.**
Verified from the signature (`sw_core.F90:520-560`): every prognostic dummy
(`delp`, `pt`, `ua`, `va`, `w`, `q_con`, `u`, `v`, `uc`, `vc`, `divg_d`,
`delpc`, `ptc`, `crx_adv`, `xfx_adv`, ...) is a **2-D** slice; only the tracer
array `q(isd:ied,jsd:jed,km,nq)` is 3-D, and it is indexed at the scalar dummy
`k`. So `d_sw` can be launched as a batched kernel set with `k` as an outer grid
dimension — no vertical sync, no vertical halo.

**But its control parameters are per-level.** `nord`, `nord_v`, `nord_w`,
`nord_t`, `damp_v`, `damp_w`, `damp_t`, `d2_bg`, `d4_bg`, `d_con`, `kgb` are all
**scalar** dummies of `d_sw`, assigned per-`k` by the caller
(`dyn_core.F90:705-765`). A batched CUDA `d_sw` therefore needs per-`k`
*parameter vectors*, not the scalars currently in `NumericalConfig`. This is the
same contract gap identified in Section 4, in a second place.

Consequence: **both divergence-damping branches are live in a single run.**
`dyn_core.F90:743/753/761` force `nord_k = 0` in the sponge layers, so the top
levels take the `nord==0` del-2 path (`sw_core.F90:1277-1358`) while all others
take the `nord>0` path (`:1359-1423`). Verified directly, including the
additional `d2_bg_k2 > 0.01` / `> 0.05` guards on the 2nd and 3rd sponge
branches. With GEOS production values (`n_sponge=9`, `d2_bg=0.0`,
`d2_bg_k1=0.2`, `d2_bg_k2=0.1`): `k=1` and `k=2..8` take `nord_k=0`; `k>=9`
takes `nord>0`. **Implementing only one branch validates everywhere except the
model top.**

**Not called by `d_sw`** (corrections to plausible assumptions, all verified):
- `mix_dp` — defined `dyn_core.F90:2033`, called only at `dyn_core.F90:835`,
  *after* the k-loop, guarded by `flagstruct%fill_dp`. It is a separate
  post-loop stage, not part of `d_sw`.
- `divergence_corner` — called from **`c_sw`** (`sw_core.F90:179`) and
  `fv_nesting.F90:247`, never from `d_sw`. `d_sw` receives `divg_d`
  pre-computed and halo-exchanged (`i_pack(3)`, `position=CORNER`).
- `deln_flux` / `copy_corners` — only indirectly, via `fv_tp_2d` and
  `del6_vt_flux`.
- `d2a2c_vect` — used by `c_sw`, before the D-grid loop.

### The 32-operation sequence, grouped into slices

`d_sw` = `sw_core.F90:520-1540`, called from `dyn_core.F90:789` inside
`do k=1,npz` (`:700`) under `!$OMP parallel do` (`:693`).

Slices are ordered by dependency. `WS` = workspace clobbered. Halo depths are
read depth beyond the compute domain (`ng=3`).

---

#### Slice B1 — Contravariant C-grid wind transform + panel-edge + corner solve

| field | value |
|---|---|
| **Fortran** | `sw_core.F90:672-880`; interior `:688-707`, panel edges `:711-779`, corner 2×2 solves `:791-863`, `grid_type>=3` identity `:867-879` (dead) |
| **Host C++ reference** | none — `wp_b_kernels.hpp` has `contravariant_edge_corrections`, but it is untested and was never cross-checked against this block |
| **Inputs** | `uc`, `vc` (halo depth 3), `cosa_u`, `cosa_v`, `rsin_u`, `rsin_v`, `sin_sg(:,:,1..4)`, `dt` |
| **Outputs** | `ut(isd:ied+1,jsd:jed)`, `vt(isd:ied,jsd:jed+1)` |
| **Workspace** | scalar `damp` per corner |
| **Halo** | 3 in `uc`, `vc` — supplied by `i_pack(9)` (CGRID_NE), completed `dyn_core.F90:640` |
| **Edge/corner** | **heavy.** Interior loop *skips* rows `j∈{0,1,npy-1,npy}` and `j∈{1,npy}`; the four panel-edge blocks then fill them with upwind one-sided `sin_sg` formulas that **drop the `rsin` factor** on the parallel-to-edge component; then four corner 2×2 implicit solves with `damp = 1/(1 - 0.0625*cosa_u*cosa_v)` |
| **Kernel boundary** | 3 kernels, strictly ordered: interior → 4 edge blocks → 4 corner solves. The corner solves are ~10 point updates each and are sequentially dependent on the edge blocks |
| **Depends on** | nothing inside `d_sw` (first executable block) |
| **Dispatch slot** | `dgrid_dynamics` (currently `nullptr`) |
| **Test strategy** | needs a genuine multi-panel or panel-edge-aware fixture; a single interior patch cannot exercise any of the interesting code |
| **Numerical risk** | **HIGHEST in all of WP-B.** The corner 2×2 solve is the most intricate construct in `d_sw` |

#### Slice B2 — Courant number and flux-area preparation, `ra_x` / `ra_y`

| field | value |
|---|---|
| **Fortran** | `sw_core.F90:882-892` (`xfx_adv`/`yfx_adv` from `ut`/`vt`), `:898-921` (upwind `sin_sg` weighting, `crx_adv`/`cry_adv`, in-place `xfx_adv`/`yfx_adv`), `:927-936` (`ra_x`/`ra_y`) |
| **Host C++ reference** | `wp_b_kernels.hpp::prepare_flux_courant` (tests only the `hord=1/8` shape) |
| **Inputs** | `ut`, `vt`, `dt`, `rdxa`, `rdya`, `dx`, `dy`, `sin_sg`, `area` |
| **Outputs** | `crx_adv`, `cry_adv`, `xfx_adv`, `yfx_adv`, `ra_x(is:ie,jsd:jed)`, `ra_y(isd:ied,js:je)` |
| **Workspace** | none |
| **Halo** | 1 |
| **Edge/corner** | none as a *special case* — but the upwind `sin_sg` selection (`:900-906`) is the mechanism by which panel-edge geometry enters the fluxes, applied over the whole domain |
| **Kernel boundary** | 3 kernels (or 2 with the in-place `xfx_adv` rewrite fused) |
| **Depends on** | B1 |
| **Dispatch slot** | `dgrid_dynamics` |
| **Test strategy** | pointwise; verifiable against hand-computed values |
| **Numerical risk** | LOW–MEDIUM. Note `crx_adv` here means "Courant number", produced *inside* `d_sw` — distinct from `update_dz_d`'s `crx_adv`, which is `edge_profile`'s vertical reconstruction. **The name collision is a real trap** (see Section 3 discrepancy 9) |

#### Slice B3 — Authoritative `fv_tp_2d` transport engine (correctness upgrade)

| field | value |
|---|---|
| **Fortran** | `tp_core.F90:109-243` (`fv_tp_2d`), `:247-322` (`copy_corners`), `:324-709` (`xppm`), `:712-1147` (`yppm`), `:1263-1411` (`deln_flux`) |
| **Host C++ reference** | `wp_b_kernels.hpp::fv_tp_2d_hord6`, `xppm_hord6`, `yppm_hord6` (untested); **CUDA already exists** in `wp_b_cuda.cu` and is correct in the tile interior only |
| **Inputs** | scalar `q` (halo 3), `crx`/`cry`, `xfx`/`yfx`, `ra_x`/`ra_y`, `hord`, `dxa`, `dya`, corner flags, optional `mfx`/`mfy`/`mass`/`nord`/`damp_c` |
| **Outputs** | `fx`, `fy` |
| **Workspace** | `q_i`, `q_j`, `fx1`, `fx2`, `fy2`, `fyy` (`tp_core.F90:136-141`); `deln_flux` adds `d2`, `fx2`, `fy2` (`:1278-1280`) |
| **Halo** | 3 in `q` |
| **Edge/corner** | `copy_corners(q,·,2)` at `:163` and `copy_corners(q,·,1)` at `:181`, guarded **only** by `.not. nested` — always active for `grid_type=0`; plus `is1`/`ie3`/`ie1` clamping, one-sided `al` overrides from `dxa`/`dya` ratios, and the `smt5` sign-test override at panel edges |
| **Kernel boundary** | keep the existing 6-pass chain; add a `copy_corners` prologue (see Section 8) and panel-edge specialisation inside the PPM passes |
| **Depends on** | B2 for its inputs; **nothing depends on B3 being new — every later slice and the already-wired `dgrid_height_transport` depend on it being correct** |
| **Dispatch slot** | shared engine; consumed by `dgrid_height_transport` (wired) and `dgrid_dynamics` |
| **Test strategy** | a fixture with `is==1` / `(ie+1)==npx` so the edge paths execute; compare interior against the current kernels to prove no interior regression |
| **Numerical risk** | MEDIUM. The interior arithmetic is already verified correct (Section 3 item 1); the risk is confined to the newly added edge/corner paths |

#### Slice B4 — Scalar transport battery (`delp`, `w`, `q_con`, `pt`) + mass-flux accumulation

| field | value |
|---|---|
| **Fortran** | `:939-940` (`delp`, `hord_dp=-6`, with `deln_flux`), `:943-960` (`cx`/`cy`/`xflux`/`yflux`), `:963-968` (zero `heat_source`/`diss_est`), `:970-985` (`del6_vt_flux` on `w`), `:986-992` (`w`, `hord_vt`, `mfx`/`mfy`), `:993-1001` (`q_con`, `hord_dp`, `mass`), `:1004-1006` (`pt`, `hord_tm`, `mass`), `:1009-1048` (state update), `:1050-1054` (`dpx`) |
| **Host C++ reference** | `wp_b_kernels.hpp::apply_mass_and_scalar_flux`, `deln_flux_stage` |
| **Inputs** | `delp`, `pt`, `w`, `q_con`, `crx_adv`/`cry_adv`, `xfx_adv`/`yfx_adv`, `ra_x`/`ra_y`, `rarea`, per-`k` `nord_t`/`nord_w`/`damp_t`/`damp_w`/`kgb` |
| **Outputs** | `delp`, `pt`, `w`, `q_con` updated; `cx`, `cy`, `xflux`, `yflux`, `dpx`, `dw`, `heat_source`, `diss_est` |
| **Workspace** | `fx`, `fy`, `gx`, `gy`, `wk`, `fx2`, `fy2` + all of B3's |
| **Halo** | 3 |
| **Edge/corner** | inherited from B3; plus `del6_vt_flux`'s 2+2·`nord_w` `copy_corners` |
| **Kernel boundary** | 4 `fv_tp_2d` invocations + 1 `del6_vt_flux` + 3 pointwise update kernels |
| **Depends on** | B3 |
| **Dispatch slot** | `dgrid_dynamics` |
| **Test strategy** | per-field transport tests; explicit ordering test for `:1034-1047` |
| **Numerical risk** | **HIGH.** (a) `hord_dp = -6` activates the positivity clamp `al = max(0.,al)` — a path never yet exercised; (b) `:1037→:1040→:1043` is strictly order-critical (mass-weight with *old* `delp`, update `delp`, divide by *new* `delp`); (c) `xflux`/`yflux`/`cx`/`cy`/`dpx` are declared `real(kind=8)` **unconditionally** (`:549-554`) and must stay double even in an R4 build |

#### Slice B5 — Authoritative `update_dz_d` height transport (repair of the wired path)

| field | value |
|---|---|
| **Fortran** | `nh_utils.F90:210-323`; `edge_profile` `:1547-1653`; `del6_vt_flux` `sw_core.F90:1544-1655` |
| **Host C++ reference** | `vertical/fv3_vertical_column.hpp::edge_profile_column` — **verified this milestone to be a faithful translation of the non-uniform branch**, including the `gk` loop-persistence idiom. Currently reachable only from `tests/vertical/test_vertical_column.cpp:15` |
| **Inputs** | `zh` (halo 3, `npz+1` levels), `crx`/`cry`/`xfx`/`yfx` (**`npz` levels**), `dp0`=`dp_ref[npz]`, `area`, `rarea`, `del6_u`, `del6_v`, `zs`, `rdt`, `dz_min`, `hord_tm`, per-level `damp[npz+1]` and `ndif[npz+1]` |
| **Outputs** | `zh` updated, `ws` |
| **Workspace** | `crx_adv`/`xfx_adv`/`cry_adv`/`yfx_adv` (`npz+1`), `gam`, `z2`, `wk2`, `fx`/`fy`, `fx2`/`fy2`, `ra_x`/`ra_y` |
| **Halo** | 3 in `zh`; none in `crx`/`cry`/`xfx`/`yfx` |
| **Edge/corner** | via B3's `copy_corners`, plus `del6_vt_flux`'s 2+2·`ndif(k)` |
| **Kernel boundary** | `edge_profile` = 1 kernel, one thread per `(i,j)` column, **serial in `k`** (Thomas algorithm: forward elimination `k=2..km`, back-substitution `k=km..1`). Then per-`k`: `ra_xy` → `fv_tp_2d` → `del6_vt_flux` → combined `zh` update. Then 1 boundary kernel (`ws` + descending clamp) |
| **Depends on** | B3. Independent of B1/B2/B4 *as a kernel*, but its `crx`/`xfx` inputs are produced by B2 in production |
| **Dispatch slot** | `dgrid_height_transport` — **already wired**, and currently computing the wrong operator |
| **Test strategy** | `edge_profile` verifiable directly against `edge_profile_column` on the host (same tridiagonal solve) — a rare chance for an exact CPU/GPU cross-check; then the two `damp(k)` branches separately |
| **Numerical risk** | MEDIUM. Confirmed this milestone: `update_dz_d` hardcodes `uniform_grid = .false.` (`nh_utils.F90:245`) and `limiter = 0`, so the **non-uniform `dp0` branch is the production path** and the limiter block is dead |

#### Slice B6 — B-grid momentum transport and kinetic energy

| field | value |
|---|---|
| **Fortran** | `:1065-1074` (`dt5`/`dt4`, index clamps), `:1077-1120` (`vb`), `:1122` `ytp_v` (`sw_core.F90:2320-2792`), `:1125-1129` (`ke`), `:1131-1177` (`ub`), `:1179` `xtp_u` (`:1950-2317`), `:1182-1186` (`ke` combine), `:1191-1216` (KE corner recomputation) |
| **Host C++ reference** | `wp_b_kernels.hpp::compute_kinetic_energy` (tests the `hord=1/8` shape only). **No host reference exists for `xtp_u`/`ytp_v`** |
| **Inputs** | `u`, `v` (halo 3), `uc`, `vc`, `ut`, `vt`, `cosa`, `rsina`, `dx`/`rdx`, `dy`/`rdy`, `hord_mt=6` |
| **Outputs** | `ke(isd:ied+1,jsd:jed+1)` |
| **Workspace** | `ub`, `vb` (each serves 4 distinct roles) |
| **Halo** | 3 in `u`/`v`; 2 in `ut`/`vt` |
| **Edge/corner** | `vb`/`ub` replace the `cosa`/`rsina` projection with 4-point one-sided interpolation of the already-contravariant `vt`/`ut` at panel edges (`:1099`, `:1103`, `:1155`); source itself comments `! corner values are incorrect` at `:1088`, `:1109`, `:1147`, `:1166`, which the KE corner block at `:1193-1215` then repairs with a `dt6` three-term formula at exactly 4 points |
| **Kernel boundary** | 6 kernels, strictly ordered (`:1131-1177` has a WAR hazard on `:1122`'s `ub`) |
| **Depends on** | B1 (needs `ut`/`vt` *before* slice B7 overwrites them) |
| **Dispatch slot** | `dgrid_dynamics` |
| **Test strategy** | requires panel-edge and corner fixtures |
| **Numerical risk** | **HIGH.** `xtp_u`/`ytp_v` are **not** `xppm`/`yppm`: they act on D-grid edge winds, output on B-grid corners, take `c` as a length (converted internally at `:2049`), use `dx`/`rdx` not `dxa`, are a **single 1-D pass with no cross-flux and no `q_i`/`q_j`**, have **no** negative-`iord` clamp, call **no** `copy_corners`, and add a panel-edge `bl`/`br` **zeroing** absent from `xppm` (`:2002-2007`). None of this can be reused from B3 |

#### Slice B7 — Relative vorticity, `w`/`q_con` un-weighting, divergence damping (both branches)

| field | value |
|---|---|
| **Fortran** | `:1219-1235` (`vt=u*dx`, `ut=v*dy`, relative vorticity `wk`), `:1237-1270` (`w`/`q_con` un-weighting), `:1277-1358` (`nord==0` del-2 damping), `:1359-1423` (`nord>0` path with 3 `fill_corners` + `smag_corner`) |
| **Host C++ reference** | `wp_b_kernels.hpp::compute_relative_vorticity`, `compute_damping_increments` |
| **Inputs** | `u`, `v`, `dx`, `dy`, `rarea`, `divg_d` (halo `nord`, from `i_pack(3)`), `divg_u`, `divg_v`, `ua`, `va`, `dyc`, `dxc`, `sina_u/v`, `cosa_u/v`, `sin_sg`, `rarea_c`, `da_min`/`da_min_c`, per-`k` `d2_bg`, `d4_bg`, `dddmp`, `nord_k` |
| **Outputs** | `wk` (relative vorticity), `vort`, `delpc`, `divg_d`, `ke += vort`; `w`, `q_con` un-weighted |
| **Workspace** | **clobbers `uc` and `vc`** (`:1381`, `:1388`); `smag_corner` needs 4 more full-halo 2-D arrays (`:1868-1871`) |
| **Halo** | 3 in `u`/`v`; `nord` in `divg_d` |
| **Edge/corner** | `nord==0`: `sin_sg` edge branches + corner extra-term removal (`:1346-1349`). `nord>0`: three `fill_corners` calls **interleaved between** the three difference loops (`:1378` XDir BGRID, `:1385` YDir BGRID, `:1392` `(vc,uc)` VECTOR DGRID with `mySign=-1`) + corner removal (`:1400-1403`) |
| **Kernel boundary** | `nord==0`: ~4 kernels. `nord>0`: the `do n=1,nord` loop (`:1371-1405`) is a **serial recurrence** — 3 kernels per iteration plus the corner fixups |
| **Depends on** | B6 (`:1219` overwrites B1's `ut`/`vt`; B6's KE corner block at `:1191-1216` is their last reader — a WAR hazard) |
| **Dispatch slot** | `dgrid_dynamics` |
| **Test strategy** | **both branches must be tested**, at sponge and non-sponge levels |
| **Numerical risk** | **HIGH.** Both branches live per-level; the `nord==0` path is the sole consumer of `ua`, `va`, `ptc`. `smag_corner` runs iff `dddmp >= 1.e-5`; `fv_arrays.F90:351` gives default `dddmp = 0.0`, so whether `smag_corner` (and with it the whole `a2b_ord4` edge apparatus) is on the critical path **must be read from the production `fvcore_layout.rc`, not assumed** |

#### Slice B8 — Absolute vorticity transport, D-wind update, vorticity damping

| field | value |
|---|---|
| **Fortran** | `:1439-1459` (`vort = wk + f0`), `:1461-1462` (`fv_tp_2d(vort, hord_vt)`), `:1463-1472` (`u`/`v` update), `:1476-1479` (`del6_vt_flux(nord_v)`), `:1523-1534` (`u += vt`, `v -= ut`) |
| **Host C++ reference** | `wp_b_kernels.hpp::update_d_winds` |
| **Inputs** | `wk`, `f0`, `crx_adv`/`cry_adv`, `xfx_adv`/`yfx_adv`, `ra_x`/`ra_y`, `ke`, `vt`/`ut`, `del6_u`/`del6_v`, per-`k` `damp_v`, `nord_v` |
| **Outputs** | `u`, `v` |
| **Workspace** | `fx`/`fy`; `del6_vt_flux` **clobbers `vort`, `ut`, `vt`** |
| **Halo** | 3 in `vort`/`wk` |
| **Edge/corner** | via B3; plus `del6_vt_flux`'s 2+2·`nord_v` `copy_corners` |
| **Kernel boundary** | 5 stages, strictly ordered |
| **Depends on** | B7 (`vort`), B6 (`ke`), B3 |
| **Dispatch slot** | `dgrid_dynamics` |
| **Test strategy** | wind-update test; explicit anti-dependence tests |
| **Numerical risk** | MEDIUM–HIGH, entirely from aliasing |

#### Explicitly NOT WP-B

- `:1425-1436` (`ub`/`vb` = `d(vort)`) and `:1481-1520` (`heat_source`,
  `diss_est` with `rsin2`/`cosa_s`) — **WP-D** (diffusive heating; existing
  `post_loop_heating` slot). Both are disabled at sponge levels because
  `d_con_k = 0.` there (`dyn_core.F90:746, 758, 763`).
- `mix_dp` (`dyn_core.F90:2033`, called `:835`) — post-loop stage, **WP-D**.
- `divergence_corner` — **WP-A** (`c_sw`).
- All halo exchanges — **WP-E**.

#### Anti-dependence hazards to encode explicitly in device state

Four array-reuse traps that will silently produce wrong answers if the stage
order is relaxed:

1. `B7:1425-1436 → B7:1439` — `vort` is overwritten; its reader must run first.
2. `B6:1191-1216 → B7:1219` — `:1219` destroys B1's contravariant `ut`/`vt`.
3. `WP-D:1481 → B8:1523` — the heat source reads *pre-diffusion* `u`/`v`.
4. `{B1, B6, B7:1277} → B7:1381` — `:1381`/`:1388` destroy `uc`/`vc`.

Aliasing load in `d_sw` (all workspace is 2-D; `sw_core.F90:562-580`):
`ut`/`vt` serve 3 roles, `ub`/`vb` serve 4, `wk` serves 3, `vort` serves 3,
`fx`/`fy` serve 3.

#### Clean savepoint boundaries for Prompt 6

Points where no live scratch aliasing is in flight: after B2
(`ut,vt,crx_adv,cry_adv,xfx_adv,yfx_adv,ra_x,ra_y`), after B4
(`delp,pt,w,q_con,cx,cy,xflux,yflux,dpx`), after B6 (`ke`), after B7
(`divg_d,delpc,ke`), and at `d_sw` exit
(`u,v,delp,pt,w,q_con,heat_source,diss_est,divg_d`).

---

## Section 7 — Recommended next WP-B CUDA sub-milestone

**Slice B5, scoped to include the B3 engine corrections it depends on.**

> **Milestone 3A.4 — Authoritative `update_dz_d` height transport, rebuilt on a
> corrected `fv_tp_2d` engine.**

Chosen over the alternatives because it is the dependency-first unit *and* the
only one that repairs code already wired into production:

- `dgrid_height_transport` is the **one WP-B function already live in
  `fv3_gpu_production_dispatch.cu`**, and Section 3 shows it currently computes
  the wrong operator. Leaving a knowingly-incorrect operator wired while
  building new slices on top of it is the worst available option.
- It is WP-C's only blocker: `launch_update_dz_d` already calls
  `dispatch.dgrid_height_transport` and needs nothing new from WP-C.
- The `fv_tp_2d` corrections it requires (`copy_corners`, panel-edge PPM) are
  the shared foundation that B4, B6 and B8 all inherit, so the work is not
  throw-away.
- It is the only slice with an **exact** CPU/GPU cross-check available today:
  `edge_profile_column` is a verified faithful host translation of the same
  tridiagonal solve.

### Exact scope

1. **`edge_profile` CUDA kernel** — one thread per `(i,j)` column, serial in
   `k`, non-uniform (`dp0`) branch only (production hardcodes
   `uniform_grid=.false.`, `limiter=0`). Solves the `(crx,xfx)` and
   `(cry,yfx)` pairs with a shared `gam` factorization. Cross-check against
   `edge_profile_column`.
2. **Contract fix** — separate `crx`/`cry`/`xfx`/`yfx` (`npz` levels, D-grid)
   from `crx_adv`/`cry_adv`/`xfx_adv`/`yfx_adv` (`npz+1`, workspace). Correct
   `wp_b_contract_cuda.cpp:110-113` to require `npz`, and stop overloading the
   `crx` FieldId. Update `test_wp_b_cuda_smoke.cpp` accordingly.
3. **Per-level parameter plumbing** — device-resident `damp[npz+1]`,
   `ndif[npz+1]`, `dp0[npz]`, populated host-side by the existing
   `dyn_core.F90:713-765` logic including the sponge overrides, with
   `damp(km+1)=damp(km)` / `ndif(km+1)=ndif(km)`. **Not** scalar
   `NumericalConfig` fields.
4. **Replace the damping call** — implement `del6_vt_flux` semantics: keep the
   existing (verified-correct) `deln_*` kernel bodies, but drop the
   `pow(damp*da_min, nord+1)` prescale in favour of raw `damp(k)`, and drop
   `deln_add_no_mass_kernel` so `fx2`/`fy2` stay separate.
5. **Correct the `zh` update** to
   `(z2*area + (fx-fx) + (fy-fy)) / ((ra_x+ra_y)-area) + ((fx2-fx2)+(fy2-fy2))*rarea`,
   with the authoritative parenthesisation.
6. **Implement the `damp(k) > 1.E-5` two-branch structure**, including the
   `z2` copy in the damped branch (which keeps `copy_corners` side effects off
   `zh`'s halo corners) and the in-place transport in the undamped branch
   (which deliberately does mutate them).
7. **Add `copy_corners`** to `fv_tp_2d` (2 calls) and to the `del6_vt_flux`
   path (2 + 2·`ndif(k)`), per Section 8's Group-1 recommendation.
8. **Add panel-edge PPM treatment** — `is1`/`ie3`/`ie1` clamping, one-sided
   `al` overrides (requires resolving `dxa`/`dya`, currently unbound), and the
   `smt5` sign-test override.
9. **Fix `ra_x`/`ra_y` extents** to `(is:ie, jsd:jed)` and `(isd:ied, js:je)`.
10. **Stop writing `delz`** in the `update_dz_d` path (authoritative does not),
    after confirming the CUDA `Riem_Solver3` assigns it unconditionally.
11. **Resolve Section 3 item 18** — audit the `ws` + descending-clamp kernel.
12. **Test** with a panel-edge-aware fixture, both `damp(k)` branches, R8 and
    R4, `compute-sanitizer` clean, via `srun` on `gpu_a100`/`rome`.

Deliberately **excluded**: `dgrid_dynamics` stays `nullptr`; B1/B2/B4/B6/B7/B8
untouched; `corner_edge_service` beyond the Group-1 fill needed here.

---

## Section 8 — `corner_edge_service` semantics: RESOLVED

**Answer: (D) — it represents multiple distinct existing Fortran operations.**
Additionally (A) applies to *one* of the four groups; (B) is definitively
**wrong**; therefore (C) is also wrong.

`sw_core.F90` contains **no MPI call of any kind** — its only `fv_mp_mod`
import is `fill_corners, XDir, YDir` (`sw_core.F90:49`). `d_sw` runs per-`k`
inside an OpenMP loop and therefore *cannot* communicate. **Every corner/edge
operation in the D-grid stage is local.** Halo exchange is a strict
*precondition*, not part of the service.

Reading bodies rather than names, there are **12 distinct operations in 4
groups**:

**Group 1 — phantom-corner fills (4 ops).** Write only the four `ng×ng`
diagonal ghost blocks; read the same array at transposed positions. No metrics.
- `copy_corners` (`tp_core.F90:247-322`) — scalar, `dir∈{1,2}`, A-grid indices.
  14 live call sites.
- `fill_corners_2d_r4/r8` (`fv_mp_mod.F90:1033/1113`) — scalar, `AGRID`/`BGRID`
  staggering. **Different index formulas from `copy_corners`; not
  interchangeable.** Used on `divg_d`.
- `fill_corners_dgrid/cgrid/agrid` via `fill_corners_xy_*` (`:1194,1338,1450,1514`)
  — vector pair, cross-fills `x` from `y`, `mySign=-1` when `VECTOR`; which
  corners flip depends on staggering (D/A: SW+NE; C: NW+SE).
- `fill_4corners`/`fill2_4corners`/`fill3_4corners` (`sw_core.F90:3289/3227/3156`)
  — 2–3 points per corner, `c_sw` only.

(Correction to a natural assumption: the `fill_corners` family is **not** in
`model/boundary.F90` — that file holds only nested-grid BC machinery. It is in
`tools/fv_mp_mod.F90`. And `copy_corners` is called from `fv_tp_2d` itself, not
from inside `xppm`/`yppm`. Both verified.)

**Group 2 — corner sum corrections (1 op, 4 instantiations).** After forming a
divergence/curl as a sum of four face fluxes at a B-grid point, add or subtract
the one flux that does not exist because only three cells meet at a cube corner.
Sites: `sw_core.F90:1346-1349`, `:1400-1403`, `:1751-1754`, `:1919-1922`. Four
scalar points each.

**Group 3 — corner reformulations (4 ops).** The formula itself changes:
the `ut`/`vt` corner 2×2 implicit solve (`:791-863`); KE corner recomputation
(`:1193-1215`); `d2a2c_vect` signed vector reflections (`:2960-2979` etc.);
3-point corner averaging (`dyn_core.F90:2304-2323`, `a2b_edge.F90:435-459`).

**Group 4 — panel-edge operations (3 ops).** Upwind one-sided `sin_sg` metric
substitution; PPM stencil clamping + edge coefficient overrides + `smt5`/`smt6`
fixes + panel-corner slope zeroing; one-sided interpolation of an
already-contravariant field.

### Interface recommendation

1. **Narrow `corner_edge_service` to Group 1 only**, and rename it to reflect
   that (e.g. `phantom_corner_fill`). It is the only group with a clean,
   numerics-free contract: `{scalar | vector} × {A,B,C,D-grid} × {XDir,YDir} ×
   {full-ng | short}`.
2. **The current `LaunchFn` slot is the wrong shape.** Group 1 fires **tens of
   times per level per acoustic substep** (2 per `fv_tp_2d` × 4–6 call sites,
   plus 2+2·`nord` per `deln_flux`/`del6_vt_flux`), and each invocation touches
   only `4·ng²` points. As a standalone `LaunchFn` this is pure launch latency.
   Expose it instead as an **inlineable `__device__` prologue helper** invoked at
   the head of each PPM/differencing kernel, guarded on per-rank corner flags.
   Keep a `LaunchFn` wrapper only for standalone testing.
3. **It cannot be hoisted or cached.** `dir=1` and `dir=2` write the *same
   memory with different formulas*, so the fill must be re-done before each
   directional sweep.
4. **Fuse Groups 2, 3 and 4 into their owning kernels.** They are 4–20-point
   edits or edge-row overrides sharing registers with the surrounding stage;
   exposing them separately multiplies launches with no reuse. Group 3's corner
   2×2 solve is additionally *sequentially dependent* on Group 4's edge
   substitution having run in the same kernel.
5. **Specialise on `grid_type`/`nested` at launch time, not per thread.** For
   production (`grid_type=0`, non-nested) every `<3` branch is taken and every
   `nested` branch is not.
6. **Keep halo exchange in WP-E.** Group 1's precondition — valid *orthogonal*
   halo strips — is already guaranteed before `dyn_core.F90:789` by
   `i_pack(9)` (`:640`), `i_pack(3)` (`:636`), `i_pack(8)` (`:504`) and
   `i_pack(1)` (`:476`/`:878`). A GPU `d_sw` can assume valid halos on entry and
   needs no communication of its own.
7. **Exclude `fill_ghost`** (`fv_grid_utils.F90:3107-3184`) — a grid-init
   sentinel poison-fill (`big_number`/`tiny_number`), not a numerical operation.
   Keep at most as a debug mode.
8. **`edge_vect_w/e/s/n` are provably unused** in `d_sw`/`dyn_core` — consumed
   only by `external_ic.F90` and `fv_update_phys.F90`. Do not include them.
   (The *different* `edge_w/e/s/n` arrays do matter, indirectly via
   `a2b_ord4`/`a2b_ord2`.)

Also worth recording: `mpp_get_boundary` (`dyn_core.F90:1094`) is an **edge
de-duplication exchange**, run once per acoustic loop at `it==n_split` to
prevent round-off accumulation on overlapped D-grid edges. Its guard is
`grid_type<4` — the only place in this analysis where the threshold is not `<3`.
It belongs to WP-E, not to the corner service.

---

## Section 9 — Numerical validation

**NOT RUN.** No Fortran reference data or savepoint was generated; no
comparison was performed; no build was run and no GPU job was submitted during
this milestone.

Everything above is source inspection and call-flow tracing. Per the project
validation hierarchy, this milestone advances only *understanding* — it does
not change the status of any stage:

| level | status after 3A.3B |
|---|---|
| IMPLEMENTED | `dgrid_height_transport` yes; `dgrid_dynamics` no |
| COMPILED | unchanged from 3A.3 (R8 and R4 PASS) |
| GPU EXECUTED | unchanged from 3A.3 (`dgrid_height_transport` smoke only) |
| NUMERICALLY VALIDATED | **NOT RUN — and Section 3 now predicts the current `dgrid_height_transport` would FAIL such a comparison** |

Section 3's discrepancies were found by reading the authoritative Fortran, not
by running anything. They are *predictions* of numerical divergence, and should
be confirmed by an actual comparison in Prompt 6.

---

## Section 11 — Source modifications

**Source files modified: NONE.**

No file under `cuda/`, no authoritative Fortran, no WP-A/C/D numerical source,
no production numerical contract, and no `geos-gtfv3` file was modified. The
only file written in this milestone is this handoff document. `@fvdycore` was
not populated. No build directory was created or changed.

The Section 5 `update_dz_d` wiring was deliberately **not** performed: the
placement is unambiguous, but the change is not purely orchestration (its inputs
do not yet exist and the operator is currently wrong), so it fell outside the
"trivial, semantics-neutral integration fix" allowance.

---

## Section 10 — Relationship to earlier handoffs

`docs/agent_handoff/milestone_3a3_wp_b_cuda.md` is **superseded in three
specific claims**. Its historical text is preserved; these are annotations:

1. **"Fortran cross-check: NOT POSSIBLE (source absent)"** — now **false**. The
   authoritative source at the pinned revision `geos/v3.0.0` / `ca30e26` is
   available on Discover (Section 1). Every WP-B stage can now be cross-checked
   by inspection.
2. **"`damp_c` mapping ambiguity — `NumericalConfig` has no `vtdm4`; wire
   `damp_c` through `set_numerical_config`"** — the premise was wrong. The
   height path never calls `deln_flux`, so no `damp_c` is needed. What is
   missing is a **per-level `(damp[npz+1], ndif[npz+1])` pair** for
   `del6_vt_flux` (Section 4).
3. **"`ra_x`/`ra_y` computation is new code, numerical semantics: N/A"** — the
   authoritative definitions exist at `nh_utils.F90:269-278` with specific,
   *different* extents; the CUDA computes a superset box (Section 3 item 16).

Its core positive finding stands and is strengthened: the order-6 PPM interior
arithmetic in `wp_b_cuda.cu` is a **correct** translation, verified
element-by-element against `tp_core.F90:523-555` this milestone.

---

## Section 12 — Completion report

| item | result |
|---|---|
| **Authoritative `@fvdycore` revision** | `GEOS-ESM/GFDL_atmos_cubed_sphere`, tag `geos/v3.0.0`, SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45`. Three independent pins agree (GEOSgcm + GEOSfvdycore `components.yaml`, `.mepo/state.0.json`). Read from a pristine clean checkout already on Discover; `@fvdycore` in this repo deliberately left unpopulated |
| **WP-B Fortran mapping** | **COMPLETE** — `fv_tp_2d`, `copy_corners`, `xppm`, `yppm`, `pert_ppm`, `deln_flux`, `del6_vt_flux`, `edge_profile`, `update_dz_d`, `d_sw` (32 operations), `xtp_u`, `ytp_v`, `smag_corner`, `fill_corners` family, `divergence_corner`, `mix_dp` — all with file/routine/call-site/callee traced from actual call flow |
| **Existing `dgrid_height_transport` reconciliation** | **DISCREPANCY** — 9 discrepancies, 3 of them critical: `edge_profile` entirely absent; the `zh` update uses an additive `rarea` increment instead of the authoritative area-weighted remap; the damping uses `deln_flux` conventions where `del6_vt_flux` is required. Also missing: the `damp(k)>1.E-5` two-branch structure, all `copy_corners`, and all panel-edge PPM treatment. The order-6 PPM **interior** arithmetic MATCHES exactly |
| **`damp_c` / `vtdm4` mapping** | **RESOLVED — DIFFERENT VARIABLES.** `update_dz_d` never calls `deln_flux`; it calls `del6_vt_flux` with raw, unscaled `damp_vt(k)`. `vtdm4 → damp_vt(k) = max(0.01, min(vtdm4, d4_bg_top/4.0))` when `do_vort_damp`, with sponge overrides. The real contract gap is a **per-level `(damp[npz+1], ndif[npz+1])` array pair**, not a scalar `damp_c`. This overturns the 3A.3 handoff's stated blocker |
| **`update_dz_d` production placement** | **RESOLVED** — `dyn_core.F90:905`, after the entire `d_sw` k-loop (closes `:829`), under `.not. hydrostatic` inside `#ifndef SW_DYNAMICS`, immediately before `Riem_Solver3` (`:918`), with no intervening halo exchange. Orchestration owned by WP-C, consuming WP-B transport as a service. **NOT wired** — not a pure-orchestration change |
| **`corner_edge_service` semantics** | **RESOLVED — (D)** multiple distinct Fortran operations: 12 ops in 4 groups. All LOCAL; zero MPI in `sw_core.F90`. Only Group 1 (phantom-corner fills, 4 variants) is a reusable service; Groups 2–4 must fuse into owning kernels. The current `LaunchFn` slot is the wrong shape — Group 1 should be an inlineable `__device__` prologue |
| **Remaining `dgrid_dynamics` dependency map** | **COMPLETE** — 32 operations grouped into 8 dependency-ordered slices (B1–B8) with all required per-slice fields, 4 named anti-dependence hazards, and 5 clean savepoint boundaries. Confirmed `d_sw` is strictly per-`k` with per-`k` *parameters* |
| **Recommended next WP-B CUDA sub-milestone** | **Slice B5 + its B3 prerequisites** — "Authoritative `update_dz_d` height transport, rebuilt on a corrected `fv_tp_2d` engine", 12 enumerated scope items (Section 7) |
| **Numerical validation** | **NOT RUN** |
| **Source files modified** | **NONE** (only this handoff document was written) |
| **Working tree preservation** | **PASS** — no reset, checkout, clean, stash or discard was performed; `@fvdycore` not populated; no build directory touched |

---

## ANNOTATION — acted on by Milestone 3A.4 (2026-08-24)

Added append-only; nothing above has been altered. See
`docs/agent_handoff/milestone_3a4_wp_b_height_transport_repair.md`.

This document's recommendation — **Slice B5, "Authoritative `update_dz_d`
height transport, rebuilt on a corrected `fv_tp_2d` engine"** — has been
implemented. All 9 reconciliation discrepancies recorded above are closed,
including the 3 critical ones (`edge_profile` absent; additive `rarea`
increment instead of the area-weighted remap; `deln_flux` damping conventions
where `del6_vt_flux` is required), plus the missing `damp(k) > 1.E-5`
two-branch structure, all `copy_corners`, and all panel-edge PPM treatment.

The "Existing `dgrid_height_transport` reconciliation: **DISCREPANCY**" row in
Section 12 therefore describes code that no longer exists. A fresh
eight-item reconciliation against the same SHA `ca30e26` was performed in 3A.4
and found **no remaining discrepancy in scope**: five items MATCH FORTRAN or
MATCH WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE, and four further defects found
during that pass (D1 operand reassociation, D2 the `nord_v` group-loop cap, D3
`deln_flux`'s missing `d2` corner rotations, D4 the corner-flag/`cubed_sphere`
caller hazard) were fixed or explicitly fenced.

Everything else in this document stands as written, including:

- the authoritative revision pin;
- the WP-B Fortran mapping;
- the `damp_c`/`vtdm4` resolution (3A.4 confirmed it in the implementation:
  the height path uses raw per-level `damp`, and a bitwise linearity test now
  guards it);
- the per-level `(damp[npz+1], ndif[npz+1])` + `dp0[npz]` contract requirement
  (3A.4 registers these as `damping_v`, `damping_ndif`, `dp_ref`, and closed a
  type-safety hazard in how `damping_ndif`'s integral values travel through
  real storage);
- the `corner_edge_service` analysis — still unimplemented, still a
  fail-adapter. 3A.4's height path was verified never to call it.

**Numerical validation remains NOT RUN.**
