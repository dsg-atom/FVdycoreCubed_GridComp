# Milestone 3A.6 — WP-B.3 Authoritative Momentum Transport

## Status summary

| Level | Result |
|---|---|
| IMPLEMENTED | YES — `xtp_u`, `ytp_v`, vb/ub, KE terms, vorticity precursor, D-wind update |
| COMPILED | YES — R8 and R4, GCC 12.3.0 + nvcc 12.5, `sm_80` |
| GPU EXECUTED | YES — NVIDIA A100-SXM4-40GB (`warpa005`), R8 and R4 |
| MEMORY-SANITIZED | YES — memcheck / initcheck / racecheck, R8 and R4, 0 errors |
| NUMERICALLY VALIDATED vs runtime Fortran | **NOT RUN** (Prompt 6 work) |

Source-level reconciliation against the authoritative Fortran is complete for
the WP-B.3 slice, with **no known discrepancy**. That is a *source* comparison
plus comparison against an independent host transcription; it is **not** a
comparison against Fortran savepoints generated at runtime.

## Git state

- Branch: `cuda/gpu-bringup`
- Starting commit: `2849401c3a819981d1fdabc3a684b29a4b94c604`
  ("Close WP-B scalar transport milestone 3A.5")
- Final commit: `c405d8032ceda7ba7f795ffbff9f9c46763f366b`
  ("Implement authoritative WP-B momentum transport CUDA")
- Not pushed, not merged.

## Authoritative Fortran revision

- Repository: `GEOS-ESM/GFDL_atmos_cubed_sphere`
- Tag: `geos/v3.0.0`
- SHA: `ca30e26305fda43831095334c02a3f5e0b8c0c45`
- Read-only clone used:
  `/discover/nobackup/projects/imvi-nse/jli30/fv3_reference/GFDL_atmos_cubed_sphere`
- Files traced: `model/sw_core.F90` (`d_sw`, `xtp_u`, `ytp_v`),
  `model/tp_core.F90` (`fv_tp_2d`, `xppm`/`yppm` for contrast),
  `tools/fv_mp_mod.F90` (`ng` parameter).

No authoritative Fortran was modified. `geos-gtfv3` was not touched.

## §1 Momentum call flow (ordered)

Traced by call flow from `d_sw`, not by routine name. The full table with
Fortran line numbers, inputs, outputs, staggering, halo reads, edge/corner
dependencies, workspace and next consumer lives in the header
`cuda/dyn_core/dgrid/wp_b_momentum_cuda.cuh` §1 and is the authoritative
in-repo copy. Condensed:

| # | Fortran | operation | CUDA launcher |
|---|---|---|---|
| 1 | `sw_core.F90:1077-1118` | vb construction (B-grid y wind) | `launch_bgrid_vb_cuda` |
| 2 | `sw_core.F90:2320` (called `:1120`) | `ytp_v(vb -> ub)` | `launch_ytp_v_cuda` |
| 3 | `sw_core.F90:1123` | `ke = vb*ub` | `launch_ke_product_cuda` |
| 4 | `sw_core.F90:1126-1167` | ub construction (B-grid x wind) | `launch_bgrid_ub_cuda` |
| 5 | `sw_core.F90:1950` (called `:1169`) | `xtp_u(ub -> vb)` | `launch_xtp_u_cuda` |
| 6 | `sw_core.F90:1172` | `ke = 0.5*(ke + ub*vb)` | `launch_ke_average_cuda` |
| 7 | `sw_core.F90:1177-1200` | four-corner KE fix (`dt6 = dt/6`) | `launch_ke_corner_fix_cuda` |
| 8 | `sw_core.F90:1205-1235` | vorticity precursor `vt=u*dx`, `ut=v*dy`, `wk` | `launch_vorticity_precursor_cuda` |
| 9 | `sw_core.F90:1440` | `vort = wk + f0` / `wk + f0*z_rat` | `launch_vorticity_absolute_cuda` |
| 9b | `sw_core.F90:1456` | `fv_tp_2d(vort, hord_vt)` | **REUSED** `launch_scalar_transport_cuda` (3A.5) |
| 10 | `sw_core.F90:1466-1478` | D-wind update `u`, `v` | `launch_dwind_update_cuda` |

Two order-critical hazards were identified and preserved:

- **`ut`/`vt` dual meaning.** On entry to the momentum block they hold the
  C-grid contravariant winds (read by rows 1, 4, 7). Row 8 overwrites them with
  `u*dx` / `v*dy`, and row 10 reads the second meaning. A real
  anti-dependence — the launchers are separate and are never fused across
  row 8.
- **`ub`/`vb` reuse chain.** vb(1) → `ytp_v` writes ub(2) → `ke=vb*ub`(3) → ub
  overwritten(4) → `xtp_u` writes vb(5) → `ke=0.5*(ke+ub*vb)`(6). Enforced by
  launch order on one stream.

Row 8 is split into two launches (`vt`/`ut` then `wk`) because the `wk` pass
reads the completed `vt`/`ut`, which the Fortran gets from loop sequencing.

## §2 Data / staggering contract

Full table in `wp_b_momentum_cuda.cuh` §2. Key points:

- `nk == npz` (**cell** levels). Contrast Milestone 3A.4's height transport,
  where `nk == npz+1` interface levels.
- D staggering: `u(isd..ied, jsd..jed+1)`, `v(isd..ied+1, jsd..jed)`;
  `dx/rdx` follow `u`, `dy/rdy` follow `v`.
- B staggering: `ub`, `vb`, `ke`, `cosa`, `rsina` at `(i,j)` corners;
  `ub`/`vb` are compute-box only (`is..ie+1 × js..je+1`, no halo).
- A staggering: `wk`, `vort`, `rarea`, `f0`, `z_rat` over `isd..ied × jsd..jed`.
- `ng = 3` (Fortran PARAMETER) drives every `isd/ied/jsd/jed` bound, distinct
  from the registered `halo_width`.
- No alias was invented. `xtp_u` omits `v` and `ytp_v` omits `u` because the
  authoritative routine bodies never reference them (only the declaration lines
  mention them), so carrying them would have been a fabricated dependency.
- `ke` is an **input** view to the D-wind update so that WP-B.5 can pass the
  divergence-damping-augmented `ke` (`sw_core.F90:1462`) unchanged.

## §3 Reuse of existing WP-B infrastructure

| Helper | Classification | Reason |
|---|---|---|
| `TransportBounds`, `LevelFilter` | REUSED UNCHANGED | geometry/topology only, no field identity |
| `plan2d`, `launch_status`, `level_active`, `as_const` | REUSED UNCHANGED | launch plumbing |
| `ppm_p1/p2/c1/c2/c3` | REUSED UNCHANGED | identical PPM constants |
| `smt5_from` | REUSED UNCHANGED | the order-6 `3|b0| < |bl-br|` test and the seam `bl*br<0` degradation |
| `seam_cell_x`, `seam_cell_y` | REUSED UNCHANGED | the WMP seam index sets in `xtp_u`/`ytp_v` are *exactly* these predicates |
| `launch_scalar_transport_cuda` | REUSED UNCHANGED | row 9b `fv_tp_2d(vort, hord_vt)`; `mass_flux_weighting=0`, `deln_flux_enabled=0`, `hord=6` |
| `launch_ra_x_cuda` / `launch_ra_y_cuda` | REUSED UNCHANGED | `ra_x`/`ra_y` for row 9b |
| `al_x_at` / `al_y_at`, `xppm_face_flux` / `yppm_face_flux` | **NOT APPLICABLE** | they substitute a one-sided `al` using `dxa`/`dya` and use a single cell-centered Courant number; `xtp_u`/`ytp_v` modify `bl`/`br` directly, use `dx`/`dy`, and rescale the CFL by `rdx`/`rdy` at the **upwind** cell |
| `copy_corners` | **NOT APPLICABLE** | neither `xtp_u` nor `ytp_v` rotates corners |
| `ra_x`/`ra_y` inside `xtp_u`/`ytp_v` | **NOT APPLICABLE** | momentum transport is not area-renormalized |

Nothing was EXTENDED: no existing 3A.4/3A.5 routine changed behavior.
`fv_tp_2d` and the PPM machinery were not duplicated, and momentum transport was
*not* forced through the scalar API — `xtp_u`/`ytp_v` have genuinely different
staggering (D-wind on a B-point face), different metric (`dx` not `dxa`), and a
different Courant rescale, so a shared entry point would have required
falsifying one of the two.

## §4 `xtp_u`

`cuda/dyn_core/dgrid/wp_b_momentum_cuda.cu`, `xtp_al`, `xtp_seam_xt`,
`xtp_u_bl_br`, `xtp_u_profile_kernel`, `xtp_u_flux_kernel`,
`launch_xtp_u_cuda`.

Audited and preserved:

- Profile index space `i ∈ is-1..ie+1`, `j ∈ js..je+1` (`u`'s D stagger).
- Interior `al(i) = p1*(u(i-1)+u(i)) + p2*(u(i-2)+u(i+1))`.
- West seam uses the **`dx`-weighted** two-sided extrapolation with `dx(0)`,
  `dx(-1)` then `dx(1)`, `dx(2)`; east uses `dx(npx-1)`, `dx(npx-2)` then
  `dx(npx)`, `dx(npx+1)`. Not `dxa` — that is `xppm`'s metric and the two are
  different fields.
- Branch order east-before-interior matching the Fortran's sequential
  west-then-east overwrite.
- On the `j==1` / `j==npy` rows, `bl(npx-1)`, `br(npx-1)`, `bl(npx)`, `br(npx)`
  are all zero: the `br(npx)` c1/c2/c3 expression sits **inside** the non-edge
  `else` branch (`sw_core.F90:1985-2045`), so the zeroing is not overwritten.
- `smt5` over `is-1..ie+1`, then the WMP degradation `smt5 = bl*br < 0` at
  `{0,1}` when `is==1` and `{npx-1,npx}` when `(ie+1)==npx`.
- Flux loop: `cfl = c*rdx(i-1,j)` for `c>0` else `c*rdx(i,j)` — the **upwind**
  cell's `rdx`; `fx0 = (1-cfl)*(br(i-1) - cfl*b0(i-1))` /
  `(1+cfl)*(bl(i) + cfl*b0(i))`; `flux = u_upwind`, then
  `if (smt5(i-1) .or. smt5(i)) flux += fx0`.

A focused launcher is exposed; there is **no host numerical fallback**.

## §5 `ytp_v`

`ytp_al`, `ytp_seam_xt`, `ytp_v_bl_br`, `ytp_v_profile_kernel`,
`ytp_v_flux_kernel`, `launch_ytp_v_cuda`.

`ytp_v` was inspected line by line rather than assumed to be a transpose. The
outcome: for the `bl`/`br`/`b0`/`smt` construction and the flux loop it **is**
the exact `(i↔j, u→v, dx→dy)` transpose, and the differences are structural
only:

1. Fortran keeps 1-D locals inside a `do j` loop in `xtp_u` but 2-D locals in
   `ytp_v`;
2. the order-6 flag is named `smt5` in `xtp_u` and `smt6` in `ytp_v`;
3. `ytp_v` evaluates the `dy`-weighted seam `xt` for the columns it then zeroes,
   whereas `xtp_u` skips the `dx`-weighted `xt` on the rows it zeroes.

Preserved differences that are *not* transposes:

- The zeroing predicate is `edge_col = (i==1) || (i==npx)` — a **column**
  condition zeroing `j ∈ {0,1}` / `{npy-1,npy}` — and it runs **after** the
  south/north blocks, overwriting `br(i,npy)` which the main `i` loop computed
  unconditionally (`sw_core.F90:2352-2432`).
- Branch order is north-before-interior.
- `cfl = c*rdy(i,j-1)` / `c*rdy(i,j)`.

## §6 D-wind update (row 10)

`dwind_update_kernel`, `launch_dwind_update_cuda`.

```
u(i,j) = vt(i,j) + ke(i,j) - ke(i+1,j) + fy(i,j)   on js..je+1 × is..ie
v(i,j) = ut(i,j) + ke(i,j) - ke(i,j+1) - fx(i,j)   on js..je   × is..ie+1
```

The two index spaces are disjoint, so both are done in one kernel without
fusing across any operation. `vt`/`ut` are read in their **row-8** meaning.
Per-level independence is preserved (`k` is a pure batch dimension). Nothing
after `sw_core.F90:1478` is included.

## §7 Kinetic-energy operations

| Fortran expression | inputs | output | consumer | fusable? | deferred? |
|---|---|---|---|---|---|
| `ke = vb*ub` (`:1123`) | vb, ub | ke | row 6 | no — row 4 overwrites `ub` | no |
| `ke = 0.5*(ke + ub*vb)` (`:1172`) | ke, ub, vb | ke | row 7 | no — needs row 5's `vb` | no |
| four corner overrides, `dt6 = dt/6` (`:1177-1200`) | ut, vt, u, v | ke | row 10 | no — reads the pre-row-8 `ut`/`vt` | no |
| `ke = ke + vort` (`:1462`) | ke, vort | ke | row 10 | — | **DEFERRED to WP-B.4/B.5** (divergence-damping product) |
| `d_con > 1.e-5` heat-source / `diss_est` block (`:1480-1520`) | — | — | — | — | **DEFERRED to WP-B.4/B.5** |

Only the operations required before the next WP-B dependency boundary were
implemented. No downstream divergence or damping work was pulled in.

## §8 Edge/corner behavior implemented

Only what rows 1–10 need immediately:

- `seam_cell_x` / `seam_cell_y` (REUSED from 3A.4/3A.5) for the WMP limiter
  degradation.
- `xtp_u`/`ytp_v` one-sided seam reconstructions and the `bl=br=0` zeroing.
- The vb/ub tile-edge branches, including the **precedence asymmetry**: vb gives
  tile-edge ROWS priority over COLUMNS (`sw_core.F90:1063-1097` writes the
  `js==1` row first and applies the `is==1` column override only inside the
  clipped `j` loop); ub gives COLUMNS priority over ROWS
  (`sw_core.F90:1143-1169` writes the `is==1` column first over the full `j`
  range and clips the `j`-loop `i` range to `is2..ie1`). Also, ub's 4-point
  one-sided formula sits on the `j==1`/`j==npy` ROWS reading `ut(j-2..j+1)`,
  while vb's sits on the `i==1`/`i==npx` COLUMNS reading `vt(-1..npx+1)`.
- The four KE corner overrides, each with its own neighbour set and third-term
  sign.

No generalized `corner_edge_service` was attempted — that stays WP-B.4. No
high-frequency local operation was routed through a generic `LaunchFn` slot.

### Fenced decomposition family (documented limitation, not a defect)

`xtp_u`'s interior reconstruction is clipped to `is3 = max(3, is-1)`,
`ie3 = min(npx-3, ie+1)` while `b0`/`smt` span `is-1..ie+1` and the flux loop
reads them. For a subdomain with `is ∈ {2,3}` (and the `ie+1`, `js`, `je+1`
analogues) the clip leaves `bl`/`br` **unwritten yet read** — the authoritative
Fortran has no defined value there. Rather than invent one, those
decompositions are rejected with `FV3_GPU_UNSUPPORTED_CONFIGURATION` before any
kernel launch (`momentum_decomposition_supported`). `npx/npy >= 8` is also
required so the west index set `{0,1,2}` stays disjoint from the east set
`{npx-2,npx-1,npx}`, which the Fortran's sequential overwrite would otherwise
resolve by position. Realistic GEOS cubed-sphere decompositions satisfy both.

## §9 Test inventory

Permanent test: `cuda/dyn_core/tests/dgrid/test_wp_b3_momentum_cuda.cpp`,
CTest target `fv3_wp_b3_momentum` (label `gpu`, `SKIP_RETURN_CODE 77`).

Every numerical part compares the CUDA result against an **independent host
transcription** written directly from `sw_core.F90` (`ref_xtp_u`, `ref_ytp_v`,
`ref_vb`, `ref_ub`, `ref_ke_product`, `ref_ke_average`, `ref_ke_corner_fix`,
`ref_vorticity`, `ref_vort_absolute`, `ref_dwind`), each transcribed in the
Fortran's own multi-pass structure and NaN-poisoned so an under-written region
fails loudly. The GPU is never compared against itself.

| Part | Coverage | Result (R8 / R4 worst rel) |
|---|---|---|
| A | `xtp_u` interior + PPM-correction liveness | PASS — 4.441e-16 / 4.768e-07 |
| B | `ytp_v` interior + `xtp_u`≠`ytp_v` distinctness | PASS — 4.441e-16 / 2.384e-07 |
| C | `xtp_u` panel edge + west/east seam liveness + zeroed-row exact-upwind | PASS — 4.281e-16 / 2.384e-07 |
| D | `ytp_v` panel edge + south/north seam liveness + zeroed-column | PASS — 4.441e-16 / 2.384e-07 |
| E | vb, ub, rows 8/9/9b/10; non-degenerate momentum change; vb≠ub | PASS — 4.385e-16 / 2.444e-07 |
| F | KE rows 3/6/7, corner fix exactly 4 cells, corners pairwise distinct | PASS — 5.639e-15 / 4.205e-06 |
| G | south-west-only block; vb/ub precedence at (1,1); dx↔dy swap detectors | PASS |
| H | precision-sensitive: plateau round-off branch; `wk` cancellation | PASS — 0 / 0 |
| I | R8 reporting | PASS — 5.547e-15 at tol 1e-11 |
| J | R4 reporting | PASS — 4.205e-06 at tol 3e-05 |
| K | 7 unsupported configurations rejected, state intact; `hord_mt` ±6 bit-identical | PASS |

Defect classes the tests detect (each demonstrated live, not merely asserted):

- **Wrong staggering / swapped x/y metrics** — G3 reruns `xtp_u` with `dy`/`rdy`
  and `ytp_v` with `dx`/`rdx` and requires the answer to change
  (488 / 489 cells).
- **Missing edge behavior** — C/D compare against an unclipped
  interior-formula counterfactual (`interior_flux_x` / `interior_flux_y`) and
  require every seam cell to differ from it (108/108 both directions).
- **Wrong sweep direction / sign** — the synthetic `c` field carries both signs,
  so both flux branches execute; B requires `xtp_u` and `ytp_v` to disagree.
- **Incorrect edge precedence** — G2 computes both candidate formulas for
  `vb(1,1)` and `ub(1,1)` by hand and requires the ROW one for `vb` and the
  COLUMN one for `ub`.
- **Stale / overwritten temporaries** — the sequence driver `gpu_sequence` uses
  the same `vb`/`ub` device buffers across rows 1–6 with snapshots in between,
  so a launcher that read a stale `ub` after row 4 would fail F; the
  NaN poison in every reference catches an under-written output.
- **Leaked corner writes** — F requires the corner fix to change exactly
  `4*nk` cells and zero non-corner cells; G1 requires exactly `nk` (SW only) on
  a block that owns one corner.
- **Degenerate fixture** — E requires ≥300 `u` cells to move, F requires the
  row-6 average to change ≥300 cells, H2 requires the `wk` cancellation
  amplification to exceed 20 (measured 4.4e4).

Harness defect found and fixed during bring-up: `gpu_xtp_u` initially passed
`(u, c)` where `launch_xtp_u_cuda` takes `(c, u)`. This was a **test-driver**
bug — the same launcher called correctly from `gpu_sequence` passed — and it is
recorded here because it is exactly the class of error the milestone is meant to
catch. No numerical source changed.

## §10 Source-level reconciliation

| Item | Classification |
|---|---|
| `xtp_u` interior reconstruction (`sw_core.F90:1985-2000`) | MATCHES FORTRAN |
| `xtp_u` west/east seam `dx`-weighted `xt` (`:2009-2045`) | MATCHES FORTRAN |
| `xtp_u` `j==1`/`j==npy` `bl=br=0` zeroing (`:2020-2045`) | MATCHES FORTRAN |
| `xtp_u` `smt5` + WMP degradation (`:2130-2150`) | MATCHES FORTRAN |
| `xtp_u` flux loop / upwind `rdx` (`:2160-2180`) | MATCHES FORTRAN |
| `ytp_v` reconstruction + seam blocks (`:2352-2432`) | MATCHES FORTRAN |
| `ytp_v` `smt6` + WMP + flux loop (`:2550-2600`) | MATCHES FORTRAN |
| vb construction incl. row-over-column precedence (`:1060-1130`) | MATCHES FORTRAN |
| ub construction incl. column-over-row precedence (`:1130-1170`) | MATCHES FORTRAN |
| `ke = vb*ub` (`:1123`), `ke = 0.5*(ke+ub*vb)` (`:1172`) | MATCHES FORTRAN |
| four-corner KE fix, `dt6 = dt/6` (`:1177-1200`) | MATCHES FORTRAN |
| vorticity precursor `vt`/`ut`/`wk` (`:1205-1235`) | MATCHES WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE — split into two launches so the `wk` pass observes completed `vt`/`ut`; the Fortran gets this from loop sequencing. Numerical semantics changed: NO |
| `vort = wk + f0[*z_rat]` (`:1440`) | MATCHES FORTRAN |
| `fv_tp_2d(vort, hord_vt)` (`:1456`) | MATCHES WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE — delegated to the Milestone 3A.5 service rather than reimplemented. Numerical semantics changed: NO |
| D-wind update (`:1466-1478`) | MATCHES FORTRAN |
| `bl`/`br`/`b0`/`smt` materialized as device arrays | MATCHES WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE — Fortran keeps them as automatic arrays inside a `j` (resp. `i`) loop; the flux loop reads index `i-1`, so on the GPU they must be materialized. Numerical semantics changed: NO |
| vb/ub written by a single per-point kernel | MATCHES WITH EXPLAINED CUDA STRUCTURAL DIFFERENCE — each output point is written exactly once by the Fortran's multi-pass write order given the decomposition fence, so a single-pass kernel with resolved branch precedence is faithful. Numerical semantics changed: NO |
| `is ∈ {2,3}` decomposition family | UNSUPPORTED BY CONSTRUCTION (rejected before launch) — not a discrepancy: the Fortran reads undefined values there, so there is no authoritative behavior to reproduce |
| `abs(hord_mt) == 6` gate | MATCHES FORTRAN — `xtp_u`/`ytp_v` have no `iord<0` branch, so ±6 are the same code path (test K asserts bit-identity) |

**Known discrepancies inside WP-B.3 scope: NONE.**

## §11 Build and GPU execution

Environment (login node, per `docs/agent_instructions/discover_platform.md`):
GCC/G++/GFortran 12.3.0, OpenMPI 4.1.6, `nvidia/nvhpc-byo-compiler/24.7`
(nvcc 12.5), CMake 3.30.3.

```bash
cmake -S cuda/dyn_core -B build_cuda_r8 -DFV3_DYN_CORE_ENABLE_CUDA=ON \
      -DFV3_DYN_CORE_PRECISION=R8 -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build_cuda_r8 -j 12
cmake -S cuda/dyn_core -B build_cuda_r4 -DFV3_DYN_CORE_ENABLE_CUDA=ON \
      -DFV3_DYN_CORE_PRECISION=R4 -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build_cuda_r4 -j 12
```

- R8 build: PASS
- R4 build: PASS

GPU execution, one A100 through Slurm (`gpu_a100`, `--constraint=rome`,
`--gres=gpu:1`, `--mem=60G`, `--cpus-per-task=12`), node `warpa005`,
NVIDIA A100-SXM4-40GB `sm_80`:

- `fv3_wp_b3_momentum_test` R8: `ALL PARTS PASS (11 parts)`, `exit=0`
- `fv3_wp_b3_momentum_test` R4: `ALL PARTS PASS (11 parts)`, `exit=0`
- `ctest build_cuda_r8`: **15/15 PASS**, 0 failed
- `ctest build_cuda_r4`: **15/15 PASS**, 0 failed

A confirming rebuild-and-rerun after the final documentation edits reproduced
`100% tests passed, 0 tests failed out of 15` for both precisions on node
`warpa004`.

Actual CUDA kernel execution is established by the comparisons themselves: every
compared field is downloaded from device memory after
`cudaDeviceSynchronize()`, and a non-launching path would leave the NaN-poisoned
or zero-initialized buffers, which fails. Output is finite everywhere compared,
momentum change is non-degenerate (468 `u` cells moved, max `|du|` 2.5e6), and
edge/seam treatment is demonstrably live (108/108 seam cells per direction).
No CUDA runtime error was reported; `main` additionally checks the residual
`cudaGetLastError()`.

No CUDA executable was run on a login node.

## §12 compute-sanitizer

`/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/compute-sanitizer`,
`--error-exitcode 9`, batch job `58064293` on `warpa005`:

| Tool | R8 | R4 |
|---|---|---|
| memcheck | `ERROR SUMMARY: 0 errors`, exit 0 | `ERROR SUMMARY: 0 errors`, exit 0 |
| initcheck | `ERROR SUMMARY: 0 errors`, exit 0 | `ERROR SUMMARY: 0 errors`, exit 0 |
| racecheck | `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`, exit 0 | same, exit 0 |

All six runs also printed `ALL PARTS PASS (11 parts)` under the sanitizer.
`racecheck` covers the profile and corner-write kernels, which are the ones with
plausibly overlapping edge/corner writes. `initcheck` is the meaningful check on
the staggered `u`/`v` bounds, the shifted halos and the reused `bl/br/b0/smt`
workspace: an out-of-range or unwritten-then-read slot would surface here.

**No unresolved sanitizer issue.**

## §13 Dispatch wiring

`NumericalDispatch.dgrid_dynamics` remains `nullptr`. The rationale comment in
`cuda/dyn_core/runtime/fv3_gpu_production_dispatch.cu` was updated to record
that 3A.4/3A.5/3A.6 have landed three of d_sw's slices and to enumerate what
still blocks the slot (divergence damping, the `d_con` heat-source block, the
`damp_v` diffusive-flux add, the generalized corner service, and the
order-critical `pt`/`delp`/`w` updates plus flux-capacitor accumulation).

`dgrid_height_transport` is wired and unchanged. Momentum transport is exposed
only as internal WP-B CUDA services that WP-B.5 can call in authoritative
order.

## §14 Regression check

`ctest` 15/15 in both precisions on the GPU node covers, unchanged:

- `fv3_wp_b_cuda_smoke` — 3A.4 height transport, `copy_corners`,
  `edge_profile`, both damping branches, `del6_vt_flux`, the area-weighted `zh`
  update, and the cross-WP path through WP-C's `launch_update_dz_d`.
- `fv3_wp_b2_scalar_transport` — 3A.5 scalar transport, `deln_flux`, `nord`
  dependence, the `(damp_c*da_min)**(nord+1)` law, mass weighting, `hord<0`
  positivity, PPM edges and corners.
- `fv3_dispatch_smoke`, `fv3_integrated_smoke` — WP-E runtime / production
  dispatch.

No 3A.4 or 3A.5 regression.

## §15 Validation level statement

This milestone is implementation + GPU execution + source-level reconciliation.

**Numerical validation against runtime Fortran: NOT RUN.** No Fortran savepoints
were generated or compared. The host transcriptions used by the tests are
independent re-derivations from `sw_core.F90` source, which is a strong
source-level cross-check but is *not* runtime Fortran reference data.

## Remaining WP-B work

1. **WP-B.4 — divergence / corner slice.** `divergence_corner`, `delpc`,
   `divg_d`, `smag_corner`, `ke += vort` (`:1462`), the `d_con > 1.e-5`
   `ub/vb/fx/fy/gx/gy` + `heat_source` / `diss_est` block (`:1480-1520`), the
   `damp_v` `del6_vt_flux` on `wk` and the `u += vt` / `v -= ut` diffusive-flux
   add (`:1522-1534`), and the generalized `corner_edge_service` /
   `fill_corners`.
2. **WP-B.5 — the ordered `d_sw` driver.** Compose 3A.4/3A.5/3A.6/WP-B.4 in
   authoritative order, add the `pt`/`delp`/`w` state updates and the
   `cx`/`cy`/`xflux`/`yflux` flux-capacitor accumulation, then populate
   `NumericalDispatch.dgrid_dynamics`.
3. **Prompt 6.** Generate authoritative Fortran savepoints at the dyn_core
   savepoint boundaries and perform the real numerical validation.

## Next recommended phase

WP-B divergence/corner slice (WP-B.4).
