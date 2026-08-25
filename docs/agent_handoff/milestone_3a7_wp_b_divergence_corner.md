# Milestone 3A.7 — WP-B.4 Authoritative Divergence Damping / Corner Slice

## Status summary

| Level | Result |
| --- | --- |
| IMPLEMENTED | YES — all 13 launchers declared by `wp_b_divergence_cuda.cuh` |
| COMPILED | R8 PASS, R4 PASS (login node, zero new warnings) |
| GPU EXECUTED | PASS — A100-SXM4-40GB (`warpa004`), 12/12 test parts and 16/16 ctest, both precisions |
| MEMORY-SANITIZED | PASS — memcheck / initcheck / racecheck, both precisions, 0 errors, 0 hazards |
| NUMERICALLY VALIDATED against authoritative Fortran runtime output | **NOT RUN** — that is Prompt 6 |

The GPU-executed and sanitized results are against **host references transcribed
from the Fortran inside this repository's test**, not against Fortran savepoint
data produced by a running model. Those are different claims and this milestone
only supports the former. See §11.

## Git state

- Branch: `cuda/gpu-bringup`
- Parent commit: `c3f34b0` *Record milestone 3A.6 commit SHA in handoff*
- Commit SHA for this milestone: `1fe108c`

Files added:

| File | Lines | Role |
| --- | --- | --- |
| `cuda/dyn_core/dgrid/wp_b_divergence_cuda.cuh` | 538 | contract / stage table / rationale |
| `cuda/dyn_core/dgrid/wp_b_divergence_cuda.cu` | 995 | the 13 launchers |
| `cuda/dyn_core/tests/dgrid/test_wp_b4_divergence_cuda.cpp` | 2062 | 12-part behavior test |

Files modified:

| File | Change |
| --- | --- |
| `cuda/dyn_core/dgrid/wp_b_cuda.cuh` | `LevelFilter::gate` (optional activation array) |
| `cuda/dyn_core/dgrid/wp_b_ppm_shared.cuh` | `level_active` tests `gate` when non-empty |
| `cuda/dyn_core/CMakeLists.txt` | new `.cu` in `fv3_wp_b_cuda`; `fv3_wp_d_cuda` link; new test target |
| `cuda/dyn_core/cgrid/fv3_cgrid_divergence.{cu,hpp}` | audit repair — see §4 |
| `cuda/dyn_core/runtime/fv3_gpu_production_dispatch.cu` | `dgrid_dynamics` rationale updated to 3A.7 |

## Authoritative Fortran revision

`GFDL_atmos_cubed_sphere`, tag `geos/v3.0.0`, SHA
`ca30e26305fda43831095334c02a3f5e0b8c0c45`, read-only clone at
`/discover/nobackup/projects/imvi-nse/jli30/fv3_reference/GFDL_atmos_cubed_sphere`.

Ported ranges:

- `model/sw_core.F90:1277-1534` — `d_sw`'s divergence-damping / corner slice
- `model/sw_core.F90:1857-1947` — `smag_corner`
- `model/sw_core.F90:1544-1604` — `del6_vt_flux` (reused, not re-transcribed)
- `tools/fv_mp_mod.F90:1113-1186` — `fill_corners` BGRID scalar
- `tools/fv_mp_mod.F90:1338-1370` — `fill_corners` DGRID vector pair
- `model/dyn_core.F90:697-760` — the per-level `nord_k` / `damp_vt` / `d_con`
  recomputation that forces the selector design below

## §1 Why every launcher takes a level selector, not a scalar `nord`

`dyn_core.F90:697-760` recomputes `nord_k` **inside** the k loop and calls
`d_sw` once per level. With `n_sponge > 0` the sponge overrides force `nord_k = 0`
on the top few levels while the rest of the column keeps `flagstruct%nord > 0`.

Consequence: **both damping branches are live in a single production run.** A
port that took a scalar `nord` and picked one arm would be wrong for every
sponge configuration. So every launcher takes `DivergenceLevels`, whose `mode`
selects which levels it acts on:

| mode | meaning |
| --- | --- |
| 0 | all levels |
| 1 | `nord_k == 0` (branch L) |
| 2 | `nord_k == nord_sel` (branch H, one nord group) |
| 3 | `d_con > 1e-5` |

The level index is clamped by `lv_clamp`, so `k >= npz` reads `npz-1` —
mirroring the Fortran's `damp(km+1) = damp(km)` convention.

## §2 Stage inventory (13 launchers)

Full ordered table with Fortran line numbers, inputs/outputs, staggering, halo
reach and corner semantics is in `wp_b_divergence_cuda.cuh` §1. Summary:

| Rows | Launcher | Fortran |
| --- | --- | --- |
| L1–L5 | `launch_divdamp_nord0_cuda` | `:1290-1332` |
| H1 | `launch_divdamp_save_delpc_cuda` | `:1364-1368` |
| H2a/H2c | `launch_fill_corners_bgrid_cuda` (dir 1 / 2) | `fv_mp_mod:1113-1186` |
| H2b/H2d/H2f/H2g | `launch_divdamp_iterate_cuda` | `:1370-1403` |
| H2e | `launch_fill_corners_dgrid_vector_cuda` | `fv_mp_mod:1338-1370` |
| H3 | `launch_divdamp_vort_zero_cuda` | `:1407-1411` |
| H3 | `launch_smag_corner_cuda` | `:1857-1947` |
| H4 | `launch_divdamp_finalize_cuda` | `:1413-1423` |
| D1 | `launch_dcon_vort_diff_cuda` | `:1425-1436` |
| D2 | `launch_dcon_scale_cuda` | `:1482-1495` |
| D3 | `launch_dcon_heat_source_cuda` | `:1497-1520` |
| V1 | `launch_vorticity_damping_cuda` | `:1476-1479` (adapter, see §5) |
| V2 | `launch_dwind_damping_add_cuda` | `:1522-1534` |

## §3 No duplicated numerical functionality

Two reuses, both deliberate, both required by the project no-duplication rule:

1. **`smag_corner` consumes WP-D's `a2b_ord4`.** `smag_corner` needs
   `a2b_ord4` to move the A-grid shear strain to cell corners.
   `operators::launch_a2b_ord4_global_cuda` already implements it, so
   `launch_smag_corner_cuda` calls it rather than transcribing a second copy.
   This is the only cross-work-package call in WP-B and is why
   `fv3_wp_b_cuda` now links `fv3_wp_d_cuda`. No cycle: `fv3_wp_d_cuda` does
   not depend on `fv3_wp_b_cuda`.
2. **`launch_vorticity_damping_cuda` is an adapter over
   `launch_del6_vt_flux_cuda`.** `:1476-1479` calls the same `del6_vt_flux`
   the 3A.4 height path already uses. The adapter contributes argument
   validation and aliasing only — zero new numerical code. Part J proves it
   bitwise (§9).

Two authoritative asymmetries preserved rather than "cleaned up":

- `smag_corner`'s tension-strain sign pattern is
  `vt(i,j-1)-vt(i,j)-ut(i-1,j)+ut(i,j)` — the `ut` terms are sign-flipped
  relative to `divergence_corner`'s `+uf(i-1,j)-uf(i,j)`.
- `smag_c` is scaled by `rarea_c` **before** its four corner removals, whereas
  `divergence_corner` applies its corner removals **before** the `rarea_c`
  scale. The corner values therefore differ by a factor of `rarea_c` between
  the two routines. This is not a typo in the Fortran and is reproduced.

`smag_corner`'s own doc comment says "work only if (grid_type==4)". The code
contains no `grid_type` test, and `d_sw` calls it for every `grid_type`
whenever `nord > 0 .and. dddmp >= 1.E-5`. The comment is stale; the behavior
is ported, not the comment.

## §4 Audit of WP-A's `divergence_corner` (task carried into this milestone)

`divergence_corner` produces `divg_d`, which `d_sw` only **consumes** (H1 saves
it; H2b/H2d/H2f iterate on it in place). It is called from `c_sw`
(`sw_core.F90:175-181`), not `d_sw`, so it belongs to WP-A and is already
implemented as `cgrid::launch_divergence_corner_level`. Per the no-duplication
rule this milestone audited that implementation instead of writing a second
one. Result: one defect repaired, one gap documented, one wrong citation fixed.

### 4.1 Citation corrected

`fv3_cgrid_divergence.cu:1` and `.hpp:7` both cited
`model/sw_core.F90:1694-1799`. The subroutine is at **`1659-1764`**; `1694` is
where the `nested` selection begins and `1799` is inside the *different*
subroutine `divergence_corner_nest`. Both citations corrected.

### 4.2 DEFECT REPAIRED — unwritten halo rings in the `grid_type > 3` arm

`divergence_corner_simple` launched from origin `(is-1, js-1)` while guarding
on `i >= is-2` (for `uf`) and `j >= js-2` (for `vf`). Those two sub-guards were
therefore **unreachable**, and the corresponding rings were never written:

- Fortran writes `uf` over `i = is-2..ie+2` (`:1702`); `uf` is declared
  `is-2:ie+2` (`:1668`).
- Fortran writes `vf` over `j = js-2..je+2` (`:1706`); `vf` is declared
  `js-2:je+2` (`:1669`).

`divergence_corner_simple_divg` then reads `uf(i-1,j)` at `i = is-1` — i.e.
`uf(is-2,j)` — and `vf(i,j-1)` at `j = js-1` — i.e. `vf(i,js-2)`. Both were
uninitialized workspace.

**Repair:** origin changed to `(is-2, js-2)` in that kernel only. The launch
extents (`ie-is+5`, `je-js+5`) were already exactly the Fortran loop counts, so
only the origin was wrong; the previously-dead guards now become live and pin
the correct per-array rectangles. `divergence_corner_simple_divg` keeps origin
`(is-1, js-1)`, which is what `:1711-1712` writes.

**Numerical semantics changed: NO.** The change writes two previously-unwritten
halo rings using the same formula the Fortran uses; no value that was already
being written changes.

**Validation level of this repair: COMPILED only (R8 and R4).** No test in the
repository calls `launch_divergence_corner_level` — it currently has **no
caller at all** — so the repair is not GPU-executed and not covered by this
milestone's sanitizer run (`fv3_wp_b4_divergence_test` does not link
`fv3_wp_a_cgrid`). A WP-A behavior test is required; see "Remaining work".

**Production exposure: none.** The defective arm is `grid_type > 3`; the
production first-run configuration uses `grid_type = 0`.

### 4.3 GAP DOCUMENTED — nested grids are unsupported but not rejected

`:1694-1698` selects `is2 = is; ie1 = ie+1` when `nested`, versus
`is2 = max(2,is); ie1 = min(npx-1,ie+1)` otherwise. `DivergenceCornerArgs`
carries **no `nested` flag**, so `divergence_corner_vf` unconditionally applies
the non-nested form. The header said "nested grids are milestone-one
unsupported", but nothing rejects a nested call — it would silently run the
global formula.

Not repaired here: closing it means adding a field to a WP-A-owned struct on a
WP-B milestone, for a launcher with no caller. Both the `.cu` and `.hpp` now
state the gap explicitly and name the obligation: whoever wires the first
caller must add the flag and reject nested.

### 4.4 The `grid_type <= 3` arm (the production path) is faithful

Checked index by index against `:1722-1760` and found correct: the `uf` range
`i=is-1..ie+1`, `j=js..je+1` including the `j==1 .or. j==npy` simple form; the
`vf` range `j=js-1..je+1`, `i=is2..ie1`; the `is==1` and `(ie+1)==npx` `vf`
column overrides; the `divg_d` range `is..ie+1 × js..je+1`; and the ordering
that applies the four corner removals to the **unscaled** `divg_d` before the
`rarea_c` scale.

One thing that looks like a bug and is not: the CUDA leaves `divg_d`'s halo
unwritten in this arm. The Fortran does too — `divg_d` is `intent(out)`
declared `(isd:ied+1, jsd:jed+1)` but the cubed-sphere arm assigns only
`is..ie+1 × js..je+1`, so its halo is undefined on return there as well.
Matching that is faithful.

## §5 `LevelFilter` gained a `gate`, and why

`sw_core.F90:1476-1479`:

```fortran
if ( damp_v > 1.E-5 ) then
     damp4 = (damp_v*gridstruct%da_min_c)**(nord_v+1)
     call del6_vt_flux(nord_v, npx, npy, damp4, wk, vort, ut, vt, ...)
```

The **activation** value (`damp_v`) and the **coefficient** (`damp4`) are
different numbers. This is unlike `update_dz_d` (`nh_utils.F90:280`), which
gates `del6_vt_flux` on the very coefficient it hands the routine — so
`LevelFilter::damp` alone was sufficient for the 3A.4 height path but is *not*
sufficient here.

`LevelFilter::gate` carries the activation array separately, and
`level_active` tests `gate(kk)` in preference to `damp(kk)`. **When `gate` is
empty the test is exactly the historical `damp(kk)` test**, so the 3A.4 height
path and 3A.5 scalar path are bit-for-bit unaffected. Part I is built
specifically to catch a regression here (§9).

## §6 `fill_corners` is a local index rotation, not communication

Both `fill_corners` variants are **LOCAL** `ng × ng` corner-block rotations
(`ng = 3`, the `fv_mp_mod.F90:108` parameter — distinct from the registered
`halo_width`). No MPI, no synchronization boundary.

Corner ownership predicates: the BGRID variant spells them `is==1` /
`ie==npx-1` / `js==1` / `je==npy-1`; the DGRID variant spells them `is==1` /
`ie+1==npx` / `js==1` / `je+1==npy`. These are **arithmetically identical** —
the Fortran simply writes the same condition two ways. (An earlier draft of
the header claimed a cell-index-vs-corner-index distinction between them; that
claim was vacuous and has been corrected.)

The DGRID vector pair uses `mySign = -1` on the **SW and NE corners only**
(`:1350-1353` for `x`, `:1362-1365` for `y`); NW and SE take `+1`. The two
Fortran loops are implemented as **two sequential kernel launches** on the same
stream, in the Fortran's order — not fused. Fusing would in fact be safe (loop
2 reads `x` only at first index in `1..ng` or `npx-ng..npx-1`, while loop 1
writes `x` only at first index `<= 0` or `>= npx`, so the sets are disjoint),
but the launches are kept separate so the port has the same statement order as
the source and does not depend on that disjointness argument remaining true.

## §7 One race found and fixed during implementation

`corner_term_removal_kernel` (H2g, `:1399-1403`) applies four scalar
single-point corner corrections. Every thread in a block whose tile covered a
corner was applying the correction, so the update was applied up to
`blockDim.x * blockDim.y` times. Fixed with an
`if (threadIdx.x != 0 || threadIdx.y != 0) return;` guard so exactly one thread
performs each single-point update. racecheck reports 0 hazards (§12).

Note that H2g runs on **every** sweep, not only when `fill_c` is true — a
detail the reference implementation in the test reproduces explicitly.

## §8 `dddmp` is not absolute-valued

`:1418` is `min(0.20, dddmp*vort(i,j))` with **no `abs()`**. A negative
`vort` therefore produces a negative limiter value, which then loses to `d2`
in the following `max`. The port reproduces this exactly; Part F's fixture
drives `dddmp = 3.0` with sign-crossing `vort` so both arms of that `max` are
exercised (169 cells took the `dddmp` arm, 338 took the `d2` arm, 71 hit the
0.20 clamp).

## §9 Test inventory — `fv3_wp_b4_divergence_test`

Every part compares CUDA output against an **independent host transcription
written from `sw_core.F90` in the Fortran's own multi-pass structure**. The GPU
code is never compared against itself. Two deliberate exceptions are stated in
the test header:

- Part G's reference `sh` comes from a direct call to WP-D's already-validated
  `launch_a2b_ord4_global_cuda` (transcribing a second `a2b_ord4` would violate
  the no-duplication rule). Parts of the chain around it — `wk` from S5–S7 —
  are still compared against the host transcription, so S5–S7 is checked
  independently of `a2b_ord4`.
- Part J is a bitwise adapter-identity check, which is the point of that part.

| Part | What it establishes |
| --- | --- |
| A | branch L at a cubed-sphere seam; limiter spread asserted |
| B | branch L in the interior (no panel edges) |
| C | `fill_corners` BGRID scalar, both directions |
| D | `fill_corners` DGRID vector pair; `mySign` live |
| E | branch H multi-sweep iterate; sweep and fill counts asserted |
| F | `vort` zeroing + the high-order finalize; all three limiter arms |
| G | `smag_corner` including its `a2b_ord4` dependency |
| H | the `d_con` heat-source chain D1/D2/D3 |
| I | the `damp_v` wind update; `gate`-vs-`damp` discrimination |
| J | bitwise identity of the V1 adapter with `launch_del6_vt_flux_cuda` |
| K | unsupported-configuration rejection; **0 words mutated** |
| L | `DivergenceLevels` selector semantics + the `npz` clamp |

Design points worth preserving if these tests are ever edited:

- **Parts E, F, G, H, I all assert that filtered-out levels are bitwise
  unchanged.** A launcher that ignored the selector and processed every level
  would fail.
- **Part E deliberately passes the wrong selector** (`mode = 1, nord_sel = 0`)
  to `launch_divdamp_iterate_cuda`, which must override it and iterate the
  nord groups itself. It asserts `sweeps == 6 && fills == 3` for
  `nord_k = {0,1,2,3}`.
- **Part I isolates `gate` from `damp`**: `gate = {0, 0.5, 0.5}` while
  `damp = {1e9, 1e9, 1e9}` on *every* level. A filter that tested `damp`
  instead of `gate` would update level 0 and fail the part.
- **Part K asserts zero mutation across all 10 output buffers** for 13 invalid
  configurations plus null-pointer, out-of-range `nord_cap` and aliased
  `x == y` cases — i.e. rejection happens *before* state is touched.
- **Part G asserts `max|sh| > 1e-12`** so the `a2b_ord4` comparison cannot pass
  vacuously (observed `6.75e-05`).

## §10 Build

Login node, validated stack (GCC 12.3.0 / OpenMPI 4.1.6 /
`nvidia/nvhpc-byo-compiler/24.7`, nvcc 12.5.82, A100 `sm_80`):

```bash
module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv
module load comp/gcc/12.3.0
module load mpi/openmpi/4.1.6/gcc-12.3.0
module load nvidia/nvhpc-byo-compiler/24.7

cmake --build build_cuda_r8 -j 12   # R8: exit 0
cmake --build build_cuda_r4 -j 12   # R4: exit 0
```

Zero new warnings in either precision. (Five pre-existing unused-variable
warnings in `cgrid/fv3_cgrid_predictor.cu` are untouched by this milestone.)

## §11 GPU execution

Node `warpa004`, NVIDIA A100-SXM4-40GB, via `srun --partition=gpu_a100
--constraint=rome --gres=gpu:1 --mem=60G --cpus-per-task=12`.

```text
R8: WP-B.4: all 12 parts PASS   ->  ctest 16/16 passed, 0 failed
R4: WP-B.4: all 12 parts PASS   ->  ctest 16/16 passed, 0 failed
```

Per-part agreement with the host references (0 mismatches everywhere):

| Part | R8 peak rel dev | R4 peak rel dev | non-degeneracy witness |
| --- | --- | --- | --- |
| A branch L, seam | 1.13e-15 | 9.82e-07 | \|delpc\|max 1.77e-4 |
| B branch L, interior | 5.62e-15 | 6.95e-07 | \|delpc\|max 1.26e-4 |
| C fill BGRID | — | — | 108 cells rotated per direction |
| D fill DGRID | — | — | 216 cells rotated (expected 216) |
| E branch H iterate | 0 | 0 | 507/507 cells moved |
| F H3/H4 finalize | 2.36e-15 | 1.27e-06 | clamp 71 / dddmp 169 / d2 338 |
| G smag_corner | 6.94e-18 | 3.73e-09 | max\|sh\| 6.75e-05 |
| H d_con chain | 0 | 0 | 288 cells moved |
| I damp_v add | — | — | gate honoured on 156 cells |
| J V1 adapter | bitwise, 1083 words | bitwise, 1083 words | 1944 output words produced |
| K rejection | — | — | 13 + 5 rejected, 0 words mutated |
| L selector | — | — | 5 modes + npz clamp correct |

Tolerances: `3e-5` (R4), `1e-11` (R8). `close_rel` returns false on NaN, and
scratch is NaN-poisoned before each part, so a kernel that failed to write a
cell fails the comparison rather than passing on a stale zero.

**What this does and does not establish.** It establishes that the CUDA
launchers reproduce host code transcribed from the authoritative Fortran, on
real hardware, in both precisions. It does **not** establish agreement with
output from a running Fortran model — no savepoint comparison has been made.
Authoritative numerical validation remains Prompt 6.

## §12 compute-sanitizer

Batch job on `gpu_a100` (`warpa004`), using the absolute tool path — the
`nvidia/nvhpc-byo-compiler/24.7` module puts `nvcc` on `PATH` but **not**
`compute-sanitizer`:

```text
/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/compute-sanitizer
```

Six runs, `--error-exitcode 9`, all exit 0:

| precision | tool | result |
| --- | --- | --- |
| R8 | memcheck | ERROR SUMMARY: 0 errors |
| R8 | initcheck | ERROR SUMMARY: 0 errors |
| R8 | racecheck | 0 hazards displayed (0 errors, 0 warnings) |
| R4 | memcheck | ERROR SUMMARY: 0 errors |
| R4 | initcheck | ERROR SUMMARY: 0 errors |
| R4 | racecheck | 0 hazards displayed (0 errors, 0 warnings) |

`OVERALL_SANITIZER_FAIL=0`. All 12 parts still PASS under every tool.

initcheck passing is the meaningful one for a port full of halo-ring writes:
it establishes that no kernel in this slice reads uninitialized device memory.
Note this run covers WP-B.4 only — `fv3_wp_b4_divergence_test` does not link
`fv3_wp_a_cgrid`, so it says nothing about §4.2's repair.

## §13 Dispatch wiring — deliberately still `nullptr`

`d.dgrid_dynamics` remains `nullptr` and `d.corner_edge_service` remains bound
to `corner_service_not_implemented`. Four of `d_sw`'s slices now exist as
reusable WP-B services (3A.4 height transport, 3A.5 scalar transport, 3A.6
momentum transport, 3A.7 divergence damping), but `d_sw` additionally needs
WP-B.5: the generalized `corner_edge_service`, the order-critical
`pt`/`delp`/`w` state updates, the `cx`/`cy`/`xflux`/`yflux` accumulation, and
then the single ordered driver that calls the four services in `d_sw`'s own
sequence with the per-level selectors `dyn_core.F90:697-760` recomputes.

Wiring a partial `dgrid_dynamics` would let the production path silently skip
stages. It stays unwired until WP-B.5 closes the ordered path. The rationale
comment in `fv3_gpu_production_dispatch.cu` has been updated to say this.

## §14 Regression check

Full `ctest` in both precisions: **16/16 passed, 0 failed**, including the
3A.4, 3A.5 and 3A.6 WP-B tests. The `LevelFilter::gate` addition did not
perturb the height or scalar paths, as designed (§5).

## §15 Validation level statement

```text
Milestone 3A.7 / WP-B.4
  IMPLEMENTED ............................. YES (13/13 launchers)
  COMPILED ................................ R8 PASS, R4 PASS
  GPU EXECUTED ............................ PASS (A100 warpa004, 12/12 parts, 16/16 ctest, R8+R4)
  MEMORY-SANITIZED ........................ PASS (memcheck/initcheck/racecheck, R8+R4, 0 errors)
  NUMERICALLY VALIDATED vs Fortran output . NOT RUN  (Prompt 6)

WP-A divergence_corner repair (§4.2)
  IMPLEMENTED ............................. YES
  COMPILED ................................ R8 PASS, R4 PASS
  GPU EXECUTED ............................ NOT TESTED (no caller, no test exists)
  NUMERICALLY VALIDATED ................... NOT RUN
```

## Remaining work

1. **WP-B.5** — the ordered `d_sw` driver: `corner_edge_service`, the
   `pt`/`delp`/`w` state updates, `cx`/`cy`/`xflux`/`yflux` accumulation, then
   populate `d.dgrid_dynamics`. This is the last WP-B slice.
2. **WP-A behavior test for `divergence_corner`** covering both `grid_type`
   arms, which would GPU-execute §4.2's repair. Currently the launcher has no
   caller and no test.
3. **Add `nested` to `DivergenceCornerArgs` and reject it** when the first
   caller is wired (§4.3).
4. **Prompt 6** — authoritative savepoint validation against the Fortran at
   the dyn_core savepoint boundaries. Nothing in milestones 3A.3–3A.7 is a
   substitute for this.

## Next recommended phase

WP-B.5, the ordered `d_sw` driver. It is the only thing standing between the
four completed WP-B transport/damping services and a populated
`d.dgrid_dynamics` slot, and therefore between this branch and integrated
`d_sw` GPU execution.

---

Commit SHA for this milestone: `1fe108c8b14361752c790d9f861550a10e3d3119`
