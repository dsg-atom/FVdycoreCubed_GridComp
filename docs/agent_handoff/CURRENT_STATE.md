# Current Project State

_Last regenerated from Git + handoffs on 2026-08-31 (§1/§4/§5/§6/§7/§8/§9/§10
updated for the Prompt 6B **`omga` exit finalization** pass). Result: **the
dyn_core exit `omga` now VALIDATES at SMALL NUMERICAL DIFFERENCE on all six
tiles** (worst max_abs 4.875354e-14, guarded max_rel 1.505773e-12, rms
≤1.073e-14), down from max_abs ≈4.7e-02 / max_rel ≈1.0 — a ~1e12× residual
reduction. **All 13 dyn_core-exit fields now validate, all 15 in-loop boundaries
still validate, and GATE C2 IS CLOSED: the CUDA dyn_core is NUMERICALLY
VALIDATED against authoritative Fortran** for the c8 / npz=10 / n_split=3 /
k_split=1 / R8 / six-tile production first-run configuration. One fix applied:
the previously **entirely absent** omega finalization path
(`dyn_core.F90:503-517` `pem` snapshot, `:1211` `(pe−pem)*rdt`, `:1218`
`adv_pe` → `:1535-1636`) is now implemented on both the host and device sides,
with `en1/en2/ec1/ec2` **derived** from the already-authoritative B-grid corner
geometry (proven, not invented — see §4) and `a2b_ord2` **reused** rather than
rewritten. **Numerical algorithm changed: NO.** Next phase is post-validation
performance retest / optimization planning; the §6 performance baseline remains
**STALE — RETEST REQUIRED**._ Read this file first; avoid re-reading older
milestone handoffs unless a specific open item below requires it.

## 1. Git State

- Repository: `FVdycoreCubed_GridComp`
  - Primary path: `/discover/nobackup/projects/imvi-nse/jli30/FVdycoreCubed_GridComp`
  - Also visible at: `/gpfsm/dnb06/projects/p295/jli30/FVdycoreCubed_GridComp`
- Branch: `cuda/gpu-bringup`
- HEAD: `a060d98ca68a2996bf54d6bd214bbc3f3f977e8e` ("Add cost-efficient Claude workflow guidance")
- Ahead of `origin/cuda/gpu-bringup` by 4 commits (`2c57d56`, `6fc409a`,
  `aa149dc`, `a060d98`) — **not pushed**.
- Uncommitted Prompt 6B work in the tree (modified, not staged, not committed):
  - `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` — `--bdt`,
    `--sp1-boundary`, `kFv3Rdgas`, `--damp-n-sponge`, `--akap-exner`,
    `namespace grid_geometry_6b`, and the `grid_lonlat`/`agrid_lonlat` store-order
    fix.
  - `cuda/dyn_core/include/fv3_gpu_internal.hpp`,
    `cuda/dyn_core/operators/wp_d_contract.cpp`,
    `cuda/dyn_core/operators/wp_d_contract_cuda.cpp`,
    `cuda/dyn_core/vertical/fv3_vertical_column.hpp`,
    `cuda/dyn_core/vertical/fv3_vertical_kernels.cu`,
    `cuda/dyn_core/tests/vertical/test_vertical_column.cpp` — the
    `NumericalConfig::akap_exner` two-kappa split plumbing.
  - `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` — the `it==1` `H_DSCALAR`
    resolve; the E11 `it==n_split` seam block + `FV3_GPU_DBG_STAGE("seam_reconcile")`;
    the post-loop `dyn_core_exit` checkpoint with six `sp3` fields appended to
    `targets[]`/`names[]`; the pre-loop heat-accumulator setup; extra dump targets.
  - **E11 / heat-accumulation work (this pass):**
    `cuda/dyn_core/include/fv3_gpu_communication.h` (`FV3_GPU_H_DWIND_SEAM` +
    `FV3_GPU_H_GROUP_LAST`), `cuda/dyn_core/runtime/fv3_gpu_halo_exchange.{hpp,cu}`
    (`ring_begin`, `dest_edges`, `halo_ring_count`/`halo_ring_ordinal`, the
    `H_DWIND_SEAM` table entry), `cuda/dyn_core/runtime/fv3_gpu_halo_topology.hpp`
    (corrected stale seam comment), `cuda/dyn_core/runtime/fv3_gpu_context.hpp`
    (`heat_accum`), new `cuda/dyn_core/runtime/fv3_gpu_heat_accumulator.{hpp,cu}`,
    `CMakeLists.txt`, `cuda/dyn_core/tests/runtime/test_halo_topology.cpp`
    (T10 branch, `live_groups` 10→11),
    `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp`
    (`kRuntimeResolvedGroups[]` + `H_DWIND_SEAM`).
  - `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` also carries the
    authoritative `n_con` derivation (§4 #11) and the authoritative
    `nf_ke = min(3, nord+1)` derivation (§4 #13).
  - **`nf_ke` / corner-service work (this pass):**
    `cuda/dyn_core/runtime/fv3_gpu_production_dispatch.cu` (the real
    `corner_edge_service_cuda`, replacing `corner_service_not_implemented`) and
    `.hpp` (stale doc paragraph corrected);
    `cuda/dyn_core/operators/wp_d_contract_cuda.cpp` (the missing dir=2
    `copy_corners` request); `cuda/dyn_core/operators/wp_d_operators.{hpp,cpp}`
    (`del2_pass` split into `del2_zonal_flux` / `del2_meridional_flux` /
    `del2_update`, with `del2_pass`/`del2_pass_sg` retained as bit-identical
    wrappers); `cuda/dyn_core/operators/wp_d_contract.cpp` (host twin uses the
    phase sequence); `cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp`
    (comment/printf only: three stale "fail-stub"/"fail-adapter" narrations
    corrected; `cfg.nf_ke` stays pinned at 1 — see §10 for why the attempt to
    raise it to 3 was reverted).
  - **`omga` finalization work (this pass):**
    `cuda/dyn_core/operators/wp_d_operators.{hpp,cpp}` (`omega_pem_snapshot`,
    `omega_from_pe`, `adv_pe`, `AdvPeMetrics`, and the `adv_pe_cell_increment`
    device/host-shared helper that derives `ec1/ec2/en1/en2`),
    `cuda/dyn_core/operators/wp_d_cuda.cu` +
    `wp_d_contract{,_cuda}.{hpp,cpp}` (the device twins and the two new
    dispatch slots), `cuda/dyn_core/include/fv3_gpu_internal.hpp`
    (`OperatorArena` grown to 12 arrays: slot 10 = omega `pem`,
    slot 11 = omega `pb`), `cuda/dyn_core/runtime/fv3_gpu_production_dispatch.cu`
    (`omega_pem_snapshot` / `omega_finalize` wiring, `:202-204`),
    `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` (the `it == n_split && end_step`
    call sites and the `omega_finalize` debug-dump stage).
    No CMakeLists change was needed (`fv3_wp_d_cuda` already links
    `fv3_wp_d_operators` PUBLIC) and `internal_contract_version` is
    deliberately unchanged (the two slots are an additive tail; `run()` skips
    nulls).
  - Untracked omega validation tooling under `tools/savepoints/`:
    `verify_omega_host.cpp` (drives the real host twins from authoritative
    savepoints, no GPU) and `verify_adv_pe_geometry.cpp` (proves the harness
    B-grid corners are the Fortran `gridstruct%grid` corners, radius-free).
  - `.gitignore`, this file,
    `docs/agent_handoff/prompt_6b_cuda_fortran_validation.md`.
  - Untracked tooling under `tools/savepoints/`:
    `compare_cuda_fortran.py` (SP1/SP2/`sp2a`/`sp2b`/`sp2c`/`sp3`),
    `convert_sp1_boundary_to_cuda.py`, `verify_damping_column.py`,
    `footprint_sp2_uv.py`, `probe_e11_map.py`, `probe_e11_effect.py`.
  - **Nine-plus passes of Prompt 6B work is now uncommitted.** Worth
    consolidating into a commit when the user asks for one.
- Working tree also holds many untracked scratch files at repo root
  (`*_3a11*.sh`, `task*_6c_cuda.sh`, `*.log`, `*.out`, `*.txt`, `rprobe_6b`,
  `dt3a11v/`) and CUDA dump directories under `tools/savepoints/task*_cuda_dump/`.
  Do not commit them; do not delete without checking they are not cited as
  reproduction scripts by a milestone doc.

## 2. Authoritative Fortran Source

- `GFDL_atmos_cubed_sphere`, GEOS-ESM fork, tag `geos/v3.0.0`,
  SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45`.
- `@fvdycore` is **NOT populated** in this CUDA repo working tree. Do not
  populate it automatically per `docs/agent_instructions/discover_platform.md`.
- Authoritative standalone Fortran fixture (separate tree):
  `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf`
  - Fixture repo HEAD `8ba7bdd4f7f647cb6e6d70060c131e20938f1f0c`
  - `@FVdycoreCubed_GridComp` layer HEAD `ac43844bae40dcc9558deda85c7aeebb4d06cf41`
  - `@fvdycore` SHA matches the pinned `ca30e26...` above.
  - Executable: `install/bin/StandAlone_FV3_Dycore.x`, Intel `ifort` build.
  - Its nested `@fvdycore/model/dyn_core.F90` carries **uncommitted
    instrumentation only** (savepoints; no production numerical change): the
    Prompt 6A SP0/SP1/SP2/SP3 blocks plus the Prompt 6B Task-J `sp2a`
    (`:986-995`), `sp2b` (`:1097-1103`), `sp2c` (`:1283-1290`) blocks and the
    `crx/xfx/cry/yfx` additions to SP2 (`:884-887`).

## 3. Target Architecture

```
GEOS/FV3 Fortran -> ISO_C_BINDING / stable C ABI -> WP-E GPU context
    -> persistent device-resident CUDA dyn_core (WP-A/B/C/D)
    -> six-tile GPU-side cubed-sphere halo exchange
```
CUDA backend optional; Fortran backend preserved. See
`docs/agent_instructions/project_architecture.md` for full detail (unchanged).

## 4. Implemented CUDA dyn_core Coverage

- Full 54-stage production first-run inventory (WP-E 10, WP-A 7, WP-B 18,
  WP-C 9, WP-D 10): **IMPLEMENTED**, **COMPILED**, **GPU EXECUTED**.
- n_split acoustic loop: IMPLEMENTED, GPU EXECUTED (n_split=3 authoritative).
- Six-tile GPU-side halo exchange: IMPLEMENTED, GPU EXECUTED, liveness
  confirmed by `fv3_dispatch_smoke` Phase E causality test (R8 and R4).
- Persistent device residency: IMPLEMENTED, GPU EXECUTED.
- Per-level damping translation layer (`fv3_damping_translation.{hpp,cpp}`):
  IMPLEMENTED, COMPILED, host-unit-tested.

### NUMERICALLY VALIDATED — all three acoustic substeps, plus 12/13 exit fields

Configuration for every result below: c8 / npz=10 / n_split=3 / k_split=1 / R8 /
6 MPI ranks = 6 tiles, started from authoritative SP0 state with authoritative
grid metrics, `da_min`/`da_min_c`, `bdt=3600`, `ptop`/`phis`/`zs`/`dp_ref`,
`rdgas`, `n_sponge=-1`, authoritative `edge_*`, and `akap_exner`. Verdict boxes
are the owned/defined boxes; the four cubed-sphere **diagonal corner halo** cells
are reported separately and excluded (they have no donor tile). Worst value
across all 6 tiles is quoted. Grading: BITWISE / ROUND-OFF SCALE (≤1e-13) /
SMALL NUMERICAL DIFFERENCE (≤1e-8) / SIGNIFICANT (>1e-8).

**SP1 `it=1`** (post-C-grid predictor, `dyn_core.F90:636-643`) — **PASS**.
Validates `c_sw`, `update_dz_c`, `Riem_Solver_c`, `p_grad_c`.

| field | max_abs | max_rel | verdict |
|-------|---------|---------|---------|
| `delpc` | 1.82e-12 | 1.82e-16 | ROUND-OFF |
| `ptc` | 3.55e-15 | 1.44e-16 | ROUND-OFF |
| `gz` | 2.91e-10 | 6.34e-16 | ROUND-OFF |
| `pkc` | 2.91e-11 | 2.91e-16 | ROUND-OFF |
| `uc` | 9.28e-14 | 6.44e-15 | ROUND-OFF |
| `vc` | 9.92e-14 | 6.89e-15 | ROUND-OFF |
| `ws3` | 9.47e-17 | 4.97e-13 | SMALL |

Caveat retained: interior `gz` levels validate `Riem_Solver_c`'s output, not
`update_dz_c`'s, because `Riem_Solver_c` overwrites `gz` in full. `ws3`'s max_abs
is below its own cancellation floor but is recorded as SMALL, not claimed as
round-off. SP0 identity (zero numerical kernels executed) is BITWISE, 60/60
fields, 6 tiles.

**SP2 `it=1`** (post-`d_sw`, `:869-887`) — **PASS, 14/14 fields.** All ROUND-OFF
SCALE on all 6 tiles: `divgd`, `cx`, `cy`, `mfx`, `mfy`, `delp`, `pt`, `u`, `v`,
plus the four fields added this pass — `crx`, `cry`, `xfx`, `yfx`. `w ≡ 0`
identically on both sides here because `d_sw` does not update `w`, so its
agreement is **vacuous** and honest `d_sw` coverage is 9/10 — but `w` is no
longer uncovered overall, see `sp2a`. `cx`/`cy` are declared over the halo range
in their non-staggered direction and their verdict is a claim about the owned box
only. Adding `crx/cry/xfx/yfx` **closes `update_dz_d`'s input contract**.

**`sp2a` `it=1`** (post-`Riem_Solver3`, `:986-995`) — **PASS.** First
authoritative coverage of `Riem_Solver3`/`SIM1_solver` at any `it`, and of
`update_dz_d`'s output (`ws` is `intent(IN)` to `Riem_Solver3`).

| field | max_abs | max_rel | verdict |
|-------|---------|---------|---------|
| `zh` | 2.91e-11 | 6.27e-16 | ROUND-OFF |
| `delz` | 3.64e-11 | 1.22e-15 | ROUND-OFF |
| `pk3` | 2.13e-14 | 7.94e-16 | ROUND-OFF |
| `delp` | 3.64e-12 | 3.64e-16 | ROUND-OFF |
| `pt` | 7.11e-15 | 2.88e-16 | ROUND-OFF |
| `pkc` | 4.03e-14 | 4.52e-12 | SMALL |
| `w` | 4.74e-14 | 8.36e-12 | SMALL |
| `ws` | 3.55e-17 | 1.94e-12 | SMALL |

**`sp2b` `it=1`** (post-`nh_p_grad`, `:1097-1103`) — **PASS.** Validates
`nh_p_grad` at `it=1`. `u` 2.26e-13 / 5.81e-13 SMALL; `v` 2.03e-13 / 5.22e-13
SMALL. The `max_rel` ranking is driven by pe4, whose `u` genuinely maxes near
0.39, so the dynamic-range floor is small there.

**`sp2c` `it=1`** (end of acoustic substep, the state carried into `it=2`,
`:1283-1290`) — **PASS.** `delp` 3.64e-12, `pt` 7.11e-15, `delz` 3.64e-11 all
ROUND-OFF; `u` 2.26e-13, `v` 2.03e-13, `w` 4.74e-14 all SMALL. The CUDA
comparand is the `nh_pressure` stage dump; that is valid on source evidence, not
assumption — at `it < n_split` nothing after `nh_p_grad` writes
`u,v,w,delz,delp,pt` (E11 is `it==n_split`-gated at `:1178`, the omega block is
`last_step`-gated, `heat_source` is applied only after the loop).

**The full in-loop grid — all 15 boundaries PASS.** Every
`{SP1, SP2, sp2a, sp2b, sp2c}` × `{it=1, it=2, it=3}` combination grades at or
better than SMALL NUMERICAL DIFFERENCE (SP2 `it=1` is ROUND-OFF SCALE), worst
value across all 6 tiles, same exclusions as above. Representative `it=2` SP1
values: `delpc` 3.64e-12, `ptc` 1.07e-14, `gz` 4.66e-10, `pkc` 2.91e-11, `uc`
1.63e-13, `vc` 1.94e-13 ROUND-OFF; `ws3` 9.47e-17 / 5.17e-13 SMALL. Re-verified
with the post-E11/post-heat-accumulation build, and again with the post-`nf_ke`
build (15/15 still PASS, no regression). Per-boundary detail:
`prompt_6b_cuda_fortran_validation.md` §20.1 and §21.6.

**dyn_core exit (`sp3`, `it=3`) — 13 of 13 fields validated.** Worst across all
6 tiles. `w` and `delz` receive genuinely **non-vacuous** validation here (their
SP2 agreement was vacuous because `w ≡ 0` there). No exit field is vacuous.

| field | max_abs | max_rel | verdict |
|-------|---------|---------|---------|
| `u` | 4.086e-13 | 4.110e-13 | SMALL |
| `v` | 3.855e-13 | 4.795e-13 | SMALL |
| `w` | 9.509e-14 | 1.413e-11 | SMALL |
| `delz` | 3.638e-11 | 1.215e-15 | ROUND-OFF |
| `delp` | 1.091e-11 | 1.090e-15 | ROUND-OFF |
| `pe` (owned box) | 4.366e-11 | 4.358e-16 | ROUND-OFF |
| `peln` | 1.776e-15 | 1.543e-16 | ROUND-OFF |
| `pk` | 2.487e-14 | 9.266e-16 | ROUND-OFF |
| `pkz` | 1.421e-13 | 5.374e-15 | ROUND-OFF |
| `ua` | 3.268e-13 | 2.214e-14 | ROUND-OFF |
| `va` | 2.807e-13 | 1.765e-14 | ROUND-OFF |
| `pt` | 1.421e-14 | 5.757e-16 | ROUND-OFF |
| `omga` | 4.875e-14 | 1.506e-12 | SMALL |

`pt` reached this state in the `nf_ke` pass (§4 #12/#13 below), from a previous
6.397e-05 / 2.592e-06 SIGNIFICANT. Per-tile detail, all ROUND-OFF SCALE:
max_abs 1.421085e-14 (pe0/pe1/pe3), 1.065814e-14 (pe2/pe4/pe5); guarded max_rel
5.756846e-16 / 4.28…4.32e-16; rms ~2.5e-15; mismatch 395–445 of 640; worst
location e.g. pe0 `(7, 5, 0)`. `pkz` moved with it — `dyn_core.F90:1348`
rewrites `pkz` at every heated level inside the same NH heating block.

`pe`'s single Fortran halo ring is also **NOT WRITTEN** by CUDA (holds the
harness 1e-15 poison sentinel, ring max|diff| 1.002e+05); the comparator box was
narrowed to the owned box and the ring classified separately.

**Run health.** Zero NaN/Inf anywhere (`nan/inf=0/` on every `[FV3DBG]` line), no
NONFINITE or FAIL lines, harness exit code 0. The previously recorded "`it=3`
NONFINITE on tiles 2/3/4" symptom no longer appears, and `it=3` is now
**compared and passing** at all five in-loop boundaries.

`omga` reached this state in the `omga` finalization pass (see "Missing
authoritative behaviour now IMPLEMENTED" below), from a previous
4.740e-02 / 1.003e+00 NOT WRITTEN. Per-tile max_abs
2.53e-14…4.88e-14, guarded max_rel 7.8e-13…1.51e-12, rms 8.07e-15…1.07e-14,
mismatch 639–640 of 640. Essentially every cell differs in its last bits by
construction: `omga` is the difference of two ~1e5 Pa pressures divided by
`dt = 1200` s, whose cancellation floor is ≈1.2e-14, so this grades SMALL and is
**not** claimed as ROUND-OFF. The comparison box is `is:ie / js:je`, which is the
authoritative write footprint exactly (`dyn_core.F90:1207-1213` plus `adv_pe`'s
own `is..ie / js..je`) — Fortran does not write `omga`'s halo ring in the
finalization either, so `omga`'s "full declared bounds" figure grades SIGNIFICANT
for the same reason `ua`/`va`'s already did.

**Scope of these claims.** All three acoustic substeps end-to-end, plus **all 13
dyn_core-exit fields**. Gate C2 is CLOSED for this configuration (c8 / npz=10 /
n_split=3 / k_split=1 / R8 / six tiles / production first-run numerics). Do not
extend the claim to other grids, precisions, or configurations, and do not treat
the 54-stage GPU-executed inventory as numerically trustworthy beyond the
results above.

### Input-fidelity / layout defects found and fixed, in order of discovery

Thirteen total. The first seven are host-side and **opt-in** (validation-only
flags; the default synthetic/performance path is provably unchanged). The eighth
is opt-in. **The ninth, tenth and eleventh change the default path too** — see
their entries.

1. `bdt` = 10.0 vs authoritative **3600.0** → `--bdt`.
2. `ptop` = 1000.0 vs authoritative **100.0** → `--sp1-boundary`.
3. Surface topography `zs`/`phis` identically 0 vs authoritative real
   topography → `--sp1-boundary`.
4. `dp_ref` from the synthetic log-spaced column (`bot_ratio` 0.613137) vs
   authoritative pure-sigma **0.500000000000** → `--sp1-boundary` (fixed `ws3`
   by ~2.8e11×).
5. `rdgas` = 287.0 vs authoritative FMS/GEOS **8314.47/28.965 =
   287.0523045054** → `kFv3Rdgas` (fixed `gz` 3.3e-04 → 6.3e-16). Traced through
   three levels of build evidence, not guessed: `nh_utils.F90:53` →
   `FMS_DIR` = `ESMA-Baselibs-8.32.0` → `-DCONSTANTS=GEOS` →
   `geos_constants.fh:44`.
6. `n_sponge` = 9 (hardcoded in `geos_production_flags()`) vs authoritative
   **−1** → `--damp-n-sponge` (corrupted 9 of 10 damping levels; fixed SP2 `it=1`
   `u`/`v` 1.86e-02 → 9.42e-06). Verified by
   `tools/savepoints/verify_damping_column.py`.
7. `edge_w/e/s/n` fabricated as `1.0 + 0.01*index` (1.00…1.13, outside the [0,1]
   range an interpolation weight can occupy at all) vs authoritative great-circle
   weights `0.298255…0.701745` → `namespace grid_geometry_6b`, a verbatim
   `fv_grid_utils.F90:1188-1293` `edge_factors` port, gated on `--grid-metrics`.
   Fixed SP2 `it=1` `u`/`v` 9.42e-06 → 4.23e-16.
8. **Two-kappa split.** The scalar `akap` driving the **Exner** variable only
   (`ptk`, `pk3`, `pk3_halo`) is the authoritative single-precision `MAPL_KAPPA`
   = `0.28571426868438721`, while the `cappa(i,j,k)` array driving every
   thermodynamic closure is FMS double dry `2/7`. Implemented as
   `NumericalConfig::akap_exner` + `--akap-exner`; `cfg.akap` stays `2/7`. Two
   earlier levers were **retracted**: a `--moist-cappa` route (rejected at
   runtime) and a *global* `--akap` override (fixed `pk3` but degraded
   `zh`/`delz`/`w`/`pkc` and would have regressed the validated SP1 `it=1`).
9. **`grid_lonlat` / `agrid_lonlat` store order — the SP1 `it=2` root cause.**
   `make_host_state` filled both arrays **interleaved** (`idx*2 + component`)
   while they are registered **component-slowest**
   (`st_ll = {1, ni_h, ni_h*nj_h}`, the correct translation of Fortran's
   `grid(isd:ied+1, jsd:jed+1, 2)`). The consumer
   (`a2b_great_circle` → `View3` component index `lo_k`/`lo_k+1`) therefore read
   lon/lat of the wrong cells, mixed together. **The only consumer of these
   arrays is `a2b_ord4`'s corner extrapolation** — the interior, edge and
   `qx`/`qy`/`qxx`/`qyy` paths use `dxa`/`dya` — which is exactly why the damage
   was confined to the four physical panel corners of `nh_p_grad`'s B-grid
   pressure fields while every interior cell agreed at round-off, and why seven
   prior audits (which checked *values*, never *store order*) missed it. Fixed by
   filling component-slowest. Corner values: `pkc` 3.73e+00 → ≤2.71e-14, `gz`
   6.74e+05 → ≤1.16e-10. **Unlike 1-8, this is NOT opt-in** — `grid_lonlat` is
   built unconditionally, so the default synthetic/performance path changes too
   (from wrong to right; the §6 timings predate it).
10. **`cv_air` was `cp_air`.** `wp_d_contract.cpp` and `wp_d_contract_cuda.cpp`
    computed `cv_air = cfg.rdgas / cfg.akap`, which *is* `cp_air` (≈1004.7),
    where `dyn_core.F90:323` requires `cv_air = cp_air - rdgas` (≈717.6). Every
    post-loop heating `dtmp` (`:1349`) was a factor `1/(1-akap) = 1.4` **too
    small**. Fixed in both twins. The hydrostatic branch's
    `cp_air = cv_air + rdgas` was written against the correct definition and
    becomes right again automatically. **Affects the default path** (post-loop
    heating runs there).
11. **`cfg.n_con` hardcoded to 1; authoritative value is `npz` = 10.** `n_con` is
    **derived**, `dyn_core.F90:402-412`:
    `if (convert_ke .or. vtdm4 > 1.E-4) n_con = npz; else if (d2_bg_k1 < 1.E-3)
    n_con = 0; else if (d2_bg_k2 < 1.E-3) n_con = 1; else n_con = 2`. The
    in-source comment justifying the `1` actually justified **`nf_ke == 1`** (the
    `corner_edge_service` stub) and then set `n_con` — two independent controls
    conflated. `n_con` bounds how many *levels* of the `pt` update run
    (`:1338 do k=1,n_con`); `nf_ke` bounds how many *`del2_cubed` passes*
    `heat_source` receives first (`:1310-1311`), and only `nf_ke` touches
    `corner_edge_service`. Authoritative `vtdm4 = 0.01` confirmed twice
    independently (fixture `input.nml:23` / `fv3.j:386`, and
    `geos_production_flags()`'s `f.vtdm4 = 0.01` at
    `fv3_damping_translation.cpp:277` ← `FV_StateMod.F90:744`) ⇒ `n_con = npz`.
    This fix alone closed the exit `pkz` divergence: per-level max|diff|
    `5.33e-15, 6.68, 7.99, 8.43, 8.31, 7.74, 6.76, 5.36, 3.54, 1.26` (CUDA had
    written only `k=1`) → **1.528e-13 / 5.777e-15 ROUND-OFF**. **Affects the
    default path** — `geos_production_flags()` also sets `vtdm4 = 0.01`, so
    `n_con` derives to `npz` there too.
12. **`corner_edge_service` was a fail-stub, and `del2_cubed`'s dir=2
    `copy_corners` had no call site at all.** Not a value defect but a *missing
    authoritative operation*: `del2_cubed` rotates the corners twice per pass,
    dir=1 at `dyn_core.F90:2419` and dir=2 at `:2437`, and CUDA issued only the
    first — into a `corner_service_not_implemented` stub that returned a hard
    `FV3_GPU_UNSUPPORTED_CONFIGURATION`. Fixed by a thin
    `corner_edge_service_cuda` adapter onto WP-B's existing
    `launch_copy_corners_cuda` (a direct port of `tp_core.F90:257-318`, already
    exercised at both directions by validated transport paths) plus the missing
    `Axis::y` request in both twins. Key source fact: `copy_corners`'s `ng` is the
    compile-time PARAMETER (`tp_core.F90:52`) = 3, so the rotated block is 3×3 on
    every pass and the request's `halo_width` is the *write band*, not a block
    size. **Affects the default path.**
13. **`cfg.nf_ke` hardcoded to 1; authoritative value is `min(3, nord+1)` = 3.**
    `dyn_core.F90:1310`. This was the entire remaining exit `pt` residual —
    it **under-smoothed** `heat_source` before the `pt` update rather than
    skipping levels of it (that was #11). Evidence it was a smoothing deficit and
    not an amplitude error, gathered before the fix: non-constant per-cell ratio
    (median 1.68, std 44.6), per-level std ratio 3.7–4.9×, correlation 0.72–0.82.
    Now derived after every `cfg.nord` override so `nord=0` still yields 1; not
    hardcoded to 3. Fixing it required #12 first, and the pin's own in-source
    justification cited the stub it was perpetuating — a self-masking pair.
    Result: exit `pt` 6.397e-05 → **1.421085e-14 ROUND-OFF**, and `pkz`
    1.528e-13 → 1.421e-13. **Affects the default path.**

Localization method for #9, worth carrying forward: the corner *inputs* were
proven identical at round-off, then the three extrapolation ratios
`r = d1/(d2-d1)` were **solved for by least squares** from
`3*qout = Σ_t [(1+r_t)q1_t − r_t q2_t]` over 22 equations. Fortran returned
`[0.4212796036, 0.4212796036, 0.4212796036]` on every tile (geometrically
congruent, as a symmetric gnomonic panel corner requires); CUDA returned
`[0.093, −68.5, 1213.3]` on pe4 and `[0.050, −2.68, 1.72]` on pe0, both fitting
at round-off — proving the arithmetic form was correct and the geometry *data*
corrupt. A standalone probe against the harness's own `cube::CubeGrid` then
returned exactly `0.4212796036` on all six tiles, proving the geometry *source*
was authoritative and the corruption lay between harness and kernel.

**Numerical algorithm changed: NO** for all thirteen.

### Missing authoritative behaviour now IMPLEMENTED

- **The `omga` finalization path — entirely absent, now implemented.** Live
  branch is `use_old_omega = .true.` (`fv_arrays.F90:465`, not overridden in the
  fixture namelist), so the authoritative sequence is: **`pem` snapshot**
  (`dyn_core.F90:503-517`, `ptop` + cumulative `delp`, gated
  `it==n_split .and. end_step` — it is **inside** the acoustic loop, not
  pre-loop); then, strictly **after** the E11 block and still inside the loop,
  `:1205 if (last_step)` / `:1206 if (use_old_omega)` /
  `:1211 omga = (pe(i,k+1,j) − pem(i,k+1,j)) * rdt`; then
  `:1218 call adv_pe` (`:1535-1636`). The `.not. use_old_omega` forms at
  `:797-801`, `:838-842`, the `om2d` block at `:1219-1237` and the `id_ws`
  hydrostatic diagnostic at `:1238` are all **dead** in this configuration, and
  `NumericalConfig` carries no `use_old_omega` knob, so only the `.true.` path is
  implementable. Before this pass CUDA's last writer of `omga` was
  `Riem_Solver_c` using it as the `w3` argument — faithful as an *intermediate*
  (Fortran aliases the same array as `c_sw`'s `wc` at `:518`) but not the exit
  value, which is why the exit dump was bit-identical to the post-`Riem_Solver_c`
  dump on all six tiles.
  - Implemented as three host twins plus their device counterparts —
    `omega_pem_snapshot`, `omega_from_pe`, `adv_pe` — and two additive dispatch
    slots. `a2b_ord2` (`a2b_edge.F90:697-800`, live branch
    `grid_type<3 .and. .not. nested`) was **REUSED**, not rewritten: the existing
    `a2b_ord2_global` host/CUDA pair was verified verbatim against it.
  - **`ec1/ec2/en1/en2` are DERIVED, not invented.** They exist in Fortran
    solely for this computation (`fv_grid_utils.F90:759-772`, preceded by the
    comment `! For omega computation:`; `get_center_vect` at `:1792-1841`), are
    **unit** vectors (so `radius` cancels out of `adv_pe` except via the
    authoritative `dx`/`dy`), and are reconstructible from each cell's own four
    B-grid corners. The derivation's one premise — that the harness
    `grid_lonlat` corners *are* the Fortran `gridstruct%grid` corners at the same
    indices and orientation — was **proven radius-free** by
    `tools/savepoints/verify_adv_pe_geometry.cpp`: the ratio
    `dx_authoritative(i,j) / great_circle_dist(corner(i,j), corner(i+1,j), 1)` is
    one constant over `adv_pe`'s exact footprint on all six tiles (worst relative
    spread **6.661e-15**, implied radius exactly 6371000 m), which a store-order,
    index-base, orientation or tile-identity error would destroy. `:308-321`
    independently confirms `grid(:,:,1) = lon`, `grid(:,:,2) = lat`, i.e. the
    component-slowest layout that §4 #9 established.
  - Workspace: `OperatorArena` grown from 10 to 12 arrays (slot 10 = `pem`,
    slot 11 = `pb`), both device-resident. **No host/device transfer added**;
    persistent residency preserved.
  - Arithmetic and derived geometry were validated **independently of the GPU
    and of the runtime wiring** by `tools/savepoints/verify_omega_host.cpp`,
    which drives the real host twins entirely from authoritative Fortran
    savepoints: worst max_abs 1.158221e-14 / guarded max_rel 2.445659e-13. That
    the agreement is **non-vacuous** was measured, not assumed — the `adv_pe`
    increment is 1.619e-03…2.334e-03 against field ranges 3.238e-02…4.736e-02,
    i.e. ~5% of the field and ~1e11× the residual.
  - Recorded caveat, deliberately not chased: Fortran computes these grid
    vectors in quad precision (`f_p = selected_real_kind(20)`; no
    `NO_QUAD_PRECISION` in the fixture's `flags.make`), so the derived unit
    vectors may differ by ≤1 double ulp ⇒ ~1e-18 absolute on `omga`, three
    orders below the `pe − pem` cancellation floor.
  - **Numerical algorithm changed: NO.** `v3·grad` is invariant under global
    rotation, so global panel orientation is irrelevant; only intra-tile index
    orientation matters, and a transpose there would show as O(1) error.
- **E11 — `it==n_split` staggered seam reconciliation.** Authoritative call
  `dyn_core.F90:1178`
  (`mpp_get_boundary(u, v, domain, ebuffery=ebuffer, nbufferx=nbuffer,
  gridtype=DGRID_NE)`), write-back `:1181-1188`, guard
  `it==n_split .and. grid_type<4 .and. .not. nested`. **The `:1142`/`:1145-1152`
  line numbers previously recorded here were STALE.** It is a ring-**0**
  reconciliation of the one shared panel seam line that two tiles compute
  redundantly — not a halo exchange. Implemented as a new group
  `FV3_GPU_H_DWIND_SEAM` plus three minimum general extensions to the existing
  halo infrastructure: `HaloGroupSpec::ring_begin` (0 here, with
  `halo_width = 0`), `HaloFieldSpec::dest_edges` (north for `u`, east for `v`),
  and `halo_ring_count`/`halo_ring_ordinal` so the nine ring-1 groups are
  unaffected. The table entry uses a **cross-component source** (`u` ← `v`,
  `v` ← `u`), which is what `gridtype=DGRID_NE` means. Verified **bitwise**, not
  by tolerance — a tolerance test cannot see it, since `sp2c` `it=3` grades SMALL
  with or without it: `tools/savepoints/probe_e11_map.py` gives max|resid|
  **0.000e+00** for all 12 `(tile, field)` pairs against a genuine max|delta| of
  3.9e-14…3.9e-13. Footprint measured: at `it=3` only `u(is:ie,je+1,:)` and
  `v(ie+1,js:je,:)` change (0.0 off-line); at `it=2` **0 of 80** cells change.
  First observable at `sp2c` `it = n_split = 3`.
- **`heat_source`/`diss_est` per-substep accumulation.** Fortran accumulates
  inside the loop (`:858`) and applies after it (`del2_cubed` `:1311`, `pt`
  update `:1318-1350`); CUDA previously zeroed it every substep, so only the last
  substep survived. Implemented with a **persistent device accumulator**
  (`fv3_gpu_context.hpp: heat_accum`, new
  `runtime/fv3_gpu_heat_accumulator.{hpp,cu}`, pre-loop setup in
  `fv3_gpu_runtime.cpp`). No host/device transfer added; persistent residency
  preserved. Post-loop gate is `n_con /= 0 .and. d_con > 1.e-5` (`:1309`);
  `d_con = 1.0` here, so it is active. Consuming fields are `pt` **and** `pkz`
  (the NH branch rewrites `pkz` at every heated level, `:1348`, before
  incrementing `pt`, `:1349`).
  Lesson worth keeping: the accumulate kernel iterates **explicit inclusive index
  ranges in the view's own index space** with a hard bounds guard returning
  `FV3_GPU_INVALID_ARGUMENT`. The first draft derived ranges from allocation
  extents and silently ran one cell past the owned box on the staggered fields.
- `apply_diffusive_heating_nh` (`wp_d_operators.cpp:549-586`) was audited
  line-by-line against `:1338-1353` and **matches exactly** (the `delt` ladder,
  the `pkz` recomputation, the `dtmp` form, the
  `sign(min(delt,|dtmp|),dtmp)/pkz` increment). `n_con` and `cv_air` were the
  only levers.
- `delt_max` **PROVEN inert**, not assumed: measured max |dtmp| across all six
  tiles = **1.0305e-03** against the smallest authoritative bound
  `delt(k=1) = 0.1*|bdt*delt_max| = 0.720` even at the non-authoritative
  `delt_max = 0.002` — 698× headroom, so the limiter never engages.

### Verdicts overturned in earlier passes (do not rediscover as defects)

- `gz`/`pkc`'s "SIGNIFICANT ~0.96 / ~0.08" at SP1 `it=1` was **entirely the four
  cubed-sphere diagonal corner halo cells**, which have no donor tile and which
  both implementations compute from meaningless inputs.
- `uc`/`vc`'s "max_rel 1.7–2.0" was **cancellation noise** (max_abs 6.1e-15 /
  4.5e-14) where `rel = |F−C|/max(|F|,|C|)` fabricates ≈1. `p_grad_c` never
  needed a fix. Comparator corrected: denominator floored at the field's own
  dynamic range; corner cells split onto their own line. The corrected comparator
  was verified to **still** rank the pre-fix dumps SIGNIFICANT.
- The `nord`/ring-width and `clamp_nord_cap` hypothesis for the SP2 `u`/`v` frame
  is **DISPROVEN** at production `nord=2`; the cause was `edge_*` (#7).

## 5. Current Correctness / Stability Status

As of commit `dd7b45e` (3A.11T/U/V consolidated) and `6fc409a`, plus the
uncommitted Prompt 6B work:

- Authoritative production first-run config (`d4_bg=0.12`, `nord=2`, `npz=10`,
  globally cube-consistent IC): 8×8, 16×16, 32×32 all **PASS**, `delp` stays
  positive, decelerating growth, zero NaN/Inf, 3/3 independent 8×8 runs
  bit-identical.
- R8 CTest: **22/22 PASS** with all current Prompt 6B fixes built in (including
  E11, the heat accumulator, the `nf_ke`/corner-service work, and the `omga`
  finalization — Slurm job 58161075, whose build log confirms it recompiled
  `wp_d_operators.cpp.o` / `wp_d_cuda.cu.o` and relinked every test binary), with
  `test_dispatch_smoke.cpp`'s `cfg.nf_ke` at its pinned 1. R4 CTest: 22/22 PASS
  (last run before the 6B fixes). Default synthetic smoke: **PASS** (Slurm job
  58164201 — `RUN exit=0`, all six tiles finite on `u,v,delp,pt,w,delz`, `delp`
  positive over the owned box).
  Note on the E11 regression: adding a tenth halo resolve point made
  `fv3_dispatch_smoke` fail **by design** (`FV3_GPU_UNSUPPORTED_CONFIGURATION`
  naming `H_DWIND_SEAM`) — that harness spells out `kRuntimeResolvedGroups[]`
  precisely so a new resolve point cannot be silently ignored.
  `FV3_GPU_H_DWIND_SEAM` was added to the list as a weak-sense declaration (lets
  the integrated path execute, asserts nothing about seam values); Phase E's real
  six-tile exchange is where the seam semantics are exercised.
- compute-sanitizer memcheck PASS (0 errors), racecheck PASS (0 hazards),
  initcheck PASS except one previously-classified dead-read signature
  (`dgrid::copy_corners_kernel`, value-harmless in both Fortran and CUDA).
- The `(d4_bg=0.16, nord=2)` fv_arrays default pairing remains a **known
  foot-gun** (exceeds the del-2n stability limit) — not a production defect (GEOS
  overrides to 0.12), not a port defect (Fortran has the identical exposure).
  Runtime warns; never clips.
- The historical "two disagreeing gravitational constants" defect was repaired in
  `dd7b45e`. Only a stale comment still asserts otherwise — see §10.

## 6. Performance Baseline

Fortran R8 median `DYN_CORE` ms/call (6 MPI ranks, one per cubed-sphere face,
Intel/AVX2, Discover `mil` nodes):

| Grid  | ms/call |
|-------|---------|
| 8×8   | 12.730  |
| 16×16 | 16.704  |
| 32×32 | 30.261  |

CUDA R8 median ms/call (1 A100, `warpa003`, PRE-VALIDATION):

| Grid  | resident | transfer-inclusive |
|-------|----------|---------------------|
| 8×8   | 28.332   | 29.358              |
| 16×16 | 29.690   | 30.896              |
| 32×32 | 28.827   | 31.178              |

Speedup (Fortran ÷ CUDA, resident): 8×8 **0.449×**, 16×16 **0.563×**, 32×32
**1.050×** — launch/orchestration-latency dominated at these sizes (a 16×
cell-count increase changes resident cost <2%).

**Not a normalized hardware-efficiency comparison** (6 CPU ranks vs 1 GPU); no
CUDA implementation-quality conclusion implied.

**STATUS: STALE — RETEST REQUIRED. This is now the FIRST item of the next
phase**, since Gate C2 is closed (§7). These numbers predate the fidelity/layout
fixes. The first eight are opt-in and cannot affect them, but
**four** now touch the default synthetic path: #9 `grid_lonlat` store order
(wrong → correct geometry layout), #10 `cv_air`, #11 `n_con` (which derives
to `npz` there too, since `geos_production_flags()` sets `vtdm4 = 0.01`, so the
post-loop heating now runs at all ten levels instead of one) and #13 `nf_ke`.
E11, the heat accumulator, the new corner service, and the `omga` finalization
(`omega_pem_snapshot` + `omega_finalize` on the `it == n_split` substep) all add
work as well. **Do not draw any performance conclusion from the table above until
it is re-taken.**

## 7. Current Validation Boundary

- Gate C1 (integrated dyn_core GPU execution): **CLOSED** (milestone 3A.9).
- Prompt 6A (authoritative Fortran savepoints): **CLOSED** — see §8 history.
- Prompt 6B Task-J in-loop instrumentation: **CLOSED.** `sp2a`/`sp2b`/`sp2c`
  exist, are labelled with the real `it`, and the c8 reference was regenerated
  once: `experiments/c8_L10_savepoint_6c_run{1,2}/savepoints`, 1206 files each,
  `diff -r -q` clean ⇒ **BIT-IDENTICAL**.
- Fortran/CUDA runtime numerical validation (Prompt 6B): **SP0 plus all 15
  in-loop boundaries — `{SP1, SP2, sp2a, sp2b, sp2c}` × `{it=1, it=2, it=3}` —
  PASS.** Per-field tables in §4. Validated operators: `c_sw`, `update_dz_c`,
  `Riem_Solver_c`, `p_grad_c`, `d_sw`, `update_dz_d`,
  `Riem_Solver3`/`SIM1_solver`, `nh_p_grad`, `adv_pe`/`a2b_ord2`, and (bitwise,
  by donor-line map) the E11 seam reconciliation.
- **dyn_core exit boundary (`sp3`, `it=3`): COMPARED, 13 of 13 fields PASS.**
  E11, the `heat_source` accumulation, the authoritative `nf_ke = 3` smoothing
  path and the `omga` finalization are all IMPLEMENTED (§4). The exit `pt`
  VALIDATES at ROUND-OFF SCALE and the exit `omga` VALIDATES at SMALL NUMERICAL
  DIFFERENCE, both on all six tiles.
- **GATE C2: PASS — CLOSED.** All six criteria hold: SP0 remains authoritative
  and BITWISE; all 15 in-loop boundaries pass; all 13 exit fields pass on their
  authoritative defined boxes; `omga` is **non-vacuously** validated (the
  `adv_pe` increment is ~5% of the field, ~1e11× the residual); every remaining
  discrepancy is explained (cancellation floors, quad-precision grid vectors,
  the diagonal corner halo cells with no donor tile); **no scientific algorithm
  was changed to force agreement**; zero NaN/Inf; and six-tile GPU-side halo
  communication is live and real (no stubs).
- Scope of the Gate C2 claim: c8 / npz=10 / n_split=3 / k_split=1 / R8 / 6 MPI
  ranks = 6 tiles / production first-run numerics. **Not** yet claimed for R4,
  other grids, other configurations, or moist (`q_con ≠ 0`) runs.
- Two recorded coverage gaps that Gate C2 does **not** cover, both unchanged by
  this pass: `pe`'s single Fortran halo ring is NOT WRITTEN by CUDA, and
  `MOIST_CAPPA`/`USE_COND` are compiled in the Fortran but not in CUDA (proven
  round-off-neutral for this fixture only). See §10.
- Do not claim numerical equivalence from the performance baseline in §6, nor
  from the Prompt 6A savepoints alone (Fortran-only).

## 8. Current Task

**Prompt 6B — CUDA vs. authoritative Fortran numerical comparison: COMPLETE.
GATE C2 CLOSED.** All three acoustic substeps validate at all five in-loop
boundaries and **all 13 dyn_core-exit fields** validate. There are no remaining
gaps in this configuration.

**The next task is post-validation performance retest / optimization planning**,
beginning with re-taking the §6 CUDA timings (STALE — RETEST REQUIRED), then the
`nsys`/`ncu` profile of the launch-bound inference in milestone 3A.11V §7.2.
Correctness-first ordering has been satisfied, so performance work is now in
scope for the first time.

**Most recent pass — `omga` exit finalization.** Reused the existing 6C
savepoints (no new fixture run, no new Fortran instrumentation). Scope was the
exit `omga` only. **One fix used**, of the three permitted: the missing omega
finalization path, implemented as `omega_pem_snapshot` + `omega_from_pe` +
`adv_pe` on both host and device, with `a2b_ord2` REUSED and
`ec1/ec2/en1/en2` DERIVED-and-VERIFIED from existing authoritative geometry
(detail in §4 under "Missing authoritative behaviour now IMPLEMENTED").
**Numerical algorithm changed: NO.** R8 ctest 22/22 PASS; default synthetic smoke
PASS; all 15 in-loop boundaries and the other 12 exit fields re-verified from the
same dump with no regression (the exit `pt` is bit-unchanged). Result: exit
`omga` from max_abs ≈4.7e-02 / max_rel ≈1.0 to **4.875354e-14 / 1.505773e-12
SMALL NUMERICAL DIFFERENCE** on all six tiles, a ~1e12× reduction. The pass ended
on **HARD STOP condition (7), "`omga` passes and Gate C2 can close"**. Detail:
`prompt_6b_cuda_fortran_validation.md` §22.

Method notes worth carrying forward from this pass:
- **Validate arithmetic before, and separately from, the GPU.**
  `tools/savepoints/verify_omega_host.cpp` drove the *real* host twins from
  authoritative savepoint data on the login node, with no GPU and no queue wait,
  and reached 1.158e-14 — so when the GPU dump later reached 4.9e-14 the
  remaining question was only about the device twin and the wiring, not about the
  formulas. This was built *because* `gpu_a100` was ~6 h deep in queue; it turned
  the wait into the decisive result.
- **Prove a derived quantity's premise with a quantity you already trust.**
  `en/ec` could not be exported without new Fortran instrumentation (which would
  have fired HARD STOP (2)), so instead the premise — harness corners ≡ Fortran
  `gridstruct%grid` corners — was tested against the *authoritative `dx`/`dy`
  export*, radius-free, as a constant-ratio test over `adv_pe`'s exact footprint.
- **Measure the increment you are claiming to validate.** The `adv_pe`
  contribution was explicitly differenced out (`omga_pre`) so that agreement on
  `omga` could be shown to be evidence about `adv_pe` rather than about the
  dominant `(pe−pem)*rdt` term alone.
- **Explain a small residual instead of just reporting it.** `omga`'s floor is
  the `pe − pem` cancellation floor (~1.2e-14 at `pe ≈ 1e5` Pa, `dt = 1200` s),
  which is why it is graded SMALL and deliberately not claimed as ROUND-OFF, and
  why `mismatch = 640/640` is expected rather than alarming.

**Previous pass — `pt` / `nf_ke` exit reconciliation.** Reused the existing 6C
savepoints (no new fixture run, no new Fortran instrumentation). Scope was the
exit `pt` only; `omga` was explicitly excluded. **Three fixes used**, all inside
the assigned categories: (1) a real **`corner_edge_service_cuda`**, a thin
adapter onto WP-B's existing, already-exercised `launch_copy_corners_cuda` rather
than a new topology subsystem, replacing `corner_service_not_implemented`;
(2) the **missing dir=2 `copy_corners` rotation** at `dyn_core.F90:2437`, absent
from both the CUDA and host twins, which required splitting the host
`del2_pass` into three phases so the interleave is expressible; (3) **`nf_ke`
derived** as `min(3, nord+1)` (`:1310`) instead of pinned to 1. **Numerical
algorithm changed: NO** for all three. R8 ctest 22/22 PASS; default synthetic
smoke PASS; all 15 in-loop boundaries re-verified with no regression.

The pass ended on **HARD STOP condition (7), "exit `pt` passes"**: `pt` went from
6.397e-05 / 2.592e-06 SIGNIFICANT to 1.421085e-14 / 5.756846e-16 ROUND-OFF SCALE
on all six tiles, a ~4.5e9× reduction that confirms the previous pass's
under-smoothing diagnosis by construction. `pkz` moved to ROUND-OFF with it.
Detail: `prompt_6b_cuda_fortran_validation.md` §21.

The whole defect had been **self-masking**: with `nf_ke` pinned to 1 there is one
pass, `nt = 0`, and *both* corner rotations are skipped — so neither the
fail-stub nor the missing `:2437` call site could ever be reached, and the pin's
own in-source justification cited the stub it was perpetuating.

Localization notes worth carrying forward:
- The exit `pt` residual was **not** an amplitude error. The CUDA/Fortran
  per-cell ratio is not constant (median 1.68, std 44.6), while per level CUDA's
  increment std is 3.7–4.9× Fortran's with correlation only 0.72–0.82 — that is
  an *under-smoothing* signature, which pointed straight at the `del2_cubed`
  pass count rather than at any coefficient.
- `pkz` is computed from the **pre-increment** `pt`/`delp`/`delz`, so its
  round-off agreement at every level proves the state *entering* the post-loop
  tail agrees, isolating the residual to `heat_source` itself.
- `omga`'s provenance was **proven, not inferred**: the exit dump is
  **bit-identical** to the post-`Riem_Solver_c` dump on all six tiles
  (max|d| = 0.000e+00).

Three design choices in the instrumentation are worth carrying forward:
- `sp2a` captures `ws`, which is `intent(IN)` to `Riem_Solver3`, so one savepoint
  discriminates `update_dz_d` from `Riem_Solver3` without a fourth capture.
- `sp2b`'s `pkc`/`pk3`/`gz` are the **B-grid (corner) transforms**, because
  `nh_p_grad` calls `a2b_ord4(..., replace=.true.)` on all three. That is what
  made the decisive comparison possible: the exact arrays the pressure gradient
  differenced are directly observable on both sides.
- `sp2c`'s `u`/`v` halos are in flight (`i_pack(8)` started, not completed), so
  only the owned box is a defined comparison target there.

Settled in the previous pass and still true: **`heat_source`/`diss_est` is INERT
for every in-loop savepoint** — Fortran accumulates inside the loop
(`dyn_core.F90:858`) but does not read the result until *after* it
(`del2_cubed` at `:1311`, `pt` update `:1318-1350`). It could never have been the
SP1 `it=2` cause, on source evidence alone. The accumulation is now IMPLEMENTED
(§4), and its effect is confined to the exit `pt`/`pkz` exactly as predicted.

Also settled: the CUDA `pk3` dump differs from Fortran at `k=1` only, and that is
a **dump artifact, not a defect**. `set_top_kernel` (`wp_d_cuda.cu:455-458`)
writes `ptk` into the **arena scratch** the pressure gradient consumes, not back
into the registered `views.pk3`. Confirmed inert empirically: after the layout
fix `u`/`v` PASS while the `pk3` `k=1` dump difference is bit-unchanged.

Config-knob matching and field-layout/registration checks are otherwise clean.
Closed blockers, in order of discovery: fabricated grid metrics; missing SP0
loader; `bdt`; the missing `it==1` pre-`c_sw` scalar halo (the former "E5" item —
a **real CUDA defect**, one `resolve(FV3_GPU_H_DSCALAR)` line, root cause of the
earlier SP1 NONFINITE failure); `ptop`/`phis`/`zs`; `dp_ref`; `rdgas`;
`n_sponge`; `edge_*`; the Exner-kappa split; the `grid_lonlat` store order; E11;
the `heat_source` accumulation; `cv_air`; and `n_con`.

Full detail: `docs/agent_handoff/prompt_6b_cuda_fortran_validation.md` §19
(previous pass) and **§20** (this pass).

<details>
<summary>Prompt 6A history (closed)</summary>

Prompt 6A closed cleanly: authoritative Fortran `dyn_core` savepoints (SP0
entry, SP1 post-C-grid-predictor, SP2 post-D-grid/d_sw, SP3 exit; c8/npz=10/
n_split=3/R8/6 MPI ranks; production first-run numerics) were generated from
two independent runs of the real `GEOSfvdycore_fortran_perf` fixture driver,
classified **BIT-IDENTICAL** across all 462 captured fields, and passed all
finite/positivity sanity checks. Full detail, one documented driver-level
deviation (`test_case` 5→14; see that doc §3.1), and one instrumentation-only
bug found/fixed (damping-table write-format mismatch) are in
`docs/agent_handoff/prompt_6a_fortran_savepoints.md`. Reusable tooling:
`tools/savepoints/{manifest_builder.py,savepoint_reader.py}`; instrumentation
diff saved at `tools/savepoints/fortran_savepoint_instrumentation.patch`
(uncommitted in the fixture's nested `@fvdycore` working tree — not this
repo). Raw savepoint data lives only on the shared filesystem (not
committed): `GEOSfvdycore_fortran_perf/experiments/c8_L10_savepoint_6{a,c}_run{1,2}/savepoints/`.

</details>

## 9. Immediate Next Steps

1. ~~SP1 `it=1` divergences~~, ~~SP2 `it=1` `u`/`v` frame~~, ~~SP3 `it=1`
   observability~~, ~~SP1 `it=2` divergence~~, ~~the `it=2` and `it=3` chains~~,
   ~~E11~~, ~~the `heat_source` accumulation~~, ~~the exit `pkz`/`pe`~~,
   ~~the exit `pt` / `nf_ke`~~, ~~the exit `omga`~~, ~~Gate C2~~ — all **DONE**.
   See §4 and §7.
2. **RESUME HERE: post-validation performance retest.** Re-take the §6 CUDA
   timings — they are STALE, and four input-fidelity fixes (#9 `grid_lonlat`,
   #10 `cv_air`, #11 `n_con`, #13 `nf_ke`) plus E11, the heat accumulator, the
   new corner service and the `omga` finalization all touch the default synthetic
   path. Retake Fortran alongside if the fixture is still reproducible, so the
   comparison stays like-for-like; keep the "6 CPU ranks vs 1 GPU is not a
   normalized hardware comparison" caveat.
3. Then the deferred profiling work: an `nsys`/`ncu` profile of the launch-bound
   inference in milestone 3A.11V §7.2 (a 16× cell-count increase changed resident
   cost <2%, which is the evidence that these sizes are orchestration-latency
   dominated). Only after that consider kernel fusion, occupancy, multi-stream,
   CUDA Graphs, CUDA-aware MPI, or multi-GPU scaling.
4. Before extending the Gate C2 claim beyond this configuration, close the
   recorded gaps in §10 in this order: R4 validation (currently only compiled and
   ctest-clean, never savepoint-compared); `MOIST_CAPPA`/`USE_COND` for any moist
   configuration; the four `a_imp <= 0.999` inert-but-non-authoritative knobs
   (`scale_m`, `m_split`, `delt_max`); `pe`'s unwritten halo ring; the
   un-localized `fv3_dispatch_smoke` Phase C failure at `nf_ke > 1`.
5. Consider consolidating the ten-plus passes of uncommitted Prompt 6B work
   into a commit (**only when the user asks** — do not push or merge). This is
   now a natural commit point: Gate C2 is closed and the tree is ctest-clean.

Carry forward the ordering that found 13 of 13 input-fidelity root causes:
**audit input fidelity — values, store order, registration, and *derived* config
quantities — before touching arithmetic.** #11 (`n_con`) and #13 (`nf_ke`) are
the newest reminders that a hardcoded config value with a plausible-looking
in-source justification is still an input-fidelity defect: in both cases the
comment justified a *different* knob, and in #13 the justification had become
self-fulfilling — `nf_ke` was pinned to 1 to avoid a stub, which then guaranteed
the stub was never reached and so never got implemented.

Two reading rules for `tools/savepoints/compare_cuda_fortran.py`, both of which
briefly produced false FAILs this pass:
- The accepted verdict is the **defined box**. The parenthetical "full declared
  bounds, incl. both sides' undefined outer halo rings" figure is excluded by the
  established methodology and grades SIGNIFICANT at SP1/`sp2a` purely from the
  four diagonal corner halo cells, which have no donor tile.
- A large `max_abs` is not by itself a failure. `mfx`/`mfy` at SP2 carry
  `max_abs` ≈ 1.5 against their own dynamic range of ~4.2e+14 — `max_rel`
  ≈ 3.6e-15, ROUND-OFF.

## 10. Known Open Issues / Deferred Items

- **`nf_ke` / corner service: IMPLEMENTED** (was Gate-C2-blocking). The exit
  `pt` now validates at ROUND-OFF SCALE; see §4 #12/#13 and
  `prompt_6b_cuda_fortran_validation.md` §21. Three things came out of it that
  are worth not re-deriving:
  - **`copy_corners`'s `ng` is the compile-time PARAMETER** imported at
    `tp_core.F90:52` (`use fv_mp_mod, only: ng`) = 3, **not** `bd%ng` and **not**
    the pass's `nt`. The rotated block is always 3×3 on every pass, so a corner
    request's `halo_width` is the requester's *write band*, never a block size.
  - **`del2_cubed` rotates the corners twice per pass**, once per flux
    direction — dir=1 at `dyn_core.F90:2419` before the `fx` loop and dir=2 at
    `:2437` between the `fx` and `fy` loops. Only the first had a call site in
    CUDA.
  - The new `corner_edge_service_cuda` supports exactly
    `operation=copy, orientation=scalar, stagger=A, axis=x|y` and refuses
    anything else with a specific message. Everything outside that surface
    (`fill`, `fill_pair`, `vector_transform`, `a_to_b_edge`, B/C/D staggers) is
    still unimplemented — extend it there when a future path needs one, rather
    than loosening the guard.
- **Deferred, NOT localized: `fv3_dispatch_smoke` Phase C fails if the test's
  `cfg.nf_ke` is raised from its pinned 1 to the authoritative 3.** The pin keeps
  `nt = 0` on the single del2 pass, so the smoke test never requests the corner
  service; raising it as optional extra coverage produced `Phase C result: FAIL`
  / `Dispatch smoke test: FAIL` (21/22) and was **reverted** — three `gpu_a100`
  attempts to capture the diagnostic all stayed `PD (Priority)`, so the cause is
  unknown, and the 22/22 recorded in §5 is the state with the pin at 1. Already
  ruled out by inspection: `heat_source` is registered on the full
  `is-3..ie+3 / js-3..je+3` box with `lower[2] == 0`, satisfying every guard the
  service applies, and Phase C does wire the production service. **Not evidence
  against the service** — it is exercised on real A100 hardware by the
  authoritative-SP0 bench on all six tiles, where the exit `pt`/`pkz` validate at
  round-off. Localize before relying on `nf_ke > 1` in that test.
- **`omga` finalization: IMPLEMENTED and VALIDATED** (was the last
  Gate-C2 blocker). Detail in §4 and `prompt_6b_cuda_fortran_validation.md` §22.
  Two documentation corrections came out of it:
  - `pem` is allocated and filled **inside** the acoustic loop at
    `dyn_core.F90:503-517` under `if (it==n_split .and. end_step)`. The
    "pre-loop at `:505-513`" previously recorded in §9 was **stale**.
  - The omega finalization runs **after** the E11 seam block (`:1176-1188`),
    still inside the loop — not after the loop.
  - Scope of the new `adv_pe`/`omega_*` operators: they support the
    `use_old_omega = .true.`, non-hydrostatic, `grid_type < 3`, non-nested path
    only. `NumericalConfig` has no `use_old_omega` knob, so the `.false.` forms
    (`:797-801`, `:838-842`, `:1219-1237`) and the `:1238` hydrostatic `id_ws`
    diagnostic remain unimplemented. Extend deliberately if a future
    configuration needs them; do not loosen a guard to reach them.
- **Gate C2 is CLOSED, but the claim is configuration-scoped.** R4 has never been
  savepoint-compared against Fortran — it is COMPILED and 22/22 ctest-clean only.
  Report R4 separately and do not infer it from the R8 result.
- **E11: IMPLEMENTED** (was Gate-C2-blocking; see §4 for the implementation and
  the bitwise verification). Two documentation corrections that came out of it,
  recorded so they are not re-broken:
  - The authoritative line numbers are `dyn_core.F90:1178` (call) and
    `:1181-1188` (write-back). The `:1142`/`:1145-1152` previously recorded here
    were **stale**.
  - `fv3_gpu_halo_exchange.cu:48-51` and `fv3_gpu_halo_topology.hpp:293-296`
    stated `mpp_get_boundary` returns into `ebuffer`/`nbuffer` and "never into
    the field array itself" — **wrong**; `:1181-1188` copies both buffers
    straight back into `u`/`v`. Corrected in `fv3_gpu_halo_topology.hpp`.
  - The inherited claim that ring-0 is "structurally unreachable" in the halo API
    is **retired**: `HaloGroupSpec::ring_begin = 0` reaches it, and E11 now uses
    that path in the integrated runtime.
- **`heat_source`/`diss_est` per-substep accumulation: IMPLEMENTED** (was
  Gate-C2-blocking; see §4). Its classification as INERT for all in-loop
  savepoints was correct and is confirmed empirically — all 15 in-loop boundaries
  are unchanged by it. Only the exit `pt`/`pkz` are affected.
- **Proven non-authoritative but proven INERT under `a_imp = 1.0`** (recorded so
  they are not rediscovered as "new" defects). `nh_core.F90:160-176` dispatches
  `Riem_Solver3` on `a_imp`; `a_imp > 0.999` selects `SIM1_solver`, which takes
  neither `ms` nor `scale_m` — confirmed on the CUDA side too, where
  `sim1_column` takes neither:
  - `cfg.scale_m = 1.0` vs authoritative `scale_z = 0.0` (`fv_arrays.F90:337`);
  - `cfg.m_split = 2` vs authoritative `m_split = 0` (`fv_arrays.F90:632`);
  - `cfg.delt_max = 0.002` vs authoritative `1.0` (`fv_arrays.F90:759`) — the
    post-loop heat-source limiter. Now **PROVEN inert at the dyn_core exit**, not
    merely assumed out of window: measured max |dtmp| = 1.0305e-03 against the
    smallest bound `delt(k=1) = 0.1*|bdt*delt_max| = 0.720` even at 0.002 —
    698× headroom, so the limiter never engages and cannot contribute to the
    exit `pt`.
  Deliberately not changed while inert (each would perturb the default synthetic
  path the 3A.11T/U/V baseline was measured on). **Must be corrected before any
  `a_imp <= 0.999` configuration and before GEOS integration.**
- **Recorded, deliberately not fixed: `test_dispatch_smoke.cpp:535-538` and
  `:756-757` carry the identical interleaved-vs-registered lon/lat layout
  mismatch** that was root-caused in the harness (§4 #9). Self-contained tests
  with their own tolerances, not compared against Fortran, so leaving them was a
  scope decision, not a correctness judgement. Fix them the next time that file
  is touched.
- **Deferred (quantified, deliberately not changed): `MOIST_CAPPA`/`USE_COND`
  are compiled in the authoritative Fortran but not in CUDA.** The fixture's
  `@fvdycore/CMakeLists.txt:88-89` sets both, confirmed in the generated
  `flags.make`. Proven round-off-neutral **for this fixture**, not assumed:
  `q_con ≡ 0.0` exactly on all 6 tiles (0/1960 nonzero) ⇒ `peg ≡ pem`. Refined
  this cycle: the `MOIST_CAPPA` build is also what **separates the Exner scalar
  from the closure kappa**, which is the authoritative structure the
  `akap_exner` split now reproduces (§4 #8) — so the residual gap is narrower
  than previously recorded, but the `(pem+dm)−pem` vs `dm` non-associativity
  (~1e-16 relative) remains. Must be closed before any moist (`q_con ≠ 0`)
  configuration or GEOS integration.
- **Stale in-source documentation at `run_small_perf_bench.cpp:808-835`**
  (documentation defect, no numerical defect). That comment asserts CUDA runs
  two disagreeing gravities, that `fv3_vertical_kernels.cu` "never references
  grav at ALL", and that the authoritative `grav` is "undetermined". All three
  are false: the plumbing was repaired in milestone 3A.11V
  (`riem_solver3_column` takes a threaded `T grav`), the launcher passes
  `fv_real(cfg.grav)`, and `geos_constants.fh:38` establishes `GRAV = 9.80665`
  as authoritative. Correct the comment on the next pass that touches that file.
- **Retracted claim, recorded so it is not rediscovered:** an intermediate
  revision of the `edge_*` fix also changed `make_host_state`'s
  `cube::CubeGrid` tile argument to `tile + 1`, on the theory that a 0-based
  MPI rank was being passed to a 1-based parameter. **Wrong** — the harness
  `tile` is already 1-based (`run_small_perf_bench.cpp:2279` derives its on-disk
  directory as `"tile" + (tile-1)`). `CubeGrid` tile 7 falls through
  `tile_rotate`'s `default` while `corner_owner` reads it as odd parity, and the
  inconsistent halo mapping produced NaN in `u`/`v`/`delp` at `it=1`. Reverted.
  Lesson: read an index-base convention off an actual caller, never infer it —
  and note the `edge_*` block's dx/dy authenticity self-check **cannot** catch
  this class of error, because distances are rotation-invariant and the six
  panels are pure rotations of one another, so it proves panel shape only, not
  tile identity.
- ~~**`it=3` not compared.**~~ **RESOLVED** — `it=3` is now compared and passing
  at all five in-loop boundaries (§4). The formerly recorded "`it=3` NONFINITE on
  tiles 2/3/4" symptom does not reproduce. E11 is live exactly at `it=3` for
  n_split=3, and its footprint there was measured directly (only
  `u(is:ie,je+1,:)` / `v(ie+1,js:je,:)` change; 0.0 off-line).
- **`pe`'s one Fortran halo ring is NOT WRITTEN by CUDA** at the dyn_core exit
  (`pe` is declared `pe(is-1:ie+1, 1:npz+1, js-1:je+1)`). The ring still holds
  the harness 1e-15 poison sentinel, so including it reported max|diff| ≈ 1.0e+05
  — a coverage gap, not a numerical disagreement. `SP3_DEFINED_BOX["pe"]` in
  `tools/savepoints/compare_cuda_fortran.py` is narrowed to the owned box with
  the ring classified separately. Decide whether the ring must be written before
  GEOS integration (it depends on what consumes `pe` downstream of dyn_core).
- **`clamp_nord_cap` (D1)** caps the WP-B group loop at `ng-1==2`, so
  `nord_k==3` silently gets zero divergence damping while still returning
  success. **Not a defect at production `nord=2`** — `clamp_nord_cap(2,3) = 2`,
  the launcher loop is inclusive, and every `del6_vt_flux` extent, flux-loop
  bound and `copy_corners` placement matches Fortran for `nord=2`/`ng=3`. What
  remains is the dead `nord_k == 3` branch (would silently under-damp if a
  future configuration requested it) and the docs' incorrect `nord=1-3` support
  claim.
- **`del6_vt_flux` n==1 uninitialized `d2` ring**: a dead-but-real uninitialized
  read, reachable only at `ndif==1` (not exercised by production `nord=2`, which
  yields `ndif ∈ {0,2}`). Value-harmless in both Fortran and CUDA. Deferred: add
  an `ndif==1` case to `test_wp_b_cuda_smoke.cpp` to reproduce under initcheck,
  then decide on widening `del6_init_kernel`.
- **k=0 damping row**: root-caused to `n_sponge = 9` and **fixed for validation
  mode**. Still open for the default/performance path, which retains
  `geos_production_flags()`'s GEOS values by design; harness-only, not implicated
  in any measured instability.
- **`docs/agent_instructions/cost_efficiency.md` does not exist yet** —
  referenced by §12 below but not yet created; create separately, do not inline a
  large policy here.
- Smaller recorded-not-fixed items (see `milestone_3A11T_...md` §7 and
  `milestone_3A11U_...md` "Genuine port defects" for detail): `divg2`
  read-without-writer guard; write-write race in the smag arena outer ring
  (appears inert); halo documentation defect at
  `fv3_gpu_halo_exchange.cu:44-52` / `fv3_gpu_halo_topology.hpp:289-297`
  (3 inaccuracies, contradicted with specifics in the 3A.11U handoff and above).

## 11. Hard Constraints

- Authoritative GFDL FV3 Fortran (`geos/v3.0.0`, `ca30e26...`) is the
  numerical source of truth.
- Do not modify `geos-gtfv3`.
- Numerical validation may now be claimed **only** within the closed Gate C2
  scope (§7): c8 / npz=10 / n_split=3 / k_split=1 / R8 / six tiles / production
  first-run numerics. Any other grid, precision or configuration is still
  unvalidated and must be reported as such.
- Preserve persistent GPU residency; no full-state H2D/D2H between work
  packages.
- Use real six-tile halo semantics; do not stub communication for integrated
  validation.
- Do not populate `@fvdycore` in this repo without a demonstrated dependency,
  documented first.
- Generated build directories, Slurm logs, sanitizer logs, scratch scripts and
  CUDA dump directories are not committed.
- Do not optimize numerical kernels before correctness localization, unless a
  task explicitly requests performance work (current phase does not).
- Do not push or merge unless explicitly requested.

## 12. Cost-Efficient Agent Rules

See `docs/agent_instructions/cost_efficiency.md` (not yet created as of this
writing — create as its own file if a dedicated policy is needed; do not
inline one here). Until then, apply directly:

- Read this file (`CURRENT_STATE.md`) first; avoid re-reading superseded
  milestone handoffs unless an open item above requires the detail.
- Use targeted greps/searches over broad repository scans.
- Use bounded, coarse-to-fine debugging when localizing numerical divergences
  (savepoint-by-savepoint, not whole-run diffs). Every Prompt 6B root
  causes (all thirteen, as it turned out) were found by auditing **input
  fidelity — values, store order, registration and *derived* config quantities —
  before touching kernel arithmetic**; keep that ordering. Where a needed
  authoritative input genuinely does not exist in the port, prefer **deriving it
  from geometry already proven authoritative and then verifying the derivation
  against a trusted export** over adding new Fortran instrumentation.
- Prefer the shortest Slurm job that answers the question (short `srun` probes
  over long `sbatch` runs where possible).
