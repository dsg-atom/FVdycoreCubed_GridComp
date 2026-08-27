# Milestone 3A.11V — Stable Authoritative Performance Benchmark

Branch: `cuda/gpu-bringup`
Base commit at start of work: `774ab59`
Checkpoint commit for this work: `dd7b45e`

This document covers the three closure tasks of 3A.11V:

1. a globally cube-consistent benchmark initial condition;
2. an authoritative per-level damping translation layer;
3. the `d4_bg_top` / `nord = 2` stability audit and its classification.

It is the companion to `milestone_3A11V_dt_sensitivity.md`, which records the
*diagnosis* this milestone acts on. Nothing here revisits that diagnosis.

Validation-level vocabulary is used strictly. In particular **nothing in this
document is a numerical validation**: no result here was compared against
authoritative Fortran savepoint output. See §6.

---

## 1. Task 1 — globally cube-consistent initial condition

### 1.1 What was wrong

Every horizontal structure in the benchmark IC was a function of the **tile-local
array index**, seeded identically on all six tiles:

| field | old seed |
|---|---|
| `pt`, `delz`, `zh`, `gz` | `0.02*sin(0.35*i)*cos(0.35*j)` |
| `uc`, `vc`, `u`, `v`, `ua`, `va` | `0.29*i`, `0.23*j` phases |
| `ptc` | `sin(0.29*i + 0.23*j)` |
| `z_rat` | `sin(0.29*i - 0.23*j)` |
| `omga` | `sin(0.29*i + 0.23*j - 0.11*k)` |
| `heat_source` | `sin(0.4*(idx % ni_h))` — i only, no j or k structure at all |
| `grid_lonlat`, `agrid_lonlat` | `0.01*i`, `0.01*j` — the index in disguise, not lon/lat of anything |
| `area`, `rarea_c`, `f0`, `f_c`, `sin_sg`/`cos_sg` scale | `0.02*sin(0.31*i)*cos(0.31*j)` |
| additionally `make_tile_asymmetric` | `0.8*sin(0.37*tile + 0.11*i)`, `0.6*cos(0.37*tile + 0.13*j)` added to `u`/`v` after the fact |

That is not a single-valued field on a cubed sphere. The value seeded into tile
T's ring-1 halo is not the value the neighbour owns at the cell that halo
shadows, so the state carried a genuine O(1) discontinuity across all twelve
panel seams. The production halo exchange then *correctly* transported the
neighbour's differing value inward, and the resulting grid-scale cliff was real
forcing — measured in 3A.11V as a ring-1 surface `pkc` overshoot of +4182.3 Pa on
tiles 1/3/5 and +1211.4 Pa on tiles 2/4/6, with the ratio matching `r^gama - 1`
to 0.069%.

### 1.2 What replaced it

New files, benchmark/test infrastructure only, deliberately **not** linked into
`fv3_dyn_core_gpu`:

- `cuda/dyn_core/benchmarks/fv3_cube_geometry.hpp`
- `cuda/dyn_core/benchmarks/fv3_cube_geometry.cpp`
- `cuda/dyn_core/tests/benchmarks/test_cube_seam_continuity.cpp` (ctest `fv3_cube_seam`)

`fv3_cube_geometry` builds the **actual** gnomonic-equidistant cubed sphere:
the tile-1 closed form from `REF/model/fv_grid_utils.F90:3462-3528`, the six
`mirror_grid` tile maps from `REF/tools/fv_grid_tools.F90:2309-2439`, the
production 10° westward shift (`shift_fac = 18`), and the corner-owner map that
resolves a halo index to the panel that actually owns it. Every horizontal field
is then a smooth function of position on the sphere, which makes it single-valued
by construction.

Field-by-field, the IC now reads:

| field | new source |
|---|---|
| `pt`, `delz`, `zh`, `gz`, `ptc` | `flow_shape` = Williamson-2 `S(lon,lat)`, `REF/tools/test_cases.F90:1013-1015` |
| `u`, `v` | `dgrid_u` / `dgrid_v`: solid-body wind projected on the D-grid stagger exactly as `init_winds defOnGrid==5`, `REF/tools/test_cases.F90:485-521` |
| `uc`, `vc` | `cgrid_uc` / `cgrid_vc` — see the caveat in §1.4 |
| `ua`, `va` | `agrid_wind` — same caveat |
| `heat_source` | `flow_shape` at the cell centre, with a real (i,j) decode |
| `grid_lonlat`, `agrid_lonlat` | the true gnomonic corner and centre lon/lat, shift included |
| `z_rat` | `shape2 = sin(2*lon)*cos(lat)` |
| `omga` | `shape3 = cos(lon)*sin(2*lat)`, times `cos(0.11*k)` |
| `area`, `rarea_c`, `f0`, `f_c`, supergrid scale | `flow_shape` |
| `make_tile_asymmetric` | **deleted** |

Amplitudes are unchanged throughout: `flow_shape`, `shape2` and `shape3` all
range over `[-1,1]` exactly as the old `sin*cos` products did.

Three or more independent shapes are kept, rather than reusing `flow_shape`
everywhere, because the old patterns were deliberately different so that a
product or difference of two of them could not cancel identically and hide a
transcription error. That property is preserved. It is free to preserve: *any*
smooth function of the true `(lon, lat)` is single-valued on the sphere.

`make_tile_asymmetric` is deleted rather than disabled. It existed to stop the
six tiles being identical, which is now handled by construction — each tile
occupies a different part of the sphere. Keeping it would have been actively
harmful: it was a second, tile-local, index-keyed writer of `u`/`v` applied
*after* `make_host_state` returned, so it would have re-introduced precisely the
seam discontinuity Task 1 removes, on top of the globally consistent winds, and
invisibly.

### 1.3 Seam-continuity test — PASS

`fv3_cube_seam` (ctest, no GPU required) runs six checks at `im = 8` and
`im = 16`, `halo = 3`.

Three of the six are constructed to be **independent of the code under test**:
T2 transcribes FV3's 13 explicit seam copies as a literal table and never
consults `corner_owner`; T4 and T5 brute-force match halo cells and edges against
all six tiles' interiors by coordinate, also without consulting `corner_owner`.
A wrong seam table therefore cannot make them pass.

| check | result at im=8 | result at im=16 |
|---|---|---|
| T1 rotations orthogonal, `det = +1`; tile-1 extent; polar caps; equidistant spacing | pass; spacing ratio 1.4033 < √2 | pass; 1.4114 < √2 |
| T2 the 13 FV3 seam copies (`fv_grid_tools.F90:720-736`) | 13/13, worst chord error 0.000e+00 | 13/13, worst 6.568e-16 |
| T3 halo corners are not the local analytic extension | 648 corners, 0 degenerate | 1224, 0 |
| T4 scalar seam continuity | 576 halo cells, 0 unmatched, 0 ambiguous; worst `flow_shape` disagreement 4.441e-16 | 1152; 0; 0; 4.718e-16 |
| T5 vector seam continuity + derived sign/swap table | 1332 halo edges, 0 unmatched, 0 sign-inconsistent; 24/24 (tile,side) seams agree with the independent FMS audit | 2580; 0; 0; 24/24 |
| T6 discrete non-divergence | diagnostic only, ratio 2.951 | diagnostic only, 8.341 |

`=== cube seam continuity: PASS (0 failures) ===`

T5 is the strongest of these. It derives the D-grid seam sign/swap relationship
*from geometry alone* — which components pair across each seam and with which
signs — and that derivation reproduces the independent FMS audit on all 24
(tile, side) seams, including the asymmetric-sign prediction on the twelve
reversed seams (N side `v=+1`/`u=-1`, W side `v=-1`/`u=+1` on odd tiles; S side
`v=+1`/`u=-1`, E side `v=-1`/`u=+1` on even tiles).

**The test found a real bug.** In its first run, 9 of 13 seam copies failed with
chord errors up to 2.000 — an exact antipode. Root cause: `global_corner` applied
the production `lon -= pi` shift *after* `tile_rotate`. That shift is itself the
z-rotation `(x,y,z) -> (-x,-y,z)`, and it commutes only with the pure-z tile maps
(tiles 1 and 2) — which is exactly why the only four copies that passed were the
equatorial E-W seams. `mirror_grid`'s own comment says it assumes tile 1 is
centred on the Greenwich meridian with `Lon ∈ [-pi, pi]`, so the flip belongs
*before* the tile rotations, in Cartesian. Without this test, four of six tiles
would have sat in the wrong hemisphere and every downstream number would have
been quietly meaningless.

Two assertions were **demoted to diagnostics** rather than tuned:

- T3's `worst_ratio < 1.0`: not a property of correct code. The gnomonic
  continuation is `tan`-based and diverges away from the panel, so at halo depth 3
  near a cube corner the local extension is legitimately far from the neighbour's
  grid. Only `n_zero == 0` is asserted (it catches a silent fallthrough); T4 and
  T5 establish halo correctness properly.
- T6's `ratio < 20.0`: it passed (2.95 / 8.34) but grew ~2.8× per doubling, so it
  would have become a false failure at `im = 32`. Choosing a threshold that makes
  it pass would be tuning, not testing.

### 1.4 What Task 1 does NOT claim

Stated explicitly because the distinction is load-bearing:

- **Globally consistent ≠ balanced.** The harness keeps its fabricated uniform
  100 km metrics (`dx = dy = dxc = dyc = 1e5`, uniform `area`). The analytic
  state is therefore not in discrete balance with respect to those metrics, and a
  *smooth* O(1) forcing remains. This is deliberate: the 3A.11V del-2n stability
  calibration is expressed in `da_min/area`, and replacing the metrics would
  invalidate it. The forcing that had to be removed was the grid-scale seam
  *discontinuity*, which the damping operators amplify; a smooth large-scale
  imbalance is a different thing. The observed consequence is a slow,
  monotonically **decelerating** adjustment — see §4.2.
- **`cgrid_uc`/`cgrid_vc`/`agrid_wind` are analogues, not ports.** FV3's `atoc` /
  `get_center_vect` are not reproduced. This is acceptable only because those
  fields are `d_sw` inputs that stage S1 overwrites before any read. Do not
  promote them to a validation reference.
- **`edge_w`/`edge_e`/`edge_s`/`edge_n` are still fabricated local-index ramps**
  (`1.0 + 0.01*index`) and are still not single-valued across a seam. This is the
  one place where Task 1's rule is knowingly not applied, and it is documented at
  the site. They are grid *metrics*, and their authoritative values come from
  `edge_factors` / `efactor_a2c_v`, which are functions of a real grid spacing this
  harness does not have — there is no faithful value consistent with a uniform
  1e5 grid. Converting only these four while `dx`/`dy`/`area` stay uniform would
  make the metric set internally *less* coherent. Residual consequence: the
  contravariant edge corrections see a smooth 1% metric inconsistency at panel
  boundaries. That is not the discontinuity that drove the instability.
- `bench_horiz_scale()`'s meaning has changed. It no longer switches off a seam
  discontinuity, because there no longer is one; a nonzero `HORIZ_SCALE` is now
  seam-consistent at any amplitude. It is retained for the exact-rest control
  (`HORIZ_SCALE=0` with `WIND_SCALE=0`). The `[BENCH]` banner text was updated so
  a future reader cannot misread a log.
- **Precision.** FV3 builds grid geometry in quad (`f_p = selected_real_kind(20)`,
  `NO_QUAD_PRECISION` is not defined in this configuration) and orders the
  `cell_center2` corner sum through `sorted_inta` for bitwise decomposition
  independence. This module uses `double` and the plain corner order. Agreement to
  ~1e-16 relative is expected; **bitwise agreement is not claimed**.

---

## 2. Task 2 — authoritative per-level damping translation

### 2.1 The gap that was closed

`dyn_core.F90` does not consume damping *scalars*. It consumes eleven per-level
quantities that it derives from the flagstruct scalars at the top of every
`dyn_core` call. Ten of those were already declared as device-resident
`immutable_grid` INPUT fields in the storage contract
(`runtime/fv3_gpu_contract.cpp:56-66`) — but **nothing under `cuda/` ever
converted the scalars into them.** Every caller that needed those arrays invented
its own values, including the 3A.11 benchmark, whose fabricated 4-level cycle
paired `d4_bg = 0.16` with `nord_k = 2` and was the sole cause of the multi-step
instability.

New files:

- `cuda/dyn_core/include/fv3_damping_translation.hpp`
- `cuda/dyn_core/runtime/fv3_damping_translation.cpp`
- `cuda/dyn_core/tests/runtime/test_damping_translation.cpp` (ctest
  `fv3_damping_translation`, 10 tests, host-only, no GPU)

`translate_damping()` is a faithful, statement-ordered reimplementation of
`dyn_core.F90:677-765`, host-only, CUDA-free, no runtime state, a total function
of its arguments. `geos_production_flags()` and `fv_arrays_defaults()` supply the
GEOS overrides (`FV_StateMod.F90:569-793`) and the unoverridden
`fv_arrays.F90` defaults respectively, as named factories, so a caller never
hand-copies a magic number.

Authoritative source: `GFDL_atmos_cubed_sphere`, GEOS-ESM fork, `geos/v3.0.0`,
SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45`. This matters: `d4dmp()`,
`d_ext()` and `n_zfilter` exist only on the fork. Do not re-derive from upstream
GFDL.

### 2.2 Ownership chain

```
host configuration scalars                    DampingFlags
    (GEOS FV_StateMod overrides, or                |
     fv_arrays defaults)                           |
                                                   v
translation                          translate_damping(flags, pfull)
    (dyn_core.F90:677-765, faithful)               |
                                                   v
per-level host arrays                        DampingColumn
    (11 vectors + 2 scalars)                       |
                                                   v
field registration                     FieldId::damping_* x10
    (fv_real width even for the three              |
     integral orders; ABI has no int type)         |
                                                   v
device-resident per-level arrays        immutable_grid, INPUT
    uploaded ONCE at initialization, never re-uploaded
```

**No per-step H2D.** Every field produced is `immutable_grid`, i.e. uploaded once
at initialization. There is deliberately **no per-step entry point** in the
translation API: adding one would reintroduce a per-`dyn_core`-step host/device
transfer, which `project_architecture.md` classifies as an architectural
regression. The values are genuinely time-invariant in the authoritative code
too — `dyn_core.F90` recomputes them every call from unchanging scalars and
caches the only expensive part (`d4dmp`) behind a saved logical at `:150`.

### 2.3 Arrays now sourced authoritatively

All eleven. Ten are contract fields; `d_ext` is produced but not yet a contract
field, so a future consumer does not have to re-derive the `n_zfilter` gate.

| output | authoritative origin |
|---|---|
| `nord_k` | `dyn_core.F90:705`, sponge arms `:743 :753 :761` |
| `ndif` (= `nord_v(k)`) | `:713`, sponge `:748 :756` |
| `nord_w` | `:714`, sponge `:745 :754 :762` |
| `d2_bg` | `:718` (capped at 0.20), sponge `:737 :743 :753 :761` |
| `d4_bg` (= `d4dmp(k)`) | `:683-684`; never touched by the sponge |
| `damp_v` (= `damp_vt(k)`) | `:721-724`, sponge `:749 :757` |
| `damp_w` | `:726`, sponge `:745 :754 :762` |
| `damp_t` | `:727` |
| `d_con` (= `d_con_k`) | `:734`, sponge `:751 :759 :763` |
| `kgb` | `:707-711` (bottom level only) |
| `d_ext` | `:729-733` (`n_zfilter` gate) |
| `nord_t` (scalar) | `:715` — see the ordering subtlety below |
| `dddmp` (scalar) | `:794`, passed through unmodified |

Before this milestone, **8 of the 10 contract arrays had no path at all** from
the Fortran configuration into CUDA runtime state. That gap is now closed for all
ten.

Two deliberate non-plumbings, recorded so they are not read as omissions:

- `d2_divg_max_k1`, `d2_divg_max_k2`, `damp_k_k1`, `damp_k_k2` are **dead** on
  this code path — referenced only under the `old_divg_damp` branch of
  `fv_control.F90:901,903`, which this `dyn_core.F90` never takes. Plumbing them
  would imply they have an effect. They do not.
- **`nord_t` is a scalar on purpose.** `dyn_core.F90:715` runs *before* the sponge
  block at `:736-765`, and no sponge arm reassigns `nord_t`. So although it is
  written inside the k loop, its value is `min(2, nord)` at *every* level,
  including the sponge levels where `nord_v(k)` is subsequently reset to 0.
  Representing it as a vector would invite a future reader to "fix" it to track
  `ndif`, which would change the numerics.

### 2.4 Fabricated tables removed — YES

The benchmark's hand-coded 4-level damping cycle is gone. It now calls
`geos_production_flags()` → `translate_damping()` and prints the resulting table
unconditionally (`[FV3-DAMP]`), so a log proves which values were actually used —
which the fabricated table made impossible.

The authoritative `npz = 10` GEOS production column, as reproduced value-for-value
by the translation and printed by the acceptance runs:

```
[FV3-DAMP]   k nord_k ndif nord_w      d2_bg      d4_bg     damp_v     damp_w     damp_t      d_con        kgb      d_ext
[FV3-DAMP]   0      0    0      0   0.200000   0.120000   0.100000   0.200000   0.010000   0.000000  0.000e+00   0.000000
[FV3-DAMP]   1      0    0      0   0.100000   0.120000   0.050000   0.100000   0.010000   0.000000  0.000e+00   0.000000
   ... k = 2..7 identical to k = 1 ...
[FV3-DAMP]   8      0    2      0   0.020000   0.120000   0.010000   0.020000   0.010000   0.000000  0.000e+00   0.000000
[FV3-DAMP]   9      2    2      2   0.000000   0.120000   0.010000   0.010000   0.010000   1.000000  0.000e+00   0.000000
```

with `nord_t = 2` at every level and `dddmp = 0.2`.

Note what this shows about the original benchmark table: production pairs
`nord_k = 2` with `d4_bg = 0.12` at exactly **one** level (k = 9, the bottom), and
runs `nord_k = 0` everywhere else. The fabricated table paired `nord_k = 2` with
`d4_bg = 0.16` on a 4-level cycle — a configuration production never produces.

`FV3_BENCH_DAMPV_SCALE` and `FV3_BENCH_D4_SCALE` remain as diagnostic knobs, both
defaulting to 1.0, and the banner prints them so a leg that silently inherited an
override is visible in the log rather than mistaken for an authoritative run.

---

## 3. Task 3 — `d4_bg_top` / `nord = 2` stability audit

### 3.1 The norm

For the divergence-damping del-2n arm, the per-acoustic-substep growth factor of
the least-stable grid mode is

```
gamma_d4 = (8 * d4_bg)^(nord_k + 1) * (da_min / area_local),   unstable for gamma > 2
```

This form is **measurement-calibrated**, not merely derived. It was obtained term
by term from the Fortran statements and then confirmed by a parameter-free
threshold measurement at two different `nord` on a seam-free IC: `nord = 1`
predicted a critical scale factor in `[1.105, 1.127]` and measured `(1.10, 1.15]`;
`nord = 2` predicted `[0.984, 0.998]` and measured `(0.98, 1.00]`. Critical
`d4_bg` is `2^(1/(nord+1))/8`:

| nord | critical `d4_bg` |
|---|---|
| 0 | 0.25000 |
| 1 | 0.17678 |
| 2 | 0.15749 |
| 3 | 0.14865 |

The `nord_k > 0` branch carries a **second, additive** term, because
`sw_core.F90:1418` applies `vort = damp2*delpc + dd8*divg_d` — the del-2 part does
not vanish when the del-2n part is active:

```
gamma_d2 = 8 * max(d2_bg, min(0.20, dddmp*|div*dt|)) * (da_min_c / area_c)
```

The Smagorinsky part is flow-dependent and therefore not knowable from
configuration alone, so **both** bounds are reported: a static bound
(Smagorinsky = 0) and a worst-case bound (Smagorinsky saturated at its own 0.20
cap). The static bound alone understates the risk; the worst-case bound alone
overstates it.

`gamma_v`, `gamma_w` and `gamma_t` use the structurally analogous form because
`sw_core.F90:1477,:973` build `damp4 = (damp_v*da_min_c)^(nord_v+1)` and
`tp_core.F90:217,237` build `damp = (damp_c*da_min)^(nord+1)` — the same shape.
They are **analogous by construction and NOT independently threshold-measured**.
They are consistent with measurement (production `damp_v = 0.10` at `nord_v = 2`
gives `gamma_v = 0.51`, and that pairing ran stably for 300 calls), but must not
be treated as calibrated to the same standard as `gamma_d4`.

### 3.2 Measured table, production first-run configuration, npz = 10

Printed unconditionally at initialization by the acceptance runs:

```
[FV3-DAMP]   k d4? gamma_d4 gd2_stat gd2_wrst gdiv_stat gdiv_wrst  gamma_v  gamma_w  gamma_t  flag
[FV3-DAMP]   0  no   0.0000   1.6000   1.6000    1.6000    1.6000   0.8000   1.6000   0.0005  ok
[FV3-DAMP]   1  no   0.0000   0.8000   1.6000    0.8000    1.6000   0.4000   0.8000   0.0005  ok
   ... k = 2..7 identical to k = 1 ...
[FV3-DAMP]   8  no   0.0000   0.1600   1.6000    0.1600    1.6000   0.0005   0.1600   0.0005  ok
[FV3-DAMP]   9 yes   0.8847   0.0000   1.6000    0.8847    2.4847   0.0005   0.0005   0.0005  unstable-if-divergent-flow
[FV3-DAMP] VERDICT: stable from configuration alone (worst gamma_div_static =
           1.6000 at k=0), but the additive Smagorinsky term at sw_core.F90:1418
           can push at least one level past 2.0 in a strongly divergent flow
           (needs |div*dt| >= 1 with dddmp=0.2).
```

The single level that takes the del-2n arm is k = 9, at `gamma_d4 = 0.8847` —
comfortably below 2. Worst static margin is `gamma_div_static = 1.6000` at k = 0,
i.e. 80% of the limit with no headroom to spare but no violation.

### 3.3 Classification

**Production first-run configuration (`d4_bg_top = d4_bg_bot = 0.12`, `nord = 2`,
`npz = 10`, GEOS overrides): `SAFE FOR CURRENT DEFAULTS BUT NOT GUARANTEED`.**

Reasoning:

- No level is unstable from configuration alone. `gamma_div_static` peaks at
  1.6000 (k = 0), and the only del-2n level is at 0.8847.
- But k = 9 reaches `gamma_div_worst = 2.4847` once the Smagorinsky term
  saturates, which requires `|div*dt| >= 1` at `dddmp = 0.2`. That is a flow
  condition, not a configuration condition, so it cannot be ruled out from the
  configuration and it is not ruled out here.
- The margin is thin and it is thin *by construction*: the `fv_arrays.F90`
  default `d4_bg_top = d4_bg_bot = 0.16` sits at `gamma_d4 = (8*0.16)^3 = 2.097`
  at `nord = 2`, i.e. **already past the limit**. The GEOS override to 0.12 is
  what makes production safe. Nothing in the configuration path enforces that
  relationship.

**`(d4_bg = 0.16, nord = 2)`: `POTENTIAL USER-CONFIG FOOT-GUN`.** This is the
pairing the fabricated benchmark table used and it is exactly `gamma_d4 = 2.097`.
It is reachable by a user who sets `nord = 2` (the GEOS cube value) and leaves
`d4_bg_top`/`d4_bg_bot` at the `fv_arrays` default. It is not a production defect
because production overrides both, and it is not a port defect because the
authoritative Fortran has the identical exposure.

**Not** classified as `ACTUAL PRODUCTION DEFECT`: the production configuration
does not violate the bound, and the authoritative Fortran behaves the same way
under the same inputs.

Also worth recording under this heading: `fv_arrays.F90:379,388` defaults
`d2_bg_k1 = 4.0` and `d2_bg_k2 = 2.0`. At `8 * 4.0 = 32` these are far past the
del-2 critical value of 0.25. The GEOS overrides to 0.2 / 0.1 are again what makes
production safe. Same classification, same reason.

### 3.4 Runtime behaviour — warn, never mutate

`damping_stability()` computes the report; `format_stability_report()` renders it;
the benchmark prints it unconditionally at initialization. It is **not** gated
behind a debug flag, because an unstable damping column should be visible in
every log, not only in a log someone thought to enable. Lines are prefixed
`[FV3-DAMP]` so they can be grepped out of a mixed multi-tile stream. Cost is one
string build per initialization.

Deliberately **no clipping and no rejection**:

- The authoritative Fortran does not clip these coefficients. Silently clipping
  would make the CUDA backend disagree with the numerical source of truth in
  exactly the configurations where the disagreement matters most, and it would do
  so invisibly.
- A hard reject was considered and rejected: it would refuse configurations that
  the authoritative Fortran accepts and runs, which is a behaviour change of the
  supported configuration space, not a safety measure.
- `translate_damping()` clamps *only* where the authoritative code clamps (the
  `min`/`max` at `:718`, `:721`, `:743`) and nowhere else.

The per-level flag values are `ok`, `unstable-if-divergent-flow` (worst-case only)
and `UNSTABLE` (static). A static violation produces a loud unconditional warning
naming the offending level, `nord`, coefficient and gamma. It does not alter
numerics.

---

## 4. Acceptance results

### 4.1 Configuration

All acceptance legs: globally consistent cube IC, authoritative per-level damping
translation, production first-run configuration, **no `FV3_BENCH_*` and no
`FV3_GPU_DEBUG_FAIL`**. The `[BENCH]` banner echoes the effective
`WIND_SCALE=1`, `HORIZ_SCALE=1`, `DAMPV_SCALE=1`, `D4_SCALE=1` in every leg, so an
accidentally inherited override would be visible rather than silent.

Note on call counts: `--warmup 20 --iters 100` steps the model 120 times, not 100.
The warmup calls are real `dyn_core` calls on the same persistent state, so every
"100 call" requirement is met with margin.

### 4.2 Results

See §5 for the tabulated matrix. Diagnostic leg (8×8, 52 calls,
`FV3_GPU_DEBUG_FAIL=1`, 192192 per-stage diagnostic lines):

- **zero** NaN or Inf at any stage, any tile, any substep;
- `delp` positive over the owned box on all six tiles;
- all six tiles evolved and evolved differently;
- no stability warning fired.

`delp` growth series, tile 1, `entry` stage, owned max:

| step | 0 | 9 | 19 | 29 | 39 | 51 |
|---|---|---|---|---|---|---|
| `delp` max (Pa) | 36904.3 | 37161.7 | 37366.3 | 37519.9 | 37663.6 | 37795.2 |
| per-step increment | 31.4 | 22.1 | 16.4 | 14.9 | 12.1 | 9.6 |

Monotonically **decelerating** — a relaxation toward balance, not growth. Owned
minimum is essentially constant at 584.78 Pa. Total excursion is 2.4% over 510 s
of model time, consistent with the smooth unbalanced-metric forcing documented in
§1.4 and not with an amplifying mode.

### 4.3 A side observation, and what it does and does not prove

At `im = 8` the `u` and `v` hashes are **bit-identical** before and after the
first `halo_in` exchange:

```
[FV3DBG tile=1 step=0] it=1 entry    u  hash=a6194f34ae4ff026 ...
[FV3DBG tile=1 step=0] it=1 halo_in  u  hash=a6194f34ae4ff026 ...
```

This is the strongest available *direct* confirmation that Task 1 worked: the
exchange delivered the neighbour's owned values into tile 1's halo, and they were
bitwise the values the analytic field had already placed there. It is consistent
with the seam test, which measured a worst D-grid component disagreement of
exactly `0.000e+00` across all 1332 halo edges at `im = 8` — signed permutation
matrices are exact in floating point, so wherever a seam relates two panels by one
of the six tile maps, the corner coordinates and hence the projected components
agree to the bit. (At `im = 16` the same quantities are `7.816e-14`, not zero, so
this is an `im = 8` property, not a general one.)

It is **not** evidence that the exchange is live. A dead exchange would produce
the identical unchanged hash. Seam liveness must therefore come from a test that
creates a genuine cross-seam mismatch and requires it to propagate — the Phase E
cross-tile influence check in `fv3_dispatch_smoke`, which perturbs one tile's
`u`/`v` and requires a *different* tile to change, plus
`fv3_halo_exchange_multitile`. Both are GPU-gated ctests. No seam-liveness claim
is made from the unchanged hash.

**Seam liveness — CONFIRMED, separately, by that test.** `fv3_dispatch_smoke`
passed in both R8 and R4 (tests #15 of 22, 0.81 s / R8). Its Phase E `pass` is
conjunctive and includes both halves of the causality argument
(`test_dispatch_smoke.cpp:2496-2517`):

- `causal`: tile 1's checksum must *move* when only its west donor tile's initial
  `u`/`v` are kicked, tile 1's own input being byte-identical to the clean run.
  The donor is **queried from the mosaic** (`t1.neighbors[TileEdge::west].tile`)
  rather than written down, so a wrong neighbour mapping cannot pass by accident.
- `kick_landed`: the kicked tile itself must move, which rules out the vacuous
  pass where the perturbation never applied and "identical" meant nothing.

A dead exchange fails `causal`. Because Phase E drives the same
`make_host_state` IC this milestone rewrote, the liveness evidence is on the new
IC, not a historical one. `fv3_halo_exchange_multitile` (#16) and
`fv3_halo_topology` (#11) also passed.

Phase E additionally reports that two identical six-thread runs agree **bitwise**
on all six tiles, i.e. the stream ordering holds under concurrency.

### 4.4 The initcheck signature that disappeared, and why

The acceptance criterion allowed "ONLY previously classified authoritative dead
reads". The run reported `ERROR SUMMARY: 0 errors`, which is *better* than
allowed — so it was checked rather than accepted, because a silently
uninstrumented sanitizer run reports zero too.

The previously classified read was 100 × `Uninitialized __global__ memory read of
size 8 bytes`, one signature only, in `dgrid::copy_corners_kernel`. Its host
backtrace (`s3_initcheck.log`) names the caller precisely:

```
launch_dgrid_dynamics_cuda -> launch_d_sw_cuda
  -> launch_del6_vt_flux_cuda -> launch_copy_corners_cuda
```

so it is `sw_core.F90`'s `del6_vt_flux`, **not** `tp_core.F90`'s `deln_flux` —
the latter was deliberately widened to stay initcheck-clean and carries a comment
saying so (`wp_b_cuda.cu:517-529`).

`launch_del6_vt_flux_cuda` (`wp_b_cuda.cu:812-822`) initialises `d2` over
`[is-1-n, ie+1+n] × [js-1-n, je+1+n]`, faithfully to Fortran, and calls
`copy_corners` over the full `b` bounds for every `n > 0`. With `ng = 3`:

| `n` | init box | covers `[is-ng, ie+ng]`? | `copy_corners` called? | initcheck |
|---|---|---|---|---|
| 0 | `[is-1, ie+1]` | no | no (`if (n > 0)`) | clean |
| 1 | `[is-2, ie+2]` | **no — outer ring undefined** | yes | **reads uninitialised** |
| 2 | `[is-3, ie+3]` | yes, exactly | yes | clean |

The groups are selected per level by `LevelFilter{mode=3, nord_sel=nord}` against
`ndif(k)`. The authoritative translation yields `ndif ∈ {0, 2}` only — 0 at
k = 0..7, 2 at k = 8..9 — because `ndif(k) = nord_v(k) = min(2, nord)`
(`dyn_core.F90:713`) with the GEOS `nord = 2`. **The `n == 1` group is therefore
empty, no thread reads, and the signature cannot occur.** The fabricated table it
replaced did populate `ndif == 1`.

Consequences, stated separately because they are different claims:

- The 0-errors result is **genuine and explained**, not an artifact.
- It is **not a fix.** The narrow init box in `del6_vt_flux` is unchanged. It is a
  latent, configuration-reachable read: any configuration with `nord = 1` puts
  every non-sponge level in the `n == 1` group and re-exposes it.
- It is **Fortran-faithful and value-harmless**: Fortran leaves the same cells
  undefined and rotates them, and the rotated results land only in `d2` cells no
  later loop reads. It is a dead read in both implementations. Widening the box —
  as `deln_flux` does — would silence it without changing any used value, and is
  recorded as a deferred cleanup rather than done here, since this milestone is
  not authorised to change WP-B numerics for tidiness.

Positive controls that the sanitizer was actually instrumenting: the initcheck
leg took 1228.5 ms/step against ~24 ms/step unsanitized, a ~50× slowdown; and
`racecheck` on the same binary emitted its own distinct `RACECHECK SUMMARY`.

### 4.5 The `n == 1` index arithmetic, and the limit of this evidence

The table in §4.4 is not an estimate. With `is = 1` and `ng = 3`, take the SW
corner of `copy_corners_kernel` at `dir == 1` (`wp_b_cuda.cu:105-107`):

```
writes q(i, j)      for i, j in [1-ng, 0] = [-2, 0]
reads  q(j, 1 - i)  so first index in [-2, 0], second in [1, 3]
```

The read therefore touches first index `-2`. `del6_init_kernel`'s box starts at
`i1 = is - 1 - n`:

| `n` | `i1` | is `-2` initialised? |
|---|---|---|
| 1 | `-1` | **no** |
| 2 | `-2` | yes |

That single cell is the whole finding. It also explains the observed count: the
hits are confined to `block (0,0,1)`-style low blocks, i.e. the `t/s` threads that
land on the outermost ring.

**Limit of this evidence, stated because it matters.** This is derived from source
and index arithmetic, *not* measured. A forward reproduction was attempted and is
**vacuous**: initcheck was run over `fv3_wp_b4_divergence` and `fv3_wp_b5_dsw`,
the two ctests that exercise `nord == 1` groups, and both reported
`ERROR SUMMARY: 0 errors`. Inspecting them afterwards shows why — every test in
the suite sets `ndif` to `2` or `0` and none to `1`
(`test_wp_b4_divergence_cuda.cpp:1562,1643`; `test_wp_b_cuda_smoke.cpp:85,760,846`).
Their `nord == 1` coverage is of `nord_k`, the *divergence*-damping selector
(`LevelFilter` mode 2), whereas `launch_del6_vt_flux_cuda` selects on `ndif`
(mode 3). So those runs could not have reproduced the signature whether the
mechanism is right or wrong, and their passing says nothing either way.

Reproducing it would need a test that sets `ndif == 1` — a one-line addition to
`test_wp_b_cuda_smoke.cpp`'s existing `s.ndif_value` knob. That is recorded as a
deferred item rather than done here: it changes a WP-B test to demonstrate a read
that is dead in both implementations, and this milestone's mandate is the
benchmark configuration, not WP-B coverage.

---

## 5. Status summary

| item | status |
|---|---|
| Global cube-consistent IC | IMPLEMENTED |
| Seam-continuity test | PASS (0 failures, im=8 and im=16) |
| Authoritative damping translation | IMPLEMENTED |
| Fabricated damping tables removed | YES |
| Per-level arrays now sourced authoritatively | `nord_k`, `ndif`, `nord_w`, `d2_bg`, `d4_bg`, `damp_v`, `damp_w`, `damp_t`, `d_con`, `kgb` (10 contract fields) + `d_ext`, `nord_t`, `dddmp` |
| `d4_bg_top`/`nord` classification | SAFE FOR CURRENT DEFAULTS BUT NOT GUARANTEED; `(0.16, nord=2)` = POTENTIAL USER-CONFIG FOOT-GUN |
| Runtime validation/warning | IMPLEMENTED (warn, never clip, never reject) |
| 8×8 R8, 6 tiles, n_split=3, 100 calls | PASS (120 calls actual) |
| 16×16 R8, 100 calls | PASS (120 calls actual) |
| 32×32 R8, 50 calls | PASS (70 calls actual) |
| Independent 8×8 runs | 3/3 PASS |
| `delp` positivity, all six tiles | PASS, every leg |
| NaN/Inf | none, 192192 diagnostic lines |
| Six tiles evolve non-degenerately | PASS (12 CHANGED, distinct per-tile deltas) |
| Seam exchange live | **PASS** — `fv3_dispatch_smoke` Phase E causality + control, R8 and R4 (§4.3) |
| Benchmark-only stability override active | NO (`WIND/HORIZ/DAMPV/D4_SCALE` all 1) |
| compute-sanitizer memcheck | PASS, 0 errors |
| compute-sanitizer racecheck | PASS, 0 hazards (0 errors, 0 warnings) |
| compute-sanitizer initcheck | PASS, 0 errors — prior dead read explained and not reproduced (§4.4/§4.5) |
| R8 CTest | 22/22 PASS |
| R4 CTest | 22/22 PASS |
| Performance timing (§7) | RESUMED and COMPLETE for CUDA resident + transfer |
| Fortran performance baseline | **BLOCKED** — `@fvdycore` absent, must not be populated (§7.3) |
| Numerical validation | **NOT RUN** |

---

## 6. What has NOT been established

Stated plainly, because every item below is a level this milestone did not reach:

- **No numerical validation.** Nothing here was compared against authoritative
  Fortran savepoint output. The seam test verifies internal geometric
  consistency, not agreement with FV3. The damping translation is a faithful
  *reading* of `dyn_core.F90:677-765` verified by host unit tests against
  hand-traced expected values, not a comparison against Fortran-produced arrays.
- **The IC is not balanced.** §1.4.
- **`gamma_v`/`gamma_w`/`gamma_t` are not threshold-measured.** §3.1.
- **The del-2 (`nord = 0`) critical coefficient is predicted, not measured.**
  Predicted 0.25. A threshold-measurement leg for it was identified and not run.
- **`cgrid_uc`/`cgrid_vc`/`agrid_wind` are analogues, not ports.** §1.4.
- **Bitwise agreement with FV3 grid geometry is not claimed.** §1.4.

### Deferred items, recorded so they are not lost

- 3A.11U §4 seam-sensitive test (the rest-state leg is still blind to vector seam
  mapping at `WIND_SCALE=0`, since exchanging a zero vector is invariant under any
  sign/swap error — though scalars now carry global structure there, so it is no
  longer blind to *scalar* seam mapping).
- Implement E5; implement E11.
- k = 0 damping row per `dyn_core.F90:721`.
- D1: `clamp_nord_cap` caps the group loop at `ng-1 == 2`, so `nord_k == 3`
  silently gets zero divergence damping. Dead code D2–D5.
- `del6_vt_flux`'s `n == 1` uninitialised `d2` ring (§4.4/§4.5): add an
  `ndif == 1` case to `test_wp_b_cuda_smoke.cpp` to reproduce it under initcheck,
  then decide whether to widen `del6_init_kernel` the way `deln_init_kernel`
  already is. Value-harmless in both implementations; not exercised by the GEOS
  production defaults.
- Halo documentation defect at `fv3_gpu_halo_exchange.cu:44-52` and
  `fv3_gpu_halo_topology.hpp:289-297`.
- `delpc` aliasing deviation.
- The `adiabatic_init`/`d4dmp` caching latent Fortran bug.
- An `nsys`/`ncu` profile to confirm the launch-bound inference in §7.
- Four reference-tree fork bugs, upstream, not ours to fix here:
  `get_gnomonic_angl_coords:3547-3548` double-assigns `p(2,i,j)`;
  `get_gnomonic_ed_coords:3491-3502` reuses one `b` array;
  `pmxn:5406` `elseif` blocks `pmin`;
  `globalsum` latlon `acapS`/`acapN` uninitialized.

---

## 7. Milestone 3A.11 performance timing — resumed

Run on the stable configuration only, after the acceptance matrix passed.
A100-SXM4-40GB, SM 1410 MHz, mem 1215 MHz, 400 W cap, `warpa003`.

Methodology as originally specified, no substitutions: **20 warmup calls,
100 timed calls, 5 repetitions, median**, both `resident` and
`transfer_inclusive` modes in one invocation (`--mode both --repeats 5`). Not
single-step timing. No `FV3_BENCH_*`, no `FV3_GPU_DEBUG_FAIL`; the banner
confirms `WIND_SCALE=1 HORIZ_SCALE=1 DAMPV_SCALE=1 D4_SCALE=1` in every leg, and
no stability warning fired in any leg. All three legs reported
`3A.11 CUDA perf benchmark: PASS`.

### 7.1 Medians, R8, npz = 10, n_split = 3, 6 tiles

| grid | owned cells | resident (ms/call) | transfer-inclusive (ms/call) | transfer cost |
|---|---|---|---|---|
| 8×8 | 3 840 | **28.332** | **29.358** | +1.026 (+3.6%) |
| 16×16 | 15 360 | **29.690** | **30.896** | +1.205 (+4.1%) |
| 32×32 | 61 440 | **28.827** | **31.178** | +2.351 (+8.2%) |

Spread across the 5 repetitions is tight — within 0.6% at 8×8 for both modes.
Repetition 1 is the *fastest* at 16×16 and 32×32 (27.48 and 26.22 ms), i.e. below
the median rather than above it; recorded as an observation, most likely a clock
boost that settles, not attributed.

### 7.2 The headline result: cost is grid-independent here

A 16× increase in owned cells (3 840 → 61 440) changes the resident cost by under
2%, and non-monotonically. That is measured, and it means these problem sizes are
**not compute bound**. The strongly supported inference is that the integrated
dyn_core is **launch- and orchestration-latency bound** at this scale:

- one call is 3 acoustic substeps; median 28.8 ms/call ≈ 9.6 ms/substep;
- a substep issues ~18 stages per tile, each stage several kernels, across 6 tiles
  driven by 6 host threads — order 10³ launches per substep;
- 10³–10⁴ launches at a few µs of issue cost each lands in exactly the observed
  range, with the device largely idle between short kernels.

Stated as an inference, not a measurement: no `nsys`/`ncu` profile was taken, and
that is the correct next step before any optimization. It is worth noting that
this is precisely the regime the deferred optimizations target — kernel fusion,
multi-stream work, and CUDA Graphs all attack launch overhead rather than
arithmetic — so the deferral ordering in `project_architecture.md` (correctness
first) has not cost anything measurable yet.

The transfer-inclusive penalty *does* grow with size (3.6% → 8.2%) because it is
genuine H2D/D2H volume, and it is the one number here that scales as expected.
Even at 32×32 the persistent-residency architecture keeps it under 10%.

### 7.3 Fortran baseline — BLOCKED, and not worked around

The original methodology also calls for a Fortran baseline. **It cannot be
produced in this repository and no substitute was invented.**

- The benchmark emits only `CSV,cuda_gpu,...`; there is no host or Fortran
  dyn_core timing path in `cuda/dyn_core/` (`grep` for
  `host_dyn_core`/`cpu_dyn_core`/`reference_dyn_core` returns nothing).
- `@fvdycore` is absent from this working tree, so the authoritative Fortran
  dyn_core is not present as a buildable target at all.
- `docs/agent_instructions/discover_platform.md` anticipates exactly this and says
  **"Do not populate `@fvdycore` automatically."** So it was not populated.

Consequence, stated plainly: **no CPU-vs-GPU speedup ratio is reported, because
none was measured.** The CUDA resident and transfer-inclusive numbers above stand
on their own as absolute cost. A speedup claim requires the Fortran baseline and
belongs to the phase that populates `@fvdycore` — the same phase that owns
authoritative numerical validation, which is also NOT RUN.
