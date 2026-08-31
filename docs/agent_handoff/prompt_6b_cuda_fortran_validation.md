# Prompt 6B — CUDA vs. Authoritative Fortran `dyn_core` Numerical Comparison

Status: **SP0 BIT-IDENTICAL. SP1 finite everywhere; `ptop`/`phis`/`zs` now
authoritative (§12); `c_sw` still validated to round-off; SP1 overall still
SIGNIFICANT DIFFERENCE.** Gate C2 remains **BLOCKED**. See §12/FINAL REPORT
(this pass) at the end of this file for the current, superseding result;
§1-11 and the first FINAL REPORT below are the pass that proved `ptop`/`phis`
wrong but had not yet corrected them — preserved as history, not current
status.

Two things changed materially in this pass, and both correct earlier claims in
this file:

1. The `p_grad_c` / halo-coverage hypothesis recorded in the previous pass
   (§5-§7 of that pass, now in the collapsed history) is **DISPROVEN**.
   `p_grad_c` is exonerated; `gz` and `pkc` halo coverage are both **PASS**.
2. The actual cause of the NONFINITE failure was the **missing `it==1`
   pre-`c_sw` scalar halo** — the "E5 missing" gap already recorded in
   `CURRENT_STATE.md` §10, whose own note predicted it "would bite with a real
   (non-benchmark) IC". It has been fixed with one added halo resolve. All NaN
   is gone.

After that fix, SP1 is finite but classifies **SIGNIFICANT DIFFERENCE**. The
remaining divergence is traced to two proven *harness input-fidelity* gaps
(`ptop`, and surface topography `zs`/`phis`), not to a demonstrated numerical
translation defect in any operator — see §6. `c_sw` itself is now validated to
round-off.

## 1. Objective

Run the CUDA `dyn_core` candidate from the authoritative Fortran SP0 state
(Prompt 6A; SP0 identity verified bit-identical at Milestone 6A.5 Section 10
and treated as ground truth here, not re-checked) through the production
first-run configuration (c8/npz=10/n_split=3/R8/six tiles, `hord_mt/vt/tm=6`,
`hord_dp=-6`, `nord=2`, `d4_bg=0.12`, `vtdm4=0.01`, `a_imp=1.0`,
`grid_type=0`, `hydrostatic=.F.`, `bdt=3600`) and compare CUDA state at
SP1/SP2/SP3 against the Fortran reference savepoints, stopping at the first
significant divergence (Gate C2).

## 2. `p_grad_c` read-box reconciliation — MATCH, no defect

Read the CUDA `p_grad_c` launcher/kernel and its direct helpers, and the
authoritative `subroutine p_grad_c` (`dyn_core.F90:1592-1651`, called at
**`dyn_core.F90:634`** — note older project docs cite `:614`, which is stale
for this fixture).

| Field | Fortran required box | CUDA required box | CUDA available box | Match? |
|---|---|---|---|---|
| `gz` | 1 ring W/S, all npz+1 interfaces, no corners | identical | full ring, all levels | **YES** |
| `pkc` | 1 ring W/S, all npz+1 interfaces, no corners | identical | full ring, all levels | **YES** |
| `delpc` | owned + 1 ring | identical | full ring | **YES** |
| `ptc` | owned + 1 ring | identical | full ring | **YES** |

Union of reads is the one-ring box **minus** the four diagonal corners; the
corners are never read.

## 3. `gz` halo coverage: **PASS**

`FV3_GPU_H_GZ_FIRST` resolves (it==1) before `cgrid_pressure`, and is a
correctly-defined live group. More decisively, no `gz` exchange is *needed*
between the Riemann solver and `p_grad_c` on either side, because
`Riem_Solver_c` itself writes `gz` over exactly the ring `p_grad_c` reads
(`fv3_riem_solver_c.cu` writeback, mirroring `nh_utils.F90:421-426`).

## 4. `pkc` halo coverage: **PASS** — the previous pass's structural-gap claim was wrong

The previous pass flagged "no halo group resolved for `pkc` before
`cgrid_pressure`" as a structural gap. Determined **from authoritative
source** that this is not a defect: the Fortran performs **no** `pkc` or `gz`
exchange between `Riem_Solver_C` (`dyn_core.F90:603-606`) and `p_grad_c`
(`:634`) either. It does not need one, because `Riem_Solver_c` writes `pkc`
over the full `is-1..ie+1 / js-1..je+1` ring at every interface level
(`nh_utils.F90:350-351,:356`).

The CUDA Riemann solver reproduces that ring faithfully, **including the top
interface** — `fv3_riem_solver_c.cu:96` contains `a.pef.at(i,j,0) = a.ptop;`
(`nh_utils.F90:365`). (An intermediate hypothesis that this write was missing
was raised and disproven by direct source read.) So `H_PKC2` resolving only
after the Riemann path is correct, not late; no earlier `pkc` exchange was
added.

## 5. First invalid read — the decisive proof

Using the existing opt-in raw-dump diagnostic, the `it=1` NaN index map on
tile0 is exactly 20 ring-perimeter columns, identical on every level:
`(i=0, j=0..9)`, `(i=1..4, j=0)`, `(i=1..4, j=9)`, `(i=9,j=0)`, `(i=9,j=9)`
(tile3 shows the expected rotated/transposed variant). This maps one-to-one
onto the observed consumer damage — `uc` NaN at `(i=1, j=1..8)` (the x-sweep
at `i=is` reading `i=is-1`) and `vc` NaN at `(i=1..4, j=1)`/`(i=1..4, j=9)`
(the y-sweep at `j=js` and `j=je+1`). The four ring corners are NaN but never
read, hence harmless — exactly as §2's read pattern predicts.

The bad *input* was not `gz`/`pkc` themselves but `delpc`/`ptc`, which the
previous pass had only checked for **finiteness**:

| location | `delpc` | `ptc` |
|---|---|---|
| owned interior | ~1e4 | physical | 
| good ring columns | 1e1–1e2 | physical |
| the 20 failing ring columns | ~1e-15 | 1.000000e-15 exactly |

Feeding `delp ≈ 1e-15` into `pm2[k] = dm[k] / fv3_log(pem[k+1]/pem[k])`
(`fv3_riem_solver_c.cu`, `nh_utils.F90:394`) makes the log argument ≈ 1, the
log ≈ 0, and the quotient NaN/overflow inside `sim1_column` — which then
lands in `gz` and `pkc`, and from there in `uc`/`vc`.

Traced one step further back: the **authoritative Fortran SP0 savepoint itself
has `delp = pt = 1e-15` across its entire halo** (verified directly on disk in
`tools/savepoints/sp0_cuda_authoritative/tile0/{delp,pt}.bin` — a clean 1e-15
frame around the 1e4 owned core). That is FV3's pre-exchange sentinel: at
`dyn_core` entry the `delp`/`pt` halos are simply not yet valid.

## 6. Root cause and fix: the missing `it==1` pre-`c_sw` scalar halo (E5)

Fortran makes those halos valid before `c_sw` consumes them; CUDA did not.

- `fv_dynamics.F90:620-627` **starts** `i_pack(1)` over
  `{q_con, cappa, delp, pt}` — i.e. outside `dyn_core`.
- `dyn_core.F90:494-497` **completes** it inside `if (it==1)`, before `c_sw`
  at `:535`.
- For `it >= 2` no pre-`c_sw` completion exists, because the previous
  substep's `:892-895`/`:926` start/complete pair already refreshed the same
  group — which is what the existing CUDA `:844`/`:878` resolve already
  mirrors. So the only missing edge was the `it==1` one.

**Fix** (`cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`, one line, inserted after
`resolve(FV3_GPU_H_W_IN)` and before `cgrid_predictor`):

```cpp
if(it==1 && !resolve(FV3_GPU_H_DSCALAR)) return rc;        // :496 -> :535
```

`FV3_GPU_H_DSCALAR`'s descriptor (`fv3_gpu_halo_exchange.hpp:220-228`) is
already field-for-field `i_pack(1)`: `q_con, delp, pt, cappa`, all CENTER
scalars at `fv3_domain_halo_width`. Same fields, same staggering, same width,
same seams — so this is **a missing resolve of an already-correct group**, not
a new or repurposed one, and squarely inside the Fix Policy's allowed
"add/move a halo resolve". `resolve_halo()` was verified to exchange
unconditionally when a halo peer is present (no staleness short-circuit), so
the added resolve is not a silent no-op. **Numerical algorithm changed: NO** —
no kernel, stencil, or expression was touched.

Why it was silent until now: every prior harness IC was uniform per tile, so
an unexchanged `delp`/`pt` halo happened to hold exactly the values a correct
exchange would have delivered. That is why 22/22 ctest, a clean sanitizer
matrix, and multi-step stability runs never surfaced it.

**Effect:** `it=1` NaN totals across all 6 tiles went from
`{uc:480, vc:480, pkc:1200, gz:1200}` to **all zero**. The previously-poisoned
ring columns now hold physical `delpc ≈ 1e4` / `ptc ≈ 24.6`.

## 7. SP1 after the fix: finite, but SIGNIFICANT DIFFERENCE

Per-field worst verdict over the defined box, all 6 tiles (see §9 for the
comparator and what "defined box" means):

| Field | Verdict |
|---|---|
| `delpc` | **ROUND-OFF SCALE** (max_abs 6.9e-11, max_rel 6.9e-15) |
| `ptc` | **ROUND-OFF SCALE** (max_abs 1.5e-13, max_rel 1.06e-14) |
| `ws3` | SIGNIFICANT DIFFERENCE |
| `gz` | SIGNIFICANT DIFFERENCE |
| `pkc` | SIGNIFICANT DIFFERENCE |
| `uc` | SIGNIFICANT DIFFERENCE |
| `vc` | SIGNIFICANT DIFFERENCE |

Region split (pe0; owned = local i,j ∈ 1..8, ring = the
`is-1`/`ie+1`/`js-1`/`je+1` perimeter):

```
field   owned max_abs  owned max_rel   ring max_abs   ring max_rel
delpc      1.8190e-12     1.8216e-16     6.9122e-11     6.9199e-15
ptc        3.5527e-15     2.5245e-16     1.5277e-13     1.0556e-14
ws3        1.9059e-04     1.0000e+00     4.7340e-03     1.0000e+00
gz         1.1400e+05     1.0000e+00     2.9265e+05     1.0000e+00
pkc        9.0000e+02     9.0000e-01     4.6098e+03     9.0000e-01
uc         1.3782e-01     1.6046e-02     2.6419e-02     7.2290e-03
vc         5.3107e-01     1.9662e+00     5.3107e-01     6.0317e-01
```

`c_sw` — the producer of `delpc`/`ptc` — is now faithful to machine precision
in **both** the owned box and the ring. That is strong independent
confirmation the §6 fix is correct, and it is the first CUDA operator in this
project to be numerically validated against authoritative Fortran at all.

### 7.1 The remaining divergence: two proven input-fidelity gaps, not an operator defect

The suspiciously exact ratios (`pkc` max_rel exactly 0.9; `gz` and `ws3`
max_rel exactly 1.0) were run down to their sources rather than left as
"unexplained". Both boundary levels are readable directly from both sides:

| Quantity | Authoritative Fortran | CUDA harness | Consequence |
|---|---|---|---|
| `ptop` | **100.0** | **1000.0** (`run_small_perf_bench.cpp:752`, `cfg.ptop = h.ptop` at `:1791`) | `pkc(:,:,1) = ptop` verbatim (`nh_utils.F90:365`) ⇒ max_rel = 900/1000 = **exactly 0.9** |
| `zs` / `phis` (surface topography) | real, `phis` ∈ **[−23.5, 1105.9]** m²/s² | **identically 0** (`zs` at `:676`; `phis = grav*zs` at `:1202`) | `gz(:,:,km+1) = hs = phis` (`nh_utils.F90:421`) ⇒ max_rel = **exactly 1.0** |

`ws3`'s signature is consistent with the same surface gap: max_abs is tiny
(1.9e-4) with max_rel exactly 1.0 against a CUDA side of 0 — i.e. flat terrain
produces ≈ zero surface vertical velocity where real topography does not.
(`ws3` is `update_dz_c`'s output per `dyn_core.F90:577`; the precise expression
linking `zs` to it was **not** re-read this pass, so this one is stated as
consistent-with rather than proven.)

Both gaps are the same *class* as the `bdt` gap fixed earlier (see history
below): a harness input that is a deliberate, self-consistent choice for the
synthetic benchmark and simply wrong for validation. Both fall through the gap
between the two existing loaders:

- `grid_metrics_6a5::load()` **explicitly excludes** `phis`/`zs` as
  "dynamical/transient" (Milestone 6A.5 §7).
- The SP0 loader's 10-field list (`u,v,w,delp,pt,delz,q_con,pe,peln,pk`)
  contains neither `phis`/`zs` nor `ptop`.

Neither is exported by any Fortran run (checked: no `phis`/`zs`/`ptop` file in
the 6A savepoint or 6A.5 grid-metric exports). **However, both are recoverable
from the existing SP1 savepoint with no new Fortran instrumentation**:
`ptop = pkc(:,:,1)` and `phis = gz(:,:,km+1)`. That recovers `phis` over the
owned + one-ring box, which is exactly the extent `Riem_Solver_c`/`p_grad_c`
need for SP1; a wider extent (for `d_sw` and later stages) would require a
proper export.

**Honest boundary:** that `ptop` and `phis` are wrong is *proven*. That they
account for the *entire* remaining SP1 divergence is **not** proven — that
requires a rerun with both corrected, which is the next task's scope, not this
one's (this pass was authorized for one bounded reconciliation and one SP1
rerun, both used).

## 8. First divergent operator

`c_sw`: **agrees to round-off**. Stage order inside SP1 is
`c_sw` → `update_dz_c` (writes `gz`, `ws3`) → `Riem_Solver_c` (writes `pkc`,
overwrites `gz`, reads `ws3`) → `p_grad_c` (writes `uc`, `vc`). `ws3` is
written only by `update_dz_c` and diverges in the owned box, so **the first
divergent stage inside SP1 is now `update_dz_c`** — but per §7.1 its two
surface/top boundary inputs are demonstrably not authoritative, so this is
**not** yet attributable to `update_dz_c`'s arithmetic.

## 9. SP2 / SP3: NOT RUN

Stopped at the first mismatch as instructed. The harness's own end-of-run
summary still prints FAIL (NaN in `u/v/delp/pt/w/delz` on tiles 4 and 5 and
`delp` positivity failures at the end of the full 3-substep step; tile6
clean) — that is downstream of SP1, in SP2/SP3 territory, and was deliberately
not investigated.

## 10. Outputs / reproducibility

- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` — the §6 one-line `it==1`
  `H_DSCALAR` resolve (the only functional change this pass), plus the
  previously-added `FV3_GPU_DEBUG_DUMP_DIR`/`FV3_GPU_DEBUG_DUMP_STAGE`-gated
  raw dump inside `debug_check_finite_stage()` (off by default).
- `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` — `g_bdt` + `--bdt` flag
  (default `10.0` preserves all pre-existing benchmark behavior).
- `tools/savepoints/compare_cuda_fortran.py` — **new permanent comparator**.
  Reuses `savepoint_reader.load_field()` (Fortran side) and
  `compare_sp0_identity.gather()` (CUDA side), so there is one index-mapping
  definition, not two. Grades BITWISE / ROUND-OFF SCALE / SMALL NUMERICAL
  DIFFERENCE / SIGNIFICANT DIFFERENCE / NONFINITE and reports max_abs,
  max_rel, rms, mismatch count and argmax location.
  ```
  python3 tools/savepoints/compare_cuda_fortran.py \
    <fortran_savepoint_dir> tools/savepoints/sp1_cuda_dump_fix1 sp1 1 cgrid_pressure
  ```
  Its `DEFINED_BOX` restricts comparison to the region each field is actually
  written at SP1. This is **not** a tolerance knob: outside those boxes
  *neither* implementation has written anything at SP1 (Fortran
  `init_ijk_mem`'s `huge_r=1e40`, CUDA's corner-ghost poison and untouched
  padding), so comparing them compares two undefined regions — the first
  comparison run in this pass returned a meaningless max_abs of 6.2e126 at
  index (0,0,0) for exactly that reason. Full-declared-bounds numbers are
  reported alongside the defined-box ones, not instead of them.
- `tools/savepoints/sp1_cuda_dump_fix1/tile{0..5}/it={1,2,3}_cgrid_pressure__{uc,vc,pkc,gz,delpc,ptc,ws3}.bin`
  — post-fix raw float64 dumps (gitignored, regenerable).
- Regression: **R8 ctest 22/22 PASS** on A100 after the runtime change
  (warranted here because `fv3_gpu_runtime.cpp` is shared runtime code, unlike
  the harness-only changes of Milestone 6A.5).
- Neither authoritative Fortran nor any CUDA numerical algorithm was modified.
- Nothing committed, pushed, or merged.

## 11. Next steps

1. Inject authoritative `ptop` (100.0) and authoritative surface topography
   (`zs`/`phis`) into the validation path, gated the same opt-in way as
   `--grid-metrics`/`--sp0-state` so the synthetic benchmark path is
   untouched. Recover both from the existing SP1 savepoint (§7.1) if the
   owned+one-ring extent suffices; otherwise add a Fortran export.
2. Rerun SP1 only and re-classify. Determine whether `update_dz_c` still
   diverges once its boundary inputs are authoritative.
3. Only then proceed to SP2/SP3.

<details>
<summary>Previous pass (superseded) — the NONFINITE report and the disproven halo-coverage hypothesis</summary>

That pass established the `bdt` configuration fix and the raw-dump mechanism
(both still in use and still correct), then reported SP1 as **FAIL —
NONFINITE**: `uc`/`vc` 80/2250 NaN per tile reaching the owned interior,
`pkc`/`gz` 200/2475 NaN confined to the halo padding ring, `delpc`/`ptc`/`ws3`
clean, uniform across all 6 tiles. `it=2`/`it=3` showed a forward-propagating
cascade, not independent failures.

**`bdt` fix (still valid):** `run_small_perf_bench.cpp` had `step.bdt`
hardcoded to `10.0` in both `run_mode_a()` and `run_mode_b()`, while the
authoritative Prompt 6A run used `bdt=3600.0` (`RUN_DT=3600`, `k_split=1`) —
a ~360x-smaller acoustic timestep. Fixed with `static double g_bdt = 10.0;`
plus a `--bdt` flag; validation runs pass `--bdt 3600`.

**Disproven hypothesis:** that pass classified the cause as a TRANSLATION
DEFECT (candidate) — a halo-coverage gap, with "`pkc` has no halo-exchange
group resolved before `cgrid_pressure` consumes it" as the leading locus, and
explicitly declined to read the `p_grad_c` kernel source. Reading it (and the
authoritative Fortran) in this pass showed both `gz` and `pkc` coverage are
correct and that the Fortran performs no such exchange either; see §2-§4. The
declined-fix decision in that pass was the right call — a speculative `pkc`
halo group would have been wrong.

Also from that pass and still valid: config-knob matching and field-layout /
registration checks are clean; `pe`/`peln` are `(i,k,j)` and
`pk`/`pkz`/`pkc`/`pk3`/`gz` are `(i,j,k)` on both sides; the existing
`FV3_GPU_DBG_STAGE` checkpoints sit at the correct SP1 (`cgrid_pressure`) and
SP2 (`dgrid_dynamics`) boundaries.

**Earlier still (both closed before that pass):** the CUDA harness's
fabricated uniform grid metrics were replaced with authoritative non-uniform
c8 metrics behind `--grid-metrics` (Milestone 6A.5), and the
Fortran-SP0-state → CUDA H2D loader was built with SP0 identity confirmed
**BIT-IDENTICAL, 60/60 fields** (Milestone 6A.5 Section 10).

</details>

## FINAL REPORT

- Validation case: c8 (8x8/tile) / npz=10 / n_split=3 / R8 / six tiles,
  production first-run config, authoritative SP0 state, authoritative grid
  metrics, authoritative global `da_min`/`da_min_c`, `bdt=3600`
- Fortran `p_grad_c` required halo: 1 ring west/south, all npz+1 interface
  levels for `gz`/`pkc`, four diagonal corners never read
- CUDA `p_grad_c` required halo: **identical — MATCH**
- `gz` coverage: **PASS**
- `pkc` coverage: **PASS** (the previous pass's structural-gap claim is
  disproven; Fortran performs no exchange there either, and the CUDA Riemann
  solver writes the full required ring including the `ptop` top interface)
- First invalid input read: `delpc`/`ptc` at ~1e-15 in the one-ring halo,
  traced to `delp`/`pt` halos carrying FV3's 1e-15 pre-exchange sentinel in
  the authoritative SP0 state
- Root cause: the missing `it==1` pre-`c_sw` scalar halo — start at
  `fv_dynamics.F90:620-627`, complete at `dyn_core.F90:496` — i.e. the
  already-documented **E5** gap
- Fix applied: one added `resolve(FV3_GPU_H_DSCALAR)` at `it==1` before
  `cgrid_predictor`
- SP1 after fix: **finite everywhere (0 NaN, all 6 tiles) but NOT numerically
  agreeing**. `delpc`/`ptc` ROUND-OFF SCALE; `ws3`/`gz`/`pkc`/`uc`/`vc`
  SIGNIFICANT DIFFERENCE
- Max SP1 errors: see the §7 table (owned-box max_abs/max_rel: `delpc`
  1.8e-12/1.8e-16, `ptc` 3.6e-15/2.5e-16, `ws3` 1.9e-4/1.0, `gz` 1.1e5/1.0,
  `pkc` 9.0e2/0.9, `uc` 1.4e-1/1.6e-2, `vc` 5.3e-1/2.0)
- Numerical algorithm changed: **NO**
- Regression: R8 ctest 22/22 PASS
- Remaining divergence attributed to: two **proven** harness input-fidelity
  gaps — `ptop` (1000.0 vs authoritative 100.0) and surface topography
  `zs`/`phis` (identically 0 vs real) — not to a demonstrated translation
  defect. That they explain the divergence *in full* is not yet proven.
- Prompt 6B: **STILL BLOCKED AT SP1**, with the divergence relocated off
  `p_grad_c` and onto `update_dz_c`'s boundary inputs
- Gate C2: BLOCKED

SP1 STILL BLOCKED — PRESSURE-GRADIENT DIVERGENCE REMAINS

(Terminal line per the required template's binary choice. Stated precisely:
the pressure-gradient operator `p_grad_c` is **exonerated** — it reads exactly
what the Fortran reads and its inputs are now finite. The divergence that
remains is upstream of it, at `update_dz_c`, and is so far attributable to
non-authoritative `ptop`/topography inputs rather than to operator
arithmetic.)

## 12. `ptop`/`phis`/`zs` correction and SP1 rerun (this pass) — SUPERSEDES §7-11/FINAL REPORT ABOVE

Implemented the fix §11 of the previous pass called for: `ptop`, `phis`, `zs`
are now injected from the authoritative Fortran SP1 savepoint instead of the
synthetic harness defaults (`ptop=1000.0`, `phis=zs=0`), gated the same
opt-in way as `--grid-metrics`/`--sp0-state`.

### 12.1 Index/units relation — confirmed from source, not assumed

- `ptop = pkc(:,:,1)` (top interface). Read directly off the SP1 savepoint:
  **uniform 100.0** across the owned+one-ring box and across all 6 tiles —
  matches the authoritative value already cited in §7.1, now *derived*
  rather than hardcoded.
- `phis = gz(:,:,km+1)` (bottom interface, km=NPZ=10), **RAW, no `*grav`**.
  Verified by computing both the raw and the `*grav` variant over the SP1
  savepoint and comparing to §7.1's previously-published `phis` range
  ([−23.5, 1105.9]): the raw variant matches (max 1105.854), the `*grav`
  variant does not (max 10844.7). `zs = phis / grav` (`grav=9.80665`, matching
  `run_small_perf_bench.cpp:754` exactly) is therefore the correct inverse of
  the harness's own seeding invariant `phis = grav*zs` (`:1202`).
- Recovered over the SP1 `DEFINED_BOX` for `gz`/`pkc` (local Fortran `i,j` in
  `[0, NI+1]`, i.e. owned + one ring) — the exact extent `Riem_Solver_c`/
  `p_grad_c` consume at SP1 (§2). Zeroed outside that box, matching
  `make_host_state()`'s own default zero-seed there (no numerical claim about
  that region).
- Full characterization (not previously published): the two polar tiles
  (pe2/pe5) have a materially different, still-real `phis` range
  (`[−3057.7, 500.3]`) from the four non-polar tiles (`[−963.1, 1105.9]`
  over owned+ring; owned-only on non-polar tiles matches §7.1's `[−23.5,
  1105.9]` exactly — the wider owned+ring box simply reaches into terrain the
  owned-only box does not).

### 12.2 Implementation

- New converter `tools/savepoints/convert_sp1_boundary_to_cuda.py`: reads
  `sp1_it1_pe<N>.meta/.bin` for `pkc`/`gz`, verifies `ptop` uniformity, writes
  `<outdir>/tile<N>/{ptop.bin,phis.bin,zs.bin}` in the harness's own CUDA flat
  2D convention (`idx = i_cuda + j_cuda*ni_h`, same as `h.phis`/`h.zs`'s own
  default-seed loop). No new Fortran instrumentation; no new binary reader
  (reuses `savepoint_reader.load_field()`).
- `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp`: new `g_sp1_boundary_dir`
  global (default empty), new `sp1_boundary_6b::load()` namespace (overwrites
  `h.ptop` directly — no sentinel needed, unlike `da_min`/`da_min_c`, since
  no competing derived value exists for `ptop` — and reuses
  `sp0_state_6a5::read_doubles_into()` for `phis`/`zs`), new `--sp1-boundary
  DIR` CLI flag, `usage()` string updated. Call site inserted in `setup()`
  strictly after `sp0_state_6a5::load()` and strictly before `make_fields()`
  — early enough that `h.ptop` is set before `install_config()`'s
  `production_config()` reads it, and `h.phis`/`h.zs` are set before
  `make_fields()` captures their pointers for H2D registration. Default
  (empty) path is provably unaffected — same "empty string skips the whole
  block" idiom as the two existing loaders.
- Rebuilt `fv3_gpu_perf_bench` cleanly (`build_cuda_r8_perf/`, GCC 12.3.0 +
  CUDA 12.5 BYO stack) — no warnings introduced.

### 12.3 Verification (§3 of task)

- **Host load**: converter's own uniformity check plus the harness's runtime
  confirmation line, both A100 runs, all 6 tiles:
  `[6B-SP1] tile N: authoritative SP1-boundary ptop=100.000000 phis/zs loaded`
  — no `FAIL: sp1-boundary` lines. `ptop.bin`/`phis.bin`/`zs.bin` per-tile
  min/max recorded in each tile's `manifest.json`, matching the savepoint scan
  in §12.1 exactly (self-consistent, same read).
- **Device load (D2H readback)**: **NOT directly performed** — unlike SP0's
  `--sp0-verify-only`/`--sp0-dump` path, no D2H-readback mechanism exists for
  `phis`/`zs` specifically (they are not in `debug_check_finite_stage()`'s
  dump target list), and adding one is outside this pass's one-loader/one-
  rerun scope. Indirect evidence they reached the device correctly: `ws3`
  (the first CUDA-computed field to consume `zs`, via
  `launch_cgrid_vertical_pre_halo`) now agrees with the authoritative Fortran
  to **median absolute difference 1.2e-11** in the bulk of its owned+ring box
  (see §12.4) — inconsistent with `zs` being zero or otherwise mistransferred
  on the device, but this is inference from downstream numerics, not a direct
  readback. Recorded here as an open verification gap, not silently skipped.
- **Synthetic/performance path preserved**: ran the existing benchmark with
  no `--sp1-boundary` (and no `--grid-metrics`/`--sp0-state`), `--grid 8
  --npz 10 --n_split 3 --warmup 0 --iters 1 --repeats 1 --mode resident` on
  the same A100 node: `=== 3A.11 CUDA perf benchmark: PASS ===`, 0
  NaN/positivity failures on any tile — the Test-B smoke-run pattern from
  Milestone 6A.5 Section 10, reused rather than reinvented.

### 12.4 SP1 rerun result — max errors (worst verdict across all 6 tiles; DEFINED_BOX)

| Field | Verdict (this pass) | Verdict (§7, before fix) | max_abs | max_rel | Note |
|---|---|---|---|---|---|
| `delpc` | ROUND-OFF SCALE | ROUND-OFF SCALE | 6.9e-11 | 6.9e-15 | unchanged — `c_sw` still validated |
| `ptc` | ROUND-OFF SCALE | ROUND-OFF SCALE | 1.5e-13 | 1.06e-14 | unchanged |
| `ws3` | SIGNIFICANT DIFFERENCE | SIGNIFICANT DIFFERENCE | 1.32e-5 (pe0) | 1.0 (pe0/pe1); 0.069 (pe2-5) | max_abs dropped ~1.4e4x from §7's 1.9e-4/4.7e-3 (owned/ring); bulk median abs diff 1.2e-11 (round-off), but ~50% of the owned+ring box still shows O(1e-5) absolute deviation against a field whose own scale is ~1e-4, so `max_rel` still saturates near 1 |
| `pkc` | SIGNIFICANT DIFFERENCE | SIGNIFICANT DIFFERENCE | 3.72e3 (pe0) | 0.080 (pe0/pe1); 0.081 (pe2-5) | max_rel dropped from **exactly 0.9** to ~0.08 — the proven `ptop` fix is visibly working, but a residual ~8% error remains |
| `gz` | SIGNIFICANT DIFFERENCE | SIGNIFICANT DIFFERENCE | 4.06e5 (pe0) | 0.961 (pe0); 0.96 (all tiles) | essentially unchanged from §7's exactly-1.0 |
| `uc` | SIGNIFICANT DIFFERENCE | SIGNIFICANT DIFFERENCE | 8.0e-5 (pe0) | 4.9e-6 (pe0/pe1/pe3/pe4); 1.2e-4 (pe2/pe5) | max_abs dropped ~1700x from §7's 1.4e-1 (owned)/2.6e-2 (ring); classified SIGNIFICANT only because `max_rel` (1e-8 threshold) is exceeded on an otherwise near-round-off field — closest to resolved of the seven |
| `vc` | SIGNIFICANT DIFFERENCE | SIGNIFICANT DIFFERENCE | 3.1e-4 (pe0) | 1.69-1.99 (pe0/pe1/pe3/pe4); 1.18e-4 (pe2/pe5) | **not improved** on 4 of 6 tiles (comparable to §7's 1.97/0.60); improved sharply on the two polar tiles |

Full command and reproduction:
```
python3 tools/savepoints/convert_sp1_boundary_to_cuda.py \
  <fortran_sp1_dir> tools/savepoints/sp1_boundary_c8
FV3_GPU_DEBUG_FAIL=1 FV3_GPU_DEBUG_DUMP_DIR=tools/savepoints/sp1_cuda_dump_fix2 \
FV3_GPU_DEBUG_DUMP_STAGE=cgrid_pressure \
./fv3_gpu_perf_bench --grid 8 --npz 10 --n_split 3 --warmup 0 --iters 1 \
  --repeats 1 --mode resident --hardware A100 --bdt 3600 \
  --grid-metrics tools/grid_metrics/c8_authoritative \
  --sp0-state tools/savepoints/sp0_cuda_authoritative \
  --sp1-boundary tools/savepoints/sp1_boundary_c8
python3 tools/savepoints/compare_cuda_fortran.py \
  <fortran_sp1_dir> tools/savepoints/sp1_cuda_dump_fix2 sp1 1 cgrid_pressure
```
Raw dumps at `tools/savepoints/sp1_cuda_dump_fix2/` and boundary conversion at
`tools/savepoints/sp1_boundary_c8/` (both gitignored, regenerable). The
end-of-run harness summary for this run reported `FAIL` — expected and
**not investigated**, per scope: it is the same downstream SP2/SP3 NaN
cascade already recorded as out-of-scope in §9, now happening one step later
because `it=1`'s NaNs are gone.

### 12.5 Decision and root-cause classification

The `ptop`/`phis`/`zs` fix is **proven partially responsible, not solely
responsible**, for the SP1 divergence:

- **Real, large improvement**: `uc` (max_abs ↓ ~1700×), `ws3`'s bulk (median
  abs diff now 1.2e-11, round-off scale), `pkc` (max_rel 0.9 → 0.08).
- **No material improvement**: `gz` (max_rel 0.96, was 1.0), `vc` on 4 of 6
  tiles (max_rel 1.7-2.0, comparable to before).
- `c_sw` (`delpc`/`ptc`) remains validated to round-off, unchanged, on both
  owned and ring cells, on all 6 tiles — reconfirmed, not re-derived.

This rules out `ptop`/`phis`/`zs` as the *complete* explanation (disproving
§7.1's open question in the negative), while confirming they were a real,
worth-fixing gap (the `uc`/`pkc` improvement is not explainable by anything
else changed this pass). The residual divergence's specific origin — a
further un-recovered harness-input gap, or a genuine `update_dz_c`/
`Riem_Solver_c`/`p_grad_c` translation defect — is **not identified** in this
pass. Per scope, no further inputs were added speculatively and no kernel
source was modified.

- **`ptop` ingestion: PASS** (uniform 100.0, derived not hardcoded, loaded on
  all 6 tiles, default path unaffected)
- **`phis`/`zs` ingestion: PASS** (index/units relation confirmed from
  source, loaded on all 6 tiles, default path unaffected)
- **Host/device identity: PARTIAL** — host load PASS (direct); device D2H
  readback NOT PERFORMED (see §12.3; no existing mechanism, building one
  ruled out of scope)
- **Synthetic mode preserved: YES** (Test-B smoke rerun, PASS, 0 NaN/failures)
- **`c_sw`: PASS** (round-off, reconfirmed, unchanged from §7)
- **`update_dz_c` (`ws3`): STILL DIVERGES** — improved by ~4 orders of
  magnitude in absolute terms but `max_rel` still saturates near 1 on a
  near-zero-scale field on half the tiles
- **SP1 overall: FAIL** (SIGNIFICANT DIFFERENCE, defined box and full bounds)
- **First remaining divergent operator**: `update_dz_c` (`ws3`), propagating
  into `Riem_Solver_c` (`pkc`/`gz`) and `p_grad_c` (`uc`/`vc`) — same stage
  order as §8, now with authoritative boundary inputs and a materially
  different (improved but not resolved) error signature
- **Root-cause classification: OTHER / NOT FULLY IDENTIFIED** — proven not
  *solely* a harness-input gap (the fix helped but did not close it) and not
  proven to be a translation defect either; genuinely unresolved pending
  further coarse-to-fine localization, which is out of this pass's scope
- **Numerical algorithm changed: NO** (no CUDA kernel or authoritative
  Fortran source was modified this pass; only a new opt-in host-side loader
  was added)
- **Prompt 6B: STILL BLOCKED AT SP1**

## FINAL REPORT (this pass — supersedes the FINAL REPORT above)

- `ptop` ingestion: **PASS**
- `phis`/`zs` ingestion: **PASS**
- Host/device identity: **PARTIAL** (host PASS; device D2H readback not
  performed, no existing mechanism, out of scope to build one this pass)
- Synthetic mode preserved: **YES**
- `c_sw`: **PASS** (round-off, all 6 tiles, owned+ring)
- `update_dz_c` (`ws3`): **FAIL** — improved ~4 orders of magnitude in
  absolute terms, still SIGNIFICANT DIFFERENCE by max_rel on ~half the tiles
- SP1: **FAIL** (SIGNIFICANT DIFFERENCE)
- First remaining divergent operator: `update_dz_c` (field `ws3`), cascading
  into `Riem_Solver_c` (`pkc`,`gz`) and `p_grad_c` (`uc`,`vc`)
- Root-cause classification: **OTHER / NOT FULLY IDENTIFIED** (ptop/phis/zs
  proven a real partial cause, disproven as the complete cause; no
  translation defect proven either)
- Numerical algorithm changed: **NO**
- Prompt 6B: **STILL BLOCKED AT SP1**

SP1 STILL BLOCKED — NUMERICAL DIVERGENCE REMAINS

---

# 13. SP1 `update_dz_c` reconciliation — `dp_ref` root cause found and fixed

Scope: one input-identity pass + one `update_dz_c` reconciliation + one SP1
rerun. Everything in §1-§12 above is preserved; this section supersedes §12's
FINAL REPORT.

## 13.1 Device identity of `phis`/`zs` (and `dp_ref`) — the §12 gap closed

§12 left device-side identity of `phis`/`zs` unverified (host print only). That
gap is now closed by D2H readback, reusing the existing
`FV3_GPU_DEBUG_DUMP_DIR` mechanism rather than building a new capture path:
`FieldId::phis`, `FieldId::zs`, `FieldId::dp_ref` were added to
`debug_check_finite_stage()`'s `targets`/`names` arrays
(`cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`). That probe is gated on
`FV3_GPU_DEBUG_FAIL` and is a no-op in every normal run.

Compared over exactly the region `update_dz_c` consumes `zs` — `is-1..ie+1`,
`js-1..je+1` (`nh_utils.F90:202-204`), i.e. local Fortran `[0,9]^2` — against
the converter's authoritative output:

| field | tiles | bounds | max_abs | max_rel | identity |
|---|---|---|---|---|---|
| `phis` | pe0..pe5 | `[0,9]^2` | 0.0 | 0.0 | **BITWISE** |
| `zs` | pe0..pe5 | `[0,9]^2` | 0.0 | 0.0 | **BITWISE** |
| `dp_ref` | pe0..pe5 | `k=1..10` | 0.0 | 0.0 | **BITWISE** |

Device `zs == phis/grav` exactly (max_abs 0.0, all tiles); device
`|zs|max = 112.7658` m (pe0/1/3/4), `311.8008` m (pe2/5). Device
`bot_ratio = 0.500000000000` on all six tiles.

**Device identity: PASS.** The §12 "PARTIAL" caveat is retired.

## 13.2 Correction to §12's spatial characterization of the `ws3` mismatch

§12 recorded the residual `ws3` mismatch as "round-off in the bulk,
still-diverging at ring-edge cells", and the task premise for this pass
carried that forward as a ring/seam problem on 2 of 6 tiles. **Direct
measurement disproves both.** Over the defined box `[0,9]^2`:

- Cells exceeding `rel > 1e-8`: **98/100** on pe0/pe1/pe3/pe4, **100/100** on
  pe2/pe5 — essentially the entire box, not a ring.
- Worst relative error `6.9269e-02` was **identical on all six tiles**, not
  2 of 6.
- The only cells reaching `max_rel` exactly `1.0` were `(i=5,j=4)` and
  `(i=5,j=5)` on pe0/pe1, where Fortran is **exactly 0.0** and CUDA is
  `2.36847579e-17`. Absolute difference `2.4e-17` — pure round-off, inflated
  to `rel = 1.0` only because `rel = diff/max(|F|,|C|)` divides by an
  exactly-zero scale. This is the same divide-by-near-zero classification
  artifact already noted elsewhere in this document, **not** a divergence, and
  it is what created the false "ring-edge" reading.

The real signature was a **uniform multiplicative bias**: `C/F` over the 592
cells with `|F| > 1e-12` had minimum `0.930730903196` — identical to **12
significant digits on all six tiles** — and
`1 - 0.930730903196 = 0.069269096804` reproduces the observed `max_rel`
exactly. A 12-digit tile-invariant constant is a **scalar** input defect, not
a halo, seam, orientation, or index defect. No seam/orientation/polar/staggering
pattern exists to report.

## 13.3 `update_dz_c` authoritative input contract

Authoritative source: `GFDL_atmos_cubed_sphere` `geos/v3.0.0`,
`ca30e26305fda43831095334c02a3f5e0b8c0c45`, `model/nh_utils.F90:70-207`;
caller `model/dyn_core.F90:597`:

```fortran
call update_dz_c(is, ie, js, je, npz, ng, dt2, flagstruct%dz_min, dp_ref, zs, &
     gridstruct%area, ut, vt, gz, ws3, npx, npy, sw_corner, se_corner, &
     ne_corner, nw_corner, bd, gridstruct%grid_type)
```

Halo/ring depths below are derived from the indexing arithmetic, not from
field names:

| input | Fortran bounds read | CUDA bounds read | staggering | ring req. | CUDA data source | identity proven? |
|---|---|---|---|---|---|---|
| `ut` | `is-1..ie+2`, `js-1..je+1` | same | C-grid x | 2 in i | `c_sw` (WP-A) | indirect — `delpc`/`ptc` round-off |
| `vt` | `is-1..ie+1`, `js-1..je+2` | same | C-grid y | 2 in j | `c_sw` (WP-A) | indirect — as above |
| `gz` (in, as `gz2`) | `is-2..ie+2` via `gz2(i-1,j)` at `i=is1` | same | center | 2 | `gz_from_delz_kernel` (owned) + `H_GZ_FIRST` | structurally matched to Fortran `:456/:468` fill + `i_pack(5)` `:459/:552` |
| `zs` | `is-1..ie+1`, `js-1..je+1` | same | center | **1** | `--sp1-boundary` loader | **YES — BITWISE (§13.1)** |
| `area` | `is-1..ie+1`, `js-1..je+1` | same | center | 1 | `--grid-metrics` (6A.5) | YES (6A.5) |
| `dp0` = `dp_ref` | `1..km` (1-D) | same | — | n/a | **was synthetic** | **NO → root cause (§13.4)** |
| `dt` = `dt2` | scalar | scalar | — | n/a | `--bdt 3600` / `n_split` | YES |
| `dz_min` | scalar | `cfg.dz_min` | — | n/a | harness `2.0` | not re-derived this pass |
| `grid_type`, corners | scalar/flags | `cgrid_vertical_config_fence` | — | n/a | production config | YES |

Note `zs` is read over **one** ring only, so the loader's one-ring coverage is
sufficient for `ws` — the suspected `zs`-ring-coverage mechanism was **not**
the defect. `gz`'s two-ring requirement is met by `H_GZ_FIRST`, not by the
loader, and CUDA's owned-only `gz_from_delz_kernel` launch correctly mirrors
Fortran's owned-only fill plus halo exchange.

## 13.4 Root cause: `dp_ref` (`dp0`) was never authoritative

`dp_ref` is a **direct** `update_dz_c` input. It sets

```fortran
top_ratio = dp0(1 ) / (dp0(1)+dp0(2))            ! nh_utils.F90:96
bot_ratio = dp0(km) / (dp0(km-1)+dp0(km))        ! nh_utils.F90:97
int_ratio = 1./(dp0(k-1)+dp0(k))                 ! nh_utils.F90:118
```

i.e. the `xfx`/`yfx` wind-extrapolation weights at **every** level including
`k=km+1` — the level that produces `gz(km+1)` and therefore `ws3`.

- **Fortran**: `dp_ref(k) = (ak(k+1)-ak(k)) + (bk(k+1)-bk(k))*1.E5`
  (`dyn_core.F90:332`). This fixture's vertical coordinate is **pure sigma**
  with `ps` identically `1.0E5`: all 384 owned `pe` columns (8x8 x 6 tiles) are
  bit-identical, per-`k` spread exactly `0.0`, `pe(km+1) == 100000.0` exactly.
  Because `pe = ak + bk*ps` and `ps == 1.0E5 ==` the literal in Fortran's own
  formula, `dp_ref(k) == pe(k+1)-pe(k)` **exactly** (not a least-squares
  recovery). Result:
  `dp_ref = [10000.0, 9988.888878 x9]`,
  `top_ratio = 0.500277932457`, **`bot_ratio = 0.500000000000`**.
- **CUDA (before this pass)**: `h.dp_ref[k] = delp_col[k]`
  (`run_small_perf_bench.cpp:1382`) — the *synthetic* log-spaced pressure
  column, `dp_ref = [584.9 ... 36904.3]`,
  `top_ratio = 0.386863`, **`bot_ratio = 0.613137`**.

`dp_ref` is in **neither** earlier ingestion milestone's field set — not one of
`grid_metrics_6a5::load()`'s 31+2 static metrics (6A.5 §7 correctly classifies
it as non-metric), and not one of the SP0 loader's 10 dynamical-state fields.
It is the **third** member of the same harness-input-fidelity family as `bdt`,
`ptop`, and `phis`/`zs`.

**Quantitative confirmation.** With `ut(i,j,km-1) = alpha * ut(i,j,km)`,

```
xfx(k=km+1) = ut(km)*(1+b) - ut(km-1)*b   =>   xfx_C/xfx_F = (1+b_C*(1-alpha)) / (1+b_F*(1-alpha))
```

and, since `ws3`'s numerator is linear in `xfx`/`yfx` while its denominator is
`area + O(xfx) ~ area`, `ws3_C/ws3_F` inherits that ratio. Inverting the
measured `0.930730903196` with `b_C = 0.613136820153`, `b_F = 0.5` gives
`alpha = 1.468758579`. The **directly measured** CUDA `ut(km-1)/ut(km)` from
the dump is `1.468729597`, uniform across essentially every cell of every tile
(the SP0 initial condition is vertically separable, which is exactly why the
bias was a tile-invariant constant). Predicted vs. measured agree to 5 decimal
places; the residual is the neglected denominator and `yfx` mixing.

**All `update_dz_c` inputs identical: NO** — `dp_ref` was the first (and only)
input differing beyond round-off. The `update_dz_c` expression itself was
therefore **NOT REACHED** for term-by-term reconciliation, and no kernel
arithmetic was touched.

## 13.5 Fix

Loader-coverage correction only, in the same opt-in `--sp1-boundary` path as
`ptop`/`phis`/`zs`:

- `tools/savepoints/convert_sp1_boundary_to_cuda.py`: new
  `authoritative_dp_ref()` derives `dp_ref(k) = pe(k+1)-pe(k)` from the
  existing SP0 savepoint's owned columns and writes `dp_ref.bin` (+ manifest
  entries incl. `top_ratio`/`bot_ratio`). Both preconditions that make the
  identity exact — global column uniformity and `ps == 1.0E5` — are
  **asserted**, with a nonzero exit and an explicit "ak/bk must be exported
  from Fortran instead" message if either fails, so the script can never emit
  an approximate `dp_ref`.
- `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp`:
  `sp1_boundary_6b::load()` reads `dp_ref.bin` into `h.dp_ref` via the
  existing `read_doubles_into()` (which hard-fails on a short read, giving a
  free size guard) and prints the resulting `top_ratio`/`bot_ratio`. CLI help
  updated.
- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`: `phis`/`zs`/`dp_ref` added to
  the `FV3_GPU_DEBUG_FAIL`-gated diagnostic field list (§13.1).

No new Fortran instrumentation, no new savepoints, no tolerance change, no
clamping, no Fortran output copied into CUDA (`pe` is an SP0 **input** already
loaded verbatim), no halo semantics bypassed.
**Numerical algorithm changed: NO.**

## 13.6 SP1 rerun result

```
FV3_GPU_DEBUG_FAIL=1 FV3_GPU_DEBUG_DUMP_DIR=tools/savepoints/sp1_cuda_dump_dpref \
FV3_GPU_DEBUG_DUMP_STAGE=cgrid_pressure \
FV3_GPU_DEBUG_FIELDS=ws3,gz,pkc,uc,vc,delpc,ptc,phis,zs,dp_ref,ut,vt,xfx,yfx \
./build_cuda_r8_perf/fv3_gpu_perf_bench --grid 8 --npz 10 --n_split 3 \
  --warmup 0 --iters 1 --repeats 1 --mode resident --hardware A100 --bdt 3600 \
  --grid-metrics tools/grid_metrics/c8_authoritative \
  --sp0-state tools/savepoints/sp0_cuda_authoritative \
  --sp1-boundary tools/savepoints/sp1_boundary_c8
python3 tools/savepoints/compare_cuda_fortran.py <fortran_sp1_dir> \
  tools/savepoints/sp1_cuda_dump_dpref sp1 1 cgrid_pressure
```

| field | §12 (after ptop/phis/zs) | this pass (after `dp_ref`) | verdict |
|---|---|---|---|
| `delpc` | round-off 6.9e-15 | unchanged | ROUND-OFF SCALE |
| `ptc` | round-off 1.1e-14 | unchanged | ROUND-OFF SCALE |
| `ws3` | max_abs **1.32e-05**, uniform 6.93% bias | max_abs **4.74e-17** (pe0/1/3/4), **9.47e-17** (pe2/5) | **round-off floor — PASS** |
| `uc` | max_abs 8.0e-5 | 7.996e-05 (unchanged) | SIGNIFICANT DIFFERENCE |
| `vc` | max_abs 3.1e-4 | 3.141e-04 (unchanged) | SIGNIFICANT DIFFERENCE |
| `pkc` | max_rel ~0.080 | 8.01e-02 / 8.11e-02 (unchanged) | SIGNIFICANT DIFFERENCE |
| `gz` | max_rel ~0.96 | 9.606e-01, max_abs 4.06e+05 (unchanged) | SIGNIFICANT DIFFERENCE |

`ws3` improved by **~2.8e11x** in absolute terms (11 orders of magnitude).

`ws3` PASS justification — the residual is at the cancellation round-off
floor, not merely small. `ws3 = (zs - gz(km+1))*rdt` is the difference of two
nearly equal quantities of magnitude `|zs|`, so its representable absolute
floor is `eps*|zs|/dt2`:

| tiles | `\|zs\|max` | floor `eps*\|zs\|/600` | measured max_abs | ratio |
|---|---|---|---|---|
| pe0/1/3/4 | 112.77 m | 4.17e-17 | 4.737e-17 | 1.14x |
| pe2/5 | 311.80 m | 1.15e-16 | 9.474e-17 | **0.82x (below floor)** |

Field scale is `max\|F\| = 4.73e-03` (pe0/1) / `8.50e-02` (pe2/3/4/5), so the
error relative to field scale is `~1e-14`. `compare_cuda_fortran.py` labels
`ws3` SMALL NUMERICAL DIFFERENCE only because its `max_rel` (2.4e-9 worst,
1.4e-12..2.2e-11 restricted to `|F| > 1e-6`) divides by individual near-zero
cell values — the same small-denominator artifact documented in §13.2, not a
real residual. **`update_dz_c` PASS.**

Honest scope limit on that claim: `update_dz_c`'s only **separately
observable** output at SP1 is `ws3`, because `Riem_Solver_c` subsequently
overwrites `gz` in full (`gz(i,j,km+1) = hs`, then
`gz(i,j,k) = gz(i,j,k+1) - dz2(i,k)*grav`, `nh_utils.F90:421-428`). So this
validates `update_dz_c`'s surface-vertical-velocity output at round-off; it
does **not** independently validate its interior `gz` levels, and no such
claim is made.

## 13.7 New first divergent operator

With `c_sw` (`delpc`/`ptc`) and `update_dz_c` (`ws3`) both at round-off, the
first remaining divergent fields are **`gz` and `pkc`**, both written by
**`Riem_Solver_c`** (`nh_utils.F90:325-432`, called at `dyn_core.F90:583`).
`gz`'s worst error sits at `k=0` — the **top** interface — with
`max_abs = 4.06e+05` against a column-top `grav*height` of order `3.8e+05`,
i.e. the CUDA column depth disagrees with Fortran's over essentially the whole
column. `uc`/`vc` are `p_grad_c` outputs consuming `gz`/`pkc` and are not
independent evidence.

Per this pass's scope (`no Riem_Solver_c investigation unless ws3 passes`, and
one reconciliation per pass), `Riem_Solver_c` was **not** investigated.
`ws3` now passes, so it is the correct next target. First thing to check there,
by analogy with the three input-fidelity gaps already found: whether any
`Riem_Solver_c` input is still synthetic (its `w3`/`omga`, `q_con`/`cappa`
thermodynamic path, `a_imp`/`p_fac`/`scale_z` scalars) **before** suspecting
kernel arithmetic. Note `pkc(:,:,1) == ptop` is already correct, so the top
*pressure* boundary is not the gap; the top *height* is.

SP2/SP3: **NOT RUN.**

## 13.8 Regression

- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` (shared runtime) was touched, so
  R8 `ctest` was run: **22/22 PASS**.
- Default synthetic path (no `--grid-metrics`/`--sp0-state`/`--sp1-boundary`):
  `=== 3A.11 CUDA perf benchmark: PASS ===`, exit 0 — proves the benchmark does
  not depend on the authoritative files.
- The validation run's own end-of-run harness verdict is `FAIL` (tile3/4/5 NaN
  at `it`>1). This is the **same** pre-existing downstream SP2/SP3 divergence
  recorded in §12 for the previous run, is expected while `Riem_Solver_c` is
  still wrong, and was **not** investigated (out of scope).
- No sanitizer matrix run (no kernel or halo code changed).

## FINAL REPORT (this pass — supersedes §12's)

- `phis`/`zs` device identity: **PASS** (BITWISE, all 6 tiles, over
  `update_dz_c`'s exact consumed region; `dp_ref` also BITWISE)
- Failing tiles: **all 6** (not 2 — §12's characterization corrected)
- Failing ring locations: **none specifically** — 98-100 of 100 cells in the
  defined box on every tile; the two `rel=1.0` cells were a
  `2.4e-17`-absolute divide-by-exact-zero artifact
- Common seam/orientation pattern: **NONE** — signature was a tile-invariant
  multiplicative constant `0.930730903196` (12 digits), i.e. a scalar defect
- First mismatching `update_dz_c` input: **`dp_ref`** (`dp0`) —
  `bot_ratio` 0.613137 vs authoritative 0.500000000000
- All `update_dz_c` inputs identical: **NO**
- Fortran/CUDA `update_dz_c` expression: **NOT REACHED** (input defect found
  first; kernel arithmetic untouched)
- Root cause: `dp_ref` was seeded from the harness's synthetic log-spaced
  pressure column and belonged to neither the grid-metric nor the SP0 field
  set; third member of the `bdt`/`ptop`/`phis`-`zs` input-fidelity family
- Fix applied: authoritative `dp_ref` derived exactly from SP0 `pe` (`ps` and
  column uniformity asserted) and loaded via the existing opt-in
  `--sp1-boundary` path; `phis`/`zs`/`dp_ref` added to the debug-gated D2H
  probe
- `ws3` after fix: **PASS** (max_abs 4.74e-17, at/below the cancellation
  round-off floor; 11 orders of magnitude better)
- SP1 after fix: **FAIL** (`gz`/`pkc` still SIGNIFICANT DIFFERENCE)
- New first divergent operator: **`Riem_Solver_c`** (`gz`, `pkc`)
- Numerical algorithm changed: **NO**
- Prompt 6B: **READY FOR NEXT SP1 OPERATOR** (`Riem_Solver_c`)

UPDATE_DZ_C PASS — CONTINUE SP1 RECONCILIATION

---

## 14. `Riem_Solver_c` reconciliation (this pass) — SUPERSEDES §13's FINAL REPORT

Bounded SP1 reconciliation restricted to `Riem_Solver_c`, entered with
`gz`/`pkc` SIGNIFICANT DIFFERENCE per §13.6.

**Result: `Riem_Solver_c` PASS.** Root cause was a fourth member of the
harness input-fidelity family: `cfg.rdgas = 287.0` versus the authoritative
FMS/GEOS `rdgas = 8314.47/28.965 = 287.0523045054`. Both `gz` and `pkc` now
classify ROUND-OFF SCALE on all 6 tiles. Two of the three divergences §13
inherited (`uc`/`vc`, "max_rel 1.7-2.0") turned out never to have been
divergences at all — they were artifacts of the comparison region and the
comparison metric. Numerical algorithm changed: **NO**.

### 14.1 Two inherited verdicts overturned before any solver work

Both were re-derived from the §13-era dump (`sp1_cuda_dump_dpref`) with no
code change, i.e. these corrections apply retroactively to §12/§13's numbers.

**(a) `gz`/`pkc`'s "SIGNIFICANT DIFFERENCE ~0.96 / ~0.08" was entirely four
cells.** `Riem_Solver_c` loops `i=is-1..ie+1`, `j=js-1..je+1`, which includes
the four *diagonal* corner halo cells (local Fortran i,j ∈ {0,9}×{0,9} for
c8). Those cells have no donor tile on the cubed sphere: `mpp_update_domains`
does not fill them and the authoritative grid metrics carry a ±1.0e8 `area`
sentinel there (`milestone_6a5_authoritative_grid_metrics.md` §8). Both
implementations compute them from meaningless inputs, so comparing them
compares two undefined regions. Splitting them out:

| field | worst non-corner max_rel | worst corner max_rel |
|-------|--------------------------|----------------------|
| `gz`  | 2.0662e-04 (pe0/1/3/4), 3.3087e-04 (pe2/5) | 9.606e-01 |
| `pkc` | 1.3776e-06 – 1.3864e-06 | 8.01e-02 – 8.11e-02 |

The real signal was 2e-04, not 0.96 — three orders of magnitude smaller, and a
completely different diagnostic target.

**(b) `uc`/`vc`'s "max_rel 1.7-2.0" was cancellation noise.** `rel =
|F-C|/max(|F|,|C|)` fabricates rel≈1 (or ≈2 on a sign flip) where both sides
have cancelled to ~0. `uc` pe3 (5,6): Fortran O(−7e-16), CUDA O(+2e-15),
max_abs **6.108e-15**. `vc` pe0 (8,5): Fortran O(2.7e-14), CUDA O(−1.8e-14),
max_abs **4.4895e-14**. The largest *real* `uc` signal (pe0 (1,4), |F|max
2.3928) had max_abs 1.1061e-05. This is the third localization this metric
artifact has corrupted; see §14.5 for the comparator fix.

### 14.2 Input contract, verified before touching solver arithmetic

`Riem_Solver_c` (`nh_utils.F90:325-434`) consumes: `ms, dt(=dt2), akap,
cappa, cp, ptop, hs(=phis), w3(=omga, c_sw's `wc`), pt(=ptc), q_con,
delp(=delpc), gz(inout), pef(out, =pkc), ws(=ws3), p_fac, a_imp, scale_m`,
plus module constants `grav`, `rdgas` from `constants_mod` (`:53`).

Verified identical: `delpc`/`ptc` (ROUND-OFF SCALE, already validated §7),
`ws3` (§13.6), `phis`/`dp_ref` (BITWISE, §13.1), `ptop` = 100.0,
`grav` = 9.80665, `akap` = 2/7, `a_imp` = 1.0, `p_fac` = 0.05,
`hydrostatic` = false, `grid_type` = 0, `dt2` = 600 (with `--bdt 3600`).
`scale_m`/`ms` are dead for `a_imp > 0.5`; `cp` is unused in this branch.

**Vertical bounds/interface reconciliation (Task §3).** Loaded Fortran
bounds, confirmed against the CUDA flat box by `gather()`:

| field | Fortran bounds | kind |
|-------|----------------|------|
| `gz`, `pkc` | `(-2:11, -2:11, 1:11)` | interface, npz+1 |
| `delpc`, `ptc` | `(-2:11, -2:11, 1:10)` | layer, npz |
| `ws3` | `(-2:11, -2:11)` | rank-2 surface |
| `uc` | `(-2:12, -2:11, 1:10)` | i-staggered layer |
| `vc` | `(-2:11, -2:12, 1:10)` | j-staggered layer |

Mapping `k_fortran = k_cuda + 1`, i.e. Fortran `1..11` ↔ CUDA `0..10` for the
interface fields. This is verified, not assumed: all 11 interfaces of `gz`
agree to ≤4.4e-11 out of 4.5e5 simultaneously (below), which an off-by-one
could not produce. `pkc(:,:,1) = ptop = 100.0` exactly and
`gz(:,:,11) = hs = phis` bit-identically anchor the two ends. SP1 has no
`(i,k,j)` field, unlike SP0's `pe`/`peln`.

### 14.3 Root cause: `rdgas`

The `gz` divergence was reduced to a single scalar by reconstructing `pem`
from the already-validated `delpc`, isolating `pe2 = pkc − pem` and
`dz = Δgz/grav`, and measuring the per-layer bias on pe0 local (i=4, j=9) —
the largest-`gz`-discrepancy column:

```
dz rel (C-F)/F, k=1..10:
 -1.7966e-4 -1.8135e-4 -1.8153e-4 -1.8163e-4 -1.8171e-4
 -1.8176e-4 -1.8181e-4 -1.8185e-4 -1.8189e-4 -1.8192e-4    mean -1.815108e-04
```

Layer-independent ⇒ a scalar, not a stencil or index defect. `SIM1_solver`'s
pressure closure is `pe2 = exp(gama*log(-dm/dz2*rgas*pt2)) - pm2`; for fixed
p and T, ρ ∝ 1/R, so `dz ∝ R` and a too-small `rgas` forces a proportionally
too-small `|dz|`. Predicted bias from `rdgas`:

```
(287.0 - 287.0523045054)/287.0523045054 = -1.822125e-04   <- matches
287.04 (GFDL variant): -1.3934e-04   (23% off, excluded)
287.05 (GFS variant):  -1.7418e-04   (4.3% off, excluded)
```

Provenance traced through three levels rather than guessed: `nh_utils.F90:53`
takes `rdgas` from FMS `constants_mod`; the fixture's
`build/CMakeCache.txt:639` points `FMS_DIR` at
`ESMA-Baselibs-8.32.0/.../FMS`; that Baselibs FMS is built with
`-DCONSTANTS=GEOS` (`ESMA-Baselibs-8.32.0/src/GNUmakefile:1003`), selecting
`constants/geos_constants.fh:44`, `RDGAS = 8314.47 /28.965`. The GFDL and GFS
constant sets are therefore not what this fixture compiled — excluded by build
evidence *and* numerically above.

Classification: **INPUT MISMATCH / CONFIGURATION** (harness scalar). Not a
CUDA kernel defect. Fourth instance of the input-fidelity family after `bdt`,
`ptop`/`phis`-`zs`, and `dp_ref`.

### 14.4 Fix

`cuda/dyn_core/benchmarks/run_small_perf_bench.cpp`, three edits, host-side
only:

1. New `static const double kFv3Rdgas = 8314.47 / 28.965;` with the full
   provenance chain in a comment. Written as the quotient, not a rounded
   decimal, so it reproduces the Fortran parameter expression exactly.
2. The synthetic balanced-column IC derivation: `const double kRdgas = 287.0`
   → `kFv3Rdgas`.
3. `production_config()`: `cfg.rdgas = 287.0` → `kFv3Rdgas`.

Edits 2 and 3 move together, so the default synthetic path's balanced-column
invariant stays self-consistent; its absolute numbers shift by 1.82e-4.
`test_dispatch_smoke.cpp:899` and `test_wp_d_contract.cpp:66` keep their own
`287.0` literals — self-contained tests with their own tolerances, not
compared against Fortran.

**Numerical algorithm changed: NO.**

### 14.5 Comparator metric fix (Task §9)

`tools/savepoints/compare_cuda_fortran.py`:

- Cubed-sphere diagonal corner cells reported on their own `corner` line,
  excluded from the verdict (§14.1a). `CORNER_CELLS`, `corner_mask()`, and a
  `mask=` argument to `classify()` that preserves array shape so `argmax`
  stays reportable as (i,j,k).
- Scale-aware denominator: `den = max(|F|, |C|, ref)` where
  `ref = max|F|` over the compared region. Every difference is measured
  against a magnitude something in the field actually has — the cell's own
  where it carries signal, the field's dynamic range where it has cancelled.
  `max_rel_raw` (unfloored per-cell) and `max_abs` are always printed, so the
  guard changes which number is *ranked*, never which numbers are *shown*.
  The BITWISE / 1e-13 / 1e-8 thresholds are untouched.

An intermediate revision floored at `1e-8*ref` instead of `ref`. That was
wrong in the opposite direction — 1e-8 of the field range is not a magnitude
anything in the field has, so dividing by it inflated round-off by 1e8 and
manufactured the very artifact the guard removes. Recorded because the two
floors give opposite verdicts on SP1's `uc`/`vc`.

**The corrected metric discriminates rather than whitewashes**: re-run
unchanged on the pre-fix `sp1_cuda_dump_dpref`, it still ranks
`gz`/`pkc`/`uc`/`vc` SIGNIFICANT DIFFERENCE and reproduces §14.1's
hand-computed corner split exactly. Only the post-fix dump passes.

### 14.6 SP1 rerun result

`sp1_cuda_dump_rdgas`, SP1 `it=1 cgrid_pressure`, all 6 tiles, defined box,
corners excluded:

| field | max_abs | max_rel | field range | verdict |
|-------|---------|---------|-------------|---------|
| `delpc` | 1.82e-12 | 1.82e-16 | 1.0e+04 | ROUND-OFF SCALE |
| `ptc` | 3.55e-15 | 2.52e-16 | 2.48e+01 | ROUND-OFF SCALE |
| `gz` | ≤2.91e-10 | ≤6.34e-16 | 4.59e+05 | **ROUND-OFF SCALE** |
| `pkc` | ≤2.91e-11 | ≤2.91e-16 | 1.00e+05 | **ROUND-OFF SCALE** |
| `uc` | ≤1.03e-13 | ≤6.44e-15 | 3.20e+01 | ROUND-OFF SCALE |
| `vc` | ≤9.92e-14 | ≤6.89e-15 | 1.44e+01 | ROUND-OFF SCALE |
| `ws3` | ≤9.47e-17 | ≤4.97e-13 | 1.91e-04 | SMALL NUMERICAL DIFFERENCE |

`gz` went from 2.07e-04/3.31e-04 to ≤6.3e-16. The representative column that
carried the constant −1.8e-4 bias (pe0 local i=4, j=9), all 11 interfaces:

```
 k_f   gz_fortran         gz_cuda            diff
   1   4.518100684e+05   4.518100684e+05   0.000e+00
   2   1.575517449e+05   1.575517449e+05   2.910e-11
   3   1.142527477e+05   1.142527477e+05   4.366e-11
   4   8.699640775e+04   8.699640775e+04   2.910e-11
   5   6.681629257e+04   6.681629257e+04   2.910e-11
   6   5.074431209e+04   5.074431209e+04   7.276e-12
   7   3.738968632e+04   3.738968632e+04   1.455e-11
   8   2.596929585e+04   2.596929585e+04   1.819e-11
   9   1.598552750e+04   1.598552750e+04   0.000e+00
  10   7.093344947e+03   7.093344947e+03   0.000e+00
  11  -9.630769043e+02  -9.630769043e+02   0.000e+00
```

`ws3` is the sole residual above round-off, at 4.97e-13 relative. Its max_abs
of 9.47e-17 is far below its own cancellation floor (`ws3` reaches only
1.9e-04 through cancellation of much larger terms), so this is consistent with
input round-off rather than an operator defect — but it is reported as SMALL
NUMERICAL DIFFERENCE, not claimed as round-off.

**Solver arithmetic was never inspected** — the input defect was found first,
per the task's ordering. §5's step-by-step trace was not entered.

### 14.7 Operator status after this pass

`c_sw`, `update_dz_c`, `Riem_Solver_c`, and `p_grad_c` all now agree at
ROUND-OFF SCALE at SP1/`it=1`. `p_grad_c` needed no fix: its apparent
divergence was §14.1b's metric artifact, and its inputs (`gz`, `pkc`) became
correct with `rdgas`. **SP1 `it=1` has no remaining significantly-divergent
field.**

### 14.8 Regression

- R8 `ctest`: **22/22 PASS** (required — `cfg.rdgas` feeds shared kernel code).
- Default synthetic path (no authoritative flags): **PASS**, 0 NaN.
- The benchmark run as a whole reports FAIL, from **pre-existing** end-of-step
  NaN on tiles 3/4/5 originating at `it=3`. Verified pre-existing, not caused
  by this change: the nonfinite counts are bit-identical between the pre-fix
  and post-fix dumps —
  `it=1` none, `it=2` none, `it=3` `{pe2: gz/pkc/uc/vc 20 each, pe3: 20 each,
  pe4: gz/pkc 20, uc/vc 10}` in **both**. `it=1` and `it=2` are clean in both.
  This is a genuine open defect at `it≥3`, outside this pass's SP1/`it=1`
  scope; see §14.9.

### 14.9 Next steps

1. **`it=3` nonfinite (new, highest priority).** SP1 now passes at `it=1`, so
   the substep-3 NaN on tiles 2/3/4 is the next blocker on the path to Gate
   C2. Same coarse-to-fine discipline; note it is absent at `it=1`/`it=2`, so
   it is a substep-accumulation or halo-staleness signature rather than a
   scalar. Compare SP1 at `it=2` first (already dumped, clean) to bound it.
2. SP2 (`dgrid_dynamics`), then SP3 — not started, per scope.
3. Gate C2 remains **BLOCKED**.

Reproduction: `sp1rerun_rdgas.sh` (uncommitted scratch, repo root). Dumps:
`tools/savepoints/sp1_cuda_dump_rdgas/` (post-fix),
`tools/savepoints/sp1_cuda_dump_dpref/` (pre-fix, retained for the §14.5
discrimination check).

## FINAL REPORT (this pass — supersedes §13's)

- Complete `Riem_Solver_c` input contract: established from
  `nh_utils.F90:325-434` (§14.2), including the `grav`/`rdgas`
  `constants_mod` dependency the earlier passes' candidate list omitted
- Vertical bounds/interfaces: **RECONCILED** (§14.2); `k_fortran = k_cuda+1`
  for the npz+1 interface fields, verified by simultaneous 11-level agreement
- All inputs identical before the fix: **NO** — first proven mismatch
  **`rdgas`** (287.0 vs 8314.47/28.965 = 287.0523045054)
- Classification: **INPUT MISMATCH / CONFIGURATION** (harness scalar; not a
  CUDA translation defect, not a bound/index defect)
- Solver arithmetic reconciliation: **NOT REACHED** (input defect found
  first, per task ordering; kernel untouched)
- Fix applied: authoritative `rdgas` as `8314.47/28.965` in three places in
  `run_small_perf_bench.cpp`, with the `-DCONSTANTS=GEOS` provenance chain
  documented in source
- Two inherited verdicts **overturned**: `gz`/`pkc`'s 0.96/0.08 was four
  undefined cubed-sphere diagonal corner cells; `uc`/`vc`'s 1.7-2.0 was
  cancellation noise at 1e-14 absolute
- Comparator: corner cells split out, denominator floored at the field's own
  range; verified to still rank the pre-fix dump SIGNIFICANT
- `gz` after fix: **PASS** (≤6.3e-16, from 3.3e-04)
- `pkc` after fix: **PASS** (≤2.9e-16)
- `uc`/`vc` after fix: **PASS** (≤6.9e-15) — `p_grad_c` needed no fix
- `ws3` after fix: SMALL NUMERICAL DIFFERENCE (4.97e-13; max_abs 9.5e-17,
  below its cancellation floor)
- SP1 `it=1`: **no remaining significantly-divergent field**
- Regression: R8 ctest **22/22 PASS**; default synthetic path PASS
- Pre-existing `it=3` nonfinite on tiles 2/3/4: bit-identical before and
  after this change; new next blocker
- Numerical algorithm changed: **NO**
- SP2/SP3: **NOT RUN**. Gate C2: **BLOCKED**

RIEM_SOLVER_C PASS — CONTINUE SP1 RECONCILIATION

---

# §15 — SP1 `it=3` NONFINITE investigation: E11 mapped, E11 EXONERATED, first divergence relocated to `it=2`

**SUPERSEDES §14's "new next blocker" framing.** §14 recorded the `it=3`
NONFINITE state as the next blocker and named E11 the strongest lead. This
pass mapped E11 authoritatively and then **disproved that lead on two
independent grounds**, and relocated the first divergence from `it=3` to
`it=2`. No source file was changed this pass; no GPU job was required.

## 15.1 E11 authoritative mapping (§1)

Read directly from the fixture's instrumented
`@fvdycore/model/dyn_core.F90`. **Line-number correction**: `CURRENT_STATE.md`
§10 and the CUDA in-source comments both cite `dyn_core.F90:1094-1095`.
`grep -n "mpp_get_boundary" dyn_core.F90` returns exactly 3 hits — `:86`
(doc comment), `:108` (`use mpp_domains_mod`), and **`:1142`, the sole call
site**. The write-back is `:1145-1152`. (`:1094-1095` vs `:1142` is consistent
with the ~20-line offset introduced by this fixture's own savepoint
instrumentation; the CUDA comments were written against the uninstrumented
file.)

Verbatim:

```fortran
1140    if( it==n_split .and. gridstruct%grid_type<4 .and. .not. gridstruct%nested) then
1141 ! Prevent accumulation of rounding errors at overlapped domain edges:
1142       call mpp_get_boundary(u, v, domain, ebuffery=ebuffer,  &
1143                             nbufferx=nbuffer, gridtype=DGRID_NE )
1145          do k=1,npz
1146             do i=is,ie
1147                u(i,je+1,k) = nbuffer(i-is+1,k)
1148             enddo
1149             do j=js,je
1150                v(ie+1,j,k) = ebuffer(j-js+1,k)
1151             enddo
1152          enddo
1153    endif
1155 #ifndef ROT3
1156    if ( it/=n_split)   &
1157         call start_group_halo_update(i_pack(8), u, v, domain, gridtype=DGRID_NE)
1158 #endif
```

| Fortran E11 operation | CUDA equivalent | Implemented? | Correct timing? | Correct fields? | Correct seam transform? |
|---|---|---|---|---|---|
| `mpp_get_boundary(u, v, gridtype=DGRID_NE)`, `:1142` | none | **NO** | n/a | n/a | n/a |
| write-back `u(is:ie, je+1, k)` (north u seam row), `k=1..npz`, `:1146-1148` | none | **NO** | n/a | n/a | n/a |
| write-back `v(ie+1, js:je, k)` (east v seam column), `k=1..npz`, `:1149-1151` | none | **NO** | n/a | n/a | n/a |
| gate `it==n_split .and. grid_type<4 .and. .not. nested` (ACTIVE in production: `grid_type=0`, not nested) | none | **NO** | n/a | n/a | n/a |
| mutually-exclusive partner `start_group_halo_update(i_pack(8), u, v)` when `it/=n_split`, `:1156-1157` | `FV3_GPU_H_DWIND_IN`, resolved **every** substep (`fv3_gpu_runtime.cpp:1727`, comment `:402 -> :504` = `:422 -> :524` instrumented) | YES | YES | YES | YES (existing rotated-seam algebra) |

Semantics: E11 exchanges **`u` and `v` only**, over the **h=0 seam lines**
(the last staggered line, `j=je+1` for `u`, `i=ie+1` for `v`), for `k=1..npz`.
Both adjacent tiles compute these lines redundantly; E11 overwrites the local
copy with the neighbour's so the two agree bitwise. Stated purpose is
round-off hygiene, not a numerical correction.

## 15.2 CUDA E11 status: MISSING (§2) — confirmed from current code, not history

Per the directive, established by inspecting the current tree, not by trusting
milestone history:

- `cuda/dyn_core/include/fv3_gpu_communication.h:7-12` — the halo-group enum
  has 11 members (`H_DWIND_IN` … `H_DWIND_NEXT`); **none** corresponds to a
  staggered ring-0 seam reconciliation.
- `grep -n "resolve(FV3_GPU_H_"` over `fv3_gpu_runtime.cpp` returns 9 sites
  (`:1727, :1728, :1753, :1763, :1770, :1775, :1778, :1783, :1784, :1817`).
  There is **no `it==n_split`-gated resolve of any kind**; the only
  substep-gated resolves are the two `it==1` ones (`H_DSCALAR`, `H_GZ_FIRST`).
- `grep -rni "get_boundary\|ebuffer\|nbuffer"` over `cuda/` matches **only
  comments**, which state the position outright:
  `fv3_gpu_halo_exchange.cu:51` — "That is exchange E11 in the §1 map; it has
  no enum group and remains …".

**Verdict: E11 is MISSING in CUDA.** (Unchanged from `CURRENT_STATE.md` §10 —
but now verified against current source rather than inherited.)

**Documentation defect found and now provably characterized.** The E11
comments at `fv3_gpu_halo_exchange.cu:48-51` and
`fv3_gpu_halo_topology.hpp:293-296` both claim `mpp_get_boundary` "returns
into 1-D buffers (ebuffer/nbuffer) and **not into the field**" / "never into
the field array itself". The verbatim Fortran above shows this is **wrong**:
`:1145-1152` copies both buffers straight back into `u` and `v`. The buffers
are a staging detail, not the endpoint. This is a third inaccuracy in the
already-recorded halo-documentation defect at those two sites
(`CURRENT_STATE.md` §10, last bullet). Not fixed this pass (comment-only, and
outside this pass's bounded scope); recorded here with the verbatim
counter-evidence so the next toucher does not re-derive it.

## 15.3 E11 is EXONERATED as the cause of the SP1 `it=3` failure

Two independent decisive arguments. Either alone is sufficient.

**(a) Timing — E11 runs *after* every SP1 and SP2 capture in the same substep.**
Savepoint call sites in the same file:

| savepoint | line | position |
|---|---|---|
| SP0 | `:299-308` | dyn_core entry, before the loop |
| loop head | `:419` | `do it=1,n_split` |
| `c_sw` | `:536` | |
| `p_grad_c` | `:634` | |
| **SP1** | **`:637-643`** | immediately after `p_grad_c` |
| `d_sw` | `:823` | |
| **SP2** | **`:871-880`** | after `d_sw` |
| **E11** | **`:1140-1153`** | ~500 lines after SP1, ~270 after SP2 |
| SP3 | `:1312` | after the loop |

At `it=3 (== n_split)` the SP1 sample is taken at `:637`, roughly 500 lines
**before** E11 executes at `:1142`. E11 therefore **cannot influence any SP1
sample at any `it`, including `it=3`**. Its earliest observable effect is
**SP3**. (It cannot affect SP2 either.)

**(b) Data — the first divergence is at `it=2`, where E11 never executes at all.**
E11 is gated `it==n_split`, so it does not run during `it=1` or `it=2`. The
measurement below shows the divergence is already total at `it=2`. A mechanism
that never fires cannot explain a divergence that precedes it.

## 15.4 First divergence relocated: SP1 `it=2`, not `it=3` (§3)

`tools/savepoints/compare_cuda_fortran.py` against
`tools/savepoints/sp1_cuda_dump_rdgas` (the dumps already on disk from §14's
pass — the current code state; no rerun needed, and none would differ, since
no source changed this pass) and the authoritative
`c8_L10_savepoint_6a_run1/savepoints`. `sp1_it1`, `sp1_it2`, `sp1_it3` are all
present on the Fortran side, so a real authoritative `it=2`/`it=3` reference
exists and this is not blocked on new Fortran data.

Per-field worst verdict over the **defined box** (corner-excluded), all 6 tiles:

| field | it=1 | it=2 | it=3 |
|---|---|---|---|
| `delpc` | ROUND-OFF SCALE | **SIGNIFICANT** | SIGNIFICANT |
| `ptc`   | ROUND-OFF SCALE | **SIGNIFICANT** | SIGNIFICANT |
| `uc`    | ROUND-OFF SCALE | **SIGNIFICANT** | NONFINITE |
| `vc`    | ROUND-OFF SCALE | **SIGNIFICANT** | NONFINITE |
| `pkc`   | ROUND-OFF SCALE | **SIGNIFICANT** | NONFINITE |
| `gz`    | ROUND-OFF SCALE | **SIGNIFICANT** | NONFINITE |
| `ws3`   | SMALL NUM DIFF  | **SIGNIFICANT** | SIGNIFICANT |
| OVERALL | SMALL NUM DIFF  | **SIGNIFICANT DIFFERENCE** | NONFINITE |

At `it=2` the divergence is **not marginal and not localized**: every field
differs, and the mismatch count is the full compared population
(`delpc` 960/960, `ptc` 958-960/960, `uc`/`vc` 720/720). `it=3` is the same
state grown by roughly 2-3x (e.g. pe5 `delpc` max_rel 3.52e-3 → 9.91e-3;
pe5 `uc` 6.97e-2 → 1.09e-1) until `sim1_column`'s
`log(pem[k+1]/pem[k])` takes a non-positive argument.

**The `it=3` NONFINITE state is a terminal symptom, not the defect.**

### Per-tile `it=2` magnitudes (authoritative comparator numbers)

| tile | `delpc` max_abs (Pa) | `delpc` max_rel | `ptc` max_rel |
|---|---|---|---|
| pe0 | 1.83e+02 | 1.79e-02 | 1.46e-04 |
| pe1 | 3.13e+01 | 3.11e-03 | 1.27e-05 |
| pe2 | 2.10e+03 | **1.85e-01** | 1.05e-03 |
| pe3 | 2.23e+03 | **1.91e-01** | 3.97e-04 |
| pe4 | 3.30e+03 | **2.49e-01** | 6.27e-04 |
| pe5 | 3.54e+01 | 3.52e-03 | 2.72e-05 |

Two structural observations, both from data:

1. **Strong tile asymmetry**: pe2/pe3/pe4 are 50-80x worse than pe1/pe5. This
   distinguishes the present defect from all five previously-found
   input-fidelity defects (`bdt`, `ptop`, `phis`/`zs`, `dp_ref`, `rdgas`),
   every one of which was a scalar or a tile-invariant profile and therefore
   hit all six tiles near-identically. A 50-80x tile spread points instead at
   a **seam-, orientation-, or tile-dependent** defect (or a tile-dependent
   input), and is the single most useful discriminator handed to the next
   pass.
2. It **predicts which tiles fail first**: §14 recorded the `it=3` NaN on
   pe2/pe3/pe4 and the `delp` positivity failure on tiles 3/4/5 (1-based =
   pe2/pe3/pe4). Those are exactly the three worst `it=2` tiles. The NaN
   is the growth of this divergence, on the tiles where it is largest.

Vertical structure of the `it=2` `delpc` error (max over all 6 tiles, per k,
normalized by the field's own range) is smooth and top-heavy — k=0 3.05e-1,
rising slightly to k=2 3.30e-1, then decaying monotonically to k=9 3.89e-2.
`ptc`'s profile is different in shape (peaks mid-column at k=6, 1.05e-3).
Recorded as observation only; **no operator attribution is claimed from it**
(see §15.6).

## 15.5 Consequent localization: the defect is in the `it=1` path *after* SP1

This follows deductively from two measured facts, with no additional
assumption:

- SP1 `it=1` is **NUMERICALLY VALIDATED** (§14: `c_sw`, `update_dz_c`,
  `Riem_Solver_c`, `p_grad_c` all ROUND-OFF SCALE from authoritative SP0
  state). So the C-grid predictor is correct **given correct inputs**.
- SP1 `it=2` is fully divergent. `c_sw` is the first operator of `it=2` and
  its output `delpc`/`ptc` already differ in 960/960 cells. Since the operator
  itself is validated, its **inputs** at `it=2` — `delp`, `pt`, `u`, `v`, `w`
  as advanced by the remainder of substep `it=1` — must already be wrong.

Therefore the first divergence lies in `dyn_core.F90` between the SP1 capture
at `:637` and the end of substep `it=1` — i.e. the D-grid stage (`d_sw` at
`:823`), the nonhydrostatic vertical solver, the D-grid pressure gradient, and
the `u`/`v`/`delp`/`pt`/`w`/`delz` updates.

**That region is exactly what SP2 measures, and this pass was directed not to
inspect SP2.** So localization stops here, at the scope boundary, rather than
continuing past it. The next pass needs explicit authorization to compare
**SP2 at `it=1`** — which is now the highest-value single measurement in the
project, because `it=1`'s C-grid inputs are proven authoritative, so any SP2
`it=1` divergence is attributable to the D-grid stage alone with no upstream
ambiguity.

## 15.6 Why E11 was NOT implemented this pass

The directive's §5 authorizes implementing E11 if it is "confirmed missing or
incorrect". It **is** confirmed missing (§15.2). It was deliberately **not**
implemented, for reasons recorded explicitly rather than silently:

- It is **proven incapable** of affecting SP1 or SP2 (§15.3a) — so
  implementing it now would produce **no measurable change** in any currently
  available validation signal, and could not be verified.
- The actual defect is proven to be elsewhere (§15.5). Implementing E11 now
  would add an unvalidated change on top of an open, larger divergence,
  making the next bisection harder rather than easier.
- Doing it correctly requires a genuinely new halo capability: a staggered
  **ring h=0** seam exchange. `CURRENT_STATE.md` §10 records ring-0 as
  "structurally unreachable" in the current descriptor API. This pass did
  **not** test that claim (it became moot once E11 was exonerated), so it
  remains an untested inherited claim, explicitly flagged as such — do not
  treat it as verified.
- Project policy: do not make changes that cannot be numerically validated,
  and finish the current gate before adjacent work.

E11 therefore **remains an open, genuine authoritative-fidelity gap** that
must be closed before GEOS integration and before any SP3 comparison can pass
— unchanged in substance from `CURRENT_STATE.md` §10, now with corrected line
numbers (`:1142` call, `:1145-1152` write-back) and corrected write-back
semantics (§15.2).

## 15.7 Cost and change accounting

- Source files changed: **none**. No CUDA, no Fortran, no tooling.
- GPU jobs run: **none** (existing `it=1/2/3` dumps from §14's pass were
  sufficient and are the current code state).
- Fortran savepoints generated: **none**.
- Comparator invocations: 3 (`it=1,2,3`) + 1 per-tile detail + 1 vertical
  profile.
- Numerical algorithm changed: **NO**.
- Nothing committed, pushed, or merged.

## 15.8 §15 FINAL REPORT

- §1 E11 authoritative mapping: **DONE** (§15.1; mapping table; line-number
  correction `:1094-1095` → `:1142`/`:1145-1152`)
- §2 CUDA E11 status: **MISSING**, confirmed from current code, not history
  (§15.2). Plus one provably-wrong in-source comment characterized.
- §3 first `it=3` degradation located: **DONE, and the premise corrected** —
  the first divergence is SP1 **`it=2`** (all 7 fields, full-population
  mismatch), not `it=3`. `it=3` NONFINITE is terminal growth (§15.4).
- §4 decisive discriminator: **E11 EXONERATED** on two independent grounds —
  timing (`:1142` runs after every SP1/SP2 capture; earliest effect is SP3)
  and data (first divergence at `it=2`, where E11 never executes) (§15.3).
- §5 E11 implementation: **NOT PERFORMED, deliberately and with reasons
  recorded** (§15.6) — unverifiable at SP1/SP2, defect proven elsewhere,
  requires untested new ring-0 capability.
- §6 rerun `it=1/2/3`: **DONE from existing dumps** (no source changed, so a
  fresh run would be bit-identical; no GPU job spent).
- §7 decision: SP1 `it=1` remains validated; SP1 `it=2`/`it=3` diverge; root
  cause is in the `it=1` path **after** SP1 = **SP2 scope**, which this pass
  was directed not to inspect. Localization stopped at the scope boundary.
- §8 cost control: one E11 mapping, zero fixes, zero GPU jobs, zero source
  changes (§15.7).
- Numerical algorithm changed: **NO**
- SP2/SP3: **NOT RUN**. Gate C2: **BLOCKED**

SP1 it=3 STILL BLOCKED — NUMERICAL DIVERGENCE REMAINS

---

# 16. SP2 at `it=1` (Task H) — first measurement of the D-grid stage

Scope as directed: SP2 `it=1` only. E11 not implemented; SP3 not inspected;
`it=2` not debugged; no broad regressions. One source-backed fix, one rerun.

**Headline: 8 of the 10 SP2 fields agree at ROUND-OFF SCALE or better. `u`/`v`
do not.** The dominant divergence had a proven INPUT/CONFIG root cause
(`n_sponge`) which was fixed; a ~1e-6 residual on `u`/`v` survives and is
localized, but not root-caused.

## 16.1 Authoritative boundary map, SP1 `it=1` → SP2 `it=1` (§1)

From current source (`dyn_core.F90`, line numbers verified this pass, not
carried over from milestone history). SP1 dumps at `:637-643`, SP2 at
`:871-880`. In order:

| # | line | operation | active here? |
|---|------|-----------|--------------|
| 1 | `:648` | `start_group_halo_update(i_pack(9), uc, vc, CGRID_NE)` | yes |
| 2 | `:661` | `complete_group_halo_update(i_pack(10))` (inline_q) | no (`inline_q=.F.`) |
| 3 | `:666` | `complete_group_halo_update(i_pack(3))` — `divgd`, CORNER | yes (`nord>0`) |
| 4 | `:670` | `complete_group_halo_update(i_pack(9))` — `uc`/`vc` | yes |
| 5 | `:672-705` | nested-grid BCs | no (not nested) |
| 6 | `:708-719` | `d4dmp` one-time init | yes, but inert (see 16.2) |
| 7 | `:730-795` | per-k damping derivation | yes — **the defect lived here** |
| 8 | `:819` | `fv3_sp_damping_record` | instrumentation only |
| 9 | `:823-836` | **`d_sw`** per k (`sw_core.F90:520-1540`) | yes |
| 10 | `:806-808`, `:845-851` | `d_ext(k)>0` a2b_ord2 / ptc-corner blocks | no (`d_ext=0`) |
| 11 | `:852-860` | `heat_source`/`diss_est` accumulation | ACTIVE (`d_con=1.0`) but inert at `it=1` (accumulates from zero) |
| 12 | `:871-880` | SP2 dump | — |

## 16.2 Input/config fidelity (§2) — a SIXTH member of the harness-input defect family

Checked before any arithmetic, per the directive's ordering. Verified clean
against the authoritative captured runtime table (below): `nord=2`,
`nord_v/nord_w/nord_t = min(2,nord) = 2`, `damp_vt` floor
`max(0.01, min(vtdm4, d4_bg_top/4)) = 0.01` with `do_vort_damp=.T.`,
`kgb = ke_bg = 0`, `d_ext = 0` at all k, `d_con = 1.0`, `hord_mt/vt/tm = 6`,
`hord_dp = -6`, `bdt = 3600`, `dp_ref`, `ptop`, `phis`/`zs`, `rdgas`.

`d4dmp` (`:708-719`) is inert for this fixture: the nonhydrostatic sine profile
weights `d4_bg_top` and `d4_bg_bot`, and both are `0.12`, so the two terms sum
to exactly `0.12` for any weight. `ptop`/`pfull` therefore cannot affect
`d4_bg`, and the CUDA short-circuit `d4_bg[k] = d4_bg_top` is exact. (This also
retires a suspicion raised and discarded this pass: the harness computes the
damping column in `make_host_state()` with `h.ptop = 1000.0`, *before*
`--sp1-boundary` sets the authoritative `100.0`. Latent hazard, not a live
defect, recorded so it is not rediscovered as a bug.)

**DEFECT: `n_sponge`.** `fv3_damping_translation.cpp` `geos_production_flags()`
hardcodes `n_sponge = 9` (`FV_StateMod.F90:694`). The authoritative fixture runs
`n_sponge = -1`. Proven two independent ways, both read directly off the
authoritative side:

1. the fixture's own namelist,
   `experiments/c8_L10_savepoint_6a_run1/scratch_.../input.nml`, `&fv_core_nml`:
   `n_sponge = -1`;
2. the fixture's own **captured runtime** damping table,
   `savepoints/sp2_it1_pe0_damping.txt`, written by `fv3_sp_damping_record` at
   `:819` — `nord_k = nord_w = nord_t = nord_v = 2`,
   `damp_vt = damp_w = damp_t = 0.01`, `d_con_k = 1.0`, `d2_divg = 0`,
   `hord 6/6/6/-6` at **every** k=1..10. That is exactly what
   `dyn_core.F90:736`'s `if (npz>1 .and. n_sponge>0)` produces when it does
   **not** fire.

With `n_sponge = 9`, `translate_damping()`'s three sponge arms (`:196-245`,
faithfully mirroring `:736-765`) corrupt **9 of the 10 levels**: k=0 →
`nord_k/nord_v/nord_w = 0`, `d2_divg = 0.2`, `damp_vt = 0.1`, `d_con = 0`;
k=1..7 → `d2_divg = 0.1`, `damp_vt = 0.05`, `d_con = 0`; k=8 → `d2_divg = 0.02`,
`d_con = 0`. Only k=9 was authoritative.

This is the sixth member of the same family as `bdt`, `ptop`, `phis`/`zs`,
`dp_ref`, `rdgas` — all host-side loader/config, none a CUDA kernel defect. It
is the first that is **per-level asymmetric** rather than scalar, which is why
it survived five prior scalar-input audits.

It also **root-causes the recorded "k=0 damping row not authoritative"**
open item: the authoritative Fortran floors `damp_vt` at 0.01 at k=1 (CUDA
k=0), while the sponge arm was applying 0.1 there.

## 16.3 Fix (§2, §6) — one change, config only, opt-in

`run_small_perf_bench.cpp`: new `--damp-n-sponge N` flag overriding
`dflags.n_sponge` after `geos_production_flags()` returns, guarded by a
`std::numeric_limits<int>::min()` sentinel. Validation passes `-1`.

`geos_production_flags()` itself was deliberately **not** changed: it *is* the
documented GEOS-production translation and is what the 3A.11T/U/V
performance/stability baseline was measured against. Changing it would have
silently moved that baseline. Mode separation verified by a default-path smoke
run (§16.7).

**Numerical algorithm changed: NO.** No kernel, no scientific expression, no
tolerance, no clamp, no stencil change, no Fortran output copied in.

## 16.4 Mandatory authoritative-damping check (§7)

New permanent tool: `tools/savepoints/verify_damping_column.py`. Compares the
harness's unconditionally-printed `[FV3-DAMP]` per-level table against the
Fortran run's own captured `sp2_it<IT>_pe<PE>_damping.txt`. Ten mapped columns
(`nord_k`, `nord_v`↔`ndif`, `nord_w`, `d2_divg`↔`d2_bg`, `damp_vt`↔`damp_v`,
`damp_w`, `damp_t`, `d_con_k`↔`d_con`, `kgb`, `d_ext`), each mapping justified
against source rather than by name similarity. Also asserts the six per-tile
blocks are identical (the column is tile-invariant in Fortran). No value is
recomputed — captured Fortran runtime values win, per the directive.

Reported, not passed over, as **NOT verifiable from this record**: `d4_bg`
(no Fortran column), `dddmp` (absent entirely), and `nord_t` beyond k=1.

```
pre-fix : authoritative damping/config in SP2 validation: FAIL (61 column mismatches)
post-fix: authoritative damping/config in SP2 validation: PASS (10 levels x 10 mapped columns agree)
```

**`authoritative damping/config in SP2 validation: PASS`** (post-fix; the run
used for every number below).

## 16.5 SP2 `it=1` comparison (§3)

`compare_cuda_fortran.py` extended with `SP2_FIELDS` (the 10 fields in the
authoritative `sp2_it1_pe0.meta`), `SP2_DEFINED_BOX` from `d_sw`'s own write
bounds, a per-tile summary, and a producer-ordered first-divergent-field
report. `fv3_gpu_runtime.cpp`'s D2H dump targets gained `mfx, mfy, cx, cy` —
without them SP2 coverage would have been 6/10 and `d_sw`'s own mass/tracer
fluxes would have gone uncompared.

Worst verdict across all 6 tiles, defined box, corners reported separately:

| field | verdict | worst max_abs | worst max_rel |
|-------|---------|---------------|---------------|
| `divgd` | ROUND-OFF SCALE | 2.85e-19 | 1.96e-14 |
| `cx` | ROUND-OFF SCALE | 9.89e-17 | 5.17e-15 |
| `cy` | ROUND-OFF SCALE | 1.11e-16 | 3.00e-15 |
| `mfx` | ROUND-OFF SCALE | 1.28e+00 (range 1.7e+14) | 7.57e-15 |
| `mfy` | ROUND-OFF SCALE | 1.44e+00 (range 4.2e+14) | 3.42e-15 |
| `delp` | ROUND-OFF SCALE | 3.64e-12 | 3.64e-16 |
| `pt` | ROUND-OFF SCALE | 7.11e-15 | 2.88e-16 |
| `w` | BITWISE — **but vacuous**, see below | 0 | 0 |
| `u` | **SIGNIFICANT DIFFERENCE** | 3.94e+01 | 9.42e-06 |
| `v` | **SIGNIFICANT DIFFERENCE** | 3.94e+01 | 9.42e-06 |

`w`'s BITWISE verdict carries **no validation weight** and is stated as such
rather than counted as a pass: `w ≡ 0` identically on both sides at SP2, because
`d_sw` does not update `w` — `update_dz_d`/`Riem_Solver3` do, and both run after
`:871`. Honest coverage is therefore **7 of 10 fields genuinely agreeing at
round-off**, one vacuous, two divergent.

Coverage limitation, stated: `cx`/`cy` are declared over the halo range in
their non-staggered direction (`cx(is:ie+1, jsd:jed)`); the verdict box is
restricted to the owned staggered box, so their halo rows are **not** compared.

## 16.6 Effect of the fix, and the residual

Per-tile `u`/`v`, owned box, before vs after the one fix:

| | pre-fix max_rel | pre-fix median_rel | post-fix max_rel | post-fix median_rel |
|---|---|---|---|---|
| pe0/pe1 (`u`), pe3/pe4 (`v`) | 1.86e-02 | 5.23e-03 | 1.15e-06 | 2.14e-08 |
| pe2/pe5 (`u` and `v`) | 1.70e-02 | 3.61e-03 | 1.10e-06 | 3.15e-09 |
| pe3/pe4 (`u`), pe0/pe1 (`v`) | 1.28e-02 | 4.52e-04 | 9.42e-06 | **3.9e-16** |

Four orders of magnitude improvement in `max_rel`; the *median* collapses to
round-off on four of the twelve (field, tile) pairs. `n_sponge` was therefore
the dominant, proven cause — but not the whole cause.

**Residual spatial signature (the localization).** The residual is not
interior-uniform. On pe0/`u`, 437 of 720 cells exceed 1e-8 relative, and their
(i,j) footprint is exactly a **~2-cell-deep perimeter frame of the owned box**
(i∈{1,2} at every j, then i≥3 only at j∈{1,2,8,9}); 48 of the 52 frame columns
at k=1, with the 20-column interior at round-off (median 3.9e-16 where the
frame is excluded). Magnitude decays monotonically with depth: max_rel
1.15e-06 at k=3 falling to 1.09e-07 at k=10.

**Every input `d_sw` consumed is bit-identical.** Measured ring by ring on the
authoritative SP2 savepoint, which still holds the pre-`d_sw` exchanged halo
(Fortran re-exchanges `u`/`v` only near `:1142`, ~270 lines after the SP2
capture), so the SP2 halo *is* what `d_sw` read:

```
field  pe   region     n     max_abs      max_rel
u      0    OWNED      72    3.8887e+01   1.149e-06
u      0    halo r=1   38    0.0000e+00   0.000e+00
u      0    halo r=2   46    0.0000e+00   0.000e+00
u      0    halo r=3   54    0.0000e+00   0.000e+00
delp   0    halo r=1..3 132  0.0000e+00   0.000e+00
```

Confirmed non-vacuous: the Fortran `u` halo is nonzero in all 1380 cells
(|halo| ∈ [1.0e-15, 3.30e+01], the 1e-15 being FV3's pre-exchange sentinel in
the donorless diagonal corners), and `delp`'s in all 1320 (∈ [9.989e3, 1.0e4]).
The CUDA six-tile halo exchange reproduced all of it exactly, sentinel
included.

So: authoritative config, bit-identical inputs, seven outputs at round-off,
interior of `u`/`v` at round-off — and a 2-deep perimeter frame of `u`/`v`
wrong at ≤9.4e-6. That places the residual inside `d_sw`'s own `u`/`v` update
(`sw_core.F90:1463-1472` and the damping/heating terms feeding it at
`:1480-1536`), in its boundary/perimeter treatment specifically.

**Root cause of the residual: NOT IDENTIFIED.** Per the directive's stop
condition, the next investigation was not started. Recorded as the resume
**hypothesis, explicitly not a finding**: a 2-deep frame at `nord=2` is the
signature of a del-2n stencil one ring short. `CURRENT_STATE.md` §10 records
`clamp_nord_cap` (D1) capping the WP-B group loop at `ng-1 == 2` and asserts it
is "not live at production `nord=2`". At `nord=2` the del-6 divergence damping
needs depth 3 while `ng-1 = 2`, so that assertion should be **re-tested, not
assumed**, as step 1 tomorrow. It must be checked against source before being
believed — the exactly-analogous "`p_grad_c` halo-coverage" hypothesis of an
earlier pass looked equally compelling and was disproven.

## 16.7 Per-tile pattern (§4) — the asymmetry discriminator

**The pe2/pe3/pe4-much-worse-than-pe0/pe1/pe5 pattern of `it=2` is ABSENT at
SP2 `it=1`.** All six tiles classify SIGNIFICANT DIFFERENCE, and the magnitudes
fall into three *symmetric* pairs — {pe0,pe1}, {pe2,pe5}, {pe3,pe4} — with `u`
and `v` exchanging roles between the halves (pe0 `u` 1.15e-06 / `v` 9.42e-06;
pe3 `u` 9.42e-06 / `v` 1.15e-06). That is ordinary cubed-sphere tile-parity
symmetry: odd and even tiles have transposed local orientation.

Interpretation, stated with its limits: this is **tile-local translation**
behavior, not a seam/topology defect and not a configuration defect (config is
now PASS). The `it=2` asymmetry is therefore a *different* phenomenon from the
`it=1` residual, and the two should not be assumed to share a cause.

## 16.8 Mode separation and regression (§8)

- Default synthetic path (no authoritative flags): `=== 3A.11 CUDA perf
  benchmark: PASS ===`, exit 0, **zero `[6B-H]` lines** — the override
  provably does not fire.
- R8 `ctest`: **22/22 PASS** (required — `fv3_gpu_runtime.cpp` is shared
  runtime code).
- Runs A/B exit 1 on the already-known `it=3` NONFINITE failure of §15, which
  occurs after the `it=1` dump and does not affect it.
- Cost: exactly one GPU job; one source-backed fix; one rerun. No `it=2`
  debugging, no SP3, no E11, no c16/c32, no R4, no sanitizer matrix.

## 16.9 Resume point for tomorrow

**Resume at `d_sw`'s `u`/`v` update perimeter treatment, `sw_core.F90:1463-1472`
plus the `damp_v>1.E-5` diffusive add at `:1523-1536`, versus the CUDA WP-B
D-grid damping path.** Step 1: re-test the `clamp_nord_cap` (D1) `ng-1 == 2`
cap against the depth-3 requirement of del-6 damping at `nord=2`, and the
`del6_u`/`del6_v` flux loop bounds, against source. Do not re-audit inputs:
`d_sw`'s halo inputs are proven bit-identical (§16.6) and its config is proven
authoritative (§16.4). Do not re-derive the boundary map (§16.1).

## 16.10 Outputs

- `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` — `--damp-n-sponge`
  (sentinel-guarded, default path bit-identical).
- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` — `mfx`/`mfy`/`cx`/`cy` added to
  the debug D2H dump targets (opt-in via `FV3_GPU_DEBUG_FIELDS`).
- `tools/savepoints/verify_damping_column.py` — new, permanent, the §7 check.
- `tools/savepoints/compare_cuda_fortran.py` — SP2 field set, boxes, per-tile
  summary, producer-ordered first-divergent report. SP1 behavior unchanged.
- `sp2_it1_taskh.sh` — reproduction script (untracked scratch, per repo policy).
- Nothing committed, pushed, or merged.

## 16.11 §16 FINAL REPORT

- SP1 `it=1`: **PASS** (§14.7, ROUND-OFF SCALE, all fields, unchanged)
- Authoritative SP2 configuration/damping: **PASS** (post-fix; **FAIL** as
  previously coded, 61 mismatches)
- SP2 `it=1`: **FAIL** — 7/10 fields genuinely ROUND-OFF SCALE, 1 vacuous
  (`w ≡ 0`), `u`/`v` SIGNIFICANT DIFFERENCE at ≤9.42e-06
- First divergent field (producer order): **`u`** (with `v`, its symmetric
  partner, written by the same block)
- First divergent operator: **`d_sw`'s `u`/`v` update**, `sw_core.F90:1463-1472`
  and the damping/heating terms at `:1480-1536`
- Per-tile pattern: symmetric three pairs {pe0,pe1}/{pe2,pe5}/{pe3,pe4}; the
  `it=2` pe2/pe3/pe4 asymmetry is **ABSENT**
- Root-cause classification: **INPUT/CONFIG (proven, fixed)** for the dominant
  1.9e-2 divergence; **UNRESOLVED** for the ≤9.4e-6 residual, localized to a
  2-deep owned-box perimeter frame
- Fix applied: one — `--damp-n-sponge` config override (`n_sponge` 9 → −1)
- Numerical algorithm changed: **NO**
- Gate C2: still **BLOCKED**

SP2 it=1 LOCALIZED — RESUME TOMORROW AT d_sw u/v update perimeter (sw_core.F90:1463-1472)

---

# 17. Task I — SP2 `it=1` closed: the `edge_*` interpolation weights

## 17.1 Result

**SP2 `it=1` PASSES.** All ten fields ROUND-OFF SCALE on all six tiles; the
2-cell perimeter frame is gone. One fix, one input: the cubed-sphere
edge-interpolation weights `edge_w/e/s/n`.

| field | worst max_abs (6 tiles) | worst max_rel | verdict |
|-------|------------------------|---------------|---------|
| `u` | 1.49e-08 (range 3.5e+07) | 4.23e-16 | ROUND-OFF SCALE |
| `v` | 1.30e-08 (range 3.5e+07) | 3.70e-16 | ROUND-OFF SCALE |
| `divgd` | 2.85e-19 | 1.96e-14 | ROUND-OFF SCALE |
| `cx` | 9.89e-17 | 5.17e-15 | ROUND-OFF SCALE |
| `cy` | 1.11e-16 | 3.00e-15 | ROUND-OFF SCALE |
| `mfx` | 1.28e+00 (range 1.7e+14) | 7.57e-15 | ROUND-OFF SCALE |
| `mfy` | 1.44e+00 (range 4.2e+14) | 3.42e-15 | ROUND-OFF SCALE |
| `delp` | 3.64e-12 | 3.63e-16 | ROUND-OFF SCALE |
| `pt` | 7.11e-15 | 2.88e-16 | ROUND-OFF SCALE |
| `w` | 0 | 0 | BITWISE — still **VACUOUS** (`w ≡ 0`, `d_sw` does not write `w`) |

`u`/`v` improved from `max_rel` ≤9.42e-06 (§16.6) to ≤4.23e-16 — ten orders of
magnitude. Honest coverage is unchanged at **9 of 10 fields**: `w`'s agreement
still carries no weight, and `cx`/`cy` are still judged on the owned staggered
box only (§16.5).

## 17.2 Ring-width hypothesis (§16.9 / CURRENT_STATE §9 step 4): DISPROVEN

Reconciled against source before any measurement, for `nord=2`, `ng=3`:

| quantity | Fortran | CUDA | match |
|----------|---------|------|-------|
| del-2n mathematical radius, `nord=2` | 3 | 3 | yes |
| `del6_vt_flux` `d2` fill extent | is-2..ie+2 / js-2..je+2 | same | yes |
| `del6_vt_flux` flux loop bounds per pass | contracting by 1 per pass | same | yes |
| `copy_corners` placement in the n-loop | before each pass | same | yes |
| WP-B group-loop cap | n/a | `clamp_nord_cap(2,3) = 2`, loop inclusive | yes |
| `u`/`v` write radius | owned staggered box | same | yes |

`clamp_nord_cap(nord=2, ng=3)` returns 2 and the launcher loop is inclusive, so
every `nord=2` damping group does run at full depth. **CURRENT_STATE §10's
suspicion that `clamp_nord_cap` is live at production `nord=2` is resolved: it
is NOT a defect there.** The dead `nord_k==3` branch remains a real (documented,
unexercised) gap.

The independent measurement agreed. The SP2 residual was shown to be an error in
`ke` alone, not in any flux or damping add: the discrete curl of (du,dv) vanished
to 2.8e-10…1.3e-09 relative on all six tiles at k=3/6/10 while the residual was
1.1e-06…9.4e-06, and an integrate-and-re-predict round trip closed to
1.2e-08…2.6e-08 against `du` magnitudes 3.7–39. That eliminated
`del6_vt_flux`, the `damp_v` add, and `fv_tp_2d(vort)` — i.e. exactly the
operators the ring-width hypothesis pointed at — and left the `damp2` →
`vort` → `ke` chain.

## 17.3 Complete `d_sw` `u`/`v` input audit (§3): one synthetic input, found

Run once, before touching any arithmetic, as instructed.

AUTHORITATIVE + VERIFIED (against the fixture's own stdout config print and its
captured runtime damping record `sp2_it1_pe0_damping.txt`): `nord = 2`,
`nord_v = 2`, `nord_k = 2` at every k, `damp_vt = 0.01`, `d2_bg = 0`,
`d4_bg = 0.12`, `dddmp = 0.2`, `d_con = 1.0`, `d_ext = 0`, `n_sponge = -1`,
`do_vort_damp = T`, `grid_type = 0`, `hord_mt/vt/tm/dp = 6/6/6/-6`,
`n_split = 3`, `a_imp = 1.0`, `bdt = 3600`, `rdgas = 8314.47/28.965`, `dt`.
All 2-D per-cell grid metrics (`dx`, `dy`, `dxc`, `dyc`, `rarea`, `rarea_c`,
`sin_sg`, `cos_sg`, `rsin2`, `cosa_*`, …), `da_min`, `da_min_c`: AUTHORITATIVE
(loaded by `--grid-metrics`). Halo inputs `u`, `v`, `delp`, `pt` on rings
r=1/2/3: **bit-identical** (§16.6, unchanged).

**SYNTHETIC DEFAULT — the only one: `edge_w`, `edge_e`, `edge_s`, `edge_n`.**
`make_host_state` fabricates them as local-index ramps `1.0 + 0.01*index`
(values 1.0…1.13), and says so at that site. `--grid-metrics` cannot override
them: the Prompt 6A capture took gridstruct's per-cell **2-D** metric arrays,
and `edge_*` are **1-D** edge vectors, so they were never in the authoritative
set. Authoritative values are ≈0.298…0.702; the fabricated ramp is not even
inside [0,1].

DERIVED (and left alone): `grid_lonlat`, `agrid_lonlat`, built by
`make_host_state` from `cube::CubeGrid`; also absent from the authoritative
metric set.

## 17.4 Why `edge_*` produce exactly this signature

`edge_*` are read only by `a2b_ord4`'s edge rows. In the SP1→SP2 window the only
`a2b_ord4` consumer is `smag_corner` (`sw_core.F90:1857-1947`), which moves the
A-grid shear strain to the B grid. `smag_corner` feeds only
`damp2 = da_min_c * max(d2_bg, min(0.20, dddmp*vort_smag))`, and with the
authoritative table (`d2_bg = 0`, `dddmp = 0.2`) `damp2` is *purely*
smag-proportional — nothing dilutes the error. `damp2` reaches `u`/`v` through
`vort` → `ke` and **nothing else**, which is why the other seven `d_sw` outputs
could pass at round-off while `u`/`v` did not.

Footprint: `a2b_ord4`'s interior is metric-authoritative and `edge_*`-free; its
edge rows are j=1,9 and i=1,9 directly, plus j=2,8 / i=2,8 through the
`c2 = -1/6` `qout` echo (≈1/12 attenuation), exact in the interior. That is a
2-deep perimeter frame with monotonically decaying magnitude — the measured
signature.

## 17.5 Grid authenticity: proven, not assumed

Recomputing `edge_*` from `cube::CubeGrid` is only legitimate if that grid *is*
the authoritative c8 grid. Proven inside the loader, before it writes anything,
by re-deriving `dx`/`dy` from `CubeGrid` and fitting a single radius against the
authoritative arrays `grid_metrics_6a5::load` has just installed:

- fitted radius **6371000.000000 m** on all six tiles (FV3 `radius` exactly);
- `max|dx,dy ratio − 1|` between **2.0e-15 and 6.4e-15**;
- the loader **fails the run** if that exceeds 1e-12.

Stated limitation, recorded in the code: great-circle distances are
rotation-invariant and the six panels are pure rotations, so this proves panel
**shape**, not tile identity. It must not be read as a check on the tile index.

No new Fortran instrumentation or savepoint was needed (`edge_*` are recomputed
from geometry, and the resulting weights are symmetric and identical on all six
tiles, `edge_w[2..8] = 0.298255158930 0.354408625979 0.423790009628
0.500000000000 0.576209990372 0.645591374021 0.701744841070`, with the exact
0.5 at the panel midpoint — the expected symmetry of a gnomonic c8 panel).

## 17.6 Fix (§6): one change, harness input only, opt-in

New `namespace grid_geometry_6b` in
`cuda/dyn_core/benchmarks/run_small_perf_bench.cpp`, called from inside the
existing `--grid-metrics` gate strictly *after* `grid_metrics_6a5::load` (so the
authenticity self-check can run against authoritative `dx`/`dy`). It is a direct
port of `edge_factors`, `REF/model/fv_grid_utils.F90:1188-1293`: A-grid-pair
spherical midpoints along each physical edge, then
`edge = d2/(d1+d2)` from great-circle distances to the B-grid corner. Entries
outside the filled band are poisoned with `big_number`, as Fortran does; CUDA's
`launch_a2b_ord4_global_cuda` reads only `[js+1, je] = [2, npx-1]` (and the same
in i), exactly the Fortran band, so no poisoned entry is read.

The radius cancels in `d2/(d1+d2)`, so unit-sphere distances are exact and no
earth-radius constant is introduced.

**Numerical algorithm changed: NO** — no numerical expression is touched, only
the values fed into one. Confined to the `--grid-metrics` gate; the
default/performance path keeps the fabricated ramps and is bit-identical
(verified: default synthetic smoke PASS, zero `[6B]` lines).

**This is the seventh member of the harness input-fidelity defect family**
(`bdt`, `ptop`, `phis`/`zs`, `dp_ref`, `rdgas`, `n_sponge`, `edge_*`). It is the
first that could not be fixed by a config scalar or by extending the existing
savepoint loader — the authoritative data simply was not captured, so it had to
be recomputed from the grid and then proven authoritative.

## 17.7 One retracted intermediate claim, recorded so it is not rediscovered

The first revision of `grid_geometry_6b` also "fixed" `make_host_state`'s
`cube::CubeGrid` tile argument, on the theory that it passed a 0-based MPI rank
to a 1-based parameter (`test_cube_seam_continuity.cpp` drives `t = 1..6`, and
`tile_rotate`'s case labels are 1..6).

**That was wrong.** The `tile` variable in the harness is already 1-based:
`grid_metrics_6a5::load` derives its directory as `"tile" + (tile - 1)`, i.e.
tile 1..6 → `tile0..tile5` on disk. `make_host_state` was correct. The `tile + 1`
produced `CubeGrid` tile 7, whose panel falls through `tile_rotate`'s `default`
while `corner_owner` reads it as odd parity; the resulting inconsistent halo
mapping generated **NaN in `u`/`v`/`delp` at `it=1`** (job 58156403). Caught
immediately by the loader's own progress print, which named the CubeGrid tile it
was using. Reverted; `grid_lonlat`/`agrid_lonlat` therefore need no rebuild
either, and the fix reduced to `edge_*` alone.

Lesson worth keeping: the authenticity self-check in §17.5 would never have
caught this, and did not. Index-base conventions must be read off a caller, not
inferred.

## 17.8 Residual structure (§9)

The perimeter frame is no longer a distinct error structure. Mean |Δ| on the
2-deep frame versus the interior, SP2 `it=1`, all six tiles:

- `u`: frame/interior mean ratio **0.92 … 2.82**
- `v`: frame/interior mean ratio **1.00 … 3.09**

against `max_rel` ≤4.2e-16 everywhere. A factor of ≈2-3 in round-off *amplitude*
is expected on the frame, because `a2b_ord4`'s edge rows evaluate a different
and longer expression than its interior; it is not a factor of 2-3 in relative
error, which stays at 1e-16. Pre-fix the frame carried 1e-06 relative against a
1e-16 interior — four orders of separation, versus none now.

The §16.6 frame-footprint characterization is left as measured and is now moot;
it was not re-derived, since the pre-fix dump it describes has been superseded.

## 17.9 §4 (cx/cy halos) — not needed

§4 was conditional on continued localization. SP2 `it=1` became acceptable
before it was reached, so no halo-coverage extension was made for `cx`/`cy`.
Their PASS remains a claim about the owned staggered box only.

## 17.10 Regression (§12)

Only validation-loader/harness code changed (`run_small_perf_bench.cpp`, behind
`--grid-metrics`), so per §12: validation run + synthetic smoke.

- SP2 `it=1` validation run: PASS (§17.1)
- SP1 `it=1` regression: **PASS, unchanged** — all fields ROUND-OFF SCALE except
  `ws3` at SMALL NUMERICAL DIFFERENCE, exactly as §14.7
- Default synthetic path: **PASS**, `[6B]` line count 0 (loader never runs)
- No R8/R4 CTest, sanitizer, or performance matrix run — no shared
  production/runtime source was modified, and §12/§13 exclude them.

The runs still exit non-zero because of the **pre-existing `it=3` NONFINITE**
failure with `FV3_GPU_DEBUG_FAIL=1` set. First NaN is at `it=3
dgrid_dynamics`; `it=1` and `it=2` are clean, and the tile 3/4/5
`u`/`v`/`delp`/`pt`/`w`/`delz` + positivity signature is the one already
recorded in CURRENT_STATE §10. Not caused by this pass.

## 17.11 §10 cheap discriminator — SP1 `it=2`, with the corrected code

Run from the existing dump (no extra GPU time). **FAIL**, essentially unchanged
by the `edge_*` fix:

- First divergent field by producer order: **`uc`**
- All seven SP1 fields SIGNIFICANT DIFFERENCE on all six tiles
- `uc` `max_rel`: pe5 4.43e-02, pe0 1.15e-01, pe1 1.75e-01, pe2 7.05e-01,
  pe3 9.99e-01, **pe4 1.83e+00**; `delpc` pe1 4.42e-03 … pe4 2.47e-01
- The strong pe2/pe3/pe4-vs-pe0/pe1/pe5 asymmetry is **still present**,
  confirming §16.7's conclusion that SP1 `it=2` is a different phenomenon from
  the (now closed) SP2 `it=1` residual.

Not investigated further, per §10.

## 17.12 Outputs

- `sp2_it1_taski.sh` (untracked scratch) — SP2 dump, SP1 regression dump,
  default smoke; Slurm job 58156554 on `warpa003`
- `tools/savepoints/sp2_cuda_dump_edge/`, `tools/savepoints/sp1_cuda_dump_edge/`
  (shared filesystem, not committed)
- `scratch_6b/frame_check.py` — diagnostic only, frame-vs-interior structure
- Comparator unchanged this pass (§11 satisfied)

## 17.13 §17 FINAL REPORT

- Initial residual: `u`/`v` SIGNIFICANT DIFFERENCE, `max_rel` ≤9.42e-06,
  2-deep owned-box perimeter frame
- nord ring-width reconciliation: **MATCH** — hypothesis DISPROVEN;
  `clamp_nord_cap` is not a defect at production `nord=2`
- `d_sw` `u`/`v` input audit: **FAIL** — one synthetic default found,
  `edge_w/e/s/n`; no synthetic/default input remains in the validated path
- First proven residual cause: fabricated `edge_*` → `a2b_ord4` edge rows →
  `smag_corner` → `damp2` → `vort` → `ke` → `u`/`v`
- Fixes applied: **1** (`grid_geometry_6b`, authoritative `edge_factors` port,
  opt-in, harness input only)
- Numerical algorithm changed: **NO**
- SP1 `it=1`: **PASS** (unchanged)
- SP2 `it=1`: **PASS** — 9/10 fields genuinely ROUND-OFF SCALE, `w` vacuous
- `u`: max_abs 1.49e-08, max_rel 4.23e-16
- `v`: max_abs 1.30e-08, max_rel 3.70e-16
- Perimeter-frame residual: **GONE**
- SP1 `it=2` discriminator: **FAIL**, first field `uc`, tile asymmetry persists
- Gate C2: still **BLOCKED** (SP1 `it=2`, `it=3` NONFINITE, SP3 not run)

SP2 it=1 PASS — NEXT BLOCKER LOCALIZED (SP1 it=2, first field `uc`)

---

# 18. SP3 `it=1` pass — HARD STOP: SP3 `it=1` is not capturable

Task scope was SP2 `it=1` → SP3 `it=1` only: map the operator sequence, run one
complete input-fidelity audit, then compare SP3 at `it=1`. §1 and §2 completed.
§3 **cannot be executed with existing data**, which is hard-stop condition 2
(new Fortran instrumentation / savepoints required). No source file was
modified; zero fixes applied; no GPU job submitted.

## 18.1 The blocker: SP3 is a dyn_core-**exit** savepoint, instrumented once

`tools/savepoints/fortran_savepoint_instrumentation.patch` places the SP3 block
**outside** the acoustic loop (immediately after
`if (allocated(heat_source)) deallocate( heat_source )`) and labels every field
with the *constant* `n_split`, not the loop variable `it`:

```
call fv3_sp_dump3d('sp3', n_split, 'u', 'i,j,k', u, isd,ied,jsd,jed+1,1,npz)
```

SP2's block, by contrast, sits inside the loop right after `d_sw` and labels
with `it`. Consequence, confirmed in **both** independent Prompt 6A runs
(`ls … | sed 's/_pe[0-9].*//' | sort -u`):

```
sp0_it0  sp1_it1 sp1_it2 sp1_it3  sp2_it1 sp2_it2 sp2_it3  sp3_it3
```

There is exactly one SP3 instance per dyn_core call and it is at `it=3`.
`sp3_it1` does not exist and cannot be produced by re-reading, re-indexing or
re-deriving the captured data. Producing it requires moving/adding Fortran
instrumentation inside the acoustic loop and re-running the fixture.

## 18.2 A second, independent reason SP3 is exit-only by design

`dyn_core.F90:424-428`:

```
if ( flagstruct%breed_vortex_inline .or. it==n_split ) then
     remap_step = .true.
else
     remap_step = .false.
```

`breed_vortex_inline` defaults `.false.`, so at `it=1` `remap_step = .false.`,
and `dyn_core.F90:971` passes `remap_step` as `Riem_Solver3`'s `last_call`.
With `last_call = .false.` the solver does **not** write `pe`, `peln`, `pk`;
`pe_halo` (`:988`) is skipped; the hydrostatic `pkz` block (`:1039`) is skipped.
Of SP3's 13 fields (`u v w delp pt delz pe peln pk pkz omga ua va`), the ones
that carry genuinely new end-of-`it=1` information are only
**`u`, `v`, `w`, `delz`** (`delp`/`pt` are merely halo-exchanged after SP2).
So even a hypothetical `sp3_it1` capture would be largely stale. The correct
instrumentation for this question is a *new* in-loop savepoint, not SP3.

Corollary for §5 (make `w` non-vacuous): **NOT REACHABLE.** No authoritative
capture of `w` or `delz` at end-of-substep-`it=1` exists anywhere in the
Prompt 6A data. `w` is first non-trivially observable at `sp3_it3`, which is
downstream of the already-known SP1 `it=2` divergence and also downstream of
E11 (`:1142`, `it==n_split`), which is MISSING in CUDA.

## 18.3 §1 — authoritative SP2 → end-of-substep-`it=1` operator sequence

From `dyn_core.F90` at the authoritative pin, with this fixture's flags. SP2 is
captured at `:871-880`, immediately after `d_sw`.

| # | line | operator | active at `it=1`? | reason |
|---|------|----------|-------------------|--------|
| 1 | `:884` | `mix_dp` | **NO** | `fill_dp = .false.` (fv_arrays default) |
| 2 | `:894-895` | `start_group_halo_update(i_pack(1), delp, pt)` | YES | |
| 3 | `:899-922` | `divg2` | degenerate | `d_ext = 0` ⇒ `divg2(:,:) = 0.` |
| 4 | `:926` | `complete_group_halo_update(i_pack(1))` | YES | |
| 5 | `:953-954` | **`update_dz_d`** | YES | writes `zh`, `delz`, `ws` |
| 6 | `:966-971` | **`Riem_Solver3`** → `SIM1_solver` | YES | writes `w`, `delz`, `zh`, `pkc`, `pk3` |
| 7 | `:982-983` | `start_group_halo_update(i_pack(4), zh, pkc)` | YES | |
| 8 | `:988` | `pe_halo` | **NO** | gated `remap_step` (false at `it=1`) |
| 9 | `:994` | `pk3_halo` | YES | `use_logp = .false.` ⇒ not `pln_halo` |
| 10 | `:1015` | `complete_group_halo_update(i_pack(4))` | YES | |
| 11 | `:1018-1025` | `gz(i,j,k) = zh(i,j,k)*grav`, `k=1..npz+1`, `j=js-2..je+2`, `i=is-2..ie+2` | YES | |
| 12 | `:1055-1068` | PG_D dispatch → **`nh_p_grad`** | YES | `beta = 0.0` ⇒ not `split_p_grad`, not `one_grad_p`; writes `u`, `v` |
| 13 | `:1100+` | `Ray_fast` | **NO** | `RF_fast = .false.`, `tau = 0.` |
| 14 | `:1142-1152` | E11 `mpp_get_boundary` + write-back | **NO** | gated `it == n_split` |
| 15 | `:1156-1157` | `start_group_halo_update(i_pack(8), u, v, DGRID_NE)` | YES | `it /= n_split` |

So exactly three numerical operators write state between SP2 `it=1` and the end
of substep `it=1`: **`update_dz_d`**, **`Riem_Solver3`/`SIM1_solver`**, and
**`nh_p_grad`**. Nested-grid BCs and `breed_vortex_inline` are inactive.

## 18.4 §2 — complete input-fidelity audit of that window

Authoritative side = fixture `input.nml &fv_core_nml` ∪ `fv_arrays.F90`
defaults ∪ the captured runtime damping table `sp2_it1_pe0_damping.txt`
(measured ground truth, not inferred). CUDA side = `run_small_perf_bench.cpp`
`cfg.*` / `h.*` and the runtime's `p.*` derivation.

### AUTHORITATIVE + VERIFIED (live in this window)

| input | authoritative | CUDA | evidence |
|-------|---------------|------|----------|
| `a_imp` | 1.0 | `cfg.a_imp = 1.0` | namelist |
| `p_fac` | 0.05 | `cfg.p_fac = 0.05` | `fv_arrays.F90:604` |
| `dz_min` | 2.0 | `cfg.dz_min = 2.0` | `fv_arrays.F90:625` |
| `lim_fac` | 1.0 | `cfg.lim_fac = 1.0` | `fv_arrays.F90:340` |
| `beta` | 0.0 | `cfg.beta = 0.0` | `fv_arrays.F90:467`; selects `nh_p_grad`, the only D-grid PG variant CUDA implements — correct branch, and `fv3_gpu_runtime.cpp:1025` fences `beta != 0` |
| `use_logp` | `.false.` | `cfg.use_logp = 0` | `fv_arrays.F90:636`; ⇒ `pk3_halo`, which `wp_d_contract_cuda.cpp:56-65` calls unconditionally, matching `:994` |
| `a2b_ord` | 4 | `cfg.a2b_order = 4` | `fv_arrays.F90:1015` |
| `hord_tm` | 6 | `cfg.hord_tm = 6` | namelist |
| `nord_v` | 2 | 2 | damping table |
| `damp_vt` | 1.0e-02 | 1.0e-02 | damping table |
| `d_con` | 1.0 | `cfg.d_con` | damping table |
| `d_ext` | 0 | `cfg.d_ext = 0` | damping table ⇒ `divg2 = 0` |
| `grav` | **9.80665** | `h.grav = 9.80665`, `cfg.grav = h.grav` | `geos_constants.fh:38` |
| `rdgas` | **8314.47/28.965** | `kFv3Rdgas` | `geos_constants.fh:44` |
| `akap` | **2/7 exactly** | `h.akap = 2.0/7.0` | `geos_constants.fh:50` `KAPPA = RDGAS/(3.5*RDGAS)` |
| `cp` | `3.5*RDGAS` | derived from `akap`,`rdgas` | `geos_constants.fh:51` `CP_AIR = RDGAS/KAPPA` |
| `ptop` | 100.0 | `--sp1-boundary` | prior pass |
| `dp_ref`, `phis`, `zs` | authoritative | `--sp1-boundary` | **BITWISE** on all 6 tiles, prior pass |
| `dt` | `bdt/n_split = 1200` | `p.dt = s->bdt/n_split` (`fv3_gpu_runtime.cpp:1510`) | `dyn_core.F90:317` |
| `fp_out` | `beta < -0.1` = `.false.` | `p.fp_out = (config.beta < -0.1)` (`:1538`) | `:971` |
| `last_call` | `remap_step` = `.false.` at `it=1` | `p.last_call = p.remap_step` (`:1715`) | `:424-428`, `:971` |
| `gama` | `1/(1-akap)` | `fv_real(1.0/(1.0-cfg.akap))` (`fv3_vertical_kernels.cu:~505`) | `nh_core.F90` |

### Proven NON-authoritative but proven INERT under `a_imp = 1.0`

`nh_core.F90:160-176` dispatches `Riem_Solver3` on `a_imp`; `a_imp > 0.999`
selects **`SIM1_solver`**, whose signature takes neither `ms` nor `scale_m`
(`ms` reaches only `RIM_2D`; `scale_m` only `SIM3p0_solver`/`SIM3_solver`/
`SIM_solver`). Confirmed against `fv3_vertical_column.hpp:696-766`, where
`sim1_column` likewise takes neither.

| input | authoritative | CUDA | why inert |
|-------|---------------|------|-----------|
| `scale_m` (`flagstruct%scale_z`) | **0.0** (`fv_arrays.F90:337`) | `cfg.scale_m = 1.0` | not a `SIM1_solver` argument |
| `m_split` | **0** (`fv_arrays.F90:632`) | `cfg.m_split = 2` → `p.vertical_ms` | `ms` not a `SIM1_solver` argument |
| `delt_max` | **1.0** (`fv_arrays.F90:759`) | `cfg.delt_max = 0.002` | `last_step` heat-source limiter, outside the `it=1` window |

These are real fidelity gaps and should be corrected before any configuration
with `a_imp <= 0.999` or before a `last_step` comparison, but they **cannot**
explain a >round-off difference in this window. Not changed this pass: a change
with no measurable target cannot be validated, and each would perturb the
default synthetic/performance path that the 3A.11T/U/V baseline was measured
against.

### AUTHORITATIVE BUT NOT VERIFIED (no capture exists)

`crx`, `cry`, `xfx`, `yfx` — `d_sw` outputs consumed directly by `update_dz_d`
(`:953-954`). They are **not** in the SP2 capture (SP2 has
`u v w delp pt divgd mfx mfy cx cy`). `cx`/`cy` are validated and are the
accumulated-Courant analogues, but they are not the same arrays. So
`update_dz_d`'s input contract is **not fully closed by existing data** — a
second reason this window needs new instrumentation rather than more analysis.

### Corrected stale in-source documentation (no code defect)

`run_small_perf_bench.cpp:808-835` carries a long comment asserting that CUDA
runs two disagreeing gravities (a hardcoded `T(9.8)` at
`fv3_vertical_column.hpp:703` vs `cfg.grav`), that `fv3_vertical_kernels.cu`
"never references grav at ALL", and that the authoritative `grav` value is
"NOT VERIFIED … undetermined". All three parts are **false today**:
`riem_solver3_column` has a threaded `T grav` parameter (repaired in milestone
3A.11V, documented in that file's own comment), `fv3_vertical_kernels.cu`
passes `fv_real(cfg.grav)`, and `geos_constants.fh:38` establishes
`GRAV = 9.80665` as authoritative. The comment is stale documentation only;
the numerics are correct. Left in place this pass (documentation-only edit with
no measurement attached); flagged for the next pass that touches that file.

**Audit verdict: no live input-fidelity defect remains in the
SP2 → end-of-`it=1` window.** This is the first Prompt 6B pass where the
input-identity-first ordering came up empty — which is itself the finding: if a
defect exists in this window it is now expected to be *arithmetic or halo
extent*, not a fabricated input.

## 18.5 Compensating localization obtained without new savepoints

Not a substitute for measurement, but it narrows the next pass:

1. SP2 `it=1` `delp` is **validated** (ROUND-OFF SCALE, all 6 tiles).
2. After SP2, `delp` is only halo-exchanged at `it=1` — `mix_dp` never runs
   (`fill_dp = .false.`) and no operator in the §18.3 table writes `delp`.
   Therefore owned-box `delp` at end-of-`it=1` **equals** the validated SP2
   value.
3. Yet SP1 `it=2` `delpc` diverges (pe1 4.42e-03 … pe4 2.47e-01). `c_sw`
   computes `delpc` from `delp`, `u`, `v`, `uc`, `vc` and grid metrics.
4. With `delp` eliminated, the corruption must enter through **`u`/`v`**, whose
   only post-SP2 writer at `it=1` is **`nh_p_grad`** (`:1067`) — or through
   `nh_p_grad`'s pressure inputs `pkc`/`gz`/`pk3`, produced by `Riem_Solver3`,
   `pk3_halo` and `gz = zh*grav`.

So the candidate set for the SP1 `it=2` divergence is
**`nh_p_grad` and its pressure inputs**, with `update_dz_d`'s `delp`-side
contribution demoted. Note also that `w`/`delz` first become non-trivial in
this window and are captured nowhere at `it=1`, so `Riem_Solver3` currently has
**zero** authoritative coverage at any `it`.

## 18.6 Required next step (new instrumentation — user decision)

To make this window measurable, add one in-loop Fortran savepoint at
end-of-substep in `dyn_core.F90` (after `:1157`, labelled with `it`), capturing
at minimum `u`, `v`, `w`, `delz`, `zh`, `pkc`, `pk3`, `gz`, `delp`, `pt`, and
re-run the c8/npz=10/n_split=3/R8 fixture. Two intermediate captures inside the
window (post-`Riem_Solver3` and post-`nh_p_grad`) would separate the two
remaining candidate operators in a single fixture run. `crx/cry/xfx/yfx` should
be added to the SP2 block at the same time to close `update_dz_d`'s input
contract.

This is exactly hard-stop condition 2 and was not undertaken: it modifies the
authoritative fixture's instrumentation and re-runs it, which is outside the
delegated scope.

## 18.7 §17 FINAL REPORT

- SP2 `it=1`: **PASS** (unchanged)
- SP3 `it=1`: **NOT RUN — not capturable**
- Complete SP3 input audit: **PASS for live inputs; FAIL as a completeness
  claim** (3 proven-inert gaps + 4 unverifiable `d_sw` outputs)
- Fixes applied: **NONE**
- Numerical algorithm changed: **NO**
- `w` validation: **NOT REACHED**
- Gate C2: **BLOCKED**

HARD STOP — SP3 `it=1` requires new Fortran instrumentation and a fixture re-run

---

# 19. Prompt 6B pass — in-loop instrumentation (Task J) + post-SP2 localization

Scope of this pass: add the minimum authoritative Fortran observability to expose
the SP2-`it=1` → SP1-`it=2` transition, regenerate the c8 reference once, extend
the existing comparator, and run the first CUDA comparison against the new
boundaries. Fix budget: 2. Both were used, both host-side, **numerical algorithm
changed: NO** for both.

**Outcome: the post-SP2 window is now fully covered and PASSES, and SP1 `it=2`
PASSES.** The blocker recorded in §15/§17 is closed. Root cause was a single
host-side memory-layout defect in the validation harness that corrupted the
`a2b_ord4` corner extrapolation only.

## 19.1 New in-loop Fortran savepoints (Task J)

Added to the fixture's `@fvdycore/model/dyn_core.F90`, in the acoustic loop, all
labelled with the real substep `it` (the defect that made the pre-existing `sp3`
block usable only at `it=n_split` — §18.1 — is not repeated). Instrumentation
only; no production numerical statement was touched, and every block is inside
`if (fv3_sp_enabled())`.

| savepoint | location | fields |
|-----------|----------|--------|
| `sp2a` | immediately after `Riem_Solver3` (`:986-995`) | `w delz zh pkc pk3 delp pt ws` |
| `sp2b` | immediately after `nh_p_grad` (`:1097-1103`) | `u v pkc pk3 gz` |
| `sp2c` | last statement before `enddo ! time split loop` (`:1283-1290`) | `u v w delz delp pt` |

Three design points that made this a 3-savepoint rather than 4-savepoint problem:

- **`sp2a` captures `ws`, which is `intent(IN)` to `Riem_Solver3`.** The value
  dumped there is therefore `update_dz_d`'s *output*, so a single savepoint
  discriminates `update_dz_d` from `Riem_Solver3` without a fourth capture.
- **`sp2b`'s `pkc`/`pk3`/`gz` are the B-grid (corner) transforms, not A-grid
  inputs**, because `nh_p_grad` calls `a2b_ord4(..., replace=.true.)` on all
  three. This is what made the decisive comparison possible (§19.3): the exact
  arrays the pressure gradient differenced are directly observable on both sides.
  `delp` is passed without `replace` and is unchanged.
- **`sp2c`'s `u`/`v` halos are in flight** (`i_pack(8)` started, not completed),
  so only the owned box is a defined comparison target for `u`/`v` there.

§2 of the task — SP2 observability gap — closed by adding `crx xfx cry yfx` to
the existing SP2 block (`:884-887`) with their authoritative staggered bounds.
This **closes `update_dz_d`'s input contract**, which §10 of `CURRENT_STATE` had
listed as "authoritative but not verifiable from existing data".

## 19.2 Reference regeneration (§4) — PASS

One regeneration, two independent runs, exact Prompt 6A production configuration
(c8 / npz=10 / n_split=3 / k_split=1 / R8 / 6 MPI ranks):

```
experiments/c8_L10_savepoint_6c_run1/savepoints   1206 files
experiments/c8_L10_savepoint_6c_run2/savepoints   1206 files
diff -r -q run1 run2   ->  clean
```

**Reference reproducibility: PASS (BIT-IDENTICAL).** All new fields finite;
`delp` positive; `it` labels present as `it=1`, `it=2`, `it=3` for `sp2a`/`sp2b`/
`sp2c` (i.e. the in-loop labelling works, unlike `sp3`). No performance runs.

## 19.3 Fix 2 of 2 — `grid_lonlat` / `agrid_lonlat` store-order defect (ROOT CAUSE)

`sp2b` `it=1` initially failed with `u`/`v` SIGNIFICANT DIFFERENCE. Localization
chain, coarse to fine, in the order actually executed:

1. **Per-`k` breakdown of `pk3`** showed its whole-box 6.46e-02 error is confined
   to `k=1` — see §19.6, it is a *dump artifact*, not a defect. This collapsed
   two apparent defects into one before any deeper work.
2. **Halo-input identity at the failing corner.** On pe4, every `a2b_ord4` input
   feeding the SW corner agrees at round-off: `q(2,2)` 3.4e-15, `q(0,1)` 3.2e-14,
   `q(-1,2)` 5.0e-15, `q(1,0)` 1.7e-14, `q(2,-1)` 4.1e-14 — while the corner
   *output* at (1,1) differed by 3.73e+00 (`pkc`) and 6.74e+05 (`gz`).
   ⇒ the inputs are authoritative; the corner operator itself is at fault.
3. **Solved for the extrapolation ratios instead of reading the kernel.**
   `extrap_corner` is `q1 + r*(q1-q2)` with `r = d1/(d2-d1)`, and
   `3*qout = Σ_t [(1+r_t)q1_t − r_t q2_t]` over the three stencil terms. A
   least-squares solve over 22 equations (11 levels × {`pkc`,`gz`}) gave:
   - Fortran, pe0 and pe4: `r = [0.4212796036, 0.4212796036, 0.4212796036]`,
     fit residual 3.7e-14 / 6.2e-14 relative. The three terms are geometrically
     congruent — exactly what a symmetric gnomonic panel corner requires.
   - CUDA, pe4: `[0.0932513932, −68.5250532755, 1213.2502819611]`; pe0:
     `[0.0497037855, −2.6761949036, 1.7246358257]` — both fitting at round-off.
   ⇒ **arithmetic form correct, geometry data corrupt and tile-dependent.**
4. **Probed the geometry source directly.** A standalone binary against the
   harness's own `cube::CubeGrid` (`fv3_cube_geometry.{hpp,cpp}`) returned
   exactly `0.4212796036` for all three terms on all six tiles. ⇒ the geometry
   *source* was always authoritative; corruption is between harness and kernel.
5. **The layout.** `run_small_perf_bench.cpp` filled `grid_lonlat`/`agrid_lonlat`
   **interleaved** (`idx*2 + component`) but registers them **component-slowest**
   (`st_ll = {1, ni_h, ni_h*nj_h}`), which is the correct translation of Fortran's
   `grid(isd:ied+1, jsd:jed+1, 2)`. The consumer
   (`a2b_great_circle` → `View3` component index `lo_k` / `lo_k+1`) therefore read
   lon/lat of the wrong cells, mixed together.

Fix: fill component-slowest to match the registration —
`h.grid_lonlat[idx]` / `h.grid_lonlat[idx + h.size_2d]` (same for `agrid_lonlat`).
**Numerical algorithm changed: NO** — a store-order correction only.

Why the damage was confined to the four physical panel corners: **only the
`a2b_ord4` corner extrapolation reads these arrays.** The interior, edge, and
`qx`/`qy`/`qxx`/`qyy` paths use `dxa`/`dya`. That is precisely why every interior
cell agreed at round-off while the corner values were wrong by 6 orders of
magnitude, and why seven prior input-fidelity audits — all of which checked
*values*, not *store order* — missed it.

**Important difference from the seven earlier input-fidelity fixes: this one is
NOT opt-in.** `grid_lonlat` is built unconditionally in `make_host_state`, so the
default synthetic/performance path changes too. It changes that path from wrong
to right, but the §6 `CURRENT_STATE` timings were taken before it.

Post-fix, corner values: `pkc` 3.73e+00 → ≤2.71e-14 (3.7e-12 rel); `gz`
6.74e+05 → ≤1.16e-10 (2.5e-16 rel).

**Recorded, deliberately not fixed (would exceed the 2-fix budget):**
`test_dispatch_smoke.cpp:535-538` and `:756-757` carry the *identical*
interleaved-vs-registered lon/lat mismatch. Self-contained tests with their own
tolerances, not compared against Fortran.

Fix 1 of 2 (applied in the preceding pass, carried here) was the **two-kappa
split**: the scalar `akap` driving the *Exner* variable only (`ptk`, `pk3`,
`pk3_halo`) is the authoritative single-precision `MAPL_KAPPA` =
`0.28571426868438721`, while the `cappa(i,j,k)` array driving every thermodynamic
closure is FMS double dry `2/7`. Implemented as `NumericalConfig::akap_exner` +
`--akap-exner`; `cfg.akap` stays `2/7`. Two earlier levers were **retracted**: a
`--moist-cappa` route (rejected at runtime) and a *global* `--akap` override
(fixed `pk3` but degraded `zh`/`delz`/`w`/`pkc` and would have regressed the
validated SP1 `it=1`).

## 19.4 `pk3` `k=1` dump artifact — NOT a defect

Recorded so it is not rediscovered. `nh_p_grad` does not interpolate `k==1`; both
sides overwrite the top level with `ptk`. Fortran writes `top_value` into the
`pk3` array itself, so its dump shows the corrected value. CUDA's
`set_top_kernel` (`wp_d_cuda.cu:455-458`) writes into the **arena scratch**
`pk_v`/`pk3_v` that the pressure gradient consumes, *not* back into the
registered `views.pk3` — so the dumped CUDA `pk3` at `k=1` still holds the
uncorrected interpolation. Empirically confirmed inert: after the layout fix
`u`/`v` PASS while the `pk3` `k=1` dump difference is **bit-unchanged**, so it
provably never entered the pressure gradient.

## 19.5 §3 `heat_source` / `diss_est` reconciliation — DEFECT, but INERT in-loop

| aspect | authoritative Fortran | CUDA |
|--------|----------------------|------|
| accumulate | `heat_source(i,j,k) = heat_source(i,j,k) + heat_s(i,j)`, `dyn_core.F90:858`, **inside** the acoustic loop, gated `d_con > 1.0E-5` (`d_con = 1.0` here ⇒ active) | zeroed every substep — accumulation **MISSING** |
| smooth | `del2_cubed(heat_source, cnst_0p20*da_min, ...)`, `:1311` — **after** `enddo ! time split loop` | present |
| apply to `pt` | `:1318-1350` — **after** the loop | present |

Classification: **DEFECT (MISSING accumulation), INERT for every in-loop
savepoint at every `it`.** Because the accumulated field is not read until after
the acoustic loop closes, it cannot perturb `u`, `v`, `w`, `delz`, `delp` or `pt`
at any of SP1/SP2/`sp2a`/`sp2b`/`sp2c`; only the dyn_core-*exit* `pt` can be
affected.

**This retires the claim, recorded in `CURRENT_STATE` §10 and §8, that
`heat_source`/`diss_est` was the "leading candidate for the SP1 `it=2`
divergence".** It could not have been, on source evidence alone. Per §3 of the
task it was not modified. It remains a real gap to close before any SP3/exit
comparison.

## 19.6 Results — all boundaries, c8 / R8 / 6 tiles, worst value across tiles

CUDA started from authoritative SP0, authoritative grid metrics, `--bdt 3600`,
`--sp1-boundary`, `--damp-n-sponge -1`, `--akap-exner 0.28571426868438721`.
Verdict boxes and corner handling exactly as established in §16.1/§17 (the four
cubed-sphere diagonal corner halo cells are reported separately and excluded).

**SP1 `it=1` — PASS** (unchanged, re-verified with both fixes in place)

| field | max_abs | max_rel | verdict |
|-------|---------|---------|---------|
| `delpc` | 1.82e-12 | 1.82e-16 | ROUND-OFF |
| `ptc` | 3.55e-15 | 1.44e-16 | ROUND-OFF |
| `gz` | 2.91e-10 | 6.34e-16 | ROUND-OFF |
| `pkc` | 2.91e-11 | 2.91e-16 | ROUND-OFF |
| `uc` | 9.28e-14 | 6.44e-15 | ROUND-OFF |
| `vc` | 9.92e-14 | 6.89e-15 | ROUND-OFF |
| `ws3` | 9.47e-17 | 4.97e-13 | SMALL |

**SP2 `it=1` — PASS, now 14/14 fields.** The ten previously reported fields
remain ROUND-OFF (§4 of `CURRENT_STATE`), and the four new ones are ROUND-OFF
as well: `crx`, `cry`, `xfx`, `yfx`. `update_dz_d`'s input contract is closed.
`w ≡ 0` on both sides here remains vacuous, so honest `d_sw` coverage is still
9/10 — but see `sp2a`, where `w` becomes non-vacuous for the first time.

**`sp2a` `it=1` (post-`Riem_Solver3`) — PASS**

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

**This is `Riem_Solver3`/`SIM1_solver`'s first authoritative coverage at any
`it`**, and `update_dz_d`'s first output coverage (`ws`). §18.6's "`w` validation
NOT REACHABLE from existing data" is resolved by the new instrumentation.

**`sp2b` `it=1` (post-`nh_p_grad`) — PASS**

| field | max_abs | max_rel | verdict |
|-------|---------|---------|---------|
| `u` | 2.26e-13 | 5.81e-13 | SMALL |
| `v` | 2.03e-13 | 5.22e-13 | SMALL |

`nh_p_grad` is validated at `it=1`. The `max_rel` ranking is driven by pe4, whose
`u` field genuinely maxes near 0.39, so the dynamic-range floor is small there;
absolute agreement is 2e-13 on a field of order 1e-1 … 1e+1.

**`sp2c` `it=1` (end of substep, state carried into `it=2`) — PASS**

| field | max_abs | max_rel | verdict |
|-------|---------|---------|---------|
| `delp` | 3.64e-12 | 3.64e-16 | ROUND-OFF |
| `pt` | 7.11e-15 | 2.88e-16 | ROUND-OFF |
| `delz` | 3.64e-11 | 1.22e-15 | ROUND-OFF |
| `u` | 2.26e-13 | 5.81e-13 | SMALL |
| `v` | 2.03e-13 | 5.22e-13 | SMALL |
| `w` | 4.74e-14 | 8.36e-12 | SMALL |

Note the CUDA comparand for `sp2c` at `it=1` is the `nh_pressure` stage dump.
That is valid, established from source rather than assumed: at `it < n_split`
nothing after `nh_p_grad` writes `u,v,w,delz,delp,pt` — the E11 `mpp_get_boundary`
write-back is `it==n_split`-gated (`:1142`/`:1145-1152`), the omega block is
`last_step`-gated, and `heat_source` is applied only after the loop (§19.5).

**SP1 `it=2` — PASS** (was: every field SIGNIFICANT DIFFERENCE, first field `uc`
at pe4 `max_rel` 1.83e+00)

| field | max_abs | max_rel | verdict |
|-------|---------|---------|---------|
| `delpc` | 3.64e-12 | 3.63e-16 | ROUND-OFF |
| `ptc` | 1.07e-14 | 4.31e-16 | ROUND-OFF |
| `gz` | 4.66e-10 | 1.02e-15 | ROUND-OFF |
| `pkc` | 2.91e-11 | 2.91e-16 | ROUND-OFF |
| `uc` | 1.63e-13 | 1.16e-14 | ROUND-OFF |
| `vc` | 1.94e-13 | 1.38e-14 | ROUND-OFF |
| `ws3` | 9.47e-17 | 5.17e-13 | SMALL |

The pe2/pe3/pe4-vs-pe0/pe1/pe5 tile asymmetry recorded in §15 is **gone** — as
expected, since it was the signature of a tile-dependent geometry corruption.

**Run health.** Every `[FV3DBG]` line reports `nan/inf=0/`; no NONFINITE or FAIL
lines anywhere; harness exit code went 1 → **0**. The "`it=3` NONFINITE on tiles
2/3/4" symptom recorded in §14 no longer appears. Per task §11 (`no SP3 it=3
debugging`) `it=3` was **not** separately compared, so this is a health
observation, **not** a validation claim for `it=3`.

**R8 ctest: 22/22 PASS** with both fixes built in.

## 19.7 §12 FINAL REPORT

- New Fortran savepoints (`sp2a`/`sp2b`/`sp2c`, `it`-labelled): **PASS**
- Reference reproducibility (6C run1 vs run2, 1206 files): **PASS, BIT-IDENTICAL**
- SP2 additional `crx`/`cry`/`xfx`/`yfx` coverage: **PASS** (14/14; closes
  `update_dz_d`'s input contract)
- `heat_source`/`diss_est` source reconciliation: **DEFECT (MISSING accumulation)
  but INERT** for all in-loop savepoints — retires it as the SP1 `it=2` candidate
- SP1 `it=1`: **PASS** · SP2 `it=1`: **PASS** · post-`Riem_Solver3` `it=1`:
  **PASS** · post-`nh_p_grad` `it=1`: **PASS** · end-substep `it=1`: **PASS** ·
  SP1 `it=2`: **PASS**
- First divergent boundary: **`sp2b` `it=1` (post-`nh_p_grad`)** — now resolved
- First divergent field: **`u`/`v`** (via B-grid `pkc`/`gz` panel corners)
- First divergent operator: **`a2b_ord4` corner extrapolation inside `nh_p_grad`**
- Root cause: **harness `grid_lonlat`/`agrid_lonlat` filled interleaved but
  registered component-slowest** — the only consumer is the corner
  extrapolation, hence corner-only, tile-dependent damage
- Fixes applied: **2** (`akap_exner` Exner-scalar split; lon/lat store order)
- Numerical algorithm changed: **NO**
- Gate C2: **BLOCKED** (SP3/exit not yet compared; E11 and the `heat_source`
  accumulation still MISSING)
- Next resume point: compare the dyn_core exit boundary (`sp3` at `it=3`, which
  does exist), which requires E11 and the post-loop `heat_source` accumulation to
  be implemented first.

SP1 it=2 PASS — CONTINUE PROMPT 6B

---

# 20. Prompt 6B — acoustic-loop closure + dyn_core-exit boundary (this pass)

Scope as assigned: validate `it=2` and `it=3` on the existing 6C savepoints,
implement E11 and the `heat_source`/`diss_est` behaviour if their source
contracts are unambiguous, then compare the dyn_core exit (`sp3`, `it=3`).
Budget: 3 proven fixes. No new fixture run (the 6C reference was reused as-is),
c8 only, R8 only, no performance work.

## 20.1 `it=2` and `it=3` in-loop chain — all 15 boundaries PASS

Same configuration as §19 (c8 / npz=10 / n_split=3 / k_split=1 / R8 / 6 MPI
ranks = 6 tiles, authoritative SP0 state, grid metrics, `bdt=3600`, `rdgas`,
`n_sponge=-1`, `edge_*`, `akap_exner`). Worst value across all 6 tiles; verdict
boxes are the owned/defined boxes; the four diagonal corner halo cells are
excluded and reported separately as always.

| boundary | it=1 | it=2 | it=3 |
|----------|------|------|------|
| SP1 (post-C-grid predictor) | PASS | PASS | PASS |
| SP2 (post-`d_sw`) | PASS (ROUND-OFF) | PASS | PASS |
| `sp2a` (post-`Riem_Solver3`) | PASS | PASS | PASS |
| `sp2b` (post-`nh_p_grad`) | PASS | PASS | PASS |
| `sp2c` (end of substep) | PASS | PASS | PASS |

Every one of the 15 boundaries grades at or better than SMALL NUMERICAL
DIFFERENCE (≤1e-8); SP2 `it=1` is ROUND-OFF SCALE. Re-verified with the
post-fix build (the `n_con`/`cv_air` fixes below touch only the post-loop tail,
so no in-loop change was expected, and none was observed). This is the first
authoritative validation of `it=2` beyond SP1 and of `it=3` at any boundary.

## 20.2 Fix 1 of 3 — E11 staggered seam reconciliation: IMPLEMENTED

**Stale line numbers corrected.** `CURRENT_STATE.md` recorded the E11 call at
`dyn_core.F90:1142` with write-back `:1145-1152`. Those are **stale**. The
current authoritative lines in the pinned source are:

- call: `dyn_core.F90:1178` —
  `call mpp_get_boundary(u, v, domain, ebuffery=ebuffer, nbufferx=nbuffer, gridtype=DGRID_NE)`
- write-back: `:1181-1188` —
  `u(is:ie, je+1, k) = nbuffer(i,k)` and `v(ie+1, js:je, k) = ebuffer(j,k)`
- guard: `it == n_split .and. grid_type < 4 .and. .not. nested`

Semantics: this is **not** a halo exchange. It is a ring-**0** reconciliation of
the single shared panel seam line that two neighbouring tiles both compute
redundantly, so that the two tiles agree bitwise rather than to round-off. Only
`u`'s north line and `v`'s east line are written; nothing outside those lines
changes.

**Implementation — minimum general extension, no redesign.** Three small,
general capabilities were added to the existing halo infrastructure rather than
a new communication path:

1. `HaloGroupSpec::ring_begin` (`fv3_gpu_halo_exchange.hpp`) — previously every
   group implicitly began at ring 1. `ring_begin = 0` with `halo_width = 0`
   expresses "the seam line itself, no halo rings", which is exactly
   `mpp_get_boundary`'s footprint. The pack/unpack kernels now take
   `ring_begin`; `halo_ring_count`/`halo_ring_ordinal` were added so the
   existing ring iteration works unchanged for the ring-1 groups.
2. `HaloFieldSpec::dest_edges` — a bitmask restricting the unpack to specific
   edges. E11 writes only north (for `u`) and only east (for `v`).
3. A new group `FV3_GPU_H_DWIND_SEAM` in `fv3_gpu_communication.h`, with
   `#define FV3_GPU_H_GROUP_LAST FV3_GPU_H_DWIND_SEAM` so the ten existing
   `FV3_GPU_H_DWIND_NEXT` sentinel sites became `FV3_GPU_H_GROUP_LAST` and stay
   correct as groups are added. Table entry: `ring_begin=0`, `halo_width=0`,
   fields `{u, stagger_north, slot_x, source v, 1<<north}` and
   `{v, stagger_east, slot_y, source u, 1<<east}` — note the **cross-component
   source**, which is what `gridtype=DGRID_NE` means and what makes this
   distinct from an ordinary vector halo.

Call site: `fv3_gpu_runtime.cpp`, immediately after the `nh_pressure` stage,
gated `it == n_split`, followed by `FV3_GPU_DBG_STAGE("seam_reconcile")` so the
seam-reconciled state is separately dumpable.

**Verification — bitwise map, not a tolerance check.** A tolerance comparison
cannot validate E11, because its whole effect is ~1e-13: `sp2c` `it=3` grades
SMALL with *or* without it. So the map was verified directly
(`tools/savepoints/probe_e11_map.py`): for all 12 `(tile, field)` pairs, the
CUDA post-E11 seam line matches the authoritative donor line with
**max|resid| = 0.000e+00**, against a genuine max|delta| of 3.9e-14…3.9e-13
(i.e. the lines really did differ before, and now match exactly). Footprint
measured: at `it=3` only `u(is:ie,je+1,:)` and `v(ie+1,js:je,:)` change, with
**0.0** change off-line; at `it=2` **0 of 80** cells change, confirming the
`it==n_split` gate.

**First boundary at which E11 is observable:** `sp2c` at `it = n_split = 3`.
It runs after both the SP1 (`:637`) and SP2 (`:871`) captures and after
`sp2b`/`sp2c` at `it < n_split`, and never executes at `it=1` or `it=2` at all.

**Numerical algorithm changed: NO** (adds a missing authoritative
reconciliation; no operator arithmetic touched).

**Regression consequence, by design.** Adding a tenth resolve point made
`fv3_dispatch_smoke` fail loudly (`FV3_GPU_UNSUPPORTED_CONFIGURATION`, naming
`H_DWIND_SEAM`) — that harness spells out `kRuntimeResolvedGroups[]` precisely
so a new resolve point cannot be silently ignored. `FV3_GPU_H_DWIND_SEAM` was
added to that list; it is a weak-sense declaration (lets the integrated path
execute, asserts nothing about seam values). Phase E, which runs a real six-tile
exchange, is where the seam semantics are actually exercised.

## 20.3 Fix 2 of 3 — `heat_source` / `diss_est` accumulation: IMPLEMENTED

**Source contract, documented before implementing.**

- Accumulated field: `heat_source` (with `diss_est` as its diagnostic twin).
- Formed inside `d_sw` (`sw_core.F90`) and accumulated **inside** the acoustic
  loop at `dyn_core.F90:858`.
- Applied **after** the loop: `del2_cubed` smoothing at `:1311`, then the `pt`
  update at `:1318-1350`.
- Post-loop gate: `if ( n_con /= 0 .and. flagstruct%d_con > 1.e-5 )`
  (`:1309`) — so `d_con` does control activation; it is 1.0 here.
- Smoothing strength: `nf_ke = min(3, flagstruct%nord+1)` (`:1310`), passed to
  `del2_cubed(heat_source, cnst_0p20*da_min, ..., nf_ke, ...)`; `del2_cubed`
  itself further caps `ntimes = min(3, nmax)` at `:2400`.
- Consuming fields: `pt` **and** `pkz` — the nonhydrostatic branch rewrites
  `pkz` at every heated level (`:1348`) before incrementing `pt` (`:1349`).

CUDA previously **zeroed** `heat_source` every substep, so only the final
substep's contribution survived.

**Implementation.** A persistent device accumulator was added
(`fv3_gpu_context.hpp`: `fv_real *heat_accum` + `heat_accum_bytes`; new
`runtime/fv3_gpu_heat_accumulator.{hpp,cu}`; added to `CMakeLists.txt`), set up
once pre-loop in `fv3_gpu_runtime.cpp` and accumulated per substep. Persistent
GPU residency preserved; no host/device transfer added. The accumulate kernel
iterates **explicit inclusive index ranges in the view's own index space** with
a hard bounds guard that returns `FV3_GPU_INVALID_ARGUMENT` rather than writing
out of range — worth keeping: the first draft derived ranges from the field's
allocation extents and silently ran one cell past the owned box on the
staggered fields.

**Numerical algorithm changed: NO.**

## 20.4 Fix 3 of 3 — post-loop application fidelity (`n_con`, `cv_air`)

Two genuine defects in the *application* of the accumulated heat, both found by
comparing the exit boundary against the authoritative source rather than by
inspection:

**(a) `cv_air` was cp_air.** `wp_d_contract.cpp` and `wp_d_contract_cuda.cpp`
computed `cv_air = cfg.rdgas / cfg.akap`, which *is* `cp_air` (≈1004.7), where
`dyn_core.F90:323` requires `cv_air = cp_air - rdgas` (≈717.6). Every `dtmp` at
`:1349` was therefore a factor `cp_air/cv_air = 1/(1-akap) = 1.4` **too small**.
Corrected in both twins to `cfg.rdgas / cfg.akap - cfg.rdgas`. Note the
hydrostatic branch's `cp_air = cv_air + rdgas` was written against the *correct*
definition and becomes right again automatically.

**(b) `cfg.n_con` was hardcoded to 1; authoritative value is `npz` = 10.**
`n_con` is **derived**, per `dyn_core.F90:402-412`:

```
if ( convert_ke .or. vtdm4 > 1.E-4 ) then   n_con = npz
else if ( d2_bg_k1 < 1.E-3 )           then n_con = 0
else if ( d2_bg_k2 < 1.E-3 )           then n_con = 1
else                                        n_con = 2
```

The in-source comment justifying the hardcoded `1` actually justified
**`nf_ke == 1`** (the `corner_edge_service` stub) and then set `n_con` — two
independent controls conflated. `n_con` bounds how many *levels* of the `pt`
update run (`:1338 do k=1,n_con`); `nf_ke` bounds how many *del2_cubed passes*
`heat_source` receives first (`:1310-1311`). Only `nf_ke` touches
`corner_edge_service`. Now derived from `h.tr_flags`. Authoritative
`vtdm4 = 0.01` confirmed twice independently: the fixture's `input.nml:23` /
`fv3.j:386`, and `geos_production_flags()`'s `f.vtdm4 = 0.01`
(`fv3_damping_translation.cpp:277`, sourced from `FV_StateMod.F90:744`) —
hence `n_con = npz`.

The `n_con` fix alone closed the exit `pkz` divergence: per-level max|diff| went
from `5.33e-15, 6.68, 7.99, 8.43, 8.31, 7.74, 6.76, 5.36, 3.54, 1.26` (CUDA had
written only `k=1`) to **ROUND-OFF SCALE, worst 1.528e-13 / 5.777e-15** across
all 6 tiles.

`apply_diffusive_heating_nh` (`wp_d_operators.cpp:549-586`) was audited
line-by-line against `:1338-1353` and matches exactly — the `delt` ladder
(`0.1*` at `k==1`, `0.5*` at `k==2`), the `pkz` recomputation, the `dtmp`
form, and the `sign(min(delt,|dtmp|),dtmp)/pkz` increment. `n_con` and `cv_air`
were the only levers.

**Numerical algorithm changed: NO** for both.

## 20.5 `sp3` / dyn_core exit — field-by-field classification

Worst value across all 6 tiles. Classification per the assigned taxonomy.

| field | max_abs | max_rel | verdict | classification |
|-------|---------|---------|---------|----------------|
| `u` | 4.086e-13 | 4.110e-13 | SMALL | NON-VACUOUS VALIDATED |
| `v` | 3.855e-13 | 4.795e-13 | SMALL | NON-VACUOUS VALIDATED |
| `w` | 9.509e-14 | 1.413e-11 | SMALL | NON-VACUOUS VALIDATED |
| `delz` | 3.638e-11 | 1.215e-15 | ROUND-OFF | NON-VACUOUS VALIDATED |
| `delp` | 1.091e-11 | 1.090e-15 | ROUND-OFF | NON-VACUOUS VALIDATED |
| `pe` (owned box) | 4.366e-11 | 4.358e-16 | ROUND-OFF | NON-VACUOUS VALIDATED |
| `peln` | 1.776e-15 | 1.543e-16 | ROUND-OFF | NON-VACUOUS VALIDATED |
| `pk` | 2.487e-14 | 9.266e-16 | ROUND-OFF | NON-VACUOUS VALIDATED |
| `pkz` | 1.528e-13 | 5.777e-15 | ROUND-OFF | NON-VACUOUS VALIDATED |
| `ua` | 3.268e-13 | 2.214e-14 | ROUND-OFF | NON-VACUOUS VALIDATED |
| `va` | 2.807e-13 | 1.765e-14 | ROUND-OFF | NON-VACUOUS VALIDATED |
| `pt` | 6.397e-05 | 2.592e-06 | **SIGNIFICANT** | explained, §20.6 |
| `omga` | 4.740e-02 | 1.003e+00 | **SIGNIFICANT** | **NOT WRITTEN**, §20.7 |
| `pe` halo ring | 1.002e+05 | — | — | **NOT WRITTEN**, §20.8 |

No field is VACUOUS at the exit. **`w` and `delz` receive genuinely
non-vacuous validation here** — `w ≡ 0` on both sides at SP2 made that
agreement vacuous, but by the exit `w` carries real nonhydrostatic structure.

Eleven of thirteen exit fields validate. Neither remaining discrepancy is
unexplained, but neither is closed.

## 20.6 Remaining `pt` residual — localized to `nf_ke = 1` vs authoritative 3

Elimination, in order:

- `pkz` is computed from the **pre-increment** `pt`, `delp`, `delz` (`:1348`).
  It now agrees at round-off at every level ⇒ the `pt` *entering* the tail
  agrees at round-off, and so do `delp` and `delz`. The residual is created
  inside the tail, not carried into it.
- `cv_air` is now authoritative (§20.4a).
- **`delt_max` PROVEN inert**, not assumed. Measured max |dtmp| across all six
  tiles = **1.0305e-03**, against the smallest authoritative bound
  `delt(k=1) = 0.1*|bdt*delt_max| = 0.720` even at the non-authoritative
  `delt_max = 0.002` — 698× headroom. The limiter never engages, so the known
  `delt_max` 0.002-vs-1.0 discrepancy cannot contribute here.

Everything left is in `heat_source` itself, and the signature says
**under-smoothing, not mis-amplitude**:

- Fortran's total heating increment (`sp3 pt` − `sp2c it=3 pt`) maxes at
  2.0286e-05 on pe0/1/3/4 and 1.1433e-05 on pe2/5.
- The CUDA/Fortran per-cell ratio is **not** a constant (median 1.68,
  std 44.6) — an amplitude error would give a constant.
- Per level, CUDA's increment std is **3.7–4.9× Fortran's**, with correlation
  only **0.72–0.82** — the CUDA field is rougher and only loosely correlated,
  which is what running fewer smoothing passes looks like.

**Cause: `cfg.nf_ke = 1` where the authoritative value is
`min(3, nord+1) = 3`** at production `nord = 2`. Three passes cannot run yet:
passes 0 and 1 have `nt = ntimes-1-pass > 0`, which reaches
`post_loop_heating`'s `x.corner_service(...)` `copy_corners` call, and the
production dispatch table binds `corner_edge_service` to
`corner_service_not_implemented` (`fv3_gpu_production_dispatch.cu:102`,
defined `:8-20`) — a hard `FV3_GPU_UNSUPPORTED_CONFIGURATION`. Closing this
needs (a) a real `copy_corners` corner service for `heat_source`, and (b) in
the CUDA twin, splitting the del2 pass to insert the **direction-2** corner
copy between `launch_del2_zonal_flux_cuda` and
`launch_del2_meridional_flux_cuda` (`del2_cubed` calls `copy_corners` twice per
pass, once per direction, `:2360-2469`). That is a distinct fourth source fix.

`nf_ke` remains 1, with the gap recorded in-source at the assignment site. It
**under-smooths** `heat_source` before the `pt` update; it does not skip levels
of it.

## 20.7 `omga` — NOT WRITTEN (a diagnostic, never finalized in CUDA)

Proven, not inferred: `it=3_dyn_core_exit__omga.bin` is **bit-identical** to
`it=3_cgrid_vertical_solver__omga.bin` on all six tiles (`max|d| = 0.000e+00`).
CUDA's last writer of `omga` is `Riem_Solver_c`, which uses it as its `w3`
argument — Fortran aliases the same array as `c_sw`'s `wc` at `:518`, so this is
a faithful *intermediate*, but it is not the exit value.

CUDA implements **neither** of the two stages that produce the authoritative
exit `omga`:

- in-loop finalization: `:801` `omga = delp`, then `:842`
  `omga = omga*(xfx(i,j,k)-xfx(i+1,j,k)+yfx(i,j,k)-yfx(i,j+1,k))*rarea*rdt`;
- the `last_step` block `:1204-1237`. `use_old_omega` defaults `.true.`
  (`fv_arrays.F90:465`) and is **not** overridden in the fixture namelist, so
  the authoritative path is `omga(i,j,k) = (pe(i,k+1,j) - pem(i,k+1,j))*rdt`
  followed by `call adv_pe(ua, va, pem, omga, ...)`. This needs `pem`, which is
  allocated and filled pre-loop at `:505-513`
  (`pem(i,1,j)=ptop`, `pem(i,k+1,j)=pem(i,k,j)+delp(i,j,k)`).

`adv_pe` (`:1535-1636`) additionally requires `a2b_ord2` (**0** references in
the CUDA port), `gridstruct%en1`/`en2` (**0** references), and `ec1`/`ec2`
(2 references) — new operators *plus* authoritative grid metrics that the
current `--grid-metrics` export does not carry. That is a distinct fifth source
fix, and the largest of the two remaining.

`omga` is a diagnostic: nothing inside `dyn_core` consumes it, so this does not
affect any prognostic field above.

## 20.8 `pe` halo ring — NOT WRITTEN (comparator box corrected)

`pe` is declared `pe(is-1:ie+1, 1:npz+1, js-1:je+1)`, one ring wider than the
owned box. The CUDA port writes only the owned box; the surrounding ring still
holds the harness's 1e-15 poison sentinel, so including it reported
max|diff| ≈ 1.0e+05 — a NOT-WRITTEN gap misreported as a numerical
disagreement. `SP3_DEFINED_BOX["pe"]` in
`tools/savepoints/compare_cuda_fortran.py` was narrowed to `(1, NI, 1, NI)`
with the ring classified separately. The owned box is ROUND-OFF on all six
tiles.

## 20.9 §11 regression

- **R8 CTest: 22/22 PASS** (after the `kRuntimeResolvedGroups[]` update of
  §20.2; the first run was 21/22 with the designed loud failure).
- **Default synthetic smoke: PASS** — `RUN exit=0`,
  `=== 3A.11 CUDA perf benchmark: PASS ===`,
  `CSV,cuda_gpu,8,8,10,6,R8,resident,1,156.388463,78.194232,A100`, all tiles
  finite, `delp` positive over the owned box.
- **No performance run** (out of scope by instruction).

**Performance baseline status: STALE — RETEST REQUIRED**, now for two
independent reasons. (1) The `grid_lonlat` store-order fix of §19.3 changed the
default synthetic path from wrong to correct geometry layout. (2) The `n_con`
derivation of §20.4b also changes the default path — `geos_production_flags()`
sets `vtdm4 = 0.01`, so `n_con` derives to `npz` there too, and the post-loop
heating now runs at all ten levels instead of one. The §6 CURRENT_STATE timings
predate both.

## 20.10 Fix budget and HARD STOP

Three proven fixes used, all inside the assigned categories:

1. **E11** seam reconciliation (§20.2).
2. **`heat_source`/`diss_est` accumulation** (§20.3).
3. **Post-loop application fidelity** — `n_con` + `cv_air` (§20.4).

Two further **distinct** source fixes are required to close the exit boundary,
both fully diagnosed with authoritative citations and measured evidence but
neither implementable inside the budget:

4. **`nf_ke = 3`** — needs a `copy_corners` corner service for `heat_source`
   plus a del2 restructuring in the CUDA twin (§20.6).
5. **`omga`** — needs the in-loop finalization, the `pem` capture, and
   `adv_pe`, which in turn needs `a2b_ord2`, `en1`/`en2`, `ec1`/`ec2` (§20.7).

**§9 HARD STOP condition (3) fires: more than 3 distinct source fixes are
required.** Gate C2 remains BLOCKED. Both remaining exit discrepancies are
*explained*, not closed.

## 20.11 §12 FINAL REPORT

- `it=2` chain {SP1, SP2, `sp2a`, `sp2b`, `sp2c`}: **PASS**
- `it=3` chain {SP1, SP2, `sp2a`, `sp2b`, `sp2c`}: **PASS**
- E11: **IMPLEMENTED** (bitwise map verified, max|resid| 0.000e+00, 12/12 pairs;
  first observable at `sp2c` `it=3`)
- `heat_source`/`diss_est`: **IMPLEMENTED** (accumulation + application
  fidelity)
- First divergent boundary: **`sp3` / dyn_core exit**
- First divergent field: **`pt`** (6.397e-05), then **`omga`** (NOT WRITTEN)
- Root cause: `pt` ⇒ `nf_ke = 1` vs authoritative `min(3,nord+1) = 3`, blocked
  on `corner_service_not_implemented`; `omga` ⇒ the in-loop finalization,
  `pem` capture and `adv_pe` are all absent
- Fixes applied: **3**
- SP3 / dyn_core exit: **FAIL** (11 of 13 fields validated)
- Non-vacuous exit coverage: 11/13, including `w` and `delz`
- Numerical algorithm changed: **NO**
- R8 regression: **PASS 22/22** · Synthetic smoke: **PASS**
- Performance baseline: **STALE — RETEST REQUIRED**
- Gate C2: **BLOCKED**
- Next resume point: implement the `heat_source` `copy_corners` corner service
  to enable `nf_ke = 3`, then the `omga` finalization/`adv_pe` subsystem.

HARD STOP — more than 3 distinct source fixes are required to close the dyn_core exit boundary

---

# 21. Prompt 6B — bounded dyn_core-exit reconciliation of `pt` / `nf_ke`

Scope of this pass, as assigned: the exit `pt` residual ONLY, via the
`nf_ke` / `heat_source` / `del2_cubed` / corner-service path. `omga` was
explicitly out of scope and is untouched. Configuration unchanged from §19/§20:
c8 / npz=10 / n_split=3 / k_split=1 / R8 / 6 MPI ranks, authoritative SP0 state,
authoritative grid metrics, `bdt=3600`, `ptop=100`, `rdgas=287.0523045054`,
`n_sponge=-1`, authoritative `edge_*`, `akap_exner`.

**Outcome: exit `pt` PASSES at ROUND-OFF SCALE on all six tiles.** §20.6's
diagnosis (under-smoothing from `nf_ke = 1` vs authoritative 3) is confirmed
by construction: implementing the missing capability and restoring the
authoritative value moved `pt` from `max_abs` 6.397e-05 to 1.421085e-14, a
factor of ~4.5e9.

## 21.1 Authoritative `nf_ke` sequence (§1 reconciliation)

`nf_ke` is derived, not configured: `dyn_core.F90:1310`

```fortran
nf_ke = min(3, flagstruct%nord+1)
```

and is passed straight into

```fortran
call del2_cubed(heat_source, cnst_0p20*gridstruct%da_min, gridstruct, &
                domain, npx, npy, npz, nf_ke, bd)
```

with `cnst_0p20 = 0.20d0` (`:147`). Production `nord = 2` ⇒ **`nf_ke = 3`**.

`del2_cubed` (`dyn_core.F90:2360-2469`) for `nmax = 3`:

- `ntimes = min(3, nmax) = 3` (`:2400`), `r3 = 1./3.`
- **one** `mpp_update_domains(q, domain, complete=.true.)` BEFORE the pass loop
- `do n = 1, ntimes` with `nt = ntimes - n` ⇒ **`nt = 2, 1, 0`**

Per pass, in order:

| step | operation | direction | in | out | region |
|------|-----------|-----------|----|-----|--------|
| 1 | four `r3` corner averages, gated on `sw/se/ne/nw_corner` | — | `q` | `q` | the four physical panel corners |
| 2 | `if (nt>0) copy_corners(q(isd,jsd,k), npx, npy, 1, ...)` (`:2419`) | **dir=1 (XDir)** | `q` | `q` | rotated 3×3 corner blocks |
| 3 | `fx = del6_v*(q(i-1,j)-q(i,j))` | x | `q` | `fx` | `[is-nt, ie+1+nt] × [js-nt, je+nt]` |
| 4 | `if (nt>0) copy_corners(q(isd,jsd,k), npx, npy, 2, ...)` (`:2437`) | **dir=2 (YDir)** | `q` | `q` | rotated 3×3 corner blocks |
| 5 | `fy = del6_u*(q(i,j-1)-q(i,j))` | y | `q` | `fy` | `[is-nt, ie+nt] × [js-nt, je+1+nt]` |
| 6 | `q += cd*rarea*(fx(i,j)-fx(i+1,j)+fy(i,j)-fy(i,j+1))` | — | `fx,fy` | `q` | `[is-nt, ie+nt] × [js-nt, je+nt]` |

The decisive detail, resolved from source rather than assumed:
**`copy_corners`'s `ng` is the compile-time PARAMETER imported at
`tp_core.F90:52` (`use fv_mp_mod, only: ng`), not `bd%ng` and not `nt`.** It is
3. The rotated block is therefore always 3×3 regardless of which pass is
running, and a corner-service request must NOT forward its write-band `nt` as a
block size. This is why `CornerServiceRequest::halo_width` is documented at the
new service as *the requester's write band*, not the corner-block extent.

`copy_corners` rotation formulas (`tp_core.F90:247-322`), `if (nested) return`:

- dir=1 (XDir): sw `q(i,j)=q(j,1-i)`; se `q(i,j)=q(npy-j,i-npx+1)`;
  ne `q(i,j)=q(j,2*npx-1-i)`; nw `q(i,j)=q(npy-j,i-1+npx)`
- dir=2 (YDir): sw `q(1-j,i)`; se `q(npy+j-1,npx-i)`;
  ne `q(2*npy-1-j,i)`; nw `q(j+1-npx,npy-i)`

Index-base equivalence used throughout: Fortran `nt = ntimes - n` for
`n = 1..ntimes` is identical to the CUDA `nt = ntimes - 1 - pass` for
`pass = 0..ntimes-1`.

## 21.2 CUDA state before the fix (§2 reconciliation)

| Fortran operation | CUDA equivalent | status before |
|---|---|---|
| `ntimes = min(3,nmax)` pass loop | `dispatch_params.diffusion_passes`, driven by `cfg.nf_ke` (`fv3_gpu_runtime.cpp`) | MATCH |
| `nt = ntimes - n` | `nt = ntimes - 1 - pass` (`wp_d_contract_cuda.cpp:167`) | MATCH |
| pre-loop `mpp_update_domains` | `H_HEAT` required on `pass == 0` only | MATCH |
| four `r3` corner averages | `launch_average_del2_corners_cuda` | MATCH |
| `copy_corners(..., 1, ...)` `:2419` | request built (`Axis::x`) but service was `corner_service_not_implemented` | **UNSUPPORTED** |
| `fx` loop | `launch_del2_zonal_flux_cuda` | MATCH |
| `copy_corners(..., 2, ...)` `:2437` | **absent from both the CUDA and host twins** | **MISSING** |
| `fy` loop | `launch_del2_meridional_flux_cuda` | MATCH |
| `q` update | `launch_del2_update_cuda` | MATCH |
| `nf_ke = min(3, nord+1)` | `cfg.nf_ke = 1`, hardcoded in the bench | **WRONG VALUE** |

Two observations made the fix small. First, the CUDA del2 was already three
separate launches per pass, so nothing needed restructuring to insert a
rotation between the flux phases — only the host twin was monolithic. Second,
`DiffusionPhase::y_corner_copy`, `diffusion_corner_request`'s acceptance of
`Axis::y`, and `validate_corner_request`'s axis-agnostic validation all already
existed; the contract had anticipated this operation and only the
implementation and the second call site were missing.

The whole defect was latent and mutually masking: with `nf_ke` pinned to 1 there
is a single pass, `nt = 0`, and **both** rotations are skipped, so neither the
fail-stub nor the missing `:2437` site could ever be reached.

## 21.3 Fixes applied (three, §6 budget)

**Fix 1 — real corner service.** `corner_edge_service_cuda` in
`cuda/dyn_core/runtime/fv3_gpu_production_dispatch.cu` replaces
`corner_service_not_implemented`. It is a thin adapter, not a new subsystem:
it translates `CornerServiceRequest` into `dgrid::TransportBounds` and calls the
**pre-existing, already-exercised** `dgrid::launch_copy_corners_cuda`
(`wp_b_cuda.cu:681`, kernel at `:94`, a direct port of `tp_core.F90:257-318`),
which existing WP-B call sites already drive at both `dir=1` and `dir=2`. Per
§3's reuse requirement, no ad-hoc `heat_source`-only topology was created.

Accepted surface, explicitly refused otherwise: `operation = copy`,
`orientation = scalar`, `stagger = A`, `axis ∈ {x, y}`. Preconditions checked
and refused with a specific message rather than silently mis-executing: rank-3
`fv_real` field; `lower[2] == 0` (the kernel derives its level from
`blockIdx.z` and indexes with that absolute `k`); field declared over the full
`isd..ied / jsd..jed` box; borrowed compute stream. Device-resident, no H2D or
D2H, no device sync.

**Fix 2 — the missing `:2437` rotation**, in both twins.
`wp_d_contract_cuda.cpp` now issues an `Axis::y` /
`DiffusionPhase::y_corner_copy` request between `launch_del2_zonal_flux_cuda`
and `launch_del2_meridional_flux_cuda`, gated `nt > 0` exactly as Fortran gates
it. The host twin required splitting the monolithic `del2_pass` into
`del2_zonal_flux` / `del2_meridional_flux` / `del2_update`
(`wp_d_operators.{hpp,cpp}`) so the same interleave is expressible;
`del2_pass` / `del2_pass_sg` are retained as wrappers over the three phases and
are **bit-identical** to the previous implementations, because the flux phases
read only unmodified `q` and the update reads only `fx`/`fy` — the old per-level
interleaving carried no cross-phase dependence. `wp_d_contract.cpp`'s host
contract uses the phase sequence directly.

**Fix 3 — authoritative `nf_ke`.** `run_small_perf_bench.cpp` drops
`cfg.nf_ke = 1` and derives

```cpp
cfg.nf_ke = (int32_t)std::min(3, cfg.nord + 1);   // dyn_core.F90:1310
```

placed **after** every `cfg.nord` override so the derivation is respected in
all damping modes: production `nord=2` ⇒ 3, `nord=0` ⇒ 1. Not hardcoded to 3.

**Numerical algorithm changed: NO** for all three. Fix 1 makes a previously
unreachable authoritative operation reachable; Fix 2 restores an omitted
authoritative call site; Fix 3 restores an authoritative derived value.

## 21.4 §5 pass-by-pass verification

No authoritative intermediate `heat_source` capture exists, and §5 permits
source reconciliation instead of new Fortran instrumentation ("If existing exit
`pt` is enough to prove the fix, prefer that"). Verification is therefore:

- **Sequencing** — verified against §21.1 step-by-step; the CUDA order is
  average → dir=1 → fx → dir=2 → fy → update, with both rotations gated `nt>0`.
- **Corner values** — the rotation is the same device kernel WP-B already uses
  in validated transport paths, at both directions, so the corner arithmetic is
  not new code on this path.
- **Band shrinkage** — `halo_band = nt` per pass gives 2, 1, 0, matching
  Fortran's `nt = ntimes - n`.
- **End-to-end** — the exit `pt` result below is a 4.5e9× residual reduction
  driven by exactly this change, which no partially-correct sequencing would
  produce.

## 21.5 §7 exit `pt` — VALIDATED

`sp3` / `it=3` / `dyn_core_exit`, worst-of-6-tiles, defined box:

| tile | classification | max_abs | max_rel (guarded) | rms | mismatch | max-err loc |
|------|----------------|---------|-------------------|-----|----------|-------------|
| pe0 | ROUND-OFF SCALE | 1.421085e-14 | 5.756846e-16 | 2.522919e-15 | 422/640 | (7, 5, 0) |
| pe1 | ROUND-OFF SCALE | 1.421085e-14 | 5.756846e-16 | 2.575143e-15 | 424/640 | (7, 4, 0) |
| pe2 | ROUND-OFF SCALE | 1.065814e-14 | 4.283986e-16 | 2.481541e-15 | 395/640 | (4, 6, 0) |
| pe3 | ROUND-OFF SCALE | 1.421085e-14 | 5.756846e-16 | 2.734847e-15 | 445/640 | (2, 5, 0) |
| pe4 | ROUND-OFF SCALE | 1.065814e-14 | 4.317634e-16 | 2.656196e-15 | 437/640 | (1, 3, 0) |
| pe5 | ROUND-OFF SCALE | 1.065814e-14 | 4.283986e-16 | 2.553030e-15 | 421/640 | (5, 6, 0) |

`pt` overall: **ROUND-OFF SCALE**, defined box and full declared bounds alike.
Tolerance was not relaxed; the guarded `max_rel` denominator is floored at the
field's own dynamic range (~2.47e+01) exactly as in §19/§20.

`pkz` moved with it, from SMALL to **ROUND-OFF SCALE** (`max_abs` 1.421085e-13,
`max_rel` 5.374156e-15). That is the expected coupling: `dyn_core.F90:1348`
rewrites `pkz` at every heated level inside the same NH heating block.

Exit fields now standing at ROUND-OFF or SMALL: `u`, `v`, `w`, `delz`, `delp`,
`pt`, `pe`, `peln`, `pk`, `pkz`, `ua`, `va` — **12 of 13**.

**`omga` remains the sole SIGNIFICANT exit field** (`max_abs` 3.248e-02 …
4.740e-02, `max_rel` ≈ 1.00, mismatch 640/640 on every tile). Out of scope by
assignment (§9); unchanged from §20.7's diagnosis — the in-loop finalization,
the `pem` capture and `adv_pe` are all absent.

## 21.6 §8 in-loop regression — no regression

All 15 in-loop savepoint × substep combinations still PASS on the defined box,
from the same run:

| savepoint | it=1 | it=2 | it=3 |
|---|---|---|---|
| SP1 (`cgrid_pressure`) | PASS | PASS | PASS |
| SP2 (`dgrid_dynamics`) | PASS | PASS | PASS |
| `sp2a` (`vertical_solver`) | PASS | PASS | PASS |
| `sp2b` (`nh_pressure`) | PASS | PASS | PASS |
| `sp2c` | PASS | PASS | PASS |

Zero NaN/Inf, no NONFINITE or FAIL lines, harness exit 0. Two comparator
reading notes worth carrying forward, both of which briefly produced false
FAILs during this pass:

- The accepted verdict is the **defined box**. The parenthetical "full declared
  bounds, incl. both sides' undefined outer halo rings" figure is excluded by
  the established methodology and grades SIGNIFICANT at SP1/`sp2a` purely from
  the four diagonal corner halo cells, which have no donor tile.
- A large `max_abs` is not by itself a failure. `mfx`/`mfy` at SP2 carry
  `max_abs` ≈ 1.5 against their own dynamic range of ~4.2e+14, i.e. `max_rel`
  ≈ 3.6e-15 — ROUND-OFF.

`sp2c` at `it<n_split` is compared against the `nh_pressure` stage dump, not
`seam_reconcile`; the latter is emitted only at `it = n_split`. That is the same
source-justified comparand documented in §19.

## 21.7 §12 regression

- **R8 CTest: 22/22 PASS**, run under a `gpu_a100` job so all 9 GPU-labelled
  tests actually execute (on a login node they are correctly *skipped*, which
  must not be read as passing).
- **Default synthetic smoke: PASS** — all six tiles finite in `u,v,delp,pt,w,delz`,
  `delp` positive over every owned box, state demonstrably changed, exit 0.
- Performance deliberately NOT re-run. **Performance baseline remains STALE.**
- The 22/22 was measured on a build behaviourally identical to the final tree
  (the only later edits to a test were comment/printf text plus the reverted
  `nf_ke` pin). A belt-and-braces re-run of the final tree was submitted as
  `gpu_a100` job **58161075** (`r8ctest_final.sh` → `final_build.log`,
  `final_ctest.log`, `r8ctf_58161075.out`); it was still `PD (Priority)` when
  this pass reported. Check those three files rather than re-deriving.

`test_dispatch_smoke.cpp` carried three stale narrations asserting that
`corner_edge_service` "remains a deliberate fail-stub", that the production path
"never REQUESTS it", and (in a Phase-B printf) that the production dispatch's
slot is a `fail-adapter (unsupported)`. All three are now false and were
corrected. Comment/printf text only — no behavioural change.

**Deferred, recorded honestly: raising the smoke test's `cfg.nf_ke` from its
pinned 1 to the authoritative 3 makes Phase C FAIL.** The pin exists only to
keep `nt = 0` on the single del2 pass, which is what let the old stub survive.
Raising it was attempted as optional extra coverage, produced
`Phase C result: FAIL` / `Dispatch smoke test: FAIL` (21/22), and was **reverted**
to 1 because localizing it needs a GPU diagnostic run that could not be
scheduled inside this task's cost budget (three `gpu_a100` attempts stayed
`PD (Priority)`; `warpa006` draining, the rest `mix`). Ruled out by inspection
already: `heat_source` is registered on the full `is-3..ie+3 / js-3..je+3` box
with `lower[2] == 0`, so it satisfies every guard the new service applies, and
Phase C does wire the production service (`dispatch.corner_edge_service =
full.corner_edge_service`). The `22/22` above is therefore the state **with the
pin at 1** — i.e. the three production fixes present, the smoke test unchanged in
behaviour.

This is a gap in *this test's* coverage, not evidence against the service: the
service is exercised on real A100 hardware by the authoritative-SP0 bench run on
all six tiles, where the dyn_core-exit `pt`/`pkz` validate at round-off scale
(§21.5). Localize the Phase C failure before relying on `nf_ke > 1` in this test.

## 21.8 Fix budget and stop condition

Three of three permitted fixes used. **§10 HARD STOP condition (7) — "exit `pt`
passes" — is satisfied**, so no further diagnose→fix cycles were run.

Gate C2 remains **BLOCKED**, now on `omga` alone.

# 22. Prompt 6B — dyn_core-exit `omga` finalization

Scope: the exit `omga` only, the last of the 13 `sp3` fields and the only
remaining Gate C2 blocker. No new Fortran instrumentation; the existing
`c8_L10_savepoint_6c_run1` savepoints were reused. c8 / npz=10 / n_split=3 /
k_split=1 / R8 / 6 tiles, as everywhere else in this document.

## 22.1 §1 — the authoritative `omga` write path, complete

`dt = bdt/n_split` (`dyn_core.F90:317`), `rdt = 1/dt` (`:319`); here
`rdt = 1/1200`. The acoustic loop is `:419`…`:1293`.

| # | step | source | gate | inputs | writes `omga`? |
|---|------|--------|------|--------|----------------|
| 1 | `pem` allocate + fill | `:503-517` | `it==n_split .and. end_step .and. use_old_omega` | `ptop`, `delp` | no (writes `pem`) |
| 2 | `last_step = .true.` | `:518` | same | — | no |
| 3 | `c_sw` uses `omga` as `wc` | `:539` | every `it` | — | yes, as **scratch** |
| 4 | `Riem_Solver_c` uses `omga` as `w3` | `:604` | every `it` | — | yes, as **scratch** |
| 5 | in-loop `omga = delp` | `:801` | `.not. use_old_omega` | — | **DEAD here** |
| 6 | in-loop `xfx/yfx` convergence form | `:838-842` | `.not. use_old_omega` | — | **DEAD here** |
| 7 | E11 seam reconciliation | `:1176-1188` | `it==n_split`, `grid_type<4`, not nested | `u`, `v` | no |
| 8 | **`omga = (pe(i,k+1,j) - pem(i,k+1,j))*rdt`** | `:1205-1211` | `last_step .and. use_old_omega` | `pe`, `pem`, `rdt` | **yes — final** |
| 9 | **`call adv_pe(ua, va, pem, omga, …)`** | `:1218` → `:1535-1636` | same | `ua`, `va`, `pem`, `ec1/ec2`, `en1/en2`, `dx`, `dy`, `rarea`, `a2b_ord2` | **yes — final** |
| 10 | `om2d` cumulative-sum form | `:1219-1237` | `.not. use_old_omega` | — | **DEAD here** |
| 11 | `id_ws` diagnostic | `:1238` | `hydrostatic` | — | **DEAD here** |
| 12 | `deallocate(pem)` | `:1397` | as #1 | — | no |

Two corrections to what this project previously recorded:

- **`pem` is snapshotted INSIDE the acoustic loop**, at the top of the final
  substep (`:503-517`), not "pre-loop at `:505-513`" as `CURRENT_STATE.md` §9
  said. It sits after `complete_group_halo_update(i_pack(1))` (`:496`) and
  before `complete_group_halo_update(i_pack(8))` (`:524`), and is taken from
  *that moment's* `delp` — not from the dyn_core entry state. Corrected in §14.
- Steps 8/9 run **after** the E11 seam block, still inside the loop. The `sp2c`
  dump (`:1284-1290`) is after them but dumps only `u,v,w,delz,delp,pt`, so
  there is no ordering conflict. Nothing between `:1218` and the `sp3` dump
  (`:1371`) writes `pe`, `ua` or `va` — `ua`/`va`'s only writers are `:539`
  (`c_sw`) and `:825` (`d_sw`) — so step 9 produces the exit value.

`adv_pe` itself, `:1535-1636`, per level `k`:

```
  k==npz:  up = ua(i,j,npz);              vp = va(i,j,npz)
  else:    up = 0.5*(ua(i,j,k)+ua(i,j,k+1)); vp = 0.5*(va(i,j,k)+va(i,j,k+1))
  v3(n,i,j) = up*ec1(n,i,j) + vp*ec2(n,i,j)            ! is..ie   / js..je
  pin(i,j)  = pem(i,k+1,j)                             ! is-1..ie+1 / js-1..je+1
  call a2b_ord2(pin, pb, …)
  pdx(n,i,j) = (pb(i,j)+pb(i+1,j))*dx(i,j)*en1(n,i,j)  ! is..ie   / js..je+1
  pdy(n,i,j) = (pb(i,j)+pb(i,j+1))*dy(i,j)*en2(n,i,j)  ! is..ie+1 / js..je
  grad(n,i,j)= pdx(n,i,j+1)-pdx(n,i,j) - pdy(n,i,j)+pdy(n,i+1,j)
  om(i,j,k)  = om(i,j,k) + 0.5*rarea(i,j)*(v3·grad)
```

Note FV3's vertical convention: `k=1` is the model **top** and `k=npz` the
**surface** layer, so the `k==npz` branch is the surface, not the top. Both
ports name that flag `surface` for exactly this reason.

## 22.2 §2 — the live branch

| flag | value | source |
|------|-------|--------|
| `use_old_omega` | `.true.` | `fv_arrays.F90:465` default, not overridden in the fixture namelist |
| `last_step` / `end_step` | true at `it==n_split==3` | `:518`; the harness sets `step.end_step = 1` |
| `hydrostatic` | false | production first-run |
| `grid_type` | 0 | production first-run |
| `nested` | false | one rank per face |
| `a2b_ord` | 2 (fixed — `adv_pe` calls `a2b_ord2` unconditionally) | `:1583` |
| `beta` | 0 | production first-run |

So exactly one branch is live: `use_old_omega = .true.` → steps 8 and 9. The
`om2d` form (step 10) is dead, and **`NumericalConfig` carries no
`use_old_omega` knob**, so it cannot be silently selected either; the port
implements only the live branch and `omega_preconditions()` says so.

## 22.3 §3 — `omga` input-fidelity audit

Every direct, non-workspace input to steps 8 and 9:

| input | classification | evidence |
|-------|----------------|----------|
| `pe` | AUTHORITATIVE + VERIFIED | exit `pe` validates ROUND-OFF over the owned box (§4 of `CURRENT_STATE.md`); the omega path reads only `is..ie / js..je`, never the unwritten halo ring |
| `pem` | WORKSPACE | operator-arena slot 10, built in-path from `ptop` + `delp` |
| `pb` | WORKSPACE | operator-arena slot 11, `a2b_ord2` output |
| `delp` (at the snapshot) | AUTHORITATIVE + VERIFIED | `sp2c` `it=2` `delp` is ROUND-OFF, and that dump *is* the state the snapshot sees |
| `ua`, `va` | AUTHORITATIVE + VERIFIED | exit `ua`/`va` are ROUND-OFF (3.27e-13 / 2.81e-13); last writer is `d_sw` at `:825` |
| `dx`, `dy`, `rarea` | AUTHORITATIVE + VERIFIED | `--grid-metrics` authoritative export |
| `grid_lonlat` (B-grid corners) | AUTHORITATIVE + VERIFIED | harness `cube::CubeGrid`, store order fixed in §4 #9, and now verified against the authoritative `dx`/`dy` — see §22.5 |
| `edge_w/e/s/n` | AUTHORITATIVE + VERIFIED | `edge_factors` port, fix #7 |
| `ec1`, `ec2`, `en1`, `en2` | **DERIVED + VERIFIED** | derived per cell from its own four B-grid corners; see §22.5 |
| `rdt` | DERIVED + VERIFIED | `DispatchParams::rdt = 1/dt`, `dt = bdt/n_split` ⇒ 1/1200 |
| `ptop` | AUTHORITATIVE + VERIFIED | 100.0 (fix #2) |
| incoming `omga` | irrelevant | fully overwritten at `:1211` before any read |
| `grav`, `rdgas`, `kappa`, `radius` | NOT REQUIRED | `ec*`/`en*` are unit vectors; the earth radius enters only through `dx`/`dy`, which are authoritative |

**Audit result: PASS — no missing authoritative inputs.** This is the first
Prompt 6B pass in which the input audit found nothing; the defect was a missing
*stage*, not a corrupt input.

## 22.4 §4 — `adv_pe` reconciliation before writing code

| aspect | authoritative | CUDA before this pass |
|--------|---------------|-----------------------|
| routine | `dyn_core.F90:1535-1636` | **MISSING** — no `adv_pe`, no caller |
| `a2b_ord2` | `a2b_edge.F90:697-800` | **PRESENT and correct** — `a2b_ord2_global` (host `wp_d_operators.cpp:259-313` and its CUDA twin) was verified verbatim against the Fortran, including the `1/3` corner form, the `edge_*`-weighted edge form and the interior-band exclusions |
| `ec1/ec2` | `get_center_vect`, `fv_grid_utils.F90:1792-1841` | **MISSING** (2 textual refs, no data) |
| `en1/en2` | `fv_grid_utils.F90:759-772` | **MISSING** (0 refs) |
| `pem` | `:505-513` | **MISSING** |
| `dx/dy/rarea` | grid metrics | **PRESENT**, already registered `immutable_grid` |
| final assignment | `:1211` + `:1218` | **MISSING** — `omga`'s last CUDA writer was `Riem_Solver_c` using it as the `w3` scratch argument, which is faithful as an intermediate (Fortran aliases the same array as `c_sw`'s `wc` at `:518`) but is not the exit value |

Classification: `adv_pe` **MISSING**; `a2b_ord2` **REUSED, no new code**;
`en/ec` **DERIVED**. Per the §4 caution, the subsystem was *not* assumed missing
merely because the symbols were absent — `a2b_ord2` was found already present
and correct, and is called rather than reimplemented.

## 22.5 §6 — `en1/en2/ec1/ec2`: derived from existing authoritative geometry

All four are functions of the B-grid corner positions alone, so no new Fortran
instrumentation and no new savepoint is needed — HARD STOP (2) does not fire.
Read verbatim from the authoritative source:

- `fv_grid_utils.F90:308-321` — `latlon2xyz(grid(i,j,1:2), grid3(1,i,j))` over
  `isd..ied+1 / jsd..jed+1`, then `get_center_vect(npx, npy, grid3, ec1, ec2)`.
  This also confirms `grid(:,:,1)=lon`, `grid(:,:,2)=lat`, i.e. exactly the
  component-slowest `grid_lonlat` layout that fix #9 established.
- `fv_grid_utils.F90:759-772`, under the in-source comment
  `! For omega computation:` — `en1/en2` exist **solely** for `adv_pe`:
  `en1(:,i,j) = normalize(grid3(i,j) × grid3(i+1,j))` over `js..je+1 / is..ie`,
  `en2(:,i,j) = normalize(grid3(i,j+1) × grid3(i,j))` over `js..je / is..ie+1`.
- `fv_grid_utils.F90:1792-1841` — `get_center_vect`'s non-`OLD_VECT` branch:
  cell centre `pc = cell_center3(4 corners)`; then for `e1`,
  `p1 = mid_pt3_cart(c(i,j), c(i,j+1))`, `p2 = mid_pt3_cart(c(i+1,j), c(i+1,j+1))`,
  `p3 = p2 × p1`, `u1 = normalize(pc × p3)`; for `e2` the same with the other
  midpoint pair. Its four diagonal-block zeroing guards are **unreachable** for
  `adv_pe`'s footprint at one rank per face.

Every cross-product argument order and index pairing above was re-checked
against `adv_pe_cell_increment` (`wp_d_operators.hpp`). All six vectors are
**unit** vectors, so no earth-radius constant enters the derivation.

**Verification, not assumption.** The derivation is faithful only if the harness
`grid_lonlat` corners *are* the Fortran `gridstruct%grid` corners at the same
indices with the same intra-tile orientation. `tools/savepoints/verify_adv_pe_geometry.cpp`
tests exactly that, radius-free: `fv_grid_tools` builds
`dx(i,j) = great_circle_dist(grid(i,j), grid(i+1,j))` and
`dy(i,j) = great_circle_dist(grid(i,j), grid(i,j+1))`, so if the corners agree
then `dx_authoritative / gcd_unit` is the *same constant* on every cell of every
tile — and a store-order, index-base, orientation or tile-identity error makes
that ratio field non-constant. Over exactly `adv_pe`'s footprint:

```
WORST RELATIVE SPREAD = 6.661e-15  -> GEOMETRY VERIFIED
implied radius = 6371000.000000000 m on all six tiles
```

So `en/ec` are **DERIVED + VERIFIED**, and HARD STOP (3) does not fire.

Caveat recorded, deliberately not chased: the authoritative build has
`f_p = selected_real_kind(20)` (no `NO_QUAD_PRECISION` in `flags.make`), so
Fortran's `normalize_vect`/`cell_center3` accumulate in quad. The derived unit
vectors can therefore differ by up to ~1 double ulp, i.e. ~1e-18 absolute on
`omga` — three to four orders below the ~1e-14 cancellation floor of `pe−pem`
computed below, hence unobservable. Also note `v3·grad` is invariant under a
global rotation of the sphere, so global grid orientation cannot matter; only
intra-tile index orientation can, and a transpose there would show up as an
O(1) error, not a round-off one.

## 22.6 §5 — implementation

One fix, additive, no existing arithmetic touched.

New host twins in `operators/wp_d_operators.{hpp,cpp}`:
`omega_pem_snapshot` (`:505-516`), `omega_from_pe` (`:1208-1212`), `adv_pe`
(`:1535-1636`), plus the header-only device-callable
`adv_pe_cell_increment` and the small `latlon2xyz`/`vect_cross`/`normalize`/
`mid_pt3_cart`/`cell_center3` helpers it needs (`fv_grid_utils.F90:1644-1668`,
`:1780-1789`, `:1792-1841`).

CUDA twins in `operators/wp_d_cuda.{hpp,cu}`: `omega_pem_kernel`,
`omega_from_pe_kernel`, `adv_pe_kernel` and the launchers
`launch_pem_snapshot_cuda` / `launch_omega_from_pe_cuda` / `launch_adv_pe_cuda`.
The kernels share `adv_pe_cell_increment` with the host twin, so there is one
definition of the arithmetic.

Contract layer, `operators/wp_d_contract.{hpp,cpp}` (+ `_cuda` twins):
`OmegaViews` / `bind_omega` / `omega_preconditions`, and two dispatch entries
`launch_omega_pem_snapshot` and `launch_omega_finalize`. `bind_omega` binds
`omga`/`pe`/`delp` mutable, `ua`/`va` const, and `dx`/`dy`/`rarea`/
`grid_lonlat`/`edge_{w,e,s,n}` from `immutable->grid`.

`pem` and `pb` live in the **WP-D operator arena** as new tail slots 10 and 11
(`array_count` 8 → 10 → 12), so `pem` survives the whole substep between the
`:503` snapshot and the `:1218` consumer **on the device**: no host/device
transfer is added and persistent residency is preserved. Only `pem` levels
`2..npz+1` are ever read, so the workspace stores exactly `npz` levels with
`pem_view(i,j,k0) = pem(i,k0+2,j)`, which makes it directly usable as
`adv_pe`'s `pin` for level `k0+1` with no shift.

Two additive tail slots in `NumericalDispatch`
(`include/fv3_gpu_internal.hpp`): `omega_pem_snapshot`, `omega_finalize`.
`internal_contract_version` deliberately does **not** move —
`validate_dispatch` does not require them and `run()` skips nulls, the same
argument the existing `cgrid_vertical_pre_halo` slot documents. Wired in
`runtime/fv3_gpu_production_dispatch.cu`.

Two call sites in `runtime/fv3_gpu_runtime.cpp`, both gated
`it == n_split && p.end_step` (the same predicate as Fortran's `last_step`):
the snapshot immediately after the `it==1` `H_DSCALAR` resolve / `halo_in`
stage, matching `:503` sitting after `:496`; and the finalize immediately
**after** the E11 `H_DWIND_SEAM` block, matching `:1204-1218` sitting after
`:1176-1188`, with a `FV3_GPU_DBG_STAGE("omega_finalize")` checkpoint.

No new translation unit, so no `CMakeLists.txt` change; `fv3_wp_d_cuda` already
links `fv3_wp_d_operators` PUBLIC. **Numerical algorithm changed: NO** — an
authoritative stage that was entirely absent is now performed.

## 22.7 Host-side validation of the arithmetic, no GPU

Before spending GPU time, the whole path was driven on the login node from
authoritative Fortran data by `tools/savepoints/verify_omega_host.cpp`, which
calls the **real** host twins (`omega_pem_snapshot` → `omega_from_pe` →
`a2b_ord2_global` → `adv_pe`) with:

- `delp` from `sp2c` `it=2` — end of `it=2` *is* the top of `it=3`, which is
  exactly the state `:503-517` snapshots;
- `pe`, `ua`, `va` from `sp3` `it=3`;
- `dx`/`dy`/`rarea` from the authoritative grid-metrics export;
- B-grid corners from `cube::CubeGrid` (verified, §22.5) and `edge_*` from the
  radius-free `edge_factors` ratio;

and compares against the authoritative `sp3` `it=3` `omga`, owned box, all six
tiles:

| tile | max_abs | guarded max_rel | rms | field range | max adv_pe increment |
|------|---------|-----------------|-----|-------------|----------------------|
| pe0 | 2.789435e-15 | 8.615285e-14 | 5.735649e-16 | 3.237775e-02 | 1.619154e-03 |
| pe1 | 4.853756e-15 | 1.499102e-13 | 1.003117e-15 | 3.237775e-02 | 1.619154e-03 |
| pe2 | 1.158221e-14 | 2.445659e-13 | 2.741039e-15 | 4.735822e-02 | 2.334083e-03 |
| pe3 | 6.530909e-15 | 2.017098e-13 | 1.119127e-15 | 3.237775e-02 | 1.619154e-03 |
| pe4 | 5.048045e-15 | 1.559109e-13 | 9.841411e-16 | 3.237775e-02 | 1.619154e-03 |
| pe5 | 6.147860e-15 | 1.298161e-13 | 1.883786e-15 | 4.735822e-02 | 2.334083e-03 |

Worst: **max_abs 1.158221e-14, guarded max_rel 2.445659e-13** — against the
pre-fix divergence of max_abs ≈4.7e-02 / max_rel ≈1.0, a ~4e12× reduction.

That residual is the **cancellation floor of `pe − pem`, not an error budget**:
`pe ≈ 1e5` Pa, one double ulp of which is ≈1.5e-11 Pa, and `omga` divides the
difference by `dt = 1200` s ⇒ ≈1.2e-14, which is what is measured. By the
project's mechanical grading (`≤1e-13` ROUND-OFF, `≤1e-8` SMALL) the guarded
`max_rel` of 2.4e-13 lands in **SMALL NUMERICAL DIFFERENCE**, which is a PASS;
it is not claimed as ROUND-OFF.

The `adv_pe` increment is 1.6e-03…2.3e-03 — about 5% of the field and ~1e11×
the residual — so agreement on `omga` is **non-vacuous** evidence about
`adv_pe`, `a2b_ord2`, `pem` and the derived `en/ec` geometry, not just about the
`(pe − pem)*rdt` term.

This isolates arithmetic from wiring: a PASS here proves the formulas and the
derived geometry; the GPU `sp3` comparison below is what proves the CUDA twin,
the arena workspace and the runtime call sites.

## 22.8 §8 — GPU validation of the exit `omga`

Run: authoritative SP0 → dyn_core exit, c8 / npz=10 / n_split=3 / k_split=1 /
R8 / 6 MPI ranks = 6 tiles, one A100 (`warpa003`), Slurm job **58162990**, with
`FV3_GPU_DEBUG_DUMP_STAGE` extended by the new `omega_finalize` stage. Run
health: `RUN exit=0`, log ends `=== 3A.11 CUDA perf benchmark: PASS ===`, 5952
`[FV3DBG]` lines all carrying `nan/inf=0/`, zero `NONFINITE`/`FAIL` lines. (The
job's `sacct` state reads `FAILED 1:0` only because the script's last command is
a `grep -c "NONFINITE\|FAIL"` that exits 1 on **zero** matches — the desired
result.)

Comparator: `tools/savepoints/compare_cuda_fortran.py <6c_run1> <dump> sp3 3
dyn_core_exit`, unchanged, tolerances unchanged.

| tile | max_abs | guarded max_rel | rms | floor | mismatch | worst (i,j,k) |
|------|---------|-----------------|-----|-------|----------|----------------|
| pe0 | 4.875354e-14 | 1.505773e-12 | 8.637913e-15 | 3.238e-02 | 639/640 | (3, 0, 9) |
| pe1 | 2.533564e-14 | 7.825015e-13 | 8.071037e-15 | 3.238e-02 | 639/640 | (3, 6, 7) |
| pe2 | 4.007211e-14 | 8.461490e-13 | 9.925673e-15 | 4.736e-02 | 640/640 | (2, 6, 8) |
| pe3 | 3.828014e-14 | 1.182298e-12 | 1.073349e-14 | 3.238e-02 | 640/640 | (7, 2, 7) |
| pe4 | 3.627133e-14 | 1.120255e-12 | 8.841662e-15 | 3.238e-02 | 640/640 | (1, 7, 9) |
| pe5 | 3.793146e-14 | 8.009478e-13 | 9.618378e-15 | 4.736e-02 | 640/640 | (3, 1, 9) |

**Worst across all six tiles: max_abs 4.875354e-14, guarded max_rel
1.505773e-12, rms ≤1.073e-14 ⇒ SMALL NUMERICAL DIFFERENCE = PASS.** Finite
everywhere. Pre-fix: max_abs ≈3.2e-02…4.7e-02, max_rel ≈1.0, 640/640 differing —
a ~1e12× residual reduction.

`mismatch = 639–640 of 640` is expected and is not a defect signal: every
`omga` cell is the difference of two ~1e5 Pa pressures divided by `dt`, so
essentially every cell differs in its last bits. The GPU residual is ~4× the
host-twin residual of §22.7 (4.9e-14 vs 1.2e-14), consistent with the CUDA path
reaching `pe`/`pem` through its own summation order rather than reading the
Fortran dump.

**The comparison box is the authoritative write footprint, not a convenience
choice.** `dyn_core.F90:1207-1213` loops `k=1,npz / j=js,je / i=is,ie`, and
`adv_pe`'s `om` update covers the same `is..ie / js..je`; nothing writes `omga`'s
halo ring in the finalization, so on the Fortran side that ring still holds
`c_sw`/`Riem_Solver_c` scratch. `omga`'s "full declared bounds" figure therefore
grades SIGNIFICANT for exactly the same reason `ua`/`va`'s already did, and is
excluded by the established methodology.

### Exit boundary, all 13 fields

`sp3` `it=3` now reads, on the defined box, worst across all six tiles:

- ROUND-OFF SCALE: `delz`, `delp`, `pe` (owned box), `peln`, `pk`, `pkz`, `pt`,
  `ua`, `va`
- SMALL NUMERICAL DIFFERENCE: `u`, `v`, `w`, **`omga`**

**13 of 13 exit fields PASS.** The exit `pt` is bit-unchanged from the previous
pass (1.421085e-14 / 5.756846e-16 on pe0/pe1/pe3, 1.065814e-14 on pe2/pe4/pe5),
so the omega work did not perturb it. `pe`'s single Fortran halo ring remains
NOT WRITTEN by CUDA — a coverage gap already recorded, unchanged by this pass.

## 22.9 §9 / §13 — regression

**All 15 in-loop boundaries re-verified from this same dump**, defined box,
worst across six tiles:

| savepoint | it=1 | it=2 | it=3 |
|-----------|------|------|------|
| SP1 (`cgrid_pressure`) | SMALL | SMALL | SMALL |
| SP2 (`dgrid_dynamics`) | **ROUND-OFF** | SMALL | SMALL |
| `sp2a` (`vertical_solver`) | SMALL | SMALL | SMALL |
| `sp2b` (`nh_pressure`) | SMALL | SMALL | SMALL |
| `sp2c` (`seam_reconcile`) | SMALL | SMALL | SMALL |

No regression against the `nf_ke` pass. `sp2c` at `it=1`/`it=2` is compared
against the `nh_pressure` stage dump, as established — the `seam_reconcile`
stage only exists at `it == n_split` because E11 is `it==n_split`-gated
(`dyn_core.F90:1178`).

**R8 CTest: 22/22 PASS** (Slurm job 58161075). `final_build.log` confirms the
run genuinely recompiled `operators/wp_d_operators.cpp.o` and
`operators/wp_d_cuda.cu.o` and relinked every test binary, so it covers the
omega operators as built. `test_dispatch_smoke.cpp`'s `cfg.nf_ke` stays at its
pinned 1 (the deferred, un-localized Phase C item).

**Default synthetic smoke: PASS** (Slurm job 58164201) — `SMOKE_PIPE_STATUS=0`,
`=== 3A.11 CUDA perf benchmark: PASS ===`, all six tiles finite on
`u,v,delp,pt,w,delz`, `delp` positive over the owned box.

**Performance: NOT retaken. The §6 baseline stays STALE — RETEST REQUIRED.**
The omega finalization adds `omega_pem_snapshot` + `omega_finalize` work to the
`it == n_split` substep on the default path, so it is one more reason the old
table cannot be reused.

## 22.10 Corrections to earlier documentation

- `CURRENT_STATE.md` §9 previously stated `pem` is "allocated and filled
  pre-loop at `:505-513`". **STALE.** `pem` is allocated and filled **inside**
  the acoustic loop at `dyn_core.F90:503-517`, under
  `if (it==n_split .and. end_step)` — i.e. it snapshots the pressure at the top
  of the *final* substep, which is why the host validation feeds it `delp` from
  `sp2c`/`it=2` rather than from a pre-loop dump. Corrected.
- The omega finalization runs **after** the E11 seam block (`:1176-1188`), still
  inside the loop, not after it.
