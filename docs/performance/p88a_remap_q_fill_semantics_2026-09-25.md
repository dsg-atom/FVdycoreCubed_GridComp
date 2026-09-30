# Prompt 88A — Map Real Production `fill` Semantics for REMAP_Q

Date: 2026-09-25
Branch: `optimization/cuda-fv3-performance`
Task type: bounded runtime/source-mapping task. **No CUDA source changes.**
No REMAP_Q numerics, FV_StateMod configuration, PT/DELZ, REMAP_UV, tracer
CUDA, R4, C512, or performance measurement touched, per the task's Do-NOT
list. Only additive documentation and one new host-only (no CUDA) test file
were created.

## 0. Background

Prompt 87 closed with CUDA REMAP_Q dispatch 0/36 in a real C32 A/B. The
`kord_tr(1)==9` fix was confirmed correct and necessary, but insufficient: a
second, independent, unconditional GEOS forcing —
`FV_StateMod.F90:568`'s `flagstruct%fill = .true.` — rejects dispatch via the
selector's `.not. fill` term regardless of kord. This left an open question
that Prompt 87 explicitly declined to resolve: what does `fill` actually do
numerically, is `fill=.false.` ever reachable in a real runnable
configuration, and how much CUDA work would `fill=.true.` support require.
Prompt 88A answers exactly that question, with no implementation.

## 1. Stage 0 — TRACE `fill` DEFINITION

Declaration: `fv_arrays.F90:820` (both repos' `fv_flags_type`):

```fortran
logical :: fill = .false.   !< Fills in negative tracer values by taking positive tracers from
                            !< the cells above and below. This option is useful when the physical
                            !< parameterizations produced negatives. The default is .false.
```

The bare GFDL default is `.false.`. Every writer found by an exhaustive
`grep -rn "%fill *="` sweep (excluding `fill_dp`/`fill_wz`/`fill_gfs`/etc.)
across both repositories:

| File | Routine | Assignment | Condition | Reachable from `StandAlone_FV3_Dycore.x`? |
|---|---|---|---|---|
| `FV_StateMod.F90:568` (fortran_perf) / `:563` (this repo) | `FV_Setup` | `flagstruct%fill = .true.` | none (unconditional) | **YES** — the only `FV_Setup` in the actual production/HS standalone call path |
| `geos-gtfv3/driver/FV_State_Utilities.F90:74` (both repos, mirrored) | (geos-gtfv3 driver init) | `flagstruct%fill = .true.` | none (unconditional) | **NO** — compiles into a separate executable target, `fv3_driver.x` (gated by `BUILD_GEOS_GTFV3_INTERFACE`, `CMakeLists.txt:260-278`), never linked into `StandAlone_FV3_Dycore.x` (built from `StandAlone_FV3_Dycore.F90`, `CMakeLists.txt:195-196`). Not part of this project's execution path. |
| `fv_control.F90` namelist pointer (`fill` declared line 268, listed in `fv_core_nml` line 674, associated `fill => Atm%flagstruct%fill` line 1264) | `run_setup` (read at `fv_control.F90:730/751`), invoked via `fv_init2` | `read(...,fv_core_nml,...)` only overwrites `fill` if the namelist file's `&fv_core_nml` block explicitly lists it | requires an explicit `fill = ...` line in `input.nml` | **Structurally reachable, but empirically never exercised** — see Stage 1/2 |

Do not assume `FV_StateMod.F90:568` is the only writer — confirmed by this
sweep it is the only one on the actual standalone execution path; the
`geos-gtfv3` writer is real but unreachable here, and the namelist path is
real but never used (below).

## 2. Stage 1 — TRACE ACTUAL STANDALONE RUNTIME

Call order inside `FV_Setup` (`FV_StateMod.F90`), confirmed by direct read:

```
fms_init -> fv_init1 (setup_pointers only; no namelist read)
  -> [unconditional GEOS-default block, lines 540-575, incl. line 568 fill=.true.]
  -> [resolution/grid-dependent defaults, lines 580-802]
  -> fv_init2 (line 808)  <-- THIS calls run_setup, which performs the
                               actual `read(input_nml_file, fv_core_nml, ...)`
                               (fv_control.F90:730/751)
  -> [no further %fill assignment anywhere after line 808 in this file]
```

**Key finding, not previously documented**: the namelist read (`fv_init2`
-> `run_setup`) executes *after* the hardcoded `fill=.true.` override at line
568, not before. Fortran namelist semantics only overwrite variables
explicitly present in the input file's `&fv_core_nml` block — a variable
absent from the file retains whatever value it already had. So *if* any
`input.nml` in this project ever set `fill = ...` explicitly, that value
would survive and take effect (the opposite of being clobbered). This is
structurally different from a simple "always reset after" pattern; it makes
Stage 2's search decisive rather than moot.

`mapn_tracer` (`fv_mapz.F90:2191`) receives `fill` as a plain `logical,
intent(in)` parameter from `Lagrangian_to_Eulerian` (`fv_mapz.F90:132`,
called from `fv_dynamics.F90:952`) with no reassignment anywhere in that call
chain. Confirmed by direct grep: no `%fill *=` in `fv_dynamics.F90` or
`fv_mapz.F90`.

**Final value at the actual REMAP_Q call, confirmed both by this trace and by
Prompt 87's own empirical run-log evidence (`FV3 fill: T`, both C32 A/B
arms): `fill = .true.`, unconditionally, for the entire traced call chain.**

## 3. Stage 2 — SEARCH RUNNABLE CONFIGURATIONS

Per Stage 1, the only way `fill=.false.` could ever reach production is an
explicit `fill = .false.` (or `.true.`, moot either way) line in some
`input.nml`'s `&fv_core_nml` block. Searched exhaustively:

- **Every `input.nml` under `GEOSfvdycore_fortran_perf/experiments/`** (57
  directories found by `find -iname input.nml`, spanning every C8/C16/C32/
  C64/C128/C256/C512 job, every Fortran/CUDA arm, R4 and R8, every prompt
  from P17 through P87): `grep -rl '^\s*fill\s*=' ... | wc -l` = **0**.
- **The job-generation template**, `src/Components/@FVdycoreCubed_GridComp/
  scripts/fv3.j` (the actual source these `input.nml` files are generated
  from): no `fill` assignment (excluding `fill_dp`/`fill_wz`/etc.) found.
- **Every `*.rc` file project-wide**: no `fill:` resource entry found.

| Config | `fill` runtime | Evidence |
|---|---|---|
| Every C8-C512 Fortran/CUDA A/B job in `GEOSfvdycore_fortran_perf/experiments/` (Prompts 17-87) | **TRUE** | `input.nml` never overrides it (0/57); `FV_StateMod.F90:568` stands; P87's own logs directly confirm `FV3 fill: T` |
| Held-Suarez track (`experiment/held-suarez-e2e`, C8-C512) | **TRUE** | Same harness family/template; no distinct `fv_core_nml` mechanism for HS; not separately overridden |
| GEOS standalone regression/gate harnesses (Gate-C2-era, P17-23) | **TRUE** | Same template lineage, same absence of a `fill` line |
| `geos-gtfv3` driver (`fv3_driver.x`) | N/A to this project's runnable configs | Not built/linked by the validated harness; a dead branch for this purpose, per Stage 0's "Reachable?" column — **not counted as a runnable config**, per the task's explicit instruction |

**No runnable configuration found producing `fill=.false.` in this project.**
`fill=.true.` is effectively universal across every standalone
Fortran/CUDA/R4/R8/HS/production-first-run job this project has ever run.

## 4. Stage 3 — MAP `fill` NUMERICAL SEMANTICS

`mapn_tracer` (`fv_mapz.F90:2191-2302`) uses `fill` in exactly one place,
after the full per-column PPM vertical remap (the `q4`/`scalar_profile` +
pressure-interval mapping loop) has already produced `q2` for all `nq`
tracers:

```fortran
if (fill) call fillz(i2-i1+1, km, nq, q2, dp2)
```

`fillz` (`fv_fill.F90:51-156`, non-`DEV_GFS_PHYS` branch — confirmed
authoritative: `DEV_GFS_PHYS` is not defined anywhere in this project's
CMake configuration) is a **separate, bounded, per-column, per-tracer
post-pass**, not a change to the core remap mathematics itself:

- **Top layer**: if negative, borrow its entire deficit from layer 2,
  zero the top.
- **Interior (levels 2..km-1)**: if negative, first try borrowing from the
  layer above (if positive), then — if still negative — from the layer
  below (if positive). Each borrow moves exactly `min(donor_mass,
  -deficit_mass)`, preserving `sum(q*dp)` exactly between donor and
  receiver (a local, zero-sum exchange).
- **Bottom layer**: if negative, borrow from the layer above.
- **Final check** (only if any level in the column was touched,
  `zfix`): recompute `sum0 = sum(q(k)*dp(k), k=2..km)`. If `sum0 > 0`
  (the column's aggregate mass in that range is still net-positive),
  rescale: `fac = sum0 / sum(max(0,q(k)*dp(k)))`, then
  `q(k) = max(0, fac*q(k)*dp(k)/dp(k))` for every level in that range —
  this exactly conserves `sum0` while clipping every remaining negative to
  zero. **If `sum0 <= 0`, the rescale does not fire and the column can be
  left with negative values** — confirmed both by direct code reading and
  by this task's own host reference test (§6, case
  `sum0_negative_fix_inert_by_design`). `fillz` is therefore a **mass-
  conserving, best-effort fixer, not an unconditional non-negativity
  guarantee**.

Other properties directly relevant to a future CUDA port:

- **No cross-tracer coupling**: the outer loop is `do ic=1,nq`, entirely
  independent per tracer — confirmed by code reading and by this task's
  batched nq=2 host test (§6).
- **No tracer-identity branching**: nothing in `fillz` inspects which
  tracer (`sphum` vs `cld_amt`) it is operating on — same code path for
  both.
- **`q_min`/positivity**: `fillz` is the *only* positivity mechanism in the
  fill=.true. path — `mapn_tracer`'s own `q_min=0.` literal is passed to
  `scalar_profile` (the PPM reconstruction), not to `fillz`. No dependence
  on `q_min` inside `fillz` itself.
- **No conservation trade-off beyond what's stated above**: mass is exactly
  conserved by every borrow step and by the final rescale (when it fires);
  it is *not* conserved to `sum0`'s original sign — a genuinely
  mass-deficient column is left mass-deficient, by design.
- No moisture/thermodynamic coupling, no Riemann-solver or `nh_utils.F90`
  interaction — `fillz` is REMAP_Q-local, operating purely on the already-
  remapped `q2`/`dp2` arrays.

## 5. Stage 4 — IDENTIFY MISSING CUDA LOGIC

Current CUDA REMAP_Q (`cuda/dyn_core/vertical/fv3_remap_q.cu`,
`remap_q_kernel`) is a per-thread-per-column kernel: reads `q1_in[0..km-1]`
from the persistent device tracer buffer, calls the existing `map_scalar_column`
device function to produce `q2_out[0..km-1]` (already fully validated,
bitwise-exact, for kord=9/9), then writes `q2_out` straight back to the
device buffer. There is currently no fillz-equivalent — the header
(`fv3_remap_q.cuh:20-53`) already documents `fill=.false.` as the current
(and, per Stages 0-2 above, actually unreachable) precondition.

| Operation | Current CUDA? | Needed for fill=true? | Reusable helper? |
|---|---|---|---|
| Top-layer borrow (1 conditional, 2 assignments) | No | Yes | NEW HELPER (small) |
| Interior borrow-above/borrow-below loop (per-level conditional, `km-2` iterations) | No | Yes | NEW HELPER (small) |
| Bottom-layer borrow | No | Yes | NEW HELPER (small) |
| Final non-local rescale (`sum0`/`sum1`/`fac`, 2 linear passes over the column) | No | Yes | NEW HELPER (small) |
| `zfix`-style "was any level touched" tracking | No | Yes (1 bool) | SMALL NEW BRANCH |
| Per-column, per-tracer independence (loop structure) | Yes (kernel is already 1 launch per tracer) | Reused as-is | REUSE DIRECTLY |
| `q2_out`/`dpeO` already resident, correctly shaped, per-thread local arrays | Yes | Reused as-is (fillz operates on exactly these) | REUSE DIRECTLY |
| Existing scratch slots (`prof_q`/`c2`/`c3`/`c4`/`gam`, each sized `km+1`, free after `map_scalar_column` returns) | Yes | One of these can hold fillz's `dm[]` array | REUSE DIRECTLY |

All four numerically-new pieces (top/interior/bottom borrow, final rescale)
are simple bookkeeping over an already-computed `q2` — no new PPM
reconstruction, no new pressure-interval search, no new algorithm class.
They compose naturally into one new `__host__ __device__` helper (e.g.
`fillz_column`, mirroring the existing `scalar_profile_column`/
`map_scalar_column` convention), callable once per tracer launch right
before the existing write-back loop.

## 6. Stage 5 — DATA/WORKSPACE IMPACT

- **New input arrays**: none. `fillz` needs only `q2`/`dp2`, both already
  computed and device-resident (`q2_out`, `dpeO`) before the existing
  write-back loop.
- **New output arrays**: none. `fillz_column` would write in place to
  `q2_out`, the same array the existing write-back loop already reads from.
- **New device residency**: none.
- **New H2D/D2H**: none.
- **New scratch**: none required beyond reusing one already-allocated,
  already-freed-by-that-point scratch slot (`km+1`-sized) for the `dm[]`
  local array; `remap_q_scratch_slots`'s slot count is unaffected.
- **New tracer metadata**: none — `fill` would be a single new boolean ABI
  parameter (mirroring how `kord1`/`kord2` were already added as extra
  per-launch parameters in Prompt 86B), not a new per-tracer identity
  concept.

**Critical question: Can `fill=.true.` still preserve 1 H2D + 1 D2H per
REMAP_Q invocation?**

**YES.** `fillz`'s entire operation happens strictly between the existing
`map_scalar_column` call and the existing single write-back loop, on data
that is already fully device-resident at that point. No new transfer of any
kind is implied.

## 7. Stage 6 — HARNESS COVERAGE

Existing host harness (`test_remap_q_tracer.cpp`, `test_remap_q_nq2_scope.cpp`)
covers only `fill=.false.` (confirmed by grep — neither file references
`fill`/`fillz` beyond one comment). **No existing coverage of `fill=.true.`
before this task.**

Added `cuda/dyn_core/tests/vertical/test_fillz_host_reference.cpp` — a
host-only (no CUDA, no project headers), mechanical, line-for-line C++ port
of the authoritative `fv_fill.F90` `fillz` (non-`DEV_GFS_PHYS` branch), wired
into `cuda/dyn_core/CMakeLists.txt` as a new permanent CTest target,
`fv3_wp_c_fillz_host_reference`. 9 fixture cases, each checking mass
conservation and (where applicable) non-negativity:

| Case | Covers |
|---|---|
| `no_negatives_is_noop` | fill branch present but inert on normal positive q |
| `near_minimum_q_no_trigger` | near-minimum q (tiny positive values) does not spuriously trigger a fix |
| `negative_top_layer` | top-layer boundary borrow |
| `negative_interior_borrow_above` | interior borrow-from-above |
| `negative_interior_borrow_below` | interior borrow-from-below (above insufficient) |
| `negative_bottom_layer` | bottom-layer boundary borrow |
| `deep_cluster_nonlocal_normalize` | final non-local rescale, `sum0>0` success case (clean non-negative + mass-conserved result) |
| `sum0_negative_fix_inert_by_design` | documents the genuine authoritative edge case where `sum0<=0` and the rescale does not enforce non-negativity (graded on mass conservation only, per Stage 3's finding) |
| `batched_nq2_per_tracer_independent` | both production tracers in one batched call — confirms fillz applies identically/independently per tracer |

Compiled (`g++ -std=c++17 -O2 -Wall`) and run directly on the login node
(pure host C++, no CUDA/GPU/Slurm required):

```
no_negatives_is_noop                   unchanged=yes PASS
near_minimum_q_no_trigger              mass_rel=0.000e+00 negative_after=no repeatable=yes PASS
negative_top_layer                     mass_rel=0.000e+00 negative_after=no repeatable=yes PASS
negative_interior_borrow_above         mass_rel=0.000e+00 negative_after=no repeatable=yes PASS
negative_interior_borrow_below         mass_rel=0.000e+00 negative_after=no repeatable=yes PASS
negative_bottom_layer                  mass_rel=0.000e+00 negative_after=no repeatable=yes PASS
deep_cluster_nonlocal_normalize        mass_rel=3.290e-16 negative_after=no repeatable=yes PASS
sum0_negative_fix_inert_by_design      mass_rel=1.348e-16 negative_after=YES (expected) PASS
batched_nq2_per_tracer_independent     tracer0_ok=yes tracer1_ok=yes PASS
RESULT=PASS (9 cases)
```

Repeatability confirmed two ways: (1) each case internally re-runs `fillz`
on the same input and on its own output (idempotence), both checked
bit-identical; (2) the whole binary was run twice as separate process
invocations and `diff`'d — byte-identical output. **Host fill=true harness:
PASS.**

## 8. Stage 7 — PORT SIZE

Estimated incremental work from the current Prompt-87 CUDA REMAP_Q
implementation:

- **CUDA LOC**: one new `__host__ __device__` helper function (~40-60 lines,
  mirroring `fillz`'s four bookkeeping stages, §5) in
  `fv3_map_scalar_column.hpp` or a small new header; one new call site + one
  new `if (a.fill)` branch in `remap_q_kernel` (~5 lines); no change to
  `map_scalar_column` itself.
- **Bridge/ABI changes**: one new boolean parameter threaded through
  `RemapQArgs`, `launch_remap_q`/`launch_remap_q_ctx`, `fv3_gpu_remap_q`'s
  bind(C) signature, and the Fortran ABI call site — same shape and size as
  Prompt 86B's `kord1`/`kord2` extension.
- **Selector changes**: drop (or invert the sense of) the `.not. fill` term
  in `fv_mapz.F90`'s `use_cuda_remap_q` — a one-line change, since fill is
  now handled rather than excluded.
- **New tests**: extend the existing GPU smoke test
  (`fv3_remap_q_nq2_prod_shape_smoke.cu`) with a fill=true case injecting a
  synthetic negative, checked bitwise against this task's new host
  reference (§7) — a single new smoke case, same pattern as the existing
  ones.
- **New residency changes**: none (§6).

**Classification: SMALL.** Smaller in scope than the original kord=8/9
REMAP_Q port itself (which required real PPM-profile reconstruction/
pressure-interval-search work) and smaller than PT/DELZ or REMAP_UV — no new
pressure-interval geometry, no new transfer architecture, no cross-tracer or
cross-column coupling, and the exact numerical target is now a validated,
repeatable, 9-case host golden reference (§7).

## 9. Stage 8 — GO/NO-GO

Checking the task's own criteria:

- `fill=.true.` **is** required for real production — it is universal
  (§3): every runnable configuration in this project forces it, with zero
  found exceptions.
- Semantics **are** cleanly separable — `fillz` is a bounded post-pass on
  already-remapped `q2`, with no cross-tracer or cross-column coupling and
  no moisture/thermodynamic interaction (§4).
- Current numerical helpers **do** cover most of the work — the entire
  existing kord=9/9 remap kernel (`map_scalar_column`) is unaffected and
  fully reused; only the four small fillz bookkeeping stages are new (§5,
  §8).
- The 1H2D+1D2H architecture **can** remain exactly as-is (§6).
- No new communication/residency problem appears (§6).

None of the STOP criteria apply: no broad tracer/moisture coupling, no
significant new numerical algorithm (fillz is elementary column arithmetic,
not a new PPM/Riemann-class method), no transfer-architecture fragmentation.

```
Standalone production fill:
TRUE

Any runnable fill=false configuration:
NO

fill changes numerics:
fillz -- a bounded, per-column, per-tracer, mass-conserving post-pass
(top/interior/bottom borrow-from-neighbor, then a final non-local
normalize-and-clip if the column's aggregate mass is still net-positive) --
runs after the core PPM vertical remap; no change to the remap math itself.
Best-effort, not an unconditional non-negativity guarantee (sum0<=0 case).

New CUDA numerical logic:
One new small __host__ __device__ helper (~40-60 LOC) implementing fillz's
4 bookkeeping stages; no new pressure-interval/PPM work; reuses the
existing map_scalar_column output (q2_out) and one existing scratch slot.

New data residency:
None -- operates entirely on already-device-resident q2/dp2 between the
existing kernel's remap step and its existing single write-back loop.

Can preserve 1 H2D + 1 D2H:
YES

Host fill=true harness:
PASS (9/9 cases, new file test_fillz_host_reference.cpp, wired into CMake
as ctest target fv3_wp_c_fillz_host_reference)

Port size:
SMALL

Decision:
PROCEED TO FILL=TRUE CUDA SUPPORT
```

---

**REMAP_Q FILL=TRUE SCOPE CLEAN — PROCEED TO CUDA EXTENSION**

**Superseded by implementation**: Prompt 88B implemented the CUDA
extension recommended here (`fillz_column<T>` helper, `fill` threaded
through REMAP_Q's C ABI, `.not. fill` selector gate removed in favor of
opt-in `RUN_CUDA_FV3_REMAPQ_ENABLE`), validated bitwise/GPU-executed/
mass-conserving, and PERFORMANCE VALIDATED POSITIVE at C512 (REMAPPING
-11.98%, TOTAL -1.51%). Verdict: KEEP AS OPT-IN INFRASTRUCTURE,
production default remains Fortran. See
`p88b_remap_q_fill_true_cuda_2026-09-25.md`.

## 10. Project Status

No CUDA source changed. No REMAP_Q numerics, FV_StateMod configuration,
PT/DELZ, REMAP_UV, tracer CUDA, or R4 touched. No C512 run, no `nsys`/`ncu`.
No commit/push performed. Files added/changed this task:

- `cuda/dyn_core/tests/vertical/test_fillz_host_reference.cpp` (new,
  host-only, no CUDA)
- `cuda/dyn_core/CMakeLists.txt` (one new `add_executable`/`add_test` pair,
  additive)
- `docs/performance/p88a_remap_q_fill_semantics_2026-09-25.md` (this file)
- `docs/performance/p87_remap_q_actual_kord9_production_2026-09-25.md`,
  `docs/performance/cuda_fv3_optimization_roadmap_2026-09-21.md`,
  `docs/agent_handoff/CURRENT_STATE.md` (updated, see each for the
  cross-reference to this finding)
