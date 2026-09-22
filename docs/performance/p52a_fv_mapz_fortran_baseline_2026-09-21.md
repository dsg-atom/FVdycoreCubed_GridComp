# Prompt 52A: fv_mapz Fortran Baseline Harness

Standalone Fortran baseline harness for `map_scalar`/`scalar_profile`
(`fv_mapz.F90`), the minimum vertical-remap CUDA porting unit selected by
Prompt 52. This is the Fortran-only baseline-generation phase; no CUDA
implementation exists yet (that is Prompt 52B).

## 0/1. Production config (locked facts, not re-derived here)

- `fv_mapz.F90` confirmed authoritative/production (compiled into both the
  main fvdycore library and the `mapz-driver` target from the identical
  source path).
- Production C512 Held-Suarez runtime: `kord_mt=kord_wz=kord_tm=9`, all
  `>7` -> the `scalar_profile` path is live; `ppm_profile` (`kord<=7`) is
  **not required** for the current C512 path. This is a **configuration-
  dependent** fact: any future reconfiguration to `kord<=7` would make
  `ppm_profile` required. Flagged as a caveat for Prompt 52B, not a
  current requirement.

## 2. map_scalar / scalar_profile contract

### `map_scalar` (public, `fv_mapz.F90:1519-1671`, 153 LOC)

Signature (order matches the source):

```fortran
subroutine map_scalar(km, pe1, q1, kn, pe2, q2, dp1, dp2, i1, i2, j, &
                       ibeg, iend, jbeg, jend, iv, kord, q_min)
```

| Arg | Type/shape | Intent | Description |
|---|---|---|---|
| `km` | integer | in | number of Lagrangian (input) layers |
| `pe1` | real, `(i1:i2, km+1)` | in | Lagrangian pressure at layer edges |
| `q1` | real, `(ibeg:iend, jbeg:jend, km)` | in | input scalar field on Lagrangian grid |
| `kn` | integer | in | number of Eulerian (output) layers |
| `pe2` | real, `(i1:i2, kn+1)` | in | Eulerian pressure at layer edges |
| `q2` | real, `(i1:i2, kn)` | out | remapped scalar field on Eulerian grid |
| `dp1` | real, `(i1:i2, km)` | in | Lagrangian layer thickness (`pe1` diffs) |
| `dp2` | real, `(i1:i2, kn)` | in | Eulerian layer thickness (`pe2` diffs) |
| `i1, i2` | integer | in | column loop bounds (this harness: `i1=i2=1`) |
| `j` | integer | in | j-index into `q1`'s 2nd dim (this harness: `j=1`) |
| `ibeg, iend, jbeg, jend` | integer | in | `q1`'s actual array bounds |
| `iv` | integer | in | field-type selector: `0`=positive-definite scalar, `1`=general (matches `qmap`/`t_map`-style production calls); other values exist in the full contract (e.g. `-1`,`-2`) but are not exercised here — not needed for the `kord=9` scalar path |
| `kord` | integer | in | remap order; `>7` dispatches to `scalar_profile`, `<=7` to `ppm_profile` |
| `q_min` | real, optional | in | lower clamp for `iv=0`/positive-definite; harness passes `0.0` for the positive-definite case, `-1e25` (effectively unbounded) otherwise |

Internally: builds a PPM-style sub-grid reconstruction of `q1` via
`scalar_profile`/`ppm_profile` (selected by `kord`), then integrates that
reconstruction analytically over each output layer's `[pe2(k), pe2(k+1)]`
pressure interval to produce `q2`.

### `scalar_profile` (private, `fv_mapz.F90:1881-2288`, 408 LOC)

Reached only via `map_scalar`'s internal `if (kord>7) call
scalar_profile(...)` dispatch — not independently callable from outside
`fv_mapz_mod` (and must not be made so; `public` statement is
authoritative Fortran source and was not touched). Builds the monotonic
piecewise-parabolic (PPM-family) sub-grid profile coefficients (left edge,
right edge, and curvature term per layer) that `map_scalar` then
integrates. Its own signature/argument list was inspected but is not
independently exercised by this harness (correctly — `map_scalar` is the
only entry point the harness or production code calls).

No `#ifdef` blocks inside either routine's body (confirmed via targeted
grep of both line ranges for `#if`/`#endif`/`#else` — none found).
`MOIST_CAPPA`/`USE_COND` (library-wide preprocessor defines) do not affect
either routine's body.

## 3. Dependency classification

`fv_mapz_mod`'s 11 module-level `use` dependencies (lines 84-95), and how
each was actually resolved in the harness after the Stage-4 pivot
(below):

| Dependency | Symbols used (module-wide) | Called by map_scalar/scalar_profile? | Classification | Harness resolution |
|---|---|---|---|---|
| `constants_mod` | radius, pi_8, rvgas, rdgas, grav, hlv, hlf, hls, cp_air, cp_vapor | No | D (unused by target routines, but part of whole-module compile) | Real (via real FMS) |
| `tracer_manager_mod` | get_tracer_index | No | D | Real (via real FMS) |
| `field_manager_mod` | MODEL_ATMOS | No | D | Real (via real FMS) |
| `fv_grid_utils_mod` | ptop_min | No | D | `mapz-driver/stubs/fv_grid_utils_stub.f90` (reused, not duplicated) |
| `fv_fill_mod` | fillz | No | D | Real (`../fv_fill.F90`) |
| `mpp_domains_mod` | mpp_update_domains, domain2d, mpp_global_sum, {NON_,}BITWISE\_{EXACT,EFP}\_SUM | No | D | Real (via real FMS) |
| `mpp_mod` | NOTE, FATAL, mpp_error, get_unit, mpp_root_pe, mpp_pe | Yes (one guarded `mpp_error(FATAL,...)` inside scalar_profile's `iv==-2 .and. .not.present(qs)` branch, never hit by any harness case) | C (called, but only on an unreachable/never-hit error path for our test cases) | Real (via real FMS) |
| `fv_arrays_mod` | fv_grid_type, fv_flags_type | No | D | Real (`../fv_arrays.F90`) |
| `fv_timing_mod` | timing_on, timing_off | No | D | Real (`../../tools/fv_timing.F90`) |
| `fv_mp_mod` | is_master | No | D | Real (`../../tools/fv_mp_mod.F90`) |
| `fv_cmp_mod` | qs_init, fv_sat_adj | No | D | Real (`../fv_cmp.F90`) |

Classification key: **A** = called directly by target routines with
nontrivial behavior; **B** = called directly, trivial/pass-through; **C**
= called directly but only on a path not exercised by our cases; **D** =
not called by target routines at all, only present because Fortran
type-checks the whole module (`Lagrangian_to_Eulerian` and other
subroutines in `fv_mapz_mod` do use these real types/interfaces).

**Revised conclusion vs. the original Stage-3 plan**: every dependency
ended up resolved with the **real** module (via real FMS/MAPL linkage),
not hand-written stubs, except `fv_grid_utils_mod`, which reuses the
existing non-trivial `mapz-driver/stubs/fv_grid_utils_stub.f90` (itself a
thin real-FMS-backed replacement, not a dummy). See Stage 4 below for why.

## 4. Harness architecture (Option A, as directed) — including the build pivot

New CMake target `map_scalar-driver`, sibling to the pre-existing
`mapz-driver`, added to the **Fortran fixture tree**:

```
GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/@fvdycore/
  model/map_scalar-driver/
    CMakeLists.txt
    main.F90
```

registered via `esma_add_subdirectories(...)` in the parent
`@fvdycore/CMakeLists.txt` (inserted between `mapz-driver` and
`tp-core-driver`). Compiles the **unmodified, authoritative** `../fv_mapz.F90`
directly — no production numerical source is copied or edited.

### First attempt (failed) and why

The initial design (per Stage-3 planning) assumed a small, fully
hand-written set of infra stubs (`stubs/fv_mapz_infra_stubs.F90`: trivial
one-liner replacements for all 11 `use` dependencies, e.g. an
`fv_grid_type` with a single dummy field) would suffice, reasoning that
`map_scalar`/`scalar_profile` themselves call almost none of these
symbols (confirmed true, per the table above). This reasoning is correct
for the two target routines' own bodies, but **Fortran type-checks every
subroutine in a compiled module at compile time, not just the ones
actually invoked** — `fv_mapz_mod` also contains `Lagrangian_to_Eulerian`
and other subroutines that reference real `fv_grid_type`/`fv_flags_type`
fields (e.g. `gridstruct%rsin2`) and the real multi-argument generic
`mpp_update_domains` interface. The trivial stubs did not provide these,
producing ~15 Intel Fortran compile errors (`#6460`, `#6633`, `#6634`,
`#6638`, `#6784`) all located outside the target routines' line ranges.

**Fix**: abandoned the from-scratch minimal-stub strategy. Rewrote
`CMakeLists.txt` to mirror the exact dependency/link/compile-define
configuration already proven to compile+link all of `fv_mapz.F90` by the
pre-existing `mapz-driver` target: real `fv_arrays.F90`, `fv_fill.F90`,
`fv_cmp.F90`, `../../tools/fv_timing.F90`, `../../tools/fv_mp_mod.F90`,
the existing `mapz-driver/stubs/fv_grid_utils_stub.f90` (referenced by
relative path, not duplicated), real `FMS::fms_r8` + `MAPL` linkage, and
`MAPL_MODE SPMD TIMING` compile defines — i.e. "reuse before create."

### Second attempt (also failed, same errors) and the actual root cause

Rebuilding with the corrected `CMakeLists.txt` reproduced the **identical**
errors (`mpp_domains_mod`'s `nest_domain_type` / `mpp_mod`'s
`mpp_broadcast` "not accessible", in `fv_fill.F90` and `fv_arrays.F90`).
Since `mapz-driver` compiles these exact same source files successfully
with a nearly-identical `CMakeLists.txt`, this was not a real dependency
problem. Root cause: the **first, abandoned** build attempt had already
written its fake stub modules' compiled `.mod` files (e.g.
`mpp_domains_mod.mod`, `mpp_mod.mod`, `fv_arrays_mod.mod`) into
`build/include/map_scalar-driver/` — the target's own
`Fortran_MODULE_DIRECTORY`. Deleting only the stub *source* directory
(`stubs/`) left these stale generated `.mod` files in place; because a
target's own module directory is searched with higher priority than the
real FMS `.mod` include path, the second build's compiler kept resolving
`mpp_domains_mod`/`mpp_mod`/`fv_arrays_mod` against the leftover **fake**
module interfaces instead of the real FMS ones.

**Fix**: `rm -rf build/include/map_scalar-driver` and
`rm -rf build/.../model/map_scalar-driver/CMakeFiles` (clearing both the
stale generated `.mod` files and the target's CMake object-file cache),
then a full `cmake .` + `make map_scalar-driver -j4` from clean. This
resolved all errors; build succeeded (see Stage 10).

Numerical semantics changed in either fix: **NO** — both fixes were pure
build-tooling/harness corrections; `fv_mapz.F90` itself was never edited.

`main.F90` (harness driver): single-column (`i1=i2=1`), no
`MPI_Init`/`fms_init`/domain-decomposition calls (confirmed unnecessary —
`map_scalar`/`scalar_profile` need no MPI/domain2d/gridstruct at runtime,
only at Fortran's whole-module compile time). Runs 8 cases, each calling
`map_scalar` twice (repeatability self-check), writing a deterministic
text fixture per case, and reporting `MAP_SCALAR_DRIVER: RESULT=PASS/FAIL`
with `rc=0/1`.

## 5. Test cases (7 required + 1 iv=0)

All `km=kn=10`, Lagrangian grid `pe1` (exponent 1.5) remapped onto a
distinct Eulerian grid `pe2` (exponent 1.4) — a genuine non-trivial remap
in every case, `kord=9` (production value) throughout:

1. `case1_smooth_monotonic` — linear decreasing profile, `iv=1`.
2. `case2_strong_gradient` — step discontinuity at mid-column, `iv=1`.
3. `case3_local_extrema` — sinusoidal non-monotonic bump, `iv=1`.
4. `case4_near_uniform` — near-constant profile (round-off floor), `iv=1`.
5. `case5_variable_layer_thickness` — power-law profile stressing
   thickness sensitivity, `iv=1`.
6. `case6_limiter_activating` — alternating up/down values, `iv=1`
   (designed to activate the monotonicity limiter).
7. `case7_realistic_c512_profile` — power-law, `npz=10`-like standard-
   atmosphere-ish shape, `iv=1`.
8. `case8_positive_definite_iv0` — same shape as case 3, `iv=0`,
   `q_min=0.0` (positive-definite path).

## 6. Fixture-vs-production-savepoint coverage (DEFERRED, per locked Stage-6 finding)

No existing production fixture captures `map_scalar`/`scalar_profile`'s
actual production-runtime arguments (confirmed prior to this stage — no
new fixture generation or GPU job attempted here). Deferred; the
synthetic cases in Stage 5 are the baseline for Prompt 52B until/unless a
real production savepoint at this exact call site is captured separately.

## 7. Output format

Plain, high-precision text (`ES25.17`), one file per case — see
`tests/fortran_baseline/fv_mapz/README.md` for the exact format. Chosen
over a new NetCDF dependency for these tiny single-column arrays.
`mapz-driver`'s existing `output/output.f90` uses an MPI-rank-keyed binary
convention appropriate for its full multi-rank `Lagrangian_to_Eulerian`
scope; not adopted here since this harness is single-column, single-rank,
and the number of values is small enough that plain text is both simpler
and directly diffable/greppable.

## 8. Self-checks

Implemented in `main.F90`'s `run_case`:

- **Repeatability**: `map_scalar` called twice with identical inputs
  within the same process; `any(q2a /= q2b)` checked (bitwise).
  Additionally verified via a **separate process invocation** (Stage 10)
  — `diff -rq` between two independent runs' output directories reported
  bitwise-identical for all 8 fixture files.
- **Finiteness**: `ieee_is_finite_wrap` (elemental, `x==x .and.
  abs(x)<huge(x)`) applied to all `q2` outputs — no NaN/Inf across any
  case.
- **Monotonicity-limiter sanity**: only checked where the routine's own
  contract guarantees it — the positive-definite case (`iv=0`, `q_min=0`)
  asserts `q2 >= -1e-10`; no monotonicity-preservation assertion is made
  for the general (`iv=1`) cases since PPM-family remaps are not
  guaranteed monotone at every interior point for irregular/discontinuous
  inputs (case 2's overshoot at k=5, `238.15` vs. neighboring `220`s, is
  expected PPM near-discontinuity behavior, not a harness bug).

## 9. Build environment

Verified via `build/CMakeCache.txt` (not assumed): `CMAKE_Fortran_COMPILER`
= ifort `2021.13.0` (`/usr/local/intel/oneapi/2024/compiler/2024.2/bin/ifort`),
`MPI_Fortran_COMPILER` = mpiifort (Intel MPI 2021.13), `FV_PRECISION=R8`
— matches `docs/agent_instructions/discover_platform.md`'s "Authoritative
Production FV3/CUDA Runtime Stack (Intel MPI)", not the GCC/OpenMPI dev
stack. Module sequence used for both configure and build:

```bash
module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv
module load comp/gcc/12.3.0
module load comp/intel/2024.2.0
module load mpi/impi/2021.13
module load nvidia/nvhpc-byo-compiler/24.7
export I_MPI_CXX=g++
```

CPU-only, login-node build — no GPU/Slurm required for this harness.
**Runtime note**: the same module environment must also be loaded before
*executing* `map_scalar-driver` (Intel's `libiomp5.so` OpenMP runtime is
not on the default `LD_LIBRARY_PATH` without it) — confirmed via a first
run attempt failing with `error while loading shared libraries:
libiomp5.so: cannot open shared object file`, resolved by loading the
same module stack in the execution shell.

## 10. Build + run results

```bash
cd /discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf/build
cmake .                          # picks up new model/map_scalar-driver subdir
make map_scalar-driver -j4       # rc=0 after the Stage-4 stale-.mod fix
```

Result: `[100%] Built target map_scalar-driver`,
`bin/map_scalar-driver` (11.9 MB). Execution (module stack loaded):

```
CASE [case1_smooth_monotonic] OK=T
CASE [case2_strong_gradient] OK=T
CASE [case3_local_extrema] OK=T
CASE [case4_near_uniform] OK=T
CASE [case5_variable_layer_thickness] OK=T
CASE [case6_limiter_activating] OK=T
CASE [case7_realistic_c512_profile] OK=T
CASE [case8_positive_definite_iv0] OK=T
MAP_SCALAR_DRIVER: 8 cases run, 0 failed
MAP_SCALAR_DRIVER: RESULT=PASS
```

`rc=0`. No NaN/Inf. Bitwise-repeatable within-process (both calls per
case) and across two independent process invocations (`diff -rq` of the
two runs' output directories: identical, all 8 files).

**Validation level for this harness**: IMPLEMENTED, COMPILED, CPU
EXECUTED (no GPU claim made or implied — this is a Fortran-only CPU
baseline harness).

## 11. Divergence-risk statement

`main.F90` calls the **unmodified authoritative** `../fv_mapz.F90`
directly (real production source, not a copy). **DIRECT AUTHORITATIVE
SOURCE — no copy, no divergence risk.** No diff/hash check against a
second copy is needed since there is no second copy.

## 12. CUDA-repo test tree

Created in this repo (`FVdycoreCubed_GridComp`, not the Fortran fixture
tree): `tests/fortran_baseline/fv_mapz/`:

```
tests/fortran_baseline/fv_mapz/
  README.md                 -- build/run instructions, case table, format
  compare_baseline.py        -- Prompt-52B comparison-metric skeleton
  fixtures/
    case1_smooth_monotonic.txt
    case2_strong_gradient.txt
    case3_local_extrema.txt
    case4_near_uniform.txt
    case5_variable_layer_thickness.txt
    case6_limiter_activating.txt
    case7_realistic_c512_profile.txt
    case8_positive_definite_iv0.txt
```

Fixtures are a committed snapshot (regeneratable via the documented
commands in `README.md`), per project convention of preferring checked-in
regeneration code over large binaries for small, cheaply-reproducible
data.

## 13. Comparison metrics (for Prompt 52B)

Defined in `compare_baseline.py` (per-case, over the `kn`-length `q2`
column):

- `max_abs = max_k |q2_cuda(k) - q2_fortran(k)|`
- `max_rel = max_k |q2_cuda(k) - q2_fortran(k)| / max(|q2_fortran(k)|, floor)`,
  `floor = 1e-12` (guards near-zero denominators, e.g. `case4_near_uniform`)
- `RMS = sqrt(mean_k (q2_cuda(k) - q2_fortran(k))^2)`
- `nan_count` — count of NaN/Inf in `q2_cuda`
- `mismatch_count` — count of `k` where `q2_cuda(k) != q2_fortran(k)` (bitwise)

**Policy**: R8 comparison is **bitwise-first** (`mismatch_count==0` and
`nan_count==0` required for PASS by default) — matching Gate C2's
observed round-off-only agreement precedent. Fall back to Gate C2's
existing round-off tolerance methodology (relative-tolerance thresholds
per field) **only if** a justified non-bitwise difference appears (e.g.
FMA contraction or reduction-order differences on GPU) — not loosened
preemptively.

## 14. Prompt 52B CUDA scope

**Required routines**: `map_scalar` + `scalar_profile` only.

**Excluded**: `ppm_profile` (`kord<=7` path) — not required for the
current production C512 Held-Suarez configuration (`kord=9` everywhere
relevant). **Caveat**: this is configuration-dependent, not a permanent
exclusion — if any future run reconfigures `kord_mt`/`kord_wz`/`kord_tm`
to `<=7`, `ppm_profile` becomes required and this scope must be revisited.

No `MOIST_CAPPA`/`USE_COND`-conditional behavior inside either target
routine (confirmed, Stage 2/Stage 0) — irrelevant to the CUDA port's
correctness surface.

## Project status

- Gate C2 (54-stage production dyn_core inventory): unaffected, still
  CLOSED — this prompt does not touch dyn_core's already-validated
  scope.
- `CURRENT_STATE.md`: intentionally **not updated** — `ppm_profile`
  confirmed not required, so no new architectural fact was produced by
  this prompt (the existing §9 "Next Recommended Task" pointer to Phase
  52A remains accurate; Phase 52B is the next step, not yet started).
- No production numerical Fortran source was modified (`fv_mapz.F90`,
  `fv_fill.F90`, `fv_cmp.F90`, `fv_arrays.F90`, `fv_mp_mod.F90`,
  `fv_timing.F90` — all read/linked-against unmodified). Only new files
  were added: `map_scalar-driver/{CMakeLists.txt,main.F90}` in the
  Fortran fixture tree, one `esma_add_subdirectories` line in
  `@fvdycore/CMakeLists.txt`, and this repo's
  `tests/fortran_baseline/fv_mapz/` + this doc file.
- Nothing committed or pushed in either repository (per directive).

## 15. Correction (Prompt 52B): fixtures were single-precision, now fixed

**This section corrects §9/§10 above.** `FV_PRECISION=R8` in
`CMakeCache.txt` selects the R8 **FMS** library variant project-wide, but
does **not** by itself promote a target's own Fortran `real` declarations
to 8 bytes — GFDL FV3 source achieves R8 via generic, un-kinded `real`
promoted by a compiler flag (`${FREAL8}`, expands to `-r8` for ifort)
applied at **target** level, not via `real(kind=8)` in source. The
production `fvdycore` library target applies this flag explicitly (its
own `CMakeLists.txt`, `FV_PRECISION STREQUAL R8` branch); `map_scalar-
driver`'s `CMakeLists.txt` (§4 above) was built by mirroring `mapz-
driver`'s dependency/link structure but never carried over this flag.
Net effect: every fixture in §10/`tests/fortran_baseline/fv_mapz/
fixtures/` was silently compiled and run in **4-byte** `real`, not 8-byte,
despite every check in §9 (compiler path, `FV_PRECISION` cache value)
being individually correct — none of those checks actually probe the
per-target compile-flag set.

**How caught**: direct inspection of `case1_smooth_monotonic.txt`'s
`q_min` sentinel in the CUDA repo (Prompt 52B, not this prompt) —
`-9.99999956202352625E+24` is a float32-quantized representation of the
intended `-1.0e25` (ULP spacing ~1e17-1e18 at that magnitude, matching
binary32, not binary64). Confirmed via `flags.make` diff: `map_scalar-
driver`'s Fortran flags lacked `-r8`; the production `fvdycore` target's
otherwise-identical flags included it.

**Fix**: added an `if (FV_PRECISION STREQUAL R8)` block to `map_scalar-
driver/CMakeLists.txt`, applying `${FREAL8}` via `target_compile_options`
exactly as the parent `@fvdycore/CMakeLists.txt` does. Test-harness/
build-config-only change — `fv_mapz.F90` and all other production
numerical source untouched. Numerical semantics changed: **NO** (for
production code); the *test harness's own compiled precision* changed
from incorrect (R4) to correct (R8), which was the intent all along.

**Verification after fix**: rebuilt `map_scalar-driver` clean, reran all
8 cases (`MAP_SCALAR_DRIVER: RESULT=PASS`, `rc=0`, unchanged from before).
New `q_min` sentinel: `-1.00000000000000009E+25` — correct double-
precision representation (ULP spacing ~1e9 at this magnitude). All other
printed values (`pe1`, `pe2`, `q2_output`) shifted from float32-quantized
patterns (visually: many trailing-zero mantissa bits, e.g.
`3.25911547851562500E+03`) to genuine double-precision values (e.g.
`3.25911538250821150E+03`) — consistent across all 8 regenerated fixture
files. `tests/fortran_baseline/fv_mapz/fixtures/*.txt` in this repo have
been overwritten with the corrected output; §9/§10 above describe the
harness architecture accurately but their specific printed numeric
examples predate this fix.

**Deferred, out of scope for this fix**: the pre-existing `mapz-driver`
target has the structurally identical omission (confirmed by direct
`CMakeLists.txt` comparison) and is very likely subject to the same bug
if ever run/relied upon for numeric output — not fixed here since it is
a separate target this prompt does not otherwise touch.
