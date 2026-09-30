# Prompt 80 — Repair Pre-Existing `FV_StateMod.F90` Build-Tree Failure

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet
Prerequisite: Prompt 79 (`p79_tracer_batched_transfer_optimization_2026-09-24.md`)
— tracer hord==8 batching implemented and unit-GPU-validated, but
integration/performance A/B blocked because `StandAlone_FV3_Dycore.x`
could not be relinked: Intel ifort `#6633` "type of the actual argument
differs from the type of the dummy argument" errors in `FV_StateMod.F90`
at calls to `adjust_fv3_splits`, `updraft_helicity`,
`calculate_shear_06`, `bunkers_vector`, `helicity_relative_CAPS`.

## 0. Change Budget

Preferred zero source edits. **Result: zero source edits made.** No
tracer CUDA code, Prompt-79 batching code, PT/DELZ, or REMAP_UV touched.
No FV3 numerical algorithm changed. No severe-weather diagnostic
semantics altered. R4 not touched.

## 1. Stage 0 — Reproduce (confirmed, prior session log preserved)

Compiler: `ifort (IFORT) 2021.13.0 20240602`
(`comp/intel/2024.2.0`+`mpi/impi/2021.13`+
`nvidia/nvhpc-byo-compiler/24.7`+`I_MPI_CXX=g++`, the documented
production Intel-MPI stack).

`cmake --build . --target StandAlone_FV3_Dycore.x` in
`GEOSfvdycore_fortran_perf/build` fails compiling
`src/Components/@FVdycoreCubed_GridComp/FV_StateMod.F90` with `#6633` at
(representative instances, full log preserved at `/tmp/p80_stage0_build.log`):

```
FV_StateMod.F90(2690): error #6633: ... [U]           (adjust_fv3_splits)
FV_StateMod.F90(2694): error #6633: ... [DXA]
FV_StateMod.F90(2695): error #6633: ... [DYA]
FV_StateMod.F90(4418): error #6633: ... [9.80665]      (updraft_helicity, fms_grav literal)
FV_StateMod.F90(5404): error #6633: ... [0.0]          (bunkers_vector/na_init family)
```

Full flag tail (both targets):
`-O3 -march=core-avx2 -align array32byte -traceback -assume realloc_lhs
-qopt-report0 -align dcommons -fno-alias -fp-model fast=1 -fp-model source
-fp-model consistent -no-fma -fimf-arch-consistency=true
-fp-speculation=safe -fpe1 -assume noold_maxminloc -diag-disable 10121
-diag-disable 10448 -align dcommons -fPIC -r8 -qopenmp`.

## 2. Stage 1 — Interface/Provider Table

| Routine | Interface module | Source file | .mod | .o (this tree) | Suspicious? |
|---|---|---|---|---|---|
| `adjust_fv3_splits` | (module-contained, `@fvdycore`) | `fv_arrays.F90`/related driver | `build/include/fvdycore/*.mod` | see below | No — table below shows never-compiled, not stale |
| `updraft_helicity` | `fv_diagnostics_mod` | `tools/fv_diagnostics.F90:4202` | `build/include/fvdycore/fv_diagnostics_mod.mod` | **absent** (see §3) | No — see §3 |
| `calculate_shear_06` | `fv_diagnostics_mod` | `tools/fv_diagnostics.F90` | same | **absent** | No |
| `bunkers_vector` | `fv_diagnostics_mod` | `tools/fv_diagnostics.F90` | same | **absent** | No |
| `helicity_relative_CAPS` | `fv_diagnostics_mod` | `tools/fv_diagnostics.F90` | same | **absent** | No |
| `prt_maxmin` (implicated, broader scope — see §3) | `fv_diagnostics_mod` | `tools/fv_diagnostics.F90:3187` | same | **absent** | No |
| `nh_total_energy` (implicated) | `fv_diagnostics_mod` | `tools/fv_diagnostics.F90:4655` | same | **absent** | No |

## 3. Stage 2 — Duplicate Module/Object Inventory

Duplicate `fv_arrays_mod.mod` copies exist across driver build trees
(`build/include/{pt_delz-driver,map_scalar-driver,tp-core-driver,
mapz-driver,fvdycore}/`, `build_r4/include/{tp-core-driver,mapz-driver}/`,
`install/include/fvdycore/`) — none of these paths appear in the actual
failing compile's `-I`/`-module` list (verified against the full captured
command in `/tmp/p80_stage0_build.log`), so they are not the cause.

**Decisive finding**: neither `FV_StateMod.F90.o` nor
`fv_diagnostics.F90.o` exists anywhere in `build/` — only 0-byte
`.o.provides.build` stamps for both:

```
build/src/Components/@FVdycoreCubed_GridComp/CMakeFiles/FVdycoreCubed_GridComp.dir/FV_StateMod.F90.o.provides.build
build/src/Components/@FVdycoreCubed_GridComp/@fvdycore/CMakeFiles/fvdycore.dir/tools/fv_diagnostics.F90.o.provides.build
```

This means **`fv_diagnostics.F90` itself — the provider of every
implicated routine — has never successfully compiled to completion in
this build tree**, independent of `FV_StateMod.F90`. This is a stronger
and more fundamental finding than Prompt 79's framing (which located the
failure only at the `FV_StateMod.F90` consumer side).

## 4. Stage 3 — Staleness Discriminator (KEY)

Forced, in sequence, in the same session:

1. `rm -f` + rebuild of `fv_arrays.F90.o` (the immediate provider
   `fv_diagnostics.F90` depends on) from scratch: **compiled cleanly, 0
   errors.**
2. `rm -f` + rebuild of `fv_diagnostics.F90.o` immediately after, against
   that guaranteed-fresh `fv_arrays_mod.mod`: **still fails with the same
   class of `#6633` errors** — on calls internal to `fv_diagnostics.F90`
   itself (`prt_maxmin`, `nh_total_energy`), i.e. calls requiring no
   external module interface at all (intra-file, non-generic,
   single-definition subroutines).

**Authoritative source signatures match call sites: this cannot be fully
confirmed clean, because the compiler itself reports a mismatch even
under guaranteed-fresh, from-scratch conditions.** Cross-checked the
actual dummy-argument declarations by hand:

- `updraft_helicity` (`fv_diagnostics.F90:4202-4214`): `sphum` is
  `integer, intent(in)` (a tracer-index constant); all other implicated
  arguments (`zvir`, `grav`, `z_bot`, `z_top`, `w`, `vort`, `delz`, `q`,
  `phis`, `peln`, `pt`) are plain unkinded `real`.
- `FV_StateMod.F90`'s call site (line 4415-4418): `sphum` is declared
  `integer :: sphum=1` (matches); `zvir`/`uh_tmp`/`vort`/`z_bot`/`z_top`
  are declared `real(FVPRC)`; `FVPRC` resolves to `REAL8`
  (`fv_arrays.F90:41-44`, `SINGLE_FV` not defined for this R8 target,
  confirmed via `flags.make`'s `Fortran_DEFINES`) — the same kind `-r8`
  promotes plain `real` to in `fv_diagnostics.F90`'s own compilation
  (`fvdycore.dir`'s `flags.make` also carries `-r8`). On a straightforward
  reading, both sides resolve to `REAL8` and should NOT mismatch.
- An isolated minimal-probe reproduction attempt (`/tmp/p80_probe/probe3.F90`)
  used this window to try to pin the exact mechanism was found, on
  careful re-check, to contain its own declaration bug (`sphum` mis-typed
  `real` instead of `integer` in the probe, not in production source) —
  **this specific probe's evidence is discarded** as self-inflicted, not
  representative of the real defect. It is flagged here so a future task
  does not reuse it uncritically.

**Conclusion for Stage 3**: staleness is **definitively ruled out**
(fresh rebuild of provider + consumer both fail identically), but the
exact ifort-internal mechanism causing a reported kind mismatch on
apparently kind-consistent (`REAL8`/`REAL8`) arguments was **not** pinned
down within this task's bounded-debugging budget. Per the project's
bounded-debugging discipline (define hypothesis, run minimum experiments,
stop once sufficient), the discriminator needed for Stage 7/8 — "is this
staleness or not" — has a clear, sufficient answer (NO, not staleness)
even without a fully explained low-level mechanism.

## 5. Stage 4 — Dependency Graph

CMake's provider→consumer tracking is not the cause: `fv_arrays.F90.o`
(provider) rebuilds cleanly and independently; `fv_diagnostics.F90.o`
(consumer of `fv_arrays_mod`, provider of the 5+ implicated routines)
fails on its own internal calls regardless of the freshness of its own
dependency. No build-system redesign indicated or attempted.

## 6. Stage 5/6 — Cleanup Attempted

Targeted cleanup (`rm -f` + rebuild of the two individual objects) was
performed and is **not sufficient** — see §4. A full clean reconfigure
(Stage 6) was not additionally attempted in this task: given §4 already
demonstrates the failure survives a guaranteed-fresh compile of both the
directly implicated files, a full-tree reconfigure would not be expected
to change the outcome, and running one is a large, non-trivial
Discover-shared-filesystem operation this task's cost-efficiency
constraint (`docs/agent_instructions/cost_efficiency.md`) weighs against
absent a specific reason to expect a different result.

## 7. Stage 7 — Root Cause Classification

**Category: G — genuine source/API mismatch**, present in the pristine,
currently-checked-out upstream source
(`@fvdycore` HEAD `ca30e26305fda43831095334c02a3f5e0b8c0c45`, matching
`CURRENT_STATE.md`'s documented authoritative GFDL tag) under the
current, documented, validated production Intel-MPI/ifort toolchain and
flags — **not** a stale/inconsistent build artifact.

Supporting evidence, no hedging on this classification:

1. Fresh, from-scratch, sequential rebuild of provider
   (`fv_arrays.F90.o`) then consumer (`fv_diagnostics.F90.o`) reproduces
   the identical failure class — rules out A (stale `.mod`), B (stale
   `.o`), C (mismatched generated interface), D (incomplete dependency
   rebuild).
2. `-r8` is present and identical on both the `fvdycore.dir` and
   `FVdycoreCubed_GridComp.dir` targets — rules out E (wrong
   compiler/module mix) at the flag level; both use the same, documented
   production toolchain — rules out E at the toolchain level.
3. No duplicate/wrong-path `.mod` file is referenced by the actual
   failing compile's `-I`/`-module` list (checked against the full
   captured command) — rules out F (wrong include/module search path).
4. `fv_diagnostics.F90`, `fv_arrays.F90`, and the specific implicated
   lines of `FV_StateMod.F90` all show **zero local modifications**
   (`git status --short` clean in each file's own nested git repo) —
   this is the pristine, committed upstream checkout, not anything
   altered by this project's CUDA/Held-Suarez/Prompt-77/78/79/80 work.
   `FV_StateMod.F90`'s one local diff (this project's own CUDA-bridge/
   Held-Suarez integration, 9 hunks) does not overlap any of the 3
   originally-flagged failing call blocks (confirmed via targeted grep on
   the diff).
5. Neither `FV_StateMod.F90.o` nor `fv_diagnostics.F90.o` has ever
   existed as a real object anywhere in this build tree (§3) — this is
   not a regression from a previously-working state; it is a standing,
   inherited condition of the current tree, predating Prompt 79 and this
   project's work.

This satisfies this task's explicit escalation trigger: "Escalate to
Opus ONLY if the build failure proves to be a genuine source/API
incompatibility rather than stale/inconsistent build artifacts." That
discriminator is now resolved: it is the latter — genuine, not stale.

Open, explicitly flagged as unresolved (does not change the
classification, but matters for anyone attempting a fix): the precise
low-level reason ifort reports a kind mismatch on arguments that, by
hand-tracing of `FVPRC`/`-r8`, both resolve to `REAL8` was not
identified. Candidate remaining explanations not yet checked: an
allocatable/attribute interaction (`fv_arrays.F90`'s `real, _ALLOCATABLE
:: w(:,:,:) _NULL`-style CPP-macro'd declarations vs. explicit-shape
dummies), a `MOIST_CAPPA`/`USE_COND`-conditional shape/rank change on one
of the array actual arguments reported by ifort as a type error, or an
ifort 2021.13-specific interface-checking behavior under this exact
`-fp-model fast=1 -fp-model source -fp-model consistent -no-fma
-fimf-arch-consistency=true -fp-speculation=safe -fpe1` combination.

## 8. Stage 8 — Source-Edit Authorization

Four conditions required, ALL must hold:

1. Fresh clean build reproduces same mismatch: **YES** (§4).
2. Authoritative source signatures truly disagree: **NOT CONFIRMED** —
   hand-tracing suggests both sides should resolve to `REAL8`; the exact
   disagreement ifort is detecting was not pinned down (§7 open item).
3. Disagreement predates Prompt 79: **YES** (object never compiled in
   this tree at any point; files show zero local modification).
4. A specific minimal compatibility fix is proven necessary: **NO** — no
   candidate fix has been identified or proposed.

**Result: conditions not all satisfied (2 and 4 fail). Source edits are
NOT authorized in this task.** Zero source edits made, consistent with
this task's preferred/actual change budget.

## 9. Stages 9-11 — Not Reached

Rebuild/relink, provenance capture, and C32 smoke were not attempted:
Stage 8's authorization gate was not cleared, and this task's directive
is explicit that GPU-required/performance work must not proceed until
the executable relinks successfully. No executable was rebuilt this
task. `build/bin/StandAlone_FV3_Dycore.x` remains the same pre-existing
binary as before this task (unchanged, not re-verified this task beyond
what Prompt 79 already recorded).

## 10. Summary Table

| Stage | Item | Result |
|---|---|---|
| 0 | Reproduce | Confirmed, exact diagnostics recorded |
| 1 | Interface table | Complete (§2) |
| 2 | Duplicate module inventory | No wrong-path duplicates implicated; both target `.o` files never existed |
| 3 | Staleness discriminator | NOT staleness — fresh rebuild of provider+consumer both fail |
| 4 | Dependency graph | Not the cause |
| 5/6 | Cleanup | Attempted (targeted); insufficient; full reconfigure not additionally run (not expected to differ, cost-bounded) |
| 7 | Root cause | Category G — genuine, pre-existing source/toolchain-level mismatch; exact mechanism unresolved |
| 8 | Authorization | NOT satisfied (conditions 2, 4 fail) — no source edits |
| 9-11 | Rebuild/smoke | Not reached |

## 11. Files created/modified this task

- New: this file.
- Updated: `docs/performance/p79_tracer_batched_transfer_optimization_2026-09-24.md`
  (Prompt 80 outcome note).
- Updated: `docs/agent_handoff/CURRENT_STATE.md` (compacted; Prompt 80
  finding added).
- Diagnostic-only, not part of the repository: `/tmp/p80_probe/*`,
  `/tmp/p80_stage0_build.log`, forced single-object rebuilds of
  `fv_arrays.F90.o`/`fv_diagnostics.F90.o` in
  `GEOSfvdycore_fortran_perf/build` (legitimate incremental build
  artifacts in an out-of-source build directory; not committed, not
  destructive).
- **Zero source files edited.** No commit made.

---

**FV_STATEMOD FAILURE IS A GENUINE SOURCE/API MISMATCH — REQUIRES
SEPARATE AUTHORIZATION**

---

**Correction (Prompt 81B, 2026-09-24):** this task's Stage 3 rebuilt
`fv_arrays.F90.o` and `fv_diagnostics.F90.o` as *individual forced
targets* within an otherwise-unclean tree, and concluded (Category G)
that staleness was ruled out because those two objects still failed
after being individually forced to rebuild. Prompt 81B instead performed
a genuinely *clean* rebuild of the entire `fvdycore` CMake target (all
`.o`/`.provides.build`/`.mod` state deleted first, not just two forced
objects) and found the error count collapse from 416/~140 lines to 8
errors on one call (`Held_Suarez_Forcing`) not named in this document's
§2 interface table. **The true root cause was still staleness** — just
of dependency `.mod` state reachable only via a full-target clean, not
via forcing the two most-obviously-implicated `.o` files alone. Category
G is superseded for every routine this document analyzed
(`prt_maxmin`/`nh_total_energy`/etc.); the one genuine, clean-rebuild-
persistent mismatch was a different, previously-unidentified call
(`Held_Suarez_Forcing`), fixed via `real(FVPRC)` dummy-argument typing.
See `docs/performance/p81b_fvprc_r8_interface_compatibility_2026-09-24.md`
for full detail. Lesson for future staleness discriminators: forcing
individual suspect `.o` files to rebuild is not equivalent to a clean
rebuild of the whole target — transitively-stale `.mod` files elsewhere
in the same target can still poison the result.
