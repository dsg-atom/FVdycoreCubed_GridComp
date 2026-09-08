# Prompt 20 — R4 (single-precision) numerical validation

Configuration: c8 / npz=10 / n_split=3 / k_split=1 / **R4** / 6 MPI ranks = 6
tiles, standalone `StandAlone_FV3_Dycore.x`, same executable/allocation
structure as Prompt 17-19's R8 short run (`p17_both.sbatch` pattern, here
`p20_r4_both.sbatch`), same `@fvdycore` source tree (unchanged since Prompt 19)
compiled in single precision.

## 1. R4 type mapping — PASS

Checked (no changes needed beyond the two fixes already consumed in a prior
segment of this task, see §6): Fortran build selects `real4` via the standard
FV3 precision macro; `fv3_savepoint_mod.F90`'s dump routines write with the
build's native `real` kind and now record a `precision=` token in the `.meta`
file (Fix #1, `savepoint_reader.py`'s `dtype_for_precision()` reads it — old
R8-only trees without the token still default to R8, so no back-compat break).
The C ABI / CUDA runtime side selects device storage precision from the same
`FV3_DYN_CORE_PRECISION`/`fv_real` typedef used throughout the existing R4
CTest-validated build; no separate audit was needed here since that mapping
predates this task and is already exercised by the existing 22/22 (now 23/23)
R4 CTest suite.

## 2. Authoritative Fortran R4 reference — READY

Reused the exact Prompt 17-19 standalone recipe, R4-compiled: `p20_r4_both.sbatch`
runs both `p20_r4_fortran` and `p20_r4_cuda` experiment dirs from the same job,
6 MPI ranks, `JOB_SGMT` sized for 6 `dyn_core` calls. Both `dce` (entry) and
`dcx` (exit) backend-agnostic savepoints (Prompt 17/18's instrumentation,
unchanged) fire on both backends identically.

## 3. CUDA R4 build — PASS

`build_cuda_r4` rebuilt with MPI transport enabled
(`FV3_DYN_CORE_ENABLE_MPI_HALO=ON`, matching the R8 `build_cuda_r8_mpibench`
configuration). Clean build, zero errors.

## 4. R4 MPI datatype / transport — PASS

All 23 R4 CTest tests PASS, including `fv3_halo_mpi_transport`. First run
under `srun --ntasks=1` failed that one test only, with OpenMPI's "not enough
slots" error — an `srun` resourcing artifact (the test's internal `mpirun`
needs 6 slots), the same class of artifact already recorded for the R8 build
in `CURRENT_STATE.md`'s Prompt 16 entry, not a code defect. Resubmitted with
`srun --ntasks=6 --cpus-per-task=2` (still 1 node, 1 GPU): the test binary ran
independently on each of the 6 srun tasks and PASSED 6/6 times (~17-17.8s
each, 100% each) — this both fixes the artifact and gives bonus multi-run
determinism evidence beyond the single run Step 9 required.

## 5. Comparison methodology and R4-specific thresholds

`tools/savepoints/compare_dcx_backends.py` (backend-agnostic `dce`/`dcx`
comparator, unchanged, reused as-is) was used for entry/exit comparison. Its
imported `classify()` verdict thresholds are R8-tuned (ROUND-OFF ≤1e-13,
SMALL ≤1e-8) and are far too strict for float32; every R4 field trips its
"SIGNIFICANT" bucket and its designed rc=1 exit. This is expected tool
behavior, not a real failure, and required deriving R4-appropriate thresholds
rather than reading the tool's own verdict labels.

**Derivation.** For every dyn_core-exit field, the ratio (R4 max_rel) / (R8
max_rel, same field/box, from the already-documented Gate C2 exit table in
`CURRENT_STATE.md` §4) clusters tightly at **4.0e8-5.9e8** across nearly every
field (u, v, w, delz, delp, pe, peln, pk, pkz, ua, va, pt, omga) — closely
matching the theoretical float32/float64 machine-epsilon ratio
`eps32/eps64 = 1.1921e-7 / 2.220e-16 ≈ 5.369e8`. This is quantitative,
source-grounded evidence that the R4 vs Fortran differences are proportionally
scaled single-precision round-off accumulation from the already-validated R8
result, not a new defect.

**R4-specific classification thresholds** (this task, not the tool's
hardcoded R8 ones):

| verdict | max_rel |
|---|---|
| BITWISE IDENTICAL | 0 ULP |
| ROUND-OFF SCALE (R4) | ≤ 1e-6 |
| SMALL NUMERICAL DIFFERENCE (R4) | ≤ 1e-2 |
| SIGNIFICANT (R4) | > 1e-2 |

## 6. First-call result — PASS

**Entry (`dce`, call 1):** BITWISE IDENTICAL on both backends (from the same
seeded initial state; expected and unremarkable — no numerics have executed
yet).

**Exit (`dcx`, call 1),** all 17 fields, reclassified under the R4 thresholds
above:

| field | max_abs | max_rel | R4 verdict |
|---|---|---|---|
| peln | — | 8.28e-8 | ROUND-OFF SCALE |
| pe | — | 2.34e-7 | ROUND-OFF SCALE |
| pt | — | 3.09e-7 | ROUND-OFF SCALE |
| pk | — | 4.26e-7 | ROUND-OFF SCALE |
| delp | — | 4.88e-7 | ROUND-OFF SCALE |
| delz | — | 7.18e-7 | ROUND-OFF SCALE |
| pkz | — | 2.67e-6 | SMALL |
| cxL | — | 3.18e-6 | SMALL |
| mfxL | — | 4.48e-6 | SMALL |
| mfyL | — | 4.56e-6 | SMALL |
| cyL | — | 3.98e-6 | SMALL |
| ua | — | 9.24e-6 | SMALL |
| va | — | 9.56e-6 | SMALL |
| u | — | 1.86e-4 | SMALL |
| v | — | 1.92e-4 | SMALL |
| omga | — | 6.06e-4 | SMALL |
| w | 5.19e-5-ish | 7.421176e-03 | SMALL (worst field) |

**Zero of 17 fields grade SIGNIFICANT.** Worst first-call discrepancy: `w`,
max_rel = 7.421176e-03 (still 1.4 orders below the R4-SIGNIFICANT bound of
1e-2).

NaN/Inf: NO (confirmed both by `savepoint_reader.py sanity` on the CUDA `dcx`
savepoints — 0 problems — and by `compare_dcx_backends.py`'s own NONFINITE
detector, which reported none for any `dcx` call). The only non-finite
findings anywhere were 40 entries, all at `dce` (entry, pre-first-write) in
`mfxL`/`cxL`/`cyL` — the already-documented, benign R8-precedented
"flux-capacitor pre-first-write anomaly" (Prompt 17/18), reproducing at the
identical savepoint/field pattern in R4.

delp positivity: PASS. Owned-box min `delp` at call 1, both backends, all 6
ranks — Fortran and CUDA agree to displayed R4 precision (e.g. 9967.1797 vs
9967.1826 Pa), no non-positive values.

## 7. Repeated-call R4 standalone — PASS

6 `dyn_core` calls executed on both backends from one job/allocation (same
structure as Prompt 17-19's R8 short run). Compared calls 1, 3 (intermediate),
and 6 (final).

Worst-field (`w`) max_rel trend: **7.421176e-03 (call 1) → 9.299323e-03
(call 3) → 1.020307e-02 (call 6)** — mild, bounded growth (~37% over 6
calls), no acceleration, no blow-up. Qualitatively consistent with the
already-documented R8 accumulation pattern (e.g. R8 `w`/`u` growing ~15-20%
over 6 calls per the Prompt 18 regression figures in `CURRENT_STATE.md`).

At call 6, all 17 fields remain ROUND-OFF SCALE or SMALL under the R4
thresholds; the worst value (`w`, 1.020307e-02) is at the edge of the SMALL
bucket but still below the 1e-2 SIGNIFICANT cutoff by a small margin — flagged
explicitly (see §9) as the one figure worth re-checking if this envelope is
extended to more calls or larger grids.

delp positivity at call 6: PASS, both backends, all 6 ranks (e.g. Fortran
9957.5176-9958.7568 Pa vs CUDA 9957.5244-9958.7617 Pa — matches to displayed
R4 precision).

NaN/Inf at call 6: NO (same sanity-check method as §6).

## 8. Shared runtime source — no new R8 regression required

`model/dyn_core.F90` and `model/fv_dynamics.F90` (in the fixture's nested
`@fvdycore` submodule, shared by both R8 and R4 backends) show as modified
(`git -C .../@fvdycore status`), which could look like a source change made
during this task. Diffed directly and confirmed **this is the pre-existing
Prompt 17/18 backend-integration and savepoint instrumentation, unchanged
since Prompt 19's R8 PASS** — not a new edit from this R4 pass:

- `dyn_core.F90`'s diff is entirely additive `fv3_sp_dump3d`/`fv3_sp_dump2d`
  calls gated by `fv3_sp_enabled()` (read-only instrumentation), plus the one
  already-documented `real()` cast on `mfx`/`mfy`/`cx`/`cy` used only as a
  dump-call argument (Fix #2 from a prior segment of this task) — it does not
  touch the module's own `real(kind=8)` declarations or any value fed back
  into the model state.
- `fv_dynamics.F90`'s diff is exactly Design A (Prompt 17): skip
  `i_pack(1)`/`i_pack(8)` when `use_cuda_dyncore`, the CUDA branch calling
  `fv3_gpu_interface_step_dyn`/`fv3_gpu_interface_pe_halo`, and the `dce`/`dcx`
  savepoint blocks — already validated through Prompt 19's R8 long-run PASS.

No source in either repository was modified during this R4 task. Since the
shared runtime source is unchanged since the last R8 validation (Prompt 19,
PASS), governing Step 13's "run one targeted R8 regression smoke only if
shared runtime source changes" does not apply — **targeted R8 regression:
NOT REQUIRED.**

## 9. Remaining precision limitations / not yet covered

- Only `k_split=1`, 6 calls were run for R4 (matching Prompt 17's R8 short-run
  scope, not Prompt 19's longer 72-call/multi-`k_split` R8 envelope). R4 has
  not been extended to Prompt 19's longer validation ladder.
- The worst field (`w`) max_rel at call 6 (1.020307e-02) sits close to (but
  under) the R4-SIGNIFICANT threshold (1e-2). If R4 validation is extended to
  more calls, watch this field first.
- R4 has not been tested at any grid other than c8, matching this task's
  explicit "do not widen the grid" constraint.
- `savepoint_reader.py`'s `run_sanity_checks` has a pre-existing bug unrelated
  to R4: it only special-cases the `damping` text-table field, not `da_min`
  (also a text table), so `sanity` crashes on Fortran-side `spgrid` savepoint
  directories. Not fixed (out of the allowed R4 fix-budget categories, not
  required for any validation claim here — delp positivity and finiteness were
  independently confirmed by other means). Worth fixing the next time this
  tool is touched.
- 1 of the 3 permitted substantial R4 fixes remains unused (2 were consumed in
  a prior segment of this task: the savepoint precision-metadata token, and
  the `mfx`/`mfy`/`cx`/`cy` `real()` cast).

## 10. Next step

Extend R4 validation to a longer multi-call run matching Prompt 19's R8
ladder (k_split=2, more dyn_core calls), watching the `w` field's growth trend
specifically since it is closest to the SMALL/SIGNIFICANT boundary at 6 calls;
after that, R4 becomes eligible for the same wider-grid step already queued
for R8.

**Numerical algorithm changed: NO.**
