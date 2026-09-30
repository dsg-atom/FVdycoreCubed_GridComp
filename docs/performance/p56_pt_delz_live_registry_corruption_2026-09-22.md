# Prompt 56 — Localize Live Registry-View Numerical Corruption (pt/delz CUDA remap)

Date: 2026-09-22. Branch: `optimization/cuda-fv3-performance`. Model: Sonnet
(no Opus escalation — the registry/view hypothesis was fully resolved by
targeted inspection + direct runtime comparison, never became ambiguous).

## 1. Scope and starting point

Prompt 55 fixed a memory-safety OOB in `pt_delz_remap_kernel` and confirmed
GPU EXECUTED (0 compute-sanitizer errors), but the first real integrated
invocation (job 58512259) showed severe, immediate, CUDA-arm-only numerical
corruption (~1.9M `Range_Warn`/~1.8M `NaN` hits, widespread `RIEM3-BAD`
diagnostics) despite every standalone/production-shape fixture test passing
at roundoff. Prompt 56's mandate: determine whether this is a live
registry-view (stride/extent/pointer-origin/layout) defect distinct from
anything a standalone fixture could reproduce, or actual kernel arithmetic —
**without changing the kernel first** — using a bounded, diagnostics-first
procedure with a max-one-narrow-fix change budget.

## 2. Stage 0-3 — static re-derivation (no source change)

Re-derived every field's live registration axis order directly from
`fv3_gpu_field_registry.F90`'s `add1`/`add3` call sites, cross-checked
against the authoritative `fv_mapz.F90` array declarations (tag
`geos/v3.0.0`):

| Field | Registration | Axis order | Halo |
|---|---|---|---|
| `pe` | `add3(is-1,ni+2, 0,npz+1, js-1,nj+2)` | (i,k,j) | 1 |
| `peln` | `add3(is,ni, 0,npz+1, js,nj)` | (i,k,j) | 0 |
| `pt`/`delz`/`delp` | `add3(isd,nih, jsd,njh, 0,npz)` | (i,j,k) | 3 |
| `pk` | `add3(is,ni, js,nj, 0,npz+1)` | (i,j,k) | 0 |
| `pkz` | `add3(is,ni, js,nj, 0,npz)` | (i,j,k) | 0 |
| `ak`/`bk` | `add1(0, npz+1)` | flat 1D | n/a |

Confirmed `pt_delz_remap_kernel`'s `.at(i,k,j)` calls for `pe`/`peln`
specifically (the Prompt 55 fix) are self-consistent with this table, and
confirmed `fv3_vertical_kernels.cu`'s `riem_solver3_kernel` (the downstream
consumer of this kernel's output) uses the identical `(i,k,j)` convention
for `pe_out`/`peln_out`, `(i,j,k)` for `pk_out` — ruling out a cross-kernel
axis-order disagreement between writer and reader.

## 3. Stage 4 — live registry-view metadata dump (diagnostic-only)

Added a one-time, env-var-gated (`RUN_CUDA_FV3_PTDELZ_DUMP_VIEWS`), host-side
diagnostic (`p56_dump_view_once`, `cuda/dyn_core/vertical/fv3_pt_delz_remap.cu`)
printing `data,rank,element_bytes,lower,extent,stride` for every field the
launcher touches. Zero kernel/device-code change; inert unless the env var
is set.

Job 58512732 (C32, `p56_stage4_c32.sbatch`, real production registry, binary
sha256 `6065cc48de290c79e8b58d690126a5cff655fc19bdc9f7798610d98191aad637`):
live dump matched the Stage 0-3 static predictions **exactly** for every
field (pointer, rank, element_bytes, lower, extent, stride) — e.g. `pe`:
`lower=(0,0,0) extent=(34,11,34) stride=(1,34,374)`, consistent with Fortran
column-major `pe(is-1:ie+1, 0:npz, js-1:je+1)`, i.e. (i,k,j) with i fastest.

**Conclusion: the registry-view hypothesis is refuted at the metadata
level.** No stride/extent/pointer-origin/layout mismatch exists between what
`fv3_gpu_field_registry.F90` registers and what the launcher's views
describe.

## 4. Stage 5 — pre-kernel INPUT value dump (diagnostic-only, one self-correction)

Added a second one-time diagnostic (`p56_dump_column_values3/1`) that reads
actual pre-kernel device memory contents via blocking `cudaMemcpyDeviceToHost`
at one fixed interior column, using the same offset arithmetic as `.at()`.
Zero kernel/device-code change.

**First attempt (job 58512808) was itself buggy**: the helper hard-coded the
`(i,j,k)` positional convention for every field, but `pe`/`peln` require the
`(i,k,j)` convention (per §2/§3). This made the diagnostic read `pe` at a
fixed pseudo-k (from the wrong argument slot) while sweeping over a
pseudo-j range of only 0-10 — producing an apparently near-constant
~50,055 Pa "profile" that looked exactly like the kind of registry
corruption Prompt 56 was hunting for, but was in fact a bug in the new
instrumentation itself, not in the production registry or kernel. This is
consistent with Prompt 56's own risk framing (a wrong diagnostic can
manufacture a false positive that looks like the thing being searched for)
and is exactly why Stage 4's independent metadata dump (§3, unaffected by
this bug) was not, by itself, sufficient to close the investigation.

**Fix**: added an `axis_ikj` parameter to `p56_dump_column_values3`, set
`true` only for the `pe`/`peln` call sites, matching `pt_delz_remap_kernel`'s
own `.at(i,k,j)` usage. Rebuilt (`fv3_wp_c_vertical`), relinked
(`StandAlone_FV3_Dycore.x`, sha256
`239502f5b1bad59b38e1e552e0107c9521e31312bf2fd161666a3900d14c6f25`), reran
(job 58512938).

**Corrected result — pre-kernel inputs are physically sane**, one interior
column (i=16,j=16, C32/L10 Held-Suarez, all 6 ranks agree within roundoff):

| Field | k range | Values | Physically sane? |
|---|---|---|---|
| `pe` | 0..10 | 100 Pa → 1.0000e5 Pa, monotonic increasing | YES (ptop=100 Pa at k=0 matches `ak(0)`, bk(0)=0; ~1e5 Pa surface at k=10) |
| `peln` | 0..10 | 4.6052 → 11.513, monotonic increasing | YES (`= ln(pe)` exactly) |
| `ak` | 0..10 | 100, 10100, 8977.8, ..., 0 | YES (matches synthetic L10 test hybrid coefficients, ak(km)=0) |
| `bk` | 0..10 | 0, 0, 0.111, ..., 1.0 | YES (bk(km)=1, standard) |
| `pt` | 0..9 | ~217-308 K, increasing with k | YES (Held-Suarez-plausible troposphere profile) |
| `delz` | 0..9 | -30169 to -690 m, all negative | YES (height decreases downward, sign convention correct) |
| `delp` | 0..9 | ~9988-10000 Pa | YES (uniform ~10 hPa layers, sane for L10) |
| `pkz` | 0..9 | 8.98 → 26.43, monotonic increasing | YES |

**Conclusion: the registry-view hypothesis is refuted at the value level
too.** Every field `pt_delz_remap_kernel` reads is registered correctly
(Stage 4) and contains correct pre-kernel data (Stage 5, corrected). The
same job (58512938) still shows the corruption signature in its production
diagnostics: `Range_Warn=1,901,587`, `NaN=1,843,495`, 385 `RIEM3-BAD` lines
in this run's log, `Abort=0` (memory-safe, no crash — consistent with
Prompt 55).

## 5. Stage 6/7 decision point — Category G vs Category H

Per Prompt 56's own decision rule: Stage 5 showing inputs already wrong
before kernel launch would mean Category G (sync/staleness), fixable within
a narrow registry/sync change. Stage 5 instead shows **all inputs are
correct** — this is Category H (the kernel's own arithmetic, or a
downstream consumer, produces wrong output from correct inputs). Per
Prompt 56's explicit stop condition, this requires **stopping and requesting
fresh authorization** rather than attempting a second kernel-math fix in
this prompt. `pt_delz_remap_kernel`'s arithmetic (`fv3_pt_delz_remap.cu`)
and `map_scalar_column`/`scalar_profile`/`cs_limiters`
(`fv3_map_scalar_column.hpp`, Gate-C2-adjacent, standalone-validated in
Prompt 52B) were **not modified or further inspected for defects** in this
prompt, consistent with the forbidden-list ban on touching Gate C2 kernels
and Riemann-solver code without authorization. Stages 8-11 (bitwise
comparison, C512 A/B) were not attempted.

## 6. Change budget accounting

One narrow, diagnostic-only file was touched:
`cuda/dyn_core/vertical/fv3_pt_delz_remap.cu` — two additive blocks
(`p56_dump_view_once` + call site; `p56_dump_column_values3/1` + call site),
both env-var-gated and inert by default, plus the one self-correction in
§4 (an `axis_ikj` parameter added to the diagnostic helper itself, not to
`pt_delz_remap_kernel` or any production code path). No kernel math, no
registry-source (`fv3_gpu_field_registry.F90`) change, no Riemann-solver
change, no MPI/runtime-topology change. This is within the "max one narrow
fix plus minimal one-time diagnostics" budget — the one fix applied was to
the diagnostic tool itself, not to production source, because the
diagnostic tool's own bug was what needed fixing to get a trustworthy
answer.

## 7. Summary

- **Registry-view hypothesis (metadata): REFUTED** — Stage 4, live dump
  matches static re-derivation exactly for all 9 fields.
- **Registry-view hypothesis (values): REFUTED** — Stage 5 (corrected),
  live pre-kernel inputs are physically sane for all 8 value-checked
  fields.
- **Root cause localized to: Category H — kernel arithmetic (inside
  `pt_delz_remap_kernel` itself, or its downstream Riemann-solver
  consumption in `fv3_vertical_kernels.cu`, unmodified) — not registry/view
  metadata, not stale/unsynchronized inputs.**
- **Not yet identified**: the specific divergent expression/line inside
  `pt_delz_remap_kernel`'s arithmetic (peO/pn2 derivation, `map_scalar`
  remap call, delz-post/pkz joint write, or the closing delp/pe/peln/pk
  update) responsible for the corruption. Localizing further requires
  either (a) a Fortran-vs-CUDA savepoint comparison of this kernel's own
  outputs (peO/pn2/remapped pt/remapped delz/pkz) against the authoritative
  `Lagrangian_to_Eulerian` computation at the same column, or (b) inspecting
  the kernel's arithmetic against `fv_mapz.F90` line-by-line — both are
  kernel-math-adjacent activities Prompt 56 explicitly reserves for a
  separately-authorized prompt.
- Fixing this is out of Prompt 56's scope by its own explicit rule.

**PT/DELZ LIVE REGISTRY BUG IDENTIFIED — FIX REQUIRES NEW KERNEL-MATH AUTHORIZATION**

## 8. Update (Prompt 57 — partial resolution)

Prompt 57 picked up Category H directly and split it into two distinct
bugs. **Bug 1**, a double-transform of `pt` (Fortran's own pre-transform
ran unconditionally before `use_cuda_pt_delz_remap` was computed, so the
CUDA kernel then re-applied the identical transform on top of it), fully
explains this document's Stage 4/5 corruption signature and is now FIXED
and VERIFIED (job 58516240, Stage 11 regression PASS). A **second,
distinct** divergence (non-round-off, growing call-over-call, isolated to
inside the pt/delz CUDA dispatch by a control experiment, job 58516315)
remains unfixed and requires its own authorization. Full detail:
`docs/performance/p57_pt_delz_kernel_first_divergence_2026-09-22.md`.
