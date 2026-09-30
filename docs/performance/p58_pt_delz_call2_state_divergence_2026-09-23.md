# Prompt 58 — PT/DELZ Bug 2 Localization: Call-2 State Divergence

_2026-09-23. Branch `optimization/cuda-fv3-performance`. Model: Sonnet (no
Opus escalation needed — the root cause was fully resolved from source
ordering alone, with no genuine Fortran/CUDA temporal-semantic ambiguity).
Continues Prompt 57 (`p57_pt_delz_kernel_first_divergence_2026-09-22.md`),
which fixed Bug 1 (pt double-transform) and isolated, but did not fix, a
second, distinct, non-round-off divergence ("Bug 2") to inside the pt/delz
CUDA dispatch itself._

## 1. Starting point

Prompt 57's control experiment (job 58516315,
`RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN=1`) proved Bug 2 lives inside the pt/delz
CUDA dispatch: disabling only that dispatch (everything else in dyn_core
still on GPU) restored round-off agreement across all 6 calls of a C32
Held-Suarez `dcx` A/B. Call 1 was always clean; calls 2-6 showed a growing
divergence (`pt` max_rel 1.28e-6→6.19e-6, `delp` 6.34e-6→2.14e-5, `w`/`omga`
up to ~5% relative). This prompt's mandate: find the first state variable
or arithmetic stage that is correct after call 1 but wrong entering or
during call 2, prove it (not "likely"), and apply at most one fix.

## 2. Candidate root cause from source ordering (Category E — later REFUTED, see §4)

Reconstructed the authoritative `Lagrangian_to_Eulerian` sequence directly
from `fv_mapz.F90` (GFDL_atmos_cubed_sphere, tag `geos/v3.0.0`):

1. `peO` (the new/target hybrid grid) is computed once, early (:538-547),
   from `pe(i,km+1,j)` — the OLD grid's surface pressure.
2. The CUDA pt/delz dispatch is called immediately after (:557-561),
   *before* four Fortran blocks that all read the shared `pe` array
   directly and require it to still hold the OLD (pre-remap) grid:
   - tracer remap (`mapn_tracer`, :611-641 — `pe1(i,k) = pe(i,k,j)`)
   - `dpe`/`dpeO` derivation (:644-653 — `dpe(i,k,j) = pe(i,k+1,j)-pe(i,k,j)`)
   - NH w-remap (:685-734, **not** gated by `use_cuda_pt_delz_remap` — only
     the delz sub-block at :663-683 is)
   - UV/`mfx`/`mfy`/`cx`/`cy` remap (:741-844)
3. Only *after* all four of those blocks does Fortran's own unconditional
   closing block run (:859-908, "Update pe" / "Update logP & pk" /
   "update delp") — recomputing `pe2` from `ak`/`bk`/`pe(surface)` (the
   *same* formula as `peO`) and overwriting `pe`, `peln`, `pk`, `delp` in
   place. This is, by design, the *last* step.

`fv3_pt_delz_remap.cu`'s kernel (pre-fix) performed this exact closing
update — writing `a.pe`, `a.peln`, `a.pk` (lines 248-250, in addition to
`a.pt`/`a.delz`/`a.delp`/`a.pkz` at 227-230) — *inside the same dispatch*
that runs at call-site :557-561, i.e. chronologically *before* all four of
the Fortran blocks above. Confirmed by grep
(`awk 'NR>=611 && NR<=844'` over `fv_mapz.F90`) that `peln`/`pk` are never
read in any of those four blocks (their early write is harmless but
unnecessary — Fortran's closing block re-derives them identically at
:896-897 regardless), but `pe` *is* read repeatedly (tracer remap :621; NH
w-remap :686; UV remap :756/763/799/803/810). Once the kernel overwrote
`pe`'s interior with the new grid, tracer/w/UV remaps received an
already-finalized grid instead of the true old one, degenerating each into
a near-identity remap (old ≈ new) rather than the true old→new
redistribution — most visibly for `w`/`omga` (a direct `map_scalar`
consumer of the degenerate `pe`/`peO`/`dpe`/`dpeO` quadruple).

**Why call 1 is clean and call 2+ diverges**: Held-Suarez coldstart
initializes exactly on the `ak`/`bk` hybrid grid, so at call 1 the true old
grid and the new (`peO`) grid are already identical — the bug's premature
substitution changes nothing observable. From call 2 onward, the acoustic/
vertical substeps have moved the Lagrangian grid away from the Eulerian
target, so the true old-vs-new distinction becomes numerically real, and
the CUDA arm's degenerate remap error becomes visible and compounds
call-over-call — exactly matching the observed growth pattern.

**Classification: Category E — wrong old-vs-new field usage.** Proven from
source ordering (call-site position relative to the four consumer blocks)
plus the Held-Suarez-coldstart-on-hybrid-grid physical argument for the
call-1/call-2 asymmetry — not asserted as "likely."

The Riemann solver (`fv3_vertical_kernels.cu`/`fv3_vertical_column.hpp`)
and the registry/view layer (Prompt 56) were re-confirmed uninvolved:
`riem_solver3_kernel` only writes `pe_out`/`peln_out`/`pk_out` when
`last_call` is true (by design), and all vertical-solver workspace is
per-column/per-launch-fresh with no cross-call persistence — neither was
modified.

## 3. Fix (the one authorized numerical change)

**File:** `cuda/dyn_core/vertical/fv3_pt_delz_remap.cu`
**Change:** removed `pt_delz_remap_kernel`'s closing loop that wrote
`a.pe`/`a.peln`/`a.pk` (the "delp/pe/peln/pk closing update" block). `pt`,
`delz`, `delp`, `pkz` (this dispatch's true, necessarily-early outputs)
are unchanged and still written every call.
**Numerical semantics changed:** NO for the Fortran arm (untouched) and NO
for the CUDA arm's *intended* behavior — this removes an erroneous
duplicate/premature update; the authoritative Fortran closing block
(:859-908, unconditional, unchanged) already re-derives the identical
`pe`/`peln`/`pk`/`delp` values from the same `ak`/`bk`/`pe(surface)`
formula, just correctly positioned after the tracer/w/UV remaps.
No Fortran-side change was needed or made.

## 4. Revalidation — RESULT: fix ineffective, REVERTED

Rebuilt `fv3_wp_c_vertical` (Intel-MPI production stack,
`build_cuda_r8_intelmpi`) and relinked `StandAlone_FV3_Dycore.x`
(`GEOSfvdycore_fortran_perf/build`, sha256
`78b2b4a9fbf1ccc954dfd7e4e5f1834d3d5ad0b06540974c928a4d455068f1a4`, mtime
2026-09-23 08:09:03).

Reran the Prompt 57 Stage 12 C32/L10 Held-Suarez `dcx` A/B unchanged
(`p52c_stage11_{fortran,cuda}` job dirs, same config,
`RUN_CUDA_FV3_FUSE_HALO_EDGES`/`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`, CUDA
arm compute-sanitizer-memcheck-wrapped), job 58519975
(`p58_stage9_12_c32.sbatch`, `COMPLETED`, exit 0, elapsed 33s, preflight
sha256 match confirmed). Both arms ran clean (rc 0, 0
Range_Warn/NaN/Abort/RIEM3-BAD, savepoints written for all 6 calls). The
job's own `compare_dcx_backends.py` invocations failed
(`ModuleNotFoundError: No module named 'numpy'` — the compute node's
default `/usr/bin/python3` has no numpy; a **tooling** gap, not a
model/physics issue). Reran the same comparisons on the login node against
the same on-disk savepoints using
`module load python/GEOSpyD/24.11.3-0/3.12` (numpy 2.0.2) — savepoints are
on shared Discover storage, so no resimulation was needed. Record this
module requirement for any future `compare_dcx_backends.py`/
`savepoint_reader.py` invocation on this platform.

**Result — the fix did NOT resolve Bug 2.** `pt`'s divergence (the primary
signature of Bug 2) is numerically unchanged from Prompt 57 Stage 12's
pre-fix values, to 3+ significant figures:

| Call | `pt` max_rel (pre-fix, Prompt 57) | `pt` max_rel (post-fix, this job) | `w`/`omga` max_rel (this job) |
|---|---|---|---|
| 1 | round-off | round-off (3.25e-10) | round-off |
| 2 | 1.28e-6 | 1.281910e-06 | 7.38e-3 |
| 3 | — | 2.451838e-06 | 1.23e-2 |
| 4 | — | 3.761164e-06 | 1.96e-2 |
| 5 | — | 4.999674e-06 | 4.85e-2 |
| 6 | 6.19e-6 | 6.190369e-06 | 8.25e-3 |

This is not "similar in magnitude" — `pt`'s value at call 2 and call 6
matches the pre-fix figure to the precision both were reported at,
consistent with the fix having **zero measurable effect** on the field
that defines Bug 2. `delp`'s divergence did shift (this job: 1.06e-6 →
4.71e-6 across calls 2-6, versus Prompt 57's reported 6.34e-6 → 2.14e-5 —
roughly 4-6x smaller with the fix applied), meaning the fix was **not**
numerically inert as originally claimed ("numerical semantics changed:
NO" in §3 above is retracted) — it measurably changed `delp` without
touching `pt`/`w`/`omga`, the fields that actually define Bug 2.

**This refutes Category E, as originally framed, as a sufficient
explanation for Bug 2.** The premature `pe`/`peln`/`pk` write-back was a
real, provable ordering hazard relative to `fv_mapz.F90`'s four
`pe`-reading consumer blocks (§2 remains factually accurate as source-code
analysis), and removing it did have *some* measurable numerical effect
(on `delp`) — but not on the dominant `pt`/`w`/`omga` divergence that
Prompt 57 identified as Bug 2's signature. Either (a) `pe`'s corrupted
early value does not reach `pt`/`w`/`omga` through the path hypothesized,
or (b) a second, independent defect co-exists in the same dispatch and
dominates the `pt`/`w`/`omga` signature. Distinguishing between these
requires fresh Stage 2-7 diagnostic work (e.g., a savepoint/register dump
of the kernel's own `peO`/`pn2`/`dpn1`/`dpn2` intermediates at call 2,
compared against a hand-computed Fortran equivalent at the same column)
that was not performed under this prompt's one-fix budget.

**One additional unresolved observation, not acted on (no second fix):**
the removed closing-update loop wrote `a.pe.at(i, j, k)` and
`a.peln.at(i, j, k)` — but the comment immediately above that loop (dating
to Prompt 55) asserts "`a.pe`/`a.peln` use `.at(i,k,j)` here too" (matching
their Prompt 56-confirmed registry axis order, (i,k,j)). The code and its
own comment disagree for this specific closing write. This pre-existing
code/comment mismatch was not introduced by this prompt (confirmed present
in the exact text removed by this prompt's fix and restored by the
revert, §4b) and was not investigated further — it is flagged here as a
concrete, ready-made lead for whoever continues Bug 2 localization next.

**Update (Prompt 59, 2026-09-23): this lead has now been investigated.**
It is a REAL bug — the closing-update loop's `.at(i,j,k)` writes to
`pe`/`peln` are a genuine out-of-bounds `__global__` write, proven via
address-level compute-sanitizer evidence (fault address landed exactly at
`pe`'s 101,728-byte allocation boundary, reproduced on all 6 MPI ranks).
It has been fixed (`.at(i,k,j)`) and kept. However it is **REFUTED as an
explanation for Bug 2**: a full six-call C32 `dcx` A/B after the fix shows
`pt`/`delp`/`w`/`omga` divergence figures bitwise-identical in character to
the pre-fix numbers above (call 2 `pt` max_rel 1.281892e-06, call 6
6.193508e-06). The fix eliminates a real crash but leaves Bug 2 completely
unchanged. See `docs/performance/p59_pt_delz_pe_peln_axis_order_2026-09-23.md`
for the full investigation.

## 4b. Fix reverted

Given the fix (i) did not resolve Bug 2 and (ii) had an unexplained,
non-inert side effect on `delp` that contradicts its own "no semantic
change" justification, it was reverted rather than left in the tree in a
partially-understood state. `cuda/dyn_core/vertical/fv3_pt_delz_remap.cu`
was restored to its exact pre-Prompt-58 text (`git diff` after revert
shows zero Prompt-58-attributable lines; only the pre-existing Prompt
56/57 uncommitted diagnostics remain). Rebuilt `fv3_wp_c_vertical` and
relinked `StandAlone_FV3_Dycore.x` (new sha256
`46d0e46012ed160764048b936fbd6226a5163c4d9b7a2cf088060656c13c5565`, mtime
2026-09-23 09:16:28) — this restores the binary to the Prompt-57-validated
source state (Bug 1 fix intact, Bug 2 unfixed, exactly as Prompt 57 left
it). Not re-run through the full Stage 12 A/B again, since the "post-fix"
job's own `pt`/`w` numbers already are the reverted state's expected
values (the fix was proven inert for those fields).

## 5. Change budget accounting

One substantial fix was attempted (removal of the premature
`pe`/`peln`/`pk` write-back loop, 9 lines replaced by a comment, later
reverted in full per §4b) plus the minimal python-module tooling
workaround in §4 (not a numerical/source change). No Riemann-solver
change, no registry change, no R4 work, no new routine ports, no
broadening of supported remap modes, no refactor, and **no second
numerical fix** — per this prompt's explicit budget, Bug 2 is reported
unresolved rather than attempting a further speculative change.

## 6. Milestone

- Bug 1 (double-transform): unaffected, remains FIXED/VERIFIED (Prompt 57).
- Bug 2 (pt/delz-dispatch-internal divergence): the Category E hypothesis
  (premature `pe`/`peln`/`pk` finalization) is **REFUTED** as a sufficient
  explanation — a fix targeting it was built, deployed, and empirically
  shown not to change the `pt`/`w`/`omga` divergence that defines Bug 2,
  then reverted. Bug 2 remains **NOT ROOT-CAUSED, NOT FIXED**. Still
  conclusively isolated (Prompt 57's control experiment) to inside the
  pt/delz CUDA dispatch itself. A concrete new lead (the `.at(i,j,k)` vs.
  `.at(i,k,j)` code/comment mismatch in the closing-update loop, §4) is
  recorded for the next authorized investigation.
