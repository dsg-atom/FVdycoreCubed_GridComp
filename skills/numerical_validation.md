# Numerical Validation

## Purpose

Prove that a new implementation's numerical output matches the legacy
authoritative implementation, at a precisely stated and justified
tolerance, and avoid the single most expensive failure mode in this kind
of project: conflating "it ran" or "it compiled" with "it is correct."

## Use When

- At every stage of a port: isolated kernel vs. host reference, host
  reference vs. legacy baseline, integrated system vs. legacy baseline,
  at every scale you intend to claim validated.
- Whenever a handoff or status report is about to state a validation
  level — use this skill's terminology before writing that sentence.

## Inputs Needed

- An authoritative legacy baseline (see `fortran_baseline_harness.md`).
- The candidate implementation's output at the same boundary/boundaries.
- A pre-declared tolerance policy: what counts as bitwise, what counts as
  acceptable round-off, what would count as a real divergence.

## Procedure

1. Always use exactly four validation levels, and never use one to imply
   another:
   - **IMPLEMENTED** — source exists, is wired into the real call graph.
   - **COMPILED** — the compiler/toolchain accepts and links it.
   - **EXECUTED** — it actually ran on the real target hardware/backend.
   - **NUMERICALLY VALIDATED** — compared against the authoritative
     baseline at a stated, justified tolerance.
2. Before comparing, confirm the real production/integration entry point
   actually invokes the candidate code path end-to-end — do not infer this
   from the presence of a kernel/function; trace the actual call path
   taken at runtime (e.g. via a log line, a debugger, or an instrumented
   counter you've confirmed is real).
3. Localize divergence coarse-to-fine: compare at major stage boundaries
   first (e.g. subsystem entry/exit), find the first mismatching stage,
   then narrow inside that stage — do not dump every field at every stage
   up front.
4. Classify any observed difference: bitwise-identical; round-off
   (consistent with floating-point reassociation/operation-order
   differences, typically ~1e-10 to 1e-15 relative at double precision);
   ULP-level transcendental difference; or genuine algorithmic divergence.
   State which category applies and why, not just a number.
5. Before accepting or rejecting based on a relative-error threshold, check
   the absolute magnitude of both values — relative error is meaningless
   (and misleadingly large) near a small-magnitude or zero reference value.
6. For global-reduction/conservation diagnostics (sums across the whole
   domain), be aware that summation order affects bit-for-bit
   reproducibility independent of any kernel bug — treat these as a
   distinct, harder validation category, not an ordinary kernel-level
   comparison.
7. Maintain a short, explicit list of known-benign artifacts (pre-existing
   in both the legacy and candidate implementations, not new divergences)
   so they are not re-investigated from scratch by a future session.
8. State the validation level per precision/configuration separately (e.g.
   R8 vs R4, or any other precision/mode split) — do not let one
   precision's PASS imply another's.

## Validation / PASS Criteria

- Every validation claim names its exact level (one of the four) and the
  exact tolerance/classification used.
- The comparison was performed at the real production entry point, with
  confirmed end-to-end reachability of the candidate code path — not just
  an isolated unit test, if the claim is about production readiness.
- Any accepted "large-looking" relative error has a documented absolute-
  magnitude explanation.

## STOP Conditions

- The real production/integration entry point cannot be confirmed to
  invoke the candidate code path — stop and fix the wiring/dispatch before
  claiming any validation level beyond COMPILED.
- A comparison shows a genuine algorithmic divergence (not explainable as
  round-off/ULP/known-benign) — stop and localize via step 3 before
  attempting any fix; do not patch based on a guess.
- Only a toy/degenerate case has been validated (e.g. a structurally-zero
  field, see `fortran_baseline_harness.md`) — do not claim validation for
  a numerical path that the test configuration never actually exercises.

## Common Failure Modes

- Declaring "NUMERICALLY VALIDATED" or "COMPLETE" based on code presence
  and a passing compile, without ever running an actual comparison against
  baseline data — this project's own documented
  `implementation_reality_audit.md` is the canonical example: a "COMPLETE"
  claim stood on a one-line stub production entry point that never called
  any of the ~76 real kernels claimed as validated.
  See `docs/agent_work_summary.md` §4 for the full account.
- Reporting a large relative error as a failure without checking whether
  the reference value is near zero (making the relative-error formula
  blow up on an otherwise-tiny absolute difference).
- Comparing only final aggregate output, missing an intermediate divergence
  that cancels out or is masked by later stages.
- Re-investigating the same known-benign artifact repeatedly because it
  was never written down as "known benign, do not re-investigate."
- Letting one precision's (e.g. R8) validated status imply another's (R4)
  is also validated, without actually running the R4 comparison.

## Cost-Control Guidance

- Localize coarse-to-fine (step 3) specifically to avoid the cost of
  dumping and analyzing every field at every stage before you know where
  to look.
- Reuse one comparison tool/format across all subsystems rather than
  writing bespoke comparison scripts per subsystem.
- Stop investigating once a divergence's category (round-off / ULP /
  known-benign / genuine) is determined with enough confidence to decide
  pass/fail — do not keep gathering confirmatory evidence past that point.

## Reusable Prompt Template

```
Numerically validate <candidate implementation> against the authoritative
<legacy> baseline for <subsystem>, at <production entry point>. First
confirm the production entry point actually reaches the candidate code
path end-to-end (not just that the code exists). Compare at
<stage boundaries>, coarse-to-fine, to localize the first mismatching
stage if any. Classify any difference as bitwise / round-off / ULP /
known-benign / genuine divergence, with a stated absolute-magnitude check
for any large-looking relative error. Report using exactly these four
levels: IMPLEMENTED / COMPILED / EXECUTED / NUMERICALLY VALIDATED — do not
use one to imply another. Validate <precision/configuration variants>
separately, do not let one imply another.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** Gate C2 validated
all 54 production first-run dyn_core stages at savepoint boundaries
(dyn_core entry, after C-grid predictor, after C pressure, after D-grid
dynamics/transport, after vertical/`update_dz_d`, after D pressure/NH,
end acoustic substep, dyn_core exit), R8 and R4 reported separately: R8
worst-case `w` max_rel ~5e-11 (accepted as round-off); R4 worst-case `w`
max_rel ~2.4e-1 (initially alarming, but traced to a small-denominator
relative-error artifact with absolute agreement ≤5.9e-5, and accepted on
that basis). Three specific benign artifacts (a subnormal halo-padding
diff in `delpc`/`pkc`, a benign FMS `range_warn` at C128 10-day, near-
surface NaN from standard MAPL masking) are explicitly tracked as
known-benign in `CURRENT_STATE.md` §4 rather than re-investigated each
session.
