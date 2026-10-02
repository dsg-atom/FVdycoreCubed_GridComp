# Modernization Stop/Go Decisions

## Purpose

Apply a consistent, evidence-based rubric for deciding whether to PORT
(make new-backend the default), KEEP (leave on the legacy/host path),
DEFER (real blocker, revisit later), or STOP (reject, don't reopen without
new evidence) for any individual porting candidate.

## Use When

- At the end of any porting effort for a bounded subsystem (after
  `cuda_kernel_porting.md`'s procedure completes).
- Whenever tempted to "just keep pushing" on an optimization that isn't
  clearing its bar.
- When revisiting a previously-stopped or deferred item, to check whether
  new evidence actually justifies reopening it.

## Inputs Needed

- A pre-declared acceptance bar (decided before implementation): numerical
  tolerance, and if performance is in scope, a minimum required
  improvement.
- Completed isolated and post-integration numerical validation results.
- Completed production-scale, paired A/B performance measurement (local
  and end-to-end).
- Completed memory-headroom check at target production density, if
  applicable.

## Procedure

1. **PORT (make new-backend default)** when: numerically validated both
   in isolation and post-integration, at or beyond the declared tolerance;
   GPU/new-backend-executed at production scale; non-negative (or
   explicitly accepted) end-to-end performance contribution; memory-safe
   at target production density.
2. **PORT conditionally** when the same bar as PORT is met only above a
   specific, measurable threshold (device count, rank density, grid
   size, etc.) — gate the default on that exact threshold rather than
   forcing it everywhere or abandoning it everywhere.
3. **KEEP (legacy/host path, do not port)** when both sides' data already
   reside in the same place at the call boundary (porting would add a
   transfer, not remove one), or when a port's correctness/performance
   case cannot be made at all.
4. **DEFER** when a specific, named blocker exists that a kernel rewrite
   cannot address alone (e.g. the bottleneck is in communication/reduction
   cost, not compute; a required numerical capability is simply not yet
   implemented) — record the exact blocker, not just "not done yet,"
   so a future revisit knows what actually needs to change.
5. **STOP** when a measured result (numerical or performance) fails the
   pre-declared bar, confirmed by at least one independent reproduction —
   record both the measured failure and, if applicable, whether the code
   path is even reachable in the real production configuration (a result
   can fail on performance AND be moot on relevance simultaneously; record
   both facts, either one independently justifies stopping).
6. Do not retroactively lower the acceptance bar after seeing a
   disappointing number — a bar changed after the fact is not evidence-
   based.
7. Document every decision (not just PORT) with the same rigor: what was
   measured, how, and why that measurement led to this specific decision
   — a STOP or DEFER deserves exactly as much documentation as a
   successful PORT, so it is never redone from scratch by a future
   session.
8. When revisiting a STOP/DEFER, require genuinely new evidence (a changed
   production configuration, a new numerical capability, new hardware)
   before reopening — "it's been a while" or "let's just try again" is not
   sufficient justification on its own.

## Validation / PASS Criteria

- Every porting candidate has exactly one of the four decisions recorded,
  with the measured evidence that produced it.
- No decision was made before its required evidence (numerical validation,
  performance measurement, memory check as applicable) was actually
  gathered.
- Any conditional PORT states its exact gating threshold.

## STOP Conditions

(This skill's own "STOP condition" is about applying it, not a modernization
decision itself.)

- Evidence is incomplete (e.g. no post-integration validation, no
  production-scale paired measurement) — do not render a PORT/KEEP/DEFER/
  STOP decision yet; gather the missing evidence first.
- The acceptance bar was not declared before implementation — stop and
  declare it now, honestly, independent of already having seen the
  result, rather than reverse-engineering a bar that happens to match
  the number you got.

## Common Failure Modes

- Declaring PORT based on isolated validation alone, without re-validating
  post-integration or checking memory at production density.
- Lowering the acceptance bar after seeing a disappointing performance
  number, so the result "passes" after the fact.
- Recording a STOP decision with no evidence trail, so a future session
  re-attempts the same rejected approach from scratch.
- Treating "the production configuration doesn't even reach this code
  path" as a side note rather than an independently sufficient reason to
  stop/deprioritize, alongside any performance finding.
- Reopening a STOP/DEFER based on time elapsed alone, without new
  evidence.

## Cost-Control Guidance

- Decide the acceptance bar once, before implementation, so no analysis
  time is spent later rationalizing a result after the fact.
- Document STOP/DEFER decisions with the same (not necessarily more)
  rigor as PORT decisions — this is what prevents re-litigating a settled
  question later, which is the expensive failure mode.
- When checking whether to reopen a STOP/DEFER, check first whether the
  named blocker or acceptance-bar context has actually changed before
  spending any implementation effort.

## Reusable Prompt Template

```
Given the following evidence for porting <subsystem> to <new backend>:
isolated numerical validation = <result>, post-integration validation =
<result>, production-scale paired performance (local, end-to-end) =
<result>, memory headroom at <target density> = <result>, against the
pre-declared acceptance bar of <tolerance / minimum speedup>: render one
of PORT / PORT-conditional(<threshold>) / KEEP / DEFER(<named blocker>) /
STOP(<failed bar + relevance to real production config>). Document the
decision with the evidence that produced it, to the same standard
regardless of which decision it is.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** See the full
decision table in `docs/agent_work_summary.md` §6. Representative cases:
REMAP_UV — PORT (bitwise, GPU-executed at scale, +~1% TOTAL, no memory
issue). `State_To_FV` — PORT conditionally (`device_count>=2`) after
failing the memory check at 1-GPU density. `FV_To_State` — KEEP (both
ends already host-resident). REMAP_TOTALENERGYCON — DEFER
(`mpp_global_sum` communication is ~89% of the timer, a communication
problem, not a kernel problem). TRACER_2D hord==8 — STOP, with both facts
recorded: it measured a +37.87% regression against a required ≥5x bar,
*and* production's `hord_mt/vt/tm=6` configuration never reaches that code
path at all, so even a fixed-performance version would be moot in
production.
