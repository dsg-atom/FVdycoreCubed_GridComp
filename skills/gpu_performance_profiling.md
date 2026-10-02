# GPU Performance Profiling

## Purpose

Measure the real, production-relevant performance impact of a port using
the model's own timers and real workload, and choose the next porting
target based on measured evidence rather than an a-priori list.

## Use When

- After a port has passed numerical validation (never before — see
  `numerical_validation.md`).
- When selecting the next subsystem to port.
- When investigating a performance regression or an unexpected result.

## Inputs Needed

- The model's own built-in timing instrumentation (preferred over an
  external microbenchmark) — if none exists, build one minimal, reusable
  timer around the subsystem boundary rather than wrapping every
  individual call site ad hoc.
- A representative, production-scale workload — not a toy case.
- Scheduler/allocation access sufficient to run paired comparisons in a
  single allocation (see `production_ab_testing.md`).
- A GPU/accelerator profiler appropriate to the platform (e.g. a timeline/
  system profiler as the default entry point; a hardware-counter profiler
  as a secondary tool if available and permitted).

## Procedure

1. Establish the current production baseline timing (legacy-only, or
   current-best-known configuration) using the model's own timers, at
   production scale, before changing anything.
2. After integrating a validated port, measure both the **local**
   subsystem timer and the **end-to-end/TOTAL** timer — report both,
   never substitute one for the other; a large local speedup does not
   guarantee a comparable end-to-end change if the subsystem's share of
   total runtime is small.
3. Use the platform's timeline/system profiler as the first-stage
   investigative tool for "where is time actually going" questions
   (kernel launches, API calls, gaps/idle time) before reaching for a
   hardware-counter profiler.
4. If a hardware-counter profiler is needed and is platform-restricted
   (e.g. a permissions limitation), run a minimal smoke test of it first
   in the current allocation rather than assuming a historical finding
   about its availability is still valid — platform permissions can
   change between sessions.
5. Any comparison across more than one job/allocation must follow
   `production_ab_testing.md`'s paired-same-allocation methodology — do
   not trust a result from comparing two different jobs run at different
   times/nodes.
6. After each port, re-profile the full workload to find the next
   bottleneck — do not work from a fixed, pre-written list of "things to
   port"; let the current profile drive the next target.
7. When a result looks surprising (e.g. "the new path is slower"), do not
   accept it at face value — first rule out measurement artifacts
   (cross-node/cross-allocation variance, stale binary, wrong module
   stack) before concluding the port itself regressed. See
   `production_ab_testing.md` and `agent_recovery_and_resume.md`'s
   preflight checklist.

## Validation / PASS Criteria

- Both local and end-to-end timer values are reported for every
  performance claim, with the measurement methodology (same-allocation
  paired A/B, number of repeats, warmup handling) stated.
- A surprising result has been checked against the known measurement-
  artifact failure modes before being reported as a real finding.
- The next porting target is justified by a current profile, not an
  assumption.

## STOP Conditions

- A performance claim is based on a single run, not a paired, repeated,
  same-allocation comparison — do not report it as a conclusion; rerun
  properly first.
- The measured local speedup is real but the end-to-end change is
  negligible or negative — this is a valid, reportable outcome (not every
  real local win moves the needle), but it must be reported as such, not
  inflated.
- Hardware-counter profiling is blocked by a platform permission issue —
  do not attempt a privilege workaround; fall back to a correctness-only
  diagnostic tool and/or static compile-time resource-usage reporting,
  and document the limitation.

## Common Failure Modes

- Comparing a run on one node/allocation against a run on a different
  node/allocation and treating the difference as a real regression —
  node-to-node hardware variance on shared HPC clusters can exceed 2x and
  will dominate a small/moderate real effect.
- Reporting a local kernel's speedup number as if it were the end-to-end
  improvement.
- Re-profiling with a stale/incorrectly-rebuilt binary and attributing the
  (non-)result to the wrong cause — always confirm which binary actually
  ran (see `agent_recovery_and_resume.md`).
- Assuming a hardware-counter profiler is unavailable (or available) based
  on a historical note, without a fresh smoke test in the current session/
  allocation.
- Working through a fixed porting roadmap decided before any profiling,
  instead of letting the current hotspot profile pick the next target.

## Cost-Control Guidance

- Prefer the model's own built-in timers (if present) over building new
  external measurement infrastructure.
- Use the lighter-weight timeline/system profiler as the default,
  reserving a heavier hardware-counter profiler for cases the timeline
  profiler can't resolve.
- Batch multiple configurations/repeats into one scheduled job/allocation
  where the scheduler allows it, rather than submitting many small jobs.

## Reusable Prompt Template

```
Profile <subsystem/workload> on <platform> using the model's own timers
and real production-scale input. Report both the local <subsystem> timer
and the end-to-end/<TOTAL> timer. Run as a same-allocation, alternating-
order, multi-repeat paired comparison against <baseline configuration>
per production_ab_testing.md. If a hardware-counter profiler is needed,
smoke-test its availability in this allocation first rather than trusting
a historical note. Identify the next-largest unaddressed hotspot from the
resulting profile as the recommended next target, with its measured
contribution.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** FV3 uses
`fv_timing_mod`'s `DYN_CORE`/`TOTAL` timers as ground truth, with `nsys`
(Nsight Systems) as the unrestricted first-stage timeline profiler and
`ncu` (Nsight Compute) blocked on this platform by a driver permission
issue (`ERR_NVGPUCTRPERM`) requiring a fresh smoke test before each new
reliance on it, per `docs/agent_instructions/discover_platform.md`.
REMAP_UV's ~10.23x local, bitwise-exact speedup produced only +~1% TOTAL
change — reported as both numbers, not inflated into "10x faster." The
porting sequence (PT/DELZ -> REMAP_UV -> TRACER_2D(stopped) -> REMAP_Q ->
REMAP_PRESSUREVARS -> REMAP_TOTALENERGYCON(blocked) -> `FV_To_State`
(kept CPU) -> `State_To_FV`) was chosen by re-profiling after each step,
not from a pre-written list (`docs/agent_work_summary.md` §5).
