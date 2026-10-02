# Production A/B Performance Testing

## Purpose

Produce performance comparisons (legacy vs. candidate, configuration A vs.
B) that are actually trustworthy on shared, multi-tenant HPC hardware,
where node-to-node variance can be larger than the real effect you're
trying to measure.

## Use When

- Any time you are about to report "X is faster/slower than Y" based on
  wall-clock or timer measurements.
- Investigating an unexpected performance result before accepting it as
  real.

## Inputs Needed

- Both configurations (A and B) ready to run with no code changes between
  runs other than the thing being compared.
- Scheduler access sufficient to request a single allocation that can run
  both configurations.
- The model's own production timers (see `gpu_performance_profiling.md`).

## Procedure

1. Request a single scheduler allocation (same node(s)) that will run both
   configuration A and configuration B — never compare timings from two
   separately-submitted jobs that may land on different hardware.
2. Alternate run order within that allocation (e.g. A, B, A, B, A, B — not
   AAA then BBB) to average out any monotonic drift (thermal, system
   load) across the allocation's lifetime.
3. Run multiple repeats of each configuration (this project used 3 reps
   per configuration as a working minimum) and compare medians, not single
   runs or means skewed by one outlier.
4. Include a warmup repeat if the system/workload has any warmup-dependent
   behavior (cache warming, JIT, first-touch page faults) and state
   explicitly whether warmup runs are included in or excluded from the
   reported statistic.
5. Confirm which exact binary/configuration ran in each case — grep the
   job's own log for a definitive binary-identity marker (not just "the
   job didn't error") before trusting the result belongs to the
   configuration you think it does.
6. Report the ratio of medians (or another robust statistic), not just
   raw numbers — and report it alongside the node/allocation identity so
   a future reader can tell whether two results are comparable.
7. If an earlier result (from a non-paired comparison) conflicts with a
   new paired result, treat the new paired result as authoritative and
   explicitly supersede the old one in documentation — do not leave both
   standing as if equally valid.

## Validation / PASS Criteria

- The comparison ran both configurations in one allocation, alternating
  order, multiple repeats.
- The exact binary/configuration identity for each run is confirmed from
  the job's own log, not assumed.
- The reported statistic (e.g. ratio of medians) and its measurement
  methodology are both stated together, not just the headline number.

## STOP Conditions

- Only one run of each configuration exists — do not report a conclusion;
  get at least 3 repeats, paired, same-allocation, before concluding
  anything.
- The two configurations were run in separate allocations/jobs/times — do
  not treat the comparison as conclusive regardless of how large the
  apparent difference is; rerun paired before reporting.
- The job's log does not clearly identify which binary/configuration
  actually ran — resolve this before trusting any number from that job.

## Common Failure Modes

- Comparing a legacy run from one job against a candidate run from a
  separate job (different time, possibly different physical node) and
  reporting the difference as a real performance regression or
  improvement — on shared multi-tenant hardware, node-to-node variance
  alone can produce a misleading difference of this size or larger.
- Running configuration A three times in a row, then B three times in a
  row, in the same allocation — order-dependent drift (e.g. thermal
  throttling building up over the allocation) can still bias this; prefer
  strict alternation.
- Trusting a stale `install/`-directory or previously-built binary copy
  instead of the freshly rebuilt one, producing a comparison that
  silently compares old-vs-old or old-vs-new rather than the intended
  comparison.
- Treating a historical "A is slower than B" finding as still valid after
  a later, properly paired rerun contradicts it, instead of explicitly
  marking the old finding as superseded.

## Cost-Control Guidance

- Combine both configurations' repeats into a single scheduled allocation
  — this is both more trustworthy and typically cheaper than separate
  submissions.
- 3 repeats per configuration, alternated, was sufficient evidence in this
  project's experience for most comparisons — escalate to more repeats
  only if the medians themselves are inconsistent across the 3.
- Once a paired comparison gives a clean, consistent result, do not keep
  re-running it "just to be sure" — treat it as settled unless new code
  changes invalidate it.

## Reusable Prompt Template

```
Compare <configuration A> vs <configuration B> for <workload> on
<platform>. Request a single allocation covering both. Run <N>
alternating repeats (A,B,A,B,...), using <production timer>. Confirm via
the job's own log which exact binary/configuration ran in each repeat.
Report the ratio of medians, explicitly stating the allocation/node
identity and repeat count alongside the number. If this contradicts an
earlier, non-paired comparison, state that the new result supersedes the
old one and why.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** The original C512
"CUDA 1.270x slower" finding came from comparing Fortran and CUDA runs on
different physical A100 nodes. A dedicated root-cause investigation
(Prompts 38-47) led to a same-node, same-allocation, alternating-order
paired rerun (F1 C1 F2 C2 F3 C3, job 58468244) giving a ratio of medians
of 0.99998 — the backends are actually comparable; the original finding
was a measurement artifact, now explicitly marked SUPERSEDED in
`CURRENT_STATE.md` §5. A separately documented lesson: "a job's own log
line is the only reliable evidence of which binary ran" — Prompt 53 found
jobs that had silently fallen back to a stale `install/bin` binary
(`docs/agent_work_summary.md` §5/§8/§12).
