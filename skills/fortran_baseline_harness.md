# Legacy-Language Baseline Harness

## Purpose

Establish an authoritative, reproducible reference output from the legacy
implementation for a bounded target subsystem, before writing any new-
language or new-backend code. Without this, there is nothing correct to
validate a port against.

## Use When

- Before starting translation/porting work on any bounded subsystem.
- Whenever "is this new implementation correct?" has no concrete answer
  because no legacy reference output exists yet for the relevant inputs.

## Inputs Needed

- The legacy subsystem's actual call signature and data layout (from the
  architecture map).
- A representative input case — ideally the smallest case that still
  exercises the real numerical path (not a degenerate trivial case that
  takes a shortcut the production case doesn't).
- A location to persist reference output (sentinel/savepoint files,
  checksums, or a comparison database).

## Procedure

1. Identify the exact subsystem boundary (inputs in, outputs out) you need
   a baseline for — as narrow as will still be meaningful.
2. Instrument the legacy code to dump its intermediate/output state at that
   boundary, in a stable, parseable format (not just stdout prose).
3. Run the instrumented legacy code at a small-but-real test case first;
   confirm the dump is deterministic (same inputs twice -> identical
   output) before trusting it as a baseline.
4. Scale the baseline to the actual target problem size(s) you'll validate
   against later — a baseline only at a toy size may miss size-dependent
   bugs (array indexing, tile boundaries, precision-dependent branches).
5. Store the baseline with its exact generating configuration (input
   parameters, compiler/version, random seed if any) so it can be
   regenerated or trusted later without guessing what produced it.
6. Reuse one project-wide savepoint/sentinel format and reader across all
   subsystems rather than inventing a new ad hoc dump format per
   subsystem — this is what makes later cross-subsystem comparison and
   tooling reuse possible.

## Validation / PASS Criteria

- Running the legacy code twice on identical inputs produces bit-identical
  baseline output (determinism confirmed).
- The baseline covers both a small development-scale case and the actual
  production-scale case(s) you ultimately need to validate against.
- The dump format is machine-parseable and documented well enough that a
  different engineer/agent can read it without guessing.

## STOP Conditions

- The legacy code is nondeterministic at the target boundary (e.g.,
  uninitialized memory, race conditions, platform-dependent floating-point
  behavior) — characterize and either fix or explicitly bound this
  nondeterminism (e.g., "differs only in known-benign padding bytes")
  before using it as a baseline; an unstable baseline poisons every
  downstream comparison.
- The legacy code cannot be instrumented without changing its numerical
  behavior (e.g., adding a print statement perturbs floating-point
  register allocation in a way that changes results) — find a
  non-perturbing instrumentation method before proceeding.

## Common Failure Modes

- Validating only against a toy/degenerate case and discovering size-
  dependent bugs much later at production scale.
- Trusting a single run's output without checking determinism first.
- Reinventing a new dump/comparison format per subsystem instead of reusing
  one project-wide mechanism — this multiplies tooling maintenance cost and
  makes cross-subsystem comparisons harder.
- Treating "the baseline run completed without crashing" as equivalent to
  "the baseline run produced correct/meaningful output" — a completed run
  with structurally-zero or default-valued fields (e.g. an uninitialized
  array that never gets seeded under the test configuration) validates
  nothing about that field; check that the baseline actually exercises the
  numerical path you intend to validate.

## Cost-Control Guidance

- Build the baseline mechanism once; reuse it for every subsequent
  subsystem rather than rebuilding per-subsystem.
- Prefer compact, structured output (binary dumps, CSV, checksums) over
  large text logs — this reduces both storage and the cost of any
  downstream comparison tooling or AI-agent context usage.
- Generate baselines at the smallest size that is still representative
  first; only generate production-scale baselines once the mechanism is
  confirmed correct at small scale.

## Reusable Prompt Template

```
Build an authoritative baseline/reference-output mechanism for
<subsystem> in <legacy codebase>, covering <boundary: inputs/outputs>.
Instrument the legacy code to dump state at that boundary in
<format: e.g. binary/CSV/checksum>. Confirm determinism by running twice
on identical inputs. Generate baselines at <small case> and
<production-scale case>. Document the exact generating configuration
alongside each baseline. Reuse <existing project mechanism, if any>
rather than building a new one. Do not modify legacy numerical behavior —
only add non-perturbing instrumentation.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** FV3's baseline
mechanism uses savepoints at dyn_core entry/exit and multiple intermediate
boundaries (after C-grid predictor, after D-grid dynamics, after
vertical/`update_dz_d`) generated from the authoritative GFDL Fortran
implementation, compared bitwise/round-off against the CUDA candidate at
the same boundaries — this is what let the project localize a divergence to
"the first mismatching stage" rather than only detecting it in aggregate
output (see `docs/agent_work_summary.md` §4/§11). A structural limitation
was also documented rather than hidden: under the default `COLDSTART=1` +
empty `ExtData.rc` configuration, tracers (`Atm%q`) are never seeded and
stay exactly zero for the whole run, so no production-shaped baseline run
can validate tracer-transport numerics at all — a dedicated non-production
seeding mechanism would be required first.
