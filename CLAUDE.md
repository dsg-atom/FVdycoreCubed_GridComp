# GEOS FV3 CUDA Modernization

Read and follow these persistent project instructions:

@docs/agent_instructions/discover_platform.md
@docs/agent_instructions/project_architecture.md
@docs/agent_instructions/cost_efficiency.md

Also read:

@docs/agent_handoff/CURRENT_STATE.md

Use `CURRENT_STATE.md` as the compact source of current project status.
Do not reconstruct project history from old handoff files unless the current task
requires it.

## Before Every Task

Inspect:

```bash
pwd
git status
git branch --show-current
git rev-parse HEAD
git log -5 --oneline --decorate
```

Git and the current source tree are authoritative.

Do not discard, reset, clean, revert, push, or merge existing work unless
explicitly instructed.

## Core Rules

- Authoritative GFDL FV3 Fortran is the numerical source of truth.
- Preserve the original Fortran backend.
- Keep CUDA optional.
- Preserve the WP-E persistent GPU runtime/state architecture.
- Preserve real six-tile cubed-sphere communication semantics.
- Do not replace integrated halo communication with stubs.
- Do not duplicate numerical functionality across work packages.
- Avoid unnecessary full-state host/device transfers.
- Do not modify scientific algorithms merely to make CUDA compile or tests pass.
- Do not modify `geos-gtfv3`.
- Do not push or merge unless explicitly requested.

Always distinguish:

- IMPLEMENTED
- COMPILED
- GPU EXECUTED
- NUMERICALLY VALIDATED

Do not claim a later validation level based on an earlier one.

## Discover Execution Model

Claude Code remains on a Discover login node.

GPU-required execution must be submitted through Slurm according to:

@docs/agent_instructions/discover_platform.md

Lack of direct GPU access from the login node is expected and is not a blocker.

## Cost-Efficient Agent Operation

Token/API cost is a project constraint.

Use the minimum context and exploration required to complete the current task.

### Context

At the start of a task, normally read only:

1. `CLAUDE.md`
2. `docs/agent_handoff/CURRENT_STATE.md`
3. files explicitly relevant to the current task

Do not reread historical milestone handoffs unless:

- `CURRENT_STATE.md` points to one;
- the current defect originated there;
- exact historical evidence is required.

### Repository Inspection

Prefer targeted inspection:

- exact files
- specific symbols
- `git grep`
- narrow `find`/`grep`

Do not perform broad repository scans by default.

Expand search only when targeted inspection is insufficient.

### Debugging

Use bounded debugging.

For each investigation:

- define the hypothesis or discriminator;
- run the minimum experiments needed;
- define a stop condition;
- stop once evidence is sufficient.

Do not continue gathering redundant evidence after a hypothesis is resolved.

For numerical divergence use coarse-to-fine localization:

1. compare major stage boundaries;
2. identify the first mismatching stage;
3. compare fields inside that stage;
4. inspect kernels/operators only after localization;
5. perform element-level comparison last.

Do not dump or analyze every field at every stage up front.

### Execution Output

Keep routine progress terse.

Report only:

- material findings;
- blockers;
- test failures that change the plan;
- corrections to previous assumptions;
- final results.

Do not repeatedly summarize:

- project history;
- files merely read;
- successful routine builds;
- already-resolved hypotheses.

Prefer machine-readable diagnostics such as CSV, JSON, hashes, and compact
summaries over large logs.

### Scope

Finish the current acceptance gate before starting adjacent work.

Record unrelated findings as deferred tasks unless they directly block the
current task.

Do not optimize numerical kernels before correctness localization unless the
current task explicitly requires performance work.

### Reuse

Before creating a new:

- harness
- test framework
- savepoint format
- reader
- build script
- runtime service

check for an existing project mechanism that can be reused or extended.

## Current Project Sequence

```text
CUDA dyn_core implementation/integration
-> integrated six-tile GPU execution
-> multi-step stability
-> small-scale Fortran/CUDA performance baseline
-> authoritative Fortran savepoints
-> CUDA vs Fortran numerical comparison
-> Gate C2 numerical agreement
-> production GEOS integration
-> profiling / optimization / scaling
```

Follow the current task prompt for phase-specific objectives and completion
criteria.

Do not repeat completed project history in task reports unless it is directly
relevant to the current decision.
