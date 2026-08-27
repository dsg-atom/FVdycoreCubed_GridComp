# Cost-Efficient Agent Workflow

API/token cost is a first-class project constraint.

The objective is to minimize unnecessary context, repository exploration, repeated
reasoning, and verbose output while preserving scientific and numerical correctness.

## 1. Start With Minimum Context

At the start of a task, normally read only:

1. `CLAUDE.md`
2. `docs/agent_handoff/CURRENT_STATE.md`
3. files explicitly named by the current task
4. the smallest set of source files directly relevant to the current symbol,
   stage, or defect

Do not automatically reread historical milestone handoffs.

Open an older handoff only when:

- `CURRENT_STATE.md` explicitly points to it;
- the current defect originated there;
- exact historical evidence is required;
- a current result conflicts with a prior accepted result.

If `CURRENT_STATE.md` conflicts with Git or the current source tree, Git/source
is authoritative. Report the discrepancy briefly.

## 2. Do Not Reconstruct Project History Unless Necessary

Do not spend tokens rebuilding project history from:

- old handoff documents;
- long Git logs;
- previous milestone reports;
- broad repository scans.

Use `CURRENT_STATE.md` as the compact source of current truth.

Historical details should be retrieved only when they affect the current
decision.

## 3. Prefer Targeted Repository Inspection

Start with the narrowest useful query.

Prefer:

- exact filenames;
- exact routines or symbols;
- `git grep`;
- narrow `grep`;
- narrow `find`;
- known directories;
- specific call sites.

Avoid broad recursive exploration by default.

Expand search scope only when targeted inspection fails to answer the current
question.

## 4. Reuse Existing Infrastructure First

Before creating a new:

- harness;
- test framework;
- savepoint format;
- parser/reader;
- build script;
- Slurm wrapper;
- runtime service;
- communication layer;

search specifically for an existing project mechanism that can be reused or
extended.

Prefer extending one established mechanism over creating parallel
infrastructure.

## 5. Use Bounded Debugging

Every debugging task should have:

- a specific hypothesis or discriminator;
- the minimum experiment set needed to test it;
- an explicit stop condition.

Do not continue open-ended tracing after the stop condition is reached.

If a bounded investigation does not isolate the defect, report the blocker and
stop rather than expanding indefinitely.

Do not gather redundant evidence once a hypothesis is decisively confirmed or
rejected.

## 6. Use Coarse-to-Fine Numerical Localization

For numerical divergence or instability:

1. compare major stage boundaries;
2. identify the first mismatching stage;
3. compare only the fields produced or consumed there;
4. inspect individual routines/operators only after stage localization;
5. inspect element-level values only as the final step.

Do not dump or analyze all fields at all stages up front.

For savepoint validation, begin with a minimal ladder such as:

- dyn_core entry;
- after C-grid vertical work;
- after D-grid dynamics / d_sw;
- dyn_core exit.

Add intermediate savepoints only inside the first mismatching interval.

## 7. Prefer Compact Machine-Readable Diagnostics

Prefer:

- CSV;
- JSON;
- YAML;
- hashes/checksums;
- min/max/norm summaries;
- concise per-stage tables.

Avoid large free-form logs when compact structured output can answer the same
question.

Do not repeatedly feed verbose scheduler, compiler, or numerical logs back into
the model when a short extracted summary is sufficient.

## 8. Keep Execution Updates Terse

During work, report only:

- material findings;
- blockers;
- failures that change the plan;
- corrections to previous assumptions;
- final results.

Do not narrate routine:

- file reads;
- grep commands;
- successful builds;
- expected test execution;
- previously known project history;
- already-resolved hypotheses.

## 9. Separate Difficult Reasoning From Mechanical Execution

When model choice is available, use the cheaper/default model for:

- CMake/build work;
- Slurm scripts;
- test harnesses;
- routine implementation;
- compile fixes;
- regression runs;
- documentation;
- mechanical refactoring.

Use the strongest reasoning model only for:

- authoritative Fortran/CUDA numerical reconciliation;
- subtle halo/topology semantics;
- concurrency/order reasoning;
- unexplained numerical divergence;
- review of risky numerical changes.

A preferred workflow is:

1. use the stronger model for a short analysis/plan;
2. switch back to the cheaper model for implementation, build, and tests.

Do not repeatedly re-derive an already-approved plan during execution.

## 10. Finish the Current Gate Before Adjacent Work

Do not start adjacent tasks merely because they become visible.

Complete the current acceptance gate first.

Record unrelated findings as deferred tasks unless they directly block the
current task.

Avoid opportunistic cleanup or optimization during correctness work.

## 11. Do Not Optimize Before Localization

Default order:

correct execution
-> numerical validation
-> profiling
-> optimization
-> scaling

Do not spend tokens designing performance optimizations before the relevant
correctness gate passes unless the current task explicitly requests performance
work.

When performance work is requested, measure first and optimize only from
evidence.

## 12. Stop When Evidence Is Sufficient

When acceptance criteria pass, stop testing unless a required regression
remains.

When a hypothesis is ruled out, close it.

When a defect is localized to a bounded region, do not continue broad tracing.

More evidence is not automatically better if it does not change the decision.

## 13. Minimize Repeated Handoff Content

At milestone completion:

- write detailed findings only in that milestone's handoff;
- update `docs/agent_handoff/CURRENT_STATE.md` with current truth only;
- do not copy the entire milestone report into `CURRENT_STATE.md`;
- remove or avoid superseded hypotheses in `CURRENT_STATE.md`.

Target `CURRENT_STATE.md` length:

approximately 100-200 lines maximum.

It should contain:

- current branch / HEAD;
- authoritative source SHA;
- current implementation status;
- current correctness status;
- current benchmark numbers;
- current open issues;
- current task;
- immediate next step;
- durable hard constraints.

## 14. Avoid Repeated Long Prompts

When persistent instructions already exist in:

- `CLAUDE.md`;
- `docs/agent_instructions/*.md`;
- `docs/agent_handoff/CURRENT_STATE.md`;

task prompts should normally contain only:

- the specific goal;
- the acceptance criteria;
- important task-specific constraints;
- files/routines to inspect if known.

Do not restate the entire project architecture in every task prompt.

## 15. Use Explicit Stop Conditions for Expensive Work

For tasks involving:

- multi-build matrices;
- sanitizer runs;
- repeated Slurm jobs;
- broad numerical comparisons;
- large savepoint generation;

define the minimum required acceptance set first.

Do not run larger matrices until the smallest reproducer or acceptance case
passes.

Do not run expensive secondary cases when the primary gate already fails.

## 16. Preserve Scientific Correctness Over Token Savings

Cost efficiency must never justify:

- skipping authoritative source verification;
- guessing array layout;
- guessing halo semantics;
- hiding failed tests;
- suppressing numerical discrepancies;
- weakening validation criteria;
- replacing required communication with stubs;
- claiming validation that has not been performed.

Spend extra tokens when necessary to protect numerical correctness.

## 17. Preferred Default Task Pattern

For most tasks:

1. read `CLAUDE.md`;
2. read `CURRENT_STATE.md`;
3. inspect Git state;
4. inspect only the directly relevant files;
5. state one bounded plan;
6. execute;
7. stop on sufficient evidence;
8. update the milestone handoff;
9. update `CURRENT_STATE.md` only if current truth changed.

## 18. Scope Expansion Rule

Expand scope only when one of these is true:

- the current task is blocked by a dependency outside the original scope;
- evidence shows the original hypothesis is wrong;
- authoritative source tracing requires following a call chain;
- acceptance cannot be determined without one additional bounded check.

When expanding scope, state briefly:

- why;
- what new bounded question is being answered;
- what the new stop condition is.
