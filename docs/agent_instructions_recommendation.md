# Recommended Top-Level Agent Instructions for a New Legacy-HPC Modernization Project

This document does **not** modify this project's existing instruction
files (`CLAUDE.md`, `docs/agent_instructions/discover_platform.md`,
`docs/agent_instructions/project_architecture.md`). It is a new,
standalone recommendation for what a reusable top-level `AGENTS.md`/
`agent.md` should contain on a *different* legacy-HPC modernization
project, generalized from language in this project's instruction files
that proved durable in practice.

Project-specific values from this project (absolute paths, platform names,
grid sizes, specific module names) are **excluded** below unless
explicitly marked "Example (this project)" — a new project should fill in
its own equivalents, not copy these verbatim.

---

## 1. Mission Statement

State, in a few sentences: what legacy system is being modernized, to
what target architecture, and why. Be specific about the staged
integration path (e.g. legacy-language -> stable ABI -> modern-backend
runtime -> accelerator kernels), since this shapes almost every later
policy decision (ABI stability requirements, validation levels, what
"done" means at each stage).

**Example (this project):** "Modernize the GEOS FV3 atmospheric
dynamical core for heterogeneous CPU/GPU execution via: GEOS/FV3 Fortran
-> ISO_C_BINDING -> stable C ABI -> C++/CUDA runtime -> persistent GPU
dyn_core subsystem."

## 2. Source-of-Truth Hierarchy

Declare, explicitly and in priority order:

1. The authoritative legacy implementation (name the exact repository/tag/
   commit if there is one) as the permanent numerical/behavioral source of
   truth — new-backend code must match it, not the other way around.
2. Any secondary reference implementation (e.g. a prior modernization
   attempt, a different team's port) that may inform architecture or
   performance expectations but is explicitly **not** authoritative for
   correctness.
3. Git and the current source tree as the authoritative record of current
   project state — always re-derive state from these, not from a prior
   session's summary or a stale document.
4. The living status document (see §9) as the compact, current-truth
   reference — consulted before any deeper historical document.

State explicitly: do not change scientific/numerical/behavioral results
merely because a secondary reference behaves differently, and do not
modify the secondary reference repository as part of this project's work.

## 3. Validation Policy

Mandate exactly four non-conflatable validation levels (adapt names to
taste, but keep exactly this shape): source exists and is wired
(IMPLEMENTED); compiler/toolchain accepts it (COMPILED); it actually ran
on the real target hardware/backend (EXECUTED); it has been compared
against the authoritative baseline at a stated tolerance
(NUMERICALLY VALIDATED). Require every status report to use these words
precisely and forbid inferring a higher level from a lower one.

Require: correctness before optimization — no performance work on a
subsystem before its numerical validation is complete; a named validation
methodology (e.g. coarse-to-fine localization at defined stage
boundaries) rather than only final-output comparison; and explicit
handling of structurally-degenerate test configurations (fields that are
zero/default by construction under the test harness) so a pass is never
claimed for a path the test never actually exercised.

## 4. Cost / Resource Policy

State the project's actual constraint (token budget, compute budget, wall-
clock budget, or some combination) and derive concrete behavior from it:

- Read the living status document plus task-relevant files at the start
  of a task; avoid rereading full historical documents unless the living
  document points to one or the current defect originated there.
- Prefer targeted inspection (exact symbols/files) over broad scans;
  expand only when targeted inspection is insufficient.
- Require bounded debugging: an explicit hypothesis, a minimum experiment
  set, and a stated stop condition for every investigation.
- Require reuse of existing harnesses/tools/formats before building new
  ones.
- Require terse routine progress reporting; reserve detail for material
  findings, blockers, and plan changes.
- Explicitly state that cost efficiency must never justify skipping
  authoritative verification, guessing at unknown layout/ordering, hiding
  a failed test, weakening a validation threshold, or claiming a
  validation level that was not actually performed. This clause is as
  important as the efficiency guidance itself — efficiency and rigor are
  not presented as a tradeoff.

## 5. Allowed Tools / Build-Run Discipline

- Name the build system and require out-of-source build directories, with
  an explicit list of what must never be committed (build artifacts,
  object files, binaries, scheduler logs, generated/temporary files).
- Require environment/toolchain verification by direct inspection
  (version-check commands actually run) rather than assuming a module/
  environment load succeeded correctly just because it didn't error.
- If more than one validated toolchain/environment stack exists for
  different purposes (e.g. a lightweight dev stack vs. an authoritative
  production stack), name both explicitly and state which one is
  authoritative for which kind of result — require every job's stack to
  be confirmed, not assumed, before trusting its output for a
  cross-stack-sensitive comparison.

## 6. GPU/Accelerator Job Policy

- State the execution model precisely: where the agent runs (e.g. a login
  node) vs. where accelerator-requiring work must run (e.g. submitted
  through a scheduler) — and state clearly that lacking direct hardware
  access from the agent's own location is expected, not a blocker.
- Provide the exact, verbatim module-load/environment-setup sequence,
  including any non-obvious ordering requirement (e.g. "clear path
  extensions before extending the search path, in that order, or the
  extension is lost").
- Provide a minimal smoke-test pattern for accelerator availability, and
  require it be run fresh in the current allocation before relying on a
  historical finding about hardware/driver availability or permissions.
- Name any known, confirmed platform limitation (e.g. a specific
  profiling tool blocked by a permissions issue) and the sanctioned
  fallback — and explicitly forbid privilege-escalation workarounds.
- Require scheduler-submitted jobs to log their own exact
  binary/configuration identity, and require that identity be checked
  before trusting a job's result.

## 7. Git Policy

- State explicitly: do not discard, reset, clean, revert, push, or merge
  existing work unless explicitly instructed. Require re-checking
  `git status`/`git log`/current branch before any action that could
  discard uncommitted work.
- Require new commits rather than amending, unless explicitly asked
  otherwise.
- State explicitly: do not push or merge without being explicitly asked,
  every time — a prior approval does not imply blanket future approval.
- If the project uses parallel branches/worktrees for independent
  ownership regions, state that opportunistic changes to another
  region's branch are not permitted without explicit assignment.

## 8. Stop Conditions

Collect, in one place, every "stop and do X instead of continuing" rule
used elsewhere in the instructions, so an agent can scan this section
alone to know when to halt: unvalidated code must not be integrated;
isolated-validation failures must not be papered over by proceeding to
integration anyway; a memory/resource check failing at target scale must
block a "production-ready" claim regardless of correctness/performance;
an acceptance bar must not be lowered after seeing a disappointing result;
ambiguous or unfamiliar repository state must be investigated, not
cleared away; a platform permission/capability limitation must not be
worked around via privilege escalation.

## 9. Documentation Expectations

- Require one compact, continuously-updated "current truth" document
  (target an explicit size ceiling) that a new session reads first, with
  deeper per-task/per-milestone documents created separately and read
  only on demand.
- Require the living document to be updated **in place** (superseded
  facts replaced, not appended as a new dated paragraph) so it does not
  grow into a second history dump.
- Require every milestone/task handoff to record: scope, what was
  measured, what passed/failed, remaining blockers, and the specific next
  recommended step — not just a narrative of what was attempted.
- Require stopped/rejected approaches to be documented with the same
  rigor as successful ones (what was tried, what was measured, why
  stopped) specifically so they are not silently re-attempted later.

## 10. Recovery Behavior

- Require state recovery in this order: git/source tree first, then the
  living status document, then deeper historical documents only if
  needed — never recovery from conversation memory alone across a session
  boundary.
- Require checking for existing evidence (prior job logs/output) before
  re-running any expensive computation.
- Require treating unfamiliar repository state (files, branches, changes)
  as potentially meaningful in-progress work to be investigated, not
  cleared away, until its origin is understood.

---

## Notes on Generalizing From This Project's Instructions

- This project's `CLAUDE.md` and `docs/agent_instructions/*` files are
  strong examples of §2 (source-of-truth hierarchy), §3 (validation
  policy), §4 (cost policy — see `docs/agent_handoff/cost_efficiency.md`
  for the fullest version found in this repository), §6 (GPU job policy,
  very platform-detailed in `discover_platform.md`), and §7 (git policy).
  Their phrasing on "never conflate IMPLEMENTED/COMPILED/EXECUTED/
  VALIDATED" and "cost efficiency must never justify skipping
  authoritative verification" generalize directly and are recommended for
  reuse nearly verbatim in a new project's instructions.
- One structural inconsistency observed in this project's own files is
  worth flagging explicitly rather than silently copying: `CLAUDE.md`
  references `docs/agent_instructions/cost_efficiency.md` via an
  `@`-include, but the actual, materially more detailed cost-efficiency
  policy lives at `docs/agent_handoff/cost_efficiency.md` instead — the
  referenced path does not exist. A new project's instruction set should
  keep its `@`-includes (or equivalent) and the files they point to in
  the same location, verified to resolve, to avoid this exact drift.
- This project's platform-specific file (`discover_platform.md`) and
  architecture-specific file (`project_architecture.md`) are kept
  separate from the cost/workflow policy — this separation (platform
  mechanics vs. architecture/ownership vs. cost/workflow policy, as
  distinct files) is itself a reusable organizational pattern worth
  adopting in a new project's instruction set, rather than one large
  undifferentiated file.
