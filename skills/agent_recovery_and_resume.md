# Agent Recovery and Resume

## Purpose

Recover accurate project state after a session/context interruption (or
when starting a new session on an ongoing project) without re-deriving
history from memory, and without wastefully re-running expensive jobs
whose evidence already exists.

## Use When

- Starting any new work session on an ongoing modernization project.
- Resuming after a context compaction, disconnect, or handoff to a
  different engineer/agent.
- Before re-running any job that might already have been run and logged.

## Inputs Needed

- The project's living status document (a compact "current truth" file,
  not the full chronological history).
- Git itself (status, log, branch) as the authoritative record of what
  code state actually exists.
- Access to prior job logs/output directories, if re-running an expensive
  computation is being considered.

## Procedure

1. Recover state from, in this order: (a) git (status, log, branch,
   current HEAD), (b) the actual source tree, (c) the living status
   document, (d) deeper historical documents — only if the living status
   document points to one or the current task's defect clearly originated
   there. Never rely on conversation memory alone across a session
   boundary.
2. Before starting any new work, run the basic orientation commands (`git
   status`, `git branch --show-current`, `git log` recent commits) and
   treat their output as authoritative over any prior belief about the
   state of the tree.
3. Before re-running an expensive job (long GPU allocation, large-scale
   run), check whether the needed evidence already exists in a completed
   job's logs/output directory — searching existing logs is almost always
   cheaper than a fresh run.
4. If an expensive job did previously run but its result is in doubt
   (unclear which binary ran, unclear configuration), prefer re-deriving
   the needed fact from the existing job's own logs (binary-identity
   line, config dump) before deciding a fresh rerun is actually necessary.
5. Do not reset, clean, revert, or discard any existing uncommitted work
   when resuming — treat any unfamiliar file/branch/change as potentially
   meaningful in-progress work until investigated, not noise to clear away.
6. When the living status document and a deeper/older document disagree,
   trust the living status document for *current* state, but note the
   discrepancy explicitly rather than silently picking one — it may
   indicate the living document is stale and needs updating, or that the
   older document is itself superseded and should be marked as such.
7. At the end of a work session or major milestone, update the living
   status document in place (replace superseded facts, don't just append
   a new dated paragraph) so the next session's recovery (step 1) stays
   cheap.

## Validation / PASS Criteria

- Current git state (branch, HEAD, uncommitted changes) has been directly
  checked, not assumed from memory of a prior session.
- Before any expensive rerun, existing logs were checked for the needed
  evidence first.
- The living status document was consulted before any deep historical
  document, and any discrepancy found was explicitly resolved or flagged.

## STOP Conditions

- Unfamiliar uncommitted changes or branches are found and their purpose
  is unclear — stop and investigate (or ask) before any destructive git
  operation; do not assume they are safe to discard.
- The living status document and git state disagree in a way that affects
  the current task — resolve the discrepancy (usually by trusting git/
  source over the document, and then fixing the document) before
  proceeding on the task itself.

## Common Failure Modes

- Reconstructing project history purely from a conversation summary
  without cross-checking it against git/source — summaries can be lossy
  or stale.
- Re-running a multi-hour/expensive job to re-derive a fact that is
  already sitting in a completed job's log from a prior session.
- Discarding "clutter" (unfamiliar scratch files, branches) during
  recovery without checking whether it represents real in-progress work.
- Letting a stale historical document's claim stand uncorrected once a
  living status document or fresh evidence contradicts it.

## Cost-Control Guidance

- Read the living status document and run basic git orientation commands
  first, every session — this is cheap and prevents much more expensive
  mistakes (duplicated work, destructive operations on misunderstood
  state).
- Only read deep historical documents when the current task specifically
  needs that level of detail — not as a default recovery step.
- Treat "check existing logs before rerunning" as a near-zero-cost
  action relative to any real GPU/HPC job — always do this check first.

## Reusable Prompt Template

```
Before starting <task>, recover current project state: run git status/
branch/log, read <living status document path>, and read only the
source files directly relevant to <task>. Do not reconstruct history from
memory of a prior session without cross-checking against git/source. If
<task> would require rerunning an expensive job, first check <existing
log/output locations> for whether the needed evidence already exists.
Do not reset/clean/revert/discard any existing uncommitted work without
first investigating what it is.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** This project's
`CLAUDE.md` explicitly requires `pwd`/`git status`/`git branch
--show-current`/`git log -5` before every task, states "git and the
current source tree are authoritative," and forbids discarding existing
work without explicit instruction. `CURRENT_STATE.md` is deliberately kept
compact (~target 100-200 lines, hard ceiling ~30,000 characters) and is
updated in place rather than appended to, with deeper detail pushed to
dated `docs/performance/p*.md`/`docs/agent_handoff/*.md` files read only
on demand. A documented recovery lesson: "before concluding no output was
produced, check the run's actual execution subdirectory, not just the
top-level rundir" — one session wrongly concluded a run produced nothing
by checking the wrong directory; a later session found the output in the
same completed job without needing a rerun (`docs/agent_work_summary.md`
§8/§12).
