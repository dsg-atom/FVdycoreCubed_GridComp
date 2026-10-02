# Agent/Colleague Handoff Manifest

Minimal file set a new engineer or AI agent should receive to continue
this project, or to use it as a reference for a different legacy-HPC
modernization effort. Grouped by necessity.

---

## ESSENTIAL (read before doing anything)

| File | Why essential |
|---|---|
| `CLAUDE.md` | Top-level binding instructions: core rules, source-of-truth, cost policy pointer, task-start checklist. Everything else is subordinate to this. |
| `docs/agent_handoff/CURRENT_STATE.md` | The compact, continuously-updated "current truth" document. Must be read before any deeper historical document — tells you exactly what is validated, what is open, and what to do next. |
| `docs/agent_instructions/project_architecture.md` | Work-package ownership, validation-level terminology, modification rules, communication-staging policy. Needed to understand *why* the codebase is shaped the way it is before touching it. |
| `docs/agent_instructions/discover_platform.md` | Platform-specific execution model (login node vs. scheduler, module stacks, GPU job patterns). Needed before running anything that touches the GPU/scheduler at all — without this, a new agent will likely waste a cycle discovering constraints documented here already. |
| `docs/agent_handoff/cost_efficiency.md` | The actual (fuller) cost/token-efficiency policy — note this lives here, not at the path `CLAUDE.md`'s include references (`docs/agent_instructions/cost_efficiency.md`, which does not exist); flagged as a source conflict in `docs/agent_work_summary.md`. |
| `docs/agent_work_summary.md` (this package's primary output) | The distilled, non-chronological knowledge-transfer document: architecture, workflow, decision framework, and lessons across every category of this project's work. Intended to make most of the deeper historical documents below optional for a first pass. |
| `skills/README.md` + `skills/*.md` | Reusable, procedural playbooks derived from this project's actual experience, usable on a different legacy-HPC modernization project without reading this project's history at all. |

## RECOMMENDED (read when working in the relevant area)

| File | When to read |
|---|---|
| `docs/agent_handoff/milestone_summary_2026-09-19.md` | The single richest chronological/evidence source (LOC accounting, full milestone list, experiment table, performance table, numerical-validation summary, limitations). Read when `CURRENT_STATE.md` or `agent_work_summary.md` isn't detailed enough and you need the underlying evidence. |
| `docs/agent_handoff/implementation_reality_audit.md` | Read before trusting any pre-existing "COMPLETE"/validated claim at face value, and as a worked example of the IMPLEMENTED/COMPILED/EXECUTED/VALIDATED conflation failure mode. |
| `docs/performance/cuda_fv3_optimization_roadmap_2026-09-21.md` | Read when deciding or re-examining the hotspot-driven porting sequence/priorities. |
| Specific `docs/performance/p*.md` / `docs/agent_handoff/prompt_*.md` files | Read only the specific file for the specific subsystem/milestone you are currently working on or debugging — not as a batch. `CURRENT_STATE.md` §11 and `milestone_summary_2026-09-19.md` point to the right file by topic. |
| `docs/agent_instructions_recommendation.md` (this package's output) | Read if setting up a *new* project's top-level agent instructions and want a generalized starting template. |
| `docs/agent_handoff_manifest.md` (this file) | Read when deciding what subset of this repository to hand to a new colleague/agent, on this project or a new one. |

## PROJECT-SPECIFIC REFERENCE (rarely needed; FV3/Discover-specific)

| File/area | Why it's reference-only, not general-purpose |
|---|---|
| `cuda/dyn_core/`, `cuda/interface/` | The actual FV3 CUDA source — essential if you are continuing *this* project's implementation work, but not useful as reading material for a different codebase; use the `skills/` procedures instead for a different project. |
| `docs/agent_handoff/wp-a.md` through `wp-e-v3.1.md` | FV3-specific work-package design evolution — useful only if deep-diving a specific work package's history on this project. |
| `docs/agent_handoff/geos_interface.md`, `integration.md`, `cuda_bringup.md` | Early FV3-specific bring-up narratives, mostly superseded by `CURRENT_STATE.md`/`milestone_summary_2026-09-19.md`; read only for exact historical detail on those specific milestones. |
| `docs/agent_handoff/milestone_3a*.md` series | FV3-specific gate-closure detail (e.g. Gate C1 closure specifics); read only if a task needs to distinguish Gate C1 from Gate C2 precisely. |
| Individual `docs/performance/p49.md`-`p116.md` | Each is a detailed, FV3-specific record of one porting/investigation step; `CURRENT_STATE.md` and `agent_work_summary.md` already condense all the generalizable content — read an individual file only for exact numbers/evidence on that specific step. |
| `Held_Suarez_ForcingMod.F90`, `FV_StateMod.F90`, other FV3 Fortran source | Atmospheric-model-specific; use only if directly continuing FV3 work. |
| Discover-specific paths, Slurm partition/QOS names, module names in `discover_platform.md` | Specific to this one HPC platform; `docs/agent_instructions_recommendation.md` generalizes the *policy shape*, not these specific values — do not copy the specific values into a new project's docs. |

---

## Rationale Summary

The ESSENTIAL set is deliberately small: it is everything needed to
understand current status, platform constraints, and reusable
methodology without reading project history. The RECOMMENDED set adds
depth for anyone actually continuing FV3 work or wanting underlying
evidence rather than the distilled summary. The PROJECT-SPECIFIC
REFERENCE set is the bulk of the repository's documentation by file
count, but is intentionally excluded from a default handoff — a new
colleague or agent should not need to read ~90 historical handoff files
or ~77 performance reports to become productive; `docs/agent_work_summary.md`
and the `skills/` package exist specifically so they don't have to.
