# Reusable Legacy-HPC Modernization Skills

This directory distills operational playbooks from the GEOS FV3 CUDA
modernization project (see `docs/agent_work_summary.md` for the full
narrative and evidence). Each file is a **procedure**, not a history —
written so an engineer or agent working on a *different* legacy-Fortran (or
other legacy-language) → C++/CUDA modernization can follow it directly.

## What a skill file is

Each skill follows the same 10-section structure:

- **Purpose** — what problem this solves.
- **Use When** — the trigger conditions for reaching for this skill.
- **Inputs Needed** — what you must have in hand before starting.
- **Procedure** — the actual steps.
- **Validation / PASS Criteria** — how you know you're done, correctly.
- **STOP Conditions** — when to halt and escalate rather than push forward.
- **Common Failure Modes** — specific, previously-observed ways this goes
  wrong, and what they look like.
- **Cost-Control Guidance** — how to do this without burning excess
  tokens/compute/wall-clock.
- **Reusable Prompt Template** — a copy-pasteable prompt for an AI agent to
  execute this skill on a new project.
- **FV3 Example** — a concrete worked example from this project, explicitly
  marked as illustrative, not a universal rule.

## How to use these

1. Read the skill file relevant to your current step before starting it —
   not all twelve up front.
2. Treat the **Procedure** as a checklist, the **STOP Conditions** as hard
   gates, and the **FV3 Example** as evidence of feasibility, not as a
   template to copy verbatim onto a different codebase/platform.
3. Adapt the **Reusable Prompt Template** by replacing project-specific
   nouns (language names, platform names, file paths) — the structure of
   the prompt (what evidence to gather, what to report, what to stop on)
   is the part designed to transfer.

## Recommended loading order

Follow this sequence for a new modernization effort; skip ahead only if a
step genuinely doesn't apply yet (e.g. you have no multi-GPU concerns until
single-GPU correctness is established):

1. `legacy_hpc_architecture_mapping.md` — understand what you're modernizing
   before writing anything.
2. `fortran_baseline_harness.md` — get an authoritative legacy-language
   reference before writing any new-language code.
3. `fortran_cpp_abi_integration.md` — establish the integration boundary
   early; everything downstream calls through it.
4. `numerical_validation.md` — read this before step 5, not after — know
   how you will judge correctness before you have something to judge.
5. `cuda_kernel_porting.md` — the actual port, one bounded subsystem at a
   time.
6. `gpu_performance_profiling.md` — once something is correct, measure
   whether it's actually faster, and where the next bottleneck is.
7. `gpu_memory_diagnostics.md` — track memory headroom as you scale up.
8. `mpi_gpu_binding.md` — once single-GPU is solid, scale across GPUs/nodes.
9. `halo_exchange_modernization.md` — if your model has domain-decomposed
   boundary exchange, this governs scaling the decomposition itself.
10. `production_ab_testing.md` — the methodology underlying every
    performance claim in steps 6-9; read it alongside them, not strictly
    after.

Two more are not steps in the sequence but apply throughout:

- `agent_recovery_and_resume.md` — read before resuming any interrupted
  session or re-running an expensive job.
- `modernization_stop_go_decisions.md` — read before deciding to KEEP,
  PORT, DEFER, or STOP any individual port — this is the decision rubric
  used at the end of nearly every other skill's procedure.

## Generic vs. FV3-specific

All twelve skills are **generic in procedure**. Each one's "FV3 Example"
section is explicitly atmospheric-model-specific (cubed-sphere halos,
Held-Suarez forcing, `fv_timing_mod` timers, Discover/Slurm specifics) and
is marked as such — do not treat any FV3-specific detail (file names, env
var names, grid names like C512, platform names like Discover) as
applicable to your project. If your legacy codebase is not a cubed-sphere
atmospheric model, you can safely ignore every FV3-labeled detail and keep
only the Purpose/Procedure/Validation/STOP/Failure-mode content.

## Adapting to a different project

- Replace "Fortran" with your actual legacy language (could be Fortran 77,
  COBOL, legacy C, MATLAB) in the Procedure steps — the ABI-boundary and
  baseline-harness concepts apply to any legacy-language-to-modern-stack
  migration, not just Fortran-to-C++.
- Replace "CUDA"/"GPU" with your actual target backend (could be a
  different accelerator, a distributed-compute framework, or even just a
  modern CPU language) — the porting/validation/profiling procedures apply
  to any migration where you must prove new-code output matches
  old-code output before trusting a performance number.
- If your legacy model has no domain decomposition / halo exchange (e.g. a
  single-process scientific code), skip `halo_exchange_modernization.md`
  and `mpi_gpu_binding.md` entirely.
- If your platform has no HPC scheduler, replace every Slurm-specific
  instruction in `gpu_performance_profiling.md`/`mpi_gpu_binding.md`/
  `production_ab_testing.md` with your platform's job-submission mechanism
  — the methodology (paired A/B, same-allocation, alternating order) is
  what to keep; the exact scheduler commands are not.
