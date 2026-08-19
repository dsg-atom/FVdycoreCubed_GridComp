# GEOS FV3 CUDA Modernization

Read and follow these persistent project instructions:

@docs/agent_instructions/discover_platform.md
@docs/agent_instructions/project_architecture.md

## Before Every Task

Inspect:

```bash
pwd
git status
git branch --show-current
git log -5 --oneline --decorate
```

Recover project state from Git, source code, and the latest handoff documents
before making changes.

Do not discard or reset existing work without explicit instruction.

## Core Rules

- Authoritative GFDL FV3 Fortran is the numerical source of truth.
- Preserve the original Fortran backend.
- Keep CUDA optional.
- Preserve the WP-E persistent GPU runtime/state architecture.
- Do not duplicate numerical functionality across work packages.
- Do not introduce unnecessary full-state host/device transfers.
- Do not modify scientific algorithms merely to make CUDA compile.
- Do not modify `geos-gtfv3`.
- Do not push or merge unless explicitly requested.

Always distinguish:

- IMPLEMENTED
- COMPILED
- GPU EXECUTED
- NUMERICALLY VALIDATED

Do not claim a later validation level based on an earlier one.

## Discover

Claude Code remains on a Discover login node.

GPU-required execution must be submitted through Slurm according to:

@docs/agent_instructions/discover_platform.md

Lack of direct GPU access from the login node is expected and is not a blocker.

## Current Project Sequence

```text
WP-A/B/C/D/E
-> dyn_core integration
-> GEOS/FV3 interface
-> CUDA bring-up
-> integrated GPU smoke execution
-> authoritative Fortran numerical validation
-> production GEOS integration
-> performance/scaling
```

Follow the current task prompt for phase-specific objectives and completion
criteria.
