# GEOS FV3 CUDA Modernization — Project Architecture Instructions

## Purpose

This file defines stable architectural context and development constraints for
the GEOS FV3 CUDA modernization project.

It should be read by all agents working on the project.

Platform-specific instructions belong in:

`docs/agent_instructions/discover_platform.md`

Phase-specific work belongs in the current task prompt.

---

## Primary Goal

Modernize the GEOS FV3 atmospheric dynamical core for heterogeneous CPU/GPU
execution.

The target implementation path is:

```text
GEOS/FV3 Fortran
    ->
ISO_C_BINDING
    ->
stable C ABI
    ->
C++/CUDA runtime
    ->
persistent GPU dyn_core subsystem
```

The purpose is NOT to create isolated CUDA versions of individual leaf
routines.

The purpose is to migrate coherent numerical subsystems while preserving the
scientific behavior of the authoritative FV3 implementation.

---

## Numerical Source of Truth

Authoritative GFDL FV3 Fortran is the numerical source of truth.

Do not change scientific or numerical behavior merely because another
implementation behaves differently.

The historical GEOS PACE / GT4Py / DaCe implementation may be used as:

- a GPU architecture reference;
- an implementation reference;
- a mapping reference;
- a performance/design reference.

It is NOT the numerical authority.

Do not modify `geos-gtfv3`.

---

## Repository Structure

The larger project contains multiple related repositories.

Primary top-level GEOS repository:

`GEOSfvdycore`

Primary FV3 component repository:

`FVdycoreCubed_GridComp`

Authoritative nested GFDL FV3 repository when populated:

`GFDL_atmos_cubed_sphere`

CUDA implementation belongs in:

`FVdycoreCubed_GridComp/cuda/dyn_core/`

The CUDA implementation must not be placed inside the historical
`geos-gtfv3` implementation.

---

## Git Development Model

The project used parallel worktrees for coherent work packages.

Important branches include:

```text
cuda/wp-a-cgrid
cuda/wp-b-dgrid
cuda/wp-c-vertical
cuda/wp-d-operators
cuda/wp-e-runtime
cuda/dyn-core-integration
cuda/geos-interface
cuda/gpu-bringup
```

Conceptual progression:

```text
WP-A/B/C/D/E
    ->
cuda/dyn-core-integration
    ->
cuda/geos-interface
    ->
cuda/gpu-bringup
    ->
validation
    ->
production GEOS integration
```

Historical WP branches should be treated as immutable inputs unless explicitly
assigned for repair.

Do not make opportunistic changes to unrelated work-package branches.

Do not merge or push unless explicitly requested.

Before editing:

```bash
git status
git branch --show-current
git log -5 --oneline --decorate
```

Do not reset, clean, revert, or discard existing work unless explicitly
authorized.

---

## Work Package Ownership

### WP-A — C-grid Predictor

Owns C-grid predictor functionality such as:

- C-grid transformations;
- predictor operations;
- C-grid divergence;
- C-grid `dz` updates;
- C-grid pressure-gradient-related preparation.

Do not duplicate WP-A algorithms in other packages.

### WP-B — D-grid Dynamics / Transport / Damping

Owns:

- D-grid dynamics;
- horizontal transport;
- order-6 transport;
- PPM paths;
- `fv_tp_2d`;
- `ra_x` / `ra_y`;
- `deln_flux`;
- D-grid damping;
- contravariant cubed-sphere edge corrections;
- horizontal transport service consumed by WP-C.

WP-C must reuse WP-B transport services rather than duplicate them.

### WP-C — Vertical / Nonhydrostatic Dynamics

Owns:

- vertical dynamics;
- Riemann solver paths;
- vertical workspace;
- C-grid vertical entry;
- D-grid/Riemann entry;
- `update_dz_d` orchestration using WP-B horizontal transport.

### WP-D — Pressure / Grid / Post-loop Operators

Owns:

- pressure reconstruction;
- pressure gradients;
- grid/update operators;
- hyperdiffusion;
- diffusive heating;
- edge/corner post-loop semantics.

### WP-E — GPU Runtime / State / API / Communication

Owns shared infrastructure:

- GPU context;
- FieldRegistry;
- device state;
- persistent workspace;
- numerical configuration;
- dispatch metadata;
- streams;
- event/dependency interfaces;
- halo/communication interfaces;
- public C ABI;
- shared grid/geometry/topology contracts.

Do not create an independent runtime inside WP-A/B/C/D.

---

## Stable GPU Architecture

The intended architecture is:

```text
GEOS/FV3 Fortran
        |
ISO_C_BINDING wrapper
        |
stable C ABI
        |
WP-E GPU context
        |
persistent device state
        |
integrated CUDA dyn_core
        |
WP-A / WP-B / WP-C / WP-D
```

Fortran must not call individual CUDA kernels.

The interface boundary should be subsystem-level.

---

## Lifecycle

The bridge should expose a coherent lifecycle conceptually equivalent to:

```text
initialize
step
sync when required
finalize
```

The GPU context should be created once and reused.

Do not create/destroy GPU state for every numerical kernel.

---

## Persistent Device Residency

A major architectural objective is persistent GPU state.

Desired pattern:

```text
initial host state
    ->
initial H2D
    ->
WP-A
    ->
WP-B
    ->
WP-C
    ->
WP-D
    ->
next acoustic/substep state remains on GPU
    ->
explicit D2H only at required boundary
```

Do not introduce:

```text
Fortran
-> H2D
-> one kernel
-> D2H
-> Fortran
```

around every numerical routine.

Full-state host/device transfer between work packages is considered an
architectural regression unless explicitly required by a defined external
boundary.

---

## Expected Synchronization Boundaries

Host synchronization may be required at defined boundaries such as:

- MPI halo exchange;
- physics;
- remapping;
- diagnostics;
- I/O;
- savepoint validation;
- finalization.

These boundaries must be explicit.

Do not introduce hidden device synchronization for convenience.

---

## Communication Evolution

Communication strategy is staged.

### Stage 1

Host-staged MPI is acceptable.

### Stage 2

CUDA-aware MPI may be introduced later.

### Stage 3

Communication/computation overlap may be optimized later.

The numerical package APIs should not need redesign when moving from Stage 1 to
later stages.

Do not prematurely optimize MPI during early correctness validation.

---

## Production First-Run Configuration

The first required production GEOS nonhydrostatic configuration is:

```text
hord_mt = 6
hord_vt = 6
hord_tm = 6
hord_dp = -6
vtdm4 > 0
grid_type = 0
hydrostatic = false
a_imp = 1.0
```

This configuration must remain supported.

Do not silently replace it with easier test settings such as:

```text
hord = 8
grid_type = 4
```

unless explicitly performing a separate test-only experiment.

Configuration values should propagate from GEOS/FV3 configuration into shared
runtime configuration objects.

Do not hard-code production configuration values inside numerical kernels.

Unsupported configurations should fail before mutating model state.

---

## Integrated Numerical Path

Prompt 4.2 established that all required numerical stages for the production
first-run dyn_core configuration were implemented in the integrated source
path.

The integrated stage inventory contains 54 required numerical stages:

- WP-E: 10
- WP-A: 7
- WP-B: 18
- WP-C: 9
- WP-D: 10

This means:

IMPLEMENTATION COVERAGE: COMPLETE

It does NOT automatically mean:

GPU EXECUTION: COMPLETE

or:

NUMERICAL VALIDATION: COMPLETE

Those must be demonstrated separately.

---

## Current Branch Milestones

Important project checkpoints include:

```text
077b729
Close GEOS production first-run dyn_core CUDA path

6baaefb
Add GEOS/FV3 CUDA interface (Prompt 5)

a395538
Update handoff with Prompt 5 commit SHA

bf4e6ea
Discover CUDA bring-up checkpoint
```

Always inspect actual Git history before relying on these identifiers, because
new commits may supersede them.

---

## GEOS/FV3 Interface

Prompt 5 established the intended bridge:

```text
GEOS/FV3 Fortran
    ->
ISO_C_BINDING
    ->
C ABI
    ->
WP-E context
    ->
CUDA dyn_core
```

Requirements:

- original Fortran backend remains available;
- CUDA backend remains optional;
- MPI communicator/domain decomposition is preserved;
- init/step/finalize lifecycle is explicit;
- Fortran does not call individual kernels;
- one coherent dyn_core subsystem interface is used;
- state/configuration mapping is explicit;
- persistent device residency is preserved.

---

## CPU / CUDA Backend Selection

The original Fortran implementation must remain available.

Desired behavior is conceptually similar to:

```text
FV3_DYNCORE_BACKEND=fortran
FV3_DYNCORE_BACKEND=cuda
```

The exact mechanism should follow repository/build conventions.

When CUDA is disabled, the original Fortran behavior should remain unchanged.

---

## Precision

Both R8 and R4 configurations are supported development targets.

Avoid maintaining independent numerical implementations for each precision.

Shared templates/types/configuration should preserve identical algorithmic
structure where possible.

Always report R8 and R4 validation separately.

---

## Validation Hierarchy

Always distinguish these levels:

### 1. IMPLEMENTED

The source code exists and the stage is wired.

### 2. COMPILED

The host/CUDA compiler accepts and links the implementation.

### 3. GPU EXECUTED

The implementation actually launches and executes on a real GPU.

### 4. NUMERICALLY VALIDATED

The CUDA result has been compared against authoritative Fortran reference data
using an accepted tolerance or bitwise criterion.

Never call a feature fully validated merely because it compiles.

Never call compilation a GPU test.

Never call successful GPU execution numerical equivalence.

---

## Prompt 5B Bring-Up Status

The initial Discover CUDA bring-up established:

- R8 CUDA compilation: PASS
- R4 CUDA compilation: PASS
- R8 regression tests: 10/10 PASS
- R4 regression tests: 10/10 PASS
- WP-E GPU runtime: PASS
- compute-sanitizer: PASS
- zero sanitizer errors reported

Package-specific statuses must be described carefully.

If only CUDA compilation was demonstrated for a package, report:

```text
CUDA compilation: PASS
GPU execution: NOT DIRECTLY TESTED
```

Do not label package-level GPU execution PASS solely from successful compilation.

At the initial Prompt 5B checkpoint, a complete integrated dyn_core GPU smoke
harness had not yet been executed.

Therefore integrated GPU execution remains a separate closure item until
demonstrated.

---

## Numerical Validation

Authoritative numerical validation must compare the CUDA candidate against the
original Fortran implementation.

Preferred methodology:

```text
Fortran baseline/savepoint
        vs
CUDA candidate/savepoint
        ->
identify first divergence
```

Important savepoint locations may include:

- dyn_core entry;
- after C-grid predictor;
- after C pressure stage;
- after D-grid dynamics/transport;
- after vertical/update_dz_d;
- after D pressure/NH stage;
- end acoustic substep;
- dyn_core exit.

Do not validate only final model output when intermediate savepoints can
identify the first numerical divergence.

---

## Performance Work Comes Later

Do not perform major optimization before numerical correctness is established.

Defer:

- kernel fusion;
- occupancy tuning;
- shared-memory tuning;
- multi-stream optimization;
- CUDA Graphs;
- CUDA-aware MPI;
- overlapping communication;
- multi-GPU scaling;
- performance benchmarking;
- production tuning.

Correctness and reproducibility come first.

---

## Modification Rules

Do not modify authoritative Fortran numerical algorithms unless explicitly
authorized.

Do not modify numerical CUDA algorithms solely to satisfy compiler behavior.

Mechanical CUDA compatibility changes are acceptable when they preserve
semantics.

For every nontrivial CUDA bring-up fix, document:

- file;
- original error;
- root cause;
- change;
- whether numerical semantics changed.

Expected answer should normally be:

```text
numerical semantics changed: NO
```

If a fix requires changing scientific semantics, stop and escalate instead of
silently making the change.

---

## Documentation / Handoff Practice

Major phases should leave a handoff document under:

`docs/agent_handoff/`

Handoffs should record:

- current branch;
- latest commit;
- work completed;
- tests executed;
- commands used;
- compile status;
- GPU execution status;
- numerical validation status;
- remaining blockers;
- next recommended phase.

Agents must recover state from:

1. Git;
2. source code;
3. handoff documents;

rather than relying only on prior conversation context.

---

## Git / Generated Artifact Policy

Do not commit:

- build directories;
- CMake-generated files;
- object files;
- binaries;
- Slurm logs;
- temporary test output;
- temporary sanitizer logs.

Commit:

- legitimate source fixes;
- build-system fixes;
- permanent tests;
- permanent infrastructure scripts;
- documentation.

Do not push automatically unless explicitly instructed.

---

## Phase Progression

Current conceptual project progression:

```text
Architecture / Gate A
    ->
WP-A/B/C/D/E
    ->
Prompt 4 / 4.2 integration
    ->
Prompt 5 GEOS/FV3 interface
    ->
Prompt 5B CUDA bring-up
    ->
integrated CUDA smoke closure
    ->
Prompt 6 authoritative numerical validation
    ->
full GEOS production integration
    ->
performance / CUDA-aware MPI / scaling
```

Do not skip numerical validation simply because CUDA compilation/runtime
bring-up succeeds.
