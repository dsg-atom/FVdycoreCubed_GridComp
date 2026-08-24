# GEOS FV3 Multi-Agent GPU Modernization Implementation Plan

> **Plan revision — August 21, 2026**
>
> This revision preserves the original architecture and agent roles while tightening
> the transition from subsystem integration to numerical validation.
>
> The key clarification is that **compiled/linked integration is not sufficient to
> enter numerical validation**. Before Prompt 6A/6B, the project must demonstrate a
> real single-rank integrated GPU execution of the production first-run `dyn_core`
> path, with WP-A, WP-B, WP-C, and WP-D actually dispatched on an NVIDIA GPU under
> WP-E persistent runtime/state management.
>
> In the current project history, Prompt 5 interface scaffolding was completed earlier
> than the original sequence specified. That work remains useful and should be reused,
> but it does **not** waive Milestone 3 / Gate C. After numerical validation, the
> existing interface must be re-verified for GEOS/FV3 hybrid execution rather than
> reimplemented from scratch.


## 1. Objective

Modernize the GEOS FV3 dynamical core for heterogeneous CPU/GPU execution using an AI-assisted, multi-agent workflow.

The Held–Suarez prototype already demonstrated the basic workflow:

```text
code map → target selection → translation spec → Fortran baseline → C++ port → numerical comparison
```

For GEOS FV3, the goal is **not** to repeat that workflow one small routine at a time. The migration unit should be substantially larger and should address the difficult production issues that the Held–Suarez experiment intentionally avoided:

- coupled computational kernels;
- persistent GPU-resident state;
- multi-routine integration;
- cubed-sphere halo exchange;
- MPI interaction;
- subsystem-level validation;
- performance across complete acoustic/dynamics timesteps.

The first major target is:

> **Replace the computational body of FV3 `dyn_core` with a GPU-resident C++/CUDA subsystem while retaining the surrounding GEOS/FV3 Fortran driver and existing domain decomposition.**

---

# 2. Target Scope

## 2.1 Existing GEOS/FV3 hierarchy

```text
GEOS / MAPL / ESMF
        |
        v
FV3 Fortran driver
        |
        v
fv_dynamics
        |
        v
dyn_core
        |
        +-- c_sw
        +-- d_sw
        +-- C-grid pressure-gradient calculations
        +-- nonhydrostatic pressure-gradient calculations
        +-- update_dz_c / update_dz_d
        +-- Riem_Solver_C
        +-- Riem_Solver3
        +-- del2_cubed / damping
        +-- halo exchanges
        +-- auxiliary calculations
```

Relevant original FV3 source includes:

```text
model/fv_dynamics.F90
model/dyn_core.F90
model/sw_core.F90
model/nh_core.F90
```

The existing pyFV3/PACE implementation provides a valuable independent reference. Its equivalent hierarchy is approximately:

```text
DynamicalCore
    |
    v
AcousticDynamics
    |
    +-- CGridShallowWaterDynamics
    +-- DGridShallowWaterLagrangianDynamics
    +-- NonhydrostaticVerticalSolverCGrid
    +-- NonhydrostaticVerticalSolver
    +-- NonHydrostaticPressureGradient
    +-- UpdateGeopotentialHeightOnCGrid
    +-- UpdateHeightOnDGrid
    +-- HyperdiffusionDamping
    +-- halo updaters
```

This existing implementation should be treated as:

1. an algorithmic cross-reference;
2. a decomposition reference;
3. a validation reference;
4. an example of whole-region GPU orchestration;

but **not** as the architecture that the C++/CUDA implementation must copy.

---

# 3. Proposed GPU Architecture

The initial implementation should use one subsystem boundary rather than many fine-grained Fortran/CUDA crossings.

```text
GEOS Fortran
    |
    v
fv_dynamics
    |
    | one subsystem call
    v
+--------------------------------------------------+
| GPU dyn_core subsystem                           |
|                                                  |
|   C-grid dynamics                                |
|   vertical C-grid solver                         |
|   C-grid pressure gradient                       |
|   D-grid dynamics                                |
|   height update                                  |
|   vertical solver                                |
|   nonhydrostatic pressure gradient               |
|   damping / diffusion                            |
|                                                  |
|   persistent GPU state                           |
|   reusable workspace                             |
|   CUDA streams/events                            |
+--------------------------------------------------+
    |
    v
fv_dynamics / remaining Fortran
```

## Core design rule

**Do not copy the entire model state from host to device between individual kernels.**

State used repeatedly within `dyn_core` should remain GPU resident for the complete subsystem execution.

---

# 4. Translation Work Packages

Instead of assigning agents one routine each, divide the subsystem into coherent work packages.

## WP0 — Architecture and Interface Contract

Owner: **Architecture Agent**

Defines before translation begins:

- subsystem entry/exit interface;
- Fortran/C ABI;
- array dimensions and indexing;
- halo width;
- staggered-grid conventions;
- precision policy;
- state ownership;
- GPU-residency rules;
- temporary/workspace ownership;
- error tolerances;
- MPI assumptions;
- CUDA stream policy;
- memory layout policy.

No translation agent should independently change these decisions.

---

## WP1 — C-grid Acoustic Dynamics

Owner: **C-Grid Agent**

Scope:

```text
c_sw
C-grid wind update
C-grid transport quantities
C-grid pressure-gradient preparation
associated local computational kernels
```

Equivalent pyFV3 reference:

```text
CGridShallowWaterDynamics
```

Deliverable:

```text
cpp/cuda/dyn_core/cgrid/
```

---

## WP2 — D-grid Acoustic Dynamics

Owner: **D-Grid Agent**

Scope:

```text
d_sw
D-grid prognostic-field update
mass fluxes
Courant numbers
vorticity / kinetic-energy terms
associated local computational kernels
```

Equivalent pyFV3 reference:

```text
DGridShallowWaterLagrangianDynamics
```

Deliverable:

```text
cpp/cuda/dyn_core/dgrid/
```

---

## WP3 — Nonhydrostatic / Vertical Dynamics

Owner: **Vertical Dynamics Agent**

Scope:

```text
Riem_Solver_C
Riem_Solver3
update_dz_c
update_dz_d
vertical pressure / interface-pressure calculations
vertical geopotential calculations
```

Equivalent pyFV3 references:

```text
NonhydrostaticVerticalSolverCGrid
NonhydrostaticVerticalSolver
UpdateGeopotentialHeightOnCGrid
UpdateHeightOnDGrid
```

Deliverable:

```text
cpp/cuda/dyn_core/vertical/
```

---

## WP4 — Pressure Gradient and Damping

Owner: **Pressure/Damping Agent**

Scope:

```text
C-grid pressure-gradient operation
nonhydrostatic pressure-gradient operation
del2_cubed
hyperdiffusion / dissipative heating pieces used inside dyn_core
```

Equivalent pyFV3 references include:

```text
p_grad_c_stencil
NonHydrostaticPressureGradient
HyperdiffusionDamping
```

Deliverable:

```text
cpp/cuda/dyn_core/operators/
```

---

## WP5 — GPU State, Memory and Communication

Owner: **Runtime Agent**

Scope:

- GPU state object;
- device allocation;
- persistent workspace;
- host/device conversion;
- CUDA stream/event management;
- halo buffers;
- MPI integration;
- CUDA-aware MPI path when available;
- fallback host-staging path;
- instrumentation.

Deliverable:

```text
cpp/cuda/runtime/
```

This agent should not rewrite numerical algorithms.

---

## WP6 — Subsystem Integration

Owner: **Integration Agent**

Scope:

Assemble WP1–WP5 into:

```cpp
fv3_gpu_dyn_core_step(...)
```

or equivalent subsystem API.

Responsibilities:

- preserve original algorithm ordering;
- preserve synchronization dependencies;
- launch kernels in correct sequence;
- call halo exchange at correct points;
- manage temporary lifetimes;
- provide initialization/finalization interfaces;
- connect to the GEOS/FV3 Fortran caller.

---

## WP7 — Validation

Owner: **Validation Agent**

Validation levels:

```text
Level 1: kernel-group comparison
Level 2: acoustic-substep comparison
Level 3: dyn_core-step comparison
Level 4: fv_dynamics-step comparison
Level 5: short GEOS integration run
Level 6: multi-day scientific regression
```

Validation should compare against the original Fortran implementation, not only pyFV3.

pyFV3 can be used as an additional diagnostic reference when discrepancies occur.

---

## WP8 — Performance Analysis

Owner: **Performance Agent**

Measure:

- total `dyn_core` wall time;
- GPU kernel time;
- host/device transfer time;
- MPI halo time;
- GPU idle time;
- kernel-launch overhead;
- memory bandwidth;
- occupancy;
- synchronization cost;
- scaling across GPUs/nodes.

Optimization begins only after subsystem-level numerical validation.

---

# 5. Multi-Agent System Architecture

## Recommended structure

Use a **supervisor + specialist agents + integration gate** model.

```text
                         Supervisor Agent
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
      Architecture Agent                Code-Mapping Agent
              |                                 |
              +----------------+----------------+
                               |
                         approved contract
                               |
        +----------------------+-----------------------+
        |              |              |                |
        v              v              v                v
    C-Grid Agent   D-Grid Agent   Vertical Agent   Operators Agent
        |              |              |                |
        +--------------+--------------+----------------+
                               |
                        Integration Agent
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
          Validation Agent             Runtime Agent
                 |                           |
                 +-------------+-------------+
                               |
                         Performance Agent
```

## Important rule

Translation agents work in parallel **only after WP0 is approved**.

The shared integration contract acts as a synchronization barrier.

---

# 6. Practical Multi-Agent Setup

Two stages are recommended.

## Stage A — Start immediately with coding agents + Git worktrees

Do not build a complicated orchestration platform before starting the FV3 work.

Create one Git branch/worktree per specialist agent.

Example:

```bash
git worktree add ../fv3-architecture agent/architecture
git worktree add ../fv3-cgrid agent/cgrid
git worktree add ../fv3-dgrid agent/dgrid
git worktree add ../fv3-vertical agent/vertical
git worktree add ../fv3-operators agent/operators
git worktree add ../fv3-runtime agent/runtime
git worktree add ../fv3-integration agent/integration
git worktree add ../fv3-validation agent/validation
```

Each agent operates only in its assigned worktree.

Benefits:

- clean ownership;
- minimal merge collisions;
- independent experiments;
- easy rollback;
- human inspection remains simple.

The Supervisor maintains:

```text
artifacts/agent_status.yaml
artifacts/dependency_graph.json
artifacts/integration_contract.yaml
```

Example status file:

```yaml
architecture:
  status: complete
  branch: agent/architecture

cgrid:
  status: translating
  depends_on:
    - architecture

dgrid:
  status: translating
  depends_on:
    - architecture
vertical:
  status: translating
  depends_on:
    - architecture
operators:
  status: translating
  depends_on:
    - architecture
integration:
  status: blocked
  depends_on:
    - cgrid
    - dgrid
    - vertical
    - operators
    - runtime
```

---

## Stage B — Automate orchestration with LangGraph

Once the manual multi-agent workflow is stable, encode it as a graph.

Conceptually:

```text
MapCode
   ↓
DefineContract
   ↓
 ┌────────────── parallel ────────────────┐
 ↓              ↓             ↓           ↓
CGrid          DGrid        Vertical    Operators
 └───────────────┬─────────────┬──────────┘
                 ↓
              Integrate
                 ↓
              Validate
           ↙ pass     fail ↘
     Performance      DebugRouter
                         |
               offending work package
```

State passed through the graph should contain metadata, not giant source-code blobs.

Suggested state:

```python
class ModernizationState:
    repo_root
    baseline_commit
    architecture_spec
    work_packages
    changed_files
    build_status
    validation_status
    numerical_metrics
    performance_metrics
    blocking_issues
```

Agents should read files from the repository directly rather than embedding the entire codebase in prompts.

---

# 7. Shared Repository Structure

Suggested layout:

```text
geos-fv3-modernization/
|
+-- agents/
|   +-- prompts/
|   +-- tools/
|
+-- artifacts/
|   +-- code_map/
|   +-- call_graph/
|   +-- agent_status.yaml
|   +-- dependency_graph.json
|   +-- integration_contract.yaml
|
+-- docs/
|   +-- architecture/
|   +-- translation_specs/
|   +-- validation/
|   +-- performance/
|
+-- translated/
|   +-- fv3/
|       +-- cpp/
|       +-- cuda/
|           +-- dyn_core/
|               +-- cgrid/
|               +-- dgrid/
|               +-- vertical/
|               +-- operators/
|           +-- runtime/
|
+-- interface/
|   +-- fortran/
|   +-- c/
|
+-- tests/
|   +-- baseline/
|   +-- subsystem/
|   +-- integration/
|   +-- reports/
|
+-- benchmarks/
```

---

# 8. Step-by-Step Agent Prompts

The prompts below are intended to be pasted directly into Codex, Claude Code, or another repository-aware coding agent.

Replace placeholders such as `<REPO_ROOT>` and `<BASELINE_BRANCH>` where needed.

---

## Prompt 0 — Supervisor: Establish Project Rules

```text
You are the supervisor for a production-scale GPU modernization of the GEOS FV3 dynamical core.

Repository: <REPO_ROOT>
Baseline branch/commit: <BASELINE_BRANCH>

The Held–Suarez proof-of-concept phase is complete. Do NOT optimize for proving that one isolated Fortran routine can be translated. The new objective is subsystem-level modernization.

Primary target:
Replace the computational body of FV3 dyn_core with a GPU-resident C++/CUDA subsystem while preserving the GEOS/FV3 outer Fortran driver and domain decomposition.

Your first task is project governance only. Do not translate numerical routines.

Create:
1. docs/architecture/project_scope.md
2. artifacts/agent_status.yaml
3. artifacts/work_packages.yaml
4. docs/architecture/coding_and_agent_rules.md

Define:
- source-of-truth rules;
- branch/worktree ownership;
- which agents may edit which directories;
- review gates;
- merge order;
- numerical validation requirements;
- rule that translation agents cannot independently change shared interfaces.

Use original Fortran as the numerical source of truth. pyFV3/PACE may be used as an independent reference but must not silently replace behavior from the Fortran source.

Do not modify production numerical code yet.
```

---

## Prompt 1 — Code-Mapping Agent: Map `dyn_core`

```text
Analyze the FV3 dyn_core subsystem in this repository.

Primary entry point:
model/dyn_core.F90

Also inspect all directly and transitively called numerical routines relevant to a normal nonhydrostatic dyn_core timestep.

Produce a subsystem-level map, not a generic repository summary.

Create:
1. artifacts/code_map/dyn_core_call_graph.json
2. artifacts/code_map/dyn_core_call_graph.md
3. docs/architecture/dyn_core_execution_sequence.md
4. docs/architecture/dyn_core_dataflow.md

For every called routine record:
- defining file/module;
- caller(s);
- inputs;
- outputs/inout arrays;
- grid staggering;
- halo requirement;
- vertical dependency;
- horizontal stencil radius;
- MPI/communication dependency;
- temporary arrays;
- compile-time/namelist branches;
- whether it is computational, orchestration, communication, diagnostic, or setup code.

Trace the exact ordering inside the acoustic-substep loop.

Cross-reference equivalent pyFV3/PACE components where confidently identifiable, but clearly label those mappings as cross-references rather than the source of truth.

Do not translate anything.
```

---

## Prompt 2 — Architecture Agent: Define the Shared GPU Contract

```text
Using the dyn_core call graph and dataflow artifacts, design the shared GPU subsystem contract.

The target architecture is one coarse-grained Fortran → GPU subsystem call for dyn_core, not one call per kernel.

Create:
1. artifacts/integration_contract.yaml
2. docs/architecture/gpu_dyn_core_interface.md
3. docs/architecture/gpu_memory_model.md
4. docs/architecture/grid_and_indexing_contract.md

Specify exactly:
- public initialization API;
- public dyn_core-step API;
- public finalize API;
- Fortran/C ABI data types;
- array shapes;
- lower-bound/index conversions;
- C-grid/D-grid/A-grid staggering conventions;
- halo widths;
- precision;
- contiguous-layout assumptions;
- persistent GPU-resident fields;
- fields copied at entry/exit;
- device workspace;
- ownership/lifetime rules;
- CUDA stream model;
- synchronization points;
- MPI communicator handling;
- CUDA-aware MPI assumptions and fallback behavior.

Minimize host/device transfers.

The contract must allow C-grid, D-grid, vertical, and operator agents to work independently without inventing incompatible interfaces.

Do not implement the numerical kernels yet.
```

### Gate A

Human/Supervisor reviews:

```text
dyn_core_call_graph.md
gpu_dyn_core_interface.md
gpu_memory_model.md
integration_contract.yaml
```

Parallel translation starts only after this gate passes.

---

## Prompt 3A — C-Grid Agent

```text
Implement the C-grid work package defined in artifacts/work_packages.yaml and artifacts/integration_contract.yaml.

Source of truth:
original FV3 Fortran implementation, especially the c_sw path reached from model/dyn_core.F90.

Independent reference:
pyFV3 CGridShallowWaterDynamics and related stencils, when useful.

Do not change the shared integration contract.
Do not modify D-grid, vertical-solver, MPI, or Fortran-interface ownership areas.

Tasks:
1. identify all numerical operations belonging to the C-grid work package;
2. write docs/translation_specs/cgrid.md;
3. implement CPU C++ reference versions where useful for debugging;
4. implement CUDA kernels;
5. expose only the work-package API specified by the integration contract;
6. add deterministic group-level tests;
7. record any mismatch between Fortran and pyFV3 behavior.

Preserve algorithm order and numerical semantics before optimizing.

Deliver a short report containing:
- files changed;
- kernels implemented;
- unsupported branches;
- numerical test status;
- interface assumptions;
- blockers for integration.
```

---

## Prompt 3B — D-Grid Agent

```text
Implement the D-grid work package defined in artifacts/work_packages.yaml and artifacts/integration_contract.yaml.

Source of truth:
original FV3 Fortran d_sw path reached from model/dyn_core.F90.

Independent reference:
pyFV3 DGridShallowWaterLagrangianDynamics and related stencils.

Do not modify the shared contract or other agents' directories.

Include the coupled calculations needed for:
- D-grid prognostic updates;
- mass fluxes;
- Courant numbers;
- relevant vorticity / kinetic-energy / transport terms.

Create docs/translation_specs/dgrid.md and the CUDA implementation under your assigned directory.

Build tests at the work-package boundary rather than one test executable per tiny helper routine.

Report numerical status and unresolved dependencies.
```

---

## Prompt 3C — Vertical Dynamics Agent

```text
Implement the nonhydrostatic and vertical-dynamics work package.

Primary Fortran targets include the paths corresponding to:
- Riem_Solver_C
- Riem_Solver3
- update_dz_c
- update_dz_d
- required interface-pressure/geopotential calculations

Use original Fortran behavior as the source of truth.
Use pyFV3 vertical solver classes only as an independent cross-reference.

Pay special attention to vertical recurrences. Do not incorrectly parallelize a k-dependent recurrence across vertical levels.

Document, for every kernel:
- whether k is parallel, forward-recursive, or backward-recursive;
- horizontal decomposition strategy;
- temporary workspace;
- synchronization requirements.

Create docs/translation_specs/vertical_dynamics.md.
Implement and test the work package behind the shared subsystem contract.
```

---

## Prompt 3D — Pressure/Damping Agent

```text
Implement the pressure-gradient and damping work package.

Scope includes the dyn_core calculations corresponding to:
- C-grid pressure gradient;
- nonhydrostatic pressure gradient;
- del2_cubed / hyperdiffusion used on this path;
- related dissipative-heating calculations that belong inside the dyn_core subsystem.

Source of truth: original FV3 Fortran.
Independent references may include pyFV3 p_grad_c, NonHydrostaticPressureGradient, and HyperdiffusionDamping.

Do not alter shared state definitions.

Favor kernel fusion only after a numerically equivalent unfused version exists and passes tests.

Create docs/translation_specs/operators.md and report all stencil radii/halo dependencies explicitly.
```

---

## Prompt 3E — Runtime Agent

```text
Implement GPU runtime support for the dyn_core subsystem according to artifacts/integration_contract.yaml.

Do not implement or alter numerical algorithms.

Implement:
- GPU state allocation;
- persistent field storage;
- reusable workspace;
- device pointer mapping;
- initialization/finalization;
- CUDA stream/event support;
- halo send/receive buffers;
- CUDA-aware MPI path where the MPI implementation supports device buffers;
- host-staging fallback path;
- timing hooks;
- error checking.

Explicitly measure all host/device transfers.

Create docs/architecture/runtime_implementation.md.
```

---

# 9. Parallel Agent Execution

After Gate A:

```text
3A C-grid       ┐
3B D-grid       │
3C Vertical     ├── run in parallel
3D Operators    │
3E Runtime      ┘
```

Each agent commits only to its own branch.

Do not merge directly into the baseline branch.

---

# 10. Integration Prompt

## Prompt 4 — Integration Agent

```text
Integrate the completed dyn_core GPU work packages.

Inputs:
- artifacts/integration_contract.yaml
- C-grid branch
- D-grid branch
- vertical branch
- operators branch
- runtime branch

Reproduce the original model/dyn_core.F90 execution ordering.

Implement a coarse-grained subsystem entry point similar to:

    gpu_dyn_core_init(...)
    gpu_dyn_core_step(...)
    gpu_dyn_core_finalize(...)

Do NOT copy fields between host and device between internal kernels.

Keep subsystem state GPU resident during gpu_dyn_core_step.

Preserve communication ordering and dependencies.

First objective: correctness.
Do not perform aggressive fusion or asynchronous overlap until validation passes.

Create:
1. docs/architecture/dyn_core_gpu_execution_sequence.md
2. docs/integration/dyn_core_integration_report.md
3. subsystem integration tests.

Integration is not complete merely because all work-package libraries compile or
link. The integrated test path must make it possible to prove that WP-A, WP-B,
WP-C, and WP-D are actually dispatched through one coherent `gpu_dyn_core_step`
execution on a real GPU.

If work packages disagree with the shared integration contract, do not silently invent an adapter. Record the incompatibility and route it back to the responsible agent/supervisor.
```

---

# 10A. Milestone 3 Integrated GPU Execution Closure

## Prompt 4C — Integrated GPU Execution Closure Agent

Use this prompt after the work packages compile and link, but before Prompt 6A/6B.

```text
Close the Milestone 3 execution gap for the integrated FV3 CUDA dyn_core.

This is NOT numerical validation.
This is NOT a collection of isolated leaf-kernel demonstrations.
This is NOT full GEOS production integration.

Goal:
Demonstrate one real single-rank integrated GPU execution of the supported
production first-run dyn_core/acoustic path.

The execution must use the shared WP-E runtime/context and must actually dispatch
the required numerical work packages in the integrated ordering:

    WP-A C-grid predictor
      ->
    WP-C C-grid vertical stage where required
      ->
    WP-B D-grid dynamics / transport / damping
      ->
    WP-C D-grid / vertical / Riemann stage
      ->
    WP-D pressure / grid-update / post-loop operators

Requirements:

1. Use the existing integrated subsystem API and shared contracts. Do not create a
   second numerical implementation solely for testing.

2. Execute on a real NVIDIA GPU.

3. Prove WP-A, WP-B, WP-C, and WP-D are reached by the integrated dispatch.
   Merely linking their libraries is not sufficient.

4. Use the supported production first-run configuration rather than an easier
   test-only path:

       hord_mt = 6
       hord_vt = 6
       hord_tm = 6
       hord_dp = -6
       vtdm4 > 0
       grid_type = 0
       hydrostatic = false
       a_imp = 1.0

5. Preserve persistent device residency:

       initial H2D
         -> integrated WP execution
         -> explicit final/validation D2H

   No unintended full-state H2D/D2H transfer is allowed between work packages.

6. Check successful CUDA launches, synchronization, and finite outputs.

7. Run compute-sanitizer memcheck on the integrated executable where available.

8. Do not compare against authoritative Fortran values yet. Prompt 6A/6B owns
   savepoint capture and numerical comparison.

9. If full integrated execution cannot be achieved because dispatch wiring,
   corner/halo services, state registration, or orchestration is incomplete,
   report the exact missing integration infrastructure and fix that integration
   layer. Do not fall back to declaring success from isolated kernel tests.

Deliver:

- integrated executable/test harness;
- evidence that WP-A/WP-B/WP-C/WP-D were executed;
- persistent-residency report;
- compute-sanitizer result;
- updated integration/handoff documentation.

Completion status must distinguish:

IMPLEMENTED
COMPILED
GPU EXECUTED
NUMERICALLY VALIDATED

End with either:

READY FOR PROMPT 6A BASELINE CAPTURE

or

MILESTONE 3 EXECUTION BLOCKED
```

### Gate C1 — Integrated execution gate

Before Prompt 6A begins, require all of the following:

```text
single-rank integrated gpu_dyn_core_step executes on a real GPU
WP-A is GPU executed through integrated dispatch
WP-B is GPU executed through integrated dispatch
WP-C is GPU executed through integrated dispatch
WP-D is GPU executed through integrated dispatch
persistent device residency is preserved across the integrated path
no unresolved CUDA runtime/memory defect remains
```

Compilation, linkage, or execution of only WP-E plus one operator kernel does not
satisfy Gate C1.


---

# 11. Fortran Integration Prompt

## Prompt 5 — GEOS/FV3 Interface Agent

> **Sequencing note:** Interface scaffolding may be implemented early to de-risk ABI
> and lifecycle design. However, Gate D cannot be claimed until Milestone 3 execution
> and numerical validation are complete. If Prompt 5 scaffolding already exists,
> reuse and re-verify it after Gate C rather than rebuilding it.

```text
Connect the validated GPU dyn_core subsystem to the existing Fortran FV3 driver.
If interface scaffolding was created earlier, audit and reuse it rather than
reimplementing it.

Requirements:
- preserve the original Fortran implementation behind a build/runtime switch;
- use ISO_C_BINDING or the approved project ABI;
- add init/step/finalize lifecycle calls;
- avoid exposing individual CUDA kernels to Fortran;
- pass one coherent dyn_core subsystem state/interface;
- preserve the existing MPI communicator/domain decomposition;
- do not alter scientific behavior outside the migration boundary.

Provide a runtime/build option similar in intent to:

    FV3_DYNCORE_BACKEND=fortran
    FV3_DYNCORE_BACKEND=cuda

Exact configuration mechanism should follow repository conventions.

Create docs/integration/fortran_gpu_bridge.md.
```

---

# 12. Validation Prompts

## Prompt 6A — Baseline Capture Agent

```text
Build a subsystem-level Fortran baseline for dyn_core.

Do not create one standalone executable for every tiny routine.

Capture state at meaningful savepoints corresponding to the computational phases:

1. dyn_core input
2. after C-grid dynamics
3. after C-grid pressure/vertical update
4. after D-grid dynamics
5. after vertical solver
6. after nonhydrostatic pressure gradient
7. dyn_core output

Store only the fields required to localize numerical differences.

Record configuration, grid dimensions, precision, timestep, and namelist settings with every baseline dataset.
```

---

## Prompt 6B — Numerical Validation Agent

```text
Compare GPU dyn_core against the original Fortran baseline.

Validation hierarchy:

1. compare major internal savepoints;
2. compare final dyn_core output;
3. compare one fv_dynamics timestep;
4. run short integrated FV3/GEOS cases;
5. run multi-day regression after timestep-level agreement is acceptable.

For each state variable report:
- max absolute error;
- max relative error;
- RMSE;
- location of maximum discrepancy;
- NaN/Inf count;
- conservation-related metrics when applicable.

Do not hide a mismatch by relaxing tolerances without documenting the reason.

When a mismatch occurs, use the savepoint sequence to identify the first divergent work package and route the bug to that agent.

Create machine-readable JSON reports plus a concise Markdown summary.
```

---

# 13. Debug Routing

The Supervisor should route failures using the **first divergent savepoint**.

```text
C-grid mismatch
     → C-Grid Agent

D-grid mismatch after C-grid matches
     → D-Grid Agent

vertical mismatch
     → Vertical Dynamics Agent

pressure/damping mismatch
     → Operators Agent

multi-rank-only mismatch
     → Runtime/Communication Agent

all packages pass separately but subsystem fails
     → Integration Agent
```

This prevents several agents from editing the same numerical code simultaneously.

---

# 14. Performance Prompt

## Prompt 7 — Performance Agent

```text
Profile the validated GPU dyn_core implementation.

Compare against the original Fortran dyn_core using the same grid, rank layout, timestep configuration, and precision.

Measure separately:
- total dyn_core time;
- kernel execution time;
- host/device copies;
- halo/MPI time;
- synchronization;
- GPU idle gaps;
- major kernel groups;
- memory allocation activity.

Use Nsight Systems for execution timeline analysis and Nsight Compute for selected hot kernels where available.

Prioritize optimizations in this order:
1. eliminate repeated host/device copies;
2. eliminate repeated allocations;
3. reduce unnecessary synchronization;
4. overlap halo exchange with independent computation;
5. fuse clearly memory-bound adjacent kernels when safe;
6. tune memory layout/access;
7. tune launch configuration.

Do not change scientific algorithms solely to improve benchmark numbers.

Create docs/performance/dyn_core_gpu_profile.md.
```

---

# 15. Milestones and Gates

## Milestone 1 — Architecture complete

Required:

```text
dyn_core call graph
dataflow map
integration contract
GPU memory model
work-package ownership
```

**Gate A: approve interfaces before translation.**

---

## Milestone 2 — Parallel work packages complete

Required:

```text
C-grid implementation
D-grid implementation
vertical implementation
operators implementation
runtime implementation
work-package-level tests
```

**Gate B: each package must compile and pass its boundary tests.**

---

## Milestone 3 — GPU dyn_core subsystem runs and validates

Milestone 3 is intentionally split into two gates so that compilation/linkage
cannot be mistaken for integrated GPU execution.

### Milestone 3A — Integrated GPU execution

Required:

```text
single-rank integrated gpu_dyn_core_step executes on a real GPU
WP-A/WP-B/WP-C/WP-D are actually dispatched
persistent GPU state across the subsystem
correct production first-run operation ordering
no unresolved CUDA runtime/memory defects
```

**Gate C1: integrated GPU execution is demonstrated.**

### Milestone 3B — Authoritative numerical validation

Required:

```text
Fortran subsystem savepoints captured
GPU savepoints captured at matching boundaries
first-divergence comparison available
subsystem output agrees with Fortran within agreed tolerances
```

**Gate C2: subsystem savepoints agree with authoritative Fortran within agreed tolerances.**

Gate C is complete only when both C1 and C2 pass.

---

## Milestone 4 — GEOS/FV3 hybrid execution

Required:

```text
Fortran → GPU dyn_core → Fortran
runtime backend switch
short integrated simulation
```

**Gate D: original and GPU backends both run from the same source tree.**

---

## Milestone 5 — Multi-GPU execution

Required:

```text
halo exchange
multi-rank execution
CUDA-aware MPI or staged fallback
```

**Gate E: multi-rank numerical regression passes.**

---

## Milestone 6 — Performance optimization

Required:

```text
profile
bottleneck ranking
optimization iterations
strong/weak scaling
```

---

# 16. What Not to Do

Avoid returning to the Held–Suarez experimental pattern of:

```text
routine 1 → translate → validate
routine 2 → translate → validate
routine 3 → translate → validate
```

for dozens of tiny FV3 routines.

Also avoid:

- one Fortran/CUDA boundary per leaf kernel;
- one host/device copy per kernel;
- independent data-layout decisions by different agents;
- optimizing kernels before subsystem integration;
- translating MPI/halo infrastructure before numerical ownership is clear;
- letting multiple agents edit the same source files simultaneously;
- replacing Fortran behavior with pyFV3 behavior when they disagree without investigation.

---

# 17. Recommended First Execution Sequence

The practical sequence is:

```text
1. Freeze a known-good baseline commit.

2. Run Prompt 0 — Supervisor setup.

3. Run Prompt 1 — dyn_core mapping.

4. Run Prompt 2 — architecture/interface contract.

5. Human review: Gate A.

6. Create worktrees for:
      C-grid
      D-grid
      Vertical
      Operators
      Runtime

7. Run Prompts 3A–3E concurrently.

8. Review/package-test each branch.

9. Run Prompt 4 — subsystem integration.

10. Run Prompt 4C — Milestone 3 integrated GPU execution closure.
       Require a real single-rank gpu_dyn_core_step that dispatches
       WP-A/WP-B/WP-C/WP-D with persistent device state.
       Gate C1 must pass.

11. Run Prompt 6A — authoritative Fortran baseline/savepoint capture.

12. Run Prompt 6B — subsystem numerical validation.
       Compare matching savepoints and identify the first divergence.

13. Fix first-divergence issues through debug routing until Gate C2 passes.

14. Run Prompt 5 — connect the validated subsystem to GEOS/FV3,
       OR, if Prompt 5 scaffolding was completed early, re-verify and
       finish that existing interface against the now-validated subsystem.

15. Validate single-rank hybrid Fortran -> GPU dyn_core -> Fortran execution.
       Gate D must pass.

16. Add multi-rank halo exchange and validate.
       Gate E must pass.

17. Run Prompt 7 — performance profiling.

18. Optimize the integrated subsystem.
```

## Current-project sequencing note

The current implementation performed Prompt 5 interface scaffolding and CUDA
bring-up before Gate C was complete. This is acceptable as de-risking work, but
the project should now return to the milestone sequence at **Step 10**:

```text
current CUDA/interface scaffolding
        ->
Prompt 4C integrated WP-A/B/C/D GPU execution
        ->
Prompt 6A authoritative Fortran savepoints
        ->
Prompt 6B numerical validation / first divergence
        ->
re-verify existing Prompt 5 interface
        ->
single-rank GEOS/FV3 hybrid execution
```

Do not interpret early completion of interface code as permission to skip Gate C.

---

# 17A. How to Run This With Codex

This section turns the execution sequence above into an operational Codex workflow. For the first GEOS/FV3 migration cycle, use Codex sessions plus Git worktrees as the multi-agent system. Do not build LangGraph first; automate only after this workflow has been exercised successfully once.

## 17A.1 Operating model

Use one supervisor session, five parallel implementation sessions, and one integration session:

```text
                           Human lead
                               |
                               v
                       Supervisor Codex
                    Prompts 0 -> 1 -> 2
                               |
                     Gate A: human review
                               |
       +---------------+-------+-------+----------------+
       |               |               |                |
       v               v               v                v
   C-grid Codex     D-grid Codex   Vertical Codex   Operators Codex
    Prompt 3A        Prompt 3B       Prompt 3C         Prompt 3D
       |               |               |                |
       +---------------+-------+-------+----------------+
                               |
                         Runtime Codex
                          Prompt 3E
                               |
                               v
                       Integration Codex
                          Prompt 4
                               |
                               v
                  Integrated GPU execution
                          Prompt 4C
                               |
                         Gate C1
                               |
                               v
                   Baseline + Validation
                      Prompts 6A, 6B
                               |
                         Gate C2
                               |
                               v
                    GEOS interface/hybrid
                Prompt 5 (or re-verify early
                    Prompt 5 scaffolding)
                               |
                               v
                         Prompt 7
```

The human lead owns architecture approval and merge decisions. Agents may propose changes to the shared contract, but implementation agents must not silently redefine it.

## 17A.2 Prepare the repository

Start from a clean, known-good repository state.

```bash
git status
git rev-parse HEAD
git branch --show-current
```

Run the existing baseline tests or reference case and record the exact commit used. Then create the supervisor branch:

```bash
git switch -c gpu/dyn-core-supervisor
```

Create a project directory for persistent handoff artifacts:

```bash
mkdir -p docs/gpu_port/agent_status
```

Recommended shared files are:

```text
docs/gpu_port/
├── baseline_commit.md
├── dyn_core_map.md
├── dyn_core_interface_contract.md
├── validation_policy.md
├── integration_status.md
└── agent_status/
```

These files are the durable communication channel between Codex sessions. Do not rely on chat history to communicate architecture decisions between agents.

## 17A.3 Supervisor session: run Prompts 0, 1, and 2

Open one Codex session at the root of the supervisor worktree/branch.

### Step 1 — Prompt 0

Paste **Prompt 0 — Supervisor: Establish Project Rules** into this session.

Require the agent to:

- inspect the repository before editing;
- record the baseline commit;
- establish directory and naming conventions;
- create or update project documentation under `docs/gpu_port/`;
- avoid changing production numerical code at this stage;
- commit its completed artifact changes.

Suggested commit:

```bash
git add docs/gpu_port
git commit -m "gpu: establish dyn_core modernization project rules"
```

### Step 2 — Prompt 1

In the same supervisor Codex session, paste **Prompt 1 — Code-Mapping Agent: Map `dyn_core`**.

Its primary durable output should be:

```text
docs/gpu_port/dyn_core_map.md
```

The map should identify at minimum:

- `fv_dynamics -> dyn_core` entry path;
- call graph below `dyn_core`;
- C-grid and D-grid operations;
- nonhydrostatic/vertical operations;
- pressure-gradient and damping operations;
- halo/MPI boundaries;
- inputs, outputs, temporaries, and state ownership;
- loop/domain shapes and halo requirements;
- conditional paths controlled by namelist/configuration options.

Commit the result before moving on.

```bash
git add docs/gpu_port/dyn_core_map.md
git commit -m "gpu: map FV3 dyn_core computational subsystem"
```

### Step 3 — Prompt 2

Still in the supervisor session, paste **Prompt 2 — Architecture Agent: Define the Shared GPU Contract**.

Require these concrete outputs:

```text
docs/gpu_port/dyn_core_interface_contract.md
docs/gpu_port/validation_policy.md
```

The interface contract must explicitly define:

- host and device ownership of every persistent field;
- array dimensions and index ordering;
- halo width and valid computational regions;
- precision and type mapping;
- Fortran/C/C++ ABI conventions;
- initialization, step, and finalize APIs;
- GPU workspace lifetime;
- stream ownership and synchronization rules;
- MPI/halo assumptions;
- error propagation;
- configuration fields passed from Fortran;
- rules for changing the contract.

A target external API may resemble:

```text
gpu_dyn_core_init(...)
gpu_dyn_core_step(...)
gpu_dyn_core_finalize(...)
```

but the architecture agent should derive the final signature from the actual source rather than forcing this exact form.

Commit the contract:

```bash
git add docs/gpu_port
git commit -m "gpu: define shared dyn_core GPU interface contract"
```

## 17A.4 Gate A — mandatory human review

Stop after Prompt 2.

Do not launch parallel translation agents until the following questions have acceptable answers:

1. Is the selected migration boundary truly the intended `dyn_core` subsystem?
2. Are all major fields and temporaries represented?
3. Is Fortran/C/C++ ownership unambiguous?
4. Can core state remain GPU-resident across the complete `dyn_core` call?
5. Are halo exchanges placed at explicit boundaries?
6. Have the agents avoided introducing one host/device copy per translated routine?
7. Are C-grid, D-grid, vertical, operators, and runtime work packages separable under the same contract?
8. Is the numerical validation policy sufficiently strict to locate the first divergence?

If the contract needs changes, make them on the supervisor branch and commit them before proceeding.

Tagging this point is useful:

```bash
git tag gpu-dyncore-gate-a
```

## 17A.5 Create parallel Git worktrees

From the main repository, create one worktree per implementation agent from the reviewed Gate A commit:

```bash
git worktree add ../fv3-cgrid     -b gpu/cgrid     gpu-dyncore-gate-a
git worktree add ../fv3-dgrid     -b gpu/dgrid     gpu-dyncore-gate-a
git worktree add ../fv3-vertical  -b gpu/vertical  gpu-dyncore-gate-a
git worktree add ../fv3-operators -b gpu/operators gpu-dyncore-gate-a
git worktree add ../fv3-runtime   -b gpu/runtime   gpu-dyncore-gate-a
```

The exact neighboring paths can be changed to match the filesystem, but each Codex session must operate in only its assigned worktree.

Check:

```bash
git worktree list
```

## 17A.6 Launch five implementation Codex sessions

Open a separate Codex session in each worktree.

Use this assignment:

| Worktree | Branch | Prompt | Responsibility |
|---|---|---|---|
| `../fv3-cgrid` | `gpu/cgrid` | 3A | C-grid acoustic dynamics |
| `../fv3-dgrid` | `gpu/dgrid` | 3B | D-grid acoustic dynamics |
| `../fv3-vertical` | `gpu/vertical` | 3C | vertical/nonhydrostatic dynamics |
| `../fv3-operators` | `gpu/operators` | 3D | pressure gradient, damping, related operators |
| `../fv3-runtime` | `gpu/runtime` | 3E | device state, runtime, ABI, communication infrastructure |

Before the specialized prompt, prepend the following common instruction to each Codex session:

```text
Before modifying code, read:

  docs/gpu_port/dyn_core_map.md
  docs/gpu_port/dyn_core_interface_contract.md
  docs/gpu_port/validation_policy.md

Treat dyn_core_interface_contract.md as authoritative.

Work only within the work package assigned in the prompt. Do not redesign shared
interfaces locally. If the contract is insufficient or incorrect, document the
required change in docs/gpu_port/agent_status/<work-package>.md and stop that
part of the implementation rather than silently changing the shared API.

Preserve the original Fortran implementation as the numerical reference.
Use pyFV3/GT4Py only as an independent implementation reference, not as a
replacement specification when it differs from the Fortran source.

Before finishing:
1. build all code that can be built in this environment;
2. run available package/unit tests;
3. document untested items and environment limitations;
4. update docs/gpu_port/agent_status/<work-package>.md;
5. commit all intended changes to the current branch;
6. report the commit SHA and files changed.
```

Then paste the corresponding Prompt 3A, 3B, 3C, 3D, or 3E from this plan.

## 17A.7 What each implementation agent must deliver

Each branch should contain four types of output:

```text
1. implementation code
2. local tests/harnesses where appropriate
3. an agent handoff/status file
4. one or more intentional Git commits
```

Recommended status files:

```text
docs/gpu_port/agent_status/cgrid.md
docs/gpu_port/agent_status/dgrid.md
docs/gpu_port/agent_status/vertical.md
docs/gpu_port/agent_status/operators.md
docs/gpu_port/agent_status/runtime.md
```

Each status file should contain:

```text
Scope completed
Source routines mapped
Files added/modified
Public/internal APIs implemented
Contract assumptions
Tests run and results
Numerical validation performed
Known deviations/TODOs
Dependencies on other work packages
Requested contract changes, if any
Final commit SHA
```

Do not judge agent completion only from a chat message; judge it from committed code plus the handoff file.

## 17A.8 Handling a requested contract change

If an implementation agent discovers that the shared contract must change:

```text
implementation agent
        |
        v
record proposal in agent_status/<name>.md
        |
        v
stop dependent interface work
        |
        v
human + supervisor review
        |
        v
update contract on supervisor branch
        |
        v
commit contract revision
        |
        v
rebase/cherry-pick revision into affected worktrees
```

Avoid allowing one specialist branch to become the de facto owner of the global architecture.

## 17A.9 Review the five branches before integration

When an implementation agent reports completion, inspect at least:

```bash
git status
git log --oneline --decorate -n 10
git diff gpu-dyncore-gate-a...HEAD --stat
git diff gpu-dyncore-gate-a...HEAD
```

Verify that the branch stayed within scope and did not modify unrelated GEOS/FV3 behavior.

Where feasible, require each worktree to build independently before integration.

## 17A.10 Create the integration worktree

Create a dedicated integration branch from the Gate A baseline:

```bash
git worktree add ../fv3-integration \
    -b gpu/dyn-core-integration \
    gpu-dyncore-gate-a
```

Open a new Codex session in `../fv3-integration` and paste **Prompt 4 — Integration Agent**.

Before Prompt 4, tell the integration agent exactly which branch/commit belongs to each work package. For example:

```text
Integrate these reviewed work-package branches:

  C-grid:     gpu/cgrid      <commit-sha>
  D-grid:     gpu/dgrid      <commit-sha>
  Vertical:   gpu/vertical   <commit-sha>
  Operators:  gpu/operators  <commit-sha>
  Runtime:    gpu/runtime    <commit-sha>

Read all files in docs/gpu_port/agent_status/ and the shared interface contract
before integrating. Resolve conflicts according to the shared contract; do not
silently choose one specialist's incompatible interface over another.
```

The integration agent can merge or cherry-pick reviewed commits as appropriate. Keep work-package history visible whenever practical.

## 17A.11 Integration target

The integration agent should produce one coherent GPU subsystem boundary rather than exposing every translated routine across the language boundary.

Conceptually:

```text
Fortran fv_dynamics
       |
       v
+-------------------------+
| GPU dyn_core interface  |
|                         |
| init                    |
| step                    |
| finalize                |
+------------+------------+
             |
             v
      persistent GPU state
             |
       +-----+------+----------+-----------+
       |            |          |           |
       v            v          v           v
     C-grid        D-grid    vertical   operators
```

Internal CUDA/C++ calls should stay behind this boundary as much as possible.

## 17A.12 Close integrated GPU execution, then run baseline capture and numerical validation

A successful build is not sufficient to begin numerical validation.

First run **Prompt 4C — Integrated GPU Execution Closure Agent** and require Gate
C1 to pass: the integrated `gpu_dyn_core_step` must actually execute WP-A,
WP-B, WP-C, and WP-D on a real GPU with persistent device residency.

Only after Gate C1 passes, use dedicated Codex/Claude Code sessions or the
integration session for:

```text
Prompt 6A — Baseline Capture Agent
Prompt 6B — Numerical Validation Agent
```

Prompt 6A should capture deterministic Fortran reference state at meaningful subsystem boundaries, not just final model output.

Useful checkpoints include states before and after major stages such as:

```text
C-grid dynamics
C-grid pressure/vertical update
D-grid dynamics
vertical solver
nonhydrostatic pressure gradient
end of acoustic substep
end of dyn_core
```

Prompt 6B should compare the GPU path against those checkpoints and identify the earliest divergence.

The debugging rule is:

```text
fix the first divergence, not the largest downstream error
```

Gate C2 passes only when the required subsystem savepoints agree with the
authoritative Fortran baseline within the accepted validation policy.

## 17A.13 Connect to GEOS only after subsystem validation

Once the integrated `dyn_core` subsystem passes Gate C1 and Gate C2, run
**Prompt 5 — GEOS/FV3 Interface Agent**.

If Prompt 5 interface scaffolding was implemented earlier, do not duplicate it.
Instead, audit and re-verify the existing Fortran/ISO_C_BINDING/C-ABI/WP-E bridge
against the validated subsystem and complete any missing production integration.

Recommended progression:

```text
1. Fortran baseline backend remains selectable
2. GPU backend compiles into FV3/GEOS
3. initialization/finalization work
4. one dyn_core call works
5. short single-rank run works
6. longer single-rank regression works
7. multi-rank halo path works
8. multi-rank regression works
```

Do not combine initial numerical debugging with MAPL/ESMF, MPI, and GPU-runtime debugging if it can be avoided.

## 17A.14 Run performance analysis last

After numerical correctness and multi-rank execution are established, paste **Prompt 7 — Performance Agent**.

Require performance reporting to distinguish at least:

```text
GPU kernel time
halo/MPI time
host -> device transfer time
device -> host transfer time
synchronization time
Fortran/C/C++ interface overhead
total dyn_core wall time
```

The major architectural metric is not simply kernel speedup. It is whether the migration maintains useful **GPU residency across the whole `dyn_core` subsystem**.

## 17A.15 Suggested daily operating rhythm

For a human-supervised first implementation cycle:

```text
Start of session
  git status / branch / HEAD
  read outstanding agent_status files
  identify approved tasks

During session
  one Codex session per worktree
  no cross-worktree editing
  agents commit completed units
  architecture changes go through supervisor review

End of session
  ensure every active branch is committed
  update agent_status files
  push branches to remote
  record unresolved issues
  verify integration branch state
```

Pushing active branches at the end of a work period is strongly recommended so that each agent's work and the architecture artifacts are recoverable and can be resumed from an exact commit.

## 17A.16 Resuming work in a later Codex session

Do not depend on the previous Codex conversation being available. Give the new session a repository-grounded resume prompt such as:

```text
Resume the GEOS/FV3 dyn_core GPU modernization work in this branch.

First inspect:
  git status
  git log --oneline -n 10
  docs/gpu_port/dyn_core_map.md
  docs/gpu_port/dyn_core_interface_contract.md
  docs/gpu_port/validation_policy.md
  docs/gpu_port/agent_status/<work-package>.md

Summarize the current branch status and the next incomplete task before making
changes. Continue from committed repository state and preserve the shared
interface contract.
```

This makes the Git repository, rather than an individual chat transcript, the persistent memory of the multi-agent project.

## 17A.17 When to automate with LangGraph

Do not automate the workflow merely because multiple agents are being used. The first manual Codex/worktree cycle should establish which dependencies and review gates are real.

Introduce LangGraph after you have evidence that the following sequence is stable:

```text
map
 -> architecture contract
 -> parallel work packages
 -> package tests
 -> integration
 -> numerical validation
 -> GEOS integration
 -> profiling
```

At that point LangGraph can encode state transitions, artifact checks, retry/debug routing, and approval gates that have already been proven manually.

---

# 18. Definition of Success

The first GEOS FV3 modernization phase is successful when all of the following are true:

1. The original Fortran `dyn_core` remains available as the baseline backend.
2. A C++/CUDA `dyn_core` backend can be selected without changing the surrounding GEOS workflow.
3. The GPU backend executes the major coupled acoustic-dynamics calculation as one subsystem.
4. Core state remains device-resident throughout the GPU `dyn_core` call.
5. Single-rank numerical regression passes.
6. Multi-rank regression passes.
7. Performance measurements separate computation, communication, and transfer overhead.
8. The workflow demonstrates that multiple AI agents can safely modernize a large scientific subsystem in parallel under a shared architecture contract.

This is a substantially stronger result than proving that individual Fortran routines can be translated to CUDA.

---

# 19. Next Expansion After `dyn_core`

After `dyn_core` is validated, expand the GPU-resident region outward rather than restarting with isolated routines.

Recommended sequence:

```text
Phase 1: dyn_core acoustic dynamics

Phase 2: tracer transport
         + finite-volume transport

Phase 3: Lagrangian-to-Eulerian remapping

Phase 4: complete fv_dynamics GPU-resident region

Phase 5: optimize GEOS ↔ dynamics data residency

Phase 6: evaluate additional GEOS atmospheric physics modules
```

The long-term target becomes:

```text
GEOS / MAPL / ESMF
       |
       v
Fortran orchestration
       |
       v
GPU-resident FV3 dynamics
       |
       +-- acoustic dynamics
       +-- transport
       +-- remapping
       +-- halo communication
       |
       v
physics / coupling / I/O
```

---

# 20. Reference Codebases

Primary Fortran reference:

```text
NOAA-GFDL/GFDL_atmos_cubed_sphere
    model/fv_dynamics.F90
    model/dyn_core.F90
    model/sw_core.F90
    model/nh_core.F90
```

GEOS integration reference:

```text
GEOS-ESM/FVdycoreCubed_GridComp
    geos-gtfv3/
```

Independent GPU/DSL reference:

```text
NOAA-GFDL/pyFV3
    pyfv3/stencils/fv_dynamics.py
    pyfv3/stencils/dyn_core.py
    pyfv3/wrappers/geos_wrapper.py
```

These should be version-pinned in the project artifacts before implementation begins.
