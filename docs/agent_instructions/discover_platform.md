# NASA NCCS Discover Platform Instructions

## Purpose

This file defines stable platform-specific instructions for developing,
building, and validating the GEOS FV3 CUDA modernization project on NASA NCCS
Discover.

These instructions should be treated as persistent project context.

Do not duplicate phase-specific task instructions here.

---

## Platform

Primary platform:

NASA NCCS Discover

Primary repository path:

`/discover/nobackup/projects/imvi-nse/jli30/FVdycoreCubed_GridComp`

Primary GPU target:

Discover `gpu_a100` partition

Current GPU architecture used for development/validation:

NVIDIA A100

Current Slurm constraint:

`rome`

---

## Claude Code Network Constraint

Claude Code MUST remain on a Discover login node.

Claude API access is not available from `gpu_a100` compute nodes.

This is expected and is NOT a blocker.

Do not attempt to start, resume, or move the Claude Code process itself onto a
GPU compute node.

The repository and build directories are on the shared Discover filesystem.

Therefore Claude should:

- inspect source on the login node;
- edit source on the login node;
- perform Git operations on the login node;
- configure CMake builds on the login node;
- compile with host compilers / nvcc on the login node when allowed;
- submit GPU-required execution through Slurm;
- inspect GPU job output from the login node.

A lack of `nvidia-smi` access on the login node is NOT a GPU failure.

GPU availability must be tested inside a Slurm GPU job.

---

## Discover Module Search Path

Before loading the validated GEOS/CUDA toolchain, add:

```bash
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
```

Because `module purge` may remove module-path changes, use the following order:

```bash
module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
```

Do not reverse this order.

---

## Validated Discover CUDA Development Environment

Use the following validated module stack for standalone CUDA development and
Prompt 5B-style CUDA bring-up:

```bash
module purge

module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15

module load GEOSenv
module load comp/gcc/12.3.0
module load mpi/openmpi/4.1.6/gcc-12.3.0
module load nvidia/nvhpc-byo-compiler/24.7
```

Expected toolchain:

- GCC: 12.3.0
- G++: 12.3.0
- GFortran: 12.3.0
- CUDA/nvcc: 12.5.x
- OpenMPI: 4.1.6
- CMake: 3.30.3

The validated CUDA toolkit version during initial bring-up was:

`nvcc V12.5.82`

---

## Why the BYO CUDA Module Is Used

Use:

`nvidia/nvhpc-byo-compiler/24.7`

rather than the full NVHPC compiler modules for the current CUDA development
workflow.

This module provides CUDA/nvcc and CUDA libraries without forcing:

- `nvc`
- `nvc++`
- `nvfortran`

as the host compiler stack.

It does not intentionally replace the selected external GCC/GFortran toolchain.

This allows the project to use:

```text
GCC/GFortran
+
OpenMPI
+
CUDA/nvcc
```

without rebuilding the broader GEOS dependency stack using NVIDIA compilers.

Do not switch to full modules such as:

- `nvidia/nvhpc-nompi/...`
- `nvidia/nvhpc-hpcx/...`
- `nvidia/nvhpc-openmpi3/...`

unless a specific task demonstrates that the validated BYO environment is
insufficient.

Do not change compiler families merely to work around a CUDA source error.

---

## Environment Verification

After loading the validated stack, verify:

```bash
module list

which gcc
gcc --version

which g++
g++ --version

which gfortran
gfortran --version

which nvcc
nvcc --version

which mpicc
mpicc --showme 2>/dev/null || mpicc -show

which mpifort
mpifort --showme 2>/dev/null || mpifort -show

which cmake
cmake --version
```

Do not assume the toolchain is correct simply because modules loaded without an
error.

---

## GPU Execution Model

Claude remains on the login node.

Any command requiring an actual GPU must execute through Slurm.

Examples include:

- `nvidia-smi`
- CUDA runtime initialization
- `cudaMalloc`
- CUDA kernel execution
- GPU runtime tests
- integrated CUDA smoke tests
- `compute-sanitizer`
- device-memory diagnostics
- later GPU numerical validation

Do not run GPU executables directly on login nodes.

---

## Preferred Short GPU Execution Pattern

For short GPU tests, prefer a one-shot `srun`.

Typical pattern:

```bash
srun \
  --partition=gpu_a100 \
  --constraint=rome \
  --gres=gpu:1 \
  --mem=60G \
  --time=00:10:00 \
  --ntasks=1 \
  --cpus-per-task=12 \
  bash -lc '
    module purge
    module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
    module load GEOSenv
    module load comp/gcc/12.3.0
    module load mpi/openmpi/4.1.6/gcc-12.3.0
    module load nvidia/nvhpc-byo-compiler/24.7

    cd /discover/nobackup/projects/imvi-nse/jli30/FVdycoreCubed_GridComp

    <GPU_COMMAND>
  '
```

Always establish the module environment inside the compute-node shell.

Do not assume the login-node module state will automatically be reproduced
correctly inside the Slurm execution context.

---

## Preferred GPU Probe

A minimal GPU availability probe can use:

```bash
srun \
  --partition=gpu_a100 \
  --constraint=rome \
  --gres=gpu:1 \
  --mem=60G \
  --time=00:05:00 \
  --ntasks=1 \
  --cpus-per-task=12 \
  bash -lc '
    module purge
    module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
    module load GEOSenv
    module load comp/gcc/12.3.0
    module load mpi/openmpi/4.1.6/gcc-12.3.0
    module load nvidia/nvhpc-byo-compiler/24.7

    echo "HOST=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    which nvcc
    nvcc --version

    nvidia-smi
  '
```

A successful probe establishes:

- Slurm GPU access;
- A100 visibility;
- correct CUDA environment on the compute node.

---

## Longer GPU Runs

Use `sbatch` instead of synchronous `srun` for work such as:

- longer `compute-sanitizer` runs;
- integrated validation;
- tests generating substantial output;
- tests likely to exceed a short interactive window.

Batch jobs should:

- request one GPU unless the task explicitly requires more;
- use `gpu_a100`;
- use `--constraint=rome`;
- load the validated module environment;
- work from the shared repository/build path;
- write logs to clearly named files;
- avoid excessive resource requests.

Claude may inspect jobs using:

```bash
squeue -u $USER
```

and completed jobs using:

```bash
sacct
```

Do not aggressively busy-loop while waiting for queued jobs.

---

## Scheduler Guidelines

For current single-GPU development:

- partition: `gpu_a100`
- constraint: `rome`
- GPU count: `1`
- memory: approximately `60G`
- CPUs: approximately `12`
- one task

Typical interactive allocation, when a human explicitly needs one:

```bash
salloc \
  --gres=gpu:1 \
  --mem=60G \
  --time=2:00:00 \
  --partition=gpu_a100 \
  --constraint=rome \
  --ntasks-per-node=1 \
  --cpus-per-task=12
```

Claude should generally prefer targeted `srun` or `sbatch` commands rather than
holding an interactive `salloc` allocation.

Do not request multiple GPUs during ordinary CUDA bring-up or single-GPU
validation.

---

## Shared Filesystem Assumption

Source and build artifacts under:

`/discover/nobackup/projects/imvi-nse/jli30/`

are available to both login and compute nodes.

Therefore the normal workflow is:

```text
login node
  -> edit
  -> configure
  -> compile
  -> submit GPU executable

compute node
  -> execute binary
  -> write output

login node
  -> inspect output
  -> modify source if necessary
```

Do not copy the repository to temporary GPU-local storage unless a specific
performance or filesystem requirement later makes that necessary.

---

## Standalone FVdycore Repository

The current standalone CUDA development repository may not contain:

`@fvdycore`

under:

`FVdycoreCubed_GridComp/@fvdycore`

This is acceptable for standalone CUDA build/runtime work if no target depends
on the authoritative GFDL source checkout.

Do not populate `@fvdycore` automatically.

If a task encounters an actual dependency on it:

1. identify the exact target;
2. identify the exact missing source or include;
3. determine whether the dependency is truly required for the current phase;
4. document it before cloning or changing repository structure.

The authoritative GFDL source will be required for later baseline numerical
validation and full FV3/GEOS integration.

---

## Full GEOS Compiler Environment

The current validated Prompt 5B stack is GCC-based.

It is suitable for standalone CUDA development and is compatible with a
GCC-based GEOS build environment.

Production GEOS may also use an Intel compiler/Intel MPI configuration.

Do not assume successful standalone GCC/CUDA bring-up automatically proves ABI
compatibility with an Intel production GEOS build.

Compiler-family/ABI decisions must be revisited explicitly before final
production GEOS linkage if the target GEOS executable uses Intel Fortran.

---

## Build Directory Policy

Use out-of-source build directories.

Typical examples:

```text
build_cuda_r8/
build_cuda_r4/
```

Do not commit:

- build directories;
- object files;
- executables;
- generated CMake files;
- Slurm logs;
- sanitizer logs;
- compiler scratch files.

Keep source/build fixes separate from generated artifacts.

---

## Platform Validation Terminology

Always distinguish:

- environment configured;
- CUDA compiled;
- GPU executed;
- memory-sanitized;
- numerically validated.

For example:

```text
CUDA compilation: PASS
```

does NOT imply:

```text
GPU execution: PASS
```

Likewise:

```text
GPU execution: PASS
```

does NOT imply:

```text
numerical validation: PASS
```

---

## Current Known Platform Bring-Up Result

The initial Discover CUDA bring-up established:

- Discover module stack: PASS
- GPU Slurm access: PASS
- A100 node used: `warpa003`
- CUDA toolkit: 12.5 / V12.5.82
- GCC/G++/GFortran: 12.3.0
- OpenMPI: 4.1.6
- R8 CUDA compilation: PASS
- R4 CUDA compilation: PASS
- R8 CPU-visible regression tests: 10/10 PASS
- R4 CPU-visible regression tests: 10/10 PASS
- WP-E real GPU runtime: PASS
- compute-sanitizer: PASS with 0 reported errors

Integrated full dyn_core device execution and authoritative numerical validation
are separate milestones and must not be inferred from these results.
