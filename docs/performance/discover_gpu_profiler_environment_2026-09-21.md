# Discover GPU Profiler Environment Discovery

Date: 2026-09-21
Status: **ENVIRONMENT DISCOVERY ONLY** — no source changed, no GPU job
submitted, no Nsight run against the FV3 executable.

All commands below were run on a Discover login node, per
`docs/agent_instructions/discover_platform.md` (Claude Code must remain on a
login node; GPU-required execution goes through Slurm, not exercised here).

---

## 1. Validated FV3/CUDA environment (extracted from project docs + existing scripts)

Per `docs/agent_handoff/CURRENT_STATE.md` §2, the MPI implementation used by
every C256/C512 production job to date is **Intel MPI 2021.13**, not the
OpenMPI dev stack in `docs/agent_instructions/discover_platform.md` (that
stack — GCC + OpenMPI 4.1.6 — is for standalone CUDA-only dev work, build dir
`build_cuda_r8_mpi`; the Intel-MPI-matched CUDA build is
`build_cuda_r8_intelmpi`/`build_cuda_r4_intelmpi`).

The exact validated Intel-MPI module sequence was recovered from the existing
project script `cuda_intelmpi_build.sh` (repo root, untracked but load-bearing
project tooling):

```bash
module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv
module load comp/gcc/12.3.0
module load comp/intel/2024.2.0
module load mpi/impi/2021.13
module load nvidia/nvhpc-byo-compiler/24.7
export I_MPI_CXX=g++
```

This was reproduced exactly (Stage 0) with no modification.

---

## 2. Stage 0 — Recorded current environment

```
Currently Loaded Modules:
  1) git/2.49.0             8) other/tree/2.1.0
  2) cmake/4.4.3            9) other/ninja/1.12.0
  3) other/mepo/2.4.0      10) GEOSenv
  4) other/tig/2.5.8       11) comp/gcc/12.3.0
  5) ImageMagick/7.1.1-15  12) comp/intel/2024.2.0
  6) git-lfs/3.7.1         13) mpi/impi/2021.13
  7) other/gh              14) nvidia/nvhpc-byo-compiler/24.7
```

| Tool | Path | Version |
|---|---|---|
| gcc | `/usr/local/other/gcc/12.3.0/bin/gcc` | GCC 12.3.0 |
| icx/ifort/ifx | `/usr/local/intel/oneapi/2024/compiler/2024.2/bin/{icx,ifort,ifx}` | present, Intel oneAPI 2024.2 |
| mpirun | `/usr/local/intel/oneapi/2024/mpi/2021.13/bin/mpirun` | Intel(R) MPI Library 2021.13, Build 20240515 |
| nvcc | `/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/bin/nvcc` | Cuda compilation tools, release 12.5, V12.5.82 |

Relevant `LD_LIBRARY_PATH` entries:
```
/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/extras/CUPTI/lib64
/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/lib64
/usr/local/intel/oneapi/2024/mpi/2021.13/libfabric/lib
/usr/local/intel/oneapi/2024/mpi/2021.13/lib/release
/usr/local/intel/oneapi/2024/mpi/2021.13/lib
/usr/local/intel/oneapi/2024/compiler/2024.2/lib
/usr/local/intel/oneapi/2024/compiler/2024.2/opt/compiler/lib
```
Note: CUPTI (the library nsys/ncu instrument through) is already on
`LD_LIBRARY_PATH` as a side effect of `nvidia/nvhpc-byo-compiler/24.7` — no
separate CUPTI module is needed.

No build was performed in this stage.

---

## 3. Stage 1 — Direct profiler availability (before any PATH change)

With only the validated stack loaded, none of `nsys`/`ncu`/`nvprof`/
`compute-sanitizer` are on `PATH`:

```
which nsys              -> not found
which ncu               -> not found
which nvprof            -> not found
which compute-sanitizer -> not found
```

However, the `nvidia/nvhpc-byo-compiler/24.7` module (already required and
already loaded by the validated stack) installs the full NVHPC SDK tree,
which **bundles Nsight Systems and Nsight Compute** — they are simply not
added to `PATH` by that module:

```
/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/profilers/Nsight_Systems/bin/nsys
/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/profilers/Nsight_Compute/ncu
/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer
/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/bin/nvprof   (legacy, deprecated for Ampere+ metrics)
```

Both ran successfully by absolute path with no modules loaded at all:

```
$ nsys --version
NVIDIA Nsight Systems version 2024.4.1.61-244134315967v0

$ nsys status -e
... CPU Profiling Environment (process-tree): OK
... CPU Profiling Environment (system-wide): Fail
(login-node CPU-side check only — GPU capability not exercised here)

$ ncu --version
NVIDIA (R) Nsight Compute Command Line Profiler
Version 2024.2.1.0 (build 34372528) (public-release)
```

---

## 4. Stage 2 — Module system search

```
$ module avail 2>&1 | grep -Ei 'nsight|profiler|cuda|nvhpc'
   cuDNN/8.9.7.29_cuda12        tview/2021.2.14
   cuDNN/9.1.1.17_cuda12 (D)    tview/2021.4.10
   nvidia/nvhpc-byo-compiler/23.9    (D)
   nvidia/nvhpc-byo-compiler/24.7          <- validated stack uses this one
   nvidia/nvhpc-hpcx-cuda11/24.7
   nvidia/nvhpc-hpcx-cuda12/23.9     (D)
   nvidia/nvhpc-hpcx-cuda12/24.7
   nvidia/nvhpc-hpcx/23.9            (D)
   nvidia/nvhpc-hpcx/24.7
   nvidia/nvhpc-nompi/23.9           (D)
   nvidia/nvhpc-nompi/24.7
   nvidia/nvhpc-openmpi3/24.7
   nvidia/nvhpc/23.9                 (D)
   nvidia/nvhpc/24.7

$ module spider nsys   -> Unable to find: "nsys"
$ module spider ncu    -> Unable to find: "ncu"
$ module spider nsight -> Unable to find: "nsight"
```

**Conclusion**: there is no dedicated Nsight Systems / Nsight Compute
module on Discover. Every `nvidia/nvhpc-*` module variant bundles the same
profilers under its own SDK tree (`<module_root>/profilers/Nsight_Systems`,
`<module_root>/profilers/Nsight_Compute`) — this is a property of the NVHPC
SDK packaging, not a separate installable tool. Since the validated stack
already loads `nvidia/nvhpc-byo-compiler/24.7` (required for `nvcc`), no
*additional* module load is needed to obtain the profiler binaries — only a
`PATH` addition.

---

## 5. Stage 3 — Software-tree search

Not required — Stage 1/2 already located both tools unambiguously inside the
already-loaded module's own install tree
(`/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/profilers/`). No broader
filesystem search was performed.

---

## 6. Stage 4 — Compatibility test (profiler PATH addition on top of validated stack)

Reproduced the full validated Intel-MPI stack, then added only the two
profiler `bin` directories to `PATH` (no additional `module load`):

```bash
NVHPC_ROOT=/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7
export PATH=$NVHPC_ROOT/profilers/Nsight_Systems/bin:$NVHPC_ROOT/profilers/Nsight_Compute:$PATH
```

Before/after `module list` is **identical** (14 modules, unchanged) —
confirming this is a pure `PATH` addition, not a module load, so it cannot
alter module-resolved dependencies.

| Check | Before PATH addition | After PATH addition |
|---|---|---|
| `module list` | 14 modules incl. `mpi/impi/2021.13`, `comp/gcc/12.3.0`, `nvidia/nvhpc-byo-compiler/24.7` | **identical, unchanged** |
| `mpirun` | Intel MPI 2021.13 | Intel MPI 2021.13 (unchanged) |
| `gcc` | GCC 12.3.0 | GCC 12.3.0 (unchanged) |
| `nvcc` | CUDA 12.5.82 | CUDA 12.5.82 (unchanged) |
| `nsys` | not found | found, v2024.4.1.61 |
| `ncu` | not found | found, v2024.2.1.0 |

**Intel MPI 2021.13 confirmed preserved.** No compiler-family, CUDA-runtime,
or GEOS-environment component changed.

---

## 7. Stage 5 — Classification table

| Tool | Module/setup | Version | Intel MPI preserved | FV3 env compatible | Status |
|---|---|---|---|---|---|
| `nsys` | No module — `PATH` addition inside already-loaded `nvidia/nvhpc-byo-compiler/24.7` tree | 2024.4.1.61 | YES | YES | **SAFE** — login-node checks (`--version`, `status -e`) pass; GPU-side capture not yet exercised (that requires a Slurm GPU allocation, not run in this task) |
| `ncu` | Same — `PATH` addition, no module | 2024.2.1.0 | YES | YES (executable/version-check level) | **PROBABLY SAFE, with a known pre-existing platform block** — see §8. The binary loads and reports its version cleanly, but every prior attempt to actually attach `ncu` to a running GPU kernel on this Discover A100 allocation (Prompts 12, 13, 24, 26) failed with `ERR_NVGPUCTRPERM` (GPU performance-counter access denied at the driver/admin level). This was not re-tested in this task (no GPU job permitted) — documented as a carried-forward, unresolved platform limitation, not new evidence. |
| `compute-sanitizer` | Same tree, `cuda/12.5/compute-sanitizer` | matches CUDA 12.5 toolkit | YES | YES | **SAFE** — already used successfully in this project (`CURRENT_STATE.md`: "compute-sanitizer: PASS with 0 reported errors" during initial bring-up) |
| `nvprof` | Same tree, `cuda/bin/nvprof` | legacy/deprecated | YES | N/A | **UNAVAILABLE (functionally)** — NVIDIA has deprecated `nvprof` metric/timeline collection for Ampere (A100) and newer GPUs in favor of `nsys`/`ncu`; not recommended even though the binary exists |
| TAU / HPCToolkit / Score-P | — | — | — | — | **UNAVAILABLE** — `module avail` search found none of these on Discover |

---

## 8. Stage 6 — Fallback / capability classification

Both required capability classes are already covered by tools confirmed
present, with one caveat:

- **System/timeline critical-path analysis** → **Nsight Systems (`nsys`)** —
  SAFE, no known platform restriction (`CURRENT_STATE.md` explicitly:
  "`nsys` unaffected" by the `ncu` block). This is the tool for the roadmap's
  Part 3 Phase A (same-node C512 critical-path capture).
- **Detailed GPU kernel/microarchitecture analysis** → **Nsight Compute
  (`ncu`)** is the correct tool in principle, but is **platform-blocked** on
  this Discover `gpu_a100` allocation via `ERR_NVGPUCTRPERM` — a driver-level
  GPU performance-counter permission restriction, confirmed repeatedly
  (Prompts 12/13/24/26) and unchanged as of the last check. No unprivileged
  workaround exists; fixing it would require an NCCS administrative change
  (opening `NVreg_RestrictProfilingToAdminUsers` to non-admin users, or a
  `setcap`/suid `ncu` install) — out of scope for this project to pursue
  directly.
  - **Fallback if the block is still in effect**: `compute-sanitizer`
    (confirmed working, but diagnostic/correctness-only — no
    occupancy/register/throughput metrics), or source-level inference from
    kernel launch configuration + register-count from `nvcc
    --resource-usage`/`-Xptxas -v` at compile time (static register/shared-
    memory counts, no dynamic occupancy/stall data). Neither is a substitute
    for real Nsight Compute counters; both were already the fallback used in
    Prompt 12's "SOURCE-READING INFERENCE ONLY (not ncu-confirmed)"
    classification.
  - This directly affects the roadmap's Part 3 Phase B and Part 10 items
    2/4 (`docs/performance/cuda_fv3_optimization_roadmap_2026-09-21.md`) —
    those steps should proceed but their output must be labeled
    "attempted, pending platform permission" unless a GPU-node smoke test
    (Stage 8) shows the restriction has been lifted.

---

## 9. Stage 7 — Recommended profiling environment for Prompt 49

Exact module/PATH sequence, derived from Stage 4's confirmed-safe test — no
speculative steps:

```bash
module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv
module load comp/gcc/12.3.0
module load comp/intel/2024.2.0
module load mpi/impi/2021.13
module load nvidia/nvhpc-byo-compiler/24.7
export I_MPI_CXX=g++

NVHPC_ROOT=/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7
export PATH=$NVHPC_ROOT/profilers/Nsight_Systems/bin:$NVHPC_ROOT/profilers/Nsight_Compute:$PATH

which nsys     # -> $NVHPC_ROOT/profilers/Nsight_Systems/bin/nsys
which ncu      # -> $NVHPC_ROOT/profilers/Nsight_Compute/ncu
which mpirun   # -> /usr/local/intel/oneapi/2024/mpi/2021.13/bin/mpirun (unchanged)
```

This is identical to the existing validated build/runtime stack
(`cuda_intelmpi_build.sh`) plus two `PATH` entries — no module substitution,
no new module, no MPI/compiler/CUDA-runtime change.

For an actual GPU-node capture (not run in this task — requires Slurm), wrap
the same block inside the standard `srun`/`sbatch` pattern from
`docs/agent_instructions/discover_platform.md`, establishing the module
environment inside the compute-node shell before invoking `nsys profile ...`
around the target C512 dyn_core run.

---

## 10. Stage 8 — Minimal non-FV3 tool check (login node only, no GPU job)

Performed, GPU-independent checks only:

```
$ nsys --version
NVIDIA Nsight Systems version 2024.4.1.61-244134315967v0

$ nsys status -e
CPU Profiling Environment (process-tree): OK
CPU Profiling Environment (system-wide): Fail   <- expected without root; irrelevant to GPU kernel-trace mode
(no GPU-side status reported — a login node has no visible GPU; this check is CPU-only)

$ ncu --version
NVIDIA (R) Nsight Compute Command Line Profiler
Version 2024.2.1.0 (build 34372528) (public-release)
```

**GPU-node execution requirement documented, not performed**: confirming
`nsys` GPU-kernel capture actually works end-to-end, and re-confirming
whether the `ncu`/`ERR_NVGPUCTRPERM` block still applies, both require a real
Slurm GPU allocation (`gpu_a100`/`rome`, per
`docs/agent_instructions/discover_platform.md`). This task did not submit
that job, per its explicit scope restriction. That smoke test is the natural
first action for Prompt 49's Phase 0 (`docs/performance/
cuda_fv3_optimization_roadmap_2026-09-21.md` Part 5).

---

## 11. Known caveats

- `ncu` GPU-kernel profiling is platform-blocked on this Discover A100
  allocation (`ERR_NVGPUCTRPERM`), confirmed repeatedly in project history
  (Prompts 12/13/24/26) and not re-tested here (no GPU job permitted this
  task). Treat Part 3 Phase B of the optimization roadmap as
  permission-gated until a fresh GPU-node smoke test either reconfirms or
  lifts the block.
- `nsys status -e`'s "system-wide" CPU profiling check reports `Fail` on the
  login node — expected without root and orthogonal to GPU kernel tracing;
  do not read this as a GPU-side failure.
- The profiler binaries live inside the `nvidia/nvhpc-byo-compiler/24.7`
  module's own SDK tree, not a separate module — if that module version is
  ever bumped (e.g. to a newer NVHPC release) for compiler/CUDA reasons, the
  `NVHPC_ROOT` path above must be updated to match, and the `ncu`/`nsys`
  versions will change accordingly (they track the NVHPC SDK release, not an
  independently versioned tool).
- No dedicated Nsight module exists on Discover (`module spider nsys/ncu/
  nsight` all report "Unable to find") — this is expected NVHPC packaging
  behavior, not a gap in the module system.
- This entire discovery was performed without building or running the FV3
  executable; the "FV3 env compatible" classifications in §7 are based on
  module/environment-variable non-interference only, not an actual build/run
  with the profiler PATH active. A GPU-node smoke test (Stage 8, deferred)
  is the first thing that should exercise the combination against a live
  process.
