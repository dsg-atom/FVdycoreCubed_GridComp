# Prompt 41: C512 CUDA-Aware MPI A/B Test

Branch: `experiment/held-suarez-e2e`. Model: Sonnet only, per task policy.
Scope: determine whether the validated Discover MPI stack supports
CUDA-aware/device-pointer transport before attempting any A/B transport
experiment for the C512 CUDA performance regression. No source
modification, no GPU job, no correctness test, no performance run —
Stage 0 returned a clear STOP before any of those stages were reached.

## Stage 0: CUDA-Aware MPI Build Support — Verified, Not Assumed

Loaded the exact validated stack (`docs/agent_instructions/
discover_platform.md`): `GEOSenv` + `comp/gcc/12.3.0` +
`mpi/openmpi/4.1.6/gcc-12.3.0` + `nvidia/nvhpc-byo-compiler/24.7`. Queried
`ompi_info` directly (login node; static build-info query, no GPU needed):

- **`opal_built_with_cuda_support`**: `"false"` (and its synonym
  `mpi_built_with_cuda_support`: `"false"`). This is a compile-time flag
  baked into the library, not a runtime-tunable MCA parameter — it
  reflects whether OpenMPI was configured `--with-cuda` at build time.
- **Configure command line** (`ompi_info --all`): `--disable-wrapper-
  rpath --disable-wrapper-runpath --with-slurm --with-hwloc=internal
  --with-libevent=internal --with-pmix=internal --without-verbs
  CC=gcc CXX=g++ FC=gfortran`. **No `--with-cuda` flag anywhere.** Built
  2023-12-06 by `mathomp4`.
- Consistent with Prompt 40's independent finding: available BTLs are
  `vader`/`tcp`/`self` only — no `smcuda` (the CUDA-aware vader variant)
  in this build.
- `pml ucx`'s `pml_ucx_tls` parameter lists `cuda_ipc`/`rocm_ipc` as
  *valid string values* the UCX PML component could accept — this is a
  static list of everything the UCX PML component knows the *name* of,
  not evidence that this OpenMPI's UCX PML was itself built against CUDA,
  and moot regardless since Prompt 40 confirmed `ob1`+`vader` (not `ucx`)
  is what actually gets selected for this same-node six-rank job.

**CUDA-aware MPI build support: NO** (confirmed via direct `ompi_info`
query, not inferred).
**Device-buffer P2P supported: NO** — passing a device pointer to
`MPI_Isend`/`MPI_Irecv` on this build is unsupported; OpenMPI would treat
it as ordinary host memory and either segfault (host `memcpy` against
un-mapped device memory) or silently corrupt data. Not attempted.

This is a **build-time property of the linked OpenMPI library**, not
something a runtime MCA variable, environment variable, or source-level
workaround in `fv3_gpu_halo_mpi.cpp` can change. Per the task's Hard Stop
Condition 1 ("OpenMPI build clearly lacks CUDA-aware support"), STOP here.

## Stage 0b: Cheapest Environment-Level Check — Already-Available CUDA-Aware Build

Per the task's Case-E instruction ("recommend the cheapest environment-
level next experiment: use a CUDA-aware MPI build if one is already
available on Discover. Do not rebuild OpenMPI in this prompt"), checked
`module avail` for an existing CUDA-aware MPI module rather than assuming
none exists:

```
module avail 2>&1 | grep -iE "openmpi|hpcx|mvapich|mpi"
```

Found `nvidia/nvhpc-hpcx-cuda12/24.7` (bundled Mellanox HPC-X MPI) and
`nvidia/nvhpc-openmpi3/24.7` already installed on Discover — both are
full NVHPC-bundled modules `docs/agent_instructions/discover_platform.md`
explicitly warns not to switch to "unless a specific task demonstrates
that the validated BYO environment is insufficient" (this Stage 0 result
is exactly that demonstration, for the CUDA-aware-MPI question
specifically — it does not by itself justify switching the whole
validated toolchain).

Loaded `nvidia/nvhpc-hpcx-cuda12/24.7` in isolation (no other module
state; login-node `ompi_info` query only, no compile/link/GPU job) to
confirm it is genuinely CUDA-aware rather than assuming from the name:

- **Ident string**: `4.1.7a1` (HPC-X's own OpenMPI build, distinct from
  the validated `mpi/openmpi/4.1.6/gcc-12.3.0` module).
- **`opal_built_with_cuda_support`**: `"true"`.

**A CUDA-aware MPI build (`nvidia/nvhpc-hpcx-cuda12/24.7`) is confirmed
already present on Discover.** It was NOT adopted in this prompt — doing
so would mean relinking the entire CUDA dyn_core + standalone Fortran
fixture against a different MPI implementation (HPC-X, not the validated
OpenMPI 4.1.6), which is a toolchain-level change out of this prompt's
2-substantial-change budget and squarely the kind of compiler/ABI
decision `discover_platform.md`/`project_architecture.md` require
revisiting explicitly (Intel-compiler production-ABI compatibility,
host-compiler family) before adopting. This is recorded as the concrete
next environment-level step, not attempted here.

## Stages 1-6: Not Reached

Stage 0 is an explicit Hard Stop condition. Per the task's own rule ("Do
not proceed with implementation if support is clearly absent"), no
further stage was attempted:

- No search for/reuse of an existing CUDA-aware transport implementation
  in `cuda/dyn_core/` (would be moot — nothing to run it against on this
  MPI build). Grep confirms none exists (`fv3_gpu_halo_mpi.hpp`/`.cpp`
  implement only the host-staged path documented as "Stage 1" of
  `project_architecture.md`'s Communication Evolution; the file's own
  header comment explicitly names CUDA-aware MPI as unimplemented
  "Stage 2... a later step").
- No source modification. Substantial changes used: **0/2**.
- No transport correctness test run (nothing new to test).
- No standalone C512 correctness gate run.
- No same-node paired A/B performance job submitted. **GPU jobs
  submitted: 0.**
- No profiling.

`fv3_gpu_halo_transport.hpp`/`fv3_gpu_halo_mpi.{hpp,cpp}` were read in
full to confirm the `HaloTransport` abstraction already anticipates a
second backend cleanly (the interface is backend-agnostic; adding a
`cuda_aware` implementation later would not require touching
`PeerHaloTransport`, the pack/unpack kernels, or any numerical code) —
this is recorded so a future prompt does not have to re-derive it, but no
code was added.

## Decision

Per the task's Case E: **CUDA-AWARE MPI BLOCKED BY MPI BUILD.**

The validated Discover OpenMPI 4.1.6/GCC-12.3.0 module used throughout
this project's CUDA bring-up was not built with CUDA GPU buffer support
(`--with-cuda` absent from its configure line; `opal_built_with_cuda_
support=false`). This is a hard blocker at the library level, not a
runtime/MCA/protocol tunable (which Prompt 40 already exhausted) and not
something `MpiHaloTransport`'s source can work around. The C512
regression's root cause chain (Prompts 38-40: message-size-driven cost
growth -> isolated to the MPI transport/protocol layer, not generic GPU
halo work -> not explained by eager/rendezvous threshold) therefore
**cannot be tested against a CUDA-aware alternative on the currently
validated toolchain.** A CUDA-aware build (`nvidia/nvhpc-hpcx-cuda12/
24.7`) does already exist on Discover as a future candidate, but
adopting it is an environment/toolchain decision requiring its own scoped
evaluation (ABI/host-compiler-family impact, revalidation of the existing
correctness matrix), not a same-prompt drop-in.

**Preferred C512 transport: HOST-STAGED** (unchanged — the only option
available under the validated toolchain; remains correct, BITWISE, and
the current default).

## Summary Table

| Stage | Result |
|---|---|
| 0. CUDA-aware MPI build support | Confirmed NO via `ompi_info` (`opal_built_with_cuda_support=false`; configure line lacks `--with-cuda`) |
| 0b. Already-available CUDA-aware build check | `nvidia/nvhpc-hpcx-cuda12/24.7` (HPC-X, OpenMPI 4.1.7a1, `cuda_support=true`) confirmed present; not adopted (toolchain-level decision, out of scope) |
| 1-6 | Not reached — Hard Stop Condition 1 triggered at Stage 0 |
| Decision | CUDA-AWARE MPI BLOCKED BY MPI BUILD |
| Preferred transport | HOST-STAGED (unchanged) |

## Final Report

**Model:** SONNET (honored throughout).

--------------------------------------------------
**CUDA-AWARE MPI SUPPORT**
--------------------------------------------------
OpenMPI version: 4.1.6 (validated module, `mpi/openmpi/4.1.6/gcc-12.3.0`)
CUDA-aware MPI build support: **NO**
Device-buffer P2P supported: **NO**
Active PML: ob1 (unchanged from Prompt 40; same-node vader path)
Active same-node transport: vader (shared memory), self for local
Existing CUDA-aware transport found: NO
Implementation required: NO (blocked before implementation stage)

--------------------------------------------------
**IMPLEMENTATION**
--------------------------------------------------
Transport modes:
host_staged: PASS (unchanged, pre-existing, not re-tested this prompt)
cuda_aware: UNSUPPORTED

Files modified: none
Substantial changes used: 0/2
Host-staged remains default: YES

--------------------------------------------------
**CORRECTNESS**
--------------------------------------------------
Transport test: NOT RUN (nothing new to test)
Standalone C512 CUDA-aware: NOT REACHED
NaN/Inf: N/A
delp positive: N/A
T@500: N/A
U@250: N/A
V@250: N/A
PS: N/A
Correctness classification: NOT REACHED

--------------------------------------------------
**SAME-NODE PAIRED A/B**
--------------------------------------------------
Node: N/A (no job submitted)
GPU: N/A
Pairs: 0
Median host-staged: N/A (not re-measured this prompt; see Prompt 40, 6393.83 ms/call on `warpa005`)
Median CUDA-aware: N/A
CUDA-aware speedup: N/A
Improvement: N/A

--------------------------------------------------
**COMMUNICATION CLOCKS**
--------------------------------------------------
Host-staged COMM_DSW: N/A (not re-measured; see Prompt 40, 1032.52 ms/call)
CUDA-aware COMM_DSW: N/A
Change: N/A
Host-staged COMM_TOTAL: N/A (see Prompt 40, 1835.10 ms/call)
CUDA-aware COMM_TOTAL: N/A
Change: N/A

--------------------------------------------------
**DECISION**
--------------------------------------------------
Classification: **CUDA-AWARE MPI UNSUPPORTED**
Preferred C512 transport: **HOST-STAGED**

--------------------------------------------------
**NEXT STEP**
--------------------------------------------------
Recommended exactly one next step: evaluate whether adopting the
already-installed `nvidia/nvhpc-hpcx-cuda12/24.7` CUDA-aware MPI module
(or another CUDA-aware MPI already on Discover) is viable for this
project — a bounded environment/toolchain-compatibility investigation
(host-compiler family, link compatibility with the existing CUDA
dyn_core build, and whether adopting it would jeopardize the Intel-
compiler production-ABI compatibility established in Prompt 30) —
BEFORE attempting any CUDA-aware transport implementation or A/B test.
Do not rebuild/reconfigure OpenMPI. Do not attempt a CUDA-aware transport
implementation until that investigation confirms a viable, ABI-compatible
CUDA-aware MPI is usable for this project's toolchain.

--------------------------------------------------
**PROJECT STATUS**
--------------------------------------------------
C512 numerical correctness: PASS (unchanged, BITWISE, Prompt 37/§9a)
C512 transport regression: CONFIRMED (unchanged, Prompts 38-40)
MPI eager threshold: RULED OUT (Prompt 40, unchanged)
PeerHalo generic halo path: EXONERATED (Prompt 39, unchanged)
CUDA-aware MPI: **BLOCKED BY MPI BUILD** (this prompt)
Numerical source changed: NO
Held-Suarez physics changed: NO
MPI topology changed: NO
Profiling used: NO
Escalation to Opus: NO
GPU jobs submitted: 0
Source files modified: 0

CURRENT_STATE size: see `CURRENT_STATE.md` (updated alongside this doc)

C512 CUDA-AWARE MPI UNSUPPORTED — READY FOR CUDA-AWARE MPI ENVIRONMENT TEST
