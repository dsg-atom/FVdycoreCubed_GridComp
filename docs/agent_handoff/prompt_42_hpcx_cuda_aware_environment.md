# Prompt 42: CUDA-Aware MPI Environment Gate — Redirected from HPC-X to Intel MPI 2021.13

Branch: `experiment/held-suarez-e2e`. Model: Sonnet only, per task policy.

**This prompt was originally scoped as an HPC-X (`nvidia/nvhpc-hpcx-cuda12/
24.7`) CUDA-aware MPI environment/ABI compatibility gate, following
Prompt 41's finding that the validated OpenMPI 4.1.6 module lacks
`--with-cuda`. Mid-investigation, a higher-priority correction to Prompts
39-41's premise was found and confirmed with the user before proceeding:
the actual C512 production/perf path does not use OpenMPI at all — it uses
Intel MPI 2021.13. The user redirected this prompt to test CUDA-aware
support against Intel MPI 2021.13 instead of HPC-X.**

## Correction to Prompts 39-41: Wrong MPI Implementation Was Characterized

Prompt 14/24 (existing project history, not new) already documented that
the standalone Fortran fixture is Intel-compiler-based (`ifort`/Intel MPI
2021.13, ESMA-Baselibs `ifort_2021.13.0-intelmpi_2021.13.0`) and that a
dedicated `build_cuda_r8_intelmpi` CUDA build exists specifically so the
CUDA/`MpiHaloTransport` code shares one MPI ABI with the Fortran
executable. This was re-confirmed directly from the actual job log used by
both Prompt 37 and Prompt 40 (`p37_c512_perf_cuda_3rep_58444655.out`,
reused unmodified by Prompt 40's job 58445328):

```
Currently Loaded Modules: ... 14) comp/intel/2024.2.0  15) mpi/impi/2021.13 ...
esma_mpirun: mpi_type = intel
/usr/local/intel/oneapi/2024/mpi/2021.13/bin/mpirun -np 6 .../StandAlone_FV3_Dycore.x
```

**The C512 six-rank job runs entirely under Intel MPI 2021.13, never
OpenMPI.** Consequences for prior prompts:

- **Prompt 40's eager/rendezvous MCA sweep was inert.**
  `OMPI_MCA_btl_vader_eager_limit` is an OpenMPI-only environment variable;
  Intel MPI's `mpirun`/`libmpi.so.12` do not read it. All three settings
  tested (default/145000B/524288B) produced *identical actual runtime
  behavior* — the observed ±1.4-1.6% deltas were measurement noise, not a
  real protocol-threshold test. The "BTL=vader, no crossover" analysis
  characterized a library that was never active in this job.
- **Prompt 41's "CUDA-aware MPI blocked by MPI build" finding is only true
  for OpenMPI 4.1.6**, which is not in the production path. The actually-
  relevant question — whether Intel MPI 2021.13 supports CUDA-aware
  device-buffer transport — was never asked in Prompt 41.
- Prompt 39's PeerHalo-vs-MPI discriminator result (MPI transport/protocol
  path is the root cause, not generic GPU halo work) is **unaffected** —
  that conclusion did not depend on which MPI implementation was active,
  only on comparing in-process vs. cross-process transport.

CURRENT_STATE.md (post-Prompt-41 compaction) did not carry forward the
Intel MPI fact from Prompt 14/24 — a real gap introduced/perpetuated by
compaction, now corrected in this update.

## Stage 2/3 (redirected target): CUDA-Aware Support Confirmed for Intel MPI 2021.13

Verified on the login node first, cheaply, before any GPU job:

- `mpicc -show` / `mpicxx -show` (module `mpi/impi/2021.13`) confirm the
  wrapper backs onto `g++`/`gcc` when `comp/gcc/12.3.0` is loaded before
  `comp/intel/2024.2.0` — the same wrapper pattern (`mpigxx`) already used
  by `build_cuda_r8_intelmpi`'s CMakeCache
  (`MPI_CXX_COMPILER=.../bin/mpigxx`, `CMAKE_CXX_COMPILER=g++`). No new
  compiler-family decision required.
- `strings` on `libmpi.so.12` show the `MPIX_Query_cuda_support` symbol
  (the same runtime-query API OpenMPI also exposes) plus internal CUDA
  topology/detection code (`MPL_CUDA_*`, `cudaGetDeviceCount`,
  `dlopen("libcuda.so")`).
- `libfabric.so.1` (Intel MPI's bundled `opt/mpi/libfabric`) exports CUDA
  HMEM support (`cuda_hmem_init`, `cuda_gdrcopy_dev_register`,
  `cuda_dev_reg_copy_to_hmem`) — genuine GPU-direct infrastructure, not
  just a name-list.
- On the login node (no GPU/CUDA driver), `MPIX_Query_cuda_support()`
  returned `0` and `I_MPI_OFFLOAD=1` produced `Could not load either Level
  Zero or CUDA` — expected with no GPU present, not evidence against
  support. Confirmed the real answer requires a GPU node.

### GPU-node verification (1 job, well within the 3-4 job budget)

Wrote a minimal, standalone 2-rank CUDA-device-buffer MPI correctness test
(`scratch_p42/intelmpi_cuda_aware_test.cu`, `scratch_p42/
p42_intelmpi_cuda_aware.sbatch`) — no FV3/Held-Suarez source touched, no
production `HaloTransport` code modified. Test: `cudaMalloc` a device
buffer on each of 2 ranks, rank 0 fills it via H2D, then `MPI_Send`/
`MPI_Recv` is called **directly on the device pointer** (rank 0 sends,
rank 1 receives) at three message sizes — 4KB, **135KB (138,650 bytes,
matching Prompt 38's derived C512 ordinary-group halo message size
exactly)**, and 256KB — followed by a D2H copy and exact `double`
comparison against the expected pattern.

Job 58450084, `gpu_a100`/`rome`, 1 GPU, 2 tasks, node `warpa003`,
COMPLETED, exit code 0, elapsed 11s. Two `mpirun` invocations in the same
job:

| Config | `MPIX_Query_cuda_support()` | Result |
|---|---|---|
| Default env (no `I_MPI_OFFLOAD`) | `0` | **SIGSEGV** (rank 0, signal 11) — device pointer passed to `MPI_Send` is treated as host memory and dereferenced directly, crashing. Rank 1 killed by signal 9 as a result. |
| `I_MPI_OFFLOAD=1`, `I_MPI_DEBUG=4` | `1` | **PASS** all three cases. `libfabric provider: mlx`. |

```
CASE=4KB   BYTES=4096   RESULT=PASS
CASE=135KB BYTES=138650 RESULT=PASS
CASE=256KB BYTES=262144 RESULT=PASS
INTELMPI_CUDA_AWARE_TEST_PASS
```

**CUDA-aware MPI build support (Intel MPI 2021.13): YES, gated behind
`I_MPI_OFFLOAD=1`.** Device-buffer P2P: PASS, bit-exact round trip, at the
exact C512 halo message size and two other sizes, on a real A100
(`warpa003`), using the `mlx` libfabric provider (same provider identified
in the earlier `I_MPI_DEBUG=5` login-node probe).

Without `I_MPI_OFFLOAD=1`, passing a device pointer is **unsafe** (matches
Prompt 41's warning about "host `memcpy` against un-mapped device memory")
— confirms `I_MPI_OFFLOAD=1` (or equivalent) must be set for any future
CUDA-aware `HaloTransport` implementation and must never be silently
assumed on.

## Stages 4/6/7 (ABI / standalone build / standalone run compatibility): Already Satisfied by Existing Project History

Because the CUDA-aware candidate is the *same* Intel MPI 2021.13 already
linked into `build_cuda_r8_intelmpi` and the currently-installed
`StandAlone_FV3_Dycore.x`, the toolchain-migration/ABI-compatibility
questions this gate exists to answer are already answered by prior,
independently-validated project work — no new build or rerun was needed:

- Compiler/ABI compatibility: established Prompt 14/24 (`build_cuda_r8_
  intelmpi` exists precisely to keep the CUDA/C++ code and the Intel-MPI-
  linked Fortran executable on one MPI ABI) and Prompt 30 (fixed-width
  `iso_c_binding` types, `MPI_Comm_f2c`/`c2f` conversion, runtime `abi_
  version`/`struct_bytes` guards — all already Intel-compiler-aware by
  design).
- Standalone build/link and host-staged correctness under this exact
  executable/MPI stack: **already PASS** — Prompt 36/37/§9a's C256/C512
  BITWISE Held-Suarez gates ran under this identical Intel MPI 2021.13
  stack (confirmed above from the job logs), not a hypothetical.
- No `build_hpcx`/new build directory was created; no rebuild of the
  standalone fixture was attempted; 0 production source files modified.

## What Remains Open (explicitly NOT attempted in this gate)

- Whether `MpiHaloTransport::transfer()` (`cuda/dyn_core/runtime/
  fv3_gpu_halo_mpi.cpp`) can be adapted to pass its existing pinned-host
  buffers' *device-side counterparts* directly to `MPI_Isend`/`MPI_Irecv`
  under `I_MPI_OFFLOAD=1`, and whether that removes the D2H/H2D staging
  currently in the critical path — **implementation work, out of scope
  for this gate, belongs in a future Prompt 43.**
- Six-rank (production topology) device-buffer correctness — only 2-rank
  was tested here, per the task's cost-control cheapest-discriminator-
  first rule; sufficient to answer the environment-gate question.
- Any performance comparison — explicitly out of scope for this gate.
- Whether `I_MPI_OFFLOAD=1` has any effect on existing host-staged
  `MpiHaloTransport` runs (should be transparent since those never pass
  device pointers, but not verified here — a one-line sanity check for
  Prompt 43, not this gate).

## Decision

**Case A-equivalent: CUDA-AWARE MPI ENVIRONMENT PASS, but for Intel MPI
2021.13, not HPC-X.** The originally-scoped HPC-X toolchain-migration
question is moot — the CUDA-aware MPI candidate is already the validated,
in-production MPI implementation. No new toolchain adoption, no ABI risk,
no host-compiler-family change. `nvidia/nvhpc-hpcx-cuda12/24.7` (Prompt
41's Stage 0b finding) remains an untouched, unneeded fallback.

**Recommended next step: Prompt 43 — implement a CUDA-aware
`MpiHaloTransport` variant** (device-pointer `MPI_Isend`/`MPI_Irecv` under
`I_MPI_OFFLOAD=1`, behind the existing `HaloTransport` interface, default
OFF) and A/B it against the current host-staged path at C512, same-node/
same-allocation, per Prompt 41's already-defined methodology.

## Final Report

**Model:** SONNET (honored throughout).

--------------------------------------------------
**CORRECTION TO PRIOR ASSUMPTIONS**
--------------------------------------------------
Prompts 39-41 characterized OpenMPI 4.1.6, which is not the MPI
implementation used by the actual C512 production/perf path (confirmed
Intel MPI 2021.13 from job logs). Prompt 40's MCA eager-limit sweep was
inert (wrong env-var namespace). Prompt 41's "CUDA-aware MPI blocked"
finding applies only to OpenMPI, not to the actually-active Intel MPI.
Prompt 39's root-cause isolation (MPI transport/protocol layer, not
generic GPU halo work) is unaffected.

--------------------------------------------------
**CUDA-AWARE MPI SUPPORT (Intel MPI 2021.13)**
--------------------------------------------------
CUDA-aware MPI build support: **YES** (gated behind `I_MPI_OFFLOAD=1`)
Device-buffer P2P supported: **YES** — verified PASS at 4KB/135KB/256KB
Active libfabric provider: `mlx`
Without `I_MPI_OFFLOAD=1`: **UNSAFE** (SIGSEGV, confirmed)
Same MPI implementation already used by: `build_cuda_r8_intelmpi`,
production `StandAlone_FV3_Dycore.x`, all Prompt 36-40 C256/C512 gates

--------------------------------------------------
**GPU VERIFICATION**
--------------------------------------------------
Job: 58450084, node `warpa003`, gpu_a100/rome, 1 GPU, 2 ranks, rc=0,
elapsed 11s
Test: `scratch_p42/intelmpi_cuda_aware_test.cu` (new, standalone, no
production source touched)
Cases: 4KB PASS, 135KB (exact C512 halo message size) PASS, 256KB PASS
GPU jobs submitted this prompt: 1 (budget was 3-4)

--------------------------------------------------
**ABI / STANDALONE COMPATIBILITY**
--------------------------------------------------
Toolchain migration required: **NO** — target MPI is already the
validated, linked, in-production implementation
New build directory created: NO
Production source modified: NO (0 files)
Existing standalone correctness under this MPI: already PASS (Prompt
36/37/§9a, BITWISE)

--------------------------------------------------
**DECISION**
--------------------------------------------------
Classification: **CUDA-AWARE MPI ENVIRONMENT PASS (Intel MPI 2021.13)**
Recommended next step: Prompt 43 — implement + A/B a CUDA-aware
`MpiHaloTransport` variant at C512, same-node/same-allocation, per
Prompt 41's methodology.

--------------------------------------------------
**PROJECT STATUS**
--------------------------------------------------
C512 numerical correctness: PASS (unchanged, BITWISE, Prompt 37/§9a)
C512 transport regression: CONFIRMED (unchanged, Prompt 38/39)
MPI eager threshold (OpenMPI): RULED OUT, but now known to have been
tested against the wrong library (this prompt's correction)
PeerHalo generic halo path: EXONERATED (Prompt 39, unchanged)
CUDA-aware MPI (Intel MPI 2021.13): **ENVIRONMENT PASS** (this prompt)
Numerical source changed: NO
Held-Suarez physics changed: NO
MPI topology changed: NO
Profiling used: NO
Escalation to Opus: NO (resolved via user redirect instead)
GPU jobs submitted: 1
Source files modified: 0 (test artifacts confined to new `scratch_p42/`)

CUDA-AWARE MPI ENVIRONMENT PASS — READY FOR PROMPT 43 TRANSPORT A/B
