# Prompt 26 — Rank/GPU Execution-Architecture Characterization

_Date: 2026-09-14. Branch `cuda/gpu-bringup`, HEAD `6817f8d`. Study scope:
execution-architecture characterization only. No CUDA-aware MPI was
implemented. No numerical algorithm, topology, or production source was
changed. All measurements are GPU EXECUTED; the numerical status of the CUDA
backend is unchanged from `CURRENT_STATE.md` §5._

Entry point: `prompt_25a_fv3_halo_characterization.md` concluded
"DO NOT IMPLEMENT [CUDA-aware MPI] YET — INVESTIGATE RANK/GPU ARCHITECTURE",
because the halo traffic is 312 tiny messages/`dyn_core` (only 8712 B and
2904 B, 100% ≤64 KB) with a device-direct-MPI ceiling of only ~1.02–1.04×,
while MPI wait time rose ~3.3× going from 2→6 ranks on the same GPU. This
prompt tests the rank/GPU hypothesis directly.

---

## 1. Current Execution Architecture (CONFIRMED FROM SOURCE + RUNTIME, NOT ASSUMED)

| Property | Value | Evidence |
|---|---|---|
| MPI ranks | 6 | `fv3.j`: `MODEL_NPES=6`, `FV_NX=1`, `FV_NY=6` |
| Tiles per rank | exactly 1 | `@fvdycore/tools/fv_mp_mod.F90:662` `Atm%tile = (gid-pelist(1))/npes_per_tile+1` with `npes_per_tile = npes/nregions` |
| CUDA contexts per rank | 1, and it is the **primary** context | no `cuCtxCreate` anywhere in `cuda/`; `cudaSetDevice` only — `fv3_gpu_runtime.cpp:1458-1470` |
| Streams per context | 2, both `cudaStreamNonBlocking` (compute, transfer) | `fv3_gpu_runtime.cpp:1458-1470` |
| Events per context | 2, `cudaEventDisableTiming` | same |
| GPU assignment | round-robin over *visible* devices: `dev = mod(comm_rank, device_count)` | `cuda/interface/fv3_gpu_dyncore_bridge.F90:155`; logged per rank ("rank N owns tile T, device D") |
| Contexts per physical GPU under `--gres=gpu:1` | **6** | all six ranks resolve `dev=0`; confirmed at runtime by `nvidia-smi --query-compute-apps` (six PIDs, one bus ID) |
| Same stage sequence per rank | yes — each rank independently replays the identical dispatch script for its own tile | production dispatch install is per-context; no cross-rank work partitioning exists |
| Cross-rank kernel overlap on one GPU | **NO** without MPS — separate contexts are time-sliced, not co-resident | NVIDIA context model; corroborated by §3/§4 measurements |
| CUDA MPS in use for this account | **NO** (was not in use for Prompt 24 either) | `echo get_server_list \| nvidia-cuda-mps-control` → "Cannot find MPS control daemon process". A daemon *does* run on these nodes but belongs to another user with a private pipe directory, so this account's processes never attached to it. Verified, not assumed. |
| GPU hardware | A100-SXM4-40GB, `compute_mode=Default` (MPS-eligible), `Gres=gpu:4` per node | `nvidia-smi --query-gpu=index,name,compute_mode` |

So the production standalone mapping today is:

```
6 MPI ranks -> 6 tiles (1/rank) -> 6 primary CUDA contexts -> 1 A100
   (time-sliced; 2 non-blocking streams each; no MPS)
```

This confirms the prompt's primary suspect exactly.

---

## 2. Architecture Candidates

| # | Architecture | Implementation scope | Topology impact | MPI impact | Halo-transport impact | Expected contention reduction | Compatible with validated `HaloPeerGroup`? |
|---|---|---|---|---|---|---|---|
| **A** | 6 ranks / 1 GPU (today) | none (baseline) | none | none | none | none (reference) | n/a |
| **B** | 3 ranks / 1 GPU, 2 tiles/rank | **HIGH** — not expressible in authoritative FMS decomposition (§2.1) | breaks 1-tile-per-PE | rank↔tile map, `pelist`, all FMS domain plumbing | needs hybrid intra-rank peer + inter-rank MPI | 2× fewer contexts | partially (intra-rank pairs) |
| **C** | 2 ranks / 1 GPU, 3 tiles/rank | HIGH, same reason | same | same | same | 3× fewer contexts | partially |
| **D** | 1 rank / 1 GPU, 6 tiles | HIGH, same reason; but is exactly the internal-harness arrangement | same | eliminates dyn_core halo MPI entirely | pure `PeerHaloTransport` | 6× fewer contexts (ceiling) | yes, fully |
| **E** | 6 ranks / 1 GPU **+ CUDA MPS** | **NONE** — operational only (daemon + 2 env vars in the job script) | none | none | none | contexts multiplexed into 1 server context ⇒ cross-rank kernel overlap | n/a (unchanged) |
| **F** | 6 ranks / multiple GPUs | **NONE in source** — `dev = mod(rank, device_count)` already does it; controlled by `CUDA_VISIBLE_DEVICES`/`--gres` | none | none | none | contexts/GPU ÷ (#GPUs) | n/a (unchanged) |

### 2.1 Why B/C/D are out of scope right now

`fv_mp_mod.F90:662` computes `Atm%tile` from the PE index with
`npes_per_tile = npes/nregions`, i.e. the authoritative GEOS/FMS cubed-sphere
decomposition assigns **exactly one tile per PE** and requires `npes` to be a
multiple of 6. Multiple tiles per rank is therefore not a CUDA-side change at
all — it would require changing the authoritative Fortran decomposition, its
`pelist` construction, and every FMS domain/halo contract built on it. That is
a high-risk production redesign, and per §7 of the task it must only be
recommended if the expected whole-call benefit clearly exceeds the
CUDA-aware-MPI ceiling. It does not need to be: E and F reach the same
contention ceiling at zero source cost (§4, §5).

---

## 3. Context-Scaling Evidence

Three independent measurements, all on one A100, all MPS-off.

### 3.1 Real dyn_core numerics, 1 context vs 6 contexts (job 58380039, warpa013)

Same binary (`fv3_gpu_perf_bench`), same flags, same c32/R8/`npz=10`/
`n_split=3`, 50 iterations × 3 repeats:

| Arrangement | ms/call (3 reps) | median | vs 1 context |
|---|---|---|---|
| B1: 1 process / 6 threads / 6 tiles / `PeerHaloTransport` → **1 context** | 28.725, 28.794, 28.651 | **28.725** | 1.000× |
| B2: 6 processes / 1 tile each / `MpiHaloTransport` → **6 contexts** | 102.966, 103.056, 102.980 | **102.980** | **3.585× slower** |

**Recorded confound (not hidden):** B1 and B2 also differ in halo transport —
the peer path is device-to-device with host barriers only and never calls
`cudaStreamSynchronize`/`cudaDeviceSynchronize` inside an exchange
(`cuda/dyn_core/include/fv3_gpu_halo_peer.hpp`), whereas the MPI path host-stages
and issues 24 `cudaEventSynchronize` per call. This single comparison therefore
cannot separate context contention from transport. §3.2 removes that confound.

### 3.2 Synthetic context-scaling probe, total work held constant (same job)

`scratch_p26/ctx_scaling_probe.cu` — six tile workers redistributed over
1/2/3/6 processes, **no MPI and no transport difference of any kind**,
reproducing only the three structural properties the Prompt 24 nsys trace
identified as dominant (≈2952 launches/call/worker, short kernels over a
38×38×11 c32 halo box, 24 host-blocking event syncs/call/worker, 2
non-blocking streams/context), with a wall-clock start barrier:

| Contexts | procs × threads | worst-worker ms/call | normalized |
|---|---|---|---|
| 1 | 1 × 6 | 28.085 | 1.000× |
| 2 | 2 × 3 | 30.711 | 1.094× |
| 3 | 3 × 2 | 35.724 | 1.272× |
| 6 | 6 × 1 | 72.484 | **2.581×** |

Total GPU work is identical in all four rows, so the 2.58× degradation is
attributable to CUDA context/launch/scheduling contention alone.
Scaling is **superlinear in context count** (1.09 → 1.27 → 2.58) — the knee is
between 3 and 6 contexts. Classification: **SEVERE CONTENTION**.

### 3.3 Real standalone path, contexts/GPU halved via a second GPU (job 58380134, warpa013)

Candidate F, real `StandAlone_FV3_Dycore.x`, identical job directory/config to
the Prompt 24 baseline (c32/R8, `bdt=1200`, 100 `dyn_core` calls/rep,
`FV3_SAVEPOINT_ENABLE=0`, `fv_timing_mod` `DYN_CORE` timer), 3 reps, six ranks
throughout — only `CUDA_VISIBLE_DEVICES` differs:

| Phase | contexts/GPU | DYN_CORE ms/call (3 reps) | median | vs P1 |
|---|---|---|---|---|
| P1 `CVD=0` | 6 | 106.007, 106.026, 106.414 | **106.007** | 1.000× |
| P3 `CVD=0,1` | 3 (verified: 3 PIDs on bus 81:00.0, 3 on C1:00.0) | 54.410, 54.531, 54.730 | **54.410** | **1.949× faster** |

0 Range_Warn, 0 NaN, rc=0 in all six runs. Kernel compute is only ~11% of
per-call cost (Prompt 24), so doubling compute resource cannot account for a
1.95× improvement — halving contexts/GPU is the mechanism. This corroborates
§3.2 on the real path.

---

## 4. CUDA MPS Test (FIRST PRIORITY) — job 58380090, warpa003

Candidate E on the real standalone path. Same node, same GPU, same c32/R8
config, same 100 calls/rep × 3 reps, same job directory and binary, no source
change and no flag change. The only difference is a private user-level MPS
daemon (`CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` under `/tmp`,
`nvidia-cuda-mps-control -d`; server confirmed live, PID 60272 in
`get_server_list`, torn down with `quit` at the end).

| Phase | DYN_CORE ms/call (3 reps) | median |
|---|---|---|
| P1 MPS **off** (6 contexts, time-sliced) | 106.261, 106.370, 106.261 | **106.261** |
| P2 MPS **on** (6 clients → 1 server context) | 27.075, 29.692, 26.763 | **27.075** |

**MPS speedup on `DYN_CORE`: 3.924×.**

P1's MPS-off median reproduces the Prompt 24 baseline (105.563 ms/call) to
within 0.7%, so the MPS-off leg is a valid re-baseline and Prompt 24's number
was genuinely MPS-free.

Whole-run and secondary timers (rep 1 logs):

| Timer | MPS off | MPS on |
|---|---|---|
| `TOTAL` | 14.7257 s | 5.8631 s (**2.51×**) |
| `DYN_CORE` | 10.6007 s | 2.7075 s |
| `COMM_TOTAL` | 0.1151 s | 0.0296 s (**3.89×**) |
| `REMAPPING` | 0.2893 s | 0.2917 s (unchanged, as expected — CPU Fortran) |

(The MPS-off `TOTAL`/`COMM_TOTAL`/`REMAPPING` rows are read from job 58380134's
P1 rep-1 log on warpa013, because job 58380090's own P1 logs in the shared job
directory were later overwritten; its `DYN_CORE` differs from 58380090's P1 by
0.24%, so the cross-node read is sound. All `DYN_CORE` figures in the table
above are from 58380090 itself, single node.)

`COMM_TOTAL` falling 3.9× under MPS is itself diagnostic: the reported "MPI
wait" was largely ranks waiting on a serialized GPU, not on the network. This
independently supports Prompt 25a's LOW rating for CUDA-aware MPI.

Against the same-node Fortran reference (19.930 ms/call, Prompt 24):

| CUDA configuration | ms/call | vs Fortran |
|---|---|---|
| 6 contexts / 1 GPU, MPS off (Prompt 24 baseline) | 106.261 | 0.188× (5.33× slower) |
| 6 contexts / 1 GPU, **MPS on** | 27.075 | **0.736× (1.36× slower)** |
| 3 contexts / GPU on 2 GPUs, MPS off | 54.410 | 0.366× (2.73× slower) — but consumes 2 GPUs |

MPS on a **single** GPU beats two GPUs without MPS (27.1 vs 54.4 ms/call), and
lands essentially at the 1-context ceiling measured in §3.1/§3.2
(28.7 / 28.1 ms/call). MPS therefore captures effectively the entire available
contention benefit without any of candidate B/C/D's redesign risk.

### 4.1 MPS correctness guard — job 58380133 (+ MPS-off control 58380203)

MPS changes only how client contexts are scheduled onto the device; it does not
change kernels, per-rank launch order, or arithmetic. That claim was tested,
not asserted: the Prompt 23 Stage B job (c32/R8, `bdt=1200`, 6 `dyn_core`
calls, savepoint dumping ON) was re-run under MPS and every savepoint
byte-compared against Prompt 23's existing non-MPS reference dump (which was
moved aside and restored, never overwritten).

```
BITWISE_RESULT identical=1230 differing=66 missing=0     (1296 files/side, rc=0)
```

All 66 differing files are `dce_it{1..6}_pe*_mfyL.bin` (36) and
`dce_it{1..6}_pe*_cxL.bin` (30) — i.e. confined to the dce-**entry** dumps that
`CURRENT_STATE.md` §5 already records as a known non-blocking c32 artifact
("`mfyL` dce-entry-only 2-cell uninitialized-memory artifact, both backends
independently… real exit-state agreement clean"). **No exit-state or physical
state field differs.** **MPS-off control (job 58380203).** The identical job was re-run against the
same reference **without MPS**, to discriminate "MPS-induced" from
"pre-existing run-to-run nondeterminism":

```
BITWISE_RESULT identical=1230 differing=66 missing=0     (rc=0, 0 Range_Warn, 0 NaN)
differing file classes:  30 x dce_*_cxL.bin   36 x dce_*_mfyL.bin
```

Identical result — same count, same 66 files, same two classes. The dce-entry
`mfyL`/`cxL` dumps therefore differ run-to-run on this path whether MPS is on or
off, confirming they are the already-documented uninitialized-read artifact and
**not** an MPS effect.

Status: **MPS is bitwise-neutral.** All 1230 state/exit savepoints are
byte-identical, and the 66 dce-entry files that differ under MPS differ
identically without it.

---

## 5. Single-Process Six-Tile vs Six-MPI-Process

See §3.1 for the numbers (28.725 vs 102.980 ms/call, 3.585×). Per the task's
instruction, this difference is **not** attributed to halo transport alone:
the confound is stated explicitly in §3.1, and §3.2's transport-free probe
independently reproduces most of the gap (2.581× from context count alone),
while §4's MPS result reproduces essentially all of it (3.924×) with the halo
transport held completely fixed. Conclusion: the dominant term is
context contention on the shared GPU, not the transport implementation.

---

## 6. Candidate Ranking

| Rank | Candidate | Expected benefit | Implementation scope | Numerical/topology risk | Operational complexity | GEOS production compatibility |
|---|---|---|---|---|---|---|
| **1** | **E — MPS, 6 ranks / 1 GPU** | **3.92× measured** on `DYN_CORE`, 2.51× whole-run | **none** (job-script only) | none (bitwise-neutral, §4.1); topology untouched | low: per-node daemon + 2 env vars + teardown; needs `compute_mode=Default` | high — no source or decomposition change; standard NVIDIA facility |
| 2 | F — 6 ranks / multiple GPUs | 1.95× measured at 3 ctx/GPU | none (already implemented) | none | moderate: requires ≥2 GPUs/node allocation; `gpu:4` was unschedulable during this study | high, but costs GPUs to buy less benefit than E; **composes with E** |
| 3 | D — 1 rank / 6 tiles | ≈1-context ceiling, ~3.6–3.9× — i.e. no better than E | HIGH (authoritative FMS decomposition) | HIGH | high | low for now |
| 4 | B/C — 2–3 tiles/rank | between D and A, < E | HIGH | HIGH | high | low |
| — | A — status quo | reference | — | — | — | — |

---

## 7. Findings

1. The production standalone mapping is `6 ranks → 6 tiles → 6 primary CUDA
   contexts → 1 A100`, time-sliced, MPS-off. Confirmed from source and runtime.
2. Context count on a shared GPU is the dominant cost in this integration.
   Superlinear degradation, measured three ways: 2.58× (synthetic, work held
   constant), 1.95× recovered by halving contexts/GPU (real path), 3.59×
   between 1-context and 6-context real dyn_core.
3. **CUDA MPS recovers 3.92× on `DYN_CORE` and 2.51× whole-run with zero source
   change**, moving CUDA from 5.33× slower than Fortran to 1.36× slower on the
   same node/GPU/config, and is bitwise-neutral on all state/exit savepoints.
4. `COMM_TOTAL` drops 3.89× under MPS. The apparent MPI wait was mostly GPU
   serialization, not network — confirming Prompt 25a's LOW CUDA-aware-MPI
   rating from a second direction.
5. Multiple tiles per MPI rank (B/C/D) is not expressible in the authoritative
   FMS cubed-sphere decomposition (one tile per PE, `fv_mp_mod.F90:662`) and
   would be a high-risk production redesign. It is **not justified**, because it
   cannot beat the 1-context ceiling that MPS already reaches for free.
6. Kernel optimization remains LOW priority for this path: GPU kernel compute
   is ~11% of per-call cost (Prompt 24) and none of this prompt's gains came
   from kernels. `ncu` remains platform-blocked (`ERR_NVGPUCTRPERM`); not
   needed for this study.

## 8. Recommendation

Adopt candidate **E (CUDA MPS)** as the operational configuration for
single-GPU six-rank CUDA FV3 runs, and compose it with **F** whenever more than
one GPU per node is available. Do **not** implement CUDA-aware MPI (Prompt 25a
ceiling ~1.02–1.04×; and §4 shows the "MPI wait" was GPU serialization). Do
**not** redesign rank/tile ownership.

Next step: re-measure the post-MPS bottleneck profile (a short `nsys` trace
under MPS) before choosing any further optimization, since MPS invalidates the
Prompt 24 bucket attribution (host synchronization ~56%) that motivated the
previous priority list. Then decide between MPS-aware operational integration
in the GEOS job scripts and the remaining production-integration track (moist
`q_con ≠ 0`, `a_imp ≤ 0.999`, Intel-compiler ABI verification).

## 9. Artifacts

Untracked scratch/experiment files (not for commit):
- `scratch_p26/ctx_scaling_probe.cu`, `scratch_p26/p26_ctx_control.sbatch`,
  `scratch_p26/p26_ctx_58380039.out`
- `<fortran_perf>/experiments/p26_mps.sbatch`, `p26_multigpu.sbatch`,
  `p26_mps_bitwise.sbatch`, `p26_nomps_bitwise.sbatch`,
  `p26_rankgpu_arch.sbatch` (original 4-phase driver; cancelled as job
  58380030 because `gpu:4` would not schedule, then split into the two jobs
  above)
- Job outputs: `p26_mps_58380090.out`, `p26_mgpu_58380134.out`,
  `p26_mpsbit_58380133.out`, `p26_nombit_58380203.out`

Jobs: 58380030 (cancelled, `gpu:4` unschedulable), 58380039 (context scaling),
58380090 (MPS off/on), 58380133 (MPS bitwise guard), 58380134 (multi-GPU),
58380203 (MPS-off bitwise control; same 66-file result as 58380133, proving
the difference is not MPS-induced).
