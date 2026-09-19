# Prompt 46: C512 Reduced-Rank-Density CUDA-Aware MPI A/B

## 1. Objective

Prompt 45 found the CUDA-aware `MpiHaloTransport` (Intel MPI 2021.13,
`I_MPI_OFFLOAD=1`) OOMs at C512's standard density of 6 MPI ranks on 1 A100
(host-staged transport alone already uses 39466/40960 MiB, 96.4%, leaving
<1.5GB headroom — insufficient for Intel MPI's GPU-direct/RDMA buffer pool).
Prompt 46 tests Prompt 45's first deferred follow-on: does lowering rank
density to 3 ranks/GPU across 2 A100s (same 6 ranks, same 6 tiles, same FV3
decomposition — only GPU *placement* changes, via `CUDA_VISIBLE_DEVICES=0,1`
and the existing `dev = mod(comm_rank, device_count)` round-robin at
`cuda/interface/fv3_gpu_dyncore_bridge.F90:156`, no source change) make
cuda-aware viable, and if so, whether it is performance-beneficial.

Constraints: SONNET only, C512 only, max 2 A100s, 6 ranks/3-per-GPU, no
topology/decomposition/numerics/physics changes, no new transport code, no
MPI tuning, no C256 rerun, no profiling, no kernel optimization.

## 2. Stage 0-1: No Source Change Needed / Job Scripts

Confirmed the existing `dev = mod(comm_rank, device_count)` mapping is
sufficient — GPU count is controlled entirely by `CUDA_VISIBLE_DEVICES` at
job-launch time, no CUDA/Fortran source touched. Three new sbatch scripts
built under `<fortran_perf>/experiments/`:
`p46_c512_2gpu_host_staged.sbatch`, `p46_c512_2gpu_cuda_aware.sbatch`,
`p46_c512_2gpu_perf_ab.sbatch` (paired A/B). All use `--gres=gpu:2`,
`CUDA_VISIBLE_DEVICES=0,1`, MPS OFF (Prompt 26 P3 2-GPU mechanism reused,
documented decision — per-device MPS would need multiple control daemons,
a new operational mechanism not exercised here), same C512 production
config as Prompt 37/44/45 (`bdt=36s`, `k_split=1`, `n_split=3`, fusions ON,
Graph OFF).

## 3. Stage 2-3: Host-Staged 2-GPU Correctness (Job 58454451)

6h simulated gate, `host_staged` transport, 2 GPUs/3-ranks-per-GPU:
`rc=0`, 0 NaN/Range_Warn/Error, 6 NC4 files, throughput 3.536 days/day.
NetCDF-level comparison (`hs_netcdf_inspect.py compare-backends`) against
the established 1-GPU `p37_c512_gate_cuda` baseline: **BITWISE identical**
(0.0 max_abs/rms on T@500hPa, U@250hPa, V@250hPa, PS, at all 3 checkpoints).
Two-GPU host-staged itself does not fail — **Case F ruled out.**

GPU memory-poll tooling defect found and fixed here: backgrounding
`nvidia-smi ... &` directly in the sbatch script's own `.batch`-step shell
produced a flat, bogus "4 MiB" reading for the entire 1h42m run (Intel
MPI's Hydra bootstraps ranks via its own internal `srun`, landing them in
a separate job step/cgroup the `.batch` step's nvidia-smi cannot see into
on this cluster). Fix (applied to the Stage 4 job, below):
`srun --overlap --jobid=$SLURM_JOB_ID -N1 -n1 nvidia-smi ... &`. Stage 2's
own peak-MiB number is therefore unmeasured (accepted gap — the
correctness gate itself does not depend on it, and the more decisive
cuda-aware viability test was prioritized over a costly re-run for
telemetry alone).

## 4. Stage 4: CUDA-Aware 2-GPU Viability (Job 58456709)

Same 6h gate, `cuda_aware` transport (`RUN_CUDA_FV3_MPI_TRANSPORT=cuda_aware`,
`I_MPI_OFFLOAD=1`), 2 GPUs/3-ranks-per-GPU: **`rc=0`, 0 NaN/Range_Warn/Error,
6 NC4 files, throughput 3.576 days/day** — no OOM, no
`cudaErrorMemoryAllocation`, no FATAL message. This directly reverses
Prompt 45's 1-GPU finding (crashed within ~39s at the first Riemann-solver
kernel launch). Peak GPU memory (with the fixed poller): **~5053 MiB/GPU
(12.3% of A100 capacity)** on both GPUs — a large headroom improvement over
Prompt 45's 96.4%-utilized crisis. **Case E ruled out — cuda-aware is
viable at 3-ranks/GPU density.**

## 5. Stage 5: Numerical Comparison, Host-Staged-2GPU vs CUDA-Aware-2GPU

`hs_netcdf_inspect.py compare-backends` on Stage 3's and Stage 4's own NC4
outputs: **BITWISE identical** (0.0 diff, T/U/V/PS, all checkpoints).
Transport choice does not affect model output at this density — expected
and consistent with all prior transport-equivalence findings (§4 in
`CURRENT_STATE.md`).

## 6. Stage 6-7: Paired Performance A/B (Job 58458525)

3 pairs, alternating `host_staged`/`cuda_aware` on the **same** 2-GPU
allocation (avoids node-to-node A100 variance per the Prompt 37/40 lesson).
Each rep: 18 simulated minutes, `bdt=36s` → 30 `dyn_core` calls. All 6 reps
`RC=0`. `fv_timing_mod` block (seconds, /30 calls = ms/call):

| Rep | Mode | DYN_CORE (s) | COMM_DSW (s) | COMM_TOTAL (s) |
|---|---|---|---|---|
| pair1 | host_staged | 140.8502 | 11.4899 | 20.8907 |
| pair1 | cuda_aware  | 140.7159 | 11.2237 | 20.5669 |
| pair2 | host_staged | 141.5440 | 12.3291 | 21.9039 |
| pair2 | cuda_aware  | 141.6443 | 11.8603 | 21.4061 |
| pair3 | host_staged | 141.0516 | 11.1965 | 20.7495 |
| pair3 | cuda_aware  | 141.0117 | 11.0066 | 20.1948 |

Median-of-3, ms/call:

| Metric | host_staged | cuda_aware | improvement% |
|---|---|---|---|
| DYN_CORE | 4701.72 | 4700.39 | 0.03% |
| COMM_DSW | 383.00 | 374.12 | 2.32% |
| COMM_TOTAL | 696.36 | 685.56 | 1.55% |

All deltas are **<5%**, and the primary `DYN_CORE` metric is statistically
flat (0.03%, smaller than the ~23ms pair-to-pair spread already present
within `host_staged` alone). `COMM_DSW`/`COMM_TOTAL` show a small,
consistent edge to `cuda_aware` (~1.5-2.3%), but not enough to change the
classification. Direction is not even consistent per-pair for `DYN_CORE`
(pair2 `cuda_aware` is marginally *slower* than pair2 `host_staged`) —
consistent with run-to-run noise, not a real effect.

**Classification: Case C — VIABLE BUT NOT PERFORMANCE-SIGNIFICANT.**

## 7. Stage 8: Operational 1-GPU vs 2-GPU Summary (Informal — Not a Controlled A/B)

Not a rigorous cross-job comparison (different jobs/times/possibly
different physical nodes — see the Prompt 37/40 node-variance lesson in
`CURRENT_STATE.md` §6). Reported as a directional signal only.

| Config | Transport | DYN_CORE ms/call | vs. Fortran (7155.63 ms/call) |
|---|---|---|---|
| 1 GPU / 6 ranks (Prompt 37) | host_staged | 9089.05 | 1.270x SLOWER (documented regression) |
| 2 GPU / 3-ranks-per-GPU (Prompt 46) | host_staged | 4701.72 | 0.657x — ~34% FASTER |
| 2 GPU / 3-ranks-per-GPU (Prompt 46) | cuda_aware | 4700.39 | 0.657x — ~34% FASTER |

Simply spreading the existing 6 ranks across 2 A100s (3/GPU) instead of 6
ranks on 1 A100 appears to cut C512's CUDA `DYN_CORE` time roughly in half,
which would flip the previously-documented C512 CUDA-vs-Fortran regression
(§6/§7 in `CURRENT_STATE.md`) into a large CUDA win — **independent of
transport choice** (host_staged alone shows the same gain as cuda_aware).
This suggests the C512 regression may be substantially a per-GPU
rank/resource-contention effect at 6-ranks-per-A100 density, not solely
the MPI transport-protocol cost growth root-caused in Prompts 38-40 (which
was diagnosed specifically at the 1-GPU/6-rank density). **This is a
strong, unplanned, directly-relevant lead, but re-deriving or re-measuring
the C512 regression under a controlled same-node 1-GPU-vs-2-GPU A/B is out
of Prompt 46's scope** (topology/decomposition-change territory) and is
flagged as the top candidate for a dedicated future prompt rather than
acted on now.

## 8. Decision

**Case C — CUDA-aware MPI transport is numerically and operationally VIABLE
at C512 under 3-ranks/GPU (2-A100) density (reversing Prompt 45's OOM
finding at 6-ranks/1-GPU density), but is NOT performance-significant
relative to host-staged at this density** (all measured deltas <5%,
primary DYN_CORE metric statistically flat). **Recommendation: use
host-staged transport for C512 at 2-GPU density** — it is simpler
(no `I_MPI_OFFLOAD`/Intel-MPI GPU-direct dependency), equally fast, and
already the established default. cuda_aware's viability is a useful new
fact (removes the "always OOMs at C512" caveat when rank density is
lowered) but does not by itself justify switching the default transport.

## 9. Deferred Follow-Ons (Not Started — Confirm Scope Before Starting)

1. **Controlled 1-GPU-vs-2-GPU C512 performance A/B** (same-node/same-job
   where feasible) to formally test whether lowering rank density alone
   (independent of transport) resolves the C512 CUDA-vs-Fortran regression
   documented in §6/§7 — the strongest new lead from this prompt (§7
   above). This is a topology/decomposition-adjacent question and should
   be scoped explicitly as its own prompt.
2. Intel MPI GPU-buffer-pool sizing controls (`I_MPI_*` env vars) — still
   not investigated; now lower priority given cuda_aware is not
   performance-significant even where it fits in memory.
3. Multi-node scaling, further resolution ladder, cuda_aware at
   4-ranks-or-more/GPU densities — all out of scope, not attempted.

## 10. Artifacts

- Jobs: 58454451 (Stage 2/3, host_staged 2-GPU), 58456709 (Stage 4,
  cuda_aware 2-GPU viability + memory poll), 58458525 (Stage 6/7, paired
  A/B, 6 reps).
- New sbatch/job scripts and fv3.j copies under
  `<fortran_perf>/experiments/{p46_c512_2gpu_host_staged,
  p46_c512_2gpu_cuda_aware,p46_c512_2gpu_perf}/`.
- Per-rep timing logs: `<fortran_perf>/experiments/p46_c512_2gpu_perf/
  perf_logs/pair{1,2,3}_{host_staged,cuda_aware}.log`.
- GPU memory CSVs: `p46_c512_2gpu_host_staged_gpumem_58454451.csv` (bogus,
  tooling bug), `p46_c512_2gpu_cuda_aware_gpumem_58456709.csv` (valid,
  fixed poller).
