# Prompt 107 — Node-Local GPU Device Binding (Multi-Node Readiness)

**Status: CLOSED. NODE-LOCAL GPU BINDING VALIDATED.**

Branch `optimization/cuda-fv3-performance`. No commit/push performed (per
task constraint) — all source changes remain in the working tree of
`FVdycoreCubed_GridComp` (this repo) and the separate
`GEOSfvdycore_fortran_perf` checkout.

## 1. Objective

Replace the single-node-only device-selection rule
`dev = mod(global_comm_rank, device_count)` in
`cuda/interface/fv3_gpu_dyncore_bridge.F90` with a node-local rule using
`MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, key, info, newcomm)` +
`dev = mod(local_rank, device_count)`, so that genuine multi-node runs bind
GPUs correctly (global rank is not node-local; the old rule silently
mis-binds beyond one node). Preserve all single-node behavior exactly; no
numerical, halo, or CUDA-aware-MPI changes.

## 2. Implementation (recap)

`fv3_gpu_bridge_init` now derives a node-local sub-communicator via
`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED, ...)`, reads its rank as
`local_rank`, and selects `dev = mod(local_rank, device_count)` where
`device_count` is `fv3_gpu_device_count()` (Prompt 106's cached getter,
unchanged). On a single node `local_rank == global comm_rank` for every
rank, so the new rule is bit-identical to the old one there — this is the
regression contract validated below.

An opt-in (default OFF) diagnostic, gated by env var
`RUN_CUDA_FV3_LOG_GPU_BINDING`, emits one line per rank:
```
FV3_GPU_BIND host=<hostname> global=<rank> local=<local_rank>/<local_size> devices=<device_count> selected=<dev>
```

**Bug found and fixed this session's build**: the diagnostic originally
used `call mpp_error(NOTE, trim(msg))`. FMS's `mpp_error` only prints
NOTE/WARNING from PE 0 by default, so only 1 of 6 bind lines ever appeared
regardless of topology — not a binding defect, a logging defect. Fixed by
switching to the codebase's existing per-rank pattern,
`write(0,'(A)') trim(msg)` (precedent: `FV_StateMod.F90`'s
`P99_EXIT_DUMP`). Confirmed fixed: `BIND_COUNT=6` in every run below (was
1 before).

## 3. Build / Provenance

Rebuilt `GEOSfvdycore_fortran_perf/build` (Intel-MPI production stack:
`comp/gcc/12.3.0`, `comp/intel/2024.2.0`, `mpi/impi/2021.13`,
`nvidia/nvhpc-byo-compiler/24.7`, `I_MPI_CXX=g++`):
```
cmake --build . --target StandAlone_FV3_Dycore.x -j 4
```
New binary `build/bin/StandAlone_FV3_Dycore.x`:
- sha256 `14efaa865e2db908b451a0a586a256288833e1fc51a19014d90b2f24352bbff2`
- Propagated via `cp -f` to `p104_correctness_cuda/`,
  `p104_correctness_fortran/`, `p107_multinode_binding/`; verified by
  `sha256sum` in every job before running (`*_PROVENANCE` lines in each
  log).

## 4. Test Matrix and Results

All runs use the standing `StandAlone_FV3_Dycore.x` correctness/smoke
harness (`fv3.j` template): C512, L10 (idealized `AK`/`BK`, `test_case=14`,
adiabatic), 6 MPI ranks (`FV_NX=1`/`FV_NY=6`), `hord_mt/vt/tm=6`,
`hord_dp=-6`, `a_imp=1.0`, `k_split=1`/`n_split=3`, `DYN_DATE_COUNT=30`
steps. This is a fast correctness/binding harness, **not** the dedicated
performance A/B benchmark used for the absolute ms/call figures in
`CURRENT_STATE.md` §5 (different grid depth/step count/methodology) — see
§7 below for how its timing numbers are used.

| Stage | Topology | Job | rc | DYN steps | Abort | BIND_COUNT | State_To_FV T/F |
|---|---|---|---|---|---|---|---|
| 7 (default) | 1 node, 6 ranks, 2 GPU, MPS | 58658676 | 0 | 30 | 0 | 6 | 180/6 (CUDA default, device_count=2) |
| 7b (FORCE_FORTRAN) | 1 node, 6 ranks, 2 GPU, MPS | 58658676 | 0 | 30 | 0 | 6 | 0/186 (all Fortran) |
| 8 | 1 node, 6 ranks, 1 GPU, MPS | 58658840 | 0 | 30 | 0 | 6 | 0/186 (Fortran default, device_count=1) |
| 9/10 | 2 nodes, 3 ranks/node, 2 GPU/node, MPS both nodes | 58658921 | 0 | 30 | 0 | 6 | 180/6 (CUDA default, device_count=2) |

### Stage 7 — single-node 2-GPU regression (new rule vs. old rule, identical on 1 node)
Bind mapping: local ranks 0,2,4 → GPU0; 1,3,5 → GPU1 — **bit-identical to
Prompt 103/106's established single-node mapping.** State_To_FV correctly
defaults to CUDA (`device_count=2`).

### Stage 7b — escape hatch re-check, same topology
`RUN_CUDA_FV3_STATE_TO_FV_FORCE_FORTRAN=1` still selects Fortran
(0/186) — escape hatch undisturbed by the binding change.

### Stage 8 — 1-GPU degenerate-case regression
All 6 ranks (`local_rank` 0-5) select device 0 (`mod(local_rank,1)==0`
for every rank) — confirms the degenerate 1-GPU case is unaffected.
State_To_FV correctly defaults to Fortran (`device_count=1<2`, Prompt
106 policy unchanged).

### Stage 9/10 — genuine multi-node divergence proof
Bind lines (sorted by global rank):
```
FV3_GPU_BIND host=warpa005 global=0 local=0/3 devices=2 selected=0
FV3_GPU_BIND host=warpa005 global=1 local=1/3 devices=2 selected=1
FV3_GPU_BIND host=warpa005 global=2 local=2/3 devices=2 selected=0
FV3_GPU_BIND host=warpa009 global=3 local=0/3 devices=2 selected=0
FV3_GPU_BIND host=warpa009 global=4 local=1/3 devices=2 selected=1
FV3_GPU_BIND host=warpa009 global=5 local=2/3 devices=2 selected=0
```
Node 1 (`warpa009`, global ranks 3,4,5) selects devices `0,1,0` under the
new node-local rule (`local%2` = 0,1,0). The **old** global-rank rule
(`global%2`) would have given `1,0,1` for the same ranks — a different
answer. This is the direct empirical proof that the fix changes behavior
on node 1 and is not a coincidental match; it is only correct because
`local_rank` (not `global rank`) is used. State_To_FV correctly defaults
to CUDA (`device_count=2`, read per-node, same on both nodes).

## 5. Platform Discoveries (new)

- **`alla100` QOS caps a single job's summed GPU/memory request**, not
  per-node: `sacctmgr show qos alla100` → `MaxTRES=cpu=288,gres/gpu=4,
  mem=510000M`. `scontrol show node warpa003` confirms each `gpu_a100`
  node has `RealMemory=510000M`/`gres:gpu:4`. Requesting
  `--gres=gpu:4 --nodes=2` (job-total 8) hit `QOSMaxGRESPerJob`; omitting
  `--mem` on a 2-node job defaulted toward full-node memory × 2 and hit
  `QOSMaxMemoryPerJob`. **Fix used here**: `--gres=gpu:2 --nodes=2`
  (job-total 4, at the cap) + explicit `--mem=60G`. Any future multi-node
  job under this account/QOS must budget GPU/memory as a job-wide total,
  not per-node.
- **Multi-node CUDA MPS requires an explicit per-node broadcast.** The
  existing `fv3_mps_lifecycle.sh` starts/stops the MPS control daemon only
  in the batch script's own shell, which runs on the allocation's head
  node only. Fixed here via
  `srun --overlap --nodes=2 --ntasks-per-node=1 bash -c '...'` wrapping
  `nvidia-cuda-mps-control -d` (and matching `quit` at teardown), with
  `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` exported in the
  parent shell so Intel MPI's environment forwarding carries them to
  every rank. This pattern is reusable for any future multi-node MPS job.
- **Stale `EXPDIR` bug is more than harmless.** `CURRENT_STATE.md` §7
  previously characterized this class of bug as affecting only
  `scratch_*.log` capture. This session's `p107_multinode_binding/fv3.j`
  had a stale `setenv EXPDIR .../p104_correctness_cuda` (copy-paste
  leftover). Because Stage 8 (job 58658840) was running concurrently and
  legitimately owns `p104_correctness_cuda` — and executes
  `rm -rf savepoints scratch_*` at its own start — the two jobs collided:
  Stage 9's in-progress scratch directory was deleted mid-run by Stage
  8's cleanup, producing a real `MPI_Abort`
  (`MAPL_HistoryGridComp.F90 status=19`) and a
  `slurmstepd: couldn't change working dir` error, not just a log-capture
  gap. Fixed by pointing `EXPDIR` at the job's own rundir
  (`p107_multinode_binding`); Stage 8's own results were independently
  confirmed clean/unaffected; Stage 9 was resubmitted alone (job
  58658921) and completed cleanly. **Any reused/copied `fv3.j` must be
  grepped for `setenv EXPDIR` before submission — this can cause a real
  crash under concurrency, not just harmless log noise.**

## 6. Numerical Sanity (Stage 11)

The 6-rank/one-tile-per-rank decomposition (`comm_size==6`) is completely
unchanged by this fix — only the integer GPU-index computed at
`fv3_gpu_bridge_init` changed. `[P99_EXIT_DUMP]` sample fields
(`u`/`v`/`pe`/`pt`/`delz`/`pkz`/`ps`) at matching PE indices are
**bitwise identical across all four arms** (Stage 7 default, Stage 7b
Fortran-forced, Stage 8 1-GPU, Stage 9 multi-node) — e.g. `pe=3` gives
`u_sample=-1.8806012728925989E-14`, `pt_sample=2.2094705734315698E+02`
identically in every arm. No numerical divergence was introduced by the
binding change, as expected — this is a routing-only change with no
numerical-kernel or halo-semantics impact, so no full T/U/V/PS savepoint
comparison (Prompt 103/105 methodology) was necessary here.

## 7. Memory Balance (Stage 12)

Post-hoc `nvidia-smi` checkpoints (`P107_S9_gpu_pre.csv`/`_post.csv`) read
near-idle (~4 MiB used both GPUs, both nodes, before and after) — expected
per the documented "post-hoc checkpoint reads near-idle" lesson (context
already destroyed by checkpoint time). The concurrent in-run poller
(`P107_S9_poll.log`, `srun --ntasks-per-node=1` broadcast every 5s) is the
only trustworthy source: GPU index 0 (2 ranks sharing under MPS) peaked at
13712 MiB used; GPU index 1 (1 rank alone) peaked at 6860 MiB — ratio
≈2.0, consistent with the expected 2:1 rank-sharing pattern under MPS.
(Absolute values reflect this test's small L10 grid, not a C512-production
memory footprint — not comparable to the 19.68 GiB/GPU figure from
Prompt 103's full production config.)

## 8. State_To_FV Interaction (Stage 13)

Prompt 106's `device_count>=2` conditional-default policy is unaffected by
this change and continues to work correctly under every topology tested:
Fortran default at `device_count=1` (Stage 8), CUDA default at
`device_count=2` whether single-node (Stage 7) or genuinely multi-node
(Stage 9) — `device_count` is read per-node via the unchanged
`fv3_gpu_device_count()` getter, which is exactly the right behavior since
GPU visibility (and hence `device_count`) is inherently node-local. No
interaction gap found.

## 9. CUDA_VISIBLE_DEVICES / Launcher Interaction (Stage 14)

`CUDA_VISIBLE_DEVICES` was left unset in every run. Slurm's per-job GPU
cgroup restriction correctly exposed exactly the allocated GPU count as
device indices `0..N-1` on each node independently: `devices=1` at Stage
8, `devices=2` at both Stage 7 (single node) and Stage 9 (each of the two
nodes). No interaction with Intel MPI's rank placement or `esma_mpirun`
was needed or observed — Slurm's cgroup-based device remapping is
sufficient, node-local, and orthogonal to the new `MPI_Comm_split_type`
logic (which operates on MPI ranks, not device visibility).

## 10. Performance Sanity (Stage 15, <5% regression guardrail)

Same-harness relative comparison (this L10 smoke config is not the
absolute-ms production benchmark — see §4 caveat — so the check here is
relative, not an absolute match to Prompt 103's production figures):

| Comparison | TOTAL | FV_DYN_LOOP | DYN_CORE |
|---|---|---|---|
| Stage 7 (CUDA default) ms | 109.3120 | 32.7615 | 14.2096 |
| Stage 7b (Fortran forced) ms | 110.3292 | 33.1609 | 14.4196 |
| Stage7/Stage7b ratio | 0.9908 | 0.9880 | 0.9854 |

CUDA is ~1-1.5% faster than Fortran at this small grid on the identical
2-GPU/single-node topology post-fix — same direction (CUDA
faster-or-comparable) as every prior production finding (Prompt 94-106),
well inside the <5% regression guardrail. No regression from the binding
change itself.

Cross-topology sanity (not a strict regression check, informational only):
Stage 8 (1-GPU) TOTAL 133.4268 ms is ~22% slower than Stage 7 (2-GPU) TOTAL
109.3120 ms — same direction as Prompt 103's validated 2-GPU-faster
finding (magnitude differs because this is a much smaller L10 config, not
the same benchmark). Stage 9 (multi-node, 2 nodes × 2 GPU, lower per-GPU
rank density than Stage 7's single-node 3-ranks/GPU) TOTAL 97.9105 ms is
faster than Stage 7 — expected, since Stage 9 has more GPU headroom per
rank (up to 1-2 ranks/GPU vs. 3 ranks/GPU), not a regression signal.

**Conclusion: no performance regression from the node-local binding
change.** All ratios are consistent with pre-existing, already-validated
CUDA/Fortran and 1-GPU/2-GPU performance relationships.

## 11. Instrumentation Cleanup (Stage 16)

`RUN_CUDA_FV3_LOG_GPU_BINDING` remains opt-in/default-OFF (follows the
project's `env_flag_enabled(name)` convention, same as
`RUN_CUDA_FV3_FUSE_HALO_EDGES` etc.) — no unconditional logging was added.
The `write(0,...)` fix is contained entirely inside the existing
diagnostic block; no further cleanup needed.

## 12. Validated Scope / C1024 Readiness (Stage 17/18)

| Axis | Status |
|---|---|
| Single-node (1, 2 GPU) | VALIDATED (Stage 7/7b, 8) — bit-identical to Prompt 103/106 |
| 1-GPU degenerate case | VALIDATED (Stage 8) |
| Genuine multi-node (2 nodes) | VALIDATED (Stage 9/10) — divergence proof against the old rule |
| Intel MPI 2021.13 | VALIDATED (all 4 jobs, production stack) |
| CUDA MPS, single-node | VALIDATED (Stage 7/7b, 8 — pre-existing `fv3_mps_lifecycle.sh`) |
| CUDA MPS, multi-node | VALIDATED (Stage 9/10 — new `srun --overlap` broadcast pattern) |
| Numerical impact | NONE (bitwise-identical exit-dump samples across all arms) |
| Performance impact | NONE / mildly positive (§10), no regression |
| State_To_FV interaction | CORRECT at every device_count tested |

`MPI_Comm_split_type`-based node-local binding is now the active mechanism
in `fv3_gpu_bridge_init` and is IMPLEMENTED, COMPILED, GPU EXECUTED, and
NUMERICALLY VALIDATED (routing-only — no numerical-kernel change to
validate against a savepoint baseline). This closes the multi-node
device-mapping limitation documented in Prompt 106 (`CURRENT_STATE.md`
§7/§9) and removes the last identified single-node assumption blocking a
genuine multi-node scaling attempt.

**Remaining blocker before an actual C1024 attempt**: this fix addresses
GPU *device binding* only. `CURRENT_STATE.md` §7's separate, larger item —
**the CUDA dyn_core bridge hardcodes `comm_size==6`** (one tile/rank, no
intra-tile/same-face halo path) — is untouched and unresolved by Prompt
107. C1024 needs more than 6 ranks to stay within per-rank memory bounds
(Prompt 102's finding), which requires a new halo-exchange architecture,
not addressed here. Node-local binding is a **necessary prerequisite**
for multi-node C1024 but does not by itself unblock it.

**Architecture now mapped (Prompt 108, design/mapping only, no source
change)**: `docs/performance/p108_c1024_intra_tile_halo_architecture_2026-09-30.md`
designs the generalization needed to remove the `comm_size==6` assumption
(target: 2×2 ranks/face, 24 ranks). Found that FMS/`fv_mp_mod.F90`'s
`domain_decomp` already supports `layout(1,2)>1` per tile natively (reused,
not reinvented); the CUDA-side gap is confined to
`fv3_gpu_halo_topology.hpp/.cpp`'s single-neighbor-per-edge `TopologyView`
data model and the bridge's `comm_size==6` gate. Classification: MODERATE
HALO GENERALIZATION. Recommended next bounded task: implement/validate
Stage-15 Phases 1-4 (CUDA-side topology/transport generalization only, small
grid, single node) — explicitly not the bridge `comm_size` gate (Phase 6) or
a real 24-rank/C1024 run (Phase 8), each its own future task.

## 13. Files Changed

- `cuda/interface/fv3_gpu_dyncore_bridge.F90` — `MPI_Comm_split_type`-based
  node-local binding (implemented prior session), `write(0,...)` logging
  fix (this session's rebuild target). No numerical/halo/CUDA-aware-MPI
  change.
- `GEOSfvdycore_fortran_perf/build/*` — rebuilt (separate checkout, not
  this repo).
- `GEOSfvdycore_fortran_perf/experiments/p107_*.sbatch` — new/updated job
  scripts (separate checkout, not source).
- `GEOSfvdycore_fortran_perf/experiments/p107_multinode_binding/fv3.j` —
  `EXPDIR` fix (separate checkout, not source).
- No changes to this repo's committed source beyond what was already
  in the working tree before this session (bridge file was already
  modified pre-session; this session only rebuilt/validated it).
- **No commit or push performed.**
