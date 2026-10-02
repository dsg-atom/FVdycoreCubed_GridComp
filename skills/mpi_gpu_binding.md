# MPI / Multi-GPU Binding

## Purpose

Correctly and verifiably bind MPI ranks to GPU devices across single- and
multi-node configurations, avoiding the global-rank-modulo anti-pattern
that silently breaks beyond one node.

## Use When

- Scaling a validated single-GPU path to multiple GPUs and/or multiple
  nodes.
- Introducing or auditing any device-selection logic (`dev =
  f(rank, device_count)`).
- Investigating a correctness or performance anomaly that only appears at
  multi-node scale.

## Inputs Needed

- The MPI implementation's node-local rank/communicator-splitting
  mechanism (e.g. `MPI_Comm_split_type` with `MPI_COMM_TYPE_SHARED`).
- Knowledge of the platform's device-visibility mechanism (e.g.
  `CUDA_VISIBLE_DEVICES`) and how the scheduler sets it per node.
- The platform's resource-cap model: confirm whether quotas
  (GPU count, memory) are enforced per-node or job-wide/summed — do not
  assume one without checking.

## Procedure

1. Never select a device using the global MPI rank modulo device count
   (`dev = mod(global_rank, device_count)`) — this is only correct within
   a single node and silently misassigns devices the moment more than one
   node is involved.
2. Split the global communicator by shared-memory/node locality (e.g.
   `MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)`) to obtain a
   node-local rank; use that node-local rank (modulo the node's visible
   device count) for device selection.
3. Validate device binding explicitly: have each rank report (device
   selected, node-local rank, hostname) and confirm the mapping is correct
   and collision-free within each node.
4. Test at both single-node, multi-GPU and genuine multi-node
   configurations — a single-node test cannot catch the global-rank-
   modulo anti-pattern; it only breaks across nodes.
5. If using a shared-device-context mechanism (e.g. MPS) to let multiple
   ranks per device run concurrently, confirm its startup/lifecycle
   mechanism is actually broadcast to every node in a multi-node job
   (many such mechanisms default to head-node-only) — use an explicit
   node-broadcast launch (e.g. `srun --overlap --nodes=N
   --ntasks-per-node=1`) if it is not.
6. Confirm resource-cap scope (per-node vs. job-wide/summed) before
   budgeting a multi-node job's device/memory request — a job-wide cap
   must be budgeted against the whole job's total request, not computed
   per-node and multiplied.
7. Re-validate bitwise-identical numerical results at every new binding
   configuration tested (1 device, N devices same-node, N devices
   multi-node) — binding changes are a correctness risk, not just a
   performance one.

## Validation / PASS Criteria

- Device selection uses a node-local rank, never the global rank, in any
  code reachable from a multi-node configuration.
- Binding has been validated (correct, collision-free mapping) at both
  single-node multi-GPU and genuine multi-node configurations.
- Numerical results are bitwise-identical across all tested binding
  configurations.
- Any shared-device-context lifecycle mechanism (e.g. MPS) is confirmed
  active on every node of a multi-node job, not assumed from head-node
  behavior.

## STOP Conditions

- Device-selection logic uses the global rank directly — stop and fix
  before any multi-node test; this is a known, confirmed-recurring
  anti-pattern, not a hypothetical risk.
- A shared-device-context mechanism's startup state has not been confirmed
  on every node — do not assume it started correctly on secondary nodes
  just because the job didn't error.
- Resource-cap scope (per-node vs. job-wide) is unconfirmed for the
  platform — look it up before submitting a multi-node job that might
  silently be rejected or throttled.

## Common Failure Modes

- `dev = mod(global_rank, device_count)` working correctly in every
  single-node test (because it happens to coincide with node-local rank
  when there's only one node) and then silently misbinding the moment a
  second node is added — the single-node tests give false confidence.
- Assuming a shared-device-context (MPS) lifecycle script run once at job
  start covers every node, when it is actually head-node-only by
  construction.
- Computing a multi-node resource request per-node and multiplying by node
  count, when the platform's actual quota is a job-wide sum — leading to
  either a silently throttled job or a rejected submission.
- Treating a correct single-node multi-GPU result as sufficient evidence
  for multi-node correctness.

## Cost-Control Guidance

- Validate single-node multi-GPU binding first (cheaper, faster to debug)
  before attempting the more expensive multi-node configuration.
- Look up the platform's resource-cap model once and record it, rather
  than discovering it empirically via a rejected/throttled job submission.
- Reuse one binding-validation diagnostic (report device/node-local-rank/
  hostname per rank) across every new configuration tested, rather than
  writing a new one each time.

## Reusable Prompt Template

```
Validate GPU device binding for <workload> across <N> devices, first at
single-node multi-GPU, then at genuine multi-node (<M> nodes). Ensure
device selection uses the node-local rank (via a shared-memory
communicator split), never the global MPI rank. Report, per rank: node-
local rank, hostname, device selected — confirm no collisions within a
node. If using <shared-device-context mechanism>, confirm it is active on
every node, not just the launching node. Confirm the platform's resource-
cap scope (per-node vs job-wide) before sizing the request. Re-validate
numerical bitwise-identity at each configuration tested.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** FV3's early
multi-GPU device selection used `dev = mod(comm_rank, device_count)`,
correct only within one node. Node-local GPU binding via
`MPI_Comm_split_type` was implemented and validated bit-identical both
single-node and across a genuine 2-node test (Prompt 107). The platform's
`alla100` QOS caps GPU/memory **job-wide** (summed across nodes), not
per-node — budgeted explicitly rather than assumed. CUDA MPS's lifecycle
script (`fv3_mps_lifecycle.sh`) is head-node-only by default; genuine
multi-node MPS required an explicit `srun --overlap --nodes=N
--ntasks-per-node=1` broadcast (`docs/agent_work_summary.md` §8).
