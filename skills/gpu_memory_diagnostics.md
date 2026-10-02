# GPU Memory Diagnostics

## Purpose

Track GPU (or other accelerator) memory headroom as a first-class,
continuously measured constraint throughout a porting effort, so a
numerically-correct and performant port is not discovered to be
production-unusable only after it ships.

## Use When

- Before declaring any GPU port "production-ready," at the actual target
  rank/device density.
- Whenever scaling up grid size, rank count, or ranks-per-device.
- When an OOM or unexpected allocation failure occurs.

## Inputs Needed

- A static inventory mechanism: ability to sum named persistent
  allocations (buffers that live for the whole run, not transient
  per-call allocations).
- A runtime telemetry mechanism (e.g. `nvidia-smi`, `cudaMemGetInfo`, or
  platform equivalent) that can be sampled *during* the run, not only
  after.
- Knowledge of the target production rank/device density (how many
  processes/ranks will actually share each device in production).

## Procedure

1. Build a static memory inventory: list every persistent (whole-run)
   allocation by name/purpose and its size, summed per rank/process.
2. Separately measure runtime memory via a *concurrent* background
   poller that samples throughout the run — never rely on a single
   post-exit snapshot (the accelerator context is often already destroyed
   by the time a job exits, making a post-hoc read near-idle and
   meaningless).
3. If multiple ranks/processes share one device (e.g. via a context-
   sharing mechanism), add explicit synchronization (e.g. a barrier) before
   each telemetry checkpoint — an unsynchronized read across shared-device
   processes cannot be cleanly attributed to any one lifecycle stage.
4. Test memory headroom at the actual target production density (rank
   count per device, grid size) — a port that fits comfortably at a
   smaller test grid or lower rank density is not evidence it fits at
   production scale.
5. If memory is insufficient, prefer a structural fix (change how ranks
   map to devices — spread across more devices) over micro-optimizing
   individual buffer sizes, if an architectural fix is available and
   affordable — this is usually both cheaper in engineering time and more
   robust to future growth.
6. Before concluding memory scales a particular way (e.g. "per-rank" vs.
   "per-device"), vary rank-density/device-count independently and
   measure each configuration — do not assume a scaling law without
   isolating the variable.
7. Re-run the memory check whenever the target scale (grid size, rank
   count) changes materially — a result at one scale does not
   automatically generalize to a larger one.

## Validation / PASS Criteria

- A static inventory and a concurrent runtime measurement both confirm
  headroom at the real target production density.
- The measurement methodology (concurrent poller, synchronized
  checkpoints if shared-device) is stated alongside the result.

## STOP Conditions

- Headroom is insufficient at the real target density — do not ship the
  port as unconditional default; gate it (conditional default based on
  device count/density) or apply a structural fix before considering it
  production-ready.
- Only a post-exit snapshot is available — do not trust it; arrange a
  concurrent in-run measurement before drawing any conclusion.

## Common Failure Modes

- Checking memory only after a job completes and concluding the port is
  memory-safe, when the context was already torn down and the reading is
  near-idle, not representative of in-run peak usage.
- Taking a single unsynchronized reading across several device-sharing
  processes and attributing it to a specific pipeline stage — the reading
  reflects whichever processes happened to be at whichever stage at that
  instant, not a clean per-stage attribution.
- Assuming memory scales "per device" when it actually scales "per
  rank/context" (or vice versa) without testing both axes independently.
- Chasing per-buffer size optimizations when a structural fix (more
  devices, different rank mapping) is available, cheaper, and more
  robust.
- Testing memory headroom only at a smaller development grid and
  extrapolating to production scale without re-measuring.

## Cost-Control Guidance

- Build the static-inventory-plus-concurrent-poller mechanism once; reuse
  it for every subsequent memory check rather than inventing a new ad hoc
  check each time.
- Decide whether a structural fix (more devices) or buffer-level
  optimization is cheaper in engineering time before starting either —
  don't default to the more labor-intensive option.
- Only re-run the full memory check at scales that materially matter for
  the current decision (e.g. the actual production density), not every
  intermediate test grid.

## Reusable Prompt Template

```
Check GPU memory headroom for <port/subsystem> at the real target
production density: <rank count>/<device count>. Build a static
inventory of persistent allocations, and a concurrent (in-run, not
post-exit) runtime measurement, synchronized across ranks if the devices
are shared. Report whether headroom exists at this exact density. If
not, evaluate whether spreading ranks across more devices (structural
fix) resolves it, and report the resulting per-device memory and any
performance side effect. Do not conclude memory-safety from a smaller
test grid or a post-exit snapshot.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** `State_To_FV`'s
CUDA port was ~440 MiB short per rank at C512/6-ranks-per-1-A100 density —
discovered via a dedicated memory inventory, not a timing run. The fix was
architectural: spreading the same 6 ranks across 2 GPUs (3 ranks/GPU)
resolved the shortfall, validated at 19.68 GiB/GPU with memory to spare,
bitwise-identical results, and a modest incidental TOTAL improvement
(1.120x) — not because 2-GPU is inherently faster, but because the memory
unblock allowed the already-correct port to actually run. Separately,
rank-density changes (1/2/3 GPUs at fixed rank count) were shown to have
no effect on timing at all — memory and performance are decoupled axes
here. A documented lesson: "A post-hoc `nvidia-smi` checkpoint after a job
exits reads near-idle, not the in-run footprint" — a concurrent poller was
required instead (`docs/agent_work_summary.md` §7).
