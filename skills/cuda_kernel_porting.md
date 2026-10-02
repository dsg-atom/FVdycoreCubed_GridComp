# CUDA (or Other Accelerator) Kernel Porting

## Purpose

Port one bounded numerical subsystem from the legacy/host implementation to
a GPU (or other accelerator) kernel while preserving exact (or explicitly
justified) numerical behavior, without destabilizing the surrounding
persistent-state architecture.

## Use When

- You have a validated host-language reference implementation (see
  `fortran_cpp_abi_integration.md`/`fortran_baseline_harness.md`) and a
  specific, bounded subsystem chosen for acceleration.
- A production profile (see `gpu_performance_profiling.md`) has identified
  this subsystem as a worthwhile target — not before.

## Inputs Needed

- A validated host-language (CPU) implementation of the exact target
  subsystem, already proven numerically equivalent to the legacy baseline.
- The surrounding persistent-state/residency architecture (what stays
  device-resident across calls, what boundaries require
  synchronization) — know this before writing a kernel, so the kernel
  doesn't introduce an unnecessary host round-trip.
- An explicit acceptance bar (tolerance for correctness; minimum required
  speedup if performance is in scope) — decided before implementation, not
  after seeing the numbers.

## Procedure

1. Confirm the target subsystem's host-language reference implementation
   is already validated against the legacy baseline, in isolation — never
   port directly from legacy source to a kernel, skipping the host
   reference step; the host reference step is what lets you validate
   algorithm translation separately from device-execution concerns.
2. Choose kernel granularity to match existing persistent-state boundaries
   — avoid new host/device round-trips around every individual kernel;
   kernels should fit inside the existing stay-on-device data flow, not
   dictate a new one.
3. Implement the kernel preserving the exact algorithmic structure of the
   validated host reference (same operation order where numerically
   significant) unless a reordering is explicitly justified and
   documented as changing only floating-point reassociation, not the
   algorithm.
4. Validate the kernel in isolation (fixture/unit-test level) against the
   host-language reference before any integration — bit-for-bit or at an
   explicitly justified round-off/ULP tolerance (see
   `numerical_validation.md`).
5. Integrate behind a default-OFF selection flag at the existing stable ABI
   boundary — do not require legacy callers to change to use the new path.
6. Re-validate numerically after integration, at the real production entry
   point, not just the isolated fixture — integration can introduce new
   divergence (data layout mismatch, uninitialized halo/padding, implicit
   synchronization gaps) that an isolated unit test cannot see.
7. Only after correctness is re-confirmed post-integration, profile the
   real production workload (not a microbenchmark) to measure actual
   contribution, local and end-to-end (see `gpu_performance_profiling.md`).
8. Apply the KEEP/PORT/DEFER/STOP decision (see
   `modernization_stop_go_decisions.md`) using the pre-declared acceptance
   bar — not a post-hoc rationalization of whatever number you got.
9. Run a memory-headroom check at the real target device/rank density
   before declaring the port production-ready (see
   `gpu_memory_diagnostics.md`) — a kernel can be numerically perfect and
   still unusable at production scale due to memory.

## Validation / PASS Criteria

- Isolated kernel-vs-host-reference numerical validation passes at the
  declared tolerance.
- Post-integration numerical validation (at the real production entry
  point) passes at the same tolerance — not just the isolated check.
- Memory-safe at the real target device/rank density (not just a smaller
  test grid).
- Default (flag-unset) legacy behavior is unchanged, confirmed by actually
  running an existing caller/test with the flag unset.

## STOP Conditions

- Isolated validation (step 4) fails — do not integrate.
- Post-integration validation (step 6) fails even though isolated
  validation passed — do not profile performance on an unvalidated
  integration; find and fix the integration-specific divergence first.
- A memory-headroom check (step 9) fails at the real target density — do
  not declare the port production-ready; resolve via
  `gpu_memory_diagnostics.md` (architectural fix) before shipping, even if
  correctness and performance both already look good.
- Measured production-scale contribution does not meet the pre-declared
  acceptance bar — apply STOP per `modernization_stop_go_decisions.md`,
  do not quietly lower the bar after seeing the number.

## Common Failure Modes

- Writing the kernel directly against legacy source, skipping host
  reference validation — any resulting bug is now entangled across both
  "did I translate the algorithm correctly" and "did I execute it
  correctly on the device," doubling debugging cost.
- Validating only in isolation and assuming integration can't introduce new
  divergence — it can (data layout, padding/halo initialization,
  synchronization ordering).
- Treating a local kernel speedup as proof of end-to-end benefit without
  measuring the real production timer.
- Declaring a port "production-ready" without a memory check at the real
  target rank/device density — a port can be numerically perfect and
  performant at small scale yet unusable in production for a reason
  invisible to a correctness or speed test.
- Choosing kernel boundaries that force a host round-trip the existing
  persistent-state architecture didn't need — this degrades the whole
  pipeline's performance even if the new kernel itself is fast.

## Cost-Control Guidance

- Do steps 1-4 (host reference, isolated kernel validation) before
  spending any GPU-scheduler time — these can mostly be developed and
  partially validated without a live accelerator allocation.
- Batch GPU-required validation (isolated kernel test, integration test,
  memory check) into as few scheduled jobs as reasonably possible rather
  than one job per micro-step.
- Decide the acceptance bar before implementation so you don't have to
  redo analysis after the fact to decide whether a result is "good enough."

## Reusable Prompt Template

```
Port <subsystem> from its validated host-language reference
implementation to a <GPU/accelerator> kernel. Preserve the exact
algorithmic structure unless you explicitly justify and document a
reordering as floating-point-reassociation-only. Validate the kernel in
isolation against the host reference at <tolerance> before any
integration. Integrate behind a default-OFF flag at <ABI boundary>.
Re-validate numerically at the real production entry point post-
integration. Profile the real production workload (not a microbenchmark)
for both local and end-to-end contribution. Check memory headroom at
<target device/rank density>. Report: isolated validation result,
post-integration validation result, local speedup, end-to-end
contribution, memory headroom, and a KEEP/PORT/DEFER/STOP recommendation
against <pre-declared acceptance bar>.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** PT/DELZ, REMAP_UV,
REMAP_Q, and REMAP_PRESSUREVARS all followed this exact sequence: host C++
reference validated first, isolated CUDA-kernel-vs-host validation next,
then integration into the real `StandAlone_FV3_Dycore.x` production call
site, re-validated bitwise/round-off there, then profiled with the real
`fv_timing_mod` timers for both local and `TOTAL` contribution before
being made CUDA-default. `State_To_FV` followed the same sequence but
failed the memory-headroom check (step 9) at 6-ranks/1-GPU density — it
was correctly held back from unconditional default and shipped instead as
conditional on `device_count>=2`, where the memory check does pass
(`docs/agent_work_summary.md` §3/§7).
