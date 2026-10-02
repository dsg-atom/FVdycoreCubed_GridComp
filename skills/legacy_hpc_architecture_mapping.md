# Legacy HPC Architecture Mapping

## Purpose

Before writing any new code, build an accurate map of what the legacy
codebase actually does: who calls what, who owns which numerical
responsibility, where data lives, and where the real integration
boundaries are. This prevents duplicating functionality across work
packages and prevents building a port against a call graph that is wrong.

## Use When

- Starting a modernization effort on an unfamiliar or partially-familiar
  legacy codebase.
- Before assigning ownership of a subsystem to a team/agent/work-package.
- Whenever a "wait, who actually calls this?" question blocks progress.

## Inputs Needed

- Read access to the full legacy source tree (not just the subsystem you
  think you care about — ownership boundaries are often surprising).
- The actual build system (so you know what's compiled/linked, not just
  what's present in the tree).
- If available, any existing architecture docs — but treat them as a
  starting hypothesis, not ground truth, until cross-checked against code.

## Procedure

1. Identify the top-level entry point(s) actually invoked at runtime (the
   real driver, not a test harness or an old deprecated entry point).
2. From each entry point, trace the call graph for the subsystem(s) in
   scope. Use `grep`/`git grep` for exact symbol names, not just file
   names — legacy codebases often have multiple files with similar names
   and only one is actually wired in.
3. For each major subsystem, record: owning module/file, inputs, outputs,
   data residency (host memory layout, aliasing, mutation in place vs.
   copy), and any communication/synchronization points (MPI calls, I/O,
   locks).
4. Classify every piece of code you touch as: owned-and-modernizable,
   external-dependency (do not modify), or vendored-reference
   (used for comparison only, not a build dependency).
5. Partition the subsystem into non-overlapping ownership regions
   (work packages) based on actual data/control flow boundaries, not file
   boundaries — two files that are always called together and share state
   should usually be one ownership region.
6. Write the map down as a short document (call graph + ownership table),
   not just institutional knowledge — this is what every later skill
   assumes exists.
7. Explicitly identify the data-residency pattern at each subsystem
   boundary: does data live on host or device, is a boundary a natural
   place for synchronization, is a transfer currently full-state or
   partial.

## Validation / PASS Criteria

- For any numerical routine in scope, you can state which module owns it
  and what calls it, without reading source live.
- Every planned work-package boundary corresponds to a real
  control/data-flow seam, not an arbitrary file split.
- You have explicitly listed the subsystem's communication/synchronization
  points, not just its numerical kernels.

## STOP Conditions

- The call graph cannot be established statically (e.g., heavy use of
  function pointers, dynamic dispatch, or preprocessor-generated code
  paths) — escalate to a dynamic-tracing approach (actually run the code
  with tracing/instrumentation) rather than guessing from static reading.
- Two proposed ownership regions both claim the same numerical
  responsibility — resolve this before any implementation starts; it is
  the single most common cause of duplicated/conflicting modernization
  work.

## Common Failure Modes

- Trusting a doc comment or old design doc over the actual call graph —
  legacy comments rot; the live call graph does not lie.
- Mapping only the "interesting" numerical subsystem and skipping the
  "boring" runtime/communication infrastructure around it — that
  infrastructure (state management, dispatch, communication) is usually
  where the real integration risk and the real work-package ownership
  conflicts live.
- Assuming a module's name describes its current responsibility — modules
  in long-lived legacy codebases accumulate responsibilities the name no
  longer reflects.

## Cost-Control Guidance

- Use targeted `grep`/`git grep` for exact symbols rather than reading
  every file in a subsystem end to end.
- Map only the subsystem(s) actually in scope for the current phase; defer
  mapping adjacent subsystems until a task actually depends on them.
- Record the map once, durably, so it does not need to be rebuilt from
  scratch in a future session — point future readers at the map file
  instead of re-deriving it.

## Reusable Prompt Template

```
Map the architecture of <subsystem> inside <legacy codebase>. Starting
from <real production entry point>, trace the actual call graph (not
file-name guesses) for <subsystem>. For each major piece: owning
module/file, inputs/outputs, data residency, and any
communication/synchronization points. Classify each piece as
owned-and-modernizable / external-dependency / vendored-reference.
Propose non-overlapping ownership regions based on real data/control-flow
boundaries. Write the result to <map file path>. Do not modify any
source file as part of this task.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** This project
partitioned the FV3 dynamical core into five ownership regions (WP-A
C-grid predictor, WP-B D-grid dynamics/transport, WP-C vertical/
nonhydrostatic, WP-D pressure/grid operators, WP-E GPU runtime/state/API/
communication infrastructure), derived from the actual numerical/data-flow
structure of `dyn_core`, documented in
`docs/agent_instructions/project_architecture.md`. WP-E was explicitly
carved out as shared infrastructure (GPU context, persistent state, ABI,
halo/communication) specifically because every other work package depends
on it — exactly the "boring infrastructure is actually the highest-risk
ownership region" pattern noted above. A later audit
(`docs/agent_handoff/implementation_reality_audit.md`) found that WP-B had
historically received zero CUDA kernels at all (host C++ only) despite
being named as a CUDA work package — a reminder that an ownership map must
be periodically re-validated against what the code actually contains, not
assumed to still hold from when it was written.
