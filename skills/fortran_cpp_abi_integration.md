# Legacy-Language / Modern-Backend ABI Integration

## Purpose

Establish a stable, explicit, low-risk integration boundary between the
legacy language runtime and the new backend (C++/CUDA or otherwise), so
that new capability can be added incrementally without touching legacy
call sites repeatedly or risking silent memory-layout mismatches.

## Use When

- Before writing any code that must be called from the legacy language and
  that lives in the new backend.
- Whenever you are about to add a new field/capability that crosses the
  legacy/modern boundary.

## Inputs Needed

- The legacy language's standard foreign-function-interface mechanism
  (e.g. Fortran's `ISO_C_BINDING`) or an existing established boundary if
  one is already in place.
- A clear list of what crosses the boundary: scalars, arrays, communicator
  handles, callback pointers — and their exact bit-widths.
- The target backend's build system and header-generation convention.

## Procedure

1. Define one stable C ABI (plain C struct/function layer) as the sole
   crossing point — never let the legacy language call directly into
   C++/CUDA-specific constructs (classes, templates, exceptions).
2. Use explicit, fixed-width types for every field crossing the boundary
   (e.g. `c_int32_t`, not a default/platform-dependent integer kind) —
   this is what prevents silent mismatches across compilers/platforms.
3. For communicator/handle types that don't have a portable C
   representation (e.g. an MPI communicator), cross the boundary as a
   plain integer and convert on the receiving side using the standard
   conversion function for that library (e.g. `MPI_Comm_f2c`) — never
   assume the legacy and modern sides agree on an opaque handle's bit
   layout.
4. Add a runtime version/size descriptor to the ABI (e.g. an `abi_version`
   and `struct_bytes` field) that both sides check at initialization, so a
   compiler-family or build mismatch between caller and library fails
   loudly at startup rather than corrupting memory later.
5. When adding new capability, extend the ABI additively: trailing fields
   with safe defaults, trailing arguments, never reordering or removing
   existing fields/arguments. Confirm a zero-initialized or
   existing-caller-shaped struct still works correctly with the new field
   defaulted.
6. Keep lifecycle explicit on both sides: initialize once, step
   repeatedly, finalize once — do not create/destroy backend state per
   call.
7. Gate every new capability behind an explicit flag, default OFF/legacy
   behavior, until it has been separately validated.

## Validation / PASS Criteria

- An existing caller, unmodified, continues to compile and behave
  identically after an additive ABI change (confirmed by actually
  rebuilding and running that existing caller, not just reasoning about
  it).
- The ABI version/size check actually fires (verified by deliberately
  testing a mismatched build, if feasible) rather than being
  unverified dead code.
- No C++-specific type (class, template, STL container) appears in any
  header included by the legacy language side.

## STOP Conditions

- A new capability cannot be expressed as an additive, trailing
  field/argument without reordering existing ABI layout — stop and
  redesign the struct/call rather than silently breaking existing callers.
- The legacy language's FFI mechanism cannot represent a required type
  (e.g. a complex nested structure) in a portable, fixed-width way —
  resolve this explicitly (usually by flattening/serializing) before
  proceeding, rather than guessing at an unverified binary layout.

## Common Failure Modes

- Using default/platform-dependent integer or real kinds at the boundary —
  works until a platform or compiler changes, then fails silently with
  corrupted data, not a clean error.
- Letting a convenience shortcut call directly into a C++ class or template
  from the legacy side — breaks the "legacy calls one stable C ABI" model
  and makes every future ABI change far riskier.
- Treating "the code builds" as equivalent to "the two sides agree on
  layout" — only an explicit runtime version/size check (or an actual
  mismatched-build test) proves this.
- Reordering or removing ABI fields "just this once" instead of adding a
  trailing field — this is the single most avoidable source of
  hard-to-debug cross-language memory corruption.

## Cost-Control Guidance

- Design the ABI's extensibility convention (additive-only, versioned)
  once, early — this is what makes every later capability addition cheap
  (small diff, no redesign) instead of expensive (every addition risks a
  breaking change).
- Validate new ABI fields against existing callers mechanically (rebuild
  and run existing tests) rather than reasoning abstractly about layout
  compatibility — this is cheap and removes doubt.

## Reusable Prompt Template

```
Add <new capability/field> to the existing stable ABI boundary between
<legacy language> and <modern backend> in <project>. Follow the
additive-only convention already established: trailing field(s)/
argument(s) only, explicit fixed-width types, safe default when absent.
Confirm the existing caller(s) at <call site(s)> still compile and run
unchanged. Gate any new behavior enabled by this field behind a
default-OFF flag until separately validated. Do not reorder or remove any
existing ABI field/argument. Report before/after behavior of at least one
existing caller as evidence of non-breakage.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** The FV3 CUDA ABI
(`fv3_gpu_api.h`, `fv3_gpu_communication.h`) uses fixed-width
`iso_c_binding` types exclusively, carries its own `abi_version`/
`struct_bytes` fields checked at init, and crosses the MPI communicator as
a plain Fortran `INTEGER` converted via `MPI_Comm_f2c` on the C++ side —
no C++/MPI type ever crosses the boundary directly. Every observed ABI
extension in this project's history (`layout_x`/`layout_y`, a
`device_count` getter) was purely additive: trailing fields/arguments with
`<=0`-defaults-to-`1` normalization, confirmed not to break an existing
zero-initialized test struct literal (`docs/performance/
p114_fms_layout_bridge_abi_plumbing_2026-09-30.md` §3a). New capability
(e.g. layout metadata) was explicitly staged as "wire the data through,
validate it arrives correctly, change zero production behavior" before any
later task was allowed to make a behavior decision based on it.
