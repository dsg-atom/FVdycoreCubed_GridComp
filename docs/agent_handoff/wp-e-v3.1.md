# WP-E v3.1 Hotfix Handoff

## Summary

Targeted Gate A.2.1 hotfix closing WP-A and WP-D shared-contract blockers.

## Blockers addressed

### WP-A blocker: corner service unreachable from LaunchFn

**Status: CLOSED**

WP-A reported that `NumericalDispatch` owns `CornerServiceFn corner_edge_service`
but numerical launchers receive only `ExecutionContext`, which had no service
reference. Thus WP-A predictor (`c_sw`, `update_dz_c`) could not interleave
directional corner fills with dependent kernels.

**Solution:** Added `CornerServiceFn corner_service` member to `ExecutionContext`.
Validation now requires non-null `corner_service`. Numerical packages can invoke
`ctx.corner_service(ctx, request, &err)` directly.

WP-A corner patterns now expressible:
- `fill2_4corners(delp, pt, 1)` → `CornerOperation::fill_pair`, `first=delp`, `second=pt`, `axis=Axis::x`
- `fill_4corners(w, 1)` → `CornerOperation::fill`, `first=w`, `axis=Axis::x`
- `fill_4corners(gz2, 2)` → `CornerOperation::fill`, `first=gz2`, `axis=Axis::y`

### WP-D blocker: edge descriptor type/rank

**Status: CLOSED**

WP-D reported that `FieldId::edge_w/e/s/n` were declared as `LogicalType::fv_real`,
rank 2, but authoritative `fv_arrays.F90` declares them as rank-1 `R_GRID` (real64).

**Verification:** Confirmed in `model/fv_arrays.F90`:
- Line 37: `R_GRID = r8_kind`
- Lines 150-153: `edge_s(:)`, `edge_n(:)`, `edge_w(:)`, `edge_e(:)` as `real(kind=R_GRID)`
- Lines 1629-1632: allocations confirm rank-1

**Solution:** Corrected `fv3_gpu_contract.cpp` specs:
- `edge_w`, `edge_e`: rank 1, `accumulator_real`, `Y_FACE` stagger (indexed by j)
- `edge_s`, `edge_n`: rank 1, `accumulator_real`, `X_FACE` stagger (indexed by i)

### WP-D blocker: A-to-B replace semantics

**Status: CLOSED**

WP-D reported that pressure requires both:
- `replace=true`: A-to-B for pp/pk/gz in-place
- `replace=false`: A-to-B for delp → wk1 (operator arena destination)

**Solution:** Added to `CornerServiceRequest`:
- `FieldId destination`: for A-to-B when destination differs from source
- `uint8_t replace`: 1=in-place, 0=write to destination

Validation enforces that `replace=0` with `a_to_b_edge` operation requires valid
registered destination field.

### WP-D blocker: diffusion corner-service subphases

**Status: CLOSED**

WP-D reported that `del2_cubed` requires interleaved phases within each pass:
x-copy → x-flux → y-copy → y-flux → update

**Solution:** Added `DiffusionPhase` enum and `diffusion_phase` member:
```cpp
enum class DiffusionPhase : uint8_t {
  none=0, x_corner_copy, x_flux, y_corner_copy, y_flux, update
};
```

## Files modified

- `cuda/dyn_core/include/fv3_gpu_internal.hpp`: Added `DiffusionPhase` enum,
  extended `CornerServiceRequest` with `destination`, `diffusion_phase`, `replace`,
  added `corner_service` to `ExecutionContext`
- `cuda/dyn_core/runtime/fv3_gpu_contract.cpp`: Corrected edge specs, updated
  validation for new request fields and context service
- `cuda/dyn_core/tests/runtime/test_contract.cpp`: Added V3.1 tests for edge
  descriptors, A-to-B replace semantics, diffusion phases, service access
- `docs/architecture/dyn_core_contract_change_log.md`: Appended Gate A.2.1 section

## Test results

| Configuration | Build | Tests |
|---|---|---|
| R8 | PASS | 4/4 PASS |
| R4 | PASS | 4/4 PASS |
| CUDA | NOT RUN | nvcc unavailable |
| GPU execution | NOT RUN | no device |

## API compatibility

- Public C ABI: unchanged
- Internal contract version: remains 3 (no version bump for compatible extension)
- Existing packages: must set `corner_service` in `ExecutionContext`

## Blocker summary

| Blocker | Status |
|---|---|
| WP-A corner service access | **CLOSED** |
| WP-D edge descriptor type/rank | **CLOSED** |
| WP-D A-to-B replace semantics | **CLOSED** |
| WP-D diffusion subphases | **CLOSED** |

## Recommended next steps

1. WP-A: Resume `c_sw`/`update_dz_c` translation using `ctx.corner_service`
2. WP-D: Bind rank-1 real64 edge views and wire A-to-B replace/non-replace paths
3. WP-D: Wire diffusion phases in `del2_cubed` orchestration
4. Integration: Ensure all `ExecutionContext` construction sets `corner_service`

WP-E V3.1 HOTFIX READY
