#pragma once
#include "fv3_gpu_types.hpp"
#include "fv3_gpu_internal.hpp"
#include <cstddef>

namespace fv3::gpu::vertical {
struct ColumnLaunch {
  int columns;
  int levels;
  std::ptrdiff_t column_stride;
  std::ptrdiff_t level_stride;
};

// Enqueue-only package API. stream is the WP-E-owned compute stream.
int launch_reconstruct_height(fv_real *zh, const fv_real *delz,
                              const fv_real *zs, ColumnLaunch layout,
                              void *stream);
int launch_height_boundary(fv_real *zh, fv_real *delz, fv_real *ws,
                           const fv_real *zs, fv_real rdt, fv_real dz_min,
                           ColumnLaunch layout, void *stream);
int launch_dgrid_height_v2(const ExecutionContext &, fv3_gpu_error *);

// Complete update_dz_d vertical/height dispatch. Orchestrates:
// 1. Pre-transport height preparation (edge_profile on zh)
// 2. WP-B horizontal transport service (hord_tm=6 via dgrid_height_transport)
// 3. Post-transport height recurrence
// 4. Boundary/monotonic height corrections
// 5. Final delz recovery
// Requires H_DSCALAR dependency satisfied and WP-B transport service wired.
int launch_update_dz_d(const ExecutionContext &, const NumericalDispatch &,
                       fv3_gpu_error *);

/* Milestone 3A.9 §5/§6: the dgrid_height dispatch slot. update_dz_d
 * (dyn_core.F90:905-906) as a LaunchFn, taking the dispatch table it needs for
 * the WP-B transport service from ExecutionContext::dispatch instead of an
 * extra parameter no dispatch slot could supply. Adds no numerical content --
 * see the block comment at its definition for the full §5 ownership resolution
 * (dgrid_height = update_dz_d; dgrid_height_transport = the WP-B body it calls;
 * launch_dgrid_height_v2 = not production, must never be wired). */
int launch_dgrid_height_stage(const ExecutionContext &, fv3_gpu_error *);

// Vertical Riemann solver dispatch. Uses a_imp from NumericalConfig to select:
// a_imp > 0.999: SIM1, a_imp <= 0.5: RIM_2D, etc.
// Requires WorkspaceId::vertical slice with sufficient capacity.
int launch_vertical_solver(const ExecutionContext &, fv3_gpu_error *);

/* C-grid vertical PRE-HALO part (Milestone 3A.11S).
 *
 * dyn_core.F90:436-455 reconstructs gz's INTERIOR (is:ie, js:je) from delz, and
 * :459 only then starts the gz halo exchange that :552 completes. The producer
 * therefore runs strictly BEFORE the exchange packs anything.
 *
 * That ordering cannot be expressed if the reconstruction lives inside the same
 * opaque LaunchFn as the post-exchange work, because the runtime has no point at
 * which to resolve H_GZ_FIRST in between. So the reconstruction is its own stage:
 *
 *   cgrid_vertical_pre_halo  ->  resolve(H_GZ_FIRST)  ->  cgrid_vertical_solver
 *
 * Gated on acoustic_index == 0, matching Fortran's `if (it==1)` at :433. A no-op
 * for it > 1, where gz is restored from zh instead of rebuilt from delz.
 *
 * Numerical algorithm changed: NO. This is the identical gz_from_delz_kernel
 * launch that previously opened launch_cgrid_vertical_solver, relocated. */
int launch_cgrid_vertical_pre_halo(const ExecutionContext &, fv3_gpu_error *);

// C-grid vertical solver entry for WP-A. Shares column solver primitives
// but has different orchestration scope (pressure outputs only).
//
// POST-HALO part as of Milestone 3A.11S: assumes launch_cgrid_vertical_pre_halo
// has already reconstructed gz's interior and that H_GZ_FIRST has been resolved
// from it. Still require()s H_GZ_FIRST, so a runtime that forgets the pre-halo
// stage fails loudly rather than running on a stale halo.
int launch_cgrid_vertical_solver(const ExecutionContext &, fv3_gpu_error *);

// Workspace sizing and layout from fv3_vertical_column.hpp.
// See VerticalWorkspaceLayout and compute_workspace_layout() for details.

} // namespace fv3::gpu::vertical
