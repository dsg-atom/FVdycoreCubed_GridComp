#pragma once

#include "fv3_gpu_internal.hpp"

namespace fv3::gpu::dgrid {

// Resolved FieldRegistry views for the dgrid_height_transport CUDA path
// (WP-C's update_dz_d dependency).
//
// Milestone 3A.4 grew this set. Milestone 3A.3 bound only the fields its
// (incomplete) flux-divergence update read; the authoritative update_dz_d
// additionally needs:
//   dxa/dya        -- xppm/yppm one-sided panel-edge interface reconstruction
//                     (tp_core.F90:373-386, 755-772 and the y-analogues)
//   dp_ref         -- edge_profile's reference thickness profile, which builds
//                     the interface Courant numbers / mass fluxes that are
//                     actually transported (nh_utils.F90:265-267)
//   damping_v      -- per-level damp(k); nh_utils.F90:280 branches on
//                     damp(k) > 1.E-5 and sw_core.F90:1590 uses the raw value
//   damping_ndif   -- per-level nord(k) selecting del-2/4/6
struct WpBHeightTransportViews {
  const DeviceViewBase *zh = nullptr;
  const DeviceViewBase *crx = nullptr, *cry = nullptr;
  const DeviceViewBase *xfx = nullptr, *yfx = nullptr;
  const DeviceViewBase *area = nullptr, *rarea = nullptr;
  const DeviceViewBase *dxa = nullptr, *dya = nullptr;
  const DeviceViewBase *del6_u = nullptr, *del6_v = nullptr;
  const DeviceViewBase *dp_ref = nullptr;
  const DeviceViewBase *damping_v = nullptr, *damping_ndif = nullptr;
};

// Scratch arena carved from WorkspaceId::transport. Each array is a batched
// View3 (all npz+1 interface levels at once) over the extended horizontal
// box [is-halo, ie+halo] x [js-halo, je+halo], which bounds every access made
// by the authoritative sequence given halo >= 3 and nord <= 2.
//
// The two flux pairs are deliberately distinct: fv_tp_2d has its own internal
// fx2/fy2 scratch (tp_core.F90:180, 201) and del6_vt_flux produces a different
// fx2/fy2 that the zh update reads separately, weighted by rarea
// (nh_utils.F90:290-297). Milestone 3A.3 conflated them.
//
// gam occupies a whole slot but uses only its first npz values; keeping every
// slot the same size keeps the sizing arithmetic and the overflow checks
// uniform.
struct WpBHeightTransportArena {
  // fx, fy, q_i, q_j, fx2_tp, fy2_tp, ra_x, ra_y, z2, d2, fx2_d6, fy2_d6,
  // crx_adv, cry_adv, xfx_adv, yfx_adv, gam
  static constexpr size_t array_count = 17;
  void *arrays[array_count] = {};
};

// Resolves views + arena for the height-transport CUDA path. Does not
// launch, synchronize, or allocate -- allocation is the WP-E workspace
// layout's job (build_workspace_layout), called once at context setup.
int bind_wp_b_height_transport(const ExecutionContext &, WpBHeightTransportViews &,
                               WpBHeightTransportArena &, fv3_gpu_error *) noexcept;

// LaunchFn-compatible entry point wired into
// NumericalDispatch::dgrid_height_transport. Executes the authoritative
// update_dz_d transport body (nh_utils.F90:255-305) on all npz+1 interface
// levels of zh: edge_profile -> ra_x/ra_y -> per-level damped/undamped split
// -> fv_tp_2d -> del6_vt_flux -> area-weighted height update.
//
// It does NOT apply the ws / dz_min monotonicity closure
// (nh_utils.F90:307-322); that is WP-C's height boundary kernel, which calls
// this operator and then finishes the column.
int launch_dgrid_height_transport_cuda(const ExecutionContext &, fv3_gpu_error *) noexcept;

} // namespace fv3::gpu::dgrid
