#pragma once

#include "fv3_gpu_internal.hpp"

namespace fv3::gpu::dgrid {

// Resolved FieldRegistry views for the dgrid_height_transport CUDA path
// (WP-C's update_dz_d dependency). Deliberately narrower than the older
// ContractViews/resolve_contract_v3 (which resolves the full D-grid
// momentum/vorticity/damping contract) -- this only covers what
// fv_tp_2d_hord6 + deln_flux_transport actually read when transporting the
// scalar field zh, per direct source audit of wp_b_kernels.hpp.
struct WpBHeightTransportViews {
  const DeviceViewBase *zh = nullptr;
  const DeviceViewBase *crx = nullptr, *cry = nullptr;
  const DeviceViewBase *xfx = nullptr, *yfx = nullptr;
  const DeviceViewBase *area = nullptr, *rarea = nullptr;
  // Only consumed by deln_flux_transport, which this milestone implements as
  // CUDA kernels but does not invoke from the production launch path (see
  // milestone_3a3_wp_b_cuda.md, "damp_c mapping ambiguity"). Still resolved
  // so the standalone deln_flux launcher/test can exercise real fields.
  const DeviceViewBase *del6_u = nullptr, *del6_v = nullptr;
};

// Scratch arena carved from WorkspaceId::transport. Each array is a batched
// View3 (all npz+1 interface levels at once) over the extended horizontal
// box [is-halo, ie+halo] x [js-halo, je+halo], matching fv_tp_2d_hord6's
// hardcoded +/-3 stencil reach and, when invoked, deln_flux_transport's
// +/-(1+nord) reach.
struct WpBHeightTransportArena {
  static constexpr size_t array_count = 8; // fx,fy,q_work,fx2,fy2,d2,ra_x,ra_y
  void *arrays[array_count] = {};
};

// Resolves views + arena for the height-transport CUDA path. Does not
// launch, synchronize, or allocate -- allocation is the WP-E workspace
// layout's job (build_workspace_layout), called once at context setup.
int bind_wp_b_height_transport(const ExecutionContext &, WpBHeightTransportViews &,
                               WpBHeightTransportArena &, fv3_gpu_error *) noexcept;

// LaunchFn-compatible entry point wired into
// NumericalDispatch::dgrid_height_transport. Executes fv_tp_2d_hord6 on zh
// for all interface levels [0, npz] using hord = config.hord_tm.
int launch_dgrid_height_transport_cuda(const ExecutionContext &, fv3_gpu_error *) noexcept;

} // namespace fv3::gpu::dgrid
