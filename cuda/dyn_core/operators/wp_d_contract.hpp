#pragma once

#include "fv3_gpu_internal.hpp"
#include "wp_d_operators.hpp"

#include <array>

namespace fv3::gpu::operators {

struct WpDViews {
  DeviceView<fv_real> u, v, delp, pt, delz, cappa, pe, peln, pkz;
  DeviceView<fv_real> gz, zh, pkc, pk3, du, dv, heat_source, divg2;
  DeviceView<const fv_real> rdx, rdy, rarea, del6_u, del6_v;
  DeviceView<const fv_real> dx, dy, rdxc, rdyc, sina_u, sina_v;
  DeviceView<const fv_real> dxa, dya, grid_lonlat, agrid_lonlat;
};

struct OperatorArena {
  static constexpr size_t array_count = 12;
  std::array<void *, array_count> arrays{};
  uint64_t values_per_array = 0;
};

/* Validate and bind WP-E v3 resources without taking ownership or changing
 * descriptor metadata.  Device pointers remain borrowed. */
int bind_wp_d(const ExecutionContext &, WpDViews &, OperatorArena &,
              fv3_gpu_error *) noexcept;

/* Coarse integration-owned dispatch entries. */
int launch_nh_pressure(const ExecutionContext &, fv3_gpu_error *) noexcept;
int launch_post_loop_heating(const ExecutionContext &, fv3_gpu_error *) noexcept;

CornerServiceRequest diffusion_corner_request(Axis axis, uint32_t halo_width,
                                               int32_t pass_index,
                                               fv3_gpu_event dependency) noexcept;
CornerServiceRequest a2b_edge_request(FieldId field, int32_t order,
                                      fv3_gpu_event dependency) noexcept;
PhysicalEdges physical_edges(const TopologyView &) noexcept;
PhysicalCorners physical_corners(const TopologyView &) noexcept;

template<class T> inline View1<T> view1(const DeviceView<T> &v) noexcept {
  return {v.base(),v.lower[0],v.stride[0]};
}
template<class T> inline View2<T> view2(const DeviceView<T> &v) noexcept {
  return {v.base(),v.lower[0],v.lower[1],v.stride[0],v.stride[1]};
}
/* This preserves descriptor axis order.  In particular, pe and peln remain
 * (i,k,j) and must not be treated as logical (i,j,k) views. */
template<class T> inline View3<T> view3(const DeviceView<T> &v) noexcept {
  return {v.base(),v.lower[0],v.lower[1],v.lower[2],v.stride[0],v.stride[1],v.stride[2]};
}

} // namespace fv3::gpu::operators
