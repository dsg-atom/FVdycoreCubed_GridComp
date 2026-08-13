#pragma once

#include "fv3_gpu_internal.hpp"
#include "wp_d_operators.hpp"

#include <array>

namespace fv3::gpu::operators {

struct WpDViews {
  DeviceView<fv_real> u, v, delp, pt, delz, cappa, pkz;
  DeviceView<fv_real> gz, zh, pkc, pk3, du, dv, heat_source;
  DeviceView<const fv_real> rdx, rdy, rarea, del6_u, del6_v;
};

struct OperatorArena {
  static constexpr size_t array_count = 12;
  std::array<void *, array_count> arrays{};
  uint64_t values_per_array = 0;
};

/* Validate and bind WP-E v2 resources without taking ownership or changing
 * descriptor metadata.  Device pointers remain borrowed. */
int bind_wp_d(const ExecutionContext &, WpDViews &, OperatorArena &,
              fv3_gpu_error *) noexcept;

/* Coarse integration-owned dispatch entries. */
int launch_nh_pressure(const ExecutionContext &, fv3_gpu_error *) noexcept;
int launch_post_loop_heating(const ExecutionContext &, fv3_gpu_error *) noexcept;

} // namespace fv3::gpu::operators
