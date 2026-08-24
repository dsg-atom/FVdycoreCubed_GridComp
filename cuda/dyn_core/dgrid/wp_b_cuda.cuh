#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"
#include "operators/wp_d_operators.hpp"

#include <cstdint>

namespace fv3::gpu::dgrid {

using fv3::gpu::operators::View2;
using fv3::gpu::operators::View3;

struct HeightTransportBounds {
  int64_t is, ie, js, je, npz;
  int64_t halo_width;
};

// Direct CUDA translation of wp_b_kernels.hpp's transport_service_for_update_dz
// (fv_tp_2d_hord6 only -- deln_flux is a separate, unwired entry point; see
// docs/agent_handoff/milestone_3a3_wp_b_cuda.md). Batches over all npz+1
// interface levels via blockIdx.z, matching operators/wp_d_cuda.cu's idiom.
// q_in/fx/fy/q_work/fx2/fy2 are the batched arena scratch views; ra_x/ra_y are
// scratch outputs from launch_ra_xy_cuda.
fv3_gpu_status launch_fv_tp_2d_hord6_cuda(
    const HeightTransportBounds &b, int32_t hord,
    View3<const fv_real> q_in, View3<const fv_real> crx, View3<const fv_real> cry,
    View3<const fv_real> xfx, View3<const fv_real> yfx, View2<const fv_real> area,
    View3<const fv_real> ra_x, View3<const fv_real> ra_y,
    View3<fv_real> fx, View3<fv_real> fy, View3<fv_real> q_work,
    View3<fv_real> fx2, View3<fv_real> fy2, void *stream) noexcept;

// ra_x(i,j,k) = area(i,j) + xfx(i,j,k) - xfx(i+1,j,k)
// ra_y(i,j,k) = area(i,j) + yfx(i,j,k) - yfx(i,j+1,k)
// New code (not a translation of an existing tested function) -- see handoff.
fv3_gpu_status launch_ra_xy_cuda(const HeightTransportBounds &b,
                                 View3<const fv_real> xfx, View3<const fv_real> yfx,
                                 View2<const fv_real> area, View3<fv_real> ra_x,
                                 View3<fv_real> ra_y, void *stream) noexcept;

// zh_new(i,j,k) = zh(i,j,k) + (fx(i,j,k)-fx(i+1,j,k)+fy(i,j,k)-fy(i,j+1,k))*rarea(i,j)
// New code modeled on wp_b_kernels.hpp's apply_mass_and_scalar_flux mass_div
// formula (same divergence pattern, no existing mass field for zh) -- see
// handoff for the ambiguity this resolves.
fv3_gpu_status launch_apply_height_flux_divergence_cuda(
    const HeightTransportBounds &b, View3<const fv_real> fx, View3<const fv_real> fy,
    View2<const fv_real> rarea, View3<fv_real> zh, void *stream) noexcept;

// Direct CUDA translation of wp_b_kernels.hpp's deln_flux_transport, has_mass
// = false path only (the only path transport_service_for_update_dz ever
// exercises). Implemented and smoke-tested but NOT invoked from
// launch_dgrid_height_transport_cuda's production path this milestone --
// NumericalConfig has no field identifiable as the Fortran vtdm4-derived
// damp_c coefficient. See handoff, "damp_c mapping ambiguity".
fv3_gpu_status launch_deln_flux_transport_cuda(
    const HeightTransportBounds &b, int32_t nord, fv_real damp, fv_real da_min,
    View3<const fv_real> q, View2<const fv_real> del6_u, View2<const fv_real> del6_v,
    View2<const fv_real> rarea, View3<fv_real> fx, View3<fv_real> fy,
    View3<fv_real> d2, View3<fv_real> fx2, View3<fv_real> fy2, void *stream) noexcept;

} // namespace fv3::gpu::dgrid
