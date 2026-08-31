#pragma once

#include "wp_d_operators.hpp"

namespace fv3::gpu::operators {

fv3_gpu_status launch_reconstruct_pe_halo_cuda(
    const HorizontalBounds &, fv_real ptop, View3<const fv_real> delp,
    View3<fv_real> pe, void *stream) noexcept;
fv3_gpu_status launch_reconstruct_pk3_halo_cuda(
    const HorizontalBounds &, fv_real ptop, fv_real akap,
    View3<const fv_real> delp, View3<fv_real> pk3, void *stream) noexcept;
fv3_gpu_status launch_reconstruct_pln_halo_cuda(
    const HorizontalBounds &, fv_real ptop, View3<const fv_real> delp,
    View3<fv_real> pln, void *stream) noexcept;
fv3_gpu_status launch_geopotential_from_height_cuda(
    const HorizontalBounds &, int halo_band, fv_real grav,
    View3<const fv_real> zh, View3<fv_real> gz, void *stream) noexcept;

fv3_gpu_status launch_a2b_ord2_centered_cuda(
    const HorizontalBounds &, int64_t levels, View3<const fv_real> input,
    View3<fv_real> output, void *stream) noexcept;
fv3_gpu_status launch_a2b_ord4_centered_cuda(
    const HorizontalBounds &, int64_t levels, View3<const fv_real> input,
    View3<fv_real> output, View3<fv_real> qx, View3<fv_real> qy,
    void *stream) noexcept;
fv3_gpu_status launch_a2b_ord2_global_cuda(
    const HorizontalBounds &, PhysicalEdges, PhysicalCorners, int64_t levels,
    View3<fv_real> input, View3<fv_real> output, const A2BGeometry &,
    bool replace, void *stream) noexcept;
fv3_gpu_status launch_a2b_ord4_global_cuda(
    const HorizontalBounds &, PhysicalEdges, PhysicalCorners, int64_t levels,
    View3<fv_real> input, View3<fv_real> output, const A2BGeometry &,
    View3<fv_real> qx, View3<fv_real> qy, View3<fv_real> qxx,
    View3<fv_real> qyy, bool replace, void *stream) noexcept;
fv3_gpu_status launch_set_pressure_top_cuda(
    const HorizontalBounds &, fv_real top_value, View3<fv_real> pp,
    View3<fv_real> pk, void *stream) noexcept;

fv3_gpu_status launch_pressure_gradient_cuda(
    const HorizontalBounds &, const PressureInputs &, const PressureOutputs &,
    const PressureConfig &, void *stream) noexcept;
fv3_gpu_status launch_one_gradient_pressure_cuda(
    const HorizontalBounds &, const OneGradientInputs &, View3<fv_real> u,
    View3<fv_real> v, fv_real dt, fv_real d_ext, void *stream) noexcept;
fv3_gpu_status launch_grad1_pressure_update_cuda(
    const HorizontalBounds &, const OneGradientInputs &, View3<fv_real> u,
    View3<fv_real> v, View3<fv_real> du, View3<fv_real> dv, fv_real dt,
    fv_real beta, void *stream) noexcept;

fv3_gpu_status launch_average_del2_corners_cuda(
    const HorizontalBounds &, PhysicalCorners, View3<fv_real> q,
    void *stream) noexcept;
fv3_gpu_status launch_del2_zonal_flux_cuda(
    const HorizontalBounds &, int halo_band, bool use_sg, View3<fv_real> q,
    const Del2Metrics &, const Del2SgMetrics &, View3<fv_real> fx,
    void *stream) noexcept;
fv3_gpu_status launch_del2_meridional_flux_cuda(
    const HorizontalBounds &, int halo_band, bool use_sg, View3<fv_real> q,
    const Del2Metrics &, const Del2SgMetrics &, View3<fv_real> fy,
    void *stream) noexcept;
fv3_gpu_status launch_del2_update_cuda(
    const HorizontalBounds &, int halo_band, accumulator_real cd,
    View3<fv_real> q, View2<const fv_real> rarea, View3<fv_real> fx,
    View3<fv_real> fy, void *stream) noexcept;
fv3_gpu_status launch_diffusive_heating_nh_cuda(
    const HorizontalBounds &, fv_real bdt, fv_real delt_max, fv_real cv_air,
    fv_real rdg, fv_real akap, bool moist_cappa, int64_t n_con,
    View3<fv_real> heat_source, View3<const fv_real> delp,
    View3<const fv_real> delz, View3<const fv_real> cappa,
    View3<fv_real> pkz, View3<fv_real> pt, void *stream) noexcept;
fv3_gpu_status launch_diffusive_heating_hydrostatic_cuda(
    const HorizontalBounds &, fv_real bdt, fv_real delt_max, fv_real cp_air,
    int64_t n_con, View3<const fv_real> heat_source,
    View3<const fv_real> delp, View3<const fv_real> pkz,
    View3<fv_real> pt, void *stream) noexcept;

fv3_gpu_status launch_pem_snapshot_cuda(
    const HorizontalBounds &, fv_real ptop, View3<const fv_real> delp,
    View3<fv_real> pem, void *stream) noexcept;
fv3_gpu_status launch_omega_from_pe_cuda(
    const HorizontalBounds &, fv_real rdt, View3<const fv_real> pe,
    View3<fv_real> pem, View3<fv_real> omga, void *stream) noexcept;
fv3_gpu_status launch_adv_pe_cuda(
    const HorizontalBounds &, View3<const fv_real> ua, View3<const fv_real> va,
    View3<fv_real> pb, const AdvPeMetrics &, View3<fv_real> om,
    void *stream) noexcept;

} // namespace fv3::gpu::operators
