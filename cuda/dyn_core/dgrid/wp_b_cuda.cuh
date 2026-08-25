#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"
#include "operators/wp_d_operators.hpp"

#include <cstdint>

namespace fv3::gpu::dgrid {

using fv3::gpu::operators::View1;
using fv3::gpu::operators::View2;
using fv3::gpu::operators::View3;

// Geometry + panel-topology parameters for the height-transport path.
//
// npx/npy are the Fortran fv_grid_bounds_type tile corner counts (the last
// cell index along a tile edge is npx-1/npy-1). They are REQUIRED: xppm/yppm
// replace the interior PPM interface reconstruction with one-sided formulas at
// a panel seam, and copy_corners rotates halo data across the seam, so a wrong
// or absent npx/npy silently applies interior stencils where the authoritative
// Fortran does not. bind_wp_b_height_transport rejects npx==0/npy==0.
//
// cubed_sphere mirrors the Fortran guard `.not.nested .and. grid_type < 3`,
// which gates BOTH the xppm/yppm edge formulas and copy_corners. west/east/
// south/north mirror `is==1`, `(ie+1)==npx`, `js==1`, `(je+1)==npy`.
struct HeightTransportBounds {
  int64_t is, ie, js, je, npz;
  // Registered domain halo width. Used only for arena geometry / bounds
  // validation, never for a Fortran loop bound -- see ng below.
  int64_t halo_width;
  // The FV3 ghost-zone count. This is a compile-time PARAMETER in the
  // authoritative Fortran (`integer, parameter:: ng = 3`,
  // tools/fv_mp_mod.F90:108), which tp_core.F90 and sw_core.F90 `use` directly.
  // Every isd/ied/jsd/jed loop bound and copy_corners' corner-block size is
  // therefore fixed at 3 regardless of how much halo the caller allocated.
  //
  // It must stay distinct from halo_width: conflating them (as this file did
  // before Milestone 3A.4) makes copy_corners rotate a halo_width x halo_width
  // corner block, so a caller registering halo_width = 4 would have four rings
  // of corner cells overwritten where the Fortran overwrites three.
  int64_t ng = 3;
  int64_t npx = 0, npy = 0;
  int32_t cubed_sphere = 1;
  int32_t nested = 0;
  int32_t west_edge = 0, east_edge = 0, south_edge = 0, north_edge = 0;
  int32_t sw_corner = 0, se_corner = 0, ne_corner = 0, nw_corner = 0;
};

// Per-level activation predicate shared by every stage of update_dz_d.
//
// Authoritative nh_utils.F90:280 splits the k loop into a damped branch
// (damp(k) > 1.E-5: transport a scratch copy z2, then add del6_vt_flux
// damping) and an undamped branch (transport zh in place, no damping). Because
// both branches run inside one batched CUDA launch over k, every kernel takes
// this filter and returns early on levels it does not own.
//
// damp/ndif are indexed at min(k, npz-1), which reproduces update_dz_d's
// `damp(km+1) = damp(km)` / `ndif(km+1) = ndif(km)` prologue without requiring
// the caller to register npz+1 damping entries or letting the GPU mutate an
// immutable grid field.
//
// ndif carries integral nord values through an fv_real array: FieldId
// damping_ndif is declared LogicalType::integer, but the public C ABI admits
// only F32/F64 element types, so the values travel as exactly-representable
// reals and are converted here with a nearest-integer round.
struct LevelFilter {
  View1<const fv_real> damp{};
  View1<const fv_real> ndif{};
  int64_t npz = 0;
  // 0 = every level, 1 = damped levels, 2 = undamped levels,
  // 3 = damped levels whose nord equals nord_sel.
  int32_t mode = 0;
  int32_t nord_sel = 0;
};

// --- edge_profile (nh_utils.F90:1547, uniform_grid = .false., limiter = 0) --
// Reconstructs the km+1 interface Courant numbers / mass fluxes that
// update_dz_d actually transports with (crx_adv/xfx_adv/cry_adv/yfx_adv) from
// the km cell-centered crx/xfx/cry/yfx. gam is the tridiagonal factor array;
// it depends only on dp_ref and k, so it is computed once for all (i,j).
fv3_gpu_status launch_edge_profile_gam_cuda(const HeightTransportBounds &b,
                                            View1<const fv_real> dp_ref,
                                            View1<fv_real> gam, void *stream) noexcept;

fv3_gpu_status launch_edge_profile_x_cuda(const HeightTransportBounds &b,
                                          View3<const fv_real> crx,
                                          View3<const fv_real> xfx,
                                          View1<const fv_real> dp_ref,
                                          View1<const fv_real> gam,
                                          View3<fv_real> crx_adv, View3<fv_real> xfx_adv,
                                          void *stream) noexcept;

fv3_gpu_status launch_edge_profile_y_cuda(const HeightTransportBounds &b,
                                          View3<const fv_real> cry,
                                          View3<const fv_real> yfx,
                                          View1<const fv_real> dp_ref,
                                          View1<const fv_real> gam,
                                          View3<fv_real> cry_adv, View3<fv_real> yfx_adv,
                                          void *stream) noexcept;

// --- copy_corners (tp_core.F90:247) --------------------------------------
// Rotates the four (halo x halo) corner blocks of a scalar in place, for the
// x direction (dir = 1) or y direction (dir = 2). All four corners are done in
// one launch: every corner's source block lies in an "interior along one axis"
// band and every destination block is a (halo, halo) corner, so the eight
// regions are pairwise disjoint and the in-place update is race-free.
// No-op when b.nested or !b.cubed_sphere.
fv3_gpu_status launch_copy_corners_cuda(const HeightTransportBounds &b, int32_t dir,
                                        const LevelFilter &lf, View3<fv_real> q,
                                        void *stream) noexcept;

// --- ra_x / ra_y (nh_utils.F90:269-278) ----------------------------------
// ra_x(i,j) = area(i,j) + (xfx_adv(i,j,k) - xfx_adv(i+1,j,k)) on is..ie x jsd..jed
// ra_y(i,j) = area(i,j) + (yfx_adv(i,j,k) - yfx_adv(i,j+1,k)) on isd..ied x js..je
// The extents differ between the two and the parenthesisation is the Fortran's.
fv3_gpu_status launch_ra_x_cuda(const HeightTransportBounds &b,
                                View3<const fv_real> xfx_adv, View2<const fv_real> area,
                                View3<fv_real> ra_x, void *stream) noexcept;

fv3_gpu_status launch_ra_y_cuda(const HeightTransportBounds &b,
                                View3<const fv_real> yfx_adv, View2<const fv_real> area,
                                View3<fv_real> ra_y, void *stream) noexcept;

// --- fv_tp_2d (tp_core.F90:109, no mfx/mfy, no deln_flux) ----------------
// Full authoritative sequence for hord == ord_in == ord_ou:
//   copy_corners(q, dir=2)
//   yppm(fy2, q,   cry_adv) on isd..ied x js..je+1
//   q_i = (q*area + yfx*fy2(j) - yfx*fy2(j+1)) / ra_y   on isd..ied x js..je
//   xppm(fx,  q_i, crx_adv) on is..ie+1  x js..je
//   copy_corners(q, dir=1)
//   xppm(fx2, q,   crx_adv) on is..ie+1  x jsd..jed
//   q_j = (q*area + xfx*fx2(i) - xfx*fx2(i+1)) / ra_x   on is..ie   x jsd..jed
//   yppm(fy,  q_j, cry_adv) on is..ie    x js..je+1
//   fx = 0.5*(fx + fx2)*xfx ; fy = 0.5*(fy + fy2)*yfx
// q is mutated in place by the two copy_corners calls, exactly as in Fortran.
// fx2/fy2 are fv_tp_2d's OWN scratch and are distinct from del6_vt_flux's
// fx2/fy2 outputs; update_dz_d's zh update reads the latter.
fv3_gpu_status launch_fv_tp_2d_cuda(const HeightTransportBounds &b, int32_t hord,
                                    const LevelFilter &lf, View3<fv_real> q,
                                    View3<const fv_real> crx_adv,
                                    View3<const fv_real> cry_adv,
                                    View3<const fv_real> xfx_adv,
                                    View3<const fv_real> yfx_adv,
                                    View2<const fv_real> area, View2<const fv_real> dxa,
                                    View2<const fv_real> dya, View3<const fv_real> ra_x,
                                    View3<const fv_real> ra_y, View3<fv_real> q_i,
                                    View3<fv_real> q_j, View3<fv_real> fx,
                                    View3<fv_real> fy, View3<fv_real> fx2,
                                    View3<fv_real> fy2, void *stream) noexcept;

// --- del6_vt_flux (sw_core.F90:1544) -------------------------------------
// del-2/4/6 damping fluxes for the damped levels of update_dz_d. Uses the raw
// per-level damp(k) coefficient (NOT the (damp_c*da_min)**(nord+1) form that
// tp_core's deln_flux uses) and emits fx2/fy2 rather than folding them into
// fx/fy -- update_dz_d applies them separately, weighted by rarea.
//
// Levels are processed in nord groups: one pass per nord value in
// [0, nord_cap], each pass selecting only the damped levels whose ndif equals
// that value.
//
// nord_cap MUST be the largest nord the port can represent (ng - 1 == 2), NOT
// the scalar NumericalConfig::nord_v. The authoritative per-level order is the
// registered damping_ndif array -- cfg.nord_v only *constructs* that array in
// Fortran (dyn_core.F90:713, nord_v(k) = min(2, nord), or 0 in sponge levels).
// Capping the group loop at cfg.nord_v instead would silently give zero
// damping to every damped level whose ndif exceeded it, and because
// NumericalConfig::nord_v defaults to 0 and no current caller populates it,
// that would drop del-4/del-6 height damping entirely while passing every
// contract check. The binder therefore rejects nord_v > 2 and passes ng - 1
// here, so every value FV3 can produce is covered.
//
// fx2/fy2 are still pre-zeroed on all damped levels first, so a malformed
// damping_ndif yields zero damping rather than uninitialized workspace.
fv3_gpu_status launch_del6_vt_flux_cuda(const HeightTransportBounds &b, int32_t nord_cap,
                                        const LevelFilter &lf, View3<fv_real> q,
                                        View2<const fv_real> del6_u,
                                        View2<const fv_real> del6_v,
                                        View2<const fv_real> rarea, View3<fv_real> d2,
                                        View3<fv_real> fx2, View3<fv_real> fy2,
                                        void *stream) noexcept;

// --- z2 copy + zh update (nh_utils.F90:281-305) --------------------------
// z2(i,j,k) = zh(i,j,k) over isd..ied x jsd..jed, damped levels only.
fv3_gpu_status launch_copy_z2_cuda(const HeightTransportBounds &b, const LevelFilter &lf,
                                   View3<const fv_real> zh, View3<fv_real> z2,
                                   void *stream) noexcept;

// Authoritative area-weighted height update, both branches in one kernel:
//   damped:   zh = (z2*area + (fx(i)-fx(i+1)) + (fy(j)-fy(j+1)))
//                  / ((ra_x+ra_y) - area)
//                + ((fx2(i)-fx2(i+1)) + (fy2(j)-fy2(j+1))) * rarea
//   undamped: zh = (zh*area + (fx(i)-fx(i+1)) + (fy(j)-fy(j+1)))
//                  / ((ra_x+ra_y) - area)
// This is NOT an additive flux-divergence increment: the transported height is
// renormalised by the swept-area denominator ((ra_x+ra_y)-area).
fv3_gpu_status launch_zh_update_cuda(const HeightTransportBounds &b, const LevelFilter &lf,
                                     View3<const fv_real> z2, View3<const fv_real> fx,
                                     View3<const fv_real> fy, View3<const fv_real> fx2,
                                     View3<const fv_real> fy2, View3<const fv_real> ra_x,
                                     View3<const fv_real> ra_y, View2<const fv_real> area,
                                     View2<const fv_real> rarea, View3<fv_real> zh,
                                     void *stream) noexcept;

// --- deln_flux (tp_core.F90:1263 / wp_b_kernels.hpp deln_flux_transport) --
// Retained from milestone 3A.3 for the later WP-B.2/WP-B.4 scalar/momentum
// transport stages, which DO call tp_core's deln_flux with the
// (damp_c*da_min)**(nord+1) coefficient and DO fold fx2/fy2 into fx/fy. It is
// a different Fortran routine from del6_vt_flux above and is not part of the
// update_dz_d height path; launch_dgrid_height_transport_cuda never calls it.
fv3_gpu_status launch_deln_flux_transport_cuda(
    const HeightTransportBounds &b, int32_t nord, fv_real damp, fv_real da_min,
    View3<const fv_real> q, View2<const fv_real> del6_u, View2<const fv_real> del6_v,
    View2<const fv_real> rarea, View3<fv_real> fx, View3<fv_real> fy,
    View3<fv_real> d2, View3<fv_real> fx2, View3<fv_real> fy2, void *stream) noexcept;

} // namespace fv3::gpu::dgrid
