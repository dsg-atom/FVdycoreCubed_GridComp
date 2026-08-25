#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"
#include "operators/wp_d_operators.hpp"

#include <cstdint>

namespace fv3::gpu::dgrid {

using fv3::gpu::operators::View1;
using fv3::gpu::operators::View2;
using fv3::gpu::operators::View3;

// Geometry + panel-topology parameters shared by every WP-B transport path
// (Milestone 3A.4's update_dz_d height transport and Milestone 3A.5's arbitrary
// scalar transport). It carries no field identity and no numerical option, so
// the two paths share it without sharing a contract.
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
struct TransportBounds {
  int64_t is, ie, js, je, npz;
  // Number of vertical slabs the batched launches cover: every kernel processes
  // k in [0, nk-1]. This is NOT always npz.
  //
  //   height transport  -> nk = npz + 1. update_dz_d transports the npz+1
  //                        INTERFACE heights zh (nh_utils.F90:255-305).
  //   scalar transport  -> nk = npz. d_sw transports the npz CELL-centered
  //                        delp/w/q_con/pt (sw_core.F90:939-1007).
  //
  // Kept separate from npz because LevelFilter still indexes the per-level
  // damping arrays at min(k, npz-1) -- that clamp reproduces update_dz_d's
  // `damp(km+1) = damp(km)` prologue and must not be driven by nk.
  int64_t nk = 0;
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
  // Optional ACTIVATION array, separate from the COEFFICIENT in damp.
  //
  // update_dz_d gates del6_vt_flux on the very coefficient it hands the routine
  // (nh_utils.F90:280), so damp alone is enough there. d_sw's vorticity damping
  // does not: sw_core.F90:1476-1479 tests `damp_v > 1.E-5` but passes
  // damp4 = (damp_v*da_min_c)**(nord_v+1), and the two numbers differ. When
  // gate is non-empty level_active tests gate(kk) instead of damp(kk).
  //
  // Leaving it empty selects the historical behaviour exactly, so Milestone
  // 3A.4's height path and 3A.5's scalar path are bit-for-bit unaffected.
  View1<const fv_real> gate{};
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
fv3_gpu_status launch_edge_profile_gam_cuda(const TransportBounds &b,
                                            View1<const fv_real> dp_ref,
                                            View1<fv_real> gam, void *stream) noexcept;

fv3_gpu_status launch_edge_profile_x_cuda(const TransportBounds &b,
                                          View3<const fv_real> crx,
                                          View3<const fv_real> xfx,
                                          View1<const fv_real> dp_ref,
                                          View1<const fv_real> gam,
                                          View3<fv_real> crx_adv, View3<fv_real> xfx_adv,
                                          void *stream) noexcept;

fv3_gpu_status launch_edge_profile_y_cuda(const TransportBounds &b,
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
fv3_gpu_status launch_copy_corners_cuda(const TransportBounds &b, int32_t dir,
                                        const LevelFilter &lf, View3<fv_real> q,
                                        void *stream) noexcept;

// --- ra_x / ra_y (nh_utils.F90:269-278) ----------------------------------
// ra_x(i,j) = area(i,j) + (xfx_adv(i,j,k) - xfx_adv(i+1,j,k)) on is..ie x jsd..jed
// ra_y(i,j) = area(i,j) + (yfx_adv(i,j,k) - yfx_adv(i,j+1,k)) on isd..ied x js..je
// The extents differ between the two and the parenthesisation is the Fortran's.
fv3_gpu_status launch_ra_x_cuda(const TransportBounds &b,
                                View3<const fv_real> xfx_adv, View2<const fv_real> area,
                                View3<fv_real> ra_x, void *stream) noexcept;

fv3_gpu_status launch_ra_y_cuda(const TransportBounds &b,
                                View3<const fv_real> yfx_adv, View2<const fv_real> area,
                                View3<fv_real> ra_y, void *stream) noexcept;

// --- fv_tp_2d (tp_core.F90:109, no deln_flux) ----------------------------
// Full authoritative sequence for hord == ord_in == ord_ou:
//   copy_corners(q, dir=2)
//   yppm(fy2, q,   cry_adv) on isd..ied x js..je+1
//   q_i = (q*area + yfx*fy2(j) - yfx*fy2(j+1)) / ra_y   on isd..ied x js..je
//   xppm(fx,  q_i, crx_adv) on is..ie+1  x js..je
//   copy_corners(q, dir=1)
//   xppm(fx2, q,   crx_adv) on is..ie+1  x jsd..jed
//   q_j = (q*area + xfx*fx2(i) - xfx*fx2(i+1)) / ra_x   on is..ie   x jsd..jed
//   yppm(fy,  q_j, cry_adv) on is..ie    x js..je+1
//   fx = 0.5*(fx + fx2)*wx ; fy = 0.5*(fy + fy2)*wy
// q is mutated in place by the two copy_corners calls, exactly as in Fortran.
// fx2/fy2 are fv_tp_2d's OWN scratch and are distinct from del6_vt_flux's
// fx2/fy2 outputs; update_dz_d's zh update reads the latter.
//
// wx/wy are the FINAL flux-averaging weights and are the only difference
// between fv_tp_2d's two optional-argument branches:
//   tp_core.F90:229-234 (`else`, mfx/mfy absent) -> wx = xfx_adv, wy = yfx_adv
//   tp_core.F90:216-221 (mfx/mfy present)        -> wx = mfx,     wy = mfy
// xfx_adv/yfx_adv are ALSO passed separately and are used unconditionally by
// the q_i/q_j intermediates (tp_core.F90:168-193 build fyy/fx1 from xfx/yfx in
// both branches), so a mass-flux-weighted call must supply both.
fv3_gpu_status launch_fv_tp_2d_cuda(const TransportBounds &b, int32_t hord,
                                    const LevelFilter &lf, View3<fv_real> q,
                                    View3<const fv_real> crx_adv,
                                    View3<const fv_real> cry_adv,
                                    View3<const fv_real> xfx_adv,
                                    View3<const fv_real> yfx_adv,
                                    View3<const fv_real> wx, View3<const fv_real> wy,
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
fv3_gpu_status launch_del6_vt_flux_cuda(const TransportBounds &b, int32_t nord_cap,
                                        const LevelFilter &lf, View3<fv_real> q,
                                        View2<const fv_real> del6_u,
                                        View2<const fv_real> del6_v,
                                        View2<const fv_real> rarea, View3<fv_real> d2,
                                        View3<fv_real> fx2, View3<fv_real> fy2,
                                        void *stream) noexcept;

// --- z2 copy + zh update (nh_utils.F90:281-305) --------------------------
// z2(i,j,k) = zh(i,j,k) over isd..ied x jsd..jed, damped levels only.
fv3_gpu_status launch_copy_z2_cuda(const TransportBounds &b, const LevelFilter &lf,
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
fv3_gpu_status launch_zh_update_cuda(const TransportBounds &b, const LevelFilter &lf,
                                     View3<const fv_real> z2, View3<const fv_real> fx,
                                     View3<const fv_real> fy, View3<const fv_real> fx2,
                                     View3<const fv_real> fy2, View3<const fv_real> ra_x,
                                     View3<const fv_real> ra_y, View2<const fv_real> area,
                                     View2<const fv_real> rarea, View3<fv_real> zh,
                                     void *stream) noexcept;

// --- deln_flux (tp_core.F90:1263) ----------------------------------------
// tp_core's deln_flux, which is a DIFFERENT Fortran routine from
// del6_vt_flux above and must not be confused with it:
//
//                      del6_vt_flux              deln_flux
//   coefficient        raw per-level damp(k)     (damp_c*da_min)**(nord+1)
//   output             separate fx2/fy2          folded into fx/fy
//   activation         damp(k) > 1.E-5           damp_c > 1.e-4
//   copy_corners       2 + 2*nord (of d2)        2 + 2*nord (of d2)
//   mass weighting     none                      optional, 0.5*damp*(m+m)
//
// Only deln_flux has the (damp_c*da_min)**(nord+1) coefficient, so its
// diffusive flux scales as damp_c**(nord+1) -- NOT linearly in damp_c the way
// del6_vt_flux does. Milestone 3A.5 test G exploits exactly that difference.
//
// damp_c is passed RAW; this routine forms (damp_c*da_min)**(nord+1) itself,
// mirroring tp_core.F90:213/231 where fv_tp_2d builds `damp` immediately before
// the call. The `damp_c > 1.e-4` activation branch is the CALLER's (it lives in
// fv_tp_2d, not deln_flux); launch_scalar_transport_cuda applies it.
//
// mass_weighted mirrors `present(mass)`:
//   false -> d2 = damp*q ; fx += fx2 ; fy += fy2                (tp_core:1276, 1388)
//   true  -> d2 = q      ; fx += 0.5*damp*(mass(i-1,j)+mass(i,j))*fx2
//                        ; fy += 0.5*damp*(mass(i,j-1)+mass(i,j))*fy2  (:1373)
// `mass` is not dereferenced when mass_weighted is false.
//
// Milestone 3A.5 closes defect D3 (fenced during 3A.4): the 2 + 2*nord
// copy_corners rotations of d2 required at tp_core.F90:1313, 1326, 1354 and
// 1366 are now performed, so nord > 0 is supported on corner-owning blocks.
fv3_gpu_status launch_deln_flux_transport_cuda(
    const TransportBounds &b, int32_t nord, fv_real damp_c, fv_real da_min,
    bool mass_weighted, const LevelFilter &lf, View3<const fv_real> q,
    View3<const fv_real> mass, View2<const fv_real> del6_u, View2<const fv_real> del6_v,
    View2<const fv_real> rarea, View3<fv_real> fx, View3<fv_real> fy,
    View3<fv_real> d2, View3<fv_real> fx2, View3<fv_real> fy2, void *stream) noexcept;

// --- WP-B.2 arbitrary scalar transport -----------------------------------
// Numerical options for one fv_tp_2d scalar-transport call. These are exactly
// fv_tp_2d's optional-argument combinations as exercised by d_sw
// (sw_core.F90:939, 986, 993, 1004); no other combination occurs:
//
//   scalar  hord      wx/wy      deln_flux         mass
//   delp    hord_dp   xfx/yfx    nord_v, damp_v    absent
//   w       hord_vt   mfx/mfy    absent            absent
//   q_con   hord_dp   mfx/mfy    nord_t, damp_t    delp
//   pt      hord_tm   mfx/mfy    nord_t, damp_t    delp
//
// mass_flux_weighting therefore also selects deln_flux's mass mode: Fortran
// passes `mass` only in the mfx/mfy branch (tp_core.F90:222-224) and never in
// the xfx/yfx branch (tp_core.F90:236-238). Encoding it as one flag makes the
// off-manifold combinations unrepresentable rather than merely rejected.
struct ScalarTransportOptions {
  // Signed hord. abs(hord) selects the PPM order; hord < 0 additionally applies
  // the al >= 0 positivity clamp (tp_core.F90:390-394). The production
  // configuration uses hord_dp = -6, hord_vt = hord_tm = 6.
  int32_t hord = 0;
  // 0 -> average fluxes against xfx_adv/yfx_adv; 1 -> against mfx/mfy.
  int32_t mass_flux_weighting = 0;
  // Mirrors `present(nord) .and. present(damp_c)` at the fv_tp_2d call site.
  int32_t deln_flux_enabled = 0;
  int32_t nord = 0;
  fv_real damp_c = fv_real(0);
  fv_real da_min = fv_real(0);
};

// Reusable internal WP-B service: transport one arbitrary scalar field.
//
// This is deliberately NOT wired into NumericalDispatch. WP-B.2 owns the
// transport machinery; the d_sw driver that calls it four times per level (and
// then applies the order-critical pt/delp/w state updates and the cx/cy/xflux/
// yflux flux-capacitor accumulation) is WP-B.3/WP-B.5 work. Exposing the
// service at this granularity is what lets dgrid_dynamics reuse it later
// without a redesign: the caller supplies views, this launches kernels.
//
// The scalar `q` is mutated in place by fv_tp_2d's two copy_corners calls,
// exactly as in Fortran. The transported result is delivered as the flux pair
// (fx, fy); applying it to model state is the caller's business, because the
// authoritative update differs per scalar (delp is an additive flux
// divergence, pt/w/q_con are mass-weighted, zh is renormalised by the swept
// area).
//
// Returns FV3_GPU_UNSUPPORTED_CONFIGURATION when abs(hord) != 6 (only the
// order-6 PPM path is ported) or when nord is outside [0, ng-1] with damping
// enabled, and does so BEFORE launching any kernel, so model state is never
// partially mutated by a rejected configuration.
fv3_gpu_status launch_scalar_transport_cuda(
    const TransportBounds &b, const ScalarTransportOptions &opt, const LevelFilter &lf,
    View3<fv_real> q, View3<const fv_real> mass, View3<const fv_real> crx_adv,
    View3<const fv_real> cry_adv, View3<const fv_real> xfx_adv,
    View3<const fv_real> yfx_adv, View3<const fv_real> mfx, View3<const fv_real> mfy,
    View2<const fv_real> area, View2<const fv_real> rarea, View2<const fv_real> dxa,
    View2<const fv_real> dya, View2<const fv_real> del6_u, View2<const fv_real> del6_v,
    View3<const fv_real> ra_x, View3<const fv_real> ra_y, View3<fv_real> q_i,
    View3<fv_real> q_j, View3<fv_real> fx2_tp, View3<fv_real> fy2_tp, View3<fv_real> d2,
    View3<fv_real> fx2_dn, View3<fv_real> fy2_dn, View3<fv_real> fx, View3<fv_real> fy,
    void *stream) noexcept;

} // namespace fv3::gpu::dgrid
