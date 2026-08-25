#pragma once

// ===========================================================================
// Milestone 3A.8 / WP-B.5 -- the authoritative FV3 d_sw driver.
//
// Authoritative source: GEOS-ESM/GFDL_atmos_cubed_sphere, tag geos/v3.0.0,
// SHA ca30e26305fda43831095334c02a3f5e0b8c0c45,
//   model/sw_core.F90       d_sw            :520-1540
//   model/dyn_core.F90      d_sw call site  :699-802
//   model/tp_core.F90       fv_tp_2d        :109-243
//
// SCOPE. This header owns the COMPLETE ordered d_sw sequence for the production
// first-run configuration, plus the state updates d_sw performs itself and that
// no earlier work package owns. It is an ORCHESTRATION layer: every numerical
// service it can reuse, it reuses.
//
//   reused from Milestone 3A.4  launch_fv_tp_2d_cuda, launch_del6_vt_flux_cuda,
//                               launch_copy_corners_cuda, launch_ra_x/y_cuda
//   reused from Milestone 3A.5  launch_scalar_transport_cuda,
//                               launch_deln_flux_transport_(level_)cuda
//   reused from Milestone 3A.6  the whole wp_b_momentum_cuda.cuh surface
//   reused from Milestone 3A.7  the whole wp_b_divergence_cuda.cuh surface
//   reused from WP-D            a2b_ord4 (through launch_smag_corner_cuda)
//
// NEW code here is confined to the twelve driver-owned kernels K1..K12 listed
// in §2 -- the contravariant wind construction, the advective-flux finalize,
// the flux capacitors, and the pt/delp/w state updates. Each carries its
// authoritative line citation.
//
// ---------------------------------------------------------------------------
// §1 ORDERED d_sw OPERATION SEQUENCE (production first-run configuration)
// ---------------------------------------------------------------------------
// Live configuration: grid_type == 0, non-nested, hydrostatic == false,
// inline_q == false, do_f3d == false, do_skeb == false, SW_DYNAMICS and
// USE_COND undefined, a_imp == 1.0, hord_mt = hord_vt = hord_tm = 6,
// hord_dp = -6.
//
//  S  | sw_core.F90 | operation                          | owner
// ----|-------------|------------------------------------|--------------------
//  1  |  672-707    | contravariant ut/vt interior       | K1
//  2  |  709-779    | ut/vt tile-edge overrides (8 seq)  | K2 (8 launches)
//  3  |  791-863    | ut/vt 2x2 corner solves (4 seq)    | K3 (4 launches)
//  4  |  882-892    | xfx_adv = dt*ut ; yfx_adv = dt*vt  | K4a
//  5  |  898-921    | upwind crx_adv/cry_adv finalize    | K4b
//  6  |  927-936    | ra_x / ra_y                        | 3A.4 launch_ra_x/y
//  7  |  939-940    | fv_tp_2d(delp, hord_dp, nord_v,    | 3A.5 scalar
//     |             |   damp_v) -> fx, fy                |   transport
//  8  |  943-960    | flux capacitors cx,cy,xflux,yflux  | K5
//  9  |  963-968    | heat_source = 0 ; diss_est = 0     | K6
// 10  |  970-985    | damp_w: del6_vt_flux(w) -> dw,     | 3A.4 del6 + K7
//     |             |   heat_source = dd8 - d_con*...    |
// 11  |  986-987    | fv_tp_2d(w, hord_vt, mfx=fx,       | 3A.5 scalar
//     |             |   mfy=fy) -> gx, gy                |   transport
// 12  |  988-992    | w = OLD delp*w + div(gx,gy)        | K8
// 13  |  993-1001   | USE_COND q_con                     | NOT LIVE
// 14  | 1004-1006   | fv_tp_2d(pt, hord_tm, mfx=fx,      | 3A.5 scalar
//     |             |   mfy=fy, mass=delp, nord_t,       |   transport
//     |             |   damp_t) -> gx, gy                |
// 15  | 1034-1047   | pt = pt*delp + div(gx,gy) ;        | K9
//     |             | delp = delp + div(fx,fy) ;         |
//     |             | pt = pt/delp                       |
// 16  | 1050-1054   | dpx += div(fx,fy)                  | K10
// 17  | 1065-1074   | dt5/dt4 and is2/ie1/js2/je1        | host arithmetic
// 18  | 1077-1120   | vb construction                    | 3A.6 bgrid_vb
// 19  | 1122-1123   | ytp_v(vb) -> ub                    | 3A.6 ytp_v
// 20  | 1125-1129   | ke = vb*ub                         | 3A.6 ke_product
// 21  | 1131-1177   | ub construction                    | 3A.6 bgrid_ub
// 22  | 1179-1180   | xtp_u(ub) -> vb                    | 3A.6 xtp_u
// 23  | 1182-1186   | ke = 0.5*(ke + ub*vb)              | 3A.6 ke_average
// 24  | 1191-1216   | KE tile-corner fix (dt6 = dt/6)    | 3A.6 ke_corner_fix
// 25  | 1219-1223   | vt = u*dx   (OVERWRITES S1-S3 vt)  | 3A.6 vorticity
// 26  | 1224-1228   | ut = v*dy   (OVERWRITES S1-S3 ut)  |   precursor
// 27  | 1231-1235   | wk = rarea*curl(vt,ut)             |   (one launcher)
// 28  | 1249-1255   | w = w / NEW delp                   | K11
// 29  | 1256-1262   | damp_w: w = w + dw                 | K12
// 30  | 1263-1269   | USE_COND q_con/delp                | NOT LIVE
// 31  | 1277-1423   | divergence damping, ke += vort     | 3A.7 (both arms)
// 32  | 1425-1436   | d_con: ub/vb = vort differences    | 3A.7 dcon_vort_diff
// 33  | 1439-1459   | vort = wk + f0                     | 3A.6 vorticity_abs
// 34  | 1461-1462   | fv_tp_2d(vort, hord_vt) -> fx, fy  | 3A.5 scalar
//     |             |                                    |   transport
// 35  | 1463-1472   | u/v D-wind update from vt,ut,ke,fx,| 3A.6 dwind_update
//     |             |   fy                               |
// 36  | 1476-1479   | damp_v: del6_vt_flux(wk) -> ut,vt  | 3A.7 vorticity_damping
// 37  | 1481-1520   | d_con heat-source chain            | 3A.7 dcon_scale +
//     |             |                                    |   dcon_heat_source
// 38  | 1523-1534   | damp_v: u += vt ; v -= ut          | 3A.7 dwind_damping_add
//
// NOT LIVE arms, all fenced rather than silently skipped (§6):
//   #ifdef SW_DYNAMICS          :560-670, :922-925, :1057-1062, :1536
//   #ifdef USE_COND             :993-1001, :1263-1269
//   inline_q                    :1009-1032
//   do_f3d / #ifdef ROT3        :1238-1248, and vort = wk + f0*z_rat at :1445
//   do_skeb                     :981-983, :1508-1513
//   grid_type >= 3              :867-880 (ut = uc, vt = vc)
//   nested                      :675-687, :1280-1296, :1069-1070, :1191, :1372
//   hydrostatic                 skips S10-S12 and S28-S29 entirely
//
// ---------------------------------------------------------------------------
// §2 DRIVER-OWNED KERNELS
// ---------------------------------------------------------------------------
// These are the operations that no work package owns because they are d_sw's
// own arithmetic rather than a reusable numerical service.
//
//  K1   contravariant ut/vt interior              :672-707
//  K2   ut/vt tile-edge overrides                 :709-779   (8 ordered)
//  K3   ut/vt 2x2 corner solves                   :791-863   (4 ordered)
//  K4a  xfx_adv/yfx_adv = dt*ut / dt*vt           :882-892
//  K4b  upwind crx_adv/cry_adv + xfx/yfx rescale  :898-921
//  K5   flux capacitors cx, cy, xflux, yflux      :943-960
//  K6   heat_source / diss_est zeroing            :963-968
//  K7   dw and the damp_w heat-source seed        :975-984
//  K8   mass-weighted w update (OLD delp)         :988-992
//  K9   pt / delp update                          :1034-1047
//  K10  dpx accumulation                          :1050-1054
//  K11  w = w / NEW delp                          :1250-1254
//  K12  w += dw                                   :1256-1262
//
// ---------------------------------------------------------------------------
// §3 ANTI-DEPENDENCE SUMMARY
// ---------------------------------------------------------------------------
//  field | OLD value's last consumer         | NEW value's first producer | workspace?
// -------|-----------------------------------|----------------------------|-----------
//  delp  | S14 (mass=delp) and S12 (w*delp)  | S15 (delp += div(fx,fy))   | NO -- K9
//        |                                   |                            | reads delp
//        |                                   |                            | once and
//        |                                   |                            | writes pt
//        |                                   |                            | and delp in
//        |                                   |                            | the SAME
//        |                                   |                            | thread, so
//        |                                   |                            | the old
//        |                                   |                            | value lives
//        |                                   |                            | in a
//        |                                   |                            | register.
//  pt    | S15 itself (pt*delp)              | S15                        | NO (same)
//  w     | S12 (delp*w)                      | S12                        | NO (same
//        |                                   |                            | thread)
//  w     | S28 divides by NEW delp, 16 rows  | S28                        | NO
//        | after S12 mass-weighted it by OLD |                            |
//        | delp. The separation is genuine   |                            |
//        | and MUST be preserved: fusing S12 |                            |
//        | and S28 would divide by the wrong |                            |
//        | delp.                             |                            |
//  ut/vt | S1-S3 meaning (contravariant C    | S25/S26 (u*dx, v*dy)       | NO, but
//        | winds) last read by S18/S21/S24   |                            | NEVER fuse
//        | ut/vt second meaning last read by | S36 (del6 diffusive flux)  | across the
//        | S35                               |                            | boundary
//  ke    | S35                               | S20                        | NO
//  vort  | S32 reads the DAMPING vort        | S33 (wk + f0)              | NO -- S32
//        | S34 reads the ABSOLUTE vort       | S36 (del6 d2 scratch)      | must precede
//        |                                   |                            | S33
//  wk    | S33 and S36                       | S27                        | NO -- S10
//        |                                   |                            | uses wk as
//        |                                   |                            | del6 scratch
//        |                                   |                            | while it is
//        |                                   |                            | dead
//  fx/fy | S16 (delp fluxes)                 | S34 (vorticity fluxes)     | NO
//        | S35 (vorticity fluxes)            | S37 (v*rdy, u*rdx)         | NO
//  uc/vc | S21/S31-L (sign switches)         | S31-H (DESTROYS both)      | NO --
//        |                                   |                            | nothing in
//        |                                   |                            | d_sw reads
//        |                                   |                            | uc/vc after
//        |                                   |                            | S31
//  dw    | S29                               | S10                        | dedicated
//        |                                   |                            | arena slot;
//        |                                   |                            | lifetime
//        |                                   |                            | spans S10..
//        |                                   |                            | S29
//
// ---------------------------------------------------------------------------
// §4 PER-LEVEL CONTROL
// ---------------------------------------------------------------------------
// dyn_core.F90:699-765 recomputes eleven controls inside its k loop. Which of
// them genuinely vary with k, and how each reaches the GPU:
//
//  control    | per level? | why                                | carried by
// ------------|------------|------------------------------------|---------------------
//  nord_k     | YES        | 0 on all 3 sponge branches         | damping_nord_k
//  nord_v(k)  | YES        | 0 on sponge 1-2 if do_vort_damp    | damping_ndif
//  nord_w     | YES        | 0 on all 3 sponge branches         | damping_nord_w
//  nord_t     | NO         | :715 assignment is never reset     | config.nord_t
//  damp_vt(k) | YES        | 0.5*d2_divg on sponge 1-2          | damping_v
//  damp_w     | YES        | d2_divg on all 3 sponge branches   | damping_w
//  damp_t     | NO         | :718 copies the PRE-sponge         | damping_t (an
//             |            | damp_vt value, which the later     | array whose
//             |            | damp_vt(k) assignment cannot       | entries happen to
//             |            | change                             | be equal)
//  d2_divg    | YES        | three distinct sponge values       | damping_d2_bg
//  d4dmp(k)   | YES        | dyn_core.F90:684                   | damping_d4_bg
//  d_con_k    | YES        | 0 on all 3 sponge branches         | damping_d_con
//  kgb        | YES        | ke_bg at k==npz, 0 elsewhere       | damping_kgb
//
// Consequence, and the production behaviour this milestone must preserve: with
// n_sponge > 0 the top levels take d_sw's nord == 0 divergence-damping arm and
// the interior levels take the higher-order arm IN THE SAME d_sw INVOCATION.
// Both branch families are therefore launched on every call, each filtered to
// the levels that own it. Test B exercises exactly this.
//
// nord_cap values are bounded from the HOST without reading any device array,
// because the Fortran expressions bound them structurally:
//   nord_k(k)  in {0, nord}          -> cap = config.nord
//   nord_v(k)  in {0, min(2,nord)}   -> cap = min(2, config.nord)
//   nord_w(k)  in {0, min(2,nord)}   -> cap = min(2, config.nord)
//   nord_t     == min(2, nord)       -> scalar
// No D2H is performed anywhere in this driver.
//
// ---------------------------------------------------------------------------
// §5 FLUX CAPACITORS
// ---------------------------------------------------------------------------
// Four real(kind=8) accumulators, all intent(INOUT) in d_sw and therefore
// carrying state ACROSS acoustic substeps (sw_core.F90:548-551):
//
//   cx(is:ie+1, jsd:jed)   += crx_adv       :944-948
//   xflux(is:ie+1, js:je)  += fx            :949-953   (dyn_core's mfx)
//   cy(isd:ied, js:je+1)   += cry_adv       :955-957
//   yflux(is:ie, js:je+1)  += fy            :958-960   (dyn_core's mfy)
//   dpx(is:ie, js:je)      += div(fx,fy)    :1050-1054
//
// All five accumulate AFTER their producer and BEFORE the prognostic update
// that would overwrite it: cx/cy read the S5 advective Courant numbers, and
// xflux/yflux/dpx read the S7 delp fluxes before S34 reuses fx/fy for the
// vorticity transport. dpx is at S16, i.e. AFTER S15 has already updated delp,
// but it reads fx/fy rather than delp so the ordering is not load-bearing --
// it is nonetheless kept at S16 because that is where Fortran puts it.
//
// No coefficient, no sign flip, no scaling: every one is a plain `+=`. The
// accumulation happens in accumulator_real (kind=8) even in an R4 build, which
// is what the Fortran declaration demands; K5 and K10 therefore promote the
// fv_real addend to accumulator_real before adding. NEITHER is initialized by
// d_sw -- dyn_core zeroes them once per k_split (dyn_core.F90:379-392), so this
// driver must never zero them. Test J accumulates across two invocations for
// exactly that reason.
//
// heat_source and diss_est are the opposite case: intent(OUT), zeroed by S9 on
// every call, accumulated only WITHIN one call (S10 seeds, S37 adds). They are
// not flux capacitors.
//
// ---------------------------------------------------------------------------
// §6 UNSUPPORTED CONFIGURATION FENCING
// ---------------------------------------------------------------------------
// launch_d_sw_cuda validates BEFORE its first launch, so a rejected
// configuration cannot leave prognostic state half-updated. Rejected with
// FV3_GPU_UNSUPPORTED_CONFIGURATION:
//
//   grid_type != 0            the >= 3 arm replaces S1-S3 with ut=uc, vt=vc
//   nested != 0               changes S1-S3, S17, S24 and S31 simultaneously,
//                             and TransportBounds::nested is the only signal
//                             the contract carries (DivergenceCornerArgs still
//                             has no nested flag -- see the 3A.7 handoff)
//   hydrostatic != 0          S10-S12 and S28-S29 vanish; w is not prognostic
//   inline_q != 0             replaces S15 with the wk-buffered form and adds
//                             an nq tracer loop
//   use_cond != 0             adds S13 and S30
//   do_f3d != 0               changes S28 and S33
//   do_skeb != 0              adds diss_est production at S10 and S37
//   rot3 != 0                 the ROT3 arm of S28
//   abs(hord_mt/vt/tm/dp) != 6  only order-6 PPM is implemented
//   nord < 0 or nord > 3      nt = nord-1 would index past ng = 3
//   npx == 0 or npy == 0      panel-seam stencils would silently be interior
//   a_imp != 1.0              not read by d_sw, but a different value means the
//                             caller is not running the production first-run
//                             configuration this driver was validated against
//   stream == nullptr         S31's smag_corner reuses WP-D's a2b_ord4, which
//                             rejects a null stream; without this fence that
//                             would surface thirty stages in, after pt, delp, w
//                             and dpx had already been mutated
//
// Nothing here is a numerical approximation of an unsupported mode.
//
// nk != npz, and any missing state / arena / geometry view, are rejected with
// FV3_GPU_INVALID_ARGUMENT (a wiring defect, not a configuration choice) -- also
// before the first launch.
//
// ---------------------------------------------------------------------------
// §7 corner_edge_service DISPOSITION
// ---------------------------------------------------------------------------
// This driver does NOT use NumericalDispatch::corner_edge_service, and that is
// deliberate rather than an omission.
//
// Every corner/edge operation the production grid_type == 0 d_sw path needs is
// a LOCAL, on-rank, ng x ng device operation with FV3-specific semantics:
//
//   K2/K3            tile-edge and 2x2 corner wind solves -- specific algebra,
//                    not a generic fill
//   copy_corners     inside fv_tp_2d / deln_flux / del6_vt_flux (3A.4/3A.5)
//   fill_corners     BGRID scalar + DGRID vector inside the S31 high-order arm
//                    (3A.7)
//   KE corner fix    S24 (3A.6)
//   delpc corner     the four extra-term removals in S31's nord == 0 arm (3A.7)
//
// Each is already implemented as a focused local kernel by the work package
// that owns the surrounding numerics. Routing them through the generic
// CornerServiceRequest abstraction would require encoding FV3-specific algebra
// into a generic operation enum, which the 3A.7 review explicitly rejected.
//
// corner_edge_service remains reserved for genuine halo/communication corner
// work at an MPI boundary. It is NOT required by this path, so it stays a
// fail-stub. Corner SUPPORT is complete for the production path; the generic
// dispatch slot is simply not the mechanism.
// ===========================================================================

#include "dgrid/wp_b_cuda.cuh"
#include "dgrid/wp_b_divergence_cuda.cuh"
#include "dgrid/wp_b_momentum_cuda.cuh"

#include <cstdint>

namespace fv3::gpu::dgrid {

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
// Every field is a d_sw or dyn_core argument that is genuinely scalar. Anything
// dyn_core recomputes per k lives in DSwLevels instead (§4).
struct DSwOptions {
  // Acoustic substep dt (dyn_core.F90:798). Signed; S10 uses abs(dt) and S31's
  // smag_corner arm uses abs(dt), while S4/S17/S24 use the signed value.
  fv_real dt = fv_real(0);

  int32_t hord_mt = 0;  // flagstruct%hord_mt -> S18-S22
  int32_t hord_vt = 0;  // flagstruct%hord_vt -> S11, S34
  int32_t hord_tm = 0;  // flagstruct%hord_tm -> S14
  int32_t hord_dp = 0;  // flagstruct%hord_dp -> S7
  fv_real lim_fac = fv_real(1);

  // Cap on the per-level integer orders; see §4 for why these are host-known.
  int32_t nord = 0;    // flagstruct%nord   -> caps nord_k
  int32_t nord_t = 0;  // min(2, nord)      -> S14, genuinely scalar

  fv_real dddmp = fv_real(0);
  fv_real da_min = fv_real(0);
  fv_real da_min_c = fv_real(0);

  int32_t grid_type = 0;
  int32_t hydrostatic = 0;
  int32_t inline_q = 0;
  int32_t use_cond = 0;
  int32_t do_f3d = 0;
  int32_t do_skeb = 0;
  int32_t rot3 = 0;
  // NumericalConfig::a_imp. Not read by d_sw; checked only as a production
  // configuration guard (§6).
  double a_imp = 1.0;
};

// Per-level controls, one View1 per dyn_core k-loop quantity that varies with
// k. Integral orders arrive in fv_real storage for the ABI reason documented on
// LevelFilter::ndif and are rounded to nearest.
struct DSwLevels {
  View1<const fv_real> nord_k{};  // damping_nord_k
  View1<const fv_real> nord_v{};  // damping_ndif
  View1<const fv_real> nord_w{};  // damping_nord_w
  View1<const fv_real> d2_bg{};   // damping_d2_bg (d2_divg)
  View1<const fv_real> d4_bg{};   // damping_d4_bg (d4dmp)
  View1<const fv_real> damp_v{};  // damping_v     (damp_vt)
  View1<const fv_real> damp_w{};  // damping_w
  View1<const fv_real> damp_t{};  // damping_t
  View1<const fv_real> d_con{};   // damping_d_con (d_con_k)
  View1<const fv_real> kgb{};     // damping_kgb
};

// Grid metrics. DivergenceGeometry already carries most of what d_sw needs, so
// it is embedded rather than re-listed; `extra` holds the metrics only the
// non-damping stages read.
struct DSwGeometry {
  DivergenceGeometry div{};
  A2BGeometry a2b{};        // for S31's smag_corner arm
  View2<const fv_real> area{};
  View2<const fv_real> dxa{};
  View2<const fv_real> dya{};
  View2<const fv_real> rdxa{};
  View2<const fv_real> rdya{};
  View2<const fv_real> del6_u{};
  View2<const fv_real> del6_v{};
  View2<const fv_real> cosa{};    // B-grid, for S18/S21
  View2<const fv_real> rsina{};   // B-grid, for S18/S21
  View2<const fv_real> rsin_u{};  // for K1
  View2<const fv_real> rsin_v{};  // for K1
  View2<const fv_real> f0{};      // for S33
};

// Prognostic and shared-temporary state. Names match the d_sw dummy arguments,
// with the dyn_core actual argument noted where they differ.
struct DSwState {
  View3<fv_real> u{}, v{}, w{}, delp{}, pt{};
  View3<fv_real> uc{}, vc{};
  View3<const fv_real> ua{}, va{};
  View3<fv_real> divgd{};   // d_sw's divg_d
  View3<fv_real> delpc{};   // dyn_core passes vt(:,:,k) here
  View3<fv_real> ptc{};
  View3<fv_real> ut{}, vt{};
  View3<fv_real> crx{}, cry{}, xfx{}, yfx{};  // crx_adv/cry_adv/xfx_adv/yfx_adv
  View3<fv_real> heat_source{};
  View3<fv_real> diss_est{};
  View3<const fv_real> z_rat{};
  // The kind=8 flux capacitors (§5).
  View3<accumulator_real> mfx{};  // d_sw's xflux
  View3<accumulator_real> mfy{};  // d_sw's yflux
  View3<accumulator_real> cx{};
  View3<accumulator_real> cy{};
  View3<accumulator_real> dpx{};
};

// ---------------------------------------------------------------------------
// Workspace arena
// ---------------------------------------------------------------------------
// Every d_sw local that is not a registered field. The runtime sizes
// WorkspaceId::transport at array_count whole (ni x nj x (npz+1)) fv_real
// arrays; runtime/fv3_gpu_runtime.cpp cites this constant, and the binder
// re-checks the slice against it, so a stale literal there fails loudly.
struct WpBDSwArena {
  // 14 d_sw locals with real lifetimes.
  View3<fv_real> ub{}, vb{}, wk{}, ke{}, vort{};
  View3<fv_real> fx{}, fy{}, gx{}, gy{};
  View3<fv_real> ra_x{}, ra_y{};
  View3<fv_real> fx2{}, fy2{};
  View3<fv_real> dw{};
  // 7 fv_tp_2d / deln_flux scratch arrays, reused sequentially by the delp, w,
  // pt and vort transports. Safe to reuse: nothing in them survives one
  // launch_scalar_transport_cuda call.
  View3<fv_real> q_i{}, q_j{}, fx2_tp{}, fy2_tp{}, d2{}, fx2_dn{}, fy2_dn{};
  // 4 PPM momentum workspace arrays (S19, S22). `smt` is a Fortran LOGICAL and
  // is kept int32 so it is never compared against a rounded real.
  View3<fv_real> bl{}, br{}, b0{};
  View3<int32_t> smt{};
  // 8 smag_corner private locals. These MUST NOT alias d_sw's ut/vt/wk: S31
  // runs while wk holds the S27 relative vorticity that S33 and S36 still need.
  SmagCornerWorkspace smag{};
  // 1 slot subdivided into the derived per-level coefficient arrays.
  View1<fv_real> deln_damp_delp{}, deln_gate_delp{};
  View1<fv_real> deln_damp_pt{}, deln_gate_pt{}, nord_t_level{};
  View1<fv_real> del6_damp_w{}, del6_damp_v{};

  // 14 + 7 + 4 + 8 + 1
  static constexpr int64_t array_count = 34;
  // Number of npz-length fv_real arrays carved from the last slot.
  static constexpr int64_t level_array_count = 7;
};

// ---------------------------------------------------------------------------
// Driver-owned launchers (K1..K12)
// ---------------------------------------------------------------------------
// Exposed individually so the permanent tests can assert each in isolation and
// so a reviewer can diff one launcher against one Fortran range. The driver
// calls them in the §1 order.

// K1 -- sw_core.F90:686-705, the `.not.nested` grid_type < 3 arm.
//   ut(i,j) = (uc - 0.25*cosa_u*(vc(i-1,j)+vc(i,j)+vc(i-1,j+1)+vc(i,j+1)))*rsin_u
//     over j in jsd..jed EXCLUDING j == 0, 1, npy-1, npy ; i in is-1..ie+2
//   vt(i,j) = (vc - 0.25*cosa_v*(uc(i,j-1)+uc(i+1,j-1)+uc(i,j)+uc(i+1,j)))*rsin_v
//     over j in js-1..je+2 EXCLUDING j == 1, npy ; i in isd..ied
// The excluded j rows are left UNTOUCHED, not zeroed: K2 writes them where the
// rank owns a tile edge, and where it does not they are halo values the caller
// supplied.
fv3_gpu_status launch_dsw_contravariant_cuda(const TransportBounds &b,
                                            View3<const fv_real> uc,
                                            View3<const fv_real> vc,
                                            View2<const fv_real> cosa_u,
                                            View2<const fv_real> cosa_v,
                                            View2<const fv_real> rsin_u,
                                            View2<const fv_real> rsin_v,
                                            View3<fv_real> ut, View3<fv_real> vt,
                                            void *stream) noexcept;

// K2 -- sw_core.F90:709-779. Eight ordered launches in the authoritative order
// West, East, South, North; within each face the sin_sg sign-switch pass must
// complete before the 4-point pass, which reads it. Each face is skipped when
// the rank does not own that tile edge.
fv3_gpu_status launch_dsw_edge_winds_cuda(const TransportBounds &b, fv_real dt,
                                          View3<const fv_real> uc,
                                          View3<const fv_real> vc,
                                          View2<const fv_real> cosa_u,
                                          View2<const fv_real> cosa_v,
                                          View3<const fv_real> sin_sg, View3<fv_real> ut,
                                          View3<fv_real> vt, void *stream) noexcept;

// K3 -- sw_core.F90:790-861. Four ordered 2x2 corner solves (sw, se, ne, nw).
// Within one corner block the four statements read only K1/K2 output, so they
// are launched as a single thread per level per corner and there is no
// intra-block read-after-write. Between blocks there is none either (the four
// corners are disjoint), but they are launched in Fortran order regardless.
fv3_gpu_status launch_dsw_corner_winds_cuda(const TransportBounds &b,
                                            View3<const fv_real> uc,
                                            View3<const fv_real> vc,
                                            View2<const fv_real> cosa_u,
                                            View2<const fv_real> cosa_v,
                                            View3<fv_real> ut, View3<fv_real> vt,
                                            void *stream) noexcept;

// K4a -- sw_core.F90:882-892. xfx_adv = dt*ut over jsd..jed x is..ie+1 ;
// yfx_adv = dt*vt over js..je+1 x isd..ied.
fv3_gpu_status launch_dsw_advective_flux_cuda(const TransportBounds &b, fv_real dt,
                                              View3<const fv_real> ut,
                                              View3<const fv_real> vt,
                                              View3<fv_real> xfx_adv,
                                              View3<fv_real> yfx_adv,
                                              void *stream) noexcept;

// K4b -- sw_core.F90:898-921. The upwind branch that produces crx_adv/cry_adv
// and rescales xfx_adv/yfx_adv IN PLACE. The rescale must happen in the same
// thread as the crx_adv computation because it consumes the pre-rescale value.
fv3_gpu_status launch_dsw_courant_finalize_cuda(const TransportBounds &b,
                                                View2<const fv_real> dx,
                                                View2<const fv_real> dy,
                                                View2<const fv_real> rdxa,
                                                View2<const fv_real> rdya,
                                                View3<const fv_real> sin_sg,
                                                View3<fv_real> crx_adv,
                                                View3<fv_real> cry_adv,
                                                View3<fv_real> xfx_adv,
                                                View3<fv_real> yfx_adv,
                                                void *stream) noexcept;

// K5 -- sw_core.F90:943-960. The four flux capacitors (§5). Plain `+=` in
// accumulator_real; no coefficient and no sign flip.
fv3_gpu_status launch_dsw_flux_capacitor_cuda(const TransportBounds &b,
                                              View3<const fv_real> crx_adv,
                                              View3<const fv_real> cry_adv,
                                              View3<const fv_real> fx,
                                              View3<const fv_real> fy,
                                              View3<accumulator_real> cx,
                                              View3<accumulator_real> cy,
                                              View3<accumulator_real> xflux,
                                              View3<accumulator_real> yflux,
                                              void *stream) noexcept;

// K6 -- sw_core.F90:963-968. heat_source = 0 and diss_est = 0 over is..ie x
// js..je. Both are intent(OUT) per call, so this is a required reset, not an
// arena-hiding zero fill.
fv3_gpu_status launch_dsw_heat_reset_cuda(const TransportBounds &b,
                                          View3<fv_real> heat_source,
                                          View3<fv_real> diss_est,
                                          void *stream) noexcept;

// K7 -- sw_core.F90:975-984.
//   dw = (fx2(i)-fx2(i+1)+fy2(j)-fy2(j+1))*rarea
//   heat_source = kgb(k)*abs(dt) - d_con(k)*dw*(w + 0.5*dw)
// Note `w` here is the PRE-transport w (S11 has not run yet) and d_con is the
// per-level d_con_k. Filtered to the levels with damp_w > 1.E-5.
fv3_gpu_status launch_dsw_w_damping_heat_cuda(const TransportBounds &b,
                                              const LevelFilter &lf, fv_real dt,
                                              View1<const fv_real> kgb,
                                              View1<const fv_real> d_con,
                                              View3<const fv_real> w,
                                              View3<const fv_real> fx2,
                                              View3<const fv_real> fy2,
                                              View2<const fv_real> rarea,
                                              View3<fv_real> dw,
                                              View3<fv_real> heat_source,
                                              void *stream) noexcept;

// K8 -- sw_core.F90:988-992. w = delp*w + div(gx,gy)*rarea, using the OLD delp.
// Must run BEFORE K9 updates delp.
fv3_gpu_status launch_dsw_w_mass_update_cuda(const TransportBounds &b,
                                             View3<const fv_real> delp,
                                             View3<const fv_real> gx,
                                             View3<const fv_real> gy,
                                             View2<const fv_real> rarea,
                                             View3<fv_real> w, void *stream) noexcept;

// K9 -- sw_core.F90:1034-1047, the `.not. inline_q` arm. One thread performs
// all three statements in Fortran order, so the OLD delp is a register value
// and no double buffer is needed:
//   pt   = pt*delp + div(gx,gy)*rarea
//   delp = delp    + div(fx,fy)*rarea
//   pt   = pt/delp
fv3_gpu_status launch_dsw_pt_delp_update_cuda(const TransportBounds &b,
                                              View3<const fv_real> fx,
                                              View3<const fv_real> fy,
                                              View3<const fv_real> gx,
                                              View3<const fv_real> gy,
                                              View2<const fv_real> rarea,
                                              View3<fv_real> pt, View3<fv_real> delp,
                                              void *stream) noexcept;

// K10 -- sw_core.F90:1050-1054. dpx += ((fx(i)-fx(i+1)) + (fy(j)-fy(j+1)))*rarea.
// The Fortran parenthesisation is preserved because dpx is kind=8 while fx/fy
// are working precision, so the grouping is observable in R4.
fv3_gpu_status launch_dsw_dpx_accumulate_cuda(const TransportBounds &b,
                                              View3<const fv_real> fx,
                                              View3<const fv_real> fy,
                                              View2<const fv_real> rarea,
                                              View3<accumulator_real> dpx,
                                              void *stream) noexcept;

// K11 -- sw_core.F90:1250-1254 (the `.not. do_f3d` arm). w = w / NEW delp.
fv3_gpu_status launch_dsw_w_normalize_cuda(const TransportBounds &b,
                                           View3<const fv_real> delp, View3<fv_real> w,
                                           void *stream) noexcept;

// K12 -- sw_core.F90:1256-1262. w += dw, on the damp_w > 1.E-5 levels only.
fv3_gpu_status launch_dsw_w_damping_add_cuda(const TransportBounds &b,
                                             const LevelFilter &lf,
                                             View3<const fv_real> dw, View3<fv_real> w,
                                             void *stream) noexcept;

// Derives the coefficient/gate/order arrays §4 needs from the registered
// per-level damping fields. Runs as one small device kernel over npz so that
// no host code ever reads a device array.
//
//   deln_damp_delp(k) = (damp_v(k)*da_min)**(nord_v(k)+1)
//   deln_gate_delp(k) = damp_v(k) > 1.e-4 ? 1 : 0
//   deln_damp_pt(k)   = (damp_t(k)*da_min)**(nord_t+1)
//   deln_gate_pt(k)   = damp_t(k) > 1.e-4 ? 1 : 0
//   nord_t_level(k)   = nord_t                     (constant, for mode-3 groups)
//   del6_damp_w(k)    = (damp_w(k)*da_min_c)**(nord_w(k)+1)
//   del6_damp_v(k)    = (damp_v(k)*da_min_c)**(nord_v(k)+1)
//
// The two 1.e-4 gates reproduce fv_tp_2d's activation branch
// (tp_core.F90:212/230); the del6 coefficients are gated at 1.E-5 by their own
// callers against the RAW damp array, matching sw_core.F90:972 and :1476.
fv3_gpu_status launch_dsw_level_coefficients_cuda(const TransportBounds &b,
                                                  const DSwOptions &opt,
                                                  const DSwLevels &lv,
                                                  const WpBDSwArena &ws,
                                                  void *stream) noexcept;

// ---------------------------------------------------------------------------
// The complete driver
// ---------------------------------------------------------------------------
// Executes S1..S38 in authoritative order on the caller's stream, entirely on
// device. No host numerical fallback, no per-stage H2D/D2H, no device
// synchronization: the persistent-residency architecture owns the state and
// this service only enqueues work.
//
// Returns FV3_GPU_UNSUPPORTED_CONFIGURATION for anything §6 fences, and
// FV3_GPU_INVALID_ARGUMENT for a missing view, both BEFORE the first launch.
fv3_gpu_status launch_d_sw_cuda(const TransportBounds &b, const DSwOptions &opt,
                                const DSwLevels &lv, const DSwGeometry &g,
                                const DSwState &st, const WpBDSwArena &ws,
                                void *stream) noexcept;

} // namespace fv3::gpu::dgrid
