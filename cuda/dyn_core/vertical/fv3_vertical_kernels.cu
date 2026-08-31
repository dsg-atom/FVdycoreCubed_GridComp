#include "fv3_vertical_kernels.cuh"
#include "fv3_vertical_column.hpp"
#include "fv3_riem_solver_c.cuh"
#include "fv3_gpu_errors.hpp"
// Milestone 3A.9 §4: the C-grid vertical entry is WP-C sequencing, but the
// update_dz_c kernels and the A-grid corner fills it drives are WP-A property
// (project_architecture.md: WP-A owns "C-grid dz updates"). WP-C calls them
// rather than transcribing a second copy, exactly as WP-B's smag_corner calls
// WP-D's a2b_ord4. fv3_wp_c_vertical therefore links fv3_wp_a_cgrid; there is
// no cycle, because fv3_wp_a_cgrid depends only on the WP-E runtime.
#include "fv3_cgrid_dz.hpp"
#include "fv3_cgrid_corners.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace fv3::gpu::vertical {
namespace {
struct View3 { fv_real *p; int64_t l0,l1,l2,s0,s1,s2; __device__ fv_real &at(int64_t i,int64_t j,int64_t k)const{return p[(i-l0)*s0+(j-l1)*s1+(k-l2)*s2];} };
struct CView3 { const fv_real *p; int64_t l0,l1,l2,s0,s1,s2; __device__ fv_real at(int64_t i,int64_t j,int64_t k)const{return p[(i-l0)*s0+(j-l1)*s1+(k-l2)*s2];} };
struct View2 { fv_real *p; int64_t l0,l1,s0,s1; __device__ fv_real &at(int64_t i,int64_t j)const{return p[(i-l0)*s0+(j-l1)*s1];} };
struct CView2 { const fv_real *p; int64_t l0,l1,s0,s1; __device__ fv_real at(int64_t i,int64_t j)const{return p[(i-l0)*s0+(j-l1)*s1];} };
__global__ void reconstruct_height_kernel(fv_real *zh, const fv_real *delz,
                                          const fv_real *zs,
                                          ColumnLaunch l) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= l.columns) return;
  Column<fv_real> z{zh + c * l.column_stride, l.level_stride};
  Column<const fv_real> dz{delz + c * l.column_stride, l.level_stride};
  reconstruct_height_column(z, dz, zs[c], l.levels);
}

__global__ void height_boundary_kernel(fv_real *zh, fv_real *delz, fv_real *ws,
                                       const fv_real *zs, fv_real rdt,
                                       fv_real dz_min, ColumnLaunch l) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= l.columns) return;
  Column<fv_real> z{zh + c * l.column_stride, l.level_stride};
  Column<fv_real> dz{delz + c * l.column_stride, l.level_stride};
  ws[c] = update_height_boundary_column(z, dz, zs[c], rdt, dz_min, l.levels);
}

// Authoritative closure of update_dz_d (nh_utils.F90:307-322): ws from the
// bottom interface, then a bottom-to-top monotonicity floor on zh.
//
// write_delz is a Milestone 3A.4 correction. Fortran update_dz_d declares delz
// intent(out) but NEVER assigns it -- delz is produced later by the Riemann
// solver from the final zh. Writing it here made the CUDA path diverge from the
// authoritative sequence, so launch_update_dz_d now passes false.
// launch_dgrid_height_v2 keeps true: it is the standalone height-boundary
// operator whose callers rely on it to derive delz, and it is not the
// update_dz_d path.
__global__ void height_boundary_v2_kernel(View3 zh,View3 delz,View2 ws,View2 zs,
 int64_t is,int64_t ie,int64_t js,int64_t je,int km,fv_real rdt,fv_real dz_min,
 bool write_delz){
  const int64_t i=is+blockIdx.x*blockDim.x+threadIdx.x,j=js+blockIdx.y*blockDim.y+threadIdx.y;
  if(i>ie||j>je)return;ws.at(i,j)=(zs.at(i,j)-zh.at(i,j,km))*rdt;
  for(int k=km-1;k>=0;--k){const fv_real floor=zh.at(i,j,k+1)+dz_min;if(zh.at(i,j,k)<floor)zh.at(i,j,k)=floor;if(write_delz)delz.at(i,j,k)=zh.at(i,j,k+1)-zh.at(i,j,k);}
}
View3 view3(const DeviceViewBase&v){return{static_cast<fv_real*>(v.data),v.lower[0],v.lower[1],v.lower[2],v.stride[0],v.stride[1],v.stride[2]};}
View2 view2(const DeviceViewBase&v){return{static_cast<fv_real*>(v.data),v.lower[0],v.lower[1],v.stride[0],v.stride[1]};}
CView3 cview3(const DeviceViewBase&v){return{static_cast<const fv_real*>(v.data),v.lower[0],v.lower[1],v.lower[2],v.stride[0],v.stride[1],v.stride[2]};}
CView2 cview2(const DeviceViewBase&v){return{static_cast<const fv_real*>(v.data),v.lower[0],v.lower[1],v.stride[0],v.stride[1]};}
} // namespace

int launch_reconstruct_height(fv_real *zh, const fv_real *delz,
                              const fv_real *zs, ColumnLaunch l, void *stream) {
  if (!zh || !delz || !zs || l.columns <= 0 || l.levels < 2) return 1;
  constexpr int threads = 128;
  reconstruct_height_kernel<<<(l.columns + threads - 1) / threads, threads, 0,
                                static_cast<cudaStream_t>(stream)>>>(zh, delz, zs, l);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

int launch_height_boundary(fv_real *zh, fv_real *delz, fv_real *ws,
                           const fv_real *zs, fv_real rdt, fv_real dz_min,
                           ColumnLaunch l, void *stream) {
  if (!zh || !delz || !ws || !zs || l.columns <= 0 || l.levels < 2) return 1;
  constexpr int threads = 128;
  height_boundary_kernel<<<(l.columns + threads - 1) / threads, threads, 0,
                            static_cast<cudaStream_t>(stream)>>>(
      zh, delz, ws, zs, rdt, dz_min, l);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

int launch_dgrid_height_v2(const ExecutionContext &x,fv3_gpu_error *error) {
  const int context_status=validate_execution_context(x,error);
  if(context_status!=FV3_GPU_SUCCESS)return context_status;
  if(x.submission.stream_id!=StreamId::compute)return FV3_GPU_INVALID_ARGUMENT;
  fv3_gpu_event ready{};if(x.dependencies->require(FV3_GPU_H_DSCALAR,&ready)!=FV3_GPU_SUCCESS)return FV3_GPU_STATE_CONFLICT;
  const auto *z=x.fields->find(FieldId::zh),*d=x.fields->find(FieldId::delz),*w=x.fields->find(FieldId::ws),*s=x.fields->find(FieldId::zs);
  if(!z||!d||!w||!s)return FV3_GPU_INVALID_ARGUMENT;
  const auto &b=x.immutable->bounds;const auto &c=x.immutable->config;
  const auto &p=*x.dispatch_params;
  if(b.npz<2||p.rdt==0)return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  const dim3 t(16,8),g((b.ie-b.is+16)/16,(b.je-b.js+8)/8);
  height_boundary_v2_kernel<<<g,t,0,static_cast<cudaStream_t>(x.submission.stream)>>>(view3(*z),view3(*d),view2(*w),view2(*s),b.is,b.ie,b.js,b.je,(int)b.npz,fv_real(p.rdt),fv_real(c.dz_min),true);
  return cudaGetLastError()==cudaSuccess?FV3_GPU_SUCCESS:FV3_GPU_CUDA_ERROR;
}

/* Milestone 3A.9: update_dz_d_post_transport_kernel was removed here.
 *
 * It was dead code with no caller anywhere in the tree, and it was not a
 * redundant copy of a correct implementation -- it was a WRONG one, which is
 * why leaving it in place was a hazard rather than clutter:
 *
 *   - it applied its own flux-divergence update to zh, as
 *     `zh += rdt*rarea*(fx(i)-fx(i+1)+fy(j)-fy(j+1))`. The authoritative update
 *     is nh_utils.F90:300-301,
 *       zh = (zh*area + (fx(i)-fx(i+1)) + (fy(j)-fy(j+1)))
 *            / ((ra_x + ra_y) - area)
 *     -- an area-weighted RATIO, not an rdt-scaled increment, and it is applied
 *     inside the WP-B transport body (launch_dgrid_height_transport_cuda), so
 *     any caller of this kernel would also have double-counted it;
 *   - it wrote delz, which on the D-grid belongs to Riem_Solver3
 *     (nh_core.F90:181); update_dz_d writes only zh and ws
 *     (nh_utils.F90:310-320).
 *
 * Its ws-then-clamp ordering and its `k = km-1 .. 0` clamp direction WERE
 * correct (nh_utils.F90:313 then :315-319), which is exactly what made the
 * kernel look plausible enough to be dangerous.
 *
 * The live path is launch_update_dz_d -> dispatch.dgrid_height_transport ->
 * height_boundary_v2_kernel(..., write_delz=false), reached through the
 * dgrid_height slot. Numerical semantics changed: NO (nothing called this).
 */

// Complete update_dz_d dispatch
int launch_update_dz_d(const ExecutionContext &x, const NumericalDispatch &dispatch,
                       fv3_gpu_error *error) {
  // Validate context
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  if (x.submission.stream_id != StreamId::compute) return FV3_GPU_INVALID_ARGUMENT;

  // Require H_DSCALAR dependency (scalar halos ready)
  fv3_gpu_event ready{};
  if (x.dependencies->require(FV3_GPU_H_DSCALAR, &ready) != FV3_GPU_SUCCESS)
    return FV3_GPU_STATE_CONFLICT;

  // Get required field views
  const auto *v_zh = x.fields->find(FieldId::zh);
  const auto *v_delz = x.fields->find(FieldId::delz);
  const auto *v_ws = x.fields->find(FieldId::ws);
  const auto *v_zs = x.fields->find(FieldId::zs);
  const auto *v_rarea = x.fields->find(FieldId::rarea);

  if (!v_zh || !v_delz || !v_ws || !v_zs || !v_rarea)
    return FV3_GPU_INVALID_ARGUMENT;

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const auto &p = *x.dispatch_params;

  if (b.npz < 2 || p.rdt == 0.0)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  // Verify hord_tm=6 for GEOS production (or at least supported order)
  // WP-B supports hord = 1, 6, 8 and negative variants
  const int32_t mord = cfg.hord_tm > 0 ? cfg.hord_tm : -cfg.hord_tm;
  if (mord != 1 && mord != 6 && mord != 8)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  // === Step 1: Call WP-B dgrid_height_transport for horizontal advection ===
  // Milestone 3A.4: the WP-B operator now implements the whole authoritative
  // transport body of update_dz_d (nh_utils.F90:255-305):
  // - edge_profile -> the interface crx_adv/xfx_adv/cry_adv/yfx_adv actually
  //   transported with
  // - ra_x on is..ie x jsd..jed, ra_y on isd..ied x js..je
  // - per-level damped/undamped split on damp(k) > 1.E-5
  // - fv_tp_2d with PPM order-6 incl. cubed-sphere panel-edge reconstruction
  //   and copy_corners
  // - del6_vt_flux (NOT tp_core's deln_flux) on the damped levels
  // - the area-weighted height update renormalised by ((ra_x+ra_y)-area)
  if (!dispatch.dgrid_height_transport) {
    // WP-B transport service not wired - integration incomplete
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  }

  int transport_status = dispatch.dgrid_height_transport(x, error);
  if (transport_status != FV3_GPU_SUCCESS) return transport_status;

  // === Step 2: Post-transport boundary/monotonicity enforcement ===
  // nh_utils.F90:307-322 applies exactly:
  // - ws = (zs - zh_bottom) * rdt
  // - Bottom-to-top monotonic: zh[k] >= zh[k+1] + dz_min
  // and nothing else. delz is intent(out) there but never assigned, so
  // write_delz = false: the Riemann solver derives delz from the final zh.
  const dim3 threads(16, 8);
  const dim3 grid((b.ie - b.is + 16) / 16, (b.je - b.js + 8) / 8);
  height_boundary_v2_kernel<<<grid, threads, 0,
                               static_cast<cudaStream_t>(x.submission.stream)>>>(
      view3(*v_zh), view3(*v_delz), view2(*v_ws), view2(*v_zs),
      b.is, b.ie, b.js, b.je, static_cast<int>(b.npz),
      fv_real(p.rdt), fv_real(cfg.dz_min), false);

  return cudaGetLastError() == cudaSuccess ? FV3_GPU_SUCCESS : FV3_GPU_CUDA_ERROR;
}

/* ==========================================================================
 * Milestone 3A.9 §5 and §6 -- the dgrid_height slot, and how
 * dgrid_height_transport becomes reachable.
 *
 * §5 RESOLUTION: dgrid_height IS update_dz_d, and the two slots are in a
 * sequencer/service relationship, not a duplication:
 *
 *   dgrid_height            = nh_utils.F90:230-322 update_dz_d, ENTIRE routine,
 *                             called once per substep from dyn_core.F90:905-906.
 *                             WP-C property (project_architecture.md: WP-C owns
 *                             "update_dz_d orchestration using WP-B horizontal
 *                             transport").
 *   dgrid_height_transport  = nh_utils.F90:255-305, the horizontal-transport
 *                             BODY that routine's k loop runs. WP-B property.
 *                             It has no authoritative call site of its own.
 *
 * So ownership is explicit and nothing is duplicated: this stage contributes
 * only what update_dz_d does around the transport -- the crx/cry edge_profile
 * inputs, the per-level damping split, and the :307-322 boundary/monotonicity
 * closure -- and delegates the transport itself.
 *
 * launch_dgrid_height_v2 (above) is NOT this stage and is NOT production. It is
 * height_boundary_v2_kernel with write_delz=true, i.e. the :307-322 closure plus
 * a delz recompute that authoritatively belongs to Riem_Solver3 (nh_core.F90:181,
 * where delz is derived from the solved zh). It has no caller anywhere in the
 * tree. It is left in place because it is harmless and referenced by comments,
 * but it must never be wired into a dispatch slot: doing so would write delz a
 * substep early, before the Riemann solve that owns it.
 *
 * §6 ROOT CAUSE: launch_update_dz_d needs the dispatch table to reach
 * dispatch.dgrid_height_transport, so its signature is
 * (ExecutionContext, NumericalDispatch, error) -- which is not a LaunchFn and
 * therefore cannot occupy a dispatch slot. dgrid_height_transport was
 * IMPLEMENTED, WIRED and yet UNREACHABLE through fv3_gpu_dyn_core_step for
 * exactly that reason. ExecutionContext::dispatch now carries the table, so this
 * thin adapter is a legitimate LaunchFn and the real production path is
 *
 *   dgrid_dynamics (WP-B d_sw)
 *     -> dgrid_height (WP-C update_dz_d)
 *          -> dgrid_height_transport (WP-B fv_tp_2d/del6_vt_flux body)
 *     -> vertical_solver (WP-C Riem_Solver3)
 *
 * which is dyn_core.F90:829 -> :905 -> :918 in order. The adapter adds no
 * numerical content of its own.
 *
 * Numerical semantics changed: NO.
 * ========================================================================== */
int launch_dgrid_height_stage(const ExecutionContext &x, fv3_gpu_error *error) {
  static constexpr const char *api = "launch_dgrid_height_stage";
  if (!x.dispatch) {
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "ExecutionContext::dispatch is null; update_dz_d cannot reach the "
                "WP-B dgrid_height_transport service it sequences");
  }
  return launch_update_dz_d(x, *x.dispatch, error);
}

/* TEMPORARY diagnostic budget for [RIEM3-BAD], Milestone 3A.11T §7.
 *
 * Once the first bad column appears the failure cascades, so an ungated probe
 * would print every column of every remaining substep -- tens of thousands of
 * lines that bury the only ones that matter. Capping globally keeps exactly the
 * FIRST few offending columns, which is what §7 asks for ("immediately before
 * the first bad invocation"; "do not dump full arrays unless needed").
 *
 * Not reset between steps on purpose: the budget is spent by the earliest
 * failure in the run, whichever step that turns out to be. */
__device__ unsigned fv3_riem3_bad_budget = 0;
static constexpr unsigned kRiem3BadMax = 6;

// Vertical Riemann solver kernel: one thread per horizontal column
__global__ void riem_solver3_kernel(
    CView3 delp, CView3 pt, CView3 q_con, CView3 cappa, CView3 zh_in, CView3 w_in,
    CView2 zs, CView2 ws_in,
    View3 w_out, View3 delz_out, View3 zh_out,
    View3 pe_out, View3 peln_out, View3 pk_out, View3 pk3_out, View3 ppe_out,
    fv_real *workspace, uint64_t workspace_stride,
    int64_t is, int64_t ie, int64_t js, int64_t je, int km,
    int ms, fv_real dt, fv_real rgas, fv_real gama, fv_real kappa,
    fv_real akap_exner,
    fv_real ptop, fv_real p_fac, fv_real a_imp, fv_real scale_m, fv_real grav,
    bool use_cond, bool moist_cappa, bool last_call, bool fp_out, bool use_logp,
    const VerticalWorkspaceLayout layout) {

  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  if (i > ie || j > je) return;

  // Compute linear column index for workspace addressing
  const int64_t col_idx = (i - is) + (j - js) * (ie - is + 1);
  fv_real *col_ws = workspace + col_idx * workspace_stride;

  // Map workspace arrays using layout offsets
  fv_real *dm = col_ws + layout.dm;
  fv_real *pm = col_ws + layout.pm;
  fv_real *pem = col_ws + layout.pem;
  fv_real *peln2 = col_ws + layout.peln2;
  fv_real *pk3_tmp = col_ws + layout.pk3_tmp;
  fv_real *gm = col_ws + layout.gm;
  fv_real *cp = col_ws + layout.cp;
  fv_real *dz = col_ws + layout.dz;
  fv_real *w2 = col_ws + layout.w2;
  fv_real *peg = col_ws + layout.peg;
  fv_real *pelng = col_ws + layout.pelng;
  fv_real *pe2 = col_ws + layout.pe2;
  fv_real *aa = col_ws + layout.aa;
  fv_real *bb = col_ws + layout.bb;
  fv_real *dd = col_ws + layout.dd;
  fv_real *w1 = col_ws + layout.w1;
  fv_real *wk = col_ws + layout.wk;
  fv_real *g_rat = col_ws + layout.g_rat;
  fv_real *gam_arr = col_ws + layout.gam;
  fv_real *pp = col_ws + layout.pp;
  fv_real *bet = col_ws + layout.bet;
  fv_real *p1 = col_ws + layout.p1;
  fv_real *dts = col_ws + layout.dts;
  fv_real *pf1 = col_ws + layout.pf1;
  fv_real *r_lo = col_ws + layout.r_lo;
  fv_real *r_hi = col_ws + layout.r_hi;
  fv_real *m_bot = col_ws + layout.m_bot;
  fv_real *m_top = col_ws + layout.m_top;
  fv_real *r_bot = col_ws + layout.r_bot;
  fv_real *r_top = col_ws + layout.r_top;
  fv_real *wbar = col_ws + layout.wbar;
  fv_real *pbar = col_ws + layout.pbar;
  fv_real *cm = col_ws + layout.cm;
  fv_real *wc = col_ws + layout.wc;
  fv_real *dm_tmp = col_ws + layout.dm_tmp;
  fv_real *wm = col_ws + layout.wm;
  fv_real *dz_tmp = col_ws + layout.dz_tmp;

  // Copy input columns to local workspace
  fv_real delp_col[128], pt_col[128], q_con_col[128], cappa_col[128];
  fv_real zh_col[129], w_col[128];
  for (int k = 0; k < km; ++k) {
    delp_col[k] = delp.at(i, j, k);
    pt_col[k] = pt.at(i, j, k);
    q_con_col[k] = q_con.at(i, j, k);
    cappa_col[k] = cappa.at(i, j, k);
    w_col[k] = w_in.at(i, j, k);
  }
  for (int k = 0; k <= km; ++k) {
    zh_col[k] = zh_in.at(i, j, k);
  }
  const fv_real surface_z = zs.at(i, j);
  const fv_real ws = ws_in.at(i, j);

  // Output arrays
  fv_real w_out_col[128], dz_out_col[128], zh_out_col[129];
  fv_real pe_col[129], peln_col[129], pk_col[129], pk3_col[129], ppe_col[129];

  // Call the column solver
  riem_solver3_column(
      ms, dt, rgas, gama, kappa, akap_exner, ptop, p_fac, a_imp, scale_m, grav,
      use_cond, moist_cappa, last_call, fp_out, use_logp,
      delp_col, pt_col, q_con_col, cappa_col, zh_col, w_col,
      surface_z, ws, w_out_col, dz_out_col, zh_out_col,
      pe_col, peln_col, pk_col, pk3_col, ppe_col,
      dm, pm, pem, peln2, pk3_tmp, gm, cp, dz, w2, peg, pelng,
      pe2, aa, bb, dd, w1, wk, g_rat, gam_arr, pp, bet, p1,
      dts, pf1, r_lo, r_hi, m_bot, m_top, r_bot, r_top, wbar, pbar,
      cm, wc, dm_tmp, wm, dz_tmp, km);

  /* TEMPORARY diagnostic, Milestone 3A.11T §7: isolate Riem_Solver3's inputs
   * immediately before the first bad invocation.
   *
   * Self-targeting on purpose. The failing columns are not known ahead of the
   * run and an (i,j) filter passed from the host would have to be guessed; this
   * fires only on a column whose OWN output went non-finite, so it prints the
   * inputs of exactly the first offending column(s) and nothing else. The
   * inputs are still live in the local copies above, so nothing is re-read from
   * global memory and no other column's arithmetic is disturbed.
   *
   * Also reports the two SIM1 preconditions that a finite-input/NaN-output
   * column implicates: layer thickness must stay strictly negative (dz < 0, an
   * upward-decreasing height), and the reconstructed edge pressure must stay
   * strictly positive and monotone so that log(pe) and pe**kappa are defined.
   * Printing the precondition verdict alongside the raw column is what
   * separates "invalid state handed in" (§8) from "valid state, bad port" (§9).
   *
   * Device printf goes to STDOUT; the host [FV3DBG] probe goes to STDERR. */
  bool bad_out = false;
  for (int k = 0; k < km; ++k)
    if (!isfinite(w_out_col[k]) || !isfinite(dz_out_col[k])) bad_out = true;
  if (bad_out && atomicAdd(&fv3_riem3_bad_budget, 1u) < kRiem3BadMax) {
    int n_dz_nonneg = 0, n_pe_nonpos = 0, n_pe_nonmono = 0;
    fv_real dz_worst = dz_out_col[0];
    for (int k = 0; k < km; ++k) {
      const fv_real d = zh_col[k + 1] - zh_col[k];
      if (!(d < (fv_real)0)) ++n_dz_nonneg;
      if (d > dz_worst) dz_worst = d;
    }
    for (int k = 0; k <= km; ++k) {
      if (!(pe_col[k] > (fv_real)0)) ++n_pe_nonpos;
      if (k > 0 && !(pe_col[k] > pe_col[k - 1])) ++n_pe_nonmono;
    }
    printf("[RIEM3-BAD i=%lld j=%lld] dz_nonneg=%d dz_worst=%.6e "
           "pe_nonpos=%d pe_nonmono=%d ws=%.6e zs=%.6e p_fac=%.3e a_imp=%.3e\n",
           (long long)i, (long long)j, n_dz_nonneg, (double)dz_worst,
           n_pe_nonpos, n_pe_nonmono, (double)ws, (double)surface_z,
           (double)p_fac, (double)a_imp);
    for (int k = 0; k < km; ++k)
      printf("[RIEM3-BAD i=%lld j=%lld] k=%d delp=%.9e pt=%.9e w=%.9e "
             "zh=%.9e zh1=%.9e cappa=%.9e | w_out=%.9e dz_out=%.9e\n",
             (long long)i, (long long)j, k, (double)delp_col[k],
             (double)pt_col[k], (double)w_col[k], (double)zh_col[k],
             (double)zh_col[k + 1], (double)cappa_col[k],
             (double)w_out_col[k], (double)dz_out_col[k]);
  }

  // Write outputs back to global memory
  for (int k = 0; k < km; ++k) {
    w_out.at(i, j, k) = w_out_col[k];
    delz_out.at(i, j, k) = dz_out_col[k];
  }
  for (int k = 0; k <= km; ++k) {
    zh_out.at(i, j, k) = zh_out_col[k];
    ppe_out.at(i, j, k) = ppe_col[k];
    pk3_out.at(i, j, k) = pk3_col[k];
    if (last_call) {
      /* Milestone 3A.11T. pe and peln are the FV3 exception to the (i,j,k)
       * convention: every one of their ~40 declaration sites in the
       * authoritative tree orders them (i, k, j), with the vertical axis in
       * the MIDDLE slot. Riem_Solver3's own dummies say so --
       *   nh_core.F90:87  pe(is-1:ie+1,km+1,js-1:je+1)
       *   nh_core.F90:88  peln(is:ie,km+1,js:je)
       * -- as do dyn_core.F90:202-203, fv_dynamics.F90:215/217, geopk's
       * :2123-2124, and pe_halo's :1396. Body-level proof: geopk writes
       * `pe(i,1,j) = ptop` (:2191) and fv_dynamics reads the surface as
       * `pe(i,npz+1,j)` (:611).
       *
       * View3::at() is purely positional -- slot 0/1/2 -- so the k index has
       * to be handed to the MIDDLE argument. It previously went to the last
       * one, transposing k against j on every write.
       *
       * pk is deliberately NOT changed: nh_core.F90:91 declares
       * pk(is:ie,js:je,km+1), genuinely (i,j,k). Only pe/peln are special. */
      pe_out.at(i, k, j) = pe_col[k];
      peln_out.at(i, k, j) = peln_col[k];
      pk_out.at(i, j, k) = pk_col[k];
    }
  }
}

// Vertical solver dispatch
int launch_vertical_solver(const ExecutionContext &x, fv3_gpu_error *error) {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  if (x.submission.stream_id != StreamId::compute) return FV3_GPU_INVALID_ARGUMENT;

  // Get required fields
  const auto *v_delp = x.fields->find(FieldId::delp);
  const auto *v_pt = x.fields->find(FieldId::pt);
  const auto *v_q_con = x.fields->find(FieldId::q_con);
  const auto *v_cappa = x.fields->find(FieldId::cappa);
  const auto *v_zh = x.fields->find(FieldId::zh);
  const auto *v_w = x.fields->find(FieldId::w);
  const auto *v_zs = x.fields->find(FieldId::zs);
  const auto *v_ws = x.fields->find(FieldId::ws);
  const auto *v_delz = x.fields->find(FieldId::delz);
  const auto *v_pe = x.fields->find(FieldId::pe);
  const auto *v_peln = x.fields->find(FieldId::peln);
  const auto *v_pk = x.fields->find(FieldId::pk);
  const auto *v_pk3 = x.fields->find(FieldId::pk3);
  const auto *v_ppe = x.fields->find(FieldId::pkc); // ppe stored in pkc slot

  if (!v_delp || !v_pt || !v_q_con || !v_cappa || !v_zh || !v_w ||
      !v_zs || !v_ws || !v_delz || !v_pe || !v_peln || !v_pk || !v_pk3 || !v_ppe)
    return FV3_GPU_INVALID_ARGUMENT;

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const auto &p = *x.dispatch_params;

  if (b.npz < 2) return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  /* gama = 1/(1-akap) below; akap == 1 is a pole, and m_split is the vertical
   * substep count Riem_Solver3 receives at dyn_core.F90:918. Neither may be
   * defaulted. */
  if (cfg.akap <= 0.0 || cfg.akap >= 1.0 || cfg.m_split < 1)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  /* grav reaches riem3_precompute_column as rgrav = 1/grav (nh_core.F90:100),
   * so a defaulted 0 would divide by zero and silently poison every column.
   * Mirrors the guard at fv3_riem_solver_c.cu:272 for the C-grid twin, which
   * already validated its own grav. Milestone 3A.11V. */
  if (!(cfg.grav > 0.0)) return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  // Get workspace slice
  const auto *ws_slice = workspace_slice(*x.workspace, WorkspaceId::vertical);
  if (!ws_slice || !ws_slice->data) return FV3_GPU_INVALID_ARGUMENT;

  const int km = static_cast<int>(b.npz);
  const auto layout = compute_workspace_layout(km);
  const uint64_t bytes_per_column = vertical_workspace_per_column<fv_real>(km);
  const uint64_t columns = static_cast<uint64_t>((b.ie - b.is + 1) * (b.je - b.js + 1));

  // Verify workspace capacity
  if (ws_slice->bytes < columns * bytes_per_column)
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;

  const uint64_t workspace_stride = layout.total_scalars;

  const dim3 threads(16, 8);
  const dim3 grid((b.ie - b.is + 16) / 16, (b.je - b.js + 8) / 8);

  riem_solver3_kernel<<<grid, threads, 0,
                         static_cast<cudaStream_t>(x.submission.stream)>>>(
      cview3(*v_delp), cview3(*v_pt), cview3(*v_q_con), cview3(*v_cappa),
      cview3(*v_zh), cview3(*v_w), cview2(*v_zs), cview2(*v_ws),
      view3(*v_w), view3(*v_delz), view3(*v_zh),
      view3(*v_pe), view3(*v_peln), view3(*v_pk), view3(*v_pk3), view3(*v_ppe),
      static_cast<fv_real *>(ws_slice->data), workspace_stride,
      b.is, b.ie, b.js, b.je, km,
      /* gama. nh_core.F90:99 `gama = 1./(1.-akap)` -- DERIVED from akap, not a
       * constant. This argument was hardcoded to 1.4, which is only the value
       * gama happens to take near akap = 2/7 (1/(1-0.2857) = 1.4000); with the
       * GEOS akap of 0.286 the authoritative value is 1.40056, so every SIM1
       * column solve was running a slightly wrong polytropic exponent.
       * Milestone 3A.9: derived. Numerical semantics changed: YES -- this is a
       * repair toward the authoritative Fortran, not a change away from it. */
      p.vertical_ms, fv_real(p.dt), fv_real(cfg.rdgas),
      fv_real(1.0 / (1.0 - cfg.akap)),
      fv_real(cfg.akap),
      /* akap_exner: the Exner-only scalar for ptk/pk3. cfg.akap remains the
       * closure kappa (gama and capa1). They are the same number unless the
       * caller supplies akap_exner, in which case the two Riem_Solver3 roles
       * separate exactly as the authoritative MOIST_CAPPA build does; see
       * NumericalConfig::akap_exner. Prompt 6B. */
      fv_real(cfg.akap_exner > 0.0 ? cfg.akap_exner : cfg.akap),
      fv_real(cfg.ptop), fv_real(cfg.p_fac),
      fv_real(cfg.a_imp), fv_real(cfg.scale_m),
      /* grav. Was hardcoded T(9.8) inside riem_solver3_column, which had no grav
       * parameter to pass it through; see fv3_vertical_column.hpp for the full
       * derivation and the measured 10-digit reproduction. Milestone 3A.11V. */
      fv_real(cfg.grav),
      cfg.use_cond != 0, cfg.moist_cappa != 0,
      p.last_call != 0, p.fp_out != 0, cfg.use_logp != 0,
      layout);

  return cudaGetLastError() == cudaSuccess ? FV3_GPU_SUCCESS : FV3_GPU_CUDA_ERROR;
}

/* ==========================================================================
 * Milestone 3A.9 §4 -- the C-grid vertical entry.
 *
 * VERDICT ON THE PRIOR IMPLEMENTATION: it was a semantic alias, and wiring it
 * would have been numerically wrong. The body read
 *
 *     // For now, delegate to the same Riemann solver.
 *     return launch_vertical_solver(x, error);
 *
 * i.e. it ran Riem_Solver3. dyn_core.F90:583 calls Riem_Solver_C, which is a
 * DIFFERENT authoritative routine (nh_utils.F90) -- different argument list,
 * different a_imp branch set (SIM3p0 / RIM_2D / SIM1 rather than Riem_Solver3's
 * families), no pe/peln/pk/ppe/pk3 outputs, no last_call/fp_out/use_logp
 * behaviour, and it consumes dt2 rather than dt. It is not an alias and must not
 * be wired as one.
 *
 * So this slot IS genuinely distinct, and it is a COMPOSITE, not a single
 * routine. dyn_core.F90's nonhydrostatic C-grid vertical group is:
 *
 *   :445-455   it==1 only: gz(:,:,npz+1) = zs; gz(k) = gz(k+1) - delz(k)
 *   :459       it==1 only: start_group_halo_update(i_pack(5), gz)
 *   :552       it==1 only: complete_group_halo_update(i_pack(5))
 *   :557-563   it==1:  zh = gz   over isd:ied, jsd:jed, k=1..npz+1
 *   :566-573   it/=1:  gz = zh   over isd:ied, jsd:jed, k=1..npz+1
 *   :577       update_dz_c(..., dt2, ...)
 *   :583       Riem_Solver_C(ms, dt2, ...)
 *
 * CUDA STRUCTURAL DIFFERENCE (§18 class "MATCHES WITH EXPLAINED CUDA STRUCTURAL
 * DIFFERENCE"): in Fortran the :445-459 block sits BEFORE c_sw (:515) so that
 * the gz halo exchange overlaps c_sw. It is hoisted into this stage here.
 * That is safe and order-preserving because c_sw writes only delpc, ptc, uc, vc,
 * ut, vt, divgd and omga, and the :445-455 reconstruction reads only delz and
 * zs -- there is no read-write or write-write overlap in either direction. The
 * only thing lost is communication/computation overlap, which is explicitly
 * deferred (project_architecture.md, "Performance Work Comes Later").
 *
 * NOT part of this slot: the hydrostatic branch (:544 geopk) and the nested
 * branches (:432-443, :586-606 nest_halo_nh -- the latter is dead code in the
 * supported configuration anyway). Both are rejected below rather than
 * silently approximated.
 * ========================================================================== */

// dyn_core.F90:445-455, non-nested branch. Reuses reconstruct_height_column
// (fv3_vertical_column.hpp:109) rather than restating the recurrence.
__global__ void gz_from_delz_kernel(View3 gz, CView3 delz, CView2 zs,
                                    int64_t is, int64_t ie, int64_t js,
                                    int64_t je, int km) {
  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  if (i > ie || j > je) return;
  // CView3::at returns by value, so address the column directly. Same offset
  // arithmetic as CView3::at with k == 0.
  Column<fv_real> z{&gz.at(i, j, 0), gz.s2};
  const Column<const fv_real> dz{
      delz.p + (i - delz.l0) * delz.s0 + (j - delz.l1) * delz.s1 +
          (0 - delz.l2) * delz.s2,
      delz.s2};
  reconstruct_height_column<fv_real>(z, dz, zs.at(i, j), km);
}

// dyn_core.F90:557-563 (zh_from_gz) and :566-573 (gz from zh). Both loop the
// FULL halo box isd:ied, jsd:jed over k=1..npz+1, so kmp1 == npz+1 levels.
__global__ void gz_zh_sync_kernel(View3 gz, View3 zh, int64_t isd, int64_t ied,
                                  int64_t jsd, int64_t jed, int kmp1,
                                  bool zh_from_gz) {
  const int64_t i = isd + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = jsd + blockIdx.y * blockDim.y + threadIdx.y;
  if (i > ied || j > jed) return;
  for (int k = 0; k < kmp1; ++k) {
    if (zh_from_gz)
      zh.at(i, j, k) = gz.at(i, j, k);
    else
      gz.at(i, j, k) = zh.at(i, j, k);
  }
}

/* Shared configuration fence for both halves of the C-grid vertical group
 * (Milestone 3A.11S). Factored out of launch_cgrid_vertical_solver verbatim so
 * the pre-halo half rejects exactly the same unsupported configurations, and so
 * every rejection still precedes the first launch of either half.
 *
 * Numerical algorithm changed: NO. Identical predicates, identical messages. */
static int cgrid_vertical_config_fence(const ExecutionContext &x,
                                       fv3_gpu_error *error, const char *api) {
  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const auto &p = *x.dispatch_params;
  const auto &topo = *x.immutable->topology;

  if (cfg.hydrostatic)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "hydrostatic C-grid pressure (dyn_core.F90:544 geopk) is not "
                "implemented; the supported configuration is hydrostatic=false");
  if (!topo.global_cubed_sphere || topo.nested || topo.regional)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "only the global non-nested cubed sphere is implemented");
  if (cfg.grid_type != 0)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "only grid_type == 0 is implemented");
  if (b.halo_width < 3u)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "update_dz_c requires halo_width >= 3");
  if (b.npz < 3)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "update_dz_c's top/bottom extrapolation requires npz >= 3");
  if (topo.npx <= 0 || topo.npy <= 0)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "fill_4corners needs the tile corner counts npx/npy; a zero "
                "value would silently skip the panel-corner fills");
  if (p.dt2 == 0.0)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "dt2 is zero; update_dz_c divides by it");
  return FV3_GPU_SUCCESS;
}

/* dyn_core.F90:433-455 -> :459. See fv3_vertical_kernels.cuh for why this is a
 * separate stage rather than the opening phase of launch_cgrid_vertical_solver.
 *
 * Writes gz's INTERIOR only (is:ie, js:je), exactly as the non-nested Fortran
 * branch at :445-455 does. The halo cells are supplied by the H_GZ_FIRST
 * exchange the runtime resolves immediately after this stage returns. */
int launch_cgrid_vertical_pre_halo(const ExecutionContext &x, fv3_gpu_error *error) {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  const char *api = "launch_cgrid_vertical_pre_halo";
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "the C-grid vertical group must be enqueued on the compute stream");

  const int fence = cgrid_vertical_config_fence(x, error, api);
  if (fence != FV3_GPU_SUCCESS) return fence;

  const auto &b = x.immutable->bounds;
  const auto &p = *x.dispatch_params;

  // it > 1 rebuilds gz from zh inside the post-halo half, not from delz, and
  // dyn_core.F90:459 correspondingly starts no exchange. Nothing to do.
  if (p.acoustic_index != 0) return FV3_GPU_SUCCESS;

  const auto *v_gz   = x.fields->find(FieldId::gz);
  const auto *v_delz = x.fields->find(FieldId::delz);
  const auto *v_zs   = x.fields->find(FieldId::zs);
  if (!v_gz || !v_delz || !v_zs)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "the C-grid vertical pre-halo stage requires gz, delz and zs");

  auto *stream = static_cast<cudaStream_t>(x.submission.stream);
  const dim3 block(16, 8);
  const dim3 g(static_cast<unsigned>((b.ie - b.is + 1 + block.x - 1) / block.x),
               static_cast<unsigned>((b.je - b.js + 1 + block.y - 1) / block.y));
  gz_from_delz_kernel<<<g, block, 0, stream>>>(
      view3(*v_gz), cview3(*v_delz), cview2(*v_zs),
      b.is, b.ie, b.js, b.je, static_cast<int>(b.npz));
  if (cudaPeekAtLastError() != cudaSuccess)
    return fail(error, FV3_GPU_CUDA_ERROR, api, "gz_from_delz launch failed");
  return FV3_GPU_SUCCESS;
}

int launch_cgrid_vertical_solver(const ExecutionContext &x, fv3_gpu_error *error) {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  const char *api = "launch_cgrid_vertical_solver";
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "the C-grid vertical group must be enqueued on the compute stream");

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const auto &p = *x.dispatch_params;
  const auto &topo = *x.immutable->topology;

  // ---- configuration fence: every rejection precedes the first launch ----
  if (cfg.hydrostatic)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "hydrostatic C-grid pressure (dyn_core.F90:544 geopk) is not "
                "implemented; the supported configuration is hydrostatic=false");
  if (!topo.global_cubed_sphere || topo.nested || topo.regional)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "only the global non-nested cubed sphere is implemented");
  if (cfg.grid_type != 0)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "only grid_type == 0 is implemented");
  // update_dz_c's gz2 corner-fill band spans is-3..ie+3 (nh_utils.F90:156-160
  // copies is-ng..ie+ng and the fill_4corners writes reach two cells out), and
  // cgrid/fv3_cgrid_dz.cu sizes that band with a literal 3.
  if (b.halo_width < 3u)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "update_dz_c requires halo_width >= 3");
  // nh_utils.F90:96-97 form top_ratio from dp0(1),dp0(2) and bot_ratio from
  // dp0(km-1),dp0(km), so km >= 3 is required for those to be distinct levels.
  if (b.npz < 3)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "update_dz_c's top/bottom extrapolation requires npz >= 3");
  if (topo.npx <= 0 || topo.npy <= 0)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "fill_4corners needs the tile corner counts npx/npy; a zero "
                "value would silently skip the panel-corner fills");
  // dyn_core.F90:577 and :583 both receive dt2, and nh_utils.F90:94 forms
  // rdt = 1/dt from it.
  if (p.dt2 == 0.0)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "dt2 is zero; update_dz_c divides by it");

  const auto *v_gz    = x.fields->find(FieldId::gz);
  const auto *v_zh    = x.fields->find(FieldId::zh);
  const auto *v_delz  = x.fields->find(FieldId::delz);
  const auto *v_zs    = x.fields->find(FieldId::zs);
  const auto *v_ut    = x.fields->find(FieldId::ut);
  const auto *v_vt    = x.fields->find(FieldId::vt);
  const auto *v_dp0   = x.fields->find(FieldId::dp_ref);
  const auto *v_area  = x.fields->find(FieldId::area);
  const auto *v_ws3   = x.fields->find(FieldId::ws3);
  if (!v_gz || !v_zh || !v_delz || !v_zs || !v_ut || !v_vt || !v_dp0 ||
      !v_area || !v_ws3)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "the C-grid vertical group requires gz, zh, delz, zs, ut, vt, "
                "dp_ref, area and ws3");

  auto *stream = static_cast<cudaStream_t>(x.submission.stream);
  const int km = static_cast<int>(b.npz);
  const int64_t halo = static_cast<int64_t>(b.halo_width);
  const int64_t isd = b.is - halo, ied = b.ie + halo;
  const int64_t jsd = b.js - halo, jed = b.je + halo;
  const dim3 block(16, 8);
  auto grid_for = [&](int64_t nx, int64_t ny) {
    return dim3(static_cast<unsigned>((nx + block.x - 1) / block.x),
                static_cast<unsigned>((ny + block.y - 1) / block.y));
  };

  /* ---- Phase 1: gz / zh synchronisation (dyn_core.F90:552-573) ----
   *
   * Milestone 3A.11S: the gz reconstruction from delz (:436-455) that used to
   * open this phase now lives in launch_cgrid_vertical_pre_halo, which the
   * runtime runs BEFORE resolving H_GZ_FIRST. That is required for correctness,
   * not style: :459 starts the exchange only after the interior is rebuilt, so
   * packing before the reconstruction exchanges the previous step's gz. */
  if (p.acoustic_index == 0) {
    /* dyn_core.F90:459 starts and :552 completes the gz halo exchange, and only
     * AFTER it completes does :557-563 propagate gz into zh across the full
     * halo box. The pre-halo stage wrote only is:ie / js:je, so the halo cells
     * zh inherits here come entirely from that exchange. Requiring the group at
     * this exact point is what prevents the copy below from broadcasting
     * uninitialised or stale halo values into zh, where they would persist for
     * every subsequent substep.
     *
     * This require is deliberately NOT satisfied by this stage. Nothing here may
     * mark a halo ready; that is the runtime's single-rank halo-readiness
     * responsibility (§8). It is also the guard that makes a runtime which
     * forgot to run the pre-halo stage fail loudly instead of silently
     * exchanging stale gz. */
    fv3_gpu_event gz_ready{};
    if (x.dependencies->require(FV3_GPU_H_GZ_FIRST, &gz_ready) != FV3_GPU_SUCCESS)
      return fail(error, FV3_GPU_STATE_CONFLICT, api,
                  "H_GZ_FIRST (dyn_core.F90:459 start / :552 complete, gz) must "
                  "be ready before zh inherits gz across the halo");
  }

  {
    // it==1: zh <- gz (:557-563). it>1: gz <- zh (:566-573).
    const bool zh_from_gz = (p.acoustic_index == 0);
    const dim3 g = grid_for(ied - isd + 1, jed - jsd + 1);
    gz_zh_sync_kernel<<<g, block, 0, stream>>>(view3(*v_gz), view3(*v_zh),
                                               isd, ied, jsd, jed, km + 1,
                                               zh_from_gz);
    if (cudaPeekAtLastError() != cudaSuccess)
      return fail(error, FV3_GPU_CUDA_ERROR, api, "gz/zh sync launch failed");
    /* :557-563 rewrites zh over the whole halo box from an exchanged gz, so zh's
     * own halo is consistent afterwards and H_ZH is NOT invalidated here. The
     * it>1 direction does not write zh at all. Either way zh's halo state is
     * unchanged by this phase, which is why there is no mark_stale for it. */
  }

  /* ---- Phase 2: update_dz_c (dyn_core.F90:577, nh_utils.F90:70-207) ---- */
  {
    /* Scratch: nh_utils.F90:84-86 declares gz2(is-ng:ie+ng,js-ng:je+ng),
     * xfx/fx(is-1:ie+2,js-1:je+1) and yfx/fy(is-1:ie+1,js-1:je+2) as per-k
     * OpenMP-private locals. The CUDA per-level launchers index them by k, so
     * each of the four flux arrays is carved as one whole
     * (ni x nj x (npz+1)) fv_real block and gz2 uses level 0 of a fifth block.
     * That is the same arena convention as WpBDSwArena (dgrid/wp_b_dsw_cuda.cuh)
     * and as the runtime's own workspace sizing.
     *
     * WorkspaceId::cgrid is safe to share with the C-grid predictor: this stage
     * runs strictly after it within the substep, and every value the predictor
     * hands over (ut, vt, uc, vc, delpc, ptc, divgd, omga) is a registered
     * field, not scratch. */
    constexpr uint64_t kUpdateDzCArrays = 5;
    const auto *slice = workspace_slice(*x.workspace, WorkspaceId::cgrid);
    if (!slice || !slice->data)
      return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                  "WorkspaceId::cgrid slice is unavailable");
    const uint64_t ni = static_cast<uint64_t>(b.ie - b.is + 1) + 2 * b.halo_width;
    const uint64_t nj = static_cast<uint64_t>(b.je - b.js + 1) + 2 * b.halo_width;
    const uint64_t block_values = ni * nj * static_cast<uint64_t>(km + 1);
    if (slice->bytes < kUpdateDzCArrays * block_values * sizeof(fv_real))
      return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                  "WorkspaceId::cgrid is too small for update_dz_c's five "
                  "(ni x nj x npz+1) scratch arrays");
    auto *arena = static_cast<fv_real *>(slice->data);
    auto carve3 = [&](uint64_t n) {
      cgrid::FieldView<fv_real, 3> v{};
      v.data = arena + n * block_values;
      v.lower[0] = isd; v.lower[1] = jsd; v.lower[2] = 0;
      v.stride[0] = 1;
      v.stride[1] = static_cast<int64_t>(ni);
      v.stride[2] = static_cast<int64_t>(ni * nj);
      return v;
    };

    cgrid::UpdateDzCArgs a{};
    a.bounds = {b.is, b.ie, b.js, b.je, b.npz};
    a.npx = static_cast<int32_t>(topo.npx);
    a.npy = static_cast<int32_t>(topo.npy);
    a.km = km;
    a.dt = static_cast<fv_real>(p.dt2);   // dyn_core.F90:577 passes dt2
    a.dz_min = static_cast<fv_real>(cfg.dz_min);
    a.grid_type = cfg.grid_type;
    a.sw_corner = topo.sw_corner; a.se_corner = topo.se_corner;
    a.ne_corner = topo.ne_corner; a.nw_corner = topo.nw_corner;
    a.ut = cgrid::make_view<const fv_real, 3>(*v_ut);
    a.vt = cgrid::make_view<const fv_real, 3>(*v_vt);
    a.dp0 = cgrid::make_view<const fv_real, 1>(*v_dp0);
    a.gz = cgrid::make_view<fv_real, 3>(*v_gz);
    a.zs = cgrid::make_view<const fv_real, 2>(*v_zs);
    a.area = cgrid::make_view<const fv_real, 2>(*v_area);
    a.ws = cgrid::make_view<fv_real, 2>(*v_ws3);   // dyn_core.F90:577 passes ws3
    a.gz2 = cgrid::level_slice<fv_real>(carve3(0), 0);
    a.xfx = carve3(1);
    a.yfx = carve3(2);
    a.fx  = carve3(3);
    a.fy  = carve3(4);

    cgrid::CornerFillArgs corner{};
    corner.npx = a.npx; corner.npy = a.npy;
    corner.sw_corner = a.sw_corner; corner.se_corner = a.se_corner;
    corner.ne_corner = a.ne_corner; corner.nw_corner = a.nw_corner;

    /* nh_utils.F90:113 `do 6000 k=1,km+1`. The loop is serialised on one stream
     * because gz2 is a single reused level buffer, exactly as it is a per-k
     * OpenMP-private local in Fortran. launch_update_dz_c_interface issues the
     * whole level body including the two fill_4corners calls at :162 and :174,
     * which sit between the gz2 copy and each flux phase and therefore cannot be
     * hoisted out of the loop. */
    for (int64_t k = 1; k <= km + 1; ++k) {
      const fv3_gpu_status s =
          cgrid::launch_update_dz_c_interface(a, corner, k, stream);
      if (s != FV3_GPU_SUCCESS)
        return fail(error, s, api, "update_dz_c interface level launch failed");
    }

    // nh_utils.F90:194-205, after the level loop closes.
    fv3_gpu_status s = cgrid::launch_update_dz_c_finalize(a, stream);
    if (s != FV3_GPU_SUCCESS)
      return fail(error, s, api, "update_dz_c finalize launch failed");
  }

  /* ---- Phase 3: Riem_Solver_C (dyn_core.F90:583) ---- */
  {
    const int rc = launch_riem_solver_c_ctx(x, error);
    if (rc != FV3_GPU_SUCCESS) return rc;
  }

  /* gz and pkc are now updated over is-1..ie+1 / js-1..je+1 only, which is
   * exactly what p_grad_c (dyn_core.F90:614) reads -- Fortran performs no halo
   * exchange between :583 and :614 either. Their halo rings are therefore stale
   * with respect to the newly written interior, and any later consumer that
   * needs a wider stencil must exchange first.
   *
   * H_ZH is NOT marked stale: this group does not write zh after phase 1, and
   * dyn_core exchanges zh separately at :931 (i_pack(4)) once d_sw has run. */
  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu::vertical
