#include "fv3_vertical_kernels.cuh"
#include "fv3_vertical_column.hpp"
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

__global__ void height_boundary_v2_kernel(View3 zh,View3 delz,View2 ws,View2 zs,
 int64_t is,int64_t ie,int64_t js,int64_t je,int km,fv_real rdt,fv_real dz_min){
  const int64_t i=is+blockIdx.x*blockDim.x+threadIdx.x,j=js+blockIdx.y*blockDim.y+threadIdx.y;
  if(i>ie||j>je)return;ws.at(i,j)=(zs.at(i,j)-zh.at(i,j,km))*rdt;
  for(int k=km-1;k>=0;--k){const fv_real floor=zh.at(i,j,k+1)+dz_min;if(zh.at(i,j,k)<floor)zh.at(i,j,k)=floor;delz.at(i,j,k)=zh.at(i,j,k+1)-zh.at(i,j,k);}
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
  height_boundary_v2_kernel<<<g,t,0,static_cast<cudaStream_t>(x.submission.stream)>>>(view3(*z),view3(*d),view2(*w),view2(*s),b.is,b.ie,b.js,b.je,(int)b.npz,fv_real(p.rdt),fv_real(c.dz_min));
  return cudaGetLastError()==cudaSuccess?FV3_GPU_SUCCESS:FV3_GPU_CUDA_ERROR;
}

// Complete update_dz_d sequence kernel: post-transport boundary/monotonicity
__global__ void update_dz_d_post_transport_kernel(
    View3 zh, View3 delz, View2 ws, CView2 zs, CView3 flux_x, CView3 flux_y,
    CView2 rarea, int64_t is, int64_t ie, int64_t js, int64_t je, int km,
    fv_real rdt, fv_real dz_min) {
  const int64_t i = is + blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t j = js + blockIdx.y * blockDim.y + threadIdx.y;
  if (i > ie || j > je) return;

  // Apply horizontal flux divergence to zh (height transport result)
  // This follows tp_core.F90 update_dz_d pattern:
  // zh = zh + rdt * rarea * (flux_x(i) - flux_x(i+1) + flux_y(j) - flux_y(j+1))
  for (int k = 0; k < km; ++k) {
    const fv_real div = rarea.at(i, j) *
        (flux_x.at(i, j, k) - flux_x.at(i + 1, j, k) +
         flux_y.at(i, j, k) - flux_y.at(i, j + 1, k));
    zh.at(i, j, k) = zh.at(i, j, k) + rdt * div;
  }

  // Surface vertical velocity ws = (zs - zh_bottom) * rdt
  ws.at(i, j) = (zs.at(i, j) - zh.at(i, j, km)) * rdt;

  // Bottom-to-top monotonic height enforcement: zh[k] >= zh[k+1] + dz_min
  for (int k = km - 1; k >= 0; --k) {
    const fv_real floor = zh.at(i, j, k + 1) + dz_min;
    if (zh.at(i, j, k) < floor) zh.at(i, j, k) = floor;
    delz.at(i, j, k) = zh.at(i, j, k + 1) - zh.at(i, j, k);
  }
}

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
  // This executes fv_tp_2d with hord_tm on the height field zh.
  // WP-B transport_service_for_update_dz handles:
  // - PPM order-6 reconstruction
  // - Y-X / X-Y directional sequencing
  // - ra_x/ra_y area corrections
  // - deln_flux damping when vtdm4 > 0
  if (!dispatch.dgrid_height_transport) {
    // WP-B transport service not wired - integration incomplete
    return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  }

  int transport_status = dispatch.dgrid_height_transport(x, error);
  if (transport_status != FV3_GPU_SUCCESS) return transport_status;

  // === Step 2: Post-transport boundary/monotonicity enforcement ===
  // The height boundary kernel applies:
  // - ws = (zs - zh_bottom) * rdt
  // - Bottom-to-top monotonic: zh[k] >= zh[k+1] + dz_min
  // - delz[k] = zh[k+1] - zh[k]
  const dim3 threads(16, 8);
  const dim3 grid((b.ie - b.is + 16) / 16, (b.je - b.js + 8) / 8);
  height_boundary_v2_kernel<<<grid, threads, 0,
                               static_cast<cudaStream_t>(x.submission.stream)>>>(
      view3(*v_zh), view3(*v_delz), view2(*v_ws), view2(*v_zs),
      b.is, b.ie, b.js, b.je, static_cast<int>(b.npz),
      fv_real(p.rdt), fv_real(cfg.dz_min));

  return cudaGetLastError() == cudaSuccess ? FV3_GPU_SUCCESS : FV3_GPU_CUDA_ERROR;
}

// Vertical Riemann solver kernel: one thread per horizontal column
__global__ void riem_solver3_kernel(
    CView3 delp, CView3 pt, CView3 q_con, CView3 cappa, CView3 zh_in, CView3 w_in,
    CView2 zs, CView2 ws_in,
    View3 w_out, View3 delz_out, View3 zh_out,
    View3 pe_out, View3 peln_out, View3 pk_out, View3 pk3_out, View3 ppe_out,
    fv_real *workspace, uint64_t workspace_stride,
    int64_t is, int64_t ie, int64_t js, int64_t je, int km,
    int ms, fv_real dt, fv_real rgas, fv_real gama, fv_real kappa,
    fv_real ptop, fv_real p_fac, fv_real a_imp, fv_real scale_m,
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
      ms, dt, rgas, gama, kappa, ptop, p_fac, a_imp, scale_m,
      use_cond, moist_cappa, last_call, fp_out, use_logp,
      delp_col, pt_col, q_con_col, cappa_col, zh_col, w_col,
      surface_z, ws, w_out_col, dz_out_col, zh_out_col,
      pe_col, peln_col, pk_col, pk3_col, ppe_col,
      dm, pm, pem, peln2, pk3_tmp, gm, cp, dz, w2, peg, pelng,
      pe2, aa, bb, dd, w1, wk, g_rat, gam_arr, pp, bet, p1,
      dts, pf1, r_lo, r_hi, m_bot, m_top, r_bot, r_top, wbar, pbar,
      cm, wc, dm_tmp, wm, dz_tmp, km);

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
      pe_out.at(i, j, k) = pe_col[k];
      peln_out.at(i, j, k) = peln_col[k];
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

  // Get workspace slice
  const auto *ws_slice = workspace_slice(*x.workspace, WorkspaceId::vertical);
  if (!ws_slice || !ws_slice->data) return FV3_GPU_INVALID_ARGUMENT;

  const int km = static_cast<int>(b.npz);
  const auto layout = compute_workspace_layout(km);
  const uint64_t bytes_per_column = vertical_workspace_per_column(km);
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
      p.vertical_ms, fv_real(p.dt), fv_real(cfg.rdgas), fv_real(1.4),
      fv_real(cfg.akap), fv_real(cfg.ptop), fv_real(cfg.p_fac),
      fv_real(cfg.a_imp), fv_real(cfg.scale_m),
      cfg.use_cond != 0, cfg.moist_cappa != 0,
      p.last_call != 0, p.fp_out != 0, cfg.use_logp != 0,
      layout);

  return cudaGetLastError() == cudaSuccess ? FV3_GPU_SUCCESS : FV3_GPU_CUDA_ERROR;
}

// C-grid vertical solver entry
int launch_cgrid_vertical_solver(const ExecutionContext &x, fv3_gpu_error *error) {
  // C-grid vertical solver shares the same column primitives but has
  // different orchestration scope (WP-A owned). This entry provides
  // the internal dispatch point.
  // For now, delegate to the same Riemann solver.
  return launch_vertical_solver(x, error);
}

} // namespace fv3::gpu::vertical
