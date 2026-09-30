//===========================================================================
// WP-C: production State_To_FV CUDA kernels + launchers (Prompt 99).
// One thread per horizontal column, plain interior domain is..ie x js..je
// (matches FV_StateMod.F90's own isc:iec/jsc:jec slices -- STATE%VARS%*
// carries no halo at all).
//
// Authoritative Fortran (numerical source of truth, read-only reference):
//   GEOSfvdycore_fortran_perf checkout, src/Components/
//   @FVdycoreCubed_GridComp/FV_StateMod.F90:2519-2702, subroutine
//   State_To_FV, confirmed against the live source (this prompt) --
//   restated precisely here because Phase C's pe/peln/pk loops turned out
//   to run over the FULL k=1,km+1 range for every interior column (no
//   boundary-pass-through special case, unlike REMAP_PRESSUREVARS's Pass
//   1) and neither w nor delz nor pkz gets a tiny_number fill statement
//   (only u, v, pe, delp, pt do):
//
//   Phase A (:2570-2593, grid_type==0/mpp_get_boundary branch only --
//     grid_type>=4 Doubly-Periodic is dead in production, not implemented):
//     u(:,:,:)=tiny_number; v(:,:,:)=tiny_number
//     u(isc:iec,jsc:jec,:) = STATE%VARS%U
//     v(isc:iec,jsc:jec,:) = STATE%VARS%V
//   [Phase B, host-only, unchanged: mpp_get_boundary + the nbuffer/ebuffer
//    edge-row scatter -- see fv3_state_to_fv.cuh's header comment.]
//   Phase C (:2595-2681, SW_DYNAMICS/ADJUST_SPLITS branches dead in
//     production, not implemented; (.not. hydrostatic) required, matching
//     Stage 8's selector gate):
//     w(isc:iec,jsc:jec,:) = STATE%VARS%W                         [:2595]
//     pe(:,:,:)=tiny_number                                       [:2604]
//     do k=1,km+1: pe(i,k,j) = STATE%VARS%PE(i,j,k)                [:2621-2627]
//     do k=1,km+1: peln(i,k,j) = log(pe(i,k,j))                    [:2630-2636]
//     do k=1,km+1: pk(i,j,k) = exp(akap*peln(i,k,j))               [:2639-2645]
//     ps(isc:iec,jsc:jec) = pe(isc:iec,km+1,jsc:jec)               [:2648]
//     delp(:,:,:)=tiny_number                                     [:2652]
//     do k=1,km: delp(i,j,k) = pe(i,k+1,j) - pe(i,k,j)             [:2654-2659]
//     pt(:,:,:)=tiny_number                                       [:2668]
//     pt(isc:iec,jsc:jec,:) = STATE%VARS%PT*STATE%VARS%PKZ         [:2669]
//     delz(isc:iec,jsc:jec,:) = STATE%VARS%DZ                     [:2680]
//     pkz(isc:iec,jsc:jec,:) = STATE%VARS%PKZ                     [:2685]
//
// akap = kappa always (SW_DYNAMICS's akap=1 branch is dead), same
// NumericalConfig::akap every other vertical-remap kernel already uses.
//
// pe/peln/pk have no boundary pass-through in this routine (unlike
// REMAP_PRESSUREVARS): every one of the km+1 levels for every interior
// column is directly overwritten from STATE%VARS%PE / log / exp, so this
// kernel does not need to read-before-overwrite the way
// remap_pressurevars_kernel does for its k=0/km boundary values.
//
// delp re-reads pe back out of the SAME (just-written) device view to
// form pe(k+1)-pe(k) -- safe because one thread performs the whole column
// (pe write, then delp read) sequentially, so there is no race with any
// other thread's column. Same safety argument as
// fv3_remap_pressurevars.cu's Pass 2.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_state_to_fv.cuh"
#include "fv3_vertical_column.hpp"
#include "fv3_gpu_errors.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace fv3::gpu::vertical {
namespace {

__global__ void state_to_fv_phase_a_kernel(const StateToFvPhaseAArgs a) {
  const int64_t i = a.is + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t j = a.js + int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (i > a.ie || j > a.je) return;

  for (int k = 0; k < a.km; ++k) {
    a.u.at(i, j, k) = a.state_u.at(i, j, k);  // :2591 u(isc:iec,jsc:jec,:)
    a.v.at(i, j, k) = a.state_v.at(i, j, k);  // :2592 v(isc:iec,jsc:jec,:)
  }
}

__global__ void state_to_fv_phase_c_kernel(const StateToFvPhaseCArgs a) {
  const int64_t i = a.is + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t j = a.js + int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (i > a.ie || j > a.je) return;

  const int km = a.km;

  // ---- w (:2595) -----------------------------------------------------
  for (int k = 0; k < km; ++k) {
    a.w.at(i, j, k) = a.state_w.at(i, j, k);
  }

  // ---- pe transpose (:2621-2627), then peln/pk (:2630-2645) ----------
  // STATE%VARS%PE is (i,j,k), Fortran k=1..km+1 -- 0-based k=0..km. pe/
  // peln are registered (i,k,j) (k MIDDLE); pk is registered (i,j,k) (k
  // last), same convention every other vertical-remap kernel documents.
  for (int k = 0; k <= km; ++k) {
    const fv_real pe_k = a.state_pe.at(i, j, k);
    a.pe.at(i, k, j) = pe_k;
    const fv_real peln_k = fv3_log<fv_real>(pe_k);
    a.peln.at(i, k, j) = peln_k;
    a.pk.at(i, j, k) = fv3_exp<fv_real>(a.akap * peln_k);
  }

  // ---- ps (:2648) -- packed into its own scratch, no FieldRegistry ----
  // identity (see fv3_state_to_fv.cuh's header comment). Fortran
  // pe(isc:iec,km+1,jsc:jec) is 0-based pe.at(i,km,j).
  a.ps_out.at(i, j) = a.pe.at(i, km, j);

  // ---- delp (:2654-2659) -- re-reads pe back out of the SAME device --
  // view this thread just finished writing above; safe, see header.
  for (int k = 0; k < km; ++k) {
    a.delp.at(i, j, k) = a.pe.at(i, k + 1, j) - a.pe.at(i, k, j);
  }

  // ---- pt = STATE%VARS%PT*STATE%VARS%PKZ (:2669) ----------------------
  for (int k = 0; k < km; ++k) {
    a.pt.at(i, j, k) = a.state_pt.at(i, j, k) * a.state_pkz.at(i, j, k);
  }

  // ---- delz (:2680), pkz (:2685) --------------------------------------
  for (int k = 0; k < km; ++k) {
    a.delz.at(i, j, k) = a.state_dz.at(i, j, k);
    a.pkz.at(i, j, k) = a.state_pkz.at(i, j, k);
  }
}

}  // namespace

fv3_gpu_status launch_state_to_fv_phase_a(const StateToFvPhaseAArgs &a,
                                          void *stream, fv3_gpu_error *error) {
  clear_error(error);
  const char *api = "launch_state_to_fv_phase_a";

  if (!stream)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "null CUDA stream"));

  if (a.ie < a.is || a.je < a.js)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "empty compute bounds"));

  if (!a.u.p || !a.v.p || !a.state_u.p || !a.state_v.p)
    return static_cast<fv3_gpu_status>(fail(
        error, FV3_GPU_INVALID_ARGUMENT, api,
        "null device field: state_to_fv_phase_a requires u, v, state_u, "
        "state_v"));

  if (a.km < 1)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km < 1: no levels to copy"));

  const dim3 threads(16, 8);
  const int64_t nx = a.ie - a.is + 1;
  const int64_t ny = a.je - a.js + 1;
  const dim3 grid(static_cast<unsigned>((nx + 15) / 16),
                  static_cast<unsigned>((ny + 7) / 8));
  auto s = static_cast<cudaStream_t>(stream);

  state_to_fv_phase_a_kernel<<<grid, threads, 0, s>>>(a);

  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(launch_status),
             static_cast<int>(launch_status)));
  return FV3_GPU_SUCCESS;
}

fv3_gpu_status launch_state_to_fv_phase_c(const StateToFvPhaseCArgs &a,
                                          void *stream, fv3_gpu_error *error) {
  clear_error(error);
  const char *api = "launch_state_to_fv_phase_c";

  if (!stream)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "null CUDA stream"));

  if (a.ie < a.is || a.je < a.js)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "empty compute bounds"));

  if (!a.w.p || !a.pe.p || !a.peln.p || !a.pk.p || !a.delp.p || !a.pt.p ||
      !a.delz.p || !a.pkz.p || !a.state_w.p || !a.state_pe.p ||
      !a.state_pt.p || !a.state_pkz.p || !a.state_dz.p || !a.ps_out.p)
    return static_cast<fv3_gpu_status>(fail(
        error, FV3_GPU_INVALID_ARGUMENT, api,
        "null device field: state_to_fv_phase_c requires w, pe, peln, pk, "
        "delp, pt, delz, pkz, state_w, state_pe, state_pt, state_pkz, "
        "state_dz, ps_out"));

  if (a.km < 1)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km < 1: no levels to compute"));

  if (a.akap <= fv_real(0) || a.akap >= fv_real(1))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "akap must be in (0,1): pk = exp(akap*peln) "
             "(FV_StateMod.F90:2639-2645)"));

  const dim3 threads(16, 8);
  const int64_t nx = a.ie - a.is + 1;
  const int64_t ny = a.je - a.js + 1;
  const dim3 grid(static_cast<unsigned>((nx + 15) / 16),
                  static_cast<unsigned>((ny + 7) / 8));
  auto s = static_cast<cudaStream_t>(stream);

  state_to_fv_phase_c_kernel<<<grid, threads, 0, s>>>(a);

  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(launch_status),
             static_cast<int>(launch_status)));
  return FV3_GPU_SUCCESS;
}

int launch_state_to_fv_phase_a_ctx(const ExecutionContext &x,
                                   fv3_gpu_error *error) noexcept {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  const char *api = "launch_state_to_fv_phase_a_ctx";
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "state_to_fv_phase_a must be enqueued on the compute stream");

  const auto *v_u = x.fields->find(FieldId::u);
  const auto *v_v = x.fields->find(FieldId::v);
  if (!v_u || !v_v)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "state_to_fv_phase_a requires u, v");

  // state_u/state_v are NOT FieldRegistry-resident here (STATE%VARS%* has
  // no FV_Atm identity, see this file's .cuh header) -- the caller
  // (fv3_gpu_runtime.cpp's fv3_gpu_state_to_fv_phase_a) stashes their
  // dedicated device-buffer pointers into DispatchParams' additive tail
  // fields before invoking dispatch.state_to_fv_phase_a(x, e), same
  // convention DispatchParams::remap_q_buffer already documents.
  const auto &dp = *x.dispatch_params;
  if (!dp.state_to_fv_u_buffer || !dp.state_to_fv_v_buffer)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "state_to_fv_phase_a requires state_to_fv_u_buffer/"
                "state_to_fv_v_buffer dispatch-param buffers to be set "
                "before dispatch");

  const auto &b = x.immutable->bounds;

  StateToFvPhaseAArgs a{};
  a.u = riem_c_view3(*v_u);
  a.v = riem_c_view3(*v_v);

  a.is = b.is; a.ie = b.ie; a.js = b.js; a.je = b.je;
  a.km = static_cast<int32_t>(b.npz);

  const int64_t ni = a.ie - a.is + 1;
  const int64_t nj = a.je - a.js + 1;

  a.state_u = StateInputView{dp.state_to_fv_u_buffer, a.is, a.js, ni, nj};
  a.state_v = StateInputView{dp.state_to_fv_v_buffer, a.is, a.js, ni, nj};

  return launch_state_to_fv_phase_a(a, x.submission.stream, error);
}

int launch_state_to_fv_phase_c_ctx(const ExecutionContext &x,
                                   fv3_gpu_error *error) noexcept {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  const char *api = "launch_state_to_fv_phase_c_ctx";
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "state_to_fv_phase_c must be enqueued on the compute stream");

  const auto *v_w = x.fields->find(FieldId::w);
  const auto *v_pe = x.fields->find(FieldId::pe);
  const auto *v_peln = x.fields->find(FieldId::peln);
  const auto *v_pk = x.fields->find(FieldId::pk);
  const auto *v_delp = x.fields->find(FieldId::delp);
  const auto *v_pt = x.fields->find(FieldId::pt);
  const auto *v_delz = x.fields->find(FieldId::delz);
  const auto *v_pkz = x.fields->find(FieldId::pkz);
  if (!v_w || !v_pe || !v_peln || !v_pk || !v_delp || !v_pt || !v_delz ||
      !v_pkz)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "state_to_fv_phase_c requires w, pe, peln, pk, delp, pt, "
                "delz, pkz");

  const auto &dp = *x.dispatch_params;
  if (!dp.state_to_fv_w_buffer || !dp.state_to_fv_pe_buffer ||
      !dp.state_to_fv_pt_buffer || !dp.state_to_fv_pkz_buffer ||
      !dp.state_to_fv_dz_buffer || !dp.state_to_fv_ps_buffer)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "state_to_fv_phase_c requires state_to_fv_w/pe/pt/pkz/dz/"
                "ps dispatch-param buffers to be set before dispatch");

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;

  if (cfg.hydrostatic)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "hydrostatic=true not implemented (FV_StateMod.F90:2595, "
                "2680 guard w/delz updates on .not. hydrostatic)");

  StateToFvPhaseCArgs a{};
  a.w = riem_c_view3(*v_w);
  a.pe = riem_c_view3(*v_pe);
  a.peln = riem_c_view3(*v_peln);
  a.pk = riem_c_view3(*v_pk);
  a.delp = riem_c_view3(*v_delp);
  a.pt = riem_c_view3(*v_pt);
  a.delz = riem_c_view3(*v_delz);
  a.pkz = riem_c_view3(*v_pkz);

  a.is = b.is; a.ie = b.ie; a.js = b.js; a.je = b.je;
  a.km = static_cast<int32_t>(b.npz);
  a.akap = fv_real(cfg.akap);

  const int64_t ni = a.ie - a.is + 1;
  const int64_t nj = a.je - a.js + 1;

  a.state_w = StateInputView{dp.state_to_fv_w_buffer, a.is, a.js, ni, nj};
  a.state_pe = StateInputView{dp.state_to_fv_pe_buffer, a.is, a.js, ni, nj};
  a.state_pt = StateInputView{dp.state_to_fv_pt_buffer, a.is, a.js, ni, nj};
  a.state_pkz = StateInputView{dp.state_to_fv_pkz_buffer, a.is, a.js, ni, nj};
  a.state_dz = StateInputView{dp.state_to_fv_dz_buffer, a.is, a.js, ni, nj};
  a.ps_out = PsPackView{dp.state_to_fv_ps_buffer, a.is, a.js, ni};

  return launch_state_to_fv_phase_c(a, x.submission.stream, error);
}

}  // namespace fv3::gpu::vertical
