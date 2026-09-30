//===========================================================================
// WP-C: production Remap_PressureVars CUDA kernel + launchers (Prompt 92).
// One thread per horizontal column, plain interior domain is..ie x js..je
// (matches fv_mapz.F90's own `do 1000 j=js,je` + `is:ie` slices -- no halo
// needed for compute, unlike Riem_Solver_c's widened range).
//
// Authoritative Fortran (numerical source of truth, read-only reference):
//   GFDL_atmos_cubed_sphere tag geos/v3.0.0 (ca30e26), model/fv_mapz.F90:
//   1239-1516 (Remap_PressureVars timer block). Exact per-statement mapping
//   and the authoritative host-reference transcription this kernel must
//   match bitwise (Pass 1: pe/peln/pk; round-off only in cappa/pkz, both
//   transcendental -- see fv3_remap_pressurevars.cuh's header comment) live
//   in tests/vertical/test_remap_pressurevars_host_reference.cpp (Prompt 91).
//
// Unlike fv3_pt_delz_remap.cu, this kernel needs NO map_scalar_column call,
// NO per-thread/workspace scratch array, and NO kMaxLevels cap: pe2/pn2/pk2
// are a direct closed-form recompute from ak/bk/pe(surface) (no monotone
// remap is involved in this Fortran block at all), and delp/cappa/pkz are
// direct closed-form pointwise expressions. Pass 1 writes pe2/pn2/pk2
// directly into the real pe/peln/pk FieldRegistry views; Pass 2 re-reads
// pe2 back out of those SAME (now-updated) views to form delp -- safe
// because one thread performs both passes sequentially over its own column,
// so there is no race with any other thread's column.
//
// MOIST_CAPPA/nwat contract: this kernel implements moist_cv's nwat=0
// case-default (cvm=cv_air, a true constant, Mayer's relation
// cv_air=cp_air-rdgas) parameterized on the REAL q(sphum) value (Prompt 92
// Stage 1: q(sphum)!=0 is cheap/closed-form and IS supported here, refining
// the coarser "cappa==akap whenever nwat==0" assumption fv3_pt_delz_remap.cu
// documents -- see this file's .cuh header for the full correction). nwat
// itself is a Fortran-only value with no NumericalConfig representation, so
// this file cannot verify nwat==0 itself -- that remains an external
// contract enforced at the Fortran dispatch call site, same as pt/delz.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_remap_pressurevars.cuh"
#include "fv3_vertical_column.hpp"
#include "fv3_gpu_errors.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace fv3::gpu::vertical {
namespace {

__global__ void remap_pressurevars_kernel(const PressureVarsArgs a) {
  const int64_t i = a.is + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t j = a.js + int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (i > a.ie || j > a.je) return;

  const int km = a.km;

  // ---- read values that Pass 1 is about to overwrite, BEFORE overwriting -
  // NOTE: pe/peln are registered (i,k,j) -- k is the MIDDLE Fortran
  // dimension, same convention fv3_pt_delz_remap.cu's read-loop documents
  // (Prompt 55) -- so .at(i,k,j), NOT .at(i,j,k). pk is registered (i,j,k)
  // (k last), so .at(i,j,k), matching every other field here.
  const fv_real pe_old_km = a.pe.at(i, km, j);      // :1259 pe(i,npz+1,j)
  const fv_real peln_old_0 = a.peln.at(i, 0, j);     // :1279 peln(i,1,j)
  const fv_real peln_old_km = a.peln.at(i, km, j);   // :1280 peln(i,npz+1,j)
  const fv_real pk_old_0 = a.pk.at(i, j, 0);         // :1281 pk(i,j,1)
  const fv_real pk_old_km = a.pk.at(i, j, km);       // :1282 pk(i,j,npz+1)

  // ---- Pass 1: pe2/pn2/pk2 (:1256-1291), written directly into the real
  //      pe/peln/pk fields -- boundary pass-through at k=0/km, else a
  //      direct closed-form recompute from ak/bk/pe(surface). ------------
  a.pe.at(i, 0, j) = a.ptop;          // :1260 pe2(i,1) = ptop
  a.pe.at(i, km, j) = pe_old_km;      // :1261 pe2(i,npz+1) = pe(i,npz+1,j)
  a.peln.at(i, 0, j) = peln_old_0;    // :1279 pn2(i,1) = peln(i,1,j)
  a.peln.at(i, km, j) = peln_old_km;  // :1280 pn2(i,npz+1) = peln(i,npz+1,j)
  a.pk.at(i, j, 0) = pk_old_0;        // :1281 pk2(i,1) = pk(i,j,1)
  a.pk.at(i, j, km) = pk_old_km;      // :1282 pk2(i,npz+1) = pk(i,j,npz+1)

  for (int k = 1; k < km; ++k) {
    const fv_real pe2_k = a.ak.at(k) + a.bk.at(k) * pe_old_km;  // :1263-1265
    const fv_real pn2_k = fv3_log<fv_real>(pe2_k);              // :1284-1286
    const fv_real pk2_k = fv3_exp<fv_real>(a.akap * pn2_k);
    a.pe.at(i, k, j) = pe2_k;
    a.peln.at(i, k, j) = pn2_k;
    a.pk.at(i, j, k) = pk2_k;
  }

  // ---- Pass 2: delp (:1296-1300), then cappa/pkz, remap_t + MOIST_CAPPA
  //      branch (:1423-1446), moist_cv nwat=0 case-default (cvm=cv_air)
  //      inlined. delp re-reads pe2 back out of a.pe -- safe: this thread
  //      already wrote every pe2[k] for its own column above, and no other
  //      thread touches this column. --------------------------------------
  const fv_real cv_air = a.rdgas / a.akap - a.rdgas;  // Mayer's relation,
                                                       // same formula as
                                                       // wp_d_contract.cpp
  const fv_real rrg = -a.rdgas / a.grav;              // :311 (pt_delz's own rrg)

  for (int k = 0; k < km; ++k) {
    const fv_real pe2_k = a.pe.at(i, k, j);
    const fv_real pe2_kp1 = a.pe.at(i, k + 1, j);
    const fv_real delp_k = pe2_kp1 - pe2_k;
    a.delp.at(i, j, k) = delp_k;

    const fv_real q_sphum_k = a.q_sphum.at(i, j, k);
    const fv_real cappa_k =
        a.rdgas / (a.rdgas + cv_air / (fv_real(1) + a.r_vir * q_sphum_k));
    const fv_real pt_k = a.pt.at(i, j, k);
    const fv_real delz_k = a.delz.at(i, j, k);
    const fv_real pkz_k = fv3_exp<fv_real>(
        cappa_k * fv3_log<fv_real>(rrg * delp_k / delz_k * pt_k));

    a.cappa.at(i, j, k) = cappa_k;
    a.pkz.at(i, j, k) = pkz_k;
  }
}

}  // namespace

fv3_gpu_status launch_remap_pressurevars(const PressureVarsArgs &a, void *stream,
                                         fv3_gpu_error *error) {
  clear_error(error);
  const char *api = "launch_remap_pressurevars";

  // ---- rejections. All of these precede every state mutation. ------------
  if (!stream)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "null CUDA stream"));

  if (a.ie < a.is || a.je < a.js)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "empty compute bounds"));

  if (!a.pe.p || !a.peln.p || !a.pk.p || !a.delp.p || !a.pkz.p || !a.pt.p ||
      !a.delz.p || !a.ak.p || !a.bk.p || !a.q_sphum.p || !a.cappa.p)
    return static_cast<fv3_gpu_status>(fail(
        error, FV3_GPU_INVALID_ARGUMENT, api,
        "null device field: remap_pressurevars requires pe, peln, pk, delp, "
        "pkz, pt, delz, ak, bk, q_sphum, cappa"));

  if (a.km < 1)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km < 1: no levels to remap"));

  if (!(a.grav > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "grav must be positive: rrg = -rdgas/grav (fv_mapz.F90:311)"));

  if (!(a.rdgas > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "rdgas must be positive"));

  if (a.akap <= fv_real(0) || a.akap >= fv_real(1))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "akap must be in (0,1): cv_air = rdgas/akap - rdgas"));

  if (!(a.ptop > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "ptop must be positive: pe2(1) = ptop (fv_mapz.F90:1260)"));

  if (!(fv_real(1) + a.r_vir * fv_real(0) > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "r_vir out of range: 1+r_vir*q(sphum) denominator must stay "
             "positive"));

  const dim3 threads(16, 8);
  const int64_t nx = a.ie - a.is + 1;
  const int64_t ny = a.je - a.js + 1;
  const dim3 grid(static_cast<unsigned>((nx + 15) / 16),
                  static_cast<unsigned>((ny + 7) / 8));
  auto s = static_cast<cudaStream_t>(stream);

  remap_pressurevars_kernel<<<grid, threads, 0, s>>>(a);

  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(launch_status),
             static_cast<int>(launch_status)));
  return FV3_GPU_SUCCESS;
}

int launch_remap_pressurevars_ctx(const ExecutionContext &x, fv3_gpu_error *error) noexcept {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  const char *api = "launch_remap_pressurevars_ctx";
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "remap_pressurevars must be enqueued on the compute stream");

  const auto *v_pe = x.fields->find(FieldId::pe);
  const auto *v_peln = x.fields->find(FieldId::peln);
  const auto *v_pk = x.fields->find(FieldId::pk);
  const auto *v_delp = x.fields->find(FieldId::delp);
  const auto *v_pkz = x.fields->find(FieldId::pkz);
  const auto *v_pt = x.fields->find(FieldId::pt);
  const auto *v_delz = x.fields->find(FieldId::delz);
  const auto *v_ak = x.fields->find(FieldId::ak);
  const auto *v_bk = x.fields->find(FieldId::bk);
  if (!v_pe || !v_peln || !v_pk || !v_delp || !v_pkz || !v_pt || !v_delz ||
      !v_ak || !v_bk)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "remap_pressurevars requires pe, peln, pk, delp, pkz, pt, "
                "delz, ak, bk");

  // q_sphum/cappa are NOT FieldRegistry-resident here (see this file's .cuh
  // header comment) -- the caller (fv3_gpu_runtime.cpp's
  // fv3_gpu_remap_pressurevars) stashes their dedicated device-buffer
  // pointers and addressing bounds into DispatchParams' additive tail
  // fields before invoking dispatch.remap_pressurevars(x, e), same
  // convention DispatchParams::remap_q_buffer already documents.
  const auto &dp = *x.dispatch_params;
  if (!dp.remap_pressurevars_qsphum_buffer || !dp.remap_pressurevars_cappa_buffer)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "remap_pressurevars requires q_sphum/cappa dispatch-param "
                "buffers to be set before dispatch");

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;

  PressureVarsArgs a{};
  a.pe = riem_c_view3(*v_pe);
  a.peln = riem_c_view3(*v_peln);
  a.pk = riem_c_view3(*v_pk);
  a.delp = riem_c_view3(*v_delp);
  a.pkz = riem_c_view3(*v_pkz);
  a.pt = riem_c_view3(*v_pt);
  a.delz = riem_c_view3(*v_delz);
  a.ak = ptdz_cview1(*v_ak);
  a.bk = ptdz_cview1(*v_bk);

  a.q_sphum.p = dp.remap_pressurevars_qsphum_buffer;
  a.q_sphum.isd = dp.remap_pressurevars_isd;
  a.q_sphum.jsd = dp.remap_pressurevars_jsd;
  a.q_sphum.nih = dp.remap_pressurevars_nih;
  a.q_sphum.njh = dp.remap_pressurevars_njh;

  a.cappa.p = dp.remap_pressurevars_cappa_buffer;
  a.cappa.isd = dp.remap_pressurevars_isd;
  a.cappa.jsd = dp.remap_pressurevars_jsd;
  a.cappa.nih = dp.remap_pressurevars_nih;
  a.cappa.njh = dp.remap_pressurevars_njh;

  a.is = b.is; a.ie = b.ie; a.js = b.js; a.je = b.je;
  a.km = static_cast<int32_t>(b.npz);

  a.akap = fv_real(cfg.akap);
  a.rdgas = fv_real(cfg.rdgas);
  a.grav = fv_real(cfg.grav);
  a.ptop = fv_real(cfg.ptop);
  a.r_vir = fv_real(cfg.zvir);

  return launch_remap_pressurevars(a, x.submission.stream, error);
}

}  // namespace fv3::gpu::vertical
