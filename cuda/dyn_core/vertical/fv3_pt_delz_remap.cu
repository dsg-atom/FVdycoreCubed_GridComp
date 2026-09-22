//===========================================================================
// WP-C: production pt/delz vertical-remap CUDA kernel + launchers
// (Prompt 52C). One thread per horizontal column, plain interior domain
// is..ie x js..je (NOT Riem_Solver_c's widened is-1..ie+1/js-1..je+1 --
// fv_mapz.F90's own `do 1000/1002 j=js,je` + `is:ie` slices need no halo).
//
// Authoritative Fortran (numerical source of truth, read-only reference):
//   GFDL_atmos_cubed_sphere tag geos/v3.0.0 (ca30e26), model/fv_mapz.F90.
// All ":NNN" comments below are fv_mapz.F90 line numbers. Validated
// standalone reference for every step below except map_scalar itself
// (already validated, Prompt 52B): pt_delz-driver/main.F90 (Prompt
// 52C-prep, docs/performance/p52c_prep_pt_delz_fixture_2026-09-21.md).
//
// map_scalar_column (Prompt 52B, fv3_map_scalar_column.hpp) is called
// exactly as the Fortran calls map_scalar -- not re-derived here. This file
// contributes only the surrounding pressure-grid/pt/delz/pkz arithmetic
// that map_scalar does not cover, exactly matching the driver.
//
// MOIST_CAPPA/nwat contract: this kernel implements ONLY the dry-degenerate
// (nwat=0) formulas (moist_cv's case-default: cvm=cv_air, cappa=akap).
// nwat and kord are Fortran-only values with no NumericalConfig
// representation, so this file cannot itself verify nwat==0 or kord==9 --
// that is an external contract enforced at the Fortran dispatch call site.
// Every launch through this entry point is trusted to have already been
// gated there. See fv3_pt_delz_remap.cuh's header comment.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_pt_delz_remap.cuh"
#include "fv3_vertical_column.hpp"
#include "fv3_map_scalar_column.hpp"
#include "fv3_gpu_errors.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace fv3::gpu::vertical {
namespace {

inline constexpr int kMaxLevels = pt_delz_remap_max_local_levels;

// fv_mapz.F90:99-100 module parameters -- fixed physical constants, not
// runtime-configurable, so not plumbed through NumericalConfig.
template <class T> FV3_PTDZ_HD inline T pt_delz_t_min() { return T(184); }
template <class T> FV3_PTDZ_HD inline T pt_delz_no_min() { return T(-1e25); }

__global__ void pt_delz_remap_kernel(const PtDelzRemapArgs a) {
  const int64_t i = a.is + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t j = a.js + int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (i > a.ie || j > a.je) return;

  const int km = a.km;

  // Per-thread local scratch (stack/local memory, no device workspace --
  // see fv3_pt_delz_remap.cuh's header comment for why one is not needed).
  fv_real pe_old[kMaxLevels + 1];
  fv_real pn1[kMaxLevels + 1];
  fv_real peO[kMaxLevels + 1];
  fv_real pn2[kMaxLevels + 1];
  fv_real prof_q[kMaxLevels + 1];  // scalar_profile_column's own q(:) scratch

  fv_real dpn1[kMaxLevels], dpn2[kMaxLevels];
  fv_real dpe[kMaxLevels], dpeO[kMaxLevels];
  fv_real pt_pre[kMaxLevels], delz_pre[kMaxLevels];
  fv_real q2_pt[kMaxLevels], q2_delz[kMaxLevels];
  fv_real c2[kMaxLevels], c3[kMaxLevels], c4[kMaxLevels], gam[kMaxLevels];
  bool extm[kMaxLevels];

  const fv_real k1k = a.akap / (fv_real(1) - a.akap);  // :308 rdgas/cv_air
  const fv_real rrg = -a.rdgas / a.grav;                // :311

  // ---- read old pe/peln BEFORE either is overwritten below ---------------
  for (int k = 0; k <= km; ++k) {
    pe_old[k] = a.pe.at(i, j, k);
    pn1[k] = a.peln.at(i, j, k);  // read the existing device field, NOT
                                  // recomputed via log(pe) (driver-vs-
                                  // production divergence noted in Stage 1).
  }

  // ---- peO / pn2 derivation, :482-520 -------------------------------------
  peO[0] = a.ptop;
  peO[km] = pe_old[km];
  pn2[0] = pn1[0];
  pn2[km] = pn1[km];
  for (int k = 1; k < km; ++k) {
    peO[k] = a.ak.at(k) + a.bk.at(k) * pe_old[km];
    pn2[k] = fv3_log<fv_real>(peO[k]);
  }
  for (int k = 0; k < km; ++k) {
    dpn1[k] = pn1[k + 1] - pn1[k];
    dpn2[k] = pn2[k + 1] - pn2[k];
    dpe[k] = pe_old[k + 1] - pe_old[k];    // :576-585
    dpeO[k] = peO[k + 1] - peO[k];
  }

  // ---- pt pre-transform (:388, density-pt -> density-temp, dry/nwat=0) ---
  for (int k = 0; k < km; ++k) {
    const fv_real ptv = a.pt.at(i, j, k);
    pt_pre[k] = ptv * fv3_exp<fv_real>(
        k1k * fv3_log<fv_real>(rrg * dpe[k] / a.delz.at(i, j, k) * ptv));
  }

  // ---- pt remap: iv=1, kord=9, q_min=t_min, coordinate = log(pe) ---------
  {
    Column<const fv_real> pn1v{pn1, 1}, dpn1v{dpn1, 1}, pn2v{pn2, 1},
        dpn2v{dpn2, 1}, ptprev{pt_pre, 1};
    Column<fv_real> q2ptv{q2_pt, 1};
    map_scalar_column<fv_real>(pn1v, dpn1v, pn2v, dpn2v, ptprev, q2ptv, c2, c3,
                                c4, gam, prof_q, extm, km, km, 1,
                                pt_delz_t_min<fv_real>());
  }

  // ---- delz pre-transform (:601, height -> specific volume/grav) ---------
  for (int k = 0; k < km; ++k) delz_pre[k] = -a.delz.at(i, j, k) / dpe[k];

  // ---- delz remap: iv=1, kord=9, no q_min, coordinate = pe ---------------
  {
    Column<const fv_real> pe1v{pe_old, 1}, dpe1v{dpe, 1}, pe2v{peO, 1},
        dpe2v{dpeO, 1}, dzprev{delz_pre, 1};
    Column<fv_real> q2dzv{q2_delz, 1};
    map_scalar_column<fv_real>(pe1v, dpe1v, pe2v, dpe2v, dzprev, q2dzv, c2, c3,
                                c4, gam, prof_q, extm, km, km, 1,
                                pt_delz_no_min<fv_real>());
  }

  // ---- delz post-transform (:611) + joint pkz (:928-930), using NEW
  //      delp(=dpeO)/delz/pt ------------------------------------------------
  for (int k = 0; k < km; ++k) {
    const fv_real new_delp = dpeO[k];
    const fv_real delz_out = -q2_delz[k] * new_delp;
    const fv_real pkz_out = fv3_exp<fv_real>(
        a.akap * fv3_log<fv_real>(rrg * new_delp / delz_out * q2_pt[k]));
    a.pt.at(i, j, k) = q2_pt[k];
    a.delz.at(i, j, k) = delz_out;
    a.delp.at(i, j, k) = new_delp;
    a.pkz.at(i, j, k) = pkz_out;
  }

  // ---- delp/pe/peln/pk closing update, :789-848 ---------------------------
  for (int k = 0; k <= km; ++k) {
    a.pe.at(i, j, k) = peO[k];
    a.peln.at(i, j, k) = pn2[k];
    a.pk.at(i, j, k) = fv3_exp<fv_real>(a.akap * pn2[k]);
  }
}

}  // namespace

fv3_gpu_status launch_pt_delz_remap(const PtDelzRemapArgs &a, void *stream,
                                    fv3_gpu_error *error) {
  clear_error(error);
  const char *api = "launch_pt_delz_remap";

  // ---- rejections. All of these precede every state mutation. ------------
  if (!stream)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "null CUDA stream"));

  if (a.ie < a.is || a.je < a.js)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "empty compute bounds"));

  if (!a.pe.p || !a.peln.p || !a.pt.p || !a.delz.p || !a.delp.p || !a.pk.p ||
      !a.pkz.p || !a.ak.p || !a.bk.p)
    return static_cast<fv3_gpu_status>(fail(
        error, FV3_GPU_INVALID_ARGUMENT, api,
        "null device field: pt/delz remap requires pe, peln, pt, delz, "
        "delp, pk, pkz, ak, bk"));

  if (a.km < 3)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km < 3: scalar_profile_column's Huynh interior loop "
             "(k=2..km-3) and bottom-layer terms require at least 3 levels"));

  if (a.km > kMaxLevels)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km > 128: the local-array column scratch path is capped at "
             "128 levels (no device-workspace fallback is implemented)"));

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
             "akap must be in (0,1): k1k = akap/(1-akap) (fv_mapz.F90:308)"));

  if (!(a.ptop > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "ptop must be positive: peO(1) = ptop (fv_mapz.F90:482)"));

  const dim3 threads(16, 8);
  const int64_t nx = a.ie - a.is + 1;
  const int64_t ny = a.je - a.js + 1;
  const dim3 grid(static_cast<unsigned>((nx + 15) / 16),
                  static_cast<unsigned>((ny + 7) / 8));
  auto s = static_cast<cudaStream_t>(stream);

  pt_delz_remap_kernel<<<grid, threads, 0, s>>>(a);

  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(launch_status),
             static_cast<int>(launch_status)));
  return FV3_GPU_SUCCESS;
}

int launch_pt_delz_remap_ctx(const ExecutionContext &x, fv3_gpu_error *error) noexcept {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  const char *api = "launch_pt_delz_remap_ctx";
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "pt/delz remap must be enqueued on the compute stream");

  const auto *v_pe = x.fields->find(FieldId::pe);
  const auto *v_peln = x.fields->find(FieldId::peln);
  const auto *v_pt = x.fields->find(FieldId::pt);
  const auto *v_delz = x.fields->find(FieldId::delz);
  const auto *v_delp = x.fields->find(FieldId::delp);
  const auto *v_pk = x.fields->find(FieldId::pk);
  const auto *v_pkz = x.fields->find(FieldId::pkz);
  const auto *v_ak = x.fields->find(FieldId::ak);
  const auto *v_bk = x.fields->find(FieldId::bk);
  if (!v_pe || !v_peln || !v_pt || !v_delz || !v_delp || !v_pk || !v_pkz ||
      !v_ak || !v_bk)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "pt/delz remap requires pe, peln, pt, delz, delp, pk, pkz, "
                "ak, bk");

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;

  PtDelzRemapArgs a{};
  a.pe = riem_c_view3(*v_pe);
  a.peln = riem_c_view3(*v_peln);
  a.pt = riem_c_view3(*v_pt);
  a.delz = riem_c_view3(*v_delz);
  a.delp = riem_c_view3(*v_delp);
  a.pk = riem_c_view3(*v_pk);
  a.pkz = riem_c_view3(*v_pkz);
  a.ak = ptdz_cview1(*v_ak);
  a.bk = ptdz_cview1(*v_bk);

  a.is = b.is; a.ie = b.ie; a.js = b.js; a.je = b.je;
  a.km = static_cast<int32_t>(b.npz);

  a.akap = fv_real(cfg.akap);
  a.rdgas = fv_real(cfg.rdgas);
  a.grav = fv_real(cfg.grav);
  a.ptop = fv_real(cfg.ptop);

  return launch_pt_delz_remap(a, x.submission.stream, error);
}

}  // namespace fv3::gpu::vertical
