//===========================================================================
// WP-C: production u/v vertical-remap (REMAP_UV) CUDA kernel + launchers
// (Prompt 71). One thread per (i,j) in the UNION bounding box
// is..ie+1 x js..je+1; per-thread guards `i<=ie` (u) / `j<=je` (v) select
// which of u/v that thread computes, exactly mirroring Fortran's single
// `do 1003 j=js,je+1` loop with an unconditional u-block and a
// `if (j<je+1)`-guarded v-block (fv_mapz.F90:907-1010).
//
// Authoritative Fortran (numerical source of truth, read-only reference):
//   GFDL_atmos_cubed_sphere tag geos/v3.0.0 (ca30e26), model/fv_mapz.F90.
// All ":NNN" comments below are fv_mapz.F90 line numbers.
//
// map_scalar_column (Prompt 52B, extended with iv=-1 branches for this
// prompt, fv3_map_scalar_column.hpp) is called exactly as the Fortran
// calls map_scalar -- not re-derived here. This file contributes only the
// surrounding u/v-specific pressure-grid arithmetic that map_scalar does
// not cover.
//
// This kernel must be enqueued on the SAME StreamId::compute stream, and
// AFTER, PT/DELZ's own remap kernel (fv3_pt_delz_remap.cu): REMAP_UV reads
// the fully-updated (post-PT/DELZ closing-update) `pe` field, matching
// Fortran's own structurally separate `Remap_UV` timer block, which always
// runs after `Remap_NH` completes (fv_mapz.F90's own loop ordering).
// Enforced by the Fortran/bridge dispatch call site, not by this file.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_remap_uv.cuh"
#include "fv3_vertical_column.hpp"
#include "fv3_map_scalar_column.hpp"
#include "fv3_gpu_errors.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace fv3::gpu::vertical {
namespace {

inline constexpr int kMaxLevels = remap_uv_max_local_levels;

// Flat per-thread arena for the workspace-less path, same convention as
// fv3_pt_delz_remap.cu's kLocalArenaScalars -- 11 slots x (128+1). km > 128
// is rejected in the launcher when no workspace is supplied, never
// silently clamped.
inline constexpr int kLocalArenaScalars =
    remap_uv_scratch_slots * (remap_uv_max_local_levels + 1);

// fv_mapz.F90's map_scalar default when q_min is not passed (:1705-1709,
// module parameter at :100): both of REMAP_UV's map_scalar calls omit
// q_min, so both use this exact floor (an effective no-op, since no wind
// value is ever this negative). Defined locally (not reused from
// fv3_pt_delz_remap.cu's own pt_delz_no_min -- that helper has internal
// linkage in a different translation unit) but bitwise-identical.
template <class T> FV3_RMUV_HD inline T remap_uv_no_min() { return T(-1e25); }

template <bool UseLocalScratch>
__global__ void remap_uv_kernel(const RemapUvArgs a) {
  const int64_t i = a.is + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t j = a.js + int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (i > a.ie + 1 || j > a.je + 1) return;

  const int km = a.km;
  const int slot = km + 1;  // scratch slot pitch; matches km/km+1 locals

  // Column scratch. Either a per-thread local arena (stack/local memory) or
  // the caller's persistent device workspace; identical slot layout in
  // both cases. Column addressing uses the UNION-domain width (ie+2-is),
  // matching remap_uv_columns's own column-count convention exactly --
  // every thread in the launch grid gets its own scratch region regardless
  // of whether it ends up computing u, v, both, or (at the (ie+1,je+1)
  // corner) neither.
  fv_real local_arena[UseLocalScratch ? kLocalArenaScalars : 1];
  fv_real *sc;
  if constexpr (UseLocalScratch) {
    sc = local_arena;
  } else {
    const int64_t nx = (a.ie + 1) - a.is + int64_t(1);
    const int64_t col = (i - a.is) + (j - a.js) * nx;
    sc = a.workspace + col * static_cast<int64_t>(a.workspace_scalars_per_column);
  }

  fv_real *pe_src  = sc + 0 * slot;
  fv_real *dpe_src = sc + 1 * slot;
  fv_real *pe_tgt  = sc + 2 * slot;
  fv_real *dpe_tgt = sc + 3 * slot;
  fv_real *q1_in   = sc + 4 * slot;
  fv_real *q2_out  = sc + 5 * slot;
  fv_real *prof_q  = sc + 6 * slot;  // scalar_profile_column's own q(:) scratch
  fv_real *c2      = sc + 7 * slot;
  fv_real *c3      = sc + 8 * slot;
  fv_real *c4      = sc + 9 * slot;
  fv_real *gam     = sc + 10 * slot;
  bool extm[kMaxLevels];  // tiny; plain per-thread local array, no workspace
                          // slot needed (same precedent as PT/DELZ's extm).

  const fv_real no_min = remap_uv_no_min<fv_real>();

  // ======================= u-block (:907-926) ============================
  // Unconditional for every j in the thread's range (u's Fortran domain is
  // is..ie x js..je+1, i.e. every j this kernel ever launches with).
  if (i <= a.ie) {
    // pe1 (source): top hardcoded to ptop (:908-910); interior (kf=2..km+1,
    // 0-based k=1..km) is the j-1/j average of the fully-updated pe (:911-
    // 915). Kept as two distinct steps, not unified via the fact that
    // pe(i,1,*)=ptop always makes them bitwise-equal, to keep this a literal
    // transcription of Fortran's own two separate code blocks.
    pe_src[0] = a.ptop;
    for (int k = 1; k <= km; ++k) {
      pe_src[k] = fv_real(0.5) * (a.pe.at(i, k, j - 1) + a.pe.at(i, k, j));
    }
    // pe2 (target): ak(k) + 0.5*bk(k) * (j-1/j-averaged surface pressure),
    // for every k=0..km (:916-921).
    for (int k = 0; k <= km; ++k) {
      const fv_real bkh = fv_real(0.5) * a.bk.at(k);
      pe_tgt[k] = a.ak.at(k) +
                  bkh * (a.pe.at(i, km, j - 1) + a.pe.at(i, km, j));
    }
    // dpe1/dpe2 (:922-926).
    for (int k = 0; k < km; ++k) {
      dpe_src[k] = pe_src[k + 1] - pe_src[k];
      dpe_tgt[k] = pe_tgt[k + 1] - pe_tgt[k];
    }
    for (int k = 0; k < km; ++k) q1_in[k] = a.u.at(i, j, k);

    {
      Column<const fv_real> pe1v{pe_src, 1}, dpe1v{dpe_src, 1}, pe2v{pe_tgt, 1},
          dpe2v{dpe_tgt, 1}, q1v{q1_in, 1};
      Column<fv_real> q2v{q2_out, 1};
      map_scalar_column<fv_real>(pe1v, dpe1v, pe2v, dpe2v, q1v, q2v, c2, c3, c4,
                                  gam, prof_q, extm, km, km, -1, no_min);
    }

    for (int k = 0; k < km; ++k) a.u.at(i, j, k) = q2_out[k];
  }

  // ======================= v-block (:928-1008) ============================
  // Guarded `j < je+1` in Fortran (:928), i.e. j<=je here -- v's Fortran
  // domain is is..ie+1 x js..je, one column short of the full thread j
  // range but one column wider in i than u.
  if (j <= a.je) {
    // pe0 (source): i-1/i average of the fully-updated pe, for every
    // k=0..km (:929-936). Unlike u's pe1, Fortran's own separate top line
    // (:929-931) and its k=2..km+1 loop (:932-936) use the IDENTICAL
    // formula -- so one uniform loop is itself the literal transcription,
    // not a simplification.
    for (int k = 0; k <= km; ++k) {
      pe_src[k] = fv_real(0.5) * (a.pe.at(i - 1, k, j) + a.pe.at(i, k, j));
    }
    // pe3 (target): ak(k) + 0.5*bk(k) * (i-1/i-averaged surface pressure),
    // for every k=0..km (:937-942).
    for (int k = 0; k <= km; ++k) {
      const fv_real bkh = fv_real(0.5) * a.bk.at(k);
      pe_tgt[k] = a.ak.at(k) +
                  bkh * (a.pe.at(i - 1, km, j) + a.pe.at(i, km, j));
    }
    // dpe0/dpe3 (:943-947).
    for (int k = 0; k < km; ++k) {
      dpe_src[k] = pe_src[k + 1] - pe_src[k];
      dpe_tgt[k] = pe_tgt[k + 1] - pe_tgt[k];
    }
    for (int k = 0; k < km; ++k) q1_in[k] = a.v.at(i, j, k);

    {
      Column<const fv_real> pe0v{pe_src, 1}, dpe0v{dpe_src, 1}, pe3v{pe_tgt, 1},
          dpe3v{dpe_tgt, 1}, q1v{q1_in, 1};
      Column<fv_real> q2v{q2_out, 1};
      map_scalar_column<fv_real>(pe0v, dpe0v, pe3v, dpe3v, q1v, q2v, c2, c3, c4,
                                  gam, prof_q, extm, km, km, -1, no_min);
    }

    for (int k = 0; k < km; ++k) a.v.at(i, j, k) = q2_out[k];
  }
}

}  // namespace

fv3_gpu_status launch_remap_uv(const RemapUvArgs &a, void *stream,
                               fv3_gpu_error *error) {
  clear_error(error);
  const char *api = "launch_remap_uv";

  // ---- rejections. All of these precede every state mutation. ------------
  if (!stream)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "null CUDA stream"));

  if (a.ie < a.is || a.je < a.js)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "empty compute bounds"));

  if (!a.pe.p || !a.u.p || !a.v.p || !a.ak.p || !a.bk.p)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api,
             "null device field: REMAP_UV requires pe, u, v, ak, bk"));

  if (a.km < 3)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km < 3: scalar_profile_column's Huynh interior loop "
             "(k=2..km-3) and bottom-layer terms require at least 3 levels"));

  const bool use_workspace = a.workspace != nullptr;
  if (use_workspace) {
    const uint64_t need_per_column = remap_uv_scalars_per_column(a.km);
    const uint64_t columns = remap_uv_columns(a.is, a.ie, a.js, a.je);
    if (a.workspace_scalars_per_column < need_per_column)
      return static_cast<fv3_gpu_status>(
          fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
               "workspace_scalars_per_column too small for km"));
    if (a.workspace_bytes <
        columns * a.workspace_scalars_per_column * sizeof(fv_real))
      return static_cast<fv3_gpu_status>(
          fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
               "workspace too small for the (ie+1-is+1) x (je+1-js+1) "
               "union-domain column count of REMAP_UV"));
  } else if (a.km > kMaxLevels) {
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km > 128 requires a device workspace: the local-array column "
             "scratch path is capped at 128 levels"));
  }

  if (!(a.ptop > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "ptop must be positive: pe1(i,1) = ptop (fv_mapz.F90:908)"));

  const dim3 threads(16, 8);
  const int64_t nx = (a.ie + 1) - a.is + 1;
  const int64_t ny = (a.je + 1) - a.js + 1;
  const dim3 grid(static_cast<unsigned>((nx + 15) / 16),
                  static_cast<unsigned>((ny + 7) / 8));
  auto s = static_cast<cudaStream_t>(stream);

  if (use_workspace)
    remap_uv_kernel<false><<<grid, threads, 0, s>>>(a);
  else
    remap_uv_kernel<true><<<grid, threads, 0, s>>>(a);

  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(launch_status),
             static_cast<int>(launch_status)));
  return FV3_GPU_SUCCESS;
}

int launch_remap_uv_ctx(const ExecutionContext &x, fv3_gpu_error *error) noexcept {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  const char *api = "launch_remap_uv_ctx";
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "REMAP_UV must be enqueued on the compute stream");

  const auto *v_pe = x.fields->find(FieldId::pe);
  const auto *v_u = x.fields->find(FieldId::u);
  const auto *v_v = x.fields->find(FieldId::v);
  const auto *v_ak = x.fields->find(FieldId::ak);
  const auto *v_bk = x.fields->find(FieldId::bk);
  if (!v_pe || !v_u || !v_v || !v_ak || !v_bk)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "REMAP_UV requires pe, u, v, ak, bk");

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;

  RemapUvArgs a{};
  a.pe = riem_c_view3(*v_pe);
  a.u = riem_c_view3(*v_u);
  a.v = riem_c_view3(*v_v);
  a.ak = ptdz_cview1(*v_ak);
  a.bk = ptdz_cview1(*v_bk);

  a.is = b.is; a.ie = b.ie; a.js = b.js; a.je = b.je;
  a.km = static_cast<int32_t>(b.npz);

  a.ptop = fv_real(cfg.ptop);

  // Prefer the persistent WP-E vertical workspace (same slice PT/DELZ and
  // every Riemann-solver vertical consumer already use); fall back to
  // per-thread local scratch only when it cannot hold this column count.
  // Safe to share: this launcher requires StreamId::compute (checked
  // above), same as every other consumer of WorkspaceId::vertical, so GPU
  // stream ordering guarantees no concurrent use of the same scratch
  // region -- in particular, PT/DELZ's own remap kernel (enqueued earlier
  // on the same stream, per this launcher's own dispatch-site contract)
  // has fully finished with its claim on this scratch region before this
  // kernel starts.
  {
    const uint64_t need_per_column = remap_uv_scalars_per_column(a.km);
    const uint64_t columns = remap_uv_columns(a.is, a.ie, a.js, a.je);
    const auto *slice = workspace_slice(*x.workspace, WorkspaceId::vertical);
    if (slice && slice->data &&
        slice->bytes >= columns * need_per_column * sizeof(fv_real)) {
      a.workspace = static_cast<fv_real *>(slice->data);
      a.workspace_scalars_per_column = need_per_column;
      a.workspace_bytes = slice->bytes;
    }
  }

  return launch_remap_uv(a, x.submission.stream, error);
}

}  // namespace fv3::gpu::vertical
