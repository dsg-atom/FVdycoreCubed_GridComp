//===========================================================================
// WP-C: production tracer vertical-remap (REMAP_Q, mapn_tracer nq=2:
// sphum kord=9, cld_amt kord=9 -- corrected by Prompt 87, see
// fv3_remap_q.cuh's header comment) CUDA kernel + launchers (Prompt 85
// nq=1/kord=8; extended to production's real nq=2 schedule by Prompt 86B
// via two sequential single-tracer launches of the SAME kernel below, then
// corrected to the actual kord=(9,9) schedule by Prompt 87 -- no new
// numerics in either step). One thread per (i,j) in the PLAIN
// interior domain is..ie x js..je -- matching fv_mapz.F90's own
// `do 1001 j=js,je` / mapn_tracer(is,ie, ...) call bounds, NOT REMAP_UV's
// widened union box (REMAP_Q has no edge-averaged u/v-style pe0/pe1 split;
// every interior column is self-contained).
//
// Authoritative Fortran (numerical source of truth, read-only reference):
//   GFDL_atmos_cubed_sphere tag geos/v3.0.0 (ca30e26), model/fv_mapz.F90.
// All ":NNN" comments below are fv_mapz.F90 line numbers.
//
// pe1/pe2/dp2 construction (:824-843) is IDENTICAL to
// fv3_pt_delz_remap.cu's own DELZ-branch peO/dpeO construction (confirmed
// Prompt 85 Stage 2/5) -- reproduced independently here rather than shared,
// since PtDelzRemapArgs's pe/peln/pt/delz/... shape does not fit REMAP_Q's
// single tracer buffer. map_scalar_column (Prompt 52B/85, iv=0/kord=8
// branch) is called exactly as Fortran's mapn_tracer calls scalar_profile
// + its own "Mapping" search/integrate section -- not re-derived here.
//
// This kernel must be enqueued on the SAME StreamId::compute stream, and
// AFTER, PT/DELZ's own remap kernel (fv3_pt_delz_remap.cu): REMAP_Q reads
// the fully-updated (post-PT/DELZ closing-update) `pe` field, matching
// Fortran's own structurally separate `Remap_Q` timer block, which always
// runs after `Remap_NH` completes. Enforced by the Fortran/bridge dispatch
// call site, not by this file.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_remap_q.cuh"
#include "fv3_vertical_column.hpp"
#include "fv3_map_scalar_column.hpp"
#include "fv3_fillz_column.hpp"
#include "fv3_gpu_errors.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace fv3::gpu::vertical {
namespace {

inline constexpr int kMaxLevels = remap_q_max_local_levels;

// Flat per-thread arena for the workspace-less path, same convention as
// fv3_pt_delz_remap.cu's/fv3_remap_uv.cu's kLocalArenaScalars.
inline constexpr int kLocalArenaScalars =
    remap_q_scratch_slots * (remap_q_max_local_levels + 1);

template <bool UseLocalScratch>
__global__ void remap_q_kernel(const RemapQArgs a) {
  const int64_t i = a.is + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t j = a.js + int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (i > a.ie || j > a.je) return;

  const int km = a.km;
  const int slot = km + 1;  // scratch slot pitch; matches km/km+1 locals

  // Column scratch. Either a per-thread local arena (stack/local memory) or
  // the caller's persistent device workspace; identical slot layout in
  // both cases. Column addressing uses the plain interior-domain width
  // (ie+1-is), matching remap_q_columns's own column-count convention.
  fv_real local_arena[UseLocalScratch ? kLocalArenaScalars : 1];
  fv_real *sc;
  if constexpr (UseLocalScratch) {
    sc = local_arena;
  } else {
    const int64_t nx = a.ie - a.is + int64_t(1);
    const int64_t col = (i - a.is) + (j - a.js) * nx;
    sc = a.workspace + col * static_cast<int64_t>(a.workspace_scalars_per_column);
  }

  fv_real *pe_old = sc + 0 * slot;
  fv_real *dpe_old = sc + 1 * slot;
  fv_real *peO = sc + 2 * slot;
  fv_real *dpeO = sc + 3 * slot;
  fv_real *q1_in = sc + 4 * slot;
  fv_real *q2_out = sc + 5 * slot;
  fv_real *prof_q = sc + 6 * slot;  // scalar_profile_column's own q(:) scratch
  fv_real *c2 = sc + 7 * slot;
  fv_real *c3 = sc + 8 * slot;
  fv_real *c4 = sc + 9 * slot;
  fv_real *gam = sc + 10 * slot;
  bool extm[kMaxLevels];  // tiny; plain per-thread local array, no workspace
                          // slot needed (same precedent as PT/DELZ/REMAP_UV).

  // pe1/dp1 (source grid, :824-830): the fully-updated pe, read straight
  // through, no averaging (unlike REMAP_UV's edge-averaged pe0/pe1).
  for (int k = 0; k <= km; ++k) pe_old[k] = a.pe.at(i, k, j);
  for (int k = 0; k < km; ++k) dpe_old[k] = pe_old[k + 1] - pe_old[k];

  // pe2/dp2 (target grid, :831-843): top = ptop, bottom = pe1(km+1)
  // (unchanged surface pressure), interior = ak(k) + bk(k)*ps -- identical
  // formula to fv3_pt_delz_remap.cu's own DELZ-branch peO construction.
  peO[0] = a.ptop;
  peO[km] = pe_old[km];
  for (int k = 1; k < km; ++k) peO[k] = a.ak.at(k) + a.bk.at(k) * pe_old[km];
  for (int k = 0; k < km; ++k) dpeO[k] = peO[k + 1] - peO[k];

  // q1 (:read from the persistent device tracer buffer, laid out exactly as
  // Fortran's own q(isd:ied,jsd:jed,km,nq): i fastest, then j, then k. a.q
  // already points at this launch's own tracer's plane base -- the nq
  // offset is applied by the caller, launch_remap_q_ctx, not here).
  const int64_t nx_d = a.ied - a.isd + 1;
  const int64_t ny_d = a.jed - a.jsd + 1;
  const int64_t base = (i - a.isd) + (j - a.jsd) * nx_d;
  const int64_t plane = nx_d * ny_d;
  for (int k = 0; k < km; ++k) q1_in[k] = a.q[base + int64_t(k) * plane];

  {
    Column<const fv_real> pe1v{pe_old, 1}, dpe1v{dpe_old, 1}, pe2v{peO, 1},
        dpe2v{dpeO, 1}, q1v{q1_in, 1};
    Column<fv_real> q2v{q2_out, 1};
    // iv=0 (scalar; mapn_tracer hardcodes iv=0), qmin=0 (mapn_tracer's own
    // call site literal), gmao_bot/gmao_top=false (mapn_tracer has no
    // GMAO-boundary concept), kord=a.kord (Prompt 86B: per-tracer -- 8 for
    // sphum, 9 for cld_amt; a single launch still handles exactly one
    // tracer/kord pair, same as Prompt 85).
    map_scalar_column<fv_real>(pe1v, dpe1v, pe2v, dpe2v, q1v, q2v, c2, c3, c4,
                                gam, prof_q, extm, km, km, /*iv=*/0,
                                /*qmin=*/fv_real(0), /*gmao_bot=*/false,
                                /*gmao_top=*/false, a.kord);
  }

  // Prompt 88B: mapn_tracer's `if (fill) call fillz(...)` (fv_mapz.F90:2291),
  // applied in place to this tracer's own remapped column, immediately
  // before write-back -- same placement Fortran uses. dpeO is the target-
  // grid dp2 fillz needs; c2 is dead/free scratch here (map_scalar_column
  // has already returned) and hosts fillz's own dm() local array -- no new
  // scratch slot, no new global-memory traffic.
  if (a.fill) fillz_column<fv_real>(q2_out, dpeO, km, c2);

  // Write back in place (:mapn_tracer's own closing q1(i,j,k,iq)=q2(i,k,iq)
  // loop) -- q2_out is a separate per-thread scratch slot from q1_in, so
  // this is not a self-aliased read-after-write within map_scalar_column.
  for (int k = 0; k < km; ++k) a.q[base + int64_t(k) * plane] = q2_out[k];
}

}  // namespace

fv3_gpu_status launch_remap_q(const RemapQArgs &a, void *stream,
                              fv3_gpu_error *error) {
  clear_error(error);
  const char *api = "launch_remap_q";

  // ---- rejections. All of these precede every state mutation. ------------
  if (!stream)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "null CUDA stream"));

  if (a.ie < a.is || a.je < a.js)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "empty compute bounds"));

  if (!a.pe.p || !a.ak.p || !a.bk.p || !a.q)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api,
             "null device field: REMAP_Q requires pe, ak, bk, q"));

  if (a.isd > a.is || a.ied < a.ie || a.jsd > a.js || a.jed < a.je)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api,
             "q buffer memory domain (isd:ied,jsd:jed) does not contain the "
             "interior compute domain (is:ie,js:je)"));

  if (a.km < 3)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km < 3: scalar_profile_column's Huynh interior loop "
             "(k=2..km-3) and bottom-layer terms require at least 3 levels"));

  if (a.kord != 8 && a.kord != 9)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "kord must be 8 (sphum) or 9 (cld_amt): the only production "
             "REMAP_Q kord values (fv_dynamics.F90:941-944)"));

  const bool use_workspace = a.workspace != nullptr;
  if (use_workspace) {
    const uint64_t need_per_column = remap_q_scalars_per_column(a.km);
    const uint64_t columns = remap_q_columns(a.is, a.ie, a.js, a.je);
    if (a.workspace_scalars_per_column < need_per_column)
      return static_cast<fv3_gpu_status>(
          fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
               "workspace_scalars_per_column too small for km"));
    if (a.workspace_bytes <
        columns * a.workspace_scalars_per_column * sizeof(fv_real))
      return static_cast<fv3_gpu_status>(
          fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
               "workspace too small for the (ie+1-is) x (je+1-js) interior "
               "column count of REMAP_Q"));
  } else if (a.km > kMaxLevels) {
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km > 128 requires a device workspace: the local-array column "
             "scratch path is capped at 128 levels"));
  }

  if (!(a.ptop > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "ptop must be positive: pe2(i,1) = ptop (fv_mapz.F90:831)"));

  const dim3 threads(16, 8);
  const int64_t nx = a.ie - a.is + 1;
  const int64_t ny = a.je - a.js + 1;
  const dim3 grid(static_cast<unsigned>((nx + 15) / 16),
                  static_cast<unsigned>((ny + 7) / 8));
  auto s = static_cast<cudaStream_t>(stream);

  if (use_workspace)
    remap_q_kernel<false><<<grid, threads, 0, s>>>(a);
  else
    remap_q_kernel<true><<<grid, threads, 0, s>>>(a);

  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(launch_status),
             static_cast<int>(launch_status)));
  return FV3_GPU_SUCCESS;
}

int launch_remap_q_ctx(const ExecutionContext &x, fv3_gpu_error *error) noexcept {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  const char *api = "launch_remap_q_ctx";
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "REMAP_Q must be enqueued on the compute stream");

  const auto *v_pe = x.fields->find(FieldId::pe);
  const auto *v_ak = x.fields->find(FieldId::ak);
  const auto *v_bk = x.fields->find(FieldId::bk);
  if (!v_pe || !v_ak || !v_bk)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "REMAP_Q requires pe, ak, bk");

  // q has no FieldRegistry identity (see fv3_remap_q.cuh's header comment):
  // its device buffer is threaded through DispatchParams::remap_q_buffer,
  // populated by fv3_gpu_remap_q immediately before this call.
  if (!x.dispatch_params || !x.dispatch_params->remap_q_buffer)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "DispatchParams::remap_q_buffer not set: fv3_gpu_remap_q "
                "must populate it before dispatching REMAP_Q");

  // Prompt 86B/87: production tracer count/kord schedule, stashed by
  // fv3_gpu_remap_q alongside remap_q_buffer (see DispatchParams comment).
  // Only the validated nq=2, kord=[9,9] (sphum, cld_amt) schedule is
  // accepted here -- anything else fails closed rather than guessing a
  // schedule, the same class of stale-selector error Prompt 85's original
  // kord==9 copy-paste bug was (do not repeat it here). Prompt 87 corrected
  // this from the originally assumed kord=[8,9]: FV_StateMod.F90:563
  // unconditionally forces flagstruct%kord_tr=9, so sphum's effective kord
  // is 9 in every GEOS-integrated runtime configuration, not the bare
  // fv_arrays.F90:333 default of 8.
  const int32_t nq = x.dispatch_params->remap_q_nq;
  const int32_t kord1 = x.dispatch_params->remap_q_kord1;
  const int32_t kord2 = x.dispatch_params->remap_q_kord2;
  // Prompt 88B: flagstruct%fill, one global value applied identically to
  // both tracers (see fv3_remap_q.cuh header comment) -- not part of the
  // nq/kord schedule check below, since both fill=0 and fill=1 are valid.
  const bool fill = x.dispatch_params->remap_q_fill != 0;
  if (nq != 2 || kord1 != 9 || kord2 != 9)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "REMAP_Q requires nq=2, kord=[9,9] (sphum, cld_amt) -- "
                "DispatchParams::remap_q_nq/kord1/kord2 did not match");

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;

  RemapQArgs a{};
  a.pe = riem_c_view3(*v_pe);
  a.ak = ptdz_cview1(*v_ak);
  a.bk = ptdz_cview1(*v_bk);

  a.is = b.is; a.ie = b.ie; a.js = b.js; a.je = b.je;
  a.isd = b.is - static_cast<int64_t>(b.halo_width);
  a.ied = b.ie + static_cast<int64_t>(b.halo_width);
  a.jsd = b.js - static_cast<int64_t>(b.halo_width);
  a.jed = b.je + static_cast<int64_t>(b.halo_width);
  a.km = static_cast<int32_t>(b.npz);

  a.ptop = fv_real(cfg.ptop);

  // Prefer the persistent WP-E vertical workspace (same slice PT/DELZ and
  // REMAP_UV already use); fall back to per-thread local scratch only when
  // it cannot hold this column count. Safe to share: this launcher requires
  // StreamId::compute (checked above), same as every other consumer of
  // WorkspaceId::vertical, so GPU stream ordering guarantees no concurrent
  // use of the same scratch region. Both tracer launches below reuse this
  // same slice sequentially -- stream-ordered on StreamId::compute, so the
  // second launch cannot observe the first's scratch mid-write (Stage 4:
  // no separate per-tracer workspace).
  {
    const uint64_t need_per_column = remap_q_scalars_per_column(a.km);
    const uint64_t columns = remap_q_columns(a.is, a.ie, a.js, a.je);
    const auto *slice = workspace_slice(*x.workspace, WorkspaceId::vertical);
    if (slice && slice->data &&
        slice->bytes >= columns * need_per_column * sizeof(fv_real)) {
      a.workspace = static_cast<fv_real *>(slice->data);
      a.workspace_scalars_per_column = need_per_column;
      a.workspace_bytes = slice->bytes;
    }
  }

  // Tracer-plane stride: the full isd:ied x jsd:jed x km block per tracer
  // (remap_q_buffer_elements), matching Fortran's q(isd:ied,jsd:jed,km,nq)
  // layout where nq is the outermost/slowest dimension -- confirmed Prompt
  // 86B Stage 0 -- so tracer 2's plane begins exactly one stride past
  // tracer 1's, within the SAME contiguous buffer fv3_gpu_remap_q already
  // sized to nq=2 and moved with its one H2D/one D2H (Stage 2/3; this
  // launcher performs no transfer of its own).
  const uint64_t stride = remap_q_buffer_elements(a.isd, a.ied, a.jsd, a.jed, a.km);
  fv_real *const base_q = x.dispatch_params->remap_q_buffer;

  a.q = base_q;
  a.kord = kord1;
  a.fill = fill;
  int rc = launch_remap_q(a, x.submission.stream, error);
  if (rc != FV3_GPU_SUCCESS) return rc;

  a.q = base_q + static_cast<int64_t>(stride);
  a.kord = kord2;
  a.fill = fill;
  return launch_remap_q(a, x.submission.stream, error);
}

}  // namespace fv3::gpu::vertical
