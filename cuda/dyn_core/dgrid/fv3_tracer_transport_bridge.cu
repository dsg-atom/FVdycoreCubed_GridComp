// Prompt 77: production dispatch bridge for tp_core.F90::fv_tp_2d at
// fv_tracer2d.F90::tracer_2d_1L's nsplt==1 tracer-transport call site.
//
// This file implements exactly one public entry point,
// fv3_gpu_fv_tp_2d_hord8 (declared in include/fv3_gpu_api.h), which marshals
// one (k, iq) tracer-transport slice from plain host array pointers into
// device memory, calls the already-validated
// fv3::gpu::dgrid::launch_fv_tp_2d_cuda (wp_b_cuda.cuh/wp_b_cuda.cu -- NOT
// modified by this file), copies fx/fy back, and returns. It duplicates no
// numerics: every PPM/limiter/flux formula lives exclusively in
// launch_fv_tp_2d_cuda, unchanged.
//
// Deliberately NOT part of fv3_gpu_context/FieldRegistry: tracer temporaries
// (q, crx/cry, xfx/yfx, mfx/mfy, ra_x/ra_y) have no FieldRegistry identity
// (see p77 Stage 0/2 notes), so this entry owns its own small, persistent,
// per-host-thread device workspace instead. thread_local is required, not
// just convenient: PeerHaloTransport's default in-process model runs one
// host thread per tile (six threads, one process), so a process-wide static
// workspace would let two tiles race on the same device buffers.
//
// Prompt 79: the ABI/signature below is UNCHANGED (no Fortran-side edit
// required). Only the internal transfer strategy changed: the original
// implementation issued 12 separate blocking cudaMemcpy H2D calls (one per
// input array) plus 3 separate blocking cudaMemcpy D2H calls (q, fx, fy) --
// 15 synchronous transfers per (k, iq) slice call, ~300/tracer_2d_1L call at
// production npz/nq (see p78 attribution doc). This version packs all 12
// inputs plus the 3 host-visible outputs into ONE contiguous pinned host
// buffer and ONE contiguous device buffer, so the whole slice moves in
// exactly one cudaMemcpyAsync H2D and one cudaMemcpyAsync D2H (q, fx, fy
// share a single contiguous prefix, enabling one D2H to cover all three),
// synchronized once at the end -- see p79 doc §2-5 for the exact layout and
// byte accounting. No numerical semantics changed: launch_fv_tp_2d_cuda
// (the only place PPM/limiter/flux formulas live) is called identically to
// before, on views into the same packed device buffer instead of into 14
// separate per-field buffers.

#include "fv3_gpu_api.h"
#include "dgrid/wp_b_cuda.cuh"
#include "fv3_gpu_types.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace fv3::gpu::dgrid {
namespace {

using fv3::gpu::fv_real;

void clear_error_local(fv3_gpu_error *e) noexcept {
  if (!e) return;
  std::memset(e, 0, sizeof(*e));
  e->abi_version = FV3_GPU_ABI_VERSION;
  e->struct_bytes = sizeof(*e);
}

int fail(fv3_gpu_error *e, fv3_gpu_status code, const char *api,
         const char *message) noexcept {
  clear_error_local(e);
  if (e) {
    e->code = code;
    std::snprintf(e->api, sizeof(e->api), "%s", api);
    std::snprintf(e->message, sizeof(e->message), "%s", message);
  }
  return code;
}

int cuda_fail(fv3_gpu_error *e, cudaError_t st, const char *api) noexcept {
  return fail(e, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(st));
}

bool header_ok(uint32_t version, uint32_t bytes, uint32_t need) noexcept {
  return version == FV3_GPU_ABI_VERSION && bytes == need;
}

// Stage 7 diagnostic only (env-gated, off by default, no effect on the
// transfer path itself): counts H2D/D2H cudaMemcpy calls issued by THIS
// bridge invocation (one (k, iq) slice), so the achieved count can be
// checked against the p79 target (<=3 each, ideally 1+1) without a
// profiler. Not a per-tracer_2d_1L aggregate -- see p79 doc §Stage 7 for
// why the Fortran k/iq loop's call count is out of scope here.
bool env_flag_enabled(const char *name) noexcept {
  const char *v = std::getenv(name);
  return v && v[0] != '\0' && v[0] != '0';
}

// Grow-only device buffer. ensure() reallocates only when the requested
// element count exceeds current capacity, so a steady-state run (fixed tile
// bounds across every (k, iq) call) allocates once and reuses thereafter --
// the Prompt 68 device-workspace-reuse precedent.
struct DeviceBuf {
  fv_real *ptr = nullptr;
  size_t capacity = 0;
  cudaError_t ensure(size_t n) noexcept {
    if (n <= capacity) return cudaSuccess;
    if (ptr) { cudaFree(ptr); ptr = nullptr; capacity = 0; }
    cudaError_t st = cudaMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(fv_real));
    if (st != cudaSuccess) return st;
    capacity = n;
    return cudaSuccess;
  }
};

// Grow-only pinned host buffer (cudaHostAlloc), same reuse discipline as
// DeviceBuf. Pinned (page-locked) memory is required for a genuine
// cudaMemcpyAsync H2D/D2H -- a copy staged through ordinary heap memory
// (std::vector) is silently downgraded to a synchronous copy by the CUDA
// driver. thread_local, same rationale as Workspace below.
struct PinnedBuf {
  fv_real *ptr = nullptr;
  size_t capacity = 0;
  ~PinnedBuf() noexcept { if (ptr) cudaFreeHost(ptr); }
  cudaError_t ensure(size_t n) noexcept {
    if (n <= capacity) return cudaSuccess;
    if (ptr) { cudaFreeHost(ptr); ptr = nullptr; capacity = 0; }
    cudaError_t st = cudaHostAlloc(reinterpret_cast<void **>(&ptr),
                                   n * sizeof(fv_real), cudaHostAllocDefault);
    if (st != cudaSuccess) return st;
    capacity = n;
    return cudaSuccess;
  }
};

// Prompt 79: one packed device buffer replaces the 14 separate per-field
// DeviceBufs used through Prompt 78. Layout (element offsets, computed
// per-call in fv3_gpu_fv_tp_2d_hord8 from the caller's own bounds -- see
// p79 doc §2 for the exact byte accounting):
//   [0)                q        (isd..ied x jsd..jed)   -- in AND out
//   [n_2d)             fx       (is..ie+1 x js..je)      -- out only
//   [n_2d+n_fx)        fy       (is..ie x js..je+1)      -- out only
//   [n_2d+n_fx+n_fy)   area, dxa, dya, crx, xfx, cry, yfx, ra_x, ra_y,
//                      mfx, mfy                          -- in only
// q/fx/fy occupy one contiguous prefix so a single cudaMemcpyAsync D2H
// covers all three host-visible outputs; the remaining fields never need
// to come back to the host at all (they are pure kernel inputs), so they
// are excluded from the D2H's byte count even though they share the one
// H2D that populates the whole buffer.
struct Workspace {
  DeviceBuf packed;                     // all 14 fields, one allocation
  DeviceBuf q_i;                        // isd..ied x js..je  (device-only scratch)
  DeviceBuf q_j;                        // is..ie x jsd..jed  (device-only scratch)
  DeviceBuf fx2;                        // is..ie+1 x jsd..jed (device-only scratch)
  DeviceBuf fy2;                        // isd..ied x js..je+1 (device-only scratch)
};

thread_local Workspace g_ws;
thread_local PinnedBuf g_pack;  // host-side mirror of Workspace::packed

// Prompt 83: move one field into (or out of) the packed pinned staging
// buffer. Every field this bridge marshals is a contiguous, same-
// representation region when T == fv_real (verified in p83 doc Stage 0/1:
// on the R8 production build, T is `double` -- the ABI's fixed element
// type -- and fv_real is also `double`, so pack_field/unpack_field's
// per-element cast was always an identity conversion over contiguous
// memory). In that case this is a plain bulk move: std::memcpy, no
// per-element loop at all. That fast path is selected at compile time via
// `if constexpr`, so it costs nothing when it does not apply.
//
// The scalar loop is retained, byte-for-byte as introduced in Prompt 79,
// for the one case where T != fv_real -- the R4 build (fv_real == float)
// receiving this bridge's fixed `double` ABI arguments, where a genuine
// narrowing/widening conversion is required and memcpy would be unsafe.
// R4 is explicitly out of this task's scope (Do-NOT: "touch R4") and its
// behavior is unchanged by this edit. __restrict__ is added so the
// compiler can assume dst/src do not alias (they never did in practice --
// dst is always this bridge's private pinned/host buffer or the caller's
// own output array, never aliased with src -- __restrict__ merely makes
// that existing invariant explicit and vectorizable).
template <class T>
void pack_field(fv_real *__restrict__ dst, const T *__restrict__ src, size_t n) noexcept {
  if constexpr (std::is_same_v<T, fv_real>) {
    std::memcpy(dst, src, n * sizeof(fv_real));
  } else {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<fv_real>(src[i]);
  }
}

template <class T>
void unpack_field(T *__restrict__ dst, const fv_real *__restrict__ src, size_t n) noexcept {
  if constexpr (std::is_same_v<T, fv_real>) {
    std::memcpy(dst, src, n * sizeof(fv_real));
  } else {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<T>(src[i]);
  }
}

template <class T>
operators::View3<T> view2d_as3(T *data, int64_t lo_i, int64_t lo_j, int64_t ni) noexcept {
  return operators::View3<T>{data, lo_i, lo_j, 0, 1, ni, 0};
}

template <class T>
operators::View2<T> view2d(T *data, int64_t lo_i, int64_t lo_j, int64_t ni) noexcept {
  return operators::View2<T>{data, lo_i, lo_j, 1, ni};
}

}  // namespace
}  // namespace fv3::gpu::dgrid

extern "C" int fv3_gpu_fv_tp_2d_hord8(const fv3_gpu_tracer_transport_bounds *b,
                                       double *q, const double *crx,
                                       const double *cry, const double *xfx,
                                       const double *yfx, const double *mfx,
                                       const double *mfy, const double *ra_x,
                                       const double *ra_y, const double *area,
                                       const double *dxa, const double *dya,
                                       double *fx_out, double *fy_out,
                                       fv3_gpu_error *err) {
  using namespace fv3::gpu::dgrid;
  using fv3::gpu::fv_real;
  clear_error_local(err);

  if (!header_ok(b ? b->abi_version : 0, b ? b->struct_bytes : 0,
                 sizeof(fv3_gpu_tracer_transport_bounds))) {
    return fail(err, FV3_GPU_INVALID_ARGUMENT, __func__,
                "fv3_gpu_tracer_transport_bounds header mismatch");
  }
  if (!q || !crx || !cry || !xfx || !yfx || !mfx || !mfy || !ra_x || !ra_y ||
      !area || !dxa || !dya || !fx_out || !fy_out) {
    return fail(err, FV3_GPU_INVALID_ARGUMENT, __func__, "null array pointer");
  }
  if (b->hord != 6 && b->hord != 8) {
    return fail(err, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "hord must be 6 or 8");
  }
  if (b->ie < b->is || b->je < b->js || b->ied < b->isd || b->jed < b->jsd) {
    return fail(err, FV3_GPU_INVALID_ARGUMENT, __func__, "degenerate bounds");
  }

  const int64_t is = b->is, ie = b->ie, js = b->js, je = b->je;
  const int64_t isd = b->isd, ied = b->ied, jsd = b->jsd, jed = b->jed;
  const int64_t nx_d = ied - isd + 1, ny_d = jed - jsd + 1;
  const int64_t nx_i = ie - is + 1, ny_j = je - js + 1;
  const int64_t nx_xedge = nx_i + 1;  // is..ie+1
  const int64_t ny_yedge = ny_j + 1;  // js..je+1

  Workspace &w = g_ws;
  cudaError_t st = cudaSuccess;
  const size_t n_2d = static_cast<size_t>(nx_d) * ny_d;               // q/area/dxa/dya
  const size_t n_xedge_full = static_cast<size_t>(nx_xedge) * ny_d;   // crx/xfx/fx2
  const size_t n_full_yedge = static_cast<size_t>(nx_d) * ny_yedge;   // cry/yfx/fy2
  const size_t n_rax = static_cast<size_t>(nx_i) * ny_d;              // ra_x/q_j
  const size_t n_ray = static_cast<size_t>(nx_d) * ny_j;              // ra_y/q_i
  const size_t n_fx = static_cast<size_t>(nx_xedge) * ny_j;           // mfx/fx
  const size_t n_fy = static_cast<size_t>(nx_i) * ny_yedge;           // mfy/fy

  // Prompt 79 packed layout -- see file header and p79 doc §2 for the
  // rationale. q/fx/fy form one contiguous host-visible-output prefix;
  // everything after it is pure kernel input, never copied back.
  const size_t off_q = 0;
  const size_t off_fx = off_q + n_2d;
  const size_t off_fy = off_fx + n_fx;
  const size_t off_area = off_fy + n_fy;
  const size_t off_dxa = off_area + n_2d;
  const size_t off_dya = off_dxa + n_2d;
  const size_t off_crx = off_dya + n_2d;
  const size_t off_xfx = off_crx + n_xedge_full;
  const size_t off_cry = off_xfx + n_xedge_full;
  const size_t off_yfx = off_cry + n_full_yedge;
  const size_t off_rax = off_yfx + n_full_yedge;
  const size_t off_ray = off_rax + n_rax;
  const size_t off_mfx = off_ray + n_ray;
  const size_t off_mfy = off_mfx + n_fx;
  const size_t total = off_mfy + n_fy;
  const size_t out_bytes = (n_2d + n_fx + n_fy) * sizeof(fv_real);

  if (st == cudaSuccess) st = w.packed.ensure(total);
  if (st == cudaSuccess) st = g_pack.ensure(total);
  if (st == cudaSuccess) st = w.fx2.ensure(n_xedge_full);
  if (st == cudaSuccess) st = w.fy2.ensure(n_full_yedge);
  if (st == cudaSuccess) st = w.q_i.ensure(n_ray);
  if (st == cudaSuccess) st = w.q_j.ensure(n_rax);
  if (st != cudaSuccess) return cuda_fail(err, st, __func__);

  // Pack all 12 inputs (q included, since it is mutated in place by
  // copy_corners) into the one pinned staging buffer, then move the
  // whole thing in exactly one cudaMemcpyAsync H2D.
  pack_field(g_pack.ptr + off_q, q, n_2d);
  pack_field(g_pack.ptr + off_area, area, n_2d);
  pack_field(g_pack.ptr + off_dxa, dxa, n_2d);
  pack_field(g_pack.ptr + off_dya, dya, n_2d);
  pack_field(g_pack.ptr + off_crx, crx, n_xedge_full);
  pack_field(g_pack.ptr + off_xfx, xfx, n_xedge_full);
  pack_field(g_pack.ptr + off_cry, cry, n_full_yedge);
  pack_field(g_pack.ptr + off_yfx, yfx, n_full_yedge);
  pack_field(g_pack.ptr + off_rax, ra_x, n_rax);
  pack_field(g_pack.ptr + off_ray, ra_y, n_ray);
  pack_field(g_pack.ptr + off_mfx, mfx, n_fx);
  pack_field(g_pack.ptr + off_mfy, mfy, n_fy);

  st = cudaMemcpyAsync(w.packed.ptr, g_pack.ptr, total * sizeof(fv_real),
                        cudaMemcpyHostToDevice, /*stream=*/nullptr);
  if (st != cudaSuccess) return cuda_fail(err, st, __func__);
  int h2d_count = 1;

  TransportBounds tb{};
  tb.is = is; tb.ie = ie; tb.js = js; tb.je = je;
  tb.npz = 1; tb.nk = 1;
  tb.halo_width = b->isd < is ? (is - b->isd) : 0;
  tb.ng = 3;
  tb.npx = b->npx; tb.npy = b->npy;
  tb.cubed_sphere = b->cubed_sphere;
  tb.nested = b->nested;
  tb.west_edge = b->west_edge; tb.east_edge = b->east_edge;
  tb.south_edge = b->south_edge; tb.north_edge = b->north_edge;
  tb.sw_corner = b->sw_corner; tb.se_corner = b->se_corner;
  tb.ne_corner = b->ne_corner; tb.nw_corner = b->nw_corner;

  LevelFilter lf{};  // empty -> no deln_flux, matching launch_fv_tp_2d_cuda's own scope

  // Views all point into the one packed device buffer, offset per the
  // Prompt 79 layout -- no per-field DeviceBuf pointers remain.
  auto q_view = view2d_as3<fv_real>(w.packed.ptr + off_q, isd, jsd, nx_d);
  auto crx_view = view2d_as3<const fv_real>(w.packed.ptr + off_crx, is, jsd, nx_xedge);
  auto cry_view = view2d_as3<const fv_real>(w.packed.ptr + off_cry, isd, js, nx_d);
  auto xfx_view = view2d_as3<const fv_real>(w.packed.ptr + off_xfx, is, jsd, nx_xedge);
  auto yfx_view = view2d_as3<const fv_real>(w.packed.ptr + off_yfx, isd, js, nx_d);
  auto wx_view = view2d_as3<const fv_real>(w.packed.ptr + off_mfx, is, js, nx_xedge);
  auto wy_view = view2d_as3<const fv_real>(w.packed.ptr + off_mfy, is, js, nx_i);
  auto area_view = view2d<const fv_real>(w.packed.ptr + off_area, isd, jsd, nx_d);
  auto dxa_view = view2d<const fv_real>(w.packed.ptr + off_dxa, isd, jsd, nx_d);
  auto dya_view = view2d<const fv_real>(w.packed.ptr + off_dya, isd, jsd, nx_d);
  auto rax_view = view2d_as3<const fv_real>(w.packed.ptr + off_rax, is, jsd, nx_i);
  auto ray_view = view2d_as3<const fv_real>(w.packed.ptr + off_ray, isd, js, nx_d);
  auto qi_view = view2d_as3<fv_real>(w.q_i.ptr, isd, js, nx_d);
  auto qj_view = view2d_as3<fv_real>(w.q_j.ptr, is, jsd, nx_i);
  auto fx_view = view2d_as3<fv_real>(w.packed.ptr + off_fx, is, js, nx_xedge);
  auto fy_view = view2d_as3<fv_real>(w.packed.ptr + off_fy, is, js, nx_i);
  auto fx2_view = view2d_as3<fv_real>(w.fx2.ptr, is, jsd, nx_xedge);
  auto fy2_view = view2d_as3<fv_real>(w.fy2.ptr, isd, js, nx_d);

  fv3_gpu_status rc = launch_fv_tp_2d_cuda(
      tb, b->hord, lf, q_view, crx_view, cry_view, xfx_view, yfx_view,
      wx_view, wy_view, area_view, dxa_view, dya_view, rax_view, ray_view,
      qi_view, qj_view, fx_view, fy_view, fx2_view, fy2_view, /*stream=*/nullptr);
  if (rc != FV3_GPU_SUCCESS) {
    return fail(err, rc, __func__, "launch_fv_tp_2d_cuda rejected this configuration");
  }

  // Single D2H covering exactly the q+fx+fy prefix (they are contiguous
  // by construction, off_q==0 through off_fy+n_fy == out_bytes), then the
  // one required synchronization at this API's ownership boundary before
  // handing results back to the Fortran caller.
  st = cudaMemcpyAsync(g_pack.ptr, w.packed.ptr, out_bytes,
                        cudaMemcpyDeviceToHost, /*stream=*/nullptr);
  if (st != cudaSuccess) return cuda_fail(err, st, __func__);
  int d2h_count = 1;

  st = cudaStreamSynchronize(nullptr);
  if (st != cudaSuccess) return cuda_fail(err, st, __func__);

  unpack_field(q, g_pack.ptr + off_q, n_2d);
  unpack_field(fx_out, g_pack.ptr + off_fx, n_fx);
  unpack_field(fy_out, g_pack.ptr + off_fy, n_fy);

  if (env_flag_enabled("RUN_CUDA_FV3_TRACER_TRANSFER_COUNT_DUMP")) {
    std::fprintf(stderr,
                 "[P79_TRANSFER_COUNT] h2d=%d d2h=%d total_bytes=%zu out_bytes=%zu\n",
                 h2d_count, d2h_count, total * sizeof(fv_real), out_bytes);
  }

  return FV3_GPU_SUCCESS;
}
