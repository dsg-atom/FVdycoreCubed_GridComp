// Prompt 83 Stage 4 host microbench: OLD (pre-fix, scalar cast-copy loop,
// no __restrict__) vs NEW (if-constexpr memcpy fast path for T==fv_real)
// pack_field/unpack_field, at C512 production-sized packed-buffer geometry
// (fv3_tracer_transport_bridge.cu's own offset/size formulas, is=1..512,
// isd=-2..515, ng=3, matching a single-tile C512 (k,iq) slice call).
// Host-only, no CUDA required. Not part of the production build; deleted
// after this task's measurement is recorded (see p83 doc for the numbers).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <type_traits>
#include <vector>

using fv_real = double;

// OLD: byte-for-byte the pre-Prompt-83 implementation.
template <class T>
void pack_field_old(fv_real *dst, const T *src, size_t n) noexcept {
  for (size_t i = 0; i < n; ++i) dst[i] = static_cast<fv_real>(src[i]);
}
template <class T>
void unpack_field_old(T *dst, const fv_real *src, size_t n) noexcept {
  for (size_t i = 0; i < n; ++i) dst[i] = static_cast<T>(src[i]);
}

// NEW: Prompt 83 fix.
template <class T>
void pack_field_new(fv_real *__restrict__ dst, const T *__restrict__ src, size_t n) noexcept {
  if constexpr (std::is_same_v<T, fv_real>) {
    std::memcpy(dst, src, n * sizeof(fv_real));
  } else {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<fv_real>(src[i]);
  }
}
template <class T>
void unpack_field_new(T *__restrict__ dst, const fv_real *__restrict__ src, size_t n) noexcept {
  if constexpr (std::is_same_v<T, fv_real>) {
    std::memcpy(dst, src, n * sizeof(fv_real));
  } else {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<T>(src[i]);
  }
}

int main() {
  // C512 single-tile geometry, exactly as fv3_gpu_fv_tp_2d_hord8 computes it.
  const int64_t is = 1, ie = 512, js = 1, je = 512;
  const int64_t ng = 3;
  const int64_t isd = is - ng, ied = ie + ng, jsd = js - ng, jed = je + ng;
  const int64_t nx_d = ied - isd + 1, ny_d = jed - jsd + 1;
  const int64_t nx_i = ie - is + 1, ny_j = je - js + 1;
  const int64_t nx_xedge = nx_i + 1, ny_yedge = ny_j + 1;

  const size_t n_2d = (size_t)nx_d * ny_d;
  const size_t n_xedge_full = (size_t)nx_xedge * ny_d;
  const size_t n_full_yedge = (size_t)nx_d * ny_yedge;
  const size_t n_rax = (size_t)nx_i * ny_d;
  const size_t n_ray = (size_t)nx_d * ny_j;
  const size_t n_fx = (size_t)nx_xedge * ny_j;
  const size_t n_fy = (size_t)nx_i * ny_yedge;

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
  const size_t out_total = n_2d + n_fx + n_fy;

  std::printf("[P83_MICROBENCH] geometry: nx_d=%ld ny_d=%ld total_elems=%zu "
              "total_bytes=%zu out_elems=%zu out_bytes=%zu\n",
              (long)nx_d, (long)ny_d, total, total * sizeof(fv_real),
              out_total, out_total * sizeof(fv_real));

  std::vector<double> src_q(n_2d, 1.0), src_area(n_2d, 2.0), src_dxa(n_2d, 3.0),
      src_dya(n_2d, 4.0), src_crx(n_xedge_full, 5.0), src_xfx(n_xedge_full, 6.0),
      src_cry(n_full_yedge, 7.0), src_yfx(n_full_yedge, 8.0), src_rax(n_rax, 9.0),
      src_ray(n_ray, 10.0), src_mfx(n_fx, 11.0), src_mfy(n_fy, 12.0);
  std::vector<fv_real> packed(total, 0.0);
  std::vector<double> q_out(n_2d, 0.0), fx_out(n_fx, 0.0), fy_out(n_fy, 0.0);

  auto do_pack_old = [&]() {
    pack_field_old(packed.data() + off_q, src_q.data(), n_2d);
    pack_field_old(packed.data() + off_area, src_area.data(), n_2d);
    pack_field_old(packed.data() + off_dxa, src_dxa.data(), n_2d);
    pack_field_old(packed.data() + off_dya, src_dya.data(), n_2d);
    pack_field_old(packed.data() + off_crx, src_crx.data(), n_xedge_full);
    pack_field_old(packed.data() + off_xfx, src_xfx.data(), n_xedge_full);
    pack_field_old(packed.data() + off_cry, src_cry.data(), n_full_yedge);
    pack_field_old(packed.data() + off_yfx, src_yfx.data(), n_full_yedge);
    pack_field_old(packed.data() + off_rax, src_rax.data(), n_rax);
    pack_field_old(packed.data() + off_ray, src_ray.data(), n_ray);
    pack_field_old(packed.data() + off_mfx, src_mfx.data(), n_fx);
    pack_field_old(packed.data() + off_mfy, src_mfy.data(), n_fy);
  };
  auto do_pack_new = [&]() {
    pack_field_new(packed.data() + off_q, src_q.data(), n_2d);
    pack_field_new(packed.data() + off_area, src_area.data(), n_2d);
    pack_field_new(packed.data() + off_dxa, src_dxa.data(), n_2d);
    pack_field_new(packed.data() + off_dya, src_dya.data(), n_2d);
    pack_field_new(packed.data() + off_crx, src_crx.data(), n_xedge_full);
    pack_field_new(packed.data() + off_xfx, src_xfx.data(), n_xedge_full);
    pack_field_new(packed.data() + off_cry, src_cry.data(), n_full_yedge);
    pack_field_new(packed.data() + off_yfx, src_yfx.data(), n_full_yedge);
    pack_field_new(packed.data() + off_rax, src_rax.data(), n_rax);
    pack_field_new(packed.data() + off_ray, src_ray.data(), n_ray);
    pack_field_new(packed.data() + off_mfx, src_mfx.data(), n_fx);
    pack_field_new(packed.data() + off_mfy, src_mfy.data(), n_fy);
  };
  auto do_unpack_old = [&]() {
    unpack_field_old(q_out.data(), packed.data() + off_q, n_2d);
    unpack_field_old(fx_out.data(), packed.data() + off_fx, n_fx);
    unpack_field_old(fy_out.data(), packed.data() + off_fy, n_fy);
  };
  auto do_unpack_new = [&]() {
    unpack_field_new(q_out.data(), packed.data() + off_q, n_2d);
    unpack_field_new(fx_out.data(), packed.data() + off_fx, n_fx);
    unpack_field_new(fy_out.data(), packed.data() + off_fy, n_fy);
  };

  const int WARMUP = 20, REPS = 500;
  for (int i = 0; i < WARMUP; ++i) { do_pack_old(); do_unpack_old(); do_pack_new(); do_unpack_new(); }

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < REPS; ++i) do_pack_old();
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < REPS; ++i) do_unpack_old();
  auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < REPS; ++i) do_pack_new();
  auto t3 = std::chrono::steady_clock::now();
  for (int i = 0; i < REPS; ++i) do_unpack_new();
  auto t4 = std::chrono::steady_clock::now();

  double pack_old_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;
  double unpack_old_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / REPS;
  double pack_new_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / REPS;
  double unpack_new_ms = std::chrono::duration<double, std::milli>(t4 - t3).count() / REPS;

  std::printf("[P83_MICROBENCH] PACK_OLD_MS=%.6f UNPACK_OLD_MS=%.6f "
              "PACK_NEW_MS=%.6f UNPACK_NEW_MS=%.6f\n",
              pack_old_ms, unpack_old_ms, pack_new_ms, unpack_new_ms);
  std::printf("[P83_MICROBENCH] PACK_SPEEDUP=%.3fx UNPACK_SPEEDUP=%.3fx COMBINED_SPEEDUP=%.3fx\n",
              pack_old_ms / pack_new_ms, unpack_old_ms / unpack_new_ms,
              (pack_old_ms + unpack_old_ms) / (pack_new_ms + unpack_new_ms));

  // Correctness self-check: old vs new must produce bit-identical output.
  std::vector<fv_real> packed_ref(total, 0.0);
  {
    auto save = packed; packed.assign(total, 0.0);
    do_pack_old(); packed_ref = packed;
    packed.assign(total, 0.0);
    do_pack_new();
    bool match = (packed == packed_ref);
    std::printf("[P83_MICROBENCH] PACK_BITWISE_MATCH=%s\n", match ? "PASS" : "FAIL");
    packed = save;
  }
  return 0;
}
