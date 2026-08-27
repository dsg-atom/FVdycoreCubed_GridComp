//===========================================================================
// Milestone 3A.11T §4 -- permanent layout-sensitive pe/peln test.
//
// pe and peln are FV3's exception to the (i,j,k) convention. EVERY authoritative
// declaration site orders them (i, k, j):
//
//   nh_core.F90:87-88     pe(is-1:ie+1,km+1,js-1:je+1)  peln(is:ie,km+1,js:je)
//   dyn_core.F90:202-203  same, with bd% bounds
//   fv_arrays.F90:1472/1474  the matching allocations
//
// with body-level proof at geopk's `pe(i,1,j) = ptop` (dyn_core.F90:2191),
// `pe(i,k,j) = p1d(i)` (:2209) and `psx(i,j) = pe(i,npz+1,j)`
// (fv_dynamics.F90:611). ~40 sites, zero exceptions. Their (i,j,k) neighbours
// pk, gz, zh, pkc and pk3 are genuinely (i,j,k) -- nh_core.F90:91 declares
// pk(is:ie,js:je,km+1) -- so this is a per-field property, not a global one.
//
// -------------------------------------------------------------------------
// WHY A TEST IS NEEDED AT ALL
// -------------------------------------------------------------------------
//
// Nothing in the CUDA stack validates axis SEMANTICS. Three facts compose:
//
//   1. make_device_view (runtime/fv3_gpu_contract.cpp:76-88) checks field id,
//      rank, stagger, access and element width, then copies lower/extent/stride
//      verbatim. It cannot check axis order, and does not try.
//   2. The FieldSpec table gives pe and peln FV3_GPU_STAGGER_INTERFACE -- the
//      same stagger as pk, gz, zh, pkc and pk3, which are (i,j,k). So the
//      stagger cannot distinguish them either.
//   3. Every kernel-side view (View3::at, DeviceView<T>::offset) is purely
//      POSITIONAL: it computes (a-l0)*s0 + (b-l1)*s1 + (c-l2)*s2 and attaches no
//      meaning to any slot.
//
// Axis order is therefore expressible ONLY through the descriptor strides, and
// is enforced ONLY by agreement between whoever fills the descriptor and
// whoever writes the .at() argument list. That agreement is exactly what
// 3A.11T found broken in both directions at once -- the harnesses registered
// pe/peln as (i,j,k) AND riem_solver3_kernel's writeback used .at(i,j,k) -- so
// the two cancelled and pe/peln round-tripped self-consistently. A latent
// contract defect that no existing test could see. This test sees it.
//
// -------------------------------------------------------------------------
// HOW IT AVOIDS BEING CIRCULAR
// -------------------------------------------------------------------------
//
// Every element carries its own provenance, encoded losslessly in its value:
//
//     value(i, j, k) = (i + 8) + 1000*(j + 8) + 1000000*k
//
// i and j run -2..11 and k runs 0..10, so i+8 and j+8 land in 6..19, well below
// the 1000 spacing -- no permutation of the three can alias. The maximum is
// 10019019 < 2^24, so the encoding is exact in float32 and float64 alike and one
// binary proves R4 and R8. Reads are DECODED back to a triple and compared to
// the triple asked for; the test never recomputes an index and compares it to
// the implementation's index.
//
// -------------------------------------------------------------------------
// WHY IT MUST FAIL UNDER A PERMUTATION -- exactly, not probabilistically
// -------------------------------------------------------------------------
//
// §4 requires that an i/j/k permutation cannot pass. For the authoritative
// (i,k,j) descriptor the buffer holds encode(i,j,k) at memory offset
//
//     O(i,k,j) = (i+2) + k*ni + (j+2)*ni*npz1
//
// A permuted read .at(i,j,k) computes O' = (i+2) + (j+2)*ni + k*ni*npz1, which
// is O(i', k', j') for i'=i, k'=j+2, j'=k-2. That cell holds
// encode(i, k-2, j+2), which equals encode(i, j, k) if and only if
//
//     k - 2 == j  and  j + 2 == k,     i.e.  k == j + 2.
//
// So the negative control is a proof, not a sampling argument, for every probe
// with k != j+2 (and j+2 <= npz, so that O' decomposes inside the k range).
// Part C asserts precisely that, and refuses to pass unless enough probes
// qualify.
//
// -------------------------------------------------------------------------
// WHAT THIS TEST DOES AND DOES NOT COVER
// -------------------------------------------------------------------------
//
// COVERS the whole production descriptor chain: fv3_gpu_field_desc ->
// FieldRegistry::insert -> make_device_view -> DeviceViewBase.lower/stride ->
// a positional device view -> a real device read and a real device write.
// Part A additionally asserts that make_device_view copied the stride and lower
// slots verbatim, which is the property the rest of the argument rests on.
//
// DOES NOT call production kernels. The positional view below is a local
// transcription of the six-line View3/CView3 that every production kernel file
// declares in its own anonymous namespace (e.g. fv3_vertical_kernels.cu:18-19),
// because those are file-local and unreachable from a test. The transcription is
// deliberate in a second way: a test that called the production helper would
// move with it under a fault and could not detect one.
//
// DOES NOT establish numerical equivalence to authoritative Fortran. Numerical
// validation against authoritative Fortran savepoints: NOT RUN.
//===========================================================================
#include "fv3_gpu_internal.hpp"
#include "fv3_gpu_api.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace fv3::gpu;

namespace {

// A domain with a genuine 3-cell halo, so the j halo probes §4 asks for are real
// halo cells and not merely interior ones.
constexpr int64_t kIs = 1, kIe = 8, kJs = 1, kJe = 8, kHalo = 3, kNpz = 10;
constexpr int64_t kNi = kIe - kIs + 1 + 2 * kHalo;   // 14
constexpr int64_t kNj = kJe - kJs + 1 + 2 * kHalo;   // 14
constexpr int64_t kLoI = kIs - kHalo;                // -2
constexpr int64_t kLoJ = kJs - kHalo;                // -2
constexpr int64_t kNpz1 = kNpz + 1;                  // 11
// Both orders are permutations of the same box, so the allocation is identical.
constexpr size_t kSize = (size_t)(kNi * kNpz1 * kNj);

struct P3 { int64_t i, j, k; };

double encode(int64_t i, int64_t j, int64_t k) {
  return (double)(i + 8) + 1000.0 * (double)(j + 8) + 1000000.0 * (double)k;
}

// Decode a value back to its provenance. Returns false if it is not a valid
// encoding at all (NaN, negative, or not an integer), which is itself a failure.
bool decode(double v, int64_t &i, int64_t &j, int64_t &k) {
  if (!(v == v) || v < 0.0) return false;
  const double r = std::round(v);
  if (std::fabs(v - r) > 1e-6) return false;
  long long n = (long long)r;
  k = n / 1000000; n -= (long long)k * 1000000;
  j = n / 1000 - 8;  n -= (long long)(j + 8) * 1000;
  i = n - 8;
  return true;
}

// Local transcription of the production positional view -- see the header note.
struct RView {
  const fv_real *p; int64_t l0, l1, l2, s0, s1, s2;
  __device__ fv_real at(int64_t a, int64_t b, int64_t c) const {
    return p[(a - l0) * s0 + (b - l1) * s1 + (c - l2) * s2];
  }
};
struct WView {
  fv_real *p; int64_t l0, l1, l2, s0, s1, s2;
  __device__ fv_real &at(int64_t a, int64_t b, int64_t c) const {
    return p[(a - l0) * s0 + (b - l1) * s1 + (c - l2) * s2];
  }
};
RView rview(const DeviceViewBase &v) {
  return {static_cast<const fv_real *>(v.data), v.lower[0], v.lower[1],
          v.lower[2], v.stride[0], v.stride[1], v.stride[2]};
}
WView wview(const DeviceViewBase &v) {
  return {static_cast<fv_real *>(v.data), v.lower[0], v.lower[1], v.lower[2],
          v.stride[0], v.stride[1], v.stride[2]};
}

// order 0 = authoritative (i,k,j); order 1 = the (i,j,k) permutation.
__global__ void read_kernel(RView v, const P3 *pr, int n, int order,
                            fv_real *out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const P3 q = pr[t];
  out[t] = (order == 0) ? v.at(q.i, q.k, q.j) : v.at(q.i, q.j, q.k);
}

__global__ void write_kernel(WView v, const P3 *pr, int n, int order) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const P3 q = pr[t];
  const fv_real val = (fv_real)((double)(q.i + 8) + 1000.0 * (double)(q.j + 8) +
                                1000000.0 * (double)q.k);
  if (order == 0) v.at(q.i, q.k, q.j) = val; else v.at(q.i, q.j, q.k) = val;
}

// order 0 = (i,j,k), the ordinary order, for the pk control in Part D.
__global__ void read_ijk_kernel(RView v, const P3 *pr, int n, fv_real *out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const P3 q = pr[t];
  out[t] = v.at(q.i, q.j, q.k);
}

struct Probe { P3 p; const char *what; };

// §4's required coverage: interior, j halo (both sides, both depths), k=0,
// k=npz, and the C-grid vertical boundary ring is-1..ie+1 / js-1..je+1 that
// Riem_Solver_c and update_dz_c work over (nh_utils.F90:325-432, :70-71).
const Probe kProbes[] = {
    {{4, 5, 3},    "interior"},
    {{6, 2, 7},    "interior"},
    {{3, 6, 0},    "k = 0 (model top interface)"},
    {{5, 1, kNpz}, "k = npz (surface interface)"},
    {{2, 0, 5},    "j halo, js-1"},
    {{7, -1, 4},   "j halo, js-2"},
    {{1, 9, 6},    "j halo, je+1"},
    {{8, 10, 2},   "j halo, je+2"},
    {{0, 4, 8},    "i halo, is-1"},
    {{9, 3, 1},    "i halo, ie+1"},
    {{0, 0, 9},    "C-grid vertical corner (is-1, js-1)"},
    {{9, 8, kNpz}, "C-grid vertical ring (ie+1, je) at k = npz"},
    {{4, -1, kNpz},"j halo at k = npz"},
    {{-2, -2, 7},  "deepest halo corner (isd, jsd)"},
};
constexpr int kNProbes = (int)(sizeof(kProbes) / sizeof(kProbes[0]));

fv3_gpu_field_desc make_desc(FieldId id, void *base, const int64_t *lower,
                             const uint64_t *extent, const int64_t *stride) {
  fv3_gpu_field_desc d{};
  d.abi_version = FV3_GPU_ABI_VERSION;
  d.struct_bytes = sizeof(d);
  d.field_id = (uint32_t)id;
  d.base = base;
  d.allocation_bytes = kSize * sizeof(fv_real);
  d.element_bytes = (uint32_t)sizeof(fv_real);
  d.element_type = sizeof(fv_real) == 4 ? FV3_GPU_ELEMENT_F32 : FV3_GPU_ELEMENT_F64;
  d.rank = 3;
  d.memory_location = FV3_GPU_MEMORY_DEVICE;
  const auto *sp = field_spec(id);
  d.stagger = sp->stagger;
  d.access = sp->access;
  for (int n = 0; n < 3; ++n) {
    d.lower[n] = lower[n];
    d.extent[n] = extent[n];
    d.stride_elements[n] = stride[n];
    d.compute_lower[n] = lower[n];
    d.compute_upper[n] = lower[n] + (int64_t)extent[n] - 1;
  }
  d.alignment_bytes = 16;
  return d;
}

// Authoritative pe/peln: slot 0 = i, slot 1 = k, slot 2 = j.
const int64_t  kPeLower[3]  = {kLoI, 0, kLoJ};
const uint64_t kPeExtent[3] = {(uint64_t)kNi, (uint64_t)kNpz1, (uint64_t)kNj};
const int64_t  kPeStride[3] = {1, kNi, kNi * kNpz1};
// pk and the other interface fields: the ordinary (i, j, k).
const int64_t  kPkLower[3]  = {kLoI, kLoJ, 0};
const uint64_t kPkExtent[3] = {(uint64_t)kNi, (uint64_t)kNj, (uint64_t)kNpz1};
const int64_t  kPkStride[3] = {1, kNi, kNi * kNj};

size_t off_ikj(int64_t i, int64_t j, int64_t k) {
  return (size_t)((i - kLoI) + k * kNi + (j - kLoJ) * kNi * kNpz1);
}
size_t off_ijk(int64_t i, int64_t j, int64_t k) {
  return (size_t)((i - kLoI) + (j - kLoJ) * kNi + k * kNi * kNj);
}

struct Ctx {
  fv_real *d_buf = nullptr, *d_out = nullptr;
  P3 *d_probes = nullptr;
  std::vector<fv_real> h_buf;
  FieldRegistry reg;
  bool ok = true;

  bool setup() {
    h_buf.assign(kSize, (fv_real)-1);
    if (cudaMalloc(&d_buf, kSize * sizeof(fv_real)) != cudaSuccess) return false;
    if (cudaMalloc(&d_out, kNProbes * sizeof(fv_real)) != cudaSuccess) return false;
    if (cudaMalloc(&d_probes, kNProbes * sizeof(P3)) != cudaSuccess) return false;
    std::vector<P3> hp(kNProbes);
    for (int n = 0; n < kNProbes; ++n) hp[n] = kProbes[n].p;
    return cudaMemcpy(d_probes, hp.data(), kNProbes * sizeof(P3),
                      cudaMemcpyHostToDevice) == cudaSuccess;
  }
  ~Ctx() {
    if (d_buf) cudaFree(d_buf);
    if (d_out) cudaFree(d_out);
    if (d_probes) cudaFree(d_probes);
  }
};

// ------------------------------------------------------------------ Part A
// Register pe as (i,k,j) and prove make_device_view carried the descriptor
// through untouched. Everything below depends on this.
bool part_a(Ctx &c) {
  bool ok = true;
  fv3_gpu_error err{};
  auto d = make_desc(FieldId::pe, c.d_buf, kPeLower, kPeExtent, kPeStride);
  if (c.reg.insert(d, c.d_buf, &err) != 0) {
    printf("  registering pe failed: %s\n", err.message);
    return false;
  }
  const DeviceViewBase *v = c.reg.find(FieldId::pe);
  if (!v) { printf("  pe not found after insert\n"); return false; }
  for (int n = 0; n < 3; ++n) {
    if (v->lower[n] != kPeLower[n]) {
      printf("  FAIL lower[%d]: registered %lld, expected %lld\n", n,
             (long long)v->lower[n], (long long)kPeLower[n]);
      ok = false;
    }
    if (v->stride[n] != kPeStride[n]) {
      printf("  FAIL stride[%d]: registered %lld, expected %lld\n", n,
             (long long)v->stride[n], (long long)kPeStride[n]);
      ok = false;
    }
  }
  printf("  pe descriptor survives make_device_view verbatim: %s\n",
         ok ? "PASS" : "FAIL");
  printf("  (stagger is INTERFACE, identical to (i,j,k) pk -- so the stagger\n"
         "   carries no axis-order information and only the strides do)\n");
  return ok;
}

// ------------------------------------------------------------------ Part B
// Fill in authoritative (i,k,j) memory order on the host, read on the device
// through .at(i,k,j), decode, and compare to the triple asked for.
bool part_b(Ctx &c) {
  for (int64_t j = kLoJ; j < kLoJ + kNj; ++j)
    for (int64_t k = 0; k < kNpz1; ++k)
      for (int64_t i = kLoI; i < kLoI + kNi; ++i)
        c.h_buf[off_ikj(i, j, k)] = (fv_real)encode(i, j, k);
  if (cudaMemcpy(c.d_buf, c.h_buf.data(), kSize * sizeof(fv_real),
                 cudaMemcpyHostToDevice) != cudaSuccess) return false;

  const DeviceViewBase *v = c.reg.find(FieldId::pe);
  read_kernel<<<1, 64>>>(rview(*v), c.d_probes, kNProbes, 0, c.d_out);
  std::vector<fv_real> out(kNProbes);
  if (cudaMemcpy(out.data(), c.d_out, kNProbes * sizeof(fv_real),
                 cudaMemcpyDeviceToHost) != cudaSuccess) return false;

  bool ok = true;
  for (int n = 0; n < kNProbes; ++n) {
    const P3 q = kProbes[n].p;
    int64_t di, dj, dk;
    if (!decode((double)out[n], di, dj, dk) || di != q.i || dj != q.j ||
        dk != q.k) {
      printf("  FAIL %-42s asked (%lld,%lld,%lld) got value %.1f",
             kProbes[n].what, (long long)q.i, (long long)q.j, (long long)q.k,
             (double)out[n]);
      if (decode((double)out[n], di, dj, dk))
        printf(" = cell (%lld,%lld,%lld)", (long long)di, (long long)dj,
               (long long)dk);
      printf("\n");
      ok = false;
    }
  }
  printf("  %d/%d device reads decoded to the exact (i,k,j) cell: %s\n",
         kNProbes, kNProbes, ok ? "PASS" : "FAIL");
  return ok;
}

// ------------------------------------------------------------------ Part C
// The negative control. A permuted .at(i,j,k) read of the same descriptor must
// NOT return the requested cell -- see the exactness proof in the header.
bool part_c(Ctx &c) {
  const DeviceViewBase *v = c.reg.find(FieldId::pe);
  read_kernel<<<1, 64>>>(rview(*v), c.d_probes, kNProbes, 1, c.d_out);
  std::vector<fv_real> out(kNProbes);
  if (cudaMemcpy(out.data(), c.d_out, kNProbes * sizeof(fv_real),
                 cudaMemcpyDeviceToHost) != cudaSuccess) return false;

  bool ok = true;
  int qualified = 0;
  for (int n = 0; n < kNProbes; ++n) {
    const P3 q = kProbes[n].p;
    // k == j+2 is the one aliasing case, and j+2 <= npz keeps the permuted
    // offset decomposable inside the k range. Both are header-proved.
    if (q.k == q.j + 2 || q.j + 2 > kNpz) continue;
    ++qualified;
    int64_t di, dj, dk;
    const bool decoded = decode((double)out[n], di, dj, dk);
    if (decoded && di == q.i && dj == q.j && dk == q.k) {
      printf("  FAIL %-42s permuted read STILL returned (%lld,%lld,%lld) --\n"
             "       the view is not honouring the descriptor's axis order\n",
             kProbes[n].what, (long long)q.i, (long long)q.j, (long long)q.k);
      ok = false;
    }
  }
  if (qualified < 8) {
    printf("  FAIL only %d probes qualify for the negative control (need >= 8)\n",
           qualified);
    ok = false;
  }
  printf("  %d qualifying probes all mis-read under the (i,j,k) permutation: %s\n",
         qualified, ok ? "PASS" : "FAIL");
  return ok;
}

// ------------------------------------------------------------------ Part D
// The WRITE direction, which is what pe/peln actually are in the CUDA path --
// riem_solver3_kernel's writeback is their only producer. Clear the buffer, let
// the device write through .at(i,k,j), then check the raw HOST offset.
bool part_d(Ctx &c) {
  std::vector<fv_real> zero(kSize, (fv_real)0);
  if (cudaMemcpy(c.d_buf, zero.data(), kSize * sizeof(fv_real),
                 cudaMemcpyHostToDevice) != cudaSuccess) return false;
  const DeviceViewBase *v = c.reg.find(FieldId::pe);
  write_kernel<<<1, 64>>>(wview(*v), c.d_probes, kNProbes, 0);
  std::vector<fv_real> back(kSize);
  if (cudaMemcpy(back.data(), c.d_buf, kSize * sizeof(fv_real),
                 cudaMemcpyDeviceToHost) != cudaSuccess) return false;

  bool ok = true;
  for (int n = 0; n < kNProbes; ++n) {
    const P3 q = kProbes[n].p;
    const double want = encode(q.i, q.j, q.k);
    const double got_ikj = (double)back[off_ikj(q.i, q.j, q.k)];
    if (got_ikj != want) {
      printf("  FAIL %-42s (%lld,%lld,%lld): authoritative (i,k,j) offset holds"
             " %.1f, expected %.1f\n", kProbes[n].what, (long long)q.i,
             (long long)q.j, (long long)q.k, got_ikj, want);
      ok = false;
    }
    // And it must NOT have landed at the (i,j,k) offset, whenever the two differ.
    const size_t o_ikj = off_ikj(q.i, q.j, q.k), o_ijk = off_ijk(q.i, q.j, q.k);
    if (o_ikj != o_ijk && (double)back[o_ijk] == want) {
      printf("  FAIL %-42s (%lld,%lld,%lld): value also appeared at the (i,j,k)"
             " offset\n", kProbes[n].what, (long long)q.i, (long long)q.j,
             (long long)q.k);
      ok = false;
    }
  }
  printf("  %d/%d device writes landed at the (i,k,j) offset and nowhere else:"
         " %s\n", kNProbes, kNProbes, ok ? "PASS" : "FAIL");
  return ok;
}

// ------------------------------------------------------------------ Part E
// The control that keeps Part B honest: pk IS (i,j,k) (nh_core.F90:91), so a
// test that simply demanded "k in slot 1" for every interface field would be
// wrong. Register pk as (i,j,k), read via .at(i,j,k), and require that to
// decode correctly too.
bool part_e(Ctx &c) {
  fv3_gpu_error err{};
  auto d = make_desc(FieldId::pk, c.d_buf, kPkLower, kPkExtent, kPkStride);
  if (c.reg.insert(d, c.d_buf, &err) != 0) {
    printf("  registering pk failed: %s\n", err.message);
    return false;
  }
  for (int64_t k = 0; k < kNpz1; ++k)
    for (int64_t j = kLoJ; j < kLoJ + kNj; ++j)
      for (int64_t i = kLoI; i < kLoI + kNi; ++i)
        c.h_buf[off_ijk(i, j, k)] = (fv_real)encode(i, j, k);
  if (cudaMemcpy(c.d_buf, c.h_buf.data(), kSize * sizeof(fv_real),
                 cudaMemcpyHostToDevice) != cudaSuccess) return false;

  const DeviceViewBase *v = c.reg.find(FieldId::pk);
  read_ijk_kernel<<<1, 64>>>(rview(*v), c.d_probes, kNProbes, c.d_out);
  std::vector<fv_real> out(kNProbes);
  if (cudaMemcpy(out.data(), c.d_out, kNProbes * sizeof(fv_real),
                 cudaMemcpyDeviceToHost) != cudaSuccess) return false;

  bool ok = true;
  for (int n = 0; n < kNProbes; ++n) {
    const P3 q = kProbes[n].p;
    int64_t di, dj, dk;
    if (!decode((double)out[n], di, dj, dk) || di != q.i || dj != q.j ||
        dk != q.k) {
      printf("  FAIL %-42s pk (i,j,k) read returned %.1f\n", kProbes[n].what,
             (double)out[n]);
      ok = false;
    }
  }
  printf("  %d/%d (i,j,k) pk reads decoded correctly, so the test distinguishes\n"
         "  the two orders rather than demanding k in slot 1: %s\n",
         kNProbes, kNProbes, ok ? "PASS" : "FAIL");
  return ok;
}

} // namespace

int main() {
  printf("pe/peln layout test (%s)\n", sizeof(fv_real) == 4 ? "R4" : "R8");
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    printf("no CUDA device available -- this test requires a GPU (SKIP)\n");
    return 77;   // matches SKIP_RETURN_CODE in cuda/dyn_core/CMakeLists.txt
  }
  Ctx c;
  if (!c.setup()) { printf("device allocation failed\n"); return 1; }

  int failed = 0;
  printf("\nPart A: pe descriptor round-trip through the production registry\n");
  if (!part_a(c)) ++failed;
  printf("\nPart B: device READ through the authoritative (i,k,j) order\n");
  if (!part_b(c)) ++failed;
  printf("\nPart C: negative control -- the (i,j,k) permutation must fail\n");
  if (!part_c(c)) ++failed;
  printf("\nPart D: device WRITE through the authoritative (i,k,j) order\n");
  if (!part_d(c)) ++failed;
  printf("\nPart E: control -- pk really is (i,j,k)\n");
  if (!part_e(c)) ++failed;

  printf("\n=== pe/peln layout test: %s ===\n", failed ? "FAIL" : "PASS");
  printf("Numerical validation against authoritative Fortran savepoints:"
         " NOT RUN.\n");
  return failed ? 1 : 0;
}
