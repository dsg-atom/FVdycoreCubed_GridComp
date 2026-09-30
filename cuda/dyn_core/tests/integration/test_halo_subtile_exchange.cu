//===========================================================================
// Prompt 111 (Phase 5, edges-only slice) -- sub-tile halo FIELD-DATA
// exchange test. Generalizes the existing, already-validated
// tests/integration/test_halo_exchange_multitile.cu (Milestone 3A.10 §13)
// to an arbitrary layout_x x layout_y-per-face synthetic decomposition, so
// that the REAL, UNMODIFIED production kernel chain --
// launch_halo_pack -> launch_halo_transfer (device-to-device) ->
// launch_halo_unpack -- is exercised and verified end to end using
// TopologyView instances built by the new ADDITIVE
// fv3_gpu_halo_subtile_view.hpp builder (make_subtile_cubed_sphere_topology)
// instead of the legacy one-rank-per-face make_cubed_sphere_topology.
//
// Per project "Reuse" instruction: this file changes only what is
// necessary to generalize from a fixed 6 tiles to N = 6*layout_x*layout_y
// synthetic ranks. The provenance-encoding scheme, the independently-
// transcribed expected_sign() oracle (anti-circularity precedent -- see
// the original file's header comment, not duplicated here), the pack/
// transfer/unpack call sequence, and the per-cell decode-and-verify logic
// are otherwise UNCHANGED.
//
// WHY THIS IS STILL A VALID 1:1 EDGE EXCHANGE AT LAYOUT=2x2: each synthetic
// sub-rank owns its own independent NXxNY local grid (same as the legacy
// per-face grid) -- sub-tile decomposition here means MORE, SAME-SIZE
// ranks, not one rank owning a fraction of a larger shared grid. Every
// edge therefore has exactly ONE donor sub-rank whose full local edge
// matches mine 1:1 (subtile_edge_neighbor's segment_index/segment_count
// bookkeeping identifies WHICH neighboring sub-rank that is; it is not a
// partial-buffer split the pack/unpack kernels need to know about). This
// is why launch_halo_pack/launch_halo_transfer/launch_halo_unpack need
// zero changes to run correctly against this builder's output.
//
// `.tile`/`.rank` CONTRACT (Prompt 112; fv3_gpu_halo_subtile_view.hpp): the
// builder sets EdgeNeighbor.tile = donor's PHYSICAL CUBE FACE (1..6, matching
// the legacy builder's own convention) and EdgeNeighbor.rank = donor's global
// rank (0..n-1, the transport peer identity launch_halo_transfer indexes
// its peer_send table by). This test's own provenance encoding is keyed on
// 1-based RANK identity (`rank1 = global_rank + 1`, this file's loop/array-
// index variable `t`), so the decode-and-verify comparison below checks
// `gr != nb.rank + 1`, not `gr != nb.tile` -- `.tile` is a physical face and
// is no longer expected to equal a rank-derived slot id beyond layout=1x1.
//
// WHAT THIS TEST DOES NOT COVER (same caveats as the legacy file, plus):
//   * Corner halo blocks / cube-vertex three-or-four-rank junctions --
//     edges only, per Prompt 111's explicit scope (Stage 9). No corner
//     message is packed, transferred, or unpacked here.
//   * Real MPI (`tests/integration/test_halo_subtile_mpi_routing.cpp`
//     covers 24-rank message ROUTING; this test covers field-data
//     CORRECTNESS through the real GPU kernels, in-process only).
//   * Any bridge/production wiring -- this builder is unreachable from
//     production; see p111 documentation for the grep evidence.
//   * Numerical validation against authoritative Fortran savepoints.
//===========================================================================
#include "fv3_gpu_halo_exchange.hpp"
#include "fv3_gpu_halo_subtile_view.hpp"
#include "fv3_gpu_errors.hpp"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace fv3::gpu;
using namespace fv3::gpu::halo;

namespace {

int failures = 0;
long long checks = 0;

void check(bool ok, const char *what) {
  ++checks;
  if (!ok && failures < 25) {
    std::printf("  FAIL: %s\n", what);
    ++failures;
  } else if (!ok) {
    ++failures;
  }
}

constexpr int64_t NX = 6, NY = 6, NPZ = 4, NG = 3;
constexpr int64_t EXT_I = NX + 1 + 2 * NG;
constexpr int64_t EXT_J = NY + 1 + 2 * NG;
constexpr int64_t LO = 1 - NG;

/* Lossless provenance encoding, keyed on RANK identity (1-based: rank1 =
 * global_rank + 1) instead of face id -- otherwise identical to the legacy
 * file's encode/decode. */
double encode(int rank1, int64_t i, int64_t j, int64_t k) {
  return (double)((((int64_t)rank1 * 64 + (i + 8)) * 64 + (j + 8)) * 8 + k);
}
void decode(double v, int &rank1, int64_t &i, int64_t &j, int64_t &k) {
  long long n = (long long)std::llround(v);
  k = n % 8; n /= 8;
  j = n % 64 - 8; n /= 64;
  i = n % 64 - 8; n /= 64;
  rank1 = (int)n;
}

struct TileArray {
  std::vector<fv_real> host;
  fv_real *dev = nullptr;
  int64_t nk = 0;
  TileArray() = default;
  void create(int rank1, int64_t nk_in) {
    nk = nk_in;
    host.assign((size_t)(EXT_I * EXT_J * nk), (fv_real)0);
    for (int64_t k = 1; k <= nk; ++k)
      for (int64_t j = LO; j <= NY + 1 + NG; ++j)
        for (int64_t i = LO; i <= NX + 1 + NG; ++i)
          host[(size_t)((i - LO) + (j - LO) * EXT_I + (k - 1) * EXT_I * EXT_J)] =
              (fv_real)encode(rank1, i, j, k);
    if (cudaMalloc(&dev, host.size() * sizeof(fv_real)) != cudaSuccess) {
      std::printf("  FAIL: cudaMalloc\n"); ++failures; dev = nullptr; return;
    }
    cudaMemcpy(dev, host.data(), host.size() * sizeof(fv_real), cudaMemcpyHostToDevice);
  }
  HaloStripView view() const {
    HaloStripView v{};
    v.data = dev;
    v.lo_i = LO; v.lo_j = LO; v.lo_k = 1;
    v.st_i = 1;  v.st_j = EXT_I; v.st_k = EXT_I * EXT_J;
    v.nk = nk;
    return v;
  }
  void fetch() {
    cudaMemcpy(host.data(), dev, host.size() * sizeof(fv_real), cudaMemcpyDeviceToHost);
  }
  double at(int64_t i, int64_t j, int64_t k) const {
    return (double)host[(size_t)((i - LO) + (j - LO) * EXT_I + (k - 1) * EXT_I * EXT_J)];
  }
  void destroy() { if (dev) cudaFree(dev); dev = nullptr; }
};

/* Independently-transcribed oracle, verbatim from the legacy file -- NOT a
 * second call into seam_component_sign()/seam_rotation(). See the legacy
 * file's header comment for the anti-circularity negative-control history;
 * not re-run here since the sign TABLE itself is untouched by Prompt 111 --
 * only the donor-IDENTIFICATION layer (which edge, which rank) is new. */
int expected_sign(VectorRole role, TileEdge mine, bool reversed) {
  if (role == VectorRole::scalar || !reversed) return +1;
  const bool ninety = (mine == TileEdge::west || mine == TileEdge::east);
  if (role == VectorRole::slot_x) return ninety ? +1 : -1;
  return ninety ? -1 : +1;
}

const char *edge_name(TileEdge e) {
  switch (e) {
    case TileEdge::west: return "W";
    case TileEdge::east: return "E";
    case TileEdge::south: return "S";
    default: return "N";
  }
}

/* Exchange ONE group across all N = 6*layout_x*layout_y synthetic sub-ranks
 * and verify every halo cell. Structurally identical to the legacy file's
 * exchange_group(); the only changes are (a) N synthetic ranks instead of a
 * fixed 6, (b) topology built via make_subtile_cubed_sphere_topology
 * instead of make_cubed_sphere_topology, (c) dynamically-sized arrays. */
void exchange_group(uint32_t group, const char *label, int layout_x, int layout_y) {
  const HaloGroupSpec spec = halo_group_spec(group);
  if (!spec.live) { std::printf("  (%s is not live; skipped)\n", label); return; }
  const int n = 6 * layout_x * layout_y;
  std::printf("\n--- %s: %d field(s), halo width %d, layout=%dx%d (%d sub-ranks) ---\n",
              label, spec.field_count, spec.halo_width, layout_x, layout_y, n);

  const HaloGeometry geom{NX, NY, NPZ};
  const int64_t buf_n = group_buffer_elements(spec, geom);

  std::vector<TopologyView> topo(n + 1);
  std::vector<std::vector<TileArray>> arr(n + 1);
  for (auto &row : arr) row.resize(max_halo_group_fields);
  std::vector<fv_real *> send(n + 1, nullptr), recv(n + 1, nullptr);
  fv3_gpu_error err{};

  for (int r = 0; r < n; ++r) {
    const int t = r + 1;
    const SubTileId me = subtile_from_global_rank(r, layout_x, layout_y);
    if (!make_subtile_cubed_sphere_topology(me, layout_x, layout_y, NX + 1, NY + 1, topo[t])) {
      std::printf("  FAIL: make_subtile_cubed_sphere_topology(rank %d)\n", r);
      ++failures; return;
    }
    for (int fld = 0; fld < spec.field_count; ++fld) {
      const bool interface_level = spec.fields[fld].id == FieldId::gz ||
                                   spec.fields[fld].id == FieldId::zh ||
                                   spec.fields[fld].id == FieldId::pkc;
      arr[t][fld].create(t, interface_level ? NPZ + 1 : NPZ);
    }
    if (cudaMalloc(&send[t], (size_t)buf_n * sizeof(fv_real)) != cudaSuccess ||
        cudaMalloc(&recv[t], (size_t)buf_n * sizeof(fv_real)) != cudaSuccess) {
      std::printf("  FAIL: cudaMalloc buffers\n"); ++failures; return;
    }
    cudaMemset(send[t], 0xFF, (size_t)buf_n * sizeof(fv_real));
    cudaMemset(recv[t], 0xFF, (size_t)buf_n * sizeof(fv_real));
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    std::printf("  FAIL: cudaStreamCreate\n"); ++failures; return;
  }

  std::vector<HaloTileFields> fields(n + 1);
  for (int t = 1; t <= n; ++t)
    for (int fld = 0; fld < spec.field_count; ++fld) {
      fields[t].view[fld] = arr[t][fld].view();
      fields[t].present[fld] = 1;
    }

  for (int t = 1; t <= n; ++t)
    check(launch_halo_pack(spec, geom, fields[t], send[t], stream, &err) == FV3_GPU_SUCCESS,
          "pack succeeded");

  /* Prompt 112: launch_halo_transfer indexes by EdgeNeighbor.rank (0..n-1),
   * not by the array's own 1-based (t=r+1) slot. Re-key once here rather
   * than restructure this test's otherwise-unchanged 1-based bookkeeping. */
  std::vector<fv_real *> send_by_rank(n, nullptr);
  for (int r = 0; r < n; ++r) send_by_rank[r] = send[r + 1];

  for (int t = 1; t <= n; ++t)
    check(launch_halo_transfer(spec, geom, topo[t], send_by_rank.data(), n, recv[t], stream,
                               &err) == FV3_GPU_SUCCESS, "device-to-device transfer succeeded");

  for (int t = 1; t <= n; ++t)
    check(launch_halo_unpack(spec, geom, topo[t], fields[t], recv[t], stream, &err) ==
              FV3_GPU_SUCCESS, "unpack succeeded");

  const cudaError_t sync = cudaStreamSynchronize(stream);
  check(sync == cudaSuccess, "exchange stream completed without error");
  if (sync != cudaSuccess) std::printf("  cuda: %s\n", cudaGetErrorString(sync));

  long long verified = 0;
  int edge_cases_checked = 0;
  for (int t = 1; t <= n; ++t) {
    for (int fld = 0; fld < spec.field_count; ++fld) arr[t][fld].fetch();

    for (int q = 0; q < 4; ++q) {
      const TileEdge mine = static_cast<TileEdge>(q);
      const EdgeNeighbor &nb = topo[t].neighbors[q];
      const bool reversed = seam_is_reversed(mine, nb.edge);
      ++edge_cases_checked;

      for (int fld = 0; fld < spec.field_count; ++fld) {
        const HaloFieldSpec &fs = spec.fields[fld];
        const StaggerShift src = source_shift(fs.stagger, reversed);
        const int64_t along = edge_seam_extent(mine, NX, NY, fs.stagger);
        const int64_t donor_ext = donor_seam_extent(nb.edge, NX, NY, src);
        const int want_sign = expected_sign(fs.role, mine, reversed);
        const int64_t nk = arr[t][fld].nk;

        for (int h = 1; h <= spec.halo_width; ++h) {
          for (int64_t a = 1; a <= along; ++a) {
            int64_t di = 0, dj = 0, si = 0, sj = 0;
            halo_dest_index(mine, a, h, NX, NY, fs.stagger, di, dj);
            const int64_t ap = seam_along_index(a, donor_ext, reversed);
            halo_src_index(nb.edge, ap, h, NX, NY, src, si, sj);

            for (int64_t k = 1; k <= nk; ++k) {
              const double got = arr[t][fld].at(di, dj, k);
              const double mag = (got < 0) ? -got : got;
              int gr = 0; int64_t gi = 0, gj = 0, gk = 0;
              decode(mag, gr, gi, gj, gk);
              ++verified;

              const int want_rank1 = nb.rank + 1;
              if (gr != want_rank1 || gi != si || gj != sj || gk != k ||
                  (got < 0) != (want_sign < 0)) {
                if (failures < 25) {
                  std::printf("  FAIL: rank %d edge %s field %d h=%d a=%lld k=%lld\n"
                              "        want rank1 %d (%lld,%lld,%lld) sign %+d\n"
                              "        got  rank1 %d (%lld,%lld,%lld) sign %+d\n",
                              t, edge_name(mine), fld, h, (long long)a, (long long)k,
                              want_rank1, (long long)si, (long long)sj, (long long)k,
                              want_sign, gr, (long long)gi, (long long)gj,
                              (long long)gk, (got < 0) ? -1 : +1);
                }
                ++failures;
                ++checks;
                goto next_field;
              }
              ++checks;
            }
          }
        }
      next_field:;
      }
    }

    for (int fld = 0; fld < spec.field_count; ++fld) {
      const HaloFieldSpec &fs = spec.fields[fld];
      bool intact = true;
      for (int64_t k = 1; k <= arr[t][fld].nk && intact; ++k)
        for (int64_t j = 1; j <= NY + fs.stagger.j && intact; ++j)
          for (int64_t i = 1; i <= NX + fs.stagger.i && intact; ++i)
            if (arr[t][fld].at(i, j, k) != (fv_real)encode(t, i, j, k)) intact = false;
      check(intact, "owned interior untouched by the exchange");
    }
  }
  std::printf("  %lld halo values decoded and verified across %d sub-ranks x 4 edges "
              "(%d edge-cases)\n", verified, n, edge_cases_checked);

  cudaStreamDestroy(stream);
  for (int t = 1; t <= n; ++t) {
    for (int fld = 0; fld < spec.field_count; ++fld) arr[t][fld].destroy();
    if (send[t]) cudaFree(send[t]);
    if (recv[t]) cudaFree(recv[t]);
  }
}

} // namespace

int main() {
  std::printf("=== Prompt 111 sub-tile GPU halo field-data exchange ===\n");
  std::printf("Precision: %s (sizeof(fv_real)=%zu)\n",
              sizeof(fv_real) == 8 ? "R8" : "R4", sizeof(fv_real));

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("SKIP: no CUDA devices available\n");
    return 77;
  }

  /* Stage 12: layout=1x1 (N=6) is a bitwise regression against the legacy
   * builder's already-proven-correct multitile behavior -- same groups, same
   * geometry, only the topology source differs (new builder vs legacy). */
  std::printf("\n########## layout=1x1 (N=6) -- Stage 12 regression ##########\n");
  exchange_group(FV3_GPU_H_DWIND_IN, "H_DWIND_IN (u,v DGRID_NE)", 1, 1);
  exchange_group(FV3_GPU_H_CWIND, "H_CWIND (uc,vc CGRID_NE)", 1, 1);
  exchange_group(FV3_GPU_H_W_IN, "H_W_IN (w, CENTER scalar)", 1, 1);
  exchange_group(FV3_GPU_H_DIVGD, "H_DIVGD (divgd, CORNER scalar)", 1, 1);
  exchange_group(FV3_GPU_H_PKC2, "H_PKC2 (pkc, CENTER, npz+1 levels)", 1, 1);
  exchange_group(FV3_GPU_H_DSCALAR, "H_DSCALAR (q_con,delp,pt,cappa)", 1, 1);

  /* Stages 5-8: layout=2x2 (N=24) -- same-face scalar edges (Stage 5),
   * cross-face segmented scalar edges (Stage 6, H_W_IN/H_DSCALAR/H_DIVGD),
   * U/V vector transform (Stage 7, H_DWIND_IN/H_CWIND), all-edge coverage
   * (Stage 8, printed edge-case count below). */
  std::printf("\n########## layout=2x2 (N=24) -- Stages 5-8 ##########\n");
  {
    struct { uint32_t g; const char *label; } groups[] = {
      {FV3_GPU_H_DWIND_IN, "H_DWIND_IN (u,v DGRID_NE)"},
      {FV3_GPU_H_CWIND, "H_CWIND (uc,vc CGRID_NE)"},
      {FV3_GPU_H_W_IN, "H_W_IN (w, CENTER scalar)"},
      {FV3_GPU_H_DIVGD, "H_DIVGD (divgd, CORNER scalar)"},
      {FV3_GPU_H_PKC2, "H_PKC2 (pkc, CENTER, npz+1 levels)"},
      {FV3_GPU_H_DSCALAR, "H_DSCALAR (q_con,delp,pt,cappa)"},
    };
    for (auto &grp : groups) {
      int before = failures;
      exchange_group(grp.g, grp.label, 2, 2);
      std::printf("  [DIAG] %s failures this group: %d\n", grp.label, failures - before);
    }
  }

  std::printf("\n=== Summary ===\n");
  std::printf("  checks: %lld   failures: %d\n", checks, failures);
  if (failures) {
    std::printf("Sub-tile halo field-data exchange: FAIL\n");
    return 1;
  }
  std::printf("Sub-tile halo field-data exchange: PASS\n");
  std::printf("  * GPU EXECUTED: real, UNMODIFIED launch_halo_pack/\n");
  std::printf("    launch_halo_transfer/launch_halo_unpack ran against\n");
  std::printf("    TopologyView instances from the new additive sub-tile\n");
  std::printf("    builder, at layout=1x1 (6 ranks) and layout=2x2 (24 ranks).\n");
  std::printf("  * NOT covered: corner halo blocks (edges only, by design --\n");
  std::printf("    no corner message is packed/transferred/unpacked here),\n");
  std::printf("    real multi-rank MPI, bridge/production wiring (this\n");
  std::printf("    builder is unreachable from production).\n");
  std::printf("  * Numerical validation against authoritative Fortran\n");
  std::printf("    savepoints: NOT RUN (not applicable -- synthetic ranks).\n");
  return 0;
}
