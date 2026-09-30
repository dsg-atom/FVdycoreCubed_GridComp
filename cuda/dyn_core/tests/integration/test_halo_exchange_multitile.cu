//===========================================================================
// Milestone 3A.10 §13 -- permanent integrated MULTI-TILE halo exchange test.
//
// All six cubed-sphere tiles, on the device, exchanged through the real
// production path: launch_halo_pack -> launch_halo_transfer (device-to-device)
// -> launch_halo_unpack. No host packing, no host staging, no
// cudaDeviceSynchronize inside the exchange.
//
// -------------------------------------------------------------------------
// HOW THIS TEST AVOIDS BEING CIRCULAR
// -------------------------------------------------------------------------
//
// §13 demands "deliberately asymmetric per-tile data so incorrect neighbor
// mapping cannot pass". Rather than comparing against a second copy of the
// index formulae, every array element carries its own PROVENANCE, encoded
// losslessly in its value:
//
//     value(tile, i, j, k) = (((tile*64 + (i+8))*64 + (j+8))*8 + k)
//
// The maximum is ~206k, well inside float's exact-integer range (2^24), so the
// encoding is exact in R4 and R8 alike and the same test binary proves both.
//
// After the exchange each halo cell is DECODED back to (tile, i, j, k) and we
// assert it came from the right tile, at the right index, with the right sign.
// That makes the following independently detectable, which is precisely the list
// §13 asks for:
//
//   * wrong neighbour tile      -> decoded tile is wrong
//   * reversed seam not applied -> decoded along-index is mirrored
//   * u/v not swapped on a rotated seam -> decoded index has the wrong
//                                          staggering shift, and the sign is wrong
//   * wrong sign                -> sign check fails
//   * halo not written at all   -> decodes to the RECEIVER's own tile, because
//                                  every array is pre-filled with its owner's
//                                  own encoding. A no-op exchange cannot pass.
//   * insufficient halo width   -> the h = 3 ring fails
//
// The index formulae themselves are validated separately, and against a genuinely
// independent oracle -- FMS's own fill_cubic_grid_halo -- by T9 of
// tests/runtime/test_halo_topology.cpp. This test validates the KERNEL CHAIN:
// buffer addressing, the rotated-seam partner remap, the sign, and the fact that
// the data actually moves on the GPU.
//
// -------------------------------------------------------------------------
// NEGATIVE CONTROLS ACTUALLY RUN (A100 warpa003, job 58077949)
// -------------------------------------------------------------------------
//
// A passing test proves nothing until it has been shown to fail. Three faults
// were injected into the IMPLEMENTATION, one at a time, rebuilt, and run:
//
//   1. seam_component_sign() forced to return +1 unconditionally
//        -> FAIL, 24 failures. Decoded tile and index correct, sign wrong.
//   2. the along-seam reversal dropped in unpack_edge_kernel (tp = t)
//        -> FAIL, 132 failures. Decoded along-index mirrored.
//   3. the rotated-seam u/v partner remap dropped in launch_halo_transfer
//        -> FAIL, 48 failures. Decoded index carries the wrong staggering shift.
//
// All three faults were then reverted and the clean build passes in R8 and R4
// alike (20484 checks, 0 failures each).
//
// An EARLIER version of this test survived fault 1. It computed its expected
// sign by calling seam_component_sign() -- the very function under test -- so
// the fault moved expectation and behaviour together. That is why expected_sign()
// below is a local transcription of the FMS table and not a second call into the
// header. The bug was in the test; the implementation was correct throughout.
//
// -------------------------------------------------------------------------
// WHAT THIS TEST DOES NOT COVER
// -------------------------------------------------------------------------
//
//   * The corner halo BLOCKS (i and j both outside the owned range). The four
//     edge strips are disjoint and stop short of them by construction. Cube
//     vertices are three-tile junctions and are a separate question (§8).
//   * The shared staggered seam LINE, which FV3 reconciles through
//     mpp_get_boundary rather than through a halo update (exchange E11, §1).
//   * Numerical validation against authoritative Fortran savepoints: NOT RUN.
//===========================================================================
#include "fv3_gpu_halo_exchange.hpp"
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

/* Geometry. Small enough to check exhaustively on the host, large enough that a
 * 3-ring halo is strictly narrower than the owned block (so a strip can never be
 * confused with the whole face). */
constexpr int64_t NX = 6, NY = 6, NPZ = 4, NG = 3;
constexpr int64_t EXT_I = NX + 1 + 2 * NG;   /* i from 1-NG to NX+1+NG */
constexpr int64_t EXT_J = NY + 1 + 2 * NG;
constexpr int64_t LO = 1 - NG;

/* Lossless provenance encoding. Exact in float and double. */
double encode(int tile, int64_t i, int64_t j, int64_t k) {
  return (double)((((int64_t)tile * 64 + (i + 8)) * 64 + (j + 8)) * 8 + k);
}
void decode(double v, int &tile, int64_t &i, int64_t &j, int64_t &k) {
  long long n = (long long)std::llround(v);
  k = n % 8; n /= 8;
  j = n % 64 - 8; n /= 64;
  i = n % 64 - 8; n /= 64;
  tile = (int)n;
}

struct TileArray {
  std::vector<fv_real> host;
  fv_real *dev = nullptr;
  int64_t nk = 0;
  TileArray() = default;
  void create(int tile, int64_t nk_in) {
    nk = nk_in;
    host.assign((size_t)(EXT_I * EXT_J * nk), (fv_real)0);
    /* Pre-fill EVERYTHING, halo included, with this tile's OWN encoding. A halo
     * cell the exchange fails to write therefore decodes to the receiver's own
     * tile and is caught, rather than silently reading as whatever was there. */
    for (int64_t k = 1; k <= nk; ++k)
      for (int64_t j = LO; j <= NY + 1 + NG; ++j)
        for (int64_t i = LO; i <= NX + 1 + NG; ++i)
          host[(size_t)((i - LO) + (j - LO) * EXT_I + (k - 1) * EXT_I * EXT_J)] =
              (fv_real)encode(tile, i, j, k);
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

/* The expected sign, transcribed LOCALLY from FMS group_update_pack.inc and
 * keyed on the receiver's own halo edge -- deliberately NOT computed with
 * seam_component_sign() or seam_rotation().
 *
 * The first version of this test called seam_component_sign() for the expected
 * value, which made the sign check compare the implementation against itself. A
 * negative control that forced seam_component_sign() to return +1 unconditionally
 * still PASSED, which is how the circularity was found. Restating the table here
 * is the whole point: this is the oracle, not a second call to the code.
 *
 *   my halo edge W or E  -> rotation NINETY        (mpp_domains_define.inc:5459/:5513)
 *   my halo edge S or N  -> rotation MINUS_NINETY  (:5486/:5540)
 *   FIELD_X: +fieldy at NINETY (:128),  -fieldy at MINUS_NINETY (:115)
 *   FIELD_Y: -fieldx at NINETY (:206),  +fieldx at MINUS_NINETY (:181)
 *   FIELD_S: +field always (:39/:51/:63/:75)
 */
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

/* Exchange ONE group across all six tiles and verify every halo cell. */
void exchange_group(uint32_t group, const char *label) {
  const HaloGroupSpec spec = halo_group_spec(group);
  if (!spec.live) { std::printf("  (%s is not live; skipped)\n", label); return; }
  std::printf("\n--- %s: %d field(s), halo width %d, all 6 tiles ---\n",
              label, spec.field_count, spec.halo_width);

  const HaloGeometry geom{NX, NY, NPZ};
  const int64_t buf_n = group_buffer_elements(spec, geom);

  int32_t rank_of_tile[6] = {0, 1, 2, 3, 4, 5};
  TopologyView topo[7];
  TileArray arr[7][max_halo_group_fields];
  fv_real *send[7] = {nullptr}, *recv[7] = {nullptr};
  fv3_gpu_error err{};

  for (int t = 1; t <= 6; ++t) {
    if (!make_cubed_sphere_topology(t, NX + 1, NY + 1, rank_of_tile, topo[t])) {
      std::printf("  FAIL: make_cubed_sphere_topology(tile %d)\n", t); ++failures; return;
    }
    for (int n = 0; n < spec.field_count; ++n) {
      /* Interface-level fields carry npz+1. Only the CENTER scalars in this
       * milestone's live groups do (gz/zh/pkc); winds and divgd carry npz. */
      const bool interface_level = spec.fields[n].id == FieldId::gz ||
                                   spec.fields[n].id == FieldId::zh ||
                                   spec.fields[n].id == FieldId::pkc;
      arr[t][n].create(t, interface_level ? NPZ + 1 : NPZ);
    }
    if (cudaMalloc(&send[t], (size_t)buf_n * sizeof(fv_real)) != cudaSuccess ||
        cudaMalloc(&recv[t], (size_t)buf_n * sizeof(fv_real)) != cudaSuccess) {
      std::printf("  FAIL: cudaMalloc buffers\n"); ++failures; return;
    }
    /* Poison the buffers, so an unwritten slot cannot masquerade as a zero that
     * happens to look plausible. */
    cudaMemset(send[t], 0xFF, (size_t)buf_n * sizeof(fv_real));
    cudaMemset(recv[t], 0xFF, (size_t)buf_n * sizeof(fv_real));
  }

  /* One stream for the whole exchange. Ordering within a stream is program
   * order, so pack -> transfer -> unpack needs no events here; the CROSS-stream
   * case is the runtime's problem and is handled by order_after (§17). */
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    std::printf("  FAIL: cudaStreamCreate\n"); ++failures; return;
  }

  HaloTileFields fields[7];
  for (int t = 1; t <= 6; ++t)
    for (int n = 0; n < spec.field_count; ++n) {
      fields[t].view[n] = arr[t][n].view();
      fields[t].present[n] = 1;
    }

  /* PACK every tile first. The transfer reads a NEIGHBOUR's send buffer, so all
   * packs must precede all transfers -- on one stream that is just ordering, and
   * it is the same barrier an MPI implementation gets from Isend/Irecv. */
  for (int t = 1; t <= 6; ++t)
    check(launch_halo_pack(spec, geom, fields[t], send[t], stream, &err) == FV3_GPU_SUCCESS,
          "pack succeeded");

  /* Prompt 112: launch_halo_transfer indexes by EdgeNeighbor.rank now, not
   * .tile. rank_of_tile[] above is the identity permutation, so rank == tile
   * - 1 for every tile here and this re-keying is exact/lossless. */
  fv_real *send_by_rank[6];
  for (int r = 0; r < 6; ++r) send_by_rank[r] = send[r + 1];

  for (int t = 1; t <= 6; ++t)
    check(launch_halo_transfer(spec, geom, topo[t], send_by_rank, 6, recv[t], stream, &err) ==
              FV3_GPU_SUCCESS, "device-to-device transfer succeeded");

  for (int t = 1; t <= 6; ++t)
    check(launch_halo_unpack(spec, geom, topo[t], fields[t], recv[t], stream, &err) ==
              FV3_GPU_SUCCESS, "unpack succeeded");

  const cudaError_t sync = cudaStreamSynchronize(stream);
  check(sync == cudaSuccess, "exchange stream completed without error");
  if (sync != cudaSuccess) std::printf("  cuda: %s\n", cudaGetErrorString(sync));

  /* ------------------------------------------------------------------
   * VERIFY. Every halo cell of every field of every tile, decoded.
   * ------------------------------------------------------------------ */
  long long verified = 0;
  for (int t = 1; t <= 6; ++t) {
    for (int n = 0; n < spec.field_count; ++n) arr[t][n].fetch();

    for (int q = 0; q < 4; ++q) {
      const TileEdge mine = static_cast<TileEdge>(q);
      const EdgeNeighbor &nb = topo[t].neighbors[q];
      const bool reversed = seam_is_reversed(mine, nb.edge);

      for (int n = 0; n < spec.field_count; ++n) {
        const HaloFieldSpec &fs = spec.fields[n];
        const StaggerShift src = source_shift(fs.stagger, reversed);
        const int64_t along = edge_seam_extent(mine, NX, NY, fs.stagger);
        const int64_t donor = donor_seam_extent(nb.edge, NX, NY, src);
        const int want_sign = expected_sign(fs.role, mine, reversed);
        const int64_t nk = arr[t][n].nk;

        for (int h = 1; h <= spec.halo_width; ++h) {
          for (int64_t a = 1; a <= along; ++a) {
            int64_t di = 0, dj = 0, si = 0, sj = 0;
            halo_dest_index(mine, a, h, NX, NY, fs.stagger, di, dj);
            const int64_t ap = seam_along_index(a, donor, reversed);
            halo_src_index(nb.edge, ap, h, NX, NY, src, si, sj);

            for (int64_t k = 1; k <= nk; ++k) {
              const double got = arr[t][n].at(di, dj, k);
              const double mag = (got < 0) ? -got : got;
              int gt = 0; int64_t gi = 0, gj = 0, gk = 0;
              decode(mag, gt, gi, gj, gk);
              ++verified;

              if (gt != nb.tile || gi != si || gj != sj || gk != k ||
                  (got < 0) != (want_sign < 0)) {
                if (failures < 25) {
                  std::printf("  FAIL: tile %d edge %s field %d h=%d a=%lld k=%lld\n"
                              "        want tile %d (%lld,%lld,%lld) sign %+d\n"
                              "        got  tile %d (%lld,%lld,%lld) sign %+d\n",
                              t, edge_name(mine), n, h, (long long)a, (long long)k,
                              (int)nb.tile, (long long)si, (long long)sj, (long long)k,
                              want_sign, gt, (long long)gi, (long long)gj,
                              (long long)gk, (got < 0) ? -1 : +1);
                }
                ++failures;
                ++checks;
                goto next_field;   /* one report per (tile, edge, field) is enough */
              }
              ++checks;
            }
          }
        }
      next_field:;
      }
    }

    /* The exchange must not have touched a single OWNED cell. */
    for (int n = 0; n < spec.field_count; ++n) {
      const HaloFieldSpec &fs = spec.fields[n];
      bool intact = true;
      for (int64_t k = 1; k <= arr[t][n].nk && intact; ++k)
        for (int64_t j = 1; j <= NY + fs.stagger.j && intact; ++j)
          for (int64_t i = 1; i <= NX + fs.stagger.i && intact; ++i)
            if (arr[t][n].at(i, j, k) != (fv_real)encode(t, i, j, k)) intact = false;
      check(intact, "owned interior untouched by the exchange");
    }
  }
  std::printf("  %lld halo values decoded and verified across 6 tiles x 4 seams\n",
              verified);

  cudaStreamDestroy(stream);
  for (int t = 1; t <= 6; ++t) {
    for (int n = 0; n < spec.field_count; ++n) arr[t][n].destroy();
    if (send[t]) cudaFree(send[t]);
    if (recv[t]) cudaFree(recv[t]);
  }
}

} // namespace

int main() {
  std::printf("=== Milestone 3A.10 multi-tile GPU halo exchange ===\n");
  std::printf("Precision: %s (sizeof(fv_real)=%zu)\n",
              sizeof(fv_real) == 8 ? "R8" : "R4", sizeof(fv_real));

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("SKIP: no CUDA devices available\n");
    return 77;
  }

  /* D-grid winds: slot_x is u, which is NORTH-staggered. C-grid winds: slot_x is
   * uc, which is EAST-staggered. Running BOTH is the point -- a single
   * staggering-keyed sign table passes one and fails the other. */
  exchange_group(FV3_GPU_H_DWIND_IN, "H_DWIND_IN (u,v DGRID_NE)");
  exchange_group(FV3_GPU_H_CWIND, "H_CWIND (uc,vc CGRID_NE)");
  exchange_group(FV3_GPU_H_W_IN, "H_W_IN (w, CENTER scalar)");
  exchange_group(FV3_GPU_H_DIVGD, "H_DIVGD (divgd, CORNER scalar)");
  exchange_group(FV3_GPU_H_PKC2, "H_PKC2 (pkc, CENTER, npz+1 levels)");
  exchange_group(FV3_GPU_H_DSCALAR, "H_DSCALAR (q_con,delp,pt,cappa)");

  std::printf("\n=== Summary ===\n");
  std::printf("  checks: %lld   failures: %d\n", checks, failures);
  if (failures) {
    std::printf("Multi-tile halo exchange: FAIL\n");
    return 1;
  }
  std::printf("Multi-tile halo exchange: PASS\n");
  std::printf("  * GPU EXECUTED: pack, device-to-device transfer and\n");
  std::printf("    seam-transforming unpack ran on a real device for all six\n");
  std::printf("    tiles, and every halo value was decoded back to its\n");
  std::printf("    originating tile, index and sign.\n");
  std::printf("  * NOT covered: corner halo blocks, the shared staggered seam\n");
  std::printf("    line (exchange E11), and multi-rank MPI.\n");
  std::printf("  * Numerical validation against authoritative Fortran\n");
  std::printf("    savepoints: NOT RUN.\n");
  return 0;
}
