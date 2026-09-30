//===========================================================================
// Prompt 112, Stage 13 -- synthetic 24-RANK REAL-MPI sub-tile halo transport
// existence proof.
//
// Combines two already-validated pieces without duplicating either:
//   * the real cross-process transport path from
//     tests/integration/test_halo_mpi_transport.cu (Milestone 3A.15 §11-§13:
//     launch_halo_pack -> MpiHaloTransport::transfer -> launch_halo_unpack,
//     one rank per sub-tile, provenance-encoding oracle, independently-
//     transcribed expected_sign());
//   * the sub-tile topology/encoding from
//     tests/integration/test_halo_subtile_exchange.cu (Prompt 111/112:
//     make_subtile_cubed_sphere_topology, layout_x x layout_y ranks per
//     face, RANK-keyed (not tile-keyed) provenance so a face shared by
//     several ranks is still disambiguated).
//
// WHAT THIS PROVES. That the Prompt 112 generalization of MpiHaloTransport's
// internal storage (Stage 7: measured std::vector<int32_t> tile_of_rank_ and
// size_, not a hardcoded 6-slot std::array) is not just compile-clean but
// GPU-EXECUTES CORRECTLY over a REAL communicator larger than 6 -- something
// the in-process device-to-device tests (test_halo_subtile_exchange.cu)
// cannot exercise, because they never call an MPI transport at all.
//
// HOW THE PRODUCTION GATE STAYS CLOSED. This test calls
// create_mpi_transport_for_testing(), never create_mpi_transport(). That
// test-only factory is unreachable from any production call site (grep
// evidence in the Prompt 112 doc) and create_mpi_transport()'s own
// size==cubed_sphere_tiles==6 gate is untouched by this file or by anything
// it calls. Run with exactly LAYOUT_X*LAYOUT_Y*6 = 24 ranks:
//
//     mpirun -np 24 ./fv3_halo_subtile_mpi_transport
//
// SCOPE (deliberately narrower than the 6-rank test, per Stage 13's own
// "existence proof, not full production-group coverage" framing): two
// groups, one scalar (H_W_IN) and one vector (H_DWIND_IN) -- enough to
// exercise both the plain-copy and the seam-reversed vector-partner-remap
// paths through the now-generalized transport. Corner halo blocks and
// numerical validation against Fortran savepoints are out of scope here,
// exactly as they are for the 6-rank test.
//===========================================================================
#include "fv3_gpu_halo_exchange.hpp"
#include "fv3_gpu_halo_mpi.hpp"
#include "fv3_gpu_halo_subtile_view.hpp"
#include "fv3_gpu_halo_transport.hpp"

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace fv3::gpu;
using namespace fv3::gpu::halo;

namespace {

constexpr int kLayoutX = 2, kLayoutY = 2;
constexpr int kNumRanks = cubed_sphere_tiles * kLayoutX * kLayoutY;  // 24

int failures = 0;
long long checks = 0;
int my_rank = -1;
SubTileId my_id{};
int32_t my_face = 0;

void check(bool ok, const char *what) {
  ++checks;
  if (!ok) {
    if (failures < 40) std::printf("  [rank %d] FAIL: %s\n", my_rank, what);
    ++failures;
  }
}

constexpr int64_t NX = 6, NY = 6, NPZ = 4, NG = 3;
constexpr int64_t EXT_I = NX + 1 + 2 * NG;
constexpr int64_t EXT_J = NY + 1 + 2 * NG;
constexpr int64_t LO = 1 - NG;

/* Same provenance scheme as test_halo_subtile_exchange.cu: keyed on the
 * GLOBAL RANK (1-based), not the physical face, because several ranks share
 * a face at this layout and only rank identity disambiguates them. */
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

/* Stream-ordered create/H2D, per Milestone 3A.18 §1 (test_halo_mpi_transport.cu):
 * issuing the initial H2D on the exchange stream, not the legacy stream,
 * keeps it ordered ahead of launch_halo_pack. */
struct TileArray {
  std::vector<fv_real> host;
  fv_real *dev = nullptr;
  int64_t nk = 0;
  void create(int rank1, int64_t nk_in, cudaStream_t s) {
    nk = nk_in;
    host.assign((size_t)(EXT_I * EXT_J * nk), (fv_real)0);
    for (int64_t k = 1; k <= nk; ++k)
      for (int64_t j = LO; j <= NY + 1 + NG; ++j)
        for (int64_t i = LO; i <= NX + 1 + NG; ++i)
          host[(size_t)((i - LO) + (j - LO) * EXT_I + (k - 1) * EXT_I * EXT_J)] =
              (fv_real)encode(rank1, i, j, k);
    if (cudaMalloc(&dev, host.size() * sizeof(fv_real)) != cudaSuccess) {
      check(false, "cudaMalloc field array");
      dev = nullptr;
      return;
    }
    cudaMemcpyAsync(dev, host.data(), host.size() * sizeof(fv_real),
                    cudaMemcpyHostToDevice, s);
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

/* Independently-transcribed sign oracle, verbatim from both prior halo
 * transport tests -- see test_halo_mpi_transport.cu's header for the
 * anti-circularity rationale (not repeated here; the table itself is
 * untouched by Prompt 112). */
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

bool interface_level_field(FieldId id) {
  return id == FieldId::gz || id == FieldId::zh || id == FieldId::pkc;
}

void exchange_group(HaloTransport *tr, const TopologyView &topo, uint32_t group,
                    const char *label, fv_real *send, fv_real *recv, int64_t buf_capacity,
                    cudaStream_t stream) {
  const HaloGroupSpec spec = halo_group_spec(group);
  if (!spec.live) {
    if (my_rank == 0) std::printf("  (%s is not live; skipped)\n", label);
    return;
  }
  if (my_rank == 0)
    std::printf("\n--- %s: %d field(s), rings %d..%d, %d synthetic ranks ---\n", label,
                spec.field_count, spec.ring_begin, spec.halo_width, kNumRanks);

  const HaloGeometry geom{NX, NY, NPZ};
  const int64_t buf_n = group_buffer_elements(spec, geom);
  check(buf_n <= buf_capacity, "group fits the buffers sized at attach");
  if (buf_n > buf_capacity) return;

  TileArray arr[max_halo_group_fields];
  for (int n = 0; n < spec.field_count; ++n)
    arr[n].create(my_rank + 1,
                  interface_level_field(spec.fields[n].id) ? NPZ + 1 : NPZ, stream);

  cudaMemsetAsync(send, 0xFF, (size_t)buf_n * sizeof(fv_real), stream);
  cudaMemsetAsync(recv, 0xFF, (size_t)buf_n * sizeof(fv_real), stream);

  HaloTileFields fields{};
  for (int n = 0; n < spec.field_count; ++n) {
    fields.view[n] = arr[n].view();
    fields.present[n] = 1;
  }

  fv3_gpu_error err{};
  const char *api = "test_halo_subtile_mpi_transport";

  check(tr->agree_on_layout(group, spec.field_count),
        "all ranks agree on this group's compacted field layout");

  check(launch_halo_pack(spec, geom, fields, send, stream, &err) == FV3_GPU_SUCCESS,
        "pack succeeded");

  const int rc = tr->transfer(spec, geom, topo, send, recv, stream, &err, api);
  check(rc == FV3_GPU_SUCCESS, "MPI transfer succeeded");
  if (rc != FV3_GPU_SUCCESS) std::printf("  [rank %d] transfer: %s\n", my_rank, err.message);

  check(launch_halo_unpack(spec, geom, topo, fields, recv, stream, &err) == FV3_GPU_SUCCESS,
        "unpack succeeded");
  check(tr->release_send(stream, &err, api) == FV3_GPU_SUCCESS, "release_send succeeded");

  const cudaError_t sy = cudaStreamSynchronize(stream);
  check(sy == cudaSuccess, "exchange stream completed without error");
  if (sy != cudaSuccess) std::printf("  [rank %d] cuda: %s\n", my_rank, cudaGetErrorString(sy));

  for (int n = 0; n < spec.field_count; ++n) arr[n].fetch();

  long long verified = 0;
  for (int q = 0; q < 4; ++q) {
    const TileEdge mine = static_cast<TileEdge>(q);
    const EdgeNeighbor &nb = topo.neighbors[q];
    const bool reversed = seam_is_reversed(mine, nb.edge);

    for (int n = 0; n < spec.field_count; ++n) {
      const HaloFieldSpec &fs = spec.fields[n];
      if (!(fs.dest_edges & (1u << q))) continue;
      const StaggerShift src = source_shift(fs.stagger, reversed);
      const int64_t along = edge_seam_extent(mine, NX, NY, fs.stagger);
      const int64_t donor = donor_seam_extent(nb.edge, NX, NY, src);
      const int want_sign = expected_sign(fs.role, mine, reversed);
      const int64_t nk = arr[n].nk;
      const int want_rank1 = nb.rank + 1;

      for (int h = spec.ring_begin; h <= spec.halo_width; ++h) {
        for (int64_t a = 1; a <= along; ++a) {
          int64_t di = 0, dj = 0, si = 0, sj = 0;
          halo_dest_index(mine, a, h, NX, NY, fs.stagger, di, dj);
          const int64_t ap = seam_along_index(a, donor, reversed);
          halo_src_index(nb.edge, ap, h, NX, NY, src, si, sj);

          for (int64_t k = 1; k <= nk; ++k) {
            const double got = arr[n].at(di, dj, k);
            const double mag = (got < 0) ? -got : got;
            int gr = 0; int64_t gi = 0, gj = 0, gk = 0;
            decode(mag, gr, gi, gj, gk);
            ++verified;
            ++checks;

            if (gr != want_rank1 || gi != si || gj != sj || gk != k ||
                (got < 0) != (want_sign < 0)) {
              ++failures;
              if (failures < 40)
                std::printf("  [rank %d] FAIL: face %d edge %s field %d h=%d a=%lld k=%lld\n"
                            "        want rank1 %d (donor face %d) (%lld,%lld,%lld) sign %+d\n"
                            "        got  rank1 %d (%lld,%lld,%lld) sign %+d\n",
                            my_rank, (int)my_face, edge_name(mine), n, h, (long long)a,
                            (long long)k, want_rank1, (int)nb.tile, (long long)si,
                            (long long)sj, (long long)k, want_sign, gr, (long long)gi,
                            (long long)gj, (long long)gk, (got < 0) ? -1 : +1);
            }
          }
        }
      }
    }
  }

  if (my_rank == 0)
    std::printf("  rank 0 decoded and verified %lld received values (each rank does "
                "the same for its own sub-tile)\n", verified);

  for (int n = 0; n < spec.field_count; ++n) arr[n].destroy();
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (size != kNumRanks) {
    if (my_rank == 0)
      std::printf("FAIL: this test requires exactly %d MPI ranks (layout %dx%d per face, "
                  "6 faces); got %d. Run with mpirun -np %d.\n",
                  kNumRanks, kLayoutX, kLayoutY, size, kNumRanks);
    MPI_Finalize();
    return 1;
  }

  /* Deterministic synthetic decomposition (Prompt 109 Stage 4 face-major
   * numbering) -- not measured via MPI_Allgather, because the mapping is a
   * pure function of my_rank and the fixed layout, known identically by
   * construction on every rank; the transport's own Allgather-based
   * production path (create_mpi_transport) is exercised separately by the
   * 6-rank test and is untouched here. */
  my_id = subtile_from_global_rank(my_rank, kLayoutX, kLayoutY);
  my_face = my_id.face;

  std::vector<int32_t> tile_of_rank(kNumRanks);
  for (int r = 0; r < kNumRanks; ++r)
    tile_of_rank[r] = subtile_from_global_rank(r, kLayoutX, kLayoutY).face;

  if (my_rank == 0) {
    std::printf("=== Prompt 112 Stage 13: synthetic 24-rank real-MPI sub-tile "
                "halo transport ===\n");
    std::printf("Precision: %s (sizeof(fv_real)=%zu), %d ranks, layout %dx%d per face\n",
                sizeof(fv_real) == 8 ? "R8" : "R4", sizeof(fv_real), size, kLayoutX, kLayoutY);
  }

  int devices = 0;
  int local_ok = (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0) ? 1 : 0;
  int all_ok = 0;
  MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!all_ok) {
    if (my_rank == 0) std::printf("SKIP: no CUDA device on at least one rank\n");
    MPI_Finalize();
    return 77;
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    std::printf("  [rank %d] FAIL: cudaStreamCreate\n", my_rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  struct GroupCase { uint32_t group; const char *label; };
  const GroupCase cases[] = {
      {FV3_GPU_H_W_IN, "H_W_IN (w, CENTER scalar)"},
      {FV3_GPU_H_DWIND_IN, "H_DWIND_IN (u,v DGRID_NE; slot_x is NORTH-staggered)"},
  };

  const HaloGeometry geom{NX, NY, NPZ};
  int64_t buf_capacity = 0;
  for (const GroupCase &c : cases) {
    const HaloGroupSpec s = halo_group_spec(c.group);
    if (s.live) {
      const int64_t need = group_buffer_elements(s, geom);
      if (need > buf_capacity) buf_capacity = need;
    }
  }

  fv_real *send = nullptr, *recv = nullptr;
  if (cudaMalloc(&send, (size_t)buf_capacity * sizeof(fv_real)) != cudaSuccess ||
      cudaMalloc(&recv, (size_t)buf_capacity * sizeof(fv_real)) != cudaSuccess) {
    std::printf("  [rank %d] FAIL: cudaMalloc halo buffers\n", my_rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  TopologyView topo{};
  if (!make_subtile_cubed_sphere_topology(my_id, kLayoutX, kLayoutY, NX + 1, NY + 1, topo)) {
    std::printf("  [rank %d] FAIL: make_subtile_cubed_sphere_topology(face %d, sub %d,%d)\n",
                my_rank, (int)my_id.face, my_id.sub_i, my_id.sub_j);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  /* THE STAGE 13 CALL. create_mpi_transport_for_testing(), NOT
   * create_mpi_transport() -- the production factory's size==6 gate is never
   * reached by this test and remains completely untouched. `tile` is the
   * PHYSICAL FACE (1..6), matching what MpiHaloTransport::attach() and
   * check_topology_once() both compare against, not the global rank. */
  std::string why;
  HaloTransport *tr = create_mpi_transport_for_testing(
      MPI_Comm_c2f(MPI_COMM_WORLD), my_face, kNumRanks, tile_of_rank,
      TransportMode::host_staged, &why);
  if (!tr) {
    std::printf("  [rank %d] FAIL: create_mpi_transport_for_testing: %s\n", my_rank,
                why.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  if (my_rank == 0) std::printf("transport: %s\n", tr->name());

  HaloMemberResources res{};
  res.send = send;
  res.recv = recv;
  res.buffer_elements = buf_capacity;
  res.transfer_stream = stream;
  if (!tr->attach(my_face, res)) {
    std::printf("  [rank %d] FAIL: attach: %s\n", my_rank, tr->attach_failure_reason());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  for (const GroupCase &c : cases)
    exchange_group(tr, topo, c.group, c.label, send, recv, buf_capacity, stream);

  tr->detach(my_face);
  destroy_mpi_host_staged_transport(tr);
  cudaFree(send);
  cudaFree(recv);
  cudaStreamDestroy(stream);

  long long totals[2] = {checks, (long long)failures};
  long long global[2] = {0, 0};
  MPI_Allreduce(totals, global, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  if (my_rank == 0) {
    std::printf("\n=== Summary (all %d ranks) ===\n", size);
    std::printf("  checks: %lld   failures: %lld\n", global[0], global[1]);
    if (global[1]) {
      std::printf("24-rank synthetic sub-tile MPI halo transport: FAIL\n");
    } else {
      std::printf("24-rank synthetic sub-tile MPI halo transport: PASS (host_staged)\n");
      std::printf("  * GPU EXECUTED across 24 SEPARATE MPI PROCESSES (layout 2x2 per\n");
      std::printf("    face): pack, generalized MpiHaloTransport transfer (Prompt 112\n");
      std::printf("    Stage 7 arbitrary-size storage) and seam-transforming unpack ran\n");
      std::printf("    on a real device on every rank; every received value decoded\n");
      std::printf("    back to its originating RANK (not just face), index and sign.\n");
      std::printf("  * Constructed via create_mpi_transport_for_testing() only;\n");
      std::printf("    create_mpi_transport()'s production comm_size==6 gate was never\n");
      std::printf("    called and remains untouched.\n");
      std::printf("  * Covered: scalar (H_W_IN) and one D-grid vector group (H_DWIND_IN).\n");
      std::printf("  * NOT covered here: corner halo blocks; the other production\n");
      std::printf("    groups; numerical validation against authoritative Fortran\n");
      std::printf("    savepoints.\n");
    }
  }

  MPI_Finalize();
  return global[1] ? 1 : 0;
}
