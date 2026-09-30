//===========================================================================
// Prompt 109 Phases 1-4, Stage 15 -- CUDA GPU-EXECUTED smoke test for the
// generalized sub-tile halo topology (fv3_gpu_halo_subtile_topology.hpp).
//
// SCOPE. Phases 1-4 are TOPOLOGY-ONLY (see the header's own top comment and
// docs/performance/p109_*.md): no CUDA field ever moves in this phase, and
// no halo-fill kernel (fv3_gpu_halo_exchange.hpp / fv3_cgrid_corners.hpp) is
// generalized here -- that is Phase 5+ per the Prompt 108 plan, out of this
// task's change budget. What DOES belong to Phases 1-4, and what this test
// closes, is the GPU-EXECUTED leg of the validation hierarchy for the
// topology functions THEMSELVES: every function in the new header is
// FV3_HALO_HD (__host__ __device__), so this test launches one thread per
// sub-tile rank (24 threads, one block), has each thread compute its own 4
// edge neighbors and 4 corner donors ON THE DEVICE, copies the results back,
// and compares them BIT-FOR-BIT against the SAME computation done on the
// HOST for the SAME 24 identities (test_halo_subtile_topology.cpp already
// established the host results are correct; this test's job is only to
// prove the device compiles/executes the identical logic, not to
// re-establish correctness).
//
// This is deliberately NOT a data-movement test: it is the device-callable
// counterpart of test_halo_subtile_topology.cpp's host-only T1-T11, scoped
// to exactly what Phases 1-4 own.
//===========================================================================
#include "fv3_gpu_halo_subtile_topology.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace fv3::gpu;
using namespace fv3::gpu::halo;

namespace {

constexpr int kLayoutX = 2;
constexpr int kLayoutY = 2;
constexpr int kNumRanks = 6 * kLayoutX * kLayoutY; // 24

// Flattened, POD device-transferable snapshot of everything
// subtile_edge_neighbor/subtile_corner_donor produce for one rank.
struct DeviceResult {
  // 4 edges: donor identity, donor_edge, orientation, signs, same_face,
  // segment index/count/donor_segment_index.
  int32_t edge_donor_face[4], edge_donor_sub_i[4], edge_donor_sub_j[4];
  int32_t edge_donor_edge[4], edge_orientation[4];
  int32_t edge_sign_x[4], edge_sign_y[4];
  int32_t edge_same_face[4], edge_seg_idx[4], edge_seg_count[4], edge_donor_seg_idx[4];
  // 4 corners: type, donor identity.
  int32_t corner_type[4];
  int32_t corner_donor_face[4], corner_donor_sub_i[4], corner_donor_sub_j[4];
};

FV3_HALO_HD inline void compute_result(int global_rank, DeviceResult &out) {
  const SubTileId me = subtile_from_global_rank(global_rank, kLayoutX, kLayoutY);
  const TileEdge edges[4] = {TileEdge::west, TileEdge::east, TileEdge::south, TileEdge::north};
  for (int e = 0; e < 4; ++e) {
    const SubTileEdgeNeighbor n = subtile_edge_neighbor(me, edges[e], kLayoutX, kLayoutY);
    out.edge_donor_face[e] = n.donor.face;
    out.edge_donor_sub_i[e] = n.donor.sub_i;
    out.edge_donor_sub_j[e] = n.donor.sub_j;
    out.edge_donor_edge[e] = static_cast<int32_t>(n.donor_edge);
    out.edge_orientation[e] = static_cast<int32_t>(n.orientation);
    out.edge_sign_x[e] = n.vector_sign_x;
    out.edge_sign_y[e] = n.vector_sign_y;
    out.edge_same_face[e] = n.same_face ? 1 : 0;
    out.edge_seg_idx[e] = n.segment_index;
    out.edge_seg_count[e] = n.segment_count;
    out.edge_donor_seg_idx[e] = n.donor_segment_index;
  }
  for (int c = 0; c < 4; ++c) {
    SeamVectorTransform xf{};
    const SubTileId donor = subtile_corner_donor(me, c, kLayoutX, kLayoutY, xf);
    out.corner_type[c] = static_cast<int32_t>(classify_subtile_corner(me, c, kLayoutX, kLayoutY));
    out.corner_donor_face[c] = donor.face;
    out.corner_donor_sub_i[c] = donor.sub_i;
    out.corner_donor_sub_j[c] = donor.sub_j;
  }
}

__global__ void compute_on_device(DeviceResult *results) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= kNumRanks) return;
  compute_result(r, results[r]);
}

bool results_equal(const DeviceResult &a, const DeviceResult &b) {
  return std::memcmp(&a, &b, sizeof(DeviceResult)) == 0;
}

#define CUDA_CHECK(expr)                                                                       \
  do {                                                                                          \
    cudaError_t _e = (expr);                                                                    \
    if (_e != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                     \
                   cudaGetErrorString(_e));                                                     \
      std::exit(1);                                                                             \
    }                                                                                            \
  } while (0)

} // namespace

int main() {
  int device_count = 0;
  const cudaError_t rc = cudaGetDeviceCount(&device_count);
  if (rc != cudaSuccess || device_count == 0) {
    std::fprintf(stderr,
                  "SKIP: no CUDA device visible (rc=%d, count=%d) -- this test requires "
                  "GPU execution, not just compilation.\n",
                  static_cast<int>(rc), device_count);
    return 77;
  }

  DeviceResult host_ref[kNumRanks];
  for (int r = 0; r < kNumRanks; ++r) {
    compute_result(r, host_ref[r]);
  }

  DeviceResult *d_results = nullptr;
  CUDA_CHECK(cudaMalloc(&d_results, sizeof(DeviceResult) * kNumRanks));
  compute_on_device<<<1, kNumRanks>>>(d_results);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  DeviceResult host_from_device[kNumRanks];
  CUDA_CHECK(cudaMemcpy(host_from_device, d_results, sizeof(DeviceResult) * kNumRanks,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_results));

  int mismatches = 0;
  for (int r = 0; r < kNumRanks; ++r) {
    if (!results_equal(host_ref[r], host_from_device[r])) {
      std::fprintf(stderr, "rank %d: DEVICE result differs from HOST result\n", r);
      ++mismatches;
    }
  }

  if (mismatches == 0) {
    std::printf(
        "PASS: sub-tile halo topology GPU-executed smoke test (%d ranks, layout=%dx%d) -- "
        "device kernel results bit-identical to host results for every edge and corner.\n",
        kNumRanks, kLayoutX, kLayoutY);
    std::printf(
        "  Covers: GPU EXECUTION (not just compilation) of "
        "fv3_gpu_halo_subtile_topology.hpp's FV3_HALO_HD functions.\n");
    std::printf(
        "  Does NOT cover: CUDA halo kernel application, actual field data movement "
        "(Phase 5+), numerical validation, MPI routing (see "
        "fv3_halo_subtile_mpi_routing_test).\n");
    return 0;
  }
  std::fprintf(stderr, "FAIL: %d/%d ranks mismatched between device and host\n", mismatches,
               kNumRanks);
  return 1;
}
