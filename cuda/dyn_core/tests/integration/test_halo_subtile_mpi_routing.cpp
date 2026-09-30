//===========================================================================
// Prompt 109 Phases 1-4, Stage 14 -- permanent 24-RANK MPI routing test for
// the generalized sub-tile (2x2-ranks-per-face) halo topology.
//
// Run with exactly 24 ranks (layout_x = layout_y = 2, 6 faces):
//
//     mpirun -np 24 ./fv3_halo_subtile_mpi_routing_test
//
// SCOPE. This is a ROUTING test, not a numerics test: it proves that the
// donor identification in fv3_gpu_halo_subtile_topology.hpp (Prompt 109
// Stages 2-13) produces a real, deadlock-free, collision-free MPI exchange
// across actual separate processes -- each rank can see only its OWN
// identity, so a wrong-donor defect (mismatched tag, wrong rank<->subtile
// map, a reciprocal-index bug) cannot be caught by comparing two local
// structs the way the host-only unit test (test_halo_subtile_topology.cpp)
// does. It is caught anyway, because every message carries the SENDER's
// own identity and edge/corner index, and the RECEIVER independently
// recomputes (via the same pure, already-unit-tested topology functions,
// called locally -- not trusted from the wire) what that identity and
// index MUST be, then compares. This mirrors the "oracle travels inside
// the data" pattern used by test_halo_mpi_transport.cu, applied one level
// up: the payload is ROUTING PROVENANCE (who/which-edge), not a field
// value, because Phases 1-4 are topology-only (no CUDA field ever moves
// here).
//
// ALL-NONBLOCKING BY DESIGN. Every rank posts ALL of its Irecvs and Isends
// (up to 4 edges + 3 corners) before waiting on any of them, then a single
// Waitall. This is required, not a style choice: a per-message BLOCKING
// Sendrecv, done in the SAME fixed edge order (west,east,south,north) on
// every rank, deadlocks in practice -- confirmed by running an earlier,
// blocking-Sendrecv version of this test, which hung with every one of the
// 24 ranks stuck inside its very first (west) Sendrecv. The reason: rank
// A's west-edge receive often expects the tag its neighbor B sends only
// from B's OWN east-edge (2nd loop iteration), but B is simultaneously
// stuck waiting on ITS west-edge (1st iteration) receive from some other
// rank -- a real circular wait with no rank able to make progress, not a
// topology/algebra bug (subtile_edge_neighbor/subtile_corner_donor
// reciprocity was independently confirmed correct offline, exhaustively,
// over all 24 ranks, before this fix). Posting everything non-blocking
// removes the lockstep ordering dependency entirely.
//
// What is covered:
//   * all 4 x 24 = 96 edge exchanges (cardinal same-face and cross-face)
//   * all mixed/diagonal corner exchanges (2 mixed + 1 diagonal per
//     sub-rank x 24 = 72; the 4th, true_cube_corner, is local by design
//     and sends no message -- verified directly, not just assumed)
//   * MPI tag collision-freedom in a REAL MPI_Comm, not just by argument
//   * reciprocity of donor identity, edge/corner index, orientation and
//     vector sign, cross-checked by a SEPARATE local computation on the
//     receiving rank (subtile_edge_neighbor/subtile_corner_donor called
//     again, from the donor's own identity, not the unit-tested call the
//     sender itself made)
//
// What this test does NOT cover:
//   * CUDA halo kernel application / actual field data movement (that is
//     Phase 5+ per the Prompt 108 plan and the still-to-be-written CUDA
//     small-grid smoke test)
//   * numerical validation against authoritative Fortran savepoints
//   * the production comm_size==6 bridge gate, which this test does not
//     touch and is not a substitute for (see docs/performance/p109_*.md
//     Stage 18)
//===========================================================================
#include "fv3_gpu_halo_subtile_topology.hpp"

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace fv3::gpu;
using namespace fv3::gpu::halo;

namespace {

constexpr int kLayoutX = 2;
constexpr int kLayoutY = 2;
constexpr int kExpectedRanks = 6 * kLayoutX * kLayoutY; // 24

struct EdgeMsg {
  int32_t face, sub_i, sub_j;
  int32_t rank;
  int32_t edge; // sender's own TileEdge, as int
  int32_t vector_sign_x, vector_sign_y;
  int32_t orientation; // sender's own EdgeOrientation, as int
};

struct CornerMsg {
  int32_t face, sub_i, sub_j;
  int32_t rank;
  int32_t corner; // sender's own corner index (0..3)
};

struct PendingEdge {
  TileEdge mine;
  SubTileEdgeNeighbor n;
  int32_t donor_rank;
  EdgeMsg send_msg;
  EdgeMsg recv_msg;
};

struct PendingCorner {
  int corner;
  SubTileId donor;
  int32_t donor_rank;
  int recip;
  CornerMsg send_msg;
  CornerMsg recv_msg;
};

bool ids_equal(const SubTileId &id, int32_t face, int32_t sub_i, int32_t sub_j) {
  return id.face == face && id.sub_i == sub_i && id.sub_j == sub_j;
}

void verify_edge(const SubTileId &me, int32_t my_rank, PendingEdge &p, int &failures) {
  // Independent re-derivation: what MUST the donor have sent, purely as a
  // function of (p.n.donor, p.n.donor_edge) -- computed HERE, not copied
  // from anything already known about `p.n` on this rank.
  const SubTileEdgeNeighbor back =
      subtile_edge_neighbor(p.n.donor, p.n.donor_edge, kLayoutX, kLayoutY);

  bool ok = true;
  ok &= ids_equal(p.n.donor, p.recv_msg.face, p.recv_msg.sub_i, p.recv_msg.sub_j);
  ok &= (p.recv_msg.rank == p.donor_rank);
  ok &= (p.recv_msg.edge == static_cast<int32_t>(p.n.donor_edge));
  ok &= (p.recv_msg.vector_sign_x == back.vector_sign_x);
  ok &= (p.recv_msg.vector_sign_y == back.vector_sign_y);
  ok &= (p.recv_msg.orientation == static_cast<int32_t>(back.orientation));
  ok &= (back.donor == me); // reciprocity: donor's own neighbor lookup must point back to me

  if (!ok) {
    std::fprintf(stderr,
                  "rank %d FAIL edge=%d: got face/i/j=%d/%d/%d rank=%d donor_edge=%d "
                  "sign=%d/%d orient=%d; expected face/i/j=%d/%d/%d rank=%d donor_edge=%d "
                  "sign=%d/%d orient=%d\n",
                  my_rank, static_cast<int>(p.mine), p.recv_msg.face, p.recv_msg.sub_i,
                  p.recv_msg.sub_j, p.recv_msg.rank, p.recv_msg.edge, p.recv_msg.vector_sign_x,
                  p.recv_msg.vector_sign_y, p.recv_msg.orientation, p.n.donor.face,
                  p.n.donor.sub_i, p.n.donor.sub_j, p.donor_rank, static_cast<int>(p.n.donor_edge),
                  back.vector_sign_x, back.vector_sign_y, static_cast<int32_t>(back.orientation));
    ++failures;
  }
}

void verify_corner(const SubTileId &me, int32_t my_rank, PendingCorner &p, int &failures) {
  // Independent re-derivation from the donor's own identity: the donor's
  // own corner `recip` must resolve back to `me`.
  SeamVectorTransform xf_back{};
  const SubTileId back = subtile_corner_donor(p.donor, p.recip, kLayoutX, kLayoutY, xf_back);

  bool ok = true;
  ok &= ids_equal(p.donor, p.recv_msg.face, p.recv_msg.sub_i, p.recv_msg.sub_j);
  ok &= (p.recv_msg.rank == p.donor_rank);
  ok &= (p.recv_msg.corner == p.recip);
  ok &= (back == me);

  if (!ok) {
    std::fprintf(stderr,
                  "rank %d FAIL corner=%d: got face/i/j=%d/%d/%d rank=%d corner=%d; "
                  "expected face/i/j=%d/%d/%d rank=%d corner=%d\n",
                  my_rank, p.corner, p.recv_msg.face, p.recv_msg.sub_i, p.recv_msg.sub_j,
                  p.recv_msg.rank, p.recv_msg.corner, p.donor.face, p.donor.sub_i, p.donor.sub_j,
                  p.donor_rank, p.recip);
    ++failures;
  }
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int world_size = 0, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size != kExpectedRanks) {
    if (world_rank == 0) {
      std::fprintf(stderr,
                    "FAIL: fv3_halo_subtile_mpi_routing_test requires exactly %d ranks "
                    "(layout_x=%d, layout_y=%d, 6 faces); got %d\n",
                    kExpectedRanks, kLayoutX, kLayoutY, world_size);
    }
    MPI_Finalize();
    return 1;
  }

  const SubTileId me = subtile_from_global_rank(world_rank, kLayoutX, kLayoutY);
  int failures = 0;

  // Sanity: rank<->subtile bijection must round-trip for MY rank (already
  // unit-tested in the abstract; here it gates whether the rest of this
  // rank's work is even meaningful).
  if (subtile_global_rank(me, kLayoutX, kLayoutY) != world_rank) {
    std::fprintf(stderr, "rank %d FAIL: rank<->subtile bijection did not round-trip\n",
                 world_rank);
    ++failures;
  }

  // ---- Phase 1: build all pending edge/corner exchanges (pure, local). ----
  const TileEdge edges[4] = {TileEdge::west, TileEdge::east, TileEdge::south, TileEdge::north};
  std::vector<PendingEdge> pending_edges;
  pending_edges.reserve(4);
  for (TileEdge e : edges) {
    PendingEdge p{};
    p.mine = e;
    p.n = subtile_edge_neighbor(me, e, kLayoutX, kLayoutY);
    p.donor_rank = subtile_global_rank(p.n.donor, kLayoutX, kLayoutY);
    p.send_msg = EdgeMsg{me.face, me.sub_i, me.sub_j, world_rank, static_cast<int32_t>(e),
                          p.n.vector_sign_x, p.n.vector_sign_y,
                          static_cast<int32_t>(p.n.orientation)};
    pending_edges.push_back(p);
  }

  int local_corner_count = 0;
  std::vector<PendingCorner> pending_corners;
  pending_corners.reserve(3);
  for (int c = 0; c < 4; ++c) {
    SeamVectorTransform xf{};
    const SubTileId donor = subtile_corner_donor(me, c, kLayoutX, kLayoutY, xf);
    if (donor == me) {
      ++local_corner_count;
      continue;
    }
    PendingCorner p{};
    p.corner = c;
    p.donor = donor;
    p.donor_rank = subtile_global_rank(donor, kLayoutX, kLayoutY);
    p.recip = subtile_corner_reciprocal_index(me, donor, kLayoutX, kLayoutY);
    if (p.recip < 0) {
      std::fprintf(stderr, "rank %d FAIL corner=%d: no reciprocal index found\n", world_rank, c);
      ++failures;
      continue;
    }
    p.send_msg = CornerMsg{me.face, me.sub_i, me.sub_j, world_rank, c};
    pending_corners.push_back(p);
  }
  if (local_corner_count != 1 || pending_corners.size() != 3) {
    std::fprintf(stderr,
                  "rank %d FAIL: expected exactly 1 local + 3 messaged corner, got %d + %zu\n",
                  world_rank, local_corner_count, pending_corners.size());
    ++failures;
  }

  // ---- Phase 2: post ALL Irecv/Isend up front, then Waitall. ----
  std::vector<MPI_Request> requests;
  requests.reserve(2 * (pending_edges.size() + pending_corners.size()));

  for (PendingEdge &p : pending_edges) {
    MPI_Request req;
    MPI_Irecv(&p.recv_msg, sizeof(p.recv_msg), MPI_BYTE, p.donor_rank,
               subtile_edge_tag(p.n.donor_edge), MPI_COMM_WORLD, &req);
    requests.push_back(req);
  }
  for (PendingCorner &p : pending_corners) {
    MPI_Request req;
    MPI_Irecv(&p.recv_msg, sizeof(p.recv_msg), MPI_BYTE, p.donor_rank,
               subtile_corner_tag(p.recip), MPI_COMM_WORLD, &req);
    requests.push_back(req);
  }
  for (PendingEdge &p : pending_edges) {
    MPI_Request req;
    MPI_Isend(&p.send_msg, sizeof(p.send_msg), MPI_BYTE, p.donor_rank,
               subtile_edge_tag(p.mine), MPI_COMM_WORLD, &req);
    requests.push_back(req);
  }
  for (PendingCorner &p : pending_corners) {
    MPI_Request req;
    MPI_Isend(&p.send_msg, sizeof(p.send_msg), MPI_BYTE, p.donor_rank,
               subtile_corner_tag(p.corner), MPI_COMM_WORLD, &req);
    requests.push_back(req);
  }

  const int wait_rc =
      MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
  if (wait_rc != MPI_SUCCESS) {
    std::fprintf(stderr, "rank %d FAIL: MPI_Waitall returned %d\n", world_rank, wait_rc);
    ++failures;
  }

  // ---- Phase 3: verify every received message against an independent,
  // locally re-derived expectation. ----
  for (PendingEdge &p : pending_edges) {
    verify_edge(me, world_rank, p, failures);
  }
  for (PendingCorner &p : pending_corners) {
    verify_corner(me, world_rank, p, failures);
  }

  int total_failures = 0;
  MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  const int rc = (total_failures == 0) ? 0 : 1;
  if (world_rank == 0) {
    if (total_failures == 0) {
      std::printf(
          "PASS: 24-rank sub-tile halo MPI routing test (%d ranks, layout=%dx%d) -- all "
          "edge and corner exchanges verified via independent local re-derivation on the "
          "receiving rank.\n",
          world_size, kLayoutX, kLayoutY);
      std::printf(
          "  Covers: real cross-process MPI routing (not just pure-function logic), "
          "collision-free tags in a live MPI_Comm, reciprocity of donor identity/edge/"
          "corner index/orientation/vector sign.\n");
      std::printf(
          "  Does NOT cover: CUDA halo kernel application, actual field data movement, "
          "numerical validation, the production comm_size==6 bridge gate.\n");
    } else {
      std::printf("FAIL: %d total failure(s) across %d ranks (see stderr above)\n",
                  total_failures, world_size);
    }
  }

  MPI_Finalize();
  return rc;
}
