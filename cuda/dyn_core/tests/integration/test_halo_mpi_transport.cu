//===========================================================================
// Milestone 3A.15 §11-§13 -- permanent SIX-RANK MPI halo transport test.
//
// One cubed-sphere tile per MPI process, which is the real GEOS/FV3 production
// decomposition (FV_NX=1, FV_NY=6). Run with exactly six ranks:
//
//     mpirun -np 6 ./fv3_halo_mpi_transport
//
// The exchange runs through the real production path with the real MPI backend:
//
//     launch_halo_pack  ->  MpiHaloTransport::transfer  ->  launch_halo_unpack
//
// i.e. D2H of the packed send buffer, MPI_Irecv/MPI_Isend/MPI_Waitall, H2D into
// the recv buffer. Nothing about the exchange's CONTENT is re-implemented here or
// in the backend; the pack and unpack kernels, the seam algebra and the sign table
// are the same ones the single-process path uses.
//
// -------------------------------------------------------------------------
// WHY THIS IS NOT A SECOND COPY OF test_halo_exchange_multitile.cu
// -------------------------------------------------------------------------
//
// That test drives all six tiles from ONE process, so its "transfer" is a
// device-to-device read out of a peer's send buffer and every tile's data is
// reachable from the verifying code. Here each rank can see only its OWN tile, so
// a wrong-donor defect cannot be caught by comparing two local arrays -- and that
// is precisely the class of defect a cross-process transport can introduce:
// mismatched MPI tags, a wrong rank<->tile map, a partner remap applied on the
// send side instead of the receive side, or a slot offset computed from the wrong
// group.
//
// It is caught anyway, because the oracle travels INSIDE THE DATA. Every array
// element on every rank carries its own provenance, encoded losslessly in its
// value (identical scheme to the multi-tile test):
//
//     value(tile, i, j, k) = (((tile*64 + (i+8))*64 + (j+8))*8 + k)
//
// After the exchange each received cell is DECODED back to (tile, i, j, k) and
// checked against the tile, index and sign the seam requires. A rank therefore
// verifies its neighbours' contribution without ever holding their arrays. What
// that makes independently detectable:
//
//   * message crossed between the wrong ranks   -> decoded tile is wrong
//   * tag collision between groups/edges/fields -> decoded index or field is wrong
//   * partner remap missing on a rotated seam   -> decoded staggering shift wrong
//   * reversal or sign wrong                    -> decoded along-index/sign wrong
//   * an edge never sent or never received      -> decodes to the RECEIVER's own
//                                                  tile, because every array is
//                                                  pre-filled with its owner's own
//                                                  encoding, so a no-op cannot pass
//   * slot offset computed for the wrong group  -> poison (0xFF) or another
//                                                  group's values decode wrong
//
// As in the multi-tile test, expected_sign() below is a LOCAL transcription of the
// FMS table, deliberately NOT a call to seam_component_sign(). An earlier version
// of that test computed its expected sign by calling the function under test and
// consequently survived a negative control that forced the sign to +1. Restating
// the table is the oracle; calling the header would be circular.
//
// -------------------------------------------------------------------------
// WHAT IS COVERED
// -------------------------------------------------------------------------
//
//   H_W_IN        scalar, CENTER                       (§13 scalar)
//   H_DWIND_IN    u,v DGRID_NE -- slot_x is NORTH-staggered
//   H_CWIND       uc,vc CGRID_NE -- slot_x is EAST-staggered
//                 Running BOTH wind groups is the point: a sign table keyed on
//                 staggering instead of on the argument slot passes one and
//                 fails the other.
//   H_DSCALAR     4 fields, halo width 3               (§13 multi-ring)
//   H_DWIND_SEAM  E11, ring h = 0, destinations are OWNED cells and each field
//                 accepts exactly one edge (dest_edges)
//
// Every group additionally asserts that no owned cell OTHER than that group's own
// declared destinations changed value -- which for the ordinary groups means no
// owned cell at all, and for E11 means exactly u's north line and v's east line
// and nothing else, matching the footprint measured by
// tools/savepoints/probe_e11_map.py.
//
// -------------------------------------------------------------------------
// WHAT THIS TEST DOES NOT COVER
// -------------------------------------------------------------------------
//
//   * The corner halo BLOCKS (i and j both outside the owned range). The four
//     edge strips are disjoint and stop short of them by construction; nothing
//     corner-specific is sent by any transport (§8).
//   * Numerical validation against authoritative Fortran savepoints: NOT RUN
//     here. That is the dyn_core-level comparison (§14), not a transport test.
//===========================================================================
#include "fv3_gpu_halo_exchange.hpp"
#include "fv3_gpu_halo_mpi.hpp"
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

int failures = 0;
long long checks = 0;
int my_rank = -1;
int32_t my_tile = 0;

/* ---------------------------------------------------------------------------
 * Milestone 3A.18 §1 -- failure observability.
 *
 * The original version reported ONE cell per (edge, field) and then skipped the
 * rest of the block, so an intermittent failure could not be told apart from a
 * whole missing message: the reported cell was always simply the first in
 * iteration order (h=1, a=1, k=1), and the check count silently dropped by the
 * size of the unchecked remainder. Both modes below count EVERY cell, so
 * `checks` is now the same in a failing run as in a passing one.
 *
 *   FV3_HALO_TEST_DEBUG=1            report every failing cell in a block, plus
 *                                    the message identity, the buffer offsets
 *                                    and a raw-bit classification of what the
 *                                    destination actually holds.
 *   FV3_HALO_TEST_MEMSET_ORDERED=1   issue the buffer poison on the exchange
 *                                    stream rather than the legacy default
 *                                    stream (see exchange_group).
 * ------------------------------------------------------------------------- */
bool dbg = false;
/* Reproduce the 3A.18 defect on demand: issue this test's own device-buffer setup
 * on the LEGACY stream, unordered against the exchange stream, the way it was
 * written before 3A.18 §1. Default OFF -- the ordered path is the correct one. */
bool legacy_setup = false;
int printed = 0;
constexpr int kMaxPrinted = 60;

bool may_print() { return printed < kMaxPrinted && ++printed <= kMaxPrinted; }

/* Per-CELL prints get their own budget so a block that fails in bulk cannot
 * starve the per-block summary, which is the line that carries the mechanism. */
int printed_cells = 0;
bool may_print_cell() { return printed_cells < 24 && ++printed_cells <= 24; }

void check(bool ok, const char *what) {
  ++checks;
  if (!ok) {
    if (may_print()) std::printf("  [rank %d] FAIL: %s\n", my_rank, what);
    ++failures;
  }
}

/* What a wrong destination cell actually contains. The three cases have
 * completely different causes and the decode alone cannot tell them apart --
 * 0xFF-filled fv_real is a NaN, and llround(NaN) is LLONG_MIN on x86-64, which
 * decodes to exactly "tile 0 (-8,-8,0)", the same text a genuine 0.0 produces. */
const char *classify(fv_real v) {
  unsigned char b[sizeof(fv_real)];
  std::memcpy(b, &v, sizeof(fv_real));
  bool all_ff = true;
  for (size_t t = 0; t < sizeof(fv_real); ++t)
    if (b[t] != 0xFFu) all_ff = false;
  if (all_ff) return "POISON-REMAINS (0xFF; device recv slot not overwritten by the H2D)";
  if (v == (fv_real)0) return "ZERO (host staging slot never written by MPI)";
  return "WRONG-VALUE (arrived, but from the wrong place)";
}

/* Diagnostic transcription of the backend's private message_tag(); reporting
 * only, never used to decide anything. */
int diag_tag(uint32_t group, TileEdge edge, int field) {
  return (static_cast<int>(group) * 4 + static_cast<int>(edge)) * max_halo_group_fields +
         field;
}

/* Same geometry as the single-process multi-tile test, so a failure here versus
 * there is a transport difference and not a geometry difference. A 3-ring halo is
 * strictly narrower than the owned block, so a strip can never be confused with a
 * whole face. */
constexpr int64_t NX = 6, NY = 6, NPZ = 4, NG = 3;
constexpr int64_t EXT_I = NX + 1 + 2 * NG;
constexpr int64_t EXT_J = NY + 1 + 2 * NG;
constexpr int64_t LO = 1 - NG;

/* Lossless provenance encoding. Max ~206k, inside float's exact-integer range
 * (2^24), so the same source is exact in R4 and R8. */
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
  void create(int tile, int64_t nk_in, cudaStream_t s) {
    nk = nk_in;
    host.assign((size_t)(EXT_I * EXT_J * nk), (fv_real)0);
    /* Pre-fill EVERYTHING, halo included, with this tile's OWN encoding, so a cell
     * the exchange fails to write decodes to the receiver's own tile and is caught
     * rather than reading as whatever happened to be there. */
    for (int64_t k = 1; k <= nk; ++k)
      for (int64_t j = LO; j <= NY + 1 + NG; ++j)
        for (int64_t i = LO; i <= NX + 1 + NG; ++i)
          host[(size_t)((i - LO) + (j - LO) * EXT_I + (k - 1) * EXT_I * EXT_J)] =
              (fv_real)encode(tile, i, j, k);
    if (cudaMalloc(&dev, host.size() * sizeof(fv_real)) != cudaSuccess) {
      check(false, "cudaMalloc field array");
      dev = nullptr;
      return;
    }
    /* ORDERING (3A.18 §1). A synchronous cudaMemcpy H2D from PAGEABLE memory
     * returns once the source has been staged for DMA, not once the device write
     * has landed, and it lands in the LEGACY stream -- which does not implicitly
     * synchronize with a cudaStreamNonBlocking stream. Issuing it there therefore
     * leaves this initialization unordered against launch_halo_pack, so the pack
     * may read the field array before it is populated. Fresh cudaMalloc memory
     * commonly reads as 0.0, and 0.0 decodes to "tile 0 (-8,-8,0)" -- exactly the
     * recorded intermittent signature. Enqueue it on the exchange stream instead. */
    if (s)
      cudaMemcpyAsync(dev, host.data(), host.size() * sizeof(fv_real),
                      cudaMemcpyHostToDevice, s);
    else
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

/* The expected sign, transcribed LOCALLY from FMS group_update_pack.inc and keyed
 * on the RECEIVER's own halo edge. Deliberately not seam_component_sign().
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

bool interface_level_field(FieldId id) {
  return id == FieldId::gz || id == FieldId::zh || id == FieldId::pkc;
}

/* Exchange ONE group on THIS rank's tile through the MPI transport, then decode
 * and verify every cell the group is required to have written.
 *
 * Every rank must call this for the same groups in the same order: transfer() is
 * point-to-point but agree_on_layout() is collective, and the tag space is
 * per-group, so a rank that skipped a group would leave its neighbours' receives
 * unmatched. */
void exchange_group(HaloTransport *tr, const TopologyView &topo, uint32_t group,
                    const char *label, fv_real *send, fv_real *recv, int64_t buf_capacity,
                    cudaStream_t stream) {
  const HaloGroupSpec spec = halo_group_spec(group);
  if (!spec.live) {
    if (my_rank == 0) std::printf("  (%s is not live; skipped)\n", label);
    return;
  }
  if (my_rank == 0)
    std::printf("\n--- %s: %d field(s), rings %d..%d, 6 ranks ---\n", label,
                spec.field_count, spec.ring_begin, spec.halo_width);

  const HaloGeometry geom{NX, NY, NPZ};
  const int64_t buf_n = group_buffer_elements(spec, geom);
  check(buf_n <= buf_capacity, "group fits the buffers sized at attach");
  if (buf_n > buf_capacity) return;

  TileArray arr[max_halo_group_fields];
  for (int n = 0; n < spec.field_count; ++n)
    arr[n].create((int)my_tile, interface_level_field(spec.fields[n].id) ? NPZ + 1 : NPZ,
                  legacy_setup ? nullptr : stream);

  /* Poison both buffers so an unwritten slot cannot masquerade as a plausible
   * zero, and so a slot offset computed for the wrong group shows up as garbage
   * rather than as another group's still-valid values.
   *
   * ORDERING (3A.18 §1). `stream` is cudaStreamNonBlocking, which by definition
   * does NOT implicitly synchronize with the legacy default stream, and plain
   * cudaMemset on device memory is asynchronous with respect to the host and is
   * issued into the legacy stream. Issuing the poison there therefore leaves it
   * unordered against the pack, the staging H2D and the unpack that follow on
   * `stream` -- the poison is allowed to land AFTER them and re-poison a region
   * the exchange had already filled correctly. cudaMemsetAsync on `stream` puts
   * it in the same ordered sequence as everything that reads and writes these
   * buffers. The legacy-stream form is retained under an env switch purely so
   * the defect can be demonstrated rather than asserted. */
  if (legacy_setup) {
    cudaMemset(send, 0xFF, (size_t)buf_n * sizeof(fv_real));
    cudaMemset(recv, 0xFF, (size_t)buf_n * sizeof(fv_real));
  } else {
    cudaMemsetAsync(send, 0xFF, (size_t)buf_n * sizeof(fv_real), stream);
    cudaMemsetAsync(recv, 0xFF, (size_t)buf_n * sizeof(fv_real), stream);
  }

  HaloTileFields fields{};
  for (int n = 0; n < spec.field_count; ++n) {
    fields.view[n] = arr[n].view();
    fields.present[n] = 1;
  }

  fv3_gpu_error err{};
  const char *api = "test_halo_mpi_transport";

  /* Collective, and the reason a cross-rank field-count disagreement cannot go
   * unnoticed: the buffers are indexed by compacted field SLOT. */
  check(tr->agree_on_layout(group, spec.field_count),
        "all six ranks agree on this group's compacted field layout");

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

  /* ------------------------------------------------------------------
   * VERIFY. Decode every destination cell, then prove nothing else in the
   * owned box moved.
   * ------------------------------------------------------------------ */
  std::vector<char> written[max_halo_group_fields];
  for (int n = 0; n < spec.field_count; ++n) {
    arr[n].fetch();
    written[n].assign((size_t)(EXT_I * EXT_J), 0);
  }

  /* The staged recv buffer as the unpack actually saw it, so a failing block can
   * be attributed to the buffer content rather than to the unpack's indexing.
   * §7: every slot lies inside the region the transport copies, because
   * slot_base(max) + slot == 4*field_count*slot == group_buffer_elements(). */
  const int32_t g_width = halo_ring_count(spec);
  const int64_t g_slot = group_slot_elements(geom, g_width);
  std::vector<fv_real> recv_host((size_t)buf_n, (fv_real)0);
  cudaMemcpy(recv_host.data(), recv, (size_t)buf_n * sizeof(fv_real),
             cudaMemcpyDeviceToHost);
  if (dbg && my_rank == 0)
    std::printf("  [dbg] width=%d slot=%lld elems buffer=%lld elems (%zu B/elem)\n",
                (int)g_width, (long long)g_slot, (long long)buf_n, sizeof(fv_real));

  long long verified = 0;
  for (int q = 0; q < 4; ++q) {
    const TileEdge mine = static_cast<TileEdge>(q);
    const EdgeNeighbor &nb = topo.neighbors[q];
    const bool reversed = seam_is_reversed(mine, nb.edge);

    for (int n = 0; n < spec.field_count; ++n) {
      const HaloFieldSpec &fs = spec.fields[n];
      /* E11's fields each accept exactly one edge, because their destinations are
       * OWNED cells; the other three edges are packed and sent but deliberately
       * not unpacked. See HaloFieldSpec::dest_edges. */
      if (!(fs.dest_edges & (1u << q))) continue;
      const StaggerShift src = source_shift(fs.stagger, reversed);
      const int64_t along = edge_seam_extent(mine, NX, NY, fs.stagger);
      const int64_t donor = donor_seam_extent(nb.edge, NX, NY, src);
      const int want_sign = expected_sign(fs.role, mine, reversed);
      const int64_t nk = arr[n].nk;
      /* Message identity for this block, both ends, for the §5 reconciliation. */
      int sn = n;
      if (reversed && fs.role != VectorRole::scalar) sn = field_slot(spec, fs.partner);
      const int64_t my_slot_off = slot_base(geom, g_width, n, mine);
      long long blockfail = 0, blockcells = 0;

      for (int h = spec.ring_begin; h <= spec.halo_width; ++h) {
        for (int64_t a = 1; a <= along; ++a) {
          int64_t di = 0, dj = 0, si = 0, sj = 0;
          halo_dest_index(mine, a, h, NX, NY, fs.stagger, di, dj);
          const int64_t ap = seam_along_index(a, donor, reversed);
          halo_src_index(nb.edge, ap, h, NX, NY, src, si, sj);
          written[n][(size_t)((di - LO) + (dj - LO) * EXT_I)] = 1;

          for (int64_t k = 1; k <= nk; ++k) {
            const double got = arr[n].at(di, dj, k);
            const double mag = (got < 0) ? -got : got;
            int gt = 0; int64_t gi = 0, gj = 0, gk = 0;
            decode(mag, gt, gi, gj, gk);
            ++verified;

            ++blockcells;
            ++checks;

            if (gt != nb.tile || gi != si || gj != sj || gk != k ||
                (got < 0) != (want_sign < 0)) {
              ++failures;
              ++blockfail;
              /* Every failing cell, not just the first: one missing cell and a
               * whole missing message have very different causes and the old
               * suppression made them indistinguishable. */
              if ((dbg || blockfail == 1) && may_print_cell())
                std::printf("  [rank %d] FAIL: tile %d edge %s field %d h=%d a=%lld k=%lld\n"
                            "        want tile %d (%lld,%lld,%lld) sign %+d\n"
                            "        got  tile %d (%lld,%lld,%lld) sign %+d  [%s]\n",
                            my_rank, (int)my_tile, edge_name(mine), n, h, (long long)a,
                            (long long)k, (int)nb.tile, (long long)si, (long long)sj,
                            (long long)k, want_sign, gt, (long long)gi, (long long)gj,
                            (long long)gk, (got < 0) ? -1 : +1,
                            classify(arr[n].host[(size_t)((di - LO) + (dj - LO) * EXT_I +
                                                          (k - 1) * EXT_I * EXT_J)]));
            }
          }
        }
      }

      /* Per-block summary: the message this block should have carried, both
       * ends' view of it, and what the staged recv slot actually held. */
      if (blockfail && may_print()) {
        long long sp = 0, sz = 0;
        for (int64_t t = 0; t < g_slot; ++t) {
          const fv_real v = recv_host[(size_t)(my_slot_off + t)];
          if (v != v) ++sp;                 /* NaN: still the 0xFF poison */
          else if (v == (fv_real)0) ++sz;   /* the backend's attach-time zero */
        }
        std::printf("  [rank %d] BLOCK FAIL %lld/%lld cells | group %u %s | my tile %d "
                    "edge %s field %d <- donor tile %d rank %d edge %s slot %d | "
                    "tag %d | my recv slot offset %lld of %lld elems, slot %lld elems "
                    "| slot census: poison %lld, zero %lld, other %lld\n",
                    my_rank, blockfail, blockcells, group, label, (int)my_tile,
                    edge_name(mine), n, (int)nb.tile, nb.rank, edge_name(nb.edge), sn,
                    diag_tag(group, nb.edge, sn), (long long)my_slot_off,
                    (long long)buf_n, (long long)g_slot, sp, sz, g_slot - sp - sz);
      }
    }
  }

  /* Nothing in the owned box may have changed except this group's own declared
   * destinations. For the ordinary groups that is "no owned cell at all"; for E11
   * it is exactly u's north line and v's east line, which is the footprint
   * probe_e11_map.py measured against authoritative Fortran. */
  for (int n = 0; n < spec.field_count; ++n) {
    const HaloFieldSpec &fs = spec.fields[n];
    long long moved = 0;
    bool first = true;
    for (int64_t k = 1; k <= arr[n].nk; ++k)
      for (int64_t j = 1; j <= NY + fs.stagger.j; ++j)
        for (int64_t i = 1; i <= NX + fs.stagger.i; ++i) {
          if (written[n][(size_t)((i - LO) + (j - LO) * EXT_I)]) continue;
          if (arr[n].at(i, j, k) == (fv_real)encode((int)my_tile, i, j, k)) continue;
          ++moved;
          if (first) {
            first = false;
            const fv_real v = arr[n].host[(size_t)((i - LO) + (j - LO) * EXT_I +
                                                   (k - 1) * EXT_I * EXT_J)];
            const double mag = ((double)v < 0) ? -(double)v : (double)v;
            int gt = 0; int64_t gi = 0, gj = 0, gk = 0;
            decode(mag, gt, gi, gj, gk);
            if (may_print())
              std::printf("  [rank %d] OWNED-BOX MOVED: group %u %s field %d first at "
                          "(%lld,%lld,%lld): want tile %d (%lld,%lld,%lld), got tile %d "
                          "(%lld,%lld,%lld) [%s]\n",
                          my_rank, group, label, n, (long long)i, (long long)j,
                          (long long)k, (int)my_tile, (long long)i, (long long)j,
                          (long long)k, gt, (long long)gi, (long long)gj, (long long)gk,
                          classify(v));
          }
        }
    check(moved == 0, "owned cells outside this group's destinations are untouched");
    if (moved && may_print())
      std::printf("  [rank %d] OWNED-BOX MOVED: group %u %s field %d -> %lld owned cells\n",
                  my_rank, group, label, n, moved);
  }

  if (my_rank == 0)
    std::printf("  rank 0 decoded and verified %lld received values (each rank does "
                "the same for its own tile)\n", verified);

  for (int n = 0; n < spec.field_count; ++n) arr[n].destroy();
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (size != cubed_sphere_tiles) {
    if (my_rank == 0)
      std::printf("FAIL: this test requires exactly %d MPI ranks (one per cubed-sphere "
                  "tile); got %d. Run with mpirun -np %d.\n",
                  cubed_sphere_tiles, size, cubed_sphere_tiles);
    MPI_Finalize();
    return 1;
  }

  /* §2: the production decomposition (FV_NX=1, FV_NY=6) maps MPI rank r to tile
   * r+1. The transport does not TRUST that -- it gathers the map and validates it
   * against the installed topology -- but this test is what produces it. */
  my_tile = (int32_t)(my_rank + 1);

  dbg = std::getenv("FV3_HALO_TEST_DEBUG") != nullptr;
  legacy_setup = std::getenv("FV3_HALO_TEST_LEGACY_SETUP") != nullptr;

  /* Prompt 43 Stage 6: which HaloTransport backend this run exercises. Default
   * unchanged (host_staged); FV3_HALO_TEST_TRANSPORT=cuda_aware selects the
   * device-pointer path added for the C512 A/B. Same test, same verification
   * logic, only the transport under test changes -- create_mpi_transport()
   * itself refuses (no silent fallback) if cuda_aware is requested and the MPI
   * runtime lacks CUDA-buffer support (e.g. I_MPI_OFFLOAD=1 not set). */
  TransportMode transport_mode = TransportMode::host_staged;
  if (const char *tm = std::getenv("FV3_HALO_TEST_TRANSPORT")) {
    if (std::string(tm) == "cuda_aware") transport_mode = TransportMode::cuda_aware;
    else if (std::string(tm) != "host_staged" && std::string(tm) != "") {
      if (my_rank == 0)
        std::printf("FAIL: unknown FV3_HALO_TEST_TRANSPORT=%s (want host_staged or "
                    "cuda_aware)\n", tm);
      MPI_Finalize();
      return 1;
    }
  }

  if (my_rank == 0) {
    std::printf("=== Milestone 3A.15 six-rank MPI halo transport ===\n");
    std::printf("Precision: %s (sizeof(fv_real)=%zu), %d ranks, one tile each\n",
                sizeof(fv_real) == 8 ? "R8" : "R4", sizeof(fv_real), size);
    std::printf("Transport requested: %s\n",
                transport_mode == TransportMode::cuda_aware ? "cuda_aware"
                                                              : "host_staged");
    std::printf("Diagnostics: debug=%s  device-buffer setup=%s\n", dbg ? "ON" : "off",
                legacy_setup ? "LEGACY STREAM (3A.18 defect reproduction)"
                             : "stream-ordered on the exchange stream");
  }

  /* A missing device must skip on EVERY rank or the survivors deadlock in the
   * first collective. */
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

  /* The groups under test, in an order every rank executes identically. */
  struct GroupCase { uint32_t group; const char *label; };
  const GroupCase cases[] = {
      {FV3_GPU_H_W_IN, "H_W_IN (w, CENTER scalar)"},
      {FV3_GPU_H_DWIND_IN, "H_DWIND_IN (u,v DGRID_NE; slot_x is NORTH-staggered)"},
      {FV3_GPU_H_CWIND, "H_CWIND (uc,vc CGRID_NE; slot_x is EAST-staggered)"},
      {FV3_GPU_H_DSCALAR, "H_DSCALAR (q_con,delp,pt,cappa; 4 fields, width 3)"},
      {FV3_GPU_H_DWIND_SEAM, "H_DWIND_SEAM (E11, ring 0, owned destinations)"},
  };

  /* Size the halo buffers ONCE, for the largest group under test, exactly as the
   * runtime's attach_halo_transport does -- the transport contract forbids
   * allocating inside an exchange, and the pinned host buffers the MPI backend
   * allocates at attach are sized from this. */
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

  int32_t rank_of_tile[cubed_sphere_tiles] = {0, 1, 2, 3, 4, 5};
  TopologyView topo{};
  if (!make_cubed_sphere_topology(my_tile, NX + 1, NY + 1, rank_of_tile, topo)) {
    std::printf("  [rank %d] FAIL: make_cubed_sphere_topology(tile %d)\n", my_rank,
                (int)my_tile);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  /* Collective: the factory queries size/rank and gathers the rank->tile map.
   * No silent fallback: if cuda_aware was requested and this MPI runtime lacks
   * CUDA-buffer support, create_mpi_transport() returns null with `why` set
   * rather than substituting host_staged. */
  std::string why;
  HaloTransport *tr = create_mpi_transport(
      MPI_Comm_c2f(MPI_COMM_WORLD), my_tile, transport_mode, &why);
  if (!tr) {
    std::printf("  [rank %d] FAIL: create_mpi_transport: %s\n", my_rank, why.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  if (my_rank == 0) std::printf("transport: %s\n", tr->name());

  HaloMemberResources res{};
  res.send = send;
  res.recv = recv;
  res.buffer_elements = buf_capacity;
  res.transfer_stream = stream;
  if (!tr->attach(my_tile, res)) {
    std::printf("  [rank %d] FAIL: attach: %s\n", my_rank, tr->attach_failure_reason());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  for (const GroupCase &c : cases)
    exchange_group(tr, topo, c.group, c.label, send, recv, buf_capacity, stream);

  tr->detach(my_tile);
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
      std::printf("Six-rank MPI halo transport: FAIL\n");
    } else {
      std::printf("Six-rank MPI halo transport: PASS (%s)\n",
                  transport_mode == TransportMode::cuda_aware ? "cuda_aware" : "host_staged");
      std::printf("  * GPU EXECUTED across SIX SEPARATE MPI PROCESSES: pack,\n");
      std::printf("    %s MPI transfer and seam-transforming unpack ran on a\n",
                  transport_mode == TransportMode::cuda_aware ? "cuda-aware" : "host-staged");
      std::printf("    real device on every rank, and every received value was decoded\n");
      std::printf("    back to its originating tile, index and sign.\n");
      std::printf("  * Covered: scalar, both wind staggerings, 4-field width-3\n");
      std::printf("    multi-ring, and E11's ring-0 owned-cell reconciliation.\n");
      std::printf("  * NOT covered here: corner halo blocks; numerical validation\n");
      std::printf("    against authoritative Fortran savepoints.\n");
    }
  }

  MPI_Finalize();
  return global[1] ? 1 : 0;
}
