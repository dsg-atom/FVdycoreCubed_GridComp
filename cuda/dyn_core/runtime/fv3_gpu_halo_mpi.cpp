/* Milestone 3A.15 §4-§7 -- the host-staged MPI halo transport.
 *
 * One cubed-sphere tile per MPI rank. This is the transport the real GEOS/FV3
 * production decomposition needs (FV_NX=1, FV_NY=6: six separate processes), and it
 * is the piece milestone 3A.10 named as the only one an MPI build has to replace:
 * "HaloPeerGroup is the only piece an MPI transport replaces"
 * (milestone_3a10_gpu_halo_exchange.md:955).
 *
 * NOTHING NUMERICAL IS DECIDED HERE. Which cells a group exchanges, which ring,
 * which staggering, which seam rotation, which sign, and -- critically -- which
 * DONOR SLOT a rotated seam must read all come from fv3_gpu_halo_exchange.hpp and
 * are shared verbatim with the in-process backend. In particular the partner remap
 * below calls the same field_slot()/seam_is_reversed() that launch_halo_transfer
 * calls; a private second copy of that rule is exactly the class of defect this
 * project has repeatedly found to be invisible downstream (it would put the wrong
 * wind component on a rotated seam and still deliver a plausible number).
 *
 * THE ONE THING THIS FILE ADDS TO THE MODEL B SEQUENCE
 *
 * Model B's transfer is a device-to-device read out of the donor's send buffer,
 * which is possible only because all six tiles are in one address space. Across
 * processes the bytes have to travel, so per group:
 *
 *   1  D2H the whole packed send region on the caller's stream (stream-ordered
 *      after the pack, which the contract guarantees is already enqueued)
 *   2  wait for THAT COPY ONLY, on this transport's own event -- not the stream,
 *      never the device (fv3_gpu_halo_transport.hpp's contract, and
 *      project_architecture.md's "no hidden device synchronization")
 *   3  MPI_Irecv x (4 edges * fields), then MPI_Isend x (4 edges * fields),
 *      then one MPI_Waitall
 *   4  H2D the whole recv region on the same stream, so the unpack the runtime
 *      enqueues next is stream-ordered behind it
 *
 * Receives are posted before sends so a slow neighbour cannot force the MPI
 * implementation to buffer an unexpected message the size of a halo group.
 *
 * The single host wait in step 2 is the intrinsic cost of host staging ("Stage 1"
 * of project_architecture.md's Communication Evolution) and the reason Stage 2
 * (CUDA-aware MPI, handing MPI the device pointers) exists as a later step. It does
 * not weaken persistent residency: only the packed EDGE buffers cross the bus, never
 * model state, and no device-resident field is ever round-tripped.
 *
 * WHY THE WHOLE REGION MOVES IN ONE COPY, NOT ONE PER SLOT
 *
 * Slots are laid out slot_base(g, w, field, edge) = (field*4 + edge) * slot, so a
 * group's used region is CONTIGUOUS from the buffer base and its length is exactly
 * group_buffer_elements(). Two bulk copies per group is therefore both correct and
 * the minimum. (Model B copies per (field, edge) instead because it must gather from
 * five different donors' buffers; here the peer traffic is MPI's job.)
 * ------------------------------------------------------------------------- */

#include "fv3_gpu_halo_mpi.hpp"

#include "fv3_gpu_errors.hpp"
#include "fv3_gpu_halo_topology.hpp"
#include "fv3_gpu_internal.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
/* MPIX_Query_cuda_support() (Prompt 43's cuda_aware capability check) is
 * declared directly in Intel MPI's mpi.h (confirmed against mpi/impi/2021.13),
 * but Open MPI declares it only in the mpi-ext.h extension header -- OPEN_MPI
 * is the macro Open MPI's own mpi.h defines to 1, so this only pulls in the
 * extra header for that implementation and is a no-op for Intel MPI/MPICH. */
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif

#include <array>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace fv3::gpu::halo {
namespace {

constexpr int kGroups = FV3_GPU_H_GROUP_LAST + 1;

/* One MPI message per (group, edge, field). The tag has to be built from the
 * SENDER's own (edge, field) and reconstructed by the receiver from the donor's
 * (edge, field) -- which the receiver knows, because the seam tells it which of the
 * donor's edges and which of the donor's field slots feeds each of its own slots.
 *
 * Max value: ((FV3_GPU_H_GROUP_LAST*4)+3)*max_halo_group_fields+3 = 207, far inside
 * the MPI_TAG_UB every implementation guarantees (>= 32767). */
inline int message_tag(uint32_t group, TileEdge edge, int field) noexcept {
  return (static_cast<int>(group) * 4 + static_cast<int>(edge)) * max_halo_group_fields +
         field;
}

inline MPI_Datatype fv_real_datatype() noexcept {
  return (sizeof(fv_real) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;
}

class MpiHaloTransport final : public HaloTransport {
 public:
  /* Prompt 112: `size`/`tile_of_rank` are now measured, arbitrary-size
   * inputs (a std::vector, not a fixed 6-slot std::array) rather than a
   * hardcoded 6-tile assumption baked into storage. create_mpi_transport()
   * still only ever calls this with size==cubed_sphere_tiles==6 (its own
   * gate, unchanged, below); create_mpi_transport_for_testing() is the one
   * new caller that can supply a larger size, for Stage 13's synthetic
   * multi-rank-per-face test only. */
  MpiHaloTransport(MPI_Comm comm, int32_t tile, int rank, int size,
                   std::vector<int32_t> tile_of_rank, TransportMode mode) noexcept
      : comm_(comm), tile_(tile), rank_(rank), size_(size),
        tile_of_rank_(std::move(tile_of_rank)), mode_(mode) {
    layout_.fill(-1);
  }

  ~MpiHaloTransport() override { release_host_buffers(); }

  const char *name() const noexcept override {
    return mode_ == TransportMode::cuda_aware ? "mpi_cuda_aware" : "mpi_host_staged";
  }

  bool attach(int32_t tile, const HaloMemberResources &r) noexcept override {
    if (attached_) {
      attach_why_ = "this MPI halo transport already has its tile attached; one rank "
                    "carries exactly one tile in the production decomposition";
      return false;
    }
    if (tile != tile_) {
      attach_why_ = "the attaching context's tile is not the tile this rank was "
                    "created for; rank->tile mapping must not differ between the "
                    "transport and the installed topology";
      return false;
    }
    if (r.buffer_elements <= 0 || !r.transfer_stream) {
      attach_why_ = "the attaching context supplied no halo buffer size or no "
                    "transfer stream";
      return false;
    }
    /* cuda_aware mode passes the DEVICE send/recv buffers straight to MPI and never
     * stages through host memory, so it allocates no pinned buffers here -- Prompt
     * 43's whole point is removing this D2H/H2D pair from the critical path. */
    if (mode_ == TransportMode::host_staged) {
      /* Sized ONCE here, for the largest live group, exactly as the device buffers
       * are -- the transport contract forbids allocating inside an exchange, and a
       * cudaHostAlloc in the acoustic loop would serialize the device just as a
       * cudaMalloc would. Pinned rather than ordinary host memory so both bulk copies
       * are asynchronous DMA rather than a staged pageable copy. */
      const size_t bytes = static_cast<size_t>(r.buffer_elements) * sizeof(fv_real);
      if (cudaHostAlloc(reinterpret_cast<void **>(&h_send_), bytes, cudaHostAllocDefault) !=
              cudaSuccess ||
          cudaHostAlloc(reinterpret_cast<void **>(&h_recv_), bytes, cudaHostAllocDefault) !=
              cudaSuccess) {
        release_host_buffers();
        attach_why_ = "could not allocate pinned host staging buffers for the MPI halo "
                      "transport";
        return false;
      }
      /* Zeroed for the same reason the device buffers are (milestone 3A.10 §20): a
       * group whose field_count is below max_halo_group_fields leaves a slot tail that
       * is copied but never read, and initcheck is right to call an uninitialized
       * copy source a defect even when the value is never consumed. Across MPI that
       * tail is also literally sent, so leaving it undefined would put uninitialized
       * host memory on the wire. */
      std::memset(h_send_, 0, bytes);
      std::memset(h_recv_, 0, bytes);
    }
    /* Reused in cuda_aware mode as the "pack is complete" event -- the same
     * narrower-than-a-stream-synchronize wait the host-staged path already relies
     * on, just waited on before MPI_Isend instead of before an H2D. */
    if (cudaEventCreateWithFlags(&ev_d2h_, cudaEventDisableTiming) != cudaSuccess) {
      release_host_buffers();
      attach_why_ = "could not create the staging-copy completion event for the MPI "
                    "halo transport";
      return false;
    }
    elements_ = r.buffer_elements;
    attached_ = true;
    return true;
  }

  void detach(int32_t tile) noexcept override {
    if (!attached_ || tile != tile_) return;
    release_host_buffers();
    attached_ = false;
    elements_ = 0;
  }

  const char *attach_failure_reason() const noexcept override { return attach_why_; }

  /* Positional field compaction has to agree across RANKS, not just across threads:
   * a group's field set is configuration-dependent (cappa only under MOIST_CAPPA,
   * heat_source only when d_con exceeds its threshold), and two tiles that compacted
   * it differently would read each other's buffers at the wrong offset and get a
   * plausible number from the wrong field. One MIN/MAX allreduce pair per group,
   * cached, is the cheapest real check.
   *
   * COLLECTIVE. Every rank reaches it for the same group in the same order because
   * every rank runs the same dispatch table; a rank that failed locally before
   * getting here would have failed identically on every rank (the field set is a
   * function of configuration, not of data). */
  bool agree_on_layout(uint32_t group, int32_t field_count) noexcept override {
    if (group >= static_cast<uint32_t>(kGroups)) return false;
    if (layout_[group] >= 0) return layout_[group] == field_count;
    int lo = field_count, hi = field_count;
    if (MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_INT, MPI_MIN, comm_) != MPI_SUCCESS ||
        MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_INT, MPI_MAX, comm_) != MPI_SUCCESS)
      return false;
    if (lo != hi) return false;
    layout_[group] = field_count;
    return true;
  }

  int transfer(const HaloGroupSpec &eff, const HaloGeometry &geom,
               const TopologyView &topo, const fv_real *send, fv_real *recv,
               void *stream, fv3_gpu_error *e, const char *api) noexcept override {
    if (!attached_)
      return fail(e, FV3_GPU_STATE_CONFLICT, api,
                  "the MPI halo transport has no attached tile");
    if (!send || !recv || !stream)
      return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                  "null halo buffer or stream reached the MPI halo transport");

    const int32_t width = halo_ring_count(eff);
    const int64_t slot = group_slot_elements(geom, width);
    const int64_t used = group_buffer_elements(eff, geom);
    if (used > elements_)
      return fail(e, FV3_GPU_INTERNAL_ERROR, api,
                  "this halo group needs more buffer than was sized at attach; the "
                  "attach-time sizing loop must cover every live group");

    const int rc_topo = check_topology_once(topo, api, e);
    if (rc_topo != FV3_GPU_SUCCESS) return rc_topo;

    auto *cs = static_cast<cudaStream_t>(stream);
    const size_t used_bytes = static_cast<size_t>(used) * sizeof(fv_real);
    const bool cuda_aware = (mode_ == TransportMode::cuda_aware);

    /* 1. host_staged: D2H, stream-ordered behind the pack the caller already
     *    enqueued. cuda_aware: no payload copy -- MPI will read `send` directly. */
    if (!cuda_aware) {
      cudaError_t cr = cudaMemcpyAsync(h_send_, send, used_bytes, cudaMemcpyDeviceToHost, cs);
      if (cr != cudaSuccess)
        return fail(e, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(cr), (int)cr);
    }
    /* 2. Wait for the pack (cuda_aware) or the D2H (host_staged) to complete on
     *    THIS transport's own event -- strictly narrower than a stream synchronize
     *    and not a device synchronize; unrelated work on the compute stream keeps
     *    running. This is the point at which `send` becomes safe for MPI to read,
     *    whether MPI reads it via a host copy or, in cuda_aware mode, directly:
     *    the wait is on ordinary host-observable stream-order completion, not on
     *    anything MPI-specific, so it is sufficient before either kind of read. */
    cudaError_t cr = cudaEventRecord(ev_d2h_, cs);
    if (cr != cudaSuccess)
      return fail(e, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(cr), (int)cr);
    cr = cudaEventSynchronize(ev_d2h_);
    if (cr != cudaSuccess)
      return fail(e, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(cr), (int)cr);

    /* 3. Point-to-point. Receives first. host_staged addresses h_send_/h_recv_
     *    (pinned host); cuda_aware addresses `send`/`recv` (device) directly --
     *    same tags, same message count, same peer mapping, same slot_base()
     *    offsets, only the pointer family differs. Requires I_MPI_OFFLOAD=1 (or
     *    equivalent) to be set by the caller; verified unsafe without it
     *    (Prompt 42, SIGSEGV). */
    fv_real *const recv_base = cuda_aware ? recv : h_recv_;
    const fv_real *const send_base = cuda_aware ? send : h_send_;
    const MPI_Datatype dt = fv_real_datatype();
    std::array<MPI_Request, 2 * 4 * max_halo_group_fields> req{};
    int nreq = 0;

    for (int q = 0; q < 4; ++q) {
      const TileEdge mine = static_cast<TileEdge>(q);
      const EdgeNeighbor &nb = topo.neighbors[q];
      if (nb.tile < 1 || nb.tile > cubed_sphere_tiles)
        return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                    "edge has no neighbour tile; a connected topology is required "
                    "before any halo group may be exchanged");
      if (nb.rank < 0 || nb.rank >= size_)
        return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                    "edge neighbour has no MPI rank in the installed topology; the "
                    "MPI halo transport cannot infer one");
      /* Same rule as launch_halo_transfer, from the same functions: on a rotated
       * seam a vector field reads its PARTNER's slot in this same group. */
      const bool reversed = seam_is_reversed(mine, nb.edge);
      for (int n = 0; n < eff.field_count; ++n) {
        const HaloFieldSpec &fs = eff.fields[n];
        int sn = n;
        if (reversed && fs.role != VectorRole::scalar) {
          sn = field_slot(eff, fs.partner);
          if (sn < 0)
            return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                        "vector field's seam partner is not a member of its own halo "
                        "group");
        }
        if (MPI_Irecv(recv_base + slot_base(geom, width, n, mine),
                      static_cast<int>(slot), dt, nb.rank,
                      message_tag(eff.group, nb.edge, sn), comm_,
                      &req[nreq++]) != MPI_SUCCESS)
          return fail(e, FV3_GPU_MPI_ERROR, api,
                      "MPI_Irecv failed while posting a halo edge receive");
      }
    }
    for (int q = 0; q < 4; ++q) {
      const TileEdge mine = static_cast<TileEdge>(q);
      const EdgeNeighbor &nb = topo.neighbors[q];
      for (int n = 0; n < eff.field_count; ++n) {
        if (MPI_Isend(const_cast<fv_real *>(send_base) + slot_base(geom, width, n, mine),
                      static_cast<int>(slot), dt, nb.rank,
                      message_tag(eff.group, mine, n), comm_,
                      &req[nreq++]) != MPI_SUCCESS)
          return fail(e, FV3_GPU_MPI_ERROR, api,
                      "MPI_Isend failed while posting a halo edge send");
      }
    }
    if (MPI_Waitall(nreq, req.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS)
      return fail(e, FV3_GPU_MPI_ERROR, api,
                  "MPI_Waitall failed while completing a halo group exchange");

    /* 4. host_staged only: H2D on the caller's stream, so the unpack enqueued
     *    next sees it. cuda_aware already wrote `recv` directly above -- by the
     *    time MPI_Waitall returns on the host, the completed MPI receive is
     *    host-observable, and the unpack kernel is only enqueued on `stream`
     *    AFTER this function returns to the caller, so ordinary host-issue-order
     *    stream semantics make it safe without an extra device-side wait. */
    if (!cuda_aware) {
      cr = cudaMemcpyAsync(recv, h_recv_, used_bytes, cudaMemcpyHostToDevice, cs);
      if (cr != cudaSuccess)
        return fail(e, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(cr), (int)cr);
    }
    return FV3_GPU_SUCCESS;
  }

  /* Nothing to do in either mode: transfer() already blocks on MPI_Waitall before
   * returning (no async overlap in this transport), so every reader of the send
   * buffer -- the D2H in host_staged mode, or MPI_Isend reading `send` directly in
   * cuda_aware mode -- has completed by the time this is called. No reader of the
   * device send buffer survives into the next group's pack. Model B needs its
   * step-5 barrier only because peers read the device buffer directly and
   * asynchronously; this transport never does. */
  int release_send(void *, fv3_gpu_error *e, const char *) noexcept override {
    clear_error(e);
    return FV3_GPU_SUCCESS;
  }

  /* There is no process-local rendezvous to release. A rank that fails mid-exchange
   * leaves its neighbours in MPI_Waitall, and no library-level action can fix that
   * safely -- the caller must MPI_Abort. Recorded here rather than papered over. */
  void abort_transport() noexcept override {}

 private:
  void release_host_buffers() noexcept {
    if (ev_d2h_) { cudaEventDestroy(ev_d2h_); ev_d2h_ = nullptr; }
    if (h_send_) { cudaFreeHost(h_send_); h_send_ = nullptr; }
    if (h_recv_) { cudaFreeHost(h_recv_); h_recv_ = nullptr; }
  }

  /* The topology's rank table and the MPI communicator have to describe the SAME
   * mapping. If they disagree, every exchange still "succeeds" and simply delivers
   * one tile's edge into another's halo -- a wrong-donor defect that no tolerance
   * test can distinguish from a physics difference. Checked once, against the tile
   * table gathered at creation, rather than assumed. */
  int check_topology_once(const TopologyView &topo, const char *api,
                          fv3_gpu_error *e) noexcept {
    if (topology_checked_) return FV3_GPU_SUCCESS;
    if (topo.tile != tile_)
      return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                  "the installed topology's tile is not the tile this MPI rank was "
                  "created for");
    for (int q = 0; q < 4; ++q) {
      const EdgeNeighbor &nb = topo.neighbors[q];
      if (nb.rank < 0 || nb.rank >= size_)
        return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                    "edge neighbour has no MPI rank in the installed topology; the "
                    "MPI halo transport cannot infer one");
      if (tile_of_rank_[nb.rank] != nb.tile)
        return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                    "the installed topology says an edge neighbour tile lives on an "
                    "MPI rank that reported a different tile; a rank<->tile "
                    "disagreement would silently exchange the wrong tiles' edges");
    }
    topology_checked_ = true;
    return FV3_GPU_SUCCESS;
  }

  MPI_Comm comm_ = MPI_COMM_NULL;
  int32_t tile_ = 0;
  int rank_ = -1;
  /* Measured at construction (Prompt 112) -- create_mpi_transport() always
   * passes cubed_sphere_tiles (its own gate, unchanged, refuses any other
   * communicator size before constructing); create_mpi_transport_for_testing()
   * may pass a larger size for a synthetic multi-rank-per-face topology. */
  int size_ = 0;
  std::vector<int32_t> tile_of_rank_;
  TransportMode mode_ = TransportMode::host_staged;
  std::array<int32_t, kGroups> layout_{};
  fv_real *h_send_ = nullptr;
  fv_real *h_recv_ = nullptr;
  int64_t elements_ = 0;
  cudaEvent_t ev_d2h_ = nullptr;
  bool attached_ = false;
  bool topology_checked_ = false;
  const char *attach_why_ = "the MPI halo transport refused the attaching tile";
};

}  // namespace

HaloTransport *create_mpi_transport(int comm_fortran_handle, int32_t tile,
                                    TransportMode mode, std::string *why) noexcept {
  const auto refuse = [&](const char *m) -> HaloTransport * {
    if (why) *why = m;
    return nullptr;
  };

  int inited = 0;
  if (MPI_Initialized(&inited) != MPI_SUCCESS || !inited)
    return refuse("MPI is not initialized; the caller owns MPI_Init and must call it "
                  "before attaching the MPI halo transport");
  if (tile < 1 || tile > cubed_sphere_tiles)
    return refuse("tile must be a 1-based cubed-sphere tile in 1..6");

  MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_fortran_handle));
  if (comm == MPI_COMM_NULL)
    return refuse("the Fortran communicator handle does not name a valid MPI "
                  "communicator");

  int size = 0, rank = -1;
  if (MPI_Comm_size(comm, &size) != MPI_SUCCESS ||
      MPI_Comm_rank(comm, &rank) != MPI_SUCCESS)
    return refuse("could not query the communicator's size and rank");
  if (size != cubed_sphere_tiles)
    return refuse("the halo communicator must have exactly 6 ranks, one per "
                  "cubed-sphere tile; a different decomposition is not supported by "
                  "this transport");

  /* Prompt 43: cuda_aware construction fails clearly here rather than silently
   * falling back to host_staged -- a silent fallback would invalidate the A/B
   * this transport exists to support. MPIX_Query_cuda_support() is the exact
   * runtime capability probe Prompt 42 validated on a real A100 (PASS with
   * I_MPI_OFFLOAD=1, SIGSEGV without it) against Intel MPI 2021.13, the MPI
   * this project's production build links. */
  if (mode == TransportMode::cuda_aware && !MPIX_Query_cuda_support())
    return refuse("cuda_aware MPI halo transport requested, but this MPI runtime "
                  "reports MPIX_Query_cuda_support() == 0; for Intel MPI 2021.13 "
                  "this means I_MPI_OFFLOAD=1 is not set in the environment "
                  "(verified required in Prompt 42) or the runtime genuinely lacks "
                  "CUDA-buffer support -- refusing rather than silently using "
                  "host_staged");

  /* Gather the whole rank->tile map instead of assuming rank r holds tile r+1. The
   * identity mapping IS what the production decomposition produces (§2), but this
   * transport routes by the topology's rank table, so the map it validates against
   * must be measured, not presumed. A duplicated or missing tile is refused here
   * rather than discovered as a wrong-donor halo later. */
  std::vector<int32_t> tiles(static_cast<size_t>(size));
  if (MPI_Allgather(&tile, 1, MPI_INT32_T, tiles.data(), 1, MPI_INT32_T, comm) !=
      MPI_SUCCESS)
    return refuse("could not gather the rank->tile map across the halo communicator");
  std::array<int, cubed_sphere_tiles + 1> seen{};
  for (int r = 0; r < size; ++r) {
    if (tiles[r] < 1 || tiles[r] > cubed_sphere_tiles)
      return refuse("a rank of the halo communicator reported a tile outside 1..6");
    if (seen[tiles[r]]++)
      return refuse("two ranks of the halo communicator reported the same tile; each "
                    "of the six tiles must be held by exactly one rank");
  }

  auto *t = new (std::nothrow) MpiHaloTransport(comm, tile, rank, size, tiles, mode);
  if (!t) return refuse("could not allocate the MPI halo transport");
  return t;
}

/* Prompt 112, Stage 13 -- TEST-ONLY factory. Constructs an MpiHaloTransport
 * over an arbitrary-size communicator (`peer_count` ranks, several possibly
 * sharing one PHYSICAL FACE `tile`), bypassing create_mpi_transport's
 * production comm_size==6 gate above (which is completely untouched by this
 * function). Exists solely so a synthetic multi-rank-per-face test can
 * exercise MpiHaloTransport's now-arbitrary-size-capable internal storage
 * (Stage 7) without making any real >6-rank production configuration
 * reachable -- no production call site references this function (verified
 * by grep, see the Prompt 112 doc). `tile_of_rank` must already be the
 * caller's own measured rank->face map (e.g. from its own MPI_Allgather); it
 * is used as-is, not re-validated the way create_mpi_transport's is, since
 * this is a test helper, not a hardened entry point. */
HaloTransport *create_mpi_transport_for_testing(int comm_fortran_handle, int32_t tile,
                                                int peer_count,
                                                const std::vector<int32_t> &tile_of_rank,
                                                TransportMode mode, std::string *why) noexcept {
  const auto refuse = [&](const char *m) -> HaloTransport * {
    if (why) *why = m;
    return nullptr;
  };

  int inited = 0;
  if (MPI_Initialized(&inited) != MPI_SUCCESS || !inited)
    return refuse("MPI is not initialized; the caller owns MPI_Init and must call it "
                  "before attaching the MPI halo transport");
  if (tile < 1 || tile > cubed_sphere_tiles)
    return refuse("tile must be a 1-based physical cube face in 1..6");
  if (peer_count < 1 || static_cast<int>(tile_of_rank.size()) != peer_count)
    return refuse("peer_count must match tile_of_rank's size");

  MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_fortran_handle));
  if (comm == MPI_COMM_NULL)
    return refuse("the Fortran communicator handle does not name a valid MPI "
                  "communicator");

  int size = 0, rank = -1;
  if (MPI_Comm_size(comm, &size) != MPI_SUCCESS ||
      MPI_Comm_rank(comm, &rank) != MPI_SUCCESS)
    return refuse("could not query the communicator's size and rank");
  if (size != peer_count)
    return refuse("communicator size does not match peer_count");

  if (mode == TransportMode::cuda_aware && !MPIX_Query_cuda_support())
    return refuse("cuda_aware MPI halo transport requested, but this MPI runtime "
                  "reports MPIX_Query_cuda_support() == 0");

  auto *t = new (std::nothrow) MpiHaloTransport(comm, tile, rank, size, tile_of_rank, mode);
  if (!t) return refuse("could not allocate the MPI halo transport");
  return t;
}

HaloTransport *create_mpi_host_staged_transport(int comm_fortran_handle, int32_t tile,
                                               std::string *why) noexcept {
  return create_mpi_transport(comm_fortran_handle, tile, TransportMode::host_staged, why);
}

void destroy_mpi_host_staged_transport(HaloTransport *t) noexcept { delete t; }

}  // namespace fv3::gpu::halo
