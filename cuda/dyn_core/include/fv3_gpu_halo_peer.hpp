#pragma once
/* Milestone 3A.10 §4/§9/§10/§17 -- the Gate C1 exchange TRANSPORT.
 *
 * §4 asks for a decision between four execution models and warns "Do not choose
 * based on convenience alone" and "Prefer the smallest model that proves the
 * production semantics". The choice recorded in the handoff is MODEL B: one
 * process represents all six cubed-sphere tiles and neighbours exchange by
 * device-to-device copy. This header is that model's rendezvous.
 *
 * ---------------------------------------------------------------------------
 * WHY A RENDEZVOUS IS REQUIRED AND CANNOT BE ELIDED
 * ---------------------------------------------------------------------------
 *
 * A halo exchange is a two-sided operation. My west halo is filled from tile 5's
 * owned cells AS THEY STAND AT THIS SUBSTEP, so tile 5 must have finished the
 * producer stage and packed before my transfer reads its send buffer. Six
 * contexts stepped one after another on a single host thread cannot express that:
 * tile 1's transfer would be enqueued before tile 5 had packed anything, and it
 * would copy the previous substep's values -- or, on the first substep,
 * uninitialised memory. It would still "run on the GPU" and it would still
 * produce finite numbers, which is exactly the kind of result §4 means when it
 * says the model must "exercise halo freshness transitions truthfully" and must
 * "not mark halos ready without producing the required data".
 *
 * So the six contexts run on six host threads and meet twice per exchange. That
 * is not an artefact of Model B: it is the same rendezvous MPI provides through
 * Isend/Irecv/Waitall, and an MPI transport replaces this class without touching
 * the pack, the transform, the unpack, the buffers or the readiness machinery
 * (§15).
 *
 * ---------------------------------------------------------------------------
 * WHAT IS SYNCHRONIZED AND WHAT IS NOT
 * ---------------------------------------------------------------------------
 *
 * The two barriers are HOST barriers over enqueue order only. No thread ever
 * calls cudaStreamSynchronize or cudaDeviceSynchronize inside an exchange, and
 * no device memory is copied to the host. Ordering between the six streams is
 * established with cudaEventRecord / cudaStreamWaitEvent, because every stream in
 * this runtime is cudaStreamNonBlocking and therefore nothing orders them
 * implicitly (§17; this is the same class of bug as the 3A.9 D2H race, which is
 * why it is done with events and not with a sync).
 *
 * Per exchange, per member:
 *
 *   compute_stream ==(event_compute_done)==> transfer_stream   producer visible
 *   pack on transfer_stream
 *   record pack_done
 *   -------- BARRIER 1: every member has ENQUEUED its pack --------
 *   transfer_stream waits on each neighbour's pack_done
 *   device-to-device transfer, then unpack, on transfer_stream
 *   record xfer_done
 *   compute_stream waits on xfer_done                          consumer ordered
 *   -------- BARRIER 2: every member has ENQUEUED its transfer --------
 *   transfer_stream waits on every member's xfer_done
 *
 * Barrier 2 and the wait that follows it are not redundant. Without them a
 * neighbour could enqueue the NEXT group's pack into its send buffer while my
 * transfer was still reading it -- two different streams, no ordering, a
 * write-after-read hazard that racecheck would be entirely right to report.
 *
 * A member that leaves the group mid-run would deadlock its peers, so the group
 * refuses to be destroyed while members are attached.
 */
#include "fv3_gpu_halo_exchange.hpp"
#include <array>
#include <condition_variable>
#include <mutex>

namespace fv3::gpu::halo {

/* One member's registration. Buffers and events are owned by the MEMBER's
 * context (allocated on its device, destroyed with it); the group only borrows
 * them, so a member's lifetime is not extended by the group. */
struct PeerMember {
  fv_real *send = nullptr;
  fv_real *recv = nullptr;
  void *pack_done = nullptr;      /* cudaEvent_t: my pack is enqueued */
  void *xfer_done = nullptr;      /* cudaEvent_t: my transfer+unpack is enqueued */
  void *transfer_stream = nullptr;
  int32_t attached = 0;
};

class HaloPeerGroup {
 public:
  /* layout_ is seeded to -1, not 0, because 0 is a legal field count for a group
   * that compacted to nothing and would otherwise read as "already agreed". */
  explicit HaloPeerGroup(int32_t tiles) noexcept : tiles_(tiles) { layout_.fill(-1); }

  int32_t tiles() const noexcept { return tiles_; }

  /* Register tile `tile` (1-based). Returns false if the slot is taken or out of
   * range -- a duplicated tile would make the barrier count wrong and one tile's
   * halo would silently come from the wrong buffer. */
  bool attach(int32_t tile, const PeerMember &m) noexcept {
    if (tile < 1 || tile > tiles_ || tile > cubed_sphere_tiles) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (member_[tile].attached) return false;
    member_[tile] = m;
    member_[tile].attached = 1;
    ++attached_;
    return true;
  }

  void detach(int32_t tile) noexcept {
    if (tile < 1 || tile > cubed_sphere_tiles) return;
    std::lock_guard<std::mutex> lk(mu_);
    if (!member_[tile].attached) return;
    member_[tile] = PeerMember{};
    --attached_;
  }

  int32_t attached() noexcept {
    std::lock_guard<std::mutex> lk(mu_);
    return attached_;
  }

  bool complete() noexcept { return attached() == tiles_; }

  const PeerMember &member(int32_t tile) const noexcept { return member_[tile]; }

  /* Snapshot of every member's send buffer, indexed by tile with entry 0 unused,
   * which is the shape launch_halo_transfer expects. */
  void send_table(fv_real *out[cubed_sphere_tiles + 1]) const noexcept {
    out[0] = nullptr;
    for (int t = 1; t <= cubed_sphere_tiles; ++t) out[t] = member_[t].send;
  }

  /* Host rendezvous over all attached members. Sense-reversing by generation, so
   * consecutive barriers cannot be conflated by a thread that arrives late.
   *
   * Returns false if the group is not fully attached: waiting for a member that
   * will never arrive is a deadlock, and a silent one, so it is refused at the
   * first barrier instead. */
  bool barrier() noexcept {
    std::unique_lock<std::mutex> lk(mu_);
    if (aborted_ || attached_ != tiles_) return false;
    const uint64_t gen = generation_;
    if (++waiting_ == tiles_) {
      waiting_ = 0;
      ++generation_;
      lk.unlock();
      cv_.notify_all();
      return true;
    }
    cv_.wait(lk, [this, gen] { return generation_ != gen || aborted_; });
    return !aborted_;
  }

  /* Deadlock brake. If ONE member fails before reaching a barrier -- a contract
   * check, an unregistered field, a CUDA error -- its five peers would wait at
   * barrier() for a thread that is never coming. A hung test is a much worse
   * failure report than a red one, because it hides which member failed and why.
   *
   * A member that gives up calls this, and every barrier from then on returns
   * false, so all six unwind through the ordinary refusal path in
   * perform_peer_exchange rather than blocking. Deliberately one-way: an aborted
   * exchange has already skipped somebody's pack, so no later barrier in the same
   * run could mean anything even if the group were revived. */
  void abort_group() noexcept {
    std::lock_guard<std::mutex> lk(mu_);
    aborted_ = true;
    waiting_ = 0;
    ++generation_;
    cv_.notify_all();
  }

  bool aborted() noexcept {
    std::lock_guard<std::mutex> lk(mu_);
    return aborted_;
  }

  /* Recorded on the first exchange and compared on every later one. Two members
   * that compacted a group's field list differently (because their registries or
   * their moist_cappa differ) would index each other's slots wrongly, and the
   * result would be a plausible number from the wrong field. */
  bool agree_on_layout(uint32_t group, int32_t field_count) noexcept {
    std::lock_guard<std::mutex> lk(mu_);
    if (group >= layout_.size()) return false;
    if (layout_[group] < 0) { layout_[group] = field_count; return true; }
    return layout_[group] == field_count;
  }

 private:
  int32_t tiles_;
  std::mutex mu_;
  std::condition_variable cv_;
  int32_t attached_ = 0;
  int32_t waiting_ = 0;
  uint64_t generation_ = 0;
  bool aborted_ = false;
  PeerMember member_[cubed_sphere_tiles + 1]{};
  std::array<int32_t, FV3_GPU_H_DWIND_NEXT + 1> layout_{};

 public:
  void reset_layout() noexcept {
    std::lock_guard<std::mutex> lk(mu_);
    layout_.fill(-1);
  }
};

}  // namespace fv3::gpu::halo
