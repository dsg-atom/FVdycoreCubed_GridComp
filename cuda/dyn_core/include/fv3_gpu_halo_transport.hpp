#pragma once
/* Milestone 3A.15 §3 -- the halo TRANSPORT boundary.
 *
 * Everything about the cubed-sphere halo that is numerical -- which cells a group
 * exchanges, which ring, which staggering, which seam rotation, which sign, which
 * donor slot a rotated seam reads -- lives in fv3_gpu_halo_exchange.hpp and its
 * pack/unpack kernels, and is IDENTICAL for every transport. What differs between
 * a single-process run and a six-MPI-rank run is only how one tile's packed edge
 * bytes reach the tile that consumes them.
 *
 * This header is that difference and nothing else:
 *
 *   halo topology/descriptor        (fv3_gpu_halo_topology.hpp)
 *        -> pack CUDA kernels       (launch_halo_pack)
 *        -> TRANSPORT BACKEND       <-- this file
 *        -> unpack CUDA kernels     (launch_halo_unpack)
 *
 * Two backends implement it:
 *
 *   PeerHaloTransport   Model B, in-process. Six host threads in one address
 *                       space; the "transfer" is a device-to-device copy out of a
 *                       peer's send buffer (launch_halo_transfer) and the
 *                       rendezvous is a std::mutex/condition_variable. Defined in
 *                       runtime/fv3_gpu_runtime.cpp, wrapping HaloPeerGroup
 *                       UNCHANGED -- see fv3_gpu_halo_peer.hpp.
 *
 *   MpiHaloTransport    Real cross-process. One tile per MPI rank; the transfer is
 *                       D2H of the packed send buffer, MPI_Isend/Irecv/Waitall,
 *                       H2D into the recv buffer. Defined in
 *                       runtime/fv3_gpu_halo_mpi.cpp, compiled only when
 *                       FV3_DYN_CORE_ENABLE_MPI_HALO is on. Created through
 *                       fv3_gpu_halo_mpi.hpp, which exposes NO MPI type.
 *
 * WHY THE SEAM IS DRAWN HERE AND NOT ONE STEP EITHER SIDE
 *
 * The exchange sequence that milestone 3A.10 §10/§17 proved correct is
 *
 *     order compute -> pack -> record pack_done
 *       -> [ make the donors' packed bytes readable, fill MY recv buffer ]
 *       -> unpack -> record xfer_done -> order the consumer behind it
 *       -> [ nobody may repack a send buffer somebody is still reading ]
 *
 * The two bracketed steps are the only transport-dependent ones. So the interface
 * has exactly two operations, transfer() and release_send(), and the runtime keeps
 * the surrounding five steps verbatim. Widening it (letting a backend own the pack
 * too, say) would let a backend get the seam algebra wrong privately, which is the
 * one class of bug this project has repeatedly found to be invisible downstream.
 *
 * Narrowing it -- e.g. handing the backend only a byte range -- would not work
 * either: a receiver must resolve WHICH donor slot feeds each of its own slots,
 * and that is a function of the seam (field_slot/seam_is_reversed), so the group
 * spec and topology have to cross the boundary.
 *
 * CONTRACT NOTES THAT ARE OBLIGATIONS, NOT DESCRIPTION
 *
 *  - transfer() is called with the pack already ENQUEUED on `stream` (and
 *    `pack_done` recorded on it). A backend may therefore rely on stream order
 *    for anything it enqueues on `stream`, and must not assume the pack has
 *    COMPLETED unless it waits for it explicitly.
 *  - On return from transfer(), `recv` must be filled by work that is ordered
 *    before anything subsequently enqueued on `stream` -- the unpack is enqueued
 *    there immediately afterwards.
 *  - Neither method may call cudaDeviceSynchronize(), nor synchronize a stream it
 *    does not own. Persistent residency (project_architecture.md) is a hard
 *    constraint; a host-staged backend needs to wait for its own D2H and nothing
 *    more.
 *  - No method may allocate device or pinned memory. Buffers are sized once at
 *    attach, for the largest live group, exactly as attach_halo_peer_group does,
 *    because a cudaMalloc inside the acoustic loop is an implicit device-wide
 *    synchronization.
 *  - agree_on_layout() exists because send and recv buffers are indexed
 *    POSITIONALLY by compacted field slot: two tiles that compacted a group's
 *    configuration-dependent fields differently would read each other's buffers at
 *    the wrong offset and get a plausible number from the wrong field. Every
 *    backend must therefore make the agreement real across its own participants,
 *    not assume it.
 * ------------------------------------------------------------------------- */

#include "fv3_gpu_halo_exchange.hpp"

namespace fv3::gpu::halo {

/* What one participant contributes to the transport. Deliberately a POD of
 * type-erased handles rather than CUDA types, so this header stays compilable in
 * a non-CUDA translation unit.
 *
 * This mirrors PeerMember (fv3_gpu_halo_peer.hpp) field for field on purpose: the
 * peer adapter converts one to the other and HaloPeerGroup is not touched. The
 * duplication is a deliberate decoupling of the abstract interface from Model B's
 * concrete rendezvous type, not an oversight.
 *
 * `pack_done` / `xfer_done` are cudaEvent_t; `transfer_stream` is a cudaStream_t.
 * A transport may ignore any of them -- MpiHaloTransport needs only the stream --
 * but every one it does use is BORROWED and owned by the attaching context. */
struct HaloMemberResources {
  fv_real *send = nullptr;
  fv_real *recv = nullptr;
  /* Element count of EACH of the two buffers above -- the largest live group's
   * requirement, computed once by attach_halo_transport. A staging backend needs it
   * to size its own host buffers at attach time instead of inside the acoustic
   * loop, which is the whole reason it is part of the contract. */
  int64_t buffer_elements = 0;
  void *pack_done = nullptr;
  void *xfer_done = nullptr;
  void *transfer_stream = nullptr;
};

class HaloTransport {
 public:
  virtual ~HaloTransport() = default;

  /* Stable identifier for diagnostics and for fv3_gpu_halo_transport_kind_of(). */
  virtual const char *name() const noexcept = 0;

  /* Register this context's tile (1..6) and its buffers/events. False means the
   * tile is already attached, which is refused rather than allowed to make a
   * rendezvous count wrong. */
  virtual bool attach(int32_t tile, const HaloMemberResources &r) noexcept = 0;
  virtual void detach(int32_t tile) noexcept = 0;

  /* Why the last attach() returned false. A backend that can fail for more than
   * one reason overrides this so the caller's error message is the true one; the
   * default is the only way the in-process rendezvous can fail. */
  virtual const char *attach_failure_reason() const noexcept {
    return "that tile is already attached to this halo transport; a duplicate would "
           "make the rendezvous count wrong and feed one tile's halo from the wrong "
           "buffer";
  }

  /* Record-then-compare the compacted field count of `group` across all
   * participants. False means a real disagreement, which must fail the exchange. */
  virtual bool agree_on_layout(uint32_t group, int32_t field_count) noexcept = 0;

  /* Make every donor's packed edge bytes for `eff` land in MY `recv` at MY slot
   * addresses -- slot_base(geom, halo_ring_count(eff), my_field, my_edge) -- with
   * the donor slot resolved by the seam exactly as launch_halo_transfer does.
   * `send` is MY packed buffer, which a cross-process backend must read out.
   *
   * `api` is threaded through only so a failure names the public entry point the
   * caller actually invoked; the transport is never the top of a call stack. */
  virtual int transfer(const HaloGroupSpec &eff, const HaloGeometry &geom,
                       const TopologyView &topo, const fv_real *send, fv_real *recv,
                       void *stream, fv3_gpu_error *e, const char *api) noexcept = 0;

  /* Barrier step 5: guarantee that nobody is still reading MY send buffer before
   * the next group's pack overwrites it. A backend that consumed the send buffer
   * synchronously inside transfer() has nothing to do here. */
  virtual int release_send(void *stream, fv3_gpu_error *e, const char *api) noexcept = 0;

  /* Release any participant blocked in a rendezvous, so a failing tile cannot
   * deadlock its peers. A backend with no rendezvous does nothing. */
  virtual void abort_transport() noexcept {}
};

}  // namespace fv3::gpu::halo
