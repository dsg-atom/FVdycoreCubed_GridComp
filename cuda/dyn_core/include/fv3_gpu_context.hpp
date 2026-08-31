#pragma once
#include "fv3_gpu_state.hpp"
#include <array>
#include <cstdint>

namespace fv3::gpu {
enum class ContextPhase { created, initialized, failed };
struct ContextData {
  fv3_gpu_create_desc create{};
  State state;
  Workspace workspace;
  ImmutableViews immutable;
  TopologyView topology;
  DispatchParams dispatch_params;
  DependencyState dependencies;
  NumericalDispatch dispatch;
  /* Milestone 3A.9 §8. Per halo group: has the CALLER explicitly taken
   * responsibility for this group's halo contents (mark_halo_caller_managed)?
   *
   * This is not a readiness flag and it is not a substitute for one. It records
   * a declaration of ownership, so that the runtime's resolve_halo() can
   * re-establish readiness at the authoritative exchange point of each substep
   * without ever inventing readiness itself: a group is resolvable only if the
   * topology makes the exchange provably vacuous, or the caller has said it
   * fills that group. Everything else is rejected.
   *
   * It has to be sticky and it has to live here rather than in DependencyState,
   * because a stage may mark a group stale MID-substep (cgrid_predictor ->
   * H_CWIND/H_DIVGD) and the caller has no opportunity to re-declare from
   * outside. Staleness stays fully live -- what is delegated is only who is
   * allowed to clear it. */
  std::array<uint8_t, FV3_GPU_H_GROUP_LAST + 1> halo_caller_managed{};
  /* Milestone 3A.10 §10/§27: provenance of each group's most recent readiness.
   * See HaloResolution in fv3_gpu_internal.hpp. Written only by the runtime's
   * resolve_halo() and by mark_halo_ready(); never by a numerical stage. Read by
   * the Gate C1 harness, which must be able to prove that no required group was
   * resolved by declaration rather than by exchange. */
  std::array<HaloResolution, FV3_GPU_H_GROUP_LAST + 1> halo_resolution{};
  void *compute_stream = nullptr;
  void *transfer_stream = nullptr;
  /* Milestone 3A.9. Cross-stream ordering handles (cudaEvent_t, held as void*
   * so this header stays compilable without the CUDA toolkit).
   *
   * Both streams are created cudaStreamNonBlocking, so NOTHING orders them
   * against each other implicitly -- not even the legacy default-stream
   * semantics. Before this milestone, fv3_gpu_sync_to_host enqueued its D2H on
   * transfer_stream with no dependency on the acoustic-loop kernels running on
   * compute_stream, and fv3_gpu_event_wait synchronized only transfer_stream.
   * A caller that did step -> sync_to_host -> event_wait, which is the exact
   * sequence project_architecture.md describes for an explicit D2H boundary,
   * could therefore read pre-step values.
   *
   * That was not theoretical: the integrated Gate C1 harness hit it. With
   * n_split = 3 the host spent long enough enqueueing kernels that the copies
   * landed after them and the data looked right; with n_split = 1 the D2H won
   * the race and every field came back byte-identical to the H2D input. The
   * n_split = 3 result was luck, not ordering.
   *
   * These two events are the ordering. They are recorded and waited on, never
   * queried, and they carry no numerical meaning. The uint64_t event handles in
   * the public ABI are a separate, host-side sequence counter and remain
   * unchanged. */
  void *event_compute_done = nullptr;
  void *event_transfer_done = nullptr;
  /* Milestone 3A.10 §9/§10. The real halo exchange's device state.
   *
   * `halo_peer` is the Model B rendezvous shared with this tile's five peers
   * (halo::HaloPeerGroup, include/fv3_gpu_halo_peer.hpp); it is BORROWED and
   * outlives nothing. `halo_peer_tile` is this context's 1-based tile, kept
   * separately from TopologyView::tile so that attaching cannot silently succeed
   * before a topology has been installed.
   *
   * The two buffers are device-resident and owned here. They are sized once, for
   * the largest live halo group, so that an exchange never allocates: allocating
   * inside the acoustic loop would be a per-substep cudaMalloc, which serializes
   * against every stream on the device and would quietly undo the persistent
   * residency the architecture requires.
   *
   * The two events are the ordering handles named in fv3_gpu_halo_peer.hpp's
   * sequence diagram. They are recorded and waited on, never queried. */
  void *halo_peer = nullptr;
  int32_t halo_peer_tile = 0;
  fv_real *halo_send = nullptr;
  fv_real *halo_recv = nullptr;
  int64_t halo_buffer_elements = 0;
  void *event_halo_pack = nullptr;
  void *event_halo_xfer = nullptr;
  /* Prompt 6B. The dyn_core-level heat_source accumulator of
   * dyn_core.F90:853-861. Device-resident, owned here, allocated lazily on the
   * first dyn_core step that needs it (d_con > 1.0E-5) and reused for the whole
   * context lifetime -- the same persistent-residency rule the halo buffers
   * above follow, and for the same reason: a per-substep cudaMalloc would
   * serialize the device inside the acoustic loop.
   *
   * It exists because the port uses the registered 3-D heat_source field as
   * d_sw's per-level scratch, which d_sw re-zeroes every substep, so the array
   * cannot also carry the running sum. See
   * runtime/fv3_gpu_heat_accumulator.hpp for the full contract. */
  fv_real *heat_accum = nullptr;
  uint64_t heat_accum_bytes = 0;
  uint64_t next_event = 1;
  uint64_t last_event = 0;
  ContextPhase phase = ContextPhase::created;
};
}
