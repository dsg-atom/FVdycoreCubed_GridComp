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
  std::array<uint8_t, FV3_GPU_H_DWIND_NEXT + 1> halo_caller_managed{};
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
  uint64_t next_event = 1;
  uint64_t last_event = 0;
  ContextPhase phase = ContextPhase::created;
};
}
