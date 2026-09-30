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
  /* Milestone 3A.15 §3/§10. The transport BACKEND that moves the packed bytes
   * above between tiles: halo::HaloTransport*, see
   * include/fv3_gpu_halo_transport.hpp.
   *
   * Everything in the exchange except this pointer is transport-independent, so
   * this is the whole difference between the in-process six-thread Model B run and
   * a real six-MPI-rank run. `resolve_halo` branches on it and on nothing else --
   * `halo_peer` above is retained only because the Model B adapter needs the group
   * it wraps and because diagnostics name it.
   *
   * `halo_transport_owned` records whether THIS context must destroy the backend
   * at detach. It is 1 for the two attach paths that construct the backend
   * themselves (attach_halo_peer_group, which wraps a caller-owned HaloPeerGroup in
   * an adapter it allocates; and fv3_gpu_attach_mpi_halo_transport, which
   * constructs the MPI backend) and 0 when a caller attaches a backend it owns.
   * Getting this wrong would either leak one adapter per context or free a
   * caller's object, so it is stored rather than inferred.
   *
   * `halo_transport_kind` is the fv3_gpu_halo_transport_kind reported through the
   * public ABI, kept as a plain integer so this header needs no ABI enum. */
  void *halo_transport = nullptr;
  uint8_t halo_transport_owned = 0;
  uint8_t halo_transport_kind = 0;
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
  /* Prompt 85. The persistent device buffer for REMAP_Q's tracer array q
   * (mapn_tracer, nq=1). q has no FieldRegistry identity (see
   * vertical/fv3_remap_q.cuh's header comment), so it cannot ride the
   * ordinary field-registry residency path pt/delz/u/v use; this is its own
   * dedicated, grow-only, context-owned allocation instead -- same
   * persistent-residency rule as heat_accum above, and for the same reason:
   * a per-call cudaMalloc would defeat the whole point of a "one contiguous
   * H2D + one contiguous D2H per call" transfer design (Stage 3/4). Sized to
   * the full isd:ied x jsd:jed x km memory domain (fv3::gpu::vertical::
   * remap_q_buffer_elements), grown (never shrunk) by fv3_gpu_remap_q
   * (fv3_gpu_runtime.cpp) on first use and on any later call whose domain is
   * larger than the current allocation -- freed at finalize alongside
   * heat_accum/halo_send/halo_recv. */
  fv_real *remap_q_buffer = nullptr;
  uint64_t remap_q_buffer_bytes = 0;
  /* Prompt 92. The persistent device buffers for REMAP_PRESSUREVARS's
   * q(sphum) (read-only input) and cappa (written for is:ie x js:je,
   * round-tripped H2D+D2H so halo bytes outside that range pass through
   * unchanged) -- same rationale/lifecycle as remap_q_buffer above: neither
   * field can ride the ordinary FieldRegistry residency path here (see
   * vertical/fv3_remap_pressurevars.cuh's header comment), so each gets its
   * own dedicated, grow-only, context-owned allocation instead. Sized to
   * the full isd:ied x jsd:jed x km memory domain, grown (never shrunk) by
   * fv3_gpu_remap_pressurevars (fv3_gpu_runtime.cpp) on first use and on
   * any later call whose domain is larger than the current allocation --
   * freed at finalize alongside remap_q_buffer/heat_accum/halo_send/
   * halo_recv. */
  fv_real *remap_pressurevars_qsphum_buffer = nullptr;
  uint64_t remap_pressurevars_qsphum_buffer_bytes = 0;
  fv_real *remap_pressurevars_cappa_buffer = nullptr;
  uint64_t remap_pressurevars_cappa_buffer_bytes = 0;
  /* Prompt 99. Persistent device buffers for FV_StateMod.F90's
   * State_To_FV: the STATE%VARS%U/V/W/PT/PKZ/PE/DZ inputs (no
   * FieldRegistry identity -- MAPL State fields, not FV_Atm fields; see
   * vertical/fv3_state_to_fv.cuh's header comment) and the ps output
   * (also no FieldRegistry identity, and unlike the inputs, no
   * downstream device consumer). Same rationale/lifecycle as
   * remap_q_buffer/remap_pressurevars_*_buffer above: each is its own
   * dedicated, grow-only, context-owned allocation. Sized to the plain
   * interior compute domain is:ie x js:je x (km or km+1) -- STATE%VARS%*
   * arrays carry no halo at all, unlike remap_q/remap_pressurevars's
   * isd:ied x jsd:jed sizing -- grown (never shrunk) by
   * fv3_gpu_state_to_fv_phase_a/_phase_c (fv3_gpu_runtime.cpp) on first
   * use and on any later call whose domain is larger than the current
   * allocation -- freed at finalize alongside remap_pressurevars_*_
   * buffer/remap_q_buffer/heat_accum/halo_send/halo_recv. */
  fv_real *state_to_fv_u_buffer = nullptr;
  uint64_t state_to_fv_u_buffer_bytes = 0;
  fv_real *state_to_fv_v_buffer = nullptr;
  uint64_t state_to_fv_v_buffer_bytes = 0;
  fv_real *state_to_fv_w_buffer = nullptr;
  uint64_t state_to_fv_w_buffer_bytes = 0;
  fv_real *state_to_fv_pe_buffer = nullptr;
  uint64_t state_to_fv_pe_buffer_bytes = 0;
  fv_real *state_to_fv_pt_buffer = nullptr;
  uint64_t state_to_fv_pt_buffer_bytes = 0;
  fv_real *state_to_fv_pkz_buffer = nullptr;
  uint64_t state_to_fv_pkz_buffer_bytes = 0;
  fv_real *state_to_fv_dz_buffer = nullptr;
  uint64_t state_to_fv_dz_buffer_bytes = 0;
  fv_real *state_to_fv_ps_buffer = nullptr;
  uint64_t state_to_fv_ps_buffer_bytes = 0;
  uint64_t next_event = 1;
  uint64_t last_event = 0;
  ContextPhase phase = ContextPhase::created;
  /* Prompt 8 (CUDA Graphs prototype). Opt-in only -- default false, so default
   * execution is provably unchanged (fv3_gpu_set_cuda_graph_mode is the only
   * writer). When enabled, the single dgrid_dynamics stage call is captured
   * once, on this tile's own compute_stream, and replayed on every later
   * occurrence (every subsequent it within this call, and every subsequent
   * fv3_gpu_dyn_core_step call) rather than re-issued. This stage was chosen
   * because it is bounded on both sides by a cross-thread halo resolve
   * (resolve(H_CWIND) before, resolve(H_DSCALAR) after -- see
   * fv3_gpu_runtime.cpp's per-substep sequence), runs entirely on
   * compute_stream with no cross-thread stream reference inside it, and its
   * entire kernel-launch parameter set (bounds, options, level tables,
   * geometry, and the persistent-residency device pointers) is invariant
   * across it and across calls for a fixed topology/config -- so replay needs
   * no per-call parameter update (cudaGraphExecKernelNodeSetParams is not
   * required). `dgrid_dynamics_graph_exec` is a cudaGraphExec_t stored as
   * void* so this header stays compilable without the CUDA toolkit, matching
   * every other stream/event field above. Owned here; destroyed in
   * fv3_gpu_finalize. */
  bool cuda_graph_enabled = false;
  bool dgrid_dynamics_graph_captured = false;
  void *dgrid_dynamics_graph_exec = nullptr;
  /* Prompt 13. The ONE host scalar that reaches a captured kernel node by value
   * and could in principle differ between calls is dgrid_dynamics's dt
   * (DispatchParams::dt = bdt/n_split, consumed as DSwOptions::dt at
   * operators/../dgrid/wp_b_dsw_contract_cuda.cpp; everything else the stage
   * reads is either immutable config/topology or live device memory at a
   * persistent address). Capture freezes it into every node that takes
   * DSwOptions by value, so replaying with a different bdt/n_split would
   * silently integrate with the wrong timestep. Rather than update thousands of
   * nodes, the capture-time pair is recorded here and replay refuses a
   * mismatch. */
  double graph_capture_bdt = 0.0;
  int32_t graph_capture_n_split = 0;
};
}
