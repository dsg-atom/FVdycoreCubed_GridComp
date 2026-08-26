#include "fv3_gpu_api.h"
#include "fv3_gpu_context.hpp"
#include "fv3_gpu_errors.hpp"
#include "fv3_gpu_halo_peer.hpp"
#include "fv3_gpu_halo_topology.hpp"
#include "fv3_gpu_types.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <string>

#ifdef FV3_GPU_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

struct fv3_gpu_context { fv3::gpu::ContextData data; };
namespace {
using namespace fv3::gpu;
bool header_ok(uint32_t version, uint32_t bytes, size_t need) {
  return version == FV3_GPU_ABI_VERSION && bytes >= need;
}
int validate_mask(const fv3_gpu_field_mask *m, fv3_gpu_error *e, const char *api) {
  if (!m || !header_ok(m->abi_version, m->struct_bytes, sizeof(*m)))
    return fail(e, FV3_GPU_INVALID_ARGUMENT, api, "invalid field-mask ABI");
  return FV3_GPU_SUCCESS;
}
[[maybe_unused]] uint64_t issue_event(ContextData &d) { d.last_event = d.next_event++; return d.last_event; }
[[maybe_unused]] bool mul_overflow(uint64_t a,uint64_t b,uint64_t &out){if(a&&b>std::numeric_limits<uint64_t>::max()/a)return true;out=a*b;return false;}
#ifdef FV3_GPU_ENABLE_CUDA
int cuda_fail(cudaError_t rc, fv3_gpu_error *e, const char *api, ContextData *d = nullptr) {
  if (rc == cudaSuccess) return FV3_GPU_SUCCESS;
  if (d) d->phase = ContextPhase::failed;
  return fail(e, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(rc), (int)rc,
              d ? d->create.mpi_rank : -1, d ? d->create.device_ordinal : -1);
}

/* Milestone 3A.9: make `waiter` wait for everything already enqueued on
 * `signaler`, using the context-owned event `ev`.
 *
 * Both streams are cudaStreamNonBlocking, so this is the ONLY thing that orders
 * them; see ContextData::event_compute_done for the race this closes and how it
 * was found. This is a dependency, not a synchronization: the host does not
 * block, which is what keeps the persistent-residency model intact. Introducing
 * a cudaStreamSynchronize here instead would have been a hidden host sync of
 * exactly the kind project_architecture.md forbids. */
cudaError_t order_after(void *signaler, void *waiter, void *ev) {
  if (!signaler || !waiter || !ev) return cudaSuccess;  // CPU-only build path
  const cudaError_t rc =
      cudaEventRecord(static_cast<cudaEvent_t>(ev), static_cast<cudaStream_t>(signaler));
  if (rc != cudaSuccess) return rc;
  return cudaStreamWaitEvent(static_cast<cudaStream_t>(waiter),
                             static_cast<cudaEvent_t>(ev), 0);
}

/* "Empty the flux capacitors" -- dyn_core.F90:368-379, executed ONCE per
 * dyn_core call, before `do it=1,n_split` (:399).
 *
 * The authoritative code is four whole-array assignments,
 *
 *     mfx(:,:,:) = 0.0        ! :372
 *     mfy(:,:,:) = 0.0        ! :373
 *      cx(:,:,:) = 0.0        ! :374
 *      cy(:,:,:) = 0.0        ! :375
 *
 * and then, ONLY when d_con exceeds the 1.0E-5 activation threshold, a freshly
 * allocated heat_source zeroed over its full (isd:ied, jsd:jed, npz) extent
 * (:377-379). Whole-array syntax covers the entire allocation including halo,
 * which is exactly a memset of allocation_bytes.
 *
 * Three deliberate non-actions, each traced:
 *   - diss_est is NOT zeroed here. It is owned by fv_dynamics
 *     (fv_dynamics.F90:607-614), not dyn_core, and dyn_core only accumulates
 *     into it (:819-826). Zeroing it per dyn_core call would destroy the
 *     k_split accumulation.
 *   - dpx is NOT zeroed here either, for the same reason (fv_dynamics.F90).
 *   - When d_con <= 1.0E-5 the Fortran never allocates heat_source at all, so
 *     there is nothing to zero and no stage reads it (the del2_cubed tail is
 *     gated on the same threshold at :1212). A caller that registers the field
 *     anyway simply gets it left alone, matching "not allocated".
 *
 * Only registered fields are touched: a production caller that does not use the
 * tracer capacitors need not register them, and their absence is not an error.
 *
 * Device epochs are intentionally left alone. Every numerical stage in this
 * runtime mutates device memory without touching host_epoch/device_epoch --
 * those counters track explicit sync_from_host/sync_to_host transfers -- and
 * inventing a different convention for this one memset would make the epoch
 * pair mean two different things depending on which code path ran.
 */
int zero_accumulators(ContextData &d, fv3_gpu_error *e, const char *api) {
  auto stream = static_cast<cudaStream_t>(d.compute_stream);
  const bool zero_heat = d.immutable.config.d_con > 1.0e-5;  // dyn_core.F90:377
  for (auto &x : d.state.fields) {
    const FieldId id = field_id_from_abi(x.host.field_id);
    const bool capacitor = id == FieldId::mfx || id == FieldId::mfy ||
                           id == FieldId::cx  || id == FieldId::cy;
    if (!capacitor && !(zero_heat && id == FieldId::heat_source)) continue;
    const cudaError_t rc =
        cudaMemsetAsync(x.device, 0, x.host.allocation_bytes, stream);
    if (rc != cudaSuccess) return cuda_fail(rc, e, api, &d);
  }
  return FV3_GPU_SUCCESS;
}
#endif

/* ---------------------------------------------------------------------------
 * Milestone 3A.9 §8/§9: halo readiness for the supported single-rank path.
 *
 * dyn_core.F90 drives its halos through mpp/fv_mp_mod: for each group it starts
 * a pack (`start_group_halo_update`) and later completes it
 * (`complete_group_halo_update`) at the point the consumer needs the data. This
 * runtime models the same thing with DependencyState: a stage calls
 * require(group) and refuses to run unless the group is `ready`.
 *
 * Nothing in the tree ever produced that readiness. `fv3_gpu_halo_callbacks`
 * (include/fv3_gpu_communication.h) declares the transport interface, but there
 * is no registration API, no storage for it in ContextData, and no consumer --
 * so a real exchange cannot be performed by this runtime at all. The only
 * existing producer is mark_halo_ready(), a C++-only setter a test can poke.
 *
 * The declared callback signature is host-staged pack/unpack -- begin() is
 * handed a send buffer and a recv buffer -- so honouring it requires per-group
 * halo PACKING (which fields, which strips, which rotation at a panel seam),
 * and none of that machinery exists. Building it is a halo-exchange
 * implementation, which §8 explicitly defers ("do NOT implement MPI in this
 * milestone unless truly unavoidable"). So this milestone does not pretend to
 * consume that interface.
 *
 * §8 forbids two shortcuts explicitly: do not implement MPI here, and do not
 * simply mark all halos ready. resolve_halo() therefore has exactly three
 * outcomes, and none of them lets the RUNTIME invent readiness:
 *
 *   1. The topology has no connected neighbour on any of the four tile edges,
 *      no corner owner, and no periodic/nested/regional wrap. Fortran's
 *      mpp_update_domains on such a domain copies nothing -- the update is
 *      genuinely vacuous -- so marking the group ready skips no operation.
 *
 *   2. The caller has explicitly declared the group caller-managed
 *      (mark_halo_caller_managed). Readiness is then the CALLER's assertion,
 *      re-established at the authoritative exchange point because a stage may
 *      have marked the group stale mid-substep and the caller cannot reach in
 *      to re-declare. The runtime asserts nothing about the data; it records
 *      whose claim it is.
 *
 *   3. Otherwise. A real exchange IS required, nothing in this runtime can
 *      perform one, no caller has claimed it, and pretending otherwise is
 *      exactly the false readiness §8 prohibits. Return
 *      FV3_GPU_UNSUPPORTED_CONFIGURATION naming the group, before any state is
 *      mutated.
 *
 * Consequence that MUST be stated plainly wherever this path is reported: a
 * true global cubed sphere puts a neighbouring tile on every edge, so a
 * production topology reaches outcome 1 for no group and depends entirely on
 * outcome 2. An integrated run under outcome 2 exercises the real stage graph,
 * the real dispatch table and the real device kernels -- it is genuine GPU
 * EXECUTION -- but its halo cells hold whatever the caller put there. It is NOT
 * a numerically meaningful cubed-sphere integration, and a caller whose
 * "management" of a group is a no-op has a falsely-ready halo in substance even
 * though the runtime never claimed otherwise.
 *
 * MILESTONE 3A.10 AMENDS THE ABOVE IN TWO WAYS. First, outcome 1's predicate
 * failed OPEN on a topology that had never been supplied, which made the
 * "runtime invents no readiness" claim untrue in one reachable case; there is
 * now a fourth outcome, `not_supplied`, which refuses -- see
 * classify_halo_topology() below. Second, readiness now carries provenance
 * (HaloResolution, recorded in ContextData::halo_resolution), so the distinction
 * this comment draws in prose between "exchanged" and "asserted by the caller"
 * is machine-checkable rather than something a report has to promise.
 *
 * mark_stale stays fully live. Groups are re-resolved at the authoritative
 * exchange points on every substep, so a stage that invalidates a group
 * (cgrid_predictor -> H_CWIND/H_DIVGD, cgrid_pressure -> H_CWIND) still forces
 * the group back through resolve_halo before its next consumer.
 * ------------------------------------------------------------------------- */
const char *halo_group_name(fv3_gpu_halo_group g) noexcept {
  switch (g) {
    case FV3_GPU_H_DWIND_IN:   return "H_DWIND_IN (u,v DGRID_NE, dyn_core.F90:402/:504)";
    case FV3_GPU_H_W_IN:       return "H_W_IN (w, :427/:507)";
    case FV3_GPU_H_GZ_FIRST:   return "H_GZ_FIRST (gz, :459/:552)";
    case FV3_GPU_H_Q_INLINE:   return "H_Q_INLINE (inline_q tracers, :419/:631)";
    case FV3_GPU_H_DIVGD:      return "H_DIVGD (divgd CORNER, :528/:636)";
    case FV3_GPU_H_CWIND:      return "H_CWIND (uc,vc CGRID_NE, :618/:640)";
    case FV3_GPU_H_DSCALAR:    return "H_DSCALAR (q_con,delp,pt, :844/:878)";
    case FV3_GPU_H_ZH:         return "H_ZH (zh, :931/:967)";
    case FV3_GPU_H_PKC2:       return "H_PKC2 (pkc, :931/:967)";
    case FV3_GPU_H_HEAT:       return "H_HEAT (heat_source, del2_cubed :2288)";
    case FV3_GPU_H_DWIND_NEXT: return "H_DWIND_NEXT (u,v for the next substep, :1109)";
    default:                   return "unknown halo group";
  }
}

/* ---------------------------------------------------------------------------
 * Milestone 3A.10 §2: the vacuity test USED TO FAIL OPEN, and that was
 * reachable false halo readiness.
 *
 * The previous predicate was:
 *
 *     for (const auto &nb : t.neighbors)
 *       if (nb.rank >= 0 || nb.tile >= 0) return false;
 *     for (const auto owner : t.corner_owner_rank)
 *       if (owner >= 0) return false;
 *     return !(t.nested || t.regional || t.periodic_x || t.periodic_y);
 *
 * A default-constructed TopologyView has every neighbour rank and tile at -1,
 * every corner_owner_rank at -1, and none of the wrap flags set -- so it
 * returned TRUE. resolve_halo then marked every group ready having moved
 * nothing, on a context whose global_cubed_sphere defaults to 1, i.e. one that
 * positively declares itself a global cubed sphere. The predicate conflated
 * "no topology information was supplied" with "this topology genuinely has
 * nothing to exchange", and those are opposite conclusions.
 *
 * It was reachable: set_topology() had no production caller at all, and the
 * Fortran bridge (cuda/interface/fv3_gpu_interface_mod.F90) never mentions
 * topology, npx, npy or tile. Nothing but luck stood in the way -- npx = npy = 0
 * makes the panel-geometry stages reject later in the substep, which masks it
 * incidentally for a FULL production dispatch but guarantees nothing for a
 * partial one.
 *
 * Vacuity now requires POSITIVE evidence and has a third outcome for absence of
 * evidence:
 *
 *   - npx/npy > 0. The contract already documents zero as "not supplied by the
 *     caller" (see TopologyView in fv3_gpu_internal.hpp), so this is the
 *     existing supplied-ness signal, not a new one.
 *   - global_cubed_sphere == 0. A global cubed sphere puts a neighbouring tile
 *     on all four edges of every tile; a TopologyView asserting both "global
 *     cubed sphere" and "no neighbours" is self-contradictory and must never be
 *     read as vacuous.
 *   - and then, as before, no connected edge, no corner owner, and no
 *     periodic/nested/regional wrap.
 * ------------------------------------------------------------------------- */
enum class HaloTopologyClass { not_supplied, connected, vacuous };

HaloTopologyClass classify_halo_topology(const TopologyView &t) noexcept {
  if (t.npx <= 0 || t.npy <= 0) return HaloTopologyClass::not_supplied;
  if (t.global_cubed_sphere) return HaloTopologyClass::connected;
  if (t.nested || t.regional || t.periodic_x || t.periodic_y)
    return HaloTopologyClass::connected;
  for (const auto &nb : t.neighbors)
    if (nb.rank >= 0 || nb.tile >= 0) return HaloTopologyClass::connected;
  for (const auto owner : t.corner_owner_rank)
    if (owner >= 0) return HaloTopologyClass::connected;
  return HaloTopologyClass::vacuous;
}

#ifdef FV3_GPU_ENABLE_CUDA
/* ---------------------------------------------------------------------------
 * Milestone 3A.10 §9/§10/§17 -- the REAL exchange, wired into readiness.
 *
 * §10 requires that "a halo may become ready only after producer stage -> pack ->
 * transfer/transform -> unpack -> completion event", and §17 requires the stream
 * ordering to be proven rather than assumed. The sequence and the reason each
 * step exists are documented once, in include/fv3_gpu_halo_peer.hpp; this is the
 * implementation of it.
 * ------------------------------------------------------------------------- */
halo::HaloGeometry geometry_of(const ContextData &d) noexcept {
  halo::HaloGeometry g{};
  g.nx = d.immutable.bounds.ie - d.immutable.bounds.is + 1;
  g.ny = d.immutable.bounds.je - d.immutable.bounds.js + 1;
  g.npz = d.immutable.bounds.npz;
  return g;
}

/* Reduce a group's declarative field table to the fields this context actually
 * has registered, and build the device views the kernels need.
 *
 * Compaction is necessary because a group's table is the AUTHORITATIVE field list
 * while a caller's registry is configuration-dependent -- cappa exists only under
 * MOIST_CAPPA, heat_source only when d_con exceeds its threshold. It is also
 * dangerous, because the send and receive sides index slots positionally: two
 * tiles that compacted differently would read each other's buffers at the wrong
 * offset and get a plausible number from the wrong field. That is why the
 * resulting field count is agreed across the peer group before any data moves.
 *
 * A VECTOR field is never compacted away. Dropping one half of a pair would leave
 * the survivor's rotated-seam partner unresolvable, and the seam would then be
 * served from the wrong component -- so a half-registered wind pair is refused. */
int build_group_fields(const ContextData &d, uint32_t group,
                       halo::HaloGroupSpec &eff, halo::HaloTileFields &f,
                       fv3_gpu_error *e, const char *api) {
  const halo::HaloGroupSpec spec = halo::halo_group_spec(group);
  if (!spec.live)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                std::string("halo group ") +
                    halo_group_name(static_cast<fv3_gpu_halo_group>(group)) +
                    " is not live and cannot be exchanged");
  eff = spec;
  eff.field_count = 0;
  for (int n = 0; n < spec.field_count; ++n) {
    const halo::HaloFieldSpec &fs = spec.fields[n];
    const DeviceViewBase *v = d.state.registry.find(fs.id);
    if (!v || !v->data) {
      if (fs.role != halo::VectorRole::scalar)
        return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                    std::string("halo group ") +
                        halo_group_name(static_cast<fv3_gpu_halo_group>(group)) +
                        " is a vector pair but only one component is registered; "
                        "a rotated seam could not resolve the missing partner");
      continue;  /* configuration-dependent scalar; genuinely absent */
    }
    if (v->rank < 3)
      return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                  "a halo group field is registered with fewer than three "
                  "dimensions; the exchange indexes (i, j, k)");
    if (!view_element_width_is(v, sizeof(fv_real)))
      return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                  "a halo group field's registered element width does not match "
                  "this build's fv_real; the exchange will not reinterpret it");
    const int slot = eff.field_count;
    eff.fields[slot] = fs;
    halo::HaloStripView view{};
    view.data = static_cast<fv_real *>(v->data);
    view.lo_i = v->lower[0]; view.lo_j = v->lower[1]; view.lo_k = v->lower[2];
    view.st_i = v->stride[0]; view.st_j = v->stride[1]; view.st_k = v->stride[2];
    view.nk = static_cast<int64_t>(v->extent[2]);
    f.view[slot] = view;
    f.present[slot] = 1;
    ++eff.field_count;
  }
  if (eff.field_count == 0)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, api,
                std::string("halo group ") +
                    halo_group_name(static_cast<fv3_gpu_halo_group>(group)) +
                    " has no registered fields, so no exchange can be performed "
                    "for it; readiness is refused rather than granted vacuously");
  return FV3_GPU_SUCCESS;
}

int perform_peer_exchange(ContextData &d, fv3_gpu_halo_group g, fv3_gpu_error *e,
                          const char *api) {
  auto *pg = static_cast<halo::HaloPeerGroup *>(d.halo_peer);
  const uint32_t group = static_cast<uint32_t>(g);
  halo::HaloGroupSpec eff{};
  halo::HaloTileFields fields{};
  int rc = build_group_fields(d, group, eff, fields, e, api);
  if (rc != FV3_GPU_SUCCESS) return rc;

  if (!pg->agree_on_layout(group, eff.field_count))
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                std::string("peers disagree on the registered field layout of ") +
                    halo_group_name(g) +
                    "; slot-indexed buffers would be read at the wrong offset");

  const halo::HaloGeometry geom = geometry_of(d);
  if (halo::group_buffer_elements(eff, geom) > d.halo_buffer_elements)
    return fail(e, FV3_GPU_INTERNAL_ERROR, api,
                "halo buffers were sized for a smaller group than the one being "
                "exchanged");

  /* 1. The pack must observe the producer stage's writes. Both streams are
   *    cudaStreamNonBlocking, so this event is the only thing that makes them
   *    visible to each other (§17). */
  cudaError_t crc = order_after(d.compute_stream, d.transfer_stream, d.event_compute_done);
  if (crc != cudaSuccess) return cuda_fail(crc, e, api, &d);

  rc = halo::launch_halo_pack(eff, geom, fields, d.halo_send, d.transfer_stream, e);
  if (rc != FV3_GPU_SUCCESS) return rc;
  crc = cudaEventRecord(static_cast<cudaEvent_t>(d.event_halo_pack),
                        static_cast<cudaStream_t>(d.transfer_stream));
  if (crc != cudaSuccess) return cuda_fail(crc, e, api, &d);

  /* 2. Every peer has now ENQUEUED its pack. This is a host rendezvous over
   *    enqueue order, not a device synchronization: no thread waits for the GPU
   *    here. It is the same guarantee MPI_Waitall on the sends would give. */
  if (!pg->barrier())
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "the halo peer group is not fully attached; waiting for a member "
                "that will never arrive would deadlock, so it is refused");

  /* 3. Wait on the DEVICE for the four donors' packs. Only the four neighbours
   *    matter, and taking them from the topology rather than from all six keeps
   *    this honest about who actually feeds this tile. */
  for (int q = 0; q < 4; ++q) {
    const int32_t nbt = d.topology.neighbors[static_cast<size_t>(q)].tile;
    if (nbt < 1 || nbt > halo::cubed_sphere_tiles)
      return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                  "an edge has no neighbour tile, so its halo cannot be exchanged");
    void *ev = pg->member(nbt).pack_done;
    if (!ev)
      return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                  "a neighbour tile is not attached to the peer group");
    crc = cudaStreamWaitEvent(static_cast<cudaStream_t>(d.transfer_stream),
                              static_cast<cudaEvent_t>(ev), 0);
    if (crc != cudaSuccess) return cuda_fail(crc, e, api, &d);
  }

  fv_real *send_table[halo::cubed_sphere_tiles + 1];
  pg->send_table(send_table);
  rc = halo::launch_halo_transfer(eff, geom, d.topology, send_table, d.halo_recv,
                                  d.transfer_stream, e);
  if (rc != FV3_GPU_SUCCESS) return rc;
  rc = halo::launch_halo_unpack(eff, geom, d.topology, fields, d.halo_recv,
                                d.transfer_stream, e);
  if (rc != FV3_GPU_SUCCESS) return rc;

  crc = cudaEventRecord(static_cast<cudaEvent_t>(d.event_halo_xfer),
                        static_cast<cudaStream_t>(d.transfer_stream));
  if (crc != cudaSuccess) return cuda_fail(crc, e, api, &d);
  /* 4. The CONSUMER runs on compute_stream and must not start before the unpack
   *    lands. This is the edge that makes readiness mean something. */
  crc = cudaStreamWaitEvent(static_cast<cudaStream_t>(d.compute_stream),
                            static_cast<cudaEvent_t>(d.event_halo_xfer), 0);
  if (crc != cudaSuccess) return cuda_fail(crc, e, api, &d);

  /* 5. Nobody may repack into a send buffer that somebody else is still reading.
   *    My next pack is on my stream; the transfer that reads my buffer is on the
   *    reader's stream, and nothing orders those two. Without this the next
   *    group's pack is a write-after-read hazard across streams. */
  if (!pg->barrier())
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "the halo peer group lost a member mid-exchange");
  for (int t = 1; t <= pg->tiles(); ++t) {
    void *ev = pg->member(static_cast<int32_t>(t)).xfer_done;
    if (!ev) continue;
    crc = cudaStreamWaitEvent(static_cast<cudaStream_t>(d.transfer_stream),
                              static_cast<cudaEvent_t>(ev), 0);
    if (crc != cudaSuccess) return cuda_fail(crc, e, api, &d);
  }
  return FV3_GPU_SUCCESS;
}
#endif  // FV3_GPU_ENABLE_CUDA

int resolve_halo(ContextData &d, fv3_gpu_halo_group g, fv3_gpu_error *e,
                 const char *api) {
  const auto slot = static_cast<size_t>(g);
  if (slot >= d.halo_caller_managed.size())
    return fail(e, FV3_GPU_INVALID_ARGUMENT, api, "invalid halo group");

  const HaloTopologyClass klass = classify_halo_topology(d.topology);

  /* Absence of a topology is never a licence to resolve, and a caller
   * declaration cannot substitute for one: a caller that has not said where its
   * neighbours are has not said what it is filling either. This is the branch
   * that used to be silently the vacuous one. */
  if (klass == HaloTopologyClass::not_supplied)
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                std::string("halo group ") + halo_group_name(g) +
                    " cannot be resolved because no topology has been supplied "
                    "(TopologyView::npx/npy are zero, which the contract "
                    "defines as not-supplied); an absent topology is refused "
                    "rather than treated as having nothing to exchange");

  HaloResolution provenance = HaloResolution::none;
  if (klass == HaloTopologyClass::vacuous) {
    provenance = HaloResolution::vacuous_topology;
  } else if (d.halo_peer != nullptr) {
    /* Milestone 3A.10: the branch that did not exist. A real exchange runs here
     * and readiness is granted only if it completed enqueuing without error, with
     * the consumer stream ordered behind the unpack. This is checked BEFORE the
     * caller declaration, deliberately: if a real exchange is available it is
     * always the truthful answer, and a caller's claim must not be able to
     * suppress it. */
#ifdef FV3_GPU_ENABLE_CUDA
    const int xrc = perform_peer_exchange(d, g, e, api);
    if (xrc != FV3_GPU_SUCCESS) return xrc;
    provenance = HaloResolution::exchanged;
#else
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "a halo peer group is attached but this build has no CUDA "
                "support, so no exchange can be performed");
#endif
  } else if (d.halo_caller_managed[slot] != 0) {
    /* Retained for callers that own mpp_update_domains themselves. It records
     * whose claim readiness is; it asserts nothing about the halo cells. A group
     * resolved here is reported as caller_declared and cannot count towards a
     * Gate C1 pass. */
    provenance = HaloResolution::caller_declared;
  } else {
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                std::string("halo group ") + halo_group_name(g) +
                    " requires a real domain exchange on this connected "
                    "topology, no exchange is installed for it, and the caller "
                    "has not declared this group caller-managed, so readiness "
                    "cannot be established");
  }

  const fv3_gpu_event ev = issue_event(d);
  const int rc = d.dependencies.mark_ready(g, ev);
  if (rc != 0)
    return fail(e, static_cast<fv3_gpu_status>(rc), api,
                std::string("could not mark halo group ") + halo_group_name(g) +
                    " ready");
  d.halo_resolution[slot] = provenance;
  return FV3_GPU_SUCCESS;
}

/* ---------------------------------------------------------------------------
 * Milestone 3A.9 §13 -- the production configuration fence, at the runtime
 * level.
 *
 * Every individual launcher already fences its own configuration: d_sw refuses
 * a non-6 hord (dgrid/wp_b_dsw_cuda.cu:895-901), Riem_Solver_C refuses
 * a_imp <= 0.999 (vertical/fv3_riem_solver_c.cu:212), the C-grid predictor
 * refuses hydrostatic (cgrid/fv3_cgrid_predictor.cu:196), and so on. Those
 * fences are the numerically meaningful ones and are not being replaced.
 *
 * They are not, however, sufficient for the requirement in
 * project_architecture.md that an unsupported configuration "should fail before
 * mutating model state". A per-stage fence fires at the stage that cares, and
 * the stages that do not care have already run by then. Concretely: with
 * grid_type = 4 the C-grid predictor takes its `grid_type >= 3` BRANCH rather
 * than refusing (cgrid/fv3_cgrid_csw.cu:151), so on the pre-3A.9 path the whole
 * C-grid half of the substep executed and wrote uc/vc/delpc/ptc/divgd to device
 * memory before WP-B's d_sw returned UNSUPPORTED_CONFIGURATION. The call failed,
 * but the state was already torn: the caller cannot retry, and cannot trust a
 * D2H either.
 *
 * So the supported subset is checked once, here, before zero_accumulators and
 * before the first launch. This is a whitelist of the documented production
 * first-run configuration, not a guess at what the kernels happen to accept.
 *
 * What is deliberately NOT checked here:
 *
 *   vtdm4 > 0  -- §13 lists it, but NumericalConfig has no vtdm4 member and
 *                 should not gain one: vtdm4 reaches the kernels already
 *                 resolved into the per-level damp_vt array (FieldId::damping_v)
 *                 and the per-level nord_v array (FieldId::damping_ndif), which
 *                 is how dyn_core.F90:699-765 itself hands it to d_sw. A scalar
 *                 vtdm4 check here would either duplicate that or contradict
 *                 the sponge levels, where the authoritative code forces the
 *                 vorticity-damping controls to zero. The live consequence of
 *                 vtdm4 > 0 -- that the do_vort_damp arms are exercised -- is
 *                 asserted by the harness through those fields instead.
 *
 *   nord/nord_t/nord_w ranges, dddmp, da_min, p_fac, delt_max -- these are
 *                 tuning parameters, not the configuration IDENTITY. They are
 *                 range-checked by the stages that consume them, and a wrong
 *                 value there does not change which code path runs.
 *
 * The fence is intentionally narrow in the other direction too: it names only
 * conditions this project has explicitly declared unsupported. It must never
 * grow into a general "reject anything unfamiliar" check, because that would
 * silently prevent a future milestone's widened subset from running while
 * looking like a correctness guarantee.
 * ------------------------------------------------------------------------- */
int fence_production_config(const NumericalConfig &cfg, const TopologyView *topo,
                           fv3_gpu_error *e, const char *api) {
  const auto reject = [&](const std::string &what) {
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "configuration outside the implemented production first-run "
                "subset: " + what +
                    " (see project_architecture.md, Production First-Run "
                    "Configuration); refused before any state was mutated");
  };

  if (cfg.grid_type != 0)
    return reject("grid_type must be 0 (gnomonic cubed sphere); grid_type >= 3 "
                  "selects different Cartesian edge/corner semantics that are "
                  "branched on but not implemented");
  if (cfg.hydrostatic)
    return reject("hydrostatic must be false; the hydrostatic path removes w, "
                  "delz and the Riemann solvers entirely");
  if (!(cfg.a_imp > 0.999))
    return reject("a_imp must be > 0.999 (the SIM1 vertical branch); the "
                  "RIM_2D, SIM3, SIM3p0 and SIM branches are not implemented");
  {
    const int32_t hord[4] = {cfg.hord_mt, cfg.hord_vt, cfg.hord_tm, cfg.hord_dp};
    const char *const name[4] = {"hord_mt", "hord_vt", "hord_tm", "hord_dp"};
    for (int i = 0; i < 4; ++i) {
      const int32_t a = hord[i] < 0 ? -hord[i] : hord[i];
      if (a != 6)
        return reject(std::string(name[i]) + " must be +/-6 (the order-6 "
                      "transport path); no other PPM order is implemented");
    }
    /* The SIGN matters and is not interchangeable: hord_dp < 0 selects the
     * positive-definite-off variant in tp_core.F90's fv_tp_2d, so hord_dp = +6
     * is a different scheme, not the same one written differently. */
    if (cfg.hord_dp > 0)
      return reject("hord_dp must be negative (-6); a positive hord_dp selects "
                    "a different fv_tp_2d variant");
  }
  if (cfg.use_cond || cfg.moist_cappa)
    return reject("use_cond / moist_cappa are not implemented in this subset "
                  "(a documented divergence from the real GEOS FV3 build, "
                  "where both are on; see the milestone 3A.9 handoff)");
  if (cfg.inline_q)
    return reject("inline_q is not implemented; tracer advection is not part "
                  "of the dyn_core subsystem in this subset");
  if (cfg.do_f3d || cfg.do_skeb || cfg.rot3)
    return reject("do_f3d / do_skeb / rot3 are not implemented");
  if (cfg.beta != 0.0)
    return reject("beta must be 0; a nonzero beta selects the backward-in-time "
                  "nh_p_grad variant and changes fp_out");
  /* m_split is checked here rather than at its old site below so that every
   * configuration refusal happens in one place, before mutation. */
  if (cfg.m_split < 1)
    return reject("m_split is not supplied; the vertical Riemann substep count "
                  "cannot be defaulted (dyn_core.F90:300 derives Riem_Solver_C's "
                  "ms = max(1, m_split/2) and :918 hands Riem_Solver3 m_split "
                  "itself, so a default would silently pick both)");
  if (!(cfg.dz_min > 0.0))
    return reject("dz_min must be > 0; the update_dz_c/update_dz_d monotonicity "
                  "clamps would otherwise admit zero-thickness layers and the "
                  "SIM1 column solve divides by dz");

  if (topo) {
    if (topo->nested || topo->regional)
      return reject("nested / regional domains are not implemented; the "
                    "nest_halo_nh path is dead code in this subset");
    if (!topo->global_cubed_sphere)
      return reject("global_cubed_sphere must be set; the supported subset is "
                    "the global gnomonic cubed sphere");
    /* npx/npy == 0 is deliberately NOT fenced here. It is a missing INPUT, not
     * an unsupported configuration, and stages that need no panel semantics are
     * legitimately drivable without it.
     *
     * CORRECTED in milestone 3A.10. This comment previously claimed that the
     * two consumers needing panel geometry -- bind_wp_b_d_sw and "WP-D's
     * physical_edges feeding a2b_ord4" -- both refuse zero at their binders.
     * Only the first half was true. WP-D CANNOT refuse a zero npx, because it
     * never reads npx/npy at all: operators/ contains zero occurrences of
     * either, and operators/wp_d_contract.cpp:398-405 derives physical_edges
     * from topology.neighbors[] instead. bind_wp_d (:53-171) therefore has no
     * value to check.
     *
     * What actually refuses a zero npx before WP-D can run is a different pair,
     * and the distinction matters because it is ORDERING rather than validation:
     *   - cgrid/fv3_cgrid_predictor.cu:251-254 and
     *     dgrid/wp_b_dsw_contract_cuda.cpp:78-80 both refuse it, and
     *   - validate_dispatch (runtime/fv3_gpu_contract.cpp:119) requires all ten
     *     stages, so a dispatch table holding WP-D without those two cannot be
     *     installed in the first place.
     * Anything that reaches WP-D has already passed through them.
     *
     * Those refusals are not pre-mutation, so a panel-geometry consumer plus an
     * npx == 0 topology can still tear device state -- a residual gap recorded
     * in the handoff rather than papered over here. Fencing it at this level
     * would have required every minimal-dispatch driver to describe a full tile
     * geometry it does not use, trading a narrow real gap for a broad false
     * requirement. */
  }
  return FV3_GPU_SUCCESS;
}
}

namespace fv3::gpu {
void clear_error(fv3_gpu_error *e) noexcept {
  if (!e) return;
  std::memset(e, 0, sizeof(*e)); e->abi_version = FV3_GPU_ABI_VERSION; e->struct_bytes = sizeof(*e);
}
int fail(fv3_gpu_error *e, fv3_gpu_status code, const char *api, const std::string &msg,
         int native, int rank, int device, uint64_t seq) noexcept {
  if (e) { clear_error(e); e->code=code; e->native_code=native; e->rank=rank; e->device=device; e->sequence=seq;
    std::strncpy(e->api, api ? api : "", sizeof(e->api)-1); std::strncpy(e->message, msg.c_str(), sizeof(e->message)-1); }
  return code;
}

int set_numerical_dispatch(fv3_gpu_context *c, const NumericalDispatch &d, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if (c->data.phase==ContextPhase::failed) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  if (d.contract_version!=internal_contract_version) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"wrong dispatch version");
  c->data.dispatch=d;
  return FV3_GPU_SUCCESS;
}
int set_numerical_config(fv3_gpu_context *c, const NumericalConfig &cfg, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if (c->data.phase==ContextPhase::failed) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  c->data.immutable.config=cfg;
  return FV3_GPU_SUCCESS;
}
int set_topology(fv3_gpu_context *c, const TopologyView &t, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if (c->data.phase==ContextPhase::failed) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  if (t.contract_version!=internal_contract_version) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"wrong topology version");
  c->data.topology=t;
  return FV3_GPU_SUCCESS;
}
/* Milestone 3A.10 §4/§9/§10 -- join the Model B peer group.
 *
 * Buffers are sized ONCE here, for the largest live group, so that no exchange
 * ever allocates. A cudaMalloc inside the acoustic loop is an implicit
 * device-wide synchronization and would quietly dismantle the persistent
 * residency model, so the allocation is hoisted out even though it makes this
 * function the place where geometry must already be known.
 *
 * Order matters and is enforced rather than documented: without a topology there
 * is no tile connectivity to size or validate against, and npx/npy == 0 is the
 * contract's "not supplied". */
int attach_halo_peer_group(fv3_gpu_context *c, halo::HaloPeerGroup *pg, int32_t tile,
                           fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if (!pg) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null peer group");
  if (c->data.phase==ContextPhase::failed)
    return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  if (c->data.halo_peer)
    return fail(e,FV3_GPU_STATE_CONFLICT,__func__,
                "this context is already attached to a halo peer group");
  if (tile < 1 || tile > halo::cubed_sphere_tiles)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"tile must be 1..6");
  if (c->data.topology.npx <= 0 || c->data.topology.npy <= 0)
    return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,
                "a topology must be installed before attaching to a peer group; "
                "npx/npy are zero, which the contract defines as not-supplied");
  if (c->data.topology.tile != tile)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,
                "the tile being attached disagrees with TopologyView::tile");
#ifndef FV3_GPU_ENABLE_CUDA
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,
              "a real halo exchange requires a CUDA build");
#else
  ContextData &d = c->data;
  const halo::HaloGeometry geom = geometry_of(d);
  if (geom.nx <= 0 || geom.ny <= 0 || geom.npz <= 0)
    return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,
                "domain bounds must be set before attaching to a peer group");
  int64_t need = 0;
  for (uint32_t g = 1; g <= static_cast<uint32_t>(FV3_GPU_H_DWIND_NEXT); ++g) {
    const halo::HaloGroupSpec spec = halo::halo_group_spec(g);
    if (!spec.live) continue;
    const int64_t n = halo::group_buffer_elements(spec, geom);
    if (n > need) need = n;
  }
  if (need <= 0)
    return fail(e,FV3_GPU_INTERNAL_ERROR,__func__,"no live halo group has a buffer size");

  void *send=nullptr, *recv=nullptr;
  cudaEvent_t ev_pack=nullptr, ev_xfer=nullptr;
  const size_t bytes = static_cast<size_t>(need) * sizeof(fv_real);
  cudaError_t rc = cudaMalloc(&send, bytes);
  if (rc == cudaSuccess) rc = cudaMalloc(&recv, bytes);
  /* Milestone 3A.10 §20, found by compute-sanitizer initcheck and fixed here.
   *
   * A slot is sized at the WORST CASE over edges and staggerings
   * (group_slot_elements uses max(nx,ny)+1 and npz+1), while the pack writes only
   * width * along_extent * nk of it. launch_halo_transfer copies the whole slot,
   * because a fixed-size copy is what makes a slot address pure arithmetic. So
   * every transfer moved a tail of never-written bytes, and initcheck reported
   * 24587 "Host API memory access error ... on access by cudaMemcpy source" for
   * the R8 six-tile run -- all from launch_halo_transfer, all at an offset in the
   * last 30% of the copy and NEVER at offset 0, which is what identifies it as
   * the slot tail rather than a pack that ran late.
   *
   * The tail is provably never READ: both sides derive the same along-extent from
   * the donor edge and the source field's staggering, and the unpack refuses a
   * mismatch outright rather than clamping. Results were therefore never
   * affected, and the multi-tile test's 20484 decoded cells confirm it.
   *
   * Zeroing once at attach is the fix rather than shortening the copy, because
   * the copy length would have to be derived from the DONOR's per-field nk, which
   * this side does not have -- and because a report that is merely rarer is
   * worse than none: 24587 known-benign records would bury the one that matters.
   * This is a one-time memset outside the acoustic loop, so it costs nothing per
   * substep and does not touch persistent residency. */
  if (rc == cudaSuccess) rc = cudaMemset(send, 0, bytes);
  if (rc == cudaSuccess) rc = cudaMemset(recv, 0, bytes);
  if (rc == cudaSuccess) rc = cudaEventCreateWithFlags(&ev_pack, cudaEventDisableTiming);
  if (rc == cudaSuccess) rc = cudaEventCreateWithFlags(&ev_xfer, cudaEventDisableTiming);
  if (rc != cudaSuccess) {
    if (send) cudaFree(send);
    if (recv) cudaFree(recv);
    if (ev_pack) cudaEventDestroy(ev_pack);
    if (ev_xfer) cudaEventDestroy(ev_xfer);
    return cuda_fail(rc, e, __func__, &d);
  }

  halo::PeerMember m{};
  m.send = static_cast<fv_real*>(send);
  m.recv = static_cast<fv_real*>(recv);
  m.pack_done = ev_pack;
  m.xfer_done = ev_xfer;
  m.transfer_stream = d.transfer_stream;
  if (!pg->attach(tile, m)) {
    cudaFree(send); cudaFree(recv);
    cudaEventDestroy(ev_pack); cudaEventDestroy(ev_xfer);
    return fail(e,FV3_GPU_STATE_CONFLICT,__func__,
                "that tile is already attached to this peer group; a duplicate "
                "would make the rendezvous count wrong and feed one tile's halo "
                "from the wrong buffer");
  }
  d.halo_peer = pg;
  d.halo_peer_tile = tile;
  d.halo_send = m.send;
  d.halo_recv = m.recv;
  d.halo_buffer_elements = need;
  d.event_halo_pack = ev_pack;
  d.event_halo_xfer = ev_xfer;
  return FV3_GPU_SUCCESS;
#endif
}

int detach_halo_peer_group(fv3_gpu_context *c, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  ContextData &d = c->data;
  if (!d.halo_peer) return FV3_GPU_SUCCESS;
  auto *pg = static_cast<halo::HaloPeerGroup*>(d.halo_peer);
  pg->detach(d.halo_peer_tile);
#ifdef FV3_GPU_ENABLE_CUDA
  /* The peers must be past their last barrier before this frees anything. That is
   * the caller's sequencing obligation -- it is the same obligation MPI_Finalize
   * imposes -- so this only synchronizes MY stream, which is a teardown boundary
   * and not an acoustic-loop one. */
  if (d.transfer_stream) cudaStreamSynchronize(static_cast<cudaStream_t>(d.transfer_stream));
  if (d.halo_send) cudaFree(d.halo_send);
  if (d.halo_recv) cudaFree(d.halo_recv);
  if (d.event_halo_pack) cudaEventDestroy(static_cast<cudaEvent_t>(d.event_halo_pack));
  if (d.event_halo_xfer) cudaEventDestroy(static_cast<cudaEvent_t>(d.event_halo_xfer));
#endif
  d.halo_send=nullptr; d.halo_recv=nullptr;
  d.event_halo_pack=nullptr; d.event_halo_xfer=nullptr;
  d.halo_buffer_elements=0; d.halo_peer=nullptr; d.halo_peer_tile=0;
  return FV3_GPU_SUCCESS;
}

HaloResolution halo_resolution_of(const fv3_gpu_context *c, fv3_gpu_halo_group g) noexcept {
  if (!c) return HaloResolution::none;
  const auto slot = static_cast<size_t>(g);
  if (slot >= c->data.halo_resolution.size()) return HaloResolution::none;
  return c->data.halo_resolution[slot];
}

int mark_halo_ready(fv3_gpu_context *c, fv3_gpu_halo_group g, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if (c->data.phase==ContextPhase::failed) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  fv3_gpu_event ev=issue_event(c->data);
  int rc=c->data.dependencies.mark_ready(g,ev);
  if (rc!=0) return fail(e,static_cast<fv3_gpu_status>(rc),__func__,"invalid halo group or event");
  /* Milestone 3A.10 §27: this entry point IS a declaration, by construction --
   * it moves no data. Record it as such so that a Gate C1 report cannot count a
   * group poked ready by a harness as one that was exchanged. */
  c->data.halo_resolution[static_cast<size_t>(g)]=HaloResolution::caller_declared;
  return FV3_GPU_SUCCESS;
}
/* Milestone 3A.9 §8: the caller takes responsibility for one halo group.
 *
 * This does NOT mark the group ready and it is not a weaker mark_halo_ready. It
 * records that the CALLER -- GEOS/FV3, which owns mpp_update_domains, or a test
 * harness that has declared what it does -- will fill this group's halo cells,
 * so that fv3_gpu_dyn_core_step's resolve_halo() may re-establish readiness at
 * the authoritative exchange point of each substep after a stage has marked the
 * group stale. Without a declaration, and on any topology whose halo exchange is
 * not provably vacuous, resolve_halo refuses rather than invent readiness.
 *
 * Declaring a group and then not filling it is a falsely-ready halo. The
 * runtime cannot detect that; the declaration is exactly where the
 * responsibility is placed, and a caller that stubs it must say so. */
int mark_halo_caller_managed(fv3_gpu_context *c, fv3_gpu_halo_group g, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!c) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if (c->data.phase==ContextPhase::failed) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  const auto slot=static_cast<size_t>(g);
  if (g<FV3_GPU_H_DWIND_IN || slot>=c->data.halo_caller_managed.size())
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid halo group");
  c->data.halo_caller_managed[slot]=1;
  return FV3_GPU_SUCCESS;
}
}

extern "C" int fv3_gpu_validate_field(const fv3_gpu_field_desc *f, fv3_gpu_error *e) {
  using namespace fv3::gpu; clear_error(e);
  if (!f || !header_ok(f->abi_version, f->struct_bytes, sizeof(*f))) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid field descriptor ABI");
  if (!f->base || f->rank == 0 || f->rank > FV3_GPU_MAX_RANK) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"base is null or rank is outside 1..4");
  const uint32_t expected = f->element_type == FV3_GPU_ELEMENT_F32 ? 4u : f->element_type == FV3_GPU_ELEMENT_F64 ? 8u : 0u;
  if (!expected || f->element_bytes != expected) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"element type and byte size disagree");
  if (f->memory_location != FV3_GPU_MEMORY_HOST && f->memory_location != FV3_GPU_MEMORY_DEVICE) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid memory location");
  uint64_t max_offset=0;
  for (uint32_t n=0;n<f->rank;++n) {
    if (!f->extent[n] || f->stride_elements[n] <= 0) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"extent must be nonzero and stride positive");
    const int64_t upper=f->lower[n]+(int64_t)f->extent[n]-1;
    if (f->compute_lower[n]<f->lower[n] || f->compute_upper[n]>upper || f->compute_lower[n]>f->compute_upper[n]) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"compute bounds lie outside allocation");
    if (f->extent[n]-1 > (std::numeric_limits<uint64_t>::max()-max_offset)/(uint64_t)f->stride_elements[n]) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"descriptor size overflow");
    max_offset += (f->extent[n]-1)*(uint64_t)f->stride_elements[n];
  }
  if (max_offset > (std::numeric_limits<uint64_t>::max()/f->element_bytes)-1 || (max_offset+1)*f->element_bytes > f->allocation_bytes) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"allocation_bytes is smaller than addressed view");
  if (f->alignment_bytes && ((uintptr_t)f->base % f->alignment_bytes)) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"base does not meet declared alignment");
  return FV3_GPU_SUCCESS;
}

extern "C" int fv3_gpu_create(const fv3_gpu_create_desc *d, fv3_gpu_context **out, fv3_gpu_error *e) {
  using namespace fv3::gpu; clear_error(e);
  if (!out) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"context output is null");
  *out=nullptr;
  if (!d || !header_ok(d->abi_version,d->struct_bytes,sizeof(*d)) || d->mpi_size<1 || d->mpi_rank<0 || d->mpi_rank>=d->mpi_size || d->is>d->ie || d->js>d->je || d->npz<1)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid create descriptor");
  if ((uint32_t)active_precision != d->precision) return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"requested precision differs from compiled FV precision");
#ifndef FV3_GPU_ENABLE_CUDA
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"runtime was built without CUDA");
#else
  int count=0; cudaError_t rc=cudaGetDeviceCount(&count); if(rc!=cudaSuccess) return cuda_fail(rc,e,__func__);
  if(count==0) return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"no CUDA devices are visible");
  int device=d->device_ordinal; if(device<0) device=d->mpi_rank%count;
  if(device<0 || device>=count) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"device ordinal is unavailable");
  rc=cudaSetDevice(device); if(rc!=cudaSuccess) return cuda_fail(rc,e,__func__);
  auto *ctx=new(std::nothrow) fv3_gpu_context; if(!ctx) return fail(e,FV3_GPU_OUT_OF_MEMORY,__func__,"context allocation failed");
  ctx->data.create=*d; ctx->data.create.device_ordinal=device;
  ctx->data.immutable.bounds={d->is,d->ie,d->js,d->je,d->npz,d->halo_width};
  ctx->data.immutable.topology=&ctx->data.topology;
  cudaStream_t compute{}, transfer{};
  rc=cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking); if(rc==cudaSuccess) rc=cudaStreamCreateWithFlags(&transfer,cudaStreamNonBlocking);
  if(rc!=cudaSuccess){ if(compute) cudaStreamDestroy(compute); delete ctx; return cuda_fail(rc,e,__func__); }
  /* cudaEventDisableTiming: these exist only to order the two non-blocking
   * streams against each other (see ContextData::event_compute_done). Timing
   * support would add measurable record overhead for a capability nothing uses. */
  cudaEvent_t compute_done{}, transfer_done{};
  rc=cudaEventCreateWithFlags(&compute_done,cudaEventDisableTiming);
  if(rc==cudaSuccess) rc=cudaEventCreateWithFlags(&transfer_done,cudaEventDisableTiming);
  if(rc!=cudaSuccess){
    if(compute_done) cudaEventDestroy(compute_done);
    cudaStreamDestroy(compute); cudaStreamDestroy(transfer); delete ctx;
    return cuda_fail(rc,e,__func__);
  }
  ctx->data.compute_stream=compute; ctx->data.transfer_stream=transfer;
  ctx->data.event_compute_done=compute_done; ctx->data.event_transfer_done=transfer_done;
  *out=ctx; return FV3_GPU_SUCCESS;
#endif
}

extern "C" int fv3_gpu_initialize_state(fv3_gpu_context *c, const fv3_gpu_state_desc *s, fv3_gpu_event *ev, fv3_gpu_error *e) {
  using namespace fv3::gpu; clear_error(e); if(ev)*ev=0;
  if(!c || !s || !ev || !header_ok(s->abi_version,s->struct_bytes,sizeof(*s)) || !s->fields || !s->field_count) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid context, state descriptor, or event output");
  if(c->data.phase!=ContextPhase::created) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context is not in created state");
#ifdef FV3_GPU_ENABLE_CUDA
  std::vector<DeviceField> staged; staged.reserve(s->field_count); FieldRegistry registry; auto stream=(cudaStream_t)c->data.transfer_stream;
  for(uint32_t i=0;i<s->field_count;++i){ if(fv3_gpu_validate_field(&s->fields[i],e)!=0) goto rollback; for(const auto &prior:staged) if(prior.host.field_id==s->fields[i].field_id){fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"duplicate field id");goto rollback;} DeviceField x; x.host=s->fields[i];
    if(cudaMalloc(&x.device,x.host.allocation_bytes)!=cudaSuccess) goto rollback; if(x.host.memory_location==FV3_GPU_MEMORY_HOST && cudaMemcpyAsync(x.device,x.host.base,x.host.allocation_bytes,cudaMemcpyHostToDevice,stream)!=cudaSuccess){cudaFree(x.device);goto rollback;} if(registry.insert(x.host,x.device,e)!=0){cudaFree(x.device);goto rollback;} x.host_epoch=x.device_epoch=1; staged.push_back(x); }
  { uint64_t ni=(uint64_t)(c->data.create.ie-c->data.create.is+1+2*c->data.create.halo_width),nj=(uint64_t)(c->data.create.je-c->data.create.js+1+2*c->data.create.halo_width),cells=0,values=0;
    if(mul_overflow(ni,nj,cells)||mul_overflow(cells,(uint64_t)c->data.create.npz+1,values)){fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"workspace dimension overflow");goto rollback;}
    uint64_t value_bytes=0,cgrid_bytes=0,transport_bytes=0,vertical_bytes=0,operator_bytes=0;
    /* WorkspaceId::transport is sized in whole (ni x nj x npz+1) fv_real
     * arrays. It is shared, sequentially, by every WP-B service:
     *   Milestone 3A.4 update_dz_d height transport -- 17 arrays
     *   Milestone 3A.8 d_sw driver                  -- 34 arrays
     * The d_sw figure is WpBDSwArena::array_count (dgrid/wp_b_dsw_cuda.cuh),
     * which enumerates every sw_core.F90 local that is not a registered field:
     * ub/vb/wk/ke/vort/fx/fy/gx/gy/ra_x/ra_y/fx2/fy2/dw (14), the fv_tp_2d +
     * deln_flux scratch reused across the delp/w/pt/vort transports (7), the
     * PPM momentum workspace (4), smag_corner's private locals plus a2b_ord4
     * scratch (8) and one slot subdivided into the per-level derived
     * coefficient arrays (1). The binder re-checks the slice against that
     * constant, so a stale literal here fails loudly instead of corrupting
     * neighbouring scratch. */
    if(mul_overflow(values,sizeof(fv_real),value_bytes)||mul_overflow(value_bytes,12,cgrid_bytes)||mul_overflow(value_bytes,34,transport_bytes)||mul_overflow(value_bytes,16,vertical_bytes)||mul_overflow(value_bytes,12,operator_bytes)){fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"workspace byte-size overflow");goto rollback;}
    WorkspaceRequest requests[]={{WorkspaceId::cgrid,cgrid_bytes,256},{WorkspaceId::transport,transport_bytes,256},{WorkspaceId::vertical,vertical_bytes,256},{WorkspaceId::operators,operator_bytes,256}};
    if(build_workspace_layout(requests,4,c->data.workspace.layout,e)!=0)goto rollback;if(cudaMalloc(&c->data.workspace.allocation,c->data.workspace.layout.total_bytes)!=cudaSuccess)goto rollback;
    for(auto &slice:c->data.workspace.layout.slices)slice.data=(char*)c->data.workspace.allocation+slice.offset; }
  c->data.state.fields=std::move(staged);c->data.state.registry=registry;c->data.immutable.grid=&c->data.state.registry;c->data.phase=ContextPhase::initialized; *ev=issue_event(c->data); return FV3_GPU_SUCCESS;
rollback: for(auto &x:staged) cudaFree(x.device);if(c->data.workspace.allocation){cudaFree(c->data.workspace.allocation);c->data.workspace={};} c->data.phase=ContextPhase::failed; if(e && e->code!=FV3_GPU_SUCCESS)return e->code; return fail(e,FV3_GPU_CUDA_ERROR,__func__,"state or workspace allocation/upload failed");
#else
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"runtime was built without CUDA");
#endif
}

extern "C" int fv3_gpu_sync_from_host(fv3_gpu_context *c,const fv3_gpu_field_mask *m,fv3_gpu_event *ev,fv3_gpu_error *e){
  using namespace fv3::gpu; clear_error(e); if(ev)*ev=0; if(!c||!ev)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context or event"); int rc=validate_mask(m,e,__func__);if(rc)return rc; if(c->data.phase!=ContextPhase::initialized)return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context is not initialized");
#ifdef FV3_GPU_ENABLE_CUDA
  auto stream=(cudaStream_t)c->data.transfer_stream; for(auto &x:c->data.state.fields) { if(x.host.field_id>=256)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"field id exceeds mask"); if((m->words[x.host.field_id/64]>>(x.host.field_id%64))&1u){ if(x.host.memory_location==FV3_GPU_MEMORY_HOST){auto z=cudaMemcpyAsync(x.device,x.host.base,x.host.allocation_bytes,cudaMemcpyHostToDevice,stream);if(z!=cudaSuccess)return cuda_fail(z,e,__func__,&c->data);} ++x.host_epoch;x.device_epoch=x.host_epoch;}} *ev=issue_event(c->data);return 0;
#else
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"runtime was built without CUDA");
#endif
}
/* Milestone 3A.10: the first public way to describe the decomposition.
 *
 * Deliberately NOT tolerant. The whole-tile seam algebra in
 * fv3_gpu_halo_topology.hpp is exact for one rank per tile and is not an
 * approximation of anything else, so a decomposition it cannot represent is
 * refused here rather than rounded into it. mpi_size is checked against the six
 * tiles because a rank owning a sub-tile block has same-tile neighbours on some
 * edges, which this entry cannot express. */
extern "C" int fv3_gpu_set_cubed_sphere_topology(fv3_gpu_context *c,int32_t tile,int64_t npx,int64_t npy,
                                                 const int32_t *rank_of_tile,fv3_gpu_error *e){
  using namespace fv3::gpu; clear_error(e);
  if(!c||!rank_of_tile)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context or rank table");
  if(c->data.phase==ContextPhase::failed)return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  if(c->data.create.mpi_size!=halo::cubed_sphere_tiles)
    return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,
                "cubed-sphere topology entry supports one rank per tile only; sub-tile decomposition is not yet implemented");
  TopologyView t{};
  if(!halo::make_cubed_sphere_topology(tile,npx,npy,rank_of_tile,t))
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"inconsistent cubed-sphere topology arguments");
  /* The rank we were handed for our own tile must be us, or the seam algebra is
   * describing somebody else's neighbours. */
  if(rank_of_tile[tile-1]!=c->data.create.mpi_rank)
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"rank_of_tile disagrees with this context's mpi_rank");
  return set_topology(c,t,e);
}
extern "C" int fv3_gpu_sync_to_host(fv3_gpu_context *c,const fv3_gpu_field_mask *m,fv3_gpu_event *ev,fv3_gpu_error *e){
  using namespace fv3::gpu; clear_error(e); if(ev)*ev=0; if(!c||!ev)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context or event"); int rc=validate_mask(m,e,__func__);if(rc)return rc; if(c->data.phase!=ContextPhase::initialized)return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context is not initialized");
#ifdef FV3_GPU_ENABLE_CUDA
  auto stream=(cudaStream_t)c->data.transfer_stream;
  /* Milestone 3A.9: this is an EXPLICIT D2H boundary, so it must read the
   * results of the compute work already enqueued -- not whatever happens to be
   * in device memory when the copy engine gets there. */
  {auto z=order_after(c->data.compute_stream,c->data.transfer_stream,c->data.event_compute_done);
   if(z!=cudaSuccess)return cuda_fail(z,e,__func__,&c->data);}
  for(auto &x:c->data.state.fields){if(x.host.field_id>=256)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"field id exceeds mask");if((m->words[x.host.field_id/64]>>(x.host.field_id%64))&1u && x.host.memory_location==FV3_GPU_MEMORY_HOST){auto z=cudaMemcpyAsync(x.host.base,x.device,x.host.allocation_bytes,cudaMemcpyDeviceToHost,stream);if(z!=cudaSuccess)return cuda_fail(z,e,__func__,&c->data);x.host_epoch=x.device_epoch;}} *ev=issue_event(c->data);return 0;
#else
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"runtime was built without CUDA");
#endif
}
extern "C" int fv3_gpu_dyn_core_step(fv3_gpu_context *c,const fv3_gpu_step_desc *s,fv3_gpu_event *ev,fv3_gpu_error *e){
  using namespace fv3::gpu; clear_error(e); if(ev)*ev=0;
  if(!c||!s||!ev||!header_ok(s->abi_version,s->struct_bytes,sizeof(*s)))
    return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"invalid context, step descriptor, or event output");
  if(c->data.phase!=ContextPhase::initialized) return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context is not initialized");
  if(s->dependency && s->dependency>c->data.last_event) return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"step dependency references an unknown event");

  const NumericalDispatch &dispatch=c->data.dispatch;
  if(dispatch.contract_version!=internal_contract_version)
    return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"numerical dispatch table is not installed");
  const bool any_stage = dispatch.cgrid_predictor||dispatch.cgrid_vertical_solver||dispatch.cgrid_pressure||
                          dispatch.dgrid_dynamics||dispatch.dgrid_height_transport||dispatch.dgrid_height||
                          dispatch.vertical_solver||dispatch.nh_pressure||dispatch.post_loop_heating;
  if(!any_stage||!dispatch.corner_edge_service)
    return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"numerical dispatch table has no usable stages wired");

  const NumericalConfig &config=c->data.immutable.config;
  const int32_t n_split=s->n_split>0?s->n_split:1;

  /* Milestone 3A.9 §13. FIRST, before zero_accumulators and before any launch,
   * so an unsupported configuration cannot leave the caller with half-updated
   * device state. See fence_production_config above for what it checks, what it
   * deliberately does not, and why a per-stage fence alone was not enough.
   *
   * Note this does NOT set ContextPhase::failed. A configuration refusal is a
   * caller error that mutated nothing, so the context is still usable -- the
   * caller can install a supported configuration and step again. Marking it
   * failed would conflate "you asked for something unimplemented" with "the
   * device state is now untrustworthy", and the whole point of fencing before
   * mutation is that those are different situations. */
  if(const int rc_cfg=fence_production_config(config,c->data.immutable.topology,e,__func__))
    return rc_cfg;

  /* -------------------------------------------------------------------------
   * Milestone 3A.9 (§7): once-per-call controls.
   *
   * dyn_core.F90:296-298, inside the `if (init_step)` block that runs before the
   * acoustic loop:
   *     dt  = bdt / real(n_split)
   *     dt2 = 0.5*dt
   *     rdt = 1.0/dt
   * The division by n_split happens HERE and nowhere else. Milestone 3A.8 had a
   * stopgap in the d_sw binder (dgrid/wp_b_dsw_contract_cuda.cpp) that divided
   * dispatch_params->dt by n_split itself, because this runtime was handing it
   * the big timestep and walking the stage list once. That stopgap has been
   * removed in this milestone; dividing in both places would have shortened the
   * integration by a factor of n_split.
   *
   * `bdt` is retained explicitly (see DispatchParams in fv3_gpu_internal.hpp):
   * the diffusive-heating tail bounds its increment with bdt*delt_max
   * (dyn_core.F90:1231/:1240) and must not reconstruct it as dt*n_split.
   * ------------------------------------------------------------------------- */
  DispatchParams &p=c->data.dispatch_params;
  p.contract_version=internal_contract_version;
  p.bdt=s->bdt;
  p.dt=s->bdt/static_cast<double>(n_split);           // dyn_core.F90:296
  p.dt2=0.5*p.dt;                                     // dyn_core.F90:297
  p.rdt=(p.dt!=0.0)?1.0/p.dt:0.0;                     // dyn_core.F90:298
  p.model_step=s->model_step; p.acoustic_sequence=s->acoustic_sequence;
  p.n_split=n_split; p.k_split=s->k_split>0?s->k_split:1;
  /* dyn_core.F90:918 -- Riem_Solver3 receives flagstruct%m_split itself.
   * Riem_Solver_C's ms is DIFFERENT (max(1, m_split/2), :300) and is derived at
   * that call site from config.m_split; it is deliberately not carried here,
   * because one DispatchParams member cannot hold both. The previous hardcoded
   * `p.vertical_ms=1` silently forced a single vertical substep on the D-grid
   * Riemann solver regardless of the namelist.
   *
   * m_split >= 1 is guaranteed by fence_production_config above, which now owns
   * every configuration refusal so that all of them happen before mutation and
   * with one consistent message. */
  p.vertical_ms=config.m_split;
  p.init_step=s->init_step; p.end_step=s->end_step;
  /* Reset so a step whose heating gate is closed cannot leave a previous call's
   * pass counters visible to anything. The post-loop block below sets them. */
  p.diffusion_pass=0; p.diffusion_passes=0;

  /* fp_out. dyn_core.F90:924 passes `beta<-0.1` into Riem_Solver3's fp_out
   * dummy (nh_core.F90:69), which selects ppe = pe2+pem instead of ppe = pe2
   * (fv3_vertical_column.hpp:189). launch_nh_pressure_cuda rejects any config
   * with beta != 0 (operators/wp_d_contract_cuda.cpp:21), so within the
   * supported production subset beta == 0 and fp_out is constantly false. It is
   * still computed from config rather than hard-coded, so that widening the beta
   * fence cannot silently leave fp_out wrong. */
  p.fp_out=(config.beta < -0.1)?1:0;                  // dyn_core.F90:924

  /* beta_d has no DispatchParams member and needs none. dyn_core.F90:474-481
   * sets beta_d = 0 on the first substep and beta thereafter, and beta_d is
   * consumed only by nh_p_grad. With the beta==0 fence above, beta_d is 0 for
   * every substep, so the it==1 special case is a no-op in the supported subset.
   * A consumer that ever needs it can derive it without a contract change:
   * beta_d = (acoustic_index == 0) ? 0 : config.beta. */

  ExecutionContext x{};
  x.contract_version=internal_contract_version;
  x.fields=&c->data.state.registry;
  x.immutable=&c->data.immutable;
  x.dispatch_params=&c->data.dispatch_params;
  x.workspace=&c->data.workspace.layout;
  x.submission=Submission{internal_contract_version,c->data.compute_stream,StreamId::compute,s->acoustic_sequence};
  x.dependencies=&c->data.dependencies;
  x.corner_service=dispatch.corner_edge_service;
  /* Milestone 3A.9 §6: the table being walked, so a stage that sequences another
   * work package's service can reach it. `dispatch` is a reference to
   * c->data.dispatch, owned by the context, so the pointer outlives this step.
   * See ExecutionContext::dispatch for why this exists at all. */
  x.dispatch=&dispatch;

  int rc=validate_execution_context(x,e);
  if(rc){ c->data.phase=ContextPhase::failed; return rc; }

#ifdef FV3_GPU_ENABLE_CUDA
  /* Milestone 3A.9: the mirror of the ordering in fv3_gpu_sync_to_host. Any H2D
   * the caller enqueued (fv3_gpu_initialize_state / fv3_gpu_sync_to_device) is
   * on the transfer stream, and the kernels below run on the compute stream, so
   * without this the first substep could read un-copied device memory. Today's
   * harnesses happen to host-block on the H2D event first, which hid it; a
   * caller following the persistent-residency model has no reason to. */
  {auto z=order_after(c->data.transfer_stream,c->data.compute_stream,c->data.event_transfer_done);
   if(z!=cudaSuccess){ return cuda_fail(z,e,__func__,&c->data); }}
  // dyn_core.F90:368-379, once per call, BEFORE the acoustic loop.
  rc=zero_accumulators(c->data,e,__func__);
  if(rc){ c->data.phase=ContextPhase::failed; return rc; }
#endif

  /* -------------------------------------------------------------------------
   * The acoustic loop: dyn_core.F90:399 `do it=1,n_split` .. :1196 `enddo`.
   *
   * PER-SUBSTEP stages, in authoritative order:
   *
   *   cgrid_predictor       d2a2c_vect + divergence_corner + c_sw over k=1..npz
   *                         (:515-523, with the nord>0 divgd corner exchange
   *                         started at :525-531)
   *   cgrid_vertical_solver gz<->zh sync (:548-575), update_dz_c (:577, dt2),
   *                         Riem_Solver_C (:583, dt2)
   *   cgrid_pressure        p_grad_c (:614, dt2)
   *   dgrid_dynamics        d_sw over k=1..npz (:700-829) and the divg2 block
   *   dgrid_height          update_dz_d (:905-906, rdt) -- WP-C sequencing over
   *                         WP-B's dgrid_height_transport service
   *   vertical_solver       Riem_Solver3 (:918-923, dt, ms=m_split)
   *   nh_pressure           pk3_halo/pe_halo, gz=zh*grav, nh_p_grad (:940-1019)
   *
   * post_loop_heating is deliberately NOT on this list. Its numerical content is
   * del2_cubed(heat_source, ...) followed by the diffusive-heating update, and
   * dyn_core runs that ONCE at :1212-1214, AFTER `enddo` at :1196 -- not per
   * substep. Looping the whole 3A.8 stage array n_split times would apply the
   * heat-source filter n_split times to a heat_source that had only partially
   * accumulated, which is why §7 forbids exactly that shortcut.
   *
   * A null slot is skipped rather than treated as an orchestration error, so a
   * partially wired dispatch table still runs what it has; a wired stage that
   * fails aborts the step, with no silent host fallback.
   *
   * dgrid_height_transport is NOT a stage on this list and must never become
   * one. It is a SERVICE: nh_utils.F90 has no standalone call to it, and
   * dyn_core.F90 reaches it only through update_dz_d (:905). It is now reached
   * per substep, but as a callee of dgrid_height -- see §5/§6 in
   * vertical/fv3_vertical_kernels.cu:launch_dgrid_height_stage.
   * ------------------------------------------------------------------------- */
  /* Milestone 3A.9 §8/§9: the substep script is now an explicit ordered
   * sequence rather than an array walk, because the authoritative halo
   * resolution points are INTERLEAVED with the stages and cannot be expressed
   * as a stage list. See resolve_halo() above for what "resolved" is allowed to
   * mean on this path.
   *
   *   HALO STATE TRANSITION TABLE (per substep it)
   *
   *   Fortran      group        entry state          producer of readiness         made stale by            next consumer
   *   :402/:504    H_DWIND_IN   unknown | ready      resolve_halo, top of substep  (never)                  cgrid_predictor  (fv3_cgrid_predictor.cu:539)
   *   :427/:507    H_W_IN       unknown | ready      resolve_halo, top of substep  (never)                  cgrid_predictor  (:549)
   *   :459/:552    H_GZ_FIRST   unknown              resolve_halo, it==1 only      (never)                  cgrid_vertical_solver (fv3_vertical_kernels.cu:604, acoustic_index==0)
   *   :528/:636    H_DIVGD      stale after predictor resolve_halo, post-predictor cgrid_predictor (:831)    dgrid_dynamics   (wp_b_contract.cpp:66)
   *   :618/:640    H_CWIND      stale after pressure resolve_halo, post-pressure   cgrid_predictor (:844),   dgrid_dynamics   (wp_b_contract.cpp:63)
   *                                                                                cgrid_pressure (fv3_cgrid.cu:122)
   *   :844/:878    H_DSCALAR    unknown | ready      resolve_halo, post-dynamics   (never)                  dgrid_height     (fv3_vertical_kernels.cu:140)
   *   :931/:967    H_ZH         unknown | ready      resolve_halo, post-Riemann    (never)                  nh_pressure      (wp_d_contract_cuda.cpp:26)
   *   :931/:967    H_PKC2       unknown | ready      resolve_halo, post-Riemann    (never)                  nh_pressure      (wp_d_contract_cuda.cpp:27)
   *   :2288        H_HEAT       unknown | ready      resolve_halo, after the loop  (never)                  post_loop_heating pass 0 (wp_d_contract_cuda.cpp:114)
   *
   *   Two groups are deliberately never resolved:
   *     H_Q_INLINE     -- inline_q is outside the supported subset (:419/:631),
   *                       so no stage requires it and no exchange occurs.
   *     H_DWIND_NEXT   -- :1109 starts i_pack(8) for the NEXT substep; that is
   *                       the same physical exchange whose completion is
   *                       H_DWIND_IN at the top of the next iteration, which the
   *                       loop already resolves. Resolving it separately would
   *                       double-count one exchange, and nothing require()s it.
   *
   * H_GZ_FIRST is the one group whose require() site cannot be reached from
   * here: it lives INSIDE launch_cgrid_vertical_solver, between the gz
   * reconstruction and the gz<->zh sync, matching Fortran's :459 -> :552 split
   * across the C-grid vertical group. So it is resolved immediately before that
   * stage. On the only topology resolve_halo will accept, the exchange copies
   * nothing, so resolving before rather than inside the stage skips no
   * operation; on any other topology resolve_halo refuses outright, so the
   * earlier placement can never manufacture a readiness claim that is false.
   */
  auto resolve=[&](fv3_gpu_halo_group g)->bool{
    rc=resolve_halo(c->data,g,e,__func__);
    if(rc!=FV3_GPU_SUCCESS){ c->data.phase=ContextPhase::failed; return false; }
    return true;
  };
  auto run=[&](LaunchFn stage)->bool{
    if(!stage) return true;   // a null slot is skipped, not an orchestration error
    rc=stage(x,e);
    if(rc!=FV3_GPU_SUCCESS){ c->data.phase=ContextPhase::failed; return false; }
    return true;
  };

  for(int32_t it=1;it<=n_split;++it){
    p.acoustic_index=it-1;

    /* remap_step: dyn_core.F90:404-408
     *   remap_step = (breed_vortex_inline .or. it==n_split)
     * breed_vortex_inline is a vortex-breeding option outside the supported
     * production subset, so it reduces to it==n_split here. */
    p.remap_step=(it==n_split)?1:0;

    /* last_call: dyn_core.F90:924 passes `remap_step` -- NOT `last_step` -- into
     * Riem_Solver3's last_call dummy (nh_core.F90:69), which gates the
     * peln/pk/pe writeback (fv3_vertical_column.hpp:188).
     *
     * This corrects a real defect: the previous code set last_call from
     * step_desc::end_step. end_step is dyn_core's `end_step` argument and feeds
     * `last_step = (it==n_split .and. end_step)` at :483, which controls the
     * omega finalization and diagnostics -- a DIFFERENT quantity. With
     * end_step=1 the old value made every substep write peln/pk/pe; with
     * end_step=0 it made none of them do so, including the final substep on
     * which the remap stage depends. */
    p.last_call=p.remap_step;

    if(!resolve(FV3_GPU_H_DWIND_IN)) return rc;               // :402 -> :504
    if(!resolve(FV3_GPU_H_W_IN)) return rc;                   // :427 -> :507
    if(!run(dispatch.cgrid_predictor)) return rc;             // :515-523
    /* :525-531 is gated on `nord > 0`; when nord == 0 the Fortran neither packs
     * nor completes the divgd corner update, and cgrid_predictor correspondingly
     * leaves H_DIVGD alone (fv3_cgrid_predictor.cu:831). The WP-B binder
     * require()s H_DIVGD unconditionally, so an nord == 0 config fails loudly at
     * dgrid_dynamics rather than running with an unproduced halo. That is the
     * correct outcome: with nord == 0 divgd has no producer at all. */
    if(config.nord>0 && !resolve(FV3_GPU_H_DIVGD)) return rc; // :528 -> :636
    if(it==1 && !resolve(FV3_GPU_H_GZ_FIRST)) return rc;      // :459 -> :552
    if(!run(dispatch.cgrid_vertical_solver)) return rc;       // :543-583
    if(!run(dispatch.cgrid_pressure)) return rc;              // :614
    if(!resolve(FV3_GPU_H_CWIND)) return rc;                  // :618 -> :640
    if(!run(dispatch.dgrid_dynamics)) return rc;              // :700-829
    if(!resolve(FV3_GPU_H_DSCALAR)) return rc;                // :844 -> :878
    if(!run(dispatch.dgrid_height)) return rc;                // :905-906
    if(!run(dispatch.vertical_solver)) return rc;             // :918-923
    if(!resolve(FV3_GPU_H_ZH)) return rc;                     // :931 -> :967
    if(!resolve(FV3_GPU_H_PKC2)) return rc;                   // :931 -> :967
    if(!run(dispatch.nh_pressure)) return rc;                 // :940-1019
  }

  /* -------------------------------------------------------------------------
   * After the loop: dyn_core.F90:1212-1214
   *
   *   if ( n_con/=0 .and. flagstruct%d_con > 1.e-5 ) then
   *     nf_ke = min(3, flagstruct%nord+1)
   *     call del2_cubed(heat_source, cnst_0p20*gridstruct%da_min, ..., nf_ke, bd)
   *
   * and then the diffusive-heating update over k=1..n_con (:1220-1265).
   *
   * del2_cubed's own `ntimes = min(3, nmax)` (dyn_core.F90:2288) with nmax=nf_ke
   * makes the pass count exactly nf_ke, since nf_ke is already <= 3.
   * launch_post_loop_heating implements ONE pass per call and requires
   * diffusion_passes == config.nf_ke, so the pass loop lives here. The heating
   * update is emitted by the final pass (pass == ntimes-1), matching the Fortran
   * ordering: all del2 passes complete, then the temperature update.
   *
   * The caller owns nf_ke = min(3, nord+1); it is validated to 1..3 by the
   * stage. Both halves of the gate are checked here rather than inside the
   * stage, because when the gate is closed the Fortran never allocates
   * heat_source at all and the stage has nothing legitimate to read.
   * ------------------------------------------------------------------------- */
  if(dispatch.post_loop_heating && config.n_con!=0 && config.d_con>1.0e-5){
    /* del2_cubed updates heat_source's halo before each of its passes
     * (dyn_core.F90:2288 onwards). launch_post_loop_heating require()s H_HEAT on
     * pass 0 only (wp_d_contract_cuda.cpp:114), so resolve it once here. */
    if(!resolve(FV3_GPU_H_HEAT)) return rc;
    const int32_t passes=config.nf_ke;
    p.diffusion_passes=passes;
    for(int32_t pass=0;pass<passes;++pass){
      p.diffusion_pass=pass;
      rc=dispatch.post_loop_heating(x,e);
      if(rc!=FV3_GPU_SUCCESS){ c->data.phase=ContextPhase::failed; return rc; }
    }
  }

  *ev=issue_event(c->data);
  return FV3_GPU_SUCCESS;
}
extern "C" int fv3_gpu_event_wait(fv3_gpu_context *c,fv3_gpu_event ev,fv3_gpu_error *e){using namespace fv3::gpu;clear_error(e);if(!c||!ev||ev>c->data.last_event)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"unknown event");
#ifdef FV3_GPU_ENABLE_CUDA
  /* Milestone 3A.9: wait for BOTH streams. This is the host's only blocking
   * wait, and callers use it to mean "the work up to this event is done and I
   * may touch host memory". Synchronizing only the transfer stream made that
   * false for a step event: fv3_gpu_dyn_core_step's kernels are on the compute
   * stream, so waiting on a step event returned immediately while the acoustic
   * loop was still running. Ordering the copies (above) fixes device-side
   * correctness; this fixes what the host is entitled to conclude.
   *
   * Neither stream is looked up by event id -- the public event handle is a
   * host-side monotonic counter, not a device object, so it cannot select a
   * stream. The wait is therefore conservative by construction; the check
   * against last_event above is what makes an unknown event an error. */
  cudaError_t wrc=cudaStreamSynchronize((cudaStream_t)c->data.compute_stream);
  if(wrc==cudaSuccess) wrc=cudaStreamSynchronize((cudaStream_t)c->data.transfer_stream);
  return cuda_fail(wrc,e,__func__,&c->data);
#else
  return fail(e,FV3_GPU_UNSUPPORTED_CONFIGURATION,__func__,"runtime was built without CUDA");
#endif
}
extern "C" int fv3_gpu_get_capabilities(fv3_gpu_context*,fv3_gpu_capabilities *p){if(!p||!header_ok(p->abi_version,p->struct_bytes,sizeof(*p)))return FV3_GPU_INVALID_ARGUMENT;
#ifdef FV3_GPU_ENABLE_CUDA
  p->compiled_with_cuda=1;int n=0;if(cudaGetDeviceCount(&n)!=cudaSuccess)n=0;p->device_count=(uint32_t)n;
#else
  p->compiled_with_cuda=0;p->device_count=0;
#endif
  p->precision_mask=(1u<<fv3::gpu::active_precision);p->communication_stage=1;p->feature_mask=0;return 0;}
extern "C" int fv3_gpu_finalize(fv3_gpu_context **pc,fv3_gpu_error *e){using namespace fv3::gpu;clear_error(e);if(!pc||!*pc)return 0;auto*c=*pc;
#ifdef FV3_GPU_ENABLE_CUDA
  cudaSetDevice(c->data.create.device_ordinal);cudaStreamSynchronize((cudaStream_t)c->data.compute_stream);cudaStreamSynchronize((cudaStream_t)c->data.transfer_stream);for(auto &x:c->data.state.fields)cudaFree(x.device);if(c->data.workspace.allocation)cudaFree(c->data.workspace.allocation);if(c->data.event_compute_done)cudaEventDestroy((cudaEvent_t)c->data.event_compute_done);if(c->data.event_transfer_done)cudaEventDestroy((cudaEvent_t)c->data.event_transfer_done);cudaStreamDestroy((cudaStream_t)c->data.compute_stream);cudaStreamDestroy((cudaStream_t)c->data.transfer_stream);
#endif
  delete c;*pc=nullptr;return 0;}
