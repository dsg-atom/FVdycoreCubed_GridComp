#include "fv3_gpu_api.h"
#include "fv3_gpu_context.hpp"
#include "fv3_gpu_errors.hpp"
#include "fv3_gpu_halo_peer.hpp"
#include "fv3_gpu_halo_topology.hpp"
#include "fv3_gpu_types.hpp"
#include "vertical/fv3_vertical_column.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <unistd.h>
#include <vector>

#ifdef FV3_GPU_ENABLE_CUDA
#include <cuda_runtime_api.h>
#include "fv3_gpu_state_poison.hpp"
#include "fv3_gpu_heat_accumulator.hpp"
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

/* Milestone 3A.11T. Restores the init_step initialization of the transient work
 * arrays that this port had dropped, and which is load-bearing rather than
 * defensive.
 *
 * dyn_core.F90:328-350, inside `if ( init_step )`:
 *
 *     allocate(    gz(isd:ied, jsd:jed ,npz+1) )
 *       call init_ijk_mem(isd,ied, jsd,jed, npz+1, gz, huge_r)
 *     ...
 *          allocate( zh(isd:ied, jsd:jed, npz+1) )
 *   !      call init_ijk_mem(isd,ied, jsd,jed, npz+1, zh, huge_r )
 *          allocate ( pk3(isd:ied,jsd:jed,npz+1) )
 *          call init_ijk_mem(isd,ied, jsd,jed, npz+1, pk3, huge_r )
 *
 * In the Fortran these arrays are ALLOCATED per init_step and deallocated at
 * end_step, so gz and pk3 are re-poisoned at the top of every dynamics call for
 * free. This runtime holds them as persistent WP-E device fields, which is the
 * intended architecture -- but persistence silently converted a Fortran
 * transient into a Fortran SAVEd array, and that changes the arithmetic:
 *
 *   :444-454  gz := height, over the OWNED box is:ie, js:je only
 *   :459/:552 halo-exchange gz (i_pack(5)) -- fills the four EDGE halos
 *   :557-563  zh := gz, over the FULL box isd:ied, jsd:jed   (it == 1)
 *   :970-977  gz := zh*grav, over is-2:ie+2, js-2:je+2       (end of substep)
 *
 * The four DIAGONAL corner ghost blocks are written by neither :444-454 (owned
 * only) nor the exchange (a four-edge exchange never fills them; FMS leaves the
 * cube-corner diagonals undefined too, which is the entire reason fill_corners
 * exists). But they ARE inside :970-977's box and inside :557-563's box. So the
 * pair of copies closes a loop through cells nothing ever defines:
 *
 *     zh(corner) := gz(corner) ; gz(corner) := zh(corner)*grav
 *
 * With a fresh huge_r each call that loop is harmless -- it reads a constant
 * sentinel and FV3 tolerates it because no stage reads gz or zh there. With a
 * persistent gz it becomes a geometric amplifier that multiplies those cells by
 * grav EVERY step. Measured on the 8x8x10 R8 bench at cell (i=10, j=9, k=1),
 * gz went 3.369e6 -> 3.303e7 -> 3.240e8 -> 3.177e9 -> 3.116e10 -> 3.055e11 over
 * steps 0..5, with zh(step N) bit-identical to gz(step N-1) and every ratio
 * exactly 9.8065 = grav. It stayed latent until a stencil reached far enough out
 * to read it: nord >= 2 makes the divergence-damping sweep run with nt /= 0,
 * whose two-ring reach touches the diagonal corner blocks, and that is exactly
 * the nord0-PASS / nord1-PASS / nord2-FAIL bisection observed.
 *
 * zh is deliberately NOT poisoned: its init_ijk_mem call is commented out at
 * :347, so the authoritative zh enters the first substep as allocate-garbage and
 * is defined by the :557-563 copy. Poisoning it here would be a change the
 * Fortran does not make.
 *
 * Numerical algorithm changed: NO. Every cell any stage legitimately reads is
 * overwritten before its first read -- gz by :444-454 plus the exchange, pk3 by
 * pk3_halo. What changes is only the value seen in cells the Fortran leaves
 * undefined, and it changes them FROM an accumulating product TO the Fortran's
 * own sentinel.
 *
 * SCOPE: the four diagonal corner ghost blocks only, not the whole allocation.
 * init_ijk_mem does cover the whole declared extent, but by the time any stage
 * reads gz the Fortran has already redefined the owned box (:444-454) and the
 * four edge halos (the :459 exchange), so huge_r SURVIVES only in the diagonal
 * corners -- which are exactly the cells the ×grav loop above amplifies. A flat
 * fill reproduces the Fortran's intermediate state but not its observable one,
 * and it destroys edge-halo values in any context where the caller rather than
 * an exchange owns them. That is not hypothetical: the flat version made
 * fv3_dispatch_smoke fail in BOTH R8 and R4 (whole Riem input columns NaN),
 * and passed again under FV3_GPU_NO_INIT_POISON=1, because that harness's
 * Phase A supplies gz's halo itself and nothing here restores it. Narrowing to
 * the corners keeps the fix and drops the collateral damage.
 *
 * Deliberately gated on init_step, not on every call: for k_split > 1 the
 * Fortran allocates once and lets gz carry across the k_split sub-calls, so
 * poisoning per call would destroy state the Fortran keeps. */
int poison_init_step_transients(ContextData &d, fv3_gpu_error *e,
                                const char *api) {
  auto stream = static_cast<cudaStream_t>(d.compute_stream);
  for (auto &x : d.state.fields) {
    const FieldId id = field_id_from_abi(x.host.field_id);
    if (id != FieldId::gz && id != FieldId::pk3) continue;
    if (x.host.rank != 3) continue;
    const int rc = fv3_fill_device_corner_ghosts(
        x.device, x.host.lower, x.host.extent, x.host.stride_elements,
        x.host.compute_lower[0], x.host.compute_upper[0],
        x.host.compute_lower[1], x.host.compute_upper[1], fv3_huge_r, stream);
    if (rc != 0) return cuda_fail(static_cast<cudaError_t>(rc), e, api, &d);
  }
  return FV3_GPU_SUCCESS;
}

/* TEMPORARY diagnostic for the Milestone 3A.11 bounded cross-step NaN pass.
 * Gated on FV3_GPU_DEBUG_FAIL and model_step>=1 (call #2 onward) so it is a
 * no-op in every normal build/run. To be reverted once the bounded pass
 * concludes (fix applied + re-verified, or PERFORMANCE BASELINE BLOCKED). */
void debug_check_finite_stage(ContextData &d, const char *stage, int64_t model_step) {
  const char *dbg = std::getenv("FV3_GPU_DEBUG_FAIL");
  if (!dbg || !*dbg) return;
  /* Milestone 3A.11V: cost filters. Without them this probe is unusable for the
   * dt matrix. It D2H-copies EVERY one of the 22 target fields at EVERY
   * checkpoint for EVERY tile, so an n_split=12 run of 300 calls would move
   * 22 fields x 17 stages x 12 substeps x 300 calls x 6 tiles x ~40 KB, on the
   * order of hundreds of GB of transfers to answer a question about three
   * fields on one tile. These knobs narrow the sampling; none of them changes a
   * single numerical result, only what is observed.
   *   FV3_GPU_DEBUG_TILE=N     sample only tile N
   *   FV3_GPU_DEBUG_FIELDS=a,b sample only these field names
   *   FV3_GPU_DEBUG_EVERY=N    sample only when model_step % N == 0 */
  if (const char *ts = std::getenv("FV3_GPU_DEBUG_TILE"))
    if (std::atoi(ts) != static_cast<int>(d.topology.tile)) return;
  if (const char *es = std::getenv("FV3_GPU_DEBUG_EVERY")) {
    const int every = std::atoi(es);
    if (every > 1 && (model_step % every) != 0) return;
  }
  const char *field_filter = std::getenv("FV3_GPU_DEBUG_FIELDS");
  /* BOTH streams. Syncing only the compute stream made the checkpoint itself
   * nondeterministic: a pending unpack on the transfer stream may or may not
   * have landed by the time the D2H copy runs, so the checksum would differ
   * run-to-run for reasons that have nothing to do with the defect under
   * investigation. Draining both makes each checkpoint a quiescent state.
   *
   * This doubles as Milestone 3A.11R §11's serialization probe: with both
   * streams drained at every stage boundary, no cross-stream ordering gap
   * inside a stage can survive. If corruption persists under it, a missing
   * stream/event edge is excluded and the defect is in kernel or stage logic. */
  cudaStreamSynchronize(static_cast<cudaStream_t>(d.compute_stream));
  cudaStreamSynchronize(static_cast<cudaStream_t>(d.transfer_stream));
  /* Milestone 3A.11T: delpc/ptc/omga/pt ADDED. The 3A.11S diagnostic covered
   * `delp` and reported it clean, which was read as "Riem_Solver_C's inputs are
   * clean". It is not: Riem_Solver_c (nh_utils.F90:325-432, called at
   * dyn_core.F90:583) consumes `delpc` and `ptc` -- the C-grid predictor's
   * outputs -- as its `delp` and `pt` dummies, NOT the D-grid `delp`/`pt`.
   * Neither delpc nor ptc was sampled, so the one hypothesis the log appeared
   * to exclude had in fact never been tested. omga is Riem's w3 dummy. */
  static const FieldId targets[] = {FieldId::u, FieldId::delp, FieldId::v, FieldId::w,
                                     FieldId::delz, FieldId::uc, FieldId::vc, FieldId::ut,
                                     FieldId::vt, FieldId::divgd, FieldId::crx, FieldId::cry,
                                     FieldId::xfx, FieldId::yfx, FieldId::gz, FieldId::zh,
                                     FieldId::pkc, FieldId::ws3,
                                     FieldId::delpc, FieldId::ptc, FieldId::omga, FieldId::pt,
  /* Prompt 6B SP1 ws3 reconciliation: update_dz_c's three remaining
   * non-dynamical inputs, added so device-side identity of the authoritative
   * --sp1-boundary load can be PROVEN by D2H readback rather than inferred
   * from the host-side print. phis/zs are static per run and dp_ref is a 1-D
   * reference profile, so sampling them costs one extra small copy per
   * checkpoint and only when FV3_GPU_DEBUG_FIELDS selects them. */
                                     FieldId::phis, FieldId::zs, FieldId::dp_ref,
  /* Prompt 6B Task H (SP2 = post-d_sw, dyn_core.F90:871-880): the four
   * remaining SP2 savepoint fields. Without them SP2 coverage would be 6/10
   * and the mass/tracer fluxes -- d_sw's own outputs, not derived state --
   * would go uncompared. Only copied when FV3_GPU_DEBUG_FIELDS selects them. */
                                     FieldId::mfx, FieldId::mfy,
                                     FieldId::cx, FieldId::cy,
  /* Prompt 6B Task J (savepoints A/B = post-Riem_Solver3 / post-nh_p_grad,
   * dyn_core.F90:972 / :1067): pk3 is Riem_Solver3's intent(out) pressure^kappa
   * on the full k+1 range and nh_p_grad's second corner-transformed input; ws
   * is update_dz_d's intent(out) and Riem_Solver3's intent(in), so it
   * discriminates the two vertical operators without a fourth savepoint.
   * Neither was sampled before, which is why the SP2 -> end-of-it=1 window had
   * no CUDA-side observable at all. */
                                     FieldId::pk3, FieldId::ws,
  /* Prompt 6B dyn_core-EXIT closure: the six sp3 fields that had no CUDA-side
   * observable at all. pe/peln/pk are Riem_Solver3's last_call writeback, pkz is
   * rewritten by the diffusive-heating tail, and ua/va are the A-grid winds the
   * exit savepoint carries. Without these, an sp3 comparison would silently be
   * 7 of 13 fields. Only copied when FV3_GPU_DEBUG_FIELDS selects them. */
                                     FieldId::pe, FieldId::peln, FieldId::pk,
                                     FieldId::pkz, FieldId::ua, FieldId::va};
  static const char *names[] = {"u", "delp", "v", "w", "delz", "uc", "vc", "ut", "vt",
                                 "divgd", "crx", "cry", "xfx", "yfx", "gz", "zh", "pkc", "ws3",
                                 "delpc", "ptc", "omga", "pt",
                                 "phis", "zs", "dp_ref",
                                 "mfx", "mfy", "cx", "cy",
                                 "pk3", "ws",
                                 "pe", "peln", "pk", "pkz", "ua", "va"};
  /* Whole-token comma-list match. Substring matching would be wrong here in a
   * way that silently widens the sample: "pt" is a substring of "ptc", "w" of
   * "ws3", "delp" of "delpc", "u" of "ut". */
  auto field_selected = [&](const char *nm) {
    if (!field_filter || !*field_filter) return true;
    const size_t n = std::strlen(nm);
    for (const char *p = field_filter; *p;) {
      const char *e = std::strchr(p, ',');
      const size_t len = e ? static_cast<size_t>(e - p) : std::strlen(p);
      if (len == n && std::strncmp(p, nm, n) == 0) return true;
      if (!e) break;
      p = e + 1;
    }
    return false;
  };
  for (int t = 0; t < static_cast<int>(sizeof(targets) / sizeof(targets[0])); ++t) {
    if (!field_selected(names[t])) continue;
    for (auto &x : d.state.fields) {
      if (field_id_from_abi(x.host.field_id) != targets[t]) continue;
      const size_t count = x.host.allocation_bytes / sizeof(fv_real);
      std::vector<fv_real> tmp(count);
      cudaMemcpy(tmp.data(), x.device, x.host.allocation_bytes, cudaMemcpyDeviceToHost);
      /* Prompt 6B: raw-value capture for SP1/SP2/SP3 CUDA-vs-Fortran
       * comparison. Off by default (FV3_GPU_DEBUG_DUMP_DIR unset) -- this is
       * purely additional D2H-already-happened disk output; no new D2H call,
       * no new synchronization, no change to the hash/extrema diagnostics
       * above or below. Reuses the exact flat layout dump_sp0_state()
       * (run_small_perf_bench.cpp) already writes, so
       * tools/savepoints/compare_sp0_identity.py's reader can consume this
       * directly without a third binary format. FV3_GPU_DEBUG_DUMP_STAGE is
       * an optional comma-list of stage-name SUBSTRINGS (not whole-token --
       * stage strings like "it=1 cgrid_pressure" have no fixed token layout
       * a whole-token matcher could rely on, and stage names don't collide
       * the way field names do, e.g. "cgrid_pressure" is not a substring of
       * "dgrid_dynamics" or vice versa). */
      if (const char *dump_dir = std::getenv("FV3_GPU_DEBUG_DUMP_DIR")) {
        if (*dump_dir) {
          bool stage_selected = true;
          if (const char *stage_filter = std::getenv("FV3_GPU_DEBUG_DUMP_STAGE")) {
            if (*stage_filter) {
              stage_selected = false;
              for (const char *p = stage_filter; *p;) {
                const char *e = std::strchr(p, ',');
                const size_t len = e ? static_cast<size_t>(e - p) : std::strlen(p);
                std::string needle(p, len);
                if (!needle.empty() && std::strstr(stage, needle.c_str())) { stage_selected = true; break; }
                if (!e) break;
                p = e + 1;
              }
            }
          }
          if (stage_selected) {
            char sclean[64];
            size_t si = 0;
            for (const char *p = stage; *p && si + 1 < sizeof(sclean); ++p)
              sclean[si++] = (*p == ' ' || *p == ',') ? '_' : *p;
            while (si > 0 && sclean[si - 1] == '_') --si;
            sclean[si] = '\0';
            const std::string tile_dir =
                std::string(dump_dir) + "/tile" + std::to_string(static_cast<int>(d.topology.tile) - 1);
            std::error_code ec;
            std::filesystem::create_directories(tile_dir, ec);
            std::vector<double> raw(tmp.begin(), tmp.end());
            const std::string path = tile_dir + "/" + sclean + "__" + names[t] + ".bin";
            std::ofstream f(path, std::ios::binary);
            if (f) {
              f.write(reinterpret_cast<const char *>(raw.data()),
                      static_cast<std::streamsize>(raw.size() * sizeof(double)));
            } else {
              std::fprintf(stderr, "[FV3DBG-DUMP] FAIL: cannot open %s for write\n", path.c_str());
            }
          }
        }
      }
      size_t bad_count = 0, first = count;
      /* Bitwise FNV-1a over the raw bytes. A hash rather than a sum because the
       * question this answers is "did two runs of identical code produce
       * identical bits at this point", and a floating-point sum can collide or
       * mask a reordered-but-equal-magnitude difference. NaN payload bits are
       * included deliberately: they carry which operation produced them. */
      uint64_t h = 1469598103934665603ull;
      const auto *raw = reinterpret_cast<const unsigned char *>(tmp.data());
      for (size_t b = 0; b < count * sizeof(fv_real); ++b) {
        h ^= raw[b];
        h *= 1099511628211ull;
      }
      /* Milestone 3A.11T: extrema over the FINITE subset, with locations.
       * Finiteness alone cannot answer the question this pass asks. SIM1_solver
       * (nh_utils.F90:1232) evaluates log(-dm2/dz2*rgas*pt2); update_dz_c's
       * dz_min clamp (nh_utils.F90:196-200) already guarantees dz2 <= -dz_min,
       * so the ONLY way that log can see a non-positive argument is delp <= 0
       * or pt <= 0. Both are perfectly finite. A finite-only probe is therefore
       * blind to the single remaining precondition, which is why min/argmin are
       * reported here rather than inferred. */
      double vmin = 0.0, vmax = 0.0;
      size_t imin = count, imax = count, finite_count = 0;
      for (size_t i = 0; i < count; ++i) {
        const double v = static_cast<double>(tmp[i]);
        if (std::isnan(v) || std::isinf(v)) {
          ++bad_count;
          if (first == count) first = i;
          continue;
        }
        if (finite_count == 0 || v < vmin) { vmin = v; imin = i; }
        if (finite_count == 0 || v > vmax) { vmax = v; imax = i; }
        ++finite_count;
      }
      /* Milestone 3A.11T: the SAME extrema restricted to the OWNED interior
       * (is..ie, js..je), reported alongside the whole-allocation pair.
       *
       * This is the measurement that separates the two surviving hypotheses
       * for delpc's drift into negative pressure thickness. The whole-box
       * extremum sat on the j=js-1 ring at the bottom level for three
       * consecutive steps, but a whole-box figure cannot distinguish
       *   (a) a genuine instability of the synthetic state, which would show
       *       up in the interior too, from
       *   (b) an edge/halo defect, where the owned interior stays healthy and
       *       only the outermost computed ring diverges.
       * Riem_Solver_c computes is-1..ie+1 / js-1..je+1 (nh_utils.F90:350-351,
       * 356), one ring WIDER than the owned box, so a ring-only defect is
       * consumed as if it were valid state and is invisible to any
       * interior-only check -- and equally invisible to a whole-box one. */
      double omin = 0.0, omax = 0.0;
      size_t oimin = count, oimax = count, own_count = 0, own_bad = 0;
      {
        const int64_t hw = static_cast<int64_t>(d.immutable.bounds.halo_width);
        const int64_t isd = d.immutable.bounds.is - hw;
        const int64_t jsd = d.immutable.bounds.js - hw;
        int64_t ni = (d.immutable.bounds.ie - d.immutable.bounds.is + 1) + 2 * hw;
        int64_t nj = (d.immutable.bounds.je - d.immutable.bounds.js + 1) + 2 * hw;
        if (ni > 0 && nj > 0 && count % static_cast<size_t>(ni * nj) != 0) { ++ni; ++nj; }
        if (ni > 0 && nj > 0 && count % static_cast<size_t>(ni * nj) == 0) {
          for (size_t idx = 0; idx < count; ++idx) {
            const int64_t ib = static_cast<int64_t>(idx % static_cast<size_t>(ni));
            const int64_t jb =
                static_cast<int64_t>((idx / static_cast<size_t>(ni)) % static_cast<size_t>(nj));
            const int64_t gi = isd + ib, gj = jsd + jb;
            if (gi < d.immutable.bounds.is || gi > d.immutable.bounds.ie ||
                gj < d.immutable.bounds.js || gj > d.immutable.bounds.je)
              continue;
            const double v = static_cast<double>(tmp[idx]);
            if (std::isnan(v) || std::isinf(v)) { ++own_bad; continue; }
            if (own_count == 0 || v < omin) { omin = v; oimin = idx; }
            if (own_count == 0 || v > omax) { omax = v; oimax = idx; }
            ++own_count;
          }
        }
      }
      /* Keyed by TILE, not by context address: a heap pointer changes every run,
       * which makes two runs' logs undiffable, and diffing them is the whole
       * point of the checksum. */
      std::fprintf(stderr,
                    "[FV3DBG tile=%d step=%lld] %-28s %-6s nan/inf=%zu/%zu first_idx=%zu hash=%016llx"
                    " min=%.9e@%zu max=%.9e@%zu ownnan=%zu ownmin=%.9e@%zu ownmax=%.9e@%zu\n",
                    static_cast<int>(d.topology.tile), static_cast<long long>(model_step), stage, names[t],
                    bad_count, count, first, static_cast<unsigned long long>(h),
                    vmin, imin, vmax, imax, own_bad, omin, oimin, omax, oimax);
      /* Milestone 3A.11V §3: the SAME numbers as the line above, but as CSV with
       * every coordinate ALREADY DECODED, plus the value at up to three named
       * probe cells. Gated on FV3_GPU_DEBUG_CSV, independent of the text form.
       *
       * This exists because 3A.11U produced a wrong intermediate conclusion by
       * parsing the text line above with fixed awk column numbers. The real
       * positions are $10=min= and $11=max=, not $11/$12, and post-loop stage
       * strings carry no `it=` token so every column shifts by one. The probe
       * printed "u NONZERO ... magnitude 0.000000e+00" -- self-contradictory --
       * and a wind table of all zeros, which would have "confirmed" the very
       * premise that was false. Emitting decoded CSV removes that entire failure
       * mode: no consumer has to know a column number or a token order.
       *
       * The probe cells answer §3 directly: the growing mode is at one specific
       * cell, and a whole-box extremum cannot show whether THAT cell is growing
       * or whether the extremum merely migrated between cells tied at the same
       * value -- which is exactly what made delp's argmin move at 3A.11U step 58
       * while every printed value stayed bit-identical. */
      if (std::getenv("FV3_GPU_DEBUG_CSV")) {
        const int64_t hw = static_cast<int64_t>(d.immutable.bounds.halo_width);
        const int64_t isd = d.immutable.bounds.is - hw;
        const int64_t jsd = d.immutable.bounds.js - hw;
        int64_t ni = (d.immutable.bounds.ie - d.immutable.bounds.is + 1) + 2 * hw;
        int64_t nj = (d.immutable.bounds.je - d.immutable.bounds.js + 1) + 2 * hw;
        if (ni > 0 && nj > 0 && count % static_cast<size_t>(ni * nj) != 0) { ++ni; ++nj; }
        const bool shaped = (ni > 0 && nj > 0 && count % static_cast<size_t>(ni * nj) == 0);
        const int64_t nk = shaped ? static_cast<int64_t>(count) / (ni * nj) : 0;
        /* Decode a flat index to (i,j,k) in GLOBAL tile coordinates. */
        auto dec_i = [&](size_t f) { return shaped ? isd + static_cast<int64_t>(f) % ni : -9999; };
        auto dec_j = [&](size_t f) { return shaped ? jsd + (static_cast<int64_t>(f) / ni) % nj : -9999; };
        auto dec_k = [&](size_t f) { return shaped ? static_cast<int64_t>(f) / (ni * nj) : -9999; };
        /* Probe cells as "i:j:k,i:j:k,..." -- the mode cell plus controls. */
        double pv[3] = {0.0, 0.0, 0.0};
        int pn = 0;
        if (const char *ps = std::getenv("FV3_GPU_DEBUG_PROBE")) {
          int pi = 0, pj = 0, pk = 0;
          const char *c = ps;
          while (pn < 3 && *c) {
            if (std::sscanf(c, "%d:%d:%d", &pi, &pj, &pk) == 3) {
              double v = std::nan("");
              if (shaped) {
                const int64_t bi = pi - isd, bj = pj - jsd;
                if (bi >= 0 && bi < ni && bj >= 0 && bj < nj && pk >= 0 && pk < nk) {
                  const size_t f = (static_cast<size_t>(pk) * static_cast<size_t>(nj) +
                                    static_cast<size_t>(bj)) * static_cast<size_t>(ni) +
                                   static_cast<size_t>(bi);
                  if (f < count) v = static_cast<double>(tmp[f]);
                }
              }
              pv[pn++] = v;
            }
            const char *comma = std::strchr(c, ',');
            if (!comma) break;
            c = comma + 1;
          }
        }
        /* `stage` carries embedded spaces ("it=1 dgrid_dynamics"), which would
         * break the field count. Collapse them so the row stays one CSV field. */
        char sclean[64];
        {
          size_t si = 0;
          for (const char *p = stage; *p && si + 1 < sizeof(sclean); ++p)
            sclean[si++] = (*p == ' ' || *p == ',') ? '_' : *p;
          while (si > 0 && sclean[si - 1] == '_') --si;   /* trim %-28s padding */
          sclean[si] = '\0';
        }
        std::fprintf(stderr,
                      "CSVPROBE,%d,%lld,%s,%s,%lld,%lld,%lld,%zu,%zu,%zu,"
                      "%.17e,%lld,%lld,%lld,%.17e,%lld,%lld,%lld,"
                      "%.17e,%lld,%lld,%lld,%.17e,%lld,%lld,%lld,"
                      "%.17e,%.17e,%.17e\n",
                      static_cast<int>(d.topology.tile), static_cast<long long>(model_step),
                      sclean, names[t],
                      static_cast<long long>(ni), static_cast<long long>(nj),
                      static_cast<long long>(nk), count, bad_count, own_bad,
                      vmin, (long long)dec_i(imin), (long long)dec_j(imin), (long long)dec_k(imin),
                      vmax, (long long)dec_i(imax), (long long)dec_j(imax), (long long)dec_k(imax),
                      omin, (long long)dec_i(oimin), (long long)dec_j(oimin), (long long)dec_k(oimin),
                      omax, (long long)dec_i(oimax), (long long)dec_j(oimax), (long long)dec_k(oimax),
                      pv[0], pv[1], pv[2]);
      }
      /* Decode the bad cells' (i,j,k). Which CELLS go bad first separates the
       * candidate mechanisms: an interior plane implicates the stage's own
       * arithmetic, an edge strip implicates the halo exchange, and a 3-wide
       * diagonal block implicates the corner halo, which a four-edge exchange
       * never fills. */
      if (bad_count > 0) {
        /* DomainBounds carries only is/ie/js/je/halo_width, so the allocated
         * halo extent is inferred: nx+2*hw for a centred field, one more for a
         * staggered one. Pick whichever divides the element count evenly, which
         * is unambiguous here because the two candidates differ by a full
         * i-row. */
        const int64_t hw = static_cast<int64_t>(d.immutable.bounds.halo_width);
        const int64_t isd = d.immutable.bounds.is - hw;
        const int64_t jsd = d.immutable.bounds.js - hw;
        int64_t ni = (d.immutable.bounds.ie - d.immutable.bounds.is + 1) + 2 * hw;
        int64_t nj = (d.immutable.bounds.je - d.immutable.bounds.js + 1) + 2 * hw;
        if (ni > 0 && nj > 0 && count % static_cast<size_t>(ni * nj) != 0) { ++ni; ++nj; }
        if (ni > 0 && nj > 0 && count % static_cast<size_t>(ni * nj) == 0) {
          std::fprintf(stderr, "    [FV3DBG-IDX tile=%d step=%lld] %s %s ni=%lld nj=%lld isd=%lld is=%lld ie=%lld :",
                       static_cast<int>(d.topology.tile), static_cast<long long>(model_step),
                       stage, names[t], static_cast<long long>(ni), static_cast<long long>(nj),
                       static_cast<long long>(isd), static_cast<long long>(d.immutable.bounds.is),
                       static_cast<long long>(d.immutable.bounds.ie));
          size_t shown = 0;
          for (size_t i = 0; i < count && shown < 24; ++i) {
            if (!(std::isnan(static_cast<double>(tmp[i])) || std::isinf(static_cast<double>(tmp[i])))) continue;
            const int64_t li = static_cast<int64_t>(i);
            std::fprintf(stderr, " (%lld,%lld,%lld)",
                         static_cast<long long>(isd + li % ni),
                         static_cast<long long>(jsd + (li / ni) % nj),
                         static_cast<long long>(li / (ni * nj)));
            ++shown;
          }
          std::fprintf(stderr, "%s\n", bad_count > shown ? " ..." : "");
        }
      }
      break;
    }
  }
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
    case FV3_GPU_H_DWIND_SEAM: return "H_DWIND_SEAM (E11 u,v shared seam line, mpp_get_boundary :1178)";
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

  const bool fuse_halo_edges = d.immutable.config.fuse_halo_edges != 0;
  rc = halo::launch_halo_pack(eff, geom, fields, d.halo_send, d.transfer_stream, e,
                              fuse_halo_edges);
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
                                d.transfer_stream, e, fuse_halo_edges);
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
  for (uint32_t g = 1; g <= static_cast<uint32_t>(FV3_GPU_H_GROUP_LAST); ++g) {
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
     * neighbouring scratch.
     *
     * WorkspaceId::vertical is NOT sized by that same (ni x nj x npz+1) x N
     * convention: its two consumers (launch_vertical_solver's Riem_Solver3
     * path and RiemSolverCArgs's Riem_Solver_c path) each pack a per-column
     * arena whose per-column scalar count is set by
     * vertical::compute_workspace_layout()/riem_solver_c_scalars_per_column()
     * -- 422 and 209 scalars respectively at km=10, not a flat multiple of
     * (npz+1). It is sized here as cells x vertical_workspace_per_column(npz),
     * using the exact Riem_Solver3 layout (its 422 > Riem_Solver_c's 209,
     * so it also covers the narrower Riem_Solver_c consumer) over the full
     * halo-inclusive cell count (>= either consumer's own column count). A
     * prior flat "x16" placeholder undersized this for any grid large enough
     * that halo padding didn't accidentally compensate (e.g. milestone
     * 3A.11's 16x16 case: it allocated ~681KB against Riem_Solver3's actual
     * ~864KB need), so launch_vertical_solver's workspace-capacity guard
     * (fv3_vertical_kernels.cu) rejected every call with
     * FV3_GPU_UNSUPPORTED_CONFIGURATION. */
    const uint64_t vertical_bytes_per_column=vertical::vertical_workspace_per_column<fv_real>((int)c->data.create.npz);
    if(mul_overflow(values,sizeof(fv_real),value_bytes)||mul_overflow(value_bytes,12,cgrid_bytes)||mul_overflow(value_bytes,34,transport_bytes)||mul_overflow(cells,vertical_bytes_per_column,vertical_bytes)||mul_overflow(value_bytes,12,operator_bytes)){fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"workspace byte-size overflow");goto rollback;}
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
/* Prompt 8 (CUDA Graphs prototype). Opt-in switch, off by default; see
 * ContextData::cuda_graph_enabled and fv3_gpu_api.h. Rejects being toggled
 * mid-lifetime once a graph has already been captured, since a context has no
 * mechanism to invalidate/re-capture -- callers set this once, right after
 * fv3_gpu_create, matching the documented "before the first
 * fv3_gpu_dyn_core_step" contract. */
extern "C" int fv3_gpu_set_cuda_graph_mode(fv3_gpu_context *c,int enabled,fv3_gpu_error *e){
  using namespace fv3::gpu; clear_error(e);
  if(!c)return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,"null context");
  if(c->data.phase==ContextPhase::failed)return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"context has failed");
  if(c->data.dgrid_dynamics_graph_captured)
    return fail(e,FV3_GPU_STATE_CONFLICT,__func__,"cannot change graph mode after a graph has been captured");
  c->data.cuda_graph_enabled=(enabled!=0);
  return 0;
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
  /* dyn_core.F90:328-350, inside `if (init_step)`, BEFORE the acoustic loop and
   * before the :444-454 height reconstruction that defines gz's owned box. See
   * poison_init_step_transients for why a persistent gz makes this load-bearing
   * rather than defensive. */
  /* TEMPORARY escape hatch, Milestone 3A.11T. Lets one job measure both arms so
   * a regression can be ATTRIBUTED to the poison rather than guessed at. Read
   * once and cached: getenv per step would be a per-call syscall on the hot path.
   * Classify KEEP/REMOVE under section 11 once stability is proven. */
  static const bool poison_disabled = [] {
    const char *s = std::getenv("FV3_GPU_NO_INIT_POISON");
    return s != nullptr && s[0] != '\0' && s[0] != '0';
  }();
  if(p.init_step && !poison_disabled){
    rc=poison_init_step_transients(c->data,e,__func__);
    if(rc){ c->data.phase=ContextPhase::failed; return rc; }
  }
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
   *   :459/:552    H_GZ_FIRST   unknown              resolve_halo, it==1 only,     cgrid_vertical_pre_halo   cgrid_vertical_solver (fv3_vertical_kernels.cu, acoustic_index==0)
   *                                                  AFTER cgrid_vertical_pre_halo (gz_from_delz_kernel)
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
   * H_GZ_FIRST's require() site is inside launch_cgrid_vertical_solver, between
   * the gz reconstruction and the gz<->zh sync, matching Fortran's :459 -> :552
   * split across the C-grid vertical group.
   *
   * CORRECTED IN MILESTONE 3A.11S. This comment previously claimed that
   * resolving the group BEFORE the stage was harmless because "on the only
   * topology resolve_halo will accept, the exchange copies nothing, so resolving
   * before rather than inside the stage skips no operation."
   *
   * That claim was FALSE. On the real six-tile grid_type == 0 cubed sphere all
   * four edge neighbours are populated and the exchange performs real
   * device-to-device data movement. Resolving before the stage therefore packed
   * gz's interior BEFORE gz_from_delz_kernel rebuilt it, so every tile's gz halo
   * carried the previous top-level step's values. :557-563 then broadcast that
   * stale halo into zh across the full halo box, where it persisted for every
   * later substep, and update_dz_c differenced gz2 across the resulting
   * interior/halo discontinuity until its denominator crossed zero. The failure
   * was deterministic, first appearing at (ie, js) and (ie, js+1) -- the
   * southeast corner, where the east and south discontinuities compound.
   *
   * The fix is the two-slot split: cgrid_vertical_pre_halo rebuilds the interior,
   * then the group is resolved, then cgrid_vertical_solver consumes it. See
   * docs/agent_handoff/milestone_3a11s_h_gz_first_ordering.md.
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

  /* Prompt 6B -- the per-substep heat_source accumulation of
   * dyn_core.F90:853-861, which this port was missing.
   *
   *     if ( flagstruct%d_con > 1.0E-5 .OR. flagstruct%do_skeb ) then
   *        do j=js,je ; do i=is,ie
   *           heat_source(i,j,k) = heat_source(i,j,k) + heat_s(i,j)
   *
   * Fortran keeps two arrays: a dyn_core-lifetime heat_source zeroed once at
   * :377-379, and d_sw's per-level intent(OUT) heat_s. This port has only the
   * registered 3-D heat_source, which d_sw re-zeroes every substep
   * (dsw_heat_reset_kernel), so at loop exit it held the LAST substep instead of
   * the sum. `heat_accum` restores the missing second array; it is persistent
   * device state so no allocation happens inside the loop.
   *
   * `do_skeb` is outside the supported production subset, so the gate is d_con
   * alone -- the same predicate zero_accumulators() already uses for the
   * once-per-call zeroing of heat_source itself.
   *
   * Numerical algorithm changed: NO -- a summation the authoritative code
   * performs is now performed; no kernel, stencil or expression is altered. */
  bool heat_accumulate = false;
  int64_t hs_i0=0, hs_i1=0, hs_j0=0, hs_j1=0, hs_k0=0, hs_k1=0;
  int64_t hs_lo_i=0, hs_lo_j=0, hs_lo_k=0, hs_st_i=0, hs_st_j=0, hs_st_k=0;
  fv_real *hs_data = nullptr;
#ifdef FV3_GPU_ENABLE_CUDA
  if(config.d_con>1.0e-5){
    const DeviceViewBase *hv=c->data.state.registry.find(FieldId::heat_source);
    uint64_t hs_bytes=0;
    for(auto &fx:c->data.state.fields)
      if(field_id_from_abi(fx.host.field_id)==FieldId::heat_source)
        hs_bytes=fx.host.allocation_bytes;
    /* The accumulation box is the AUTHORITATIVE loop nest -- dyn_core.F90:855-857
     * is `do j=js,je / do i=is,ie` -- taken from the domain bounds, not from
     * heat_source's own compute_lower/compute_upper. On this path the caller
     * registers heat_source with its compute box set to the full halo box
     * (-2..12 at c8), so using it would accumulate the halo as well. That is not
     * what Fortran does, and the halo is refreshed by FV3_GPU_H_HEAT before
     * del2_cubed reads it in any case. */
    const auto &hb=c->data.immutable.bounds;
    /* heat_source unregistered, or registered without a usable rank-3 view, is
     * the "d_con active but the caller does not use the diffusive-heating tail"
     * case. post_loop_heating is skipped for the same reason further down, so
     * this is silently inactive rather than an error. */
    if(hv && hv->data && hv->rank>=3 && hs_bytes>0 && hb.ie>=hb.is && hb.je>=hb.js
       && hb.npz>0 && static_cast<int64_t>(hv->extent[2])>=hb.npz){
      hs_data=static_cast<fv_real*>(hv->data);
      hs_lo_i=hv->lower[0];  hs_lo_j=hv->lower[1];  hs_lo_k=hv->lower[2];
      hs_st_i=hv->stride[0]; hs_st_j=hv->stride[1]; hs_st_k=hv->stride[2];
      hs_i0=hb.is; hs_i1=hb.ie;
      hs_j0=hb.js; hs_j1=hb.je;
      /* k is stated in the VIEW'S index space. The registered heat_source view
       * has lower[2] == 0 here while Fortran's loop is `do k=1,npz`, so the npz
       * levels are view indices lo_k .. lo_k+npz-1. Reconstructing this from npz
       * alone is exactly the mistake that addresses one level past the
       * allocation. */
      hs_k0=hs_lo_k; hs_k1=hs_lo_k+hb.npz-1;
      if(c->data.heat_accum==nullptr || c->data.heat_accum_bytes<hs_bytes){
        if(c->data.heat_accum) cudaFree(c->data.heat_accum);
        c->data.heat_accum=nullptr; c->data.heat_accum_bytes=0;
        void *acc=nullptr;
        const cudaError_t arc=cudaMalloc(&acc,static_cast<size_t>(hs_bytes));
        if(arc!=cudaSuccess) return cuda_fail(arc,e,__func__,&c->data);
        c->data.heat_accum=static_cast<fv_real*>(acc);
        c->data.heat_accum_bytes=hs_bytes;
      }
      /* Zeroed once per dyn_core call, mirroring the Fortran's allocate-and-zero
       * at :377-379 rather than the per-substep reset d_sw applies to its own
       * scratch. Whole-allocation, so the halo starts defined too. */
      const cudaError_t zrc=cudaMemsetAsync(c->data.heat_accum,0,
                                            static_cast<size_t>(hs_bytes),
                                            static_cast<cudaStream_t>(c->data.compute_stream));
      if(zrc!=cudaSuccess) return cuda_fail(zrc,e,__func__,&c->data);
      /* Refuse rather than corrupt. The kernel addresses dst/src with
       * heat_source's own lowers and strides, so an accumulation box that can
       * reach past the allocation writes into whichever field cudaMalloc placed
       * next -- a silent, per-tile-random corruption of an unrelated field, which
       * is the hardest class of defect to attribute. This guard is what caught
       * the original off-by-one-level form of this code. */
      const int64_t hs_elems=static_cast<int64_t>(hs_bytes/sizeof(fv_real));
      const int64_t hs_max_off=(hs_i1-hs_lo_i)*hs_st_i+(hs_j1-hs_lo_j)*hs_st_j
                              +(hs_k1-hs_lo_k)*hs_st_k;
      const int64_t hs_min_off=(hs_i0-hs_lo_i)*hs_st_i+(hs_j0-hs_lo_j)*hs_st_j
                              +(hs_k0-hs_lo_k)*hs_st_k;
      if(hs_min_off<0 || hs_max_off>=hs_elems)
        return fail(e,FV3_GPU_INVALID_ARGUMENT,__func__,
                    "the heat_source accumulation box does not fit inside "
                    "heat_source's registered allocation; the per-substep "
                    "accumulation is refused rather than writing outside it");
      heat_accumulate=true;
    }
  }
#endif

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

#ifdef FV3_GPU_ENABLE_CUDA
    char dbg_lbl[80];
#define FV3_GPU_DBG_STAGE(name) \
    (std::snprintf(dbg_lbl, sizeof(dbg_lbl), "it=%d " name, it), \
     debug_check_finite_stage(c->data, dbg_lbl, p.model_step))
#else
#define FV3_GPU_DBG_STAGE(name) do {} while (0)
#endif

    /* Milestone 3A.11V §5: a checkpoint BEFORE the first halo resolve. Every
     * other checkpoint fires after its stage, so on step 0 / it=1 the earliest
     * observable state was `halo_in`, i.e. already post-exchange. §5 asks
     * specifically whether the ~4.7e-10 bottom-level wind exists "immediately
     * after initialization", which no existing checkpoint could answer: a
     * nonzero value at halo_in is equally consistent with the IC carrying it and
     * with the exchange creating it. This separates those two. */
    FV3_GPU_DBG_STAGE("entry");
    if(!resolve(FV3_GPU_H_DWIND_IN)) return rc;               // :402 -> :504
    if(!resolve(FV3_GPU_H_W_IN)) return rc;                   // :427 -> :507
    /* Prompt 6B "E5": the pre-c_sw scalar halo. dyn_core.F90:494-497 completes
     * i_pack(1) inside `if (it==1)`, before c_sw at :535; the matching START is
     * fv_dynamics.F90:620-627, i.e. OUTSIDE dyn_core, over {q_con, cappa, delp,
     * pt}. For it >= 2 no pre-c_sw completion exists because the previous
     * substep's :892-895/:926 start/complete pair already refreshed the same
     * group -- which is exactly what the existing :844/:878 resolve below
     * mirrors. So the only missing edge is the it==1 one, and this is it.
     *
     * FV3_GPU_H_DSCALAR is reused rather than a new group added because its
     * descriptor (fv3_gpu_halo_exchange.hpp:220-228) is already field-for-field
     * i_pack(1): q_con, delp, pt, cappa, all CENTER scalars at
     * fv3_domain_halo_width. Same fields, same staggering, same width, same
     * seams -- so this is a missing resolve of a correct group, not a new or
     * repurposed one.
     *
     * Why it was silent until now: every prior harness IC was uniform per tile,
     * so an unexchanged delp/pt halo held the same values a correct exchange
     * would have delivered. Run from real authoritative Fortran SP0 state the
     * halo instead carries FV3's 1e-15 pre-exchange sentinel, c_sw consumes it,
     * and delpc/ptc emerge wrong over the whole is-1..ie+1 / js-1..je+1 ring;
     * on the subset of ring columns where that reaches Riem_Solver_c's log/
     * divide arguments it becomes NaN in gz and pkc, which p_grad_c then reads
     * into uc/vc. Numerical algorithm changed: NO -- no kernel, stencil or
     * expression is touched; a halo the Fortran exchanges is now exchanged. */
    if(it==1 && !resolve(FV3_GPU_H_DSCALAR)) return rc;        // :496 -> :535
    FV3_GPU_DBG_STAGE("halo_in");
    /* Prompt 6B -- pem snapshot, dyn_core.F90:503-517.
     *
     *     if ( it==n_split .and. end_step ) then
     *        if ( flagstruct%use_old_omega ) then
     *           allocate ( pem(is-1:ie+1,npz+1,js-1:je+1) )
     *           pem(i,1,j) = ptop
     *           pem(i,k+1,j) = pem(i,k,j) + delp(i,j,k)
     *
     * The snapshot is INSIDE the acoustic loop, at the top of the final substep,
     * and it is taken from that moment's delp -- not from the dyn_core entry
     * state. Its position relative to the H_DSCALAR completion above matches the
     * Fortran: :503 sits after complete_group_halo_update(i_pack(1)) at :496.
     * The i/j range is one ring wide, which the previous substep's post-dynamics
     * H_DSCALAR resolve (:844 -> :878) leaves current.
     *
     * pem then has to survive the whole substep until the finalize at :1205, so
     * it lives in the WP-D operator arena (slot 10) rather than on the host --
     * no transfer is added and device residency is preserved. */
    if(it==n_split && p.end_step && !run(dispatch.omega_pem_snapshot)) return rc;
    if(!run(dispatch.cgrid_predictor)) return rc;             // :515-523
    FV3_GPU_DBG_STAGE("cgrid_predictor");
    /* :525-531 is gated on `nord > 0`; when nord == 0 the Fortran neither packs
     * nor completes the divgd corner update, and cgrid_predictor correspondingly
     * leaves H_DIVGD alone (fv3_cgrid_predictor.cu:831). The WP-B binder
     * require()s H_DIVGD unconditionally, so an nord == 0 config fails loudly at
     * dgrid_dynamics rather than running with an unproduced halo. That is the
     * correct outcome: with nord == 0 divgd has no producer at all. */
    if(config.nord>0 && !resolve(FV3_GPU_H_DIVGD)) return rc; // :528 -> :636
    /* Milestone 3A.11S. The gz interior must be rebuilt from delz BEFORE the
     * exchange packs it (:436-455 precedes :459), so the pre-halo half runs
     * first, then the exchange, then the consumer. Resolving H_GZ_FIRST ahead of
     * the reconstruction -- as this loop did through 3A.11R -- exchanged the
     * PREVIOUS step's gz interior and left every tile's gz halo one step stale. */
    if(!run(dispatch.cgrid_vertical_pre_halo)) return rc;     // :436-455
    if(it==1 && !resolve(FV3_GPU_H_GZ_FIRST)) return rc;      // :459 -> :552
    if(!run(dispatch.cgrid_vertical_solver)) return rc;       // :552-583
    FV3_GPU_DBG_STAGE("cgrid_vertical_solver");
    if(!run(dispatch.cgrid_pressure)) return rc;              // :614
    FV3_GPU_DBG_STAGE("cgrid_pressure");
    if(!resolve(FV3_GPU_H_CWIND)) return rc;                  // :618 -> :640
#ifdef FV3_GPU_ENABLE_CUDA
    if(c->data.cuda_graph_enabled){
      /* Prompt 8. This one stage is bounded on both sides by a cross-thread
       * halo resolve (H_CWIND above, H_DSCALAR below) and runs entirely on
       * compute_stream, so thread-local capture is safe here even though
       * capturing the whole substep is not (see the CUDA Graphs handoff for
       * why). Captured once; every later occurrence -- every subsequent it in
       * this call, and every later fv3_gpu_dyn_core_step call -- replays. No
       * per-call parameter update is needed: dgrid_dynamics's entire
       * kernel-launch argument set (bounds/options/level-tables/geometry
       * built from immutable config+topology, and the persistent-residency
       * device pointers) is invariant across occurrences for a fixed
       * context. */
      if(!c->data.dgrid_dynamics_graph_captured){
        /* Prompt 8 diagnostic only, not a numerical change: report the
         * one-time capture+instantiate cost so it can be excluded from the
         * steady-state per-call timing the benchmark reports. */
        const auto graph_t0=std::chrono::steady_clock::now();
        cudaStream_t cs=(cudaStream_t)c->data.compute_stream;
        cudaError_t z=cudaStreamBeginCapture(cs,cudaStreamCaptureModeThreadLocal);
        if(z!=cudaSuccess) return cuda_fail(z,e,__func__,&c->data);
        bool ok=run(dispatch.dgrid_dynamics);
        cudaGraph_t graph=nullptr;
        cudaError_t ez=cudaStreamEndCapture(cs,&graph);
        if(!ok){ if(graph) cudaGraphDestroy(graph); return rc; }
        if(ez!=cudaSuccess) return cuda_fail(ez,e,__func__,&c->data);
        cudaGraphExec_t exec=nullptr;
        cudaError_t iz=cudaGraphInstantiate(&exec,graph,0);
        cudaGraphDestroy(graph);
        if(iz!=cudaSuccess) return cuda_fail(iz,e,__func__,&c->data);
        c->data.dgrid_dynamics_graph_exec=exec;
        c->data.dgrid_dynamics_graph_captured=true;
        const auto graph_t1=std::chrono::steady_clock::now();
        const double graph_setup_ms=
            std::chrono::duration<double,std::milli>(graph_t1-graph_t0).count();
        std::fprintf(stderr,"[FV3_GPU_GRAPH] capture+instantiate_ms=%.3f pid=%d device=%d\n",
                     graph_setup_ms,(int)getpid(),(int)c->data.create.device_ordinal);
      } else {
        cudaError_t z=cudaGraphLaunch((cudaGraphExec_t)c->data.dgrid_dynamics_graph_exec,
                                       (cudaStream_t)c->data.compute_stream);
        if(z!=cudaSuccess) return cuda_fail(z,e,__func__,&c->data);
      }
    } else
#endif
    if(!run(dispatch.dgrid_dynamics)) return rc;              // :700-829
    FV3_GPU_DBG_STAGE("dgrid_dynamics");
#ifdef FV3_GPU_ENABLE_CUDA
    /* :853-861, immediately after the d_sw k-loop closes at :862 and therefore
     * before anything else in the substep -- in particular before the post-loop
     * resolve(FV3_GPU_H_HEAT), which is what refreshes the halo del2_cubed
     * needs. At it < n_split the substep's contribution is banked; at
     * it == n_split the bank is folded back into heat_source, which is the array
     * post_loop_heating reads. The result is the same sum the Fortran holds at
     * :1311, reached in n_split adds rather than n_split adds into a separate
     * array, because the port's heat_source doubles as d_sw's scratch. */
    if(heat_accumulate){
      const int hrc = (it<n_split)
        ? fv3_heat_source_accumulate(c->data.heat_accum, hs_data,
                                     hs_lo_i,hs_lo_j,hs_lo_k, hs_st_i,hs_st_j,hs_st_k,
                                     hs_i0,hs_i1, hs_j0,hs_j1, hs_k0,hs_k1,
                                     c->data.compute_stream)
        : fv3_heat_source_accumulate(hs_data, c->data.heat_accum,
                                     hs_lo_i,hs_lo_j,hs_lo_k, hs_st_i,hs_st_j,hs_st_k,
                                     hs_i0,hs_i1, hs_j0,hs_j1, hs_k0,hs_k1,
                                     c->data.compute_stream);
      if(hrc!=static_cast<int>(cudaSuccess))
        return cuda_fail(static_cast<cudaError_t>(hrc),e,__func__,&c->data);
    }
#endif
    if(!resolve(FV3_GPU_H_DSCALAR)) return rc;                // :844 -> :878
    if(!run(dispatch.dgrid_height)) return rc;                // :905-906
    FV3_GPU_DBG_STAGE("dgrid_height");
    if(!run(dispatch.vertical_solver)) return rc;             // :918-923
    FV3_GPU_DBG_STAGE("vertical_solver");
    if(!resolve(FV3_GPU_H_ZH)) return rc;                     // :931 -> :967
    if(!resolve(FV3_GPU_H_PKC2)) return rc;                   // :931 -> :967
    if(!run(dispatch.nh_pressure)) return rc;                 // :940-1019
    FV3_GPU_DBG_STAGE("nh_pressure");

    /* E11 -- dyn_core.F90:1176-1189, the last statement of the acoustic substep
     * that touches u or v:
     *
     *   if( it==n_split .and. gridstruct%grid_type<4 .and. .not.gridstruct%nested)
     *      call mpp_get_boundary(u, v, domain, ebuffery=ebuffer,
     *                            nbufferx=nbuffer, gridtype=DGRID_NE)
     *      u(is:ie, je+1, k) = nbuffer(...);  v(ie+1, js:je, k) = ebuffer(...)
     *
     * Not a halo exchange: u's je+1 line and v's ie+1 line are OWNED cells that
     * the neighbour tile computes redundantly, and this replaces my copy with the
     * neighbour's so the two cannot drift apart across substeps. Implemented as
     * the FV3_GPU_H_DWIND_SEAM group, which is the existing seam algebra
     * evaluated at ring h = 0 with an unpack restricted to one edge per field;
     * see halo_group_spec() for the map and for the evidence that it reproduces
     * the authoritative Fortran result bit for bit.
     *
     * Placement: after nh_pressure, which is the last writer of u/v in the
     * substep (:1637-1660 via nh_p_grad), and inside the loop so the guard is
     * the authoritative `it==n_split` rather than a post-loop assumption. All
     * three clauses of the Fortran guard are reproduced; grid_type and nested are
     * fenced elsewhere to 0 and false for this configuration, so the gate reduces
     * to it==n_split here, but the full condition is written out because a nested
     * or grid_type>=4 configuration must NOT perform this operation.
     *
     * Numerical algorithm changed: NO -- an authoritative operation that was
     * absent is now performed; no kernel, stencil or expression is altered. */
    if(it==n_split && config.grid_type<4 && !c->data.topology.nested){
      if(!resolve(FV3_GPU_H_DWIND_SEAM)) return rc;           // :1178 -> :1188
      FV3_GPU_DBG_STAGE("seam_reconcile");
    }

    /* Prompt 6B -- omega finalization, dyn_core.F90:1204-1218.
     *
     *     if ( last_step ) then
     *       if ( flagstruct%use_old_omega ) then
     *         omga(i,j,k) = (pe(i,k+1,j) - pem(i,k+1,j)) * rdt
     *         call adv_pe(ua, va, pem, omga, gridstruct, bd, npx, npy, npz, ng)
     *
     * Placement is authoritative: strictly AFTER the E11 seam block above (:1176-
     * :1188) and still inside the acoustic loop. `last_step` is the same
     * it==n_split .and. end_step predicate the :503 snapshot uses (:518 sets it).
     * Nothing between here and the dyn_core exit writes pe, ua or va, so this is
     * the final value of omga.
     *
     * The om2d cumulative-sum form at :1219-1237 is the `.not. use_old_omega`
     * branch and is DEAD for this configuration (use_old_omega defaults .true.,
     * fv_arrays.F90:465, and is not overridden). It is deliberately not
     * implemented, and there is no NumericalConfig knob that could select it --
     * see omega_preconditions() in wp_d_contract.cpp.
     *
     * Numerical algorithm changed: NO -- an authoritative stage that was entirely
     * absent is now performed. Before this, omga's last CUDA writer was
     * Riem_Solver_c using it as the `w3` scratch argument (faithful as an
     * intermediate, since Fortran aliases the same array at :518, but not the
     * exit value). */
    if(it==n_split && p.end_step){
      if(!run(dispatch.omega_finalize)) return rc;             // :1205 -> :1218
      FV3_GPU_DBG_STAGE("omega_finalize");
    }
  }
#ifdef FV3_GPU_ENABLE_CUDA
#undef FV3_GPU_DBG_STAGE
#endif

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
#ifdef FV3_GPU_ENABLE_CUDA
    debug_check_finite_stage(c->data, "post_loop_heating", p.model_step);
#endif
  }

#ifdef FV3_GPU_ENABLE_CUDA
  /* Prompt 6B: the dyn_core EXIT checkpoint, the CUDA comparand for the
   * authoritative sp3 savepoint (dyn_core.F90, after the acoustic loop and after
   * the heat_source application). Deliberately OUTSIDE the post_loop_heating
   * gate: when that gate is closed the exit state is still a valid comparison
   * target, and a checkpoint that disappears with a configuration knob is worse
   * than useless. Labelled with n_split so the dump filename carries the same
   * `it` the Fortran sp3 block records. */
  {
    char exit_lbl[80];
    std::snprintf(exit_lbl, sizeof(exit_lbl), "it=%d dyn_core_exit",
                  static_cast<int>(n_split));
    debug_check_finite_stage(c->data, exit_lbl, p.model_step);
  }
#endif

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
  cudaSetDevice(c->data.create.device_ordinal);cudaStreamSynchronize((cudaStream_t)c->data.compute_stream);cudaStreamSynchronize((cudaStream_t)c->data.transfer_stream);for(auto &x:c->data.state.fields)cudaFree(x.device);if(c->data.workspace.allocation)cudaFree(c->data.workspace.allocation);if(c->data.heat_accum){cudaFree(c->data.heat_accum);c->data.heat_accum=nullptr;c->data.heat_accum_bytes=0;}if(c->data.event_compute_done)cudaEventDestroy((cudaEvent_t)c->data.event_compute_done);if(c->data.event_transfer_done)cudaEventDestroy((cudaEvent_t)c->data.event_transfer_done);if(c->data.dgrid_dynamics_graph_exec)cudaGraphExecDestroy((cudaGraphExec_t)c->data.dgrid_dynamics_graph_exec);cudaStreamDestroy((cudaStream_t)c->data.compute_stream);cudaStreamDestroy((cudaStream_t)c->data.transfer_stream);
#endif
  delete c;*pc=nullptr;return 0;}
