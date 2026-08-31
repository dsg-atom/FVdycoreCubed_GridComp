#include "fv3_gpu_production_dispatch.hpp"
#include "fv3_gpu_errors.hpp"
#include "../cgrid/fv3_cgrid.hpp"
#include "../vertical/fv3_vertical_kernels.cuh"
#include "../operators/wp_d_contract_cuda.hpp"
#include "../operators/wp_d_contract.hpp"
#include "../dgrid/wp_b_contract_cuda.hpp"
#include "../dgrid/wp_b_cuda.cuh"

namespace fv3::gpu {
namespace {

/* Prompt 6B -- the production corner service.
 *
 * This slot was a deliberate fail-stub through the previous milestone, which is
 * why the harness had to pin config.nf_ke = 1: del2_cubed calls copy_corners
 * between its two flux directions on every pass with nt > 0
 * (dyn_core.F90:2419 and :2437), so nf_ke > 1 could not run at all.
 *
 * The authoritative operation is copy_corners (tp_core.F90:247-322): an in-place
 * rotation of the four ng x ng halo corner blocks of a SCALAR, in the x
 * direction (dir = 1) or the y direction (dir = 2). A faithful device port
 * already exists and is exercised by WP-B for both directions
 * (dgrid/wp_b_cuda.cu:94 copy_corners_kernel, launcher at :681), so this is an
 * adapter onto existing cubed-sphere topology infrastructure, NOT a second
 * corner subsystem -- per the reuse rule in CLAUDE.md.
 *
 * ng is the fv_mp_mod compile-time PARAMETER (= 3, tools/fv_mp_mod.F90:108)
 * that tp_core.F90 `use`s directly, so the rotated block is always 3 x 3
 * regardless of the caller's nt. CornerServiceRequest::halo_width therefore
 * describes the requester's WRITE band (nt), not the corner-block size, and is
 * deliberately not forwarded to the kernel.
 *
 * Numerical semantics changed: NO -- it makes a previously unreachable
 * authoritative operation reachable.
 */
int corner_edge_service_cuda(const ExecutionContext &x, const CornerServiceRequest &r,
                             fv3_gpu_error *e) noexcept {
  int rc = validate_corner_request(x, r, e);
  if (rc) return rc;
  if (r.operation != CornerOperation::copy ||
      r.orientation != ComponentOrientation::scalar ||
      r.stagger != FV3_GPU_STAGGER_A ||
      (r.axis != Axis::x && r.axis != Axis::y))
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "the production corner service implements only the scalar A-grid "
                "copy_corners rotation of tp_core.F90:247 (operation=copy, "
                "orientation=scalar, stagger=A, axis=x|y)");
  if (x.submission.stream_id != StreamId::compute)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "the corner service requires the borrowed compute stream");

  const DeviceViewBase *base = x.fields->find(r.first);
  if (!base || base->rank != 3 || !view_element_width_is(base, sizeof(fv_real)))
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                "copy_corners requires a rank-3 fv_real field");
  /* copy_corners_kernel derives its level from blockIdx.z and indexes the view
   * with that absolute k, so the descriptor must be 0-based in the vertical --
   * the same convention riem_solver3_kernel documents. */
  if (base->lower[2] != 0)
    return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
                "copy_corners requires a 0-based vertical descriptor");

  const auto &db = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const auto &topo = *x.immutable->topology;
  constexpr int64_t ng = 3;
  /* The rotation reads and writes i,j in [1-ng, 0] and [npx, npx+ng-1], i.e.
   * exactly the isd..ied / jsd..jed box of the authoritative declaration. */
  for (uint32_t axis = 0; axis < 2; ++axis) {
    const int64_t lo = axis == 0 ? db.is - ng : db.js - ng;
    const int64_t hi = axis == 0 ? db.ie + ng : db.je + ng;
    const int64_t view_hi =
        base->lower[axis] + static_cast<int64_t>(base->extent[axis]) - 1;
    if (base->lower[axis] > lo || view_hi < hi)
      return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                  "copy_corners requires a field declared over isd..ied/jsd..jed");
  }

  dgrid::TransportBounds b{};
  b.is = db.is; b.ie = db.ie; b.js = db.js; b.je = db.je;
  b.npz = db.npz;
  b.nk = static_cast<int64_t>(base->extent[2]);
  b.halo_width = static_cast<int64_t>(db.halo_width);
  b.ng = ng;
  b.npx = topo.npx;
  b.npy = topo.npy;
  // Fortran guard: .not. nested .and. grid_type < 3 (tp_core.F90:255 / :349).
  b.nested = topo.nested ? 1 : 0;
  b.cubed_sphere = (topo.global_cubed_sphere && cfg.grid_type < 3) ? 1 : 0;
  b.west_edge = (db.is == 1) ? 1 : 0;
  b.east_edge = (db.ie + 1 == topo.npx) ? 1 : 0;
  b.south_edge = (db.js == 1) ? 1 : 0;
  b.north_edge = (db.je + 1 == topo.npy) ? 1 : 0;
  b.sw_corner = topo.sw_corner ? 1 : 0;
  b.se_corner = topo.se_corner ? 1 : 0;
  b.ne_corner = topo.ne_corner ? 1 : 0;
  b.nw_corner = topo.nw_corner ? 1 : 0;

  DeviceView<fv_real> field{};
  static_cast<DeviceViewBase &>(field) = *base;
  // LevelFilter{} carries mode = 0, which level_active reads as "every level"
  // (dgrid/wp_b_ppm_shared.cuh:51). copy_corners is unconditional in the
  // authoritative source; there is no per-level gate to reproduce here.
  const int32_t dir = (r.axis == Axis::x) ? 1 : 2;
  const fv3_gpu_status s = dgrid::launch_copy_corners_cuda(
      b, dir, dgrid::LevelFilter{}, operators::view3<fv_real>(field),
      x.submission.stream);
  if (s != FV3_GPU_SUCCESS)
    return fail(e, s, __func__, "copy_corners CUDA launch failed");
  return FV3_GPU_SUCCESS;
}
} // namespace

int fv3_build_production_dispatch(NumericalDispatch &out, fv3_gpu_error *e) noexcept {
  clear_error(e);
  NumericalDispatch d{};
  d.contract_version = internal_contract_version;
  // Milestone 3A.9 §3: the complete per-level c_sw group of dyn_core.F90:514-524
  // -- d2a2c_vect, the nord>0 divergence_corner, the interleaved
  // fill2_4corners/fill_4corners direction pairs, the ut/vt scaling, both flux
  // directions, delpc/ptc/wc, the kinetic-energy and vorticity chain, and the
  // time-centred uc/vc update. It was a fail-stub through 3A.8.
  d.cgrid_predictor = &fv3::gpu::cgrid::launch_cgrid_predictor_v3;
  // Milestone 3A.9 §4: the C-grid vertical group, a COMPOSITE, not an alias.
  // Through 3A.8 this slot's implementation delegated to launch_vertical_solver
  // (Riem_Solver3); dyn_core.F90:583 calls Riem_Solver_C, a different routine
  // with a different argument list and dt2 rather than dt, so wiring the old body
  // would have been numerically wrong rather than merely redundant.
  // Milestone 3A.11S split the group in two so H_GZ_FIRST can be resolved
  // between the gz reconstruction and its first consumer, matching dyn_core.F90
  // :436-455 (rebuild) -> :459 (start) -> :552 (complete) -> :557 (zh <- gz).
  //
  // pre_halo runs: gz from delz/zs on the first substep (:445-455).
  d.cgrid_vertical_pre_halo = &fv3::gpu::vertical::launch_cgrid_vertical_pre_halo;
  // cgrid_vertical_solver now runs: the gz<->zh sync over the full halo box
  // (:557-573), update_dz_c per interface level (:577, dt2) with both
  // fill_4corners(gz2) inside the level body, and Riem_Solver_C (:583, dt2).
  d.cgrid_vertical_solver = &fv3::gpu::vertical::launch_cgrid_vertical_solver;
  d.cgrid_pressure = &fv3::gpu::cgrid::launch_cgrid_pressure_v3;
  // Milestone 3A.8 (WP-B.5) populates this slot. It was deliberately nullptr
  // through 3A.7, when only d_sw's individual slices existed:
  //   3A.4 -- update_dz_d height transport (launch_dgrid_height_transport_cuda)
  //   3A.5 -- arbitrary scalar transport   (launch_scalar_transport_cuda)
  //   3A.6 -- momentum transport           (launch_xtp_u_cuda / launch_ytp_v_cuda,
  //           the vb/ub constructions, the kinetic-energy terms, the vorticity
  //           precursor and the D-wind update)
  //   3A.7 -- divergence damping           (both nord_k branches, the two
  //           fill_corners rotations, smag_corner, the d_con heat-source chain
  //           and the damp_v del6_vt_flux add -- wp_b_divergence_cuda.cuh)
  // Wiring a PARTIAL dgrid_dynamics would have let the production path silently
  // skip stages, so the rule was: not until the complete ordered driver exists.
  //
  // It now does. launch_d_sw_cuda (dgrid/wp_b_dsw_cuda.cuh) executes the full
  // sw_core.F90 d_sw sequence S1..S38 -- the four services above plus the twelve
  // driver-owned kernels (contravariant/edge/corner winds, advective-flux
  // finalize, the joint pt/delp update, the w chain, the flux capacitors and the
  // heat_source reset) -- with the per-level nord_k / nord_v / nord_w / damp_vt /
  // damp_w / damp_t / d_con / kgb selectors that dyn_core.F90:699-765 recomputes
  // inside its k loop, so both divergence-damping branch families can be live in
  // one call. Anything it has not been validated against
  // (grid_type != 0, nested, hydrostatic, hord_* other than the production
  // first-run set, a_imp != 1, inline_q, use_cond, do_f3d, do_skeb, rot3) is
  // rejected with FV3_GPU_UNSUPPORTED_CONFIGURATION before it mutates any state.
  d.dgrid_dynamics = &fv3::gpu::dgrid::launch_dgrid_dynamics_cuda;
  // Milestone 3A.4: the authoritative update_dz_d transport body
  // (nh_utils.F90:255-305) -- edge_profile -> ra_x/ra_y -> per-level
  // damped/undamped split -> fv_tp_2d -> del6_vt_flux -> area-weighted height
  // update -- wired for WP-C's update_dz_d dependency.
  //
  // Milestone 3A.3's "damp_c mapping ambiguity" is resolved and that note is
  // gone: the zh path calls sw_core.F90's del6_vt_flux, which uses the raw
  // per-level damp(k) and emits separate fx2/fy2, NOT tp_core.F90's deln_flux
  // with its (damp_c*da_min)**(nord+1) coefficient folded into fx/fy. There
  // was no coefficient to map.
  //
  // Milestone 3A.5 repaired launch_deln_flux_transport_cuda to the authoritative
  // tp_core.F90:1263 deln_flux semantics (coefficient (damp_c*da_min)**(nord+1),
  // 2+2*nord copy_corners rotations, flux folded into fx/fy, 1.e-4 damp_c
  // threshold) and consumes it from launch_scalar_transport_cuda. It is still
  // never invoked from the height path -- the two damping operators stay
  // distinct, exactly as the Fortran keeps them distinct.
  d.dgrid_height_transport = &fv3::gpu::dgrid::launch_dgrid_height_transport_cuda;
  // Milestone 3A.9 §5/§6: dgrid_height IS update_dz_d (dyn_core.F90:905-906),
  // the WP-C sequencer that CONSUMES the WP-B dgrid_height_transport service
  // above. Wiring it is what finally makes that service reachable through
  // fv3_gpu_dyn_core_step -- it was implemented and wired but had no caller on
  // the production path, because its only caller (launch_update_dz_d) needed the
  // dispatch table and so could not itself be a LaunchFn.
  //
  // NOT launch_dgrid_height_v2. That launcher recomputes delz, which
  // authoritatively belongs to Riem_Solver3 (nh_core.F90:181); wiring it here
  // would write delz a substep early. See the §5 block comment in
  // vertical/fv3_vertical_kernels.cu.
  d.dgrid_height = &fv3::gpu::vertical::launch_dgrid_height_stage;
  d.vertical_solver = &fv3::gpu::vertical::launch_vertical_solver;
  d.nh_pressure = &fv3::gpu::operators::launch_nh_pressure_cuda;
  d.post_loop_heating = &fv3::gpu::operators::launch_post_loop_heating_cuda;
  // Prompt 6B: was corner_service_not_implemented, which forced config.nf_ke=1.
  // See the block comment above corner_edge_service_cuda.
  d.corner_edge_service = &corner_edge_service_cuda;
  // Prompt 6B omega finalization (dyn_core.F90:503-517 and :1205-1218).
  d.omega_pem_snapshot = &fv3::gpu::operators::launch_omega_pem_snapshot_cuda;
  d.omega_finalize = &fv3::gpu::operators::launch_omega_finalize_cuda;
  out = d;
  return FV3_GPU_SUCCESS;
}

} // namespace fv3::gpu
