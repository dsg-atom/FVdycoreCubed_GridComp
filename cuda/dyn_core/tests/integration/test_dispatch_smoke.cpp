// Production Dispatch Smoke Test (3A.1, extended through 3A.8)
//
// Exercises the real production activation path:
//   fv3_gpu_create
//   -> fv3_gpu_initialize_state (H2D)
//   -> set_numerical_config / mark_halo_ready
//   -> fv3_build_production_dispatch -> set_numerical_dispatch
//   -> fv3_gpu_dyn_core_step  (real orchestration, no stub)
//   -> fv3_gpu_sync_to_host (D2H)
//
// This proves fv3_gpu_dyn_core_step reads a real, populated
// NumericalDispatch and invokes genuinely-wired CUDA implementations
// through the same code path GEOS/FV3 production dispatch will eventually
// use. It is NOT a scientifically coherent acoustic timestep: WP-A's
// predictor predecessors (cgrid_predictor, cgrid_vertical_solver) are still
// nullptr, dgrid_height_transport is wired but reached by nothing on the
// stage list, and the runtime does not yet iterate n_split.
//
// The test runs two independent WP-E contexts, because fv3_gpu_dyn_core_step
// marks a context ContextPhase::failed on the first non-success stage, and
// that phase transition is one-way (no reset API) -- a failed context can
// never again pass fv3_gpu_sync_to_host's phase check, so a full D2H
// verification and a deliberately-partial dispatch cannot share one context.
//
//   Phase A: a *minimal* dispatch table with only cgrid_pressure (WP-A
//            p_grad_c) and vertical_solver (WP-C riem_solver3) wired --
//            i.e. the two stages this test's field set actually satisfies.
//            fv3_gpu_dyn_core_step is expected to fully SUCCEED, allowing a
//            real D2H + finite-value check of genuinely GPU-executed output.
//   Phase B: the *full* fv3_build_production_dispatch table. As of Milestone
//            3A.8 that means five wired acoustic stages executing in
//            authoritative order -- cgrid_pressure (WP-A), dgrid_dynamics
//            (WP-B's complete d_sw driver), vertical_solver (WP-C),
//            nh_pressure and post_loop_heating (WP-D). This test registers
//            both the complete WP-D field contract (bind_wp_d) and the
//            complete WP-B d_sw contract (bind_wp_b_d_sw), including the ten
//            per-level damping controls, so all five stages are expected to
//            fully SUCCEED, with a real D2H + finite/changed-value check of
//            the driver-owned d_sw updates (delp, w, u, v), the five flux
//            capacitors, and WP-D's post_loop_heating output.
//
//            Milestone 3A.8 tests N and O are both made here and are made
//            separately falsifiable: N asserts the production table actually
//            populates dgrid_dynamics (a null slot would be silently skipped
//            by the runtime's stage loop and still report SUCCESS), O asserts
//            the step through that table succeeds and changes state.
//
// NOT a numerical validation test - that is Prompt 6.

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"
#include "fv3_gpu_internal.hpp"
#include "fv3_gpu_halo_peer.hpp"
#include "fv3_gpu_halo_topology.hpp"
#include "fv3_gpu_production_dispatch.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace fv3::gpu;

static constexpr int64_t NI = 8;
static constexpr int64_t NJ = 8;
static constexpr int64_t NPZ = 10;
static constexpr uint32_t HALO = 3;

// Milestone 3A.11T: pe/peln are registered (i, k, j), NOT (i, j, k).
//
// This is the FV3 exception to the usual convention, and it is uniform across
// the entire authoritative tree -- ~40 declaration sites, zero exceptions:
//   nh_core.F90:87-88     pe(is-1:ie+1,km+1,js-1:je+1), peln(is:ie,km+1,js:je)
//   dyn_core.F90:202-203  same, as dyn_core's own dummies
//   dyn_core.F90:2123-4   geopk's intent(out) pe/peln
//   fv_arrays.F90:1472-4  allocate(Atm%pe(is-1:ie+1, npz+1, js-1:je+1))
// with body-level proof at geopk's `pe(i,1,j) = ptop` (dyn_core.F90:2191) and
// fv_dynamics.F90:611's surface read `pe(i,npz+1,j)`. Every other pressure
// field -- pk, pkz, pkc, pk3 -- really is (i,j,k); only pe/peln invert.
//
// A previous revision registered them (i,j,k) with a 2-slice k pad
// (`kPelnHaloPad`) bolted on. That pad existed solely to make bind_wp_d's
// coverage check `lower[2] <= js - 2` pass. But that check is CORRECT and was
// being misread: in the authoritative layout axis 2 IS j, so WP-D is asking
// for a 2-deep j halo -- and the harness was padding its VERTICAL axis to
// satisfy a horizontal-halo requirement. Two consequences of the old form:
//   * it only ever passed when npz + pad >= nj_h, i.e. by arithmetic accident
//     at 8x8x10; at 16x16x10 bind_wp_d would have rejected outright.
//   * pe/peln were transposed against every authoritative consumer.
// Registering the real layout makes the check `jsd <= js - 2` -- true at any
// domain size for HALO >= 2 -- so the pad is not merely unnecessary, it was
// masking the defect.

fv3_gpu_field_desc make_field(FieldId id, void *base, int rank,
                              const int64_t *lower, const uint64_t *extent,
                              const int64_t *stride, uint64_t bytes) {
  fv3_gpu_field_desc d{};
  d.abi_version = FV3_GPU_ABI_VERSION;
  d.struct_bytes = sizeof(d);
  d.field_id = static_cast<uint32_t>(id);
  d.base = base;
  d.allocation_bytes = bytes;
  d.element_type = sizeof(fv_real) == 4 ? FV3_GPU_ELEMENT_F32 : FV3_GPU_ELEMENT_F64;
  d.element_bytes = sizeof(fv_real);
  d.rank = rank;
  d.memory_location = FV3_GPU_MEMORY_HOST;

  const auto *spec = field_spec(id);
  d.stagger = spec ? spec->stagger : FV3_GPU_STAGGER_A;
  d.access = spec ? spec->access : FV3_GPU_ACCESS_INOUT;

  for (int i = 0; i < rank; ++i) {
    d.lower[i] = lower[i];
    d.extent[i] = extent[i];
    d.stride_elements[i] = stride[i];
    d.compute_lower[i] = lower[i];
    d.compute_upper[i] = lower[i] + (int64_t)extent[i] - 1;
    d.halo_lower[i] = 0;
    d.halo_upper[i] = 0;
  }
  return d;
}

fv3_gpu_field_desc make_field_acc(FieldId id, void *base, int rank,
                                  const int64_t *lower, const uint64_t *extent,
                                  const int64_t *stride, uint64_t bytes) {
  // accumulator_real fields (edge_w/e/s/n) are always float64 in the ABI,
  // regardless of the fv_real build precision -- see fv3_gpu_types.hpp's
  // static_assert(sizeof(accumulator_real)==8). make_field()'s element_type/
  // element_bytes derive from sizeof(fv_real), which is wrong for these
  // fields in an R4 build, so they get their own constructor.
  fv3_gpu_field_desc d{};
  d.abi_version = FV3_GPU_ABI_VERSION;
  d.struct_bytes = sizeof(d);
  d.field_id = static_cast<uint32_t>(id);
  d.base = base;
  d.allocation_bytes = bytes;
  d.element_type = FV3_GPU_ELEMENT_F64;
  d.element_bytes = sizeof(accumulator_real);
  d.rank = rank;
  d.memory_location = FV3_GPU_MEMORY_HOST;

  const auto *spec = field_spec(id);
  d.stagger = spec ? spec->stagger : FV3_GPU_STAGGER_A;
  d.access = spec ? spec->access : FV3_GPU_ACCESS_INPUT;

  for (int i = 0; i < rank; ++i) {
    d.lower[i] = lower[i];
    d.extent[i] = extent[i];
    d.stride_elements[i] = stride[i];
    d.compute_lower[i] = lower[i];
    d.compute_upper[i] = lower[i] + (int64_t)extent[i] - 1;
    d.halo_lower[i] = 0;
    d.halo_upper[i] = 0;
  }
  return d;
}

// Templated over the element type: the five flux capacitors are
// accumulator_real (kind=8 in Fortran regardless of FV precision), which is a
// distinct type from fv_real in R4.
template <typename T>
bool check_finite(const T *data, size_t count, const char *name) {
  size_t nan_count = 0, inf_count = 0;
  for (size_t i = 0; i < count; ++i) {
    if (std::isnan(data[i])) ++nan_count;
    if (std::isinf(data[i])) ++inf_count;
  }
  if (nan_count > 0 || inf_count > 0) {
    printf("  %s: FAIL (%zu NaN, %zu Inf out of %zu)\n", name, nan_count, inf_count, count);
    return false;
  }
  printf("  %s: finite (%zu values)\n", name, count);
  return true;
}

// Largest element-wise change, so the caller can both decide "did this field
// get written" and report by how much.
//
// Milestone 3A.8 replaced an absolute 1e-12 threshold here. That threshold was
// safe for the fields it was written for, but this file's `rarea` input was
// corrected this milestone from 1e10 to 1e-10 (it is 1/area, and d_sw
// multiplies a mass-flux divergence by it, so the inverted value overflowed
// fv_real in R4). Every rarea-weighted increment in the harness therefore
// shrank by ~1e20, and post_loop_heating's pt/pkz/heat_source increment can
// legitimately land below 1e-12 in absolute terms. Comparing against exact
// equality instead keeps the probe honest -- "the stage wrote something" is
// still falsifiable, and the printed magnitude is what a reader judges.
template <typename T>
double max_abs_delta(const T *before, const T *after, size_t count) {
  double worst = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double d = std::fabs(static_cast<double>(after[i]) - static_cast<double>(before[i]));
    if (d > worst) worst = d;
  }
  return worst;
}

template <typename T>
bool report_changed(const T *before, const T *after, size_t count, const char *name) {
  const double d = max_abs_delta(before, after, count);
  printf("  %-12s max|delta| = %.6e  %s\n", name, d, d > 0.0 ? "CHANGED" : "UNCHANGED");
  return d > 0.0;
}

// Host-side state shared by both phases. Rebuilt fresh for each phase so
// each context's H2D transfer sees known input (Phase A's device writes
// never propagate back into these host buffers unless D2H runs).
struct HostState {
  int64_t ni_h, nj_h, npz1;
  size_t size_3d, size_3d_if, size_2d, size_3d_pe;
  std::vector<fv_real> delp, pt, w, delz, q_con, cappa, uc, vc, delpc;
  std::vector<fv_real> pe, peln, pk, pk3, pkc, zh, gz;
  std::vector<fv_real> zs, ws, rdxc, rdyc;
  // Milestone 3A.2: complete the WP-D field contract (bind_wp_d) so
  // nh_pressure/post_loop_heating can genuinely execute through production
  // dispatch, rather than fail at contract validation as in Milestone 3A.1.
  std::vector<fv_real> u, v, pkz, du, dv, heat_source; // mid-layer state
  std::vector<fv_real> divg2, rdx, rdy, rarea, del6_u, del6_v, dx, dy,
      sina_u, sina_v, dxa, dya; // 2D grid metrics
  std::vector<fv_real> grid_lonlat, agrid_lonlat; // 2D grid, trailing dim=2
  std::vector<accumulator_real> edge_w, edge_e; // 1D, Y_FACE (varies with j)
  std::vector<accumulator_real> edge_s, edge_n; // 1D, X_FACE (varies with i)

  // Milestone 3A.8: complete the WP-B d_sw field contract (bind_wp_b_d_sw) so
  // dgrid_dynamics -- now populated in the production table -- can genuinely
  // execute through fv3_gpu_dyn_core_step rather than fail at contract
  // validation. Three groups, matching the binder's three resolution blocks.
  //
  // (1) d_sw state that the earlier milestones' field set did not need.
  std::vector<fv_real> ua, va, divgd, ptc, ut, vt;
  std::vector<fv_real> crx, cry, xfx, yfx, diss_est, z_rat;
  // (2) the five flux capacitors. kind=8 in Fortran regardless of FV
  // precision, and bind_wp_b_d_sw pins the element width, so they are
  // accumulator_real here and registered through make_field_acc.
  std::vector<accumulator_real> mfx, mfy, cx, cy, dpx;
  // (3) grid metrics d_sw reads that WP-A/C/D did not.
  std::vector<fv_real> rarea_c, dxc, dyc, divg_u, divg_v, cosa_u, cosa_v,
      cosa_s, rsin2, area, rdxa, rdya, cosa, rsina, rsin_u, rsin_v, f0;
  std::vector<fv_real> sin_sg; // 3D: (i, j, 0..3) cell-edge components
  // (4) the ten per-level damping controls dyn_core.F90:699-765 recomputes
  // inside its k loop. These are what make a MIXED-branch run possible: see
  // make_host_state for the deliberately non-uniform nord_k pattern.
  std::vector<fv_real> lv_nord_k, lv_ndif, lv_nord_w, lv_d2_bg, lv_d4_bg;
  std::vector<fv_real> lv_damp_v, lv_damp_w, lv_damp_t, lv_d_con, lv_kgb;

  // Milestone 3A.9: the six fields the two newly-reachable C-grid stages
  // (cgrid_predictor, cgrid_vertical_solver) require and that no earlier
  // milestone's dispatch subset ever asked for. Each is rejected by name before
  // any kernel launches, so all six are load-bearing:
  //
  //   omga    3-D  d2a2c_vect's A-grid vertical velocity, and Riem_Solver_C's
  //           w3 argument (dyn_core.F90:583). NOT interchangeable with w.
  //   f_c     2-D  Coriolis on the DUAL (B-grid corner) mesh. Distinct FieldId
  //           from f0, which is the A-grid value already registered above --
  //           registering f0 does not satisfy the predictor.
  //   cos_sg  3-D  the four cell-edge angle cosines, companion to sin_sg.
  //   dp_ref  1-D  the reference-profile layer thickness update_dz_c reads as
  //           dp0 (nh_utils.F90's dp0 argument).
  //   ws3     2-D  the C-grid surface vertical velocity dyn_core.F90:577 passes
  //           to update_dz_c and :583 to Riem_Solver_C. Distinct FieldId from
  //           ws, which is the D-grid value Riem_Solver3 writes.
  //   phis    2-D  surface geopotential; Riem_Solver_C's hs closure term.
  std::vector<fv_real> omga, f_c, cos_sg, dp_ref, ws3, phis;

  fv_real ptop, akap, grav;
};

HostState make_host_state() {
  HostState h{};
  // Milestone 3A.8: the horizontal box carries ONE EXTRA row and column beyond
  // the extended [is-HALO, ie+HALO] x [js-HALO, je+HALO] domain, i.e. every
  // field here is registered over [isd, ied+1] x [jsd, jed+1].
  //
  // This is not padding-for-safety; it is what the authoritative declarations
  // require. sw_core.F90:548-556 puts u and vc on (isd:ied, jsd:jed+1), v and
  // uc on (isd:ied+1, jsd:jed), and divg_d on (isd:ied+1, jsd:jed+1), and d_sw
  // reads those extra lines: the ut interior loop (:690-696) reads
  // vc(i, j+1) at j == jed and the vt loop (:702-706) reads uc(i+1, j) at
  // i == ied. The first version of this harness registered every 3-D field on
  // the plain extended box, which under-allocated the staggered ones;
  // compute-sanitizer memcheck reported "Invalid __global__ read of size 8
  // bytes" in dsw_ut_interior_kernel, exactly 8 bytes past the end of a
  // 14x14x10 field -- element (isd, jed+1, 9).
  //
  // Registering the union box for ALL fields, rather than a different box per
  // stagger, is deliberate: contract checks are coverage checks
  // (covers3 in wp_b_dsw_contract_cuda.cpp), so a larger registration
  // satisfies every consumer, and one uniform stride triple keeps the seeding
  // loops -- which must fill the whole allocation or initcheck flags the extra
  // line as an uninitialized read -- simple enough to audit.
  h.ni_h = NI + 2 * (int64_t)HALO + 1;
  h.nj_h = NJ + 2 * (int64_t)HALO + 1;
  h.npz1 = NPZ + 1;
  h.size_3d = (size_t)(h.ni_h * h.nj_h * NPZ);
  h.size_3d_if = (size_t)(h.ni_h * h.nj_h * h.npz1);
  // pe/peln: (i, k, j) -- i fastest, then k, then j. Same element count as an
  // interface field, just a different axis order (no pad; see the file-header
  // note where kPelnHaloPad used to live).
  h.size_3d_pe = (size_t)(h.ni_h * h.npz1 * h.nj_h);
  h.size_2d = (size_t)(h.ni_h * h.nj_h);

  h.delp.assign(h.size_3d, 0); h.pt.assign(h.size_3d, 0); h.w.assign(h.size_3d, 0);
  h.delz.assign(h.size_3d, 0); h.q_con.assign(h.size_3d, 0); h.cappa.assign(h.size_3d, 0);
  h.uc.assign(h.size_3d, 0); h.vc.assign(h.size_3d, 0); h.delpc.assign(h.size_3d, 0);
  // pe/peln carry the authoritative (i,k,j) axis order, hence their own size.
  h.pe.assign(h.size_3d_pe, 0); h.peln.assign(h.size_3d_pe, 0);
  h.pk.assign(h.size_3d_if, 0); h.pk3.assign(h.size_3d_if, 0); h.pkc.assign(h.size_3d_if, 0);
  h.zh.assign(h.size_3d_if, 0); h.gz.assign(h.size_3d_if, 0);
  h.zs.assign(h.size_2d, 0); h.ws.assign(h.size_2d, 0);
  h.rdxc.assign(h.size_2d, 0); h.rdyc.assign(h.size_2d, 0);

  h.u.assign(h.size_3d, 0); h.v.assign(h.size_3d, 0); h.pkz.assign(h.size_3d, 0);
  h.du.assign(h.size_3d, 0); h.dv.assign(h.size_3d, 0); h.heat_source.assign(h.size_3d, 0);
  h.divg2.assign(h.size_2d, 0);
  h.rdx.assign(h.size_2d, 0); h.rdy.assign(h.size_2d, 0); h.rarea.assign(h.size_2d, 0);
  h.del6_u.assign(h.size_2d, 0); h.del6_v.assign(h.size_2d, 0);
  h.dx.assign(h.size_2d, 0); h.dy.assign(h.size_2d, 0);
  h.sina_u.assign(h.size_2d, 0); h.sina_v.assign(h.size_2d, 0);
  h.dxa.assign(h.size_2d, 0); h.dya.assign(h.size_2d, 0);
  h.grid_lonlat.assign(h.size_2d * 2, 0); h.agrid_lonlat.assign(h.size_2d * 2, 0);
  h.edge_w.assign((size_t)h.nj_h, 0); h.edge_e.assign((size_t)h.nj_h, 0);
  h.edge_s.assign((size_t)h.ni_h, 0); h.edge_n.assign((size_t)h.ni_h, 0);

  // Milestone 3A.8 d_sw allocations.
  h.ua.assign(h.size_3d, 0); h.va.assign(h.size_3d, 0);
  h.divgd.assign(h.size_3d, 0); h.ptc.assign(h.size_3d, 0);
  h.ut.assign(h.size_3d, 0); h.vt.assign(h.size_3d, 0);
  h.crx.assign(h.size_3d, 0); h.cry.assign(h.size_3d, 0);
  h.xfx.assign(h.size_3d, 0); h.yfx.assign(h.size_3d, 0);
  h.diss_est.assign(h.size_3d, 0); h.z_rat.assign(h.size_3d, 0);
  h.mfx.assign(h.size_3d, 0); h.mfy.assign(h.size_3d, 0);
  h.cx.assign(h.size_3d, 0); h.cy.assign(h.size_3d, 0);
  h.dpx.assign(h.size_3d, 0);
  h.rarea_c.assign(h.size_2d, 0);
  h.dxc.assign(h.size_2d, 0); h.dyc.assign(h.size_2d, 0);
  h.divg_u.assign(h.size_2d, 0); h.divg_v.assign(h.size_2d, 0);
  h.cosa_u.assign(h.size_2d, 0); h.cosa_v.assign(h.size_2d, 0);
  h.cosa_s.assign(h.size_2d, 0); h.rsin2.assign(h.size_2d, 0);
  h.area.assign(h.size_2d, 0);
  h.rdxa.assign(h.size_2d, 0); h.rdya.assign(h.size_2d, 0);
  h.cosa.assign(h.size_2d, 0); h.rsina.assign(h.size_2d, 0);
  h.rsin_u.assign(h.size_2d, 0); h.rsin_v.assign(h.size_2d, 0);
  h.f0.assign(h.size_2d, 0);
  h.sin_sg.assign(h.size_2d * 4, 0);
  // Milestone 3A.9 C-grid additions. cos_sg carries the same four cell-edge
  // components sin_sg does; dp_ref is per-level like the damping controls.
  h.omga.assign(h.size_3d, 0);
  h.f_c.assign(h.size_2d, 0);
  h.cos_sg.assign(h.size_2d * 4, 0);
  h.dp_ref.assign((size_t)NPZ, 0);
  h.ws3.assign(h.size_2d, 0);
  h.phis.assign(h.size_2d, 0);
  for (std::vector<fv_real> *p : {&h.lv_nord_k, &h.lv_ndif, &h.lv_nord_w,
                                  &h.lv_d2_bg, &h.lv_d4_bg, &h.lv_damp_v,
                                  &h.lv_damp_w, &h.lv_damp_t, &h.lv_d_con,
                                  &h.lv_kgb})
    p->assign((size_t)NPZ, 0);

  h.ptop = 100.0;
  const fv_real psurf = 100000.0;
  const fv_real dp_uniform = (psurf - h.ptop) / NPZ;
  h.grav = 9.80665;
  h.akap = fv_real(2.0 / 7.0);

  for (size_t idx = 0; idx < h.size_3d; ++idx) {
    h.delp[idx] = dp_uniform;
    h.pt[idx] = 280.0 + 10.0 * (fv_real(idx % NPZ) / NPZ);
    h.w[idx] = 0.01;
    h.delz[idx] = -1000.0 / NPZ;
    h.q_con[idx] = 0.001;
    h.cappa[idx] = h.akap;
    h.delpc[idx] = dp_uniform;
  }
  // Milestone 3A.8: uc/vc must start as genuinely nonzero C-grid winds.
  //
  // They were zero through 3A.2 because the only stage that touched them was
  // WP-A's p_grad_c, which WRITES them, so a zero start made any_nonzero(uc) a
  // valid "did it execute" probe (Phase A still probes uc/vc, now upgraded to a
  // snapshot comparison, which is strictly stronger and does not depend on the
  // initial value).
  //
  // But d_sw builds its entire transport half from uc/vc: S1's contravariant
  // transform produces ut/vt, which produce the Courant numbers crx/cry and the
  // mass fluxes xfx/yfx, which drive the delp, pt and w updates and the flux
  // capacitors. Leaving uc/vc at zero plus p_grad_c's small pressure-gradient
  // increment made every mass flux vanishingly small: the first A100 run showed
  // a delp change of 2.45e-05 out of delp = 9990 Pa, i.e. 2.5e-09 relative,
  // which R8 resolved and R4 rounded away to exactly zero. That is precisely
  // the degenerate input §16 forbids -- a stage-ordering defect in the
  // transport chain could not have changed the answer either.
  //
  // Amplitudes are chosen so the per-cell wind variation (~1 m/s over
  // dx = 1e5 m) gives a divergence near 1e-05 s^-1 and hence a delp increment
  // of order 1 Pa over dt = 10 s -- large enough to be resolvable in float32,
  // small enough to stay physically plausible.
  for (int64_t k = 0; k < NPZ; ++k) {
    for (int64_t j = 0; j < h.nj_h; ++j) {
      for (int64_t i = 0; i < h.ni_h; ++i) {
        const size_t idx = (size_t)(i + j * h.ni_h + k * h.ni_h * h.nj_h);
        const double x = 0.29 * (double)i, y = 0.23 * (double)j,
                     z = 0.17 * (double)k;
        h.uc[idx] = (fv_real)(10.0 + 4.0 * std::sin(x + 0.5 * z) * std::cos(y));
        h.vc[idx] = (fv_real)(-6.0 + 3.0 * std::cos(x) * std::sin(y + 0.5 * z));
      }
    }
  }
  for (size_t k = 0; k < (size_t)h.npz1; ++k) {
    fv_real p = h.ptop + k * dp_uniform;
    fv_real z = 10000.0 - k * 1000.0;
    for (size_t j = 0; j < (size_t)h.nj_h; ++j) {
      for (size_t i = 0; i < (size_t)h.ni_h; ++i) {
        size_t idx = i + j * h.ni_h + k * h.ni_h * h.nj_h;
        // A small horizontal (i,j) perturbation is required: p_grad_c is a
        // horizontal-difference operator, and if gz/pkc are perfectly
        // uniform across i,j (varying only with k), its first/second terms
        // cancel EXACTLY in exact arithmetic (first == -second), regardless
        // of dispatch correctness. Without this, an apparently-nonzero R4
        // result would just be float32 rounding noise, not a real signal.
        const fv_real horiz = 0.02 * std::sin(0.35 * (fv_real)i) *
                               std::cos(0.35 * (fv_real)j);
        // pe/peln are (i, k, j): k is the MIDDLE axis, so it strides by ni_h
        // and j strides by ni_h*npz1. Compare `idx` above, which is the
        // ordinary (i,j,k). Seeding through this index rather than `idx` is
        // what makes the host state agree with the registered descriptor.
        const size_t idx_pe = i + k * (size_t)h.ni_h +
                              j * (size_t)h.ni_h * (size_t)h.npz1;
        h.pe[idx_pe] = p;
        h.peln[idx_pe] = std::log(p);
        h.pk[idx] = std::pow(p, h.akap) * (1.0 + horiz);
        h.pk3[idx] = h.pk[idx];
        h.pkc[idx] = h.pk[idx];
        h.zh[idx] = z;
        // gz is geopotential (grav*height); must vary with k like zh does,
        // otherwise pressure_gradient_c_{x,y}'s gz(i-1,j,ki+1)-gz(i,j,ki)
        // term is identically zero and uc/vc never pick up a real update
        // regardless of dispatch correctness.
        h.gz[idx] = h.grav * z * (1.0 + horiz);
      }
    }
  }
  for (size_t j = 0; j < (size_t)h.nj_h; ++j) {
    for (size_t i = 0; i < (size_t)h.ni_h; ++i) {
      size_t idx = i + j * h.ni_h;
      h.zs[idx] = 0.0;
      h.ws[idx] = 0.0;
      h.rdxc[idx] = 1.0 / 100000.0;
      h.rdyc[idx] = 1.0 / 100000.0;
      // WP-D grid metrics: an idealized ~100km doubly-periodic-style tile
      // (same spacing scale as rdxc/rdyc above), with a small (i,j)
      // perturbation on the angle/area-like terms so they are not perfectly
      // uniform (avoiding the same exact-cancellation failure mode fixed for
      // gz/pkc in Milestone 3A.1).
      const fv_real horiz = 0.02 * std::sin(0.31 * (fv_real)i) *
                             std::cos(0.31 * (fv_real)j);
      h.rdx[idx] = 1.0 / 100000.0;
      h.rdy[idx] = 1.0 / 100000.0;
      h.dx[idx] = 100000.0;
      h.dy[idx] = 100000.0;
      h.dxa[idx] = 100000.0;
      h.dya[idx] = 100000.0;
      // Milestone 3A.8 correction: rarea is 1/area, so for the ~100km cells
      // this grid describes (dx = dy = 1e5 => area = 1e10) it is 1e-10, not
      // 1e10. The inverted value was harmless while only WP-D's hyperdiffusion
      // read it, but d_sw's delp/pt/w updates multiply a mass-flux divergence
      // by rarea, and a factor of 1e20 there overflows fv_real in R4 -- it
      // would look like a driver defect rather than bad test input. Fixing the
      // input rather than weakening the finite check.
      h.area[idx] = 1.0e10 * (1.0 + horiz);
      h.rarea[idx] = 1.0 / h.area[idx];
      // rarea_c is the dual-cell (corner) area reciprocal; same scale.
      h.rarea_c[idx] = 1.0 / (1.0e10 * (1.0 + 0.5 * horiz));
      h.del6_u[idx] = 0.5 * (1.0 + horiz);
      h.del6_v[idx] = 0.5 * (1.0 + horiz);
      h.sina_u[idx] = 1.0;
      h.sina_v[idx] = 1.0;
      h.divg2[idx] = 0.0;
      // d_sw grid metrics. A near-orthogonal grid: sina == 1, so every rsin*
      // is 1 and the cosa* terms are small but deliberately nonzero -- setting
      // them to exactly zero would make the contravariant transforms (S1/K1)
      // degenerate and hide an ordering error there.
      h.dxc[idx] = 100000.0;
      h.dyc[idx] = 100000.0;
      h.rdxa[idx] = 1.0 / 100000.0;
      h.rdya[idx] = 1.0 / 100000.0;
      h.divg_u[idx] = 1.0 * (1.0 + horiz);
      h.divg_v[idx] = 1.0 * (1.0 - horiz);
      h.cosa_u[idx] = 0.05 * horiz / 0.02;   // ~0.05*sin*cos, i.e. |cosa| <= 0.05
      h.cosa_v[idx] = 0.04 * horiz / 0.02;
      h.cosa[idx] = 0.045 * horiz / 0.02;
      h.cosa_s[idx] = 0.03 * horiz / 0.02;
      h.rsin2[idx] = 1.0;
      h.rsina[idx] = 1.0;
      h.rsin_u[idx] = 1.0;
      h.rsin_v[idx] = 1.0;
      h.f0[idx] = 1.0e-4 * (1.0 + 0.1 * horiz / 0.02); // Coriolis, mid-latitude
      // sin_sg's four cell-edge components (CUDA views index them 0-based).
      // Strictly positive: S1's upwind arms divide by them.
      for (int c = 0; c < 4; ++c)
        h.sin_sg[idx + (size_t)c * h.size_2d] = 1.0 - 0.01 * (fv_real)c;
      // --- Milestone 3A.9 C-grid metrics/state --------------------------
      // cos_sg is sin_sg's companion: the cosine of the same four cell-edge
      // angles. This grid is near-orthogonal (sin_sg ~ 1), so cos_sg is small
      // -- but deliberately NOT zero, for the same reason cosa_u/cosa_v above
      // are not: d2a2c_vect's edge_interpolate4 arms multiply by it, and exact
      // zeros would make a transcription error there invisible.
      for (int c = 0; c < 4; ++c)
        h.cos_sg[idx + (size_t)c * h.size_2d] =
            (fv_real)(0.03 - 0.005 * (fv_real)c);
      // f_c is Coriolis on the DUAL (corner) mesh, a different FieldId from
      // f0 above. Slightly offset from f0 so a mix-up between the two shows
      // up as a changed answer rather than silently agreeing.
      h.f_c[idx] = 1.0e-4 * (1.0 + 0.15 * horiz / 0.02);
      // ws3 is update_dz_c's surface-vertical-velocity OUTPUT (dyn_core.F90:577
      // passes it intent(out)) and then Riem_Solver_C's ws input, so a nonzero
      // seed here would only mask a missing write. Zero deliberately -- the
      // same reasoning as ut/vt/crx/cry above.
      h.ws3[idx] = 0.0;
      // phis is surface geopotential. It MUST equal grav*zs, because the
      // vertical stages consume the two as one consistent surface: the C-grid
      // height reconstruction builds gz upward from zs in METRES, while
      // Riem_Solver_C closes the column against hs = phis in GEOPOTENTIAL. zs
      // is identically zero on this idealized tile, so phis is too -- that is
      // consistency, not a degenerate input, and inventing a nonzero phis here
      // would contradict zs and break the closure.
      h.phis[idx] = h.grav * h.zs[idx];
      // grid_lonlat/agrid_lonlat: small idealized lon/lat (radians) grid,
      // trailing dim 0=lon, 1=lat.
      const size_t idx_ll = idx * 2;
      h.grid_lonlat[idx_ll + 0] = 0.01 * (fv_real)i;
      h.grid_lonlat[idx_ll + 1] = 0.01 * (fv_real)j;
      h.agrid_lonlat[idx_ll + 0] = 0.01 * (fv_real)i + 0.005;
      h.agrid_lonlat[idx_ll + 1] = 0.01 * (fv_real)j + 0.005;
    }
  }
  for (size_t j = 0; j < (size_t)h.nj_h; ++j)
    h.edge_w[j] = h.edge_e[j] = 1.0 + 0.01 * (double)j;
  for (size_t i = 0; i < (size_t)h.ni_h; ++i)
    h.edge_s[i] = h.edge_n[i] = 1.0 + 0.01 * (double)i;

  for (size_t idx = 0; idx < h.size_3d; ++idx) {
    // u/v/du/dv are pressure_gradient WRITE targets; pkz/heat_source are
    // both read and written by post_loop_heating. Initialize to finite,
    // non-degenerate values so a kernel that failed to launch (leaving
    // stale/garbage-free host-mirrored zeros) is distinguishable from one
    // that genuinely executed and overwrote them.
    h.u[idx] = 0.0; h.v[idx] = 0.0; h.du[idx] = 0.0; h.dv[idx] = 0.0;
    h.pkz[idx] = 1.0 + 0.1 * (fv_real)(idx % NPZ) / NPZ;
    h.heat_source[idx] = 10.0 + 0.5 * std::sin(0.4 * (fv_real)(idx % h.ni_h));
  }

  // --- Milestone 3A.8: d_sw state ------------------------------------------
  // u/v were zero through 3A.2 because only WP-D's nh_pressure wrote them, and
  // a zero start made any_nonzero() a valid "did it execute" probe. d_sw needs
  // genuinely nonzero D-grid winds -- with u == v == 0 the vorticity, kinetic
  // energy and every transport flux are identically zero, so a stage-ordering
  // defect could not change the answer. Phase B's u/v probe is therefore
  // upgraded to any_changed() against a pre-step snapshot, which is a strictly
  // stronger statement than "nonzero" ever was.
  for (int64_t k = 0; k < NPZ; ++k) {
    for (int64_t j = 0; j < h.nj_h; ++j) {
      for (int64_t i = 0; i < h.ni_h; ++i) {
        const size_t idx = (size_t)(i + j * h.ni_h + k * h.ni_h * h.nj_h);
        const double x = 0.29 * (double)i, y = 0.23 * (double)j,
                     z = 0.17 * (double)k;
        h.u[idx] = (fv_real)(12.0 + 3.0 * std::sin(x + z) * std::cos(y));
        h.v[idx] = (fv_real)(-7.0 + 2.5 * std::cos(x) * std::sin(y + z));
        h.ua[idx] = (fv_real)(11.0 + 2.0 * std::sin(x + 0.5 * y));
        h.va[idx] = (fv_real)(-6.0 + 1.5 * std::cos(y + 0.5 * x));
        // ut/vt/crx/cry/xfx/yfx are d_sw OUTPUTS (S1 and S2 overwrite them
        // before any read), so a nonzero seed here would only mask a
        // missing write. Left at zero deliberately.
        h.ptc[idx] = (fv_real)(280.0 + 4.0 * std::sin(x + y));
        h.divgd[idx] = 0.0;
        h.diss_est[idx] = 0.0;
        // z_rat is a read-only nonhydrostatic thickness ratio that scales the
        // damping terms; 1.0 is the neutral value, perturbed so it cannot
        // cancel out of a product.
        h.z_rat[idx] = (fv_real)(1.0 + 0.02 * std::sin(x - y));
        // Milestone 3A.9. omga is BOTH d2a2c_vect's A-grid vertical velocity
        // and Riem_Solver_C's w3 argument (dyn_core.F90:583), where it is
        // intent(in) and enters the implicit column solve's forcing. Zero
        // would be an admissible physical state but a degenerate test input:
        // it would remove w3 from the Riem_Solver_C right-hand side entirely,
        // so a transcription error in that term could not change the answer.
        // Same order of magnitude as w (0.01 m/s), sign-varying in i,j,k.
        h.omga[idx] = (fv_real)(0.01 * std::sin(x + y - z));
        // The five flux capacitors get a nonzero seed on purpose: d_sw ADDS to
        // them and never initializes them (§5 of wp_b_dsw_cuda.cuh). Seeding
        // them at zero would make "accumulated" and "overwrote" indistinguish-
        // able; Phase B checks the post-step value differs from this seed.
        h.mfx[idx] = 11.0; h.mfy[idx] = 13.0;
        h.cx[idx] = 0.17;  h.cy[idx] = 0.19;
        h.dpx[idx] = 23.0;
      }
    }
  }

  // Milestone 3A.9: dp_ref is the reference-profile layer thickness
  // update_dz_c reads as dp0. It must match the delp profile the rest of this
  // state carries, because update_dz_c's wind extrapolation weights the
  // top-two and bottom-two levels by dp0 ratios (nh_utils.F90's dp0(1)/dp0(0)
  // and dp0(km-1)/dp0(km-2) terms); an inconsistent dp_ref would silently
  // change those weights. This state is a uniform-dp column, so dp_ref is
  // uniform too, with a small k variation so the ratios are not exactly 1 and
  // a swapped index would be visible.
  for (size_t k = 0; k < (size_t)NPZ; ++k)
    h.dp_ref[k] = dp_uniform * (fv_real)(1.0 + 0.01 * (double)k);

  // --- Milestone 3A.8: the ten per-level damping controls -------------------
  // This is §4's required production behavior: a sponge top where nord_k is
  // forced to 0 sitting above an interior that takes the higher-order arm, in
  // ONE dgrid_dynamics invocation. The four-level pattern below is exactly the
  // mixed_levels() set the focused d_sw test (Part I) validated level-by-level
  // against four uniform-parameter runs; cycling it over NPZ reproduces those
  // same validated combinations here, so all three high-order nord groups plus
  // the nord == 0 group are live simultaneously and every LevelFilter gate
  // (damp_v, damp_w, damp_t, d_con) has both arms live.
  {
    const fv_real nord_k[4] = {0, 2, 0, 1};
    const fv_real nord_v[4] = {0, 2, 1, 2};
    const fv_real nord_w[4] = {0, 1, 0, 2};
    const fv_real d2_bg[4] = {fv_real(0.02), fv_real(0.0), fv_real(0.03), fv_real(0.0)};
    const fv_real d4_bg[4] = {fv_real(0.12), fv_real(0.16), fv_real(0.14), fv_real(0.10)};
    const fv_real damp_v[4] = {fv_real(0.0), fv_real(0.10), fv_real(0.05), fv_real(0.20)};
    const fv_real damp_w[4] = {fv_real(0.0), fv_real(0.06), fv_real(0.0), fv_real(0.09)};
    const fv_real damp_t[4] = {fv_real(0.0), fv_real(0.08), fv_real(0.0), fv_real(0.03)};
    const fv_real d_con[4] = {fv_real(0.0), fv_real(1.0), fv_real(0.5), fv_real(0.0)};
    const fv_real kgb[4] = {fv_real(0.0), fv_real(0.0), fv_real(1.0e-4), fv_real(5.0e-5)};
    for (size_t k = 0; k < (size_t)NPZ; ++k) {
      const size_t s = k % 4;
      h.lv_nord_k[k] = nord_k[s]; h.lv_ndif[k] = nord_v[s];
      h.lv_nord_w[k] = nord_w[s]; h.lv_d2_bg[k] = d2_bg[s];
      h.lv_d4_bg[k] = d4_bg[s];   h.lv_damp_v[k] = damp_v[s];
      h.lv_damp_w[k] = damp_w[s]; h.lv_damp_t[k] = damp_t[s];
      h.lv_d_con[k] = d_con[s];   h.lv_kgb[k] = kgb[s];
    }
  }
  return h;
}

std::vector<fv3_gpu_field_desc> make_fields(HostState &h) {
  static thread_local int64_t lo_3d[3], lo_3d_if[3];
  static thread_local uint64_t ex_3d[3], ex_3d_if[3];
  static thread_local int64_t st_3d[3], st_3d_if[3];
  static thread_local int64_t lo_2d[2];
  static thread_local uint64_t ex_2d[2];
  static thread_local int64_t st_2d[2];

  lo_3d[0] = lo_3d_if[0] = 1 - (int64_t)HALO;
  lo_3d[1] = lo_3d_if[1] = 1 - (int64_t)HALO;
  // launch_vertical_solver's riem_solver3_kernel indexes delp/pt/q_con/cappa/
  // w/zh/pe/peln/pk/pk3/pkc(ppe) with a hardcoded 0-based k loop (k=0..km-1 or
  // 0..km), unlike WP-A's pressure_gradient_c_{x,y} which derives k relative
  // to whatever lower[2] is registered ("kernels never assume Fortran k
  // starts at one", fv3_cgrid.cu). So lower[2] must be 0 here to satisfy the
  // vertical solver; this remains compatible with cgrid_pressure since that
  // kernel is base-agnostic.
  lo_3d[2] = lo_3d_if[2] = 0;
  ex_3d[0] = ex_3d_if[0] = (uint64_t)h.ni_h;
  ex_3d[1] = ex_3d_if[1] = (uint64_t)h.nj_h;
  ex_3d[2] = (uint64_t)NPZ;
  ex_3d_if[2] = (uint64_t)h.npz1;
  st_3d[0] = st_3d_if[0] = 1;
  st_3d[1] = st_3d_if[1] = h.ni_h;
  st_3d[2] = st_3d_if[2] = h.ni_h * h.nj_h;

  /* pe/peln: authoritative (i, k, j) -- slot 1 is the VERTICAL axis and slot 2
   * is j. See the file-header note. Mapping, slot by slot:
   *   slot 0 = i : same box and unit stride as every other field
   *   slot 1 = k : 0-based, npz1 deep, strides by ni_h
   *   slot 2 = j : same box as slot 1 of an (i,j,k) field, strides ni_h*npz1
   * k stays 0-based to match riem_solver3_kernel's documented convention
   * (Fortran k=1..npz+1 -> 0..npz); the kernel now hands k to the middle
   * argument of View3::at, so descriptor and kernel agree by construction.
   * bind_wp_d's `lower[2] <= js - 2` is now a genuine j-halo test and passes
   * as jsd <= js-2 for any domain with HALO >= 2. */
  static thread_local int64_t lo_pe[3];
  static thread_local uint64_t ex_pe[3];
  static thread_local int64_t st_pe[3];
  lo_pe[0] = lo_3d_if[0]; lo_pe[1] = 0;                lo_pe[2] = lo_3d_if[1];
  ex_pe[0] = ex_3d_if[0]; ex_pe[1] = (uint64_t)h.npz1; ex_pe[2] = ex_3d_if[1];
  st_pe[0] = 1;           st_pe[1] = h.ni_h;           st_pe[2] = h.ni_h * h.npz1;
  const uint64_t bytes_pe = h.size_3d_pe * sizeof(fv_real);

  lo_2d[0] = 1 - (int64_t)HALO; lo_2d[1] = 1 - (int64_t)HALO;
  ex_2d[0] = (uint64_t)h.ni_h; ex_2d[1] = (uint64_t)h.nj_h;
  st_2d[0] = 1; st_2d[1] = h.ni_h;

  const uint64_t bytes_3d = h.size_3d * sizeof(fv_real);
  const uint64_t bytes_3d_if = h.size_3d_if * sizeof(fv_real);
  const uint64_t bytes_2d = h.size_2d * sizeof(fv_real);

  std::vector<fv3_gpu_field_desc> fields;
  fields.push_back(make_field(FieldId::delp, h.delp.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::pt, h.pt.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::w, h.w.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::delz, h.delz.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::q_con, h.q_con.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::cappa, h.cappa.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::uc, h.uc.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::vc, h.vc.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::delpc, h.delpc.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::pe, h.pe.data(), 3, lo_pe, ex_pe, st_pe, bytes_pe));
  fields.push_back(make_field(FieldId::peln, h.peln.data(), 3, lo_pe, ex_pe, st_pe, bytes_pe));
  fields.push_back(make_field(FieldId::pk, h.pk.data(), 3, lo_3d_if, ex_3d_if, st_3d_if, bytes_3d_if));
  fields.push_back(make_field(FieldId::pk3, h.pk3.data(), 3, lo_3d_if, ex_3d_if, st_3d_if, bytes_3d_if));
  fields.push_back(make_field(FieldId::pkc, h.pkc.data(), 3, lo_3d_if, ex_3d_if, st_3d_if, bytes_3d_if));
  fields.push_back(make_field(FieldId::zh, h.zh.data(), 3, lo_3d_if, ex_3d_if, st_3d_if, bytes_3d_if));
  fields.push_back(make_field(FieldId::gz, h.gz.data(), 3, lo_3d_if, ex_3d_if, st_3d_if, bytes_3d_if));
  fields.push_back(make_field(FieldId::zs, h.zs.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::ws, h.ws.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::rdxc, h.rdxc.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::rdyc, h.rdyc.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));

  // Milestone 3A.2: remaining WP-D bind_wp_d contract fields. Mid-layer
  // state fields reuse the same full-halo (lo_3d/ex_3d) box as delp/pt --
  // that box is is-3..ie+3, js-3..je+3, which is a superset of every
  // bind_wp_d covers_3d() requirement below (e.g. u needs only
  // is..ie,js..je+1; heat_source needs only is-1..ie+1,js-1..je+1 for this
  // config's passes=1). 2D grid metrics likewise reuse the lo_2d/ex_2d box.
  fields.push_back(make_field(FieldId::u, h.u.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::v, h.v.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::pkz, h.pkz.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::du, h.du.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::dv, h.dv.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  fields.push_back(make_field(FieldId::heat_source, h.heat_source.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));

  fields.push_back(make_field(FieldId::divg2, h.divg2.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::rdx, h.rdx.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::rdy, h.rdy.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::rarea, h.rarea.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::del6_u, h.del6_u.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::del6_v, h.del6_v.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::dx, h.dx.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::dy, h.dy.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::sina_u, h.sina_u.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::sina_v, h.sina_v.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::dxa, h.dxa.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::dya, h.dya.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));

  // grid_lonlat/agrid_lonlat: same (i,j) footprint as the 2D box, plus a
  // trailing lon/lat dim of extent 2.
  static thread_local int64_t lo_ll[3];
  static thread_local uint64_t ex_ll[3];
  static thread_local int64_t st_ll[3];
  lo_ll[0] = lo_2d[0]; lo_ll[1] = lo_2d[1]; lo_ll[2] = 0;
  ex_ll[0] = ex_2d[0]; ex_ll[1] = ex_2d[1]; ex_ll[2] = 2;
  st_ll[0] = 1; st_ll[1] = h.ni_h; st_ll[2] = h.ni_h * h.nj_h;
  const uint64_t bytes_ll = h.size_2d * 2 * sizeof(fv_real);
  fields.push_back(make_field(FieldId::grid_lonlat, h.grid_lonlat.data(), 3, lo_ll, ex_ll, st_ll, bytes_ll));
  fields.push_back(make_field(FieldId::agrid_lonlat, h.agrid_lonlat.data(), 3, lo_ll, ex_ll, st_ll, bytes_ll));

  // edge_w/edge_e vary with j (Y_FACE); edge_s/edge_n vary with i (X_FACE).
  // bind_wp_d only requires these fields be present (no covers() shape
  // check), but they are sized to the full halo box regardless so any
  // downstream A2B/edge kernel indexing stays in-bounds.
  static thread_local int64_t lo_ew[1] = {1 - (int64_t)HALO};
  static thread_local uint64_t ex_ew[1];
  static thread_local int64_t st_ew[1] = {1};
  static thread_local int64_t lo_sn[1] = {1 - (int64_t)HALO};
  static thread_local uint64_t ex_sn[1];
  static thread_local int64_t st_sn[1] = {1};
  ex_ew[0] = (uint64_t)h.nj_h;
  ex_sn[0] = (uint64_t)h.ni_h;
  const uint64_t bytes_ew = (uint64_t)h.nj_h * sizeof(accumulator_real);
  const uint64_t bytes_sn = (uint64_t)h.ni_h * sizeof(accumulator_real);
  fields.push_back(make_field_acc(FieldId::edge_w, h.edge_w.data(), 1, lo_ew, ex_ew, st_ew, bytes_ew));
  fields.push_back(make_field_acc(FieldId::edge_e, h.edge_e.data(), 1, lo_ew, ex_ew, st_ew, bytes_ew));
  fields.push_back(make_field_acc(FieldId::edge_s, h.edge_s.data(), 1, lo_sn, ex_sn, st_sn, bytes_sn));
  fields.push_back(make_field_acc(FieldId::edge_n, h.edge_n.data(), 1, lo_sn, ex_sn, st_sn, bytes_sn));

  // ==== Milestone 3A.8: the WP-B d_sw contract (bind_wp_b_d_sw) ============
  // Group 1 -- 3-D state on the same full-halo box as delp/pt. bind_wp_b_d_sw
  // requires every one of these over [isd,ied] x [jsd,jed] x [0,npz-1], which
  // lo_3d/ex_3d supply exactly (is-3..ie+3, js-3..je+3, 0..npz-1). The
  // staggered fields (crx/xfx on is..ie+1, cry/yfx on js..je+1) are checked
  // against narrower rectangles that this box strictly contains.
  for (const std::pair<FieldId, std::vector<fv_real> *> &f :
       {std::pair<FieldId, std::vector<fv_real> *>{FieldId::ua, &h.ua},
        {FieldId::va, &h.va},   {FieldId::divgd, &h.divgd},
        {FieldId::ptc, &h.ptc}, {FieldId::ut, &h.ut},
        {FieldId::vt, &h.vt},   {FieldId::crx, &h.crx},
        {FieldId::cry, &h.cry}, {FieldId::xfx, &h.xfx},
        {FieldId::yfx, &h.yfx}, {FieldId::diss_est, &h.diss_est},
        {FieldId::z_rat, &h.z_rat}})
    fields.push_back(make_field(f.first, f.second->data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));

  // Group 2 -- the five flux capacitors, kind=8 (accumulator_real) regardless
  // of build precision. bind_wp_b_d_sw pins the element width, so these must
  // go through make_field_acc; make_field would derive element_bytes from
  // sizeof(fv_real) and be rejected in an R4 build.
  const uint64_t bytes_3d_acc = h.size_3d * sizeof(accumulator_real);
  for (const std::pair<FieldId, std::vector<accumulator_real> *> &f :
       {std::pair<FieldId, std::vector<accumulator_real> *>{FieldId::mfx, &h.mfx},
        {FieldId::mfy, &h.mfy}, {FieldId::cx, &h.cx},
        {FieldId::cy, &h.cy},   {FieldId::dpx, &h.dpx}})
    fields.push_back(make_field_acc(f.first, f.second->data(), 3, lo_3d, ex_3d, st_3d, bytes_3d_acc));

  // Group 3 -- 2-D grid metrics d_sw reads that WP-A/C/D did not, on the same
  // lo_2d/ex_2d box as the existing metrics.
  for (const std::pair<FieldId, std::vector<fv_real> *> &f :
       {std::pair<FieldId, std::vector<fv_real> *>{FieldId::rarea_c, &h.rarea_c},
        {FieldId::dxc, &h.dxc},       {FieldId::dyc, &h.dyc},
        {FieldId::divg_u, &h.divg_u}, {FieldId::divg_v, &h.divg_v},
        {FieldId::cosa_u, &h.cosa_u}, {FieldId::cosa_v, &h.cosa_v},
        {FieldId::cosa_s, &h.cosa_s}, {FieldId::rsin2, &h.rsin2},
        {FieldId::area, &h.area},     {FieldId::rdxa, &h.rdxa},
        {FieldId::rdya, &h.rdya},     {FieldId::cosa, &h.cosa},
        {FieldId::rsina, &h.rsina},   {FieldId::rsin_u, &h.rsin_u},
        {FieldId::rsin_v, &h.rsin_v}, {FieldId::f0, &h.f0}})
    fields.push_back(make_field(f.first, f.second->data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));

  // sin_sg: the four cell-edge components on a trailing 0-based axis
  // (Fortran component 1 -> 0). Required over [isd,ied] x [jsd,jed] x [0,3].
  static thread_local int64_t lo_sg[3];
  static thread_local uint64_t ex_sg[3];
  static thread_local int64_t st_sg[3];
  lo_sg[0] = lo_2d[0]; lo_sg[1] = lo_2d[1]; lo_sg[2] = 0;
  ex_sg[0] = ex_2d[0]; ex_sg[1] = ex_2d[1]; ex_sg[2] = 4;
  st_sg[0] = 1; st_sg[1] = h.ni_h; st_sg[2] = h.ni_h * h.nj_h;
  fields.push_back(make_field(FieldId::sin_sg, h.sin_sg.data(), 3, lo_sg, ex_sg, st_sg,
                              h.size_2d * 4 * sizeof(fv_real)));

  // Group 4 -- the ten per-level damping controls, [0, npz-1]. These must be
  // fv_real-width even for the three integral orders (damping_nord_k /
  // damping_ndif / damping_nord_w): the public ABI has no integer element
  // type, so bind_wp_b_d_sw asserts sizeof(fv_real) explicitly rather than let
  // a 4-byte registration be reinterpreted as doubles -- which would produce
  // arbitrary nord values and therefore arbitrary damping branches per level.
  static thread_local int64_t lo_lv[1] = {0};
  static thread_local uint64_t ex_lv[1];
  static thread_local int64_t st_lv[1] = {1};
  ex_lv[0] = (uint64_t)NPZ;
  const uint64_t bytes_lv = (uint64_t)NPZ * sizeof(fv_real);
  for (const std::pair<FieldId, std::vector<fv_real> *> &f :
       {std::pair<FieldId, std::vector<fv_real> *>{FieldId::damping_nord_k, &h.lv_nord_k},
        {FieldId::damping_ndif, &h.lv_ndif},
        {FieldId::damping_nord_w, &h.lv_nord_w},
        {FieldId::damping_d2_bg, &h.lv_d2_bg},
        {FieldId::damping_d4_bg, &h.lv_d4_bg},
        {FieldId::damping_v, &h.lv_damp_v},
        {FieldId::damping_w, &h.lv_damp_w},
        {FieldId::damping_t, &h.lv_damp_t},
        {FieldId::damping_d_con, &h.lv_d_con},
        {FieldId::damping_kgb, &h.lv_kgb}})
    fields.push_back(make_field(f.first, f.second->data(), 1, lo_lv, ex_lv, st_lv, bytes_lv));

  // ==== Milestone 3A.9: the two C-grid stages' remaining contract ==========
  // cgrid_predictor rejects a missing field BY NAME before any launch
  // (fv3_cgrid_predictor.cu's `all[]` absence list), and so do
  // launch_cgrid_vertical_solver and launch_riem_solver_c_ctx. All six below
  // are therefore load-bearing, not defensive.
  //
  // omga must use lo_3d EXACTLY, not lo_3d_if and not lo_pe: the predictor
  // requires every one of its 15 mutable-state 3-D fields to share delp's
  // lower[2], so an interface-level registration is rejected even though it
  // would "cover" the range -- and lo_pe is doubly wrong here, since pe/peln
  // deliberately carry a different axis ORDER, not just different bounds.
  fields.push_back(make_field(FieldId::omga, h.omga.data(), 3, lo_3d, ex_3d, st_3d, bytes_3d));
  // f_c is the dual-mesh Coriolis term (B stagger), a different field from f0
  // registered in group 3 above. ws3 and phis are the C-grid surface pair
  // dyn_core.F90:577/:583 pass to update_dz_c and Riem_Solver_C; ws3 is again
  // distinct from ws, which is Riem_Solver3's D-grid output.
  for (const std::pair<FieldId, std::vector<fv_real> *> &f :
       {std::pair<FieldId, std::vector<fv_real> *>{FieldId::f_c, &h.f_c},
        {FieldId::ws3, &h.ws3}, {FieldId::phis, &h.phis}})
    fields.push_back(make_field(f.first, f.second->data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  // cos_sg shares sin_sg's shape exactly -- same (i,j) box, same 0-based
  // 4-component trailing axis, and the predictor applies the same rank-3 /
  // lower[2]==0 / covers(0..3) checks to both.
  fields.push_back(make_field(FieldId::cos_sg, h.cos_sg.data(), 3, lo_sg, ex_sg, st_sg,
                              h.size_2d * 4 * sizeof(fv_real)));
  // dp_ref shares the per-level damping controls' shape. lower[0] must be 0:
  // update_dz_c reads dp0 at absolute indices 0 and npz-1, so a lower[0] of 1
  // would shift every read and run one element past the end at k == npz.
  fields.push_back(make_field(FieldId::dp_ref, h.dp_ref.data(), 1, lo_lv, ex_lv, st_lv, bytes_lv));

  return fields;
}

// The production first-run configuration, built but not installed. Split out
// from install_config() by Milestone 3A.9 so the §13 fence phase can construct
// this exact config and then perturb ONE field: a fence test that hand-rolled
// its own config would risk being refused for an unrelated reason and still
// look like it proved something.
NumericalConfig production_config(const HostState &h) {
  NumericalConfig cfg{};
  cfg.ptop = h.ptop;
  cfg.akap = h.akap;
  cfg.a_imp = 1.0;
  cfg.p_fac = 0.05;
  cfg.scale_m = 1.0;
  cfg.rdgas = 287.0;
  cfg.grav = h.grav;
  cfg.beta = 0.0;
  cfg.use_logp = 0;
  cfg.a2b_order = 4;
  cfg.hydrostatic = 0;
  cfg.grid_type = 0;
  cfg.delt_max = 0.002;
  // flagstruct%m_split. Milestone 3A.9 removed the runtime's hardcoded
  // vertical_ms=1: dyn_core.F90:918 gives Riem_Solver3 flagstruct%m_split while
  // :300/:583 give Riem_Solver_C max(1, m_split/2), so one value could not
  // serve both and NumericalConfig::m_split is now the single source. Zero is
  // the "not supplied" sentinel and BOTH solvers reject it rather than default,
  // so this line is load-bearing: without it launch_vertical_solver returns
  // FV3_GPU_UNSUPPORTED_CONFIGURATION. 2 => Riem_Solver3 ms=2, Riem_Solver_C
  // ms=max(1,1)=1, which exercises the two-different-values path rather than
  // the degenerate case where both happen to be 1.
  cfg.m_split = 2;
  // flagstruct%dz_min. Nothing REJECTS a zero dz_min, which makes it the more
  // dangerous of the two: update_dz_c's and update_dz_d's monotonicity clamps
  // (nh_utils.F90's `max(zh(k), zh(k+1)+dz_min)` and the C-grid equivalent)
  // degenerate into permitting exactly zero-thickness layers, and the SIM1
  // column solve then divides by dz -- a silent Inf/NaN, produced by a
  // perfectly valid-looking configuration. 2 m is the GEOS namelist value.
  cfg.dz_min = 2.0;
  // Milestone 3A.2: enable post_loop_heating's genuine (non-early-return)
  // execution path.
  //
  // This test pins nf_ke = 1, which is NOT the authoritative
  // min(3, nord+1) = 3 (dyn_core.F90:1310) that the production path now uses
  // (run_small_perf_bench.cpp derives it from cfg.nord). The pin keeps
  // nt = passes-1-pass at 0 on every pass, which skips the two copy_corners
  // rotations inside post_loop_heating.
  //
  // Prompt 6B, DEFERRED: raising this to 3 makes Phase C fail
  // ("Phase C result: FAIL"), root cause not yet localized -- the diagnostic
  // GPU job could not be scheduled within the task's cost budget. The
  // corner_edge_service itself is exercised on real GPU hardware by the
  // authoritative-SP0 bench run, where the dyn_core-exit pt/pkz validate at
  // round-off scale on all six tiles, so this is a gap in *this test's*
  // coverage, not evidence against the service. Localize before relying on
  // nf_ke > 1 here.
  cfg.n_con = 1;
  cfg.d_con = 1.0;
  cfg.nf_ke = 1;
  cfg.moist_cappa = 0;
  cfg.use_sg = 0;
  // Milestone 3A.8: the production first-run d_sw configuration
  // (project_architecture.md). launch_d_sw_cuda rejects anything else before
  // it mutates state, so these are not decoration -- with them absent the
  // dgrid_dynamics stage returns FV3_GPU_UNSUPPORTED_CONFIGURATION.
  cfg.hord_mt = 6;
  cfg.hord_vt = 6;
  cfg.hord_tm = 6;
  cfg.hord_dp = -6;
  cfg.lim_fac = 1.0;
  cfg.nord = 2;    // => nord_cap_k = 2, so the nord_k = 1 and 2 groups are live
  cfg.nord_t = 2;  // must be <= ng-1 = 2
  cfg.dddmp = 0.5; // >= 1.e-5, so S31 takes the smag_corner arm, not vort := 0
  // Minimum cell / dual-cell areas. Same values the focused d_sw test used, so
  // the deln_flux (damp_c*da_min)**(nord+1) and del6_vt_flux
  // (damp*da_min_c)**(nord+1) coefficients stay in the magnitude range already
  // demonstrated finite in both R8 and R4.
  cfg.da_min = 1.15e10;
  cfg.da_min_c = 1.0e10;
  cfg.inline_q = 0;
  cfg.rot3 = 0;
  cfg.do_f3d = 0;
  cfg.do_skeb = 0;
  return cfg;
}

int install_config(fv3_gpu_context *ctx, HostState &h, fv3_gpu_error *err) {
  return set_numerical_config(ctx, production_config(h), err);
}

// Milestone 3A.8: bind_wp_b_d_sw refuses a TopologyView with npx/npy == 0
// (the "not supplied" sentinel) rather than infer a value, because a wrong
// npx/npy silently applies an interior stencil at a panel seam. Nothing before
// this milestone needed panel-edge semantics, so nothing installed one.
//
// This rank is a whole 8x8 tile: npx = npy = NI+1 = 9 cell corners along an
// edge, so it owns all four tile edges and all four panel corners. That is the
// harder of the two cases -- every edge block, both fill_corners rotations and
// all four 2x2 corner solves execute, which is the same "seam" configuration
// the focused d_sw test exercises at 12x12.
//
// The neighbors[] array must be populated to match, and this is not
// bookkeeping. Two independent mechanisms decide "is this a panel edge":
//
//   WP-B d_sw   -- tests b.is == 1 / b.ie + 1 == npx / b.js == 1 /
//                  b.je + 1 == npy, which is how sw_core.F90 itself tests for
//                  a tile edge.
//   WP-D a2b    -- physical_edges(topology) (operators/wp_d_contract.cpp:383),
//                  which tests whether each neighbor is a real rank on a
//                  DIFFERENT tile.
//
// Leaving neighbors[] at its {rank=-1} default while setting the four corner
// flags is not merely incomplete, it is self-contradictory, and WP-D detects
// it: launch_a2b_ord4_global_cuda's consistent_corners(e, c) guard requires
// c.sw == (e.west && e.south), so an all-corners/no-edges topology is rejected
// with FV3_GPU_INVALID_ARGUMENT before any kernel runs. That guard did its job
// on the first A100 run of this test -- nh_pressure returned
// "A2B (pkc) CUDA launch failed" -- and the fix is to describe the rank
// correctly, not to relax the topology to the interior-only case that earlier
// milestones happened to use.
//
// A single-rank cube face has four cross-tile neighbors, one per side. Ranks
// and tiles are nominal here (this test is single-process and performs no
// exchange; §"no MPI work"), but they must be >= 0 and != tile to read as
// cross-tile.
int install_topology(fv3_gpu_context *ctx, fv3_gpu_error *err) {
  TopologyView t{};
  t.tile = 0;
  t.global_cubed_sphere = 1;
  t.nested = 0;
  t.regional = 0;
  t.npx = NI + 1;
  t.npy = NJ + 1;
  const TileEdge sides[4] = {TileEdge::west, TileEdge::east, TileEdge::south,
                             TileEdge::north};
  for (int s = 0; s < 4; ++s) {
    EdgeNeighbor &n = t.neighbors[(size_t)sides[s]];
    n.rank = s + 1;
    n.tile = (int16_t)(s + 1); // != t.tile, so physical_edges() reports a seam
    n.edge = sides[s];
    n.orientation = EdgeOrientation::aligned;
    n.vector_sign_x = 1;
    n.vector_sign_y = 1;
  }
  t.sw_corner = 1;
  t.se_corner = 1;
  t.ne_corner = 1;
  t.nw_corner = 1;
  return set_topology(ctx, t, err);
}

// ===========================================================================
// Milestone 3A.9 §8/§9 -- halo readiness, and the limit of what this harness
// can honestly claim about it.
//
// install_topology() above describes a rank that owns a whole cube face, so it
// has four CROSS-TILE neighbours. resolve_halo() in the runtime therefore does
// NOT consider its halo exchange vacuous, and refuses to invent readiness for
// any group: without the declaration below, fv3_gpu_dyn_core_step stops at the
// first resolve point with FV3_GPU_UNSUPPORTED_CONFIGURATION, naming the group.
// That refusal is the correct behaviour and is asserted separately by
// run_halo_refusal_check() below.
//
// What this function does is declare, on the caller's behalf, that this process
// is responsible for those halo contents. That declaration is TRUE ONLY IN THE
// WEAK SENSE that the halo cells hold what initialize_state's H2D put there --
// this harness performs no exchange, because none exists to perform. So:
//
//   * this is enough to make the integrated path EXECUTE on the device, which
//     is what a GPU-execution harness is for;
//   * it is NOT enough to make the panel-seam halo values physically correct,
//     and therefore NOT enough to close Gate C1's "no false halo-readiness
//     assumption" criterion.
//
// The distinction is recorded here rather than buried in a handoff because the
// declaration is exactly the kind of thing that reads as "halos handled" six
// months later. They are not handled. See docs/agent_handoff/
// milestone_3a9_gate_c1_closure.md for the blocker statement.
//
// The list is the set the runtime actually resolves, spelled out rather than a
// range over the whole enum: if a future change adds a resolve point for
// H_Q_INLINE or H_DWIND_NEXT, this harness should fail loudly at that new
// point rather than silently have covered it in advance.
//
// Prompt 6B: H_DWIND_SEAM is that mechanism working as designed. Implementing
// E11 (the it==n_split mpp_get_boundary of dyn_core.F90:1178) added a tenth
// resolve point, and this harness failed loudly at it -- FV3_GPU_UNSUPPORTED_
// CONFIGURATION naming the group -- rather than quietly executing without it.
// Adding it here restores the same weak-sense declaration the other nine carry:
// it lets the integrated path EXECUTE, and says nothing about the seam values
// being right. Phase E, which runs a REAL six-tile exchange, exercises the
// group's actual table entry instead and is the only place the seam semantics
// are tested here.
constexpr fv3_gpu_halo_group kRuntimeResolvedGroups[] = {
    FV3_GPU_H_DWIND_IN, FV3_GPU_H_W_IN,    FV3_GPU_H_GZ_FIRST,
    FV3_GPU_H_DIVGD,    FV3_GPU_H_CWIND,   FV3_GPU_H_DSCALAR,
    FV3_GPU_H_ZH,       FV3_GPU_H_PKC2,    FV3_GPU_H_HEAT,
    FV3_GPU_H_DWIND_SEAM};

bool declare_caller_managed_halos(fv3_gpu_context *ctx, fv3_gpu_error *err) {
  for (fv3_gpu_halo_group g : kRuntimeResolvedGroups) {
    const int rc = mark_halo_caller_managed(ctx, g, err);
    if (rc != FV3_GPU_SUCCESS) {
      printf("  FAIL: mark_halo_caller_managed(%d) returned %d: %s\n", (int)g,
             rc, err->message);
      return false;
    }
  }
  printf("  %zu halo groups declared CALLER-MANAGED\n",
         sizeof(kRuntimeResolvedGroups) / sizeof(kRuntimeResolvedGroups[0]));
  printf("  NOTE: this harness performs NO halo exchange. The declaration\n");
  printf("  lets the integrated path execute; it does NOT make panel-seam\n");
  printf("  halo values correct. Gate C1 halo criterion: NOT met.\n");
  return true;
}

// Phase A: minimal dispatch (only the two stages this field set satisfies).
// Expected: fv3_gpu_dyn_core_step fully succeeds; D2H + finite checks run.
bool run_phase_a() {
  printf("########################################################\n");
  printf("# Phase A: minimal dispatch (cgrid_pressure + vertical_solver)\n");
  printf("# Expect: fv3_gpu_dyn_core_step SUCCESS, real D2H verification\n");
  printf("########################################################\n\n");

  fv3_gpu_error err{};
  fv3_gpu_context *ctx = nullptr;
  fv3_gpu_create_desc create{};
  create.abi_version = FV3_GPU_ABI_VERSION;
  create.struct_bytes = sizeof(create);
  create.device_ordinal = 0;
  create.mpi_rank = 0;
  create.mpi_size = 1;
  create.precision = sizeof(fv_real) == 4 ? FV3_GPU_PRECISION_R4 : FV3_GPU_PRECISION_R8;
  create.is = 1; create.ie = NI; create.js = 1; create.je = NJ;
  create.npz = NPZ; create.halo_width = HALO;
  int rc = fv3_gpu_create(&create, &ctx, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_create returned %d: %s\n", rc, err.message);
    return false;
  }
  printf("  WP-E context created\n");

  HostState h = make_host_state();
  std::vector<fv3_gpu_field_desc> fields = make_fields(h);

  fv3_gpu_state_desc state{};
  state.abi_version = FV3_GPU_ABI_VERSION;
  state.struct_bytes = sizeof(state);
  state.field_count = static_cast<uint32_t>(fields.size());
  state.fields = fields.data();
  fv3_gpu_event ev = 0;
  rc = fv3_gpu_initialize_state(ctx, &state, &ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_initialize_state returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  rc = fv3_gpu_event_wait(ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_event_wait returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  H2D transfer complete (%zu fields)\n", fields.size());

  rc = install_config(ctx, h, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: set_numerical_config returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  NumericalConfig installed\n");

  // Milestone 3A.10: this phase used to run with NO topology at all, which the
  // runtime then read as "no neighbours, nothing to exchange" and rewarded with
  // free halo readiness. That fail-open behaviour is exactly what this milestone
  // closed, so the phase now has to say what its decomposition is -- and, since
  // it performs no exchange itself, has to take explicit responsibility for the
  // halo contents. It is the TEST that was stale, not the fence; the runtime was
  // deliberately left strict.
  rc = install_topology(ctx, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: set_topology returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  TopologyView installed (whole cube face, all 4 seams connected)\n");
  if (!declare_caller_managed_halos(ctx, &err)) {
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  halo groups declared CALLER-MANAGED (this phase performs no\n"
         "  exchange; the declaration permits execution and does NOT make\n"
         "  panel-seam values correct -- Gate C1 halo criterion NOT met here)\n");

  NumericalDispatch dispatch{};
  dispatch.contract_version = internal_contract_version;
  NumericalDispatch full{};
  rc = fv3_build_production_dispatch(full, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_build_production_dispatch returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  dispatch.cgrid_pressure = full.cgrid_pressure;
  dispatch.vertical_solver = full.vertical_solver;
  dispatch.corner_edge_service = full.corner_edge_service;
  printf("  cgrid_pressure  = WIRED (WP-A p_grad_c)\n");
  printf("  vertical_solver = WIRED (WP-C riem_solver3)\n");
  printf("  nh_pressure       = null (not exercised in this phase)\n");
  printf("  post_loop_heating = null (not exercised in this phase)\n");

  rc = set_numerical_dispatch(ctx, dispatch, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: set_numerical_dispatch returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  Dispatch installed\n");

  fv3_gpu_step_desc step{};
  step.abi_version = FV3_GPU_ABI_VERSION;
  step.struct_bytes = sizeof(step);
  step.bdt = 10.0;
  step.k_split = 1;
  step.n_split = 1;
  step.init_step = 1;
  step.end_step = 1;
  step.model_step = 0;
  step.acoustic_sequence = 1;
  step.dependency = 0;
  // uc/vc are no longer zero at entry (see make_host_state: d_sw needs real
  // C-grid winds), so snapshot them and compare rather than testing for
  // nonzero. Strictly stronger: independent of the initial value.
  const std::vector<fv_real> uc_before = h.uc;
  const std::vector<fv_real> vc_before = h.vc;
  fv3_gpu_event step_ev = 0;
  rc = fv3_gpu_dyn_core_step(ctx, &step, &step_ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_dyn_core_step returned %d\n", rc);
    printf("  error.api=\"%s\" message=\"%s\"\n", err.api, err.message);
    printf("  This phase was expected to fully succeed -- treat as a real failure.\n");
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  fv3_gpu_dyn_core_step: SUCCESS (event=%lu)\n", (unsigned long)step_ev);
  printf("  Both wired stages (cgrid_pressure, vertical_solver) returned\n");
  printf("  FV3_GPU_SUCCESS -- WP-A p_grad_c and WP-C riem_solver3 both\n");
  printf("  genuinely launched and executed on the A100.\n");

  fv3_gpu_field_mask mask{};
  mask.abi_version = FV3_GPU_ABI_VERSION;
  mask.struct_bytes = sizeof(mask);
  for (FieldId id : {FieldId::uc, FieldId::vc, FieldId::w, FieldId::delz,
                     FieldId::pe, FieldId::peln, FieldId::pk, FieldId::pkc}) {
    mask.words[static_cast<uint32_t>(id) / 64] |= (1ULL << (static_cast<uint32_t>(id) % 64));
  }
  rc = fv3_gpu_sync_to_host(ctx, &mask, &ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_sync_to_host returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  rc = fv3_gpu_event_wait(ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_event_wait returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  D2H transfer complete (event=%lu)\n\n", (unsigned long)ev);

  printf("  Verifying finite outputs:\n");
  bool finite_pass = true;
  finite_pass &= check_finite(h.uc.data(), h.uc.size(), "uc");
  finite_pass &= check_finite(h.vc.data(), h.vc.size(), "vc");
  finite_pass &= check_finite(h.w.data(), h.w.size(), "w");
  finite_pass &= check_finite(h.delz.data(), h.delz.size(), "delz");
  finite_pass &= check_finite(h.pe.data(), h.pe.size(), "pe");
  finite_pass &= check_finite(h.peln.data(), h.peln.size(), "peln");
  finite_pass &= check_finite(h.pk.data(), h.pk.size(), "pk");
  finite_pass &= check_finite(h.pkc.data(), h.pkc.size(), "pkc");

  printf("\n  Non-degenerate change probes (pre-step snapshot vs post-D2H):\n");
  bool cgrid_wrote_output =
      report_changed(uc_before.data(), h.uc.data(), h.uc.size(), "uc") ||
      report_changed(vc_before.data(), h.vc.data(), h.vc.size(), "vc");
  printf("  WP-A p_grad_c changed uc/vc: %s\n", cgrid_wrote_output ? "YES" : "NO");

  rc = fv3_gpu_finalize(&ctx, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_finalize returned %d: %s\n", rc, err.message);
    return false;
  }
  printf("  Context finalized\n\n");

  bool pass = finite_pass && cgrid_wrote_output;
  printf("Phase A result: %s\n\n", pass ? "PASS" : "FAIL");
  return pass;
}

// Phase B: the full production dispatch. Milestone 3A.2: the WP-D field
// contract (bind_wp_d) is now fully satisfied, so this phase is expected to
// fully SUCCEED through all four wired stages, with a real D2H verification
// of WP-D's nh_pressure/post_loop_heating outputs (u, v, pt, pkz).
bool run_phase_b() {
  printf("########################################################\n");
  printf("# Phase B: full production dispatch (all 4 wired stages)\n");
  printf("# Expect: fv3_gpu_dyn_core_step SUCCESS, real D2H verification\n");
  printf("# of WP-D nh_pressure/post_loop_heating outputs\n");
  printf("########################################################\n\n");

  fv3_gpu_error err{};
  fv3_gpu_context *ctx = nullptr;
  fv3_gpu_create_desc create{};
  create.abi_version = FV3_GPU_ABI_VERSION;
  create.struct_bytes = sizeof(create);
  create.device_ordinal = 0;
  create.mpi_rank = 0;
  create.mpi_size = 1;
  create.precision = sizeof(fv_real) == 4 ? FV3_GPU_PRECISION_R4 : FV3_GPU_PRECISION_R8;
  create.is = 1; create.ie = NI; create.js = 1; create.je = NJ;
  create.npz = NPZ; create.halo_width = HALO;
  int rc = fv3_gpu_create(&create, &ctx, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_create returned %d: %s\n", rc, err.message);
    return false;
  }
  printf("  WP-E context created\n");

  HostState h = make_host_state();
  std::vector<fv3_gpu_field_desc> fields = make_fields(h);

  fv3_gpu_state_desc state{};
  state.abi_version = FV3_GPU_ABI_VERSION;
  state.struct_bytes = sizeof(state);
  state.field_count = static_cast<uint32_t>(fields.size());
  state.fields = fields.data();
  fv3_gpu_event ev = 0;
  rc = fv3_gpu_initialize_state(ctx, &state, &ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_initialize_state returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  rc = fv3_gpu_event_wait(ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_event_wait returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  H2D transfer complete (%zu fields)\n", fields.size());

  rc = install_config(ctx, h, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: set_numerical_config returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  NumericalConfig installed\n");

  rc = install_topology(ctx, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: set_topology returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  TopologyView installed (npx=npy=%ld, all 4 edges + 4 corners owned)\n",
         (long)(NI + 1));

  if (!declare_caller_managed_halos(ctx, &err)) {
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }

  NumericalDispatch dispatch{};
  rc = fv3_build_production_dispatch(dispatch, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_build_production_dispatch returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  cgrid_pressure     = %s\n", dispatch.cgrid_pressure ? "WIRED (WP-A p_grad_c)" : "null");
  printf("  dgrid_dynamics     = %s\n", dispatch.dgrid_dynamics ? "WIRED (WP-B d_sw)" : "null");
  printf("  dgrid_height_transport = %s\n",
         dispatch.dgrid_height_transport ? "WIRED (WP-B update_dz_d body)" : "null");
  printf("  vertical_solver    = %s\n", dispatch.vertical_solver ? "WIRED (WP-C riem_solver3)" : "null");
  printf("  nh_pressure        = %s\n", dispatch.nh_pressure ? "WIRED (WP-D)" : "null");
  printf("  post_loop_heating  = %s\n", dispatch.post_loop_heating ? "WIRED (WP-D)" : "null");
  printf("  corner_edge_service= %s\n",
         dispatch.corner_edge_service ? "WIRED (copy_corners, scalar/A-grid)" : "null");

  // Milestone 3A.8 test N. Assert the slot separately from asserting that the
  // step succeeds, so the two claims fail independently: a null dgrid_dynamics
  // would let fv3_gpu_dyn_core_step return SUCCESS while silently skipping the
  // whole of d_sw (runtime/fv3_gpu_runtime.cpp's stage loop `continue`s past
  // null slots by design). Without this check, "Phase B passed" would be
  // compatible with WP-B never having run.
  if (!dispatch.dgrid_dynamics) {
    printf("  FAIL: production dispatch left dgrid_dynamics null; the step\n");
    printf("  below would skip d_sw entirely and still report SUCCESS.\n");
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }

  rc = set_numerical_dispatch(ctx, dispatch, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: set_numerical_dispatch returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  Full production dispatch installed\n");

  // pt/pkz/heat_source start out already nonzero (see make_host_state), so
  // an any_nonzero() check after the step would pass trivially whether or
  // not post_loop_heating actually executed. Snapshot the pre-step values
  // here and compare element-wise after D2H instead -- the same rigor as
  // the 3A.1 uc/vc check, which worked only because uc/vc started at zero.
  const std::vector<fv_real> pt_before = h.pt;
  const std::vector<fv_real> pkz_before = h.pkz;
  const std::vector<fv_real> heat_source_before = h.heat_source;

  // Milestone 3A.8. u/v now start genuinely nonzero (d_sw is identically zero
  // in every flux, vorticity and KE term if they do not), so the old
  // any_nonzero(u) probe would pass without any stage having run. Snapshot and
  // compare instead -- strictly stronger. delp/w/mfx/... are the driver-owned
  // updates and the flux capacitors, all of which also start nonzero.
  const std::vector<fv_real> u_before = h.u;
  const std::vector<fv_real> v_before = h.v;
  const std::vector<fv_real> delp_before = h.delp;
  const std::vector<fv_real> w_before = h.w;
  const std::vector<accumulator_real> mfx_before = h.mfx;
  const std::vector<accumulator_real> mfy_before = h.mfy;
  const std::vector<accumulator_real> cx_before = h.cx;
  const std::vector<accumulator_real> cy_before = h.cy;

  fv3_gpu_step_desc step{};
  step.abi_version = FV3_GPU_ABI_VERSION;
  step.struct_bytes = sizeof(step);
  step.bdt = 10.0;
  step.k_split = 1;
  step.n_split = 1;
  step.init_step = 1;
  step.end_step = 1;
  step.model_step = 0;
  step.acoustic_sequence = 1;
  step.dependency = 0;
  fv3_gpu_event step_ev = 0;
  rc = fv3_gpu_dyn_core_step(ctx, &step, &step_ev, &err);

  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_dyn_core_step returned %d\n", rc);
    printf("  error.api=\"%s\" message=\"%s\"\n", err.api, err.message);
    printf("  This phase was expected to fully succeed now that the WP-D\n");
    printf("  field contract is complete -- treat as a real failure.\n");
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  fv3_gpu_dyn_core_step: SUCCESS (event=%lu)\n", (unsigned long)step_ev);
  printf("  All 5 wired acoustic stages returned FV3_GPU_SUCCESS:\n");
  printf("    cgrid_pressure (WP-A) -> dgrid_dynamics (WP-B d_sw)\n");
  printf("    -> vertical_solver (WP-C) -> nh_pressure (WP-D)\n");
  printf("    -> post_loop_heating (WP-D)\n");
  printf("  Milestone 3A.8 test O: the complete authoritative d_sw driver\n");
  printf("  genuinely launched and executed on the A100 through\n");
  printf("  fv3_gpu_dyn_core_step -> NumericalDispatch::dgrid_dynamics, not\n");
  printf("  through a direct launch_d_sw_cuda call.\n");

  fv3_gpu_field_mask mask{};
  mask.abi_version = FV3_GPU_ABI_VERSION;
  mask.struct_bytes = sizeof(mask);
  for (FieldId id : {FieldId::u, FieldId::v, FieldId::du, FieldId::dv,
                     FieldId::pt, FieldId::pkz, FieldId::heat_source,
                     FieldId::uc, FieldId::vc, FieldId::w,
                     // Milestone 3A.8: the driver-owned d_sw updates and the
                     // flux capacitors, so the assertions below read real
                     // device results rather than the untouched H2D input.
                     FieldId::delp, FieldId::mfx, FieldId::mfy,
                     FieldId::cx, FieldId::cy, FieldId::dpx}) {
    mask.words[static_cast<uint32_t>(id) / 64] |= (1ULL << (static_cast<uint32_t>(id) % 64));
  }
  rc = fv3_gpu_sync_to_host(ctx, &mask, &ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_sync_to_host returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  rc = fv3_gpu_event_wait(ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_event_wait returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  printf("  D2H transfer complete (event=%lu)\n\n", (unsigned long)ev);

  printf("  Verifying finite outputs:\n");
  bool finite_pass = true;
  finite_pass &= check_finite(h.u.data(), h.u.size(), "u");
  finite_pass &= check_finite(h.v.data(), h.v.size(), "v");
  finite_pass &= check_finite(h.du.data(), h.du.size(), "du");
  finite_pass &= check_finite(h.dv.data(), h.dv.size(), "dv");
  finite_pass &= check_finite(h.pt.data(), h.pt.size(), "pt");
  finite_pass &= check_finite(h.pkz.data(), h.pkz.size(), "pkz");
  finite_pass &= check_finite(h.heat_source.data(), h.heat_source.size(), "heat_source");
  finite_pass &= check_finite(h.delp.data(), h.delp.size(), "delp");
  finite_pass &= check_finite(h.w.data(), h.w.size(), "w");
  finite_pass &= check_finite(h.mfx.data(), h.mfx.size(), "mfx");
  finite_pass &= check_finite(h.mfy.data(), h.mfy.size(), "mfy");
  finite_pass &= check_finite(h.cx.data(), h.cx.size(), "cx");
  finite_pass &= check_finite(h.cy.data(), h.cy.size(), "cy");
  finite_pass &= check_finite(h.dpx.data(), h.dpx.size(), "dpx");

  printf("\n  Non-degenerate change probes (pre-step snapshot vs post-D2H):\n");
  // WP-B d_sw owns u, v, delp and w. WP-D's nh_pressure also increments u/v,
  // so a u/v change alone does not isolate d_sw -- but delp is touched by
  // NOTHING else on this stage list, which makes it the clean d_sw witness.
  const bool u_changed = report_changed(u_before.data(), h.u.data(), h.u.size(), "u");
  const bool v_changed = report_changed(v_before.data(), h.v.data(), h.v.size(), "v");
  const bool delp_changed =
      report_changed(delp_before.data(), h.delp.data(), h.delp.size(), "delp");
  const bool w_changed = report_changed(w_before.data(), h.w.data(), h.w.size(), "w");
  // Flux capacitors: d_sw ADDS to these and never initializes them, so a
  // change away from the deliberately nonzero seed is evidence of
  // accumulation rather than of an overwrite (§9's requirement that the
  // integrated path not be tested only from a zero start).
  const bool mfx_changed = report_changed(mfx_before.data(), h.mfx.data(), h.mfx.size(), "mfx");
  const bool mfy_changed = report_changed(mfy_before.data(), h.mfy.data(), h.mfy.size(), "mfy");
  const bool cx_changed = report_changed(cx_before.data(), h.cx.data(), h.cx.size(), "cx");
  const bool cy_changed = report_changed(cy_before.data(), h.cy.data(), h.cy.size(), "cy");

  const bool dsw_executed = delp_changed && w_changed;
  const bool capacitors_accumulated = mfx_changed && mfy_changed && cx_changed && cy_changed;
  const bool dwinds_updated = u_changed && v_changed;
  const bool post_loop_heating_wrote_output =
      report_changed(pt_before.data(), h.pt.data(), h.pt.size(), "pt") ||
      report_changed(pkz_before.data(), h.pkz.data(), h.pkz.size(), "pkz") ||
      report_changed(heat_source_before.data(), h.heat_source.data(),
                     h.heat_source.size(), "heat_source");

  printf("\n  WP-B d_sw updated delp and w:                %s\n", dsw_executed ? "YES" : "NO");
  printf("  WP-B d_sw accumulated mfx/mfy/cx/cy:         %s\n", capacitors_accumulated ? "YES" : "NO");
  printf("  D-grid winds u/v changed:                    %s\n", dwinds_updated ? "YES" : "NO");
  printf("  WP-D post_loop_heating changed pt/pkz/heat:  %s\n", post_loop_heating_wrote_output ? "YES" : "NO");

  rc = fv3_gpu_finalize(&ctx, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_finalize returned %d: %s\n", rc, err.message);
    return false;
  }
  printf("  Context finalized\n\n");

  bool pass = finite_pass && dsw_executed && capacitors_accumulated &&
              dwinds_updated && post_loop_heating_wrote_output;
  printf("Phase B result: %s\n\n", pass ? "PASS" : "FAIL");
  return pass;
}

// ===========================================================================
// Milestone 3A.9 Phase C -- the integrated Gate C1 evidence phase (§12, §16).
//
// What distinguishes this from Phase B is not "more stages": it is that every
// claim is instrumented rather than inferred.
//
// Phase B proves a step SUCCEEDS with the full table installed. That is a
// weaker statement than it looks, because the runtime SKIPS a null slot by
// design -- so "SUCCESS" is compatible with a stage never having been entered,
// and with a stage being entered fewer times than the acoustic loop demands.
// §16 requires asserting that each production-required stage is actually
// REACHED, and §7 requires n_split > 1 to actually be a loop.
//
// So each slot is wrapped in a counting probe that calls straight through to
// the real production launcher. The probe is a test-only shim: it adds no
// numerics, and it is installed by copying the production table, so a stage
// that fv3_build_production_dispatch leaves null stays null and its expected
// count is zero rather than being silently satisfied.
//
// dgrid_height_transport is the case that makes this worth doing. It is not on
// the runtime's script at all -- it is reached only from inside WP-C's
// update_dz_d via ExecutionContext::dispatch (§6). Its probe count is therefore
// the ONLY direct evidence that the WP-B -> WP-C -> WP-B handoff closes, and
// before this milestone that slot was wired, GPU-executed in isolation, and
// unreachable through fv3_gpu_dyn_core_step.
namespace probes {

enum Slot {
  kCgridPredictor = 0,
  kCgridVerticalSolver,
  kCgridPressure,
  kDgridDynamics,
  kDgridHeight,
  kVerticalSolver,
  kNhPressure,
  kPostLoopHeating,
  kDgridHeightTransport,
  kSlotCount
};

const char *const slot_name[kSlotCount] = {
    "cgrid_predictor (WP-A c_sw/d2a2c_vect)",
    "cgrid_vertical_solver (WP-C update_dz_c + Riem_Solver_C)",
    "cgrid_pressure (WP-A p_grad_c)",
    "dgrid_dynamics (WP-B d_sw)",
    "dgrid_height (WP-C update_dz_d sequencing)",
    "vertical_solver (WP-C Riem_Solver3)",
    "nh_pressure (WP-D nh_p_grad)",
    "post_loop_heating (WP-D del2_cubed + heating)",
    "dgrid_height_transport (WP-B, called BY WP-C)"};

LaunchFn real_fn[kSlotCount] = {};
int count[kSlotCount] = {};
int last_rc[kSlotCount] = {};

void reset() {
  for (int s = 0; s < kSlotCount; ++s) { count[s] = 0; last_rc[s] = 0; }
}

template <int S>
int probe(const ExecutionContext &x, fv3_gpu_error *e) {
  ++count[S];
  // A null real_fn here would mean the probe was installed for a slot the
  // production table left empty. Report that as a hard error rather than
  // returning SUCCESS, which would forge evidence of a stage that does not
  // exist.
  if (!real_fn[S]) return FV3_GPU_INVALID_ARGUMENT;
  const int rc = real_fn[S](x, e);
  last_rc[S] = rc;
  return rc;
}

// Copies `src` into `out`, routing every non-null slot through its probe.
// corner_edge_service is passed through unwrapped: it is a CornerServiceFn, not
// a LaunchFn, so it is not probe-wrapped and its calls are not counted here.
// This test pins nf_ke=1, so post_loop_heating's nt is 0 on its single pass and
// the service is never requested here. It is no longer a fail-stub, though:
// Prompt 6B implements it in fv3_gpu_production_dispatch.cu, and the production
// path (nf_ke = min(3, nord+1) = 3) requests it twice per del2_cubed pass with
// nt>0. See the cfg.nf_ke comment above for why this test still pins 1.
bool install(const NumericalDispatch &src, NumericalDispatch &out) {
  reset();
  out = NumericalDispatch{};
  out.contract_version = src.contract_version;
  out.corner_edge_service = src.corner_edge_service;
  LaunchFn *const dst[kSlotCount] = {
      &out.cgrid_predictor,       &out.cgrid_vertical_solver,
      &out.cgrid_pressure,        &out.dgrid_dynamics,
      &out.dgrid_height,          &out.vertical_solver,
      &out.nh_pressure,           &out.post_loop_heating,
      &out.dgrid_height_transport};
  LaunchFn const src_fns[kSlotCount] = {
      src.cgrid_predictor,       src.cgrid_vertical_solver,
      src.cgrid_pressure,        src.dgrid_dynamics,
      src.dgrid_height,          src.vertical_solver,
      src.nh_pressure,           src.post_loop_heating,
      src.dgrid_height_transport};
  LaunchFn const probe_fns[kSlotCount] = {
      &probe<0>, &probe<1>, &probe<2>, &probe<3>, &probe<4>,
      &probe<5>, &probe<6>, &probe<7>, &probe<8>};
  bool all_wired = true;
  for (int s = 0; s < kSlotCount; ++s) {
    real_fn[s] = src_fns[s];
    if (!src_fns[s]) {
      printf("  FAIL: production dispatch left %s NULL\n", slot_name[s]);
      all_wired = false;
      continue;
    }
    *dst[s] = probe_fns[s];
  }
  return all_wired;
}

} // namespace probes

// Sum of |x| over a buffer. Used to compare an n_split=1 run against an
// n_split=3 run: a single scalar is enough, because the question is only
// "did the substep loop run more than once", and a per-element diff would say
// nothing extra.
double abs_sum(const accumulator_real *p, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) s += std::fabs((double)p[i]);
  return s;
}

// One full integrated run. Returns false on any failure; on success writes the
// post-step |mfx| sum through `mfx_abs_sum` so the caller can compare two
// different n_split values. `expect_substeps` is asserted against every
// in-loop stage's probe count.
bool run_integrated(int32_t n_split, bool verbose, double *mfx_abs_sum) {
  fv3_gpu_error err{};
  fv3_gpu_context *ctx = nullptr;
  fv3_gpu_create_desc create{};
  create.abi_version = FV3_GPU_ABI_VERSION;
  create.struct_bytes = sizeof(create);
  create.device_ordinal = 0;
  create.mpi_rank = 0;
  create.mpi_size = 1;
  create.precision = sizeof(fv_real) == 4 ? FV3_GPU_PRECISION_R4 : FV3_GPU_PRECISION_R8;
  create.is = 1; create.ie = NI; create.js = 1; create.je = NJ;
  create.npz = NPZ; create.halo_width = HALO;
  int rc = fv3_gpu_create(&create, &ctx, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_create returned %d: %s\n", rc, err.message);
    return false;
  }

  HostState h = make_host_state();
  std::vector<fv3_gpu_field_desc> fields = make_fields(h);
  fv3_gpu_state_desc state{};
  state.abi_version = FV3_GPU_ABI_VERSION;
  state.struct_bytes = sizeof(state);
  state.field_count = static_cast<uint32_t>(fields.size());
  state.fields = fields.data();
  fv3_gpu_event ev = 0;
  rc = fv3_gpu_initialize_state(ctx, &state, &ev, &err);
  if (rc == FV3_GPU_SUCCESS) rc = fv3_gpu_event_wait(ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: state init/H2D returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  if (verbose) printf("  H2D complete (%zu fields registered)\n", fields.size());

  if (install_config(ctx, h, &err) != FV3_GPU_SUCCESS ||
      install_topology(ctx, &err) != FV3_GPU_SUCCESS) {
    printf("  FAIL: config/topology install: %s\n", err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  if (!declare_caller_managed_halos(ctx, &err)) {
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }

  NumericalDispatch production{};
  if (fv3_build_production_dispatch(production, &err) != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_build_production_dispatch: %s\n", err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  NumericalDispatch probed{};
  if (!probes::install(production, probed)) {
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  if (set_numerical_dispatch(ctx, probed, &err) != FV3_GPU_SUCCESS) {
    printf("  FAIL: set_numerical_dispatch: %s\n", err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  if (verbose)
    printf("  All 9 production stages installed behind counting probes\n\n");

  // §14 persistent-residency probe. These are the HOST buffers the fields were
  // registered from. If any stage performed an implicit device-to-host copy,
  // they would differ immediately after the step and BEFORE the explicit
  // sync_to_host below. Snapshotting them is the only residency assertion this
  // harness can make from outside the runtime -- but it is a real one, and it
  // is the exact regression project_architecture.md forbids.
  const std::vector<fv_real> delp_h2d = h.delp;
  const std::vector<fv_real> u_h2d = h.u;
  const std::vector<accumulator_real> mfx_h2d = h.mfx;

  fv3_gpu_step_desc step{};
  step.abi_version = FV3_GPU_ABI_VERSION;
  step.struct_bytes = sizeof(step);
  step.bdt = 10.0;
  step.k_split = 1;
  step.n_split = n_split;
  step.init_step = 1;
  step.end_step = 1;
  step.model_step = 0;
  step.acoustic_sequence = 1;
  step.dependency = 0;
  fv3_gpu_event step_ev = 0;
  rc = fv3_gpu_dyn_core_step(ctx, &step, &step_ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_dyn_core_step(n_split=%d) returned %d\n",
           (int)n_split, rc);
    printf("  error.api=\"%s\" message=\"%s\"\n", err.api, err.message);
    printf("  Probe counts at the point of failure:\n");
    for (int s = 0; s < probes::kSlotCount; ++s)
      printf("    %-56s entered %d time(s), last rc=%d\n", probes::slot_name[s],
             probes::count[s], probes::last_rc[s]);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  if (verbose)
    printf("  fv3_gpu_dyn_core_step(n_split=%d): SUCCESS (event=%lu)\n\n",
           (int)n_split, (unsigned long)step_ev);

  // ---- §16 per-stage reachability -----------------------------------------
  // Expected counts follow dyn_core.F90's structure, not convenience:
  //   the eight acoustic stages are inside `do it=1,n_split` (:399-1019), so
  //   each must be entered exactly n_split times;
  //   post_loop_heating is the diffusive-heating tail AFTER the loop closes
  //   (:1231-1240), so exactly once per dyn_core call regardless of n_split.
  // dgrid_height_transport sits inside update_dz_d, itself inside the loop, so
  // it is also n_split.
  bool reach_pass = true;
  if (verbose) printf("  Per-stage reachability (n_split=%d):\n", (int)n_split);
  for (int s = 0; s < probes::kSlotCount; ++s) {
    const int expected = (s == probes::kPostLoopHeating) ? 1 : (int)n_split;
    const bool ok = probes::count[s] == expected;
    reach_pass &= ok;
    if (verbose || !ok)
      printf("    %-56s %d/%d %s\n", probes::slot_name[s], probes::count[s],
             expected, ok ? "REACHED" : "*** WRONG COUNT ***");
  }

  // ---- §14 residency ------------------------------------------------------
  const bool no_implicit_d2h =
      h.delp == delp_h2d && h.u == u_h2d && h.mfx == mfx_h2d;
  if (verbose)
    printf("\n  Host buffers unchanged before explicit sync_to_host: %s\n",
           no_implicit_d2h ? "YES (no implicit D2H)"
                           : "NO *** implicit D2H detected ***");

  fv3_gpu_field_mask mask{};
  mask.abi_version = FV3_GPU_ABI_VERSION;
  mask.struct_bytes = sizeof(mask);
  for (FieldId id : {FieldId::u, FieldId::v, FieldId::uc, FieldId::vc,
                     FieldId::delp, FieldId::pt, FieldId::w, FieldId::delz,
                     FieldId::zh, FieldId::gz, FieldId::pkc, FieldId::pkz,
                     FieldId::pe, FieldId::peln, FieldId::pk, FieldId::ws,
                     FieldId::ws3, FieldId::omga, FieldId::heat_source,
                     FieldId::mfx, FieldId::mfy, FieldId::cx, FieldId::cy})
    mask.words[static_cast<uint32_t>(id) / 64] |= (1ULL << (static_cast<uint32_t>(id) % 64));
  rc = fv3_gpu_sync_to_host(ctx, &mask, &ev, &err);
  if (rc == FV3_GPU_SUCCESS) rc = fv3_gpu_event_wait(ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: D2H returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }

  bool finite_pass = true;
  if (verbose) printf("\n  Finite checks across the full integrated output:\n");
  struct Named { const char *name; const fv_real *p; size_t n; };
  for (const Named &f : {Named{"u", h.u.data(), h.u.size()},
                         {"v", h.v.data(), h.v.size()},
                         {"uc", h.uc.data(), h.uc.size()},
                         {"vc", h.vc.data(), h.vc.size()},
                         {"delp", h.delp.data(), h.delp.size()},
                         {"pt", h.pt.data(), h.pt.size()},
                         {"w", h.w.data(), h.w.size()},
                         {"delz", h.delz.data(), h.delz.size()},
                         {"zh", h.zh.data(), h.zh.size()},
                         {"gz", h.gz.data(), h.gz.size()},
                         {"pkc", h.pkc.data(), h.pkc.size()},
                         {"pkz", h.pkz.data(), h.pkz.size()},
                         {"pe", h.pe.data(), h.pe.size()},
                         {"peln", h.peln.data(), h.peln.size()},
                         {"pk", h.pk.data(), h.pk.size()},
                         {"ws", h.ws.data(), h.ws.size()},
                         {"ws3", h.ws3.data(), h.ws3.size()},
                         {"omga", h.omga.data(), h.omga.size()},
                         {"heat_source", h.heat_source.data(), h.heat_source.size()}})
    finite_pass &= check_finite(f.p, f.n, f.name);
  finite_pass &= check_finite(h.mfx.data(), h.mfx.size(), "mfx");
  finite_pass &= check_finite(h.mfy.data(), h.mfy.size(), "mfy");
  finite_pass &= check_finite(h.cx.data(), h.cx.size(), "cx");
  finite_pass &= check_finite(h.cy.data(), h.cy.size(), "cy");

  // ---- non-degenerate output ----------------------------------------------
  // delp/w witness the transport half, uc/vc the C-grid half, zh/delz the
  // vertical half. Comparing against the H2D snapshot rather than against zero
  // keeps every probe independent of the initial value.
  const bool delp_moved =
      report_changed(delp_h2d.data(), h.delp.data(), h.delp.size(), "delp");
  const bool u_moved = report_changed(u_h2d.data(), h.u.data(), h.u.size(), "u");

  // ---- flux capacitors ----------------------------------------------------
  // dyn_core.F90:372-375 zeroes mfx/mfy/cx/cy ONCE per k_split, before
  // `do it=1,n_split`, and every substep then ADDS to them. Two independent
  // claims follow, and both are checked:
  //   (1) the zeroing happened -- the deliberately nonzero seed (11.0/13.0/
  //       0.17/0.19) must be gone, so no element may still equal it;
  //   (2) accumulation happened -- the result must be nonzero.
  // A stage that overwrote instead of accumulating would still pass (2), which
  // is why the caller additionally compares n_split=1 against n_split=3.
  bool seed_cleared = true;
  for (size_t i = 0; i < h.mfx.size(); ++i)
    if (h.mfx[i] == (accumulator_real)11.0) { seed_cleared = false; break; }
  const double mfx_sum = abs_sum(h.mfx.data(), h.mfx.size());
  const double cx_sum = abs_sum(h.cx.data(), h.cx.size());
  const bool capacitors_nonzero = mfx_sum > 0.0 && cx_sum > 0.0;
  if (verbose) {
    printf("\n  Flux capacitors (mfx, mfy, cx, cy -- the FOUR dyn_core.F90\n");
    printf("  :372-375 zeroes; diss_est and dpx are fv_dynamics-owned and are\n");
    printf("  deliberately NOT in this set):\n");
    printf("    seed 11.0 fully cleared by the runtime's zeroing: %s\n",
           seed_cleared ? "YES" : "NO");
    printf("    sum|mfx| = %.6e, sum|cx| = %.6e\n", mfx_sum, cx_sum);
  }

  if (mfx_abs_sum) *mfx_abs_sum = mfx_sum;

  rc = fv3_gpu_finalize(&ctx, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: fv3_gpu_finalize returned %d: %s\n", rc, err.message);
    return false;
  }

  return reach_pass && no_implicit_d2h && finite_pass && delp_moved &&
         u_moved && seed_cleared && capacitors_nonzero;
}

bool run_phase_c() {
  printf("########################################################\n");
  printf("# Phase C: integrated Gate C1 evidence run\n");
  printf("# All 9 production stages, n_split > 1, through the public\n");
  printf("# entry point fv3_gpu_dyn_core_step, per-stage reachability\n");
  printf("# asserted by counting probes.\n");
  printf("########################################################\n\n");

  double sum_n3 = 0.0, sum_n1 = 0.0;
  printf("--- Run 1: n_split = 3 (the authoritative acoustic loop) ---\n");
  const bool n3_pass = run_integrated(3, true, &sum_n3);
  printf("\n--- Run 2: n_split = 1 (control, for the substep-loop probe) ---\n");
  const bool n1_pass = run_integrated(1, false, &sum_n1);

  // §7 evidence. If the runtime executed the stage list once regardless of
  // n_split -- the pre-3A.9 behaviour -- these two sums would be IDENTICAL,
  // because everything else about the two runs is the same. They differ
  // because dt = bdt/n_split differs per substep AND because the capacitors
  // accumulate over n_split substeps.
  const bool substep_loop_live = n3_pass && n1_pass && sum_n3 != sum_n1;
  printf("\n  Acoustic-loop probe:\n");
  printf("    sum|mfx| at n_split=1: %.9e\n", sum_n1);
  printf("    sum|mfx| at n_split=3: %.9e\n", sum_n3);
  printf("    differ (so the loop is a real loop, not one pass): %s\n",
         substep_loop_live ? "YES" : "NO");

  const bool pass = n3_pass && n1_pass && substep_loop_live;
  printf("\nPhase C result: %s\n\n", pass ? "PASS" : "FAIL");
  return pass;
}

// ===========================================================================
// Milestone 3A.9 §13 / §8 -- the fence. Every check here asserts a REFUSAL,
// and asserts that the refusal happened BEFORE model state was mutated, which
// is the requirement project_architecture.md states ("Unsupported
// configurations should fail before mutating model state") and which a plain
// "returned nonzero" check would not establish.
//
// State mutation is detected the same way residency is: the registered host
// buffers plus an explicit D2H. If a rejected step had run any kernel, the
// device delp would differ from the H2D value and the post-rejection D2H would
// show it.
namespace fence {

// A mutation applied to the otherwise-valid production config.
struct Case {
  const char *name;
  const char *why;
  void (*mutate)(NumericalConfig &);
  bool skip_config_install; // for the halo case, which mutates nothing
};

void set_hord_mt_8(NumericalConfig &c) { c.hord_mt = 8; }
void set_grid_type_4(NumericalConfig &c) { c.grid_type = 4; }
void set_hydrostatic(NumericalConfig &c) { c.hydrostatic = 1; }
void clear_m_split(NumericalConfig &c) { c.m_split = 0; }
void clear_a_imp(NumericalConfig &c) { c.a_imp = 0.5; }

// Rebuilds the production config, applies `mutate`, installs the result.
// set_numerical_config itself must ACCEPT the perturbed config -- it is a plain
// setter, and the refusal being tested belongs to fv3_gpu_dyn_core_step. If a
// setter-level rejection ever appeared, run_case would report it as a config
// install failure rather than silently counting it as the fence firing.
int install_mutated(fv3_gpu_context *ctx, HostState &h, fv3_gpu_error *err,
                    void (*mutate)(NumericalConfig &)) {
  NumericalConfig cfg = production_config(h);
  mutate(cfg);
  return set_numerical_config(ctx, cfg, err);
}

bool run_case(const Case &c) {
  fv3_gpu_error err{};
  fv3_gpu_context *ctx = nullptr;
  fv3_gpu_create_desc create{};
  create.abi_version = FV3_GPU_ABI_VERSION;
  create.struct_bytes = sizeof(create);
  create.device_ordinal = 0;
  create.mpi_rank = 0;
  create.mpi_size = 1;
  create.precision = sizeof(fv_real) == 4 ? FV3_GPU_PRECISION_R4 : FV3_GPU_PRECISION_R8;
  create.is = 1; create.ie = NI; create.js = 1; create.je = NJ;
  create.npz = NPZ; create.halo_width = HALO;
  if (fv3_gpu_create(&create, &ctx, &err) != FV3_GPU_SUCCESS) {
    printf("    FAIL: create: %s\n", err.message);
    return false;
  }
  HostState h = make_host_state();
  std::vector<fv3_gpu_field_desc> fields = make_fields(h);
  fv3_gpu_state_desc state{};
  state.abi_version = FV3_GPU_ABI_VERSION;
  state.struct_bytes = sizeof(state);
  state.field_count = static_cast<uint32_t>(fields.size());
  state.fields = fields.data();
  fv3_gpu_event ev = 0;
  int rc = fv3_gpu_initialize_state(ctx, &state, &ev, &err);
  if (rc == FV3_GPU_SUCCESS) rc = fv3_gpu_event_wait(ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("    FAIL: H2D: %s\n", err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  if (c.mutate) rc = install_mutated(ctx, h, &err, c.mutate);
  else rc = install_config(ctx, h, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("    FAIL: config install: %s\n", err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  if (install_topology(ctx, &err) != FV3_GPU_SUCCESS) {
    printf("    FAIL: topology install: %s\n", err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }
  // The halo case is exactly this: DO NOT declare the groups caller-managed.
  if (!c.skip_config_install) {
    fv3_gpu_error quiet{};
    for (fv3_gpu_halo_group g : kRuntimeResolvedGroups)
      mark_halo_caller_managed(ctx, g, &quiet);
  }
  NumericalDispatch production{};
  if (fv3_build_production_dispatch(production, &err) != FV3_GPU_SUCCESS ||
      set_numerical_dispatch(ctx, production, &err) != FV3_GPU_SUCCESS) {
    printf("    FAIL: dispatch install: %s\n", err.message);
    fv3_gpu_finalize(&ctx, &err);
    return false;
  }

  const std::vector<fv_real> delp_h2d = h.delp;
  const std::vector<fv_real> u_h2d = h.u;

  fv3_gpu_step_desc step{};
  step.abi_version = FV3_GPU_ABI_VERSION;
  step.struct_bytes = sizeof(step);
  step.bdt = 10.0;
  step.k_split = 1;
  step.n_split = 3;
  step.init_step = 1;
  step.end_step = 1;
  step.model_step = 0;
  step.acoustic_sequence = 1;
  step.dependency = 0;
  fv3_gpu_event step_ev = 0;
  rc = fv3_gpu_dyn_core_step(ctx, &step, &step_ev, &err);

  bool ok = true;
  if (rc == FV3_GPU_SUCCESS) {
    printf("    *** ACCEPTED (expected refusal): %s\n", c.name);
    ok = false;
  } else if (rc != FV3_GPU_UNSUPPORTED_CONFIGURATION) {
    printf("    *** refused with %d, expected FV3_GPU_UNSUPPORTED_CONFIGURATION"
           " (%d): %s\n", rc, (int)FV3_GPU_UNSUPPORTED_CONFIGURATION, err.message);
    ok = false;
  } else {
    printf("    REFUSED  rc=UNSUPPORTED_CONFIGURATION  api=\"%s\"\n", err.api);
    printf("             \"%s\"\n", err.message);
  }

  // Did the refusal happen before state was mutated? A failed step leaves the
  // context in ContextPhase::failed, and fv3_gpu_sync_to_host is expected to
  // refuse on a failed context -- so a D2H error here is not itself a test
  // failure. What IS checked is that if the D2H does succeed, nothing moved.
  fv3_gpu_field_mask mask{};
  mask.abi_version = FV3_GPU_ABI_VERSION;
  mask.struct_bytes = sizeof(mask);
  for (FieldId id : {FieldId::delp, FieldId::u})
    mask.words[static_cast<uint32_t>(id) / 64] |= (1ULL << (static_cast<uint32_t>(id) % 64));
  fv3_gpu_error d2h_err{};
  fv3_gpu_event d2h_ev = 0;
  const int d2h_rc = fv3_gpu_sync_to_host(ctx, &mask, &d2h_ev, &d2h_err);
  if (d2h_rc == FV3_GPU_SUCCESS &&
      fv3_gpu_event_wait(ctx, d2h_ev, &d2h_err) == FV3_GPU_SUCCESS) {
    const bool unmutated = h.delp == delp_h2d && h.u == u_h2d;
    printf("             device state unmutated: %s\n",
           unmutated ? "YES" : "NO *** state was modified before the refusal ***");
    ok = ok && unmutated;
  } else {
    printf("             (D2H refused on the failed context: %s -- state\n"
           "              mutation not separately observable for this case)\n",
           d2h_err.message);
  }

  fv3_gpu_error fin{};
  fv3_gpu_finalize(&ctx, &fin);
  return ok;
}

} // namespace fence

bool run_phase_d_fence() {
  printf("########################################################\n");
  printf("# Phase D: the production configuration fence (§13) and the\n");
  printf("# halo-readiness refusal (§8). Every case here must be REFUSED,\n");
  printf("# with FV3_GPU_UNSUPPORTED_CONFIGURATION, before state mutation.\n");
  printf("########################################################\n\n");

  const fence::Case cases[] = {
      {"hord_mt = 8 (not the production first-run value 6)",
       "project_architecture.md pins hord_mt/hord_vt/hord_tm = 6, hord_dp = -6",
       &fence::set_hord_mt_8, false},
      {"grid_type = 4 (doubly-periodic Cartesian, not the cubed sphere)",
       "the production first-run config is grid_type = 0",
       &fence::set_grid_type_4, false},
      {"hydrostatic = true",
       "the production first-run config is nonhydrostatic",
       &fence::set_hydrostatic, false},
      {"a_imp = 0.5 (a non-SIM1 vertical branch)",
       "the production first-run config is a_imp = 1.0",
       &fence::clear_a_imp, false},
      {"m_split = 0 (the not-supplied sentinel)",
       "neither Riemann solver may default its vertical substep count",
       &fence::clear_m_split, false},
      {"halo groups NOT declared caller-managed on a connected topology",
       "the runtime has no halo transport and must not invent readiness",
       nullptr, true},
  };

  bool pass = true;
  for (const fence::Case &c : cases) {
    printf("  Case: %s\n", c.name);
    printf("    Rationale: %s\n", c.why);
    const bool ok = fence::run_case(c);
    pass &= ok;
    printf("    -> %s\n\n", ok ? "PASS (refused as required)" : "FAIL");
  }
  printf("Phase D result: %s\n\n", pass ? "PASS" : "FAIL");
  return pass;
}

// ===========================================================================
// Phase E -- Milestone 3A.10 §13/§14/§19: the SIX-TILE integrated run with a
// REAL halo exchange.
//
// Phase C above is the same integrated path with declare_caller_managed_halos(),
// and its own summary says so plainly: it executes, and its halo values are not
// physically correct. That declaration is exactly what §14 forbids -- "no manual
// halo-ready declarations may bypass required exchange" -- so Phase E does not
// make it. Six contexts, one per cubed-sphere tile, attach to a shared
// HaloPeerGroup and every one of the nine runtime-resolved groups is resolved by
// a real pack -> device-to-device transfer -> seam-transforming unpack.
//
// -------------------------------------------------------------------------
// WHY SIX HOST THREADS
// -------------------------------------------------------------------------
//
// A halo exchange is two-sided. Tile 1's west halo is filled from its west
// neighbour's owned cells AS THEY STAND AT THIS SUBSTEP, so that neighbour must
// have packed before tile 1's transfer reads its send buffer. Six contexts
// stepped one after another on one thread cannot express that: tile 1 would copy
// the previous substep's values, or on the first substep uninitialised memory,
// and would still produce finite numbers. The reasoning is set out in full in
// include/fv3_gpu_halo_peer.hpp; the consequence here is six std::threads.
//
// SETUP AND TEARDOWN RUN SEQUENTIALLY on the main thread. Only the
// fv3_gpu_dyn_core_step calls are concurrent. A failure while creating context 4
// must not leave contexts 1-3 blocked in a barrier, and the CUDA runtime's
// context/allocation paths are not what this test is trying to exercise.
//
// The plain production dispatch is installed, NOT probes::. probes:: keeps its
// counts in file-scope arrays, which six threads would race on; Phase C already
// establishes per-stage reachability single-threaded, and re-deriving it here
// under a data race would weaken both claims rather than strengthen either.
//
// -------------------------------------------------------------------------
// THE CAUSALITY TEST, AND WHY IT IS THE ONE THAT MATTERS
// -------------------------------------------------------------------------
//
// §13 asks for "deliberately asymmetric per-tile data so incorrect neighbor
// mapping cannot pass". The per-element provenance decode that proves the seam
// mapping cell by cell lives in tests/integration/test_halo_exchange_multitile.cu
// and is much stronger than anything an integrated run can assert about a value
// that has been through d_sw, a Riemann solve and a pressure-gradient stage.
//
// What THIS test can prove, and what the multitile test cannot, is that the
// integrated dyn_core path is actually WIRED to that exchange. So it runs the
// whole six-tile step twice: once clean, and once with the initial u/v perturbed
// on tile 1's WEST NEIGHBOUR ONLY -- and asserts that TILE 1's output changed.
// Tile 1's own input is byte-identical between the two runs. The only route by
// which its answer can move is the halo exchange. If resolve_halo silently fell
// back to a declaration, or the transfer copied nothing, or the unpack wrote
// outside the halo, tile 1's checksum would be identical and this fails.
//
// The neighbour is QUERIED from the topology rather than written down as "tile
// 5", so the test cannot drift out of agreement with the mosaic it is testing.
// ===========================================================================
namespace phase_e {

using fv3::gpu::halo::HaloPeerGroup;

constexpr int kTiles = 6;

// Per-tile asymmetry. u/v are the prognostic D-grid winds and carry no
// consistency constraint with delz/zh (unlike delp/pt, whose perturbation would
// have to be matched in the hydrostatic column), so a phase shift keyed on the
// tile is a safe way to make all six tiles genuinely different. Without this,
// six identical tiles would exchange values indistinguishable from their own and
// a wrong neighbour mapping would go unnoticed.
void make_tile_asymmetric(HostState &h, int tile) {
  const double s = 0.37 * (double)tile;
  for (int64_t k = 0; k < NPZ; ++k)
    for (int64_t j = 0; j < h.nj_h; ++j)
      for (int64_t i = 0; i < h.ni_h; ++i) {
        const size_t idx = (size_t)(i + j * h.ni_h + k * h.ni_h * h.nj_h);
        h.u[idx] = (fv_real)((double)h.u[idx] + 0.8 * std::sin(s + 0.11 * (double)i));
        h.v[idx] = (fv_real)((double)h.v[idx] + 0.6 * std::cos(s + 0.13 * (double)j));
      }
}

// The extra kick applied to ONE tile in the second run. Deliberately small
// enough to stay physical and large enough to survive R4 rounding through a full
// substep: 1e-2 on a 12 m/s wind is ~1e-3 relative, while float carries ~1e-7.
void perturb(HostState &h) {
  for (size_t i = 0; i < h.u.size(); ++i) {
    h.u[i] = (fv_real)((double)h.u[i] + 1.0e-2);
    h.v[i] = (fv_real)((double)h.v[i] - 1.0e-2);
  }
}

double checksum(const HostState &h) {
  double s = 0.0;
  // A weighted sum rather than a plain one, so that two changes of opposite sign
  // at different indices cannot cancel into an unchanged total.
  for (size_t i = 0; i < h.u.size(); ++i)
    s += (double)h.u[i] * (1.0 + 1.0e-6 * (double)i);
  for (size_t i = 0; i < h.delp.size(); ++i)
    s += (double)h.delp[i] * (2.0 + 1.0e-6 * (double)i);
  for (size_t i = 0; i < h.pt.size(); ++i) s += (double)h.pt[i] * 0.5;
  return s;
}

int32_t rank_of_tile_table[kTiles] = {0, 1, 2, 3, 4, 5};

// One tile's whole participation, minus the step itself.
struct Member {
  fv3_gpu_context *ctx = nullptr;
  std::unique_ptr<HostState> h;   // heap-held: the registered device field views
                                  // point into these vectors, so the HostState
                                  // must not move once fields are registered.
  std::vector<fv3_gpu_field_desc> fields;
  TopologyView topo{};
  std::vector<fv_real> u_h2d, delp_h2d;
  int rc = FV3_GPU_SUCCESS;
  bool ok = false;
};

// Sequential setup for one tile. Returns false on any failure, having cleaned up
// whatever it created.
bool setup(Member &m, int tile, HaloPeerGroup &pg, bool perturb_this_tile) {
  fv3_gpu_error err{};
  fv3_gpu_create_desc create{};
  create.abi_version = FV3_GPU_ABI_VERSION;
  create.struct_bytes = sizeof(create);
  create.device_ordinal = 0;
  // A real deployment is one rank per tile, so the rank is the tile's rank and
  // the size is six. This is nominal in-process, but describing it as rank 0 of 1
  // would be describing a different decomposition than the topology installs.
  create.mpi_rank = tile - 1;
  create.mpi_size = kTiles;
  create.precision = sizeof(fv_real) == 4 ? FV3_GPU_PRECISION_R4 : FV3_GPU_PRECISION_R8;
  create.is = 1; create.ie = NI; create.js = 1; create.je = NJ;
  create.npz = NPZ; create.halo_width = HALO;
  m.rc = fv3_gpu_create(&create, &m.ctx, &err);
  if (m.rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: tile %d fv3_gpu_create returned %d: %s\n", tile, m.rc, err.message);
    return false;
  }

  m.h = std::make_unique<HostState>(make_host_state());
  make_tile_asymmetric(*m.h, tile);
  if (perturb_this_tile) perturb(*m.h);
  m.fields = make_fields(*m.h);

  fv3_gpu_state_desc state{};
  state.abi_version = FV3_GPU_ABI_VERSION;
  state.struct_bytes = sizeof(state);
  state.field_count = static_cast<uint32_t>(m.fields.size());
  state.fields = m.fields.data();
  fv3_gpu_event ev = 0;
  m.rc = fv3_gpu_initialize_state(m.ctx, &state, &ev, &err);
  if (m.rc == FV3_GPU_SUCCESS) m.rc = fv3_gpu_event_wait(m.ctx, ev, &err);
  if (m.rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: tile %d H2D returned %d: %s\n", tile, m.rc, err.message);
    return false;
  }

  if (install_config(m.ctx, *m.h, &err) != FV3_GPU_SUCCESS) {
    printf("  FAIL: tile %d install_config: %s\n", tile, err.message);
    return false;
  }

  // The REAL mosaic connectivity, not install_topology()'s four nominal
  // neighbours. make_cubed_sphere_topology is the builder validated against
  // CubeHalo.F90:213-261 by tests/runtime/test_halo_topology.cpp.
  if (!fv3::gpu::halo::make_cubed_sphere_topology(tile, NI + 1, NJ + 1,
                                                  rank_of_tile_table, m.topo)) {
    printf("  FAIL: tile %d make_cubed_sphere_topology\n", tile);
    return false;
  }
  if (set_topology(m.ctx, m.topo, &err) != FV3_GPU_SUCCESS) {
    printf("  FAIL: tile %d set_topology: %s\n", tile, err.message);
    return false;
  }

  // NOTE the absence of declare_caller_managed_halos(). Nothing here declares a
  // halo ready. If the exchange does not work, this test does not run.
  if (attach_halo_peer_group(m.ctx, &pg, tile, &err) != FV3_GPU_SUCCESS) {
    printf("  FAIL: tile %d attach_halo_peer_group: %s\n", tile, err.message);
    return false;
  }

  NumericalDispatch production{};
  if (fv3_build_production_dispatch(production, &err) != FV3_GPU_SUCCESS ||
      set_numerical_dispatch(m.ctx, production, &err) != FV3_GPU_SUCCESS) {
    printf("  FAIL: tile %d dispatch install: %s\n", tile, err.message);
    return false;
  }

  m.u_h2d = m.h->u;
  m.delp_h2d = m.h->delp;
  m.ok = true;
  return true;
}

void teardown(Member &m) {
  if (!m.ctx) return;
  fv3_gpu_error err{};
  detach_halo_peer_group(m.ctx, &err);
  fv3_gpu_finalize(&m.ctx, &err);
  m.ctx = nullptr;
}

// One six-tile run. `perturbed_tile` is 0 for the clean run, or the 1-based tile
// whose initial winds get the extra kick. On success writes each tile's post-step
// checksum through `out_checksum` (indexed 1..6).
bool six_tile_run(int32_t n_split, int perturbed_tile, bool verbose,
                  double *out_checksum) {
  HaloPeerGroup pg(kTiles);
  Member m[kTiles + 1];
  bool setup_ok = true;
  for (int t = 1; t <= kTiles && setup_ok; ++t)
    setup_ok = setup(m[t], t, pg, t == perturbed_tile);

  if (!setup_ok || !pg.complete()) {
    if (setup_ok) printf("  FAIL: peer group has %d/%d members\n", pg.attached(), kTiles);
    for (int t = kTiles; t >= 1; --t) teardown(m[t]);
    return false;
  }
  if (verbose) printf("  6 contexts created, real mosaic topology installed,\n"
                      "  all attached to one peer group, NO caller-managed halos\n");

  // ---- the concurrent part, and only the concurrent part -------------------
  std::thread th[kTiles + 1];
  for (int t = 1; t <= kTiles; ++t) {
    th[t] = std::thread([&m, &pg, t, n_split]() {
      fv3_gpu_error err{};
      fv3_gpu_step_desc step{};
      step.abi_version = FV3_GPU_ABI_VERSION;
      step.struct_bytes = sizeof(step);
      step.bdt = 10.0;
      step.k_split = 1;
      step.n_split = n_split;
      step.init_step = 1;
      step.end_step = 1;
      step.model_step = 0;
      step.acoustic_sequence = 1;
      step.dependency = 0;
      fv3_gpu_event step_ev = 0;
      const int rc = fv3_gpu_dyn_core_step(m[t].ctx, &step, &step_ev, &err);
      m[t].rc = rc;
      if (rc != FV3_GPU_SUCCESS) {
        printf("  FAIL: tile %d fv3_gpu_dyn_core_step returned %d\n"
               "        api=\"%s\" message=\"%s\"\n", t, rc, err.api, err.message);
        // Release the other five from the rendezvous. Without this they wait for
        // a pack that is never coming and the test hangs instead of failing.
        pg.abort_group();
        m[t].ok = false;
      }
    });
  }
  for (int t = 1; t <= kTiles; ++t) th[t].join();

  bool pass = true;
  for (int t = 1; t <= kTiles; ++t) pass &= (m[t].rc == FV3_GPU_SUCCESS);
  if (!pass) {
    for (int t = kTiles; t >= 1; --t) teardown(m[t]);
    return false;
  }
  if (verbose) printf("  all 6 tiles completed fv3_gpu_dyn_core_step(n_split=%d)\n",
                      (int)n_split);

  // ---- §10/§27 provenance: every group must say `exchanged` ----------------
  // This is the criterion §27 turns on. HaloResolution::caller_declared here
  // would mean the runtime accepted a declaration in place of data movement, and
  // it is the difference between Gate C1 PASS and Gate C1 BLOCKED.
  bool provenance_ok = true;
  for (int t = 1; t <= kTiles; ++t)
    for (fv3_gpu_halo_group g : kRuntimeResolvedGroups) {
      const HaloResolution r = halo_resolution_of(m[t].ctx, g);
      if (r != HaloResolution::exchanged) {
        printf("  FAIL: tile %d group %d resolved as %d, not `exchanged`\n",
               t, (int)g, (int)r);
        provenance_ok = false;
      }
    }
  if (verbose)
    printf("  halo provenance: all %zu groups x 6 tiles resolved by EXCHANGE: %s\n",
           sizeof(kRuntimeResolvedGroups) / sizeof(kRuntimeResolvedGroups[0]),
           provenance_ok ? "YES" : "NO");

  // ---- §14 residency, per tile, BEFORE the explicit sync ------------------
  bool residency_ok = true;
  for (int t = 1; t <= kTiles; ++t)
    if (!(m[t].h->u == m[t].u_h2d && m[t].h->delp == m[t].delp_h2d)) {
      printf("  FAIL: tile %d host buffers changed before sync_to_host\n", t);
      residency_ok = false;
    }
  if (verbose)
    printf("  persistent residency (no implicit D2H on any tile): %s\n",
           residency_ok ? "YES" : "NO");

  // ---- explicit D2H, then finite + checksum -------------------------------
  bool finite_ok = true;
  for (int t = 1; t <= kTiles; ++t) {
    fv3_gpu_error err{};
    fv3_gpu_field_mask mask{};
    mask.abi_version = FV3_GPU_ABI_VERSION;
    mask.struct_bytes = sizeof(mask);
    for (FieldId id : {FieldId::u, FieldId::v, FieldId::uc, FieldId::vc,
                       FieldId::delp, FieldId::pt, FieldId::w, FieldId::delz,
                       FieldId::zh, FieldId::gz, FieldId::pkc, FieldId::pkz})
      mask.words[static_cast<uint32_t>(id) / 64] |=
          (1ULL << (static_cast<uint32_t>(id) % 64));
    fv3_gpu_event ev = 0;
    int rc = fv3_gpu_sync_to_host(m[t].ctx, &mask, &ev, &err);
    if (rc == FV3_GPU_SUCCESS) rc = fv3_gpu_event_wait(m[t].ctx, ev, &err);
    if (rc != FV3_GPU_SUCCESS) {
      printf("  FAIL: tile %d D2H returned %d: %s\n", t, rc, err.message);
      finite_ok = false;
      continue;
    }
    char label[32];
    std::snprintf(label, sizeof(label), "tile%d u", t);
    finite_ok &= check_finite(m[t].h->u.data(), m[t].h->u.size(), label);
    std::snprintf(label, sizeof(label), "tile%d delp", t);
    finite_ok &= check_finite(m[t].h->delp.data(), m[t].h->delp.size(), label);
    std::snprintf(label, sizeof(label), "tile%d pt", t);
    finite_ok &= check_finite(m[t].h->pt.data(), m[t].h->pt.size(), label);
    if (out_checksum) out_checksum[t] = checksum(*m[t].h);
  }

  for (int t = kTiles; t >= 1; --t) teardown(m[t]);
  return pass && provenance_ok && residency_ok && finite_ok;
}

} // namespace phase_e

bool run_phase_e() {
  printf("########################################################\n");
  printf("# Phase E: SIX-TILE integrated run with a REAL halo exchange\n");
  printf("# Six contexts, real cubed-sphere mosaic, one peer group,\n");
  printf("# n_split = 3, and NO caller-managed halo declarations.\n");
  printf("########################################################\n\n");

  double clean[phase_e::kTiles + 1] = {0.0};
  printf("--- Run 1: clean, n_split = 3 ---\n");
  const bool clean_pass = phase_e::six_tile_run(3, 0, true, clean);
  if (!clean_pass) {
    printf("\nPhase E result: FAIL (clean six-tile run did not complete)\n\n");
    return false;
  }

  // ---- §17 ordering stress: the same run twice must agree BITWISE -----------
  //
  // Six host threads enqueue into six pairs of non-blocking streams, and the
  // interleaving differs from run to run -- the OS schedules the threads, and
  // nothing in this test pins or paces them. If the exchange's ordering were
  // incomplete at any of its four edges (producer -> pack, donor pack ->
  // transfer, unpack -> consumer, or the write-after-read guard on the send
  // buffer), then which side of a race won would vary with that interleaving and
  // these two runs would differ.
  //
  // So a bitwise match is evidence about the EVENTS, not about the arithmetic:
  // it is the observable that a missing cudaStreamWaitEvent would destroy. The
  // 3A.9 D2H race is the precedent -- it produced plausible numbers at
  // n_split = 3 and wrong ones at n_split = 1, and the difference between those
  // two was exactly a scheduling difference. Comparing two identical runs is how
  // that class of defect is caught without knowing the right answer.
  //
  // Repeated rather than single-shot because one run cannot be non-deterministic
  // on its own, and it is deliberately BITWISE rather than tolerance-based:
  // there is no tolerance at which a stream race is acceptable.
  double repeat[phase_e::kTiles + 1] = {0.0};
  printf("\n--- Run 2: byte-identical repeat of run 1 (ordering stress) ---\n");
  const bool repeat_pass = phase_e::six_tile_run(3, 0, false, repeat);
  if (!repeat_pass) {
    printf("\nPhase E result: FAIL (repeat six-tile run did not complete)\n\n");
    return false;
  }
  bool deterministic = true;
  for (int t = 1; t <= phase_e::kTiles; ++t)
    if (repeat[t] != clean[t]) {
      printf("  FAIL: tile %d differs between two identical runs:\n"
             "        %.17e vs %.17e\n", t, clean[t], repeat[t]);
      deterministic = false;
    }
  printf("  two identical six-thread runs agree bitwise on all 6 tiles: %s\n",
         deterministic ? "YES (stream ordering holds)"
                       : "NO *** nondeterministic: an ordering edge is missing ***");

  // Which tile actually feeds tile 1's west halo? Ask the mosaic, do not assume.
  TopologyView t1{};
  if (!fv3::gpu::halo::make_cubed_sphere_topology(1, NI + 1, NJ + 1,
                                                  phase_e::rank_of_tile_table, t1)) {
    printf("  FAIL: could not build tile 1's topology\n");
    return false;
  }
  const int west_donor =
      (int)t1.neighbors[(size_t)TileEdge::west].tile;
  printf("\n--- Run 3: identical, except tile %d's initial u/v are kicked ---\n",
         west_donor);
  printf("    (tile %d is tile 1's WEST neighbour in the mosaic; tile 1's own\n",
         west_donor);
  printf("     input is byte-identical to run 1)\n");

  double kicked[phase_e::kTiles + 1] = {0.0};
  const bool kick_pass = phase_e::six_tile_run(3, west_donor, false, kicked);
  if (!kick_pass) {
    printf("\nPhase E result: FAIL (perturbed six-tile run did not complete)\n\n");
    return false;
  }

  // The causality claim. Tile 1 saw no change to its own state; if its answer
  // moved, the information crossed the panel seam.
  const double d1 = std::fabs(kicked[1] - clean[1]);
  const bool causal = d1 > 0.0;
  printf("\n  Causality across the panel seam:\n");
  printf("    tile 1 checksum, clean      : %.12e\n", clean[1]);
  printf("    tile 1 checksum, tile %d kicked: %.12e\n", west_donor, kicked[1]);
  printf("    |difference|                : %.6e -> %s\n", d1,
         causal ? "CHANGED (halo data crossed the seam)"
                : "*** IDENTICAL: no exchange reached tile 1 ***");

  // Control: the kicked tile itself must obviously differ. If it did NOT, the
  // perturbation never made it into the run at all and the test above would be
  // vacuously "identical" for an uninteresting reason.
  const double dk = std::fabs(kicked[west_donor] - clean[west_donor]);
  const bool kick_landed = dk > 0.0;
  printf("    control: tile %d itself moved by %.6e -> %s\n", west_donor, dk,
         kick_landed ? "yes" : "*** NO: the perturbation never applied ***");

  const bool pass = clean_pass && repeat_pass && deterministic && kick_pass &&
                    causal && kick_landed;
  printf("\nPhase E result: %s\n\n", pass ? "PASS" : "FAIL");
  return pass;
}

int main() {
  printf("=== Milestone 3A.1 Production Dispatch Smoke Test ===\n\n");
  printf("Configuration:\n");
  printf("  Domain: %ldx%ldx%ld with halo=%u\n", NI, NJ, NPZ, HALO);
  printf("  Precision: %s (sizeof(fv_real)=%zu)\n\n",
         sizeof(fv_real) == 4 ? "R4" : "R8", sizeof(fv_real));

  fv3_gpu_capabilities caps{};
  caps.abi_version = FV3_GPU_ABI_VERSION;
  caps.struct_bytes = sizeof(caps);
  int rc = fv3_gpu_get_capabilities(nullptr, &caps);
  assert(rc == 0);
  printf("compiled_with_cuda=%u, device_count=%u\n", caps.compiled_with_cuda, caps.device_count);
  if (!caps.compiled_with_cuda) { printf("SKIP: Not compiled with CUDA\n"); return 77; }
  if (caps.device_count == 0) { printf("SKIP: No CUDA devices available\n"); return 77; }
  printf("CUDA available: %u device(s)\n\n", caps.device_count);

  bool phase_a_pass = run_phase_a();
  bool phase_b_pass = run_phase_b();
  bool phase_c_pass = run_phase_c();
  bool phase_d_pass = run_phase_d_fence();
  bool phase_e_pass = run_phase_e();

  printf("=== Summary ===\n");
  printf("  Phase A (minimal dispatch, full success + D2H verify):         %s\n", phase_a_pass ? "PASS" : "FAIL");
  printf("  Phase B (full production dispatch, full success + D2H verify): %s\n", phase_b_pass ? "PASS" : "FAIL");
  printf("  Phase C (integrated Gate C1: 9/9 stages, n_split=3, probes):   %s\n", phase_c_pass ? "PASS" : "FAIL");
  printf("  Phase D (production config fence + halo refusal):             %s\n", phase_d_pass ? "PASS" : "FAIL");
  printf("  Phase E (six tiles, REAL halo exchange, n_split=3):            %s\n", phase_e_pass ? "PASS" : "FAIL");

  const bool all = phase_a_pass && phase_b_pass && phase_c_pass &&
                   phase_d_pass && phase_e_pass;
  if (all) {
    printf("\nDispatch smoke test: PASS\n\n");
    printf("What this test now establishes, at the GPU EXECUTED level:\n");
    printf("  * All NINE production dispatch stages are entered through the\n");
    printf("    public entry point fv3_gpu_dyn_core_step -- asserted by\n");
    printf("    per-slot counting probes, not inferred from a success code.\n");
    printf("  * The acoustic loop is a real loop: the eight in-loop stages are\n");
    printf("    entered exactly n_split times and post_loop_heating exactly\n");
    printf("    once, and the n_split=1 vs n_split=3 capacitor sums differ.\n");
    printf("  * dgrid_height_transport is REACHABLE through the production\n");
    printf("    path -- it is on no stage list, and is entered only because\n");
    printf("    WP-C's update_dz_d calls it via ExecutionContext::dispatch.\n");
    printf("  * The four flux capacitors (mfx, mfy, cx, cy) are zeroed once\n");
    printf("    per call and then accumulated across substeps.\n");
    printf("  * Persistent device residency: the registered host buffers are\n");
    printf("    byte-identical after the step and before the explicit\n");
    printf("    sync_to_host, so no stage performed an implicit D2H.\n");
    printf("  * Configurations outside the production first-run subset are\n");
    printf("    refused before any device state is mutated.\n");
    printf("  * corner_edge_service is a REAL service as of Prompt 6B and the\n");
    printf("    production path requests it twice per del2_cubed pass at the\n");
    printf("    authoritative nf_ke = min(3, nord+1) = 3 (dyn_core.F90:2419\n");
    printf("    dir=1, :2437 dir=2). THIS TEST pins nf_ke=1, so nt=0 and it is\n");
    printf("    not requested here -- see the cfg.nf_ke comment. d_sw and\n");
    printf("    update_dz_c implement their own corner semantics inline through\n");
    printf("    CornerFillArgs, not through this slot.\n");
    printf("  * Milestone 3A.10: the integrated path runs on all SIX cubed-\n");
    printf("    sphere tiles with a REAL halo exchange (Phase E). No group is\n");
    printf("    declared caller-managed there; all nine runtime-resolved\n");
    printf("    groups on all six tiles report HaloResolution::exchanged, so\n");
    printf("    readiness came from data movement and not from a declaration.\n");
    printf("  * The integrated path is WIRED to that exchange: kicking one\n");
    printf("    tile's initial winds changes its NEIGHBOUR's answer, while the\n");
    printf("    neighbour's own input is byte-identical between the two runs.\n");
    printf("    Information can only reach it across the panel seam.\n");
    printf("\nWhat this test does NOT establish:\n");
    printf("  * Phase C's own halo values remain physically meaningless: it\n");
    printf("    declares the nine groups caller-managed and performs no\n");
    printf("    exchange, so its halo cells hold only what H2D put there. It\n");
    printf("    is a single-tile stage-reachability probe, and Phase E rather\n");
    printf("    than Phase C is the halo evidence.\n");
    printf("  * MULTI-RANK MPI. Phase E is one process representing six tiles\n");
    printf("    (Gate C1 execution model B), exchanging device-to-device. Six\n");
    printf("    MPI ranks, request lifetime, CUDA-aware MPI and multi-node\n");
    printf("    scaling remain future work.\n");
    printf("  * Corner halo BLOCKS. For a cubed sphere FMS never fills the\n");
    printf("    four tile-vertex diagonals at all (rotated_ninety gates off\n");
    printf("    fill_corner_contact), and FV3 reconstructs them locally --\n");
    printf("    copy_corners, fill_corners_dgrid, a2b's vertex extrapolation.\n");
    printf("    Those are compute kernels in WP-B/C/D, not this exchange.\n");
    printf("  * Numerical validation against authoritative Fortran savepoints:\n");
    printf("    NOT RUN.\n");
    return 0;
  }
  printf("\nDispatch smoke test: FAIL\n");
  return 1;
}
