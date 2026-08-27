// Milestone 3A.11 -- small-scale Fortran-vs-CUDA performance benchmark,
// CUDA/GPU side.
//
// PRE-VALIDATION PERFORMANCE CHARACTERIZATION. This measures wall-clock cost
// of fv3_gpu_dyn_core_step at three small grid sizes. It does NOT claim
// numerical agreement with authoritative Fortran, and it does NOT change any
// kernel or architecture in pursuit of a faster number -- both are out of
// scope for this milestone by explicit instruction ("measure first").
//
// Everything through the phase_e::teardown() definition below is copied
// verbatim from tests/integration/test_dispatch_smoke.cpp's Phase E harness
// (the permanent, already-validated Gate C1 six-tile / real-halo-exchange
// path), with exactly one change: NI/NJ/NPZ became runtime-settable globals
// instead of compile-time constants, so one binary can serve the
// 16x16x10 / 32x32x10 / 64x64x10 benchmark matrix from the CLI. Phase E's own
// test file is untouched by this change -- this is a new, separate file.
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

static int64_t NI = 8;      // runtime-configurable for the perf benchmark (3A.11)
static int64_t NJ = 8;      // runtime-configurable for the perf benchmark (3A.11)
static int64_t NPZ = 10;    // runtime-configurable for the perf benchmark (3A.11)
static constexpr uint32_t HALO = 3;

// bind_wp_d's contract (wp_d_contract.cpp) requires peln's k-slot (axis 2)
// to satisfy lower[2] <= js-2 (its hardcoded p=2 halo constant; pe uses
// p=1), but launch_vertical_solver's riem_solver3_kernel requires that same
// slot's registered lower[2] to make every (k - lower[2]) offset for
// k=0..npz land in-bounds (see the lo_3d/lo_3d_if comment below for the
// original 0-based-k fix). A single shared lower[2]=0 satisfies the vertical
// solver but not bind_wp_d's peln check (0 <= -1 is false); lower[2]=0 only
// satisfies bind_wp_d's looser pe check by coincidence at this domain size.
// Give pe/peln their own k-padded buffer instead: kPelnHaloPad dummy leading
// k-slices that the vertical solver's k=0..npz loop never addresses (its
// offsets land at buffer positions kPelnHaloPad..kPelnHaloPad+npz), while
// lower[2] = -kPelnHaloPad lets bind_wp_d's coverage check pass. Both WP-C's
// view3()/cview3() (fv3_vertical_kernels.cu) and WP-D's view3_const()
// (wp_d_operators.hpp) build device views generically from each field's own
// registered lower[]/stride[], so this is a per-field bookkeeping choice,
// not a kernel change -- no field data used by any executed kernel moves or
// changes value, it just lives kPelnHaloPad slices further into its buffer.
static constexpr int64_t kPelnHaloPad = 2;

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
  h.size_3d_pe = (size_t)(h.ni_h * h.nj_h * (h.npz1 + kPelnHaloPad));
  h.size_2d = (size_t)(h.ni_h * h.nj_h);

  h.delp.assign(h.size_3d, 0); h.pt.assign(h.size_3d, 0); h.w.assign(h.size_3d, 0);
  h.delz.assign(h.size_3d, 0); h.q_con.assign(h.size_3d, 0); h.cappa.assign(h.size_3d, 0);
  h.uc.assign(h.size_3d, 0); h.vc.assign(h.size_3d, 0); h.delpc.assign(h.size_3d, 0);
  // pe/peln get their own k-padded buffer -- see kPelnHaloPad's comment.
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
    // delz is the per-LAYER thickness and must be consistent with the zh/gz
    // column built below, which spans 10000 m -> 0 m in steps of 1000 m
    // (z = 10000 - k*1000). Total depth 10000 m over NPZ layers therefore
    // gives -10000/NPZ = -1000 m per layer.
    //
    // This previously read -1000.0/NPZ = -100 m, understating the layer
    // thickness by exactly 10x. That is not a benign scaling of the test
    // input: gz_from_delz_kernel reconstructs gz's INTERIOR from delz, while
    // gz's HALO keeps the analytic grav*z from the zh column, and H_GZ_FIRST
    // resolves only at it==1 (before that reconstruction). The halo is
    // therefore never made consistent with the interior, leaving a ~9x
    // geopotential cliff exactly one cell outside each tile edge. At it==2
    // update_dz_c differences gz2 across that cliff, and update_gz's
    // denominator (area + xfx_diff + yfx_diff) crosses zero -> NaN, first at
    // the southeast corner where the east and south cliffs compound.
    h.delz[idx] = -10000.0 / NPZ;
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
        // pe/peln live kPelnHaloPad slices further into their own
        // k-padded buffer (see kPelnHaloPad's comment) -- the vertical
        // solver's k=0..npz loop reaches exactly this shifted range via
        // its view's registered lower[2] = -kPelnHaloPad.
        const size_t idx_pe = i + j * (size_t)h.ni_h +
                              (k + (size_t)kPelnHaloPad) * (size_t)h.ni_h * (size_t)h.nj_h;
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

  // pe/peln: own k-padded shape (see kPelnHaloPad's comment) -- same (i,j)
  // box and stride as lo_3d_if, but lower[2] shifted to -kPelnHaloPad and
  // extent[2] grown to match, so bind_wp_d's coverage check on axis 2
  // (lower[2] <= js-p) can pass while the vertical solver's k=0..npz
  // accesses still land on the real data at buffer offset kPelnHaloPad+k.
  static thread_local int64_t lo_pe[3];
  static thread_local uint64_t ex_pe[3];
  static thread_local int64_t st_pe[3];
  lo_pe[0] = lo_3d_if[0]; lo_pe[1] = lo_3d_if[1]; lo_pe[2] = -kPelnHaloPad;
  ex_pe[0] = ex_3d_if[0]; ex_pe[1] = ex_3d_if[1];
  ex_pe[2] = (uint64_t)(h.npz1 + kPelnHaloPad);
  st_pe[0] = st_3d_if[0]; st_pe[1] = st_3d_if[1]; st_pe[2] = st_3d_if[2];
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
  // requires every one of its 15 mutable-state 3-D fields to share
  // delp's lower[2], so a -kPelnHaloPad or interface-level registration is
  // rejected even though it would "cover" the range.
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
  // execution path. nf_ke must be exactly 1 because fv3_gpu_dyn_core_step
  // sets dispatch_params.diffusion_passes = config.nf_ke and runs that many
  // del2_cubed passes (fv3_gpu_runtime.cpp:620-623) --
  // launch_post_loop_heating_cuda requires diffusion_passes==nf_ke. With
  // nf_ke=1, passes=1 and nt=passes-1=0, so the corner_service
  // call inside post_loop_heating (only reached when nt>0) is skipped,
  // avoiding this milestone's unimplemented corner_edge_service stub.
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
constexpr fv3_gpu_halo_group kRuntimeResolvedGroups[] = {
    FV3_GPU_H_DWIND_IN, FV3_GPU_H_W_IN,    FV3_GPU_H_GZ_FIRST,
    FV3_GPU_H_DIVGD,    FV3_GPU_H_CWIND,   FV3_GPU_H_DSCALAR,
    FV3_GPU_H_ZH,       FV3_GPU_H_PKC2,    FV3_GPU_H_HEAT};
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

} // namespace phase_e (reopened below -- this closes the copied Phase E
  // foundation before new benchmark-only code adds to the same namespace)

// ===========================================================================
// Everything above this point (through Member/setup()/teardown()) is the
// copied Phase E foundation. What follows is new: a warmup+timed loop around
// repeated fv3_gpu_dyn_core_step calls, in two separately-measured modes.
//
// ---------------------------------------------------------------------------
// TIMING METHODOLOGY (3A.11 SS6)
// ---------------------------------------------------------------------------
//
// SS6 asks for cudaEventRecord/cudaEventElapsedTime and warns against an
// "unsynchronized host timer around async calls". The public C ABI
// (fv3_gpu_api.h) does not expose the runtime's internal CUDA stream, so a
// caller cannot legally record a cudaEvent_t onto the same stream
// fv3_gpu_dyn_core_step enqueues work on. What the ABI DOES expose, by
// design (WP-E "event/dependency interfaces", project_architecture.md), is
// fv3_gpu_event: a step's out-event can be threaded into the NEXT step's
// step_desc.dependency, and fv3_gpu_event_wait blocks the host until a given
// event's work has genuinely completed on the device.
//
// This benchmark uses that mechanism instead of a raw cudaEvent_t:
//   - every call's step_desc.dependency is the PREVIOUS call's out-event,
//     so the N calls form one real device-side dependency chain, not N
//     independently-launched, possibly-reordered operations;
//   - the host timer's stop point is gated by fv3_gpu_event_wait on the
//     LAST call's out-event, which does not return until that call's device
//     work is actually done.
// The host clock therefore never races ahead of the device; the wait is the
// synchronization SS6 requires, expressed through the runtime's own
// documented primitive rather than by reaching past the public ABI for a
// stream handle it does not expose. This is recorded here, and in the
// milestone 3A.11 handoff, as the documented equivalent SS6 permits in place
// of a bare cudaEventRecord pair.
//
// Warmup calls are chained and drained (event_wait) BEFORE the timed clock
// starts, so no warmup work can bleed into the timed window and no timed
// work is left in flight when the clock stops.
// ---------------------------------------------------------------------------

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace phase_e {

// One tile's warmup+timed loop for MODE A (resident): H2D already happened
// in setup(); this loop is pure repeated fv3_gpu_dyn_core_step, chained by
// dependency, with no per-iteration host/device transfer. Returns the timed
// elapsed milliseconds for THIS tile, or a negative value on failure.
double run_mode_a(Member &m, int32_t n_split, int warmup, int iters) {
  fv3_gpu_error err{};
  fv3_gpu_event prev_ev = 0;
  uint64_t model_step = 0;

  auto one_step = [&]() -> bool {
    fv3_gpu_step_desc step{};
    step.abi_version = FV3_GPU_ABI_VERSION;
    step.struct_bytes = sizeof(step);
    step.bdt = 10.0;
    step.k_split = 1;
    step.n_split = n_split;
    step.init_step = 1;
    step.end_step = 1;
    step.model_step = model_step++;
    step.acoustic_sequence = 1;
    step.dependency = prev_ev;
    fv3_gpu_event step_ev = 0;
    const int rc = fv3_gpu_dyn_core_step(m.ctx, &step, &step_ev, &err);
    if (rc != FV3_GPU_SUCCESS) {
      printf("  FAIL: dyn_core_step (mode A) returned %d api=[%s] msg=[%s]\n", rc,
             err.api, err.message);
      return false;
    }
    prev_ev = step_ev;
    return true;
  };

  for (int w = 0; w < warmup; ++w) if (!one_step()) return -1.0;
  if (fv3_gpu_event_wait(m.ctx, prev_ev, &err) != FV3_GPU_SUCCESS) return -1.0;

  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it) if (!one_step()) return -1.0;
  if (fv3_gpu_event_wait(m.ctx, prev_ev, &err) != FV3_GPU_SUCCESS) return -1.0;
  const auto t1 = std::chrono::steady_clock::now();

  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// The field mask used to round-trip "resident state" both directions in
// MODE B, and for the end-of-run D2H in MODE A. Same set Phase E's own D2H
// check already uses.
fv3_gpu_field_mask resident_field_mask() {
  fv3_gpu_field_mask mask{};
  mask.abi_version = FV3_GPU_ABI_VERSION;
  mask.struct_bytes = sizeof(mask);
  for (FieldId id : {FieldId::u, FieldId::v, FieldId::uc, FieldId::vc,
                     FieldId::delp, FieldId::pt, FieldId::w, FieldId::delz,
                     FieldId::zh, FieldId::gz, FieldId::pkc, FieldId::pkz})
    mask.words[static_cast<uint32_t>(id) / 64] |=
        (1ULL << (static_cast<uint32_t>(id) % 64));
  return mask;
}

// MODE B (transfer-inclusive): each measured iteration is
// sync_from_host -> dyn_core_step -> sync_to_host, chained by dependency and
// DRAINED before the next iteration starts -- SS5 asks for the timed section
// to include the transfer cost of EACH measured iteration, which requires
// serializing iterations rather than letting them pipeline.
double run_mode_b(Member &m, int32_t n_split, int warmup, int iters) {
  fv3_gpu_error err{};
  const fv3_gpu_field_mask mask = resident_field_mask();
  uint64_t model_step = 0;

  auto one_round_trip = [&]() -> bool {
    fv3_gpu_event ev = 0;
    int rc = fv3_gpu_sync_from_host(m.ctx, &mask, &ev, &err);
    if (rc != FV3_GPU_SUCCESS) {
      printf("  FAIL: sync_from_host (mode B) returned %d: %s\n", rc, err.message);
      return false;
    }
    fv3_gpu_step_desc step{};
    step.abi_version = FV3_GPU_ABI_VERSION;
    step.struct_bytes = sizeof(step);
    step.bdt = 10.0;
    step.k_split = 1;
    step.n_split = n_split;
    step.init_step = 1;
    step.end_step = 1;
    step.model_step = model_step++;
    step.acoustic_sequence = 1;
    step.dependency = ev;
    fv3_gpu_event step_ev = 0;
    rc = fv3_gpu_dyn_core_step(m.ctx, &step, &step_ev, &err);
    if (rc != FV3_GPU_SUCCESS) {
      printf("  FAIL: dyn_core_step (mode B) returned %d: %s\n", rc, err.message);
      return false;
    }
    fv3_gpu_event d2h_ev = 0;
    rc = fv3_gpu_sync_to_host(m.ctx, &mask, &d2h_ev, &err);
    if (rc != FV3_GPU_SUCCESS) {
      printf("  FAIL: sync_to_host (mode B) returned %d: %s\n", rc, err.message);
      return false;
    }
    return fv3_gpu_event_wait(m.ctx, d2h_ev, &err) == FV3_GPU_SUCCESS;
  };

  for (int w = 0; w < warmup; ++w) if (!one_round_trip()) return -1.0;

  const auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it) if (!one_round_trip()) return -1.0;
  const auto t1 = std::chrono::steady_clock::now();

  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// SS16: verify the timed work was not skipped/optimized away. Runs a D2H of
// the resident mask (outside all timing) and reports finiteness plus
// max|delta| against the H2D-uploaded initial value captured by setup().
// Returns false if the state did not change or is non-finite.
bool verify_work_executed(Member &m, int tile) {
  fv3_gpu_error err{};
  const fv3_gpu_field_mask mask = resident_field_mask();
  fv3_gpu_event ev = 0;
  int rc = fv3_gpu_sync_to_host(m.ctx, &mask, &ev, &err);
  if (rc == FV3_GPU_SUCCESS) rc = fv3_gpu_event_wait(m.ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("  FAIL: tile %d verify D2H returned %d: %s\n", tile, rc, err.message);
    return false;
  }
  char label[32];
  std::snprintf(label, sizeof(label), "tile%d u", tile);
  bool ok = check_finite(m.h->u.data(), m.h->u.size(), label);
  std::snprintf(label, sizeof(label), "tile%d delp", tile);
  ok &= check_finite(m.h->delp.data(), m.h->delp.size(), label);
  const bool u_changed = report_changed(m.u_h2d.data(), m.h->u.data(), m.h->u.size(), "u");
  const bool delp_changed =
      report_changed(m.delp_h2d.data(), m.h->delp.data(), m.h->delp.size(), "delp");
  return ok && (u_changed || delp_changed);
}

} // namespace phase_e

// ---------------------------------------------------------------------------
// Driver: one six-tile setup, one repeat of one mode. Fresh setup/teardown
// per repeat (matches Phase E's proven pattern) so repeats cannot leak state
// into each other. Prints one CSV row per repeat, schema:
//   implementation,grid_x,grid_y,levels,tiles,n_split,precision,mode,repeat,
//   elapsed_ms,ms_per_step,hardware
// ---------------------------------------------------------------------------
namespace {

bool run_one_repeat(int32_t n_split, int warmup, int iters, const char *mode,
                    int repeat, const char *hardware) {
  phase_e::HaloPeerGroup pg(phase_e::kTiles);
  phase_e::Member m[phase_e::kTiles + 1];
  bool setup_ok = true;
  for (int t = 1; t <= phase_e::kTiles && setup_ok; ++t)
    setup_ok = phase_e::setup(m[t], t, pg, false);
  if (!setup_ok || !pg.complete()) {
    printf("  FAIL: six-tile setup did not complete (repeat %d, mode %s)\n", repeat, mode);
    for (int t = phase_e::kTiles; t >= 1; --t) phase_e::teardown(m[t]);
    return false;
  }

  double elapsed_ms[phase_e::kTiles + 1] = {0.0};
  bool ok = true;
  std::thread th[phase_e::kTiles + 1];
  for (int t = 1; t <= phase_e::kTiles; ++t) {
    th[t] = std::thread([&, t]() {
      const double e = (std::strcmp(mode, "resident") == 0)
                           ? phase_e::run_mode_a(m[t], n_split, warmup, iters)
                           : phase_e::run_mode_b(m[t], n_split, warmup, iters);
      elapsed_ms[t] = e;
      // A mid-loop failure on this tile must not leave the other five tiles
      // blocked forever on the next step's halo-exchange rendezvous -- same
      // reasoning as six_tile_run()'s own pg.abort_group() on failure.
      if (e < 0.0) pg.abort_group();
    });
  }
  for (int t = 1; t <= phase_e::kTiles; ++t) th[t].join();
  for (int t = 1; t <= phase_e::kTiles; ++t) ok &= (elapsed_ms[t] >= 0.0);

  if (ok)
    for (int t = 1; t <= phase_e::kTiles; ++t)
      ok &= phase_e::verify_work_executed(m[t], t);

  for (int t = phase_e::kTiles; t >= 1; --t) phase_e::teardown(m[t]);
  if (!ok) {
    printf("  FAIL: repeat %d mode %s did not complete cleanly\n", repeat, mode);
    return false;
  }

  // SS9: the per-repeat statistic is the MAX across the six tiles -- the
  // resident step is bounded by whichever tile is slowest, not by an average
  // that could hide one stalled tile.
  double max_ms = 0.0;
  for (int t = 1; t <= phase_e::kTiles; ++t) max_ms = std::max(max_ms, elapsed_ms[t]);
  const double ms_per_step = max_ms / (double)iters;

  printf("CSV,cuda_gpu,%lld,%lld,%lld,%d,%s,%s,%d,%.6f,%.6f,%s\n",
         (long long)NI, (long long)NJ, (long long)NPZ, phase_e::kTiles,
         sizeof(fv_real) == 8 ? "R8" : "R4", mode, repeat, max_ms, ms_per_step,
         hardware);
  return true;
}

void usage() {
  printf("usage: fv3_gpu_perf_bench --grid N --npz N --n_split N --warmup N "
        "--iters N --repeats N --mode resident|transfer|both "
        "--hardware STRING\n");
}

} // namespace

int main(int argc, char **argv) {
  int64_t grid = 16, npz = 10;
  int32_t n_split = 3;
  int warmup = 20, iters = 100, repeats = 5;
  std::string mode = "both";
  std::string hardware = "unspecified";

  for (int i = 1; i < argc; ++i) {
    auto next = [&](int &i) { return (i + 1 < argc) ? argv[++i] : ""; };
    const std::string a = argv[i];
    if (a == "--grid") grid = std::atoll(next(i));
    else if (a == "--npz") npz = std::atoll(next(i));
    else if (a == "--n_split") n_split = std::atoi(next(i));
    else if (a == "--warmup") warmup = std::atoi(next(i));
    else if (a == "--iters") iters = std::atoi(next(i));
    else if (a == "--repeats") repeats = std::atoi(next(i));
    else if (a == "--mode") mode = next(i);
    else if (a == "--hardware") hardware = next(i);
    else if (a == "--help") { usage(); return 0; }
    else { printf("unknown argument: %s\n", a.c_str()); usage(); return 2; }
  }

  NI = grid; NJ = grid; NPZ = npz;

  fv3_gpu_capabilities caps{};
  caps.abi_version = FV3_GPU_ABI_VERSION;
  caps.struct_bytes = sizeof(caps);
  fv3_gpu_get_capabilities(nullptr, &caps);
  if (!caps.compiled_with_cuda) {
    printf("SKIP: not compiled with CUDA\n");
    return 77;
  }
  if (caps.device_count == 0) {
    printf("SKIP: no CUDA devices available\n");
    return 77;
  }

  printf("=== Milestone 3A.11 CUDA perf benchmark ===\n");
  printf("PRE-VALIDATION PERFORMANCE CHARACTERIZATION -- not numerical validation\n");
  printf("grid=%lldx%lld npz=%lld n_split=%d precision=%s warmup=%d iters=%d "
        "repeats=%d hardware=%s\n",
        (long long)grid, (long long)grid, (long long)npz, n_split,
        sizeof(fv_real) == 8 ? "R8" : "R4", warmup, iters, repeats,
        hardware.c_str());
  printf("CSVHEADER,implementation,grid_x,grid_y,levels,tiles,n_split,precision,"
        "mode,repeat,elapsed_ms,ms_per_step,hardware\n");

  const bool do_resident = (mode == "resident" || mode == "both");
  const bool do_transfer = (mode == "transfer" || mode == "both");

  bool all_ok = true;
  if (do_resident)
    for (int r = 1; r <= repeats; ++r)
      all_ok &= run_one_repeat(n_split, warmup, iters, "resident", r, hardware.c_str());
  if (do_transfer)
    for (int r = 1; r <= repeats; ++r)
      all_ok &= run_one_repeat(n_split, warmup, iters, "transfer_inclusive", r,
                               hardware.c_str());

  printf("\n=== 3A.11 CUDA perf benchmark: %s ===\n", all_ok ? "PASS" : "FAIL");
  printf("Numerical validation against authoritative Fortran savepoints: NOT RUN.\n");
  return all_ok ? 0 : 1;
}
