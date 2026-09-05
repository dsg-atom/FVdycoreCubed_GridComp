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
// Milestone 3A.11V Task 2: the benchmark no longer invents per-level damping
// arrays; it calls the shared authoritative translation of dyn_core.F90:677-765.
#include "fv3_damping_translation.hpp"
// 3A.11V Task 1: globally single-valued cubed-sphere geometry and analytic
// fields. Benchmark/test infrastructure only -- deliberately NOT linked into
// fv3_dyn_core_gpu. See the header for the provenance table and, importantly,
// for what it does NOT provide (grid metrics: the harness keeps its fabricated
// uniform 100 km metrics, so the IC is globally consistent but not balanced).
#include "fv3_cube_geometry.hpp"

// Milestone 3A.15 SS12: the six-MPI-process mode. Guarded so an MPI-off build of
// this benchmark is bit-identical to before the flag existed -- the option is
// OFF by default (cuda/dyn_core/CMakeLists.txt), and with it off neither the MPI
// header nor a single line below is compiled.
//
// The public communication API is included unconditionally because it declares
// only C types; the MPI header is not.
#include "fv3_gpu_communication.h"
#if defined(FV3_DYN_CORE_ENABLE_MPI_HALO)
#include <mpi.h>
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace fv3::gpu;
namespace cube = fv3::bench::cube;

static int64_t NI = 8;      // runtime-configurable for the perf benchmark (3A.11)
static int64_t NJ = 8;      // runtime-configurable for the perf benchmark (3A.11)
static int64_t NPZ = 10;    // runtime-configurable for the perf benchmark (3A.11)
static constexpr uint32_t HALO = 3;

// Milestone 6A.5: when set (via --grid-metrics <dir>), setup() overwrites the
// fabricated uniform grid metrics fv3_cube_geometry.hpp/make_host_state()
// seed with the authoritative non-uniform c8 spherical cubed-sphere metrics
// exported from the real Fortran gridstruct (tools/grid_metrics/
// convert_fortran_grid_metrics.py). Empty string (default) preserves the
// existing fabricated-metric benchmark path unchanged.
static std::string g_grid_metrics_dir;

// Milestone 6A.5 Section 10: when set (via --sp0-state <dir>), setup()
// overwrites HostState's dynamical-state seed (u,v,w,delp,pt,delz,q_con,pe,
// peln,pk) and da_min/da_min_c with the authoritative Fortran Prompt 6A SP0
// savepoint (converted by tools/savepoints/convert_sp0_to_cuda.py). Empty
// string (default) leaves the existing IC path unchanged.
static std::string g_sp0_state_dir;

// Milestone 6A.5 Section 10: when set (via --sp0-verify-only), main() loads
// SP0 state into all 6 tiles, performs H2D, immediately reads the state back
// (D2H, no kernel launch), and exits -- it never calls fv3_gpu_dyn_core_step.
static bool g_sp0_verify_only = false;
// Directory to dump the post-H2D/D2H readback into, one subdir per tile, for
// offline comparison against the Fortran SP0 savepoint by
// tools/savepoints/compare_sp0_identity.py. Required when
// g_sp0_verify_only is set.
static std::string g_sp0_dump_dir;

// Prompt 6B: the acoustic-loop timestep (dyn_core's own `bdt` dummy; Fortran
// divides it internally as dt = bdt/n_split, dyn_core.F90:317). The default
// (10.0) is an arbitrary benchmark-only value with no tie to any production
// config; the authoritative Prompt 6A savepoint run used bdt=3600.0
// (RUN_DT=3600 / HEARTBEAT_DT, k_split=1, from
// experiments/c8_L10_savepoint_6a_run1/fv3.j + input.nml). Getting this wrong
// would desync every dt-scaled quantity (Courant numbers, Riemann solver,
// damping) from the authoritative run for reasons having nothing to do with
// CUDA kernel fidelity, so validation-mode runs MUST pass --bdt 3600. Default
// unchanged so the synthetic/performance path is unaffected.
static double g_bdt = 10.0;

// Prompt 6B (Riem_Solver_c reconciliation): the AUTHORITATIVE dry-air gas
// constant. nh_utils.F90:53 takes `rdgas` from FMS `constants_mod`, and the
// Baselibs FMS the GEOSfvdycore_fortran_perf fixture links
// (ESMA-Baselibs-8.32.0, FMS_DIR in the fixture's build/CMakeCache.txt) is
// configured with -DCONSTANTS=GEOS (ESMA-Baselibs-8.32.0/src/GNUmakefile:1003),
// which selects constants/geos_constants.fh:44:
//     RDGAS = 8314.47 / 28.965      ! == 287.0523045054...
// (the same MAPL_RUNIV/MAPL_AIRMW pair as MAPL_RDRY; the GFDL variant's 287.04
// and the GFS variant's 287.05 are NOT what this fixture compiled).
//
// This harness previously used a round 287.0 both to derive its synthetic
// balanced column and as cfg.rdgas. That is a -1.822125e-4 relative error, and
// because SIM1_solver's pressure closure is p = -dm/dz*rgas*pt, too small an
// rgas forces too small a |dz| by the same relative amount: it biased every
// Riem_Solver_c layer thickness by -1.82e-4 and was the sole remaining cause of
// the SP1 gz/pkc divergence. Written as the quotient rather than a rounded
// decimal so it reproduces the Fortran parameter expression exactly.
static const double kFv3Rdgas = 8314.47 / 28.965;

// Prompt 6B (SP1 rerun): when set (via --sp1-boundary <dir>), setup()
// overwrites h.ptop and h.phis/h.zs with values recovered from the
// authoritative Fortran SP1 savepoint (ptop = pkc(:,:,1), phis = gz(:,:,km+1)
// RAW, zs = phis/grav; tools/savepoints/convert_sp1_boundary_to_cuda.py).
// Neither is exported by the SP0 loader or grid_metrics_6a5 (both explicitly
// exclude them -- see docs/agent_handoff/prompt_6b_cuda_fortran_validation.md
// Sec 7). Empty string (default) leaves h.ptop=1000.0 and h.phis/h.zs=0
// (make_host_state()'s synthetic-benchmark seed) completely unaffected.
static std::string g_sp1_boundary_dir;

// Prompt 6B Task H (SP2 it=1): the AUTHORITATIVE n_sponge for the
// GEOSfvdycore_fortran_perf c8/npz=10 validation fixture is -1, i.e. the
// sponge branch is DISABLED. Proven two independent ways:
//   1. the fixture's own namelist, experiments/c8_L10_savepoint_6a_run1/
//      scratch_.../input.nml &fv_core_nml: n_sponge = -1;
//   2. the authoritative captured runtime damping table,
//      savepoints/sp2_it1_pe0_damping.txt, which records
//      nord_k=nord_w=nord_t=nord_v=2, damp_vt=damp_w=damp_t=0.01,
//      d_con_k=1.0, d2_divg=0 at EVERY k=1..10 -- exactly what
//      dyn_core.F90:736's `if (npz>1 .and. n_sponge>0)` produces when it does
//      NOT fire.
// fv3_damping_translation.cpp's geos_production_flags() hardcodes n_sponge = 9
// (FV_StateMod.F90:694), which is correct for the GEOS production namelist it
// models but wrong for this fixture; with 9 the sponge arms corrupt 9 of the
// 10 levels (k=0: nord_k/nord_v/nord_w->0, d2_divg 0.2, damp_vt 0.1, d_con 0;
// k=1..7: d2_divg 0.1, damp_vt 0.05; k=8: d2_divg 0.02) and only k=9 matches.
//
// The override is a flag rather than a change to geos_production_flags()
// because that function IS the documented GEOS-production translation and is
// what the performance/stability baseline (milestone 3A.11T/U/V) was measured
// against. The sentinel below keeps the default path bit-identical.
static const int kDampNSpongeUnset = std::numeric_limits<int>::min();
static int g_damp_n_sponge = kDampNSpongeUnset;

// Prompt 6B Task J (sp2a it=1): the AUTHORITATIVE akap for this fixture is NOT
// 2/7. It is 0.28571426868438721, which is the SINGLE-PRECISION evaluation of
// R/(3.5*R) -- exactly one float32 ULP BELOW float32(2/7), i.e. 5.96e-08
// (= 2^-24) relative below the double 2/7 this harness used.
//
// Provenance, traced to the declaration rather than fitted:
//   MAPL PhysicalConstants.F90:49-56 declares, all as DEFAULT REAL (single):
//       MAPL_AIRMW = 28.965 ; MAPL_RUNIV = 8314.47
//       MAPL_RDRY  = MAPL_RUNIV/MAPL_AIRMW
//       MAPL_CPDRY = 3.5*MAPL_RDRY
//       MAPL_KAPPA = MAPL_RDRY/MAPL_CPDRY
//     so the whole chain rounds in float32 three times and lands one ULP low.
//   FV_StateMod.F90:491   kappa = MAPL_KAPPA        (widens to REAL8, value kept)
//   FV_StateMod.F90:1833  call fv_dynamics(..., kappa, ...)
//   fv_dynamics.F90:175   real, intent(IN) :: kappa
//   fv_dynamics.F90:419   akap = kappa
//   fv_dynamics.F90:645   call dyn_core(..., akap, ...)
//   dyn_core.F90:316      ptk  = ptop ** akap
//   dyn_core.F90:337      k1k  = akap / (1.-akap)
// and dyn_core passes akap on to Riem_Solver3 -> SIM1_solver.
//
// FMS constants_mod is NOT the source, and that matters because
// geos_constants.fh:50 spells KAPPA with the identical RDGAS/(3.5*RDGAS)
// expression: if FMS were the source, RDGAS would also be single, and
// Riem_Solver_c's dz (which is directly proportional to rgas) would then carry
// a ~9e-09 relative error. The validated SP1 it=1 gz/pkc agreement at
// <=6.3e-16 rules that out, so FMS is built with default real = 8 bytes and its
// KAPPA is the double 2/7. Only the MAPL chain is single.
//
// Measured decisively off the authoritative dump: Riem_Solver3 writes
// pk3(i,j,1) = ptk = ptop**akap with no other dependence (nh_core.F90:122), and
// sp2a_it1_pe0_pk3 has pk3(k=1) = 3.7275934279762497 uniformly over the owned
// box with ptop = 100, which inverts to akap = 0.28571426868438721. The double
// 2/7 gives 3.7275937203149399 and float32(2/7) gives 3.7275939395689739 --
// neither matches; the single-precision MAPL value reproduces it.
//
// Flag rather than an unconditional constant: 2/7 is the self-consistent value
// for the synthetic/performance path (h.akap also derives that path's pk/pkz/
// cappa seed), and the sentinel keeps that path bit-identical.
static const double kAkapUnset = -1.0;
static double g_akap = kAkapUnset;

// Prompt 6B Task J, second half: the authoritative Fortran uses TWO DIFFERENT
// kappas inside a single Riem_Solver3 call, and CUDA conflates them.
//
//   * the scalar `akap` (MAPL single, above) is used for the Exner variable:
//       nh_core.F90:102  ptk = exp(akap*peln1)
//       nh_core.F90:138  pk3(i,j,k) = exp(akap*peln2(i,k))
//   * the `cappa(i,j,k)` ARRAY drives the SIM1 thermodynamic closure, because
//     the fixture compiles both MOIST_CAPPA and USE_COND
//     (@fvdycore/CMakeLists.txt:88-89):
//       nh_core.F90:114  cp2(i,k) = cappa(i,j,k)        [#ifdef MOIST_CAPPA]
//       nh_core.F90:148  gm2(i,k) = 1./(1.-cp2(i,k))    [USE_COND+MOIST_CAPPA]
//       nh_core.F90:170  call SIM1_solver(..., gama, gm2, cp2, akap, ...)
//     and SIM1_solver uses gm2/cp2 in place of gama/kappa throughout.
//     cappa itself is built at fv_dynamics.F90:454 as
//       cappa = rdgas/(rdgas + cvm/(1.+dp1))
//     from FMS DOUBLE constants; dry (q_con == 0 on all 6 tiles here) gives
//     cvm = cv_air = 2.5*rdgas, hence cappa = 1/3.5 = 2/7 in double.
//
// So the authoritative pair for this fixture is akap = 0.28571426868438721
// (single) for pk3/ptk and cappa = 2/7 (double) for the closure. CUDA derives
// both from the single cfg.akap: fv3_vertical_kernels.cu:522-523 passes
// `1/(1-cfg.akap)` as gama and `cfg.akap` as kappa. That conflation is exactly
// what the measurement showed -- with cfg.akap = 2/7 the closure outputs
// (zh, delz, w, pkc) agreed to <=1.2e-15 while pk3 was off by 1.96e-07; with
// cfg.akap = the single value pk3 fell to 5.4e-11 and the closure outputs rose
// to ~1.3e-07, i.e. the error simply moved between the two roles.
//
// This retires the standing assumption that MOIST_CAPPA/USE_COND are
// round-off-neutral here. That argument was "cappa == akap == 2/7, so
// gm2 == gama"; cappa IS 2/7, but akap is NOT, so enabling the pair is what
// SEPARATES the two roles rather than what perturbs them.
//
// RETRACTED APPROACH, recorded so it is not retried: the first attempt at this
// enabled cfg.moist_cappa + cfg.use_cond and set the cappa array to 2/7, on the
// reasoning that CUDA already implements both branches. It cannot run --
// fv3_gpu_dyn_core_step rejects the configuration outright ("use_cond /
// moist_cappa are not implemented in this subset"), cfg-wide, before any state
// is mutated, and lifting that guard would switch on an unimplemented
// configuration across operators far outside this window.
//
// The viable separation is narrower and does not touch the moist branch at all.
// With cappa uniform and q_con == 0, Fortran's moist branch is algebraically
// identical to the dry branch evaluated at kappa == cappa == 2/7 (gm2 == gama,
// cp2 == kappa; only t1g's multiply order differs, at ~1e-16), which is exactly
// what CUDA's dry branch already computes with cfg.akap = 2/7. So the CLOSURE
// role is already authoritative by default; only the EXNER role needs the other
// constant. Hence akap_exner: a second scalar consumed only by ptk/pk3
// (riem3_precompute_column) and pk3_halo, leaving cfg.akap as the closure kappa
// everywhere -- including Riem_Solver_c, whose gama/cp2 at SP1 it=1 are
// validated at <=6.3e-16 with 2/7 and would REGRESS under a global akap
// override. Flag-gated; 0 means "not supplied" and is bit-identical to before.
static const double kAkapExnerUnset = -1.0;
static double g_akap_exner = kAkapExnerUnset;

// Prompt 8 (CUDA Graphs prototype). Opt-in only; false is bit-identical to
// before this flag existed. When true, every tile's context enables
// dgrid_dynamics graph capture/replay via fv3_gpu_set_cuda_graph_mode
// immediately after fv3_gpu_create -- see that function's contract.
static bool g_cuda_graph = false;

// Prompt 9 (kernel-fusion prototype). Opt-in only; false is bit-identical to
// before this flag existed. When true, c_sw's fill2_4corners(delp,pt,dir) +
// fill_4corners(w,dir) pair is issued as one launch_fill_corner_triplet call
// instead of two -- see fv3_cgrid_corners.hpp. Numerical semantics changed: NO.
static bool g_fuse_corner_fill = false;

// Milestone 3A.15 SS12: run as SIX SEPARATE MPI PROCESSES, one cubed-sphere tile
// per rank, exchanging halos through the host-staged MPI transport instead of the
// in-process HaloPeerGroup. Opt-in only; false is the six-host-thread path
// unchanged.
//
// This is the decomposition real production uses (FV_NX=1, FV_NY=6), and it is the
// configuration the single-process path could never represent: HaloPeerGroup
// copies between raw device pointers valid in one address space only.
//
// The tile is the rank, +1 for FV3's 1-based tiles -- the identity mapping SS2
// confirmed against CubeHalo.F90. The transport does not trust that: it gathers
// the rank->tile map across the communicator and refuses a duplicate or a
// mismatch with the installed topology.
static bool g_mpi_halo = false;
static int g_mpi_rank = 0;   // valid only when g_mpi_halo
static int g_mpi_tile = 1;   // == g_mpi_rank + 1

// Prompt 11: fuse pack_edge_kernel/unpack_edge_kernel's per-edge loop (4
// launches/field) into one launch/field covering all four TileEdge values --
// see fv3_gpu_halo_exchange.cu (pack_edge4_kernel/unpack_edge4_kernel).
// Numerical semantics changed: NO.
static bool g_fuse_halo_edges = false;

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
  // inside its k loop, as fv_real device-upload buffers. Milestone 3A.11V Task 2:
  // these are filled from fv3::gpu::damping::translate_damping(), not from a
  // pattern invented here.
  std::vector<fv_real> lv_nord_k, lv_ndif, lv_nord_w, lv_d2_bg, lv_d4_bg;
  std::vector<fv_real> lv_damp_v, lv_damp_w, lv_damp_t, lv_d_con, lv_kgb;
  // The translated column and flags, retained in double precision so the
  // stability report printed at config time reports on the same numbers the
  // translation produced rather than on their fv_real round-trip. Also the
  // scalars NumericalConfig needs, so a scalar cannot disagree with the column.
  fv3::gpu::damping::DampingFlags tr_flags;
  fv3::gpu::damping::DampingColumn tr_col;
  int tr_nord = 0;
  int tr_nord_t = 0;
  fv_real tr_dddmp = 0;
  fv_real tr_d_con = 0;

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

  // Milestone 6A.5 Section 10: when set (sp0_state_6a5::load(), gated by
  // --sp0-state), the authoritative Fortran GLOBAL da_min/da_min_c
  // (fv_grid_utils.F90 global_mx/global_mx_c over all 6 tiles, identical by
  // construction) override production_config()'s per-tile-local derivation.
  // Negative is the "not set" sentinel -- production_config()'s existing
  // per-tile-local min(area)/min(area_c) derivation is unchanged when unset,
  // so the synthetic/performance path is unaffected.
  fv_real override_da_min = -1;
  fv_real override_da_min_c = -1;

  /* Milestone 3A.11T diagnostic switch.  When set (FV3_BENCH_ORTHO_METRIC=1)
   * the grid metric collapses to the exactly orthogonal case
   * cos_sg == 0, sin_sg == 1 that fv_grid_utils.F90:492-493 uses for
   * grid_type >= 3.  The default remains a small-angle NON-orthogonal metric,
   * which is what a grid_type == 0 cubed sphere actually presents; the switch
   * exists so that "is the residual growth driven by the angle terms at all?"
   * can be answered without rebuilding. */
  bool ortho_metric;

  /* Milestone 3A.11T diagnostic scale on del6_u/del6_v only
   * (FV3_BENCH_DEL6_SCALE).  1.0 is the authoritative value for this geometry
   * (fv_grid_utils.F90:787/:794 with dx == dyc); before 3A.11T the harness used
   * 0.5, i.e. half the diffusion coefficient the grid implies.  The switch
   * exists so the effect of that magnitude alone can be separated from the
   * angle-metric repair it shipped with. */
  fv_real del6_scale;
};

/* --- Milestone 3A.11T §7 damping bisection --------------------------------
 *
 * FV3_BENCH_DAMP selects which damping families are live, so the residual
 * multi-step growing mode can be attributed to one of them without touching
 * production code.  Default (unset) is the full production-coverage set.  This
 * is a DIAGNOSTIC control on benchmark INPUT only; it disables no guard and
 * changes no kernel.
 *
 *   none       every damping family off
 *   nodivdamp  divergence damping off (d2_bg / d4_bg / dddmp), del6 flux on
 *   nodel6     del6_vt_flux family off (damp_v / damp_w / damp_t), divg on
 *   noheat     d_con / n_con off (the KE-dissipation heating path)
 *
 * The per-level arrays must be zeroed inside make_host_state, NOT in
 * production_config: setup() runs fv3_gpu_initialize_state (the H2D of every
 * registered field, the lv_* damping controls included) BEFORE install_config,
 * so a mutation made at config time would never reach the device.
 */
/* FV3_BENCH_DIVDAMP bisects WITHIN the divergence-damping family, which the
 * FV3_BENCH_DAMP=nodivdamp leg proved is the sole necessary ingredient:
 *   d2only   d4_bg := 0            -- second-order arm only
 *   d4only   d2_bg := 0            -- high-order arm only
 *   nosmag   dddmp := 0            -- sw_core.F90:1407-1411 takes `vort := 0`
 *                                     instead of calling smag_corner
 *   nord0    nord_k := 0 (all k)   -- pure branch L, sw_core.F90:1277-1358
 *   nord1    nord_k := 1 (all k)   -- branch H, one n-sweep, fill_c never fires
 *   nord2    nord_k := 2 (all k)   -- branch H, two n-sweeps, fill_c fires at n=1
 */
struct DampSelection {
  bool no_divdamp, no_del6, no_heat;
  bool d2only, d4only, nosmag;
  int force_nord;  // -1 = leave the per-level pattern alone
  /* FV3_BENCH_D4_SCALE multiplies every level's d4_bg. Milestone 3A.11T: an
   * audit showed the CUDA divg_d nord loop is index-exact against
   * sw_core.F90:1359-1423, so the nord1-PASS / nord2-FAIL split is not a
   * transcription defect in that loop. The remaining candidate is that the
   * bench's (da_min, d4_bg) pair sits above the del-2n stability limit for
   * TWO sweeps but below it for one -- note dd8 = da_min*d4_bg**(nord+1) is
   * actually SMALLER at nord=2 (4.7e7) than nord=1 (2.9e8), so magnitude alone
   * is not the discriminator; the two-sweep operator's larger spectral radius
   * is. Scaling d4_bg separates the two explanations: if nord2 becomes stable
   * at a reduced d4_bg with the operator unchanged, the configuration is
   * marginal (harness), not the port.
   *
   * MILESTONE 3A.11V -- ANSWERED: the configuration is marginal, not the port.
   * The del-2n limit was re-derived term-by-term from the Fortran statements as
   *
   *     gamma_d4 = (8*d4_bg)**(nord+1) * (da_min/area_local),   unstable > 2
   *
   * (divergence_corner contributes 2 PER DIRECTION, not 2 total; an earlier
   * estimate here counted a 1-D mode and was exactly 2x too small, which is why
   * the "gamma_d == 2.055" prediction below was contradicted). Critical d4_bg is
   * 2**(1/(nord+1))/8 = 0.2500 / 0.17678 / 0.15749 for nord 0/1/2. This table
   * pairs d4_bg = 0.16 with nord_k = 2 at k%4 == 1, i.e. 1.6% OVER the limit.
   *
   * Confirmed by a PARAMETER-FREE threshold measurement at two different nord,
   * on a seam-free IC (FV3_BENCH_HORIZ_SCALE=0), classified by functional form:
   *     nord=1  predicted s* in [1.105, 1.127]   measured s* in (1.10, 1.15]
   *     nord=2  predicted s* in [0.984, 0.998]   measured s* in (0.98, 1.00]
   * The competing "ported operator is effectively too strong" hypothesis needed
   * s* = 1.221 at nord=1 and is refuted. Growth is +0.0685 per acoustic substep
   * and dt-INDEPENDENT (constant to +-2.5% over a 24x dt range via n_split
   * 1..24), so it is not a CFL/timestep instability.
   *
   * The authoritative code never forms this pairing: FV_StateMod.F90:681 sets
   * nord=2 while :683 sets d4_bg_top=0.12, giving gamma_d4 = 0.885 (2.26x
   * margin). Raw fv_arrays.F90 defaults pair 0.16 with nord=1, gamma = 1.638.
   * The (0.16, nord_k=2) pairing is an invention of this harness.
   *
   * Note the general limit has a SECOND, additive term on the nord>0 branch:
   * sw_core.F90:1418 applies damp2*delpc + dd8*divg_d, contributing
   * +8*max(d2_bg, min(0.20, dddmp*|div*dt|))*(da_min_c/area_c). It does not
   * affect the thresholds above because d2_bg == 0.0 at exactly the k%4 == 1
   * group that is critical, and a quiescent seam-free run leaves the
   * Smagorinsky part ~0. It WOULD matter for any other group or a divergent
   * flow. Per-group gamma at nord=2, s=1: 1.018 / 2.035 / 1.603 / 0.497. */
  fv_real d4_scale;
  /* FV3_BENCH_DAMPV_SCALE multiplies every level's damp_v, i.e. the del^6
   * WIND-damping coefficient, leaving nord_v and every other damping arm alone.
   *
   * Why a separate knob from FV3_BENCH_DEL6_SCALE: del6_u/del6_v are consumed by
   * all three del6 arms (damp_v at wp_b_dsw_cuda.cu:1238, damp_w at :1020,
   * damp_t at :1044), so scaling the metric moves three coefficients at once and
   * also fabricates a metric no grid corresponds to. This scales exactly one
   * table column.
   *
   * What it tests. For a checkerboard mode on this uniform fabricated metric
   * (dx == dy == dxc == dyc == 1e5, rarea == 1/dx^2, da_min_c == 9.900108e9,
   * del6_u == del6_v == 1) each del^6 sweep in del6_vt_flux contributes -8*rarea
   * and the final flux stage -2, damp4 == (damp_v*da_min_c)**(nord_v+1), and the
   * consumer applies *rdx (wp_b_divergence_cuda.cu:534-560). The per-substep
   * forward-Euler amplification is therefore
   *
   *     gamma_v = 4 * damp_v**(nord_v+1) * 8**nord_v
   *
   * which needs gamma_v <= 2. The fabricated table at :874/:878 gives
   *
   *     k%4 == 1  damp_v 0.10  nord_v 2  ->  0.256   stable
   *     k%4 == 2  damp_v 0.05  nord_v 1  ->  0.080   stable
   *     k%4 == 3  damp_v 0.20  nord_v 2  ->  2.048   UNSTABLE
   *
   * The nord_v == 2 ceiling is damp_v <= (2/256)**(1/3) == 0.198, so 0.20 is over
   * it. That is the same order as FV3's documented 0.16 divergence-damping
   * ceiling, which is a nord == 1 limit and does not carry to nord == 2.
   *
   * This is a diagnostic, not the fix. If a reduced damp_v is stable with the
   * operator order untouched, the finding is that the HARNESS table pairs a
   * high nord_v with a coefficient above that pairing's stability limit. */
  fv_real dampv_scale;
};

DampSelection damp_selection() {
  DampSelection d{false,        false, false, false,
                  false,        false, -1,    (fv_real)1.0,
                  (fv_real)1.0};
  if (const char *sel = std::getenv("FV3_BENCH_DAMP")) {
    const bool none = std::strcmp(sel, "none") == 0;
    d.no_divdamp = none || std::strcmp(sel, "nodivdamp") == 0;
    d.no_del6 = none || std::strcmp(sel, "nodel6") == 0;
    d.no_heat = none || std::strcmp(sel, "noheat") == 0;
  }
  if (const char *sel = std::getenv("FV3_BENCH_DIVDAMP")) {
    d.d2only = std::strcmp(sel, "d2only") == 0;
    d.d4only = std::strcmp(sel, "d4only") == 0;
    d.nosmag = std::strcmp(sel, "nosmag") == 0;
    if (std::strcmp(sel, "nord0") == 0) d.force_nord = 0;
    if (std::strcmp(sel, "nord1") == 0) d.force_nord = 1;
    if (std::strcmp(sel, "nord2") == 0) d.force_nord = 2;
  }
  if (const char *s = std::getenv("FV3_BENCH_D4_SCALE")) {
    const double v = std::atof(s);
    /* Reject <= 0 rather than silently accepting it: 0 would be d2only under a
     * different name and a negative value would flip the damping's sign, both
     * of which would be read as "the scale test", which they are not. */
    if (v > 0.0) d.d4_scale = (fv_real)v;
  }
  if (const char *s = std::getenv("FV3_BENCH_DAMPV_SCALE")) {
    const double v = std::atof(s);
    /* Same reasoning as d4_scale: 0 is "no del^6 wind damping" under a different
     * name and a negative value reverses the damping's sign. Both would be read
     * as a coefficient measurement and are not one. */
    if (v > 0.0) d.dampv_scale = (fv_real)v;
  }
  return d;
}

/* Milestone 3A.11U: a scale factor on the INITIAL wind field only (uc, vc, u,
 * v, ua, va). Diagnostic, not a configuration knob -- it exists to separate two
 * explanations of the monotone delp growth measured at the owned cube corner
 * (ie, js) on all six tiles, at both 8x8 and 16x16, from the very first
 * acoustic substep:
 *
 *   PORT DEFECT   -- delp changes even with the state at rest. d_sw's delp
 *                    update (sw_core.F90:1013) is strict flux form,
 *                    delp = wk + (fx(i)-fx(i+1)+fy(j)-fy(j+1))*rarea, and with
 *                    zero winds every flux is identically zero, so ANY change
 *                    to delp at WIND_SCALE=0 is arithmetic that does not belong
 *                    there, independent of the state.
 *   STATE-DRIVEN  -- delp is bit-constant at rest and the growth rate scales
 *                    with the wind amplitude. Then the carrier is the synthetic
 *                    field itself: it is defined in LOCAL index space
 *                    (x = 0.29*i, y = 0.23*j) identically on all six tiles and
 *                    carries a large mean (u = +12, v = -7), so it is not a
 *                    single globally consistent vector field on the cube. The
 *                    D-grid seam transform then correctly delivers a halo u of
 *                    about -10 against an owned +12 -- a real ~22 m/s shear at
 *                    every panel seam, compounding where two seams meet.
 *
 * 0.0 is accepted and is the whole point, so unlike d4_scale/dampv_scale this
 * does NOT reject zero. Negative is rejected: it would reverse the flow, which
 * is a different experiment that would read as this one.
 *
 * READ THE FOLLOWING BEFORE USING WIND_SCALE=0 AS A REST STATE. The dichotomy
 * above is INCOMPLETE, and reading it as complete produced a wrong conclusion
 * that had to be retracted. WIND_SCALE=0 gives zero WIND, and nothing more. It
 * does NOT give a horizontally uniform state, because the SCALARS carry their
 * own local-index horizontal structure which this knob never touched:
 * horiz_pert on pt/delz/zh/gz, the metric `horiz` on area/rarea_c/f0, and the
 * standalone patterns on ptc, z_rat, omga and heat_source. With those live,
 * p_grad_c produces nonzero uc from a real gz gradient, d_sw's fluxes are
 * legitimately nonzero, and delp legitimately changes -- so "delp changed at
 * WIND_SCALE=0" does NOT imply a port defect. Use FV3_BENCH_HORIZ_SCALE=0
 * together with this knob for an actually uniform state. */
fv_real bench_wind_scale() {
  if (const char *s = std::getenv("FV3_BENCH_WIND_SCALE")) {
    const double v = std::atof(s);
    if (v >= 0.0) return (fv_real)v;
  }
  return (fv_real)1.0;
}

/* Scales the AMPLITUDE of the horizontal shape function in the synthetic initial
 * state. Defaults to 1.0.
 *
 * 3A.11V Task 1 CHANGED WHAT THIS KNOB MEANS. It no longer switches off a seam
 * discontinuity, because there no longer is one: every horizontal perturbation
 * is now cube::flow_shape(center(i,j)), a smooth single-valued function of
 * position on the real gnomonic cubed sphere, so it is seam-consistent at any
 * amplitude including 1.0. What the knob still does is set how strong the
 * large-scale horizontal structure is, and 0.0 still collapses the horizontal to
 * exactly uniform, which is what the exact-rest invariant below needs.
 *
 * So the ORIGINAL justification for the knob's existence is now obsolete and the
 * knob is retained for the exact-rest control only. Everything from "WHY THIS
 * KNOB HAS TO EXIST" down to the blank line before the exact-rest paragraph is
 * HISTORICAL: it records the defect that motivated Task 1 and the measurement
 * that proved it was IC forcing rather than a port defect. Do not read it as a
 * description of the current state.
 *
 * Two caveats that do survive Task 1, because they are not about the seam:
 *   - The fabricated uniform 100 km metrics are unchanged, so the state is
 *     globally consistent but NOT in discrete balance. A smooth O(1) forcing
 *     remains at scale 1.0 and this knob still scales it.
 *   - area, rarea_c, f0 and f_c still carry a (1 + horiz) factor built from the
 *     same shape function, so this knob perturbs the METRICS too, not only the
 *     state. That is unchanged from before and is documented at those sites.
 *
 * --- HISTORICAL, describes the pre-3A.11V IC ---------------------------------
 *
 * WHY THIS KNOB HAS TO EXIST. Each of those perturbations is a function of the
 * per-tile LOCAL array indices, seeded identically on all six tiles. That is
 * not a single-valued field on a cubed sphere: the value seeded into tile T's
 * ring-1 halo is not the value the neighbour tile owns at the cell that halo
 * shadows, so the state has a genuine discontinuity across all twelve panel
 * edges. The seam exchange then CORRECTLY transports the neighbour's differing
 * value inward, and the resulting cliff is real input, not a port defect.
 *
 * This was measured, and the agreement is what makes it certain rather than
 * plausible. At WIND_SCALE=0 the ring-1 surface pkc overshoot was +4182.3 Pa on
 * tiles 1,3,5 at (i=2,j=0) and +1211.4 Pa on tiles 2,4,6 at (i=0,j=1). Both
 * LOCATIONS follow from seam_along_index plus the odd/even parity rule (odd
 * tiles are aligned on SOUTH, even tiles on WEST), and the RATIO follows from
 * r^gama - 1 with r = (1+hp_owned)/(1+hp_ghost) and gama = 1/(1-akap) = 1.4:
 * predicted 3.45464 against measured 4182.3132/1211.4443 = 3.45227, i.e. 0.069%.
 * Both endpoints of the cube geometry and the thermodynamics are therefore
 * accounted for, leaving nothing for a kernel defect to explain.
 *
 * --- end HISTORICAL ----------------------------------------------------------
 *
 * At HORIZ_SCALE=0 with WIND_SCALE=0 the state becomes exactly uniform in the
 * horizontal and exactly at rest, and THEN the exact-rest invariant is a real
 * test: every horizontal difference is exactly 0.0, p_grad_c's numerator is
 * A*B + (-A)*B which is exactly 0.0 because IEEE subtraction is exactly
 * antisymmetric, every flux is exactly 0, and every field must stay
 * bit-constant forever. Any change at that point IS a port defect.
 *
 * Note the limitation, so it is not claimed as coverage: at HORIZ_SCALE=0 and
 * WIND_SCALE=0 all six tiles are identical and every exchanged quantity is
 * uniform, so this leg is BLIND to neighbour-mapping, stagger and sign defects
 * -- a wrong seam map returns values indistinguishable from the right one.
 * Those need the separate seam-sensitive test.
 *
 * 0.0 is accepted and is the point. Negative is rejected as a different
 * experiment that would read as this one. */
fv_real bench_horiz_scale() {
  if (const char *s = std::getenv("FV3_BENCH_HORIZ_SCALE")) {
    const double v = std::atof(s);
    if (v >= 0.0) return (fv_real)v;
  }
  return (fv_real)1.0;
}

// Milestone 3A.11V Task 1: the state is now built on the ACTUAL gnomonic
// cubed-sphere geometry, so every horizontal field is a smooth single-valued
// function of position on the sphere rather than a function of the tile-local
// array index. `tile` is therefore a required argument: it selects which part of
// the sphere this state occupies, which is the whole point.
//
// What that buys, and what it does not:
//   - It removes the O(1) discontinuity the old local-index patterns carried
//     across all twelve panel seams. Verified by fv3_cube_seam_test: the
//     analytic scalar agrees across every seam to 4.4e-16 and the D-grid
//     components to 0.0/7.8e-14 over thousands of halo edges.
//   - It does NOT make the state balanced. The harness keeps its fabricated
//     uniform 100 km metrics (see the metric block below and the header of
//     fv3_cube_geometry.hpp for why), so a smooth large-scale imbalance remains.
//     "Globally consistent" and "balanced" are different claims; only the first
//     is made here.
//
// Tiles now differ because they occupy different parts of the sphere, which is
// why make_tile_asymmetric() is gone -- see the note at its former site.
HostState make_host_state(int tile) {
  HostState h{};
  // NI == NJ is enforced by the CLI (both come from --grid), and CubeGrid is
  // square by construction. Assert rather than silently mis-index.
  if (NI != NJ) {
    printf("  FAIL: make_host_state requires NI == NJ (got %lld x %lld)\n",
           (long long)NI, (long long)NJ);
    std::abort();
  }
  const cube::CubeGrid cg((int)NI, tile, (int)HALO);
  const cube::FlowParams flow{};
  // Array index -> FV3 index. The harness allocates lo_3d = 1 - HALO = -2, so
  // array slot 0 is FV3 index -2 and owned index i lives at slot i + HALO - 1.
  auto fv = [](int64_t a) { return (int)(a - (int64_t)HALO + 1); };
  /* Scales every INITIAL velocity component -- uc, vc, u, v, ua, va and the
   * vertical w. w is included deliberately: at ws == 0 the intent is a state
   * genuinely at rest, and a nonzero w alone would keep the nonhydrostatic
   * solver evolving, which would make a "delp changed at rest" reading
   * ambiguous. Defaults to 1.0, so the default configuration is unchanged. */
  const fv_real ws = bench_wind_scale();
  {
    const char *e = std::getenv("FV3_BENCH_ORTHO_METRIC");
    h.ortho_metric = (e != nullptr && e[0] == '1');
    const char *s = std::getenv("FV3_BENCH_DEL6_SCALE");
    h.del6_scale = (s != nullptr) ? (fv_real)std::atof(s) : (fv_real)1.0;
  }
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

  /* Vertical grid: LOG-SPACED in pressure, pe(k) = ptop*(psurf/ptop)^(k/npz).
   *
   * Milestone 3A.11T. The uniform-delp column this replaces was thermodynamically
   * balanced -- the derivation below is unchanged and still correct -- but
   * geometrically pathological, and that was a SECOND, independent harness
   * defect. Uniform delp from 100 Pa to 100000 Pa puts a pressure ratio of 100.9
   * across the topmost layer ALONE. Measured by tests/host/
   * check_bench_column_balance.cpp, the balanced thicknesses then run -890 m at
   * the surface to -37810 m at the top -- a 42.5:1 spread with a single 37.8 km
   * layer -- and the interface pk increment that nh_p_grad scales its wind
   * increment by spans 12.8:1 over the column. Empirically that column is not
   * integrable: uc/vc grew 15 -> 64 m/s within one acoustic substep and the
   * D-grid Riemann solver went non-finite at step 2, while the C-grid stages
   * stayed perfectly clean through step 0 (delpc 9974..9996, ptc 10.72..31.51,
   * flat across all three substeps). So the amplifier was the vertical grid, not
   * any C-grid stage -- and thermodynamic balance alone is not sufficient.
   *
   * Log spacing is the well-conditioned choice for a nonhydrostatic solver:
   * uniform d(ln p) means uniform delz. With H = RT/g ~ 8 km the layers come out
   * -3774 m at the top and -3895 m at the bottom, a 1.0:1 spread, and the pk
   * increment spans 3.3:1. ptop moves to 1000 Pa (10 hPa), a plausible top for a
   * 10-level column, and the total depth is ln(100)*H ~ 38.3 km -- which is what
   * a real atmosphere does between 10 hPa and the surface.
   *
   * This is a HARNESS grid choice. No production CUDA numerics and no
   * production guard were touched to accommodate it. */
  h.ptop = 1000.0;
  const fv_real psurf = 100000.0;
  h.grav = 9.80665;
  /* FV3_BENCH_GRAV -- DIAGNOSTIC ONLY. It exists because the CUDA path is
   * internally inconsistent about g in a way the authoritative Fortran cannot be:
   *   - Riem_Solver3's precompute hardcodes T(9.8) (fv3_vertical_column.hpp:703),
   *     consumed as rgrav = 1/grav scaling dm (:161,:175), and
   *     fv3_vertical_kernels.cu never references grav at ALL, so cfg.grav does
   *     not reach the D-grid Riemann path;
   *   - Riem_Solver_c uses cfg.grav (fv3_riem_solver_c.cu:404) and so does
   *     gz = grav*zh (wp_d_contract_cuda.cpp:42).
   * In nh_core.F90:100 / nh_utils.F90:348 both read the SAME constants_mod grav,
   * so the Fortran is self-consistent at whatever value that is (NOT VERIFIED
   * here: FMS constants_mod is not on this filesystem, and FMS has historically
   * used 9.80 rather than 9.80665, so which side is faithful is undetermined).
   * The CUDA path runs TWO gravities inside one acoustic substep regardless.
   *
   * Why this knob is the clean test: the balanced column below is derived with
   * h.grav (:717) and its SIM1 invariant (:622) is stated in terms of it, and
   * cfg.grav = h.grav (:1411). Setting this to 9.8 therefore makes the IC, the
   * runtime config and the hardcoded riem3 value all agree -- the one
   * configuration in which an exact rest state should be an exact fixed point of
   * Riem_Solver3. Predicted effect if the inconsistency is the mechanism: the
   * spurious first-substep w at the vertical_solver stage falls from ~9.2e-02 to
   * roundoff, because pe stops being inflated by (9.80665/9.8)^gama - 1 = 9.5e-4
   * of pm2 (i.e. 1.2 Pa at k=0 rising to 76 Pa at k=9, largest at the bottom
   * level where the mode localizes).
   *
   * This is NOT a fix. It changes only the harness; the production inconsistency
   * at fv3_vertical_column.hpp:703 is untouched, and a rest state preserved
   * under this knob is evidence about the mechanism, not restored correctness. */
  if (const char *s = std::getenv("FV3_BENCH_GRAV")) {
    const double v = std::atof(s);
    /* Reject <= 0 rather than silently accepting: rgrav = 1/grav, so 0 divides
     * and a negative g inverts the column, and either would still be read as a
     * "consistent-gravity measurement", which it would not be. */
    if (v > 0.0) {
      h.grav = (fv_real)v;
      std::fprintf(stderr,
                   "[BENCH] FV3_BENCH_GRAV=%.9g -> IC + cfg.grav "
                   "(riem3 precompute stays hardcoded 9.8)\n",
                   (double)h.grav);
    }
  }
  h.akap = fv_real(2.0 / 7.0);
  // See kAkapUnset above. Applied here, before the balanced-column derivation,
  // so pk/pkz/cappa stay self-consistent with whichever akap is in force.
  if (g_akap > 0.0) {
    h.akap = (fv_real)g_akap;
    if (tile == 1) {
      std::fprintf(stderr, "[6B] akap override: 2/7=%.17g -> %.17g\n",
                   2.0 / 7.0, (double)h.akap);
    }
  }
  // Must equal cfg.rdgas below: the balanced column is derived with it, and the
  // solver evaluates the same invariant with cfg's value.
  const double kRdgas = kFv3Rdgas;

  // Interface pressures, and the layer thicknesses they imply. delp is NOT
  // uniform any more -- every consumer below reads delp_col[k].
  std::vector<double> pe_face((size_t)NPZ + 1), delp_col((size_t)NPZ);
  {
    const double lr = std::log((double)psurf / (double)h.ptop) / (double)NPZ;
    for (int64_t k = 0; k <= NPZ; ++k)
      pe_face[(size_t)k] = (double)h.ptop * std::exp(lr * (double)k);
    pe_face[(size_t)NPZ] = (double)psurf; // pin the surface face exactly
    for (int64_t k = 0; k < NPZ; ++k)
      delp_col[(size_t)k] = pe_face[(size_t)k + 1] - pe_face[(size_t)k];
  }

  /* ------------------------------------------------------------------------
   * Milestone 3A.11T. A BALANCED column, derived rather than asserted.
   *
   * SIM1_solver's first statement (nh_utils.F90:1232) is
   *
   *   pe(i,k) = exp(gama*log(-dm2/dz2*rgas*pt2)) - pm2      gama = 1/(1-akap)
   *
   * where dm2 = delp/grav, dz2 = delz < 0 and pm2 is the layer's log-mean
   * pressure (Riem_Solver_c:396). `pe` is a PERTURBATION: Riem_Solver_c adds
   * the hydrostatic component back at :417. So the state must satisfy
   *
   *   X ^ gama == pm2,   X = (rdgas/grav) * delp / (-delz) * pt
   *
   * and dyn_core.F90:1075 raises the SAME X to akap/(1-akap) to get pkz, so
   * the relation also fixes pkz = pm2^akap, the Exner function.
   *
   * The state this harness used through 3A.11S violated that by two compounding
   * factors and was therefore unintegrable:
   *
   *   1. pt was seeded as an ABSOLUTE temperature, 280..290 K. It is not one.
   *      fv_dynamics.F90:552 executes `pt(i,j,k) = pt(i,j,k)/pkz(i,j,k)`
   *      BEFORE calling dyn_core, so every pt inside dyn_core -- including
   *      Riem_Solver_c's `pt` dummy at nh_utils.F90:333 -- is a virtual
   *      temperature divided by the Exner function. For this column that is
   *      order 11 at the surface and 31 at the top, not 280.
   *   2. delz was seeded as a uniform -1000 m to match an assumed 10 km deep
   *      zh column. The ideal-gas law does not permit that alongside uniform
   *      delp: the balanced thicknesses run -890 m at the surface to -37810 m
   *      in the topmost layer, and the column is 56.8 km deep, not 10 km.
   *
   * Measured consequence, by tests/host/check_bench_column_balance.cpp:
   * X^gama/pm2 ran 83 at k=9 to 3490 at k=0, i.e. SIM1 saw a perturbation
   * pressure of 7.8e6 Pa -- 77 ATMOSPHERES -- on the very first call, at every
   * level, before a single step was integrated. That is what pef/pkc carried
   * into p_grad_c; the resulting uc/vc increments are orders of magnitude
   * beyond anything physical, the D-grid transport drove delp down (owned
   * interior 9990 -> 3175 Pa over six steps, NOT edge-confined), and once the
   * first delpc went negative SIM1's log argument went negative -> NaN. The
   * authoritative Fortran has no guard there and would NaN identically; this
   * was never a port defect.
   *
   * So: keep delp uniform and keep a physical temperature profile, then let the
   * two authoritative relations above determine delz, pt and pkz, and build
   * zh/gz from that delz. The horizontal signal p_grad_c needs now rides on
   * TEMPERATURE, which is how a real baroclinic state carries it -- pe/pk stay
   * horizontally uniform (they must, being an exact pk = pe^akap pair) while
   * gz acquires a genuine horizontal gradient through delz. The previous
   * `(1.0 + horiz)` factors on pk and gz are gone: they perturbed the answer
   * without perturbing the state it was derived from, which is just a second
   * imbalance.
   * ---------------------------------------------------------------------- */
  std::vector<double> pm2_col(NPZ), temp_col(NPZ), pkz_col(NPZ);
  for (int64_t k = 0; k < NPZ; ++k) {
    const double pe_top = pe_face[(size_t)k];
    const double pe_bot = pe_face[(size_t)k + 1];
    pm2_col[k] = (pe_bot - pe_top) / std::log(pe_bot / pe_top);
    temp_col[k] = 280.0 + 10.0 * ((double)k / (double)NPZ);
    pkz_col[k] = std::exp((double)h.akap * std::log(pm2_col[k]));
  }
  // Horizontal temperature perturbation: +-2%, about +-5.7 K, a realistic
  // gradient. It must be nonzero or p_grad_c's two terms cancel exactly in
  // exact arithmetic and the stage cannot be distinguished from a no-op.
  // Scaled by FV3_BENCH_HORIZ_SCALE.
  //
  // 3A.11V Task 1: this was `0.02*sin(0.35*i)*cos(0.35*j)` on the tile-local
  // array index, identical on all six tiles, and therefore discontinuous across
  // every panel seam -- the forcing whose consequence was measured at +4182.3 Pa
  // / +1211.4 Pa of ring-1 pkc overshoot. It is now the Williamson-2 shape
  // function S(lon,lat) evaluated at the cell's TRUE position on the cubed
  // sphere, which is single-valued by construction. Amplitude is unchanged at
  // +-2% (about +-5.7 K) because S is in [-1,1] exactly as the old sin*cos was.
  // It reaches pt, delz, zh and gz below.
  const double hscale = (double)bench_horiz_scale();
  auto horiz_pert = [&](int64_t ai, int64_t aj) {
    return hscale * 0.02 * cube::flow_shape(cg.center(fv(ai), fv(aj)), flow);
  };

  // Two further INDEPENDENT global shapes, both in [-1,1] like flow_shape.
  //
  // The reason these exist rather than reusing flow_shape everywhere: the fields
  // that consumed the old tile-local patterns used DIFFERENT patterns on purpose
  // (`sin(x+y)` for ptc against `sin(x-y)` for z_rat), so that a product or
  // difference of two of them could not cancel identically and hide a
  // transcription error. Making every field proportional to one shape would throw
  // that away.
  //
  // The reason it is free to do this: ANY smooth function of the TRUE (lon,lat)
  // is single-valued on the sphere, so independence between patterns costs
  // nothing and cannot reintroduce a seam discontinuity. Both are finite at the
  // poles (the cos(lat) / sin(2*lat) factors vanish there) and both are 2*pi
  // periodic in lon, so there is no branch-cut artifact at the lon = +-pi seam
  // that global_corner's wrap_pi produces.
  auto shape2 = [&](int64_t ai, int64_t aj) {
    const cube::LonLat p = cg.center(fv(ai), fv(aj));
    return std::sin(2.0 * p.lon) * std::cos(p.lat);
  };
  auto shape3 = [&](int64_t ai, int64_t aj) {
    const cube::LonLat p = cg.center(fv(ai), fv(aj));
    return std::cos(p.lon) * std::sin(2.0 * p.lat);
  };

  for (size_t idx = 0; idx < h.size_3d; ++idx) {
    // idx = i + j*ni_h + k*ni_h*nj_h, so the level is the quotient. delp is
    // per-level now, so it cannot be hoisted out of the loop.
    const size_t k = idx / (size_t)(h.ni_h * h.nj_h);
    h.delp[idx] = (fv_real)delp_col[k];
    h.w[idx] = (fv_real)(ws * (fv_real)0.01);
    h.q_con[idx] = 0.001;
    h.cappa[idx] = h.akap;
    h.delpc[idx] = (fv_real)delp_col[k];
  }
  /* pt, delz, pkz and the zh/gz column, all from the one invariant.
   *
   * The 3A.7 note this replaces was right about the MECHANISM -- delz and the
   * zh/gz column must agree, because gz_from_delz_kernel rebuilds gz's interior
   * from delz while gz's halo keeps whatever the harness seeded, and H_GZ_FIRST
   * resolves only at it==1, so any mismatch leaves a geopotential cliff one
   * cell outside each tile edge for update_dz_c to difference across. It was
   * wrong only in fixing the disagreement by asserting a uniform -1000 m on
   * both sides. Deriving both from delp/pm2/T satisfies that consistency
   * requirement AND the thermodynamic one at the same time.
   *
   * Loop over (i,j,k) explicitly. The flat `idx % NPZ` this replaces was not a
   * vertical index at all -- with idx = i + j*ni_h + k*ni_h*nj_h and
   * ni_h = 15, `idx % 10` walks a diagonal through the box, so the intended
   * 280..290 K vertical profile was actually being laid down horizontally. */
  std::vector<double> zh_col((size_t)h.npz1);
  for (int64_t j = 0; j < h.nj_h; ++j) {
    for (int64_t i = 0; i < h.ni_h; ++i) {
      const double hp = horiz_pert(i, j);
      // Build the layer thicknesses for this column, then integrate zh upward
      // from the surface so zh and delz agree by construction.
      zh_col[(size_t)NPZ] = 0.0; // zs = 0 everywhere in this harness
      for (int64_t k = NPZ - 1; k >= 0; --k) {
        const double T = temp_col[k] * (1.0 + hp);
        const double rho = pm2_col[k] / (kRdgas * T);
        const double dz = -delp_col[(size_t)k] / (rho * (double)h.grav);
        const size_t idx = (size_t)(i + j * h.ni_h + k * h.ni_h * h.nj_h);
        h.delz[idx] = (fv_real)dz;
        h.pt[idx] = (fv_real)(T / pkz_col[k]);
        h.pkz[idx] = (fv_real)pkz_col[k];
        zh_col[(size_t)k] = zh_col[(size_t)(k + 1)] - dz;
      }
      for (int64_t k = 0; k <= NPZ; ++k) {
        const size_t idx = (size_t)(i + j * h.ni_h + k * h.ni_h * h.nj_h);
        h.zh[idx] = (fv_real)zh_col[(size_t)k];
        /* gz is seeded in METRES, matching zh, NOT as a geopotential.
         *
         * gz carries two different unit conventions inside one substep, and the
         * seed belongs to the first:
         *   dyn_core.F90:433-455 (the it==1 fill)  gz(npz+1)=zs; gz(k)=gz(k+1)-delz(k)
         *       -- no grav, so METRES, since zs is a height and delz is in metres.
         *   update_dz_c (nh_utils.F90:188-203)     ws=(zs-gz(km+1))*rdt -- metres.
         *   Riem_Solver_c tail (nh_utils.F90:421-426)  gz(km+1)=hs;
         *       gz(k)=gz(k+1)-dz2(k)*grav -- converts to grav*height on OUTPUT,
         *       per its own comment "Compute Height * grav (for p-gradient
         *       computation)", because p_grad_c consumes a geopotential.
         * dyn_core.F90:552-560 then copies zh = gz over the full box, so zh is
         * metres too and the two seeds must agree.
         *
         * The previous grav* factor made them disagree by 9.80665. It was mostly
         * invisible because the it==1 fill overwrites gz's interior and the edge
         * exchange overwrites rings 1-3 -- but neither reaches the four DIAGONAL
         * halo corner blocks, so exactly those cells kept a value 9.8x too large.
         * That is the root of the gz/zh xgrav corner recurrence, which had
         * previously only been characterised as a device-residency artefact.
         *
         * Numerical semantics changed: NO for every cell any consumer reads in
         * this configuration; the corrected cells are ones the authoritative
         * Fortran never reads either (it reallocates gz per init_step). */
        h.gz[idx] = (fv_real)zh_col[(size_t)k];
      }
    }
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
        // 3A.11V Task 1: was sin/cos of 0.29*i, 0.23*j, 0.17*k -- a tile-local
        // index pattern, identical on all six tiles, discontinuous at every
        // seam. Now the solid-body wind projected onto this tile's C-grid
        // stagger, which is a genuinely global tangent vector field and so
        // actually exercises the halo exchange's vector sign/swap transform
        // (fv3_gpu_halo_topology.hpp seam_vector_transform) instead of feeding
        // it a field no transform could make consistent.
        //
        // The k-dependence the old pattern carried is retained as a mild shear
        // so the levels are not degenerate copies of one another; without it a
        // vertical-index defect could not change the answer.
        const double shear = 1.0 + 0.03 * (double)k;
        h.uc[idx] = (fv_real)(ws * shear * cube::cgrid_uc(cg, fv(i), fv(j), flow));
        h.vc[idx] = (fv_real)(ws * shear * cube::cgrid_vc(cg, fv(i), fv(j), flow));
      }
    }
  }
  // Interface pressures. zh/gz are NOT set here any more -- they come from the
  // balanced delz above, which is the only way the two can agree. pk is an
  // exact pk = pe^akap pair with pe and carries no horizontal perturbation:
  // the pressure column is identical in every column of this state, and the
  // horizontal signal p_grad_c needs lives in gz (hence in temperature).
  for (size_t k = 0; k < (size_t)h.npz1; ++k) {
    fv_real p = (fv_real)pe_face[k];
    for (size_t j = 0; j < (size_t)h.nj_h; ++j) {
      for (size_t i = 0; i < (size_t)h.ni_h; ++i) {
        size_t idx = i + j * h.ni_h + k * h.ni_h * h.nj_h;
        // pe/peln are (i, k, j): k is the MIDDLE axis, so it strides by ni_h
        // and j strides by ni_h*npz1. Compare `idx` above, which is the
        // ordinary (i,j,k). Seeding through this index rather than `idx` is
        // what makes the host state agree with the registered descriptor.
        const size_t idx_pe = i + k * (size_t)h.ni_h +
                              j * (size_t)h.ni_h * (size_t)h.npz1;
        h.pe[idx_pe] = p;
        h.peln[idx_pe] = std::log(p);
        h.pk[idx] = std::pow(p, h.akap);
        h.pk3[idx] = h.pk[idx];
        h.pkc[idx] = h.pk[idx];
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
      // Scaled by FV3_BENCH_HORIZ_SCALE: at 0 the metrics become exactly
      // uniform, which is what makes the exact-rest invariant rigorous rather
      // than approximate -- a horizontally varying area alone is enough to make
      // the (gz2*area)/area round-trip in update_dz_c_update_gz round
      // differently per cell, which shows up as a hash change at ulp level.
      // 3A.11V Task 1: was `0.02*sin(0.31*i)*cos(0.31*j)` -- a tile-local index
      // pattern, identical on all six tiles, hence NOT single-valued across a
      // seam. area and rarea_c matter here: rarea multiplies the mass-flux
      // divergence in d_sw, so a seam-discontinuous area is a discontinuity in
      // the divergence operator itself, sitting exactly where the damping
      // operators are most sensitive.
      //
      // Replaced by the same global shape function used for the state. This is
      // safe for the del-2n stability calibration in a way that replacing dx/dy/
      // area outright would NOT be: the AMPLITUDE is unchanged. sin*cos and
      // flow_shape both range over [-1,1], so area still spans 1e10*(1 +- 0.02)
      // and da_min/area is unchanged to within that same 2%. The shape's only
      // requirement here was "small and not perfectly uniform" (see above), and
      // a globally consistent small perturbation satisfies that strictly better
      // than a tile-local one at no cost.
      //
      // Contrast edge_w/e/s/n below, which are deliberately NOT converted --
      // there the authoritative value is a function of real grid spacing this
      // harness does not have, so there is nothing faithful to write.
      const fv_real horiz = (fv_real)bench_horiz_scale() * 0.02 *
                             (fv_real)cube::flow_shape(cg.center(fv(i), fv(j)),
                                                       flow);
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
      h.divg2[idx] = 0.0;
      h.dxc[idx] = 100000.0;
      h.dyc[idx] = 100000.0;
      h.rdxa[idx] = 1.0 / 100000.0;
      h.rdya[idx] = 1.0 / 100000.0;
      h.f0[idx] = 1.0e-4 * (1.0 + 0.1 * horiz / 0.02); // Coriolis, mid-latitude
      /* --- Milestone 3A.11T: authoritative metric assembly ---------------
       *
       * Every angle-like grid metric is now DERIVED from cos_sg using the
       * exact relations in fv_grid_utils.F90, instead of each being given an
       * independent made-up value.  The previous block set
       *   rsin2 = rsina = rsin_u = rsin_v = 1.0   exactly
       * while simultaneously setting cosa_u/cosa_v/cosa/cosa_s to nonzero
       * values up to 0.05, and paired sin_sg = 1-0.01c with
       * cos_sg = 0.03-0.005c.  That is not a metric: it violates
       * sin_sg**2 + cos_sg**2 == 1 (:430) by up to 3%, and it violates
       * rsin_u == 1/sina_u**2 (:579), rsin_v == 1/sina_v**2 (:587),
       * rsin2 == 1/sin_sg5**2 (:595) and rsina == 1/sina**2 (:612).  The
       * cube-vertex corner algebra (divergence_corner's corner-term removal,
       * fill_4corners, a2b_ord4's extrap_corner) is derived assuming those
       * identities hold, so feeding it a non-metric leaves it with no
       * restoring geometry.
       *
       * The angles stay small and spatially varying, so the deliberate
       * coverage the old comment defended is preserved: the contravariant
       * transforms (S1/K1) and d2a2c_vect's edge_interpolate4 arms still
       * multiply by nonzero cosa/cos_sg, and an ordering error there is still
       * visible.  What changes is only that the values are now mutually
       * consistent.
       *
       * FV3 sg component numbering is 1..9; the CUDA views carry the four
       * cell-edge components 1..4 as 0-based 0..3.  sg5 (A-grid cell centre)
       * and sg6/sg8 (B-grid corner) are not stored as fields -- they enter
       * only through cosa_s/rsin2 and cosa/rsina -- so representative small
       * values are used for them here.
       */
      const fv_real sg_scale = 1.0 + 0.5 * horiz / 0.02;  // |sg_scale-1| <= 0.5
      fv_real cs[6];
      for (int c = 0; c < 4; ++c)                      // sg1..sg4
        cs[c] = (fv_real)(0.03 - 0.005 * (fv_real)c) * sg_scale;
      cs[4] = (fv_real)0.030 * sg_scale;               // sg5, cell centre
      cs[5] = (fv_real)0.028 * sg_scale;               // sg6/sg8, cell corner
      if (h.ortho_metric)                              // exact flat metric,
        for (int c = 0; c < 6; ++c) cs[c] = 0.0;       // fv_grid_utils.F90:492-493
      auto sn = [](fv_real c) {
        return std::sqrt(std::max((fv_real)0.0, (fv_real)1.0 - c * c));
      };                                               // :430
      for (int c = 0; c < 4; ++c) {
        h.cos_sg[idx + (size_t)c * h.size_2d] = cs[c];
        h.sin_sg[idx + (size_t)c * h.size_2d] = sn(cs[c]);
      }
      // :576-577 / :579 -- sg3 and sg1 are bench components 2 and 0.
      const fv_real cosa_u = 0.5 * (cs[2] + cs[0]);
      const fv_real sina_u = 0.5 * (sn(cs[2]) + sn(cs[0]));
      h.cosa_u[idx] = cosa_u;
      h.sina_u[idx] = sina_u;
      h.rsin_u[idx] = 1.0 / (sina_u * sina_u);
      // :584-585 / :587 -- sg4 and sg2 are bench components 3 and 1.
      const fv_real cosa_v = 0.5 * (cs[3] + cs[1]);
      const fv_real sina_v = 0.5 * (sn(cs[3]) + sn(cs[1]));
      h.cosa_v[idx] = cosa_v;
      h.sina_v[idx] = sina_v;
      h.rsin_v[idx] = 1.0 / (sina_v * sina_v);
      // :593 / :595
      h.cosa_s[idx] = cs[4];
      h.rsin2[idx] = 1.0 / (sn(cs[4]) * sn(cs[4]));
      // :563-564 / :612
      h.cosa[idx] = cs[5];
      h.rsina[idx] = 1.0 / (sn(cs[5]) * sn(cs[5]));
      // :786-787 / :793-794 -- with dx == dyc == dxc == dy == 1e5 the length
      // ratios are 1, so these reduce to the sines.  del6_u was 0.5 before,
      // which corresponded to no grid at all.
      h.divg_u[idx] = sina_v * h.dyc[idx] * h.rdx[idx];
      h.del6_u[idx] = h.del6_scale * sina_v * h.dx[idx] / h.dyc[idx];
      h.divg_v[idx] = sina_u * h.dxc[idx] * h.rdy[idx];
      h.del6_v[idx] = h.del6_scale * sina_u * h.dy[idx] / h.dxc[idx];
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
      // grid_lonlat/agrid_lonlat: trailing dim 0=lon, 1=lat, radians.
      //
      // 3A.11V Task 1: these were `0.01*i` and `0.01*j` -- the array index in
      // disguise, identical on all six tiles, and not lon/lat of anything. They
      // are now the TRUE gnomonic corner and cell-centre coordinates of this
      // tile, including the production 10-degree westward shift, so a consumer
      // that computes anything from them (a Coriolis term, a metric, a
      // diagnostic) now gets a coherent cubed sphere instead of six identical
      // patches of a fictional plane.
      const cube::LonLat gll = cg.corner(fv(i), fv(j));
      const cube::LonLat all_ = cg.center(fv(i), fv(j));
      /* Prompt 6B: LAYOUT FIX. These two arrays were filled INTERLEAVED
       * (`idx*2 + component`), but they are registered with
       *   st_ll = {1, ni_h, ni_h*nj_h}
       * i.e. component-SLOWEST, two contiguous (i,j) planes -- which is the
       * correct translation of Fortran's grid(isd:ied+1, jsd:jed+1, 2).
       * The consumer (a2b_great_circle -> View3 component index lo_k / lo_k+1)
       * therefore read lon/lat of the wrong cells, mixed together.
       *
       * Only the a2b_ord4 CORNER extrapolation reads these -- the interior,
       * edge, qx/qy/qxx/qyy paths use dxa/dya -- which is exactly why the
       * damage was confined to the four physical panel corners of the B-grid
       * output while every interior cell agreed at round-off.
       * Measured: at sp2b/it=1 the implied extrapolation ratio d1/(d2-d1) came
       * out as [0.093, -68.5, 1213.3] on tile 5 and [0.050, -2.68, 1.72] on
       * tile 1, against the authoritative 0.4212796036 (identical for all three
       * stencil terms on all six tiles, which the harness cube::CubeGrid
       * reproduces exactly -- so the geometry SOURCE was always right and only
       * the store order was wrong).
       * Numerical algorithm changed: NO. */
      h.grid_lonlat[idx] = (fv_real)gll.lon;
      h.grid_lonlat[idx + h.size_2d] = (fv_real)gll.lat;
      h.agrid_lonlat[idx] = (fv_real)all_.lon;
      h.agrid_lonlat[idx + h.size_2d] = (fv_real)all_.lat;
    }
  }
  // edge_w/e/s/n are the cubed-sphere edge-interpolation weights consumed by the
  // contravariant edge corrections. They are DELIBERATELY LEFT as fabricated
  // local-index ramps, and that has to be stated rather than left implicit,
  // because it is the one place where 3A.11V Task 1's "no tile-local index
  // seeding" rule is knowingly not applied.
  //
  // Why left alone:
  //   - These are GRID METRICS, not state. Task 1's scope is the initial state;
  //     the harness's metrics stay fabricated as a whole (dx = dy = dxc = dyc =
  //     1e5, uniform area), because the 3A.11V del-2n stability calibration is
  //     expressed in da_min/area and replacing the metrics would invalidate it.
  //     Replacing only these four while dx/dy/area stay uniform would produce a
  //     metric set that is internally LESS coherent, not more.
  //   - The authoritative values come from edge_factors / efactor_a2c_v
  //     (REF/tools/fv_grid_tools.F90), which are functions of the real grid
  //     spacing this harness does not have. There is no faithful value to write
  //     here that is consistent with a uniform 1e5 grid.
  //
  // What is given up, so it is not silently claimed: these four arrays are not
  // single-valued across a seam -- edge_w[j] on tile T and the corresponding
  // weight on its west neighbour disagree -- so the edge corrections see an O(1)
  // metric inconsistency at the panel boundaries. That is a residual forcing in
  // this benchmark. It is bounded (a smooth 1% ramp, not a grid-scale cliff) and
  // it is NOT the discontinuity that drove the multi-step instability, which was
  // in the STATE and was measured at +4182.3 Pa in ring-1 pkc. Do not read this
  // note as reopening that finding.
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
    // pkz is NOT re-seeded here. It is set from pm2^akap alongside pt and delz,
    // because dyn_core.F90:1075 derives it from the same X the SIM1 invariant
    // fixes; pkz and pt cannot be chosen independently. The value this replaces
    // was 1.0..1.1, which made pt = T/pkz numerically indistinguishable from an
    // absolute temperature and so hid the imbalance. Real pkz for this column
    // is 5.7 at the top and 26.4 at the surface.
    // 3A.11V Task 1: was `sin(0.4 * (idx % ni_h))`. Besides being tile-local and
    // seam-discontinuous, `idx % ni_h` is only the i index -- it carried no j or
    // k structure at all. Now the global shape function at the cell centre, so
    // heat_source is single-valued across seams like everything else.
    const size_t k_hs = idx / (size_t)(h.ni_h * h.nj_h);
    const size_t rem_hs = idx % (size_t)(h.ni_h * h.nj_h);
    const int64_t j_hs = (int64_t)(rem_hs / (size_t)h.ni_h);
    const int64_t i_hs = (int64_t)(rem_hs % (size_t)h.ni_h);
    (void)k_hs;
    h.heat_source[idx] =
        (fv_real)(10.0 + (double)bench_horiz_scale() * 0.5 *
                             cube::flow_shape(cg.center(fv(i_hs), fv(j_hs)), flow));
  }

  // --- Milestone 3A.8: d_sw state ------------------------------------------
  // u/v were zero through 3A.2 because only WP-D's nh_pressure wrote them, and
  // a zero start made any_nonzero() a valid "did it execute" probe. d_sw needs
  // genuinely nonzero D-grid winds -- with u == v == 0 the vorticity, kinetic
  // energy and every transport flux are identically zero, so a stage-ordering
  // defect could not change the answer. Phase B's u/v probe is therefore
  // upgraded to any_changed() against a pre-step snapshot, which is a strictly
  // stronger statement than "nonzero" ever was.
  const double hs = (double)bench_horiz_scale();
  for (int64_t k = 0; k < NPZ; ++k) {
    for (int64_t j = 0; j < h.nj_h; ++j) {
      for (int64_t i = 0; i < h.ni_h; ++i) {
        const size_t idx = (size_t)(i + j * h.ni_h + k * h.ni_h * h.nj_h);
        // 3A.11V Task 1: the prognostic D-grid winds, projected from the global
        // solid-body flow onto this tile's D-grid stagger exactly as
        // REF/tools/test_cases.F90:485-521 (init_winds, defOnGrid==5) does it.
        // Same k-shear rationale as uc/vc above. These are the fields the seam
        // test verifies edge-by-edge, and the sign/swap table it derives from
        // them from geometry alone matches the independent FMS audit on all 24
        // (tile, side) seams.
        const double shear = 1.0 + 0.03 * (double)k;
        h.u[idx] = (fv_real)(ws * shear * cube::dgrid_u(cg, fv(i), fv(j), flow));
        h.v[idx] = (fv_real)(ws * shear * cube::dgrid_v(cg, fv(i), fv(j), flow));
        double ua = 0.0, va = 0.0;
        cube::agrid_wind(cg, fv(i), fv(j), flow, &ua, &va);
        h.ua[idx] = (fv_real)(ws * shear * ua);
        h.va[idx] = (fv_real)(ws * shear * va);
        // ut/vt/crx/cry/xfx/yfx are d_sw OUTPUTS (S1 and S2 overwrite them
        // before any read), so a nonzero seed here would only mask a
        // missing write. Left at zero deliberately.
        // ptc is a DIRECT Riem_Solver_C input, and the historical record is that
        // its local-index horizontal pattern was one of the two columns whose
        // mutual inconsistency at ring 1 produced the measured pkc overshoot.
        // 3A.11V Task 1: that pattern (`sin(0.29*i + 0.23*j)`) is replaced by the
        // global shape function, so ptc is now single-valued across every seam and
        // is consistent with the pt/delz/zh/gz column, which uses the same shape.
        // Amplitude unchanged: both old and new factors are in [-1,1].
        // Scaled by FV3_BENCH_HORIZ_SCALE.
        h.ptc[idx] =
            (fv_real)(280.0 + hs * 4.0 *
                                  cube::flow_shape(cg.center(fv(i), fv(j)), flow));
        h.divgd[idx] = 0.0;
        h.diss_est[idx] = 0.0;
        // z_rat is a read-only nonhydrostatic thickness ratio that scales the
        // damping terms; 1.0 is the neutral value, perturbed so it cannot
        // cancel out of a product.
        // 3A.11V Task 1: `sin(0.29*i - 0.23*j)` -> shape2, a global function
        // independent of the one ptp/ptc use, preserving the original intent that
        // z_rat cannot cancel out of a product with them.
        h.z_rat[idx] = (fv_real)(1.0 + hs * 0.02 * shape2(i, j));
        // Milestone 3A.9. omga is BOTH d2a2c_vect's A-grid vertical velocity
        // and Riem_Solver_C's w3 argument (dyn_core.F90:583), where it is
        // intent(in) and enters the implicit column solve's forcing. Zero
        // would be an admissible physical state but a degenerate test input:
        // it would remove w3 from the Riem_Solver_C right-hand side entirely,
        // so a transcription error in that term could not change the answer.
        // Same order of magnitude as w (0.01 m/s), sign-varying in i,j,k.
        // omga is w3 in Riem_Solver_C, so at HORIZ_SCALE=0 it must go to exactly
        // zero along with the winds -- a nonzero w3 forcing would keep the
        // nonhydrostatic column solve evolving and make the rest-state reading
        // ambiguous, which is the same reason w itself is scaled by WIND_SCALE.
        // 3A.11V Task 1: `sin(0.29*i + 0.23*j - 0.11*k)` -> shape3 times a
        // separate cos(0.11*k). The k dependence is kept because it was there
        // deliberately -- omga is Riem_Solver_C's w3 and a k-uniform forcing would
        // be a weaker input -- but it now multiplies a global horizontal shape
        // instead of being folded into a local-index phase. Still exactly zero at
        // HORIZ_SCALE=0, which the rest-state leg requires.
        h.omga[idx] =
            (fv_real)(hs * 0.01 * shape3(i, j) * std::cos(0.11 * (double)k));
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
  // change those weights.
  //
  // 3A.11T: this is now simply delp_col, the log-spaced profile the state
  // actually carries. The previous `dp_uniform * (1 + 0.01*k)` fudge existed to
  // keep the dp0 ratios away from exactly 1 so a swapped index would be visible;
  // log spacing gives a ratio of 1.585 between adjacent levels on its own, so
  // the fudge is no longer needed and would now be an inconsistency.
  for (size_t k = 0; k < (size_t)NPZ; ++k)
    h.dp_ref[k] = (fv_real)delp_col[k];

  // --- The ten per-level damping controls ----------------------------------
  //
  // MILESTONE 3A.11V TASK 2. These are no longer fabricated here.
  //
  // What used to be here was a hand-invented four-level cycle
  //     nord_k {0,2,0,1}  nord_v {0,2,1,2}  nord_w {0,1,0,2}
  //     d2_bg {0.02,0,0.03,0}  d4_bg {0.12,0.16,0.14,0.10}
  //     damp_v {0,0.10,0.05,0.12}  damp_w {0,0.06,0,0.09}
  //     damp_t {0,0.08,0,0.03}  d_con {0,1.0,0.5,0}  kgb {0,0,1e-4,5e-5}
  // cycled as k % 4. It was written to maximize kernel-path COVERAGE -- every
  // LevelFilter gate live in one dgrid_dynamics invocation -- and it succeeded
  // at that. But 8 of its 10 arrays were STRUCTURALLY unreachable from the
  // authoritative configuration path, not merely mis-valued, and one of the
  // pairings it invented (d4_bg = 0.16 at nord_k = 2) violates the del-2n
  // stability bound. That single fabricated pairing produced a slow amplifying
  // mode which was, across three milestones, repeatedly suspected of being a
  // CUDA port defect. It was not. Full account:
  //   docs/agent_handoff/milestone_3A11V_dt_sensitivity.md
  //
  // The two measurements that settled it, both on a seam-free IC:
  //   d4_bg = 0.16, nord_k = 2  ->  gamma_d4 = 2.097 > 2, nonfinite at ~570
  //                                 acoustic substeps, growth +0.0685 per
  //                                 substep and dt-INDEPENDENT over a 24x dt
  //                                 range (so not a CFL failure).
  //   d4_bg = 0.12, nord_k = 2  ->  gamma_d4 = 0.885, bounded for 300 calls at
  //                                 8x8x10 (three bitwise-identical repeats),
  //                                 16x16 and 32x32.
  // The threshold was then bracketed parameter-free at two different nord and
  // matched the derived critical d4_bg of 2^(1/(nord+1))/8 in both cases.
  //
  // So the values now come from the shared translation layer, which is a
  // statement-ordered reimplementation of dyn_core.F90:677-765 with the GEOS
  // scalar overrides of FV_StateMod.F90:569-793:
  //     include/fv3_damping_translation.hpp
  //     runtime/fv3_damping_translation.cpp
  //     tests/runtime/test_damping_translation.cpp   (asserts the column)
  // The same layer is what the GEOS bridge should call, so the benchmark and
  // production cannot drift apart again.
  //
  // COVERAGE COST, stated plainly rather than glossed. The authoritative
  // npz = 10 column is 90% sponge: nord_k = 0 at k = 0..8 and nord_k = 2 only
  // at k = 9; nord_v = 0 at k = 0..7 and 2 at k = 8,9. So this benchmark now
  // exercises FEWER nord combinations than the invented cycle did -- the
  // nord == 1 arm is not reachable at all under the GEOS overrides, since
  // FV_StateMod.F90:681 sets nord = 2 and dyn_core.F90:713 takes min(2,nord).
  // That is the correct trade: this is a PERFORMANCE and STABILITY harness for
  // the production configuration, and per-arm kernel coverage belongs in the
  // focused d_sw test (tests/dgrid/test_wp_b_cuda_smoke.cpp Part I), which
  // still drives all four groups from uniform parameters. Do not restore a
  // synthetic cycle here to recover coverage; add the case there instead.
  //
  // The env knobs below (FV3_BENCH_DAMP / DIVDAMP / D4_SCALE / DAMPV_SCALE)
  // still work, and still perturb these arrays, because the diagnostic
  // discriminators of 3A.11V need to be reproducible. All of them default to
  // no-ops, and the applied table is printed unconditionally, so an acceptance
  // run with no environment set is self-evidently unmodified in its own log.
  {
    namespace dmp = fv3::gpu::damping;
    /* high_resolution selects the FV_StateMod.F90:685/:687 d4_bg_bot branch.
     * false is the npx*CEILING(stretch_fac) < 1200 case, which is what an
     * 8x8/16x16/32x32 benchmark grid is. */
    dmp::DampingFlags dflags =
        dmp::geos_production_flags((int)NPZ, (double)h.ptop,
                                   /*high_resolution=*/false);
    /* Prompt 6B Task H: authoritative n_sponge override (see
     * g_damp_n_sponge's declaration for the two-way proof that this fixture
     * runs n_sponge = -1). Sentinel-guarded: unset leaves
     * geos_production_flags()'s value untouched. */
    if (g_damp_n_sponge != kDampNSpongeUnset) {
      std::fprintf(stderr,
                   "[6B-H] n_sponge override: %d -> %d (authoritative fixture "
                   "namelist value; sponge branch %s)\n",
                   dflags.n_sponge, g_damp_n_sponge,
                   g_damp_n_sponge > 0 ? "ACTIVE" : "DISABLED");
      dflags.n_sponge = g_damp_n_sponge;
    }
    /* pm2_col is the layer-mean pressure of the column this harness built, i.e.
     * FV3's pfull(1:npz). Read only by the d4dmp() sine profile, which is
     * inactive at d4_bg_top == d4_bg_bot; passed anyway so the profile is
     * correct if a future leg sets high_resolution. */
    const dmp::DampingColumn dcol = dmp::translate_damping(dflags, pm2_col.data());
    for (size_t k = 0; k < (size_t)NPZ; ++k) {
      h.lv_nord_k[k] = (fv_real)dcol.nord_k[k];
      h.lv_ndif[k]   = (fv_real)dcol.ndif[k];
      h.lv_nord_w[k] = (fv_real)dcol.nord_w[k];
      h.lv_d2_bg[k]  = (fv_real)dcol.d2_bg[k];
      h.lv_d4_bg[k]  = (fv_real)dcol.d4_bg[k];
      h.lv_damp_v[k] = (fv_real)dcol.damp_v[k];
      h.lv_damp_w[k] = (fv_real)dcol.damp_w[k];
      h.lv_damp_t[k] = (fv_real)dcol.damp_t[k];
      h.lv_d_con[k]  = (fv_real)dcol.d_con[k];
      h.lv_kgb[k]    = (fv_real)dcol.kgb[k];
    }
    /* nord, nord_t, dddmp and d_con are scalars in NumericalConfig; carry the
     * translated values through so a scalar cannot disagree with the column
     * derived from it. Consumed where cfg is filled. */
    h.tr_flags = dflags;
    h.tr_col = dcol;
    h.tr_nord = dflags.nord;
    h.tr_nord_t = dcol.nord_t;
    h.tr_dddmp = (fv_real)dcol.dddmp;
    h.tr_d_con = (fv_real)dflags.d_con;
    const DampSelection d = damp_selection();
    for (size_t k = 0; k < (size_t)NPZ; ++k) {
      if (d.no_divdamp) {
        h.lv_d2_bg[k] = 0.0; h.lv_d4_bg[k] = 0.0; h.lv_nord_k[k] = 0.0;
      }
      if (d.no_del6) {
        h.lv_damp_v[k] = 0.0; h.lv_damp_w[k] = 0.0; h.lv_damp_t[k] = 0.0;
        h.lv_ndif[k] = 0.0;   h.lv_nord_w[k] = 0.0; h.lv_kgb[k] = 0.0;
      }
      if (d.no_heat) h.lv_d_con[k] = 0.0;
      if (d.d2only) h.lv_d4_bg[k] = 0.0;
      if (d.d4only) h.lv_d2_bg[k] = 0.0;
      if (d.force_nord >= 0) h.lv_nord_k[k] = (fv_real)d.force_nord;
      /* Last, so it scales whatever the branch selections above left behind
       * (and is a no-op on the d2only leg, where d4_bg is already 0). */
      h.lv_d4_bg[k] *= d.d4_scale;
      h.lv_damp_v[k] *= d.dampv_scale;
    }
    if (d.no_divdamp || d.no_del6 || d.no_heat)
      std::fprintf(stderr, "[BENCH] FV3_BENCH_DAMP: divdamp=%d del6=%d heat=%d\n",
                   d.no_divdamp ? 0 : 1, d.no_del6 ? 0 : 1, d.no_heat ? 0 : 1);
    if (d.d4_scale != (fv_real)1.0)
      std::fprintf(stderr,
                   "[BENCH] FV3_BENCH_D4_SCALE=%.6g -> d4_bg[0]=%.6g\n",
                   (double)d.d4_scale, (double)h.lv_d4_bg[0]);
    /* Milestone 3A.11V Task 2: mirror the env-var perturbations back into
     * h.tr_col so the stability report printed at config time reports on the
     * coefficients AS APPLIED, not on the unmodified translation. Without this
     * a diagnostic leg would print a clean table while running a perturbed one
     * -- the exact failure mode the old "print the knob, not the value" habit
     * caused twice during 3A.11. */
    for (size_t k = 0; k < (size_t)NPZ; ++k) {
      h.tr_col.nord_k[k] = (int)h.lv_nord_k[k];
      h.tr_col.ndif[k] = (int)h.lv_ndif[k];
      h.tr_col.nord_w[k] = (int)h.lv_nord_w[k];
      h.tr_col.d2_bg[k] = (double)h.lv_d2_bg[k];
      h.tr_col.d4_bg[k] = (double)h.lv_d4_bg[k];
      h.tr_col.damp_v[k] = (double)h.lv_damp_v[k];
      h.tr_col.damp_w[k] = (double)h.lv_damp_w[k];
      h.tr_col.damp_t[k] = (double)h.lv_damp_t[k];
      h.tr_col.d_con[k] = (double)h.lv_d_con[k];
      h.tr_col.kgb[k] = (double)h.lv_kgb[k];
    }
    if (d.nosmag) h.tr_col.dddmp = 0.0;
    if (d.no_del6) h.tr_col.nord_t = 0;
    /* Printed unconditionally so a leg whose env var never reached the binary is
     * visible as WIND_SCALE=1 rather than being read as a rest-state result. */
    std::fprintf(stderr, "[BENCH] WIND_SCALE=%.6g (scales initial uc,vc,u,v,ua,va,w)\n",
                 (double)ws);
    /* Same reason, and it is the more dangerous of the two to miss: WIND_SCALE=0
     * alone still leaves every scalar's horizontal structure live, so a leg
     * intended as a rest state but run without HORIZ_SCALE=0 would show
     * legitimately changing delp and read as a port defect. It did, once.
     *
     * 3A.11V: the word "local-index" is gone from this message on purpose. The
     * structure is now the global cube shape function, so a nonzero HORIZ_SCALE
     * no longer implies a seam discontinuity -- reading it that way would be
     * reading the pre-3A.11V harness. */
    std::fprintf(stderr, "[BENCH] HORIZ_SCALE=%.6g (scales the global cube shape "
                         "function on pt,delz,zh,gz,ptc,z_rat,omga,heat_source "
                         "and the area/rarea_c/f0/f_c metrics)\n",
                 (double)bench_horiz_scale());
    std::fprintf(stderr, "[BENCH] DAMPV_SCALE=%.6g  D4_SCALE=%.6g (both 1 means "
                         "the authoritative translation is used unmodified)\n",
                 (double)d.dampv_scale, (double)d.d4_scale);
    /* The full applied per-level table. Replaces the old four-row k%%4 gamma_v
     * summary, which could only describe a 4-cycle and printed a formula that
     * later turned out to be 2x too small. The stability table itself is
     * printed at config time, where da_min and da_min_c exist. */
    std::fprintf(stderr, "%s",
                 fv3::gpu::damping::format_damping_column(h.tr_col).c_str());
    if (d.d2only || d.d4only || d.nosmag || d.force_nord >= 0)
      std::fprintf(stderr,
                   "[BENCH] FV3_BENCH_DIVDAMP: d2only=%d d4only=%d nosmag=%d "
                   "force_nord=%d\n",
                   d.d2only, d.d4only, d.nosmag, d.force_nord);
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
  lo_pe[0] = lo_3d_if[0]; lo_pe[1] = 0;               lo_pe[2] = lo_3d_if[1];
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
  cfg.rdgas = kFv3Rdgas;  // authoritative FMS/GEOS value; see kFv3Rdgas
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
  /* Prompt 6B: n_con is DERIVED, per dyn_core.F90:402-412, not chosen.
   *
   *     if ( flagstruct%convert_ke .or. flagstruct%vtdm4 > 1.E-4 ) then
   *          n_con = npz
   *     else if ( d2_bg_k1 < 1.E-3 ) then ; n_con = 0
   *     else if ( d2_bg_k2 < 1.E-3 ) then ; n_con = 1
   *     else                              ; n_con = 2
   *
   * The value was previously hardcoded to 1 with a comment that justified
   * nf_ke == 1 (the corner_edge_service stub) and then set n_con, not nf_ke --
   * two independent controls conflated. n_con selects how many LEVELS of the
   * post-loop diffusive-heating pt update run (dyn_core.F90:1338 `do k=1,n_con`);
   * nf_ke selects how many del2_cubed smoothing passes heat_source receives
   * first (:1310-1311). Only nf_ke interacts with corner_edge_service.
   *
   * The fixture's namelist sets vtdm4 = 0.01 and geos_production_flags()
   * independently sets the same 0.01 from FV_StateMod.F90:744, so the first
   * branch fires on both sides and the authoritative value is npz. convert_ke
   * is irrelevant here (fv_arrays.F90:447 default .false., not overridden) but
   * is honoured below if the translation ever carries it.
   *
   * Effect at the dyn_core exit: with n_con = 1 only k=1's pt was heated, while
   * the authoritative run heats all npz levels -- the measured exit `pt`
   * signature (nonzero at all 640 owned cells, per-level max decaying
   * 4.74e-05 -> 4.88e-07) and the `pkz` signature (agreeing at k=1, diverging
   * by O(1) for k=2..npz, because :1348 rewrites pkz at every heated level).
   *
   * Numerical algorithm changed: NO -- the authoritative loop bound is now used.
   */
  cfg.n_con = (h.tr_flags.vtdm4 > 1.0e-4)
                  ? (int32_t)NPZ
                  : (h.tr_flags.d2_bg_k1 < 1.0e-3
                         ? 0
                         : (h.tr_flags.d2_bg_k2 < 1.0e-3 ? 1 : 2));
  // 3A.11V Task 2: from the translation. FV_StateMod.F90:742 sets d_con = 1.0,
  // so the value is unchanged -- but it is now sourced rather than asserted.
  cfg.d_con = h.tr_d_con;
  /* nf_ke is DERIVED from nord, per dyn_core.F90:1310
   *     nf_ke = min(3, flagstruct%nord+1)
   * (del2_cubed then applies its own `ntimes = min(3,nmax)` at :2400, so the
   * cap is expressed twice authoritatively). It is assigned AFTER the
   * damp_selection block below, because that block may still override cfg.nord.
   *
   * It was pinned to 1 while corner_edge_service was a fail-stub: passes with
   * nt = ntimes-1-pass > 0 call copy_corners, which had no implementation. That
   * stub is gone (fv3_gpu_production_dispatch.cu), so the authoritative value
   * is now reachable. */
  cfg.moist_cappa = 0;
  // See kAkapExnerUnset: the Exner-only kappa, separated from the closure kappa
  // (cfg.akap) exactly as the authoritative MOIST_CAPPA build separates them.
  if (g_akap_exner > 0.0) {
    cfg.akap_exner = g_akap_exner;
    std::fprintf(stderr,
                 "[6B] akap_exner (ptk/pk3/pk3_halo) = %.17g; closure akap "
                 "stays %.17g\n",
                 g_akap_exner, (double)cfg.akap);
  }
  cfg.fuse_corner_fill = g_fuse_corner_fill ? 1 : 0;
  cfg.fuse_halo_edges = g_fuse_halo_edges ? 1 : 0;
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
  // Milestone 3A.11V Task 2: these three came from the same invented table as
  // the per-level arrays and are now taken from the translation, so a scalar
  // cannot disagree with the column that was derived from it.
  //   nord   FV_StateMod.F90:681 sets 2, which also caps nord_cap_k at 2.
  //   nord_t dyn_core.F90:715, min(2,nord) at every level. Must be <= ng-1 = 2.
  //   dddmp  FV_StateMod.F90:682 sets 0.2. The previous value here was 0.5,
  //          chosen only to clear the 1.e-5 gate so S31 takes the smag_corner
  //          arm rather than vort := 0. 0.2 clears it just as well, so that
  //          coverage is unchanged -- but 0.2 is also what production runs, and
  //          it halves the worst-case Smagorinsky contribution to the del-2n
  //          stability metric (a saturated term needs |div*dt| >= 0.20/dddmp,
  //          i.e. 1.0 at 0.2 versus 0.4 at 0.5).
  cfg.nord = (int32_t)h.tr_nord;
  cfg.nord_t = (int32_t)h.tr_nord_t;
  cfg.dddmp = h.tr_dddmp;
  // Minimum cell / dual-cell areas. DERIVED from the fabricated metric arrays,
  // not hard-coded: fv_grid_utils.F90:809-812 defines these as
  //     call global_mx  (area,   ..., da_min,   da_max)
  //     call global_mx_c(area_c, ..., da_min_c, da_max_c)
  // i.e. da_min IS min(area) and da_min_c IS min(area_c), by construction.
  //
  // Milestone 3A.11T: the previous hard-coded 1.15e10 was not merely imprecise,
  // it was larger than this harness's own max(area) = 1.02e10, so da_min/area
  // came out > 1 at every cell where real FV3 guarantees <= 1 everywhere. That
  // inflates every (coeff*da_min)**(nord+1) damping coefficient -- the del-2n
  // divergence damping by 1.17x, deln_flux by 1.17**3 = 1.61x at nord = 2 --
  // which is the wrong direction for a stability margin and makes the bench a
  // harder test than production for a reason that has nothing to do with the
  // port. Computing the minima cannot drift when the metric fabrication changes.
  //
  // rarea_c is stored rather than area_c, so the dual-cell minimum is the
  // reciprocal of the rarea_c MAXIMUM. Zeros are skipped: the arrays span the
  // padding row/column, which the metric loop never writes.
  {
    fv_real amin = std::numeric_limits<fv_real>::max();
    fv_real rcmax = 0;
    for (size_t n = 0; n < h.area.size(); ++n)
      if (h.area[n] > 0 && h.area[n] < amin) amin = h.area[n];
    for (size_t n = 0; n < h.rarea_c.size(); ++n)
      if (h.rarea_c[n] > rcmax) rcmax = h.rarea_c[n];
    cfg.da_min = amin;
    cfg.da_min_c = (rcmax > 0) ? (fv_real)(1.0 / rcmax) : amin;
    std::fprintf(stderr, "[BENCH] da_min=%.6e da_min_c=%.6e (derived)\n",
                 (double)cfg.da_min, (double)cfg.da_min_c);
    // Milestone 6A.5 Section 10: da_min/da_min_c are GLOBAL (all-6-tile)
    // Fortran reductions (fv_grid_utils.F90:809-812's global_mx/global_mx_c),
    // not per-tile-local minima -- see milestone_6a5_authoritative_grid_
    // metrics.md Sec 7. In authoritative validation mode (h.override_da_min
    // set by sp0_state_6a5::load()), use the exported Fortran global value
    // instead of the derivation above. The synthetic/performance path (no
    // --sp0-state) is unchanged: override_da_min defaults to -1 and this
    // branch is not taken.
    if (h.override_da_min >= 0) {
      cfg.da_min = h.override_da_min;
      cfg.da_min_c = h.override_da_min_c;
      std::fprintf(stderr,
                   "[BENCH] da_min=%.6e da_min_c=%.6e (AUTHORITATIVE Fortran "
                   "global override, Milestone 6A.5 Sec 10)\n",
                   (double)cfg.da_min, (double)cfg.da_min_c);
    }
  }
  cfg.inline_q = 0;
  cfg.rot3 = 0;
  cfg.do_f3d = 0;
  cfg.do_skeb = 0;

  /* --- Milestone 3A.11T §7 damping bisection --------------------------------
   *
   * FV3_BENCH_DAMP selects which damping families are live, so the residual
   * multi-step growing mode can be attributed to one of them without touching
   * production code.  Default (unset) is the full production-coverage set
   * above.  This is a DIAGNOSTIC control on benchmark INPUT only; it disables
   * no guard and changes no kernel.
   *
   *   none       every damping family off
   *   nodivdamp  divergence damping off (d2_bg / d4_bg / dddmp), del6 flux on
   *   nodel6     del6_vt_flux family off (damp_v / damp_w / damp_t), divg on
   *   noheat     d_con / n_con off (the KE-dissipation heating path)
   */
  {
    const DampSelection d = damp_selection();
    if (d.no_divdamp) {
      cfg.dddmp = 0.0;   // < 1e-5 => the smag_corner arm is skipped, vort := 0
      cfg.nord = 0;
    }
    if (d.no_del6) cfg.nord_t = 0;
    if (d.no_heat) { cfg.d_con = 0.0; cfg.n_con = 0; }
    if (d.nosmag) cfg.dddmp = 0.0;
    // cfg.nord caps nord_cap_k, so branch H is only reachable when it is >= 1.
    if (d.force_nord >= 0) cfg.nord = d.force_nord;
  }

  /* dyn_core.F90:1310 -- nf_ke = min(3, nord+1). Derived, never hardcoded, and
   * placed here so every override of cfg.nord above is respected: nord=2
   * (production) gives 3, nord=0 (--damp none / nodivdamp / nord0) gives 1. */
  cfg.nf_ke = (int32_t)std::min(3, cfg.nord + 1);

  /* --- Milestone 3A.11V Task 3: the permanent stability table ---------------
   *
   * Printed here rather than in make_host_state because da_min and da_min_c are
   * only known once the metric arrays exist, and the metric ratio is part of
   * the norm. Printed UNCONDITIONALLY, not behind a debug flag: an unstable
   * damping column should be visible in every log, including the one nobody
   * thought to instrument. The whole report is string building over npz rows,
   * so its cost is not measurable against a 100-call benchmark.
   *
   * da_min is passed as BOTH the numerator and the local-area denominator, so
   * the ratio is exactly 1. That is the worst case over the panel, not an
   * approximation: da_min is by definition min(area), so da_min/area <= 1
   * everywhere and the ratio attains 1 at the smallest cell. A stability
   * verdict has to be taken at the least stable cell, which is that one.
   *
   * This REPORTS and never mutates. dyn_core.F90:677-765 does not clip these
   * coefficients for stability, so neither does this: silently clipping would
   * make the CUDA backend disagree with the numerical source of truth in
   * exactly the configurations where the disagreement matters. */
  {
    namespace dmp = fv3::gpu::damping;
    const dmp::StabilityReport rep =
        dmp::damping_stability(h.tr_col, (double)cfg.da_min, (double)cfg.da_min,
                               (double)cfg.da_min_c, (double)cfg.da_min_c);
    std::fprintf(stderr, "%s",
                 dmp::format_stability_report(h.tr_col, rep).c_str());
    if (rep.any_unstable_static || rep.any_unstable_other) {
      /* Loud, and on its own line, because this is the signature that three
       * milestones of 3A.11 chased as a suspected port defect. Not fatal: a
       * diagnostic leg may deliberately want an unstable column in order to
       * measure a threshold, which is exactly how the 3A.11V root cause was
       * established. An ACCEPTANCE run must not print this. */
      std::fprintf(stderr,
                   "[BENCH] *** WARNING: the damping column above is UNSTABLE by"
                   " the del-2n metric. Any growth observed in this run is"
                   " EXPECTED and is NOT evidence of a port defect. ***\n");
    }
  }
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

// HISTORICAL NOTE -- the function this describes no longer exists. Retained
// because it records what the old asymmetry knob was for and what it measured;
// the 3A.11V note that follows it explains why it was removed. Read the two
// together, and read the second as authoritative where they conflict.
//
// Per-tile asymmetry. u/v are the prognostic D-grid winds and carry no
// consistency constraint with delz/zh (unlike delp/pt, whose perturbation would
// have to be matched in the hydrostatic column), so a phase shift keyed on the
// tile is a safe way to make all six tiles genuinely different. Without this,
// six identical tiles would exchange values indistinguishable from their own and
// a wrong neighbour mapping would go unnoticed.
// This perturbation is ADDITIVE and is applied at :1599, after make_host_state has
// already returned, so it is a SECOND writer of u/v that FV3_BENCH_WIND_SCALE must
// also cover. It is scaled by the same factor for one reason: WIND_SCALE=0 has to
// mean "no wind from any source". Measured before this was scaled, WIND_SCALE=0 still
// left u over [-0.600, +0.800] and v over [-0.656, +0.800] -- exactly the 0.8 and 0.6
// amplitudes below -- so the intended rest state was never actually at rest and any
// "delp changed at rest" reading taken from it would have been unsound.
//
// Note what is given up at ws == 0, so it is not claimed as coverage: with the tile
// phase scaled away all six tiles become identical, which is precisely the situation
// the comment above warns about -- a wrong neighbour mapping cannot be detected,
// because a tile exchanging with an identical neighbour sees its own values back. The
// rest-state leg is therefore blind to seam mapping, stagger and sign defects, and
// exchanging a uniform scalar or a zero vector is invariant under any of them. Those
// need the separate seam-sensitive test, not this knob.
// 3A.11V Task 1: make_tile_asymmetric() is DELETED, not disabled. It existed to
// stop the six tiles being identical -- "a wrong neighbour mapping would go
// unnoticed", as the note above says -- by adding a tile-keyed phase shift
// 0.8*sin(0.37*tile + 0.11*i) / 0.6*cos(0.37*tile + 0.13*j) to u and v.
//
// It is now both unnecessary and actively wrong:
//   - Unnecessary: make_host_state(tile) builds each tile on its own part of the
//     real cubed sphere, so the six tiles differ by construction and differ in
//     the way the physical problem actually differs. A wrong neighbour mapping
//     now shows up as a genuine field discontinuity, not as a mismatch between
//     two arbitrary phase offsets.
//   - Actively wrong: it was a SECOND, tile-local, index-keyed writer of u and v
//     applied after make_host_state returned. It would have re-introduced
//     precisely the seam discontinuity Task 1 removes, on top of the globally
//     consistent winds, and it would have done so invisibly.
//
// The rest-state caveat recorded in that old note still stands and still needs
// the separate seam-sensitive test rather than this harness: at ws == 0 the wind
// field is zero everywhere, and exchanging a zero vector is invariant under any
// sign, swap or stagger error. What has changed is that the SCALARS are no longer
// uniform at ws == 0 -- pt, delz, zh, gz and heat_source still carry the global
// shape function -- so the rest-state leg is no longer blind to scalar seam
// mapping the way it was when all six tiles were identical.

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

// Milestone 6A.5: overwrite HostState's fabricated static grid metrics with
// the authoritative ones exported from the real Fortran gridstruct (see
// tools/grid_metrics/convert_fortran_grid_metrics.py). Called between
// make_host_state(tile) and make_fields(*h) in setup(), so this is the only
// place metric ingestion happens -- no device/kernel code changes, and the
// H2D transfer in fv3_gpu_initialize_state carries whatever h holds at that
// point regardless of where it came from.
//
// Validation-only: g_grid_metrics_dir is empty unless --grid-metrics was
// passed on the CLI, in which case the existing fabricated-metric benchmark
// path is completely unaffected.
namespace grid_metrics_6a5 {

// CUDA HostState field name -> vector pointer, single-component metrics.
// Must match tools/grid_metrics/convert_fortran_grid_metrics.py's
// SINGLE_FIELDS mapping exactly (name-for-name; the Fortran source-field
// name is recorded there, not needed here).
std::vector<std::pair<const char *, std::vector<fv_real> *>>
single_field_table(HostState &h) {
  return {
      {"dx", &h.dx}, {"dy", &h.dy}, {"dxa", &h.dxa}, {"dya", &h.dya},
      {"rdx", &h.rdx}, {"rdy", &h.rdy}, {"rdxa", &h.rdxa}, {"rdya", &h.rdya},
      {"dxc", &h.dxc}, {"dyc", &h.dyc}, {"rdxc", &h.rdxc}, {"rdyc", &h.rdyc},
      {"area", &h.area}, {"rarea", &h.rarea}, {"rarea_c", &h.rarea_c},
      {"f0", &h.f0}, {"f_c", &h.f_c},
      {"cosa_u", &h.cosa_u}, {"sina_u", &h.sina_u}, {"rsin_u", &h.rsin_u},
      {"cosa_v", &h.cosa_v}, {"sina_v", &h.sina_v}, {"rsin_v", &h.rsin_v},
      {"cosa_s", &h.cosa_s}, {"rsin2", &h.rsin2},
      {"cosa", &h.cosa}, {"rsina", &h.rsina},
      {"divg_u", &h.divg_u}, {"del6_u", &h.del6_u},
      {"divg_v", &h.divg_v}, {"del6_v", &h.del6_v},
  };
}

bool read_doubles(const std::string &path, size_t n, std::vector<double> *out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    printf("  FAIL: grid-metrics: cannot open %s\n", path.c_str());
    return false;
  }
  out->assign(n, 0.0);
  f.read(reinterpret_cast<char *>(out->data()),
         static_cast<std::streamsize>(n * sizeof(double)));
  const std::streamsize got = f.gcount();
  if (got != static_cast<std::streamsize>(n * sizeof(double))) {
    printf("  FAIL: grid-metrics: %s: read %lld bytes, expected %zu\n",
           path.c_str(), (long long)got, n * sizeof(double));
    return false;
  }
  return true;
}

// tile is 1-based (matches setup()'s convention); the exporter/converter use
// 0-based Fortran MPI rank == tile - 1.
bool load(HostState &h, int tile, const std::string &dir) {
  const std::string tile_dir = dir + "/tile" + std::to_string(tile - 1);
  const size_t n2d = (size_t)h.ni_h * (size_t)h.nj_h;

  for (auto &kv : single_field_table(h)) {
    std::vector<double> raw;
    if (!read_doubles(tile_dir + "/" + kv.first + ".bin", n2d, &raw)) return false;
    if (raw.size() != kv.second->size()) {
      printf("  FAIL: grid-metrics: %s size mismatch (file %zu, host %zu)\n",
             kv.first, raw.size(), kv.second->size());
      return false;
    }
    for (size_t i = 0; i < raw.size(); ++i) (*kv.second)[i] = (fv_real)raw[i];
  }

  for (auto &kv : std::vector<std::pair<const char *, std::vector<fv_real> *>>{
           {"sin_sg", &h.sin_sg}, {"cos_sg", &h.cos_sg}}) {
    std::vector<double> raw;
    if (!read_doubles(tile_dir + "/" + kv.first + ".bin", 4 * n2d, &raw)) return false;
    if (raw.size() != kv.second->size()) {
      printf("  FAIL: grid-metrics: %s size mismatch (file %zu, host %zu)\n",
             kv.first, raw.size(), kv.second->size());
      return false;
    }
    for (size_t i = 0; i < raw.size(); ++i) (*kv.second)[i] = (fv_real)raw[i];
  }

  printf("  [6A.5] tile %d: authoritative grid metrics loaded from %s\n",
         tile, tile_dir.c_str());
  return true;
}

}  // namespace grid_metrics_6a5

// ---------------------------------------------------------------------------
// Prompt 6B: authoritative cubed-sphere edge-interpolation weights and
// B-/A-grid lon/lat, for validation runs only.
//
// One input-fidelity defect is corrected here: edge_w/e/s/n. They are inputs to
// a2b_ord4, and in the SP1->SP2 window the only a2b_ord4 consumer is
// smag_corner, whose output reaches d_sw through damp2 -> vort -> ke and
// therefore affects the u/v update and NOTHING else. That is exactly the
// observed SP2 it=1 signature.
//
// edge_w/e/s/n were fabricated as local-index ramps 1.0 + 0.01*index by
// make_host_state, which states the fabrication outright at that site.
// Authoritative values are ~0.298..0.702; the ramp reaches 1.13, outside [0,1]
// entirely. --grid-metrics cannot override them because edge_* are not in the
// authoritative metric set (the Prompt 6A capture took gridstruct's per-cell
// 2-D metric arrays; edge_* are 1-D edge vectors). They are recomputed instead,
// from the real grid, by a direct port of edge_factors,
// REF/model/fv_grid_utils.F90:1188-1293.
//
// RETRACTED, recorded so it is not rediscovered: an earlier revision of this
// block also "fixed" make_host_state's cube::CubeGrid tile argument, on the
// theory that it passed a 0-based MPI rank to a 1-based parameter. That was
// wrong -- `tile` here is already 1-based (grid_metrics_6a5::load derives its
// directory as "tile" + (tile-1), i.e. tile 1..6 -> tile0..tile5 on disk), so
// make_host_state was correct. The `tile + 1` produced CubeGrid tile 7, whose
// panel falls through tile_rotate's `default` while corner_owner reads it as
// odd parity, and the resulting inconsistent halo mapping generated NaN in
// u/v/delp at it=1. grid_lonlat/agrid_lonlat therefore need no rebuild either.
//
// Deliberately confined to the --grid-metrics gate. The default/performance
// path keeps make_host_state's fabricated ramps and so stays bit-identical; the
// 3A.11V del-2n stability calibration was measured against those values.
//
// Numerical algorithm changed: NO. No numerical expression is touched, only the
// values fed into one.
// ---------------------------------------------------------------------------
namespace grid_geometry_6b {

// Fortran poisons edge_* with big_number outside the band it fills, so that an
// out-of-domain read fails loudly rather than silently using a plausible
// number. Reproduced here: CUDA's a2b_ord4 reads j (and i) over
// [output_j0, output_j1] = [js+1, je] = [2, npx-1] in
// launch_a2b_ord4_global_cuda -- exactly the Fortran band -- so no poisoned
// entry is ever read, and if one ever is, it will be obvious.
static const double kBigNumber = 1.0e30;

bool apply(HostState &h, int tile) {
  const int im = (int)NI;
  const int npx = im + 1;
  // `tile` is 1-based here, which is also CubeGrid's convention; see the
  // retraction note above.
  const cube::CubeGrid cg(im, tile, (int)HALO);

  // FV3 index -> harness array slot. The 2-D box and the 1-D edge vectors are
  // both allocated from lo = 1 - HALO, so owned index a lives at a - 1 + HALO.
  auto s2 = [&h](int i, int j) {
    return (size_t)(i - 1 + (int)HALO) +
           (size_t)(j - 1 + (int)HALO) * (size_t)h.ni_h;
  };
  auto s1 = [](int a) { return (size_t)(a - 1 + (int)HALO); };

  // -- grid authenticity self-check, before anything is written ------------
  // Re-derive dx/dy from CubeGrid and fit ONE radius against the authoritative
  // dx/dy that grid_metrics_6a5::load has just installed. If CubeGrid were not
  // the authoritative c8 grid, the edge weights computed from it would not be
  // authoritative either, and this whole correction would be a different set of
  // fabricated numbers. Distances are rotation-invariant and the six panels are
  // pure rotations of one another, so this proves the panel SHAPE only, not the
  // tile identity; do not read it as a check on the tile index.
  {
    double sum_r = 0.0;
    int nr = 0;
    for (int j = 1; j <= npx; ++j)
      for (int i = 1; i <= im; ++i) {
        const double u =
            cube::great_circle_dist(cg.corner(i, j), cg.corner(i + 1, j), 1.0);
        if (u > 0.0) { sum_r += (double)h.dx[s2(i, j)] / u; ++nr; }
      }
    for (int j = 1; j <= im; ++j)
      for (int i = 1; i <= npx; ++i) {
        const double u =
            cube::great_circle_dist(cg.corner(i, j), cg.corner(i, j + 1), 1.0);
        if (u > 0.0) { sum_r += (double)h.dy[s2(i, j)] / u; ++nr; }
      }
    if (nr == 0) {
      printf("  FAIL: [6B] tile %d: no usable dx/dy for the grid self-check\n",
             tile);
      return false;
    }
    const double radius = sum_r / (double)nr;
    double worst = 0.0;
    for (int j = 1; j <= npx; ++j)
      for (int i = 1; i <= im; ++i) {
        const double u = radius * cube::great_circle_dist(
                                      cg.corner(i, j), cg.corner(i + 1, j), 1.0);
        worst = std::fmax(worst, std::fabs(u / (double)h.dx[s2(i, j)] - 1.0));
      }
    for (int j = 1; j <= im; ++j)
      for (int i = 1; i <= npx; ++i) {
        const double u = radius * cube::great_circle_dist(
                                      cg.corner(i, j), cg.corner(i, j + 1), 1.0);
        worst = std::fmax(worst, std::fabs(u / (double)h.dy[s2(i, j)] - 1.0));
      }
    printf("  [6B] tile %d: grid self-check fitted radius %.6f m, "
           "max|dx,dy ratio-1| %.3e\n", tile, radius, worst);
    if (!(worst <= 1.0e-12)) {
      printf("  FAIL: [6B] tile %d: CubeGrid is not the authoritative grid "
             "(max|ratio-1| %.3e > 1e-12); edge_* would not be authoritative\n",
             tile, worst);
      return false;
    }
  }

  // grid_lonlat/agrid_lonlat are left as make_host_state built them: it uses the
  // same CubeGrid with the same (1-based) tile, so recomputing them here would
  // be a no-op.

  // -- edge_factors (fv_grid_utils.F90:1188-1293), verbatim ---------------
  for (int64_t j = 0; j < h.nj_h; ++j)
    h.edge_w[j] = h.edge_e[j] = (accumulator_real)kBigNumber;
  for (int64_t i = 0; i < h.ni_h; ++i)
    h.edge_s[i] = h.edge_n[i] = (accumulator_real)kBigNumber;

  // py/px hold the A-grid-pair spherical midpoints along each physical edge.
  // Fortran fills them over max(1,js-1)..min(npy-1,je+1); with one tile per
  // rank (is=js=1, ie=je=im) that is 1..npx-1. The A-grid points at index 0
  // and npx are halo cells, i.e. real cells of the neighbouring panel, so this
  // depends on CubeGrid's halo mapping being right as well as its panel shape.
  std::vector<cube::LonLat> pyw(npx + 1), pye(npx + 1), pxs(npx + 1),
      pxn(npx + 1);
  for (int j = 1; j <= npx - 1; ++j) {
    pyw[j] = cube::mid_pt_sphere(cg.center(0, j), cg.center(1, j));
    pye[j] = cube::mid_pt_sphere(cg.center(npx - 1, j), cg.center(npx, j));
    pxs[j] = cube::mid_pt_sphere(cg.center(j, 0), cg.center(j, 1));
    pxn[j] = cube::mid_pt_sphere(cg.center(j, npx - 1), cg.center(j, npx));
  }
  // The radius cancels in d2/(d1+d2), so unit-sphere distances are exact here
  // and no earth-radius constant needs to be introduced.
  auto ratio = [](cube::LonLat prev, cube::LonLat curr, cube::LonLat corner) {
    const double d1 = cube::great_circle_dist(prev, corner, 1.0);
    const double d2 = cube::great_circle_dist(curr, corner, 1.0);
    return d2 / (d1 + d2);
  };
  for (int j = 2; j <= npx - 1; ++j) {
    h.edge_w[s1(j)] =
        (accumulator_real)ratio(pyw[j - 1], pyw[j], cg.corner(1, j));
    h.edge_e[s1(j)] =
        (accumulator_real)ratio(pye[j - 1], pye[j], cg.corner(npx, j));
    h.edge_s[s1(j)] =
        (accumulator_real)ratio(pxs[j - 1], pxs[j], cg.corner(j, 1));
    h.edge_n[s1(j)] =
        (accumulator_real)ratio(pxn[j - 1], pxn[j], cg.corner(j, npx));
  }

  printf("  [6B] tile %d: authoritative edge_w[2..%d] =", tile, npx - 1);
  for (int j = 2; j <= npx - 1; ++j)
    printf(" %.12f", (double)h.edge_w[s1(j)]);
  printf("\n");
  return true;
}

}  // namespace grid_geometry_6b

// Milestone 6A.5 Section 10: load the authoritative Fortran Prompt 6A SP0
// dyn_core dynamical state (u,v,w,delp,pt,delz,q_con,pe,peln,pk) plus the
// global da_min/da_min_c scalars, in place of whatever IC HostState would
// otherwise hold. Reads flat binaries pre-remapped into the harness's own
// index convention by tools/savepoints/convert_sp0_to_cuda.py (which itself
// reuses tools/savepoints/savepoint_reader.load_field() -- no new savepoint
// format). Gated by --sp0-state; empty (default) leaves HostState/
// production_config() completely unaffected.
namespace sp0_state_6a5 {

std::vector<std::pair<const char *, std::vector<fv_real> *>>
field_table(HostState &h) {
  return {
      {"u", &h.u}, {"v", &h.v}, {"w", &h.w}, {"delp", &h.delp},
      {"pt", &h.pt}, {"delz", &h.delz}, {"q_con", &h.q_con},
      {"pk", &h.pk}, {"pe", &h.pe}, {"peln", &h.peln},
  };
}

bool read_doubles_into(const std::string &path, std::vector<fv_real> *out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    printf("  FAIL: sp0-state: cannot open %s\n", path.c_str());
    return false;
  }
  std::vector<double> raw(out->size(), 0.0);
  f.read(reinterpret_cast<char *>(raw.data()),
         static_cast<std::streamsize>(raw.size() * sizeof(double)));
  const std::streamsize got = f.gcount();
  if (got != static_cast<std::streamsize>(raw.size() * sizeof(double))) {
    printf("  FAIL: sp0-state: %s: read %lld bytes, expected %zu\n",
           path.c_str(), (long long)got, raw.size() * sizeof(double));
    return false;
  }
  for (size_t i = 0; i < raw.size(); ++i) (*out)[i] = (fv_real)raw[i];
  return true;
}

// tile is 1-based (matches setup()'s convention); the converter's output
// directories are 0-based Fortran-pe-indexed, same convention as
// grid_metrics_6a5::load().
bool load(HostState &h, int tile, const std::string &dir) {
  const std::string tile_dir = dir + "/tile" + std::to_string(tile - 1);

  for (auto &kv : field_table(h)) {
    if (!read_doubles_into(tile_dir + "/" + std::string(kv.first) + ".bin", kv.second))
      return false;
  }

  std::vector<double> scalars(2, 0.0);
  {
    std::ifstream f(tile_dir + "/da_min_da_min_c.bin", std::ios::binary);
    if (!f) {
      printf("  FAIL: sp0-state: cannot open %s/da_min_da_min_c.bin\n", tile_dir.c_str());
      return false;
    }
    f.read(reinterpret_cast<char *>(scalars.data()), 2 * sizeof(double));
    if (f.gcount() != 2 * (std::streamsize)sizeof(double)) {
      printf("  FAIL: sp0-state: %s/da_min_da_min_c.bin: short read\n", tile_dir.c_str());
      return false;
    }
  }
  h.override_da_min = (fv_real)scalars[0];
  h.override_da_min_c = (fv_real)scalars[1];

  printf("  [6A.5-Sec10] tile %d: authoritative SP0 state + global "
         "da_min=%.6e da_min_c=%.6e loaded from %s\n",
         tile, (double)h.override_da_min, (double)h.override_da_min_c,
         tile_dir.c_str());
  return true;
}

}  // namespace sp0_state_6a5

// Prompt 6B (SP1 rerun): overwrite h.ptop and h.phis/h.zs with the
// authoritative Fortran SP1-savepoint-derived boundary values (see
// g_sp1_boundary_dir's comment above). Gated by --sp1-boundary; empty
// (default) leaves HostState's synthetic seed completely unaffected.
namespace sp1_boundary_6b {

bool load(HostState &h, int tile, const std::string &dir) {
  const std::string tile_dir = dir + "/tile" + std::to_string(tile - 1);

  double ptop_val = 0.0;
  {
    std::ifstream f(tile_dir + "/ptop.bin", std::ios::binary);
    if (!f) {
      printf("  FAIL: sp1-boundary: cannot open %s/ptop.bin\n", tile_dir.c_str());
      return false;
    }
    f.read(reinterpret_cast<char *>(&ptop_val), sizeof(double));
    if (f.gcount() != (std::streamsize)sizeof(double)) {
      printf("  FAIL: sp1-boundary: %s/ptop.bin: short read\n", tile_dir.c_str());
      return false;
    }
  }
  h.ptop = (fv_real)ptop_val;

  if (!sp0_state_6a5::read_doubles_into(tile_dir + "/phis.bin", &h.phis)) return false;
  if (!sp0_state_6a5::read_doubles_into(tile_dir + "/zs.bin", &h.zs)) return false;

  /* dp_ref == update_dz_c's `dp0` dummy (dyn_core.F90:597). It sets
   * top_ratio/bot_ratio/int_ratio, i.e. the xfx/yfx wind-extrapolation weights
   * at every k INCLUDING k=km+1 -- the level that produces gz(km+1) and hence
   * ws3 (nh_utils.F90:96-98, :113-137). HostState seeds it from the synthetic
   * log-spaced delp column (:1382), which for this fixture gave
   * bot_ratio=0.613137 against the authoritative 0.500000000000 and biased ws3
   * by a uniform ~6.93% on all six tiles. dp_ref belongs to neither earlier
   * ingestion milestone's field set (not a grid metric, not one of SP0's 10
   * dynamical fields), which is why it stayed synthetic until now.
   * Authoritative values come from the converter; nothing is derived here. */
  if (!sp0_state_6a5::read_doubles_into(tile_dir + "/dp_ref.bin", &h.dp_ref)) return false;

  const size_t km = h.dp_ref.size();
  const double top_ratio = (km >= 2)
      ? (double)h.dp_ref[0] / ((double)h.dp_ref[0] + (double)h.dp_ref[1]) : 0.0;
  const double bot_ratio = (km >= 2)
      ? (double)h.dp_ref[km - 1] / ((double)h.dp_ref[km - 2] + (double)h.dp_ref[km - 1]) : 0.0;
  printf("  [6B-SP1] tile %d: authoritative SP1-boundary ptop=%.6f phis/zs "
         "loaded, dp_ref[0]=%.6f dp_ref[%zu]=%.6f top_ratio=%.12f "
         "bot_ratio=%.12f from %s\n", tile, (double)h.ptop, (double)h.dp_ref[0],
         km - 1, (double)h.dp_ref[km - 1], top_ratio, bot_ratio, tile_dir.c_str());
  return true;
}

}  // namespace sp1_boundary_6b

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
  if (g_cuda_graph) {
    int grc = fv3_gpu_set_cuda_graph_mode(m.ctx, 1, &err);
    if (grc != FV3_GPU_SUCCESS) {
      printf("  FAIL: tile %d fv3_gpu_set_cuda_graph_mode returned %d: %s\n", tile, grc, err.message);
      return false;
    }
  }

  // 3A.11V Task 1: the tile index now enters the IC here, at the ONE place that
  // builds the state, and it enters as geometry (which part of the sphere this
  // panel occupies) rather than as an additive tile-keyed phase. There is no
  // longer a second post-hoc writer of u/v; see the note at
  // make_tile_asymmetric's former site.
  m.h = std::make_unique<HostState>(make_host_state(tile));
  if (perturb_this_tile) perturb(*m.h);
  if (!g_grid_metrics_dir.empty()) {
    if (!grid_metrics_6a5::load(*m.h, tile, g_grid_metrics_dir)) return false;
    // Strictly after the metric load: grid_geometry_6b's authenticity
    // self-check validates CubeGrid against the authoritative dx/dy that load
    // has just installed, and only then writes edge_*/grid_lonlat.
    if (!grid_geometry_6b::apply(*m.h, tile)) return false;
  }
  // Milestone 6A.5 Section 10: strictly after grid metrics, strictly before
  // make_fields() -- same "override host buffers before H2D registration"
  // idiom grid_metrics_6a5::load() already uses. Load order required by
  // Section 4: metrics -> global scalars -> SP0 dynamical state, all before
  // any device transfer.
  if (!g_sp0_state_dir.empty()) {
    if (!sp0_state_6a5::load(*m.h, tile, g_sp0_state_dir)) return false;
  }
  // Prompt 6B (SP1 rerun): strictly after SP0 state, strictly before
  // make_fields() -- ptop must be set before install_config() reads it
  // (production_config()), and phis/zs must be set before make_fields()
  // captures their pointers for H2D registration.
  if (!g_sp1_boundary_dir.empty()) {
    if (!sp1_boundary_6b::load(*m.h, tile, g_sp1_boundary_dir)) return false;
  }
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
  //
  // Milestone 3A.15 SS12: the ONLY difference between the six-thread and the
  // six-process configurations. Everything above and below -- state, geometry,
  // config, topology, dispatch -- is identical, which is what makes the SS14
  // bitwise comparison of the two paths' dyn_core exits a test of the transport
  // rather than of two different setups.
#if defined(FV3_DYN_CORE_ENABLE_MPI_HALO)
  if (g_mpi_halo) {
    // The communicator crosses the C ABI as its Fortran INTEGER handle; see
    // fv3_gpu_communication.h for why MPI_Comm itself cannot.
    const int rc_mpi = fv3_gpu_attach_mpi_halo_transport(
        m.ctx, MPI_Comm_c2f(MPI_COMM_WORLD), tile, &err);
    if (rc_mpi != FV3_GPU_SUCCESS) {
      printf("  FAIL: tile %d fv3_gpu_attach_mpi_halo_transport returned %d: %s\n",
             tile, rc_mpi, err.message);
      return false;
    }
    if (fv3_gpu_halo_transport_kind_of(m.ctx) !=
        FV3_GPU_HALO_TRANSPORT_MPI_HOST_STAGED) {
      // Verified, not assumed: a silent fallback to the in-process transport
      // would make a six-process run report success while exchanging nothing.
      printf("  FAIL: tile %d asked for the MPI halo transport but the context "
             "reports kind %d\n", tile, fv3_gpu_halo_transport_kind_of(m.ctx));
      return false;
    }
  } else
#endif
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
  // Transport-agnostic: this is the same implementation the public
  // fv3_gpu_detach_halo_transport exposes (fv3_gpu_runtime.cpp:1592-1597), so it
  // releases whichever transport is attached. No MPI-specific branch is needed,
  // and none is added -- but note the ordering obligation it inherits: every rank
  // must be past its last exchange before any rank detaches.
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
    step.bdt = g_bdt;
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
  // Guard on prev_ev: with warmup == 0 no step has run, so prev_ev is still the
  // null event and the wait fails. That used to abort the repeat before a single
  // fv3_gpu_dyn_core_step call, and the only symptom was "did not complete
  // cleanly" with no per-stage diagnostics at all -- indistinguishable, from the
  // log, from a mid-loop numerical failure.
  if (prev_ev != 0 &&
      fv3_gpu_event_wait(m.ctx, prev_ev, &err) != FV3_GPU_SUCCESS)
    return -1.0;

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

// Milestone 6A.5 Section 10: exactly the ten Prompt 6A SP0 dynamical-state
// fields, for the D2H readback that proves device state (not just host
// parsing) after sp0_state_6a5::load() + H2D, with no kernel launched.
fv3_gpu_field_mask sp0_field_mask() {
  fv3_gpu_field_mask mask{};
  mask.abi_version = FV3_GPU_ABI_VERSION;
  mask.struct_bytes = sizeof(mask);
  for (FieldId id : {FieldId::u, FieldId::v, FieldId::w, FieldId::delp,
                     FieldId::pt, FieldId::delz, FieldId::q_con,
                     FieldId::pe, FieldId::peln, FieldId::pk})
    mask.words[static_cast<uint32_t>(id) / 64] |=
        (1ULL << (static_cast<uint32_t>(id) % 64));
  return mask;
}

// Milestone 6A.5 Section 10: write the current (post-D2H) HostState SP0
// fields to <dir>/tile<pe>/<field>.bin, float64, same flat layout
// tools/savepoints/convert_sp0_to_cuda.py produces -- so
// tools/savepoints/compare_sp0_identity.py can classify_diff() them directly
// against the Fortran SP0 savepoint without a third data format.
bool dump_sp0_state(const HostState &h, int tile, const std::string &dir) {
  const std::string tile_dir = dir + "/tile" + std::to_string(tile - 1);
  std::error_code ec;
  std::filesystem::create_directories(tile_dir, ec);
  const std::vector<std::pair<const char *, const std::vector<fv_real> *>> table = {
      {"u", &h.u}, {"v", &h.v}, {"w", &h.w}, {"delp", &h.delp},
      {"pt", &h.pt}, {"delz", &h.delz}, {"q_con", &h.q_con},
      {"pk", &h.pk}, {"pe", &h.pe}, {"peln", &h.peln},
  };
  for (auto &kv : table) {
    std::vector<double> raw(kv.second->begin(), kv.second->end());
    std::ofstream f(tile_dir + "/" + std::string(kv.first) + ".bin", std::ios::binary);
    if (!f) {
      printf("  FAIL: sp0-dump: cannot open %s/%s.bin for write\n",
             tile_dir.c_str(), kv.first);
      return false;
    }
    f.write(reinterpret_cast<const char *>(raw.data()),
            (std::streamsize)(raw.size() * sizeof(double)));
  }
  printf("  [6A.5-Sec10] tile %d: post-H2D/D2H SP0 state dumped to %s\n",
         tile, tile_dir.c_str());
  return true;
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
    step.bdt = g_bdt;
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

// Milestone 3A.11V: scan the OWNED box only.
//
// The harness allocates every 3-D field over [1-HALO, NI+HALO] x same x [0,npz),
// i.e. ni_h = NI + 2*HALO + 1, so owned index i in [1, NI] lives at array
// offset i - (1 - HALO) = i + HALO - 1. Halo cells are deliberately excluded
// from the positivity criterion for two reasons:
//
//   1. The four outer cube-corner halo blocks have no owner anywhere on the
//      cube, so no exchange ever writes them. Whatever the harness seeded is
//      what stays there; judging the model state by those cells would report
//      initialization, not evolution.
//   2. gz/pk3 are deliberately poisoned with huge_r each substep (restored in
//      Milestone 3A.11-series work to match FV3), so "the whole array is
//      physical" is not a property this state has by design.
//
// "delp > 0 over the owned box" is the criterion the acceptance matrix asks
// for and the one that is physically meaningful.
template <typename T>
bool check_positive_owned(const T *data, int64_t ni_h, int64_t nj_h, int64_t npz,
                          const char *name) {
  const int64_t off = (int64_t)HALO - 1; // owned i -> array index i + off
  size_t bad = 0;
  double worst = 1.0e300;
  int64_t wi = 0, wj = 0, wk = 0;
  for (int64_t k = 0; k < npz; ++k)
    for (int64_t j = 1; j <= NJ; ++j)
      for (int64_t i = 1; i <= NI; ++i) {
        const double v =
            (double)data[(size_t)((k * nj_h + (j + off)) * ni_h + (i + off))];
        if (!(v > 0.0)) {
          ++bad;
          if (!(v >= worst)) { worst = v; wi = i; wj = j; wk = k; }
        }
      }
  if (bad > 0) {
    printf("  %s: FAIL positivity (%zu owned cells <= 0; worst %.6e at "
           "i=%lld j=%lld k=%lld)\n",
           name, bad, worst, (long long)wi, (long long)wj, (long long)wk);
    return false;
  }
  printf("  %s: positive over owned box (%lldx%lldx%lld)\n", name,
         (long long)NI, (long long)NJ, (long long)npz);
  return true;
}

// SS16: verify the timed work was not skipped/optimized away. Runs a D2H of
// the resident mask (outside all timing) and reports finiteness plus
// max|delta| against the H2D-uploaded initial value captured by setup().
// Returns false if the state did not change or is non-finite.
//
// Milestone 3A.11V widened this from {u, delp} finite to the full set the
// acceptance matrix names: delp positivity over the owned box, and finiteness
// of the representative winds (u, v) and scalars (delp, pt, w, delz). Before
// this change a run could report PASS with a NaN in pt or a negative delp,
// because neither was inspected. gz/pk3/pkc are intentionally NOT checked --
// see check_positive_owned's note 2 on the deliberate huge_r poison.
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
  char label[40];
  bool ok = true;
  const struct { const std::vector<fv_real> *v; const char *n; } fin[] = {
      {&m.h->u, "u"}, {&m.h->v, "v"}, {&m.h->delp, "delp"},
      {&m.h->pt, "pt"}, {&m.h->w, "w"}, {&m.h->delz, "delz"},
  };
  for (const auto &f : fin) {
    std::snprintf(label, sizeof(label), "tile%d %s", tile, f.n);
    ok &= check_finite(f.v->data(), f.v->size(), label);
  }
  std::snprintf(label, sizeof(label), "tile%d delp", tile);
  ok &= check_positive_owned(m.h->delp.data(), m.h->ni_h, m.h->nj_h, NPZ, label);
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

// Milestone 6A.5 Section 10: load order is exactly metrics -> global scalars
// -> SP0 dynamical state -> H2D/bind (all inside phase_e::setup()) -> verify
// device state (D2H via fv3_gpu_sync_to_host below) -> STOP. No
// fv3_gpu_dyn_core_step call anywhere in this function -- cgrid predictor,
// d_sw, and the vertical solver are never invoked.
bool run_sp0_verify_only() {
  phase_e::HaloPeerGroup pg(phase_e::kTiles);
  phase_e::Member m[phase_e::kTiles + 1];
  bool setup_ok = true;
  for (int t = 1; t <= phase_e::kTiles && setup_ok; ++t)
    setup_ok = phase_e::setup(m[t], t, pg, false);
  if (!setup_ok || !pg.complete()) {
    printf("  FAIL: six-tile SP0 setup did not complete\n");
    for (int t = phase_e::kTiles; t >= 1; --t) phase_e::teardown(m[t]);
    return false;
  }

  bool ok = true;
  const fv3_gpu_field_mask mask = phase_e::sp0_field_mask();
  for (int t = 1; t <= phase_e::kTiles; ++t) {
    fv3_gpu_error err{};
    fv3_gpu_event ev = 0;
    int rc = fv3_gpu_sync_to_host(m[t].ctx, &mask, &ev, &err);
    if (rc == FV3_GPU_SUCCESS) rc = fv3_gpu_event_wait(m[t].ctx, ev, &err);
    if (rc != FV3_GPU_SUCCESS) {
      printf("  FAIL: tile %d SP0 D2H returned %d: %s\n", t, rc, err.message);
      ok = false;
      continue;
    }
    const struct { const std::vector<fv_real> *v; const char *n; } fin[] = {
        {&m[t].h->u, "u"}, {&m[t].h->v, "v"}, {&m[t].h->w, "w"},
        {&m[t].h->delp, "delp"}, {&m[t].h->pt, "pt"}, {&m[t].h->delz, "delz"},
        {&m[t].h->q_con, "q_con"}, {&m[t].h->pe, "pe"}, {&m[t].h->peln, "peln"},
        {&m[t].h->pk, "pk"},
    };
    char label[40];
    for (const auto &f : fin) {
      std::snprintf(label, sizeof(label), "tile%d %s", t, f.n);
      ok &= check_finite(f.v->data(), f.v->size(), label);
    }
    std::snprintf(label, sizeof(label), "tile%d delp", t);
    ok &= phase_e::check_positive_owned(m[t].h->delp.data(), m[t].h->ni_h, m[t].h->nj_h, NPZ, label);
    if (!g_sp0_dump_dir.empty()) ok &= phase_e::dump_sp0_state(*m[t].h, t, g_sp0_dump_dir);
  }

  for (int t = phase_e::kTiles; t >= 1; --t) phase_e::teardown(m[t]);
  printf("\n=== Milestone 6A.5 Section 10: SP0 verify-only: %s ===\n",
         ok ? "PASS" : "FAIL");
  printf("Numerical kernels executed: NO\n");
  return ok;
}

#if defined(FV3_DYN_CORE_ENABLE_MPI_HALO)
/* Milestone 3A.15 SS12: one repeat under the real production decomposition -- this
 * process owns exactly ONE tile and spawns no threads. The five peers are other
 * OS processes, and the only thing that couples them is the halo exchange.
 *
 * Deliberately a separate function rather than a branch inside run_one_repeat: the
 * six-thread path's structure (HaloPeerGroup, pg.complete(), std::thread fan-out,
 * pg.abort_group() on failure, MAX over a local array) has no meaning here, and
 * interleaving the two would obscure which lines the default path executes.
 *
 * FAILURE HANDLING IS MPI_Abort, NOT `return false`. A rank that fails between
 * exchanges leaves its neighbours blocked in MPI_Waitall forever; there is no
 * library-level rendezvous to poison the way pg.abort_group() poisons the
 * in-process one (see MpiHaloTransport::abort_transport, which records exactly
 * this). Aborting the job is the honest response and it is what makes a failure
 * visible in a Slurm log instead of a 40-minute timeout. */
bool run_one_repeat_mpi(int32_t n_split, int warmup, int iters, const char *mode,
                        int repeat, const char *hardware) {
  const int tile = g_mpi_tile;
  phase_e::HaloPeerGroup pg(phase_e::kTiles);  // constructed, never attached
  phase_e::Member m{};
  if (!phase_e::setup(m, tile, pg, false)) {
    printf("  FAIL: rank %d tile %d setup did not complete (repeat %d, mode %s)\n",
           g_mpi_rank, tile, repeat, mode);
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }

  const double e = (std::strcmp(mode, "resident") == 0)
                       ? phase_e::run_mode_a(m, n_split, warmup, iters)
                       : phase_e::run_mode_b(m, n_split, warmup, iters);
  bool ok = (e >= 0.0);
  if (ok) ok = phase_e::verify_work_executed(m, tile);
  if (!ok) {
    printf("  FAIL: rank %d tile %d repeat %d mode %s did not complete cleanly\n",
           g_mpi_rank, tile, repeat, mode);
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }

  /* Detach before the reduction, so the collective below cannot be reached by a
   * rank that is still mid-exchange. */
  phase_e::teardown(m);

  /* SS9's statistic unchanged in kind: the MAX across the six tiles, which now
   * needs an allreduce because the six tiles are six processes. */
  double max_ms = e;
  MPI_Allreduce(MPI_IN_PLACE, &max_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (g_mpi_rank == 0) {
    printf("CSV,cuda_gpu_mpi,%lld,%lld,%lld,%d,%s,%s,%d,%.6f,%.6f,%s\n",
           (long long)NI, (long long)NJ, (long long)NPZ, phase_e::kTiles,
           sizeof(fv_real) == 8 ? "R8" : "R4", mode, repeat, max_ms,
           max_ms / (double)iters, hardware);
  }
  return true;
}
#endif  // FV3_DYN_CORE_ENABLE_MPI_HALO

bool run_one_repeat(int32_t n_split, int warmup, int iters, const char *mode,
                    int repeat, const char *hardware) {
#if defined(FV3_DYN_CORE_ENABLE_MPI_HALO)
  if (g_mpi_halo)
    return run_one_repeat_mpi(n_split, warmup, iters, mode, repeat, hardware);
#endif
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
        "--hardware STRING "
        "[--grid-metrics DIR]  (Milestone 6A.5: load authoritative grid "
        "metrics from DIR/tile{0..5}/*.bin instead of the fabricated "
        "uniform metrics; validation-only, default off)\n"
        "[--sp0-state DIR --sp0-verify-only --sp0-dump DUMPDIR]  "
        "(Milestone 6A.5 Section 10: load the authoritative Fortran Prompt "
        "6A SP0 dynamical state + global da_min/da_min_c from DIR "
        "(tools/savepoints/convert_sp0_to_cuda.py output), perform H2D, "
        "immediately D2H and dump to DUMPDIR, then exit -- no numerical "
        "kernel is launched. Requires --grid 8 --npz 10 to match the SP0 "
        "savepoint's c8/npz=10 config.)\n"
        "[--sp1-boundary DIR]  (Prompt 6B: overwrite h.ptop, h.phis/h.zs and h.dp_ref "
        "with the authoritative Fortran SP1-savepoint-derived values from "
        "DIR/tile{0..5}/*.bin (tools/savepoints/convert_sp1_boundary_to_cuda.py "
        "output) instead of the synthetic ptop=1000/phis=zs=0 seed; "
        "validation-only, default off)\n"
        "[--akap VALUE]  (Prompt 6B: override akap/cappa. The authoritative "
        "value for the GEOSfvdycore_fortran_perf fixture is "
        "0.28571426868438721 -- the single-precision MAPL_KAPPA, one float32 "
        "ULP below 2/7; the default is the double 2/7. validation-only, "
        "default off)\n"
        "[--akap-exner VALUE]  (Prompt 6B: use VALUE as the EXNER-only kappa "
        "(ptk, pk3, pk3_halo) while cfg.akap stays the thermodynamic-closure "
        "kappa. The authoritative fixture separates the two because it compiles "
        "MOIST_CAPPA/USE_COND: VALUE = 0.28571426868438721 (single-precision "
        "MAPL_KAPPA) for Exner, 2/7 for the closure; validation-only, "
        "default off)\n"
        "[--cuda-graph]  (Prompt 8: enable CUDA Graph capture/replay for the "
        "dgrid_dynamics stage only via fv3_gpu_set_cuda_graph_mode; every "
        "other stage and all six-tile halo communication are unchanged; "
        "default off)\n"
        "[--fuse-corner-fill]  (Prompt 9: fuse c_sw's per-level "
        "fill2_4corners(delp,pt,dir)+fill_4corners(w,dir) into one "
        "launch_fill_corner_triplet call; bit-identical, launch-count-only "
        "change; default off)\n"
        "[--fuse-halo-edges]  (Prompt 11: fuse pack_edge_kernel/"
        "unpack_edge_kernel's per-edge loop (4 launches/field) into one "
        "launch/field covering all four TileEdge values; bit-identical, "
        "launch-count-only change; default off)\n"
        "[--mpi-halo]  (Milestone 3A.15: run as SIX SEPARATE MPI PROCESSES, one "
        "cubed-sphere tile per rank (rank r owns tile r+1), exchanging halos "
        "through the host-staged MPI transport instead of the in-process "
        "HaloPeerGroup. This is the real production decomposition (FV_NX=1, "
        "FV_NY=6). Requires a build with -DFV3_DYN_CORE_ENABLE_MPI_HALO=ON and "
        "exactly 6 ranks; default off)\n");
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
    else if (a == "--grid-metrics") g_grid_metrics_dir = next(i);
    else if (a == "--sp0-state") g_sp0_state_dir = next(i);
    else if (a == "--sp0-verify-only") g_sp0_verify_only = true;
    else if (a == "--sp0-dump") g_sp0_dump_dir = next(i);
    else if (a == "--bdt") g_bdt = std::atof(next(i));
    else if (a == "--sp1-boundary") g_sp1_boundary_dir = next(i);
    else if (a == "--damp-n-sponge") g_damp_n_sponge = std::atoi(next(i));
    else if (a == "--akap") g_akap = std::atof(next(i));
    else if (a == "--akap-exner") g_akap_exner = std::atof(next(i));
    else if (a == "--cuda-graph") g_cuda_graph = true;
    else if (a == "--fuse-corner-fill") g_fuse_corner_fill = true;
    else if (a == "--fuse-halo-edges") g_fuse_halo_edges = true;
    else if (a == "--mpi-halo") g_mpi_halo = true;
    else if (a == "--help") { usage(); return 0; }
    else { printf("unknown argument: %s\n", a.c_str()); usage(); return 2; }
  }

  // Milestone 3A.15 SS12. MPI is initialized here, by the caller, not by the
  // transport -- fv3_gpu_attach_mpi_halo_transport documents that the caller owns
  // MPI_Init/MPI_Finalize, because in the real deployment FV3's Fortran already
  // owns them and a library that called MPI_Init would be wrong there.
  if (g_mpi_halo) {
#if !defined(FV3_DYN_CORE_ENABLE_MPI_HALO)
    printf("FAIL: --mpi-halo requires a build configured with "
           "-DFV3_DYN_CORE_ENABLE_MPI_HALO=ON\n");
    return 2;
#else
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
      printf("FAIL: MPI_Init failed\n");
      return 2;
    }
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
    if (size != phase_e::kTiles) {
      if (g_mpi_rank == 0)
        printf("FAIL: --mpi-halo needs exactly %d ranks (one cubed-sphere tile "
               "each); got %d\n", phase_e::kTiles, size);
      MPI_Finalize();
      return 2;
    }
    g_mpi_tile = g_mpi_rank + 1;
    // The in-process six-tile paths cannot run here: each of the two below spawns
    // or loops over all six tiles in ONE process, which is the decomposition
    // --mpi-halo exists to replace. Refused rather than silently reinterpreted.
    if (g_sp0_verify_only) {
      if (g_mpi_rank == 0)
        printf("FAIL: --sp0-verify-only is an in-process six-tile path and is not "
               "supported with --mpi-halo\n");
      MPI_Finalize();
      return 2;
    }
#endif
  }

  NI = grid; NJ = grid; NPZ = npz;

  fv3_gpu_capabilities caps{};
  caps.abi_version = FV3_GPU_ABI_VERSION;
  caps.struct_bytes = sizeof(caps);
  fv3_gpu_get_capabilities(nullptr, &caps);
  int usable = (caps.compiled_with_cuda && caps.device_count > 0) ? 1 : 0;
#if defined(FV3_DYN_CORE_ENABLE_MPI_HALO)
  /* Milestone 3A.15 SS12: the skip decision must be UNANIMOUS. If one rank
   * returned 77 on its own, the other five would run on and block forever in the
   * first halo MPI_Waitall against a rank that has already exited -- a hang, not a
   * skip. MPI_MIN makes any rank's "no device" everybody's skip. */
  if (g_mpi_halo)
    MPI_Allreduce(MPI_IN_PLACE, &usable, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
  if (!usable) {
    if (!caps.compiled_with_cuda) printf("SKIP: not compiled with CUDA\n");
    else if (caps.device_count == 0) printf("SKIP: no CUDA devices available\n");
    else printf("SKIP: another rank of this run has no usable CUDA device\n");
#if defined(FV3_DYN_CORE_ENABLE_MPI_HALO)
    if (g_mpi_halo) MPI_Finalize();
#endif
    return 77;
  }

  if (g_sp0_verify_only) {
    if (g_sp0_state_dir.empty()) {
      printf("FAIL: --sp0-verify-only requires --sp0-state DIR\n");
      return 2;
    }
    if (g_sp0_dump_dir.empty()) {
      printf("FAIL: --sp0-verify-only requires --sp0-dump DUMPDIR\n");
      return 2;
    }
    printf("=== Milestone 6A.5 Section 10: SP0 verify-only ===\n");
    printf("Load order: metrics -> global scalars -> SP0 dynamical state -> "
           "H2D -> verify device state -> STOP\n");
    printf("grid=%lldx%lld npz=%lld sp0-state=%s grid-metrics=%s\n",
          (long long)grid, (long long)grid, (long long)npz,
          g_sp0_state_dir.c_str(),
          g_grid_metrics_dir.empty() ? "(unset)" : g_grid_metrics_dir.c_str());
    const bool ok = run_sp0_verify_only();
    return ok ? 0 : 1;
  }

  if (g_mpi_halo)
    printf("=== Milestone 3A.15: SIX-MPI-PROCESS run, rank %d owns tile %d, "
           "halo transport = host-staged MPI ===\n", g_mpi_rank, g_mpi_tile);
  printf("=== Milestone 3A.11 CUDA perf benchmark ===\n");
  printf("PRE-VALIDATION PERFORMANCE CHARACTERIZATION -- not numerical validation\n");
  printf("grid=%lldx%lld npz=%lld n_split=%d precision=%s warmup=%d iters=%d "
        "repeats=%d hardware=%s cuda_graph=%d\n",
        (long long)grid, (long long)grid, (long long)npz, n_split,
        sizeof(fv_real) == 8 ? "R8" : "R4", warmup, iters, repeats,
        hardware.c_str(), g_cuda_graph ? 1 : 0);
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
#if defined(FV3_DYN_CORE_ENABLE_MPI_HALO)
  /* Every rank is past teardown (run_one_repeat_mpi detaches before its own
   * reduction), so finalizing here is safe. A rank that failed never reaches this
   * line -- it called MPI_Abort. */
  if (g_mpi_halo) {
    int agreed = all_ok ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &agreed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    all_ok = (agreed != 0);
    MPI_Finalize();
  }
#endif
  return all_ok ? 0 : 1;
}
