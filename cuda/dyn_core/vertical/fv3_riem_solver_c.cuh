#pragma once
//===========================================================================
// WP-C: C-grid Riemann solver entry -- Riem_Solver_c
//
// Authoritative Fortran source of truth:
//   GFDL_atmos_cubed_sphere (tag geos/v3.0.0, ca30e26)
//   model/nh_utils.F90:325-432   subroutine Riem_Solver_c
//
// Unique Fortran call site: dyn_core.F90:583-587
//   call Riem_Solver_C( ms, dt2, is, ie, js, je, npz, ng, akap, cappa, cp,
//                       ptop, phis, omga, ptc, q_con, delpc, gz, pkc, ws3,
//                       flagstruct%p_fac, flagstruct%a_imp, flagstruct%scale_z )
//
// This is NOT Riem_Solver3 and is NOT an alias of it. The differences that
// this file must preserve (see the .cu for the per-statement transcription):
//   * j = js-1..je+1, i = is-1..ie+1   (Riem_Solver3: js..je, is..ie)
//   * closure gz(k) = gz(k+1) - dz2(k)*grav, gz(km+1) = hs   (grav-weighted
//     geopotential, NOT metres as in Riem_Solver3's zh)
//   * pef = pe2 + pem written UNCONDITIONALLY (no fp_out gate)
//   * pm2 = dm/log(pem(k+1)/pem(k))   (Riem_Solver3: dm/(peln2(k+1)-peln2(k)))
//   * no peln2, no pk3, no pk, no pe, no peln outputs; no delz/w writeback
//   * three a_imp branches (nh_utils.F90:402-411), not five
//   * RIM_2D would be called with .true. (nh_utils.F90:407), not .false.
//
// Implemented configuration only: hydrostatic = .false., a_imp = 1.0
// (-> SIM1_solver, nh_utils.F90:409-410), grid_type = 0, non-nested.
// Every other configuration is rejected before any state is mutated.
//===========================================================================

#include "fv3_gpu_types.hpp"
#include "fv3_gpu_internal.hpp"
#include <cstdint>

namespace fv3::gpu::vertical {

#if defined(__CUDACC__)
#define FV3_RIEM_C_HD __host__ __device__
#else
#define FV3_RIEM_C_HD
#endif

// Typed device views for this entry. Zero-based level index k = 0..km-1 for
// layer fields and k = 0..km for interface fields, matching the convention
// already used by riem_solver3_kernel in fv3_vertical_kernels.cu (Fortran
// k = 1..km maps to k = 0..km-1).
struct RiemCViewC3 {
  const fv_real *p = nullptr;
  int64_t l0 = 0, l1 = 0, l2 = 0, s0 = 0, s1 = 0, s2 = 0;
  FV3_RIEM_C_HD inline fv_real at(int64_t i, int64_t j, int64_t k) const {
    return p[(i - l0) * s0 + (j - l1) * s1 + (k - l2) * s2];
  }
};
struct RiemViewC3 {
  fv_real *p = nullptr;
  int64_t l0 = 0, l1 = 0, l2 = 0, s0 = 0, s1 = 0, s2 = 0;
  FV3_RIEM_C_HD inline fv_real &at(int64_t i, int64_t j, int64_t k) const {
    return p[(i - l0) * s0 + (j - l1) * s1 + (k - l2) * s2];
  }
};
struct RiemCViewC2 {
  const fv_real *p = nullptr;
  int64_t l0 = 0, l1 = 0, s0 = 0, s1 = 0;
  FV3_RIEM_C_HD inline fv_real at(int64_t i, int64_t j) const {
    return p[(i - l0) * s0 + (j - l1) * s1];
  }
};

// Convenience adapters from the WP-E registry descriptors.
inline RiemCViewC3 riem_c_cview3(const DeviceViewBase &v) {
  return {static_cast<const fv_real *>(v.data), v.lower[0], v.lower[1],
          v.lower[2], v.stride[0], v.stride[1], v.stride[2]};
}
inline RiemViewC3 riem_c_view3(const DeviceViewBase &v) {
  return {static_cast<fv_real *>(v.data), v.lower[0], v.lower[1], v.lower[2],
          v.stride[0], v.stride[1], v.stride[2]};
}
inline RiemCViewC2 riem_c_cview2(const DeviceViewBase &v) {
  return {static_cast<const fv_real *>(v.data), v.lower[0], v.lower[1],
          v.stride[0], v.stride[1]};
}

// Per-column scratch slot count. 19 slots of (km+1) scalars each:
//   Fortran locals (nh_utils.F90:341-342): dm, dz2, w2, pm2, gm2, cp2,
//                                          pem, pe2, peg                (9)
//   pt column staged for the SIM1 call's pt(is1:ie1,j,1:km) actual arg   (1)
//   SIM1_solver work arrays owned by the caller in sim1_column():
//     aa, bb, dd, w1, g_rat, gam, pp, bet, p1                           (9)
inline constexpr int riem_solver_c_scratch_slots = 19;

// Level cap for the per-thread local-array scratch path, i.e. when
// RiemSolverCArgs::workspace is null. Same 128 precedent as
// riem_solver3_kernel; km > this is REJECTED, never wrapped.
inline constexpr int riem_solver_c_max_local_levels = 128;

FV3_RIEM_C_HD inline uint64_t riem_solver_c_scalars_per_column(int km) noexcept {
  return static_cast<uint64_t>(riem_solver_c_scratch_slots) *
         static_cast<uint64_t>(km + 1);
}

// Column count actually solved: (ie+1)-(is-1)+1 by (je+1)-(js-1)+1, i.e. the
// WIDER Riem_Solver_c range of nh_utils.F90:356/359, not is..ie / js..je.
inline uint64_t riem_solver_c_columns(int64_t is, int64_t ie, int64_t js,
                                      int64_t je) noexcept {
  const int64_t nx = (ie + 1) - (is - 1) + 1;
  const int64_t ny = (je + 1) - (js - 1) + 1;
  if (nx <= 0 || ny <= 0) return 0;
  return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny);
}

inline uint64_t riem_solver_c_workspace_bytes(int km, int64_t is, int64_t ie,
                                              int64_t js, int64_t je) noexcept {
  return riem_solver_c_columns(is, ie, js, je) *
         riem_solver_c_scalars_per_column(km) * sizeof(fv_real);
}

struct RiemSolverCArgs {
  // ---- intent(in) fields (nh_utils.F90:330-336) -------------------------
  RiemCViewC3 delp;   // delp   <- FieldId::delpc  (km levels)
  RiemCViewC3 pt;     // pt     <- FieldId::ptc    (km levels)
  RiemCViewC3 w3;     // w3     <- FieldId::omga   (km levels)
  RiemCViewC3 q_con;  // q_con  <- FieldId::q_con  (km levels; read only if use_cond)
  RiemCViewC3 cappa;  // cappa  <- FieldId::cappa  (km levels; read only if moist_cappa)
  RiemCViewC2 hs;     // hs     <- FieldId::phis
  RiemCViewC2 ws;     // ws     <- FieldId::ws3

  // ---- intent(inout) / intent(out) fields (nh_utils.F90:338-339) --------
  RiemViewC3 gz;      // gz     <- FieldId::gz     (km+1 levels, inout)
  RiemViewC3 pef;     // pef    <- FieldId::pkc    (km+1 levels, out)

  // ---- compute bounds: pass the Fortran is/ie/js/je. The launcher and
  //      kernel expand to is-1..ie+1 / js-1..je+1 themselves. -------------
  int64_t is = 0, ie = 0, js = 0, je = 0;
  int32_t km = 0;                 // Fortran km (= npz)

  // ---- scalars ----------------------------------------------------------
  /* Fortran ms. NOT DispatchParams::vertical_ms, which carries the OTHER value:
   * dyn_core.F90:918 hands Riem_Solver3 flagstruct%m_split directly, while :300
   * defines the C-grid's ms as max(1, flagstruct%m_split/2) and :583 passes
   * that. Derive this from NumericalConfig::m_split at the call site. */
  int32_t ms = 1;
  fv_real dt = fv_real(0);        // Fortran dt == dt2 at dyn_core.F90:583
  fv_real akap = fv_real(0);      // gama = 1/(1-akap) is derived, :347
  fv_real ptop = fv_real(0);
  fv_real p_fac = fv_real(0);
  fv_real a_imp = fv_real(0);     // must be > 0.999 (SIM1 branch, :408-410)
  fv_real scale_m = fv_real(0);   // flagstruct%scale_z; unused on the SIM1
                                  // branch (Fortran passes it only to
                                  // SIM3p0_solver, :403-404). Carried so the
                                  // contract does not change if SIM3p0 lands.
  fv_real rdgas = fv_real(0);     // constants_mod rdgas (NumericalConfig::rdgas)
  fv_real grav = fv_real(0);      // constants_mod grav  (NumericalConfig::grav)

  // ---- compile-time Fortran cpp switches, now runtime flags -------------
  bool use_cond = false;          // USE_COND    (:367,:377,:386)
  bool moist_cappa = false;       // MOIST_CAPPA (:389-390); requires use_cond

  // ---- configuration guards; nonzero/true is rejected ------------------
  bool hydrostatic = false;       // Riem_Solver_c is a nonhydrostatic path
  bool nested = false;
  int32_t grid_type = 0;

  // ---- optional column scratch -----------------------------------------
  // workspace == nullptr  -> per-thread local arrays, km <= 128 enforced.
  // workspace != nullptr  -> device scratch of at least
  //   riem_solver_c_workspace_bytes(km, is, ie, js, je), addressed with
  //   workspace_scalars_per_column stride per column (>= scalars_per_column).
  fv_real *workspace = nullptr;
  uint64_t workspace_scalars_per_column = 0;
  uint64_t workspace_bytes = 0;

  /* Milestone 3A.11T diagnostic, default OFF and off in every normal run.
   * When set, the kernel evaluates SIM1_solver's own preconditions
   * (nh_utils.F90:1232 log argument, :1244 pem ratio, :1272/:1320 p_fac
   * clamp argument) on the exact values it is about to pass to sim1_column,
   * and device-printfs the first violating level per column. This answers
   * "were Riem_Solver_c's inputs physically valid before the first NaN"
   * at the point of use rather than by inference from a neighbouring
   * checkpoint. Enabled from FV3_GPU_DEBUG_FAIL in launch_riem_solver_c_ctx. */
  bool diag_preconditions = false;
  int32_t diag_tile = -1;
  int64_t diag_step = -1;
};

// Enqueue-only. `stream` is the WP-E-owned compute stream (cudaStream_t).
// No host/device transfer and no device synchronization is performed.
fv3_gpu_status launch_riem_solver_c(const RiemSolverCArgs &a, void *stream,
                                   fv3_gpu_error *error = nullptr);

// WP-E ExecutionContext entry, LaunchFn-compatible. Field mapping as above;
// scalars from ImmutableViews::config; dt from DispatchParams::dt2 (the call
// site passes dt2, dyn_core.F90:583); ms = max(1, NumericalConfig::m_split/2),
// which is dyn_core.F90:300 -- deliberately not DispatchParams::vertical_ms,
// since that member holds Riem_Solver3's larger value (:918).
// Uses WorkspaceId::vertical when it is present and large enough, otherwise
// falls back to the local-array path (km <= 128).
int launch_riem_solver_c_ctx(const ExecutionContext &, fv3_gpu_error *) noexcept;

} // namespace fv3::gpu::vertical
