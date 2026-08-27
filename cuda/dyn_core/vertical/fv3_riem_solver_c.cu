//===========================================================================
// WP-C: Riem_Solver_c CUDA kernel + launchers.
//
// Authoritative Fortran (numerical source of truth, read-only reference):
//   GFDL_atmos_cubed_sphere tag geos/v3.0.0 (ca30e26)
//   model/nh_utils.F90:325-432   subroutine Riem_Solver_c
// All ":NNN" comments below are nh_utils.F90 line numbers unless another file
// is named. The transcription is one CUDA statement per Fortran statement,
// with the Fortran i/j loops replaced by one thread per (i,j) column.
//
// Implemented configuration ONLY:
//   hydrostatic = .false., a_imp = 1.0, grid_type = 0, non-nested.
// a_imp = 1.0 selects the third branch at :408 -> SIM1_solver (:409-410),
// which is reused here through the existing device function
// sim1_column() in fv3_vertical_column.hpp:117-148. SIM1 is NOT re-derived.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_riem_solver_c.cuh"
#include "fv3_vertical_column.hpp"
#include "fv3_gpu_errors.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>  // std::getenv, for the 3A.11T precondition-probe gate

namespace fv3::gpu::vertical {
namespace {

// Flat per-thread arena for the workspace-less path: 19 slots x (128+1).
// riem_solver3_kernel sets the [128]/[129] precedent; km > 128 is rejected in
// the launcher rather than silently clamped.
inline constexpr int kLocalArenaScalars =
    riem_solver_c_scratch_slots * (riem_solver_c_max_local_levels + 1);

// One thread per horizontal column of the WIDER Riem_Solver_c index range:
// i = is-1..ie+1 (:350-351,:359) and j = js-1..je+1 (:356). The Fortran
// `do 2000 j` loop is the y dimension of the launch grid.
template <bool UseLocalScratch>
__global__ void riem_solver_c_kernel(const RiemSolverCArgs a) {
  const int64_t is1 = a.is - int64_t(1);  // :350  is1 = is - 1
  const int64_t ie1 = a.ie + int64_t(1);  // :351  ie1 = ie + 1
  const int64_t js1 = a.js - int64_t(1);  // :356  do 2000 j = js-1, ...
  const int64_t je1 = a.je + int64_t(1);  // :356  ...           , je+1

  const int64_t i = is1 + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t j = js1 + int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (i > ie1 || j > je1) return;

  const int km = a.km;
  const int slot = km + 1;  // scratch slot pitch; Fortran locals are km or km+1

  // Column scratch. Either a per-thread local arena or the caller's device
  // workspace; identical slot layout in both cases.
  fv_real local_arena[UseLocalScratch ? kLocalArenaScalars : 1];
  fv_real *sc;
  if constexpr (UseLocalScratch) {
    sc = local_arena;
  } else {
    const int64_t nx = ie1 - is1 + int64_t(1);
    const int64_t col = (i - is1) + (j - js1) * nx;
    sc = a.workspace +
         col * static_cast<int64_t>(a.workspace_scalars_per_column);
  }

  // Fortran locals, :341-342
  fv_real *dm  = sc + 0 * slot;   // dm (is-1:ie+1, km)
  fv_real *dz2 = sc + 1 * slot;   // dz2
  fv_real *w2  = sc + 2 * slot;   // w2
  fv_real *pm2 = sc + 3 * slot;   // pm2
  fv_real *gm2 = sc + 4 * slot;   // gm2   (MOIST_CAPPA only)
  fv_real *cp2 = sc + 5 * slot;   // cp2   (MOIST_CAPPA only)
  fv_real *pem = sc + 6 * slot;   // pem (is-1:ie+1, km+1)
  fv_real *pe2 = sc + 7 * slot;   // pe2
  fv_real *peg = sc + 8 * slot;   // peg   (USE_COND only)
  // Staging for the pt(is1:ie1,j,1:km) actual argument at :410.
  fv_real *pt2 = sc + 9 * slot;
  // SIM1_solver locals, owned by the caller in the CUDA column form
  // (fv3_vertical_column.hpp:117-123 == nh_utils.F90:1215-1217).
  fv_real *aa    = sc + 10 * slot;
  fv_real *bb    = sc + 11 * slot;
  fv_real *dd    = sc + 12 * slot;
  fv_real *w1    = sc + 13 * slot;
  fv_real *g_rat = sc + 14 * slot;
  fv_real *gam   = sc + 15 * slot;
  fv_real *pp    = sc + 16 * slot;
  fv_real *bet   = sc + 17 * slot;
  fv_real *p1    = sc + 18 * slot;

  const fv_real gama = fv_real(1) / (fv_real(1) - a.akap);  // :347
  const fv_real rgrav = fv_real(1) / a.grav;                // :348

  // :358-362   dm(i,k) = delp(i,j,k)
  for (int k = 0; k < km; ++k) dm[k] = a.delp.at(i, j, k);  // :360

  a.pef.at(i, j, 0) = a.ptop;                        // :365  pef(i,j,1) = ptop
  pem[0] = a.ptop;                                   // :366  pem(i,1)   = ptop
  if (a.use_cond) peg[0] = a.ptop;                   // :368  peg(i,1)   = ptop

  // :372-380
  for (int k = 1; k <= km; ++k) {
    pem[k] = pem[k - 1] + dm[k - 1];                 // :374
    if (a.use_cond)                                  // :377
      peg[k] = peg[k - 1] + dm[k - 1] *
               (fv_real(1) - a.q_con.at(i, j, k - 1));
  }

  // :382-399
  for (int k = 0; k < km; ++k) {
    dz2[k] = a.gz.at(i, j, k + 1) - a.gz.at(i, j, k); // :384
    if (a.use_cond) {
      // :386  NOTE: log of the RATIO, not a difference of logs. Riem_Solver3
      // uses (pelng(k+1)-pelng(k)); Riem_Solver_c keeps no pelng at all.
      pm2[k] = (peg[k + 1] - peg[k]) / fv3_log(peg[k + 1] / peg[k]);
      if (a.moist_cappa) {
        cp2[k] = a.cappa.at(i, j, k);                // :389
        gm2[k] = fv_real(1) / (fv_real(1) - cp2[k]); // :390
      }
    } else {
      // :394  dm is still the UNSCALED delp here; rgrav is applied at :396.
      pm2[k] = dm[k] / fv3_log(pem[k + 1] / pem[k]);
    }
    dm[k] = dm[k] * rgrav;                           // :396
    w2[k] = a.w3.at(i, j, k);                        // :397
    // pt(is1:ie1,j,1:km) actual argument staged for :410.
    pt2[k] = a.pt.at(i, j, k);
  }

  /* ---- Milestone 3A.11T precondition probe (diagnostic only) ------------
   * Evaluates SIM1_solver's own implicit preconditions on the exact arrays
   * about to be handed to sim1_column. Neither Riem_Solver_c nor SIM1_solver
   * guards any of these in the authoritative Fortran, so a violation here
   * means the Fortran would produce a NaN from the same input -- which is the
   * discriminator between a port defect and an invalid input state.
   *
   *   dz2 < 0        : update_dz_c's dz_min clamp (:196-200) must guarantee it
   *   -dm/dz2*R*pt>0 : the log argument at :1232
   *   pem increasing : the log(pem(k+1)/pem(k)) ratio at :394
   *   pg > 0         : the max(p_fac*pm2, p1+pm2) log argument at :1272/:1320
   *                    (only its pm2-dependent lower bound is checkable here)
   * Reports at most one level per column to keep the output bounded. */
  if (a.diag_preconditions) {
    for (int k = 0; k < km; ++k) {
      const fv_real logarg = -dm[k] / dz2[k] * a.rdgas * pt2[k];
      const bool bad_dz = !(dz2[k] < fv_real(0));
      const bool bad_log = !(logarg > fv_real(0));
      const bool bad_pem = !(pem[k + 1] > pem[k]);
      const bool bad_pm = !(a.p_fac * pm2[k] > fv_real(0));
      if (bad_dz || bad_log || bad_pem || bad_pm) {
        printf("[RIEMC-PRE tile=%d step=%lld] i=%lld j=%lld k=%d "
               "dz2=%.9e dm=%.9e pt=%.9e pm2=%.9e pem_k=%.9e pem_k1=%.9e "
               "logarg=%.9e bad{dz=%d log=%d pem=%d pm=%d}\n",
               a.diag_tile, static_cast<long long>(a.diag_step),
               static_cast<long long>(i), static_cast<long long>(j), k,
               double(dz2[k]), double(dm[k]), double(pt2[k]), double(pm2[k]),
               double(pem[k]), double(pem[k + 1]), double(logarg),
               int(bad_dz), int(bad_log), int(bad_pem), int(bad_pm));
        break;
      }
    }
  }

  // :402-411  a_imp branch selection. Riem_Solver_c has exactly THREE
  // branches (SIM3p0 for a_imp < -0.01, RIM_2D for a_imp <= 0.5 with the
  // final argument .true., else SIM1_solver). Only the SIM1 branch is
  // implemented; the launcher rejects everything else, so the guard here is
  // unreachable in a validated launch and exists to keep the branch structure
  // visible against the Fortran.
  //
  // :409-410
  //   call SIM1_solver(dt, is1, ie1, km, rdgas, gama, gm2, cp2, akap, pe2,
  //                    dm, pm2, pem, w2, dz2, pt(is1:ie1,j,1:km),
  //                    ws(is1,j), p_fac)
  sim1_column<fv_real>(a.dt, a.rdgas, gama, a.akap, a.p_fac, a.moist_cappa,
                       gm2, cp2, dm, pm2, pem, w2, dz2, pt2, a.ws.at(i, j),
                       pe2, aa, bb, dd, w1, g_rat, gam, pp, bet, p1, km);

  // :413-417  pef(i,j,k) = pe2(i,k) + pem(i,k), k = 2..km+1.
  // Unconditional: unlike Riem_Solver3's ppe there is no fp_out gate.
  for (int k = 1; k <= km; ++k)
    a.pef.at(i, j, k) = pe2[k] + pem[k];             // :415

  // :419-428  Height * grav for the p-gradient computation. Riem_Solver_c
  // closes gz in GEOPOTENTIAL (dz2*grav, surface = hs); Riem_Solver3 closes
  // zh in metres with zh(km+1) = zs.
  a.gz.at(i, j, km) = a.hs.at(i, j);                 // :421  gz(i,j,km+1) = hs
  for (int k = km - 1; k >= 0; --k)                  // :424  do k = km,1,-1
    a.gz.at(i, j, k) = a.gz.at(i, j, k + 1) - dz2[k] * a.grav;  // :426

  // No w writeback (w3 is intent(in), :336) and no delz writeback: dz2 is a
  // local of Riem_Solver_c and is consumed only by the gz closure above.
}

} // namespace

fv3_gpu_status launch_riem_solver_c(const RiemSolverCArgs &a, void *stream,
                                    fv3_gpu_error *error) {
  clear_error(error);
  const char *api = "launch_riem_solver_c";

  // ---- rejections. All of these precede every state mutation; the kernel is
  // the only thing in this function that writes anything. ------------------
  if (!stream)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "null CUDA stream"));

  if (a.ie < a.is || a.je < a.js)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "empty compute bounds"));

  if (!a.delp.p || !a.pt.p || !a.w3.p || !a.hs.p || !a.ws.p || !a.gz.p ||
      !a.pef.p)
    return static_cast<fv3_gpu_status>(fail(
        error, FV3_GPU_INVALID_ARGUMENT, api,
        "null device field: Riem_Solver_c requires delp(delpc), pt(ptc), "
        "w3(omga), hs(phis), ws(ws3), gz, pef(pkc)"));

  if (a.use_cond && !a.q_con.p)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api,
             "use_cond requires a non-null q_con device field (nh_utils.F90:377)"));

  if (a.moist_cappa && !a.cappa.p)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api,
             "moist_cappa requires a non-null cappa device field (nh_utils.F90:389)"));

  if (a.hydrostatic)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "Riem_Solver_c is a nonhydrostatic path; hydrostatic=.true. is "
             "not implemented (required: hydrostatic=.false.)"));

  if (a.nested)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "nested grids are not implemented (required: non-nested)"));

  if (a.grid_type != 0)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "only grid_type=0 is implemented"));

  // nh_utils.F90:402-411 selects SIM1_solver for every a_imp > 0.5, but only
  // a_imp = 1.0 is implemented/validated here, so the narrower 0.999 test is
  // used deliberately: 0.5 < a_imp <= 0.999 is rejected rather than run.
  if (!(a.a_imp > fv_real(0.999)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "only a_imp = 1.0 (SIM1_solver branch, nh_utils.F90:408-410) is "
             "implemented; SIM3p0 (a_imp < -0.01) and RIM_2D (a_imp <= 0.5) "
             "branches of Riem_Solver_c are not"));

  if (a.moist_cappa && !a.use_cond)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "moist_cappa without use_cond is not a reachable Fortran "
             "configuration: nh_utils.F90:388-391 nests MOIST_CAPPA inside "
             "USE_COND, so gm2/cp2 would be undefined"));

  if (a.km < 2)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km < 2: the SIM1 tridiagonal sweep requires at least 2 levels"));

  if (!(a.dt != fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "dt = 0: SIM1_solver forms rdt = 1/dt (must be dt2 from the "
             "dyn_core.F90:583 call site)"));

  if (!(a.grav > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "grav must be positive: rgrav = 1/grav (nh_utils.F90:348)"));

  if (!(a.rdgas > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "rdgas must be positive"));

  if (a.akap == fv_real(1))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "akap = 1 makes gama = 1/(1-akap) undefined (nh_utils.F90:347)"));

  if (!(a.ptop > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "ptop must be positive: pem(1) = ptop seeds log(pem) at "
             "nh_utils.F90:366,394"));

  const bool use_workspace = a.workspace != nullptr;
  if (use_workspace) {
    const uint64_t need_per_column = riem_solver_c_scalars_per_column(a.km);
    const uint64_t columns = riem_solver_c_columns(a.is, a.ie, a.js, a.je);
    if (a.workspace_scalars_per_column < need_per_column)
      return static_cast<fv3_gpu_status>(
          fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
               "workspace_scalars_per_column too small for km"));
    if (a.workspace_bytes <
        columns * a.workspace_scalars_per_column * sizeof(fv_real))
      return static_cast<fv3_gpu_status>(
          fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
               "workspace too small for the is-1..ie+1 x js-1..je+1 column "
               "count of Riem_Solver_c"));
  } else if (a.km > riem_solver_c_max_local_levels) {
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km > 128 requires a device workspace: the local-array column "
             "scratch path is capped at 128 levels"));
  }

  // ---- launch. i = is-1..ie+1, j = js-1..je+1 (nh_utils.F90:356,:359). ----
  const int64_t nx = (a.ie + 1) - (a.is - 1) + 1;
  const int64_t ny = (a.je + 1) - (a.js - 1) + 1;
  const dim3 threads(16, 8);
  const dim3 grid(static_cast<unsigned>((nx + 15) / 16),
                  static_cast<unsigned>((ny + 7) / 8));
  auto s = static_cast<cudaStream_t>(stream);

  if (use_workspace)
    riem_solver_c_kernel<false><<<grid, threads, 0, s>>>(a);
  else
    riem_solver_c_kernel<true><<<grid, threads, 0, s>>>(a);

  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_CUDA_ERROR, api,
             cudaGetErrorString(launch_status),
             static_cast<int>(launch_status)));
  return FV3_GPU_SUCCESS;
}

int launch_riem_solver_c_ctx(const ExecutionContext &x,
                             fv3_gpu_error *error) noexcept {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  const char *api = "launch_riem_solver_c_ctx";
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "Riem_Solver_c must be enqueued on the compute stream");

  // Field mapping established for this repository.
  const auto *v_hs    = x.fields->find(FieldId::phis);   // hs
  const auto *v_w3    = x.fields->find(FieldId::omga);   // w3
  const auto *v_pt    = x.fields->find(FieldId::ptc);    // pt
  const auto *v_delp  = x.fields->find(FieldId::delpc);  // delp
  const auto *v_gz    = x.fields->find(FieldId::gz);     // gz   (inout)
  const auto *v_pef   = x.fields->find(FieldId::pkc);    // pef  (out)
  const auto *v_ws    = x.fields->find(FieldId::ws3);    // ws
  const auto *v_cappa = x.fields->find(FieldId::cappa);
  const auto *v_q_con = x.fields->find(FieldId::q_con);
  if (!v_hs || !v_w3 || !v_pt || !v_delp || !v_gz || !v_pef || !v_ws)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "Riem_Solver_c requires phis, omga, ptc, delpc, gz, pkc, ws3");

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;
  const auto &p = *x.dispatch_params;

  // i = is-1..ie+1 and j = js-1..je+1 read delpc/ptc/omga/gz one cell outside
  // the compute domain, so at least one halo cell must exist.
  if (b.halo_width < 1u)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "Riem_Solver_c spans is-1..ie+1 / js-1..je+1 and needs "
                "halo_width >= 1");
  if (x.immutable->topology->nested)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "nested grids are not implemented");

  RiemSolverCArgs a{};
  a.delp  = riem_c_cview3(*v_delp);
  a.pt    = riem_c_cview3(*v_pt);
  a.w3    = riem_c_cview3(*v_w3);
  if (v_q_con) a.q_con = riem_c_cview3(*v_q_con);
  if (v_cappa) a.cappa = riem_c_cview3(*v_cappa);
  a.hs    = riem_c_cview2(*v_hs);
  a.ws    = riem_c_cview2(*v_ws);
  a.gz    = riem_c_view3(*v_gz);
  a.pef   = riem_c_view3(*v_pef);

  a.is = b.is; a.ie = b.ie; a.js = b.js; a.je = b.je;
  a.km = static_cast<int32_t>(b.npz);

  /* Fortran ms. dyn_core.F90:300 `ms = max(1, flagstruct%m_split/2)`, and :583
   * passes that -- NOT flagstruct%m_split, which is what :918 gives
   * Riem_Solver3. DispatchParams::vertical_ms carries the :918 value, so taking
   * it here would run the C-grid solver with twice the vertical substeps the
   * Fortran uses. Derive the :300 value from config instead. */
  if (cfg.m_split < 1)
    return fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
                "NumericalConfig::m_split is not supplied; Riem_Solver_C's ms "
                "cannot be defaulted (dyn_core.F90:300)");
  a.ms      = cfg.m_split / 2 > 1 ? cfg.m_split / 2 : 1;
  a.dt      = fv_real(p.dt2);        // dyn_core.F90:583 passes dt2, not dt
  a.akap    = fv_real(cfg.akap);
  a.ptop    = fv_real(cfg.ptop);
  a.p_fac   = fv_real(cfg.p_fac);
  a.a_imp   = fv_real(cfg.a_imp);
  a.scale_m = fv_real(cfg.scale_m);  // flagstruct%scale_z
  a.rdgas   = fv_real(cfg.rdgas);
  a.grav    = fv_real(cfg.grav);

  a.use_cond    = cfg.use_cond != 0;
  a.moist_cappa = cfg.moist_cappa != 0;
  a.hydrostatic = cfg.hydrostatic != 0;
  a.nested      = x.immutable->topology->nested != 0;
  a.grid_type   = static_cast<int32_t>(cfg.grid_type);

  /* Milestone 3A.11T: same gate as the runtime's stage checkpoint, so one env
   * var turns the whole diagnostic set on and neither costs anything when it is
   * unset. getenv is called once per launch, which is acceptable for a
   * diagnostic and keeps the flag out of the ABI config. */
  {
    const char *dbg = std::getenv("FV3_GPU_DEBUG_FAIL");
    a.diag_preconditions = (dbg && *dbg);
    a.diag_tile = static_cast<int32_t>(x.immutable->topology->tile);
    a.diag_step = p.model_step;
  }

  // Prefer the persistent WP-E vertical workspace; fall back to per-thread
  // local scratch only when it cannot hold this (wider) column count.
  const uint64_t need_per_column = riem_solver_c_scalars_per_column(a.km);
  const uint64_t columns = riem_solver_c_columns(a.is, a.ie, a.js, a.je);
  const auto *slice = workspace_slice(*x.workspace, WorkspaceId::vertical);
  if (slice && slice->data &&
      slice->bytes >= columns * need_per_column * sizeof(fv_real)) {
    a.workspace = static_cast<fv_real *>(slice->data);
    a.workspace_scalars_per_column = need_per_column;
    a.workspace_bytes = slice->bytes;
  }

  return launch_riem_solver_c(a, x.submission.stream, error);
}

} // namespace fv3::gpu::vertical
