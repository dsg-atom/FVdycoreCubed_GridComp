//===========================================================================
// WP-C: production pt/delz vertical-remap CUDA kernel + launchers
// (Prompt 52C). One thread per horizontal column, plain interior domain
// is..ie x js..je (NOT Riem_Solver_c's widened is-1..ie+1/js-1..je+1 --
// fv_mapz.F90's own `do 1000/1002 j=js,je` + `is:ie` slices need no halo).
//
// Authoritative Fortran (numerical source of truth, read-only reference):
//   GFDL_atmos_cubed_sphere tag geos/v3.0.0 (ca30e26), model/fv_mapz.F90.
// All ":NNN" comments below are fv_mapz.F90 line numbers. Validated
// standalone reference for every step below except map_scalar itself
// (already validated, Prompt 52B): pt_delz-driver/main.F90 (Prompt
// 52C-prep, docs/performance/p52c_prep_pt_delz_fixture_2026-09-21.md).
//
// map_scalar_column (Prompt 52B, fv3_map_scalar_column.hpp) is called
// exactly as the Fortran calls map_scalar -- not re-derived here. This file
// contributes only the surrounding pressure-grid/pt/delz/pkz arithmetic
// that map_scalar does not cover, exactly matching the driver.
//
// MOIST_CAPPA/nwat contract: this kernel implements ONLY the dry-degenerate
// (nwat=0) formulas (moist_cv's case-default: cvm=cv_air, cappa=akap).
// nwat and kord are Fortran-only values with no NumericalConfig
// representation, so this file cannot itself verify nwat==0 or kord==9 --
// that is an external contract enforced at the Fortran dispatch call site.
// Every launch through this entry point is trusted to have already been
// gated there. See fv3_pt_delz_remap.cuh's header comment.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_pt_delz_remap.cuh"
#include "fv3_vertical_column.hpp"
#include "fv3_map_scalar_column.hpp"
#include "fv3_gpu_errors.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace fv3::gpu::vertical {
namespace {

inline constexpr int kMaxLevels = pt_delz_remap_max_local_levels;

// Prompt 68: flat per-thread arena for the workspace-less path, same
// convention as fv3_riem_solver_c.cu's kLocalArenaScalars -- 17 slots x
// (128+1). km > 128 is rejected in the launcher when no workspace is
// supplied, never silently clamped.
inline constexpr int kLocalArenaScalars =
    pt_delz_remap_scratch_slots * (pt_delz_remap_max_local_levels + 1);

// Prompt 56 Stage 4: minimal, one-time, host-side registry-view diagnostic.
// Prints exactly what fv3_gpu_field_registry.F90 handed this launcher for
// each field -- pointer, rank, lower/extent/stride -- with zero kernel
// changes and zero effect unless RUN_CUDA_FV3_PTDELZ_DUMP_VIEWS is set.
// Not a numerical change; removable after Prompt 56 closes.
void p56_dump_view_once(const char *name, const DeviceViewBase &v) {
  std::fprintf(stderr,
      "[P56_VIEW_DUMP] field=%-6s data=%p rank=%u element_bytes=%u "
      "lower=(%lld,%lld,%lld) extent=(%llu,%llu,%llu) stride=(%lld,%lld,%lld)\n",
      name, v.data, v.rank, v.element_bytes,
      static_cast<long long>(v.lower[0]), static_cast<long long>(v.lower[1]),
      static_cast<long long>(v.lower[2]),
      static_cast<unsigned long long>(v.extent[0]),
      static_cast<unsigned long long>(v.extent[1]),
      static_cast<unsigned long long>(v.extent[2]),
      static_cast<long long>(v.stride[0]), static_cast<long long>(v.stride[1]),
      static_cast<long long>(v.stride[2]));
  std::fflush(stderr);
}

// Prompt 56 Stage 5: minimal, one-time, host-side pre-kernel INPUT value
// dump for one fixed interior column, via blocking cudaMemcpyDeviceToHost
// (no kernel/device-code change; the launcher issues this before enqueuing
// pt_delz_remap_kernel, and the blocking copy is stream-ordered behind
// whatever wrote these buffers last, so it reads genuine pre-kernel state).
// axis_ikj must be true for pe/peln (registered (i,k,j), same convention
// pt_delz_remap_kernel itself uses via .at(i,k,j) -- see the NOTE above
// pe_old[k]/pn1[k] below) and false for every other field here (registered
// (i,j,k), .at(i,j,k)). Getting this wrong reads a different, still-in-
// bounds memory location -- silently wrong values, not a crash.
void p56_dump_column_values3(const char *name, const RiemViewC3 &v,
                              int64_t i, int64_t j, int k0, int k1,
                              bool axis_ikj = false) {
  for (int k = k0; k <= k1; ++k) {
    const int64_t off = axis_ikj
        ? (i - v.l0) * v.s0 + (k - v.l1) * v.s1 + (j - v.l2) * v.s2
        : (i - v.l0) * v.s0 + (j - v.l1) * v.s1 + (k - v.l2) * v.s2;
    fv_real hv = fv_real(0);
    cudaMemcpy(&hv, v.p + off, sizeof(fv_real), cudaMemcpyDeviceToHost);
    std::fprintf(stderr,
        "[P56_VALUE_DUMP] field=%-6s i=%lld j=%lld k=%d val=%.10e\n", name,
        static_cast<long long>(i), static_cast<long long>(j), k,
        static_cast<double>(hv));
  }
  std::fflush(stderr);
}
void p56_dump_column_values1(const char *name, const PtDelzView1 &v, int k0,
                              int k1) {
  for (int k = k0; k <= k1; ++k) {
    const int64_t off = (k - v.l0) * v.s0;
    fv_real hv = fv_real(0);
    cudaMemcpy(&hv, v.p + off, sizeof(fv_real), cudaMemcpyDeviceToHost);
    std::fprintf(stderr, "[P56_VALUE_DUMP] field=%-6s k=%d val=%.10e\n", name,
                 k, static_cast<double>(hv));
  }
  std::fflush(stderr);
}

// fv_mapz.F90:99-100 module parameters -- fixed physical constants, not
// runtime-configurable, so not plumbed through NumericalConfig.
template <class T> FV3_PTDZ_HD inline T pt_delz_t_min() { return T(184); }
template <class T> FV3_PTDZ_HD inline T pt_delz_no_min() { return T(-1e25); }

template <bool UseLocalScratch>
__global__ void pt_delz_remap_kernel(const PtDelzRemapArgs a) {
  const int64_t i = a.is + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t j = a.js + int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (i > a.ie || j > a.je) return;

  const int km = a.km;
  const int slot = km + 1;  // scratch slot pitch; matches km/km+1 locals

  // Column scratch. Either a per-thread local arena (stack/local memory,
  // the Prompt 67/68 OOM source -- see fv3_pt_delz_remap.cuh's header
  // comment) or the caller's persistent device workspace; identical slot
  // layout in both cases.
  fv_real local_arena[UseLocalScratch ? kLocalArenaScalars : 1];
  fv_real *sc;
  if constexpr (UseLocalScratch) {
    sc = local_arena;
  } else {
    const int64_t nx = a.ie - a.is + int64_t(1);
    const int64_t col = (i - a.is) + (j - a.js) * nx;
    sc = a.workspace + col * static_cast<int64_t>(a.workspace_scalars_per_column);
  }

  fv_real *pe_old  = sc + 0 * slot;
  fv_real *pn1     = sc + 1 * slot;
  fv_real *peO     = sc + 2 * slot;
  fv_real *pn2     = sc + 3 * slot;
  fv_real *prof_q  = sc + 4 * slot;  // scalar_profile_column's own q(:) scratch

  fv_real *dpn1     = sc + 5 * slot;
  fv_real *dpn2     = sc + 6 * slot;
  fv_real *dpe      = sc + 7 * slot;
  fv_real *dpeO     = sc + 8 * slot;
  fv_real *pt_pre   = sc + 9 * slot;
  fv_real *delz_pre = sc + 10 * slot;
  fv_real *q2_pt    = sc + 11 * slot;
  fv_real *q2_delz  = sc + 12 * slot;
  fv_real *c2       = sc + 13 * slot;
  fv_real *c3       = sc + 14 * slot;
  fv_real *c4       = sc + 15 * slot;
  fv_real *gam      = sc + 16 * slot;
  // extm is tiny (<=128 bytes/thread even at kMaxLevels) and not the
  // source of the local-memory-reservation defect (Prompt 68) -- kept as a
  // plain per-thread local array in both branches, same as riem_solver_c's
  // own bool-free precedent; no workspace slot needed for it.
  bool extm[kMaxLevels];

  const fv_real k1k = a.akap / (fv_real(1) - a.akap);  // :308 rdgas/cv_air
  const fv_real rrg = -a.rdgas / a.grav;                // :311

  // ---- read old pe/peln BEFORE either is overwritten below ---------------
  // NOTE (Prompt 55 fix): FV3_FID_PE/FV3_FID_PELN are registered from
  // Atm%pe(is-1:ie+1,npz+1,js-1:je+1) / Atm%peln(is:ie,npz+1,js:je), whose
  // physical Fortran dimension order is (i,k,j) -- k is the MIDDLE
  // dimension, unlike pt/delz/delp/pk/pkz which all have k last. add3's
  // registered slot1/slot2 therefore hold (k,j), not (j,k). RiemViewC3::at
  // is a plain positional accessor with no per-field axis awareness, so it
  // must be called as .at(i,k,j) for these two fields specifically to
  // address the real memory layout; every other field here keeps .at(i,j,k)
  // since its own registration already lists axes in (i,j,k) order.
  for (int k = 0; k <= km; ++k) {
    pe_old[k] = a.pe.at(i, k, j);
    pn1[k] = a.peln.at(i, k, j);  // read the existing device field, NOT
                                  // recomputed via log(pe) (driver-vs-
                                  // production divergence noted in Stage 1).
  }
  // Prompt 62: read the OLD pk boundary values BEFORE the closing update
  // below overwrites a.pk, mirroring pe_old/pn1 above. Fortran preserves
  // pk(i,j,1)/pk(i,j,km+1) as an exact pass-through (fv_mapz.F90:952-953)
  // rather than recomputing them via exp(akap*log(pe)) -- see the closing
  // update below for why a fresh recompute is NOT bitwise-equivalent.
  const fv_real pk_old_top = a.pk.at(i, j, 0);
  const fv_real pk_old_bot = a.pk.at(i, j, km);

  // ---- peO / pn2 derivation, :482-520 -------------------------------------
  peO[0] = a.ptop;
  peO[km] = pe_old[km];
  pn2[0] = pn1[0];
  pn2[km] = pn1[km];
  for (int k = 1; k < km; ++k) {
    peO[k] = a.ak.at(k) + a.bk.at(k) * pe_old[km];
    pn2[k] = fv3_log<fv_real>(peO[k]);
  }
  for (int k = 0; k < km; ++k) {
    dpn1[k] = pn1[k + 1] - pn1[k];
    dpn2[k] = pn2[k + 1] - pn2[k];
    dpe[k] = pe_old[k + 1] - pe_old[k];    // :576-585
    dpeO[k] = peO[k + 1] - peO[k];
  }

  // ---- pt pre-transform (:388, density-pt -> density-temp, dry/nwat=0) ---
  // Prompt 57: p57_pt_raw_dbg captures the raw kernel-input pt at the
  // debug column, before this transform is applied -- see the write-back
  // below and fv_mapz.F90's own [P57_FORTRAN_DUMP] pt_raw/
  // pt_post_pretransform prints for the cross-language comparison.
  //
  // Prompt 57 ROOT CAUSE (fixed at the call site, not here): this kernel's
  // internal pretransform is correct in isolation (matches every pt_delz
  // fixture/production-shape smoke test, which all supply true raw pt).
  // The bug was that fv_mapz.F90's OWN pt pretransform ran unconditionally
  // before this kernel's H2D sync, on both arms alike, so this kernel used
  // to receive an ALREADY-transformed pt and squared the transform.
  // Confirmed empirically (job 58516096, C32): [P57_FORTRAN_DUMP]
  // pt_post_pretransform == [P57_CUDA_DUMP] pt_in to 10 significant digits
  // (261.75501519), then pt_kernel_pretransform blew this up ~72x further
  // (18997.69) -- producing the Range_Warn/NaN corruption seen since
  // Prompt 54. Fixed by gating fv_mapz.F90's pretransform on
  // `.not. use_cuda_pt_delz_remap` (mirroring the delz pretransform's
  // existing, already-correct gate) so this kernel keeps receiving true
  // raw pt, as originally designed/validated. See
  // docs/performance/p57_pt_delz_kernel_first_divergence_2026-09-22.md.
  fv_real p57_pt_raw_dbg = fv_real(0);
  for (int k = 0; k < km; ++k) {
    const fv_real ptv = a.pt.at(i, j, k);
    pt_pre[k] = ptv * fv3_exp<fv_real>(
        k1k * fv3_log<fv_real>(rrg * dpe[k] / a.delz.at(i, j, k) * ptv));
    if (a.p57_dbg != nullptr && i == a.p57_dbg_i && j == a.p57_dbg_j &&
        k == a.p57_dbg_k) {
      p57_pt_raw_dbg = ptv;
    }
  }

  // ---- pt remap: iv=1, kord=9, q_min=t_min, coordinate = log(pe) ---------
  // Prompt 61: pt's Fortran call site (fv_mapz.F90:622-633) passes
  // optional_top=gmao_top_bc, optional_bot=gmao_bot_bc -- both .true.,
  // the unconditional GEOS production default (FV_StateMod.F90:558-559).
  // delz's own call site (:712-716) does NOT pass these -- its remap call
  // below is deliberately left with gmao_bot/gmao_top at their default
  // (false), unchanged from pre-Prompt-61 behavior. See
  // docs/performance/p61_map_scalar_gmao_boundary_conditions_2026-09-23.md.
  {
    Column<const fv_real> pn1v{pn1, 1}, dpn1v{dpn1, 1}, pn2v{pn2, 1},
        dpn2v{dpn2, 1}, ptprev{pt_pre, 1};
    Column<fv_real> q2ptv{q2_pt, 1};
    map_scalar_column<fv_real>(pn1v, dpn1v, pn2v, dpn2v, ptprev, q2ptv, c2, c3,
                                c4, gam, prof_q, extm, km, km, 1,
                                pt_delz_t_min<fv_real>(),
                                /*gmao_bot=*/true, /*gmao_top=*/true);
  }

  // ---- delz pre-transform (:601, height -> specific volume/grav) ---------
  for (int k = 0; k < km; ++k) delz_pre[k] = -a.delz.at(i, j, k) / dpe[k];

  // ---- delz remap: iv=1, kord=9, no q_min, coordinate = pe ---------------
  {
    Column<const fv_real> pe1v{pe_old, 1}, dpe1v{dpe, 1}, pe2v{peO, 1},
        dpe2v{dpeO, 1}, dzprev{delz_pre, 1};
    Column<fv_real> q2dzv{q2_delz, 1};
    map_scalar_column<fv_real>(pe1v, dpe1v, pe2v, dpe2v, dzprev, q2dzv, c2, c3,
                                c4, gam, prof_q, extm, km, km, 1,
                                pt_delz_no_min<fv_real>());
  }

  // ---- delz post-transform (:611) + joint pkz (:928-930), using NEW
  //      delp(=dpeO)/delz/pt ------------------------------------------------
  for (int k = 0; k < km; ++k) {
    const fv_real new_delp = dpeO[k];
    const fv_real delz_out = -q2_delz[k] * new_delp;
    const fv_real pkz_out = fv3_exp<fv_real>(
        a.akap * fv3_log<fv_real>(rrg * new_delp / delz_out * q2_pt[k]));
    a.pt.at(i, j, k) = q2_pt[k];
    a.delz.at(i, j, k) = delz_out;
    a.delp.at(i, j, k) = new_delp;
    a.pkz.at(i, j, k) = pkz_out;
  }

  // ---- Prompt 57: single-column pt-pretransform diagnostic write-back.
  // Inert unless a.p57_dbg is non-null (launcher-gated, opt-in only). Not a
  // numerical change: writes to a caller-owned scratch buffer, never read
  // back by this kernel or any other production code path.
  if (a.p57_dbg != nullptr && i == a.p57_dbg_i && j == a.p57_dbg_j) {
    a.p57_dbg[0] = p57_pt_raw_dbg;         // pt as handed to this kernel
    a.p57_dbg[1] = pt_pre[a.p57_dbg_k];    // kernel's OWN pre-transform of it
    a.p57_dbg[2] = q2_pt[a.p57_dbg_k];     // final remapped pt written back
  }

  // ---- delp/pe/peln/pk closing update, :789-848 ---------------------------
  // Prompt 55: a.pe/a.peln use .at(i,k,j) here too -- see the read-loop note
  // above. a.pk keeps .at(i,j,k): Atm%pk(is:ie,js:je,npz+1) has k last, so
  // its registration already matches RiemViewC3's (i,j,k) convention.
  //
  // Prompt 62: pk at the two vertical boundary edges (k=0 top, k=km bottom)
  // is preserved as the OLD value read above, matching Fortran's exact
  // pass-through (fv_mapz.F90:952-953: pk2(i,1)=pk(i,j,1),
  // pk2(i,km+1)=pk(i,j,km+1) -- read back unchanged, never recomputed).
  // Only the interior edges (0<k<km, matching Fortran's k=2..km) recompute
  // via exp(akap*pn2[k]), unchanged from before. peln needs no equivalent
  // fix: pn2[0]/pn2[km] are already set to pn1[0]/pn1[km] (the old peln,
  // read above) rather than a fresh log(), already matching Fortran's own
  // pn2(i,1)=peln(i,1,j)/pn2(i,km+1)=peln(i,km+1,j) pass-through -- this is
  // why peln was already validated bitwise in Prompt 61.
  for (int k = 0; k <= km; ++k) {
    a.pe.at(i, k, j) = peO[k];
    a.peln.at(i, k, j) = pn2[k];
    if (k == 0) {
      a.pk.at(i, j, k) = pk_old_top;
    } else if (k == km) {
      a.pk.at(i, j, k) = pk_old_bot;
    } else {
      a.pk.at(i, j, k) = fv3_exp<fv_real>(a.akap * pn2[k]);
    }
  }
}

}  // namespace

fv3_gpu_status launch_pt_delz_remap(const PtDelzRemapArgs &a, void *stream,
                                    fv3_gpu_error *error) {
  clear_error(error);
  const char *api = "launch_pt_delz_remap";

  // ---- rejections. All of these precede every state mutation. ------------
  if (!stream)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "null CUDA stream"));

  if (a.ie < a.is || a.je < a.js)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_INVALID_ARGUMENT, api, "empty compute bounds"));

  if (!a.pe.p || !a.peln.p || !a.pt.p || !a.delz.p || !a.delp.p || !a.pk.p ||
      !a.pkz.p || !a.ak.p || !a.bk.p)
    return static_cast<fv3_gpu_status>(fail(
        error, FV3_GPU_INVALID_ARGUMENT, api,
        "null device field: pt/delz remap requires pe, peln, pt, delz, "
        "delp, pk, pkz, ak, bk"));

  if (a.km < 3)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km < 3: scalar_profile_column's Huynh interior loop "
             "(k=2..km-3) and bottom-layer terms require at least 3 levels"));

  const bool use_workspace = a.workspace != nullptr;
  if (use_workspace) {
    const uint64_t need_per_column = pt_delz_remap_scalars_per_column(a.km);
    const uint64_t columns = pt_delz_remap_columns(a.is, a.ie, a.js, a.je);
    if (a.workspace_scalars_per_column < need_per_column)
      return static_cast<fv3_gpu_status>(
          fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
               "workspace_scalars_per_column too small for km"));
    if (a.workspace_bytes <
        columns * a.workspace_scalars_per_column * sizeof(fv_real))
      return static_cast<fv3_gpu_status>(
          fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
               "workspace too small for the is..ie x js..je column count of "
               "pt/delz remap"));
  } else if (a.km > kMaxLevels) {
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "km > 128 requires a device workspace: the local-array column "
             "scratch path is capped at 128 levels"));
  }

  if (!(a.grav > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "grav must be positive: rrg = -rdgas/grav (fv_mapz.F90:311)"));

  if (!(a.rdgas > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "rdgas must be positive"));

  if (a.akap <= fv_real(0) || a.akap >= fv_real(1))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "akap must be in (0,1): k1k = akap/(1-akap) (fv_mapz.F90:308)"));

  if (!(a.ptop > fv_real(0)))
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_UNSUPPORTED_CONFIGURATION, api,
             "ptop must be positive: peO(1) = ptop (fv_mapz.F90:482)"));

  const dim3 threads(16, 8);
  const int64_t nx = a.ie - a.is + 1;
  const int64_t ny = a.je - a.js + 1;
  const dim3 grid(static_cast<unsigned>((nx + 15) / 16),
                  static_cast<unsigned>((ny + 7) / 8));
  auto s = static_cast<cudaStream_t>(stream);

  // Prompt 68 Stage 2: one-time device-memory probe bracketing the FIRST
  // pt_delz_remap_kernel launch in this process, opt-in only, zero effect
  // unless RUN_CUDA_FV3_PTDELZ_MEMPROBE is set. Tests the hypothesis that
  // the OOM (Prompt 67) is CUDA's implicit per-context local-memory backing
  // pool, sized to the device's max resident-thread capacity (not this
  // launch's actual grid), reserved lazily on first launch of a kernel
  // needing this much per-thread local memory (kMaxLevels=128-capped local
  // arrays; see the km>128 rejection message above). Not a numerical
  // change; removable after Prompt 68 closes.
  {
    static bool probed = false;
    if (!probed && std::getenv("RUN_CUDA_FV3_PTDELZ_MEMPROBE")) {
      probed = true;
      size_t free_before = 0, total_before = 0;
      cudaDeviceSynchronize();
      cudaMemGetInfo(&free_before, &total_before);
      std::fprintf(stderr,
          "[P68_MEMPROBE] before_launch free=%zu total=%zu grid=(%u,%u) "
          "threads=(%u,%u)\n",
          free_before, total_before, grid.x, grid.y, threads.x, threads.y);
      std::fflush(stderr);

      if (use_workspace)
        pt_delz_remap_kernel<false><<<grid, threads, 0, s>>>(a);
      else
        pt_delz_remap_kernel<true><<<grid, threads, 0, s>>>(a);
      const cudaError_t probe_launch_status = cudaGetLastError();
      cudaDeviceSynchronize();
      size_t free_after = 0, total_after = 0;
      cudaMemGetInfo(&free_after, &total_after);
      std::fprintf(stderr,
          "[P68_MEMPROBE] after_launch free=%zu total=%zu launch_status=%d "
          "(%s) delta_bytes=%lld\n",
          free_after, total_after, static_cast<int>(probe_launch_status),
          cudaGetErrorString(probe_launch_status),
          static_cast<long long>(static_cast<int64_t>(free_before) -
                                  static_cast<int64_t>(free_after)));
      std::fflush(stderr);

      if (probe_launch_status != cudaSuccess)
        return static_cast<fv3_gpu_status>(fail(
            error, FV3_GPU_CUDA_ERROR, api,
            cudaGetErrorString(probe_launch_status),
            static_cast<int>(probe_launch_status)));
      return FV3_GPU_SUCCESS;
    }
  }

  if (use_workspace)
    pt_delz_remap_kernel<false><<<grid, threads, 0, s>>>(a);
  else
    pt_delz_remap_kernel<true><<<grid, threads, 0, s>>>(a);

  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess)
    return static_cast<fv3_gpu_status>(
        fail(error, FV3_GPU_CUDA_ERROR, api, cudaGetErrorString(launch_status),
             static_cast<int>(launch_status)));
  return FV3_GPU_SUCCESS;
}

int launch_pt_delz_remap_ctx(const ExecutionContext &x, fv3_gpu_error *error) noexcept {
  const int context_status = validate_execution_context(x, error);
  if (context_status != FV3_GPU_SUCCESS) return context_status;
  const char *api = "launch_pt_delz_remap_ctx";
  if (x.submission.stream_id != StreamId::compute)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "pt/delz remap must be enqueued on the compute stream");

  const auto *v_pe = x.fields->find(FieldId::pe);
  const auto *v_peln = x.fields->find(FieldId::peln);
  const auto *v_pt = x.fields->find(FieldId::pt);
  const auto *v_delz = x.fields->find(FieldId::delz);
  const auto *v_delp = x.fields->find(FieldId::delp);
  const auto *v_pk = x.fields->find(FieldId::pk);
  const auto *v_pkz = x.fields->find(FieldId::pkz);
  const auto *v_ak = x.fields->find(FieldId::ak);
  const auto *v_bk = x.fields->find(FieldId::bk);
  if (!v_pe || !v_peln || !v_pt || !v_delz || !v_delp || !v_pk || !v_pkz ||
      !v_ak || !v_bk)
    return fail(error, FV3_GPU_INVALID_ARGUMENT, api,
                "pt/delz remap requires pe, peln, pt, delz, delp, pk, pkz, "
                "ak, bk");

  const auto &b = x.immutable->bounds;
  const auto &cfg = x.immutable->config;

  // Prompt 56 Stage 4: one-time live-registry-view dump, opt-in only.
  {
    static bool dumped = false;
    if (!dumped && std::getenv("RUN_CUDA_FV3_PTDELZ_DUMP_VIEWS")) {
      dumped = true;
      std::fprintf(stderr,
          "[P56_VIEW_DUMP] bounds is=%lld ie=%lld js=%lld je=%lld npz=%lld\n",
          static_cast<long long>(b.is), static_cast<long long>(b.ie),
          static_cast<long long>(b.js), static_cast<long long>(b.je),
          static_cast<long long>(b.npz));
      p56_dump_view_once("pe", *v_pe);
      p56_dump_view_once("peln", *v_peln);
      p56_dump_view_once("pt", *v_pt);
      p56_dump_view_once("delz", *v_delz);
      p56_dump_view_once("delp", *v_delp);
      p56_dump_view_once("pk", *v_pk);
      p56_dump_view_once("pkz", *v_pkz);
      p56_dump_view_once("ak", *v_ak);
      p56_dump_view_once("bk", *v_bk);
    }
  }

  PtDelzRemapArgs a{};
  a.pe = riem_c_view3(*v_pe);
  a.peln = riem_c_view3(*v_peln);
  a.pt = riem_c_view3(*v_pt);
  a.delz = riem_c_view3(*v_delz);
  a.delp = riem_c_view3(*v_delp);
  a.pk = riem_c_view3(*v_pk);
  a.pkz = riem_c_view3(*v_pkz);
  a.ak = ptdz_cview1(*v_ak);
  a.bk = ptdz_cview1(*v_bk);

  a.is = b.is; a.ie = b.ie; a.js = b.js; a.je = b.je;
  a.km = static_cast<int32_t>(b.npz);

  a.akap = fv_real(cfg.akap);
  a.rdgas = fv_real(cfg.rdgas);
  a.grav = fv_real(cfg.grav);
  a.ptop = fv_real(cfg.ptop);

  // Prompt 68: prefer the persistent WP-E vertical workspace (same slice
  // Riem_Solver_c/Riem_Solver3 already use); fall back to per-thread local
  // scratch only when it cannot hold this column count. Safe to share: this
  // launcher requires StreamId::compute (checked above), same as every
  // other vertical-solver consumer of WorkspaceId::vertical, so GPU stream
  // ordering guarantees no concurrent use of the same scratch region.
  {
    const uint64_t need_per_column = pt_delz_remap_scalars_per_column(a.km);
    const uint64_t columns = pt_delz_remap_columns(a.is, a.ie, a.js, a.je);
    const auto *slice = workspace_slice(*x.workspace, WorkspaceId::vertical);
    if (slice && slice->data &&
        slice->bytes >= columns * need_per_column * sizeof(fv_real)) {
      a.workspace = static_cast<fv_real *>(slice->data);
      a.workspace_scalars_per_column = need_per_column;
      a.workspace_bytes = slice->bytes;
    }
  }

  // Prompt 56 Stage 5: pre-kernel INPUT value dump, same one-time guard as
  // Stage 4 above (RUN_CUDA_FV3_PTDELZ_DUMP_VIEWS), one fixed interior
  // column well away from any halo/edge (midpoint of is..ie/js..je).
  {
    static bool dumped_values = false;
    if (!dumped_values && std::getenv("RUN_CUDA_FV3_PTDELZ_DUMP_VIEWS")) {
      dumped_values = true;
      const int64_t di = (b.is + b.ie) / 2;
      const int64_t dj = (b.js + b.je) / 2;
      const int dkm = static_cast<int>(b.npz);
      std::fprintf(stderr, "[P56_VALUE_DUMP] column i=%lld j=%lld km=%d\n",
                   static_cast<long long>(di), static_cast<long long>(dj), dkm);
      p56_dump_column_values3("pe", a.pe, di, dj, 0, dkm, /*axis_ikj=*/true);
      p56_dump_column_values3("peln", a.peln, di, dj, 0, dkm, /*axis_ikj=*/true);
      p56_dump_column_values3("pt", a.pt, di, dj, 0, dkm - 1);
      p56_dump_column_values3("delz", a.delz, di, dj, 0, dkm - 1);
      p56_dump_column_values3("delp", a.delp, di, dj, 0, dkm - 1);
      p56_dump_column_values3("pkz", a.pkz, di, dj, 0, dkm - 1);
      p56_dump_column_values1("ak", a.ak, 0, dkm);
      p56_dump_column_values1("bk", a.bk, 0, dkm);
    }
  }

  // Prompt 57: single-column pt-pretransform diagnostic, opt-in only, zero
  // effect unless RUN_CUDA_FV3_PTDELZ_DUMP_PRETRANSFORM is set. Localizes
  // the pt/delz kernel numerical divergence (Prompt 56, Category H) by
  // comparing this kernel's own pre-kernel input / internal pre-transform
  // output against fv_mapz.F90's own [P57_FORTRAN_DUMP] pt_raw/
  // pt_post_pretransform prints at the same column. New device
  // synchronization occurs only inside this opt-in block.
  fv_real *p57_dbg_dev = nullptr;
  int32_t p57_dbg_k = -1;
  {
    static bool p57_dumped = false;
    if (!p57_dumped && std::getenv("RUN_CUDA_FV3_PTDELZ_DUMP_PRETRANSFORM")) {
      p57_dumped = true;
      const int64_t di = (b.is + b.ie) / 2;
      const int64_t dj = (b.js + b.je) / 2;
      // 0-based; matches fv_mapz.F90's guarded k==5 (1-based) at the same
      // physical level.
      p57_dbg_k = 4;
      cudaMalloc(&p57_dbg_dev, 3 * sizeof(fv_real));
      a.p57_dbg = p57_dbg_dev;
      a.p57_dbg_i = di;
      a.p57_dbg_j = dj;
      a.p57_dbg_k = p57_dbg_k;
      std::fprintf(stderr,
          "[P57_CUDA_DUMP] target column i=%lld j=%lld k=%d (bounds is=%lld "
          "ie=%lld js=%lld je=%lld)\n",
          static_cast<long long>(di), static_cast<long long>(dj), p57_dbg_k,
          static_cast<long long>(b.is), static_cast<long long>(b.ie),
          static_cast<long long>(b.js), static_cast<long long>(b.je));
    }
  }

  const int launch_rc = launch_pt_delz_remap(a, x.submission.stream, error);

  // Prompt 59 Stage 4: minimal, one-time, launcher-side-only (no kernel
  // change) POST-kernel sentinel for the closing-update pe/peln write.
  // Reuses Prompt 56's p56_dump_column_values3 unchanged. Reads the SAME
  // device memory twice with two different offset conventions at the same
  // debug column: axis_ikj=true is the correct (i,k,j) address (per the
  // read-loop note above and the Stage 0 registry re-derivation); axis_ikj=
  // false is the address the closing-update loop's current `.at(i,j,k)`
  // call actually targets. Zero effect unless
  // RUN_CUDA_FV3_PTDELZ_DUMP_POSTWRITE is set.
  {
    static bool p59_dumped = false;
    if (!p59_dumped && std::getenv("RUN_CUDA_FV3_PTDELZ_DUMP_POSTWRITE")) {
      p59_dumped = true;
      cudaStreamSynchronize(static_cast<cudaStream_t>(x.submission.stream));
      const int64_t di = (b.is + b.ie) / 2;
      const int64_t dj = (b.js + b.je) / 2;
      const int dkm = static_cast<int>(b.npz);
      std::fprintf(stderr,
          "[P59_POSTWRITE] column i=%lld j=%lld km=%d -- correct (i,k,j) "
          "address vs. code's actual (i,j,k)-positional address\n",
          static_cast<long long>(di), static_cast<long long>(dj), dkm);
      p56_dump_column_values3("pe_correct_ikj", a.pe, di, dj, 0, dkm,
                               /*axis_ikj=*/true);
      p56_dump_column_values3("pe_code_ijk", a.pe, di, dj, 0, dkm,
                               /*axis_ikj=*/false);
      p56_dump_column_values3("peln_correct_ikj", a.peln, di, dj, 0, dkm,
                               /*axis_ikj=*/true);
      p56_dump_column_values3("peln_code_ijk", a.peln, di, dj, 0, dkm,
                               /*axis_ikj=*/false);
    }
  }

  if (p57_dbg_dev != nullptr) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(x.submission.stream));
    fv_real hv[3] = {fv_real(0), fv_real(0), fv_real(0)};
    cudaMemcpy(hv, p57_dbg_dev, 3 * sizeof(fv_real), cudaMemcpyDeviceToHost);
    std::fprintf(stderr,
        "[P57_CUDA_DUMP] pt_in=%.10e pt_kernel_pretransform=%.10e "
        "pt_out=%.10e\n",
        static_cast<double>(hv[0]), static_cast<double>(hv[1]),
        static_cast<double>(hv[2]));
    std::fflush(stderr);
    cudaFree(p57_dbg_dev);
  }

  return launch_rc;
}

}  // namespace fv3::gpu::vertical
