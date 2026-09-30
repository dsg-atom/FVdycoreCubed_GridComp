#pragma once
//===========================================================================
// WP-C: fv_mapz_mod Remap_PressureVars entry (Prompt 92).
//
// Authoritative Fortran source of truth (numerical source of truth,
// read-only reference):
//   GFDL_atmos_cubed_sphere (tag geos/v3.0.0, ca30e26)
//   model/fv_mapz.F90:1239-1516  Remap_PressureVars timer block.
// Mapping/harness precedent (Prompt 91):
//   docs/performance/p91_remap_pressurevars_cuda_port_plan_2026-09-27.md
//   tests/vertical/test_remap_pressurevars_host_reference.cpp (authoritative
//   Option C host-reference transcription; this kernel must match it).
//
// Scope implemented (matches the validated production config, confirmed by
// Prompt 91's source mapping):
//   hydrostatic=false (else branch, :1381), remap_te=false (dead, :1382-1421
//   skipped), remap_t=true (MOIST_CAPPA branch, :1423-1446 -- the only live
//   branch), remap_pt=false (dead, :1470+ skipped), do_omega=false (dead,
//   skipped). moist_cv's nwat=0 case-default (cvm=cv_air, a true constant --
//   fv_mapz.F90:4192) is inlined; nwat!=0 (real MOIST_CAPPA/USE_COND with a
//   non-constant cvm) is OUT OF SCOPE, same triple-gated-rejection contract
//   already documented project-wide (project_architecture.md).
//
// q(sphum) IS supported (Prompt 92 Stage 1: cheap, closed-form, GO): cappa's
// formula divides by (1 + r_vir*q_sphum), so q_sphum==0 is NOT required for
// this port even though it is the only value exercised by every validated
// run to date (cold-start tracer-zero limitation, CURRENT_STATE.md §7).
//
// pe/peln/pk/delp/pkz ride the ordinary FieldRegistry residency path (same
// FieldIds pt/delz remap already uses). cappa and q(sphum) do NOT: cappa's
// FieldId (FV3_FID_CAPPA) is bound Fortran-side to an unrelated constant
// shadow array (kappa_dry), not the real Lagrangian_to_Eulerian-local cappa
// (fv3_gpu_field_registry.F90:629-637), and q has no FieldRegistry identity
// at all (same fact fv3_remap_q.cuh already documents). Both are handled the
// same way fv3_gpu_remap_q.cuh handles q: a dedicated, persistent, grow-only,
// context-owned device buffer sized to the full isd:ied x jsd:jed x km
// memory domain, with the caller (fv3_gpu_runtime.cpp's
// fv3_gpu_remap_pressurevars) owning ONE flat H2D/D2H transfer directly
// against the real host pointers (cappa's own array is itself fully
// contiguous over that domain -- fv_mapz.F90's local `cappa` argument is
// declared `dimension(isd:,jsd:,1:)`, matching q's per-tracer-plane
// contiguity). cappa's transfer round-trips (H2D then D2H) so halo bytes
// outside is:ie x js:je pass through unchanged; q(sphum) is H2D-only
// (read-only input).
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_gpu_types.hpp"
#include "fv3_gpu_internal.hpp"
#include "fv3_riem_solver_c.cuh"       // RiemViewC3 + riem_c_view3
#include "fv3_pt_delz_remap.cuh"       // PtDelzView1 + ptdz_cview1 (ak/bk)
#include <cstdint>

namespace fv3::gpu::vertical {

#if defined(__CUDACC__)
#define FV3_PVARS_HD __host__ __device__
#else
#define FV3_PVARS_HD
#endif

// Dedicated view for the two fields with no FieldRegistry identity here
// (cappa, q(sphum)) -- a flat, contiguous, full isd:ied x jsd:jed x km
// memory domain, i fastest, then j, then k (matches both the real Fortran
// `cappa(isd:,jsd:,1:)` local array layout and a single q(:,:,:,sphum)
// tracer plane's layout, since nq is the outermost/slowest Fortran
// dimension). Unlike RiemViewC3, this is NOT read out of the FieldRegistry;
// the caller (fv3_gpu_runtime.cpp) constructs it directly from a persistent
// device buffer pointer plus the domain bounds.
struct PressureVarsPlaneView {
  fv_real *p = nullptr;
  int64_t isd = 0, jsd = 0, nih = 0, njh = 0;
  FV3_PVARS_HD inline fv_real &at(int64_t i, int64_t j, int k) const {
    return p[(i - isd) + (j - jsd) * nih +
             static_cast<int64_t>(k) * nih * njh];
  }
};

// Per-column scalars needed: no monotone remap (map_scalar_column) is
// involved in this block at all -- pe2/pn2/pk2 are a direct closed-form
// recompute from ak/bk/pe(surface), and cappa/pkz are a direct closed-form
// pointwise expression. Unlike fv3_pt_delz_remap.cuh, this kernel needs NO
// per-thread local scratch array and NO device workspace: pe/peln/pk are
// written directly into their own FieldRegistry views in a first pass, then
// re-read from those SAME (now-updated) views in a second pass to form
// delp -- a single thread performs both passes sequentially, so there is no
// race. This eliminates the Prompt 68 local-memory-reservation risk by
// construction (no kMaxLevels-capped local array of any kind).

inline uint64_t remap_pressurevars_columns(int64_t is, int64_t ie, int64_t js,
                                           int64_t je) noexcept {
  const int64_t nx = ie - is + 1;
  const int64_t ny = je - js + 1;
  if (nx <= 0 || ny <= 0) return 0;
  return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny);
}

// Elements needed for the cappa/q(sphum) dedicated buffers: the FULL
// isd:ied x jsd:jed x km memory domain (not just the interior is:ie x
// js:je compute footprint) -- same convention as fv3_remap_q.cuh's
// remap_q_buffer, so the caller's single flat memcpy can address the real
// host array pointer directly with no host-side pack/unpack loop.
inline uint64_t remap_pressurevars_plane_elements(int64_t isd, int64_t ied,
                                                  int64_t jsd, int64_t jed,
                                                  int32_t km) noexcept {
  const int64_t nih = ied - isd + 1;
  const int64_t njh = jed - jsd + 1;
  if (nih <= 0 || njh <= 0 || km <= 0) return 0;
  return static_cast<uint64_t>(nih) * static_cast<uint64_t>(njh) *
         static_cast<uint64_t>(km);
}

struct PressureVarsArgs {
  // ---- intent(in), vertical-only, shared by every column ------------------
  PtDelzView1 ak;  // ak <- FieldId::ak (km+1, vertical-only)
  PtDelzView1 bk;  // bk <- FieldId::bk (km+1, vertical-only)

  // ---- intent(inout), FieldRegistry-resident -------------------------------
  RiemViewC3 pe;    // pe    <- FieldId::pe    (km+1; old pe(:,km+1,j) read,
                     //          then fully overwritten with pe2)
  RiemViewC3 peln;  // peln  <- FieldId::peln  (km+1; old boundary read,
                     //          interior recomputed)
  RiemViewC3 pk;    // pk    <- FieldId::pk    (km+1; old boundary read,
                     //          interior recomputed)
  RiemViewC3 delp;  // delp  <- FieldId::delp  (km levels; fully overwritten)
  RiemViewC3 pkz;   // pkz   <- FieldId::pkz   (km levels; fully overwritten)

  // ---- intent(in), FieldRegistry-resident, read-only for this kernel -------
  RiemViewC3 pt;    // pt    <- FieldId::pt    (km levels)
  RiemViewC3 delz;  // delz  <- FieldId::delz  (km levels)

  // ---- intent(in)/intent(out), NOT FieldRegistry-resident (see header) ----
  PressureVarsPlaneView q_sphum;  // read-only input, full isd:ied x jsd:jed
  PressureVarsPlaneView cappa;    // written for is:ie x js:je; halo bytes
                                  // outside that range are round-tripped
                                  // unchanged by the caller's H2D/D2H, never
                                  // touched by this kernel.

  int64_t is = 0, ie = 0, js = 0, je = 0;
  int32_t km = 0;  // Fortran km (= npz)

  fv_real akap = fv_real(0);
  fv_real rdgas = fv_real(0);
  fv_real grav = fv_real(0);
  fv_real ptop = fv_real(0);
  fv_real r_vir = fv_real(0);  // cfg.zvir == fv_mapz.F90's r_vir (rvgas/rdgas-1)
};

// Enqueue-only. `stream` is the WP-E-owned compute stream (cudaStream_t).
// No host/device transfer and no device synchronization is performed.
fv3_gpu_status launch_remap_pressurevars(const PressureVarsArgs &a, void *stream,
                                         fv3_gpu_error *error = nullptr);

// WP-E ExecutionContext entry, LaunchFn-compatible. Reads the dedicated
// cappa/q_sphum buffer pointers out of x.dispatch_params (same
// "additive tail field" convention DispatchParams::remap_q_buffer already
// documents), stashed there by fv3_gpu_remap_pressurevars immediately before
// invoking dispatch.remap_pressurevars(x, e).
int launch_remap_pressurevars_ctx(const ExecutionContext &, fv3_gpu_error *) noexcept;

} // namespace fv3::gpu::vertical
