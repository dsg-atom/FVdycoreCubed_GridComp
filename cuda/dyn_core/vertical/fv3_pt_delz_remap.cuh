#pragma once
//===========================================================================
// WP-C: fv_mapz_mod vertical remap -- production pt/delz entry (Prompt 52C).
//
// Authoritative Fortran source of truth (numerical source of truth,
// read-only reference):
//   GFDL_atmos_cubed_sphere (tag geos/v3.0.0, ca30e26)
//   model/fv_mapz.F90:482-536   peO/pn1/pn2/dpn1/dpn2 derivation + pt remap
//   model/fv_mapz.F90:576-613   dpe/dpeO derivation + delz remap
//   model/fv_mapz.F90:789-848   delp/pe/peln/pk closing update
//   model/fv_mapz.F90:900-935   non-hydrostatic pkz (MOIST_CAPPA/else, both
//                                degenerate to the same dry formula at
//                                nwat=0 -- see the launcher's rejection of
//                                nwat!=0, enforced Fortran-side only).
// Validated, already-passing standalone reference (surrounding arithmetic
// only, map_scalar itself untouched): pt_delz-driver/main.F90, Prompt
// 52C-prep (docs/performance/p52c_prep_pt_delz_fixture_2026-09-21.md).
//
// Scope implemented ONLY (matches the validated C512 Held-Suarez config):
//   remap_option=0 (remap_t), gmao_remap=0, kord_mt=kord_wz=kord_tm=9,
//   iv=1 for both pt and delz, nwat=0 (MOIST_CAPPA/USE_COND dry-degenerate).
//   mapn_tracer, wind remap, w remap, iv=-1/-2, ppm_profile, gmao_remap,
//   kord<=7 are OUT OF SCOPE and not touched by this file.
//
// kord and nwat do not exist in NumericalConfig (Fortran-only values) and
// therefore CANNOT be independently verified on the C++ side. Every launch
// through this entry point is trusted to have already been gated at the
// Fortran dispatch call site on kord_mt==kord_wz==kord_tm==9, nwat==0,
// remap_option==0, gmao_remap==0. This file's own guards check only what
// is representable in NumericalConfig/DomainBounds (akap, rdgas, grav,
// ptop, km bounds).
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_gpu_types.hpp"
#include "fv3_gpu_internal.hpp"
#include "fv3_riem_solver_c.cuh"  // reuse RiemCViewC3/RiemViewC3 + adapters
#include <cstdint>

namespace fv3::gpu::vertical {

#if defined(__CUDACC__)
#define FV3_PTDZ_HD __host__ __device__
#else
#define FV3_PTDZ_HD
#endif

// ak/bk are pure vertical hybrid-coordinate coefficients (km+1 values),
// identical for every horizontal column -- a rank-1 registry field, unlike
// every other view in this file.
struct PtDelzView1 {
  const fv_real *p = nullptr;
  int64_t l0 = 0, s0 = 0;
  FV3_PTDZ_HD inline fv_real at(int64_t k) const { return p[(k - l0) * s0]; }
};
inline PtDelzView1 ptdz_cview1(const DeviceViewBase &v) {
  return {static_cast<const fv_real *>(v.data), v.lower[0], v.stride[0]};
}

// Per-thread local-scratch level cap. Same precedent/rationale as
// riem_solver_c_max_local_levels: km > this is REJECTED, never wrapped.
// This entry's per-column scratch (~15 arrays of km/km+1 scalars) is far
// smaller than Riem_Solver_c's 19 slots, so no device-workspace path is
// provided at all -- local/stack arrays are sufficient at every km this
// project targets (production km=10; cap kept at the same 128 precedent).
inline constexpr int pt_delz_remap_max_local_levels = 128;

struct PtDelzRemapArgs {
  // ---- intent(in), vertical-only, shared by every column ------------------
  PtDelzView1 ak;  // ak <- FieldId::ak (km+1, vertical-only)
  PtDelzView1 bk;  // bk <- FieldId::bk (km+1, vertical-only)

  // ---- intent(inout) (read on entry, remapped/closed in place) ------------
  RiemViewC3 pe;    // pe    <- FieldId::pe    (km+1; old grid in, peO out)
  RiemViewC3 peln;  // peln  <- FieldId::peln  (km+1; old pn1 in, pn2 out)
  RiemViewC3 pt;    // pt    <- FieldId::pt    (km levels)
  RiemViewC3 delz;  // delz  <- FieldId::delz  (km levels)
  RiemViewC3 delp;  // delp  <- FieldId::delp  (km levels; becomes dpeO)
  RiemViewC3 pk;    // pk    <- FieldId::pk    (km+1 levels)
  RiemViewC3 pkz;   // pkz   <- FieldId::pkz   (km levels)

  int64_t is = 0, ie = 0, js = 0, je = 0;
  int32_t km = 0;  // Fortran km (= npz)

  fv_real akap = fv_real(0);
  fv_real rdgas = fv_real(0);
  fv_real grav = fv_real(0);
  fv_real ptop = fv_real(0);
};

// Enqueue-only. `stream` is the WP-E-owned compute stream (cudaStream_t).
// No host/device transfer and no device synchronization is performed.
fv3_gpu_status launch_pt_delz_remap(const PtDelzRemapArgs &a, void *stream,
                                    fv3_gpu_error *error = nullptr);

// WP-E ExecutionContext entry, LaunchFn-compatible.
int launch_pt_delz_remap_ctx(const ExecutionContext &, fv3_gpu_error *) noexcept;

} // namespace fv3::gpu::vertical
