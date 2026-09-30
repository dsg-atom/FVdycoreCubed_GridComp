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
// riem_solver_c_max_local_levels: km > this is REJECTED when no device
// workspace is supplied.
//
// Prompt 68: contrary to this file's original claim ("local/stack arrays
// are sufficient"), a kMaxLevels=128-capped per-thread local array declared
// unconditionally (independent of the actual runtime km) forces the CUDA
// driver to reserve a DEVICE-WIDE local-memory backing pool sized to the
// device's theoretical max resident-thread count, not the launched grid --
// measured directly via cudaMemGetInfo bracketing a single tiny (1024-
// thread) kernel launch: a 3,210,739,712-byte (~3.06 GiB) one-time drop in
// free device memory on an otherwise-idle A100 (job 58530194,
// RUN_CUDA_FV3_PTDELZ_MEMPROBE=1). At the authoritative 6-ranks-share-
// 1-A100 production density, this reservation alone is ~18.4 GiB (6x),
// sufficient by itself to exhaust a 40 GiB A100 alongside the existing
// per-rank persistent dyncore state -- root-causing the Prompt 67 OOM. See
// docs/performance/p68_pt_delz_c512_memory_footprint_2026-09-23.md.
//
// Fix (mirrors riem_solver_c_kernel's existing, already-validated
// workspace/local-scratch dual path exactly): prefer the persistent WP-E
// vertical workspace (WorkspaceId::vertical) via
// PtDelzRemapArgs::workspace, falling back to the per-thread local-array
// path below only when no workspace is supplied or it is too small. This
// entry's own per-column need (17 slots x (km+1) = 187 scalars at km=10)
// is well under the 422 scalars/column that slice is already sized for
// (Riem_Solver3's larger need) -- reuse requires zero additional device
// bytes. Safe by construction: both this kernel and every vertical solver
// that also uses WorkspaceId::vertical are required to enqueue on the same
// StreamId::compute stream, so GPU stream ordering (not overlap) guarantees
// any prior consumer of that scratch region has finished before this
// kernel starts.
inline constexpr int pt_delz_remap_max_local_levels = 128;

// Per-column scratch slot count. 17 uniform slots of (km+1) scalars each
// (same uniform-(km+1)-slot convention as riem_solver_c_scratch_slots, so
// the km-sized arrays waste one trailing scalar rather than needing a
// second stride):
//   pe_old, pn1, peO, pn2, prof_q                                   (5)
//   dpn1, dpn2, dpe, dpeO, pt_pre, delz_pre, q2_pt, q2_delz,
//   c2, c3, c4, gam                                                 (12)
inline constexpr int pt_delz_remap_scratch_slots = 17;

FV3_PTDZ_HD inline uint64_t
pt_delz_remap_scalars_per_column(int km) noexcept {
  return static_cast<uint64_t>(pt_delz_remap_scratch_slots) *
         static_cast<uint64_t>(km + 1);
}

// Column count actually solved: the plain interior domain is..ie x js..je
// (NOT Riem_Solver_c's widened is-1..ie+1/js-1..je+1 -- see this file's top
// comment).
inline uint64_t pt_delz_remap_columns(int64_t is, int64_t ie, int64_t js,
                                      int64_t je) noexcept {
  const int64_t nx = ie - is + 1;
  const int64_t ny = je - js + 1;
  if (nx <= 0 || ny <= 0) return 0;
  return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny);
}

inline uint64_t pt_delz_remap_workspace_bytes(int km, int64_t is, int64_t ie,
                                              int64_t js,
                                              int64_t je) noexcept {
  return pt_delz_remap_columns(is, ie, js, je) *
         pt_delz_remap_scalars_per_column(km) * sizeof(fv_real);
}

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

  // ---- Prompt 68: optional device workspace (WorkspaceId::vertical). -----
  // workspace == nullptr -> per-thread local arrays, km <= 128 enforced.
  // workspace != nullptr -> device scratch of at least
  //   pt_delz_remap_workspace_bytes(km, is, ie, js, je), addressed with
  //   workspace_scalars_per_column stride per column (>= scalars_per_column).
  fv_real *workspace = nullptr;
  uint64_t workspace_scalars_per_column = 0;
  uint64_t workspace_bytes = 0;

  // ---- Prompt 57: optional, additive, single-column pt-pretransform
  // diagnostic. nullptr/-1 unless RUN_CUDA_FV3_PTDELZ_DUMP_PRETRANSFORM is
  // set by the launcher; zero effect on every other thread/launch. Not a
  // numerical change -- writes to a caller-owned scratch buffer never read
  // back by production code.
  fv_real *p57_dbg = nullptr;
  int64_t p57_dbg_i = -1, p57_dbg_j = -1;
  int32_t p57_dbg_k = -1;
};

// Enqueue-only. `stream` is the WP-E-owned compute stream (cudaStream_t).
// No host/device transfer and no device synchronization is performed.
fv3_gpu_status launch_pt_delz_remap(const PtDelzRemapArgs &a, void *stream,
                                    fv3_gpu_error *error = nullptr);

// WP-E ExecutionContext entry, LaunchFn-compatible.
int launch_pt_delz_remap_ctx(const ExecutionContext &, fv3_gpu_error *) noexcept;

} // namespace fv3::gpu::vertical
