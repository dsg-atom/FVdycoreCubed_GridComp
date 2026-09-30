#pragma once
//===========================================================================
// WP-C: fv_mapz_mod vertical remap -- production u/v (REMAP_UV) entry
// (Prompt 71).
//
// Authoritative Fortran source of truth (numerical source of truth,
// read-only reference):
//   GFDL_atmos_cubed_sphere (tag geos/v3.0.0, ca30e26)
//   model/fv_mapz.F90:907-1010  Lagrangian_to_Eulerian, timer 'Remap_UV':
//     :907-926   u-block: pe1 (source, j-averaged pe)/pe2 (target,
//                ak/bk + j-averaged surface pressure), dpe1/dpe2,
//                map_scalar(iv=-1, kord=kord_mt), write-back u(is:ie,j,:).
//                Unconditional every j=js..je+1.
//     :928-1008  v-block: pe0 (source, i-averaged pe)/pe3 (target,
//                ak/bk + i-averaged surface pressure), dpe0/dpe3,
//                map_scalar(iv=-1, kord=kord_mt), write-back
//                v(is:ie+1,j,:). Guarded `if (j < je+1)`.
//
// Scope implemented ONLY (matches the validated production first-run
// config, Prompt 70's source map): kord_mt=9, iv=-1 for both u and v,
// q_min absent (-> map_scalar's own no_min=-1e25 default),
// optional_bot/optional_top absent (-> both .false., i.e. the GMAO
// boundary-condition branches are unreachable from REMAP_UV in
// production -- Prompt 70 Section 4). The 4 additional `mfx`/`mfy`/`cx`/
// `cy` map_scalar calls in the same source block are confirmed dead in
// production (never passed at the fv_dynamics.F90 call site) and are
// OUT OF SCOPE -- not implemented here. iv=-2 (w) and tracer remap
// (mapn_tracer) are separate routines, also OUT OF SCOPE.
//
// kord is a Fortran-only value with no NumericalConfig representation and
// therefore CANNOT be independently verified on the C++ side. Every
// launch through this entry point is trusted to have already been gated
// at the Fortran dispatch call site on kord_mt==9 (mirrors
// fv3_pt_delz_remap.cuh's identical kord trust boundary).
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_gpu_types.hpp"
#include "fv3_gpu_internal.hpp"
#include "fv3_pt_delz_remap.cuh"  // reuse PtDelzView1/ptdz_cview1 for ak/bk
#include "fv3_riem_solver_c.cuh"  // reuse RiemViewC3/riem_c_view3 for u/v/pe
#include <cstdint>

namespace fv3::gpu::vertical {

#if defined(__CUDACC__)
#define FV3_RMUV_HD __host__ __device__
#else
#define FV3_RMUV_HD
#endif

// Per-thread local-scratch level cap. Same precedent/rationale as
// pt_delz_remap_max_local_levels/riem_solver_c_max_local_levels: km > this
// is REJECTED when no device workspace is supplied, never silently
// truncated. See fv3_pt_delz_remap.cuh's header comment for the Prompt 68
// local-memory-reservation defect this avoids by preferring the shared
// device workspace path.
inline constexpr int remap_uv_max_local_levels = 128;

// Per-column scratch slot count. u and v are computed strictly
// sequentially within one thread (u's map_scalar_column call fully
// completes and writes a.u back BEFORE v's computation begins -- no
// concurrency within a thread), so the SAME 11 slots are reused for both
// fields rather than allocating 11 slots each:
//   pe_src, dpe_src, pe_tgt, dpe_tgt, q1_in, q2_out, prof_q,
//   c2, c3, c4, gam                                                (11)
// (uniform (km+1)-scalar slot pitch, same convention as
// pt_delz_remap_scratch_slots -- km-sized arrays waste one trailing
// scalar rather than needing a second stride.)
inline constexpr int remap_uv_scratch_slots = 11;

FV3_RMUV_HD inline uint64_t remap_uv_scalars_per_column(int km) noexcept {
  return static_cast<uint64_t>(remap_uv_scratch_slots) *
         static_cast<uint64_t>(km + 1);
}

// Column count actually launched: the UNION bounding box is..ie+1 x
// js..je+1 (NOT the plain interior is..ie x js..je) -- one thread per
// (i,j) in this box, with per-thread guards `i<=ie` (u) / `j<=je` (v)
// selecting which of u/v (or both, or neither at the (ie+1,je+1) corner)
// that thread computes, exactly mirroring Fortran's own single
// `do 1003 j=js,je+1` loop with an unconditional u-block and a
// `if (j<je+1)`-guarded v-block.
inline uint64_t remap_uv_columns(int64_t is, int64_t ie, int64_t js,
                                 int64_t je) noexcept {
  const int64_t nx = (ie + 1) - is + 1;
  const int64_t ny = (je + 1) - js + 1;
  if (nx <= 0 || ny <= 0) return 0;
  return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny);
}

inline uint64_t remap_uv_workspace_bytes(int km, int64_t is, int64_t ie,
                                         int64_t js, int64_t je) noexcept {
  return remap_uv_columns(is, ie, js, je) * remap_uv_scalars_per_column(km) *
         sizeof(fv_real);
}

struct RemapUvArgs {
  // ---- intent(in), vertical-only, shared by every column ------------------
  PtDelzView1 ak;  // ak <- FieldId::ak (km+1, vertical-only)
  PtDelzView1 bk;  // bk <- FieldId::bk (km+1, vertical-only)

  // ---- intent(in) ----------------------------------------------------------
  RiemViewC3 pe;  // pe <- FieldId::pe (km+1; fully updated by PT/DELZ's own
                  // closing update, which this kernel must be enqueued after
                  // on the same stream -- see the .cu launcher note)

  // ---- intent(inout) (read on entry as q1, remapped/written back) --------
  RiemViewC3 u;  // u <- FieldId::u (km levels; domain is..ie x js..je+1)
  RiemViewC3 v;  // v <- FieldId::v (km levels; domain is..ie+1 x js..je)

  int64_t is = 0, ie = 0, js = 0, je = 0;
  int32_t km = 0;  // Fortran km (= npz)

  fv_real ptop = fv_real(0);

  // ---- optional device workspace (WorkspaceId::vertical), same pattern as
  // PtDelzRemapArgs (Prompt 68). workspace == nullptr -> per-thread local
  // arrays, km <= 128 enforced. workspace != nullptr -> device scratch of
  // at least remap_uv_workspace_bytes(km, is, ie, js, je), addressed with
  // workspace_scalars_per_column stride per column (>= scalars_per_column).
  fv_real *workspace = nullptr;
  uint64_t workspace_scalars_per_column = 0;
  uint64_t workspace_bytes = 0;
};

// Enqueue-only. `stream` is the WP-E-owned compute stream (cudaStream_t).
// No host/device transfer and no device synchronization is performed.
fv3_gpu_status launch_remap_uv(const RemapUvArgs &a, void *stream,
                               fv3_gpu_error *error = nullptr);

// WP-E ExecutionContext entry, LaunchFn-compatible.
int launch_remap_uv_ctx(const ExecutionContext &, fv3_gpu_error *) noexcept;

}  // namespace fv3::gpu::vertical
