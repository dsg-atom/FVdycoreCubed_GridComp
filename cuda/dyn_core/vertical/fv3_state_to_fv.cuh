#pragma once
//===========================================================================
// WP-C/WP-E: FV_StateMod.F90 State_To_FV entry (Prompt 99).
//
// Authoritative Fortran source of truth (numerical source of truth,
// read-only reference):
//   GEOSfvdycore_fortran_perf checkout, src/Components/
//   @FVdycoreCubed_GridComp/FV_StateMod.F90:2519-2702, subroutine
//   State_To_FV. Called once per k_split sub-iteration, immediately before
//   fv_dynamics().
// Mapping/harness precedent (Prompt 98):
//   docs/performance/p98_state_to_fv_cuda_port_plan_2026-09-28.md
//   tests/host/test_state_to_fv_host_reference.cpp (authoritative host
//   reference this kernel must match).
//
// Stage 0/1 exact contract (Prompt 99). State_To_FV's body splits into
// exactly three phases by strict Fortran program order relative to the one
// `mpp_get_boundary` call it makes:
//
//   PHASE A (this file's launch_state_to_fv_phase_a/_ctx) -- everything
//     BEFORE mpp_get_boundary:
//       u(:,:,:) = tiny_number                       [full FieldRegistry
//       v(:,:,:) = tiny_number                        buffer fill]
//       u(isc:iec,jsc:jec,:) = STATE%VARS%U           [interior copy]
//       v(isc:iec,jsc:jec,:) = STATE%VARS%V           [interior copy]
//     (grid_type>=4 Doubly-Periodic branch is DEAD in production --
//     grid_type==0 is required by the Stage 8 selector below -- and is not
//     implemented.)
//
//   PHASE B (UNCHANGED, host-only, not in this file) -- the real
//     mpp_get_boundary(u,v,domain,...,gridtype=DGRID_NE,complete=.true.)
//     call itself. This CUDA port does not touch it: the Fortran-side
//     bridge (fv3_gpu_interface_state_to_fv_phase_a/_c,
//     fv3_gpu_dyncore_bridge.F90) D2H's the FULL u/v FieldRegistry buffers
//     via the EXISTING fv3_gpu_sync_to_host mechanism (a u/v-only mask),
//     lets the unmodified Fortran mpp_get_boundary run, performs the tiny
//     nbuffer/ebuffer edge-row scatter on the host (already-resident data,
//     see that file's header comment for why this one tiny copy stays on
//     the host rather than round-tripping nbuffer/ebuffer through a third
//     GPU kernel), then H2D's u/v back via the SAME existing mechanism
//     before Phase C runs.
//
//   PHASE C (this file's launch_state_to_fv_phase_c/_ctx) -- everything
//     AFTER mpp_get_boundary (the u/v edge-row scatter itself is handled
//     host-side as part of the Phase B bridge, see above; everything below
//     is genuinely independent of the boundary result and is the CUDA
//     candidate):
//       w(isc:iec,jsc:jec,:) = STATE%VARS%W           [interior copy;
//                                                       hydrostatic=false
//                                                       required by Stage 8]
//       pe(:,:,:) = tiny_number                       [full buffer fill]
//       pe(i,k,j) = STATE%VARS%PE(i,j,k)               [transpose+interior;
//                                                       (i,j,k)->(i,k,j)]
//       peln(i,k,j) = log(pe(i,k,j))                  [pointwise]
//       pk(i,j,k) = exp(akap*peln(i,k,j))             [pointwise]
//       ps(isc:iec,jsc:jec) = pe(isc:iec,km+1,jsc:jec) [packed into a
//                                                       dedicated scratch
//                                                       buffer, D2H'd
//                                                       directly to the
//                                                       caller's host ps
//                                                       pointer -- ps has
//                                                       NO FieldRegistry
//                                                       identity]
//       delp(:,:,:) = tiny_number                     [full buffer fill]
//       delp(i,j,k) = pe(i,k+1,j) - pe(i,k,j)          [pointwise]
//       pt(:,:,:) = tiny_number                       [full buffer fill]
//       pt(isc:iec,jsc:jec,:) = STATE%VARS%PT*STATE%VARS%PKZ [pointwise]
//       delz(isc:iec,jsc:jec,:) = STATE%VARS%DZ        [interior copy]
//       pkz(isc:iec,jsc:jec,:) = STATE%VARS%PKZ        [interior copy]
//     (SW_DYNAMICS branch is DEAD in production -- akap=kappa always taken
//     -- and is not implemented. ADJUST_SPLITS is DEAD in production and
//     is not implemented.)
//
// Index-order contract (Stage 0, confirmed against fv3_gpu_field_registry.
// F90's add3 registrations): u/v/w/delp/pt/delz/pkz are (i,j,k) (k last);
// pe/peln are (i,k,j) (k MIDDLE); pk is (i,j,k) despite sitting next to
// pe/peln in the field list. u is registered over isd:ied x jsd:jed+1 (so
// u(i,jec+1,k) is in-bounds); v is registered over isd:ied+1 x jsd:jed (so
// v(iec+1,j,k) is in-bounds). This file reuses RiemViewC3/riem_c_view3
// (fv3_riem_solver_c.cuh) for every FieldRegistry-resident field, exactly
// as fv3_remap_pressurevars.cu does -- the view is generic over index
// order via its own registered stride, so .at(i,k,j) vs .at(i,j,k) is
// simply which positional argument carries which stride, not a different
// view type.
//
// STATE%VARS%U/V/W/PT/PKZ/PE/DZ have NO FieldRegistry identity (they are
// MAPL State fields, not FV_Atm fields) -- same situation as REMAP_Q's q
// and REMAP_PRESSUREVARS's q(sphum)/cappa. Each gets its own dedicated,
// persistent, grow-only, context-owned device buffer (fv3_gpu_context.hpp)
// sized to the plain interior compute domain is:ie x js:je (STATE%VARS%*
// arrays carry no halo at all, unlike q/cappa's isd:ied x jsd:jed sizing),
// populated by ONE contiguous H2D copy per field before the relevant
// kernel launches (Stage 3/4: no per-level/per-column marshaling loop).
// `StateInputView` below is the accessor for these buffers, analogous to
// PressureVarsPlaneView but with `is,js` lower bounds (no ghost offset)
// since these buffers carry no halo.
//
// ps has no FieldRegistry identity either, and (unlike the inputs above)
// no downstream CUDA WP-A/B/C/D kernel reads it on-device (grep-confirmed,
// Prompt 99 Stage 2) -- it is a pure Fortran/host-consumed derived
// quantity. Rather than give it a full FieldRegistry entry (a duplicate
// storage mechanism the reuse mandate forbids), Phase C packs it into its
// own small persistent is:ie x js:je scratch buffer and the caller
// (fv3_gpu_state_to_fv_phase_c, fv3_gpu_runtime.cpp) D2H's that buffer
// directly into the caller's host `ps` pointer -- ONE contiguous D2H, same
// convention as REMAP_PRESSUREVARS's cappa D2H.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_gpu_types.hpp"
#include "fv3_gpu_internal.hpp"
#include "fv3_riem_solver_c.cuh"  // RiemViewC3 + riem_c_view3
#include <cstdint>

namespace fv3::gpu::vertical {

#if defined(__CUDACC__)
#define FV3_S2FV_HD __host__ __device__
#else
#define FV3_S2FV_HD
#endif

// Accessor for the STATE%VARS%* input buffers: plain contiguous is:ie x
// js:je (x km or km+1), i fastest, then j, then k -- no ghost/halo offset,
// no stagger. Matches STATE%VARS%U(isc:iec,jsc:jec,km)'s own declared
// Fortran layout exactly, so the caller's single H2D is a straight
// cudaMemcpy of the real host array with no repacking.
struct StateInputView {
  fv_real *p = nullptr;
  int64_t is = 0, js = 0, ni = 0, nj = 0;
  FV3_S2FV_HD inline fv_real at(int64_t i, int64_t j, int k) const {
    return p[(i - is) + (j - js) * ni + static_cast<int64_t>(k) * ni * nj];
  }
};

// Output accessor for the packed ps scratch buffer: plain contiguous
// is:ie x js:je, i fastest -- matches the caller's host `ps(isc:iec,
// jsc:jec)` array layout exactly, so the caller's D2H is a straight
// cudaMemcpy with no repacking.
struct PsPackView {
  fv_real *p = nullptr;
  int64_t is = 0, js = 0, ni = 0;
  FV3_S2FV_HD inline fv_real &at(int64_t i, int64_t j) const {
    return p[(i - is) + (j - js) * ni];
  }
};

inline uint64_t state_to_fv_columns(int64_t is, int64_t ie, int64_t js,
                                    int64_t je) noexcept {
  const int64_t nx = ie - is + 1;
  const int64_t ny = je - js + 1;
  if (nx <= 0 || ny <= 0) return 0;
  return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny);
}

// Element count of a plain is:ie x js:je x km STATE%VARS%* buffer (km may
// be the Fortran km or km+1, per field -- see call sites in
// fv3_gpu_runtime.cpp's fv3_gpu_state_to_fv_phase_a/_c).
inline uint64_t state_to_fv_plane_elements(int64_t is, int64_t ie,
                                           int64_t js, int64_t je,
                                           int32_t km) noexcept {
  const uint64_t cols = state_to_fv_columns(is, ie, js, je);
  if (cols == 0 || km <= 0) return 0;
  return cols * static_cast<uint64_t>(km);
}

//---------------------------------------------------------------------------
// Phase A: u/v tiny_number fill + interior copy. No per-thread scratch, no
// vertical dependency chain -- pure elementwise, one thread per (i,j,k).
//---------------------------------------------------------------------------
struct StateToFvPhaseAArgs {
  RiemViewC3 u;  // u <- FieldId::u (isd:ied x jsd:jed+1 x 0:npz)
  RiemViewC3 v;  // v <- FieldId::v (isd:ied+1 x jsd:jed x 0:npz)

  StateInputView state_u;  // STATE%VARS%U, is:ie x js:je x km
  StateInputView state_v;  // STATE%VARS%V, is:ie x js:je x km

  int64_t is = 0, ie = 0, js = 0, je = 0;  // interior compute domain
  int32_t km = 0;                          // Fortran km (= npz)
};

// Enqueue-only. Does NOT perform the tiny_number full-buffer fills -- the
// caller (fv3_gpu_state_to_fv_phase_a, fv3_gpu_runtime.cpp) issues those via
// the existing fv3_fill_device_constant utility (fv3_gpu_state_poison.hpp,
// the same mechanism dyn_core.F90:330's init_ijk_mem poison call already
// uses) immediately before this launch, on the same stream.
fv3_gpu_status launch_state_to_fv_phase_a(const StateToFvPhaseAArgs &a,
                                          void *stream,
                                          fv3_gpu_error *error = nullptr);

int launch_state_to_fv_phase_a_ctx(const ExecutionContext &,
                                   fv3_gpu_error *) noexcept;

//---------------------------------------------------------------------------
// Phase C: w copy, pe transpose, peln/pk, ps pack, delp, pt, delz, pkz.
// One thread per (i,j) column, looping over k -- pe must be fully written
// for a column before peln/pk/delp read it back, and one thread performs
// the whole column sequentially, so there is no race (same pattern as
// fv3_remap_pressurevars.cu's Pass 1/Pass 2).
//---------------------------------------------------------------------------
struct StateToFvPhaseCArgs {
  RiemViewC3 w;     // w    <- FieldId::w    (km levels)
  RiemViewC3 pe;    // pe   <- FieldId::pe   (km+1 levels; (i,k,j))
  RiemViewC3 peln;  // peln <- FieldId::peln (km+1 levels; (i,k,j))
  RiemViewC3 pk;    // pk   <- FieldId::pk   (km+1 levels; (i,j,k))
  RiemViewC3 delp;  // delp <- FieldId::delp (km levels)
  RiemViewC3 pt;    // pt   <- FieldId::pt   (km levels)
  RiemViewC3 delz;  // delz <- FieldId::delz (km levels)
  RiemViewC3 pkz;   // pkz  <- FieldId::pkz  (km levels)

  StateInputView state_w;    // STATE%VARS%W,   is:ie x js:je x km
  StateInputView state_pe;   // STATE%VARS%PE,  is:ie x js:je x (km+1), (i,j,k)
  StateInputView state_pt;   // STATE%VARS%PT,  is:ie x js:je x km
  StateInputView state_pkz;  // STATE%VARS%PKZ, is:ie x js:je x km
  StateInputView state_dz;   // STATE%VARS%DZ,  is:ie x js:je x km

  PsPackView ps_out;  // packed is:ie x js:je scratch, D2H'd by the caller

  int64_t is = 0, ie = 0, js = 0, je = 0;  // interior compute domain
  int32_t km = 0;                          // Fortran km (= npz)

  fv_real akap = fv_real(0);
};

// Enqueue-only. Does NOT perform the pe/delp/pt tiny_number full-buffer
// fills -- the caller (fv3_gpu_state_to_fv_phase_c, fv3_gpu_runtime.cpp)
// issues those via fv3_fill_device_constant immediately before this
// launch, on the same stream, in the same order the Fortran source writes
// them (pe, then delp, then pt) so that halo/skipped cells end up holding
// tiny_number exactly as the authoritative routine leaves them.
fv3_gpu_status launch_state_to_fv_phase_c(const StateToFvPhaseCArgs &a,
                                          void *stream,
                                          fv3_gpu_error *error = nullptr);

int launch_state_to_fv_phase_c_ctx(const ExecutionContext &,
                                   fv3_gpu_error *) noexcept;

}  // namespace fv3::gpu::vertical
