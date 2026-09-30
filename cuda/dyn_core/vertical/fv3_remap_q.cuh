#pragma once
//===========================================================================
// WP-C: fv_mapz_mod vertical remap -- production tracer entry (REMAP_Q,
// mapn_tracer with nq=1), Prompt 85.
//
// Authoritative Fortran source of truth (numerical source of truth,
// read-only reference):
//   GFDL_atmos_cubed_sphere (tag geos/v3.0.0, ca30e26)
//   model/fv_mapz.F90:817-847   Lagrangian_to_Eulerian, timer 'Remap_Q':
//                                pe1/pe2/dp2 derivation (identical formula to
//                                fv3_pt_delz_remap.cu's DELZ-branch peO/dpeO
//                                construction, confirmed Prompt 85 Stage 2/5),
//                                then the mapn_tracer(...) call.
//   model/fv_mapz.F90:2081-2192 mapn_tracer: dp1, then per-tracer
//                                scalar_profile(iv=0, kord(iq)) + the
//                                "Mapping" search/integrate section, which is
//                                -- for nq=1 -- byte-identical to
//                                map_scalar_column's existing
//                                gmao_bot=gmao_top=false path (already
//                                validated for pt/delz/u/v). fillz is
//                                conditionally applied afterward but is
//                                UNREACHABLE in production: `fill` is always
//                                .false. (fv_arrays.F90:820 compiled default,
//                                confirmed not overridden by any checked
//                                production namelist or by the
//                                map_scalar-driver fixture generator's own 4
//                                mapn_tracer call sites -- Prompt 85 Stage 5
//                                completeness check). Not ported here.
//
// Scope implemented (matches the validated production first-run config):
// nq=2 [sphum kord=9, cld_amt kord=9 -- fv_dynamics.F90:941-944 combined
// with FV_StateMod.F90:563's unconditional flagstruct%kord_tr=9 override;
// see docs/performance/p87_remap_q_actual_kord9_production_2026-09-25.md
// Stage 0 for the full source chain], iv=0 (mapn_tracer hardcodes iv=0),
// q_min=0.0 (the literal mapn_tracer's own call site passes). Prompt 85
// implemented nq=1/kord=8 only; Prompt 86B extended this to production's
// assumed nq=2/(8,9) schedule via two sequential single-tracer launches
// sharing one H2D/D2H and one workspace slice (RemapQArgs::kord,
// launch_remap_q_ctx); Prompt 87 corrected the kord schedule itself to the
// actual (9,9) runtime value -- same launch mechanism, no new numerics.
// Arbitrary nq (a genuinely different fixture/loop shape -- case23 in the
// Prompt 84B fixture set) remains OUT OF SCOPE.
//
// fill=.true. -- FV_StateMod.F90:568 unconditionally forces
// flagstruct%fill=.true. for every GEOS FV_State (confirmed universal
// across every runnable config in this project, zero fill=.false. found;
// see docs/performance/p88a_remap_q_fill_semantics_2026-09-25.md). Prompt
// 88B implements this: RemapQArgs::fill, when true, applies fv_fill_mod's
// `fillz` (see fv3_fillz_column.hpp) to each tracer's remapped column
// immediately before write-back, exactly mirroring mapn_tracer's own
// `if (fill) call fillz(...)` placement (fv_mapz.F90:2291) after its core
// PPM/monotonic remap. fillz operates entirely on already-resident
// per-thread scratch (q2_out, dpeO) and reuses one existing dead scratch
// slot (c2) for its own dm() local array -- no new global-memory traffic,
// no new scratch slot, no new H2D/D2H transfer (remap_q_scratch_slots
// stays 11). fill is applied identically to both tracers in a given
// invocation (flagstruct%fill is one global value, not per-tracer --
// confirmed by fillz(im,km,nq,q,dp)'s own signature, which applies
// identical per-column logic across the whole nq-tracer block with no
// tracer-specific branching).
//
// fill (like kord) is now also checked on the C++ side via
// DispatchParams::remap_q_fill (see fv3_gpu_internal.hpp) -- both fill
// states (0 and 1) are valid/accepted; there is no rejection based on the
// fill value itself, since fill=false must remain a working, regression-
// tested no-op path. nq/kord continue to be checked as before
// (launch_remap_q_ctx requires DispatchParams::remap_q_nq==2/kord1==9/
// kord2==9; launch_remap_q requires RemapQArgs::kord in {8,9}) as
// defense-in-depth against a stale Fortran-side selector -- Prompt 85's
// original selector bug was exactly this class of error.
//
// Unlike pt_delz_remap/remap_uv, the tracer array q has NO FieldRegistry
// identity (FieldId::q exists as an enum value but is never populated or
// looked up anywhere in the CUDA runtime -- confirmed by grep, Prompt 85
// Stage 2). It is therefore NOT resident on the GPU by any existing
// mechanism. This file's Args carries a dedicated device buffer pointer
// (`q`) for it, populated by fv3_gpu_remap_q's own persistent, module-
// owned, grow-only buffer (fv3_remap_q.cu) via ONE contiguous H2D copy
// before the kernel launches and ONE contiguous D2H copy after -- not a
// per-column or per-field marshaling loop (the exact defect Prompt 82
// root-caused in the unrelated TRACER_2D CUDA path). The buffer mirrors
// Fortran's own q(isd:ied,jsd:jed,km,nq) memory layout exactly (i fastest,
// then j, then k, then nq slowest/outermost), which is why a single
// cudaMemcpy still suffices for nq=2 (Prompt 86B): the whole 2-tracer block
// is one contiguous region, addressed here as two same-size planes
// (remap_q_buffer_elements(isd,ied,jsd,jed,km) apart) rather than as two
// separately-transferred arrays (Prompt 86B Stage 2/3). Transfer
// architecture unchanged by Prompt 87's kord correction.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_gpu_types.hpp"
#include "fv3_gpu_internal.hpp"
#include "fv3_pt_delz_remap.cuh"  // reuse PtDelzView1/ptdz_cview1 for ak/bk
#include "fv3_riem_solver_c.cuh"  // reuse RiemViewC3/riem_c_view3 for pe
#include <cstdint>

namespace fv3::gpu::vertical {

#if defined(__CUDACC__)
#define FV3_RMQ_HD __host__ __device__
#else
#define FV3_RMQ_HD
#endif

// Per-thread local-scratch level cap. Same precedent/rationale as
// pt_delz_remap_max_local_levels/remap_uv_max_local_levels.
inline constexpr int remap_q_max_local_levels = 128;

// Per-column scratch slot count (uniform (km+1)-scalar pitch, same
// convention as pt_delz_remap_scratch_slots/remap_uv_scratch_slots):
//   pe_old, peO, dpe_old, dpeO, q1_in, q2_out, prof_q, c2, c3, c4, gam  (11)
inline constexpr int remap_q_scratch_slots = 11;

FV3_RMQ_HD inline uint64_t remap_q_scalars_per_column(int km) noexcept {
  return static_cast<uint64_t>(remap_q_scratch_slots) *
         static_cast<uint64_t>(km + 1);
}

// Column count actually launched: the PLAIN interior domain is..ie x js..je
// (matches fv_mapz.F90:817's `do 1001 j=js,je` / the mapn_tracer i1=is,i2=ie
// call arguments -- NOT REMAP_UV's widened union box).
inline uint64_t remap_q_columns(int64_t is, int64_t ie, int64_t js,
                                int64_t je) noexcept {
  const int64_t nx = ie - is + 1;
  const int64_t ny = je - js + 1;
  if (nx <= 0 || ny <= 0) return 0;
  return static_cast<uint64_t>(nx) * static_cast<uint64_t>(ny);
}

inline uint64_t remap_q_workspace_bytes(int km, int64_t is, int64_t ie,
                                        int64_t js, int64_t je) noexcept {
  return remap_q_columns(is, ie, js, je) * remap_q_scalars_per_column(km) *
         sizeof(fv_real);
}

// Element count of the persistent q device buffer: the FULL
// isd:ied x jsd:jed x km memory domain (not just the interior compute
// domain), because that whole block is what is contiguous in Fortran's
// q(isd:ied,jsd:jed,km,1) layout and therefore what a single cudaMemcpy
// moves. The kernel only reads/writes the interior is:ie x js:je subset
// of this buffer; the halo cells it also carries are moved but never
// touched by any thread.
inline uint64_t remap_q_buffer_elements(int64_t isd, int64_t ied,
                                        int64_t jsd, int64_t jed,
                                        int32_t km) noexcept {
  const int64_t nx_d = ied - isd + 1;
  const int64_t ny_d = jed - jsd + 1;
  if (nx_d <= 0 || ny_d <= 0 || km <= 0) return 0;
  return static_cast<uint64_t>(nx_d) * static_cast<uint64_t>(ny_d) *
         static_cast<uint64_t>(km);
}

struct RemapQArgs {
  // ---- intent(in), vertical-only, shared by every column ------------------
  PtDelzView1 ak;  // ak <- FieldId::ak (km+1, vertical-only)
  PtDelzView1 bk;  // bk <- FieldId::bk (km+1, vertical-only)

  // ---- intent(in) ----------------------------------------------------------
  RiemViewC3 pe;  // pe <- FieldId::pe (km+1; fully updated by PT/DELZ's own
                  // closing update, which this kernel must be enqueued after
                  // on the same stream -- same ordering contract as
                  // REMAP_UV's own pe read).

  // ---- intent(inout), NOT a FieldRegistry view (see header comment) -------
  // Device buffer of remap_q_buffer_elements(isd,ied,jsd,jed,km) elements,
  // laid out i-fastest/j/k-slowest (Fortran's own q(isd:ied,jsd:jed,km,1)
  // memory order). Read as q1 and overwritten in place with q2 for every
  // interior column -- mirrors mapn_tracer's own q1(i,j,k,iq)=q2(i,k,iq)
  // closing loop, which aliases the same actual Fortran array on entry/exit
  // even though mapn_tracer's internal q2 is a separate local temporary
  // (this kernel likewise uses a small per-thread q2 scratch slot, never
  // aliasing its read of q1 with its write-back).
  fv_real *q = nullptr;

  int64_t is = 0, ie = 0, js = 0, je = 0;      // interior compute domain
  int64_t isd = 0, ied = 0, jsd = 0, jed = 0;  // q buffer's full memory domain
  int32_t km = 0;  // Fortran km (= npz)

  // Prompt 86B Stage 1. scalar_profile_column's kord, previously a kernel-
  // literal 8 (nq=1's only production value at the time, kord_tr(1)==8).
  // Default preserved at 8 so every existing kord=8-only call site
  // (Prompt 85's nq=1 fixtures/smoke test, retained as non-production
  // regression coverage, none of which set this field) is bitwise
  // unaffected. Production nq=2 requires two launches of this same kernel
  // per invocation, one per tracer -- Prompt 87 corrected the actual
  // schedule to kord=9 (sphum) then kord=9 (cld_amt), both tracers, per
  // FV_StateMod.F90:563's unconditional flagstruct%kord_tr=9 override (the
  // originally assumed kord=8-then-9 split never occurs in a
  // GEOS-integrated runtime) -- see launch_remap_q_ctx. Only 8/9 are
  // valid; anything else is rejected by launch_remap_q.
  int32_t kord = 8;

  // Prompt 88B. mapn_tracer's flagstruct%fill, applied identically to every
  // tracer in the invocation (see header comment). Default false preserves
  // bitwise behavior of every existing fill-unaware call site (Prompt 85's
  // nq=1 fixtures, Prompt 86B/87's nq=2 fill=false smoke/regression tests).
  bool fill = false;

  fv_real ptop = fv_real(0);

  // ---- optional device workspace (WorkspaceId::vertical), same pattern as
  // PtDelzRemapArgs/RemapUvArgs (Prompt 68). workspace == nullptr ->
  // per-thread local arrays, km <= 128 enforced. workspace != nullptr ->
  // device scratch of at least remap_q_workspace_bytes(km, is, ie, js, je),
  // addressed with workspace_scalars_per_column stride per column (>=
  // scalars_per_column).
  fv_real *workspace = nullptr;
  uint64_t workspace_scalars_per_column = 0;
  uint64_t workspace_bytes = 0;
};

// Enqueue-only. `stream` is the WP-E-owned compute stream (cudaStream_t).
// No host/device transfer and no device synchronization is performed --
// a.q is assumed already populated on entry (the caller, fv3_gpu_remap_q
// in fv3_gpu_runtime.cpp, is responsible for the H2D/D2H around this call).
fv3_gpu_status launch_remap_q(const RemapQArgs &a, void *stream,
                              fv3_gpu_error *error = nullptr);

// WP-E ExecutionContext entry, LaunchFn-compatible. Reads pe/ak/bk from
// x.fields (same as launch_remap_uv_ctx); reads the q device buffer pointer
// and its isd/ied/jsd/jed memory-domain extent from
// x.dispatch_params->remap_q_buffer (see fv3_gpu_internal.hpp's
// DispatchParams::remap_q_buffer comment) rather than from x.fields, since
// q has no FieldRegistry identity.
int launch_remap_q_ctx(const ExecutionContext &, fv3_gpu_error *) noexcept;

}  // namespace fv3::gpu::vertical
