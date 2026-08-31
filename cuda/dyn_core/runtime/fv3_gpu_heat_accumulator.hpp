#pragma once

#include "fv3_gpu_types.hpp"

#include <cstdint>

namespace fv3::gpu {

/* Prompt 6B -- the per-substep heat_source accumulation of dyn_core.F90:853-861.
 *
 * WHY THIS EXISTS AT ALL, i.e. why the CUDA port cannot just let d_sw accumulate
 * in place.
 *
 * The authoritative code uses TWO arrays where the CUDA port had one:
 *
 *   heat_source(isd:ied, jsd:jed, npz)   dyn_core.F90:245, allocated and zeroed
 *                                        ONCE per dyn_core call at :398-399
 *   heat_s(is:ie, js:je)                 sw_core.F90:546, d_sw's per-level
 *                                        intent(OUT) scratch
 *
 * d_sw zeroes heat_s (sw_core.F90:965-966), builds it up across its own internal
 * stages (:979-981 and onward) and finalizes it as
 * `heat_source = delp*(heat_source - damp*rsin2*(...))` at :1508/:1513. Because
 * that finalization multiplies through, d_sw's output is NOT an increment that
 * can be added to whatever the array already held -- d_sw REQUIRES its scratch to
 * start at zero. dyn_core then does the accumulating itself, once per substep:
 *
 *     if ( flagstruct%d_con > 1.0E-5 .OR. flagstruct%do_skeb ) then
 *        do j=js,je
 *           do i=is,ie
 *              heat_source(i,j,k) = heat_source(i,j,k) + heat_s(i,j)
 *
 * and reads the result only AFTER the acoustic loop: del2_cubed at :1311, then
 * the pt update at :1324/:1328/:1350.
 *
 * The CUDA port uses the registered 3-D `heat_source` field as d_sw's scratch --
 * dsw_heat_reset_kernel (dgrid/wp_b_dsw_cuda.cu:444-449) zeroes it every substep
 * -- so at loop exit it held only the LAST substep's contribution instead of the
 * sum. That is a genuine authoritative-fidelity gap, inert at every in-loop
 * savepoint (nothing inside the loop reads heat_source) and observable only in
 * the dyn_core-exit `pt` and `pkz`.
 *
 * The repair keeps d_sw's role exactly as it is and adds the missing dyn_core-
 * level accumulator as persistent device state owned by the runtime, matching the
 * Fortran's own two-array structure rather than inventing a third scheme.
 *
 * Numerical algorithm changed: NO -- the summation the authoritative code
 * performs is now performed; no kernel, stencil or expression is altered.
 *
 * ---------------------------------------------------------------------------
 * `dst` and `src` must share one descriptor geometry: the accumulator is
 * allocated with heat_source's own element count and is addressed with
 * heat_source's own lowers and strides, so one offset serves both.
 *
 * The six range arguments are INCLUSIVE bounds in the FIELD'S OWN INDEX SPACE,
 * i.e. the space `lo_*` is expressed in -- not Fortran's 1-based level number.
 * They are passed explicitly, rather than reconstructed here from a level count,
 * because the registered heat_source view on this path has lower[2] == 0 while
 * the authoritative loop is `do k=1,npz`: deriving the k range from npz alone
 * silently addresses one full level past the allocation, whose only symptom is a
 * corrupted neighbouring field. The runtime therefore states the range and this
 * routine does not guess it.
 *
 * The caller is expected to pass the authoritative loop nest -- the owned box
 * is..ie / js..je and all npz levels. The halo is deliberately untouched,
 * because del2_cubed refreshes it through FV3_GPU_H_HEAT afterwards.
 *
 * Returns a cudaError_t widened to int (0 == cudaSuccess), so this header stays
 * free of CUDA headers and remains includable from a host-only translation unit.
 */
int fv3_heat_source_accumulate(fv_real *dst, const fv_real *src,
                               std::int64_t lo_i, std::int64_t lo_j, std::int64_t lo_k,
                               std::int64_t st_i, std::int64_t st_j, std::int64_t st_k,
                               std::int64_t i0, std::int64_t i1,
                               std::int64_t j0, std::int64_t j1,
                               std::int64_t k0, std::int64_t k1,
                               void *stream) noexcept;

} // namespace fv3::gpu
