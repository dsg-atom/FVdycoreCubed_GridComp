#pragma once

#include "fv3_gpu_types.hpp"

#include <cstddef>
#include <cstdint>

namespace fv3::gpu {

/* dyn_core.F90:216-218. The authoritative poison constant that init_ijk_mem
 * writes into the transient work arrays at every init_step:
 *
 *     #ifdef OVERLOAD_R4
 *         real, parameter:: huge_r = 1.E8
 *     #else
 *         real, parameter:: huge_r = 1.E40
 *     #endif
 *
 * The precision branch is not cosmetic: 1.E40 is not representable as a float,
 * so an R4 build that used the R8 value would poison with +Inf and turn every
 * arithmetic touch into a NaN instead of a large number. */
inline constexpr fv_real fv3_huge_r =
    static_cast<fv_real>(sizeof(fv_real) == 4 ? 1.0e8 : 1.0e40);

/* Fills bytes/sizeof(fv_real) elements of a device allocation with `value`.
 *
 * Exists because cudaMemsetAsync fills BYTES: it can express 0, which is what
 * zero_accumulators needs, but it cannot express huge_r.
 *
 * Returns a cudaError_t widened to int (0 == cudaSuccess) so this header stays
 * free of CUDA headers and remains includable from a host-only translation
 * unit; the caller converts it with cuda_fail. */
int fv3_fill_device_constant(void *device, std::size_t bytes, fv_real value,
                             void *stream) noexcept;

/* Fills only the four DIAGONAL corner ghost blocks of a rank-3 (i,j,k) field --
 * the cells with i outside [ci_lo,ci_hi] AND j outside [cj_lo,cj_hi] -- leaving
 * the owned box and the four edge halos untouched.
 *
 * This is narrower than fv3_fill_device_constant on purpose. The authoritative
 * init_ijk_mem does cover the whole declared extent, but by the time any stage
 * reads gz the Fortran has redefined the owned box (dyn_core.F90:444-454) and
 * the four edge halos (the :459 exchange), so huge_r SURVIVES only in the
 * diagonal corners. Filling the whole allocation reproduces the Fortran's
 * intermediate state but not its observable one, and it destroys edge-halo
 * values in any context where the caller -- not an exchange -- owns them.
 * Measured: the flat fill made fv3_dispatch_smoke fail in both R8 and R4,
 * because its Phase A supplies gz's halo itself and nothing restores it.
 *
 * Corners are the cells whose accumulating ×grav product was the actual defect,
 * so narrowing loses none of the fix. lower/extent/stride are the descriptor's
 * own, so padding beyond the Fortran's isd:ied is included (nothing defines it
 * either, and a sentinel there is preferable to allocate-garbage).
 *
 * Returns a cudaError_t widened to int (0 == cudaSuccess). */
int fv3_fill_device_corner_ghosts(void *device, const std::int64_t lower[3],
                                  const std::uint64_t extent[3],
                                  const std::int64_t stride[3],
                                  std::int64_t ci_lo, std::int64_t ci_hi,
                                  std::int64_t cj_lo, std::int64_t cj_hi,
                                  fv_real value, void *stream) noexcept;

} // namespace fv3::gpu
