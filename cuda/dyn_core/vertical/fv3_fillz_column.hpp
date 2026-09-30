#pragma once
//===========================================================================
// WP-C: fv_fill_mod's `fillz` (fv_fill.F90:51-156, non-DEV_GFS_PHYS branch --
// DEV_GFS_PHYS is not defined anywhere in this project's build, confirmed by
// grep of CMakeLists.txt/CMakeCache.txt). Mass-conservative fix-up of
// negative tracer values within a single column, applied by mapn_tracer
// (fv_mapz.F90:2291, `if (fill) call fillz(...)`) immediately after the core
// PPM/monotonic remap. Mapped/host-validated by Prompt 88A
// (docs/performance/p88a_remap_q_fill_semantics_2026-09-25.md); implemented
// here by Prompt 88B as a standalone, dependency-free, __host__ __device__
// helper -- deliberately NOT folded into fv3_map_scalar_column.hpp, so that
// file's core scalar_profile/map_scalar logic is untouched -- so it can be
// (a) called from the REMAP_Q CUDA kernel's per-column thread body
// (fv3_remap_q.cu) immediately before its existing write-back, with zero new
// global-memory traffic (operates on the kernel's already-resident q2_out/
// dpeO scratch, reusing one existing free scratch slot for dm_scratch), and
// (b) unit-tested on the host, bit-for-bit, against the mechanical port in
// cuda/dyn_core/tests/vertical/test_fillz_host_reference.cpp's fillz_host().
//
// Preserves the authoritative operation order exactly (fv_fill.F90:86-152):
// top-layer borrow, then interior borrow-above-then-below (a single zfix
// flag suffices here -- this is a single-column, single-tracer, per-thread
// routine, so Fortran's zfix(im) collapses to one bool), then bottom-layer
// borrow, then a zfix-gated, sum0>0-gated non-local rescale-and-clip over
// Fortran k=2..km (0-indexed here: index 1..km-1). NOT algebraically
// simplified relative to the Fortran -- e.g. the interior loop still tests
// q[k]<0 twice (once to enter the branch, once again after the "borrow from
// above" sub-block, exactly as fv_fill.F90:98/:106 does), and the two
// borrow amounts are computed as separate min() expressions, not merged.
//
// NOT an unconditional non-negativity guarantee -- see fv_fill.F90:140
// (`if (sum0 > 0.)`) and test_fillz_host_reference.cpp's
// sum0_negative_fix_inert_by_design case: when a column's aggregate mass
// over k=2..km is net non-positive, fillz leaves a residual negative in
// place (mass is still exactly conserved). Do not "improve" this here.
//
// q, dp are km-length, 0-indexed (Fortran's q(i,:,ic)/dp(i,:) for a single i,
// single tracer ic -- fillz has no cross-tracer coupling, confirmed Prompt
// 88A Stage 3/6, so a single-tracer, single-column signature is a faithful
// unit for mapn_tracer's own do-ic=1,nq wrapping loop). dm_scratch must
// have at least km elements of caller-supplied scratch (only indices
// 1..km-1 are ever written) -- this function performs no allocation of its
// own.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#if defined(__CUDACC__)
#define FV3_FILLZ_HD __host__ __device__
#else
#define FV3_FILLZ_HD
#endif

namespace fv3::gpu::vertical {

template <typename T>
FV3_FILLZ_HD inline void fillz_column(T *q, const T *dp, int km,
                                       T *dm_scratch) noexcept {
  // Top layer (fv_fill.F90:87-92; Fortran k=1 -> index 0)
  if (q[0] < T(0)) {
    q[1] = q[1] + q[0] * dp[0] / dp[1];
    q[0] = T(0);
  }

  // Interior (fv_fill.F90:96-114; Fortran k=2..km-1 -> index 1..km-2)
  bool zfix = false;
  for (int k = 1; k <= km - 2; ++k) {
    if (q[k] < T(0)) {
      zfix = true;
      if (q[k - 1] > T(0)) {
        // Borrow from above
        const T dq = (q[k - 1] * dp[k - 1] < -q[k] * dp[k])
                         ? q[k - 1] * dp[k - 1]
                         : -q[k] * dp[k];
        q[k - 1] = q[k - 1] - dq / dp[k - 1];
        q[k] = q[k] + dq / dp[k];
      }
      if (q[k] < T(0) && q[k + 1] > T(0)) {
        // Borrow from below
        const T dq = (q[k + 1] * dp[k + 1] < -q[k] * dp[k])
                         ? q[k + 1] * dp[k + 1]
                         : -q[k] * dp[k];
        q[k + 1] = q[k + 1] - dq / dp[k + 1];
        q[k] = q[k] + dq / dp[k];
      }
    }
  }

  // Bottom layer (fv_fill.F90:116-128; Fortran k=km -> index km-1)
  {
    const int k = km - 1;
    if (q[k] < T(0) && q[k - 1] > T(0)) {
      zfix = true;
      const T qup = q[k - 1] * dp[k - 1];
      const T qly = -q[k] * dp[k];
      const T dup = (qly < qup) ? qly : qup;
      q[k - 1] = q[k - 1] - dup / dp[k - 1];
      q[k] = q[k] + dup / dp[k];
    }
  }

  // Final check and non-local fix if needed (fv_fill.F90:130-152; Fortran
  // k=2..km -> index 1..km-1)
  if (zfix) {
    T sum0 = T(0);
    for (int k = 1; k <= km - 1; ++k) {
      dm_scratch[k] = q[k] * dp[k];
      sum0 = sum0 + dm_scratch[k];
    }
    if (sum0 > T(0)) {
      T sum1 = T(0);
      for (int k = 1; k <= km - 1; ++k) {
        const T d = dm_scratch[k];
        sum1 = sum1 + (d > T(0) ? d : T(0));
      }
      const T fac = sum0 / sum1;
      for (int k = 1; k <= km - 1; ++k) {
        const T v = fac * dm_scratch[k] / dp[k];
        q[k] = (v > T(0)) ? v : T(0);
      }
    }
  }
}

}  // namespace fv3::gpu::vertical

#undef FV3_FILLZ_HD
