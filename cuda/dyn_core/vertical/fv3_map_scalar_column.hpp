#pragma once
//===========================================================================
// WP-C: fv_mapz_mod vertical remap -- map_scalar / scalar_profile /
// cs_limiters, one-column primitives (Prompt 52B).
//
// Authoritative Fortran source of truth (numerical source of truth,
// read-only reference):
//   GFDL_atmos_cubed_sphere (tag geos/v3.0.0, ca30e26)
//   model/fv_mapz.F90:1519-1671   subroutine map_scalar
//   model/fv_mapz.F90:1881-2288   subroutine scalar_profile
//   model/fv_mapz.F90:2699-2774   subroutine cs_limiters
// All ":NNN" comments below are fv_mapz.F90 line numbers. One CUDA/host
// thread per horizontal column; the Fortran i-loops collapse to "this
// column" and the k-loops stay serial, exactly as edge_profile_column and
// friends already do in fv3_vertical_column.hpp (same file's Column<T>,
// fv3_fabs/fv3_max are reused here, not redefined).
//
// Scope implemented (matches the Prompt 52A fixtures and the production
// Held-Suarez C512 config kord_mt=kord_wz=kord_tm=9):
//   * kord == 9 branch of scalar_profile only (:2095-2127). Other abs(kord)
//     branches (7,10,12,13,14,15,16, and the <16 catchall) are NOT ported.
//   * iv in {0,1,-1} (map_scalar's "temp"/"other", "constituent", and
//     "winds" modes). iv=-1's top/bottom sign-consistency clamps
//     (fv_mapz.F90:2357-2364 top, :2581-2588 bottom -- REMAP_UV, Prompt 71)
//     are ported exactly, mirroring the existing iv==0 branch shape. iv=2
//     (cs remap-temp) is NOT ported.
//   * the "else" (iv/=-2) tridiagonal solve for q(:) only (:1926-1954).
//     iv==-2 (qs-bounded tridiagonal, :1901-1925) is NOT ported -- the
//     fixture/production driver never passes qs.
//   * abs(kord)>9 is false whenever kord==9 exactly, so ext5/ext6
//     (:2029-2037) are dead code in this scope and are not computed.
//   * map_scalar's "standard FV3 remapping" branch (:1770-1811 in the
//     current authoritative source) PLUS the 5 GMAO top/bottom
//     boundary-condition branches (:1744-1767), gated on the new trailing
//     gmao_bot/gmao_top parameters (Prompt 61) -- ported exactly, one-for-
//     one, from map_scalar's if/else-if chain, including the LM1/LP0
//     search (:1733-1743) computed unconditionally every k, matching
//     Fortran. Both new parameters default to false, so every pre-Prompt-
//     61 call site (map_scalar-driver fixtures, the delz remap call site,
//     the GPU smoke kernel) is numerically unchanged. Only the production
//     pt remap call site (fv3_pt_delz_remap.cu) passes gmao_bot=gmao_top=
//     true, matching FV_StateMod.F90:558-559's unconditional GEOS
//     production default (optional_bot=gmao_bot_bc, optional_top=
//     gmao_top_bc at pt's own map_scalar call site, fv_mapz.F90:622-633).
//     See docs/performance/p61_map_scalar_gmao_boundary_conditions_
//     2026-09-23.md for the branch-by-branch derivation and validation.
//   * (Prompt 85) scalar_profile's abs(kord)<9 branch (:2487-2501) added as
//     an explicit `kord` parameter (default 9, so every pre-existing call
//     site is byte-for-byte unchanged) on scalar_profile_column and
//     map_scalar_column. Only kord==8 is exercised/validated (the
//     production REMAP_Q kord_tr value); no other abs(kord)<9 value has
//     fixture coverage. mapn_tracer's own nq-batched Mapping loop
//     (fv_mapz.F90:2081-2192) is NOT re-ported here -- for nq==1 (the only
//     validated REMAP_Q dispatch case) it is byte-for-byte the same
//     "standard FV3 remapping" computation as map_scalar_column's own
//     gmao_bot=false/gmao_top=false path, so REMAP_Q reuses
//     map_scalar_column directly (iv=0, kord=8) instead of duplicating it.
//     nq>1 (multi-tracer batching) is out of scope and not supported by
//     this reuse.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_vertical_column.hpp"

// fv3_vertical_column.hpp #undefs FV3_VERTICAL_HD at its own end (its own
// header-hygiene convention, not modified here) -- redefine the identical
// macro locally so it is available for this header's own declarations, and
// #undef it again at the end of this file to match that same convention.
#if defined(__CUDACC__)
#define FV3_VERTICAL_HD __host__ __device__
#else
#define FV3_VERTICAL_HD
#endif

namespace fv3::gpu::vertical {

// fv3_vertical_column.hpp defines fv3_max<T> but not fv3_min<T>; add the
// missing counterpart here rather than editing that shared file.
template <class T>
FV3_VERTICAL_HD inline T fv3_min(T x, T y) { return x < y ? x : y; }

// cs_limiters (fv_mapz.F90:2699-2774), specialized to a single column (im=1
// in the caller's own loop nest). `iv` here is cs_limiters' own mode
// argument (0, 1, or "else" == cs_limiters' iv not in {0,1}); it is NOT
// map_scalar's iv and every call site below passes it as a literal, exactly
// matching the Fortran call sites' hardcoded literals.
template <class T>
FV3_VERTICAL_HD inline void cs_limiters_column(int iv, bool extm, T a1, T &a2,
                                               T &a3, T &a4) {
  const T r12 = T(1) / T(12);
  if (iv == 0) {
    // :2708-2733 Positive definite constraint
    if (a1 <= T(0)) {
      a2 = a1; a3 = a1; a4 = T(0);
    } else if (fv3_fabs<T>(a3 - a2) < -a4) {
      if ((a1 + T(0.25) * (a3 - a2) * (a3 - a2) / a4 + a4 * r12) < T(0)) {
        if (a1 < a3 && a1 < a2) {
          a3 = a1; a2 = a1; a4 = T(0);
        } else if (a3 > a2) {
          a4 = T(3) * (a2 - a1);
          a3 = a2 - a4;
        } else {
          a4 = T(3) * (a3 - a1);
          a2 = a3 - a4;
        }
      }
    }
  } else if (iv == 1) {
    // :2734-2752 standard PPM monotonicity
    if ((a1 - a2) * (a1 - a3) >= T(0)) {
      a2 = a1; a3 = a1; a4 = T(0);
    } else {
      const T da1 = a3 - a2;
      const T da2 = da1 * da1;
      const T a6da = a4 * da1;
      if (a6da < -da2) {
        a4 = T(3) * (a2 - a1);
        a3 = a2 - a4;
      } else if (a6da > da2) {
        a4 = T(3) * (a3 - a1);
        a2 = a3 - a4;
      }
    }
  } else {
    // :2753-2773 "Standard PPM constraint" (extm-gated)
    if (extm) {
      a2 = a1; a3 = a1; a4 = T(0);
    } else {
      const T da1 = a3 - a2;
      const T da2 = da1 * da1;
      const T a6da = a4 * da1;
      if (a6da < -da2) {
        a4 = T(3) * (a2 - a1);
        a3 = a2 - a4;
      } else if (a6da > da2) {
        a4 = T(3) * (a3 - a1);
        a2 = a3 - a4;
      }
    }
  }
}

// scalar_profile (fv_mapz.F90:1881-2288), kord==9 / iv in {0,1} / qs absent
// only (see file header). `c1` is a4(1,:) (== q1, read-only throughout the
// authoritative routine); c2/c3/c4 are a4(2,:)/a4(3,:)/a4(4,:), written here.
// Caller-owned scratch, sized exactly to the Fortran locals' extents (no
// oversized cap, unlike the riem_solver_c local-array convention -- this
// header has no device __global__ of its own, so there is no fixed-size
// register-array requirement to satisfy yet; that is a launcher-layer
// decision for whoever wraps this in a kernel):
//   gam[0..km-1], q[0..km], extm[0..km-1].
template <class T>
FV3_VERTICAL_HD inline void scalar_profile_column(
    const Column<const T> c1, Column<T> c2, Column<T> c3, Column<T> c4,
    const Column<const T> delp, T *gam, T *q, bool *extm, int km, int iv,
    T qmin, int kord = 9) {
  // ---- tridiagonal solve for q(:), "else" branch only (:1926-1954) -------
  {
    const T grat = delp[1] / delp[0];
    const T bet = grat * (grat + T(0.5));
    q[0] = ((grat + grat) * (grat + T(1)) * c1[0] + c1[1]) / bet;
    gam[0] = (T(1) + grat * (grat + T(1.5))) / bet;
  }
  T d4 = T(0);
  for (int k = 1; k < km; ++k) {
    d4 = delp[k - 1] / delp[k];
    const T bet = T(2) + d4 + d4 - gam[k - 1];
    q[k] = (T(3) * (c1[k - 1] + d4 * c1[k]) - q[k - 1]) / bet;
    gam[k] = d4 / bet;
  }
  {
    const T a_bot = T(1) + d4 * (d4 + T(1.5));
    q[km] = (T(2) * d4 * (d4 + T(1)) * c1[km - 1] + c1[km - 2] -
             a_bot * q[km - 1]) /
            (d4 * (d4 + T(0.5)) - a_bot * gam[km - 1]);
  }
  for (int k = km - 1; k >= 0; --k) {
    q[k] = q[k] - gam[k] * q[k + 1];
  }

  // abs(kord) > 16 "perfectly linear scheme" early return (:1957-1966) is
  // dead code in this scope (kord == 9) and is not translated.

  // ---- Apply *large-scale* constraints (:1974-2010) ----------------------
  q[1] = fv3_min<T>(q[1], fv3_max<T>(c1[0], c1[1]));
  q[1] = fv3_max<T>(q[1], fv3_min<T>(c1[0], c1[1]));

  for (int k = 1; k < km; ++k) gam[k] = c1[k] - c1[k - 1];

  for (int k = 2; k <= km - 2; ++k) {
    if (gam[k - 1] * gam[k + 1] > T(0)) {
      q[k] = fv3_min<T>(q[k], fv3_max<T>(c1[k - 1], c1[k]));
      q[k] = fv3_max<T>(q[k], fv3_min<T>(c1[k - 1], c1[k]));
    } else if (gam[k - 1] > T(0)) {
      q[k] = fv3_max<T>(q[k], fv3_min<T>(c1[k - 1], c1[k]));
    } else {
      q[k] = fv3_min<T>(q[k], fv3_max<T>(c1[k - 1], c1[k]));
      if (iv == 0) q[k] = fv3_max<T>(T(0), q[k]);
    }
  }

  q[km - 1] = fv3_min<T>(q[km - 1], fv3_max<T>(c1[km - 2], c1[km - 1]));
  q[km - 1] = fv3_max<T>(q[km - 1], fv3_min<T>(c1[km - 2], c1[km - 1]));

  for (int k = 0; k < km; ++k) {
    c2[k] = q[k];
    c3[k] = q[k + 1];
  }

  // ---- extm (:2019-2038); ext5/ext6 skipped -- dead code at kord==9 ------
  for (int k = 0; k < km; ++k) {
    if (k == 0 || k == km - 1) {
      extm[k] = (c2[k] - c1[k]) * (c3[k] - c1[k]) > T(0);
    } else {
      extm[k] = gam[k] * gam[k + 1] < T(0);
    }
  }

  // ---- top-layer (k=1) mode adjustment (fv_mapz.F90:2357-2367) -----------
  // iv==2 branch is out of scope (not used by any ported CUDA path).
  if (iv == 0) {
    c2[0] = fv3_max<T>(T(0), c2[0]);
  } else if (iv == -1) {
    if (c2[0] * c1[0] <= T(0)) c2[0] = T(0);
  }
  c4[0] = T(3) * (T(2) * c1[0] - (c2[0] + c3[0]));
  cs_limiters_column<T>(1, extm[0], c1[0], c2[0], c3[0], c4[0]);

  // ---- k=2 (:2069-2073) ---------------------------------------------------
  c4[1] = T(3) * (T(2) * c1[1] - (c2[1] + c3[1]));
  cs_limiters_column<T>(2, extm[1], c1[1], c2[1], c3[1], c4[1]);

  // ---- Huynh's 2nd constraint for interior (:2486-2673) ------------------
  // kord==9 (default; unchanged from the pre-Prompt-85 port, :2503-2535) vs
  // abs(kord)<9 (Prompt 85, REMAP_Q kord==8 only, :2487-2501) are mutually
  // exclusive per fv_mapz.F90's own if/elseif chain -- no other kord value
  // (10,12,13,14,15,16,<16-catchall) is ported; a non-9 kord here always
  // means kord==8, the only abs(kord)<9 value the CUDA REMAP_Q dispatch
  // ever passes (see fv3_pt_delz_remap.cu / the REMAP_Q wrapper).
  if (kord == 9) {
    for (int k = 2; k <= km - 3; ++k) {
      if ((extm[k] && extm[k - 1]) || (extm[k] && extm[k + 1]) ||
          (extm[k] && c1[k] < qmin)) {
        c2[k] = c1[k]; c3[k] = c1[k]; c4[k] = T(0);
      } else {
        c4[k] = T(3) * (T(2) * c1[k] - (c2[k] + c3[k]));
        if (fv3_fabs<T>(c4[k]) > fv3_fabs<T>(c2[k] - c3[k])) {
          const T pmp_1 = c1[k] - T(2) * gam[k + 1];
          const T lac_1 = pmp_1 + T(1.5) * gam[k + 2];
          c2[k] = fv3_min<T>(fv3_max<T>(c2[k], fv3_min<T>(c1[k], fv3_min<T>(pmp_1, lac_1))),
                              fv3_max<T>(c1[k], fv3_max<T>(pmp_1, lac_1)));
          const T pmp_2 = c1[k] + T(2) * gam[k];
          const T lac_2 = pmp_2 - T(1.5) * gam[k - 1];
          c3[k] = fv3_min<T>(fv3_max<T>(c3[k], fv3_min<T>(c1[k], fv3_min<T>(pmp_2, lac_2))),
                              fv3_max<T>(c1[k], fv3_max<T>(pmp_2, lac_2)));
          c4[k] = T(3) * (T(2) * c1[k] - (c2[k] + c3[k]));
        }
      }
      if (iv == 0) cs_limiters_column<T>(0, extm[k], c1[k], c2[k], c3[k], c4[k]);
    }
  } else {
    // abs(kord)<9 (:2487-2501): unconditional PMP/LAC clamp every k, no
    // extm/qmin gating inside this branch itself (extm is still consulted
    // by the shared trailing cs_limiters call below, exactly as :2671 does
    // for every kord branch).
    for (int k = 2; k <= km - 3; ++k) {
      const T pmp_1 = c1[k] - T(2) * gam[k + 1];
      const T lac_1 = pmp_1 + T(1.5) * gam[k + 2];
      c2[k] = fv3_min<T>(fv3_max<T>(c2[k], fv3_min<T>(c1[k], fv3_min<T>(pmp_1, lac_1))),
                          fv3_max<T>(c1[k], fv3_max<T>(pmp_1, lac_1)));
      const T pmp_2 = c1[k] + T(2) * gam[k];
      const T lac_2 = pmp_2 - T(1.5) * gam[k - 1];
      c3[k] = fv3_min<T>(fv3_max<T>(c3[k], fv3_min<T>(c1[k], fv3_min<T>(pmp_2, lac_2))),
                          fv3_max<T>(c1[k], fv3_max<T>(pmp_2, lac_2)));
      c4[k] = T(3) * (T(2) * c1[k] - (c2[k] + c3[k]));
      if (iv == 0) cs_limiters_column<T>(0, extm[k], c1[k], c2[k], c3[k], c4[k]);
    }
  }

  // ---- Bottom layer subgrid constraints (fv_mapz.F90:2579-2588) ----------
  if (iv == 0) {
    c3[km - 1] = fv3_max<T>(T(0), c3[km - 1]);
  } else if (iv == -1) {
    if (c3[km - 1] * c1[km - 1] <= T(0)) c3[km - 1] = T(0);
  }

  c4[km - 2] = T(3) * (T(2) * c1[km - 2] - (c2[km - 2] + c3[km - 2]));
  cs_limiters_column<T>(2, extm[km - 2], c1[km - 2], c2[km - 2], c3[km - 2], c4[km - 2]);

  c4[km - 1] = T(3) * (T(2) * c1[km - 1] - (c2[km - 1] + c3[km - 1]));
  cs_limiters_column<T>(1, extm[km - 1], c1[km - 1], c2[km - 1], c3[km - 1], c4[km - 1]);
}

// map_scalar (fv_mapz.F90:1519-1671 for the header's original line range;
// current authoritative source has the subroutine at :1665-1817): "standard
// FV3 remapping" branch (:1770-1811) PLUS the 5 GMAO top/bottom boundary-
// condition branches (:1744-1767), see file header. q1 doubles as a4(1,:)
// (never mutated by scalar_profile_column); c2/c3/c4 are caller scratch of
// length km. gam/q/extm are scalar_profile_column's own scratch (see above).
// gmao_bot/gmao_top are map_scalar's optional_bot/optional_top, ported as
// plain booleans (both default false, preserving every pre-existing call
// site's behavior unchanged) rather than as Fortran `optional` arguments --
// this header has no "argument absent" concept, and false is exactly
// map_scalar's own `.false.` default when optional_bot/optional_top are not
// present (:1705-1709).
template <class T>
FV3_VERTICAL_HD inline void map_scalar_column(
    const Column<const T> pe1, const Column<const T> dpe1,
    const Column<const T> pe2, const Column<const T> dpe2,
    const Column<const T> q1, Column<T> q2, T *c2, T *c3, T *c4, T *gam,
    T *q, bool *extm, int km, int kn, int iv, T qmin,
    bool gmao_bot = false, bool gmao_top = false, int kord = 9) {
  Column<T> c2v{c2, 1}, c3v{c3, 1}, c4v{c4, 1};
  scalar_profile_column<T>(q1, c2v, c3v, c4v, dpe1, gam, q, extm, km, iv, qmin, kord);

  const T r3 = T(1) / T(3);
  const T r23 = T(2) / T(3);

  int k0 = 0;
  for (int k = 0; k < kn; ++k) {
    // ---- LM1/LP0 search (:1733-1743), computed unconditionally every k,
    // exactly matching Fortran -- LM1/LP0 are kept 1-based (as in the
    // Fortran source) and translated to 0-based only at each array-access
    // use site below (Fortran index X -> C++ index X-1), so every branch's
    // arithmetic below is a literal transcription, not a re-derivation.
    int LM1 = 1, LP0 = 1;
    while (LP0 <= km) {
      if (pe1[LP0 - 1] < pe2[k]) {
        LP0 = LP0 + 1;
      } else {
        break;
      }
    }
    LM1 = (LP0 - 1 > 1) ? (LP0 - 1) : 1;
    LP0 = (LP0 < km) ? LP0 : km;

    if (gmao_bot && pe2[k] >= pe1[km]) {
      // Entire grid below old ps (:1746-1747)
      q2[k] = q1[km - 1];
    } else if (gmao_bot && LM1 == km && LP0 == km) {
      // Extrapolate linearly below last model level (:1750-1752)
      const T pfac0 = (pe2[k] - pe1[km - 1]) / (pe1[km - 1] - pe1[km - 2]);
      q2[k] = q1[km - 1] + (q1[km - 1] - q1[km - 2]) * pfac0;
    } else if (gmao_bot && LP0 == km) {
      // Interpolate linearly between levels, bottom variant (:1755-1757)
      const T pfac0 = (pe2[k] - pe1[LP0 - 1]) / (pe1[LM1 - 1] - pe1[LP0 - 1]);
      q2[k] = q1[LP0 - 1] + (q1[LM1 - 1] - q1[LP0 - 1]) * pfac0;
    } else if (gmao_top && LM1 == 1 && LP0 == 1) {
      // Extrapolate linearly above first model level (:1760-1762)
      const T pfac0 = (pe2[k] - pe1[0]) / (pe1[1] - pe1[0]);
      q2[k] = q1[0] + (q1[1] - q1[0]) * pfac0;
    } else if (gmao_top && LM1 == 1) {
      // Interpolate linearly between levels, top variant (:1765-1767)
      const T pfac0 = (pe2[k] - pe1[LP0 - 1]) / (pe1[LM1 - 1] - pe1[LP0 - 1]);
      q2[k] = q1[LP0 - 1] + (q1[LM1 - 1] - q1[LP0 - 1]) * pfac0;
    } else {
      // Standard FV3 remapping (:1770-1811) -- unchanged from pre-Prompt-61.
      for (int l = k0; l < km; ++l) {
        if (pe2[k] >= pe1[l] && pe2[k] <= pe1[l + 1]) {
          const T pl = (pe2[k] - pe1[l]) / dpe1[l];
          if (pe2[k + 1] <= pe1[l + 1]) {
            // entire new interval within old layer l (:1629-1637)
            const T pr = (pe2[k + 1] - pe1[l]) / dpe1[l];
            const T pfac1 = pr + pl;
            const T pfac2 = r3 * (pr * pfac1 + pl * pl);
            q2[k] = c2[l] + T(0.5) * (c4[l] + c3[l] - c2[l]) * pfac1 - c4[l] * pfac2;
            k0 = l;
          } else {
            // fractional area, spans one or more further old layers
            // (:1638-1660)
            const T pfac0 = pe1[l + 1] - pe2[k];
            const T pfac1 = T(1) + pl;
            const T pfac2 = r3 * (T(1) + pl * pfac1);
            T qsum = pfac0 * (c2[l] + T(0.5) * (c4[l] + c3[l] - c2[l]) * pfac1 -
                              c4[l] * pfac2);
            for (int m = l + 1; m < km; ++m) {
              if (pe2[k + 1] > pe1[m + 1]) {
                qsum += dpe1[m] * q1[m];
              } else {
                const T dp = pe2[k + 1] - pe1[m];
                const T esl = dp / dpe1[m];
                const T pfac0b = T(1) - r23 * esl;
                qsum += dp * (c2[m] + T(0.5) * esl * (c3[m] - c2[m] + c4[m] * pfac0b));
                k0 = m;
                break;
              }
            }
            q2[k] = qsum / dpe2[k];
          }
          break;
        }
      }
    }
  }
}

} // namespace fv3::gpu::vertical

#undef FV3_VERTICAL_HD
