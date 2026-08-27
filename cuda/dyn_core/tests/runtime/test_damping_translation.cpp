//===========================================================================
// Milestone 3A.11V Task 2 -- host oracle for the per-level damping translation.
//
// WHAT THIS TEST IS FOR
//
// Before Milestone 3A.11V nothing under cuda/ turned damping configuration
// scalars into the ten per-level arrays the storage contract declares, so every
// caller invented its own. The 3A.11 performance benchmark's invented table
// paired d4_bg = 0.16 with nord_k = 2, which violates the del-2n stability
// bound and was the sole cause of a multi-step instability that was, for
// several milestones, suspected of being a CUDA port defect. This test exists
// so that class of mistake is caught on a login node in milliseconds.
//
// WHERE THE EXPECTED VALUES COME FROM
//
// The npz = 10 production column asserted in t1 was derived by hand from the
// authoritative sources -- GFDL_atmos_cubed_sphere (GEOS-ESM fork, geos/v3.0.0,
// ca30e26) model/dyn_core.F90:700-765 combined with the GEOS scalar overrides
// at FV_StateMod.F90:569-793 -- BEFORE fv3_damping_translation.cpp was written,
// and is transcribed here as literals. So t1 is a genuine oracle, not the
// implementation compared against itself. If someone edits the translation, t1
// fails; if someone edits t1 to match a broken translation, the derivation
// comments below no longer support the literals and the edit is visible in
// review.
//
// t2-t6 check the four load-bearing ORDERING subtleties of the Fortran block
// and the sponge-arm boundaries, which are the parts most likely to be "tidied
// up" into something that no longer matches.
//
// t7-t9 check the stability metric arithmetic only. They deliberately do NOT
// assert physical stability claims beyond what Milestone 3A.11V actually
// measured -- see the calibration note in the header.
//
// Runs on the host with no GPU and no CUDA.
//===========================================================================

#include "fv3_damping_translation.hpp"

#include <cmath>
#include <cstdio>

using namespace fv3::gpu::damping;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    ++failures;
  }
}

void check_eq(int got, int want, const char* what) {
  if (got != want) {
    std::printf("  FAIL: %s (got %d, want %d)\n", what, got, want);
    ++failures;
  }
}

/* All expected values below are exact binary fractions or exact products of
 * them, so an exact-equality comparison would in principle work. A tight
 * tolerance is used instead because 0.2*0.1 and 0.5*0.2 are evaluated in a
 * different statement order here than in the translation, and asserting
 * bit-equality across that would be asserting something the Fortran does not
 * guarantee either. */
void check_near(double got, double want, const char* what) {
  const double tol = 1.0e-14 * (1.0 + std::fabs(want));
  if (!(std::fabs(got - want) <= tol)) {
    std::printf("  FAIL: %s (got %.17g, want %.17g)\n", what, got, want);
    ++failures;
  }
}

/* ---------------------------------------------------------------------------
 * t1: the GEOS npz=10 nonhydrostatic production column, value for value.
 *
 * Derivation, 0-based k (Fortran kf = k+1), with n_sponge = 9, nord = 2,
 * d2_bg = 0, d2_bg_k1 = 0.2, d2_bg_k2 = 0.1, vtdm4 = 0.01, d4_bg_top = 0.12,
 * do_vort_damp = .true., d_con = 1.0, ke_bg = 0, d_ext = 0, n_zfilter = 0:
 *
 *   d4_bg   : top == bot, so d4dmp(:) collapses to 0.12 at every level.
 *   damp_t  : set at :727 from damp_vt before the sponge and never reassigned,
 *             so max(0.01, min(0.01, 0.12/4)) = 0.01 at EVERY level.
 *   nord_t  : set at :715 from min(2,nord) before the sponge, so 2 everywhere.
 *   kgb     : ke_bg only at kf == npz, and ke_bg == 0, so 0 everywhere.
 *   d_ext   : nonhydrostatic, npz != 1, n_zfilter == 0, so 0 everywhere.
 *
 *   kf = 1  (k=0)  sponge arm 1 (:741). nord_k=0. d2_divg = max(0.01, 0, 0.2)
 *                  = 0.20. nord_w=0, damp_w=0.20. do_vort_damp so nord_v=0,
 *                  damp_vt = 0.5*0.20 = 0.10. d_con=0.
 *   kf = 2..8 (k=1..7)  arm 2 (:752): kf <= MAX(2, 9-1) = 8 and 0.1 > 0.01.
 *                  nord_k=0. d2_divg = max(0, 0.1) = 0.10. nord_w=0,
 *                  damp_w=0.10. nord_v=0, damp_vt = 0.5*0.10 = 0.05. d_con=0.
 *   kf = 9  (k=8)  arm 2 fails (9 > 8); arm 3 (:760): 9 <= MAX(3,9) = 9 and
 *                  0.1 > 0.05. nord_k=0. d2_divg = max(0, 0.2*0.1) = 0.02.
 *                  nord_w=0, damp_w=0.02. d_con=0. Arm 3 does NOT touch
 *                  nord_v or damp_vt, so they keep their pre-sponge values:
 *                  nord_v = min(2,2) = 2 and damp_vt = 0.01.
 *   kf = 10 (k=9)  no arm fires. d2_divg = min(0.20, 0) = 0.00.
 *                  nord_k = nord = 2. nord_v = 2, nord_w = 2.
 *                  damp_vt = 0.01, damp_w = damp_vt = 0.01. d_con = 1.0.
 * ------------------------------------------------------------------------ */
void t1_geos_production_npz10() {
  std::printf("t1: GEOS npz=10 nonhydrostatic production column\n");

  /* ptop is irrelevant here because top == bot collapses the d4dmp profile;
   * a realistic 1 Pa is passed anyway. high_resolution = false selects the
   * FV_StateMod.F90:687 branch, d4_bg_bot = 0.12. */
  const DampingFlags f = geos_production_flags(10, 1.0, /*high_resolution=*/false);
  const DampingColumn c = translate_damping(f, nullptr);

  check_eq((int)c.nord_k.size(), 10, "npz levels produced");
  check_eq(c.nord_t, 2, "nord_t == min(2,nord) == 2 (dyn_core.F90:715)");
  check_near(c.dddmp, 0.2, "dddmp passthrough (FV_StateMod.F90:682)");
  check(!c.d4dmp_profile_active, "d4dmp profile inactive when top == bot");

  const int want_nord_k[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 2};
  const int want_ndif[10]   = {0, 0, 0, 0, 0, 0, 0, 0, 2, 2};
  const int want_nord_w[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 2};
  const double want_d2[10]  = {0.20, 0.10, 0.10, 0.10, 0.10,
                               0.10, 0.10, 0.10, 0.02, 0.00};
  const double want_d4[10]  = {0.12, 0.12, 0.12, 0.12, 0.12,
                               0.12, 0.12, 0.12, 0.12, 0.12};
  const double want_dv[10]  = {0.10, 0.05, 0.05, 0.05, 0.05,
                               0.05, 0.05, 0.05, 0.01, 0.01};
  const double want_dw[10]  = {0.20, 0.10, 0.10, 0.10, 0.10,
                               0.10, 0.10, 0.10, 0.02, 0.01};
  const double want_dt[10]  = {0.01, 0.01, 0.01, 0.01, 0.01,
                               0.01, 0.01, 0.01, 0.01, 0.01};
  const double want_dcon[10]= {0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 1.0};

  char buf[96];
  for (int k = 0; k < 10; ++k) {
    std::snprintf(buf, sizeof(buf), "k=%d nord_k", k);
    check_eq(c.nord_k[(size_t)k], want_nord_k[k], buf);
    std::snprintf(buf, sizeof(buf), "k=%d ndif (nord_v)", k);
    check_eq(c.ndif[(size_t)k], want_ndif[k], buf);
    std::snprintf(buf, sizeof(buf), "k=%d nord_w", k);
    check_eq(c.nord_w[(size_t)k], want_nord_w[k], buf);
    std::snprintf(buf, sizeof(buf), "k=%d d2_bg", k);
    check_near(c.d2_bg[(size_t)k], want_d2[k], buf);
    std::snprintf(buf, sizeof(buf), "k=%d d4_bg", k);
    check_near(c.d4_bg[(size_t)k], want_d4[k], buf);
    std::snprintf(buf, sizeof(buf), "k=%d damp_v", k);
    check_near(c.damp_v[(size_t)k], want_dv[k], buf);
    std::snprintf(buf, sizeof(buf), "k=%d damp_w", k);
    check_near(c.damp_w[(size_t)k], want_dw[k], buf);
    std::snprintf(buf, sizeof(buf), "k=%d damp_t", k);
    check_near(c.damp_t[(size_t)k], want_dt[k], buf);
    std::snprintf(buf, sizeof(buf), "k=%d d_con", k);
    check_near(c.d_con[(size_t)k], want_dcon[k], buf);
    std::snprintf(buf, sizeof(buf), "k=%d kgb", k);
    check_near(c.kgb[(size_t)k], 0.0, buf);
    std::snprintf(buf, sizeof(buf), "k=%d d_ext", k);
    check_near(c.d_ext[(size_t)k], 0.0, buf);
  }

  /* The specific pairing that broke 3A.11: nord_k > 0 must never coexist with
   * a d4_bg at or above the del-2n critical value 2^(1/(nord+1))/8. At
   * nord_k = 2 that is 0.157490, and the authoritative column supplies 0.12.
   * This is the assertion the fabricated benchmark table would have failed. */
  for (int k = 0; k < 10; ++k) {
    if (c.nord_k[(size_t)k] > 0) {
      const double crit =
          std::pow(2.0, 1.0 / (double)(c.nord_k[(size_t)k] + 1)) / 8.0;
      std::snprintf(buf, sizeof(buf),
                    "k=%d d4_bg below del-2n critical value", k);
      check(c.d4_bg[(size_t)k] < crit, buf);
    }
  }
}

/* ---------------------------------------------------------------------------
 * t2: ORDERING SUBTLETY 1 and 2 -- nord_t and damp_t are set before the sponge
 * and never reassigned, so they are constant over the column even where the
 * sponge resets nord_v and damp_vt.
 *
 * The discriminating configuration is one where the sponge DOES fire and DOES
 * change nord_v/damp_vt, so a translation that (wrongly) tracked nord_v with
 * nord_t would produce a k-dependent nord_t and fail here.
 * ------------------------------------------------------------------------ */
void t2_nord_t_and_damp_t_are_column_constants() {
  std::printf("t2: nord_t / damp_t are pre-sponge and column-constant\n");
  const DampingFlags f = geos_production_flags(10, 1.0, false);
  const DampingColumn c = translate_damping(f, nullptr);

  /* Precondition: the sponge really did move nord_v and damp_v, otherwise the
   * test proves nothing. */
  check(c.ndif[0] != c.ndif[9], "precondition: sponge changed nord_v");
  check(c.damp_v[0] != c.damp_v[9], "precondition: sponge changed damp_v");

  const double dt0 = c.damp_t[0];
  for (int k = 1; k < 10; ++k) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "damp_t[%d] == damp_t[0]", k);
    check_near(c.damp_t[(size_t)k], dt0, buf);
  }
  /* And its value is the pre-sponge damp_vt, not the post-sponge one. */
  check_near(dt0, 0.01, "damp_t == pre-sponge max(0.01,min(vtdm4,d4_bg_top/4))");
  check_eq(c.nord_t, 2, "nord_t == min(2,nord), not the sponge's nord_v == 0");
}

/* ---------------------------------------------------------------------------
 * t3: ORDERING SUBTLETY 3 -- damp_w IS overwritten by every sponge arm, unlike
 * damp_t. In the sponge, damp_w takes d2_divg; outside it, damp_vt.
 * ------------------------------------------------------------------------ */
void t3_damp_w_tracks_the_sponge() {
  std::printf("t3: damp_w is overwritten by the sponge, damp_t is not\n");
  const DampingFlags f = geos_production_flags(10, 1.0, false);
  const DampingColumn c = translate_damping(f, nullptr);

  for (int k = 0; k <= 8; ++k) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "sponge k=%d: damp_w == d2_divg", k);
    check_near(c.damp_w[(size_t)k], c.d2_bg[(size_t)k], buf);
  }
  check_near(c.damp_w[9], c.damp_v[9],
             "non-sponge k=9: damp_w == damp_vt (dyn_core.F90:726)");
  check(c.damp_w[0] != c.damp_t[0],
        "damp_w and damp_t diverge in the sponge despite adjacent assignment");
}

/* ---------------------------------------------------------------------------
 * t4: ORDERING SUBTLETY 4 -- the third sponge arm (:760-763) does not touch
 * nord_v or damp_vt, and has no do_vort_damp guard. Its level therefore keeps
 * nord_v = min(2,nord) while its divergence and w operators drop to del-2.
 *
 * Verified structurally rather than by restating k=8: the level selected by
 * arm 3 must satisfy nord_k == 0 and nord_w == 0 (arm 3 set them) while
 * ndif == min(2,nord) (arm 3 did not).
 * ------------------------------------------------------------------------ */
void t4_third_sponge_arm_is_asymmetric() {
  std::printf("t4: third sponge arm leaves nord_v/damp_vt alone\n");
  const DampingFlags f = geos_production_flags(10, 1.0, false);
  const DampingColumn c = translate_damping(f, nullptr);

  /* Arm 3 covers Fortran kf in (MAX(2,n_sponge-1), MAX(3,n_sponge)] = (8, 9],
   * i.e. kf = 9, i.e. 0-based k = 8. Recomputed from the flags, not hardcoded,
   * so the test still means what it says if n_sponge changes. */
  const int arm2_last = (f.n_sponge - 1 > 2) ? f.n_sponge - 1 : 2;
  const int arm3_last = (f.n_sponge > 3) ? f.n_sponge : 3;
  check(arm3_last > arm2_last, "precondition: arm 3 covers at least one level");
  for (int kf = arm2_last + 1; kf <= arm3_last; ++kf) {
    const size_t k = (size_t)(kf - 1);
    char buf[96];
    std::snprintf(buf, sizeof(buf), "arm3 kf=%d: nord_k reset to 0", kf);
    check_eq(c.nord_k[k], 0, buf);
    std::snprintf(buf, sizeof(buf), "arm3 kf=%d: nord_w reset to 0", kf);
    check_eq(c.nord_w[k], 0, buf);
    std::snprintf(buf, sizeof(buf), "arm3 kf=%d: nord_v NOT reset", kf);
    check_eq(c.ndif[k], (f.nord < 2 ? f.nord : 2), buf);
    std::snprintf(buf, sizeof(buf), "arm3 kf=%d: damp_vt NOT reset", kf);
    check_near(c.damp_v[k], 0.01, buf);
  }
}

/* ---------------------------------------------------------------------------
 * t5: the sponge disable branch (:736-737) discards the min(0.20, ...) cap
 * applied at :718, so d2_divg can exceed 0.20 when n_sponge <= 0. Reproduced
 * faithfully because the authoritative code does it; asserted here so nobody
 * "fixes" it silently.
 * ------------------------------------------------------------------------ */
void t5_no_sponge_bypasses_the_d2_cap() {
  std::printf("t5: n_sponge <= 0 bypasses the min(0.20, d2_bg) cap\n");
  DampingFlags f = geos_production_flags(10, 1.0, false);
  f.d2_bg = 0.5;

  f.n_sponge = 9;
  const DampingColumn with_sponge = translate_damping(f, nullptr);
  /* With the sponge on, level kf=10 takes the capped path at :718. */
  check_near(with_sponge.d2_bg[9], 0.20,
             "sponge on, non-sponge level: d2_divg capped at 0.20");

  f.n_sponge = 0;
  const DampingColumn no_sponge = translate_damping(f, nullptr);
  for (int k = 0; k < 10; ++k) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "sponge off k=%d: d2_divg uncapped", k);
    check_near(no_sponge.d2_bg[(size_t)k], 0.5, buf);
  }
  check_eq(no_sponge.nord_k[0], f.nord,
           "sponge off: nord_k stays at flagstruct%nord even at the top");
}

/* ---------------------------------------------------------------------------
 * t6: kgb and the d4dmp sine profile.
 *
 * kgb is ke_bg at the bottom level and zero elsewhere (:707-711) -- checked
 * with a nonzero ke_bg so the assertion is not vacuous under the production
 * value of 0.
 *
 * The profile is exercised with top != bot, which is the only configuration
 * that reads pfull. Its endpoints are checked rather than its interior: at
 * pfull == ptop the weight is sin(pi/2) = 1 so d4_bg == d4_bg_top, and for
 * pfull >= 25000 the log flips sign, the weight clamps to 0, and
 * d4_bg == d4_bg_bot. Both are exact and independent of pi's precision.
 * ------------------------------------------------------------------------ */
void t6_kgb_and_the_d4dmp_profile() {
  std::printf("t6: kgb bottom-only, and the d4dmp sine profile endpoints\n");

  DampingFlags f = geos_production_flags(10, 100.0, false);
  f.ke_bg = 1.5e-4;
  DampingColumn c = translate_damping(f, nullptr);
  for (int k = 0; k < 9; ++k) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "kgb[%d] == 0", k);
    check_near(c.kgb[(size_t)k], 0.0, buf);
  }
  check_near(c.kgb[9], 1.5e-4, "kgb[npz-1] == ke_bg (dyn_core.F90:707-711)");

  /* Now the profile. ptop = 100 Pa; pfull[0] == ptop -> weight 1;
   * pfull[9] = 90000 > 25000 -> weight clamped to 0. */
  f = geos_production_flags(10, 100.0, /*high_resolution=*/true);
  check(f.d4_bg_top != f.d4_bg_bot, "precondition: top != bot");
  double pfull[10] = {100.0,  300.0,  1000.0,  3000.0,  8000.0,
                      15000.0, 25000.0, 45000.0, 70000.0, 90000.0};
  c = translate_damping(f, pfull);
  check(c.d4dmp_profile_active, "profile active when top != bot and pfull given");
  check_near(c.d4_bg[0], f.d4_bg_top, "pfull == ptop -> d4_bg == d4_bg_top");
  check_near(c.d4_bg[9], f.d4_bg_bot, "pfull >> 25000 -> d4_bg == d4_bg_bot");
  check_near(c.d4_bg[6], f.d4_bg_bot, "pfull == 25000 -> weight 0 exactly");
  /* Monotone decreasing from top to bot, since the sine argument decreases
   * monotonically with increasing pfull over this range. */
  for (int k = 1; k < 10; ++k) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "d4_bg[%d] <= d4_bg[%d]", k, k - 1);
    check(c.d4_bg[(size_t)k] <= c.d4_bg[(size_t)(k - 1)] + 1e-15, buf);
  }
  /* Passing nullptr with the profile active must fall back to the constant,
   * matching the hydrostatic shortcut, not produce garbage. */
  const DampingColumn cn = translate_damping(f, nullptr);
  check(!cn.d4dmp_profile_active, "nullptr pfull reports profile inactive");
  check_near(cn.d4_bg[9], f.d4_bg_top, "nullptr pfull -> constant d4_bg_top");
}

/* ---------------------------------------------------------------------------
 * t7: the stability metric computes the documented formula, and reports
 * gamma_d4 == 0 where nord_k == 0 because d_sw does not read d4_bg there.
 * ------------------------------------------------------------------------ */
void t7_stability_metric_arithmetic() {
  std::printf("t7: stability metric arithmetic\n");
  const DampingFlags f = geos_production_flags(10, 1.0, false);
  const DampingColumn c = translate_damping(f, nullptr);

  /* Conservative area ratios of 1.0: pass da_min for both areas. */
  const double da_min = 9.800216e9;
  const double da_min_c = 9.900108e9;
  const StabilityReport r =
      damping_stability(c, da_min, da_min, da_min_c, da_min_c);

  check_eq((int)r.level.size(), 10, "one stability row per level");

  for (int k = 0; k <= 8; ++k) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "k=%d nord_k==0 -> d4 inactive", k);
    check(!r.level[(size_t)k].d4_active, buf);
    std::snprintf(buf, sizeof(buf), "k=%d gamma_d4 reported as 0", k);
    check_near(r.level[(size_t)k].gamma_d4, 0.0, buf);
  }
  check(r.level[9].d4_active, "k=9 nord_k==2 -> d4 active");
  /* (8*0.12)^3 = 0.96^3 = 0.884736, times an area ratio of 1. */
  check_near(r.level[9].gamma_d4, std::pow(0.96, 3.0), "k=9 gamma_d4 == 0.96^3");
  /* d2_divg == 0 at k=9, so the static del-2 term vanishes there. */
  check_near(r.level[9].gamma_d2_static, 0.0, "k=9 gamma_d2_static == 0");
  /* k=0: d2_divg = 0.20 -> 8*0.20 = 1.6. */
  check_near(r.level[0].gamma_d2_static, 1.6, "k=0 gamma_d2_static == 1.6");

  /* The production column is stable from configuration alone. This IS a
   * measured claim: 3A.11V ran d4_bg = 0.12 at nord = 2 for 300 calls at
   * 8x8x10 (three bitwise-identical repetitions), 16x16 and 32x32, all
   * bounded. See docs/agent_handoff/milestone_3A11V_dt_sensitivity.md. */
  check(!r.any_unstable_static,
        "GEOS production column: no level unstable by configuration");
  check_eq(r.worst_level, 0, "worst static gamma is the top sponge level");
  check(r.worst_gamma < kGammaLimit, "worst static gamma below the limit");
}

/* ---------------------------------------------------------------------------
 * t8: the metric flags the exact pairing that the fabricated benchmark table
 * used, and Milestone 3A.11V measured to be unstable.
 *
 * d4_bg = 0.16 with nord_k = 2 gives (8*0.16)^3 = 1.28^3 = 2.097 > 2. The
 * threshold measurement bracketed the critical scale at (0.98, 1.00] times
 * 0.16, i.e. the pairing is 1.6% over. This is the one row of the audit that
 * is anchored to measurement rather than derivation.
 * ------------------------------------------------------------------------ */
void t8_metric_flags_the_fabricated_pairing() {
  std::printf("t8: metric flags the fabricated (d4_bg=0.16, nord_k=2) pairing\n");

  /* Built by hand rather than through translate_damping(), because
   * translate_damping() cannot produce this pairing -- which is the point. */
  DampingColumn bad;
  bad.nord_k = {2};
  bad.ndif = {2};
  bad.nord_w = {2};
  bad.d2_bg = {0.0};
  bad.d4_bg = {0.16};
  bad.damp_v = {0.10};
  bad.damp_w = {0.06};
  bad.damp_t = {0.08};
  bad.d_con = {1.0};
  bad.kgb = {0.0};
  bad.d_ext = {0.0};
  bad.nord_t = 2;
  bad.dddmp = 0.5;

  const double da_min = 9.800216e9;
  const double da_min_c = 9.900108e9;
  const StabilityReport r =
      damping_stability(bad, da_min, da_min, da_min_c, da_min_c);

  check_near(r.level[0].gamma_d4, std::pow(1.28, 3.0), "gamma_d4 == 1.28^3");
  check(r.level[0].gamma_d4 > kGammaLimit, "1.28^3 exceeds the limit");
  check(r.level[0].unstable_static, "level flagged UNSTABLE-BY-CONFIG");
  check(r.any_unstable_static, "report flagged unstable by configuration");

  /* And the safe scaling that 3A.11V used to get a bounded 300-call run
   * (d4_bg *= 0.75, i.e. 0.12) must NOT be flagged. */
  bad.d4_bg[0] = 0.12;
  const StabilityReport ok =
      damping_stability(bad, da_min, da_min, da_min_c, da_min_c);
  check(!ok.any_unstable_static, "d4_bg = 0.12 at nord_k = 2 not flagged");
}

/* ---------------------------------------------------------------------------
 * t9: the raw fv_arrays defaults translate without crashing and produce the
 * documented (d4_bg_top = 0.16, nord = 1) pairing.
 *
 * gamma_d4 there is (8*0.16)^2 = 1.6384, i.e. under the limit -- so the raw
 * defaults are NOT the unstable pairing. Only nord = 2 combined with 0.16 is,
 * and no shipped configuration forms it. That is the substance of the Task 3
 * classification: the risk is user-reachable, not present in any default.
 *
 * This test also pins the sponge behaviour of the large default d2_bg_k1/k2
 * (4.0 / 2.0), which bypass the 0.20 cap through the sponge arms. It asserts
 * the translated VALUES, which are facts about the Fortran, and deliberately
 * makes no stability claim about them: the del-2 threshold has not been
 * threshold-measured, only corroborated by the existence of the min(0.20,...)
 * cap at dyn_core.F90:718 sitting just under the derived critical 0.25.
 * ------------------------------------------------------------------------ */
void t9_fv_arrays_defaults() {
  std::printf("t9: raw fv_arrays defaults\n");
  const DampingFlags f = fv_arrays_defaults(10, 1.0);
  const DampingColumn c = translate_damping(f, nullptr);

  check_eq(c.nord_t, 1, "nord_t == min(2,1) == 1");
  for (int k = 0; k < 10; ++k) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "k=%d d4_bg == 0.16", k);
    check_near(c.d4_bg[(size_t)k], 0.16, buf);
    /* do_vort_damp is false by default, so every vorticity coefficient is
     * exactly zero -- dyn_core.F90:723. */
    std::snprintf(buf, sizeof(buf), "k=%d damp_v == 0 (do_vort_damp false)", k);
    check_near(c.damp_v[(size_t)k], 0.0, buf);
  }
  check_near(c.damp_t[0], 0.0, "damp_t == 0 when do_vort_damp is false");

  /* n_sponge = 1: arm 1 at kf=1, arm 2 at kf=2 (MAX(2,0) = 2), arm 3 at kf=3
   * (MAX(3,1) = 3). Levels kf=4..10 take no arm. */
  check_near(c.d2_bg[0], 4.0, "kf=1: max(0.01, d2_bg, d2_bg_k1) == 4.0");
  check_near(c.d2_bg[1], 2.0, "kf=2: max(d2_bg, d2_bg_k2) == 2.0");
  check_near(c.d2_bg[2], 0.4, "kf=3: max(d2_bg, 0.2*d2_bg_k2) == 0.4");
  check_near(c.d2_bg[3], 0.0, "kf=4: min(0.20, d2_bg) == 0.0");
  check_eq(c.nord_k[0], 0, "kf=1 nord_k reset to 0");
  check_eq(c.nord_k[3], 1, "kf=4 nord_k == flagstruct%nord == 1");

  /* d_ext: default 0.02, n_zfilter 0, nonhydrostatic -> zero at every level. */
  for (int k = 0; k < 10; ++k) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "k=%d d_ext gated off by n_zfilter", k);
    check_near(c.d_ext[(size_t)k], 0.0, buf);
  }

  const double da_min = 9.800216e9;
  const StabilityReport r = damping_stability(c, da_min, da_min, da_min, da_min);
  check_near(r.level[3].gamma_d4, std::pow(1.28, 2.0),
             "defaults gamma_d4 == (8*0.16)^2 == 1.6384");
  check(r.level[3].gamma_d4 < kGammaLimit,
        "the raw default (0.16, nord=1) pairing is NOT over the limit");
}

/* ---------------------------------------------------------------------------
 * t10: degenerate inputs must not crash or produce partially-sized vectors.
 * ------------------------------------------------------------------------ */
void t10_degenerate_inputs() {
  std::printf("t10: degenerate inputs\n");
  DampingFlags f = geos_production_flags(0, 1.0, false);
  DampingColumn c = translate_damping(f, nullptr);
  check(c.nord_k.empty(), "npz == 0 produces an empty column");
  StabilityReport r = damping_stability(c, 1.0, 1.0, 1.0, 1.0);
  check(r.level.empty(), "empty column produces an empty report");
  check_eq(r.worst_level, -1, "empty report has worst_level == -1");

  /* npz == 1 takes the :736 disable branch regardless of n_sponge. */
  f = geos_production_flags(1, 1.0, false);
  c = translate_damping(f, nullptr);
  check_eq((int)c.nord_k.size(), 1, "npz == 1 produces one level");
  check_near(c.d2_bg[0], f.d2_bg, "npz == 1: d2_divg == raw d2_bg");
  check_near(c.d_ext[0], f.d_ext, "npz == 1: d_ext ungated");
  check_near(c.kgb[0], f.ke_bg, "npz == 1: the only level is the bottom one");
}

}  // namespace

int main() {
  std::printf("=== Milestone 3A.11V Task 2: per-level damping translation ===\n");
  t1_geos_production_npz10();
  t2_nord_t_and_damp_t_are_column_constants();
  t3_damp_w_tracks_the_sponge();
  t4_third_sponge_arm_is_asymmetric();
  t5_no_sponge_bypasses_the_d2_cap();
  t6_kgb_and_the_d4dmp_profile();
  t7_stability_metric_arithmetic();
  t8_metric_flags_the_fabricated_pairing();
  t9_fv_arrays_defaults();
  t10_degenerate_inputs();

  if (failures) {
    std::printf("\nFAILED: %d check(s)\n", failures);
    return 1;
  }
  std::printf("\nAll damping translation checks passed.\n");
  std::printf("  * This validates the HOST-SIDE TRANSLATION only:\n");
  std::printf("    configuration scalars -> per-level arrays, against a\n");
  std::printf("    hand-derived reading of dyn_core.F90:677-765.\n");
  std::printf("  * It does NOT validate that the CUDA operators consume these\n");
  std::printf("    values correctly, and it is NOT a numerical validation\n");
  std::printf("    against authoritative Fortran output.\n");

  /* Print the production column and its stability table, so a CTest log
   * carries the actual numbers rather than only a pass count. */
  const DampingFlags f = geos_production_flags(10, 1.0, false);
  const DampingColumn c = translate_damping(f, nullptr);
  const double da_min = 9.800216e9, da_min_c = 9.900108e9;
  const StabilityReport r =
      damping_stability(c, da_min, da_min, da_min_c, da_min_c);
  std::printf("\n%s", format_damping_column(c).c_str());
  std::printf("%s", format_stability_report(c, r).c_str());
  return 0;
}
