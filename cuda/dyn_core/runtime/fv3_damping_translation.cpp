//===========================================================================
// Milestone 3A.11V, Task 2 -- implementation of the authoritative per-level
// damping translation declared in include/fv3_damping_translation.hpp.
//
// A statement-ordered reimplementation of
//   GFDL_atmos_cubed_sphere (GEOS-ESM fork, geos/v3.0.0, ca30e26)
//   model/dyn_core.F90:677-690   the cached d4dmp() profile
//   model/dyn_core.F90:700-765   the per-level loop and the sponge
//
// The Fortran is reproduced as comments immediately above each translated
// statement, so a future reader can diff this against the source without
// opening it. Statement ORDER is part of the specification here and is
// preserved exactly; see the four ordering subtleties called out below.
//
// numerical semantics changed: NO. This file introduces no numerics of its
// own. It replaces hand-invented per-level tables with the values the
// authoritative code computes.
//===========================================================================

#include "fv3_damping_translation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace fv3::gpu::damping {

namespace {

/* FMS constants_mod pi. FMS is not present on this filesystem (see the
 * grav note in docs/agent_handoff/milestone_3A11V_dt_sensitivity.md), so
 * this uses the mathematical constant rather than claiming to have read
 * FMS's value. It is read ONLY by the d4dmp() sine profile, which is
 * inactive whenever d4_bg_top == d4_bg_bot -- the production case -- so no
 * production number depends on this line. */
constexpr double kPi = 3.14159265358979323846;

/* dyn_core.F90:683, the pressure at which the profile weight crosses. */
constexpr double kD4dmpPref = 25000.0;

/* Coefficient gates in the consuming operators. Below these the operator is
 * skipped entirely, so a coefficient under the gate cannot destabilize
 * anything regardless of its stability metric.
 *   sw_core.F90 damp_v / damp_w path : damp > 1.E-5
 *   tp_core.F90:217,237 deln_flux    : damp_c > 1.e-4 */
constexpr double kDampVGate = 1.0e-5;
constexpr double kDampTGate = 1.0e-4;

/* The del-2n growth norm. See the header for its derivation and for the
 * measurement that calibrated it. */
double gamma_del2n(double coeff, int order, double area_ratio) {
  if (coeff <= 0.0) return 0.0;
  return std::pow(8.0 * coeff, (double)(order + 1)) * area_ratio;
}

}  // namespace

DampingColumn translate_damping(const DampingFlags& f, const double* pfull) {
  DampingColumn c;
  const int npz = f.npz;
  if (npz <= 0) return c;

  c.nord_k.assign((size_t)npz, 0);
  c.ndif.assign((size_t)npz, 0);
  c.nord_w.assign((size_t)npz, 0);
  c.d2_bg.assign((size_t)npz, 0.0);
  c.d4_bg.assign((size_t)npz, 0.0);
  c.damp_v.assign((size_t)npz, 0.0);
  c.damp_w.assign((size_t)npz, 0.0);
  c.damp_t.assign((size_t)npz, 0.0);
  c.d_con.assign((size_t)npz, 0.0);
  c.kgb.assign((size_t)npz, 0.0);
  c.d_ext.assign((size_t)npz, 0.0);
  c.dddmp = f.dddmp;

  /* -------------------------------------------------------------------
   * dyn_core.F90:677-690  -- the d4dmp() profile.
   *
   *   if ( .not. d4dmp_initialized ) then
   *      allocate( d4dmp(npz) )
   *      if ( .not. hydrostatic ) then
   *          do k=1,npz
   *             d4dmp(k) = MAX(0.0,MIN(1.0,SIN(0.5*pi*LOG(25000.0/pfull(k)) &
   *                                                  /LOG(25000.0/ptop))))
   *             d4dmp(k) = flagstruct%d4_bg_top*d4dmp(k) &
   *                      + flagstruct%d4_bg_bot*(1.0-d4dmp(k))
   *          end do
   *       else
   *          d4dmp(:) = flagstruct%d4_bg_top
   *       endif
   *       d4dmp_initialized = .true.
   *   endif
   *
   * The `d4dmp_initialized` saved logical at dyn_core.F90:150 means this runs
   * ONCE per executable, not once per call. That caching is why the
   * adiabatic_init temporary override at FV_StateMod.F90:5340-5346 -- which
   * forces d4_bg_top = d4_bg_bot = 0.0 and restores them at :5536-5540 -- can
   * freeze d4dmp(:) at zero for an entire production run if it happens to run
   * first. That latent Fortran issue is recorded in the 3A.11V handoff; it is
   * NOT reproduced here, because a translation layer that is called once by
   * its caller has no equivalent hazard, and reproducing a bug is not
   * fidelity.
   *
   * The profile is a no-op when top == bot: the two terms sum to
   * d4_bg_top*w + d4_bg_top*(1-w) = d4_bg_top for any w. Taking the shortcut
   * in that case is therefore exact, not an approximation, and it means the
   * production configuration never depends on pfull, on kPi, or on the
   * single-precision Fortran literals in the SIN argument.
   * ---------------------------------------------------------------- */
  const bool profile_needed =
      (!f.hydrostatic) && (f.d4_bg_top != f.d4_bg_bot) && pfull != nullptr;
  c.d4dmp_profile_active = profile_needed;

  if (profile_needed) {
    const double denom = std::log(kD4dmpPref / f.ptop);
    for (int k = 0; k < npz; ++k) {
      double w = 0.0;
      if (denom != 0.0 && pfull[k] > 0.0) {
        w = std::sin(0.5 * kPi * std::log(kD4dmpPref / pfull[k]) / denom);
      }
      w = std::max(0.0, std::min(1.0, w));
      c.d4_bg[(size_t)k] = f.d4_bg_top * w + f.d4_bg_bot * (1.0 - w);
    }
  } else {
    /* dyn_core.F90:687  d4dmp(:) = flagstruct%d4_bg_top */
    for (int k = 0; k < npz; ++k) c.d4_bg[(size_t)k] = f.d4_bg_top;
  }

  /* -------------------------------------------------------------------
   * dyn_core.F90:700-765 -- the per-level loop.
   *
   * Fortran k runs 1..npz; here kf = k0 + 1 is the Fortran index so every
   * comparison below can be read directly against the source.
   * ---------------------------------------------------------------- */
  for (int k0 = 0; k0 < npz; ++k0) {
    const int kf = k0 + 1;

    /* :705  nord_k = flagstruct%nord */
    int nord_k = f.nord;

    /* :707-711  if ( k==npz ) kgb = flagstruct%ke_bg else kgb = 0. */
    const double kgb = (kf == npz) ? f.ke_bg : 0.0;

    /* :713  nord_v(k) = min(2, flagstruct%nord) */
    int nord_v = std::min(2, f.nord);

    /* :714  nord_w = nord_v(k)
     * :715  nord_t = nord_v(k)
     *
     * ORDERING SUBTLETY 1. nord_t is read HERE, before the sponge, and no
     * sponge arm below reassigns it. So nord_t == min(2,nord) at every level
     * even where nord_v(k) is later reset to 0. It is therefore a scalar in
     * DampingColumn, matching NumericalConfig::nord_t. */
    int nord_w = nord_v;
    const int nord_t = nord_v;

    /* :718  d2_divg = min(0.20, flagstruct%d2_bg) */
    double d2_divg = std::min(0.20, f.d2_bg);

    /* :720-724
     *   if ( flagstruct%do_vort_damp ) then
     *        damp_vt(k) = max(0.01,min(flagstruct%vtdm4,flagstruct%d4_bg_top/4.0))
     *   else
     *        damp_vt(k) = 0.
     *   endif
     *
     * Note the max(0.01, ...) FLOOR: with do_vort_damp true, damp_vt can
     * never fall below 0.01 no matter how small vtdm4 is. */
    double damp_vt = f.do_vort_damp
                         ? std::max(0.01, std::min(f.vtdm4, f.d4_bg_top / 4.0))
                         : 0.0;

    /* :726  damp_w = damp_vt(k)
     * :727  damp_t = damp_vt(k)
     *
     * ORDERING SUBTLETY 2. damp_t is read HERE and, like nord_t, is never
     * reassigned by the sponge -- so damp_t is constant over the whole
     * column. damp_w, by contrast, IS overwritten in all three sponge arms
     * (subtlety 3). The two lines look symmetric and are not. */
    double damp_w = damp_vt;
    const double damp_t = damp_vt;

    /* :729-733
     *   if ( npz==1 .or. k<=flagstruct%n_zfilter .or. hydrostatic ) then
     *      d_ext(k) = flagstruct%d_ext
     *   else
     *      d_ext(k) = 0.0
     *   endif */
    const double d_ext =
        (npz == 1 || kf <= f.n_zfilter || f.hydrostatic) ? f.d_ext : 0.0;

    /* :734  d_con_k = flagstruct%d_con */
    double d_con_k = f.d_con;

    /* :736-765  the sponge. */
    if (npz == 1 || f.n_sponge <= 0) {
      /* :737  d2_divg = flagstruct%d2_bg
       *
       * Note this is the UNCAPPED d2_bg -- the min(0.20, ...) at :718 is
       * discarded on this branch. With the fv_arrays default d2_bg = 0.0 that
       * is invisible, but a user setting d2_bg = 0.5 with n_sponge = 0 gets
       * 0.5 here and 0.20 with a sponge. Faithfully reproduced; flagged
       * because it looks like a Fortran oversight and is not this project's
       * to fix. */
      d2_divg = f.d2_bg;
    } else if (kf == 1) {
      /* :741-751 */
      nord_k = 0;
      d2_divg = std::max(0.01, std::max(f.d2_bg, f.d2_bg_k1));
      nord_w = 0;
      damp_w = d2_divg;
      if (f.do_vort_damp) {
        nord_v = 0;
        damp_vt = 0.5 * d2_divg;
      }
      d_con_k = 0.0;
    } else if (kf <= std::max(2, f.n_sponge - 1) && f.d2_bg_k2 > 0.01) {
      /* :752-759 */
      nord_k = 0;
      d2_divg = std::max(f.d2_bg, f.d2_bg_k2);
      nord_w = 0;
      damp_w = d2_divg;
      if (f.do_vort_damp) {
        nord_v = 0;
        damp_vt = 0.5 * d2_divg;
      }
      d_con_k = 0.0;
    } else if (kf <= std::max(3, f.n_sponge) && f.d2_bg_k2 > 0.05) {
      /* :760-763
       *
       * ORDERING SUBTLETY 4. This third arm does NOT touch nord_v(k) or
       * damp_vt(k), asymmetrically with the two arms above it, and has no
       * do_vort_damp guard at all. So the deepest sponge level keeps
       * nord_v = min(2,nord) and damp_vt = max(0.01,min(vtdm4,d4_bg_top/4))
       * while its divergence and w operators drop to del-2. That asymmetry is
       * exactly what produces nord_v = 2 at k = npz-1 in the GEOS npz=10
       * production column, and it is deliberate in the source, not a typo:
       * removing it would change the numerics. */
      nord_k = 0;
      d2_divg = std::max(f.d2_bg, 0.2 * f.d2_bg_k2);
      nord_w = 0;
      damp_w = d2_divg;
      d_con_k = 0.0;
    }

    /* :789-802 -- what actually crosses into d_sw. */
    c.nord_k[(size_t)k0] = nord_k;
    c.ndif[(size_t)k0] = nord_v;   /* d_sw's nord_v dummy */
    c.nord_w[(size_t)k0] = nord_w;
    c.d2_bg[(size_t)k0] = d2_divg;
    c.damp_v[(size_t)k0] = damp_vt;
    c.damp_w[(size_t)k0] = damp_w;
    c.damp_t[(size_t)k0] = damp_t;
    c.d_con[(size_t)k0] = d_con_k;
    c.kgb[(size_t)k0] = kgb;
    c.d_ext[(size_t)k0] = d_ext;
    c.nord_t = nord_t;   /* invariant over k, see subtlety 1 */
  }

  return c;
}

DampingFlags geos_production_flags(int npz, double ptop, bool high_resolution) {
  /* FV_StateMod.F90:569-793, the `if (ntiles == 6)` arm. Line numbers are
   * given per member so each can be re-checked independently. */
  DampingFlags f;
  f.npz = npz;
  f.ptop = ptop;
  f.nord = 2;                 /* :681 */
  f.dddmp = 0.2;              /* :682 */
  f.d2_bg = 0.0;              /* :689 */
  f.d4_bg_top = 0.12;         /* :683 */
  f.d4_bg_bot = high_resolution ? 0.06 : 0.12;  /* :685 / :687 */
  f.d2_bg_k1 = 0.2;           /* :695 */
  f.d2_bg_k2 = 0.1;           /* :696 */
  f.vtdm4 = 0.01;             /* :744 */
  f.do_vort_damp = true;      /* :743 */
  f.d_con = 1.0;              /* :742 */
  f.d_ext = 0.0;              /* :690 */
  f.n_zfilter = 0;            /* not set for npz=10; fv_arrays.F90:522 default */
  f.ke_bg = 0.0;              /* :560 */
  f.n_sponge = 9;             /* :694 */
  f.hydrostatic = false;      /* :735-739 */
  return f;
}

DampingFlags fv_arrays_defaults(int npz, double ptop) {
  /* fv_arrays.F90 initializers, i.e. what a caller gets with no namelist and
   * no GEOS override. Retained because the stability audit has to report on
   * configurations a user can actually reach, and this is the most reachable
   * one: it pairs d4_bg_top = 0.16 with nord = 1. */
  DampingFlags f;
  f.npz = npz;
  f.ptop = ptop;
  f.nord = 1;                 /* :342 */
  f.dddmp = 0.0;
  f.d2_bg = 0.0;
  f.d4_bg_top = 0.16;         /* :359 */
  f.d4_bg_bot = 0.16;         /* :365 */
  f.d2_bg_k1 = 4.0;           /* :379 */
  f.d2_bg_k2 = 2.0;           /* :388 */
  f.vtdm4 = 0.0;              /* :371 */
  f.do_vort_damp = false;     /* :453 */
  f.d_con = 0.0;              /* :766 */
  f.d_ext = 0.02;             /* :527 */
  f.n_zfilter = 0;            /* :522 */
  f.ke_bg = 0.0;              /* :771 */
  f.n_sponge = 1;             /* :524 */
  f.hydrostatic = false;
  return f;
}

StabilityReport damping_stability(const DampingColumn& c,
                                  double da_min, double area_local,
                                  double da_min_c, double area_c_local) {
  StabilityReport r;
  const size_t n = c.nord_k.size();
  if (n == 0) return r;

  /* On the authoritative gnomonic cube da_min is the global minimum cell
   * area, so these ratios are <= 1 and a caller who cannot supply a local
   * area gets the conservative value 1.0. */
  const double ar = (area_local > 0.0) ? (da_min / area_local) : 1.0;
  const double ar_c = (area_c_local > 0.0) ? (da_min_c / area_c_local) : 1.0;

  r.level.resize(n);
  for (size_t k = 0; k < n; ++k) {
    LevelStability s;
    s.k = (int)k;

    /* The del-2n divergence arm is taken only when nord_k > 0. On the
     * nord_k == 0 branch d_sw never reads d4_bg, so reporting a gamma for it
     * would be reporting on an unread number. */
    s.d4_active = (c.nord_k[k] > 0);
    if (s.d4_active) {
      s.gamma_d4 = gamma_del2n(c.d4_bg[k], c.nord_k[k], ar);
    }

    /* The additive del-2 term. sw_core.F90:1418 applies
     * vort = damp2*delpc + dd8*divg_d with
     * damp2 = da_min_c*max(d2_bg, min(0.20, dddmp*|div*dt|)), so it is
     * present on BOTH branches. Static: Smagorinsky = 0. Worst case:
     * Smagorinsky saturated at its own 0.20 cap, which needs |div*dt| >= 0.20
     * / dddmp and is therefore a property of the flow, not the config. */
    s.gamma_d2_static = 8.0 * c.d2_bg[k] * ar_c;
    s.gamma_d2_worst = 8.0 * std::max(c.d2_bg[k], std::min(0.20, c.dddmp > 0.0 ? 0.20 : 0.0)) * ar_c;

    s.gamma_div_static = s.gamma_d4 + s.gamma_d2_static;
    s.gamma_div_worst = s.gamma_d4 + s.gamma_d2_worst;

    /* Analogous-by-construction norms; see the header's calibration note. */
    if (c.damp_v[k] > kDampVGate) s.gamma_v = gamma_del2n(c.damp_v[k], c.ndif[k], ar_c);
    if (c.damp_w[k] > kDampVGate) s.gamma_w = gamma_del2n(c.damp_w[k], c.nord_w[k], ar_c);
    if (c.damp_t[k] > kDampTGate) s.gamma_t = gamma_del2n(c.damp_t[k], c.nord_t, ar);

    s.unstable_static = (s.gamma_div_static > kGammaLimit);
    s.unstable_worst_case = (!s.unstable_static) && (s.gamma_div_worst > kGammaLimit);
    s.unstable_vorticity = (s.gamma_v > kGammaLimit);
    s.unstable_w = (s.gamma_w > kGammaLimit);
    s.unstable_scalar = (s.gamma_t > kGammaLimit);

    if (s.unstable_static) r.any_unstable_static = true;
    if (s.unstable_worst_case) r.any_unstable_worst_case = true;
    if (s.unstable_vorticity || s.unstable_w || s.unstable_scalar)
      r.any_unstable_other = true;

    if (r.worst_level < 0 || s.gamma_div_static > r.worst_gamma) {
      r.worst_gamma = s.gamma_div_static;
      r.worst_level = (int)k;
    }
    r.level[k] = s;
  }
  return r;
}

std::string format_damping_column(const DampingColumn& c) {
  std::string out;
  char buf[512];
  std::snprintf(buf, sizeof(buf),
      "[FV3-DAMP] per-level damping from the authoritative translation "
      "(dyn_core.F90:677-765); nord_t=%d dddmp=%.6g d4dmp_profile=%s\n",
      c.nord_t, c.dddmp, c.d4dmp_profile_active ? "sine" : "constant");
  out += buf;
  std::snprintf(buf, sizeof(buf),
      "[FV3-DAMP]   k nord_k ndif nord_w      d2_bg      d4_bg     damp_v"
      "     damp_w     damp_t      d_con        kgb      d_ext\n");
  out += buf;
  for (size_t k = 0; k < c.nord_k.size(); ++k) {
    std::snprintf(buf, sizeof(buf),
        "[FV3-DAMP] %3zu %6d %4d %6d %10.6f %10.6f %10.6f %10.6f %10.6f"
        " %10.6f %10.3e %10.6f\n",
        k, c.nord_k[k], c.ndif[k], c.nord_w[k], c.d2_bg[k], c.d4_bg[k],
        c.damp_v[k], c.damp_w[k], c.damp_t[k], c.d_con[k], c.kgb[k],
        c.d_ext[k]);
    out += buf;
  }
  return out;
}

std::string format_stability_report(const DampingColumn& c,
                                    const StabilityReport& r) {
  std::string out;
  char buf[640];
  std::snprintf(buf, sizeof(buf),
      "[FV3-DAMP] del-2n stability, gamma = (8*coeff)^(nord+1)*(da_min/area),"
      " unstable above %.1f\n", kGammaLimit);
  out += buf;
  std::snprintf(buf, sizeof(buf),
      "[FV3-DAMP]   gamma_d4/d2/div are measurement-calibrated (3A.11V);"
      " gamma_v/w/t are analogous-by-construction, NOT threshold-measured\n");
  out += buf;
  std::snprintf(buf, sizeof(buf),
      "[FV3-DAMP]   k d4? gamma_d4 gd2_stat gd2_wrst gdiv_stat gdiv_wrst"
      "  gamma_v  gamma_w  gamma_t  flag\n");
  out += buf;
  for (const LevelStability& s : r.level) {
    const char* flag = "ok";
    if (s.unstable_static) flag = "UNSTABLE-BY-CONFIG";
    else if (s.unstable_worst_case) flag = "unstable-if-divergent-flow";
    else if (s.unstable_vorticity) flag = "UNSTABLE-VORTICITY";
    else if (s.unstable_w) flag = "UNSTABLE-W";
    else if (s.unstable_scalar) flag = "UNSTABLE-SCALAR";
    std::snprintf(buf, sizeof(buf),
        "[FV3-DAMP] %3d %3s %8.4f %8.4f %8.4f %9.4f %9.4f %8.4f %8.4f %8.4f"
        "  %s\n",
        s.k, s.d4_active ? "yes" : "no", s.gamma_d4, s.gamma_d2_static,
        s.gamma_d2_worst, s.gamma_div_static, s.gamma_div_worst, s.gamma_v,
        s.gamma_w, s.gamma_t, flag);
    out += buf;
  }
  if (r.any_unstable_static) {
    std::snprintf(buf, sizeof(buf),
        "[FV3-DAMP] VERDICT: UNSTABLE BY CONFIGURATION. Worst level k=%d,"
        " nord_k=%d, d4_bg=%.6g, d2_bg=%.6g, gamma=%.4f > %.1f. The"
        " authoritative Fortran does not clip these coefficients, so neither"
        " does this runtime: the numerics are unchanged and the run will"
        " amplify. Reduce d4_bg_top or nord, or raise the sponge depth.\n",
        r.worst_level, c.nord_k[(size_t)r.worst_level],
        c.d4_bg[(size_t)r.worst_level], c.d2_bg[(size_t)r.worst_level],
        r.worst_gamma, kGammaLimit);
  } else if (r.any_unstable_worst_case) {
    std::snprintf(buf, sizeof(buf),
        "[FV3-DAMP] VERDICT: stable from configuration alone (worst"
        " gamma_div_static = %.4f at k=%d), but the additive Smagorinsky term"
        " at sw_core.F90:1418 can push at least one level past %.1f in a"
        " strongly divergent flow (needs |div*dt| >= %.3g with dddmp=%.6g).\n",
        r.worst_gamma, r.worst_level, kGammaLimit,
        c.dddmp > 0.0 ? 0.20 / c.dddmp : 0.0, c.dddmp);
  } else if (r.any_unstable_other) {
    std::snprintf(buf, sizeof(buf),
        "[FV3-DAMP] VERDICT: divergence damping stable (worst"
        " gamma_div_static = %.4f at k=%d) but a vorticity/w/scalar arm"
        " exceeds %.1f; see the per-level flags above.\n",
        r.worst_gamma, r.worst_level, kGammaLimit);
  } else {
    std::snprintf(buf, sizeof(buf),
        "[FV3-DAMP] VERDICT: STABLE. Worst gamma_div_static = %.4f at k=%d,"
        " margin %.2fx.\n",
        r.worst_gamma, r.worst_level,
        r.worst_gamma > 0.0 ? kGammaLimit / r.worst_gamma : 0.0);
  }
  out += buf;
  return out;
}

}  // namespace fv3::gpu::damping
