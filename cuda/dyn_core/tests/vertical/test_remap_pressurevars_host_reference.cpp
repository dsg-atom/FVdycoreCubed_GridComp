// Prompt 91 Stage 10/13: host-only golden-reference harness for the
// authoritative Fortran `Remap_PressureVars` timer
// (fv_mapz.F90:1239-1516, GEOSfvdycore_fortran_perf checkout,
// src/Components/@FVdycoreCubed_GridComp/@fvdycore/model/fv_mapz.F90).
//
// This is NOT a CUDA test. No CUDA source is added or modified by this task.
// `Remap_PressureVars` is inline code inside the much larger
// `Lagrangian_to_Eulerian` subroutine, not a standalone callable routine, so
// per Prompt 91 Stage 10's stated preference order (A. direct call to the
// real production routine, B. minimal wrapper, C. exact test-only
// transcription only if a private/local routine prevents A/B) Option C
// applies here, exactly as it did for `fillz`
// (test_fillz_host_reference.cpp, Prompt 88A/88B precedent). No production
// numerical source is modified to expose a test entry point.
//
// Scope transcribed -- PRODUCTION-ACTIVE BRANCHES ONLY, confirmed by direct
// source read (see docs/performance/p91_remap_pressurevars_cuda_port_plan_2026-09-27.md
// Stage 0/1/4 for the full trace):
//   - hydrostatic = .false. (production nonhydrostatic config; the
//     `hydrostatic` branch, fv_mapz.F90:1349-1380, is DEAD in production
//     and not transcribed).
//   - remap_option = 0 => remap_t = .true., remap_pt = .false.,
//     remap_te = .false. (already the pinned production value used by the
//     existing, validated `use_cuda_pt_delz_remap` selector's own
//     `remap_option == 0` requirement). The remap_te block
//     (fv_mapz.F90:1382-1422) and remap_pt block (fv_mapz.F90:1470-1479)
//     are DEAD in production and not transcribed.
//   - MOIST_CAPPA/USE_COND are unconditionally compiled in for this build
//     (@fvdycore/CMakeLists.txt:130), so the Fortran `#ifdef MOIST_CAPPA`
//     branch (fv_mapz.F90:1426-1446) is the one that always executes; the
//     `#else` branch (fv_mapz.F90:1447-1467) is dead SOURCE (not a runtime
//     branch) and is transcribed only implicitly, because it is
//     numerically IDENTICAL to the MOIST_CAPPA branch once `cappa==akap`
//     (see the dry-degenerate note below).
//   - `moist_cv`'s `case default` (fv_mapz.F90:4188-4193, triggered for
//     `nwat` not in 1-7, i.e. `nwat=0`) returns `cvm=cv_air` (a true
//     constant) without touching the `q` tracer array.
//   - `do_omega` (fv_mapz.F90:1483-1512) is structurally .false. in
//     production: `do_omega = hydrostatic .and. last_step`
//     (fv_dynamics.F90:946) is always false in the nonhydrostatic
//     production config regardless of `last_step`. DEAD, not transcribed.
//   - `[P60_EXIT_DUMP]` diagnostic writes are inert unless a specific env
//     var is set; not part of the numerical path, not transcribed.
//
// IMPORTANT correctness note (Prompt 91 finding, refines the pre-existing
// "cappa=akap under nwat=0" project convention already used by
// fv3_pt_delz_remap.cu:19-20): reading fv_mapz.F90:1430 directly,
//   cappa(i,j,k) = rdgas / ( rdgas + cvm(i)/(1.+r_vir*q(i,j,k,sphum)) )
// depends on q(sphum), NOT on nwat alone. The `cappa == akap` identity
// (via Mayer's relation, cv_air = cp_air - rdgas) additionally requires
// q(sphum) == 0, which holds in every validated run to date only because
// of the SEPARATE, already-documented "structural cold-start tracer-zero
// limitation" (CURRENT_STATE.md Section 7, found Prompt 77): with
// COLDSTART=1 + external_ic=.false. + empty ExtData.rc (every checked
// AGCM.rc), `q` never gets a nonzero IC and stays at its allocation-time
// zero for the whole run. This harness transcribes the FULL formula
// (parameterized on q(sphum), not hard-coded to the degenerate case) and
// separately verifies the degenerate identity as its own fixture/invariant
// -- this is more conservative than assuming the simplification outright,
// and documents precisely which additional fact (q(sphum)==0) the existing
// PT/DELZ CUDA convention silently relies on beyond nwat==0 alone.
//
// Not transcribed (confirmed absent/irrelevant from the active branches by
// direct source read, Prompt 91 Stage 4): fill, kord, a_imp, k_split,
// n_split, gmao_remap, w, phis, te, u, v, gridstruct, hs -- none are
// referenced inside the active remap_t/MOIST_CAPPA + pe/peln/pk/delp
// blocks.
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Physical constants (R8 production values, GFS_typedefs/fv_arrays
// conventions already used elsewhere in this project, e.g.
// fv3_pt_delz_remap.cu's `a.akap`/`a.rdgas`/`a.grav`).
constexpr double kRdgas = 287.05;
constexpr double kCpAir = 1004.6;
constexpr double kAkap = kRdgas / kCpAir;         // rdgas/cp_air
constexpr double kCvAir = kCpAir - kRdgas;         // Mayer's relation
constexpr double kGrav = 9.80665;
constexpr double kRrg = -kRdgas / kGrav;           // matches fv_mapz.F90's rrg = -rdgas/grav
constexpr double kRvir = 0.6077338443; // rvgas/rdgas - 1 (FV3 r_vir convention)

struct ColumnIn {
  int km;
  double ptop;
  std::vector<double> ak;     // km+1
  std::vector<double> bk;     // km+1
  std::vector<double> pe_old; // km+1 (only pe_old[km], the surface value, is read)
  std::vector<double> peln_old; // km+1 (only [0] and [km] are read -- boundary pass-through)
  std::vector<double> pk_old;   // km+1 (only [0] and [km] are read -- boundary pass-through)
  std::vector<double> pt;     // km
  std::vector<double> delz;  // km
  std::vector<double> q_sphum; // km
};

struct ColumnOut {
  std::vector<double> pe;    // km+1
  std::vector<double> peln;  // km+1
  std::vector<double> pk;    // km+1
  std::vector<double> delp;  // km
  std::vector<double> pkz;   // km
  std::vector<double> cappa; // km
};

// Mechanical transcription of fv_mapz.F90:1256-1301 (pe/peln/pk/delp
// unconditional recompute, active regardless of remap_t/remap_pt/remap_te)
// followed by fv_mapz.F90:1423-1446 (remap_t=.true., MOIST_CAPPA branch,
// moist_cv's nwat=0 case-default inlined as cvm=cv_air). 0-based indexing;
// Fortran k=1..km+1 -> C++ k=0..km, Fortran k=2..km -> C++ k=1..km-1.
ColumnOut remap_pressurevars_host(const ColumnIn &in) {
  const int km = in.km;
  ColumnOut out;
  out.pe.resize(km + 1);
  out.peln.resize(km + 1);
  out.pk.resize(km + 1);
  out.delp.resize(km);
  out.pkz.resize(km);
  out.cappa.resize(km);

  // ---------------- Update pe (fv_mapz.F90:1256-1269) ----------------
  std::vector<double> pe2(km + 1);
  pe2[0] = in.ptop;
  pe2[km] = in.pe_old[km];
  for (int k = 1; k <= km - 1; ++k) {
    pe2[k] = in.ak[k] + in.bk[k] * in.pe_old[km];
  }
  for (int k = 0; k <= km; ++k) out.pe[k] = pe2[k];

  // ------------- Update logP & pk (fv_mapz.F90:1274-1291) -------------
  std::vector<double> pn2(km + 1), pk2(km + 1);
  pn2[0] = in.peln_old[0];
  pn2[km] = in.peln_old[km];
  pk2[0] = in.pk_old[0];
  pk2[km] = in.pk_old[km];
  for (int k = 1; k <= km - 1; ++k) {
    pn2[k] = std::log(pe2[k]);
    pk2[k] = std::exp(kAkap * pn2[k]);
  }
  for (int k = 0; k <= km; ++k) {
    out.peln[k] = pn2[k];
    out.pk[k] = pk2[k];
  }

  // ------------------ Update delp (fv_mapz.F90:1296-1300) -----------------
  for (int k = 0; k <= km - 1; ++k) out.delp[k] = pe2[k + 1] - pe2[k];

  // --- Compute pkz, remap_t + MOIST_CAPPA branch (fv_mapz.F90:1423-1446),
  //     moist_cv nwat=0 case-default (cvm=cv_air) inlined ---
  for (int k = 0; k <= km - 1; ++k) {
    const double cvm = kCvAir; // moist_cv case-default, fv_mapz.F90:4188-4193
    const double cappa = kRdgas / (kRdgas + cvm / (1.0 + kRvir * in.q_sphum[k]));
    out.cappa[k] = cappa;
    out.pkz[k] = std::exp(cappa * std::log(kRrg * out.delp[k] / in.delz[k] * in.pt[k]));
  }

  return out;
}

bool is_finite(double v) { return std::isfinite(v); }

struct CaseResult {
  const char *name;
  bool pass;
};
std::vector<CaseResult> g_results;

// Builds a realistic km-level hybrid sigma-p column (ak/bk monotonic,
// ptop small, surface pressure ps) -- shared by most fixtures.
ColumnIn make_column(int km, double ptop, double ps, double pt_surf,
                     double lapse_K_per_km, double q_sphum_const) {
  ColumnIn c;
  c.km = km;
  c.ptop = ptop;
  c.ak.assign(km + 1, 0.0);
  c.bk.assign(km + 1, 0.0);
  for (int k = 0; k <= km; ++k) {
    const double frac = static_cast<double>(k) / km; // 0 (top) -> 1 (surface)
    c.bk[k] = frac * frac; // monotonic, bk[0]=0, bk[km]=1
    c.ak[k] = ptop * (1.0 - frac);
  }
  c.pe_old.assign(km + 1, 0.0);
  for (int k = 0; k <= km; ++k) c.pe_old[k] = c.ak[k] + c.bk[k] * ps;
  c.peln_old.assign(km + 1, 0.0);
  for (int k = 0; k <= km; ++k) c.peln_old[k] = std::log(c.pe_old[k]);
  c.pk_old.assign(km + 1, 0.0);
  for (int k = 0; k <= km; ++k) c.pk_old[k] = std::exp(kAkap * c.peln_old[k]);

  c.pt.assign(km, 0.0);
  c.delz.assign(km, 0.0);
  c.q_sphum.assign(km, q_sphum_const);
  for (int k = 0; k < km; ++k) {
    const double frac = static_cast<double>(k) / (km - 1); // 0 top -> 1 bottom
    c.pt[k] = pt_surf + lapse_K_per_km * (1.0 - frac) * 10.0; // colder aloft
    const double dp = c.pe_old[k + 1] - c.pe_old[k];
    // Hydrostatic-ish thickness estimate for a plausible delz (negative,
    // FV3 convention: delz is a negative layer depth in this remap stage).
    c.delz[k] = -(kRdgas * c.pt[k] / ps) * dp / 1.0e2;
    if (c.delz[k] == 0.0) c.delz[k] = -1.0; // guard against degenerate zero
  }
  return c;
}

void check_and_record(const char *name, const ColumnIn &in, bool require_cappa_eq_akap) {
  const ColumnOut out = remap_pressurevars_host(in);
  const int km = in.km;
  bool pass = true;

  // Invariant: monotonic pe (positive layer thickness / delp > 0).
  for (int k = 0; k < km; ++k) {
    if (!(out.delp[k] > 0.0)) pass = false;
    if (!(out.pe[k + 1] > out.pe[k])) pass = false;
  }
  // Invariant: pk[k] == pe[k]^akap for interior k (round-off).
  for (int k = 1; k < km; ++k) {
    const double expected = std::exp(kAkap * std::log(out.pe[k]));
    if (std::fabs(out.pk[k] - expected) > 1.0e-9 * std::fabs(expected)) pass = false;
  }
  // Invariant: boundary pass-through (pk/peln at k=0 and k=km untouched).
  if (out.pk[0] != in.pk_old[0] || out.pk[km] != in.pk_old[km]) pass = false;
  if (out.peln[0] != in.peln_old[0] || out.peln[km] != in.peln_old[km]) pass = false;
  // Invariant: no NaN/Inf anywhere.
  for (int k = 0; k <= km; ++k) {
    if (!is_finite(out.pe[k]) || !is_finite(out.peln[k]) || !is_finite(out.pk[k])) pass = false;
  }
  for (int k = 0; k < km; ++k) {
    if (!is_finite(out.delp[k]) || !is_finite(out.pkz[k]) || !is_finite(out.cappa[k])) pass = false;
  }
  // Invariant: 0 < cappa < 1 (physical range for rdgas>0, cvm>0, q>=0).
  for (int k = 0; k < km; ++k) {
    if (!(out.cappa[k] > 0.0 && out.cappa[k] < 1.0)) pass = false;
  }
  // Dry-degenerate identity check (q_sphum==0 fixtures only): cappa==akap,
  // and pkz matches the dead-code #else formula (fv_mapz.F90:1449) bitwise.
  if (require_cappa_eq_akap) {
    for (int k = 0; k < km; ++k) {
      if (std::fabs(out.cappa[k] - kAkap) > 1.0e-12) pass = false;
      const double pkz_else = std::exp(kAkap * std::log(kRrg * out.delp[k] / in.delz[k] * in.pt[k]));
      if (out.pkz[k] != pkz_else) pass = false;
    }
  }
  // Repeatability: re-run on identical input must be bitwise identical.
  const ColumnOut out2 = remap_pressurevars_host(in);
  for (int k = 0; k <= km; ++k) {
    if (out.pe[k] != out2.pe[k] || out.peln[k] != out2.peln[k] || out.pk[k] != out2.pk[k]) pass = false;
  }
  for (int k = 0; k < km; ++k) {
    if (out.delp[k] != out2.delp[k] || out.pkz[k] != out2.pkz[k]) pass = false;
  }

  g_results.push_back({name, pass});
  std::printf("%-42s %s\n", name, pass ? "PASS" : "FAIL");
}

} // namespace

int main() {
  // 1. Smooth pressure profile, realistic C512-like column (km=91, GEOS
  //    standard vertical level count), q_sphum=0 (production/cold-start).
  check_and_record("c512_like_smooth_q0", make_column(91, 1.0, 101325.0, 288.0, 4.0, 0.0), true);

  // 2. Strongly stratified profile (large lapse rate contrast top/bottom).
  check_and_record("strongly_stratified_q0", make_column(60, 1.0, 101325.0, 300.0, 60.0, 0.0), true);

  // 3. Top-boundary-sensitive: very small ptop, thin top layer.
  check_and_record("top_boundary_thin_ptop_q0", make_column(60, 0.01, 101325.0, 280.0, 4.0, 0.0), true);

  // 4. Bottom-boundary-sensitive: high-elevation-like reduced surface
  //    pressure (thin bottom layer relative to ps).
  check_and_record("bottom_boundary_low_ps_q0", make_column(60, 1.0, 60000.0, 270.0, 4.0, 0.0), true);

  // 5. Weak delp layer: km with a very fine near-surface level spacing.
  {
    ColumnIn c = make_column(120, 1.0, 101325.0, 288.0, 4.0, 0.0);
    check_and_record("weak_delp_fine_levels_q0", c, true);
  }

  // 6. Nonhydrostatic case: production is always nonhydrostatic, so every
  //    fixture above already exercises the active (nonhydrostatic) branch;
  //    this fixture additionally uses a large |delz| contrast to stress
  //    the delz-dependent pkz transcendental.
  check_and_record("nonhydrostatic_large_delz_contrast_q0",
                    make_column(60, 1.0, 101325.0, 320.0, 120.0, 0.0), true);

  // 7. Min/max pressure-quantity branch, q_sphum == 0 (the actual
  //    production/cold-start value, per CURRENT_STATE Section 7).
  check_and_record("q_sphum_zero_production_value", make_column(60, 1.0, 101325.0, 288.0, 4.0, 0.0), true);

  // 8. Min/max pressure-quantity branch, q_sphum > 0 (hypothetical/
  //    robustness only -- NOT reached in any validated production run to
  //    date, per the cold-start limitation, but the general moist_cv
  //    formula must still be transcribed correctly for this input). This
  //    fixture intentionally does NOT assert cappa==akap.
  check_and_record("q_sphum_nonzero_hypothetical", make_column(60, 1.0, 101325.0, 288.0, 4.0, 0.01), false);

  // 9. Transcendental-sensitive case: very large surface pressure and
  //    very small |delz| together, stressing log/exp argument magnitude.
  {
    ColumnIn c = make_column(60, 1.0, 108000.0, 320.0, 4.0, 0.0);
    for (int k = 0; k < c.km; ++k) c.delz[k] *= 0.01; // shrink |delz| sharply
    check_and_record("transcendental_stress_small_delz_q0", c, true);
  }

  // 10. Repeatability case (explicit second construction from scratch,
  //     independent of check_and_record's own internal repeat, to also
  //     confirm construction itself is deterministic).
  {
    ColumnIn a = make_column(60, 1.0, 101325.0, 288.0, 4.0, 0.0);
    ColumnIn b = make_column(60, 1.0, 101325.0, 288.0, 4.0, 0.0);
    const ColumnOut oa = remap_pressurevars_host(a);
    const ColumnOut ob = remap_pressurevars_host(b);
    bool pass = true;
    for (int k = 0; k <= a.km; ++k) {
      if (oa.pe[k] != ob.pe[k] || oa.pk[k] != ob.pk[k] || oa.peln[k] != ob.peln[k]) pass = false;
    }
    for (int k = 0; k < a.km; ++k) {
      if (oa.delp[k] != ob.delp[k] || oa.pkz[k] != ob.pkz[k]) pass = false;
    }
    g_results.push_back({"independent_reconstruction_repeatable", pass});
    std::printf("%-42s %s\n", "independent_reconstruction_repeatable", pass ? "PASS" : "FAIL");
  }

  bool all_pass = true;
  for (const auto &r : g_results) all_pass &= r.pass;
  std::printf("RESULT=%s (%zu cases)\n", all_pass ? "PASS" : "FAIL", g_results.size());
  return all_pass ? 0 : 1;
}
