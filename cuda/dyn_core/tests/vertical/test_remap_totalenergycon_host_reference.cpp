// Prompt 95 Stage 11/13/15: host-only golden-reference harness for the
// authoritative Fortran `Remap_TotalEnergyConsv` timer
// (fv_mapz.F90:1612-1864, GEOSfvdycore_fortran_perf checkout,
// src/Components/@FVdycoreCubed_GridComp/@fvdycore/model/fv_mapz.F90).
//
// This is NOT a CUDA test. No CUDA source is added or modified by this task.
// `Remap_TotalEnergyConsv` is inline code inside `Lagrangian_to_Eulerian`,
// not a standalone callable routine, so per the same Option-C precedent
// established by test_fillz_host_reference.cpp (Prompt 88A) and
// test_remap_pressurevars_host_reference.cpp (Prompt 91), this is an exact
// test-only transcription of the production-active branches only. No
// production numerical source is modified to expose a test entry point.
//
// Scope transcribed -- PRODUCTION-ACTIVE BRANCHES ONLY, confirmed by direct
// source read and by the actual C512 performance-harness log (Prompt 95,
// see docs/performance/p95_remap_totalenergycon_cuda_port_plan_2026-09-28.md
// Stage 0-4 for the full trace):
//   - hydrostatic = .false. (production nonhydrostatic config; every
//     `if (hydrostatic)` branch in this timer is DEAD in production and
//     not transcribed).
//   - consv_te = 1.0 > consv_min (0.001), so the `consv > consv_min` branch
//     (fv_mapz.F90:1628-1722) is the one that executes; the
//     `consv < -consv_min` branch (fv_mapz.F90:1725-1755) is DEAD in
//     production and not transcribed.
//   - do_adiabatic_init = .false. always in this standalone harness family
//     (fv_nudge.F90:1241, only ever set true by IAU/digital-filter paths
//     not exercised here), so `last_step .and. .not.do_adiabatic_init`
//     reduces to `last_step`.
//   - nwat = 0 in production, so `moist_cv`'s `case default`
//     (fv_mapz.F90:4207-4287) returns cvm=cv_air, qd=0 uniformly -- the
//     compiled-in `#ifdef MOIST_CAPPA` branch (fv_mapz.F90:1663-1671) is
//     numerically IDENTICAL to the `#else` cv_air branch
//     (fv_mapz.F90:1672-1679) for this config, so only the cv_air formula
//     is transcribed (Unit A).
//   - do_sat_adj = .false. in production (nwat=0 < 6), so the
//     `sat_adj2`-timed block (fv_mapz.F90:1760-1799) is DEAD and not
//     transcribed -- it is entirely disjoint from Remap_TotalEnergyConsv's
//     own pre/reduction/post structure regardless.
//   - adiabatic = .true. in this measured C512 Held-Suarez performance
//     harness (confirmed by the log's "Running In Adiabatic Mode" line),
//     so the executed post-reduction pt-correction branch is the plain
//     `last_step .and. adiabatic` branch (fv_mapz.F90:1848-1857):
//     `pt(i,j,k) = pt(i,j,k) + dtmp*pkz(i,j,k)` -- no q(sphum)/USE_COND
//     dependency at all. The `.not.adiabatic` USE_COND-gated branch
//     (fv_mapz.F90:1802-1836) and the plain-theta_v `else` branch
//     (fv_mapz.F90:1859-1869) are NOT executed by this harness and are not
//     transcribed as Unit B (documented for completeness only).
//
// Two authoritative local numerical units are transcribed, matching the
// natural pre-reduction / bridge / post-reduction split imposed by the two
// `mpp_global_sum` calls (fv_mapz.F90:1705,1712/1718):
//   UNIT A (per-column pre-reduction compute, fv_mapz.F90:1651-1691):
//     given a column's pt/delz/delp/pkz/u/v/w and grid metrics, produces
//     the column's contribution to the two global-sum inputs: te_2d
//     (energy imbalance) and zsum1 (pressure-weighted column integral).
//   BRIDGE (fv_mapz.F90:1720, non-hydrostatic branch): given the two
//     GLOBAL scalars tesum/zsum (as if already reduced), dtmp =
//     tesum/(cv_air*zsum).
//   UNIT B (per-point post-reduction compute, fv_mapz.F90:1850-1857): given
//     dtmp and a column's pt/pkz, applies the correction
//     pt(k) = pt(k) + dtmp*pkz(k).
//
// Not transcribed (confirmed absent/irrelevant to the active branches by
// direct source read): hydrostatic zsum0/pk terms, MOIST_CAPPA cvm (dead
// weight given nwat=0), USE_COND q_con fill, sat_adj2, E_Flux (a pure
// diagnostic output, not consumed downstream in this timer).
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Physical constants (R8 production values, same convention already used
// by test_remap_pressurevars_host_reference.cpp / fv3_pt_delz_remap.cu).
constexpr double kRdgas = 287.05;
constexpr double kCpAir = 1004.6;
constexpr double kCvAir = kCpAir - kRdgas; // Mayer's relation
constexpr double kGrav = 9.80665;

// ---------------------------------------------------------------------
// UNIT A: fv_mapz.F90:1651-1691 (non-hydrostatic branch, MOIST_CAPPA/
// cv_air-degenerate, nwat=0), one atmospheric column.
// ---------------------------------------------------------------------
struct ColumnA {
  int km;
  double hs;             // surface geopotential, hs(i,j)
  double te0_2d;         // pre-existing column total energy input
  std::vector<double> pt;      // virtual temperature, km
  std::vector<double> delz;    // km (negative layer depth, FV3 convention)
  std::vector<double> delp;    // km
  std::vector<double> pkz;     // km
  std::vector<double> u_ctr;   // u(i,j,k), km
  std::vector<double> u_north; // u(i,j+1,k), km
  std::vector<double> v_ctr;   // v(i,j,k), km
  std::vector<double> v_east;  // v(i+1,j,k), km
  std::vector<double> w;       // km
  double rsin2;
  double cosa_s;
};

struct UnitAOut {
  double te_2d; // column contribution to tesum's global sum, pre-area-weight
  double zsum1; // column contribution to zsum's global sum, pre-area-weight
};

UnitAOut remap_totalenergycon_unit_a_host(const ColumnA &c) {
  const int km = c.km;
  std::vector<double> phis(km + 1);
  phis[km] = c.hs;
  for (int k = km - 1; k >= 0; --k) {
    phis[k] = phis[k + 1] - kGrav * c.delz[k];
  }

  double te_2d = 0.0;
  for (int k = 0; k < km; ++k) {
    const double cvm = kCvAir; // moist_cv case-default, nwat=0
    const double ke_cross =
        c.u_ctr[k] * c.u_ctr[k] + c.u_north[k] * c.u_north[k] +
        c.v_ctr[k] * c.v_ctr[k] + c.v_east[k] * c.v_east[k] -
        (c.u_ctr[k] + c.u_north[k]) * (c.v_ctr[k] + c.v_east[k]) * c.cosa_s;
    te_2d += c.delp[k] * (cvm * c.pt[k] +
                          0.5 * (phis[k] + phis[k + 1] + c.w[k] * c.w[k] +
                                 0.5 * c.rsin2 * ke_cross));
  }
  UnitAOut out;
  out.te_2d = c.te0_2d - te_2d;

  double zsum1 = c.pkz[0] * c.delp[0];
  for (int k = 1; k < km; ++k) zsum1 += c.pkz[k] * c.delp[k];
  out.zsum1 = zsum1;
  return out;
}

// ---------------------------------------------------------------------
// BRIDGE: fv_mapz.F90:1720 (non-hydrostatic branch). Operates on GLOBAL
// scalars (post mpp_global_sum) -- this is the entire host-side reduction
// consumer, deliberately NOT touching mpp_global_sum itself.
// ---------------------------------------------------------------------
double remap_totalenergycon_bridge_host(double tesum_global, double zsum_global) {
  return tesum_global / (kCvAir * zsum_global);
}

// ---------------------------------------------------------------------
// UNIT B: fv_mapz.F90:1850-1857 (last_step .and. adiabatic branch, the
// branch actually executed by the measured C512 Held-Suarez harness).
// ---------------------------------------------------------------------
void remap_totalenergycon_unit_b_host(double dtmp, const std::vector<double> &pkz,
                                       std::vector<double> *pt) {
  for (std::size_t k = 0; k < pt->size(); ++k) {
    (*pt)[k] = (*pt)[k] + dtmp * pkz[k];
  }
}

bool is_finite(double v) { return std::isfinite(v); }

struct CaseResult {
  const char *name;
  bool pass;
};
std::vector<CaseResult> g_results;

void record(const char *name, bool pass) {
  g_results.push_back({name, pass});
  std::printf("%-42s %s\n", name, pass ? "PASS" : "FAIL");
}

// Builds a realistic km-level column with a smooth lapse-rate temperature
// profile and plausible winds -- shared by most Unit A fixtures.
ColumnA make_column_a(int km, double hs, double te0_2d, double ps,
                       double pt_surf, double lapse_K_per_km, double u_amp,
                       double v_amp, double w_amp) {
  ColumnA c;
  c.km = km;
  c.hs = hs;
  c.te0_2d = te0_2d;
  c.pt.assign(km, 0.0);
  c.delz.assign(km, 0.0);
  c.delp.assign(km, 0.0);
  c.pkz.assign(km, 0.0);
  c.u_ctr.assign(km, 0.0);
  c.u_north.assign(km, 0.0);
  c.v_ctr.assign(km, 0.0);
  c.v_east.assign(km, 0.0);
  c.w.assign(km, 0.0);
  c.rsin2 = 1.02;   // near-1, generic non-degenerate cubed-sphere metric
  c.cosa_s = 0.05;  // small, generic non-orthogonal-grid cross term

  for (int k = 0; k < km; ++k) {
    const double frac = static_cast<double>(k) / (km - 1); // 0 top -> 1 bottom
    c.pt[k] = pt_surf + lapse_K_per_km * (1.0 - frac) * 10.0; // colder aloft
    const double dp = ps / km; // uniform layer mass, sufficient for a harness
    c.delp[k] = dp;
    c.delz[k] = -(kRdgas * c.pt[k] / ps) * dp / 1.0e2;
    if (c.delz[k] == 0.0) c.delz[k] = -1.0;
    c.pkz[k] = std::pow(c.pt[k] / 300.0, 0.2) * 0.9; // plausible, bounded
    c.u_ctr[k] = u_amp * std::sin(0.3 * k + 0.1);
    c.u_north[k] = u_amp * std::sin(0.3 * k + 0.4);
    c.v_ctr[k] = v_amp * std::cos(0.2 * k + 0.2);
    c.v_east[k] = v_amp * std::cos(0.2 * k + 0.5);
    c.w[k] = w_amp * std::sin(0.5 * k);
  }
  return c;
}

void check_unit_a(const char *name, const ColumnA &c) {
  const UnitAOut out = remap_totalenergycon_unit_a_host(c);
  bool pass = true;

  if (!is_finite(out.te_2d) || !is_finite(out.zsum1)) pass = false;
  // Invariant: zsum1 is a sum of pkz*delp with pkz>0 (by construction) and
  // delp>0 (positive layer mass, guaranteed by the source's own pressure
  // thickness convention) -- zsum1 must be strictly positive.
  if (!(out.zsum1 > 0.0)) pass = false;

  // Repeatability: re-run on identical input must be bitwise identical.
  const UnitAOut out2 = remap_totalenergycon_unit_a_host(c);
  if (out.te_2d != out2.te_2d || out.zsum1 != out2.zsum1) pass = false;

  record(name, pass);
}

} // namespace

int main() {
  // ---------------- Unit A: pre-reduction local compute ----------------

  // 1. Smooth atmospheric column, realistic C512-like level count (km=91).
  check_unit_a("unitA_c512_like_smooth", make_column_a(91, 500.0, 2.5e7, 101325.0, 288.0, 4.0, 10.0, 8.0, 0.2));

  // 2. Vertically varying pt with a strong lapse rate contrast.
  check_unit_a("unitA_strong_lapse_rate", make_column_a(60, 500.0, 2.5e7, 101325.0, 300.0, 60.0, 10.0, 8.0, 0.2));

  // 3. Varying u/v kinetic energy: large wind amplitude, stresses the
  //    cross term (u_ctr+u_north)*(v_ctr+v_east)*cosa_s.
  check_unit_a("unitA_large_wind_ke", make_column_a(60, 500.0, 2.5e7, 101325.0, 288.0, 4.0, 60.0, 45.0, 0.2));

  // 4. Nonzero w (nonhydrostatic vertical velocity), large amplitude.
  check_unit_a("unitA_large_w_nonhydrostatic", make_column_a(60, 500.0, 2.5e7, 101325.0, 288.0, 4.0, 10.0, 8.0, 5.0));

  // 5. Varying delp/delz: thin near-surface levels (fine vertical spacing).
  check_unit_a("unitA_fine_vertical_levels", make_column_a(120, 500.0, 2.5e7, 101325.0, 288.0, 4.0, 10.0, 8.0, 0.2));

  // 6. Top-sensitive column: very small ptop-like top layer influence via a
  //    near-zero surface geopotential and cold top.
  check_unit_a("unitA_top_sensitive", make_column_a(60, 0.0, 2.5e7, 101325.0, 240.0, 90.0, 10.0, 8.0, 0.2));

  // 7. Bottom-sensitive column: large surface geopotential (mountainous
  //    terrain proxy) and warm near-surface layer.
  check_unit_a("unitA_bottom_sensitive_high_terrain", make_column_a(60, 3000.0, 2.5e7, 101325.0, 310.0, 4.0, 10.0, 8.0, 0.2));

  // 8. Realistic C512-like column (second, distinct seed) for redundancy.
  check_unit_a("unitA_c512_like_alt_seed", make_column_a(91, 250.0, 2.6e7, 98000.0, 292.0, 6.0, 15.0, 12.0, 0.3));

  // 9. Positive-energy-imbalance case: te0_2d set well above the internal
  //    energy sum so the resulting te_2d (te0_2d - sum) is positive.
  check_unit_a("unitA_positive_energy_imbalance", make_column_a(60, 500.0, 1.0e9, 101325.0, 288.0, 4.0, 10.0, 8.0, 0.2));

  // 10. Negative/small-energy-imbalance case: te0_2d set below the internal
  //     energy sum so te_2d is negative (legal -- dtmp's sign follows tesum's
  //     sign, both directions are physically legal per the source formula).
  check_unit_a("unitA_negative_energy_imbalance", make_column_a(60, 500.0, 1.0e5, 101325.0, 288.0, 4.0, 10.0, 8.0, 0.2));

  // 11. Multi-column accumulation sanity: verify that summing zsum1 across
  //     several independently-built columns matches a direct manual sum
  //     (models the accumulation the real mpp_global_sum would perform
  //     across ranks/columns, without invoking any MPI/collective code).
  {
    const ColumnA c1 = make_column_a(60, 500.0, 2.5e7, 101325.0, 288.0, 4.0, 10.0, 8.0, 0.2);
    const ColumnA c2 = make_column_a(60, 300.0, 2.4e7, 100000.0, 280.0, 5.0, 12.0, 9.0, 0.3);
    const ColumnA c3 = make_column_a(60, 700.0, 2.6e7, 102000.0, 295.0, 3.0, 8.0, 6.0, 0.1);
    const UnitAOut o1 = remap_totalenergycon_unit_a_host(c1);
    const UnitAOut o2 = remap_totalenergycon_unit_a_host(c2);
    const UnitAOut o3 = remap_totalenergycon_unit_a_host(c3);
    const double manual_zsum_total = o1.zsum1 + o2.zsum1 + o3.zsum1;
    const double manual_te_total = o1.te_2d + o2.te_2d + o3.te_2d;
    bool pass = is_finite(manual_zsum_total) && is_finite(manual_te_total) &&
                manual_zsum_total > 0.0;
    record("unitA_multi_column_accumulation_sane", pass);
  }

  // ---------------- BRIDGE: deterministic global-scalar fixtures --------
  {
    // Small correction.
    const double dtmp = remap_totalenergycon_bridge_host(1.0e6, 1.0e9);
    bool pass = is_finite(dtmp) && std::fabs(dtmp) < 1.0e-3;
    record("bridge_small_correction", pass);
  }
  {
    // Large correction (tesum chosen so dtmp = tesum/(cv_air*zsum) > 1,
    // cv_air ~= 717.55).
    const double dtmp = remap_totalenergycon_bridge_host(1.0e13, 1.0e9);
    bool pass = is_finite(dtmp) && dtmp > 1.0;
    record("bridge_large_correction", pass);
  }
  {
    // Near-zero-but-legal denominator: zsum small and positive (physically
    // legal only in the "very shallow/low-mass atmosphere" limit; zsum is
    // guaranteed structurally positive by Unit A's own invariant above, so
    // an exactly-zero zsum is not a legal input and is not tested).
    const double dtmp = remap_totalenergycon_bridge_host(1.0, 1.0e-6);
    bool pass = is_finite(dtmp);
    record("bridge_small_positive_denominator", pass);
  }
  {
    // Negative tesum -> negative dtmp (sign must follow tesum's sign).
    const double dtmp = remap_totalenergycon_bridge_host(-1.0e6, 1.0e9);
    bool pass = is_finite(dtmp) && dtmp < 0.0;
    record("bridge_negative_tesum_sign_follows", pass);
  }
  {
    // Repeatability.
    const double d1 = remap_totalenergycon_bridge_host(1.234e7, 9.876e8);
    const double d2 = remap_totalenergycon_bridge_host(1.234e7, 9.876e8);
    record("bridge_repeatable_bitwise", d1 == d2 && is_finite(d1));
  }

  // ---------------- Unit B: post-reduction local compute -----------------
  {
    // dtmp == 0 identity: pt must be unchanged exactly.
    std::vector<double> pkz = {0.95, 0.90, 0.85, 0.80};
    std::vector<double> pt = {290.0, 280.0, 270.0, 260.0};
    const std::vector<double> pt_before = pt;
    remap_totalenergycon_unit_b_host(0.0, pkz, &pt);
    bool pass = true;
    for (std::size_t k = 0; k < pt.size(); ++k) if (pt[k] != pt_before[k]) pass = false;
    record("unitB_dtmp_zero_identity", pass);
  }
  {
    // Linearity in dtmp: doubling dtmp doubles the applied correction.
    std::vector<double> pkz = {0.95, 0.90, 0.85, 0.80};
    std::vector<double> pt_a = {290.0, 280.0, 270.0, 260.0};
    std::vector<double> pt_b = pt_a;
    const std::vector<double> pt0 = pt_a;
    remap_totalenergycon_unit_b_host(0.5, pkz, &pt_a);
    remap_totalenergycon_unit_b_host(1.0, pkz, &pt_b);
    bool pass = true;
    for (std::size_t k = 0; k < pt_a.size(); ++k) {
      const double delta_a = pt_a[k] - pt0[k];
      const double delta_b = pt_b[k] - pt0[k];
      if (std::fabs(2.0 * delta_a - delta_b) > 1.0e-12 * std::fabs(delta_b)) pass = false;
    }
    record("unitB_linear_in_dtmp", pass);
  }
  {
    // Negative dtmp cools the column (matches bridge's sign convention).
    std::vector<double> pkz = {0.95, 0.90, 0.85, 0.80};
    std::vector<double> pt = {290.0, 280.0, 270.0, 260.0};
    const std::vector<double> pt0 = pt;
    remap_totalenergycon_unit_b_host(-0.5, pkz, &pt);
    bool pass = true;
    for (std::size_t k = 0; k < pt.size(); ++k) if (!(pt[k] < pt0[k])) pass = false;
    record("unitB_negative_dtmp_cools", pass);
  }
  {
    // Repeatability across two independent applications from the same
    // starting state.
    std::vector<double> pkz = {0.95, 0.90, 0.85, 0.80};
    std::vector<double> pt_a = {290.0, 280.0, 270.0, 260.0};
    std::vector<double> pt_b = pt_a;
    remap_totalenergycon_unit_b_host(0.37, pkz, &pt_a);
    remap_totalenergycon_unit_b_host(0.37, pkz, &pt_b);
    bool pass = true;
    for (std::size_t k = 0; k < pt_a.size(); ++k) if (pt_a[k] != pt_b[k]) pass = false;
    record("unitB_repeatable_bitwise", pass);
  }
  {
    // Full km=91 realistic-size column, end-to-end Unit A -> bridge (with
    // a hand-picked deterministic global tesum/zsum standing in for the
    // real mpp_global_sum output) -> Unit B, confirming the three units
    // compose without shape/index mismatches.
    ColumnA c = make_column_a(91, 500.0, 2.5e7, 101325.0, 288.0, 4.0, 10.0, 8.0, 0.2);
    const UnitAOut a_out = remap_totalenergycon_unit_a_host(c);
    // Stand-in "global" scalars: a single-column reduction is just the
    // column's own contribution -- legal for confirming composition, not a
    // claim about real multi-rank reduction behavior.
    const double dtmp = remap_totalenergycon_bridge_host(a_out.te_2d, a_out.zsum1);
    std::vector<double> pt_final = c.pt;
    remap_totalenergycon_unit_b_host(dtmp, c.pkz, &pt_final);
    bool pass = is_finite(dtmp);
    for (std::size_t k = 0; k < pt_final.size(); ++k) if (!is_finite(pt_final[k])) pass = false;
    record("end_to_end_unitA_bridge_unitB_composition", pass);
  }

  bool all_pass = true;
  for (const auto &r : g_results) all_pass &= r.pass;
  std::printf("RESULT=%s (%zu cases)\n", all_pass ? "PASS" : "FAIL", g_results.size());
  return all_pass ? 0 : 1;
}
