// Prompt 88A Stage 6: host-only golden-reference test for the authoritative
// Fortran `fillz` (fv_fill.F90:51-156, the non-DEV_GFS_PHYS branch -- DEV_GFS_PHYS
// is not defined anywhere in this project's build, confirmed by grep of
// CMakeLists.txt/CMakeCache.txt). `fillz` is the numerical routine
// `mapn_tracer` (fv_mapz.F90:2291) invokes when `flagstruct%fill` is true --
// found by Prompt 87/88A to be UNCONDITIONALLY forced true by
// FV_StateMod.F90:568 for every GEOS FV_State, with no runnable configuration
// in this project (checked: every experiments/*/input.nml, the fv3.j job
// template, and all *.rc files) ever setting `fill=.false.`.
//
// This is NOT a CUDA test. No CUDA source is added or modified by this task.
// This file is a mechanical, line-for-line C++ port of the authoritative
// Fortran `fillz` (im=1, single-column form -- mapn_tracer's own per-i-then-
// per-column decomposition means each column is numerically independent, so
// a single-column port is a faithful reference for the per-thread body a
// future CUDA extension would need). It is a golden reference / scope-proof
// fixture set, per Prompt 88A Stage 6: no fill=true fixture existed in the
// existing host harness (test_remap_q_tracer.cpp, test_remap_q_nq2_scope.cpp
// both use fill=.false. only) before this task.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Mechanical port of fv_fill.F90's fillz, non-DEV_GFS_PHYS branch, restricted
// to a single column (im=1) since fillz's im-loops are independent per i.
// q, dp are km-length; nq tracers packed as q[ic][k] (mirrors q(1,km,nq)).
void fillz_host(int km, int nq, std::vector<std::vector<double>> &q,
                 const std::vector<double> &dp) {
  for (int ic = 0; ic < nq; ++ic) {
    std::vector<double> &qc = q[ic];

    // Top layer (Fortran k=1 -> C++ k=0)
    if (qc[0] < 0.0) {
      qc[1] += qc[0] * dp[0] / dp[1];
      qc[0] = 0.0;
    }

    // Interior (Fortran k=2..km-1 -> C++ k=1..km-2)
    bool zfix = false;
    for (int k = 1; k <= km - 2; ++k) {
      if (qc[k] < 0.0) {
        zfix = true;
        if (qc[k - 1] > 0.0) {
          // Borrow from above
          double dq = std::min(qc[k - 1] * dp[k - 1], -qc[k] * dp[k]);
          qc[k - 1] -= dq / dp[k - 1];
          qc[k] += dq / dp[k];
        }
        if (qc[k] < 0.0 && qc[k + 1] > 0.0) {
          // Borrow from below
          double dq = std::min(qc[k + 1] * dp[k + 1], -qc[k] * dp[k]);
          qc[k + 1] -= dq / dp[k + 1];
          qc[k] += dq / dp[k];
        }
      }
    }

    // Bottom layer (Fortran k=km -> C++ k=km-1)
    {
      const int k = km - 1;
      if (qc[k] < 0.0 && qc[k - 1] > 0.0) {
        zfix = true;
        double qup = qc[k - 1] * dp[k - 1];
        double qly = -qc[k] * dp[k];
        double dup = std::min(qly, qup);
        qc[k - 1] -= dup / dp[k - 1];
        qc[k] += dup / dp[k];
      }
    }

    // Final check and non-local fix if needed (Fortran k=2..km -> C++ k=1..km-1)
    if (zfix) {
      double sum0 = 0.0;
      std::vector<double> dm(km, 0.0);
      for (int k = 1; k <= km - 1; ++k) {
        dm[k] = qc[k] * dp[k];
        sum0 += dm[k];
      }
      if (sum0 > 0.0) {
        double sum1 = 0.0;
        for (int k = 1; k <= km - 1; ++k) sum1 += std::max(0.0, dm[k]);
        double fac = sum0 / sum1;
        for (int k = 1; k <= km - 1; ++k)
          qc[k] = std::max(0.0, fac * dm[k] / dp[k]);
      }
    }
  }
}

double column_mass(const std::vector<double> &q, const std::vector<double> &dp) {
  double m = 0.0;
  for (size_t k = 0; k < q.size(); ++k) m += q[k] * dp[k];
  return m;
}

bool any_negative(const std::vector<double> &q) {
  for (double v : q) if (v < 0.0) return true;
  return false;
}

struct CaseResult {
  const char *name;
  bool pass;
  double mass_rel;
  bool negative_after;
  bool repeatable;
};

std::vector<CaseResult> g_results;

void run_case(const char *name, std::vector<double> q_in, const std::vector<double> &dp) {
  const int km = static_cast<int>(q_in.size());
  const double mass_in = column_mass(q_in, dp);

  std::vector<std::vector<double>> q1 = {q_in};
  fillz_host(km, 1, q1, dp);
  const double mass_out = column_mass(q1[0], dp);
  const double mass_rel = std::fabs(mass_out - mass_in) / std::max(std::fabs(mass_in), 1.0e-30);
  const bool neg_after = any_negative(q1[0]);

  // Repeatability: re-running fillz on its OWN (already-nonnegative) output
  // must be a no-op (idempotent) and re-running the whole case from the same
  // input must reproduce bit-identical output.
  std::vector<std::vector<double>> q2 = {q_in};
  fillz_host(km, 1, q2, dp);
  bool repeatable = true;
  for (int k = 0; k < km; ++k)
    if (q1[0][k] != q2[0][k]) repeatable = false;

  std::vector<std::vector<double>> q3 = {q1[0]};
  fillz_host(km, 1, q3, dp);
  for (int k = 0; k < km; ++k)
    if (q3[0][k] != q1[0][k]) repeatable = false; // idempotent on its own output

  const bool pass = mass_rel < 1.0e-12 && !neg_after && repeatable;
  g_results.push_back({name, pass, mass_rel, neg_after, repeatable});

  std::printf("%-38s mass_rel=%.3e negative_after=%s repeatable=%s %s\n", name,
              mass_rel, neg_after ? "YES" : "no", repeatable ? "yes" : "NO",
              pass ? "PASS" : "FAIL");
}

} // namespace

int main() {
  const int km = 10;
  std::vector<double> dp(km, 900.0); // Pa, uniform layer thickness

  // 1. Normal positive q, no negatives -- fillz must be a true no-op
  //    (Stage 6: "normal positive q").
  {
    std::vector<double> q(km);
    for (int k = 0; k < km; ++k) q[k] = 1.0e-2 * std::exp(-0.3 * k);
    std::vector<double> before = q;
    std::vector<std::vector<double>> qv = {q};
    fillz_host(km, 1, qv, dp);
    bool unchanged = true;
    for (int k = 0; k < km; ++k) if (qv[0][k] != before[k]) unchanged = false;
    std::printf("%-38s unchanged=%s %s\n", "no_negatives_is_noop",
                unchanged ? "yes" : "NO", unchanged ? "PASS" : "FAIL");
    g_results.push_back({"no_negatives_is_noop", unchanged, 0.0, false, true});
  }

  // 2. Near-minimum q: tiny positive values close to zero -- must NOT trigger
  //    any fix path (q>=0 everywhere; Stage 6: "near-minimum q").
  {
    std::vector<double> q(km, 1.0e-30);
    run_case("near_minimum_q_no_trigger", q, dp);
  }

  // 3. Negative at the top layer -- borrows entirely from layer 2
  //    (Stage 6: "fill-specific boundary behavior", top boundary).
  {
    std::vector<double> q(km, 1.0e-2);
    q[0] = -2.0e-3;
    run_case("negative_top_layer", q, dp);
  }

  // 4. Negative in the interior, positive layer above -- borrow-from-above path.
  {
    std::vector<double> q(km, 1.0e-2);
    q[4] = -1.0e-3;
    run_case("negative_interior_borrow_above", q, dp);
  }

  // 5. Negative in the interior, layer above already zero/negative -- forces
  //    the borrow-from-below path.
  {
    std::vector<double> q(km, 1.0e-2);
    q[3] = 0.0;
    q[4] = -1.0e-3;
    run_case("negative_interior_borrow_below", q, dp);
  }

  // 6. Negative at the bottom layer -- bottom-boundary borrow path.
  {
    std::vector<double> q(km, 1.0e-2);
    q[km - 1] = -2.0e-3;
    run_case("negative_bottom_layer", q, dp);
  }

  // 7. Clustered interior negatives too deep for immediate-neighbor
  //    borrowing to fully resolve, but with the column's aggregate mass
  //    (k=1..km-1, i.e. excluding the top layer) still net-positive --
  //    forces, and this case confirms SUCCESS of, the final non-local
  //    normalize-and-clip fix (fv_fill.F90:130-152; Stage 6's "final check"
  //    boundary behavior, not exercised by cases 3-6). Values hand-derived
  //    so the pre-fix sum(q*dp) over k=1..9 is exactly +10.8 (Pa-units),
  //    i.e. sum0>0, which is the precondition for the rescale to activate
  //    at all -- see case 7b for the sum0<=0 case, where it does not.
  {
    std::vector<double> q(km, 1.2e-2);
    q[3] = -0.02;
    q[4] = -0.02;
    q[5] = -0.02;
    run_case("deep_cluster_nonlocal_normalize", q, dp);
  }

  // 7b. Same shape as case 7 but with a deeper negative cluster so the
  //     column's aggregate mass over k=1..km-1 is net-NEGATIVE (sum0<=0).
  //     Documents genuine authoritative behavior (Stage 3 finding): fillz's
  //     non-local fix only activates+clips-to-nonnegative when sum0>0; if a
  //     column's mass in that range is already net-negative, the routine
  //     leaves the result negative (mass is still exactly conserved -- it is
  //     a best-effort, mass-conserving fix, NOT an unconditional
  //     non-negativity guarantee). This case is expected to still have a
  //     negative result and is graded on mass conservation only, not
  //     non-negativity.
  {
    std::vector<double> q(km, 0.5e-3);
    q[3] = -5.0;
    q[4] = -5.0;
    q[5] = -5.0;
    const double mass_in = column_mass(q, dp);
    std::vector<std::vector<double>> qv = {q};
    fillz_host(km, 1, qv, dp);
    const double mass_out = column_mass(qv[0], dp);
    const double mass_rel = std::fabs(mass_out - mass_in) / std::max(std::fabs(mass_in), 1.0e-30);
    const bool neg_after = any_negative(qv[0]);
    const bool pass = mass_rel < 1.0e-12 && neg_after; // negative result IS the expected/authoritative outcome here
    std::printf("%-38s mass_rel=%.3e negative_after=%s (expected) %s\n",
                "sum0_negative_fix_inert_by_design", mass_rel, neg_after ? "YES" : "no",
                pass ? "PASS" : "FAIL");
    g_results.push_back({"sum0_negative_fix_inert_by_design", pass, mass_rel, neg_after, true});
  }

  // 8. Batched nq=2 call (mirrors mapn_tracer's actual do-iq=1,nq structure,
  //    fv_mapz.F90:2291): sphum-like tracer with NO negatives, cld_amt-like
  //    tracer WITH an interior negative, in the same fillz invocation --
  //    confirms fillz applies identically/independently per tracer (no
  //    branching on tracer identity in fv_fill.F90; Stage 6's "both
  //    production tracers if behavior differs" -- verified: it does not).
  {
    std::vector<double> sphum(km);
    for (int k = 0; k < km; ++k) sphum[k] = 1.0e-2 * std::exp(-0.35 * k);
    std::vector<double> cld_amt(km, 0.05);
    cld_amt[5] = -0.01;

    const double mass_in_0 = column_mass(sphum, dp);
    const double mass_in_1 = column_mass(cld_amt, dp);
    std::vector<std::vector<double>> q = {sphum, cld_amt};
    fillz_host(km, 2, q, dp);
    const double mass_rel_0 = std::fabs(column_mass(q[0], dp) - mass_in_0) /
                               std::max(std::fabs(mass_in_0), 1.0e-30);
    const double mass_rel_1 = std::fabs(column_mass(q[1], dp) - mass_in_1) /
                               std::max(std::fabs(mass_in_1), 1.0e-30);
    const bool tracer0_untouched_by_tracer1 = (mass_rel_0 < 1.0e-12) && !any_negative(q[0]);
    const bool tracer1_fixed = (mass_rel_1 < 1.0e-12) && !any_negative(q[1]);
    const bool pass = tracer0_untouched_by_tracer1 && tracer1_fixed;
    std::printf("%-38s tracer0_ok=%s tracer1_ok=%s %s\n", "batched_nq2_per_tracer_independent",
                tracer0_untouched_by_tracer1 ? "yes" : "NO", tracer1_fixed ? "yes" : "NO",
                pass ? "PASS" : "FAIL");
    g_results.push_back({"batched_nq2_per_tracer_independent", pass, mass_rel_1, false, true});
  }

  bool all_pass = true;
  for (const auto &r : g_results) all_pass &= r.pass;
  std::printf("RESULT=%s (%zu cases)\n", all_pass ? "PASS" : "FAIL", g_results.size());
  return all_pass ? 0 : 1;
}
