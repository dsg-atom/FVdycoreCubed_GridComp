// Prompt 88B Stage 5: host-side bitwise comparison of the new device-portable
// fillz_column<T> helper (cuda/dyn_core/vertical/fv3_fillz_column.hpp)
// against the Prompt 88A golden-reference fillz_host() oracle
// (test_fillz_host_reference.cpp). Not a CUDA test -- fillz_column<T> is
// __host__ __device__ and this file compiles/runs it purely on the host,
// exercising the exact same code path a device thread would run (the
// FV3_FILLZ_HD macro only adds CUDA qualifiers when __CUDACC__ is defined,
// which it is not for a plain g++ compile).
//
// Uses all 9 of Prompt 88A's fixtures, reproduced here (not #included --
// test_fillz_host_reference.cpp is itself a standalone main()) so each case
// can be run through BOTH implementations and diffed bit-for-bit.
#include "../../vertical/fv3_fillz_column.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using fv3::gpu::vertical::fillz_column;

// Mechanical port of fv_fill.F90's fillz, non-DEV_GFS_PHYS branch -- same
// oracle as test_fillz_host_reference.cpp's fillz_host(), reproduced here
// verbatim so this file has no build-system dependency on that test binary.
void fillz_host(int km, int nq, std::vector<std::vector<double>> &q,
                 const std::vector<double> &dp) {
  for (int ic = 0; ic < nq; ++ic) {
    std::vector<double> &qc = q[ic];
    if (qc[0] < 0.0) {
      qc[1] += qc[0] * dp[0] / dp[1];
      qc[0] = 0.0;
    }
    bool zfix = false;
    for (int k = 1; k <= km - 2; ++k) {
      if (qc[k] < 0.0) {
        zfix = true;
        if (qc[k - 1] > 0.0) {
          double dq = std::min(qc[k - 1] * dp[k - 1], -qc[k] * dp[k]);
          qc[k - 1] -= dq / dp[k - 1];
          qc[k] += dq / dp[k];
        }
        if (qc[k] < 0.0 && qc[k + 1] > 0.0) {
          double dq = std::min(qc[k + 1] * dp[k + 1], -qc[k] * dp[k]);
          qc[k + 1] -= dq / dp[k + 1];
          qc[k] += dq / dp[k];
        }
      }
    }
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

struct CaseResult {
  const char *name;
  bool bitwise;
  int first_diff_k;
};

std::vector<CaseResult> g_results;

// Runs one column through both fillz_host (oracle) and fillz_column<double>
// (device helper, host-compiled), then diffs bit-for-bit.
void run_case(const char *name, std::vector<double> q_in,
              const std::vector<double> &dp) {
  const int km = static_cast<int>(q_in.size());

  std::vector<std::vector<double>> oracle_q = {q_in};
  fillz_host(km, 1, oracle_q, dp);

  std::vector<double> device_q = q_in;
  std::vector<double> dm_scratch(km, 0.0);
  fillz_column<double>(device_q.data(), dp.data(), km, dm_scratch.data());

  int first_diff_k = -1;
  for (int k = 0; k < km; ++k) {
    if (oracle_q[0][k] != device_q[k]) {
      first_diff_k = k;
      break;
    }
  }
  const bool bitwise = (first_diff_k < 0);
  g_results.push_back({name, bitwise, first_diff_k});

  std::printf("%-38s bitwise=%s%s\n", name, bitwise ? "YES" : "NO",
              bitwise ? "" : " (first differing k reported above)");
  if (!bitwise) {
    std::printf("  first_diff_k=%d oracle=%.17e device=%.17e\n", first_diff_k,
                oracle_q[0][first_diff_k], device_q[first_diff_k]);
  }
}

}  // namespace

int main() {
  const int km = 10;
  std::vector<double> dp(km, 900.0);

  // 1. No negatives -- true no-op for both implementations.
  {
    std::vector<double> q(km);
    for (int k = 0; k < km; ++k) q[k] = 1.0e-2 * std::exp(-0.3 * k);
    run_case("no_negatives_is_noop", q, dp);
  }

  // 2. Near-minimum q, no trigger.
  { run_case("near_minimum_q_no_trigger", std::vector<double>(km, 1.0e-30), dp); }

  // 3. Negative top layer.
  {
    std::vector<double> q(km, 1.0e-2);
    q[0] = -2.0e-3;
    run_case("negative_top_layer", q, dp);
  }

  // 4. Interior negative, borrow-from-above path.
  {
    std::vector<double> q(km, 1.0e-2);
    q[4] = -1.0e-3;
    run_case("negative_interior_borrow_above", q, dp);
  }

  // 5. Interior negative, borrow-from-below path.
  {
    std::vector<double> q(km, 1.0e-2);
    q[3] = 0.0;
    q[4] = -1.0e-3;
    run_case("negative_interior_borrow_below", q, dp);
  }

  // 6. Bottom-layer negative.
  {
    std::vector<double> q(km, 1.0e-2);
    q[km - 1] = -2.0e-3;
    run_case("negative_bottom_layer", q, dp);
  }

  // 7. Deep cluster, sum0>0 -- exercises the non-local rescale-and-clip.
  {
    std::vector<double> q(km, 1.2e-2);
    q[3] = -0.02;
    q[4] = -0.02;
    q[5] = -0.02;
    run_case("deep_cluster_nonlocal_normalize", q, dp);
  }

  // 7b. Deep cluster, sum0<=0 -- rescale must NOT fire; residual negative is
  //     legal/expected in BOTH implementations (see fv3_fillz_column.hpp's
  //     header comment -- must not be "improved").
  {
    std::vector<double> q(km, 0.5e-3);
    q[3] = -5.0;
    q[4] = -5.0;
    q[5] = -5.0;
    run_case("sum0_negative_fix_inert_by_design", q, dp);
  }

  // 8. Batched nq=2 (sphum-like clean, cld_amt-like with an interior
  //    negative) -- fillz_column<T> is single-tracer, so this drives it
  //    twice, mirroring mapn_tracer's own do-iq=1,nq loop and REMAP_Q's own
  //    two-sequential-launches structure (one call per tracer).
  {
    std::vector<double> sphum(km);
    for (int k = 0; k < km; ++k) sphum[k] = 1.0e-2 * std::exp(-0.35 * k);
    run_case("batched_nq2_tracer0_sphum", sphum, dp);

    std::vector<double> cld_amt(km, 0.05);
    cld_amt[5] = -0.01;
    run_case("batched_nq2_tracer1_cld_amt", cld_amt, dp);
  }

  bool all_bitwise = true;
  for (const auto &r : g_results) all_bitwise &= r.bitwise;
  std::printf("RESULT=%s (%zu cases, bitwise=%s)\n",
              all_bitwise ? "PASS" : "FAIL", g_results.size(),
              all_bitwise ? "YES" : "NO");
  return all_bitwise ? 0 : 1;
}
