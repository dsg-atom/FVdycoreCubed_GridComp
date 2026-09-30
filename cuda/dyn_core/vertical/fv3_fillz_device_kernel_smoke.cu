//===========================================================================
// WP-C fv_fill_mod fillz -- dedicated device-kernel GPU smoke test (Prompt
// 88B, Stage 6). fillz_column<T> (fv3_fillz_column.hpp) is already
// bitwise-validated on the HOST against the Prompt 88A fillz_host() oracle
// (fv3_wp_c_fillz_cuda_vs_host, Stage 5). This file GPU-EXECUTES the exact
// same templated function -- not a reimplementation -- from a __global__
// kernel, one thread per column, to confirm the identical code compiles and
// runs correctly as an actual CUDA device function (not just __host__
// under g++), before trusting its use inside remap_q_kernel
// (fv3_remap_q.cu).
//
// Rationale for testing fillz in isolation here, rather than only through
// the full REMAP_Q pipeline (see fv3_remap_q_nq2_fill_true_prod_shape_smoke
// .cu, this file's sibling): bounded probing (3 smooth production-like
// tracer profiles + 3 profiles with a deliberately pre-existing negative
// layer, all under the real kord=9 monotonic map_scalar_column path on the
// same C32-shaped pe1/pe2 grid used elsewhere in this project) found that
// kord=9's monotonic/extm constraint resolves every one of those synthetic
// negatives BEFORE fillz would ever see them -- i.e. fillz's actual
// borrow/rescale branches are not organically reachable from realistic
// positive-definite tracer profiles in a small synthetic column. This
// matches fv_fill.F90's own framing of fillz as a belt-and-suspenders
// safety net (Prompt 88A), not a routinely-active code path. Reusing Prompt
// 88A's adversarial post-remap fixtures directly (bypassing map_scalar_
// column) is therefore the bounded, representative way to GPU-execute
// fillz's actual corrective branches -- top-layer borrow, interior
// borrow-above/borrow-below, bottom-layer borrow, and the zfix-gated
// sum0>0/sum0<=0 non-local rescale-or-inert paths -- rather than continuing
// an open-ended search for a naturally-triggering PPM input.
//
// Numerical semantics changed relative to the Fortran: NO.
//===========================================================================

#include "fv3_fillz_column.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using fv3::gpu::vertical::fillz_column;

namespace {

#define CUDA_CHECK(x)                                                        \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(_e));                                  \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

constexpr int kMaxLevels = 32;

// One thread per column. Flat batch layout: column c's q/dp/scratch occupy
// [c*km .. c*km+km-1].
__global__ void fillz_smoke_kernel(double *q, const double *dp, int km,
                                    double *dm_scratch, int ncols) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncols) return;
  if (km > kMaxLevels) return;
  fillz_column<double>(q + c * km, dp + c * km, km, dm_scratch + c * km);
}

// Mechanical port of fv_fill.F90's fillz -- identical oracle to Prompt
// 88A's fillz_host() / this task's test_fillz_cuda_vs_host.cpp, reproduced
// standalone (this is a .cu smoke binary, not linked against the host test
// executable).
void fillz_host(int km, std::vector<double> &q, const std::vector<double> &dp) {
  if (q[0] < 0.0) {
    q[1] += q[0] * dp[0] / dp[1];
    q[0] = 0.0;
  }
  bool zfix = false;
  for (int k = 1; k <= km - 2; ++k) {
    if (q[k] < 0.0) {
      zfix = true;
      if (q[k - 1] > 0.0) {
        double dq = std::min(q[k - 1] * dp[k - 1], -q[k] * dp[k]);
        q[k - 1] -= dq / dp[k - 1];
        q[k] += dq / dp[k];
      }
      if (q[k] < 0.0 && q[k + 1] > 0.0) {
        double dq = std::min(q[k + 1] * dp[k + 1], -q[k] * dp[k]);
        q[k + 1] -= dq / dp[k + 1];
        q[k] += dq / dp[k];
      }
    }
  }
  {
    const int k = km - 1;
    if (q[k] < 0.0 && q[k - 1] > 0.0) {
      zfix = true;
      double qup = q[k - 1] * dp[k - 1];
      double qly = -q[k] * dp[k];
      double dup = std::min(qly, qup);
      q[k - 1] -= dup / dp[k - 1];
      q[k] += dup / dp[k];
    }
  }
  if (zfix) {
    double sum0 = 0.0;
    std::vector<double> dm(km, 0.0);
    for (int k = 1; k <= km - 1; ++k) {
      dm[k] = q[k] * dp[k];
      sum0 += dm[k];
    }
    if (sum0 > 0.0) {
      double sum1 = 0.0;
      for (int k = 1; k <= km - 1; ++k) sum1 += std::max(0.0, dm[k]);
      double fac = sum0 / sum1;
      for (int k = 1; k <= km - 1; ++k) q[k] = std::max(0.0, fac * dm[k] / dp[k]);
    }
  }
}

struct Case {
  const char *name;
  std::vector<double> q;
  std::vector<double> dp;
};

} // namespace

int main() {
  const int km = 10;
  const std::vector<double> dp(km, 900.0);

  std::vector<Case> cases;

  {
    std::vector<double> q(km);
    for (int k = 0; k < km; ++k) q[k] = 1.0e-2 * std::exp(-0.3 * k);
    cases.push_back({"no_negatives_is_noop", q, dp});
  }
  cases.push_back({"near_minimum_q_no_trigger", std::vector<double>(km, 1.0e-30), dp});
  {
    std::vector<double> q(km, 1.0e-2);
    q[0] = -2.0e-3;
    cases.push_back({"negative_top_layer", q, dp});
  }
  {
    std::vector<double> q(km, 1.0e-2);
    q[4] = -1.0e-3;
    cases.push_back({"negative_interior_borrow_above", q, dp});
  }
  {
    std::vector<double> q(km, 1.0e-2);
    q[3] = 0.0;
    q[4] = -1.0e-3;
    cases.push_back({"negative_interior_borrow_below", q, dp});
  }
  {
    std::vector<double> q(km, 1.0e-2);
    q[km - 1] = -2.0e-3;
    cases.push_back({"negative_bottom_layer", q, dp});
  }
  {
    std::vector<double> q(km, 1.2e-2);
    q[3] = -0.02;
    q[4] = -0.02;
    q[5] = -0.02;
    cases.push_back({"deep_cluster_nonlocal_normalize", q, dp});
  }
  {
    std::vector<double> q(km, 0.5e-3);
    q[3] = -5.0;
    q[4] = -5.0;
    q[5] = -5.0;
    cases.push_back({"sum0_negative_fix_inert_by_design", q, dp});
  }
  {
    std::vector<double> q(km);
    for (int k = 0; k < km; ++k) q[k] = 1.0e-2 * std::exp(-0.35 * k);
    cases.push_back({"batched_nq2_tracer0_sphum", q, dp});
  }
  {
    std::vector<double> q(km, 0.05);
    q[5] = -0.01;
    cases.push_back({"batched_nq2_tracer1_cld_amt", q, dp});
  }

  const int ncols = static_cast<int>(cases.size());
  std::vector<double> q_h(ncols * km), dp_h(ncols * km);
  std::vector<std::vector<double>> oracle(ncols);
  for (int c = 0; c < ncols; ++c) {
    for (int k = 0; k < km; ++k) {
      q_h[c * km + k] = cases[c].q[k];
      dp_h[c * km + k] = cases[c].dp[k];
    }
    oracle[c] = cases[c].q;
    fillz_host(km, oracle[c], cases[c].dp);
  }

  double *d_q, *d_dp, *d_dm;
  CUDA_CHECK(cudaMalloc(&d_q, q_h.size() * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dp, dp_h.size() * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dm, q_h.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_q, q_h.data(), q_h.size() * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dp, dp_h.data(), dp_h.size() * sizeof(double), cudaMemcpyHostToDevice));

  const int threads = 32;
  const int blocks = (ncols + threads - 1) / threads;
  fillz_smoke_kernel<<<blocks, threads>>>(d_q, d_dp, km, d_dm, ncols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> q_out(q_h.size());
  CUDA_CHECK(cudaMemcpy(q_out.data(), d_q, q_h.size() * sizeof(double), cudaMemcpyDeviceToHost));

  int nfail = 0;
  for (int c = 0; c < ncols; ++c) {
    bool bitwise = true;
    int first_diff_k = -1;
    for (int k = 0; k < km; ++k) {
      if (q_out[c * km + k] != oracle[c][k]) {
        bitwise = false;
        if (first_diff_k < 0) first_diff_k = k;
      }
    }
    std::printf("[%s] GPU-EXECUTED bitwise=%s%s\n", cases[c].name,
                bitwise ? "YES" : "NO", bitwise ? "" : " MISMATCH");
    if (!bitwise) {
      std::printf("  first_diff_k=%d oracle=%.17e device=%.17e\n", first_diff_k,
                  oracle[c][first_diff_k], q_out[c * km + first_diff_k]);
      ++nfail;
    }
  }

  std::printf("FILLZ_DEVICE_KERNEL_SMOKE: %d/%d cases failed\n", nfail, ncols);

  cudaFree(d_q);
  cudaFree(d_dp);
  cudaFree(d_dm);
  return nfail == 0 ? 0 : 1;
}
