// Prompt 76 Stage 1: dedicated unit test for the device pert_ppm(iv=1)
// helper (dgrid/wp_b_ppm_shared.cuh::pert_ppm_iv1), independent of the
// hord==8 face-flux fixtures. Five named cases per the task's Stage 1
// requirement: monotonic profile, local extremum, sign change, boundary
// case, near-flat case. Bitwise comparison against an independent host
// transcription of tp_core.F90:1238-1252 (iv/=0 branch).

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "dgrid/wp_b_ppm_shared.cuh"

using fv3::gpu::fv_real;

// Independent host transcription (not a call into the device header) of
// tp_core.F90:1238-1252, iv/=0 branch.
static void h_pert_ppm_iv1(fv_real &al, fv_real &ar) {
  if (al * ar < fv_real(0)) {
    const fv_real da1 = al - ar;
    const fv_real da2 = da1 * da1;
    const fv_real a6da = fv_real(3) * (al + ar) * da1;
    if (a6da < -da2) {
      ar = fv_real(-2) * al;
    } else if (a6da > da2) {
      al = fv_real(-2) * ar;
    }
  } else {
    al = fv_real(0);
    ar = fv_real(0);
  }
}

__global__ void pert_ppm_iv1_kernel(const fv_real *al_in, const fv_real *ar_in,
                                    fv_real *al_out, fv_real *ar_out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  fv_real al = al_in[i], ar = ar_in[i];
  fv3::gpu::dgrid::pert_ppm_iv1(al, ar);
  al_out[i] = al;
  ar_out[i] = ar;
}

struct Case {
  const char *name;
  fv_real al, ar;
};

int main() {
  // Five named fixtures per Stage 1:
  static const Case cases[] = {
      // monotonic profile: al, ar same sign, product >= 0 -> al=ar=0 branch
      {"monotonic", fv_real(0.4), fv_real(0.6)},
      // local extremum: al, ar opposite sign, |a6da| small vs da2 -> no clamp
      {"local_extremum", fv_real(0.05), fv_real(-0.06)},
      // sign change: opposite sign, a6da very negative -> ar = -2*al clamp
      {"sign_change", fv_real(1.0), fv_real(-0.01)},
      // boundary case: al*ar exactly zero (one side flat)
      {"boundary", fv_real(0.0), fv_real(0.3)},
      // near-flat: both tiny, opposite sign, near da2 threshold
      {"near_flat", fv_real(1.0e-7), fv_real(-1.0e-7)},
  };
  const int n = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

  fv_real *h_al_in = new fv_real[n], *h_ar_in = new fv_real[n];
  fv_real *h_al_ref = new fv_real[n], *h_ar_ref = new fv_real[n];
  for (int i = 0; i < n; ++i) {
    h_al_in[i] = cases[i].al;
    h_ar_in[i] = cases[i].ar;
    h_al_ref[i] = cases[i].al;
    h_ar_ref[i] = cases[i].ar;
    h_pert_ppm_iv1(h_al_ref[i], h_ar_ref[i]);
  }

  fv_real *d_al_in, *d_ar_in, *d_al_out, *d_ar_out;
  cudaMalloc(&d_al_in, n * sizeof(fv_real));
  cudaMalloc(&d_ar_in, n * sizeof(fv_real));
  cudaMalloc(&d_al_out, n * sizeof(fv_real));
  cudaMalloc(&d_ar_out, n * sizeof(fv_real));
  cudaMemcpy(d_al_in, h_al_in, n * sizeof(fv_real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ar_in, h_ar_in, n * sizeof(fv_real), cudaMemcpyHostToDevice);

  pert_ppm_iv1_kernel<<<1, n>>>(d_al_in, d_ar_in, d_al_out, d_ar_out, n);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("KERNEL LAUNCH FAILED: %s\n", cudaGetErrorString(err));
    return 1;
  }

  fv_real *h_al_out = new fv_real[n], *h_ar_out = new fv_real[n];
  cudaMemcpy(h_al_out, d_al_out, n * sizeof(fv_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ar_out, d_ar_out, n * sizeof(fv_real), cudaMemcpyDeviceToHost);

  bool all_pass = true;
  for (int i = 0; i < n; ++i) {
    const bool bitwise = (h_al_out[i] == h_al_ref[i]) && (h_ar_out[i] == h_ar_ref[i]);
    printf("  %-16s al: dev=% .17e ref=% .17e | ar: dev=% .17e ref=% .17e | %s\n",
           cases[i].name, (double)h_al_out[i], (double)h_al_ref[i], (double)h_ar_out[i],
           (double)h_ar_ref[i], bitwise ? "BITWISE PASS" : "FAIL");
    all_pass &= bitwise;
  }

  printf(all_pass ? "PERT_PPM_IV1: ALL 5 CASES BITWISE PASS\n"
                  : "PERT_PPM_IV1: SOME CASES FAILED\n");

  delete[] h_al_in;
  delete[] h_ar_in;
  delete[] h_al_ref;
  delete[] h_ar_ref;
  delete[] h_al_out;
  delete[] h_ar_out;
  cudaFree(d_al_in);
  cudaFree(d_ar_in);
  cudaFree(d_al_out);
  cudaFree(d_ar_out);
  return all_pass ? 0 : 1;
}
