// Prompt 86A Stage 8: host-only nq=2 REMAP_Q scope-proof test. Kord schedule
// corrected by Prompt 87: the actual GEOS-integrated runtime forces
// kord_tr=9 unconditionally (FV_StateMod.F90:563), so BOTH tracers use
// kord=9 -- not the (8,9) split originally assumed here.
//
// Purpose: prove the existing __host__ __device__ map_scalar_column<double>
// (fv3_map_scalar_column.hpp), already used by the CUDA REMAP_Q kernel and
// by REMAP_UV (both kord=9, production-validated), is directly usable for a
// production-shaped nq=2 tracer set (sphum-like, kord=9; cld_amt-like,
// kord=9 -- fv_dynamics.F90:942-943's "kord_tracer(iq)=flagstruct%kord_tr;
// if (iq==cld_amt) kord_tracer(iq)=9", with flagstruct%kord_tr itself forced
// to 9 by FV_StateMod.F90:563) -- via two independent per-tracer calls on
// the SAME shared pe1/pe2 pressure grid (mapn_tracer's own shared-geometry
// structure, fv_mapz.F90:2154-2265).
//
// This is NOT a CUDA test and does NOT claim GPU execution or bitwise
// equivalence to any new CUDA kernel (none is implemented by this task).
// It checks the two production-relevant safety properties mapn_tracer/
// fillz care about -- column mass conservation and positivity -- for a
// synthetic but production-shaped 2-tracer case, using ONLY the existing,
// already-CUDA-validated (kord=8) / already-production-validated (kord=9,
// via REMAP_UV) host reference. No new numerical code is added.
#include "fv3_map_scalar_column.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using fv3::gpu::vertical::Column;

namespace {

struct TracerResult {
  std::vector<double> q2;
  double mass_in = 0.0, mass_out = 0.0;
  int nan_count = 0;
  bool any_negative = false;
};

// Runs one tracer through the existing host map_scalar_column path with the
// given kord, on a shared (pe1, pe2) pressure grid -- mirroring mapn_tracer's
// per-iq copy loop over shared geometry (fv_mapz.F90:2226-2253).
TracerResult run_tracer(const std::vector<double> &pe1, const std::vector<double> &pe2,
                         const std::vector<double> &q1, int kord, double q_min) {
  const int km = static_cast<int>(q1.size());
  std::vector<double> dpe1(km), dpe2(km);
  for (int k = 0; k < km; ++k) dpe1[k] = pe1[k + 1] - pe1[k];
  for (int k = 0; k < km; ++k) dpe2[k] = pe2[k + 1] - pe2[k];

  TracerResult r;
  r.q2.assign(km, 0.0);
  std::vector<double> c2(km), c3(km), c4(km), gam(km), qs(km + 1);
  std::vector<char> extm(km);

  Column<const double> pe1_c{pe1.data(), 1}, dpe1_c{dpe1.data(), 1};
  Column<const double> pe2_c{pe2.data(), 1}, dpe2_c{dpe2.data(), 1};
  Column<const double> q1_c{q1.data(), 1};
  Column<double> q2_c{r.q2.data(), 1};

  // iv=0 (scalar; mapn_tracer hardcodes iv=0), gmao_bot/gmao_top=false
  // (mapn_tracer has no GMAO-boundary concept), q_min shared across all
  // tracers at the production call site (fv_mapz.F90:903, literal 0.).
  fv3::gpu::vertical::map_scalar_column<double>(
      pe1_c, dpe1_c, pe2_c, dpe2_c, q1_c, q2_c, c2.data(), c3.data(), c4.data(),
      gam.data(), qs.data(), reinterpret_cast<bool *>(extm.data()), km, km,
      /*iv=*/0, q_min, /*gmao_bot=*/false, /*gmao_top=*/false, kord);

  for (int k = 0; k < km; ++k) {
    r.mass_in += q1[k] * dpe1[k];
    r.mass_out += r.q2[k] * dpe2[k];
    if (std::isnan(r.q2[k]) || std::isinf(r.q2[k])) ++r.nan_count;
    if (r.q2[k] < 0.0) r.any_negative = true;
  }
  return r;
}

} // namespace

int main() {
  const int km = 10;
  // Shared pressure grid (mapn_tracer's pe1/pe2 are computed once per j and
  // reused across the do-iq=1,nq loop, fv_mapz.F90:2154-2225) -- a modest
  // Lagrangian-to-Eulerian shift, same shape family as case22's
  // c512_like_profile fixture.
  std::vector<double> pe1(km + 1), pe2(km + 1);
  for (int k = 0; k <= km; ++k) {
    pe1[k] = 1000.0 + 900.0 * k;               // Pa, km+1 edges
    pe2[k] = 1000.0 + 900.0 * k + 15.0 * std::sin(0.6 * k); // perturbed target
  }
  pe2[0] = pe1[0];
  pe2[km] = pe1[km]; // total-column-preserving boundary, as in production

  // Tracer 1: sphum-like -- smooth, strictly positive, decaying-with-height
  // profile (kg/kg-shaped magnitude), kord=9 (Prompt 87: actual
  // flagstruct%kord_tr runtime value per FV_StateMod.F90:563, not the bare
  // fv_arrays.F90:333 default of 8 Prompt 85/86A assumed).
  std::vector<double> sphum_like(km);
  for (int k = 0; k < km; ++k) sphum_like[k] = 1.0e-2 * std::exp(-0.35 * k);

  // Tracer 2: cld_amt-like -- bounded-fraction-shaped profile with a sharp
  // front (cloud-fraction magnitude/shape), kord=9 (fv_dynamics.F90:943,
  // unconditional override -- the production value for this tracer,
  // regardless of flagstruct%kord_tr). kord=9 is scalar_profile_column's
  // OWN default branch (already production-validated via REMAP_UV's
  // kord_mt=kord_wz=kord_tm=9 production configuration) -- no new numerics.
  std::vector<double> cld_amt_like(km);
  for (int k = 0; k < km; ++k)
    cld_amt_like[k] = (k >= 3 && k <= 6) ? (0.7 + 0.05 * (k - 3)) : 0.05;

  const double q_min = 0.0; // shared literal at the production call site

  const TracerResult sphum_r = run_tracer(pe1, pe2, sphum_like, /*kord=*/9, q_min);
  const TracerResult cld_r = run_tracer(pe1, pe2, cld_amt_like, /*kord=*/9, q_min);

  bool all_ok = true;
  struct Named { const char *name; const TracerResult *r; };
  Named cases[] = {{"sphum_like(kord=9)", &sphum_r}, {"cld_amt_like(kord=9)", &cld_r}};
  for (const auto &c : cases) {
    const double mass_rel = std::fabs(c.r->mass_out - c.r->mass_in) /
                             std::max(std::fabs(c.r->mass_in), 1.0e-30);
    const bool ok = c.r->nan_count == 0 && !c.r->any_negative && mass_rel < 1.0e-12;
    all_ok &= ok;
    std::printf("%-22s mass_in=%.15e mass_out=%.15e mass_rel=%.3e nan=%d "
                "any_negative=%s %s\n",
                c.name, c.r->mass_in, c.r->mass_out, mass_rel, c.r->nan_count,
                c.r->any_negative ? "YES" : "no", ok ? "PASS" : "FAIL");
  }

  std::printf("RESULT=%s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? 0 : 1;
}
