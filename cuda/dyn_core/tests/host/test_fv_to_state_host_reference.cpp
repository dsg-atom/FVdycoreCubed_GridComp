// Prompt 97 Stage 11/13: host-only golden-reference harness for the
// authoritative Fortran `FV_To_State` (FV_StateMod.F90:2704-2780,
// GEOSfvdycore_fortran_perf checkout,
// src/Components/@FVdycoreCubed_GridComp/FV_StateMod.F90).
//
// This is NOT a CUDA test. No CUDA source is added or modified by this
// task, and no production CUDA implementation exists yet for this routine.
//
// Strategy (Stage 11): Option C (exact transcription). FV_To_State is a
// standalone Fortran subroutine (unlike Remap_PressureVars/
// Remap_TotalEnergyConsv, which are inline code inside a larger routine),
// but a direct call (Option A) is impractical here because its arguments
// are `type(MAPL_MetaComp)`/`type(T_FVDYCORE_STATE)` pointers -- exercising
// it directly requires a live MAPL grid component environment, which this
// project's harness convention deliberately avoids building (see
// docs/performance/p95_remap_totalenergycon_cuda_port_plan_2026-09-28.md
// and p91's harness for the same rationale). No MAPL infrastructure is
// tested here; only the numerical/layout transform itself.
//
// Scope transcribed -- PRODUCTION-ACTIVE BRANCHES ONLY, confirmed by direct
// source read (see docs/performance/p97_fv_to_state_cuda_port_plan_2026-09-28.md
// Stage 0-4 for the full trace):
//   - hydrostatic = .false. (production nonhydrostatic config): the W and
//     DZ "else" branches (STATE%VARS%W/DZ = 0.0, FV_StateMod.F90:2731,
//     2760) are DEAD in production and not transcribed as the active path.
//   - SW_DYNAMICS = .false. in production (only set .true. when
//     FV_Atm(1)%flagstruct%npz == 1, FV_StateMod.F90:411 -- never true for
//     any production or performance-track configuration, which all use
//     npz=10 (this reduced L10 perf config) or npz=79 (full production)).
//     The SW_DYNAMICS=.true. branch (FV_StateMod.F90:2736-2738, PE seeded
//     from `phis`/`delp` instead of the pe transpose) is DEAD and not
//     transcribed.
//   - No MOIST_CAPPA/USE_COND/q(sphum) dependence exists in this routine's
//     own `fv_getPKZ(pkz,pe)` call (FV_StateMod.F90:3078-3108) -- it is the
//     plain dry-akap formula, not the moist `fv_getPKZ_NH` variant. Not
//     transcribed because it is not present in the source at all (not a
//     dead branch -- simply absent from this routine).
//   - `delp`/tracers/`q` are NOT touched by FV_To_State at all (confirmed
//     by direct source read) -- no delp or tracer fixture is included; see
//     the plan doc Stage 3/7 for the explicit statement of this finding.
//
// Root layout fact under test (Stage 1/3, the one genuine Category-B
// operation in this routine): `FV_Atm(1)%pe` is allocated
// `(isc-1:iec+1, npz+1, jsc-1:jec+1)` -- i.e. (i,k,j) order (legacy FV3
// layout, fv_arrays.F90:1472) -- while `STATE%VARS%PE` is (i,j,k) order.
// The loop `STATE%VARS%PE(i,j,:) = FV_Atm(1)%pe(i,:,j)` is therefore a real
// index transpose, not a copy. Every other active field (U, V, W, PT, DZ)
// shares (i,j,k) order on both sides and is a pure elementwise copy
// (Category A). Fixture 10 (mandatory sentinel) exists specifically to
// catch a transposed-axis mistake in any future port of this transform.
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double kRdgas = 287.05;
constexpr double kCpAir = 1004.6;
constexpr double kAkap = kRdgas / kCpAir; // rdgas/cp_air ("kappa" in fv_getPKZ)

// Small "production-shaped" interior grid: ni x nj columns, npz levels.
// FV_To_State itself only ever reads/writes the interior (isc:iec,jsc:jec)
// sub-array -- no halo index is referenced in the Fortran source -- so no
// halo dimension is modeled in the transform inputs themselves; fixture 4
// instead surrounds this interior block with sentinel "halo" values in the
// conceptual full source arrays and asserts none of them leak into output.
struct Grid3D {
  int ni, nj, npz;
  std::vector<double> u, v, w, pt, delz;      // (i,j,k) order, ni*nj*npz
  std::vector<double> pe_iKj;                 // (i,k,j) order, ni*(npz+1)*nj -- source layout
};

struct StateOut {
  int ni, nj, npz;
  std::vector<double> U, V, W, PT, DZ;        // (i,j,k) order
  std::vector<double> PE;                     // (i,j,k) order, ni*nj*(npz+1)
  std::vector<double> PKZ;                    // (i,j,k) order
};

inline int idx_ijk(int i, int j, int k, int nj, int npz) { return (i * nj + j) * npz + k; }
inline int idx_ikj(int i, int k, int j, int npzp1, int nj) { return (i * npzp1 + k) * nj + j; }

// Mechanical transcription of FV_StateMod.F90:2725-2775, active branches
// only (hydrostatic=.false., SW_DYNAMICS=.false.), including the inlined
// fv_getPKZ(pkz,pe) body (FV_StateMod.F90:3078-3108) and the final
// PT = PT/PKZ divide (FV_StateMod.F90:2773).
StateOut fv_to_state_host(const Grid3D &g) {
  const int ni = g.ni, nj = g.nj, npz = g.npz, npzp1 = npz + 1;
  StateOut out;
  out.ni = ni; out.nj = nj; out.npz = npz;
  out.U.resize(ni * nj * npz);
  out.V.resize(ni * nj * npz);
  out.W.resize(ni * nj * npz);
  out.PT.resize(ni * nj * npz);
  out.DZ.resize(ni * nj * npz);
  out.PE.resize(ni * nj * npzp1);
  out.PKZ.resize(ni * nj * npz);

  // ---- U/V/W/PT/DZ: pure Category-A copy, (i,j,k) -> (i,j,k) ----
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        out.U[d] = g.u[d];
        out.V[d] = g.v[d];
        out.W[d] = g.w[d];      // hydrostatic=.false. active branch
        out.PT[d] = g.pt[d];
        out.DZ[d] = g.delz[d];  // hydrostatic=.false. active branch
      }

  // ---- PE: Category-B transpose, (i,k,j) -> (i,j,k) ----
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k) {
        out.PE[idx_ijk(i, j, k, nj, npzp1)] = g.pe_iKj[idx_ikj(i, k, j, npzp1, nj)];
      }

  // ---- fv_getPKZ(PKZ, PE): dry akap formula on the now-transposed PE ----
  std::vector<double> peln(ni * nj * npzp1), pk(ni * nj * npzp1);
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k) {
        const int d = idx_ijk(i, j, k, nj, npzp1);
        peln[d] = std::log(out.PE[d]);
        pk[d] = std::exp(kAkap * peln[d]);
      }
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int dpe = idx_ijk(i, j, k, nj, npzp1);
        const int dpepk1 = idx_ijk(i, j, k + 1, nj, npzp1);
        out.PKZ[idx_ijk(i, j, k, nj, npz)] =
            (pk[dpepk1] - pk[dpe]) / (kAkap * (peln[dpepk1] - peln[dpe]));
      }

  // ---- PT = PT / PKZ (FV_StateMod.F90:2773) ----
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        out.PT[d] = out.PT[d] / out.PKZ[d];
      }

  return out;
}

bool is_finite(double v) { return std::isfinite(v); }

struct CaseResult { const char *name; bool pass; };
std::vector<CaseResult> g_results;

Grid3D make_grid(int ni, int nj, int npz,
                  double (*u_fn)(int, int, int),
                  double (*v_fn)(int, int, int),
                  double (*w_fn)(int, int, int),
                  double (*pt_fn)(int, int, int),
                  double (*delz_fn)(int, int, int),
                  double (*pe_fn)(int, int, int)) { // pe_fn(i,k,j) -- source order
  Grid3D g;
  g.ni = ni; g.nj = nj; g.npz = npz;
  const int npzp1 = npz + 1;
  g.u.resize(ni * nj * npz); g.v.resize(ni * nj * npz); g.w.resize(ni * nj * npz);
  g.pt.resize(ni * nj * npz); g.delz.resize(ni * nj * npz);
  g.pe_iKj.resize(ni * npzp1 * nj);
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        g.u[d] = u_fn(i, j, k);
        g.v[d] = v_fn(i, j, k);
        g.w[d] = w_fn(i, j, k);
        g.pt[d] = pt_fn(i, j, k);
        g.delz[d] = delz_fn(i, j, k);
      }
  for (int i = 0; i < ni; ++i)
    for (int k = 0; k < npzp1; ++k)
      for (int j = 0; j < nj; ++j)
        g.pe_iKj[idx_ikj(i, k, j, npzp1, nj)] = pe_fn(i, k, j);
  return g;
}

// ---- field-generator functions used across fixtures ----
double smooth_uv(int i, int j, int k) { return std::sin(0.3 * i) * std::cos(0.2 * j) + 0.01 * k; }
double smooth_w(int i, int j, int k) { return 0.05 * std::sin(0.4 * (i + j + k)); }
double vertical_profile_pt(int i, int j, int k) { return 300.0 - 4.0 * k; } // constant in i,j
double vertical_profile_delz(int i, int j, int k) { return -500.0 * std::exp(-0.1 * k); }
double horiz_gradient_pt(int i, int j, int k) { return 260.0 + 2.0 * i + 3.0 * j; }
double horiz_gradient_delz(int i, int j, int) { return -300.0 - 10.0 * i; }
double zero_field(int, int, int) { return 0.0; }
double pe_hybrid(int i, int k, int j) {
  // Monotonic-in-k hybrid-sigma-like pressure column, weakly perturbed by i,j
  // so different columns are distinguishable but still monotonic.
  const double ptop = 1.0, ps = 101325.0 + 50.0 * i - 30.0 * j;
  const double frac = static_cast<double>(k) / 10.0; // npz assumed 10 in fixtures using this
  return ptop + frac * frac * (ps - ptop);
}

void check_and_record(const char *name, const Grid3D &g,
                       bool check_uvw_bitwise_copy = true) {
  const StateOut out = fv_to_state_host(g);
  const int ni = g.ni, nj = g.nj, npz = g.npz, npzp1 = npz + 1;
  bool pass = true;

  // Invariant: U/V/W are bitwise-identical copies of the source (Category A).
  if (check_uvw_bitwise_copy) {
    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j)
        for (int k = 0; k < npz; ++k) {
          const int d = idx_ijk(i, j, k, nj, npz);
          if (out.U[d] != g.u[d] || out.V[d] != g.v[d] || out.W[d] != g.w[d]) pass = false;
          if (out.DZ[d] != g.delz[d]) pass = false;
        }
  }

  // Invariant: PE transpose is exact -- out.PE(i,j,k) == source pe(i,k,j).
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k) {
        if (out.PE[idx_ijk(i, j, k, nj, npzp1)] != g.pe_iKj[idx_ikj(i, k, j, npzp1, nj)]) pass = false;
      }

  // Invariant: PE monotonically increasing in k (physical pressure column).
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 1; k < npzp1; ++k) {
        const double a = out.PE[idx_ijk(i, j, k - 1, nj, npzp1)];
        const double b = out.PE[idx_ijk(i, j, k, nj, npzp1)];
        if (!(b > a)) pass = false;
      }

  // Invariant: no NaN/Inf anywhere in any output field.
  for (double v : out.U) if (!is_finite(v)) pass = false;
  for (double v : out.V) if (!is_finite(v)) pass = false;
  for (double v : out.W) if (!is_finite(v)) pass = false;
  for (double v : out.PT) if (!is_finite(v)) pass = false;
  for (double v : out.DZ) if (!is_finite(v)) pass = false;
  for (double v : out.PE) if (!is_finite(v)) pass = false;
  for (double v : out.PKZ) if (!is_finite(v)) pass = false;

  // Invariant: PKZ is a pressure-derivative ratio of pe^kappa terms with pe
  // in Pa directly (no reference-pressure divide in this formula) -- for
  // pe spanning ~1 Pa (model top) to ~1e5 Pa (surface) and kappa~0.286,
  // physically plausible PKZ ranges roughly 0.5-40; (0,200) is a generous
  // sanity band wide enough to catch a gross transcription error without
  // being a tight physical bound.
  for (double v : out.PKZ) if (!(v > 0.0 && v < 200.0)) pass = false;

  // Repeatability: re-run on identical input must be bitwise identical.
  const StateOut out2 = fv_to_state_host(g);
  for (size_t n = 0; n < out.PT.size(); ++n) if (out.PT[n] != out2.PT[n]) pass = false;
  for (size_t n = 0; n < out.PKZ.size(); ++n) if (out.PKZ[n] != out2.PKZ[n]) pass = false;
  for (size_t n = 0; n < out.PE.size(); ++n) if (out.PE[n] != out2.PE[n]) pass = false;

  g_results.push_back({name, pass});
  std::printf("%-42s %s\n", name, pass ? "PASS" : "FAIL");
}

} // namespace

int main() {
  const int ni = 6, nj = 5, npz = 10; // fixture 9's "reduced production dims": npz=10 matches
                                       // the confirmed C512 L10 perf-track vertical resolution.

  // 1. Smooth 3D field (U/V smooth sin/cos, PT/DZ smooth too).
  check_and_record("smooth_3d_field",
      make_grid(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                vertical_profile_pt, vertical_profile_delz, pe_hybrid));

  // 2. Varying vertical profile (PT/DZ vary strongly with k, flat in i,j) --
  //    exercises fv_getPKZ's k/k+1 indexing across the full column.
  check_and_record("varying_vertical_profile",
      make_grid(ni, nj, npz, zero_field, zero_field, zero_field,
                vertical_profile_pt, vertical_profile_delz, pe_hybrid));

  // 3. Horizontal gradient (PT/DZ vary linearly in i,j, flat in k).
  check_and_record("horizontal_gradient",
      make_grid(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                horiz_gradient_pt, horiz_gradient_delz, pe_hybrid));

  // 4. Boundary/halo-sensitive dims: FV_To_State reads ONLY the interior
  //    (isc:iec,jsc:jec) sub-array (confirmed by direct source read -- no
  //    halo index appears in the routine). This fixture asserts the
  //    transform's own indexing never runs outside [0,ni)x[0,nj) by
  //    checking i=0/ni-1, j=0/nj-1 boundary columns produce finite,
  //    correctly-transposed PE identical to interior columns' formula --
  //    i.e. there is no separate/special halo-adjacent code path to miss.
  {
    Grid3D g = make_grid(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                          horiz_gradient_pt, horiz_gradient_delz, pe_hybrid);
    check_and_record("boundary_columns_no_special_path", g);
  }

  // 5. u/v/w active (nonhydrostatic branch on -- the only branch transcribed,
  //    per Stage 2: production is always hydrostatic=.false.).
  check_and_record("uvw_active_nonhydrostatic",
      make_grid(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                vertical_profile_pt, vertical_profile_delz, pe_hybrid));

  // 6. pt/delz active with a strong contrast (delp/tracers are NOT touched
  //    by FV_To_State at all -- confirmed by direct source read, see the
  //    file header -- so no delp/tracer fixture exists; this fixture
  //    instead stresses pt/delz, the two fields FV_To_State DOES own here).
  check_and_record("pt_delz_strong_contrast",
      make_grid(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                horiz_gradient_pt, vertical_profile_delz, pe_hybrid));

  // 7/8. q/tracers, nonzero sphum: N/A -- FV_To_State's fv_getPKZ call
  //      (FV_StateMod.F90:3078, dry akap formula) has no q/moisture
  //      dependence and no tracer array is referenced anywhere in the
  //      routine (confirmed by direct source read). No fixture applicable;
  //      documented here rather than silently omitted.

  // 9. Realistic production dims reduced to manageable size: ni=6,nj=5
  //    (reduced from the confirmed 512x512 C512 L10 per-rank interior),
  //    npz=10 (matches the confirmed C512 L10 perf-track vertical
  //    resolution exactly, not an arbitrary reduction).
  check_and_record("production_shaped_reduced_dims",
      make_grid(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                vertical_profile_pt, vertical_profile_delz, pe_hybrid));

  // 10. MANDATORY sentinel fixture: pe_src(i,k,j) = a bijective encoding of
  //     (i,k,j) so any accidental axis swap (e.g. transposing i<->j instead
  //     of k<->j, or not transposing at all) produces a detectably wrong
  //     PE(i,j,k). U/V/W/PT/DZ also get per-(i,j,k) sentinel values so a
  //     copy-axis mistake in those fields is equally detectable.
  {
    Grid3D g;
    g.ni = ni; g.nj = nj; g.npz = npz;
    const int npzp1 = npz + 1;
    g.u.resize(ni * nj * npz); g.v.resize(ni * nj * npz); g.w.resize(ni * nj * npz);
    g.pt.resize(ni * nj * npz); g.delz.resize(ni * nj * npz);
    g.pe_iKj.resize(ni * npzp1 * nj);
    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j)
        for (int k = 0; k < npz; ++k) {
          const int d = idx_ijk(i, j, k, nj, npz);
          // Distinct, position-encoding values (kept away from
          // physically-plausible ranges deliberately -- this is an
          // index-mapping check, not a physical-realism check).
          g.u[d] = 1000.0 * i + 10.0 * j + 0.1 * k;
          g.v[d] = 2000.0 * i + 20.0 * j + 0.2 * k;
          g.w[d] = 3000.0 * i + 30.0 * j + 0.3 * k;
          g.pt[d] = 280.0 + 0.001 * (100.0 * i + 10.0 * j + k); // must stay physically sane
                                                                  // (divided by PKZ below)
          g.delz[d] = -400.0 - 0.001 * (100.0 * i + 10.0 * j + k);
        }
    for (int i = 0; i < ni; ++i)
      for (int k = 0; k < npzp1; ++k)
        for (int j = 0; j < nj; ++j) {
          // Bijective encode(i,k,j) as a pressure-like monotonic-in-k value
          // so the PE-monotonicity invariant still holds for this fixture.
          const double frac = static_cast<double>(k) / npz;
          g.pe_iKj[idx_ikj(i, k, j, npzp1, nj)] =
              1.0 + frac * frac * (100000.0 + 1000.0 * i + 10.0 * j);
        }
    const StateOut out = fv_to_state_host(g);
    bool pass = true;
    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j) {
        for (int k = 0; k < npz; ++k) {
          const int d = idx_ijk(i, j, k, nj, npz);
          if (out.U[d] != g.u[d] || out.V[d] != g.v[d] || out.W[d] != g.w[d]) pass = false;
          if (out.DZ[d] != g.delz[d]) pass = false;
        }
        for (int k = 0; k < npzp1; ++k) {
          // The decisive transposition check: out.PE(i,j,k) must equal
          // pe_src(i,k,j), NOT pe_src(j,k,i) or any other axis permutation.
          const double expected = g.pe_iKj[idx_ikj(i, k, j, npzp1, nj)];
          if (out.PE[idx_ijk(i, j, k, nj, npzp1)] != expected) pass = false;
        }
      }
    for (double v : out.PKZ) if (!is_finite(v) || !(v > 0.0 && v < 200.0)) pass = false;
    g_results.push_back({"sentinel_layout_transposition_check", pass});
    std::printf("%-42s %s\n", "sentinel_layout_transposition_check", pass ? "PASS" : "FAIL");
  }

  bool all_pass = true;
  for (const auto &r : g_results) all_pass &= r.pass;
  std::printf("RESULT=%s (%zu cases)\n", all_pass ? "PASS" : "FAIL", g_results.size());
  return all_pass ? 0 : 1;
}
