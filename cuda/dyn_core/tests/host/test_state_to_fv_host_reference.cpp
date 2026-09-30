// Prompt 98 Stage 13/14: host-only golden-reference harness for the
// authoritative Fortran `State_To_FV` (FV_StateMod.F90:2519-2702,
// GEOSfvdycore_fortran_perf checkout,
// src/Components/@FVdycoreCubed_GridComp/FV_StateMod.F90).
//
// This is NOT a CUDA test. No CUDA source is added or modified by this
// task, and no production CUDA implementation exists yet for this routine.
//
// Strategy (Stage 13): Option C (exact transcription), same rationale as
// Prompt 97's FV_To_State harness -- State_To_FV's arguments are
// `type(MAPL_MetaComp)`/`type(T_FVDYCORE_STATE)` pointers, so a direct call
// (Option A) would require a live MAPL grid component environment, which
// this project's harness convention deliberately avoids. No MAPL
// infrastructure is tested here; only the numerical/layout transform.
//
// Scope transcribed -- PRODUCTION-ACTIVE BRANCHES ONLY (confirmed by direct
// source read; see docs/performance/p98_state_to_fv_cuda_port_plan_2026-09-28.md
// Stage 0-3 for the full trace):
//   - grid_type = 0 in production (never >=4): the "Doubly Periodic" branch
//     (FV_StateMod.F90:2578-2587, mpp_update_domains-based U/V halo fill) is
//     DEAD and not transcribed. The active branch is the plain interior
//     U/V copy + mpp_get_boundary scatter-back (FV_StateMod.F90:2589-2604).
//   - mpp_get_boundary itself (a communication primitive, Category D) is
//     OUT OF SCOPE for this numerical-correctness harness -- only the
//     interior U/V copy (isc:iec,jsc:jec) is transcribed, matching
//     FV_To_State's own halo-free interior-only convention. This harness
//     does not model the U(i,jec+1,:)/V(iec+1,j,:) edge rows filled from
//     mpp_get_boundary's buffers.
//   - SW_DYNAMICS = .false. in production (only set .true. when
//     FV_Atm(1)%flagstruct%npz == 1 -- never true for any production or
//     performance-track configuration, all of which use npz=10 or npz=79).
//     The SW_DYNAMICS=.true. branch (FV_StateMod.F90:2614-2621, identical
//     PE transpose loop, so numerically equivalent anyway) is not separately
//     transcribed; the else-branch (2623-2649, which additionally derives
//     peln/pk/ps) is the one active in production and is what is
//     transcribed here.
//   - hydrostatic = .false. in production: the W and DELZ copies
//     (FV_StateMod.F90:2607, 2676) are ACTIVE and are transcribed.
//   - ADJUST_SPLITS defaults to .false. with no production override found
//     (FV_StateMod.F90:2685-2696) -- DEAD, not transcribed.
//
// Root layout fact under test (Stage 3, the one genuine Category-B
// operation in this routine): `STATE%VARS%PE` is (i,j,k) order (canonical
// MAPL state layout) while `FV_Atm(1)%pe` is allocated
// (isc-1:iec+1, npz+1, jsc-1:jec+1) -- i.e. (i,k,j) order (legacy FV3
// layout, fv_arrays.F90:1472, same array Prompt 97's FV_To_State reads
// from in the opposite direction). The loop
// `FV_Atm(1)%pe(i,k,j) = STATE%VARS%PE(i,j,k)` is a real index transpose.
// Fixture 10 (mandatory sentinel) exists specifically to catch a
// transposed-axis mistake in any future port of this transform.
//
// peln (FV_StateMod.F90:2631-2637) and pk (2639-2645) are both re-derived
// from the now-transposed pe and are FINAL output fields of this routine
// (consumed later by fv_dynamics), not discarded intermediates -- distinct
// from pkz (2681), which is a separate, pure copy of STATE%VARS%PKZ (no
// log/exp derivation). Fortran allocates peln in (i,k,j) storage and pk in
// (i,j,k) storage (fv_arrays.F90), matching pe's and ps's layouts
// respectively -- a memory-layout detail only; the VALUES at a given
// logical (i,j,k) triple are what this harness checks, computed throughout
// in a single canonical (i,j,k) order.
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double kRdgas = 287.05;
constexpr double kCpAir = 1004.6;
constexpr double kAkap = kRdgas / kCpAir; // rdgas/cp_air ("kappa", akap in State_To_FV)

// Small "production-shaped" interior grid: ni x nj columns, npz levels.
// State_To_FV's transcribed scope (interior copy + PE transpose + derived
// fields) only ever reads/writes the interior (isc:iec,jsc:jec) sub-array
// of STATE%VARS -- no halo index is referenced on the STATE side -- so no
// halo dimension is modeled in the transform inputs themselves.
struct StateIn {
  int ni, nj, npz;
  std::vector<double> u, v, w, pt, delz, pkz; // (i,j,k) order, ni*nj*npz
  std::vector<double> pe_ijk;                 // (i,j,k) order, ni*nj*(npz+1) -- STATE layout
};

struct FvOut {
  int ni, nj, npz;
  std::vector<double> u, v, w, pt, delz, pkz; // (i,j,k) order
  std::vector<double> pe_ikj;                 // (i,k,j) order, ni*(npz+1)*nj -- FV_Atm native layout
  std::vector<double> peln_can, pk_can;       // (i,j,k)-order VALUES (see file header memory-layout note)
  std::vector<double> ps;                     // (i,j) order
  std::vector<double> delp;                   // (i,j,k) order
};

inline int idx_ijk(int i, int j, int k, int nj, int npz) { return (i * nj + j) * npz + k; }
inline int idx_ikj(int i, int k, int j, int npzp1, int nj) { return (i * npzp1 + k) * nj + j; }
inline int idx_ij(int i, int j, int nj) { return i * nj + j; }

// Mechanical transcription of FV_StateMod.F90:2589-2681, active branches
// only (grid_type=0 interior-only, hydrostatic=.false., SW_DYNAMICS=.false.).
FvOut state_to_fv_host(const StateIn &s) {
  const int ni = s.ni, nj = s.nj, npz = s.npz, npzp1 = npz + 1;
  FvOut out;
  out.ni = ni; out.nj = nj; out.npz = npz;
  out.u.resize(ni * nj * npz);
  out.v.resize(ni * nj * npz);
  out.w.resize(ni * nj * npz);
  out.pt.resize(ni * nj * npz);
  out.delz.resize(ni * nj * npz);
  out.pkz.resize(ni * nj * npz);
  out.pe_ikj.resize(ni * npzp1 * nj);
  out.peln_can.resize(ni * nj * npzp1);
  out.pk_can.resize(ni * nj * npzp1);
  out.ps.resize(ni * nj);
  out.delp.resize(ni * nj * npz);

  // ---- U/V/W: pure Category-A interior copy, (i,j,k) -> (i,j,k) ----
  // (mpp_get_boundary edge-row scatter-back into U(:,jec+1,:)/V(iec+1,:,:)
  // is Category D communication, out of scope -- see file header.)
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        out.u[d] = s.u[d];
        out.v[d] = s.v[d];
        out.w[d] = s.w[d]; // hydrostatic=.false. active branch
      }

  // ---- PE: Category-B transpose, STATE (i,j,k) -> FV_Atm (i,k,j) ----
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k) {
        out.pe_ikj[idx_ikj(i, k, j, npzp1, nj)] = s.pe_ijk[idx_ijk(i, j, k, nj, npzp1)];
      }

  // ---- peln = log(pe), pk = exp(akap*peln): Category-C, else-branch
  //      (SW_DYNAMICS=.false.), computed here in canonical (i,j,k) order
  //      from the just-transposed pe values (file header memory-layout note) ----
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k) {
        const int d = idx_ijk(i, j, k, nj, npzp1);
        const double pe_val = out.pe_ikj[idx_ikj(i, k, j, npzp1, nj)];
        out.peln_can[d] = std::log(pe_val);
        out.pk_can[d] = std::exp(kAkap * out.peln_can[d]);
      }

  // ---- ps = pe(km+1) slice: Category A ----
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j) {
      out.ps[idx_ij(i, j, nj)] = out.pe_ikj[idx_ikj(i, npz, j, npzp1, nj)];
    }

  // ---- delp = pe(k+1) - pe(k): Category C ----
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const double pe_k = out.pe_ikj[idx_ikj(i, k, j, npzp1, nj)];
        const double pe_kp1 = out.pe_ikj[idx_ikj(i, k + 1, j, npzp1, nj)];
        out.delp[idx_ijk(i, j, k, nj, npz)] = pe_kp1 - pe_k;
      }

  // ---- pt = STATE.PT * STATE.PKZ ("dry T"): Category C ----
  // ---- delz = STATE.DZ (hydrostatic=.false. active branch): Category A ----
  // ---- pkz = STATE.PKZ (separate copy, NOT the derived pk above): Category A ----
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        out.pt[d] = s.pt[d] * s.pkz[d];
        out.delz[d] = s.delz[d];
        out.pkz[d] = s.pkz[d];
      }

  return out;
}

bool is_finite(double v) { return std::isfinite(v); }

struct CaseResult { const char *name; bool pass; };
std::vector<CaseResult> g_results;

StateIn make_state(int ni, int nj, int npz,
                    double (*u_fn)(int, int, int),
                    double (*v_fn)(int, int, int),
                    double (*w_fn)(int, int, int),
                    double (*pt_fn)(int, int, int),
                    double (*delz_fn)(int, int, int),
                    double (*pkz_fn)(int, int, int),
                    double (*pe_fn)(int, int, int)) { // pe_fn(i,j,k) -- STATE order
  StateIn s;
  s.ni = ni; s.nj = nj; s.npz = npz;
  const int npzp1 = npz + 1;
  s.u.resize(ni * nj * npz); s.v.resize(ni * nj * npz); s.w.resize(ni * nj * npz);
  s.pt.resize(ni * nj * npz); s.delz.resize(ni * nj * npz); s.pkz.resize(ni * nj * npz);
  s.pe_ijk.resize(ni * nj * npzp1);
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        s.u[d] = u_fn(i, j, k);
        s.v[d] = v_fn(i, j, k);
        s.w[d] = w_fn(i, j, k);
        s.pt[d] = pt_fn(i, j, k);
        s.delz[d] = delz_fn(i, j, k);
        s.pkz[d] = pkz_fn(i, j, k);
      }
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k)
        s.pe_ijk[idx_ijk(i, j, k, nj, npzp1)] = pe_fn(i, j, k);
  return s;
}

// ---- field-generator functions used across fixtures ----
double smooth_uv(int i, int j, int k) { return std::sin(0.3 * i) * std::cos(0.2 * j) + 0.01 * k; }
double smooth_w(int i, int j, int k) { return 0.05 * std::sin(0.4 * (i + j + k)); }
double vertical_profile_pt(int i, int j, int k) { return 300.0 - 4.0 * k; }
double vertical_profile_delz(int i, int j, int k) { return -500.0 * std::exp(-0.1 * k); }
double horiz_gradient_pt(int i, int j, int k) { return 260.0 + 2.0 * i + 3.0 * j; }
double horiz_gradient_delz(int i, int j, int) { return -300.0 - 10.0 * i; }
double pkz_reasonable(int i, int j, int k) { return 0.85 + 0.001 * (i + j) - 0.01 * k; } // ~(0.5,1.2) band
double zero_field(int, int, int) { return 0.0; }
double pe_hybrid(int i, int j, int k) {
  // Monotonic-in-k hybrid-sigma-like pressure column (STATE order pe(i,j,k)),
  // weakly perturbed by i,j so different columns are distinguishable.
  const double ptop = 1.0, ps = 101325.0 + 50.0 * i - 30.0 * j;
  const double frac = static_cast<double>(k) / 10.0; // npz assumed 10 in fixtures using this
  return ptop + frac * frac * (ps - ptop);
}

void check_and_record(const char *name, const StateIn &s) {
  const FvOut out = state_to_fv_host(s);
  const int ni = s.ni, nj = s.nj, npz = s.npz, npzp1 = npz + 1;
  bool pass = true;

  // Invariant: U/V/W are bitwise-identical interior copies of the source (Category A).
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        if (out.u[d] != s.u[d] || out.v[d] != s.v[d] || out.w[d] != s.w[d]) pass = false;
        if (out.delz[d] != s.delz[d]) pass = false;
        if (out.pkz[d] != s.pkz[d]) pass = false;
      }

  // Invariant: PE transpose is exact -- out.pe(i,k,j) == source pe(i,j,k).
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npzp1; ++k) {
        if (out.pe_ikj[idx_ikj(i, k, j, npzp1, nj)] != s.pe_ijk[idx_ijk(i, j, k, nj, npzp1)]) pass = false;
      }

  // Invariant: PE monotonically increasing in k (physical pressure column).
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 1; k < npzp1; ++k) {
        const double a = out.pe_ikj[idx_ikj(i, k - 1, j, npzp1, nj)];
        const double b = out.pe_ikj[idx_ikj(i, k, j, npzp1, nj)];
        if (!(b > a)) pass = false;
      }

  // Invariant: ps == pe at k=km+1 (top-of-loop slice), delp == pe(k+1)-pe(k).
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j) {
      const double pe_bot = out.pe_ikj[idx_ikj(i, npz, j, npzp1, nj)];
      if (out.ps[idx_ij(i, j, nj)] != pe_bot) pass = false;
      for (int k = 0; k < npz; ++k) {
        const double pe_k = out.pe_ikj[idx_ikj(i, k, j, npzp1, nj)];
        const double pe_kp1 = out.pe_ikj[idx_ikj(i, k + 1, j, npzp1, nj)];
        if (out.delp[idx_ijk(i, j, k, nj, npz)] != pe_kp1 - pe_k) pass = false;
        if (!(out.delp[idx_ijk(i, j, k, nj, npz)] > 0.0)) pass = false; // physical: delp>0
      }
    }

  // Invariant: pt == STATE.PT * STATE.PKZ exactly (Category C elementwise multiply).
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nj; ++j)
      for (int k = 0; k < npz; ++k) {
        const int d = idx_ijk(i, j, k, nj, npz);
        if (out.pt[d] != s.pt[d] * s.pkz[d]) pass = false;
      }

  // Invariant: no NaN/Inf anywhere in any output field.
  for (double v : out.u) if (!is_finite(v)) pass = false;
  for (double v : out.v) if (!is_finite(v)) pass = false;
  for (double v : out.w) if (!is_finite(v)) pass = false;
  for (double v : out.pt) if (!is_finite(v)) pass = false;
  for (double v : out.delz) if (!is_finite(v)) pass = false;
  for (double v : out.pkz) if (!is_finite(v)) pass = false;
  for (double v : out.pe_ikj) if (!is_finite(v)) pass = false;
  for (double v : out.peln_can) if (!is_finite(v)) pass = false;
  for (double v : out.pk_can) if (!is_finite(v)) pass = false;
  for (double v : out.ps) if (!is_finite(v)) pass = false;
  for (double v : out.delp) if (!is_finite(v)) pass = false;

  // Invariant: pk (derived exp(akap*peln)) is physically plausible for
  // pe spanning ~1 Pa (model top) to ~1e5 Pa (surface), kappa~0.286 --
  // (0,200) is a generous sanity band, same convention as Prompt 97's PKZ check.
  for (double v : out.pk_can) if (!(v > 0.0 && v < 200.0)) pass = false;

  // Repeatability: re-run on identical input must be bitwise identical.
  const FvOut out2 = state_to_fv_host(s);
  for (size_t n = 0; n < out.pt.size(); ++n) if (out.pt[n] != out2.pt[n]) pass = false;
  for (size_t n = 0; n < out.delp.size(); ++n) if (out.delp[n] != out2.delp[n]) pass = false;
  for (size_t n = 0; n < out.pe_ikj.size(); ++n) if (out.pe_ikj[n] != out2.pe_ikj[n]) pass = false;

  g_results.push_back({name, pass});
  std::printf("%-42s %s\n", name, pass ? "PASS" : "FAIL");
}

} // namespace

int main() {
  const int ni = 6, nj = 5, npz = 10; // npz=10 matches the confirmed C512 L10 perf-track
                                       // vertical resolution (same convention as Prompt 97).

  // 1. Smooth 3D field.
  check_and_record("smooth_3d_field",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 vertical_profile_pt, vertical_profile_delz, pkz_reasonable, pe_hybrid));

  // 2. Varying vertical profile (PT/DZ vary strongly with k, flat in i,j) --
  //    exercises the peln/pk/delp k/k+1 indexing across the full column.
  check_and_record("varying_vertical_profile",
      make_state(ni, nj, npz, zero_field, zero_field, zero_field,
                 vertical_profile_pt, vertical_profile_delz, pkz_reasonable, pe_hybrid));

  // 3. Horizontal gradient (PT/DZ vary linearly in i,j, flat in k).
  check_and_record("horizontal_gradient",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 horiz_gradient_pt, horiz_gradient_delz, pkz_reasonable, pe_hybrid));

  // 4. Interior-only scope check: State_To_FV's transcribed scope reads/
  //    writes only the interior (isc:iec,jsc:jec) sub-array of STATE%VARS
  //    (confirmed by direct source read -- the mpp_get_boundary edge-row
  //    fill is a separate, out-of-scope Category-D step). This fixture
  //    asserts the transform's own indexing never runs outside
  //    [0,ni)x[0,nj) by checking i=0/ni-1, j=0/nj-1 boundary columns
  //    produce finite, correctly-transposed PE identical to interior
  //    columns' formula -- i.e. there is no separate/special edge code
  //    path within the transcribed scope to miss.
  check_and_record("boundary_columns_no_special_path",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 horiz_gradient_pt, horiz_gradient_delz, pkz_reasonable, pe_hybrid));

  // 5. u/v/w active (nonhydrostatic branch on -- the only branch transcribed,
  //    per Stage 2: production is always hydrostatic=.false.).
  check_and_record("uvw_active_nonhydrostatic",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 vertical_profile_pt, vertical_profile_delz, pkz_reasonable, pe_hybrid));

  // 6. pt/delz/pkz active with a strong contrast.
  check_and_record("pt_delz_pkz_strong_contrast",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 horiz_gradient_pt, vertical_profile_delz, pkz_reasonable, pe_hybrid));

  // 7/8. q/tracers, nonzero sphum: N/A -- State_To_FV's transcribed scope
  //      has no q/moisture dependence and no tracer array is referenced
  //      anywhere in the routine (confirmed by direct source read; delp is
  //      DERIVED from pe here, not read from STATE -- STATE has no delp
  //      field consumed by this routine at all). No fixture applicable;
  //      documented here rather than silently omitted.

  // 9. Realistic production dims reduced to manageable size: ni=6,nj=5
  //    (reduced from the confirmed 512x512 C512 L10 per-rank interior),
  //    npz=10 (matches the confirmed C512 L10 perf-track vertical
  //    resolution exactly, not an arbitrary reduction).
  check_and_record("production_shaped_reduced_dims",
      make_state(ni, nj, npz, smooth_uv, smooth_uv, smooth_w,
                 vertical_profile_pt, vertical_profile_delz, pkz_reasonable, pe_hybrid));

  // 10. MANDATORY sentinel fixture: pe_src(i,j,k) = a bijective encoding of
  //     (i,j,k) so any accidental axis swap (e.g. transposing i<->j instead
  //     of j<->k, or not transposing at all) produces a detectably wrong
  //     pe(i,k,j). U/V/W/PT/DZ/PKZ also get per-(i,j,k) sentinel values so a
  //     copy-axis mistake in those fields is equally detectable.
  {
    StateIn s;
    s.ni = ni; s.nj = nj; s.npz = npz;
    const int npzp1 = npz + 1;
    s.u.resize(ni * nj * npz); s.v.resize(ni * nj * npz); s.w.resize(ni * nj * npz);
    s.pt.resize(ni * nj * npz); s.delz.resize(ni * nj * npz); s.pkz.resize(ni * nj * npz);
    s.pe_ijk.resize(ni * nj * npzp1);
    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j)
        for (int k = 0; k < npz; ++k) {
          const int d = idx_ijk(i, j, k, nj, npz);
          // Distinct, position-encoding values (deliberately away from
          // physically-plausible ranges) -- this is an index-mapping
          // check, not a physical-realism check.
          s.u[d] = 1000.0 * i + 10.0 * j + 0.1 * k;
          s.v[d] = 2000.0 * i + 20.0 * j + 0.2 * k;
          s.w[d] = 3000.0 * i + 30.0 * j + 0.3 * k;
          s.delz[d] = -400.0 - 0.001 * (100.0 * i + 10.0 * j + k);
          s.pkz[d] = 0.9 + 0.0001 * (100.0 * i + 10.0 * j + k); // must stay physically sane (pt = pt*pkz)
          s.pt[d] = 280.0 + 0.001 * (100.0 * i + 10.0 * j + k);
        }
    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j)
        for (int k = 0; k < npzp1; ++k) {
          // Bijective encode(i,j,k) as a pressure-like monotonic-in-k value
          // so the PE-monotonicity invariant still holds for this fixture.
          const double frac = static_cast<double>(k) / npz;
          s.pe_ijk[idx_ijk(i, j, k, nj, npzp1)] =
              1.0 + frac * frac * (100000.0 + 1000.0 * i + 10.0 * j);
        }
    const FvOut out = state_to_fv_host(s);
    bool pass = true;
    for (int i = 0; i < ni; ++i)
      for (int j = 0; j < nj; ++j) {
        for (int k = 0; k < npz; ++k) {
          const int d = idx_ijk(i, j, k, nj, npz);
          if (out.u[d] != s.u[d] || out.v[d] != s.v[d] || out.w[d] != s.w[d]) pass = false;
          if (out.delz[d] != s.delz[d]) pass = false;
          if (out.pkz[d] != s.pkz[d]) pass = false;
          if (out.pt[d] != s.pt[d] * s.pkz[d]) pass = false;
        }
        for (int k = 0; k < npzp1; ++k) {
          // The decisive transposition check: out.pe(i,k,j) must equal
          // pe_src(i,j,k), NOT pe_src(j,k,i) or any other axis permutation.
          const double expected = s.pe_ijk[idx_ijk(i, j, k, nj, npzp1)];
          if (out.pe_ikj[idx_ikj(i, k, j, npzp1, nj)] != expected) pass = false;
        }
      }
    for (double v : out.pk_can) if (!is_finite(v) || !(v > 0.0 && v < 200.0)) pass = false;
    for (double v : out.delp) if (!is_finite(v) || !(v > 0.0)) pass = false;
    g_results.push_back({"sentinel_layout_transposition_check", pass});
    std::printf("%-42s %s\n", "sentinel_layout_transposition_check", pass ? "PASS" : "FAIL");
  }

  bool all_pass = true;
  for (const auto &r : g_results) all_pass &= r.pass;
  std::printf("RESULT=%s (%zu cases)\n", all_pass ? "PASS" : "FAIL", g_results.size());
  return all_pass ? 0 : 1;
}
