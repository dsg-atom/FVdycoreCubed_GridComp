// Milestone 3A.11T, section 8 -- host-only check of the synthetic benchmark
// column against SIM1_solver's own thermodynamic invariant.
//
// This runs on a login node. It needs no GPU, no CUDA and no runtime: the
// question it answers is purely about the INITIAL STATE the harness hands to
// Riem_Solver_c, and that state is built by closed-form arithmetic in
// benchmarks/run_small_perf_bench.cpp.
//
// THE INVARIANT
//
// SIM1_solver's first executable statement (nh_utils.F90:1232) is
//
//     pe(i,k) = exp(gama*log(-dm2(i,k)/dz2(i,k)*rgas*pt2(i,k))) - pm2(i,k)
//
// with gama = 1/(1-akap), dm2 = delp/grav and dz2 = delz < 0. Write
//
//     X(k) = (rdgas/grav) * delp(k) / (-delz(k)) * pt(k)
//
// so the statement reads pe = X^gama - pm2, where pm2 is the layer's log-mean
// pressure (Riem_Solver_c:396, pm2 = dm/log(pem(k+1)/pem(k))). `pe` here is a
// PERTURBATION pressure: SIM1 solves for the departure from the hydrostatic
// component, which Riem_Solver_c adds back at :417
// (pef = pe2 + pem). So a balanced state requires
//
//     X(k)^gama  ~=  pm2(k)
//
// and the same X is what dyn_core.F90:1075 raises to akap/(1-akap) to get pkz,
// i.e. pkz = pm2^akap = the Exner function. The two are the same relation.
//
// Note what pt therefore IS. fv_dynamics.F90:552 divides by pkz
//
//     pt(i,j,k) = pt(i,j,k)/pkz(i,j,k)
//
// BEFORE calling dyn_core, so every pt inside dyn_core -- including
// Riem_Solver_c's `pt` dummy -- is a virtual temperature divided by the Exner
// function, not an absolute temperature. Inverting the invariant:
//
//     pt(k) = pm2(k)^(1-akap) * grav * (-delz(k)) / (rdgas * delp(k))
//
// which for this column is order 10, not order 280.
//
// Build:
//   g++ -O2 -std=c++17 -o check_bench_column_balance \
//       cuda/dyn_core/tests/host/check_bench_column_balance.cpp

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Verbatim from run_small_perf_bench.cpp's initialisation block.
constexpr int NPZ = 10;
constexpr double kPtop = 1000.0;        // log-spaced grid top, 10 hPa
constexpr double kPsurf = 100000.0;
constexpr double kGrav = 9.80665;
constexpr double kAkap = 2.0 / 7.0;
constexpr double kRdgas = 287.0;

// The OLD uniform-delp grid, kept so the two can be compared side by side. It
// was thermodynamically balanced after the first 3A.11T fix but geometrically
// pathological: see the conditioning report at the end of main().
constexpr double kPtopOld = 100.0;

struct Column {
  const char *name;
  double ptop;
  std::vector<double> delp, delz, pt, temp;
};

// pm2(k) = delp(k) / log(pe(k+1)/pe(k)), Riem_Solver_c nh_utils.F90:396.
double log_mean_p(double pe_top, double pe_bot) {
  return (pe_bot - pe_top) / std::log(pe_bot / pe_top);
}

void report(const Column &c) {
  const double gama = 1.0 / (1.0 - kAkap);
  std::printf("\n=== %s ===\n", c.name);
  std::printf("%3s %12s %12s %10s %12s %12s %12s %10s\n", "k", "delp[Pa]",
              "delz[m]", "pt", "pm2[Pa]", "X^gama[Pa]", "SIM1 pe[Pa]", "ratio");
  double worst = 0.0;
  int worst_k = -1;
  double pe_face = c.ptop;
  for (int k = 0; k < NPZ; ++k) {
    const double pe_top = pe_face;
    const double pe_bot = pe_face + c.delp[k];
    pe_face = pe_bot;
    const double pm2 = log_mean_p(pe_top, pe_bot);
    const double X = (kRdgas / kGrav) * c.delp[k] / (-c.delz[k]) * c.pt[k];
    const double p_implied = std::exp(gama * std::log(X));
    const double pert = p_implied - pm2;
    const double ratio = p_implied / pm2;
    if (std::fabs(pert) > std::fabs(worst)) { worst = pert; worst_k = k; }
    std::printf("%3d %12.2f %12.2f %10.4f %12.2f %12.4e %12.4e %10.3f\n", k,
                c.delp[k], c.delz[k], c.pt[k], pm2, p_implied, pert, ratio);
  }
  std::printf("worst |SIM1 perturbation pressure| = %.4e Pa at k=%d"
              "  (%.1f atmospheres)\n",
              std::fabs(worst), worst_k, std::fabs(worst) / 101325.0);
}

} // namespace

// Balance a column against the invariant, given its interface pressures:
//   rho(k)  = pm2(k) / (rdgas * T(k))
//   delz(k) = -delp(k) / (rho(k) * grav)
//   pkz(k)  = pm2(k)^akap
//   pt(k)   = T(k) / pkz(k)      <- fv_dynamics.F90:552
Column balance(const char *name, const std::vector<double> &face) {
  Column c{name, face.front(), {}, {}, {}, {}};
  for (int k = 0; k < NPZ; ++k) {
    const double dp = face[k + 1] - face[k];
    const double pm2 = log_mean_p(face[k], face[k + 1]);
    const double T = 280.0 + 10.0 * double(k) / NPZ;
    const double pkz = std::exp(kAkap * std::log(pm2));
    const double rho = pm2 / (kRdgas * T);
    c.delp.push_back(dp);
    c.delz.push_back(-dp / (rho * kGrav));
    c.temp.push_back(T);
    c.pt.push_back(T / pkz);
  }
  return c;
}

// nh_p_grad divides the wind increment by the INTERFACE pk increment, so the
// spread of that increment across the column is what decides whether one layer
// dominates the D-grid wind update. Report it alongside the delz spread.
void conditioning(const char *name, const std::vector<double> &face,
                  const Column &c) {
  double dz_lo = 1e300, dz_hi = 0.0, pk_lo = 1e300, pk_hi = 0.0;
  for (int k = 0; k < NPZ; ++k) {
    const double dz = std::fabs(c.delz[k]);
    const double dpk = std::exp(kAkap * std::log(face[k + 1])) -
                       std::exp(kAkap * std::log(face[k]));
    if (dz < dz_lo) dz_lo = dz;
    if (dz > dz_hi) dz_hi = dz;
    if (dpk < pk_lo) pk_lo = dpk;
    if (dpk > pk_hi) pk_hi = dpk;
  }
  double depth = 0.0;
  for (int k = 0; k < NPZ; ++k) depth += std::fabs(c.delz[k]);
  std::printf("%-26s |delz| %8.1f .. %9.1f m  ratio %6.1f : 1   "
              "d(pk) %6.3f .. %7.3f  ratio %6.1f : 1   depth %6.1f km\n",
              name, dz_lo, dz_hi, dz_hi / dz_lo, pk_lo, pk_hi, pk_hi / pk_lo,
              depth / 1000.0);
}

int main() {
  // ---- (a) uniform delp, the grid 3A.11T started from -------------------
  std::vector<double> face_uni(NPZ + 1);
  {
    const double dp = (kPsurf - kPtopOld) / NPZ;
    for (int k = 0; k <= NPZ; ++k) face_uni[k] = kPtopOld + dp * double(k);
  }

  // The ORIGINAL pre-3A.11T state: pt seeded as an absolute temperature and
  // delz asserted as a uniform -1000 m from an assumed 10 km column.
  Column orig{"ORIGINAL harness state (pre-3A.11T)", kPtopOld, {}, {}, {}, {}};
  for (int k = 0; k < NPZ; ++k) {
    orig.delp.push_back(face_uni[k + 1] - face_uni[k]);
    orig.delz.push_back(-10000.0 / NPZ);
    orig.temp.push_back(280.0 + 10.0 * double(k) / NPZ);
    orig.pt.push_back(280.0 + 10.0 * double(k) / NPZ);
  }
  report(orig);

  // ---- (b) uniform delp, balanced. Fixes the thermodynamics only. --------
  Column uni = balance("BALANCED, uniform delp", face_uni);
  report(uni);

  // ---- (c) log-spaced delp, balanced. The grid the harness now uses. -----
  std::vector<double> face_log(NPZ + 1);
  {
    const double lr = std::log(kPsurf / kPtop) / double(NPZ);
    for (int k = 0; k <= NPZ; ++k) face_log[k] = kPtop * std::exp(lr * double(k));
    face_log[NPZ] = kPsurf;
  }
  Column lg = balance("BALANCED, log-spaced delp", face_log);
  report(lg);

  // ---- geometric conditioning: why (b) still could not be integrated -----
  std::printf("\n=== geometric conditioning ===\n");
  conditioning("uniform delp  (was)", face_uni, uni);
  conditioning("log-spaced    (now)", face_log, lg);
  std::printf("Both (b) and (c) satisfy the thermodynamic invariant exactly, so\n"
              "thermodynamic balance alone does not make a column integrable.\n"
              "Only (c) is well conditioned. (b) spans 42.5:1 in |delz| and 12.8:1\n"
              "in the interface d(pk) that nh_p_grad's wind increment is scaled by,\n"
              "with a single 37.8 km top layer; (c) spans 1.0:1 and 3.3:1.\n");

  // The zh/gz column implied by (c), for the harness to seed consistently.
  std::printf("\n=== zh column implied by the log-spaced balanced delz (zs = 0) ===\n");
  std::printf("%3s %14s %16s\n", "k", "zh[m]", "gz = grav*zh");
  std::vector<double> zh(NPZ + 1, 0.0);
  for (int k = NPZ - 1; k >= 0; --k) zh[k] = zh[k + 1] - lg.delz[k];
  for (int k = 0; k <= NPZ; ++k)
    std::printf("%3d %14.2f %16.4e\n", k, zh[k], kGrav * zh[k]);
  std::printf("column depth = %.1f m\n", zh[0]);
  return 0;
}
