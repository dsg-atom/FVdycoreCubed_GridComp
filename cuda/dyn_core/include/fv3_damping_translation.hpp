#pragma once
//===========================================================================
// Milestone 3A.11V, Task 2 -- authoritative per-level damping translation.
//
// PURPOSE
//
// dyn_core.F90 does not consume damping *scalars*. It consumes, once per
// acoustic substep and once per level, eleven per-level quantities that it
// derives from the flagstruct scalars at the top of every dyn_core call. The
// CUDA runtime's storage contract already declares ten of those as
// device-resident immutable_grid INPUT fields (fv3_gpu_contract.cpp:56-66):
//
//     damping_nord_k  damping_ndif   damping_nord_w  damping_d2_bg
//     damping_d4_bg   damping_v      damping_w       damping_t
//     damping_d_con   damping_kgb
//
// but nothing under cuda/ ever converted the scalars into them. Before this
// file, every caller that needed those arrays invented its own values --
// including the 3A.11 performance benchmark, whose fabricated 4-level cycle
// paired d4_bg = 0.16 with nord_k = 2 and was the sole cause of the multi-step
// instability investigated in Milestone 3A.11V. See
// docs/agent_handoff/milestone_3A11V_dt_sensitivity.md.
//
// This header closes that gap. It is the single place where
//
//     host configuration scalars  ->  per-level damping arrays
//
// happens, for the benchmark, for the unit tests, and for the eventual
// GEOS/FV3 bridge alike.
//
// AUTHORITATIVE SOURCE OF TRUTH
//
//   GFDL_atmos_cubed_sphere, GEOS-ESM fork, branch geos/v3.0.0,
//   SHA ca30e26305fda43831095334c02a3f5e0b8c0c45
//   (checked out at /discover/nobackup/projects/imvi-nse/jli30/fv3_reference/)
//
//   model/dyn_core.F90:677-690    d4dmp(1:npz), computed once and cached
//   model/dyn_core.F90:700-765    the per-level loop, including the sponge
//   model/dyn_core.F90:789-802    the d_sw call site, i.e. exactly which of
//                                 these values crosses into the numerics
//
// NOTE that this is the GEOS-ESM fork, not upstream GFDL: the d4dmp() array,
// the d_ext() array and n_zfilter exist only on the fork. Do not re-derive
// this translation from upstream GFDL sources; they differ.
//
// The GEOS-side scalar overrides come from
//   GEOSfvdycore .../FV_StateMod.F90:569-793   the `if (ntiles == 6)` arm
// and the unoverridden defaults from
//   GFDL_atmos_cubed_sphere model/fv_arrays.F90
// Both are reproduced as named factory functions below so a caller never has
// to hand-copy a magic number.
//
// PURITY
//
// Host-only, CUDA-free, no allocation beyond the returned vectors, no runtime
// state, no I/O. translate_damping() is a total function of its arguments, so
// the whole translation is unit-testable on a login node without a GPU -- the
// same discipline fv3_gpu_halo_topology.hpp follows.
//
// RESIDENCY
//
// Every field this produces is declared immutable_grid in the storage
// contract, i.e. uploaded once at initialization and never re-uploaded. A
// caller must therefore call translate_damping() once, upload once, and reuse
// the device arrays for the whole run. There is deliberately no per-step entry
// point here: adding one would reintroduce a per-dyn_core-step H2D transfer,
// which project_architecture.md classifies as an architectural regression.
// The values are genuinely time-invariant in the authoritative code too --
// dyn_core.F90 recomputes them every call from unchanging scalars, and caches
// the only expensive part (d4dmp) behind a saved logical at :150.
//===========================================================================

#include <cstddef>
#include <string>
#include <vector>

namespace fv3::gpu::damping {

/* ------------------------------------------------------------------------
 * INPUT: the flagstruct scalars that dyn_core.F90:677-765 actually reads.
 *
 * Field-for-field, this is the complete read set of that block. If a member
 * is absent here it is because the authoritative block does not read it. In
 * particular the following flagstruct members look like damping controls but
 * are DEAD on this code path -- they are referenced only under the
 * `old_divg_damp` branch of fv_control.F90:901,903, which this dyn_core.F90
 * never takes:
 *
 *     d2_divg_max_k1   d2_divg_max_k2   damp_k_k1   damp_k_k2
 *
 * They are intentionally not plumbed. Adding them would imply they have an
 * effect, and they do not.
 * --------------------------------------------------------------------- */
struct DampingFlags {
  /* Vertical extent. Levels are indexed 0..npz-1 here; Fortran k = index+1. */
  int npz = 0;

  /* dyn_core.F90:705 nord_k = flagstruct%nord, and :713 nord_v = min(2,nord).
   * fv_arrays.F90:342 default 1; FV_StateMod.F90:681 sets 2 for the cube. */
  int nord = 1;

  /* Smagorinsky-type divergence damping coefficient, passed straight through
   * to d_sw (dyn_core.F90:794). Not per-level. fv_arrays default 0.0;
   * FV_StateMod.F90:682 sets 0.2. */
  double dddmp = 0.0;

  /* Background second-order divergence damping. dyn_core.F90:718 caps it at
   * 0.20; the sponge arms below can raise it again. fv_arrays default 0.0;
   * FV_StateMod.F90:689 sets 0.0. */
  double d2_bg = 0.0;

  /* Background del-2n divergence damping, blended over the column by the
   * d4dmp() sine profile at dyn_core.F90:683-684. Equal top/bot collapses
   * that profile to a constant, which is the production case.
   * fv_arrays.F90:359,365 default 0.16 / 0.16;
   * FV_StateMod.F90:683 sets top = 0.12, and :685/:687 set bot = 0.12 for a
   * resolution below npx*CEILING(stretch_fac) < 1200, else 0.06. */
  double d4_bg_top = 0.16;
  double d4_bg_bot = 0.16;

  /* Sponge-layer d2 floors, dyn_core.F90:743 and :753/:761.
   * fv_arrays.F90:379,388 default 4.0 / 2.0 (these are LARGE, see the
   * stability note in damping_stability());
   * FV_StateMod.F90:695,696 set 0.2 / 0.1. */
  double d2_bg_k1 = 4.0;
  double d2_bg_k2 = 2.0;

  /* Vorticity/w/theta damping coefficient source. dyn_core.F90:721-724 forms
   * damp_vt(k) = max(0.01, min(vtdm4, d4_bg_top/4)) when do_vort_damp.
   * fv_arrays.F90:371,453 default 0.0 / false;
   * FV_StateMod.F90:744,743 set 0.01 / true. */
  double vtdm4 = 0.0;
  bool do_vort_damp = false;

  /* Kinetic-energy -> heat conversion, dyn_core.F90:734, zeroed in every
   * sponge arm. fv_arrays.F90:766 default 0.0; FV_StateMod.F90:742 sets 1.0. */
  double d_con = 0.0;

  /* Extra divergence damping, gated by n_zfilter at dyn_core.F90:729-733.
   * fv_arrays.F90:527,522 default 0.02 / 0; FV_StateMod.F90:690 sets 0.0. */
  double d_ext = 0.02;
  int n_zfilter = 0;

  /* Background KE source applied at the bottom level only,
   * dyn_core.F90:707-711. fv_arrays.F90:771 default 0.0;
   * FV_StateMod.F90:560 sets 0.0. */
  double ke_bg = 0.0;

  /* Sponge depth. dyn_core.F90:736 disables the whole sponge for n_sponge<=0
   * or npz==1. fv_arrays.F90:524 default 1; FV_StateMod.F90:694 sets 9. */
  int n_sponge = 1;

  /* dyn_core.F90:680 selects the d4dmp() profile, and :729 the d_ext gate.
   * Production first-run configuration is nonhydrostatic. */
  bool hydrostatic = false;

  /* Model-top pressure, the denominator of the d4dmp() log profile
   * (dyn_core.F90:683). Only read when d4_bg_top != d4_bg_bot and
   * .not. hydrostatic. Units: Pa. */
  double ptop = 1.0;
};

/* ------------------------------------------------------------------------
 * OUTPUT: exactly what dyn_core.F90:789-802 hands to d_sw, per level.
 *
 * Index 0 corresponds to Fortran k = 1 (model top).
 *
 * nord_t and dddmp are scalars, not vectors, and that is not a
 * simplification -- see the ordering note on nord_t below.
 * --------------------------------------------------------------------- */
struct DampingColumn {
  /* dyn_core.F90:705 / :743 / :753 / :761. Order of the divergence damping
   * operator: 0 selects the del-2 arm in d_sw, >0 the del-2n arm. */
  std::vector<int> nord_k;

  /* dyn_core.F90:713 / :748 / :756. Order of the vorticity (damp_vt)
   * operator. NOTE this is nord_v(k), a genuinely per-level array in the
   * authoritative code, declared (npz+1)-sized at dyn_core.F90:260 and
   * re-read by update_dz_d at :905. The storage contract calls it
   * damping_ndif. */
  std::vector<int> ndif;

  /* dyn_core.F90:714 / :745 / :754 / :762. Order of the w operator. */
  std::vector<int> nord_w;

  /* dyn_core.F90:718 / :737 / :743 / :753 / :761. The d2_divg actually passed
   * to d_sw -- already capped and already sponge-adjusted. */
  std::vector<double> d2_bg;

  /* dyn_core.F90:683-684, i.e. d4dmp(k). Never touched by the sponge. */
  std::vector<double> d4_bg;

  /* dyn_core.F90:721-724 / :749 / :757, i.e. damp_vt(k). */
  std::vector<double> damp_v;

  /* dyn_core.F90:726 / :745 / :754 / :762, i.e. damp_w. */
  std::vector<double> damp_w;

  /* dyn_core.F90:727, i.e. damp_t. See the ordering note below: constant. */
  std::vector<double> damp_t;

  /* dyn_core.F90:734 / :751 / :759 / :763, i.e. d_con_k. */
  std::vector<double> d_con;

  /* dyn_core.F90:707-711, i.e. kgb. Nonzero at the bottom level only. */
  std::vector<double> kgb;

  /* dyn_core.F90:729-733, i.e. d_ext(k). Not currently a contract field --
   * produced here because the authoritative block produces it, so a future
   * consumer does not have to re-derive the n_zfilter gate. */
  std::vector<double> d_ext;

  /* dyn_core.F90:715, nord_t = nord_v(k).
   *
   * LOAD-BEARING ORDERING SUBTLETY. Line :715 runs BEFORE the sponge block
   * at :736-765, and no sponge arm reassigns nord_t. So although it is
   * written inside the k loop, its value is min(2, nord) at EVERY level,
   * including the sponge levels where nord_v(k) is subsequently reset to 0.
   * Representing it as a vector would invite a future reader to "fix" it to
   * track ndif, which would change the numerics. It is a scalar here on
   * purpose, matching NumericalConfig::nord_t. */
  int nord_t = 0;

  /* Likewise dyn_core.F90:794 passes flagstruct%dddmp unmodified. Carried
   * here so a caller has one object to consult. */
  double dddmp = 0.0;

  /* True when translate_damping() ran the cached-d4dmp sine profile rather
   * than the constant shortcut, i.e. when d4_bg_top != d4_bg_bot and the
   * configuration is nonhydrostatic. Diagnostic only. */
  bool d4dmp_profile_active = false;
};

/* ------------------------------------------------------------------------
 * THE TRANSLATION.
 *
 * A faithful, statement-ordered reimplementation of dyn_core.F90:677-765.
 *
 * pfull must point to npz layer-mean pressures in Pa, ascending with index
 * (index 0 = model top), i.e. FV3's pfull(1:npz). It is read ONLY by the
 * d4dmp() profile, so a caller whose d4_bg_top == d4_bg_bot -- the production
 * case -- may pass nullptr. Passing nullptr when the profile is active is a
 * caller error and yields d4_bg = d4_bg_top at every level, matching the
 * hydrostatic shortcut at dyn_core.F90:687.
 *
 * No numerical semantics are invented here. Where the authoritative code
 * clamps (the min/max at :718, :721, :743), this clamps identically; where it
 * does not clamp, neither does this. In particular translate_damping() does
 * NOT enforce del-2n stability -- see damping_stability() for that, which
 * reports and never mutates.
 * --------------------------------------------------------------------- */
DampingColumn translate_damping(const DampingFlags& f, const double* pfull);

/* GEOS production first-run configuration for a global cubed sphere: the
 * scalars as FV_StateMod.F90:569-793 leaves them. d4_bg_bot follows the
 * :685/:687 resolution branch, so the caller supplies the branch outcome
 * rather than this function guessing a resolution it cannot see.
 *
 * high_resolution == false  ->  d4_bg_bot = 0.12  (npx*CEILING(stretch_fac) < 1200)
 * high_resolution == true   ->  d4_bg_bot = 0.06
 *
 * ptop must be the actual model-top pressure; it only matters when
 * high_resolution is true, because only then do top and bot differ. */
DampingFlags geos_production_flags(int npz, double ptop, bool high_resolution);

/* The unoverridden fv_arrays.F90 defaults, for tests that need to show the
 * translation reproduces them and for the stability audit, which has to
 * report on configurations a user could actually reach. */
DampingFlags fv_arrays_defaults(int npz, double ptop);

/* ------------------------------------------------------------------------
 * TASK 3: del-2n stability metrics.
 *
 * A REPORT, never a mutation. The authoritative Fortran does not clip these
 * coefficients, so neither does this project: silently clipping would make
 * the CUDA backend disagree with the numerical source of truth in exactly the
 * configurations where the disagreement matters most.
 *
 * NORM. For the divergence-damping del-2n arm the growth factor per acoustic
 * substep of the least-stable grid mode is
 *
 *     gamma_d4 = (8 * d4_bg)^(nord_k + 1) * (da_min / area_local)
 *
 * with instability at gamma > 2. This form is MEASUREMENT-CALIBRATED: it was
 * derived term by term from the Fortran statements and then confirmed by a
 * parameter-free threshold measurement at two different nord on a seam-free
 * initial condition (nord=1 predicted s* in [1.105,1.127], measured
 * (1.10,1.15]; nord=2 predicted [0.984,0.998], measured (0.98,1.00]).
 * Milestone 3A.11V, docs/agent_handoff/milestone_3A11V_dt_sensitivity.md.
 * Critical d4_bg is 2^(1/(nord+1))/8 = 0.2500 / 0.17678 / 0.15749 / 0.14865
 * for nord 0/1/2/3.
 *
 * The nord_k > 0 branch carries a SECOND, ADDITIVE term: sw_core.F90:1418
 * applies vort = damp2*delpc + dd8*divg_d, so the del-2 part does not vanish
 * when the del-2n part is active. Its contribution is
 *
 *     gamma_d2 = 8 * max(d2_bg, min(0.20, dddmp*|div*dt|)) * (da_min_c / area_c)
 *
 * The Smagorinsky part is flow-dependent and therefore not knowable from
 * configuration alone. Both a static bound (Smagorinsky = 0) and a worst-case
 * bound (Smagorinsky saturated at its own 0.20 cap) are reported, because the
 * static bound alone understates the risk and the worst-case bound alone
 * overstates it.
 *
 * CALIBRATION STATUS OF THE OTHER TWO NORMS. gamma_v and gamma_t use the
 * structurally analogous form, because sw_core.F90:1477,:973 build
 * damp4 = (damp_v*da_min_c)^(nord_v+1) and tp_core.F90:217,237 build
 * damp = (damp_c*da_min)^(nord+1) -- the same shape as the d4 operator. They
 * are ANALOGOUS BY CONSTRUCTION and are NOT independently threshold-measured.
 * They are consistent with measurement (the production damp_v = 0.10 at
 * nord_v = 2 gives gamma_v = 0.51, and that pairing ran stably for 300 calls
 * in 3A.11V), but a caller must not treat them as calibrated to the same
 * standard as gamma_d4.
 * --------------------------------------------------------------------- */
struct LevelStability {
  int k = 0;

  /* Which operator arm this level actually takes. When nord_k == 0 the
   * del-2n coefficient d4_bg is not read at all, so gamma_d4 is reported as
   * 0.0 and d4_active is false. Reporting a nonzero gamma for an unread
   * coefficient was a real source of confusion in 3A.11V. */
  bool d4_active = false;

  double gamma_d4 = 0.0;       /* del-2n divergence damping */
  double gamma_d2_static = 0.0;/* del-2 part, Smagorinsky = 0 */
  double gamma_d2_worst = 0.0; /* del-2 part, Smagorinsky saturated at 0.20 */
  double gamma_div_static = 0.0;/* gamma_d4 + gamma_d2_static */
  double gamma_div_worst = 0.0; /* gamma_d4 + gamma_d2_worst */
  double gamma_v = 0.0;        /* vorticity/damp_vt operator */
  double gamma_w = 0.0;        /* w operator (damp_w at order nord_w) */
  double gamma_t = 0.0;        /* scalar/deln_flux operator (damp_t, nord_t) */

  /* gamma_div_static > kGammaLimit: unstable from configuration alone. */
  bool unstable_static = false;
  /* gamma_div_worst > kGammaLimit but not unstable_static: reachable only
   * with a strongly divergent flow. */
  bool unstable_worst_case = false;
  bool unstable_vorticity = false;
  bool unstable_w = false;
  bool unstable_scalar = false;
};

/* Instability threshold of the norms above. */
constexpr double kGammaLimit = 2.0;

struct StabilityReport {
  std::vector<LevelStability> level;

  /* Any level with unstable_static, i.e. a configuration that is unstable
   * without help from the flow. This is the condition a caller should refuse
   * or loudly warn on. */
  bool any_unstable_static = false;
  /* Any level reachable-unstable only through the Smagorinsky term. */
  bool any_unstable_worst_case = false;
  bool any_unstable_other = false;   /* vorticity, w or scalar arm */

  /* Worst gamma_div_static and the level attaining it, for a one-line
   * summary. -1 when there are no levels. */
  double worst_gamma = 0.0;
  int worst_level = -1;
};

/* Compute the metrics. da_min / da_min_c and the local cell areas come from
 * the grid, not from the damping configuration, so they are explicit
 * arguments: area_local is the smallest A-grid cell area on the panel and
 * area_c_local the smallest C-grid one. On the authoritative gnomonic cube
 * da_min IS the global minimum of those areas, so da_min/area_local <= 1 and
 * a caller that passes da_min for both gets the conservative ratio 1.0. */
StabilityReport damping_stability(const DampingColumn& c,
                                  double da_min, double area_local,
                                  double da_min_c, double area_c_local);

/* Render the report as a human-readable table, one line per level plus a
 * verdict line. Cheap: pure string building, no I/O. Intended for a
 * one-shot print at initialization, which is why it is not gated behind a
 * debug flag -- an unstable damping column should be visible in every log,
 * not only in a log someone thought to enable.
 *
 * Lines are prefixed [FV3-DAMP] so they can be grepped out of a mixed
 * multi-tile stream. */
std::string format_stability_report(const DampingColumn& c,
                                    const StabilityReport& r);

/* Render the translated column itself, one line per level. Lets a caller
 * prove in its own log which values it actually used, which is what the
 * benchmark's fabricated table made impossible. */
std::string format_damping_column(const DampingColumn& c);

}  // namespace fv3::gpu::damping
