#pragma once

// ===========================================================================
// Milestone 3A.7 / WP-B.4 -- authoritative FV3 divergence damping, Smagorinsky
// corner strain, d_con heating prerequisites and the local cubed-sphere
// corner/edge semantics that d_sw needs.
//
// Authoritative source: GEOS-ESM/GFDL_atmos_cubed_sphere, tag geos/v3.0.0,
// SHA ca30e26305fda43831095334c02a3f5e0b8c0c45, model/sw_core.F90 and
// model/dyn_core.F90.
//
// SCOPE. This header covers exactly the block of d_sw that Milestone 3A.6
// stopped in front of: sw_core.F90:1273-1538. It provides numerical services
// only. It is NOT the d_sw driver, and nothing here is wired into
// NumericalDispatch -- see fv3_gpu_production_dispatch.cu.
//
// -------------------------------------------------------------------------
// §1 ORDERED OPERATION TABLE (sw_core.F90:1273-1538)
// -------------------------------------------------------------------------
// Rows are in authoritative order. "stagger" is the grid location of the
// row's OUTPUT: A = cell centre, B = cell corner, D(u)/D(v) = D-grid wind
// faces, C(u)/C(v) = C-grid wind faces. Branch L is d_sw's `nord==0` arm and
// branch H its `else` arm; per-level nord_k means BOTH are live in one run
// (see §3).
//
//  #   | Fortran         | operation                                  | in                            | out       | stagger | reads beyond box     | corner/edge dependency                    | workspace   | next consumer
// -----|-----------------|--------------------------------------------|-------------------------------|-----------|---------|----------------------|-------------------------------------------|-------------|---------------
//  L1  | :1298-1313      | ptc (upwind sin_sg at j==1/npy)            | u,va,vc,dyc,cosa_v,sina_v,    | ptc       | D(u)    | va(i,j-1)            | j==1 / j==npy switch to the one-sided     | ptc         | L3
//      |                 |   launch_divdamp_nord0_cuda                 | sin_sg                        |           |         |                      | sin_sg form keyed on sign(vc)             |             |
//  L2  | :1316-1335      | vort (upwind sin_sg at i==1/npx)           | v,ua,uc,dxc,cosa_u,sina_u,    | vort      | D(v)    | ua(i-1,j)            | is==1 / ie+1==npx override columns        | vort        | L3,L4
//      |                 |   launch_divdamp_nord0_cuda                 | sin_sg                        |           |         |                      |                                           |             |
//  L3  | :1338-1342      | delpc = vort(j-1)-vort(j)+ptc(i-1)-ptc(i)  | vort,ptc                      | delpc     | B       | vort(i,j-1),ptc(i-1) | none                                      | delpc       | L4
//  L4  | :1345-1349      | delpc corner term removal                  | delpc,vort                    | delpc     | B corner| vort(1,0),vort(np,0) | the four tile corners only                | delpc       | L5
//  L5  | :1351-1358      | rarea_c scale, damp, vort, ke += vort      | delpc,rarea_c,d2_bg,dddmp,dt  | delpc,    | B       | none                 | none                                      | -           | D1, dwind
//      |                 |   launch_divdamp_nord0_cuda                 |                               | vort,ke   |         |                      |                                           |             |
// -----|-----------------|--------------------------------------------|-------------------------------|-----------|---------|----------------------|-------------------------------------------|-------------|---------------
//  H1  | :1364-1368      | delpc = divg_d (external-mode filter save)  | divg_d (from WP-A c_sw)       | delpc     | B       | none                 | none                                      | delpc       | H4
//  H2a | :1377           | fill_corners(divg_d, XDir, BGRID)          | divg_d                        | divg_d    | B       | ng x ng corner blocks| LOCAL rotation, only when nt/=0           | -           | H2b
//      |                 |   launch_fill_corners_bgrid_cuda(dir=1)     |                               |           |         |                      |                                           |             |
//  H2b | :1378-1382      | vc = (divg_d(i+1)-divg_d(i))*divg_u        | divg_d,divg_u                 | vc        | C(v)    | divg_d(i+1,j)        | needs H2a's rotated corners               | vc          | H2e,H2f
//  H2c | :1384           | fill_corners(divg_d, YDir, BGRID)          | divg_d                        | divg_d    | B       | ng x ng corner blocks| LOCAL rotation, only when nt/=0           | -           | H2d
//      |                 |   launch_fill_corners_bgrid_cuda(dir=2)     |                               |           |         |                      |                                           |             |
//  H2d | :1385-1389      | uc = (divg_d(i,j+1)-divg_d(i,j))*divg_v    | divg_d,divg_v                 | uc        | C(u)    | divg_d(i,j+1)        | needs H2c's rotated corners               | uc          | H2e,H2f,H2g
//  H2e | :1391           | fill_corners(vc,uc,VECTOR,DGRID)           | vc,uc                         | vc,uc     | C       | ng x ng corner blocks| LOCAL rotation with mySign = -1           | -           | H2f
//      |                 |   launch_fill_corners_dgrid_vector_cuda     |                               |           |         |                      |                                           |             |
//  H2f | :1392-1396      | divg_d = uc(j-1)-uc(j)+vc(i-1)-vc(i)       | uc,vc                         | divg_d     | B       | uc(i,j-1),vc(i-1,j)  | none                                      | -           | H2g / next n
//  H2g | :1399-1403      | divg_d corner term removal                 | divg_d,uc                     | divg_d     | B corner| uc(1,0),uc(npx,0)    | the four tile corners only                | -           | next n / H4
//      |                 |   H2a..H2g are one launcher:                                                                                                                                        |
//      |                 |   launch_divdamp_high_order_cuda (H1 + the whole n-loop)                                                                                                             |
//  H3  | :1407-1411      | vort = 0  OR  smag_corner(|dt|,u,v)        | u,v or nothing                | vort       | B       | see S1..S9           | see S1..S9                                | see §6      | H4
//      |                 |   launch_divdamp_vort_zero_cuda / launch_smag_corner_cuda                                                                                                            |
//  H4  | :1413-1421      | damp2, vort = damp2*delpc + dd8*divg_d,     | delpc,divg_d,vort,d2_bg,d4_bg | vort,ke    | B       | none                 | none                                      | -           | D1, dwind
//      |                 | ke += vort                                 |                               |            |         |                      |                                           |             |
//      |                 |   launch_divdamp_high_order_finalize_cuda                                                                                                                            |
// -----|-----------------|--------------------------------------------|-------------------------------|------------|---------|----------------------|-------------------------------------------|-------------|---------------
//  D1  | :1425-1436      | ub = vort(i)-vort(i+1); vb = vort(j)-      | vort                          | ub,vb      | B       | vort(i+1),vort(j+1)  | none                                      | ub,vb       | D2
//      |                 | vort(j+1)                                  |                               |            |         |                      |                                           |             |
//      |                 |   launch_dcon_vort_diff_cuda               |                               |            |         |                      |                                           |             |
//  --  | :1438-1471      | vorticity transport + D-wind update        | ALREADY IMPLEMENTED by Milestone 3A.6 (launch_vorticity_absolute_cuda, launch_scalar_transport_cuda, launch_dwind_update_cuda)   |
//  V1  | :1476-1479      | del6_vt_flux(nord_v, damp4, wk,vort,ut,vt) | wk,del6_u,del6_v,rarea        | vort,ut,vt | see §8  | ng-wide halo         | 2+2*nord_v copy_corners of vort           | vort,ut,vt  | D2, V2
//      |                 |   REUSED: launch_del6_vt_flux_cuda via launch_vorticity_damping_cuda -- no new numerical code                                                                          |
//  D2  | :1482-1495      | ub=(ub+vt)*rdx, fy=u*rdx, gy=fy*ub;        | ub,vb,ut,vt,u,v,rdx,rdy       | ub,vb,     | B / faces| none                | none                                      | gx,gy       | D3
//      |                 | vb=(vb-ut)*rdy, fx=v*rdy, gx=fx*vb         |                               | fx,fy,gx,gy|         |                      |                                           |             |
//      |                 |   launch_dcon_scale_cuda                   |                               |            |         |                      |                                           |             |
//  D3  | :1497-1520      | heat_source (total-energy conserving)      | delp,ub,vb,fx,fy,gx,gy,rsin2, | heat_source| A       | ub(i,j+1),vb(i+1,j)  | none                                      | -           | WP-D heating
//      |                 |   launch_dcon_heat_source_cuda             | cosa_s,d_con                  |            |         | fy(i,j+1),fx(i+1,j)  |                                           |             |
//  V2  | :1522-1534      | u += vt ; v -= ut                          | ut,vt                         | u,v        | D       | none                 | none                                      | -           | WP-B.5 exit
//      |                 |   launch_dwind_damping_add_cuda            |                               |            |         |                      |                                           |             |
// -----|-----------------|--------------------------------------------|-------------------------------|------------|---------|----------------------|-------------------------------------------|-------------|---------------
//  X1  | :1659-1764      | divergence_corner (produces divg_d)        | u,v,ua,va,dxc,dyc,rarea_c,    | divg_d     | B       | uf(is-1),vf(j-1)     | four corner removals + the is==1/ie+1==npx| uf,vf       | H1, H2b
//      |                 |   ALREADY IMPLEMENTED by WP-A:             | sin_sg,cos_sg                 |            |         |                      | vf column overrides                       |             |
//      |                 |   cgrid::launch_divergence_corner_level -- called from c_sw (sw_core.F90:175-181), NOT from d_sw. See §2.                                                            |
//
// -------------------------------------------------------------------------
// §2 divergence_corner IS A c_sw PRODUCER, NOT A d_sw ROUTINE
// -------------------------------------------------------------------------
// sw_core.F90:175-181 -- inside c_sw, not d_sw:
//
//   if ( nord > 0 ) then
//        if (nested) then; call divergence_corner_nest(...)
//        else;             call divergence_corner(u,v,ua,va,divg_d,...)
//
// d_sw only CONSUMES divg_d: H1 saves it and H2b/H2d/H2f iterate on it. So
// divergence_corner belongs to WP-A's C-grid stage and is already implemented
// as cgrid::launch_divergence_corner_level
// (cuda/dyn_core/cgrid/fv3_cgrid_divergence.cu). Per the project's
// no-duplication rule this milestone AUDITED that implementation rather than
// writing a second one. The audit repaired one defect (a launch origin in the
// grid_type > 3 arm that left uf(is-2,.) and vf(.,js-2) unwritten while the
// divg kernel read them) and documented one gap it could not close (no
// `nested` flag on DivergenceCornerArgs, so nested grids are unsupported but
// not rejected). Full result:
// docs/agent_handoff/milestone_3a7_wp_b_divergence_corner.md §4. The
// production first-run configuration uses grid_type = 0, so the repaired arm
// is not on the production path.
//
// The same applies to `delpc` and `divg_d`, which are NOT separate Fortran
// subroutines despite reading like them:
//   * delpc is d_sw's local corner divergence -- rows L1..L5 in branch L, or
//     the single copy H1 in branch H.
//   * divg_d is the array divergence_corner fills and that H2 iterates on
//     in place; it survives across acoustic substeps and is halo-exchanged at
//     dyn_core.F90:528 with position=CORNER.
//
// -------------------------------------------------------------------------
// §3 BOTH DAMPING BRANCHES ARE LIVE IN ONE PRODUCTION RUN
// -------------------------------------------------------------------------
// dyn_core.F90:697-760 recomputes nord_k INSIDE the k loop and calls d_sw once
// per level. With n_sponge > 0 the sponge overrides force
//
//   k == 1                                     -> nord_k = 0
//   k <= max(2, n_sponge-1), d2_bg_k2 > 0.01   -> nord_k = 0
//   k <= max(3, n_sponge),   d2_bg_k2 > 0.05   -> nord_k = 0
//   otherwise                                  -> nord_k = flagstruct%nord > 0
//
// and simultaneously overrides d2_divg, nord_v(k), damp_vt(k) and d_con_k. A
// batched CUDA launch over k therefore sees BOTH d_sw arms at once. That is
// why every launcher below takes a DivergenceLevels selector instead of a
// scalar nord: collapsing the branch structure to whichever arm one test
// configuration happens to exercise would silently drop the sponge levels.
//
// The per-level quantities are nord_k, d2_divg, d4dmp(k) and d_con_k
// (DivergenceLevels) plus damp_vt(k) and nord_v(k) (LevelFilter, §8). dddmp,
// da_min, da_min_c, grid_type and dt are genuinely scalar.
//
// -------------------------------------------------------------------------
// §4 ANTI-DEPENDENCES AND WORKSPACE ALIASING (all authoritative)
// -------------------------------------------------------------------------
//  (a) H2b/H2d OVERWRITE vc and uc -- WP-A's C-grid winds. d_sw declares them
//      intent(inout) and nothing after :1405 reads them; dyn_core rebuilds
//      them from c_sw next substep. Branch L, by contrast, READS vc and uc
//      (the upwind sin_sg switches at L1/L2). The two arms are therefore
//      level-disjoint by construction, and callers must not run branch H's
//      iterate on a level that branch L still needs.
//  (b) divg_d is an in-place accumulator across the n-loop; H2f must observe
//      completed H2b/H2d, so the passes are separate launches on one stream.
//  (c) L5 writes vort after L3/L4 have read it. Same array, three roles in
//      branch L: the v-based term (L2), then the damping increment (L5).
//  (d) smag_corner's ut/vt/wk are the routine's OWN locals
//      (sw_core.F90:1868-1871) and are NOT d_sw's ut/vt/wk. Passing d_sw's
//      arrays would destroy the contravariant winds that the D-wind update
//      still needs. SmagCornerWorkspace exists precisely to make that
//      impossible to get wrong.
//  (e) V1's del6_vt_flux(nord_v, npx, npy, damp4, wk, vort, ut, vt) gives
//      ut/vt a THIRD meaning (del6 fluxes fx2/fy2) and overwrites vort (as
//      d2) after H4/L5 produced it and D1 consumed it. It also mutates wk in
//      place via copy_corners.
//  (f) D2 overwrites fx/fy -- the vorticity fluxes from fv_tp_2d -- which the
//      D-wind update at :1461-1471 has already consumed.
//  (g) If damp_v <= 1.E-5 the del6 call is skipped, so D2's `(ub + vt)*rdx`
//      reads vt in its ROW-8 meaning (u*dx), not as a diffusive flux. That is
//      authoritative behaviour, not a bug, and must not be "fixed".
//
// -------------------------------------------------------------------------
// §5 LOCAL CORNER/EDGE ARCHITECTURE
// -------------------------------------------------------------------------
// Every corner/edge operation in this milestone is LOCAL cubed-sphere index
// algebra on data the rank already owns. None of it is communication.
// Classification (see the handoff §11 for the full table):
//
//   inline in the numerical kernel : the four corner-term removals (L4, H2g,
//                                    S4) and the is==1/ie+1==npx/j==1/j==npy
//                                    one-sided formulas (L1, L2)
//   focused local CUDA kernel      : fill_corners BGRID scalar rotation and
//                                    fill_corners DGRID vector-pair rotation
//                                    (launch_fill_corners_*_cuda below)
//   reusable WP-B internal service : tp_core copy_corners
//                                    (launch_copy_corners_cuda, 3A.4)
//   true communication service     : NONE in WP-B.4
//
// NumericalDispatch::corner_edge_service is a CornerServiceFn keyed on
// FieldId/CornerOperation/DiffusionPhase (include/fv3_gpu_internal.hpp:193).
// It cannot express "rotate these two device views with mySign = -1" without
// first registering vc/uc/divg_d as global FieldIds, and it mixes MPI-facing
// halo semantics with local panel math. Populating it here would be false, so
// it stays bound to corner_service_not_implemented. Its disposition is a
// WP-B.5 cleanup item, recorded in the handoff.
//
// -------------------------------------------------------------------------
// §6 smag_corner AND ITS a2b_ord4 DEPENDENCY
// -------------------------------------------------------------------------
// smag_corner (sw_core.F90:1857-1947) needs a2b_ord4 to move the A-grid shear
// strain to cell corners. a2b_ord4 is ALREADY implemented by WP-D as
// operators::launch_a2b_ord4_global_cuda, so launch_smag_corner_cuda consumes
// it instead of transcribing a second copy. That is the only cross-work-package
// call in this file and it is why fv3_wp_b_cuda links fv3_wp_d_cuda.
//
// Two authoritative asymmetries that a symmetric-looking port would lose:
//
//   * the tension-strain sign pattern is (vt(i,j-1)-vt(i,j)-ut(i-1,j)+ut(i,j)),
//     with the ut terms sign-flipped relative to divergence_corner's
//     (vf(i,j-1)-vf(i,j)+uf(i-1,j)-uf(i,j));
//   * smag_c is scaled by rarea_c BEFORE the four corner removals, whereas
//     divergence_corner applies its corner removals BEFORE the rarea_c scale.
//     The corner values therefore differ by a factor of rarea_c between the
//     two routines. This is not a typo in the Fortran; it is reproduced.
//
// smag_corner's doc comment says "work only if (grid_type==4)". The code has
// no grid_type test at all, and d_sw calls it for every grid_type whenever
// nord > 0 .and. dddmp >= 1.E-5. The comment is stale; the behaviour ported
// here is the code's, and grid_type == 0 reaches it through a2b_ord4's
// cubed-sphere arm.
//
// -------------------------------------------------------------------------
// §7 DATA / STAGGERING CONTRACT
// -------------------------------------------------------------------------
// All prognostic/diagnostic fields are View3 with k batched over [0, nk) and
// nk == npz (CELL levels -- the divergence-damping block has no interface
// level). Grid metrics are k-independent View2, except sin_sg which is View3
// indexed (i, j, component) with the component 0-BASED (Fortran sin_sg(:,:,4)
// is component 3).
//
//  field       | Fortran meaning                       | stagger | mode | i extent   | j extent   | lifetime
// -------------|---------------------------------------|---------|------|------------|------------|-------------------
//  u           | D-grid zonal wind                     | D(u)    | R/W  | isd..ied   | jsd..jed+1 | persistent state
//  v           | D-grid meridional wind                | D(v)    | R/W  | isd..ied+1 | jsd..jed   | persistent state
//  ua,va       | A-grid winds from d2a2c_vect          | A       | R    | isd..ied   | jsd..jed   | WP-A output
//  uc,vc       | C-grid winds                          | C       | R/W  | see (a)    | see (a)    | WP-A output, CLOBBERED by H2
//  ut,vt       | contravariant C-winds -> u*dx/v*dy    | see 3A.6| R/W  | isd..ied+1 | jsd..jed   | d_sw-local, three meanings
//              |   -> del6 fluxes fx2/fy2              |         |      |            |            |
//  divg_d      | corner divergence                     | B       | R/W  | isd..ied+1 | jsd..jed+1 | persistent, halo-exchanged
//  delpc       | saved divergence (external-mode filter)| B      | W    | isd..ied   | jsd..jed   | d_sw output
//  ptc         | branch-L u-based corner flux          | D(u)    | W    | isd..ied   | jsd..jed   | d_sw output
//  vort        | branch-L v term / damping increment / | B then A| R/W  | isd..ied   | jsd..jed   | d_sw-local, reused
//              | smag_c / absolute vorticity / del6 d2 |         |      |            |            |
//  ke          | kinetic energy on B points            | B       | R/W  | isd..ied+1 | jsd..jed+1 | d_sw-local
//  wk          | relative vorticity                    | A       | R/W  | isd..ied   | jsd..jed   | d_sw-local
//  ub,vb       | B-grid scratch (5th and 6th meaning)  | B       | R/W  | is..ie+1   | js..je+1   | d_sw-local
//  fx,fy       | vorticity fluxes -> u*rdx / v*rdy     | faces   | R/W  | is..ie+1   | js..je+1   | d_sw-local
//  gx,gy       | flux * wind product for the heating   | faces   | W    | is..ie+1   | js..je+1   | d_sw-local
//  delp        | layer thickness                       | A       | R    | isd..ied   | jsd..jed   | persistent state
//  heat_source | damping heat                          | A       | R/W  | is..ie     | js..je     | accumulated, see (b)
//
//  (a) uc is (isd:ied+1, jsd:jed) and vc is (isd:ied, jsd:jed+1). H2b writes
//      vc over is-1-nt..ie+1+nt x js-nt..je+1+nt and H2d writes uc over
//      is-nt..ie+1+nt x js-1-nt..je+1+nt. With nord <= 3 (so nt <= 2) both
//      stay inside the ng = 3 halo; nord > 3 would run off the array and is
//      rejected.
//  (b) heat_source is declared intent(OUT) in d_sw yet the expression reads
//      heat_source(i,j) on the right-hand side (sw_core.F90:1508), so it is
//      genuinely an accumulator. It is treated as in-out here.
//
// Precision: every field is fv_real. No mixed-precision accumulation is
// introduced.
//
// -------------------------------------------------------------------------
// §8 damp_v / VORTICITY DAMPING: WHY LevelFilter GAINED A `gate`
// -------------------------------------------------------------------------
// sw_core.F90:1476-1479 is
//
//   if ( damp_v > 1.E-5 ) then
//        damp4 = (damp_v*gridstruct%da_min_c)**(nord_v+1)
//        call del6_vt_flux(nord_v, npx, npy, damp4, wk, vort, ut, vt, ...)
//
// so the ACTIVATION value (damp_v) and the COEFFICIENT (damp4) are different
// numbers -- unlike update_dz_d, where del6_vt_flux is gated on the very
// coefficient it is handed. LevelFilter::gate carries the activation array
// separately; when it is empty the filter falls back to `damp`, so Milestone
// 3A.4's height path is bit-for-bit unaffected. launch_vorticity_damping_cuda
// documents the argument aliasing (q = wk, d2 = vort, fx2 = ut, fy2 = vt),
// which is the single easiest thing to get wrong at this call site.
// ===========================================================================

#include "dgrid/wp_b_cuda.cuh"

#include <cstdint>

namespace fv3::gpu::dgrid {

using fv3::gpu::operators::A2BGeometry;

// Scalar (genuinely k-independent) parameters of the divergence-damping block.
// Everything that dyn_core recomputes inside its k loop lives in
// DivergenceLevels instead -- see §3.
struct DivergenceDampingOptions {
  // flagstruct%grid_type. Only grid_type == 0 is supported: `fill_c` at
  // sw_core.F90:1373 tests grid_type < 3, and the >= 3 arms are different
  // algorithms that are not ported.
  int32_t grid_type = 0;
  // flagstruct%do_skeb. Rejected when non-zero. sw_core.F90:1481 widens D2's
  // gate to `d_con > 1.e-5 .or. do_skeb`, so with do_skeb and d_con <= 1.e-5
  // the Fortran reads ub/vb that D1 never wrote -- they still hold xtp_u's and
  // ytp_v's fluxes. Reproducing that stale read is possible but not
  // validatable against the production first-run configuration, which has
  // do_skeb = false, so it is rejected rather than guessed at.
  int32_t do_skeb = 0;
  // flagstruct%dddmp. Scalar; drives both the max(d2_bg, min(0.20, ...))
  // limiter and the dddmp < 1.E-5 choice between vort = 0 and smag_corner.
  fv_real dddmp = fv_real(0);
  // gridstruct%da_min and gridstruct%da_min_c.
  fv_real da_min = fv_real(0);
  fv_real da_min_c = fv_real(0);
};

// Per-level selector for the divergence-damping block, the counterpart of
// LevelFilter for quantities dyn_core recomputes per k.
//
// nord_k carries integral values through an fv_real array for the same reason
// LevelFilter::ndif does: the public C ABI admits only F32/F64 element types,
// so the values travel as exactly-representable reals and are converted with a
// nearest-integer round.
//
// All four arrays are indexed at min(k, npz-1), matching LevelFilter's clamp.
struct DivergenceLevels {
  View1<const fv_real> nord_k{};  // dyn_core.F90:702 + the sponge overrides
  View1<const fv_real> d2_bg{};   // dyn_core.F90:710 d2_divg (per level)
  View1<const fv_real> d4_bg{};   // dyn_core.F90:684 d4dmp(k)
  View1<const fv_real> d_con{};   // dyn_core.F90:719 d_con_k
  int64_t npz = 0;
  // 0 = every level
  // 1 = levels taking d_sw's nord==0 arm      (nord_k == 0)
  // 2 = levels taking the higher-order arm with nord_k == nord_sel (>= 1)
  // 3 = levels with d_con > 1.e-5
  int32_t mode = 0;
  int32_t nord_sel = 0;
};

// Grid metrics consumed by this block, mirroring the gridstruct pointers d_sw
// and smag_corner dereference. Each launcher documents which members it needs;
// unused members may be left empty.
struct DivergenceGeometry {
  View2<const fv_real> rarea_c{};  // B-grid inverse corner area
  View2<const fv_real> rarea{};    // A-grid inverse area
  View2<const fv_real> dxc{};      // (isd:ied+1, jsd:jed)
  View2<const fv_real> dyc{};      // (isd:ied,   jsd:jed+1)
  View2<const fv_real> dx{};       // (isd:ied,   jsd:jed+1)
  View2<const fv_real> dy{};       // (isd:ied+1, jsd:jed)
  View2<const fv_real> rdx{};      // 1/dx
  View2<const fv_real> rdy{};      // 1/dy
  View2<const fv_real> divg_u{};   // (isd:ied,   jsd:jed+1), vc's stagger
  View2<const fv_real> divg_v{};   // (isd:ied+1, jsd:jed),   uc's stagger
  View2<const fv_real> cosa_u{};
  View2<const fv_real> cosa_v{};
  View2<const fv_real> sina_u{};
  View2<const fv_real> sina_v{};
  View2<const fv_real> cosa_s{};   // A-grid
  View2<const fv_real> rsin2{};    // A-grid
  // gridstruct%sin_sg, component index 0-BASED: Fortran sin_sg(:,:,4) is
  // component 3, sin_sg(:,:,2) is component 1, sin_sg(:,:,3) is component 2,
  // sin_sg(:,:,1) is component 0.
  View3<const fv_real> sin_sg{};
};

// smag_corner's private workspaces. These are the routine's OWN locals
// (sw_core.F90:1868-1871) plus WP-D's a2b_ord4 scratch. They must NOT alias
// d_sw's ut/vt/wk -- see §4(d). launch_smag_corner_cuda rejects the call when
// any of them is empty.
struct SmagCornerWorkspace {
  View3<fv_real> ut{};   // (isd:ied+1, jsd:jed)
  View3<fv_real> vt{};   // (isd:ied,   jsd:jed+1)
  View3<fv_real> wk{};   // (isd:ied,   jsd:jed)
  View3<fv_real> sh{};   // (isd:ied,   jsd:jed) -- a2b_ord4 output
  View3<fv_real> qx{};   // a2b_ord4 scratch
  View3<fv_real> qy{};
  View3<fv_real> qxx{};
  View3<fv_real> qyy{};
};

// --- branch L: rows L1..L5 (sw_core.F90:1277-1358, `nord==0` arm) ---------
// One launcher, five ordered launches, all filtered to lv.mode == 1 levels.
// It is not decomposed further because the five passes are strictly
// sequential (ptc -> vort -> delpc -> corners -> scale/damp/ke) and a caller
// has no legitimate reason to interleave anything between them.
//
// Requires geometry: rarea_c, dxc, dyc, cosa_u, cosa_v, sina_u, sina_v,
// sin_sg. `uc`/`vc` are read only for the sign switch in the one-sided
// sin_sg formulas at the tile edges.
//
// Outputs: ptc, vort, delpc and ke (ke is accumulated, not overwritten).
fv3_gpu_status launch_divdamp_nord0_cuda(
    const TransportBounds &b, const DivergenceDampingOptions &opt,
    const DivergenceLevels &lv, fv_real dt, View3<const fv_real> u,
    View3<const fv_real> v, View3<const fv_real> ua, View3<const fv_real> va,
    View3<const fv_real> uc, View3<const fv_real> vc, const DivergenceGeometry &g,
    View3<fv_real> ptc, View3<fv_real> vort, View3<fv_real> delpc, View3<fv_real> ke,
    void *stream) noexcept;

// --- branch H iterate: rows H1..H2g (sw_core.F90:1360-1405) ---------------
// H1's save plus the complete n-loop. Levels are processed in nord groups:
// one group per nord value in [1, nord_cap], each selecting the levels whose
// nord_k equals that value, so a single call covers a column in which
// different levels have different nord_k. nord_cap must be the largest nord
// the caller can produce; values above 3 are rejected because nt = nord-1
// would index past the ng = 3 halo.
//
// Requires geometry: divg_u, divg_v.
//
// DESTRUCTIVE: uc and vc are overwritten with divergence differences on every
// level this call touches -- see §4(a).
fv3_gpu_status launch_divdamp_high_order_cuda(
    const TransportBounds &b, const DivergenceDampingOptions &opt,
    const DivergenceLevels &lv, int32_t nord_cap, const DivergenceGeometry &g,
    View3<fv_real> divg_d, View3<fv_real> uc, View3<fv_real> vc, View3<fv_real> delpc,
    void *stream) noexcept;

// --- row H3, dddmp < 1.E-5 arm (sw_core.F90:1408) -------------------------
// `vort(:,:) = 0.` -- the WHOLE isd..ied x jsd..jed array, not just the
// compute box. Filtered to the higher-order levels so branch L's vort
// survives; pass lv.mode = 2 with the appropriate nord_sel, or call once per
// nord group.
fv3_gpu_status launch_divdamp_vort_zero_cuda(const TransportBounds &b,
                                             const DivergenceLevels &lv,
                                             View3<fv_real> vort, void *stream) noexcept;

// --- row H3, smag_corner arm (sw_core.F90:1857-1947) ---------------------
// `dt` is the Fortran call's abs(dt) (sw_core.F90:1410); the launcher does not
// take the absolute value for you, so a caller passing a signed dt gets a
// signed smag_c and will notice.
//
// Requires geometry: rarea_c, rarea, dxc, dyc, dx, dy.
//
// The three smag_c passes (S3, S4, S9) are filtered to lv; the private
// workspace passes (S1, S2, S5, S6, S7) and the a2b_ord4 call are NOT,
// because WP-D's a2b launcher has no per-level filter and reading a
// half-initialised wk would be an initcheck error. sh on the levels this call
// does not own is computed and discarded.
fv3_gpu_status launch_smag_corner_cuda(
    const TransportBounds &b, const DivergenceDampingOptions &opt,
    const DivergenceLevels &lv, fv_real dt, View3<const fv_real> u,
    View3<const fv_real> v, const DivergenceGeometry &g, const A2BGeometry &a2b,
    const SmagCornerWorkspace &ws, View3<fv_real> smag_c, void *stream) noexcept;

// --- row H4 (sw_core.F90:1413-1421) --------------------------------------
//   dd8   = da_min * d4_bg(k)**(nord_k(k)+1)
//   damp2 = da_min_c * max(d2_bg(k), min(0.20, dddmp*vort(i,j)))
//   vort  = damp2*delpc + dd8*divg_d
//   ke   += vort
// Note there is no abs() on vort inside the limiter here, unlike branch L's
// abs(delpc*dt). vort arrives as smag_c (>= 0) or as zero.
//
// Requires no geometry. Reads d2_bg, d4_bg and nord_k from lv.
fv3_gpu_status launch_divdamp_high_order_finalize_cuda(
    const TransportBounds &b, const DivergenceDampingOptions &opt,
    const DivergenceLevels &lv, View3<const fv_real> delpc,
    View3<const fv_real> divg_d, View3<fv_real> vort, View3<fv_real> ke,
    void *stream) noexcept;

// --- fill_corners, BGRID scalar (tools/fv_mp_mod.F90:1113-1186) ----------
// The LOCAL ng x ng corner rotation for a B-grid (CORNER-positioned) scalar.
// dir = 1 is FILL=XDir, dir = 2 is FILL=YDir; the Fortran's `case default`
// is identical to XDir. Corner ownership follows the Fortran predicates
// exactly: is==1 / ie==npx-1 / js==1 / je==npy-1 (fv_mp_mod.F90:1125-1128).
// The DGRID variant at :1351-1354 spells the same two conditions as
// ie+1==npx / je+1==npy; those are arithmetically identical, so the two
// variants agree on which rank owns which corner and TransportBounds'
// east_edge/north_edge flags serve both without reinterpretation.
//
// No-op when b.nested or !b.cubed_sphere, mirroring `fill_c`'s guard.
fv3_gpu_status launch_fill_corners_bgrid_cuda(const TransportBounds &b, int32_t dir,
                                              const DivergenceLevels &lv,
                                              View3<fv_real> q, void *stream) noexcept;

// --- fill_corners, DGRID vector pair (tools/fv_mp_mod.F90:1338-1370) -----
// `call fill_corners(vc, uc, npx, npy, VECTOR=.true., DGRID=.true.)`, so
// x = vc, y = uc and mySign = -1, which the Fortran applies to the SW and NE
// corners only (fv_mp_mod.F90:1351-1354 for x, :1363-1366 for y).
//
// Those two ranges are the LIVE code. The lines immediately above each of them
// -- :1347-1350 and :1359-1362 -- are a commented-out D-grid variant that puts
// the signs on NW/SE instead and reads different indices. It is dead, and it is
// a genuine trap: its sign placement coincides with the LIVE C-grid routine
// (fill_corners_cgrid_r8, :1450-1474), so a reader who lands on it can find
// apparent corroboration for the wrong convention. Cite :1351/:1363, not
// :1350/:1362.
//
// The two Fortran loops are kept as TWO sequential kernel launches on the same
// stream (fill_corners_dgrid_x_kernel then fill_corners_dgrid_y_kernel), in
// the Fortran's order. They are not fused. Fusing would in fact be safe --
// loop 2 reads x only at first index in 1..ng or npx-ng..npx-1, while loop 1
// writes x only at first index <= 0 or >= npx, so the sets are disjoint and
// the sequencing is not numerically load-bearing -- but the launches are left
// separate so that the port has the same statement order as the authoritative
// source and does not depend on that disjointness argument staying true.
fv3_gpu_status launch_fill_corners_dgrid_vector_cuda(const TransportBounds &b,
                                                     const DivergenceLevels &lv,
                                                     View3<fv_real> x, View3<fv_real> y,
                                                     void *stream) noexcept;

// --- row D1 (sw_core.F90:1425-1436) --------------------------------------
// ub(i,j) = vort(i,j) - vort(i+1,j)   over is..ie   x js..je+1
// vb(i,j) = vort(i,j) - vort(i,j+1)   over is..ie+1 x js..je
// Filtered to lv (pass mode = 3 for the `d_con > 1.e-5` gate).
fv3_gpu_status launch_dcon_vort_diff_cuda(const TransportBounds &b,
                                          const DivergenceLevels &lv,
                                          View3<const fv_real> vort, View3<fv_real> ub,
                                          View3<fv_real> vb, void *stream) noexcept;

// --- row D2 (sw_core.F90:1482-1495) --------------------------------------
// ub = (ub + vt)*rdx ; fy = u*rdx ; gy = fy*ub    over is..ie   x js..je+1
// vb = (vb - ut)*rdy ; fx = v*rdy ; gx = fx*vb    over is..ie+1 x js..je
// Requires geometry: rdx, rdy. ut/vt must hold whatever meaning §4(g)
// describes for the caller's damp_v.
fv3_gpu_status launch_dcon_scale_cuda(const TransportBounds &b,
                                      const DivergenceLevels &lv,
                                      const DivergenceGeometry &g, View3<const fv_real> u,
                                      View3<const fv_real> v, View3<const fv_real> ut,
                                      View3<const fv_real> vt, View3<fv_real> ub,
                                      View3<fv_real> vb, View3<fv_real> fx,
                                      View3<fv_real> fy, View3<fv_real> gx,
                                      View3<fv_real> gy, void *stream) noexcept;

// --- row D3 (sw_core.F90:1497-1520) --------------------------------------
// The total-energy-conserving conversion of divergence-damping KE loss to
// heat, over is..ie x js..je. damp = 0.25*d_con(k) is formed per level.
// heat_source is an accumulator (§7(b)). diss_est is NOT produced: it exists
// only under do_skeb, which this milestone rejects.
//
// Requires geometry: rsin2, cosa_s.
fv3_gpu_status launch_dcon_heat_source_cuda(
    const TransportBounds &b, const DivergenceDampingOptions &opt,
    const DivergenceLevels &lv, const DivergenceGeometry &g, View3<const fv_real> delp,
    View3<const fv_real> ub, View3<const fv_real> vb, View3<const fv_real> fx,
    View3<const fv_real> fy, View3<const fv_real> gx, View3<const fv_real> gy,
    View3<fv_real> heat_source, void *stream) noexcept;

// --- row V1 (sw_core.F90:1476-1479) --------------------------------------
// Thin, deliberately non-numerical adapter over Milestone 3A.4's
// launch_del6_vt_flux_cuda. It exists to pin down the argument aliasing that
// the Fortran call site hides:
//
//   del6_vt_flux(nord_v, npx, npy, damp4, wk, vort, ut, vt)
//        q  = wk    (relative vorticity, mutated in place by copy_corners)
//        d2 = vort  (OVERWRITTEN -- the damping increment is gone afterwards)
//        fx2 = ut   (THIRD meaning of ut)
//        fy2 = vt   (THIRD meaning of vt)
//
// `lf.gate` must carry damp_v(k) and `lf.damp` must carry
// (damp_v(k)*da_min_c)**(nord_v(k)+1); `lf.ndif` carries nord_v(k) and
// lf.mode must be 3 (the launcher iterates nord groups itself). See §8.
fv3_gpu_status launch_vorticity_damping_cuda(
    const TransportBounds &b, int32_t nord_cap, const LevelFilter &lf, View3<fv_real> wk,
    View2<const fv_real> del6_u, View2<const fv_real> del6_v, View2<const fv_real> rarea,
    View3<fv_real> vort, View3<fv_real> ut, View3<fv_real> vt, void *stream) noexcept;

// --- row V2 (sw_core.F90:1522-1534) --------------------------------------
// u(i,j) += vt(i,j)   over is..ie   x js..je+1
// v(i,j) -= ut(i,j)   over is..ie+1 x js..je
// Gated per level by `damp_v > 1.E-5`: pass a LevelFilter with mode = 1 and
// gate = damp_v(k).
fv3_gpu_status launch_dwind_damping_add_cuda(const TransportBounds &b,
                                             const LevelFilter &lf,
                                             View3<const fv_real> ut,
                                             View3<const fv_real> vt, View3<fv_real> u,
                                             View3<fv_real> v, void *stream) noexcept;

} // namespace fv3::gpu::dgrid
