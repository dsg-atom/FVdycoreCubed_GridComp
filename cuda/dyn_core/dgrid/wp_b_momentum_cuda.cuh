#pragma once

// ===========================================================================
// Milestone 3A.6 / WP-B.3 -- authoritative FV3 D-grid momentum transport.
//
// Authoritative source: GEOS-ESM/GFDL_atmos_cubed_sphere, tag geos/v3.0.0,
// SHA ca30e26305fda43831095334c02a3f5e0b8c0c45, model/sw_core.F90.
//
// SCOPE. This header covers exactly the momentum slice of d_sw: the two
// B-grid contravariant wind constructions, xtp_u, ytp_v, the kinetic-energy
// terms that consume their output, the vorticity precursor, and the D-wind
// update. It deliberately stops at the divergence-damping boundary. It is NOT
// the d_sw driver; nothing here is wired into NumericalDispatch.
//
// -------------------------------------------------------------------------
// §1 ORDERED OPERATION TABLE (d_sw momentum block, sw_core.F90:1055-1478)
// -------------------------------------------------------------------------
// Every row is one operation in authoritative order. "stagger" is the grid
// location of the row's OUTPUT. Halo/edge columns state what the row reads
// beyond the compute box.
//
//  # | Fortran            | routine / expression        | in                                   | out       | stagger | halo read              | edge/corner dep                  | workspace     | next consumer
// ---|--------------------|-----------------------------|--------------------------------------|-----------|---------|------------------------|----------------------------------|---------------|---------------
//  1 | sw_core.F90:1077   | vb construction             | vc,uc,cosa,rsina,vt                  | vb        | B (i,j) | vc(i-1),uc(i,j-1)      | js==1 / je+1==npy 2-pt vt row;   | vb            | (2) ytp_v
//    |         -1118      |   launch_bgrid_vb_cuda      |                                      |           |         | vt(-1..npx+1)          | is==1 / ie+1==npx 4-pt vt column |               |
//  2 | sw_core.F90:2320   | ytp_v(vb -> ub)             | vb(=c), v, dy, rdy                   | ub(=flux) | B (i,j) | v(j-3..j+2), dy 2 ghost| south/north one-sided PPM +      | bl,br,b0,smt6 | (3) ke, (5) xtp_u
//    |         (:1120)    |   launch_ytp_v_cuda         |                                      |           |         |                        | i==1/i==npx bl=br=0 zeroing      |               |
//  3 | sw_core.F90:1123   | ke = vb*ub                  | vb, ub                                | ke        | B (i,j) | none                   | none                             | ke            | (6)
//    |                    |   launch_ke_product_cuda    |                                      |           |         |                        |                                  |               |
//  4 | sw_core.F90:1126   | ub construction             | uc,vc,cosa,rsina,ut                  | ub        | B (i,j) | uc(i,j-1),vc(i-1,j)    | j==1 / j==npy 4-pt ut row;       | ub (REUSED)   | (5) xtp_u
//    |         -1167      |   launch_bgrid_ub_cuda      |                                      |           |         | ut(j-2..j+1)           | is==1 / ie+1==npx 2-pt ut column |               |
//  5 | sw_core.F90:1950   | xtp_u(ub -> vb)             | ub(=c), u, dx, rdx                   | vb(=flux) | B (i,j) | u(i-3..i+2), dx 2 ghost| west/east one-sided PPM +        | bl,br,b0,smt5 | (6)
//    |         (:1169)    |   launch_xtp_u_cuda         |                                      |           |         |                        | j==1/j==npy bl=br=0 zeroing      | (REUSED)      |
//  6 | sw_core.F90:1172   | ke = 0.5*(ke + ub*vb)       | ke, ub, vb                            | ke        | B (i,j) | none                   | none                             | ke            | (7)
//    |                    |   launch_ke_average_cuda    |                                      |           |         |                        |                                  |               |
//  7 | sw_core.F90:1177   | KE corner fix (dt6=dt/6)    | ut,vt,u,v                             | ke        | B corner| ut(1,0),vt(0,1),u(0,1) | the four tile corners only       | ke            | (10)
//    |         -1200      |   launch_ke_corner_fix_cuda |                                      |           |         |                        |                                  |               |
//  8 | sw_core.F90:1205   | vorticity precursor         | u,v,dx,dy,rarea                       | vt,ut,wk  | see note| full isd..ied+1 halo   | none                             | vt,ut,wk      | (9),(10)
//    |         -1235      |   launch_vorticity_precursor_cuda | OVERWRITES ut/vt (anti-dependence on 1,4,7)      |         |                        |                                  |               |
//  9 | sw_core.F90:1440   | vort = wk + f0[*z_rat]      | wk, f0, z_rat                         | vort      | A (i,j) | full isd..ied halo     | none                             | vort          | fv_tp_2d
//    |                    |   launch_vorticity_absolute_cuda |                                 |           |         |                        |                                  |               |
// 9b | sw_core.F90:1456   | fv_tp_2d(vort, hord_vt)     | vort,crx_adv,cry_adv,xfx_adv,yfx_adv  | fx,fy     | faces   | (see WP-B.2)           | (see WP-B.2)                     | (see WP-B.2)  | (10)
//    |                    |   REUSED: launch_scalar_transport_cuda (Milestone 3A.5) -- no new code                                                                     |
// 10 | sw_core.F90:1466   | D-wind update               | vt,ut,ke,fx,fy                        | u,v       | D       | ke(i+1,j),ke(i,j+1)    | none                             | -             | WP-B.4/B.5
//    |         -1478      |   launch_dwind_update_cuda  |                                      |           |         |                        |                                  |               |
//
// NOTE on row 8 staggering: vt(i,j)=u(i,j)*dx(i,j) is written over
// jsd..jed+1 x isd..ied (u's D stagger) and ut(i,j)=v(i,j)*dy(i,j) over
// jsd..jed x isd..ied+1 (v's D stagger); wk is A-grid over isd..ied x
// jsd..jed. Row 8 therefore does NOT have a single output stagger and is
// launched as three separate index spaces.
//
// -------------------------------------------------------------------------
// §2 DATA / STAGGERING CONTRACT
// -------------------------------------------------------------------------
// Fortran d_sw operates on 2-D slices; dyn_core calls it inside a k loop. All
// prognostic/diagnostic fields below are therefore View3 with k batched over
// [0, nk) and nk == npz (CELL levels -- momentum has no interface level, unlike
// Milestone 3A.4's zh path where nk == npz+1). Grid metrics are k-independent
// and stay View2.
//
//  field  | Fortran meaning                        | stagger | mode  | i extent    | j extent    | halo | lifetime
// --------|----------------------------------------|---------|-------|-------------|-------------|------|------------------
//  u      | D-grid zonal wind                      | D (u)   | R/W   | isd..ied    | jsd..jed+1  | ng   | persistent state
//  v      | D-grid meridional wind                 | D (v)   | R/W   | isd..ied+1  | jsd..jed    | ng   | persistent state
//  uc     | C-grid zonal wind                      | C (u)   | R     | isd..ied+1  | jsd..jed    | ng   | WP-A output
//  vc     | C-grid meridional wind                 | C (v)   | R     | isd..ied    | jsd..jed+1  | ng   | WP-A output
//  ut     | contravariant C-u, THEN v*dy           | see note| R/W   | isd..ied+1  | jsd..jed    | ng   | d_sw-local, dual meaning
//  vt     | contravariant C-v, THEN u*dx            | see note| R/W   | isd..ied    | jsd..jed+1  | ng   | d_sw-local, dual meaning
//  ub     | B-grid x-wind, THEN xtp_u's c           | B       | R/W   | is..ie+1    | js..je+1    | 0    | d_sw-local, reused
//  vb     | B-grid y-wind, THEN xtp_u's flux        | B       | R/W   | is..ie+1    | js..je+1    | 0    | d_sw-local, reused
//  ke     | kinetic energy on B points              | B       | R/W   | isd..ied+1  | jsd..jed+1  | ng   | d_sw-local
//  wk     | relative vorticity (rarea-scaled curl)  | A       | W     | isd..ied    | jsd..jed    | ng   | d_sw-local
//  vort   | absolute vorticity                      | A       | W     | isd..ied    | jsd..jed    | ng   | d_sw-local
//  fx,fy  | vorticity fluxes from fv_tp_2d           | faces   | R     | is..ie+1    | js..je+1    | 0    | d_sw-local
//  cosa   | gridstruct%cosa                          | B       | R     | isd..ied+1  | jsd..jed+1  | ng   | grid metric
//  rsina  | gridstruct%rsina                         | B       | R     | isd..ied+1  | jsd..jed+1  | ng   | grid metric
//  dx,rdx | gridstruct%dx / rdx                      | D (u)   | R     | isd..ied    | jsd..jed+1  | ng   | grid metric
//  dy,rdy | gridstruct%dy / rdy                      | D (v)   | R     | isd..ied+1  | jsd..jed    | ng   | grid metric
//  rarea  | gridstruct%rarea                         | A       | R     | isd..ied    | jsd..jed    | ng   | grid metric
//  f0     | gridstruct%f0 (Coriolis)                 | A       | R     | isd..ied    | jsd..jed    | ng   | grid metric
//  z_rat  | do_f3d Coriolis scaling                  | A       | R     | isd..ied    | jsd..jed    | ng   | per-substep
//
// ut/vt dual meaning: on entry to the momentum block they hold the C-grid
// CONTRAVARIANT winds computed earlier in d_sw (rows 1, 4 and 7 read them in
// that meaning). Row 8 OVERWRITES them with u*dx / v*dy, and row 10 reads them
// in the second meaning. This is a real anti-dependence in the Fortran, so the
// launchers are kept separate and are NOT fused across row 8.
//
// ub/vb reuse chain: vb(1) -> ytp_v writes ub(2) -> ke=vb*ub(3) -> ub is
// OVERWRITTEN(4) -> xtp_u writes vb(5) -> ke uses both(6). Row 3 MUST complete
// before row 4 and row 6 MUST see row 5's vb. Preserved by launch ordering.
//
// Precision: every field is fv_real (R8 double / R4 float). No mixed-precision
// accumulation is introduced -- the Fortran accumulates in the model's working
// precision and so does this.
// ===========================================================================

#include "dgrid/wp_b_cuda.cuh"

#include <cstdint>

namespace fv3::gpu::dgrid {

// Non-numerical parameters of the momentum slice. Kept separate from
// TransportBounds (which carries geometry/topology only) and from
// ScalarTransportOptions (whose hord/mass/deln fields have no meaning here).
struct MomentumOptions {
  // flagstruct%hord_mt. Signed, mirroring the Fortran namelist. xtp_u and
  // ytp_v contain NO iord<0 branch -- unlike xppm/yppm, which add an al>=0
  // clamp -- so hord_mt = -6 executes bit-identically to +6 here. The
  // launchers accept abs(hord_mt)==6 and reject everything else rather than
  // silently running a different limiter.
  int32_t hord_mt = 0;
  // flagstruct%lim_fac. Read only by the iord==1 limiter, carried so the
  // contract is complete and so a future iord==1 bring-up needs no signature
  // change. Unused at abs(hord)==6.
  fv_real lim_fac = fv_real(1);
  // flagstruct%grid_type. Only grid_type==0 (the production first-run
  // configuration) is supported; the >=3 doubly-periodic branches and the
  // nested branches are rejected.
  int32_t grid_type = 0;
  // flagstruct%do_f3d: selects f0*z_rat instead of f0 in row 9.
  int32_t do_f3d = 0;
};

// Per-row workspaces for xtp_u / ytp_v. The Fortran keeps bl/br/b0/smt5 as
// thread-private automatic arrays inside a j (resp. i) loop; on the GPU the
// flux loop reads index i-1 (resp. j-1) of all four, so they must be
// materialized. They are caller-supplied for the same reason WP-B.2's
// q_i/fx2_tp/d2 are: the persistent-residency architecture owns the arena, not
// the kernel.
//
//   xtp_u: bl,br,b0,smt over (is-1 .. ie+1) x (js .. je+1)
//   ytp_v: bl,br,b0,smt over (is   .. ie+1) x (js-1 .. je+1)
//
// `smt` is int32 rather than fv_real because it is a Fortran LOGICAL; storing
// it as a float would invite an == comparison against a rounded value.
struct PpmMomentumWorkspace {
  View3<fv_real> bl{};
  View3<fv_real> br{};
  View3<fv_real> b0{};
  View3<int32_t> smt{};
};

// --- row 1: vb construction (sw_core.F90:1077-1118) ----------------------
// vb(i,j) = dt5*(vc(i-1,j)+vc(i,j) - (uc(i,j-1)+uc(i,j))*cosa(i,j))*rsina(i,j)
// over the clipped interior, with the authoritative 2-point vt rows at the
// south/north tile edges and the 4-point vt columns at the west/east edges.
// dt5 = 0.5*dt is formed inside the launcher, exactly as d_sw forms it.
fv3_gpu_status launch_bgrid_vb_cuda(const TransportBounds &b,
                                    const MomentumOptions &opt, fv_real dt,
                                    View3<const fv_real> uc, View3<const fv_real> vc,
                                    View3<const fv_real> ut, View3<const fv_real> vt,
                                    View2<const fv_real> cosa, View2<const fv_real> rsina,
                                    View3<fv_real> vb, void *stream) noexcept;

// --- row 4: ub construction (sw_core.F90:1126-1167) ----------------------
// NOT the transpose of row 1. The 4-point one-sided formula sits on the
// SOUTH/NORTH rows here (j==1 / j==npy) and on the WEST/EAST columns there,
// and the 2-point formula correspondingly swaps. See §5 of the handoff.
fv3_gpu_status launch_bgrid_ub_cuda(const TransportBounds &b,
                                    const MomentumOptions &opt, fv_real dt,
                                    View3<const fv_real> uc, View3<const fv_real> vc,
                                    View3<const fv_real> ut, View3<const fv_real> vt,
                                    View2<const fv_real> cosa, View2<const fv_real> rsina,
                                    View3<fv_real> ub, void *stream) noexcept;

// --- row 5: xtp_u (sw_core.F90:1950-2317) --------------------------------
// `v` is in the Fortran argument list but is never referenced in the routine
// body at any iord, so it is not taken here. Adding it would fabricate a data
// dependence the authoritative code does not have.
fv3_gpu_status launch_xtp_u_cuda(const TransportBounds &b, const MomentumOptions &opt,
                                 View3<const fv_real> c, View3<const fv_real> u,
                                 View2<const fv_real> dx, View2<const fv_real> rdx,
                                 const PpmMomentumWorkspace &ws, View3<fv_real> flux,
                                 void *stream) noexcept;

// --- row 2: ytp_v (sw_core.F90:2320-2792) --------------------------------
// Symmetrically, `u` is declared but never referenced in ytp_v's body.
fv3_gpu_status launch_ytp_v_cuda(const TransportBounds &b, const MomentumOptions &opt,
                                 View3<const fv_real> c, View3<const fv_real> v,
                                 View2<const fv_real> dy, View2<const fv_real> rdy,
                                 const PpmMomentumWorkspace &ws, View3<fv_real> flux,
                                 void *stream) noexcept;

// --- rows 3 / 6 / 7: kinetic energy --------------------------------------
// Row 3: ke(i,j) = vb(i,j)*ub(i,j)                    (sw_core.F90:1123)
fv3_gpu_status launch_ke_product_cuda(const TransportBounds &b, View3<const fv_real> vb,
                                      View3<const fv_real> ub, View3<fv_real> ke,
                                      void *stream) noexcept;

// Row 6: ke(i,j) = 0.5*(ke(i,j) + ub(i,j)*vb(i,j))    (sw_core.F90:1172)
fv3_gpu_status launch_ke_average_cuda(const TransportBounds &b, View3<const fv_real> ub,
                                      View3<const fv_real> vb, View3<fv_real> ke,
                                      void *stream) noexcept;

// Row 7: the four tile-corner KE overrides, dt6 = dt/6 (sw_core.F90:1177-1200).
// Applied only when the corresponding gridstruct corner flag is set and only
// for .not.nested.
fv3_gpu_status launch_ke_corner_fix_cuda(const TransportBounds &b, fv_real dt,
                                         View3<const fv_real> ut,
                                         View3<const fv_real> vt, View3<const fv_real> u,
                                         View3<const fv_real> v, View3<fv_real> ke,
                                         void *stream) noexcept;

// --- row 8: vorticity precursor (sw_core.F90:1205-1235) ------------------
// Writes vt = u*dx, ut = v*dy, then wk = rarea*(vt(i,j)-vt(i,j+1)
//                                              -ut(i,j)+ut(i+1,j)).
// vt/ut are IN-OUT because this destroys their contravariant-wind contents.
// The wk pass is a separate kernel launch so it observes the completed vt/ut.
fv3_gpu_status launch_vorticity_precursor_cuda(const TransportBounds &b,
                                               View3<const fv_real> u,
                                               View3<const fv_real> v,
                                               View2<const fv_real> dx,
                                               View2<const fv_real> dy,
                                               View2<const fv_real> rarea,
                                               View3<fv_real> ut, View3<fv_real> vt,
                                               View3<fv_real> wk, void *stream) noexcept;

// --- row 9: absolute vorticity (sw_core.F90:1440-1451) -------------------
// vort = wk + f0, or wk + f0*z_rat when do_f3d. The hydrostatic and
// non-hydrostatic .not.do_f3d branches are the same expression in the
// authoritative source, so one launcher covers both.
fv3_gpu_status launch_vorticity_absolute_cuda(const TransportBounds &b,
                                              const MomentumOptions &opt,
                                              View3<const fv_real> wk,
                                              View2<const fv_real> f0,
                                              View3<const fv_real> z_rat,
                                              View3<fv_real> vort, void *stream) noexcept;

// --- row 10: D-wind update (sw_core.F90:1466-1478) -----------------------
// u(i,j) = vt(i,j) + ke(i,j) - ke(i+1,j) + fy(i,j)   over is..ie,   js..je+1
// v(i,j) = ut(i,j) + ke(i,j) - ke(i,j+1) - fx(i,j)   over is..ie+1, js..je
//
// `ke` is an input view rather than something this slice re-derives, so WP-B.5
// can pass the divergence-damping-augmented ke (ke = ke + vort, sw_core.F90
// :1462) without any signature change. ut/vt must already hold their row-8
// meaning.
fv3_gpu_status launch_dwind_update_cuda(const TransportBounds &b,
                                        View3<const fv_real> ut, View3<const fv_real> vt,
                                        View3<const fv_real> ke, View3<const fv_real> fx,
                                        View3<const fv_real> fy, View3<fv_real> u,
                                        View3<fv_real> v, void *stream) noexcept;

} // namespace fv3::gpu::dgrid
