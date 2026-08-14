#include "fv3_vertical_column.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

using fv3::gpu::vertical::Column;

template <class T>
bool finite_negative(T val) { return std::isfinite(val) && val < 0; }

int main() {
  double q1[] = {1., 2., 3., 4.};
  double q2[] = {-1., -2., -3., -4.};
  double e1[5]{}, e2[5]{}, gamma[4]{}, dp[] = {1., 1., 1., 1.};
  fv3::gpu::vertical::edge_profile_column(
      Column<const double>{q1, 1}, Column<const double>{q2, 1},
      Column<double>{e1, 1}, Column<double>{e2, 1}, gamma, dp, 4, true, 0);
  for (int k = 0; k < 5; ++k) assert(std::isfinite(e1[k]) && e2[k] == -e1[k]);

  double zh[] = {8., 6., 5., 0.}, dz[3]{};
  const double ws = fv3::gpu::vertical::update_height_boundary_column(
      Column<double>{zh, 1}, Column<double>{dz, 1}, 1., 0.5, 2., 3);
  assert(ws == 0.5);
  assert(zh[3] == 0. && zh[2] == 5. && zh[1] == 7. && zh[0] == 9.);
  assert(dz[0] == -2. && dz[1] == -2. && dz[2] == -5.);

  double out[4]{}, thickness[] = {-2., -3., -4.};
  fv3::gpu::vertical::reconstruct_height_column(
      Column<double>{out, 1}, Column<const double>{thickness, 1}, 10., 3);
  assert(out[3] == 10. && out[2] == 14. && out[1] == 17. && out[0] == 19.);

  constexpr int n=2;
  double dm[n]={100.,120.},pm[n]={900.,1010.},pem[n+1]={10.,110.,230.};
  double sw[n]={.1,.2},sdz[n]={-20.,-22.},spt[n]={300.,301.},gm[n]={1.4,1.4},cp[n]={.286,.286};
  double pe[n+1]{},aa[n+1]{},bb[n+1]{},dd[n+1]{},w1[n+1]{},gr[n+1]{},gam[n+1]{},pp[n+1]{},bet[n+1]{},p1[n+1]{};
  fv3::gpu::vertical::sim1_column(.1,287.,1.4,.286,.1,false,gm,cp,dm,pm,pem,sw,sdz,spt,0.,pe,aa,bb,dd,w1,gr,gam,pp,bet,p1,n);
  assert(pe[0]==0. && std::isfinite(pe[2]) && sdz[0]<0. && sdz[1]<0.);

  double rdelp[n]={100.,120.},rpt[n]={300.,301.},rq[n]={.01,.02},rc[n]={.286,.286};
  double rzh[n+1]={30.,10.,0.},rw[n]={1.,2.},rdm[n+1]{},rpm[n+1]{},rpem[n+1]{},rlog[n+1]{},rpk3[n+1]{},rgm[n+1]{},rcp[n+1]{},rdz[n+1]{},rw2[n+1]{},rpeg[n+1]{},rpelng[n+1]{};
  fv3::gpu::vertical::riem3_precompute_column(rdelp,rpt,rq,rc,rzh,rw,10.,.286,9.8,true,true,rdm,rpm,rpem,rlog,rpk3,rgm,rcp,rdz,rw2,rpeg,rpelng,n);
  assert(rpem[0]==10. && rpem[1]==110. && rpem[2]==230. && rdz[0]==-20. && rdz[1]==-10.);
  double rpe2[n+1]={0.,1.,2.},rdelz[n]{},routzh[n+1]{},rpe[n+1]{},rpeln[n+1]{},rpk[n+1]{},rppe[n+1]{};
  fv3::gpu::vertical::riem3_finalize_column(rpem,rlog,rpe2,rdz,rw,rw2,rdelz,routzh,5.,rpe,rpeln,rpk,rpk3,true,true,false,rppe,n);
  assert(rpe[2]==230. && rppe[2]==232. && routzh[2]==5. && routzh[1]==15. && routzh[0]==35.);

  printf("Basic tests passed. Testing SIM3p0/SIM3/SIM/RIM_2D solvers...\n");

  constexpr int km = 4;
  double s3_dm[km] = {1000., 2000., 3000., 4000.};
  double s3_pt[km] = {300., 295., 290., 285.};
  double s3_pem[km+1] = {1000., 2000., 4000., 7000., 11000.};
  double s3_w[km] = {0.1, 0.2, 0.15, 0.05};
  double s3_dz[km] = {-1000., -1500., -2000., -2500.};
  double s3_pe2[km+1]{}, s3_aa[km+1]{}, s3_bb[km+1]{}, s3_dd[km+1]{};
  double s3_w1[km+1]{}, s3_gr[km+1]{}, s3_gam[km+1]{}, s3_pp[km+1]{};
  double s3_bet[km+1]{}, s3_p1[km+1]{};

  fv3::gpu::vertical::sim3p0_column(
      10., 287., 1.4, 0.286, 0.1, 0.0,
      s3_dm, s3_pt, s3_pem, s3_w, s3_dz, 0.01,
      s3_pe2, s3_aa, s3_bb, s3_dd, s3_w1, s3_gr, s3_gam, s3_pp, s3_bet, s3_p1, km);

  assert(s3_pe2[0] == 0.);
  for (int k = 0; k < km; ++k) {
    assert(finite_negative(s3_dz[k]));
    assert(std::isfinite(s3_w[k]));
  }
  printf("  SIM3p0 solver: PASS\n");

  double s3b_dm[km] = {1000., 2000., 3000., 4000.};
  double s3b_pt[km] = {300., 295., 290., 285.};
  double s3b_pem[km+1] = {1000., 2000., 4000., 7000., 11000.};
  double s3b_w[km] = {0.1, 0.2, 0.15, 0.05};
  double s3b_dz[km] = {-1000., -1500., -2000., -2500.};
  double s3b_pe2[km+1]{}, s3b_aa[km+1]{}, s3b_bb[km+1]{}, s3b_dd[km+1]{};
  double s3b_w1[km+1]{}, s3b_wk[km+1]{}, s3b_gr[km+1]{}, s3b_gam[km+1]{};
  double s3b_pp[km+1]{}, s3b_bet[km+1]{}, s3b_p1[km+1]{};

  fv3::gpu::vertical::sim3_column(
      10., 287., 1.4, 0.286, 0.75, 0.1, 0.0,
      s3b_dm, s3b_pt, s3b_pem, s3b_w, s3b_dz, 0.01,
      s3b_pe2, s3b_aa, s3b_bb, s3b_dd, s3b_w1, s3b_wk, s3b_gr, s3b_gam,
      s3b_pp, s3b_bet, s3b_p1, km);

  assert(std::isfinite(s3b_pe2[km]));
  for (int k = 0; k < km; ++k) {
    assert(finite_negative(s3b_dz[k]));
    assert(std::isfinite(s3b_w[k]));
  }
  printf("  SIM3 solver (alpha=0.75): PASS\n");

  double sim_dm[km] = {1000., 2000., 3000., 4000.};
  double sim_pm[km] = {1500., 3000., 5500., 9000.};
  double sim_pt[km] = {300., 295., 290., 285.};
  double sim_pem[km+1] = {1000., 2000., 4000., 7000., 11000.};
  double sim_gm[km] = {1.4, 1.4, 1.4, 1.4};
  double sim_cp[km] = {0.286, 0.286, 0.286, 0.286};
  double sim_w[km] = {0.1, 0.2, 0.15, 0.05};
  double sim_dz[km] = {-1000., -1500., -2000., -2500.};
  double sim_pe2[km+1]{}, sim_aa[km+1]{}, sim_bb[km+1]{}, sim_dd[km+1]{};
  double sim_w1[km+1]{}, sim_wk[km+1]{}, sim_gr[km+1]{}, sim_gam[km+1]{};
  double sim_pp[km+1]{}, sim_bet[km+1]{}, sim_p1[km+1]{};

  fv3::gpu::vertical::sim_column(
      10., 287., 1.4, 0.286, 0.75, 0.1, 0.0, false,
      sim_gm, sim_cp, sim_dm, sim_pm, sim_pt, sim_pem, sim_w, sim_dz, 0.01,
      sim_pe2, sim_aa, sim_bb, sim_dd, sim_w1, sim_wk, sim_gr, sim_gam,
      sim_pp, sim_bet, sim_p1, km);

  for (int k = 0; k < km; ++k) {
    assert(finite_negative(sim_dz[k]));
    assert(std::isfinite(sim_w[k]));
  }
  printf("  SIM solver (alpha=0.75, dry): PASS\n");

  double rim_dm[km] = {1000., 2000., 3000., 4000.};
  double rim_pm[km] = {1500., 3000., 5500., 9000.};
  double rim_pt[km] = {300., 295., 290., 285.};
  double rim_gm[km] = {1.4, 1.4, 1.4, 1.4};
  double rim_w[km] = {0.1, 0.2, 0.15, 0.05};
  double rim_dz[km] = {-1000., -1500., -2000., -2500.};
  double rim_pe2[km+1]{}, rim_dts[km+1]{}, rim_pf1[km+1]{};
  double rim_r_lo[km+1]{}, rim_r_hi[km+1]{};
  double rim_m_bot[km+2]{}, rim_m_top[km+2]{};
  double rim_r_bot[km+2]{}, rim_r_top[km+2]{};
  double rim_wbar[km+2]{}, rim_pbar[km+2]{};
  double rim_cm[km+1]{}, rim_wc[km+1]{}, rim_pp[km+1]{};
  double rim_dm_tmp[km+1]{}, rim_wm[km+1]{}, rim_dz_tmp[km+1]{};

  fv3::gpu::vertical::rim_2d_column(
      2, 20., 287., 1.4, false, rim_gm, rim_dm, rim_pm, rim_pt, 0.01,
      rim_dz, rim_w, rim_pe2,
      rim_dts, rim_pf1, rim_r_lo, rim_r_hi, rim_m_bot, rim_m_top,
      rim_r_bot, rim_r_top, rim_wbar, rim_pbar, rim_cm, rim_wc, rim_pp,
      rim_dm_tmp, rim_wm, rim_dz_tmp, km);

  assert(rim_pe2[0] == 0.);
  for (int k = 0; k < km; ++k) {
    assert(finite_negative(rim_dz[k]));
    assert(std::isfinite(rim_w[k]));
  }
  printf("  RIM_2D solver (ms=2): PASS\n");

  double r3_delp[km] = {1000., 2000., 3000., 4000.};
  double r3_pt[km] = {300., 295., 290., 285.};
  double r3_q_con[km] = {0.01, 0.015, 0.02, 0.025};
  double r3_cappa[km] = {0.286, 0.286, 0.286, 0.286};
  double r3_zh_in[km+1] = {7000., 5500., 3500., 1500., 0.};
  double r3_w_in[km] = {0.1, 0.2, 0.15, 0.05};

  double r3_w_out[km]{}, r3_dz_out[km]{}, r3_zh_out[km+1]{};
  double r3_pe_out[km+1]{}, r3_peln_out[km+1]{}, r3_pk_out[km+1]{};
  double r3_pk3_out[km+1]{}, r3_ppe_out[km+1]{};

  double r3_dm[km+1]{}, r3_pm[km+1]{}, r3_pem[km+1]{}, r3_peln2[km+1]{};
  double r3_pk3_tmp[km+1]{}, r3_gm[km+1]{}, r3_cp[km+1]{};
  double r3_dz[km+1]{}, r3_w2[km+1]{}, r3_peg[km+1]{}, r3_pelng[km+1]{};
  double r3_pe2[km+1]{}, r3_aa[km+1]{}, r3_bb[km+1]{}, r3_dd[km+1]{};
  double r3_w1[km+1]{}, r3_wk[km+1]{}, r3_gr[km+1]{}, r3_gam[km+1]{};
  double r3_pp[km+1]{}, r3_bet[km+1]{}, r3_p1[km+1]{};
  double r3_dts[km+1]{}, r3_pf1[km+1]{}, r3_r_lo[km+1]{}, r3_r_hi[km+1]{};
  double r3_m_bot[km+2]{}, r3_m_top[km+2]{}, r3_r_bot[km+2]{}, r3_r_top[km+2]{};
  double r3_wbar[km+2]{}, r3_pbar[km+2]{}, r3_cm[km+1]{}, r3_wc[km+1]{};
  double r3_dm_tmp[km+1]{}, r3_wm[km+1]{}, r3_dz_tmp[km+1]{};

  fv3::gpu::vertical::riem_solver3_column(
      2, 10., 287., 1.4, 0.286, 10., 0.1, 1.0, 0.0,
      true, false, true, false, false,
      r3_delp, r3_pt, r3_q_con, r3_cappa, r3_zh_in, r3_w_in,
      0., 0.01, r3_w_out, r3_dz_out, r3_zh_out,
      r3_pe_out, r3_peln_out, r3_pk_out, r3_pk3_out, r3_ppe_out,
      r3_dm, r3_pm, r3_pem, r3_peln2, r3_pk3_tmp, r3_gm, r3_cp,
      r3_dz, r3_w2, r3_peg, r3_pelng, r3_pe2, r3_aa, r3_bb, r3_dd,
      r3_w1, r3_wk, r3_gr, r3_gam, r3_pp, r3_bet, r3_p1,
      r3_dts, r3_pf1, r3_r_lo, r3_r_hi, r3_m_bot, r3_m_top,
      r3_r_bot, r3_r_top, r3_wbar, r3_pbar, r3_cm, r3_wc,
      r3_dm_tmp, r3_wm, r3_dz_tmp, km);

  assert(r3_zh_out[km] == 0.);
  assert(r3_pe_out[0] == 10.);
  for (int k = 0; k < km; ++k) {
    assert(finite_negative(r3_dz_out[k]));
    assert(std::isfinite(r3_w_out[k]));
  }
  for (int k = 0; k <= km; ++k) {
    assert(std::isfinite(r3_pk3_out[k]));
  }
  printf("  Riem_Solver3 full (a_imp=1.0, SIM1): PASS\n");

  fv3::gpu::vertical::riem_solver3_column(
      2, 10., 287., 1.4, 0.286, 10., 0.1, 0.3, 0.0,
      true, false, false, true, false,
      r3_delp, r3_pt, r3_q_con, r3_cappa, r3_zh_in, r3_w_in,
      0., 0.01, r3_w_out, r3_dz_out, r3_zh_out,
      r3_pe_out, r3_peln_out, r3_pk_out, r3_pk3_out, r3_ppe_out,
      r3_dm, r3_pm, r3_pem, r3_peln2, r3_pk3_tmp, r3_gm, r3_cp,
      r3_dz, r3_w2, r3_peg, r3_pelng, r3_pe2, r3_aa, r3_bb, r3_dd,
      r3_w1, r3_wk, r3_gr, r3_gam, r3_pp, r3_bet, r3_p1,
      r3_dts, r3_pf1, r3_r_lo, r3_r_hi, r3_m_bot, r3_m_top,
      r3_r_bot, r3_r_top, r3_wbar, r3_pbar, r3_cm, r3_wc,
      r3_dm_tmp, r3_wm, r3_dz_tmp, km);

  assert(r3_zh_out[km] == 0.);
  for (int k = 0; k < km; ++k) {
    assert(finite_negative(r3_dz_out[k]));
  }
  printf("  Riem_Solver3 full (a_imp=0.3, RIM_2D): PASS\n");

  fv3::gpu::vertical::riem_solver3_column(
      2, 10., 287., 1.4, 0.286, 10., 0.1, -1.0, 0.0,
      true, false, false, false, false,
      r3_delp, r3_pt, r3_q_con, r3_cappa, r3_zh_in, r3_w_in,
      0., 0.01, r3_w_out, r3_dz_out, r3_zh_out,
      r3_pe_out, r3_peln_out, r3_pk_out, r3_pk3_out, r3_ppe_out,
      r3_dm, r3_pm, r3_pem, r3_peln2, r3_pk3_tmp, r3_gm, r3_cp,
      r3_dz, r3_w2, r3_peg, r3_pelng, r3_pe2, r3_aa, r3_bb, r3_dd,
      r3_w1, r3_wk, r3_gr, r3_gam, r3_pp, r3_bet, r3_p1,
      r3_dts, r3_pf1, r3_r_lo, r3_r_hi, r3_m_bot, r3_m_top,
      r3_r_bot, r3_r_top, r3_wbar, r3_pbar, r3_cm, r3_wc,
      r3_dm_tmp, r3_wm, r3_dz_tmp, km);

  assert(r3_zh_out[km] == 0.);
  for (int k = 0; k < km; ++k) {
    assert(finite_negative(r3_dz_out[k]));
  }
  printf("  Riem_Solver3 full (a_imp=-1.0, SIM3p0): PASS\n");

  printf("All vertical column solver tests: PASS\n");
  return 0;
}
