#include "fv3_vertical_column.hpp"
#include <cassert>
#include <cmath>

using fv3::gpu::vertical::Column;

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
}
