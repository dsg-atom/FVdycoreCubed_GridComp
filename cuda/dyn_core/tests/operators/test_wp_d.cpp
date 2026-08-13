#include "wp_d_operators.hpp"

#include <cassert>
#include <cmath>
#include <vector>

using namespace fv3::gpu;
using namespace fv3::gpu::operators;

namespace {
constexpr int lo=-3, hi=6, nz=3, ni=hi-lo+1;
size_t off(int i,int j,int k=0) { return size_t(i-lo)+size_t(j-lo)*ni+size_t(k)*ni*ni; }
View2<fv_real> v2(std::vector<fv_real>& a) { return {a.data(),lo,lo,1,ni}; }
View2<const fv_real> cv2(const std::vector<fv_real>& a) { return {a.data(),lo,lo,1,ni}; }
View3<fv_real> v3(std::vector<fv_real>& a) { return {a.data(),lo,lo,0,1,ni,ni*ni}; }
View3<const fv_real> cv3(const std::vector<fv_real>& a) { return {a.data(),lo,lo,0,1,ni,ni*ni}; }
bool near(fv_real a,fv_real b) {
  return std::abs(a-b)<(sizeof(fv_real)==4 ? fv_real(2e-5) : fv_real(1e-11));
}
}

int main() {
  const HorizontalBounds b{0,3,0,3,nz};
  std::vector<fv_real> dp(ni*ni*nz, fv_real(2));
  std::vector<fv_real> iface(ni*ni*(nz+1), fv_real(-1));
  reconstruct_pe_halo(b,fv_real(10),cv3(dp),v3(iface));
  assert(iface[off(-1,2,3)]==fv_real(16));
  assert(iface[off(2,-1,2)]==fv_real(14));
  std::fill(iface.begin(),iface.end(),fv_real(-1));
  reconstruct_pln_halo(b,fv_real(10),cv3(dp),v3(iface));
  assert(near(iface[off(-2,1,1)],std::log(fv_real(12))));
  reconstruct_pk3_halo(b,fv_real(10),fv_real(.5),cv3(dp),v3(iface));
  assert(near(iface[off(5,1,3)],fv_real(4)));

  std::vector<fv_real> q(ni*ni,fv_real(7)), out(ni*ni), qx(ni*ni), qy(ni*ni);
  a2b_ord2_periodic(b,cv2(q),v2(out));
  assert(out[off(2,2)]==fv_real(7));
  a2b_ord4_periodic(b,cv2(q),v2(out),v2(qx),v2(qy));
  assert(near(out[off(2,2)],fv_real(7)));

  std::vector<fv_real> field(ni*ni*nz,fv_real(3)), fx(field.size()), fy(field.size());
  std::vector<fv_real> metric(ni*ni,fv_real(1));
  del2_pass(b,0,fv_real(.2),v3(field),{cv2(metric),cv2(metric),cv2(metric)},v3(fx),v3(fy));
  for(int j=0;j<=3;++j) for(int i=0;i<=3;++i) assert(field[off(i,j)]==fv_real(3));

  std::vector<fv_real> heat(field.size(),fv_real(100));
  std::vector<fv_real> delz(field.size(),fv_real(2));
  std::vector<fv_real> pkz(field.size(),fv_real(0));
  std::vector<fv_real> pt(field.size(),fv_real(300));
  std::vector<fv_real> cappa(field.size(),fv_real(.2));
  apply_diffusive_heating_nh(b,fv_real(2),fv_real(.5),fv_real(1),fv_real(1),
                             3,v3(heat),cv3(dp),cv3(delz),cv3(cappa),v3(pkz),v3(pt));
  const fv_real expected_pkz=std::exp(fv_real(.25)*std::log(fv_real(300)));
  assert(near(pkz[off(1,1,0)],expected_pkz));
  assert(near(pt[off(1,1,0)],fv_real(300)+fv_real(.1)/expected_pkz));

  // Uniform pressure/geopotential has exactly zero gradient and only applies metrics.
  std::vector<fv_real> pp(ni*ni*(nz+1),fv_real(0));
  std::vector<fv_real> gz(pp.size(),fv_real(5));
  std::vector<fv_real> pk(pp.size());
  for(int k=0;k<=nz;++k) for(int j=lo;j<=hi;++j) for(int i=lo;i<=hi;++i)
    pk[off(i,j,k)]=fv_real(10+2*k);
  std::vector<fv_real> u(field.size(),fv_real(4)),v(field.size(),fv_real(6));
  std::vector<fv_real> du(field.size(),fv_real(0)),dv(field.size(),fv_real(0));
  pressure_gradient(b,{v3(pp),v3(gz),v3(dp),v3(pk),cv2(metric),cv2(metric)},
                    {v3(u),v3(v),v3(du),v3(dv)},
                    {PressureVariant::normal_nh,fv_real(.1),fv_real(0)});
  assert(u[off(1,1,0)]==fv_real(4));
  assert(v[off(1,1,0)]==fv_real(6));
}
