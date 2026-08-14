#include "wp_d_operators.hpp"

#include <cassert>
#include <cmath>
#include <vector>

using namespace fv3::gpu;
using namespace fv3::gpu::operators;

namespace {
constexpr int lo=-3, hi=6, nz=3, ni=hi-lo+1;
size_t off(int i,int j,int k=0) { return size_t(i-lo)+size_t(j-lo)*ni+size_t(k)*ni*ni; }
size_t peoff(int i,int k,int j) { return size_t(i-lo)+size_t(k)*ni+size_t(j-lo)*ni*(nz+1); }
View2<fv_real> v2(std::vector<fv_real>& a) { return {a.data(),lo,lo,1,ni}; }
View2<const fv_real> cv2(const std::vector<fv_real>& a) { return {a.data(),lo,lo,1,ni}; }
View1<const accumulator_real> cv1(const std::vector<accumulator_real>& a) {
  return {a.data(),lo,1};
}
View3<fv_real> v3(std::vector<fv_real>& a) { return {a.data(),lo,lo,0,1,ni,ni*ni}; }
View3<const fv_real> cv3(const std::vector<fv_real>& a) { return {a.data(),lo,lo,0,1,ni,ni*ni}; }
View3<fv_real> v3k(std::vector<fv_real>& a,int64_t k) {
  return {a.data(),lo,lo,k,1,ni,ni*ni};
}
View3<const fv_real> cv3k(const std::vector<fv_real>& a,int64_t k) {
  return {a.data(),lo,lo,k,1,ni,ni*ni};
}
bool near(fv_real a,fv_real b) {
  return std::abs(a-b)<(sizeof(fv_real)==4 ? fv_real(2e-5) : fv_real(1e-11));
}
}

int main() {
  const HorizontalBounds b{0,3,0,3,nz};
  std::vector<fv_real> dp(ni*ni*nz, fv_real(2));
  std::vector<fv_real> iface(ni*ni*(nz+1), fv_real(-1));
  View3<fv_real> pe{iface.data(),lo,1,lo,1,ni,ni*(nz+1)};
  reconstruct_pe_halo(b,fv_real(10),cv3(dp),pe);
  assert(iface[peoff(-1,3,2)]==fv_real(16));
  assert(iface[peoff(2,2,-1)]==fv_real(14));
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

  std::vector<fv_real> qxx(ni*ni),qyy(ni*ni),dxa(ni*ni,fv_real(1));
  std::vector<fv_real> dya(ni*ni,fv_real(1));
  std::vector<fv_real> grid(2*ni*ni),agrid(2*ni*ni);
  for(int j=lo;j<=hi;++j) for(int i=lo;i<=hi;++i) {
    grid[off(i,j)]=fv_real(.1)*(fv_real(i)-fv_real(.5));
    grid[off(i,j,1)]=fv_real(.05)*(fv_real(j)-fv_real(.5));
    agrid[off(i,j)]=fv_real(.1)*fv_real(i);
    agrid[off(i,j,1)]=fv_real(.05)*fv_real(j);
  }
  std::vector<accumulator_real> edge_weight(ni,accumulator_real(.25));
  A2BGeometry geometry{cv2(dxa),cv2(dya),cv3k(grid,0),cv3k(agrid,0),
                       cv1(edge_weight),cv1(edge_weight),cv1(edge_weight),
                       cv1(edge_weight)};
  const PhysicalEdges all_edges{true,true,true,true};
  const PhysicalCorners all_corners{true,true,true,true};
  const std::vector<fv_real> constant_q=q;
  a2b_ord2_global(b,all_edges,all_corners,v2(q),v2(out),geometry,false);
  assert(q==constant_q);
  for(int j=b.js;j<=b.je+1;++j)
    for(int i=b.is;i<=b.ie+1;++i) assert(near(out[off(i,j)],fv_real(7)));
  a2b_ord4_global(b,all_edges,all_corners,v2(q),v2(out),geometry,
                  v2(qx),v2(qy),v2(qxx),v2(qyy),true);
  for(int j=b.js;j<=b.je+1;++j)
    for(int i=b.is;i<=b.ie+1;++i) {
      assert(near(out[off(i,j)],fv_real(7)));
      assert(near(q[off(i,j)],fv_real(7)));
    }
  for(int j=lo;j<=hi;++j) for(int i=lo;i<=hi;++i)
    q[off(i,j)]=fv_real(i+10*j);
  const fv_real original_west=q[off(b.is,2)];
  a2b_ord2_global(b,{true,false,false,false},{},v2(q),v2(out),geometry,false);
  assert(near(out[off(b.is,2)],fv_real(17)));
  assert(q[off(b.is,2)]==original_west);
  a2b_ord4_global(b,{true,false,false,false},{},v2(q),v2(out),geometry,
                  v2(qx),v2(qy),v2(qxx),v2(qyy),false);
  assert(near(out[off(b.is,2)],fv_real(17)));
  assert(near(out[off(2,2)],fv_real(16.5)));
  assert(q[off(b.is,2)]==original_west);
  edge_weight[size_t(2-lo)]=accumulator_real(.50000001);
  q[off(b.is-1,1)]=q[off(b.is,1)]=fv_real(1.0e8);
  q[off(b.is-1,2)]=q[off(b.is,2)]=fv_real(-1.0e8);
  a2b_ord2_global(b,{true,false,false,false},{},v2(q),v2(out),geometry,false);
  const fv_real mixed_expected=static_cast<fv_real>(
      edge_weight[size_t(2-lo)]*accumulator_real(1.0e8)+
      (accumulator_real(1)-edge_weight[size_t(2-lo)])*
          accumulator_real(-1.0e8));
  assert(out[off(b.is,2)]==mixed_expected);
  if(sizeof(fv_real)==4) {
    const fv_real rounded_weight=static_cast<fv_real>(edge_weight[size_t(2-lo)]);
    const fv_real rounded_result=rounded_weight*fv_real(1.0e8)+
                                 (fv_real(1)-rounded_weight)*fv_real(-1.0e8);
    assert(mixed_expected!=rounded_result);
  }

  std::vector<fv_real> field(ni*ni*nz,fv_real(3)), fx(field.size()), fy(field.size());
  std::vector<fv_real> metric(ni*ni,fv_real(1));
  del2_pass(b,0,accumulator_real(.2),v3k(field,4),
            {cv2(metric),cv2(metric),cv2(metric)},v3k(fx,-2),v3k(fy,7));
  for(int j=0;j<=3;++j) for(int i=0;i<=3;++i) assert(field[off(i,j)]==fv_real(3));
  std::vector<fv_real> impulse(field.size(),fv_real(0));
  std::fill(fx.begin(),fx.end(),fv_real(0));
  std::fill(fy.begin(),fy.end(),fv_real(0));
  impulse[off(1,1)]=fv_real(1);
  del2_pass(b,0,accumulator_real(.2),v3(impulse),
            {cv2(metric),cv2(metric),cv2(metric)},v3(fx),v3(fy));
  assert(near(impulse[off(1,1)],fv_real(.2)));
  std::fill(impulse.begin(),impulse.end(),fv_real(0));
  impulse[off(1,1)]=fv_real(1);
  del2_pass_sg(b,0,accumulator_real(.2),v3(impulse),
               {cv2(metric),cv2(metric),cv2(metric),cv2(metric),cv2(metric),
                cv2(metric),cv2(metric)},v3(fx),v3(fy));
  assert(near(impulse[off(1,1)],fv_real(.2)));
  field[off(0,0)]=fv_real(3);field[off(-1,0)]=fv_real(6);field[off(0,-1)]=fv_real(9);
  average_del2_corners(b,{true,false,false,false},v3(field));
  assert(field[off(0,0)]==fv_real(6));
  assert(field[off(-1,0)]==fv_real(6)&&field[off(0,-1)]==fv_real(6));

  std::vector<fv_real> heat(field.size(),fv_real(100));
  std::vector<fv_real> delz(field.size(),fv_real(2));
  std::vector<fv_real> pkz(field.size(),fv_real(0));
  std::vector<fv_real> pt(field.size(),fv_real(300));
  std::vector<fv_real> cappa(field.size(),fv_real(.2));
  apply_diffusive_heating_nh(b,fv_real(2),fv_real(.5),fv_real(1),fv_real(1),
                             fv_real(.2),true,3,v3k(heat,-3),cv3k(dp,5),
                             cv3k(delz,8),cv3k(cappa,-7),v3k(pkz,11),
                             v3k(pt,2));
  const fv_real expected_pkz=std::exp(fv_real(.25)*std::log(fv_real(300)));
  assert(near(pkz[off(1,1,0)],expected_pkz));
  assert(near(pt[off(1,1,0)],fv_real(300)+fv_real(.1)/expected_pkz));
  std::fill(pt.begin(),pt.end(),fv_real(300));
  std::fill(pkz.begin(),pkz.end(),fv_real(0));
  std::fill(cappa.begin(),cappa.end(),fv_real(.3));
  apply_diffusive_heating_nh(b,fv_real(2),fv_real(.5),fv_real(1),fv_real(1),
                             fv_real(.2),false,3,v3(heat),cv3(dp),cv3(delz),
                             cv3(cappa),v3(pkz),v3(pt));
  assert(near(pkz[off(1,1,0)],expected_pkz));

  std::vector<fv_real> hydro_pt(field.size(),fv_real(100));
  std::vector<fv_real> hydro_pkz(field.size(),fv_real(2));
  apply_diffusive_heating_hydrostatic(
      b,fv_real(2),fv_real(.5),fv_real(1),3,cv3(heat),cv3(dp),
      cv3(hydro_pkz),v3(hydro_pt));
  assert(near(hydro_pt[off(1,1,0)],fv_real(125)));

  std::vector<fv_real> height(ni*ni*(nz+1),fv_real(2));
  std::vector<fv_real> geopotential(height.size(),fv_real(0));
  geopotential_from_height(b,2,fv_real(9),cv3k(height,-4),v3k(geopotential,6));
  assert(geopotential[off(-2,-2,3)]==fv_real(18));

  // Uniform pressure/geopotential has exactly zero gradient and only applies metrics.
  std::vector<fv_real> pp(ni*ni*(nz+1),fv_real(0));
  std::vector<fv_real> gz(pp.size(),fv_real(5));
  std::vector<fv_real> pk(pp.size());
  for(int k=0;k<=nz;++k) for(int j=lo;j<=hi;++j) for(int i=lo;i<=hi;++i)
    pk[off(i,j,k)]=fv_real(10+2*k);
  std::vector<fv_real> u(field.size(),fv_real(4)),v(field.size(),fv_real(6));
  std::vector<fv_real> du(field.size(),fv_real(0)),dv(field.size(),fv_real(0));
  pressure_gradient(b,{v3k(pp,-2),v3k(gz,4),v3k(dp,8),v3k(pk,-7),
                       cv2(metric),cv2(metric)},
                    {v3k(u,3),v3k(v,-5),v3k(du,9),v3k(dv,-11)},
                    {PressureVariant::normal_nh,fv_real(.1),fv_real(0)});
  assert(u[off(1,1,0)]==fv_real(4));
  assert(v[off(1,1,0)]==fv_real(6));

  std::fill(u.begin(),u.end(),fv_real(2));
  std::fill(v.begin(),v.end(),fv_real(2));
  std::fill(du.begin(),du.end(),fv_real(4));
  std::fill(dv.begin(),dv.end(),fv_real(4));
  pressure_gradient(b,{v3(pp),v3(gz),v3(dp),v3(pk),cv2(metric),cv2(metric)},
                    {v3(u),v3(v),v3(du),v3(dv)},
                    {PressureVariant::split,fv_real(.1),fv_real(.25)});
  assert(u[off(1,1,0)]==fv_real(3));
  assert(v[off(1,1,0)]==fv_real(3));
  assert(du[off(1,1,0)]==fv_real(0));
  assert(dv[off(1,1,0)]==fv_real(0));

  std::vector<fv_real> divg(ni*ni,fv_real(0));
  divg[off(1,1)]=fv_real(2);divg[off(2,1)]=fv_real(1);
  std::fill(u.begin(),u.end(),fv_real(0));std::fill(v.begin(),v.end(),fv_real(0));
  one_gradient_pressure(b,{v3(pk),v3(gz),v3(dp),cv2(divg),cv2(metric),cv2(metric)},
                        v3(u),v3(v),fv_real(.1),fv_real(1));
  assert(u[off(1,1,0)]==fv_real(1));
  assert(v[off(1,1,0)]==fv_real(2));

  std::fill(u.begin(),u.end(),fv_real(1));std::fill(v.begin(),v.end(),fv_real(1));
  std::fill(du.begin(),du.end(),fv_real(4));std::fill(dv.begin(),dv.end(),fv_real(4));
  grad1_pressure_update(
      b,{v3(pk),v3(gz),v3(dp),cv2(divg),cv2(metric),cv2(metric)},
      v3(u),v3(v),v3(du),v3(dv),fv_real(.1),fv_real(.25));
  assert(u[off(1,1,0)]==fv_real(3));
  assert(v[off(1,1,0)]==fv_real(4));
}
