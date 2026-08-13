#include "wp_d_operators.hpp"

#include <algorithm>
#include <cmath>

namespace fv3::gpu::operators {

void reconstruct_pe_halo(const HorizontalBounds &b, fv_real ptop,
                         View3<const fv_real> dp, View3<fv_real> pe) {
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i : {b.is - 1, b.ie + 1}) {
      pe(i, j, 0) = ptop;
      for (int64_t k = 0; k < b.npz; ++k)
        pe(i, j, k + 1) = pe(i, j, k) + dp(i, j, k);
    }
  }
  for (int64_t i = b.is - 1; i <= b.ie + 1; ++i) {
    for (int64_t j : {b.js - 1, b.je + 1}) {
      pe(i, j, 0) = ptop;
      for (int64_t k = 0; k < b.npz; ++k)
        pe(i, j, k + 1) = pe(i, j, k) + dp(i, j, k);
    }
  }
}

template <class Transform>
static void reconstruct_two_ring(const HorizontalBounds &b, fv_real ptop,
                                 View3<const fv_real> dp, View3<fv_real> out,
                                 Transform transform) {
  for (int64_t j = b.js; j <= b.je; ++j)
    for (int64_t i : {b.is - 2, b.is - 1, b.ie + 1, b.ie + 2}) {
      fv_real p = ptop;
      for (int64_t k = 0; k < b.npz; ++k) {
        p += dp(i, j, k);
        out(i, j, k + 1) = transform(p);
      }
    }
  for (int64_t i = b.is - 2; i <= b.ie + 2; ++i)
    for (int64_t j : {b.js - 2, b.js - 1, b.je + 1, b.je + 2}) {
      fv_real p = ptop;
      for (int64_t k = 0; k < b.npz; ++k) {
        p += dp(i, j, k);
        out(i, j, k + 1) = transform(p);
      }
    }
}

void reconstruct_pk3_halo(const HorizontalBounds &b, fv_real ptop,
                          fv_real akap, View3<const fv_real> dp,
                          View3<fv_real> pk3) {
  reconstruct_two_ring(b, ptop, dp, pk3,
                       [=](fv_real p) { return std::exp(akap * std::log(p)); });
}

void reconstruct_pln_halo(const HorizontalBounds &b, fv_real ptop,
                          View3<const fv_real> dp, View3<fv_real> pln) {
  reconstruct_two_ring(b, ptop, dp, pln,
                       [](fv_real p) { return std::log(p); });
}

void pressure_gradient(const HorizontalBounds &b, const PressureInputs &p,
                       const PressureOutputs &o, const PressureConfig &c) {
  const fv_real alpha = fv_real(1) - c.beta;
  for (int64_t k = 0; k < b.npz; ++k) {
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie; ++i) {
        const fv_real wk_l = p.pk_b(i, j, k + 1) - p.pk_b(i, j, k);
        const fv_real wk_r = p.pk_b(i + 1, j, k + 1) - p.pk_b(i + 1, j, k);
        const fv_real hydro = c.dt / (wk_l + wk_r) *
          ((p.gz(i,j,k+1)-p.gz(i+1,j,k))*(p.pk_b(i+1,j,k+1)-p.pk_b(i,j,k)) +
           (p.gz(i,j,k)-p.gz(i+1,j,k+1))*(p.pk_b(i,j,k+1)-p.pk_b(i+1,j,k)));
        const fv_real nh = c.dt / (p.delp_b(i,j,k)+p.delp_b(i+1,j,k)) *
          ((p.gz(i,j,k+1)-p.gz(i+1,j,k))*(p.pp(i+1,j,k+1)-p.pp(i,j,k)) +
           (p.gz(i,j,k)-p.gz(i+1,j,k+1))*(p.pp(i,j,k+1)-p.pp(i+1,j,k)));
        if (c.variant == PressureVariant::split) {
          o.u(i,j,k) += c.beta * o.du(i,j,k);
          o.du(i,j,k) = hydro;
          o.u(i,j,k) = (o.u(i,j,k) + alpha*hydro + nh) * p.rdx(i,j);
        } else {
          o.u(i,j,k) = (o.u(i,j,k) + hydro + nh) * p.rdx(i,j);
        }
      }
    for (int64_t j = b.js; j <= b.je; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i) {
        const fv_real wk_s = p.pk_b(i, j, k + 1) - p.pk_b(i, j, k);
        const fv_real wk_n = p.pk_b(i, j + 1, k + 1) - p.pk_b(i, j + 1, k);
        const fv_real hydro = c.dt / (wk_s + wk_n) *
          ((p.gz(i,j,k+1)-p.gz(i,j+1,k))*(p.pk_b(i,j+1,k+1)-p.pk_b(i,j,k)) +
           (p.gz(i,j,k)-p.gz(i,j+1,k+1))*(p.pk_b(i,j,k+1)-p.pk_b(i,j+1,k)));
        const fv_real nh = c.dt / (p.delp_b(i,j,k)+p.delp_b(i,j+1,k)) *
          ((p.gz(i,j,k+1)-p.gz(i,j+1,k))*(p.pp(i,j+1,k+1)-p.pp(i,j,k)) +
           (p.gz(i,j,k)-p.gz(i,j+1,k+1))*(p.pp(i,j,k+1)-p.pp(i,j+1,k)));
        if (c.variant == PressureVariant::split) {
          o.v(i,j,k) += c.beta * o.dv(i,j,k);
          o.dv(i,j,k) = hydro;
          o.v(i,j,k) = (o.v(i,j,k) + alpha*hydro + nh) * p.rdy(i,j);
        } else {
          o.v(i,j,k) = (o.v(i,j,k) + hydro + nh) * p.rdy(i,j);
        }
      }
  }
}

void a2b_ord2_periodic(const HorizontalBounds &b, View2<const fv_real> q,
                       View2<fv_real> out) {
  for (int64_t j=b.js; j<=b.je+1; ++j)
    for (int64_t i=b.is; i<=b.ie+1; ++i)
      out(i,j)=fv_real(.25)*(q(i-1,j-1)+q(i,j-1)+q(i-1,j)+q(i,j));
}

void a2b_ord4_periodic(const HorizontalBounds &b, View2<const fv_real> q,
                       View2<fv_real> out, View2<fv_real> qx,
                       View2<fv_real> qy) {
  constexpr fv_real b1=fv_real(7.0/12.0), b2=fv_real(-1.0/12.0);
  constexpr fv_real a1=fv_real(9.0/16.0), a2=fv_real(-1.0/16.0);
  for (int64_t j=b.js-2;j<=b.je+2;++j)
    for (int64_t i=b.is;i<=b.ie+1;++i)
      qx(i,j)=b1*(q(i-1,j)+q(i,j))+b2*(q(i-2,j)+q(i+1,j));
  for (int64_t j=b.js;j<=b.je+1;++j)
    for (int64_t i=b.is-2;i<=b.ie+2;++i)
      qy(i,j)=b1*(q(i,j-1)+q(i,j))+b2*(q(i,j-2)+q(i,j+1));
  for (int64_t j=b.js;j<=b.je+1;++j)
    for (int64_t i=b.is;i<=b.ie+1;++i)
      out(i,j)=fv_real(.5)*(a1*(qx(i,j-1)+qx(i,j)+qy(i-1,j)+qy(i,j))+
                            a2*(qx(i,j-2)+qx(i,j+1)+qy(i-2,j)+qy(i+1,j)));
}

void del2_pass(const HorizontalBounds &b, int nt, fv_real cd, View3<fv_real> q,
               const Del2Metrics &m, View3<fv_real> fx, View3<fv_real> fy) {
  for (int64_t k=0;k<b.npz;++k) {
    for (int64_t j=b.js-nt;j<=b.je+nt;++j)
      for (int64_t i=b.is-nt;i<=b.ie+1+nt;++i)
        fx(i,j,k)=m.del6_v(i,j)*(q(i-1,j,k)-q(i,j,k));
    for (int64_t j=b.js-nt;j<=b.je+1+nt;++j)
      for (int64_t i=b.is-nt;i<=b.ie+nt;++i)
        fy(i,j,k)=m.del6_u(i,j)*(q(i,j-1,k)-q(i,j,k));
    for (int64_t j=b.js-nt;j<=b.je+nt;++j)
      for (int64_t i=b.is-nt;i<=b.ie+nt;++i)
        q(i,j,k)=q(i,j,k)+cd*m.rarea(i,j)*
          (fx(i,j,k)-fx(i+1,j,k)+fy(i,j,k)-fy(i,j+1,k));
  }
}

void apply_diffusive_heating_nh(const HorizontalBounds &b, fv_real bdt,
                             fv_real delt_max, fv_real cv_air, fv_real rdg,
                             int64_t n_con, View3<fv_real> heat,
                             View3<const fv_real> delp,
                             View3<const fv_real> delz,
                             View3<const fv_real> cappa,
                             View3<fv_real> pkz, View3<fv_real> pt) {
  for (int64_t k=0;k<std::min(b.npz,n_con);++k) {
    fv_real delt=std::abs(bdt*delt_max);
    if (k==0) delt*=fv_real(.1);
    if (k==1) delt*=fv_real(.5);
    for (int64_t j=b.js;j<=b.je;++j)
      for (int64_t i=b.is;i<=b.ie;++i) {
        const fv_real kap=cappa(i,j,k);
        pkz(i,j,k)=std::exp(kap/(fv_real(1)-kap)*
                            std::log(rdg*delp(i,j,k)/delz(i,j,k)*pt(i,j,k)));
        const fv_real dtmp=heat(i,j,k)/(cv_air*delp(i,j,k));
        pt(i,j,k)+=std::copysign(std::min(delt,std::abs(dtmp)),dtmp)/pkz(i,j,k);
      }
  }
}

} // namespace fv3::gpu::operators
