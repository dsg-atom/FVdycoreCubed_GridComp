#include "wp_d_operators.hpp"

#include <algorithm>
#include <cmath>

namespace fv3::gpu::operators {

void reconstruct_pe_halo(const HorizontalBounds &b, fv_real ptop,
                         View3<const fv_real> dp, View3<fv_real> pe) {
  for (int64_t j = b.js; j <= b.je; ++j) {
    for (int64_t i : {b.is - 1, b.ie + 1}) {
      // pe preserves its actual Fortran (i,k,j) descriptor order.
      pe(i, pe.lo_j, j) = ptop;
      for (int64_t level = 0; level < b.npz; ++level)
        pe(i, pe.lo_j+level+1, j) = pe(i, pe.lo_j+level, j) +
                                    dp(i, j, dp.lo_k+level);
    }
  }
  for (int64_t i = b.is - 1; i <= b.ie + 1; ++i) {
    for (int64_t j : {b.js - 1, b.je + 1}) {
      pe(i, pe.lo_j, j) = ptop;
      for (int64_t level = 0; level < b.npz; ++level)
        pe(i, pe.lo_j+level+1, j) = pe(i, pe.lo_j+level, j) +
                                    dp(i, j, dp.lo_k+level);
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
      for (int64_t level = 0; level < b.npz; ++level) {
        p += dp(i, j, dp.lo_k+level);
        out(i, j, out.lo_k+level+1) = transform(p);
      }
    }
  for (int64_t i = b.is - 2; i <= b.ie + 2; ++i)
    for (int64_t j : {b.js - 2, b.js - 1, b.je + 1, b.je + 2}) {
      fv_real p = ptop;
      for (int64_t level = 0; level < b.npz; ++level) {
        p += dp(i, j, dp.lo_k+level);
        out(i, j, out.lo_k+level+1) = transform(p);
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

void geopotential_from_height(const HorizontalBounds &b, int halo_band,
                              fv_real grav, View3<const fv_real> zh,
                              View3<fv_real> gz) {
  for(int64_t level=0;level<=b.npz;++level)
    for(int64_t j=b.js-halo_band;j<=b.je+halo_band;++j)
      for(int64_t i=b.is-halo_band;i<=b.ie+halo_band;++i)
        gz(i,j,gz.lo_k+level)=grav*zh(i,j,zh.lo_k+level);
}

void set_pressure_top(const HorizontalBounds &b, fv_real top_value,
                      View3<fv_real> pp, View3<fv_real> pk) {
  for (int64_t j = b.js; j <= b.je + 1; ++j)
    for (int64_t i = b.is; i <= b.ie + 1; ++i) {
      pp(i, j, pp.lo_k) = fv_real(0);
      pk(i, j, pk.lo_k) = top_value;
    }
}

void pressure_gradient(const HorizontalBounds &b, const PressureInputs &p,
                       const PressureOutputs &o, const PressureConfig &c) {
  const fv_real alpha = fv_real(1) - c.beta;
  for (int64_t level = 0; level < b.npz; ++level) {
    const int64_t kd=p.delp_b.lo_k+level, kp=p.pk_b.lo_k+level;
    const int64_t kg=p.gz.lo_k+level, kpp=p.pp.lo_k+level;
    const int64_t ku=o.u.lo_k+level, kv=o.v.lo_k+level;
    for (int64_t j = b.js; j <= b.je + 1; ++j)
      for (int64_t i = b.is; i <= b.ie; ++i) {
        const fv_real wk_l = p.pk_b(i,j,kp+1)-p.pk_b(i,j,kp);
        const fv_real wk_r = p.pk_b(i+1,j,kp+1)-p.pk_b(i+1,j,kp);
        const fv_real hydro = c.dt / (wk_l + wk_r) *
          ((p.gz(i,j,kg+1)-p.gz(i+1,j,kg))*(p.pk_b(i+1,j,kp+1)-p.pk_b(i,j,kp))+
           (p.gz(i,j,kg)-p.gz(i+1,j,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i+1,j,kp)));
        const fv_real nh = c.dt / (p.delp_b(i,j,kd)+p.delp_b(i+1,j,kd)) *
          ((p.gz(i,j,kg+1)-p.gz(i+1,j,kg))*(p.pp(i+1,j,kpp+1)-p.pp(i,j,kpp))+
           (p.gz(i,j,kg)-p.gz(i+1,j,kg+1))*(p.pp(i,j,kpp+1)-p.pp(i+1,j,kpp)));
        if (c.variant == PressureVariant::split) {
          o.u(i,j,ku) += c.beta * o.du(i,j,o.du.lo_k+level);
          o.du(i,j,o.du.lo_k+level) = hydro;
          o.u(i,j,ku) = (o.u(i,j,ku) + alpha*hydro + nh) * p.rdx(i,j);
        } else {
          o.u(i,j,ku) = (o.u(i,j,ku) + hydro + nh) * p.rdx(i,j);
        }
      }
    for (int64_t j = b.js; j <= b.je; ++j)
      for (int64_t i = b.is; i <= b.ie + 1; ++i) {
        const fv_real wk_s=p.pk_b(i,j,kp+1)-p.pk_b(i,j,kp);
        const fv_real wk_n=p.pk_b(i,j+1,kp+1)-p.pk_b(i,j+1,kp);
        const fv_real hydro = c.dt / (wk_s + wk_n) *
          ((p.gz(i,j,kg+1)-p.gz(i,j+1,kg))*(p.pk_b(i,j+1,kp+1)-p.pk_b(i,j,kp))+
           (p.gz(i,j,kg)-p.gz(i,j+1,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i,j+1,kp)));
        const fv_real nh = c.dt / (p.delp_b(i,j,kd)+p.delp_b(i,j+1,kd)) *
          ((p.gz(i,j,kg+1)-p.gz(i,j+1,kg))*(p.pp(i,j+1,kpp+1)-p.pp(i,j,kpp))+
           (p.gz(i,j,kg)-p.gz(i,j+1,kg+1))*(p.pp(i,j,kpp+1)-p.pp(i,j+1,kpp)));
        if (c.variant == PressureVariant::split) {
          o.v(i,j,kv) += c.beta * o.dv(i,j,o.dv.lo_k+level);
          o.dv(i,j,o.dv.lo_k+level) = hydro;
          o.v(i,j,kv) = (o.v(i,j,kv) + alpha*hydro + nh) * p.rdy(i,j);
        } else {
          o.v(i,j,kv) = (o.v(i,j,kv) + hydro + nh) * p.rdy(i,j);
        }
      }
  }
}

void one_gradient_pressure(const HorizontalBounds &b,
                           const OneGradientInputs &p, View3<fv_real> u,
                           View3<fv_real> v, fv_real dt, fv_real d_ext) {
  for(int64_t level=0;level<b.npz;++level) {
    const int64_t kp=p.pk_b.lo_k+level,kg=p.gz_b.lo_k+level;
    const int64_t km=p.mass_b.lo_k+level,ku=u.lo_k+level,kv=v.lo_k+level;
    for(int64_t j=b.js;j<=b.je+1;++j)
      for(int64_t i=b.is;i<=b.ie;++i) {
        const fv_real ext=d_ext>fv_real(0)?p.divg2(i,j)-p.divg2(i+1,j):fv_real(0);
        u(i,j,ku)=p.rdx(i,j)*(ext+u(i,j,ku)+dt/(p.mass_b(i,j,km)+p.mass_b(i+1,j,km))*
          ((p.gz_b(i,j,kg+1)-p.gz_b(i+1,j,kg))*(p.pk_b(i+1,j,kp+1)-p.pk_b(i,j,kp))+
           (p.gz_b(i,j,kg)-p.gz_b(i+1,j,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i+1,j,kp))));
      }
    for(int64_t j=b.js;j<=b.je;++j)
      for(int64_t i=b.is;i<=b.ie+1;++i) {
        const fv_real ext=d_ext>fv_real(0)?p.divg2(i,j)-p.divg2(i,j+1):fv_real(0);
        v(i,j,kv)=p.rdy(i,j)*(ext+v(i,j,kv)+dt/(p.mass_b(i,j,km)+p.mass_b(i,j+1,km))*
          ((p.gz_b(i,j,kg+1)-p.gz_b(i,j+1,kg))*(p.pk_b(i,j+1,kp+1)-p.pk_b(i,j,kp))+
           (p.gz_b(i,j,kg)-p.gz_b(i,j+1,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i,j+1,kp))));
      }
  }
}

void grad1_pressure_update(const HorizontalBounds &b,
                           const OneGradientInputs &p, View3<fv_real> u,
                           View3<fv_real> v, View3<fv_real> du,
                           View3<fv_real> dv, fv_real dt, fv_real beta) {
  const fv_real alpha=fv_real(1)-beta;
  for(int64_t level=0;level<b.npz;++level) {
    const int64_t kp=p.pk_b.lo_k+level,kg=p.gz_b.lo_k+level;
    const int64_t km=p.mass_b.lo_k+level,ku=u.lo_k+level,kv=v.lo_k+level;
    for(int64_t j=b.js;j<=b.je+1;++j)
      for(int64_t i=b.is;i<=b.ie;++i) {
        const int64_t kd=du.lo_k+level;
        u(i,j,ku)+=beta*du(i,j,kd);
        du(i,j,kd)=dt/(p.mass_b(i,j,km)+p.mass_b(i+1,j,km))*
          ((p.gz_b(i,j,kg+1)-p.gz_b(i+1,j,kg))*(p.pk_b(i+1,j,kp+1)-p.pk_b(i,j,kp))+
           (p.gz_b(i,j,kg)-p.gz_b(i+1,j,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i+1,j,kp)));
        u(i,j,ku)=(u(i,j,ku)+p.divg2(i,j)-p.divg2(i+1,j)+alpha*du(i,j,kd))*p.rdx(i,j);
      }
    for(int64_t j=b.js;j<=b.je;++j)
      for(int64_t i=b.is;i<=b.ie+1;++i) {
        const int64_t kd=dv.lo_k+level;
        v(i,j,kv)+=beta*dv(i,j,kd);
        dv(i,j,kd)=dt/(p.mass_b(i,j,km)+p.mass_b(i,j+1,km))*
          ((p.gz_b(i,j,kg+1)-p.gz_b(i,j+1,kg))*(p.pk_b(i,j+1,kp+1)-p.pk_b(i,j,kp))+
           (p.gz_b(i,j,kg)-p.gz_b(i,j+1,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i,j+1,kp)));
        v(i,j,kv)=(v(i,j,kv)+p.divg2(i,j)-p.divg2(i,j+1)+alpha*dv(i,j,kd))*p.rdy(i,j);
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

namespace {

fv_real weighted_edge_value(accumulator_real weight, fv_real previous,
                            fv_real current) {
  const accumulator_real value =
      weight * static_cast<accumulator_real>(previous) +
      (accumulator_real(1) - weight) *
          static_cast<accumulator_real>(current);
  return static_cast<fv_real>(value);
}

fv_real great_circle_distance(View3<const fv_real> points, int64_t i,
                              int64_t j, View3<const fv_real> target,
                              int64_t ti, int64_t tj) {
  const int64_t kp = points.lo_k;
  const int64_t kt = target.lo_k;
  const accumulator_real dlon =
      (static_cast<accumulator_real>(points(i,j,kp)) -
       static_cast<accumulator_real>(target(ti,tj,kt))) /
      accumulator_real(2);
  const accumulator_real dlat =
      (static_cast<accumulator_real>(points(i,j,kp+1)) -
       static_cast<accumulator_real>(target(ti,tj,kt+1))) /
      accumulator_real(2);
  const accumulator_real lat = points(i,j,kp+1);
  const accumulator_real target_lat = target(ti,tj,kt+1);
  const accumulator_real haversine =
      std::sin(dlat)*std::sin(dlat) + std::cos(lat)*std::cos(target_lat)*
      std::sin(dlon)*std::sin(dlon);
  return static_cast<fv_real>(accumulator_real(2)*
                              std::asin(std::sqrt(haversine)));
}

fv_real extrapolate_corner(View3<const fv_real> grid,
                           View3<const fv_real> agrid, int64_t target_i,
                           int64_t target_j, int64_t first_i,
                           int64_t first_j, int64_t second_i,
                           int64_t second_j, fv_real first,
                           fv_real second) {
  const fv_real x1 = great_circle_distance(agrid,first_i,first_j,
                                           grid,target_i,target_j);
  const fv_real x2 = great_circle_distance(agrid,second_i,second_j,
                                           grid,target_i,target_j);
  return first + x1/(x2-x1)*(first-second);
}

void replace_a2b_output(const HorizontalBounds &b, View2<fv_real> input,
                        View2<fv_real> output) {
  for (int64_t j=b.js;j<=b.je+1;++j)
    for (int64_t i=b.is;i<=b.ie+1;++i)
      input(i,j)=output(i,j);
}

} // namespace

void a2b_ord2_global(const HorizontalBounds &b, PhysicalEdges edge,
                     PhysicalCorners corner, View2<fv_real> q,
                     View2<fv_real> out, const A2BGeometry &g,
                     bool replace) {
  constexpr fv_real quarter=fv_real(0.25);
  constexpr fv_real third=fv_real(1.0/3.0);
  const int64_t interior_i0=edge.west?b.is+1:b.is;
  const int64_t interior_i1=edge.east?b.ie:b.ie+1;
  const int64_t interior_j0=edge.south?b.js+1:b.js;
  const int64_t interior_j1=edge.north?b.je:b.je+1;
  for(int64_t j=interior_j0;j<=interior_j1;++j)
    for(int64_t i=interior_i0;i<=interior_i1;++i)
      out(i,j)=quarter*(q(i-1,j-1)+q(i,j-1)+q(i-1,j)+q(i,j));

  if(corner.sw)
    out(b.is,b.js)=third*(q(b.is,b.js)+q(b.is,b.js-1)+q(b.is-1,b.js));
  if(corner.se)
    out(b.ie+1,b.js)=third*(q(b.ie,b.js)+q(b.ie,b.js-1)+q(b.ie+1,b.js));
  if(corner.ne)
    out(b.ie+1,b.je+1)=third*(q(b.ie,b.je)+q(b.ie+1,b.je)+q(b.ie,b.je+1));
  if(corner.nw)
    out(b.is,b.je+1)=third*(q(b.is,b.je)+q(b.is-1,b.je)+q(b.is,b.je+1));

  const int64_t j1=edge.north?b.je:b.je+1;
  const int64_t je0=edge.south?b.js+1:b.js;
  if(edge.west)
    for(int64_t j=je0;j<=j1;++j) {
      const fv_real qprev=fv_real(.5)*(q(b.is-1,j-1)+q(b.is,j-1));
      const fv_real qcurr=fv_real(.5)*(q(b.is-1,j)+q(b.is,j));
      out(b.is,j)=weighted_edge_value(g.edge_w(j),qprev,qcurr);
    }
  if(edge.east)
    for(int64_t j=je0;j<=j1;++j) {
      const fv_real qprev=fv_real(.5)*(q(b.ie,j-1)+q(b.ie+1,j-1));
      const fv_real qcurr=fv_real(.5)*(q(b.ie,j)+q(b.ie+1,j));
      out(b.ie+1,j)=weighted_edge_value(g.edge_e(j),qprev,qcurr);
    }

  const int64_t i1=edge.east?b.ie:b.ie+1;
  const int64_t ie0=edge.west?b.is+1:b.is;
  if(edge.south)
    for(int64_t i=ie0;i<=i1;++i) {
      const fv_real qprev=fv_real(.5)*(q(i-1,b.js-1)+q(i-1,b.js));
      const fv_real qcurr=fv_real(.5)*(q(i,b.js-1)+q(i,b.js));
      out(i,b.js)=weighted_edge_value(g.edge_s(i),qprev,qcurr);
    }
  if(edge.north)
    for(int64_t i=ie0;i<=i1;++i) {
      const fv_real qprev=fv_real(.5)*(q(i-1,b.je)+q(i-1,b.je+1));
      const fv_real qcurr=fv_real(.5)*(q(i,b.je)+q(i,b.je+1));
      out(i,b.je+1)=weighted_edge_value(g.edge_n(i),qprev,qcurr);
    }

  if(replace) replace_a2b_output(b,q,out);
}

void a2b_ord4_global(const HorizontalBounds &b, PhysicalEdges edge,
                     PhysicalCorners corner, View2<fv_real> q,
                     View2<fv_real> out, const A2BGeometry &g,
                     View2<fv_real> qx, View2<fv_real> qy,
                     View2<fv_real> qxx, View2<fv_real> qyy,
                     bool replace) {
  constexpr fv_real r3=fv_real(1.0/3.0);
  constexpr fv_real b1=fv_real(7.0/12.0),b2=fv_real(-1.0/12.0);
  constexpr fv_real a1=fv_real(9.0/16.0),a2=fv_real(-1.0/16.0);
  constexpr fv_real c1=fv_real(2.0/3.0),c2=fv_real(-1.0/6.0);

  if(corner.sw) {
    out(b.is,b.js)=r3*(
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.is,b.js,
                         b.is,b.js,b.is+1,b.js+1,q(b.is,b.js),q(b.is+1,b.js+1))+
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.is,b.js,
                         b.is-1,b.js,b.is-2,b.js+1,q(b.is-1,b.js),q(b.is-2,b.js+1))+
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.is,b.js,
                         b.is,b.js-1,b.is+1,b.js-2,q(b.is,b.js-1),q(b.is+1,b.js-2)));
  }
  if(corner.se) {
    out(b.ie+1,b.js)=r3*(
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.js,
                         b.ie,b.js,b.ie-1,b.js+1,q(b.ie,b.js),q(b.ie-1,b.js+1))+
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.js,
                         b.ie,b.js-1,b.ie-1,b.js-2,q(b.ie,b.js-1),q(b.ie-1,b.js-2))+
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.js,
                         b.ie+1,b.js,b.ie+2,b.js+1,q(b.ie+1,b.js),q(b.ie+2,b.js+1)));
  }
  if(corner.ne) {
    out(b.ie+1,b.je+1)=r3*(
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.je+1,
                         b.ie,b.je,b.ie-1,b.je-1,q(b.ie,b.je),q(b.ie-1,b.je-1))+
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.je+1,
                         b.ie+1,b.je,b.ie+2,b.je-1,q(b.ie+1,b.je),q(b.ie+2,b.je-1))+
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.je+1,
                         b.ie,b.je+1,b.ie-1,b.je+2,q(b.ie,b.je+1),q(b.ie-1,b.je+2)));
  }
  if(corner.nw) {
    out(b.is,b.je+1)=r3*(
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.is,b.je+1,
                         b.is,b.je,b.is+1,b.je-1,q(b.is,b.je),q(b.is+1,b.je-1))+
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.is,b.je+1,
                         b.is-1,b.je,b.is-2,b.je-1,q(b.is-1,b.je),q(b.is-2,b.je-1))+
      extrapolate_corner(g.grid_lonlat,g.agrid_lonlat,b.is,b.je+1,
                         b.is,b.je+1,b.is+1,b.je+2,q(b.is,b.je+1),q(b.is+1,b.je+2)));
  }

  const int64_t qx_j0=edge.south?b.js:b.js-2;
  const int64_t qx_j1=edge.north?b.je:b.je+2;
  const int64_t qx_i0=edge.west?b.is+2:b.is;
  const int64_t qx_i1=edge.east?b.ie-1:b.ie+1;
  for(int64_t j=qx_j0;j<=qx_j1;++j)
    for(int64_t i=qx_i0;i<=qx_i1;++i)
      qx(i,j)=b2*(q(i-2,j)+q(i+1,j))+b1*(q(i-1,j)+q(i,j));

  const int64_t edge_j1=edge.north?b.je:b.je+1;
  const int64_t output_j0=edge.south?b.js+1:b.js;
  if(edge.west) {
    for(int64_t j=output_j0;j<=edge_j1;++j) {
      const fv_real prev=(q(b.is-1,j-1)*g.dxa(b.is,j-1)+
                          q(b.is,j-1)*g.dxa(b.is-1,j-1))/
                         (g.dxa(b.is-1,j-1)+g.dxa(b.is,j-1));
      const fv_real curr=(q(b.is-1,j)*g.dxa(b.is,j)+
                          q(b.is,j)*g.dxa(b.is-1,j))/
                         (g.dxa(b.is-1,j)+g.dxa(b.is,j));
      out(b.is,j)=weighted_edge_value(g.edge_w(j),prev,curr);
    }
    for(int64_t j=qx_j0;j<=qx_j1;++j) {
      const fv_real gin=g.dxa(b.is+1,j)/g.dxa(b.is,j);
      const fv_real gout=g.dxa(b.is-2,j)/g.dxa(b.is-1,j);
      qx(b.is,j)=fv_real(.5)*(
        ((fv_real(2)+gin)*q(b.is,j)-q(b.is+1,j))/(fv_real(1)+gin)+
        ((fv_real(2)+gout)*q(b.is-1,j)-q(b.is-2,j))/(fv_real(1)+gout));
      qx(b.is+1,j)=(fv_real(3)*(gin*q(b.is,j)+q(b.is+1,j))-
                       (gin*qx(b.is,j)+qx(b.is+2,j)))/(fv_real(2)+fv_real(2)*gin);
    }
  }
  if(edge.east) {
    for(int64_t j=output_j0;j<=edge_j1;++j) {
      const fv_real prev=(q(b.ie,j-1)*g.dxa(b.ie+1,j-1)+
                          q(b.ie+1,j-1)*g.dxa(b.ie,j-1))/
                         (g.dxa(b.ie,j-1)+g.dxa(b.ie+1,j-1));
      const fv_real curr=(q(b.ie,j)*g.dxa(b.ie+1,j)+
                          q(b.ie+1,j)*g.dxa(b.ie,j))/
                         (g.dxa(b.ie,j)+g.dxa(b.ie+1,j));
      out(b.ie+1,j)=weighted_edge_value(g.edge_e(j),prev,curr);
    }
    for(int64_t j=qx_j0;j<=qx_j1;++j) {
      const fv_real gin=g.dxa(b.ie-1,j)/g.dxa(b.ie,j);
      const fv_real gout=g.dxa(b.ie+2,j)/g.dxa(b.ie+1,j);
      qx(b.ie+1,j)=fv_real(.5)*(
        ((fv_real(2)+gin)*q(b.ie,j)-q(b.ie-1,j))/(fv_real(1)+gin)+
        ((fv_real(2)+gout)*q(b.ie+1,j)-q(b.ie+2,j))/(fv_real(1)+gout));
      qx(b.ie,j)=(fv_real(3)*(q(b.ie-1,j)+gin*q(b.ie,j))-
                    (gin*qx(b.ie+1,j)+qx(b.ie-1,j)))/(fv_real(2)+fv_real(2)*gin);
    }
  }

  const int64_t qy_i0=edge.west?b.is:b.is-2;
  const int64_t qy_i1=edge.east?b.ie:b.ie+2;
  const int64_t qy_j0=edge.south?b.js+2:b.js;
  const int64_t qy_j1=edge.north?b.je-1:b.je+1;
  for(int64_t j=qy_j0;j<=qy_j1;++j)
    for(int64_t i=qy_i0;i<=qy_i1;++i)
      qy(i,j)=b2*(q(i,j-2)+q(i,j+1))+b1*(q(i,j-1)+q(i,j));

  const int64_t edge_i1=edge.east?b.ie:b.ie+1;
  const int64_t output_i0=edge.west?b.is+1:b.is;
  if(edge.south) {
    for(int64_t i=output_i0;i<=edge_i1;++i) {
      const fv_real prev=(q(i-1,b.js-1)*g.dya(i-1,b.js)+
                          q(i-1,b.js)*g.dya(i-1,b.js-1))/
                         (g.dya(i-1,b.js-1)+g.dya(i-1,b.js));
      const fv_real curr=(q(i,b.js-1)*g.dya(i,b.js)+
                          q(i,b.js)*g.dya(i,b.js-1))/
                         (g.dya(i,b.js-1)+g.dya(i,b.js));
      out(i,b.js)=weighted_edge_value(g.edge_s(i),prev,curr);
    }
    for(int64_t i=qy_i0;i<=qy_i1;++i) {
      const fv_real gin=g.dya(i,b.js+1)/g.dya(i,b.js);
      const fv_real gout=g.dya(i,b.js-2)/g.dya(i,b.js-1);
      qy(i,b.js)=fv_real(.5)*(
        ((fv_real(2)+gin)*q(i,b.js)-q(i,b.js+1))/(fv_real(1)+gin)+
        ((fv_real(2)+gout)*q(i,b.js-1)-q(i,b.js-2))/(fv_real(1)+gout));
      qy(i,b.js+1)=(fv_real(3)*(gin*q(i,b.js)+q(i,b.js+1))-
                        (gin*qy(i,b.js)+qy(i,b.js+2)))/(fv_real(2)+fv_real(2)*gin);
    }
  }
  if(edge.north) {
    for(int64_t i=output_i0;i<=edge_i1;++i) {
      const fv_real prev=(q(i-1,b.je)*g.dya(i-1,b.je+1)+
                          q(i-1,b.je+1)*g.dya(i-1,b.je))/
                         (g.dya(i-1,b.je)+g.dya(i-1,b.je+1));
      const fv_real curr=(q(i,b.je)*g.dya(i,b.je+1)+
                          q(i,b.je+1)*g.dya(i,b.je))/
                         (g.dya(i,b.je)+g.dya(i,b.je+1));
      out(i,b.je+1)=weighted_edge_value(g.edge_n(i),prev,curr);
    }
    for(int64_t i=qy_i0;i<=qy_i1;++i) {
      const fv_real gin=g.dya(i,b.je-1)/g.dya(i,b.je);
      const fv_real gout=g.dya(i,b.je+2)/g.dya(i,b.je+1);
      qy(i,b.je+1)=fv_real(.5)*(
        ((fv_real(2)+gin)*q(i,b.je)-q(i,b.je-1))/(fv_real(1)+gin)+
        ((fv_real(2)+gout)*q(i,b.je+1)-q(i,b.je+2))/(fv_real(1)+gout));
      qy(i,b.je)=(fv_real(3)*(q(i,b.je-1)+gin*q(i,b.je))-
                    (gin*qy(i,b.je+1)+qy(i,b.je-1)))/(fv_real(2)+fv_real(2)*gin);
    }
  }

  const int64_t qxx_j0=edge.south?b.js+2:b.js;
  const int64_t qxx_j1=edge.north?b.je-1:b.je+1;
  const int64_t qxx_i0=edge.west?b.is+1:b.is;
  const int64_t qxx_i1=edge.east?b.ie:b.ie+1;
  for(int64_t j=qxx_j0;j<=qxx_j1;++j)
    for(int64_t i=qxx_i0;i<=qxx_i1;++i)
      qxx(i,j)=a2*(qx(i,j-2)+qx(i,j+1))+a1*(qx(i,j-1)+qx(i,j));
  if(edge.south)
    for(int64_t i=qxx_i0;i<=qxx_i1;++i)
      qxx(i,b.js+1)=c1*(qx(i,b.js)+qx(i,b.js+1))+
                     c2*(out(i,b.js)+qxx(i,b.js+2));
  if(edge.north)
    for(int64_t i=qxx_i0;i<=qxx_i1;++i)
      qxx(i,b.je)=c1*(qx(i,b.je-1)+qx(i,b.je))+
                   c2*(out(i,b.je+1)+qxx(i,b.je-1));

  const int64_t final_j0=edge.south?b.js+1:b.js;
  const int64_t final_j1=edge.north?b.je:b.je+1;
  const int64_t qyy_i0=edge.west?b.is+2:b.is;
  const int64_t qyy_i1=edge.east?b.ie-1:b.ie+1;
  for(int64_t j=final_j0;j<=final_j1;++j) {
    for(int64_t i=qyy_i0;i<=qyy_i1;++i)
      qyy(i,j)=a2*(qy(i-2,j)+qy(i+1,j))+a1*(qy(i-1,j)+qy(i,j));
    if(edge.west)
      qyy(b.is+1,j)=c1*(qy(b.is,j)+qy(b.is+1,j))+
                     c2*(out(b.is,j)+qyy(b.is+2,j));
    if(edge.east)
      qyy(b.ie,j)=c1*(qy(b.ie-1,j)+qy(b.ie,j))+
                   c2*(out(b.ie+1,j)+qyy(b.ie-1,j));
    for(int64_t i=qxx_i0;i<=qxx_i1;++i)
      out(i,j)=fv_real(.5)*(qxx(i,j)+qyy(i,j));
  }

  if(replace) replace_a2b_output(b,q,out);
}

void del2_pass(const HorizontalBounds &b, int nt, accumulator_real cd,
               View3<fv_real> q,
               const Del2Metrics &m, View3<fv_real> fx, View3<fv_real> fy) {
  for (int64_t level=0;level<b.npz;++level) {
    const int64_t kq=q.lo_k+level,kx=fx.lo_k+level,ky=fy.lo_k+level;
    for (int64_t j=b.js-nt;j<=b.je+nt;++j)
      for (int64_t i=b.is-nt;i<=b.ie+1+nt;++i)
        fx(i,j,kx)=m.del6_v(i,j)*(q(i-1,j,kq)-q(i,j,kq));
    for (int64_t j=b.js-nt;j<=b.je+1+nt;++j)
      for (int64_t i=b.is-nt;i<=b.ie+nt;++i)
        fy(i,j,ky)=m.del6_u(i,j)*(q(i,j-1,kq)-q(i,j,kq));
    for (int64_t j=b.js-nt;j<=b.je+nt;++j)
      for (int64_t i=b.is-nt;i<=b.ie+nt;++i)
        q(i,j,kq)=q(i,j,kq)+cd*m.rarea(i,j)*
          (fx(i,j,kx)-fx(i+1,j,kx)+fy(i,j,ky)-fy(i,j+1,ky));
  }
}

void average_del2_corners(const HorizontalBounds &b, PhysicalCorners c,
                          View3<fv_real> q) {
  constexpr fv_real third=fv_real(1.0/3.0);
  for(int64_t level=0;level<b.npz;++level) {
    const int64_t k=q.lo_k+level;
    if(c.sw){q(b.is,b.js,k)=third*(q(b.is,b.js,k)+q(b.is-1,b.js,k)+q(b.is,b.js-1,k));q(b.is-1,b.js,k)=q(b.is,b.js,k);q(b.is,b.js-1,k)=q(b.is,b.js,k);}
    if(c.se){q(b.ie,b.js,k)=third*(q(b.ie,b.js,k)+q(b.ie+1,b.js,k)+q(b.ie,b.js-1,k));q(b.ie+1,b.js,k)=q(b.ie,b.js,k);q(b.ie,b.js-1,k)=q(b.ie,b.js,k);}
    if(c.ne){q(b.ie,b.je,k)=third*(q(b.ie,b.je,k)+q(b.ie+1,b.je,k)+q(b.ie,b.je+1,k));q(b.ie+1,b.je,k)=q(b.ie,b.je,k);q(b.ie,b.je+1,k)=q(b.ie,b.je,k);}
    if(c.nw){q(b.is,b.je,k)=third*(q(b.is,b.je,k)+q(b.is-1,b.je,k)+q(b.is,b.je+1,k));q(b.is-1,b.je,k)=q(b.is,b.je,k);q(b.is,b.je+1,k)=q(b.is,b.je,k);}
  }
}

void del2_pass_sg(const HorizontalBounds &b, int nt, accumulator_real cd,
                  View3<fv_real> q, const Del2SgMetrics &m,
                  View3<fv_real> fx, View3<fv_real> fy) {
  for(int64_t level=0;level<b.npz;++level) {
    const int64_t kq=q.lo_k+level,kx=fx.lo_k+level,ky=fy.lo_k+level;
    for(int64_t j=b.js-nt;j<=b.je+nt;++j)
      for(int64_t i=b.is-nt;i<=b.ie+1+nt;++i)
        fx(i,j,kx)=m.dy(i,j)*m.sina_u(i,j)*(q(i-1,j,kq)-q(i,j,kq))*m.rdxc(i,j);
    for(int64_t j=b.js-nt;j<=b.je+1+nt;++j)
      for(int64_t i=b.is-nt;i<=b.ie+nt;++i)
        fy(i,j,ky)=m.dx(i,j)*m.sina_v(i,j)*(q(i,j-1,kq)-q(i,j,kq))*m.rdyc(i,j);
    for(int64_t j=b.js-nt;j<=b.je+nt;++j)
      for(int64_t i=b.is-nt;i<=b.ie+nt;++i)
        q(i,j,kq)=q(i,j,kq)+cd*m.rarea(i,j)*
          (fx(i,j,kx)-fx(i+1,j,kx)+fy(i,j,ky)-fy(i,j+1,ky));
  }
}

void apply_diffusive_heating_nh(const HorizontalBounds &b, fv_real bdt,
                             fv_real delt_max, fv_real cv_air, fv_real rdg,
                             fv_real akap, bool moist_cappa,
                             int64_t n_con, View3<fv_real> heat,
                             View3<const fv_real> delp,
                             View3<const fv_real> delz,
                             View3<const fv_real> cappa,
                             View3<fv_real> pkz, View3<fv_real> pt) {
  for (int64_t level=0;level<std::min(b.npz,n_con);++level) {
    fv_real delt=std::abs(bdt*delt_max);
    if (level==0) delt*=fv_real(.1);
    if (level==1) delt*=fv_real(.5);
    const int64_t kh=heat.lo_k+level,kd=delp.lo_k+level,kz=delz.lo_k+level;
    const int64_t kc=cappa.lo_k+level,kp=pkz.lo_k+level,kt=pt.lo_k+level;
    for (int64_t j=b.js;j<=b.je;++j)
      for (int64_t i=b.is;i<=b.ie;++i) {
        const fv_real kap=moist_cappa?cappa(i,j,kc):akap;
        pkz(i,j,kp)=std::exp(kap/(fv_real(1)-kap)*
                            std::log(rdg*delp(i,j,kd)/delz(i,j,kz)*pt(i,j,kt)));
        const fv_real dtmp=heat(i,j,kh)/(cv_air*delp(i,j,kd));
        pt(i,j,kt)+=std::copysign(std::min(delt,std::abs(dtmp)),dtmp)/pkz(i,j,kp);
      }
  }
}

void apply_diffusive_heating_hydrostatic(
    const HorizontalBounds &b, fv_real bdt, fv_real delt_max, fv_real cp_air,
    int64_t n_con, View3<const fv_real> heat, View3<const fv_real> delp,
    View3<const fv_real> pkz, View3<fv_real> pt) {
  for(int64_t level=0;level<std::min(b.npz,n_con);++level)
    for(int64_t j=b.js;j<=b.je;++j)
      for(int64_t i=b.is;i<=b.ie;++i) {
        const fv_real dtmp=heat(i,j,heat.lo_k+level)/
                            (cp_air*delp(i,j,delp.lo_k+level));
        const fv_real delta=level<2?dtmp:std::copysign(
          std::min(std::abs(bdt)*delt_max,std::abs(dtmp)),dtmp);
        pt(i,j,pt.lo_k+level)+=delta/pkz(i,j,pkz.lo_k+level);
      }
}

} // namespace fv3::gpu::operators
