#include "wp_d_cuda.hpp"

#include <cuda_runtime.h>

namespace fv3::gpu::operators {
namespace {
const dim3 block2(16,8,1);
constexpr unsigned block1=128;

template<class T> __device__ T dabs(T x){return x<T(0)?-x:x;}

struct A2BRange {
  int64_t i0,i1,j0,j1;
};

__global__ void pe_x_kernel(HorizontalBounds b,fv_real ptop,
                            View3<const fv_real> dp,View3<fv_real> pe){
  int64_t j=b.js+blockIdx.x*blockDim.x+threadIdx.x;
  int side=blockIdx.y*blockDim.y+threadIdx.y;if(j>b.je||side>=2)return;
  int64_t i=side?b.ie+1:b.is-1;pe(i,pe.lo_j,j)=ptop;
  for(int64_t n=0;n<b.npz;++n)pe(i,pe.lo_j+n+1,j)=pe(i,pe.lo_j+n,j)+dp(i,j,dp.lo_k+n);
}
__global__ void pe_y_kernel(HorizontalBounds b,fv_real ptop,
                            View3<const fv_real> dp,View3<fv_real> pe){
  int64_t i=b.is-1+blockIdx.x*blockDim.x+threadIdx.x;
  int side=blockIdx.y*blockDim.y+threadIdx.y;if(i>b.ie+1||side>=2)return;
  int64_t j=side?b.je+1:b.js-1;pe(i,pe.lo_j,j)=ptop;
  for(int64_t n=0;n<b.npz;++n)pe(i,pe.lo_j+n+1,j)=pe(i,pe.lo_j+n,j)+dp(i,j,dp.lo_k+n);
}
template<bool Logarithm>
__global__ void pressure_ring_x(HorizontalBounds b,fv_real ptop,fv_real akap,
                                View3<const fv_real> dp,View3<fv_real> out){
  int64_t j=b.js+blockIdx.x*blockDim.x+threadIdx.x;
  int col=blockIdx.y*blockDim.y+threadIdx.y;if(j>b.je||col>=4)return;
  int64_t i=col<2?b.is-2+col:b.ie-1+col;fv_real p=ptop;
  for(int64_t n=0;n<b.npz;++n){p+=dp(i,j,dp.lo_k+n);out(i,j,out.lo_k+n+1)=Logarithm?log(p):exp(akap*log(p));}
}
template<bool Logarithm>
__global__ void pressure_ring_y(HorizontalBounds b,fv_real ptop,fv_real akap,
                                View3<const fv_real> dp,View3<fv_real> out){
  int64_t i=b.is-2+blockIdx.x*blockDim.x+threadIdx.x;
  int row=blockIdx.y*blockDim.y+threadIdx.y;if(i>b.ie+2||row>=4)return;
  int64_t j=row<2?b.js-2+row:b.je-1+row;fv_real p=ptop;
  for(int64_t n=0;n<b.npz;++n){p+=dp(i,j,dp.lo_k+n);out(i,j,out.lo_k+n+1)=Logarithm?log(p):exp(akap*log(p));}
}
__global__ void geopotential_kernel(HorizontalBounds b,int h,fv_real grav,
                                    View3<const fv_real> zh,View3<fv_real> gz){
  int64_t i=b.is-h+blockIdx.x*blockDim.x+threadIdx.x;
  int64_t j=b.js-h+blockIdx.y*blockDim.y+threadIdx.y;int64_t n=blockIdx.z;
  if(i>b.ie+h||j>b.je+h||n>b.npz)return;gz(i,j,gz.lo_k+n)=grav*zh(i,j,zh.lo_k+n);
}
__global__ void a2b2_kernel(HorizontalBounds b,int64_t levels,
                            View3<const fv_real> q,View3<fv_real> out){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x;
  int64_t j=b.js+blockIdx.y*blockDim.y+threadIdx.y;int64_t n=blockIdx.z;
  if(i>b.ie+1||j>b.je+1||n>=levels)return;
  out(i,j,out.lo_k+n)=fv_real(.25)*(q(i-1,j-1,q.lo_k+n)+q(i,j-1,q.lo_k+n)+q(i-1,j,q.lo_k+n)+q(i,j,q.lo_k+n));
}
__global__ void a2b4_qx_kernel(HorizontalBounds b,int64_t levels,
                               View3<const fv_real> q,View3<fv_real> qx){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x;
  int64_t j=b.js-2+blockIdx.y*blockDim.y+threadIdx.y;int64_t n=blockIdx.z;
  if(i>b.ie+1||j>b.je+2||n>=levels)return;constexpr fv_real b1=fv_real(7.0/12.0),b2=fv_real(-1.0/12.0);
  qx(i,j,qx.lo_k+n)=b2*(q(i-2,j,q.lo_k+n)+q(i+1,j,q.lo_k+n))+b1*(q(i-1,j,q.lo_k+n)+q(i,j,q.lo_k+n));
}
__global__ void a2b4_qy_kernel(HorizontalBounds b,int64_t levels,
                               View3<const fv_real> q,View3<fv_real> qy){
  int64_t i=b.is-2+blockIdx.x*blockDim.x+threadIdx.x;
  int64_t j=b.js+blockIdx.y*blockDim.y+threadIdx.y;int64_t n=blockIdx.z;
  if(i>b.ie+2||j>b.je+1||n>=levels)return;constexpr fv_real b1=fv_real(7.0/12.0),b2=fv_real(-1.0/12.0);
  qy(i,j,qy.lo_k+n)=b2*(q(i,j-2,q.lo_k+n)+q(i,j+1,q.lo_k+n))+b1*(q(i,j-1,q.lo_k+n)+q(i,j,q.lo_k+n));
}
__global__ void a2b4_out_kernel(HorizontalBounds b,int64_t levels,
                                View3<fv_real> out,View3<fv_real> qx,View3<fv_real> qy){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x;
  int64_t j=b.js+blockIdx.y*blockDim.y+threadIdx.y;int64_t n=blockIdx.z;
  if(i>b.ie+1||j>b.je+1||n>=levels)return;constexpr fv_real a1=fv_real(9.0/16.0),a2=fv_real(-1.0/16.0);
  out(i,j,out.lo_k+n)=fv_real(.5)*(a1*(qx(i,j-1,qx.lo_k+n)+qx(i,j,qx.lo_k+n)+qy(i-1,j,qy.lo_k+n)+qy(i,j,qy.lo_k+n))+a2*(qx(i,j-2,qx.lo_k+n)+qx(i,j+1,qx.lo_k+n)+qy(i-2,j,qy.lo_k+n)+qy(i+1,j,qy.lo_k+n)));
}

__device__ fv_real a2b_weighted_edge(accumulator_real weight,
                                      fv_real previous,fv_real current){
  const accumulator_real value=
      weight*static_cast<accumulator_real>(previous)+
      (accumulator_real(1)-weight)*static_cast<accumulator_real>(current);
  return static_cast<fv_real>(value);
}

__device__ fv_real a2b_great_circle(View3<const fv_real> points,int64_t i,
                                    int64_t j,View3<const fv_real> target,
                                    int64_t ti,int64_t tj){
  const int64_t kp=points.lo_k,kt=target.lo_k;
  const accumulator_real dlon=
      (static_cast<accumulator_real>(points(i,j,kp))-
       static_cast<accumulator_real>(target(ti,tj,kt)))/accumulator_real(2);
  const accumulator_real dlat=
      (static_cast<accumulator_real>(points(i,j,kp+1))-
       static_cast<accumulator_real>(target(ti,tj,kt+1)))/accumulator_real(2);
  const accumulator_real lat=static_cast<accumulator_real>(points(i,j,kp+1));
  const accumulator_real target_lat=
      static_cast<accumulator_real>(target(ti,tj,kt+1));
  const accumulator_real sdlat=sin(dlat),sdlon=sin(dlon);
  accumulator_real haversine=sdlat*sdlat+
      cos(lat)*cos(target_lat)*sdlon*sdlon;
  if(haversine<accumulator_real(0))haversine=accumulator_real(0);
  if(haversine>accumulator_real(1))haversine=accumulator_real(1);
  return static_cast<fv_real>(accumulator_real(2)*asin(sqrt(haversine)));
}

__device__ fv_real a2b_extrapolate(View3<const fv_real> grid,
                                   View3<const fv_real> agrid,
                                   int64_t target_i,int64_t target_j,
                                   int64_t first_i,int64_t first_j,
                                   int64_t second_i,int64_t second_j,
                                   fv_real first,fv_real second){
  const fv_real x1=a2b_great_circle(agrid,first_i,first_j,
                                    grid,target_i,target_j);
  const fv_real x2=a2b_great_circle(agrid,second_i,second_j,
                                    grid,target_i,target_j);
  return first+x1/(x2-x1)*(first-second);
}

__global__ void a2b2_global_interior_kernel(A2BRange r,int64_t levels,
                                             View3<fv_real> q,
                                             View3<fv_real> out){
  const int64_t i=r.i0+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t j=r.j0+blockIdx.y*blockDim.y+threadIdx.y;
  const int64_t n=blockIdx.z;
  if(i>r.i1||j>r.j1||n>=levels)return;
  const int64_t k=q.lo_k+n,ko=out.lo_k+n;
  out(i,j,ko)=fv_real(.25)*(q(i-1,j-1,k)+q(i,j-1,k)+
                            q(i-1,j,k)+q(i,j,k));
}

__global__ void a2b2_global_corner_kernel(HorizontalBounds b,
                                           PhysicalCorners c,int64_t levels,
                                           View3<fv_real> q,
                                           View3<fv_real> out){
  const int64_t n=blockIdx.x*blockDim.x+threadIdx.x;
  if(n>=levels)return;
  const int64_t k=q.lo_k+n,ko=out.lo_k+n;
  constexpr fv_real r3=fv_real(1.0/3.0);
  if(c.sw)out(b.is,b.js,ko)=r3*(q(b.is,b.js,k)+q(b.is,b.js-1,k)+q(b.is-1,b.js,k));
  if(c.se)out(b.ie+1,b.js,ko)=r3*(q(b.ie,b.js,k)+q(b.ie,b.js-1,k)+q(b.ie+1,b.js,k));
  if(c.ne)out(b.ie+1,b.je+1,ko)=r3*(q(b.ie,b.je,k)+q(b.ie+1,b.je,k)+q(b.ie,b.je+1,k));
  if(c.nw)out(b.is,b.je+1,ko)=r3*(q(b.is,b.je,k)+q(b.is-1,b.je,k)+q(b.is,b.je+1,k));
}

__global__ void a2b2_global_x_edge_kernel(HorizontalBounds b,bool east,
                                           int64_t first,int64_t last_j,
                                           int64_t levels,View3<fv_real> q,
                                           View3<fv_real> out,A2BGeometry g){
  const int64_t j=first+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t n=blockIdx.y;
  if(j>last_j||n>=levels)return;
  const int64_t k=q.lo_k+n,ko=out.lo_k+n;
  if(east){
    const fv_real prev=fv_real(.5)*(q(b.ie,j-1,k)+q(b.ie+1,j-1,k));
    const fv_real curr=fv_real(.5)*(q(b.ie,j,k)+q(b.ie+1,j,k));
    out(b.ie+1,j,ko)=a2b_weighted_edge(g.edge_e(j),prev,curr);
  }else{
    const fv_real prev=fv_real(.5)*(q(b.is-1,j-1,k)+q(b.is,j-1,k));
    const fv_real curr=fv_real(.5)*(q(b.is-1,j,k)+q(b.is,j,k));
    out(b.is,j,ko)=a2b_weighted_edge(g.edge_w(j),prev,curr);
  }
}

__global__ void a2b2_global_y_edge_kernel(HorizontalBounds b,bool north,
                                           int64_t first,int64_t last_i,
                                           int64_t levels,View3<fv_real> q,
                                           View3<fv_real> out,A2BGeometry g){
  const int64_t i=first+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t n=blockIdx.y;
  if(i>last_i||n>=levels)return;
  const int64_t k=q.lo_k+n,ko=out.lo_k+n;
  if(north){
    const fv_real prev=fv_real(.5)*(q(i-1,b.je,k)+q(i-1,b.je+1,k));
    const fv_real curr=fv_real(.5)*(q(i,b.je,k)+q(i,b.je+1,k));
    out(i,b.je+1,ko)=a2b_weighted_edge(g.edge_n(i),prev,curr);
  }else{
    const fv_real prev=fv_real(.5)*(q(i-1,b.js-1,k)+q(i-1,b.js,k));
    const fv_real curr=fv_real(.5)*(q(i,b.js-1,k)+q(i,b.js,k));
    out(i,b.js,ko)=a2b_weighted_edge(g.edge_s(i),prev,curr);
  }
}

__global__ void a2b4_global_corner_kernel(HorizontalBounds b,
                                           PhysicalCorners c,int64_t levels,
                                           View3<fv_real> q,
                                           View3<fv_real> out,A2BGeometry g){
  const int64_t n=blockIdx.y;
  if(n>=levels)return;
  const int64_t k=q.lo_k+n,ko=out.lo_k+n;
  constexpr fv_real r3=fv_real(1.0/3.0);
  switch(blockIdx.x){
  case 0:
    if(c.sw){
      out(b.is,b.js,ko)=r3*(
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.is,b.js,
                        b.is,b.js,b.is+1,b.js+1,q(b.is,b.js,k),q(b.is+1,b.js+1,k))+
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.is,b.js,
                        b.is-1,b.js,b.is-2,b.js+1,q(b.is-1,b.js,k),q(b.is-2,b.js+1,k))+
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.is,b.js,
                        b.is,b.js-1,b.is+1,b.js-2,q(b.is,b.js-1,k),q(b.is+1,b.js-2,k)));
    }
    break;
  case 1:
    if(c.se){
      out(b.ie+1,b.js,ko)=r3*(
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.js,
                        b.ie,b.js,b.ie-1,b.js+1,q(b.ie,b.js,k),q(b.ie-1,b.js+1,k))+
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.js,
                        b.ie,b.js-1,b.ie-1,b.js-2,q(b.ie,b.js-1,k),q(b.ie-1,b.js-2,k))+
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.js,
                        b.ie+1,b.js,b.ie+2,b.js+1,q(b.ie+1,b.js,k),q(b.ie+2,b.js+1,k)));
    }
    break;
  case 2:
    if(c.ne){
      out(b.ie+1,b.je+1,ko)=r3*(
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.je+1,
                        b.ie,b.je,b.ie-1,b.je-1,q(b.ie,b.je,k),q(b.ie-1,b.je-1,k))+
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.je+1,
                        b.ie+1,b.je,b.ie+2,b.je-1,q(b.ie+1,b.je,k),q(b.ie+2,b.je-1,k))+
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.ie+1,b.je+1,
                        b.ie,b.je+1,b.ie-1,b.je+2,q(b.ie,b.je+1,k),q(b.ie-1,b.je+2,k)));
    }
    break;
  case 3:
    if(c.nw){
      out(b.is,b.je+1,ko)=r3*(
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.is,b.je+1,
                        b.is,b.je,b.is+1,b.je-1,q(b.is,b.je,k),q(b.is+1,b.je-1,k))+
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.is,b.je+1,
                        b.is-1,b.je,b.is-2,b.je-1,q(b.is-1,b.je,k),q(b.is-2,b.je-1,k))+
        a2b_extrapolate(g.grid_lonlat,g.agrid_lonlat,b.is,b.je+1,
                        b.is,b.je+1,b.is+1,b.je+2,q(b.is,b.je+1,k),q(b.is+1,b.je+2,k)));
    }
    break;
  }
}

__global__ void a2b4_global_qx_kernel(A2BRange r,int64_t levels,
                                       View3<fv_real> q,View3<fv_real> qx){
  const int64_t i=r.i0+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t j=r.j0+blockIdx.y*blockDim.y+threadIdx.y;
  const int64_t n=blockIdx.z;
  if(i>r.i1||j>r.j1||n>=levels)return;
  const int64_t k=q.lo_k+n,kx=qx.lo_k+n;
  constexpr fv_real b1=fv_real(7.0/12.0),b2=fv_real(-1.0/12.0);
  qx(i,j,kx)=b2*(q(i-2,j,k)+q(i+1,j,k))+
             b1*(q(i-1,j,k)+q(i,j,k));
}

__global__ void a2b4_global_x_output_kernel(HorizontalBounds b,bool east,
                                             int64_t first,int64_t last_j,
                                             int64_t levels,View3<fv_real> q,
                                             View3<fv_real> out,A2BGeometry g){
  const int64_t j=first+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t n=blockIdx.y;
  if(j>last_j||n>=levels)return;
  const int64_t k=q.lo_k+n,ko=out.lo_k+n;
  if(east){
    const fv_real prev=(q(b.ie,j-1,k)*g.dxa(b.ie+1,j-1)+
                        q(b.ie+1,j-1,k)*g.dxa(b.ie,j-1))/
                       (g.dxa(b.ie,j-1)+g.dxa(b.ie+1,j-1));
    const fv_real curr=(q(b.ie,j,k)*g.dxa(b.ie+1,j)+
                        q(b.ie+1,j,k)*g.dxa(b.ie,j))/
                       (g.dxa(b.ie,j)+g.dxa(b.ie+1,j));
    out(b.ie+1,j,ko)=a2b_weighted_edge(g.edge_e(j),prev,curr);
  }else{
    const fv_real prev=(q(b.is-1,j-1,k)*g.dxa(b.is,j-1)+
                        q(b.is,j-1,k)*g.dxa(b.is-1,j-1))/
                       (g.dxa(b.is-1,j-1)+g.dxa(b.is,j-1));
    const fv_real curr=(q(b.is-1,j,k)*g.dxa(b.is,j)+
                        q(b.is,j,k)*g.dxa(b.is-1,j))/
                       (g.dxa(b.is-1,j)+g.dxa(b.is,j));
    out(b.is,j,ko)=a2b_weighted_edge(g.edge_w(j),prev,curr);
  }
}

__global__ void a2b4_global_x_adjust_kernel(HorizontalBounds b,bool east,
                                             int64_t first,int64_t last_j,
                                             int64_t levels,View3<fv_real> q,
                                             View3<fv_real> qx,A2BGeometry g){
  const int64_t j=first+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t n=blockIdx.y;
  if(j>last_j||n>=levels)return;
  const int64_t k=q.lo_k+n,kx=qx.lo_k+n;
  if(east){
    const fv_real gin=g.dxa(b.ie-1,j)/g.dxa(b.ie,j);
    const fv_real gout=g.dxa(b.ie+2,j)/g.dxa(b.ie+1,j);
    qx(b.ie+1,j,kx)=fv_real(.5)*(
      ((fv_real(2)+gin)*q(b.ie,j,k)-q(b.ie-1,j,k))/(fv_real(1)+gin)+
      ((fv_real(2)+gout)*q(b.ie+1,j,k)-q(b.ie+2,j,k))/(fv_real(1)+gout));
    qx(b.ie,j,kx)=(fv_real(3)*(q(b.ie-1,j,k)+gin*q(b.ie,j,k))-
                       (gin*qx(b.ie+1,j,kx)+qx(b.ie-1,j,kx)))/
                      (fv_real(2)+fv_real(2)*gin);
  }else{
    const fv_real gin=g.dxa(b.is+1,j)/g.dxa(b.is,j);
    const fv_real gout=g.dxa(b.is-2,j)/g.dxa(b.is-1,j);
    qx(b.is,j,kx)=fv_real(.5)*(
      ((fv_real(2)+gin)*q(b.is,j,k)-q(b.is+1,j,k))/(fv_real(1)+gin)+
      ((fv_real(2)+gout)*q(b.is-1,j,k)-q(b.is-2,j,k))/(fv_real(1)+gout));
    qx(b.is+1,j,kx)=(fv_real(3)*(gin*q(b.is,j,k)+q(b.is+1,j,k))-
                         (gin*qx(b.is,j,kx)+qx(b.is+2,j,kx)))/
                        (fv_real(2)+fv_real(2)*gin);
  }
}

__global__ void a2b4_global_qy_kernel(A2BRange r,int64_t levels,
                                       View3<fv_real> q,View3<fv_real> qy){
  const int64_t i=r.i0+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t j=r.j0+blockIdx.y*blockDim.y+threadIdx.y;
  const int64_t n=blockIdx.z;
  if(i>r.i1||j>r.j1||n>=levels)return;
  const int64_t k=q.lo_k+n,ky=qy.lo_k+n;
  constexpr fv_real b1=fv_real(7.0/12.0),b2=fv_real(-1.0/12.0);
  qy(i,j,ky)=b2*(q(i,j-2,k)+q(i,j+1,k))+
             b1*(q(i,j-1,k)+q(i,j,k));
}

__global__ void a2b4_global_y_output_kernel(HorizontalBounds b,bool north,
                                             int64_t first,int64_t last_i,
                                             int64_t levels,View3<fv_real> q,
                                             View3<fv_real> out,A2BGeometry g){
  const int64_t i=first+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t n=blockIdx.y;
  if(i>last_i||n>=levels)return;
  const int64_t k=q.lo_k+n,ko=out.lo_k+n;
  if(north){
    const fv_real prev=(q(i-1,b.je,k)*g.dya(i-1,b.je+1)+
                        q(i-1,b.je+1,k)*g.dya(i-1,b.je))/
                       (g.dya(i-1,b.je)+g.dya(i-1,b.je+1));
    const fv_real curr=(q(i,b.je,k)*g.dya(i,b.je+1)+
                        q(i,b.je+1,k)*g.dya(i,b.je))/
                       (g.dya(i,b.je)+g.dya(i,b.je+1));
    out(i,b.je+1,ko)=a2b_weighted_edge(g.edge_n(i),prev,curr);
  }else{
    const fv_real prev=(q(i-1,b.js-1,k)*g.dya(i-1,b.js)+
                        q(i-1,b.js,k)*g.dya(i-1,b.js-1))/
                       (g.dya(i-1,b.js-1)+g.dya(i-1,b.js));
    const fv_real curr=(q(i,b.js-1,k)*g.dya(i,b.js)+
                        q(i,b.js,k)*g.dya(i,b.js-1))/
                       (g.dya(i,b.js-1)+g.dya(i,b.js));
    out(i,b.js,ko)=a2b_weighted_edge(g.edge_s(i),prev,curr);
  }
}

__global__ void a2b4_global_y_adjust_kernel(HorizontalBounds b,bool north,
                                             int64_t first,int64_t last_i,
                                             int64_t levels,View3<fv_real> q,
                                             View3<fv_real> qy,A2BGeometry g){
  const int64_t i=first+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t n=blockIdx.y;
  if(i>last_i||n>=levels)return;
  const int64_t k=q.lo_k+n,ky=qy.lo_k+n;
  if(north){
    const fv_real gin=g.dya(i,b.je-1)/g.dya(i,b.je);
    const fv_real gout=g.dya(i,b.je+2)/g.dya(i,b.je+1);
    qy(i,b.je+1,ky)=fv_real(.5)*(
      ((fv_real(2)+gin)*q(i,b.je,k)-q(i,b.je-1,k))/(fv_real(1)+gin)+
      ((fv_real(2)+gout)*q(i,b.je+1,k)-q(i,b.je+2,k))/(fv_real(1)+gout));
    qy(i,b.je,ky)=(fv_real(3)*(q(i,b.je-1,k)+gin*q(i,b.je,k))-
                       (gin*qy(i,b.je+1,ky)+qy(i,b.je-1,ky)))/
                      (fv_real(2)+fv_real(2)*gin);
  }else{
    const fv_real gin=g.dya(i,b.js+1)/g.dya(i,b.js);
    const fv_real gout=g.dya(i,b.js-2)/g.dya(i,b.js-1);
    qy(i,b.js,ky)=fv_real(.5)*(
      ((fv_real(2)+gin)*q(i,b.js,k)-q(i,b.js+1,k))/(fv_real(1)+gin)+
      ((fv_real(2)+gout)*q(i,b.js-1,k)-q(i,b.js-2,k))/(fv_real(1)+gout));
    qy(i,b.js+1,ky)=(fv_real(3)*(gin*q(i,b.js,k)+q(i,b.js+1,k))-
                         (gin*qy(i,b.js,ky)+qy(i,b.js+2,ky)))/
                        (fv_real(2)+fv_real(2)*gin);
  }
}

__global__ void a2b4_global_qxx_kernel(A2BRange r,int64_t levels,
                                        View3<fv_real> qx,
                                        View3<fv_real> qxx){
  const int64_t i=r.i0+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t j=r.j0+blockIdx.y*blockDim.y+threadIdx.y;
  const int64_t n=blockIdx.z;
  if(i>r.i1||j>r.j1||n>=levels)return;
  const int64_t kx=qx.lo_k+n,kxx=qxx.lo_k+n;
  constexpr fv_real a1=fv_real(9.0/16.0),a2=fv_real(-1.0/16.0);
  qxx(i,j,kxx)=a2*(qx(i,j-2,kx)+qx(i,j+1,kx))+
               a1*(qx(i,j-1,kx)+qx(i,j,kx));
}

__global__ void a2b4_global_qxx_adjust_kernel(HorizontalBounds b,bool north,
                                               int64_t first,int64_t last_i,
                                               int64_t levels,
                                               View3<fv_real> out,
                                               View3<fv_real> qx,
                                               View3<fv_real> qxx){
  const int64_t i=first+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t n=blockIdx.y;
  if(i>last_i||n>=levels)return;
  const int64_t ko=out.lo_k+n,kx=qx.lo_k+n,kxx=qxx.lo_k+n;
  constexpr fv_real c1=fv_real(2.0/3.0),c2=fv_real(-1.0/6.0);
  if(north)
    qxx(i,b.je,kxx)=c1*(qx(i,b.je-1,kx)+qx(i,b.je,kx))+
                    c2*(out(i,b.je+1,ko)+qxx(i,b.je-1,kxx));
  else
    qxx(i,b.js+1,kxx)=c1*(qx(i,b.js,kx)+qx(i,b.js+1,kx))+
                      c2*(out(i,b.js,ko)+qxx(i,b.js+2,kxx));
}

__global__ void a2b4_global_qyy_kernel(A2BRange r,int64_t levels,
                                        View3<fv_real> qy,
                                        View3<fv_real> qyy){
  const int64_t i=r.i0+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t j=r.j0+blockIdx.y*blockDim.y+threadIdx.y;
  const int64_t n=blockIdx.z;
  if(i>r.i1||j>r.j1||n>=levels)return;
  const int64_t ky=qy.lo_k+n,kyy=qyy.lo_k+n;
  constexpr fv_real a1=fv_real(9.0/16.0),a2=fv_real(-1.0/16.0);
  qyy(i,j,kyy)=a2*(qy(i-2,j,ky)+qy(i+1,j,ky))+
               a1*(qy(i-1,j,ky)+qy(i,j,ky));
}

__global__ void a2b4_global_qyy_adjust_kernel(HorizontalBounds b,bool east,
                                               int64_t first,int64_t last_j,
                                               int64_t levels,
                                               View3<fv_real> out,
                                               View3<fv_real> qy,
                                               View3<fv_real> qyy){
  const int64_t j=first+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t n=blockIdx.y;
  if(j>last_j||n>=levels)return;
  const int64_t ko=out.lo_k+n,ky=qy.lo_k+n,kyy=qyy.lo_k+n;
  constexpr fv_real c1=fv_real(2.0/3.0),c2=fv_real(-1.0/6.0);
  if(east)
    qyy(b.ie,j,kyy)=c1*(qy(b.ie-1,j,ky)+qy(b.ie,j,ky))+
                    c2*(out(b.ie+1,j,ko)+qyy(b.ie-1,j,kyy));
  else
    qyy(b.is+1,j,kyy)=c1*(qy(b.is,j,ky)+qy(b.is+1,j,ky))+
                      c2*(out(b.is,j,ko)+qyy(b.is+2,j,kyy));
}

__global__ void a2b4_global_output_kernel(A2BRange r,int64_t levels,
                                           View3<fv_real> out,
                                           View3<fv_real> qxx,
                                           View3<fv_real> qyy){
  const int64_t i=r.i0+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t j=r.j0+blockIdx.y*blockDim.y+threadIdx.y;
  const int64_t n=blockIdx.z;
  if(i>r.i1||j>r.j1||n>=levels)return;
  out(i,j,out.lo_k+n)=fv_real(.5)*(qxx(i,j,qxx.lo_k+n)+
                                    qyy(i,j,qyy.lo_k+n));
}

__global__ void a2b_replace_kernel(HorizontalBounds b,int64_t levels,
                                    View3<fv_real> input,
                                    View3<fv_real> output){
  const int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x;
  const int64_t j=b.js+blockIdx.y*blockDim.y+threadIdx.y;
  const int64_t n=blockIdx.z;
  if(i>b.ie+1||j>b.je+1||n>=levels)return;
  input(i,j,input.lo_k+n)=output(i,j,output.lo_k+n);
}

__global__ void set_top_kernel(HorizontalBounds b,fv_real top,View3<fv_real> pp,View3<fv_real> pk){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x;int64_t j=b.js+blockIdx.y*blockDim.y+threadIdx.y;
  if(i>b.ie+1||j>b.je+1)return;pp(i,j,pp.lo_k)=fv_real(0);pk(i,j,pk.lo_k)=top;
}
__global__ void pressure_u_kernel(HorizontalBounds b,PressureInputs p,PressureOutputs o,PressureConfig c){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x;int64_t j=b.js+blockIdx.y*blockDim.y+threadIdx.y;int64_t n=blockIdx.z;
  if(i>b.ie||j>b.je+1||n>=b.npz)return;int64_t kd=p.delp_b.lo_k+n,kp=p.pk_b.lo_k+n,kg=p.gz.lo_k+n,kpp=p.pp.lo_k+n,ku=o.u.lo_k+n;
  fv_real wl=p.pk_b(i,j,kp+1)-p.pk_b(i,j,kp),wr=p.pk_b(i+1,j,kp+1)-p.pk_b(i+1,j,kp);
  fv_real hydro=c.dt/(wl+wr)*((p.gz(i,j,kg+1)-p.gz(i+1,j,kg))*(p.pk_b(i+1,j,kp+1)-p.pk_b(i,j,kp))+(p.gz(i,j,kg)-p.gz(i+1,j,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i+1,j,kp)));
  fv_real nh=c.dt/(p.delp_b(i,j,kd)+p.delp_b(i+1,j,kd))*((p.gz(i,j,kg+1)-p.gz(i+1,j,kg))*(p.pp(i+1,j,kpp+1)-p.pp(i,j,kpp))+(p.gz(i,j,kg)-p.gz(i+1,j,kg+1))*(p.pp(i,j,kpp+1)-p.pp(i+1,j,kpp)));
  if(c.variant==PressureVariant::split){int64_t kdout=o.du.lo_k+n;o.u(i,j,ku)+=c.beta*o.du(i,j,kdout);o.du(i,j,kdout)=hydro;o.u(i,j,ku)=(o.u(i,j,ku)+(fv_real(1)-c.beta)*hydro+nh)*p.rdx(i,j);}else o.u(i,j,ku)=(o.u(i,j,ku)+hydro+nh)*p.rdx(i,j);
}
__global__ void pressure_v_kernel(HorizontalBounds b,PressureInputs p,PressureOutputs o,PressureConfig c){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x;int64_t j=b.js+blockIdx.y*blockDim.y+threadIdx.y;int64_t n=blockIdx.z;
  if(i>b.ie+1||j>b.je||n>=b.npz)return;int64_t kd=p.delp_b.lo_k+n,kp=p.pk_b.lo_k+n,kg=p.gz.lo_k+n,kpp=p.pp.lo_k+n,kv=o.v.lo_k+n;
  fv_real ws=p.pk_b(i,j,kp+1)-p.pk_b(i,j,kp),wn=p.pk_b(i,j+1,kp+1)-p.pk_b(i,j+1,kp);
  fv_real hydro=c.dt/(ws+wn)*((p.gz(i,j,kg+1)-p.gz(i,j+1,kg))*(p.pk_b(i,j+1,kp+1)-p.pk_b(i,j,kp))+(p.gz(i,j,kg)-p.gz(i,j+1,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i,j+1,kp)));
  fv_real nh=c.dt/(p.delp_b(i,j,kd)+p.delp_b(i,j+1,kd))*((p.gz(i,j,kg+1)-p.gz(i,j+1,kg))*(p.pp(i,j+1,kpp+1)-p.pp(i,j,kpp))+(p.gz(i,j,kg)-p.gz(i,j+1,kg+1))*(p.pp(i,j,kpp+1)-p.pp(i,j+1,kpp)));
  if(c.variant==PressureVariant::split){int64_t kdout=o.dv.lo_k+n;o.v(i,j,kv)+=c.beta*o.dv(i,j,kdout);o.dv(i,j,kdout)=hydro;o.v(i,j,kv)=(o.v(i,j,kv)+(fv_real(1)-c.beta)*hydro+nh)*p.rdy(i,j);}else o.v(i,j,kv)=(o.v(i,j,kv)+hydro+nh)*p.rdy(i,j);
}
__global__ void one_gradient_u_kernel(HorizontalBounds b,OneGradientInputs p,
                                      View3<fv_real> u,fv_real dt,
                                      fv_real d_ext){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x,j=b.js+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;
  if(i>b.ie||j>b.je+1||n>=b.npz)return;int64_t kp=p.pk_b.lo_k+n,kg=p.gz_b.lo_k+n,km=p.mass_b.lo_k+n,ku=u.lo_k+n;
  fv_real ext=d_ext>fv_real(0)?p.divg2(i,j)-p.divg2(i+1,j):fv_real(0);
  u(i,j,ku)=p.rdx(i,j)*(ext+u(i,j,ku)+dt/(p.mass_b(i,j,km)+p.mass_b(i+1,j,km))*((p.gz_b(i,j,kg+1)-p.gz_b(i+1,j,kg))*(p.pk_b(i+1,j,kp+1)-p.pk_b(i,j,kp))+(p.gz_b(i,j,kg)-p.gz_b(i+1,j,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i+1,j,kp))));
}
__global__ void one_gradient_v_kernel(HorizontalBounds b,OneGradientInputs p,
                                      View3<fv_real> v,fv_real dt,
                                      fv_real d_ext){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x,j=b.js+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;
  if(i>b.ie+1||j>b.je||n>=b.npz)return;int64_t kp=p.pk_b.lo_k+n,kg=p.gz_b.lo_k+n,km=p.mass_b.lo_k+n,kv=v.lo_k+n;
  fv_real ext=d_ext>fv_real(0)?p.divg2(i,j)-p.divg2(i,j+1):fv_real(0);
  v(i,j,kv)=p.rdy(i,j)*(ext+v(i,j,kv)+dt/(p.mass_b(i,j,km)+p.mass_b(i,j+1,km))*((p.gz_b(i,j,kg+1)-p.gz_b(i,j+1,kg))*(p.pk_b(i,j+1,kp+1)-p.pk_b(i,j,kp))+(p.gz_b(i,j,kg)-p.gz_b(i,j+1,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i,j+1,kp))));
}
__global__ void grad1_u_kernel(HorizontalBounds b,OneGradientInputs p,
                               View3<fv_real> u,View3<fv_real> du,
                               fv_real dt,fv_real beta){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x,j=b.js+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;
  if(i>b.ie||j>b.je+1||n>=b.npz)return;int64_t kp=p.pk_b.lo_k+n,kg=p.gz_b.lo_k+n,km=p.mass_b.lo_k+n,ku=u.lo_k+n,kd=du.lo_k+n;
  u(i,j,ku)+=beta*du(i,j,kd);du(i,j,kd)=dt/(p.mass_b(i,j,km)+p.mass_b(i+1,j,km))*((p.gz_b(i,j,kg+1)-p.gz_b(i+1,j,kg))*(p.pk_b(i+1,j,kp+1)-p.pk_b(i,j,kp))+(p.gz_b(i,j,kg)-p.gz_b(i+1,j,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i+1,j,kp)));
  u(i,j,ku)=(u(i,j,ku)+p.divg2(i,j)-p.divg2(i+1,j)+(fv_real(1)-beta)*du(i,j,kd))*p.rdx(i,j);
}
__global__ void grad1_v_kernel(HorizontalBounds b,OneGradientInputs p,
                               View3<fv_real> v,View3<fv_real> dv,
                               fv_real dt,fv_real beta){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x,j=b.js+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;
  if(i>b.ie+1||j>b.je||n>=b.npz)return;int64_t kp=p.pk_b.lo_k+n,kg=p.gz_b.lo_k+n,km=p.mass_b.lo_k+n,kv=v.lo_k+n,kd=dv.lo_k+n;
  v(i,j,kv)+=beta*dv(i,j,kd);dv(i,j,kd)=dt/(p.mass_b(i,j,km)+p.mass_b(i,j+1,km))*((p.gz_b(i,j,kg+1)-p.gz_b(i,j+1,kg))*(p.pk_b(i,j+1,kp+1)-p.pk_b(i,j,kp))+(p.gz_b(i,j,kg)-p.gz_b(i,j+1,kg+1))*(p.pk_b(i,j,kp+1)-p.pk_b(i,j+1,kp)));
  v(i,j,kv)=(v(i,j,kv)+p.divg2(i,j)-p.divg2(i,j+1)+(fv_real(1)-beta)*dv(i,j,kd))*p.rdy(i,j);
}
__global__ void del2_corner_kernel(HorizontalBounds b,PhysicalCorners c,View3<fv_real> q){
  int64_t n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=b.npz)return;int64_t k=q.lo_k+n;constexpr fv_real r3=fv_real(1.0/3.0);
  if(c.sw){q(b.is,b.js,k)=r3*(q(b.is,b.js,k)+q(b.is-1,b.js,k)+q(b.is,b.js-1,k));q(b.is-1,b.js,k)=q(b.is,b.js,k);q(b.is,b.js-1,k)=q(b.is,b.js,k);}if(c.se){q(b.ie,b.js,k)=r3*(q(b.ie,b.js,k)+q(b.ie+1,b.js,k)+q(b.ie,b.js-1,k));q(b.ie+1,b.js,k)=q(b.ie,b.js,k);q(b.ie,b.js-1,k)=q(b.ie,b.js,k);}if(c.ne){q(b.ie,b.je,k)=r3*(q(b.ie,b.je,k)+q(b.ie+1,b.je,k)+q(b.ie,b.je+1,k));q(b.ie+1,b.je,k)=q(b.ie,b.je,k);q(b.ie,b.je+1,k)=q(b.ie,b.je,k);}if(c.nw){q(b.is,b.je,k)=r3*(q(b.is,b.je,k)+q(b.is-1,b.je,k)+q(b.is,b.je+1,k));q(b.is-1,b.je,k)=q(b.is,b.je,k);q(b.is,b.je+1,k)=q(b.is,b.je,k);}
}
__global__ void del2_x_kernel(HorizontalBounds b,int nt,bool sg,View3<fv_real> q,Del2Metrics m,Del2SgMetrics s,View3<fv_real> fx){
  int64_t i=b.is-nt+blockIdx.x*blockDim.x+threadIdx.x,j=b.js-nt+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;if(i>b.ie+1+nt||j>b.je+nt||n>=b.npz)return;int64_t k=q.lo_k+n,kf=fx.lo_k+n;fv_real dq=q(i-1,j,k)-q(i,j,k);fx(i,j,kf)=sg?s.dy(i,j)*s.sina_u(i,j)*dq*s.rdxc(i,j):m.del6_v(i,j)*dq;
}
__global__ void del2_y_kernel(HorizontalBounds b,int nt,bool sg,View3<fv_real> q,Del2Metrics m,Del2SgMetrics s,View3<fv_real> fy){
  int64_t i=b.is-nt+blockIdx.x*blockDim.x+threadIdx.x,j=b.js-nt+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;if(i>b.ie+nt||j>b.je+1+nt||n>=b.npz)return;int64_t k=q.lo_k+n,kf=fy.lo_k+n;fv_real dq=q(i,j-1,k)-q(i,j,k);fy(i,j,kf)=sg?s.dx(i,j)*s.sina_v(i,j)*dq*s.rdyc(i,j):m.del6_u(i,j)*dq;
}
__global__ void del2_update_kernel(HorizontalBounds b,int nt,accumulator_real cd,View3<fv_real> q,View2<const fv_real> rarea,View3<fv_real> fx,View3<fv_real> fy){
  int64_t i=b.is-nt+blockIdx.x*blockDim.x+threadIdx.x,j=b.js-nt+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;if(i>b.ie+nt||j>b.je+nt||n>=b.npz)return;int64_t k=q.lo_k+n,kx=fx.lo_k+n,ky=fy.lo_k+n;q(i,j,k)=q(i,j,k)+cd*rarea(i,j)*(fx(i,j,kx)-fx(i+1,j,kx)+fy(i,j,ky)-fy(i,j+1,ky));
}
__global__ void heating_nh_kernel(HorizontalBounds b,fv_real bdt,fv_real delt_max,fv_real cv_air,fv_real rdg,fv_real akap,bool moist,int64_t ncon,View3<fv_real> heat,View3<const fv_real> dp,View3<const fv_real> dz,View3<const fv_real> cappa,View3<fv_real> pkz,View3<fv_real> pt){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x,j=b.js+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;if(i>b.ie||j>b.je||n>=b.npz||n>=ncon)return;fv_real delt=dabs(bdt*delt_max);if(n==0)delt*=fv_real(.1);if(n==1)delt*=fv_real(.5);fv_real kap=moist?cappa(i,j,cappa.lo_k+n):akap;fv_real pz=exp(kap/(fv_real(1)-kap)*log(rdg*dp(i,j,dp.lo_k+n)/dz(i,j,dz.lo_k+n)*pt(i,j,pt.lo_k+n)));pkz(i,j,pkz.lo_k+n)=pz;fv_real dtmp=heat(i,j,heat.lo_k+n)/(cv_air*dp(i,j,dp.lo_k+n));pt(i,j,pt.lo_k+n)+=copysign(fmin(delt,dabs(dtmp)),dtmp)/pz;
}
__global__ void heating_hydro_kernel(HorizontalBounds b,fv_real bdt,
                                     fv_real delt_max,fv_real cp_air,
                                     int64_t ncon,View3<const fv_real> heat,
                                     View3<const fv_real> dp,
                                     View3<const fv_real> pkz,
                                     View3<fv_real> pt){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x,j=b.js+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;
  if(i>b.ie||j>b.je||n>=b.npz||n>=ncon)return;fv_real dtmp=heat(i,j,heat.lo_k+n)/(cp_air*dp(i,j,dp.lo_k+n));fv_real delta=n<2?dtmp:copysign(fmin(dabs(bdt)*delt_max,dabs(dtmp)),dtmp);pt(i,j,pt.lo_k+n)+=delta/pkz(i,j,pkz.lo_k+n);
}

/* omega finalization -- dyn_core.F90:505-516, :1208-1212 and :1535-1636.
 * See the block comment in wp_d_operators.hpp for the authoritative mapping. */
__global__ void omega_pem_kernel(HorizontalBounds b,fv_real ptop,
                                 View3<const fv_real> dp,View3<fv_real> pem){
  int64_t i=b.is-1+blockIdx.x*blockDim.x+threadIdx.x,j=b.js-1+blockIdx.y*blockDim.y+threadIdx.y;
  if(i>b.ie+1||j>b.je+1)return;
  fv_real p=ptop;
  for(int64_t n=0;n<b.npz;++n){p+=dp(i,j,dp.lo_k+n);pem(i,j,pem.lo_k+n)=p;}
}
__global__ void omega_from_pe_kernel(HorizontalBounds b,fv_real rdt,
                                     View3<const fv_real> pe,
                                     View3<fv_real> pem,View3<fv_real> om){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x,j=b.js+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;
  if(i>b.ie||j>b.je||n>=b.npz)return;
  // pe keeps its Fortran (i,k,j) descriptor order: Fortran pe(i,k+1,j) with
  // k=n+1 is pe(i, pe.lo_j+n+1, j).
  om(i,j,om.lo_k+n)=(pe(i,pe.lo_j+n+1,j)-pem(i,j,pem.lo_k+n))*rdt;
}
__global__ void adv_pe_kernel(HorizontalBounds b,View3<const fv_real> ua,
                              View3<const fv_real> va,View3<fv_real> pb,
                              AdvPeMetrics m,View3<fv_real> om){
  int64_t i=b.is+blockIdx.x*blockDim.x+threadIdx.x,j=b.js+blockIdx.y*blockDim.y+threadIdx.y,n=blockIdx.z;
  if(i>b.ie||j>b.je||n>=b.npz)return;
  const int64_t ku=ua.lo_k+n,kv=va.lo_k+n;
  const bool surface=(n==b.npz-1);                       // Fortran k==npz
  const fv_real up=surface?ua(i,j,ku):fv_real(0.5)*(ua(i,j,ku)+ua(i,j,ku+1));
  const fv_real vp=surface?va(i,j,kv):fv_real(0.5)*(va(i,j,kv)+va(i,j,kv+1));
  const accumulator_real inc=adv_pe_cell_increment(m,pb,pb.lo_k+n,i,j,up,vp);
  const int64_t ko=om.lo_k+n;
  om(i,j,ko)=static_cast<fv_real>(
    static_cast<accumulator_real>(om(i,j,ko))+inc);
}

cudaStream_t stream_cast(void *p){return static_cast<cudaStream_t>(p);}
fv3_gpu_status last(){return cudaGetLastError()==cudaSuccess?FV3_GPU_SUCCESS:FV3_GPU_CUDA_ERROR;}
dim3 grid2(int64_t nx,int64_t ny,int64_t nz=1){return dim3((unsigned(nx)+block2.x-1)/block2.x,(unsigned(ny)+block2.y-1)/block2.y,(unsigned)nz);}
dim3 grid1(int64_t n,int64_t levels){
  return dim3((static_cast<uint64_t>(n)+block1-1)/block1,
              static_cast<uint64_t>(levels),1);
}
bool valid_bounds(const HorizontalBounds &b){
  return b.is<=b.ie && b.js<=b.je && b.npz>0;
}
bool valid_range(const A2BRange &r){return r.i0<=r.i1&&r.j0<=r.j1;}
bool any_corner(PhysicalCorners c){return c.sw||c.se||c.ne||c.nw;}
bool consistent_corners(PhysicalEdges e,PhysicalCorners c){
  return c.sw==(e.west&&e.south)&&c.se==(e.east&&e.south)&&
         c.ne==(e.east&&e.north)&&c.nw==(e.west&&e.north);
}
bool valid_a2b2_geometry(PhysicalEdges e,const A2BGeometry &g){
  return (!e.west||g.edge_w.data)&&(!e.east||g.edge_e.data)&&
         (!e.south||g.edge_s.data)&&(!e.north||g.edge_n.data);
}
bool valid_a2b4_geometry(PhysicalEdges e,PhysicalCorners c,
                         const A2BGeometry &g){
  return valid_a2b2_geometry(e,g)&&
         (!(e.west||e.east)||g.dxa.data)&&
         (!(e.south||e.north)||g.dya.data)&&
         (!any_corner(c)||(g.grid_lonlat.data&&g.agrid_lonlat.data));
}
bool valid_a2b4_domain(const HorizontalBounds &b,PhysicalEdges e){
  return (!e.west||b.ie-b.is+1>=3)&&(!e.east||b.ie-b.is+1>=3)&&
         (!e.south||b.je-b.js+1>=3)&&(!e.north||b.je-b.js+1>=3);
}
}

fv3_gpu_status launch_reconstruct_pe_halo_cuda(const HorizontalBounds&b,fv_real p,View3<const fv_real>d,View3<fv_real>o,void*s)noexcept{if(!s||!valid_bounds(b)||!d.data||!o.data)return FV3_GPU_INVALID_ARGUMENT;pe_x_kernel<<<grid2(b.je-b.js+1,2),block2,0,stream_cast(s)>>>(b,p,d,o);if(last())return FV3_GPU_CUDA_ERROR;pe_y_kernel<<<grid2(b.ie-b.is+3,2),block2,0,stream_cast(s)>>>(b,p,d,o);return last();}
fv3_gpu_status launch_reconstruct_pk3_halo_cuda(const HorizontalBounds&b,fv_real p,fv_real a,View3<const fv_real>d,View3<fv_real>o,void*s)noexcept{if(!s||!valid_bounds(b)||!d.data||!o.data)return FV3_GPU_INVALID_ARGUMENT;pressure_ring_x<false><<<grid2(b.je-b.js+1,4),block2,0,stream_cast(s)>>>(b,p,a,d,o);if(last())return FV3_GPU_CUDA_ERROR;pressure_ring_y<false><<<grid2(b.ie-b.is+5,4),block2,0,stream_cast(s)>>>(b,p,a,d,o);return last();}
fv3_gpu_status launch_reconstruct_pln_halo_cuda(const HorizontalBounds&b,fv_real p,View3<const fv_real>d,View3<fv_real>o,void*s)noexcept{if(!s||!valid_bounds(b)||!d.data||!o.data)return FV3_GPU_INVALID_ARGUMENT;pressure_ring_x<true><<<grid2(b.je-b.js+1,4),block2,0,stream_cast(s)>>>(b,p,0,d,o);if(last())return FV3_GPU_CUDA_ERROR;pressure_ring_y<true><<<grid2(b.ie-b.is+5,4),block2,0,stream_cast(s)>>>(b,p,0,d,o);return last();}
fv3_gpu_status launch_geopotential_from_height_cuda(const HorizontalBounds&b,int h,fv_real g,View3<const fv_real>z,View3<fv_real>o,void*s)noexcept{if(!s||!valid_bounds(b)||h<0||!z.data||!o.data)return FV3_GPU_INVALID_ARGUMENT;geopotential_kernel<<<grid2(b.ie-b.is+1+2*h,b.je-b.js+1+2*h,b.npz+1),block2,0,stream_cast(s)>>>(b,h,g,z,o);return last();}
fv3_gpu_status launch_a2b_ord2_centered_cuda(const HorizontalBounds&b,int64_t l,View3<const fv_real>q,View3<fv_real>o,void*s)noexcept{if(!s||!valid_bounds(b)||l<1||!q.data||!o.data)return FV3_GPU_INVALID_ARGUMENT;a2b2_kernel<<<grid2(b.ie-b.is+2,b.je-b.js+2,l),block2,0,stream_cast(s)>>>(b,l,q,o);return last();}
fv3_gpu_status launch_a2b_ord4_centered_cuda(const HorizontalBounds&b,int64_t l,View3<const fv_real>q,View3<fv_real>o,View3<fv_real>x,View3<fv_real>y,void*s)noexcept{if(!s||!valid_bounds(b)||l<1||!q.data||!o.data||!x.data||!y.data)return FV3_GPU_INVALID_ARGUMENT;a2b4_qx_kernel<<<grid2(b.ie-b.is+2,b.je-b.js+5,l),block2,0,stream_cast(s)>>>(b,l,q,x);if(last())return FV3_GPU_CUDA_ERROR;a2b4_qy_kernel<<<grid2(b.ie-b.is+5,b.je-b.js+2,l),block2,0,stream_cast(s)>>>(b,l,q,y);if(last())return FV3_GPU_CUDA_ERROR;a2b4_out_kernel<<<grid2(b.ie-b.is+2,b.je-b.js+2,l),block2,0,stream_cast(s)>>>(b,l,o,x,y);return last();}
fv3_gpu_status launch_a2b_ord2_global_cuda(
    const HorizontalBounds&b,PhysicalEdges e,PhysicalCorners c,int64_t l,
    View3<fv_real>q,View3<fv_real>o,const A2BGeometry&g,bool replace,
    void*s)noexcept{
  if(!s||!valid_bounds(b)||l<1||l>65535||!q.data||!o.data||q.data==o.data||
     !consistent_corners(e,c)||!valid_a2b2_geometry(e,g))
    return FV3_GPU_INVALID_ARGUMENT;
  const cudaStream_t stream=stream_cast(s);
  const A2BRange interior{e.west?b.is+1:b.is,e.east?b.ie:b.ie+1,
                          e.south?b.js+1:b.js,e.north?b.je:b.je+1};
  if(valid_range(interior)){
    a2b2_global_interior_kernel<<<
      grid2(interior.i1-interior.i0+1,interior.j1-interior.j0+1,l),
      block2,0,stream>>>(interior,l,q,o);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(any_corner(c)){
    a2b2_global_corner_kernel<<<
      (static_cast<uint64_t>(l)+block1-1)/block1,block1,0,stream>>>
      (b,c,l,q,o);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  const int64_t first_j=e.south?b.js+1:b.js;
  const int64_t last_j=e.north?b.je:b.je+1;
  if(e.west){
    a2b2_global_x_edge_kernel<<<grid1(last_j-first_j+1,l),block1,0,stream>>>
      (b,false,first_j,last_j,l,q,o,g);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(e.east){
    a2b2_global_x_edge_kernel<<<grid1(last_j-first_j+1,l),block1,0,stream>>>
      (b,true,first_j,last_j,l,q,o,g);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  const int64_t first_i=e.west?b.is+1:b.is;
  const int64_t last_i=e.east?b.ie:b.ie+1;
  if(e.south){
    a2b2_global_y_edge_kernel<<<grid1(last_i-first_i+1,l),block1,0,stream>>>
      (b,false,first_i,last_i,l,q,o,g);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(e.north){
    a2b2_global_y_edge_kernel<<<grid1(last_i-first_i+1,l),block1,0,stream>>>
      (b,true,first_i,last_i,l,q,o,g);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(replace){
    a2b_replace_kernel<<<grid2(b.ie-b.is+2,b.je-b.js+2,l),block2,0,stream>>>
      (b,l,q,o);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  return FV3_GPU_SUCCESS;
}

fv3_gpu_status launch_a2b_ord4_global_cuda(
    const HorizontalBounds&b,PhysicalEdges e,PhysicalCorners c,int64_t l,
    View3<fv_real>q,View3<fv_real>o,const A2BGeometry&g,
    View3<fv_real>qx,View3<fv_real>qy,View3<fv_real>qxx,
    View3<fv_real>qyy,bool replace,void*s)noexcept{
  if(!s||!valid_bounds(b)||l<1||l>65535||!q.data||!o.data||!qx.data||!qy.data||
     !qxx.data||!qyy.data||q.data==o.data||q.data==qx.data||q.data==qy.data||
     q.data==qxx.data||q.data==qyy.data||o.data==qx.data||o.data==qy.data||
     o.data==qxx.data||o.data==qyy.data||qx.data==qy.data||qx.data==qxx.data||
     qx.data==qyy.data||qy.data==qxx.data||qy.data==qyy.data||
     qxx.data==qyy.data||!consistent_corners(e,c)||
     !valid_a2b4_geometry(e,c,g)||!valid_a2b4_domain(b,e))
    return FV3_GPU_INVALID_ARGUMENT;
  const cudaStream_t stream=stream_cast(s);
  if(any_corner(c)){
    a2b4_global_corner_kernel<<<
      dim3(4,static_cast<unsigned>(l),1),1,0,stream>>>
      (b,c,l,q,o,g);
    if(last())return FV3_GPU_CUDA_ERROR;
  }

  const A2BRange qx_range{e.west?b.is+2:b.is,e.east?b.ie-1:b.ie+1,
                           e.south?b.js:b.js-2,e.north?b.je:b.je+2};
  if(valid_range(qx_range)){
    a2b4_global_qx_kernel<<<
      grid2(qx_range.i1-qx_range.i0+1,qx_range.j1-qx_range.j0+1,l),
      block2,0,stream>>>(qx_range,l,q,qx);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  const int64_t output_j0=e.south?b.js+1:b.js;
  const int64_t output_j1=e.north?b.je:b.je+1;
  if(e.west){
    a2b4_global_x_output_kernel<<<grid1(output_j1-output_j0+1,l),block1,0,stream>>>
      (b,false,output_j0,output_j1,l,q,o,g);
    if(last())return FV3_GPU_CUDA_ERROR;
    a2b4_global_x_adjust_kernel<<<
      grid1(qx_range.j1-qx_range.j0+1,l),block1,0,stream>>>
      (b,false,qx_range.j0,qx_range.j1,l,q,qx,g);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(e.east){
    a2b4_global_x_output_kernel<<<grid1(output_j1-output_j0+1,l),block1,0,stream>>>
      (b,true,output_j0,output_j1,l,q,o,g);
    if(last())return FV3_GPU_CUDA_ERROR;
    a2b4_global_x_adjust_kernel<<<
      grid1(qx_range.j1-qx_range.j0+1,l),block1,0,stream>>>
      (b,true,qx_range.j0,qx_range.j1,l,q,qx,g);
    if(last())return FV3_GPU_CUDA_ERROR;
  }

  const A2BRange qy_range{e.west?b.is:b.is-2,e.east?b.ie:b.ie+2,
                           e.south?b.js+2:b.js,e.north?b.je-1:b.je+1};
  if(valid_range(qy_range)){
    a2b4_global_qy_kernel<<<
      grid2(qy_range.i1-qy_range.i0+1,qy_range.j1-qy_range.j0+1,l),
      block2,0,stream>>>(qy_range,l,q,qy);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  const int64_t output_i0=e.west?b.is+1:b.is;
  const int64_t output_i1=e.east?b.ie:b.ie+1;
  if(e.south){
    a2b4_global_y_output_kernel<<<grid1(output_i1-output_i0+1,l),block1,0,stream>>>
      (b,false,output_i0,output_i1,l,q,o,g);
    if(last())return FV3_GPU_CUDA_ERROR;
    a2b4_global_y_adjust_kernel<<<
      grid1(qy_range.i1-qy_range.i0+1,l),block1,0,stream>>>
      (b,false,qy_range.i0,qy_range.i1,l,q,qy,g);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(e.north){
    a2b4_global_y_output_kernel<<<grid1(output_i1-output_i0+1,l),block1,0,stream>>>
      (b,true,output_i0,output_i1,l,q,o,g);
    if(last())return FV3_GPU_CUDA_ERROR;
    a2b4_global_y_adjust_kernel<<<
      grid1(qy_range.i1-qy_range.i0+1,l),block1,0,stream>>>
      (b,true,qy_range.i0,qy_range.i1,l,q,qy,g);
    if(last())return FV3_GPU_CUDA_ERROR;
  }

  const A2BRange qxx_range{e.west?b.is+1:b.is,e.east?b.ie:b.ie+1,
                            e.south?b.js+2:b.js,e.north?b.je-1:b.je+1};
  if(valid_range(qxx_range)){
    a2b4_global_qxx_kernel<<<
      grid2(qxx_range.i1-qxx_range.i0+1,qxx_range.j1-qxx_range.j0+1,l),
      block2,0,stream>>>(qxx_range,l,qx,qxx);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(e.south){
    a2b4_global_qxx_adjust_kernel<<<
      grid1(qxx_range.i1-qxx_range.i0+1,l),block1,0,stream>>>
      (b,false,qxx_range.i0,qxx_range.i1,l,o,qx,qxx);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(e.north){
    a2b4_global_qxx_adjust_kernel<<<
      grid1(qxx_range.i1-qxx_range.i0+1,l),block1,0,stream>>>
      (b,true,qxx_range.i0,qxx_range.i1,l,o,qx,qxx);
    if(last())return FV3_GPU_CUDA_ERROR;
  }

  const int64_t final_j0=e.south?b.js+1:b.js;
  const int64_t final_j1=e.north?b.je:b.je+1;
  const A2BRange qyy_range{e.west?b.is+2:b.is,e.east?b.ie-1:b.ie+1,
                            final_j0,final_j1};
  if(valid_range(qyy_range)){
    a2b4_global_qyy_kernel<<<
      grid2(qyy_range.i1-qyy_range.i0+1,qyy_range.j1-qyy_range.j0+1,l),
      block2,0,stream>>>(qyy_range,l,qy,qyy);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(e.west){
    a2b4_global_qyy_adjust_kernel<<<
      grid1(final_j1-final_j0+1,l),block1,0,stream>>>
      (b,false,final_j0,final_j1,l,o,qy,qyy);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(e.east){
    a2b4_global_qyy_adjust_kernel<<<
      grid1(final_j1-final_j0+1,l),block1,0,stream>>>
      (b,true,final_j0,final_j1,l,o,qy,qyy);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  const A2BRange final_range{qxx_range.i0,qxx_range.i1,final_j0,final_j1};
  if(valid_range(final_range)){
    a2b4_global_output_kernel<<<
      grid2(final_range.i1-final_range.i0+1,final_range.j1-final_range.j0+1,l),
      block2,0,stream>>>(final_range,l,o,qxx,qyy);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  if(replace){
    a2b_replace_kernel<<<grid2(b.ie-b.is+2,b.je-b.js+2,l),block2,0,stream>>>
      (b,l,q,o);
    if(last())return FV3_GPU_CUDA_ERROR;
  }
  return FV3_GPU_SUCCESS;
}
fv3_gpu_status launch_set_pressure_top_cuda(const HorizontalBounds&b,fv_real t,View3<fv_real>p,View3<fv_real>k,void*s)noexcept{if(!s||!valid_bounds(b)||!p.data||!k.data)return FV3_GPU_INVALID_ARGUMENT;set_top_kernel<<<grid2(b.ie-b.is+2,b.je-b.js+2),block2,0,stream_cast(s)>>>(b,t,p,k);return last();}
fv3_gpu_status launch_pressure_gradient_cuda(const HorizontalBounds&b,const PressureInputs&p,const PressureOutputs&o,const PressureConfig&c,void*s)noexcept{if(!s||!valid_bounds(b)||!p.pp.data||!p.gz.data||!p.delp_b.data||!p.pk_b.data||!p.rdx.data||!p.rdy.data||!o.u.data||!o.v.data||(c.variant==PressureVariant::split&&(!o.du.data||!o.dv.data)))return FV3_GPU_INVALID_ARGUMENT;pressure_u_kernel<<<grid2(b.ie-b.is+1,b.je-b.js+2,b.npz),block2,0,stream_cast(s)>>>(b,p,o,c);if(last())return FV3_GPU_CUDA_ERROR;pressure_v_kernel<<<grid2(b.ie-b.is+2,b.je-b.js+1,b.npz),block2,0,stream_cast(s)>>>(b,p,o,c);return last();}
fv3_gpu_status launch_one_gradient_pressure_cuda(const HorizontalBounds&b,const OneGradientInputs&p,View3<fv_real>u,View3<fv_real>v,fv_real dt,fv_real de,void*s)noexcept{if(!s||!valid_bounds(b)||!p.pk_b.data||!p.gz_b.data||!p.mass_b.data||!p.divg2.data||!p.rdx.data||!p.rdy.data||!u.data||!v.data)return FV3_GPU_INVALID_ARGUMENT;one_gradient_u_kernel<<<grid2(b.ie-b.is+1,b.je-b.js+2,b.npz),block2,0,stream_cast(s)>>>(b,p,u,dt,de);if(last())return FV3_GPU_CUDA_ERROR;one_gradient_v_kernel<<<grid2(b.ie-b.is+2,b.je-b.js+1,b.npz),block2,0,stream_cast(s)>>>(b,p,v,dt,de);return last();}
fv3_gpu_status launch_grad1_pressure_update_cuda(const HorizontalBounds&b,const OneGradientInputs&p,View3<fv_real>u,View3<fv_real>v,View3<fv_real>du,View3<fv_real>dv,fv_real dt,fv_real beta,void*s)noexcept{if(!s||!valid_bounds(b)||!p.pk_b.data||!p.gz_b.data||!p.mass_b.data||!p.divg2.data||!p.rdx.data||!p.rdy.data||!u.data||!v.data||!du.data||!dv.data)return FV3_GPU_INVALID_ARGUMENT;grad1_u_kernel<<<grid2(b.ie-b.is+1,b.je-b.js+2,b.npz),block2,0,stream_cast(s)>>>(b,p,u,du,dt,beta);if(last())return FV3_GPU_CUDA_ERROR;grad1_v_kernel<<<grid2(b.ie-b.is+2,b.je-b.js+1,b.npz),block2,0,stream_cast(s)>>>(b,p,v,dv,dt,beta);return last();}
fv3_gpu_status launch_average_del2_corners_cuda(const HorizontalBounds&b,PhysicalCorners c,View3<fv_real>q,void*s)noexcept{if(!s||!valid_bounds(b)||!q.data)return FV3_GPU_INVALID_ARGUMENT;del2_corner_kernel<<<(unsigned(b.npz)+127)/128,128,0,stream_cast(s)>>>(b,c,q);return last();}
fv3_gpu_status launch_del2_zonal_flux_cuda(const HorizontalBounds&b,int n,bool sg,View3<fv_real>q,const Del2Metrics&m,const Del2SgMetrics&g,View3<fv_real>f,void*s)noexcept{if(!s||!valid_bounds(b)||n<0||!q.data||!f.data||(sg?(!g.dy.data||!g.sina_u.data||!g.rdxc.data):!m.del6_v.data))return FV3_GPU_INVALID_ARGUMENT;del2_x_kernel<<<grid2(b.ie-b.is+2+2*n,b.je-b.js+1+2*n,b.npz),block2,0,stream_cast(s)>>>(b,n,sg,q,m,g,f);return last();}
fv3_gpu_status launch_del2_meridional_flux_cuda(const HorizontalBounds&b,int n,bool sg,View3<fv_real>q,const Del2Metrics&m,const Del2SgMetrics&g,View3<fv_real>f,void*s)noexcept{if(!s||!valid_bounds(b)||n<0||!q.data||!f.data||(sg?(!g.dx.data||!g.sina_v.data||!g.rdyc.data):!m.del6_u.data))return FV3_GPU_INVALID_ARGUMENT;del2_y_kernel<<<grid2(b.ie-b.is+1+2*n,b.je-b.js+2+2*n,b.npz),block2,0,stream_cast(s)>>>(b,n,sg,q,m,g,f);return last();}
fv3_gpu_status launch_del2_update_cuda(const HorizontalBounds&b,int n,accumulator_real cd,View3<fv_real>q,View2<const fv_real>r,View3<fv_real>x,View3<fv_real>y,void*s)noexcept{if(!s||!valid_bounds(b)||n<0||!q.data||!r.data||!x.data||!y.data)return FV3_GPU_INVALID_ARGUMENT;del2_update_kernel<<<grid2(b.ie-b.is+1+2*n,b.je-b.js+1+2*n,b.npz),block2,0,stream_cast(s)>>>(b,n,cd,q,r,x,y);return last();}
fv3_gpu_status launch_diffusive_heating_nh_cuda(const HorizontalBounds&b,fv_real bt,fv_real dm,fv_real cv,fv_real rdg,fv_real ak,bool moist,int64_t nc,View3<fv_real>h,View3<const fv_real>dp,View3<const fv_real>dz,View3<const fv_real>c,View3<fv_real>pk,View3<fv_real>pt,void*s)noexcept{if(!s||!valid_bounds(b)||nc<0||!h.data||!dp.data||!dz.data||(moist&&!c.data)||!pk.data||!pt.data)return FV3_GPU_INVALID_ARGUMENT;if(nc==0)return FV3_GPU_SUCCESS;heating_nh_kernel<<<grid2(b.ie-b.is+1,b.je-b.js+1,(nc<b.npz?nc:b.npz)),block2,0,stream_cast(s)>>>(b,bt,dm,cv,rdg,ak,moist,nc,h,dp,dz,c,pk,pt);return last();}
fv3_gpu_status launch_diffusive_heating_hydrostatic_cuda(const HorizontalBounds&b,fv_real bt,fv_real dm,fv_real cp,int64_t nc,View3<const fv_real>h,View3<const fv_real>dp,View3<const fv_real>pk,View3<fv_real>pt,void*s)noexcept{if(!s||!valid_bounds(b)||nc<0||!h.data||!dp.data||!pk.data||!pt.data)return FV3_GPU_INVALID_ARGUMENT;if(nc==0)return FV3_GPU_SUCCESS;heating_hydro_kernel<<<grid2(b.ie-b.is+1,b.je-b.js+1,(nc<b.npz?nc:b.npz)),block2,0,stream_cast(s)>>>(b,bt,dm,cp,nc,h,dp,pk,pt);return last();}
fv3_gpu_status launch_pem_snapshot_cuda(const HorizontalBounds&b,fv_real ptop,View3<const fv_real>dp,View3<fv_real>pem,void*s)noexcept{if(!s||!valid_bounds(b)||!dp.data||!pem.data)return FV3_GPU_INVALID_ARGUMENT;omega_pem_kernel<<<grid2(b.ie-b.is+3,b.je-b.js+3),block2,0,stream_cast(s)>>>(b,ptop,dp,pem);return last();}
fv3_gpu_status launch_omega_from_pe_cuda(const HorizontalBounds&b,fv_real rdt,View3<const fv_real>pe,View3<fv_real>pem,View3<fv_real>om,void*s)noexcept{if(!s||!valid_bounds(b)||!pe.data||!pem.data||!om.data)return FV3_GPU_INVALID_ARGUMENT;omega_from_pe_kernel<<<grid2(b.ie-b.is+1,b.je-b.js+1,b.npz),block2,0,stream_cast(s)>>>(b,rdt,pe,pem,om);return last();}
fv3_gpu_status launch_adv_pe_cuda(const HorizontalBounds&b,View3<const fv_real>ua,View3<const fv_real>va,View3<fv_real>pb,const AdvPeMetrics&m,View3<fv_real>om,void*s)noexcept{if(!s||!valid_bounds(b)||!ua.data||!va.data||!pb.data||!om.data||!m.grid_lonlat.data||!m.dx.data||!m.dy.data||!m.rarea.data)return FV3_GPU_INVALID_ARGUMENT;adv_pe_kernel<<<grid2(b.ie-b.is+1,b.je-b.js+1,b.npz),block2,0,stream_cast(s)>>>(b,ua,va,pb,m,om);return last();}

} // namespace fv3::gpu::operators
