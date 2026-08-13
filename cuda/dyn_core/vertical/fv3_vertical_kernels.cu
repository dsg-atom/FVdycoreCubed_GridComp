#include "fv3_vertical_kernels.cuh"
#include "fv3_vertical_column.hpp"
#include <cuda_runtime.h>

namespace fv3::gpu::vertical {
namespace {
struct View3 { fv_real *p; int64_t l0,l1,l2,s0,s1,s2; __device__ fv_real &at(int64_t i,int64_t j,int64_t k)const{return p[(i-l0)*s0+(j-l1)*s1+(k-l2)*s2];} };
struct View2 { fv_real *p; int64_t l0,l1,s0,s1; __device__ fv_real &at(int64_t i,int64_t j)const{return p[(i-l0)*s0+(j-l1)*s1];} };
__global__ void reconstruct_height_kernel(fv_real *zh, const fv_real *delz,
                                          const fv_real *zs,
                                          ColumnLaunch l) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= l.columns) return;
  Column<fv_real> z{zh + c * l.column_stride, l.level_stride};
  Column<const fv_real> dz{delz + c * l.column_stride, l.level_stride};
  reconstruct_height_column(z, dz, zs[c], l.levels);
}

__global__ void height_boundary_kernel(fv_real *zh, fv_real *delz, fv_real *ws,
                                       const fv_real *zs, fv_real rdt,
                                       fv_real dz_min, ColumnLaunch l) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= l.columns) return;
  Column<fv_real> z{zh + c * l.column_stride, l.level_stride};
  Column<fv_real> dz{delz + c * l.column_stride, l.level_stride};
  ws[c] = update_height_boundary_column(z, dz, zs[c], rdt, dz_min, l.levels);
}

__global__ void height_boundary_v2_kernel(View3 zh,View3 delz,View2 ws,View2 zs,
 int64_t is,int64_t ie,int64_t js,int64_t je,int km,fv_real rdt,fv_real dz_min){
  const int64_t i=is+blockIdx.x*blockDim.x+threadIdx.x,j=js+blockIdx.y*blockDim.y+threadIdx.y;
  if(i>ie||j>je)return;ws.at(i,j)=(zs.at(i,j)-zh.at(i,j,km))*rdt;
  for(int k=km-1;k>=0;--k){const fv_real floor=zh.at(i,j,k+1)+dz_min;if(zh.at(i,j,k)<floor)zh.at(i,j,k)=floor;delz.at(i,j,k)=zh.at(i,j,k+1)-zh.at(i,j,k);}
}
View3 view3(const DeviceViewBase&v){return{static_cast<fv_real*>(v.data),v.lower[0],v.lower[1],v.lower[2],v.stride[0],v.stride[1],v.stride[2]};}
View2 view2(const DeviceViewBase&v){return{static_cast<fv_real*>(v.data),v.lower[0],v.lower[1],v.stride[0],v.stride[1]};}
} // namespace

int launch_reconstruct_height(fv_real *zh, const fv_real *delz,
                              const fv_real *zs, ColumnLaunch l, void *stream) {
  if (!zh || !delz || !zs || l.columns <= 0 || l.levels < 2) return 1;
  constexpr int threads = 128;
  reconstruct_height_kernel<<<(l.columns + threads - 1) / threads, threads, 0,
                                static_cast<cudaStream_t>(stream)>>>(zh, delz, zs, l);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

int launch_height_boundary(fv_real *zh, fv_real *delz, fv_real *ws,
                           const fv_real *zs, fv_real rdt, fv_real dz_min,
                           ColumnLaunch l, void *stream) {
  if (!zh || !delz || !ws || !zs || l.columns <= 0 || l.levels < 2) return 1;
  constexpr int threads = 128;
  height_boundary_kernel<<<(l.columns + threads - 1) / threads, threads, 0,
                            static_cast<cudaStream_t>(stream)>>>(
      zh, delz, ws, zs, rdt, dz_min, l);
  return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

int launch_dgrid_height_v2(const ExecutionContext &x,fv3_gpu_error *error) {
  const int context_status=validate_execution_context(x,error);
  if(context_status!=FV3_GPU_SUCCESS)return context_status;
  if(x.submission.stream_id!=StreamId::compute)return FV3_GPU_INVALID_ARGUMENT;
  fv3_gpu_event ready{};if(x.dependencies->require(FV3_GPU_H_DSCALAR,&ready)!=FV3_GPU_SUCCESS)return FV3_GPU_STATE_CONFLICT;
  const auto *z=x.fields->find(FieldId::zh),*d=x.fields->find(FieldId::delz),*w=x.fields->find(FieldId::ws),*s=x.fields->find(FieldId::zs);
  if(!z||!d||!w||!s)return FV3_GPU_INVALID_ARGUMENT;
  const auto &b=x.immutable->bounds;const auto &c=x.immutable->config;
  const auto &p=*x.dispatch_params;
  if(b.npz<2||p.rdt==0)return FV3_GPU_UNSUPPORTED_CONFIGURATION;
  const dim3 t(16,8),g((b.ie-b.is+16)/16,(b.je-b.js+8)/8);
  height_boundary_v2_kernel<<<g,t,0,static_cast<cudaStream_t>(x.submission.stream)>>>(view3(*z),view3(*d),view2(*w),view2(*s),b.is,b.ie,b.js,b.je,(int)b.npz,fv_real(p.rdt),fv_real(c.dz_min));
  return cudaGetLastError()==cudaSuccess?FV3_GPU_SUCCESS:FV3_GPU_CUDA_ERROR;
}
} // namespace fv3::gpu::vertical
