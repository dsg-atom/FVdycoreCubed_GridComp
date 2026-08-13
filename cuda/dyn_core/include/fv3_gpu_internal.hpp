#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_communication.h"
#include "fv3_gpu_types.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace fv3::gpu {

inline constexpr uint32_t internal_contract_version = 3;

enum class FieldId : uint32_t {
  invalid=0, u, v, w, delz, pt, delp, q, q_con, cappa, pe, pk, peln,
  pkz, phis, omga, ua, va, uc, vc, mfx, mfy, cx, cy, dpx, diss_est, ws,
  delpc, ptc, ut, vt, divgd, crx, xfx, cry, yfx, gz, zh, pkc, pk3, du,
  dv, heat_source, ws3, z_rat, divg2,
  ak, bk, pfull, dp_ref, zs, area, rarea, dx, dy, dxc, dyc, rdx, rdy,
  rdxc, rdyc, cosa_u, cosa_v, sina_u, sina_v, rsin_u, rsin_v, sin_sg,
  cos_sg, f0, del6_u, del6_v,
  cosa_s, rsin2, dxa, dya, rarea_c, area_c, f_c,
  rdxa, rdya, cosa, rsina, divg_u, divg_v,
  grid_lonlat, agrid_lonlat, edge_w, edge_e, edge_s, edge_n,
  damping_d2_bg, damping_d4_bg, damping_v, damping_w, damping_t,
  damping_kgb, damping_ndif,
  count
};

enum class FieldClass : uint8_t { mutable_state, immutable_grid, shared_temporary };
enum class LogicalType : uint8_t { fv_real, accumulator_real, integer, bytes };

struct FieldSpec {
  FieldId id;
  const char *name;
  FieldClass field_class;
  LogicalType logical_type;
  uint8_t rank;
  fv3_gpu_stagger stagger;
  fv3_gpu_access access;
};

const FieldSpec *field_spec(FieldId id) noexcept;
FieldId field_id_from_abi(uint32_t id) noexcept;

struct DeviceViewBase {
  void *data = nullptr;
  uint32_t rank = 0;
  std::array<int64_t,FV3_GPU_MAX_RANK> lower{};
  std::array<uint64_t,FV3_GPU_MAX_RANK> extent{};
  std::array<int64_t,FV3_GPU_MAX_RANK> stride{};
  fv3_gpu_stagger stagger = FV3_GPU_STAGGER_A;
  fv3_gpu_access access = FV3_GPU_ACCESS_INPUT;
};

template<class T> struct DeviceView : DeviceViewBase {
  using element_type = T;
  T *base() const noexcept { return static_cast<T*>(data); }
  int64_t offset(int64_t i, int64_t j=0, int64_t k=0, int64_t q=0) const noexcept {
    const int64_t x[4]={i,j,k,q}; int64_t result=0;
    for(uint32_t n=0;n<rank;++n) result+=(x[n]-lower[n])*stride[n];
    return result;
  }
};

int make_device_view(const fv3_gpu_field_desc &, const FieldSpec &,
                     DeviceViewBase &, fv3_gpu_error *) noexcept;

struct FieldRegistry {
  std::array<DeviceViewBase,static_cast<size_t>(FieldId::count)> views{};
  std::array<uint8_t,static_cast<size_t>(FieldId::count)> present{};
  int insert(const fv3_gpu_field_desc &, void *device, fv3_gpu_error *) noexcept;
  const DeviceViewBase *find(FieldId id) const noexcept;
};

struct DomainBounds { int64_t is,ie,js,je,npz; uint32_t halo_width; };
struct NumericalConfig {
  double ptop=0, akap=0, a_imp=1, p_fac=0, scale_m=1, dz_min=0;
  double rdgas=0, grav=0;
  double beta=0, d_ext=0, dddmp=0, d_con=0, lim_fac=0, zvir=0;
  double da_min=0, da_min_c=0, delt_max=0;
  double rayleigh_tau=0, rayleigh_ztop=0;
  int32_t hord_tr=0, hord_mt=0, hord_vt=0, hord_tm=0, hord_dp=0;
  int32_t hord_height=0, a2b_order=4;
  int32_t nord=0, nord_v=0, nord_w=0, nord_t=0;
  int32_t nf_ke=0, n_con=0, vertical_solver=0;
  uint8_t use_cond=0, moist_cappa=0, use_logp=0, inline_q=0;
  uint8_t hydrostatic=0, rot3=0, do_f3d=0, do_skeb=0;
  uint8_t use_sg=0, rayleigh_enabled=0, grid_type=0, reserved=0;
};

enum class TileEdge : uint8_t { west, east, south, north };
enum class EdgeOrientation : uint8_t { aligned, reversed, swap_xy, swap_xy_reversed };
struct EdgeNeighbor { int32_t rank=-1; int16_t tile=-1; TileEdge edge=TileEdge::west; EdgeOrientation orientation=EdgeOrientation::aligned; int8_t vector_sign_x=1, vector_sign_y=1; };
struct TopologyView {
  uint32_t contract_version=internal_contract_version;
  int32_t tile=0;
  std::array<EdgeNeighbor,4> neighbors{};
  uint8_t global_cubed_sphere=1, nested=0, regional=0, periodic_x=0, periodic_y=0;
  uint8_t sw_corner=0, se_corner=0, ne_corner=0, nw_corner=0;
  std::array<int8_t,4> corner_owner_rank{{-1,-1,-1,-1}};
};
struct ImmutableViews {
  uint32_t contract_version=internal_contract_version;
  DomainBounds bounds{};
  NumericalConfig config{};
  const FieldRegistry *grid=nullptr;
  const TopologyView *topology=nullptr;
};

struct DispatchParams {
  uint32_t contract_version=internal_contract_version;
  double dt=0, dt2=0, rdt=0;
  uint64_t model_step=0, acoustic_sequence=0;
  int32_t acoustic_index=0, n_split=1, k_split=1, vertical_ms=1;
  int32_t diffusion_pass=0, diffusion_passes=0;
  uint8_t init_step=0, end_step=0, last_call=0, fp_out=0, remap_step=0;
};

enum class WorkspaceId : uint8_t { cgrid, transport, vertical, operators, count };
struct WorkspaceRequest { WorkspaceId id; uint64_t bytes; uint32_t alignment; };
struct WorkspaceSlice { void *data=nullptr; uint64_t offset=0, bytes=0; uint32_t alignment=1; };
struct WorkspaceLayout {
  uint64_t total_bytes=0;
  std::array<WorkspaceSlice,static_cast<size_t>(WorkspaceId::count)> slices{};
};
int build_workspace_layout(const WorkspaceRequest *, size_t, WorkspaceLayout &,
                           fv3_gpu_error *) noexcept;
const WorkspaceSlice *workspace_slice(const WorkspaceLayout &, WorkspaceId) noexcept;

enum class StreamId : uint8_t { compute, transfer_pack };
struct Submission {
  uint32_t contract_version=internal_contract_version;
  void *stream=nullptr;                 // borrowed; runtime retains ownership
  StreamId stream_id=StreamId::compute;
  uint64_t sequence=0;
};

enum class HaloState : uint8_t { unknown, ready, stale, exchange_pending };
struct DependencyState {
  std::array<HaloState,FV3_GPU_H_DWIND_NEXT+1> halo{};
  std::array<fv3_gpu_event,FV3_GPU_H_DWIND_NEXT+1> ready_event{};
  int require(fv3_gpu_halo_group, fv3_gpu_event *) const noexcept;
  int mark_stale(fv3_gpu_halo_group) noexcept;
  int mark_pending(fv3_gpu_halo_group, fv3_gpu_event) noexcept;
  int mark_ready(fv3_gpu_halo_group, fv3_gpu_event) noexcept;
};

struct ExecutionContext {
  uint32_t contract_version=internal_contract_version;
  const FieldRegistry *fields=nullptr;
  const ImmutableViews *immutable=nullptr;
  const DispatchParams *dispatch_params=nullptr;
  const WorkspaceLayout *workspace=nullptr;
  Submission submission{};
  DependencyState *dependencies=nullptr;
};

enum class CornerOperation : uint8_t { fill, copy, fill_pair, vector_transform, a_to_b_edge };
enum class Axis : uint8_t { x, y, both };
enum class ComponentOrientation : uint8_t { scalar, x_component, y_component, vector_pair };
struct CornerServiceRequest {
  uint32_t contract_version=internal_contract_version;
  CornerOperation operation=CornerOperation::fill;
  FieldId first=FieldId::invalid, second=FieldId::invalid;
  fv3_gpu_stagger stagger=FV3_GPU_STAGGER_A;
  Axis axis=Axis::x;
  ComponentOrientation orientation=ComponentOrientation::scalar;
  uint32_t halo_width=1;
  int32_t interpolation_order=0;
  int32_t pass_index=0;
  int8_t sign_x=1, sign_y=1;
  fv3_gpu_event dependency=0;
};
using CornerServiceFn = int (*)(const ExecutionContext &, const CornerServiceRequest &, fv3_gpu_error *);
int validate_corner_request(const ExecutionContext &, const CornerServiceRequest &, fv3_gpu_error *) noexcept;

using LaunchFn = int (*)(const ExecutionContext &, fv3_gpu_error *);
struct NumericalDispatch {
  uint32_t contract_version=internal_contract_version;
  LaunchFn cgrid_predictor=nullptr;
  LaunchFn cgrid_vertical_solver=nullptr;
  LaunchFn cgrid_pressure=nullptr;
  LaunchFn dgrid_dynamics=nullptr;
  LaunchFn dgrid_height_transport=nullptr; // WP-B operator called by WP-C sequencing
  LaunchFn dgrid_height=nullptr;
  LaunchFn vertical_solver=nullptr;
  LaunchFn nh_pressure=nullptr;
  LaunchFn post_loop_heating=nullptr;
  CornerServiceFn corner_edge_service=nullptr;
};

int validate_execution_context(const ExecutionContext &, fv3_gpu_error *) noexcept;
int validate_dispatch(const NumericalDispatch &, fv3_gpu_error *) noexcept;

}
