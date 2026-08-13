#pragma once

#include "fv3_gpu_api.h"
#include "fv3_gpu_communication.h"
#include "fv3_gpu_types.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace fv3::gpu {

inline constexpr uint32_t internal_contract_version = 2;

enum class FieldId : uint32_t {
  invalid=0, u, v, w, delz, pt, delp, q, q_con, cappa, pe, pk, peln,
  pkz, phis, omga, ua, va, uc, vc, mfx, mfy, cx, cy, dpx, diss_est, ws,
  delpc, ptc, ut, vt, divgd, crx, xfx, cry, yfx, gz, zh, pkc, pk3, du,
  dv, heat_source, ws3, z_rat, divg2,
  ak, bk, pfull, dp_ref, zs, area, rarea, dx, dy, dxc, dyc, rdx, rdy,
  rdxc, rdyc, cosa_u, cosa_v, sina_u, sina_v, rsin_u, rsin_v, sin_sg,
  cos_sg, f0, del6_u, del6_v, edge_geometry, corner_geometry,
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
  double rdgas=0, grav=0, bdt=0;
  int32_t hord=0, nord=0, n_split=1, k_split=1, vertical_solver=0;
  uint8_t use_cond=0, moist_cappa=0, use_logp=0, inline_q=0;
  uint8_t hydrostatic=0, nested=0, rot3=0, beta_nonzero=0;
};
struct ImmutableViews {
  uint32_t contract_version=internal_contract_version;
  DomainBounds bounds{};
  NumericalConfig config{};
  const FieldRegistry *grid=nullptr;
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
  const WorkspaceLayout *workspace=nullptr;
  Submission submission{};
  DependencyState *dependencies=nullptr;
};

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
  LaunchFn local_corner_fill=nullptr;
};

int validate_execution_context(const ExecutionContext &, fv3_gpu_error *) noexcept;
int validate_dispatch(const NumericalDispatch &, fv3_gpu_error *) noexcept;

}
