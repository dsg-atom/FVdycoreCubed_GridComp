// Milestone 3A.3 WP-B height-transport CUDA smoke test.
//
// Exercises fv3::gpu::dgrid::launch_dgrid_height_transport_cuda (the new
// WP-B CUDA path for WP-C's update_dz_d dependency) directly against a real
// GPU context, then separately exercises the standalone
// launch_deln_flux_transport_cuda kernels (implemented but not wired into
// the production launch path -- see docs/agent_handoff/milestone_3a3_wp_b_cuda.md,
// "damp_c mapping ambiguity").
//
// GPU EXECUTED verification only -- NOT a numerical validation test (no
// authoritative Fortran reference is available in this checkout; see
// discover_platform.md).

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"
#include "fv3_gpu_internal.hpp"
#include "fv3_gpu_context.hpp"
#include "fv3_gpu_errors.hpp"
#include "dgrid/wp_b_cuda.cuh"
#include "dgrid/wp_b_contract_cuda.hpp"
#include "vertical/fv3_vertical_kernels.cuh"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace fv3::gpu;

static constexpr int64_t IS = 1, IE = 8, JS = 1, JE = 8;
static constexpr int64_t NPZ = 10;
static constexpr uint32_t HALO = 3;
static constexpr int64_t NI_H = (IE - IS + 1) + 2 * (int64_t)HALO;
static constexpr int64_t NJ_H = (JE - JS + 1) + 2 * (int64_t)HALO;
static constexpr int64_t NPZ1 = NPZ + 1;

fv3_gpu_field_desc make_field(FieldId id, void *base, int rank,
                              const int64_t *lower, const uint64_t *extent,
                              const int64_t *stride, uint64_t bytes) {
  fv3_gpu_field_desc d{};
  d.abi_version = FV3_GPU_ABI_VERSION;
  d.struct_bytes = sizeof(d);
  d.field_id = static_cast<uint32_t>(id);
  d.base = base;
  d.allocation_bytes = bytes;
  d.element_type = sizeof(fv_real) == 4 ? FV3_GPU_ELEMENT_F32 : FV3_GPU_ELEMENT_F64;
  d.element_bytes = sizeof(fv_real);
  d.rank = rank;
  d.memory_location = FV3_GPU_MEMORY_HOST;
  const auto *spec = field_spec(id);
  d.stagger = spec ? spec->stagger : FV3_GPU_STAGGER_A;
  d.access = spec ? spec->access : FV3_GPU_ACCESS_INOUT;
  for (int i = 0; i < rank; ++i) {
    d.lower[i] = lower[i];
    d.extent[i] = extent[i];
    d.stride_elements[i] = stride[i];
    d.compute_lower[i] = lower[i];
    d.compute_upper[i] = lower[i] + (int64_t)extent[i] - 1;
    d.halo_lower[i] = 0;
    d.halo_upper[i] = 0;
  }
  return d;
}

bool check_finite(const fv_real *data, size_t count, const char *name) {
  size_t nan_count = 0, inf_count = 0;
  for (size_t i = 0; i < count; ++i) {
    if (std::isnan(data[i])) ++nan_count;
    if (std::isinf(data[i])) ++inf_count;
  }
  if (nan_count > 0 || inf_count > 0) {
    printf("  %s: FAIL (%zu NaN, %zu Inf out of %zu)\n", name, nan_count, inf_count, count);
    return false;
  }
  printf("  %s: finite (%zu values)\n", name, count);
  return true;
}

bool any_changed(const fv_real *before, const fv_real *after, size_t count) {
  for (size_t i = 0; i < count; ++i)
    if (std::fabs(static_cast<double>(after[i]) - static_cast<double>(before[i])) > 1e-12)
      return true;
  return false;
}

// Local View3/View2 construction matching operators::View3/View2's actual
// field order (data, lo_i, lo_j, lo_k, si, sj, sk) -- i.e. DeviceViewBase's
// axis 0/1/2 map directly to i/j/k. This mirrors wp_b_contract_cuda.cpp's
// to_view3/to_view2 (dgrid/wp_b_contract_cuda.cpp:12-22), which use the same
// direct mapping. NOTE: tests/integration/test_integrated_smoke.cpp builds
// View3 with a different (transposed) positional mapping and mislabels the
// constructor arguments in its comments; that file is unrelated to WP-B and
// is not modified by this milestone, but the discrepancy is flagged in
// docs/agent_handoff/milestone_3a3_wp_b_cuda.md as an out-of-scope finding.
template <class T>
operators::View3<T> to_view3(const DeviceViewBase *v) {
  return operators::View3<T>{static_cast<T *>(v->data), v->lower[0], v->lower[1], v->lower[2],
                             v->stride[0], v->stride[1], v->stride[2]};
}

template <class T>
operators::View2<T> to_view2(const DeviceViewBase *v) {
  return operators::View2<T>{static_cast<T *>(v->data), v->lower[0], v->lower[1], v->stride[0],
                             v->stride[1]};
}

// Matches dgrid/wp_b_contract_cuda.cpp's to_arena_view3: each arena array is
// a batched View3 over the full extended horizontal box [is-h,ie+h] x
// [js-h,je+h] x [0,npz].
operators::View3<fv_real> to_arena_view3(void *p, const DomainBounds &b) {
  const int64_t h = static_cast<int64_t>(b.halo_width);
  const int64_t ni = (b.ie + h) - (b.is - h) + 1;
  const int64_t nj = (b.je + h) - (b.js - h) + 1;
  return operators::View3<fv_real>{static_cast<fv_real *>(p), b.is - h, b.js - h, 0, 1, ni, ni * nj};
}

int corner_service_unused(const ExecutionContext &, const CornerServiceRequest &,
                          fv3_gpu_error *e) noexcept {
  return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, __func__,
              "corner_edge_service is not exercised by this smoke test");
}

int main() {
  printf("=== WP-B dgrid_height_transport CUDA Smoke Test (Milestone 3A.3) ===\n\n");
  printf("Domain: is=%ld ie=%ld js=%ld je=%ld npz=%ld halo=%u\n", IS, IE, JS, JE, NPZ, HALO);
  printf("Precision: %s (sizeof(fv_real)=%zu)\n\n", sizeof(fv_real) == 4 ? "R4" : "R8",
         sizeof(fv_real));

  fv3_gpu_capabilities caps{};
  caps.abi_version = FV3_GPU_ABI_VERSION;
  caps.struct_bytes = sizeof(caps);
  int rc = fv3_gpu_get_capabilities(nullptr, &caps);
  assert(rc == 0);
  if (!caps.compiled_with_cuda || caps.device_count == 0) {
    printf("SKIP: no CUDA device available (compiled_with_cuda=%u, device_count=%u)\n",
           caps.compiled_with_cuda, caps.device_count);
    return 77;
  }

  fv3_gpu_error err{};
  fv3_gpu_context *ctx = nullptr;
  fv3_gpu_create_desc create{};
  create.abi_version = FV3_GPU_ABI_VERSION;
  create.struct_bytes = sizeof(create);
  create.device_ordinal = 0;
  create.mpi_rank = 0;
  create.mpi_size = 1;
  create.precision = sizeof(fv_real) == 4 ? FV3_GPU_PRECISION_R4 : FV3_GPU_PRECISION_R8;
  create.is = IS; create.ie = IE; create.js = JS; create.je = JE;
  create.npz = NPZ; create.halo_width = HALO;

  rc = fv3_gpu_create(&create, &ctx, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: fv3_gpu_create returned %d: %s\n", rc, err.message);
    return 1;
  }
  printf("WP-E context created\n\n");

  // ---- Host field storage ----
  const size_t size_if = (size_t)NI_H * NJ_H * NPZ1;      // zh: box A (k in [0,npz])
  const size_t size_face = (size_t)(NI_H + 1) * (NJ_H + 1) * NPZ1; // crx/cry/xfx/yfx: box B
  const size_t size_2d = (size_t)NI_H * NJ_H;              // area/rarea/del6_u/del6_v: box C

  std::vector<fv_real> h_zh(size_if), h_crx(size_face), h_cry(size_face);
  std::vector<fv_real> h_xfx(size_face), h_yfx(size_face);
  std::vector<fv_real> h_area(size_2d), h_rarea(size_2d), h_del6_u(size_2d), h_del6_v(size_2d);

  // Additional fields for Part 3 (launch_update_dz_d cross-WP check): delz is
  // a mass-layer field (npz levels, k in [0,npz)); ws/zs are 2D surface
  // fields. Sized identically to zh/area's boxes respectively.
  const size_t size_delz = (size_t)NI_H * NJ_H * NPZ;
  std::vector<fv_real> h_delz(size_delz, fv_real(0));
  std::vector<fv_real> h_ws(size_2d, fv_real(0));
  std::vector<fv_real> h_zs(size_2d, fv_real(0));

  const fv_real cell_area = 1.0e8; // representative ~100km cell (m^2)
  for (int64_t k = 0; k < NPZ1; ++k) {
    fv_real z = 10000.0 - (fv_real)k * 1000.0;
    for (int64_t j = 0; j < NJ_H; ++j) {
      for (int64_t i = 0; i < NI_H; ++i) {
        size_t idx = (size_t)i + (size_t)j * NI_H + (size_t)k * NI_H * NJ_H;
        h_zh[idx] = z + 5.0 * std::sin(0.3 * (fv_real)i) * std::cos(0.3 * (fv_real)j);
      }
    }
  }
  for (int64_t k = 0; k < NPZ1; ++k) {
    for (int64_t j = 0; j < NJ_H + 1; ++j) {
      for (int64_t i = 0; i < NI_H + 1; ++i) {
        size_t idx = (size_t)i + (size_t)j * (NI_H + 1) + (size_t)k * (NI_H + 1) * (NJ_H + 1);
        h_crx[idx] = 0.10; h_cry[idx] = 0.08;
        h_xfx[idx] = 0.02 * cell_area; h_yfx[idx] = 0.015 * cell_area;
      }
    }
  }
  for (size_t idx = 0; idx < size_2d; ++idx) {
    h_area[idx] = cell_area;
    h_rarea[idx] = 1.0 / cell_area;
    h_del6_u[idx] = 1.0;
    h_del6_v[idx] = 1.0;
    h_zs[idx] = -500.0; // surface height well below zh's lowest interface
  }
  for (int64_t k = 0; k < NPZ; ++k)
    for (int64_t j = 0; j < NJ_H; ++j)
      for (int64_t i = 0; i < NI_H; ++i) {
        size_t idx = (size_t)i + (size_t)j * NI_H + (size_t)k * NI_H * NJ_H;
        h_delz[idx] = -1000.0; // placeholder; height_boundary_v2_kernel recomputes this
      }

  const std::vector<fv_real> h_zh_before = h_zh;

  const int64_t lo_if[3] = {IS - (int64_t)HALO, JS - (int64_t)HALO, 0};
  const uint64_t ex_if[3] = {(uint64_t)NI_H, (uint64_t)NJ_H, (uint64_t)NPZ1};
  const int64_t st_if[3] = {1, NI_H, NI_H * NJ_H};
  const uint64_t bytes_if = size_if * sizeof(fv_real);

  const int64_t lo_face[3] = {IS - (int64_t)HALO, JS - (int64_t)HALO, 0};
  const uint64_t ex_face[3] = {(uint64_t)(NI_H + 1), (uint64_t)(NJ_H + 1), (uint64_t)NPZ1};
  const int64_t st_face[3] = {1, NI_H + 1, (NI_H + 1) * (NJ_H + 1)};
  const uint64_t bytes_face = size_face * sizeof(fv_real);

  const int64_t lo_2d[2] = {IS - (int64_t)HALO, JS - (int64_t)HALO};
  const uint64_t ex_2d[2] = {(uint64_t)NI_H, (uint64_t)NJ_H};
  const int64_t st_2d[2] = {1, NI_H};
  const uint64_t bytes_2d = size_2d * sizeof(fv_real);

  const int64_t lo_delz[3] = {IS - (int64_t)HALO, JS - (int64_t)HALO, 0};
  const uint64_t ex_delz[3] = {(uint64_t)NI_H, (uint64_t)NJ_H, (uint64_t)NPZ};
  const int64_t st_delz[3] = {1, NI_H, NI_H * NJ_H};
  const uint64_t bytes_delz = size_delz * sizeof(fv_real);

  std::vector<fv3_gpu_field_desc> fields;
  fields.push_back(make_field(FieldId::zh, h_zh.data(), 3, lo_if, ex_if, st_if, bytes_if));
  fields.push_back(make_field(FieldId::crx, h_crx.data(), 3, lo_face, ex_face, st_face, bytes_face));
  fields.push_back(make_field(FieldId::cry, h_cry.data(), 3, lo_face, ex_face, st_face, bytes_face));
  fields.push_back(make_field(FieldId::xfx, h_xfx.data(), 3, lo_face, ex_face, st_face, bytes_face));
  fields.push_back(make_field(FieldId::yfx, h_yfx.data(), 3, lo_face, ex_face, st_face, bytes_face));
  fields.push_back(make_field(FieldId::area, h_area.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::rarea, h_rarea.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::del6_u, h_del6_u.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::del6_v, h_del6_v.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::delz, h_delz.data(), 3, lo_delz, ex_delz, st_delz, bytes_delz));
  fields.push_back(make_field(FieldId::ws, h_ws.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));
  fields.push_back(make_field(FieldId::zs, h_zs.data(), 2, lo_2d, ex_2d, st_2d, bytes_2d));

  fv3_gpu_state_desc state{};
  state.abi_version = FV3_GPU_ABI_VERSION;
  state.struct_bytes = sizeof(state);
  state.field_count = static_cast<uint32_t>(fields.size());
  state.fields = fields.data();

  fv3_gpu_event ev = 0;
  rc = fv3_gpu_initialize_state(ctx, &state, &ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: fv3_gpu_initialize_state returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }
  rc = fv3_gpu_event_wait(ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: fv3_gpu_event_wait (H2D) returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }
  printf("H2D transfer complete\n\n");

  NumericalConfig cfg{};
  cfg.hord_tm = 6; cfg.hord_mt = 6; cfg.hord_vt = 6; cfg.hord_dp = -6;
  cfg.grid_type = 0; cfg.hydrostatic = 0; cfg.a_imp = 1.0; cfg.dz_min = 2.0;
  rc = set_numerical_config(ctx, cfg, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: set_numerical_config returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }
  printf("NumericalConfig installed (hord_tm=6)\n\n");

  auto *c = reinterpret_cast<ContextData *>(ctx);

  ExecutionContext x{};
  x.contract_version = internal_contract_version;
  x.fields = &c->state.registry;
  x.immutable = &c->immutable;
  x.dispatch_params = &c->dispatch_params;
  x.workspace = &c->workspace.layout;
  x.submission = Submission{internal_contract_version, c->compute_stream, StreamId::compute, 1};
  x.dependencies = &c->dependencies;
  x.corner_service = &corner_service_unused;

  // ========== Part 1: launch_dgrid_height_transport_cuda ==========
  printf("Part 1: launch_dgrid_height_transport_cuda (fv_tp_2d_hord6 + flux divergence on zh)\n");
  rc = dgrid::launch_dgrid_height_transport_cuda(x, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: launch_dgrid_height_transport_cuda returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }
  cudaStreamSynchronize(static_cast<cudaStream_t>(c->compute_stream));
  printf("  launch succeeded, compute stream synchronized\n\n");

  fv3_gpu_field_mask mask{};
  mask.abi_version = FV3_GPU_ABI_VERSION;
  mask.struct_bytes = sizeof(mask);
  mask.words[static_cast<uint32_t>(FieldId::zh) / 64] |=
      (1ULL << (static_cast<uint32_t>(FieldId::zh) % 64));
  rc = fv3_gpu_sync_to_host(ctx, &mask, &ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: fv3_gpu_sync_to_host returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }
  rc = fv3_gpu_event_wait(ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: fv3_gpu_event_wait (D2H) returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }

  bool part1_pass = check_finite(h_zh.data(), h_zh.size(), "zh (after transport)");
  bool zh_changed = any_changed(h_zh_before.data(), h_zh.data(), h_zh.size());
  printf("  zh changed vs. pre-call snapshot: %s\n", zh_changed ? "YES" : "NO");
  part1_pass = part1_pass && zh_changed;
  printf("  Part 1: %s\n\n", part1_pass ? "PASS" : "FAIL");

  // ========== Part 2: launch_deln_flux_transport_cuda (standalone) ==========
  printf("Part 2: launch_deln_flux_transport_cuda (standalone, has_mass=false path)\n");
  dgrid::WpBHeightTransportViews views2;
  dgrid::WpBHeightTransportArena arena2;
  rc = dgrid::bind_wp_b_height_transport(x, views2, arena2, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: bind_wp_b_height_transport returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }

  const auto &db = c->immutable.bounds;
  dgrid::HeightTransportBounds b{db.is, db.ie, db.js, db.je, db.npz,
                                 static_cast<int64_t>(db.halo_width)};

  operators::View3<const fv_real> q = to_view3<const fv_real>(views2.zh);
  operators::View2<const fv_real> del6_u = to_view2<const fv_real>(views2.del6_u);
  operators::View2<const fv_real> del6_v = to_view2<const fv_real>(views2.del6_v);
  operators::View2<const fv_real> rarea_v = to_view2<const fv_real>(views2.rarea);
  // arena layout: fx=0, fy=1, q_work=2, fx2=3, fy2=4, d2=5, ra_x=6, ra_y=7
  operators::View3<fv_real> fx = to_arena_view3(arena2.arrays[0], db);
  operators::View3<fv_real> fy = to_arena_view3(arena2.arrays[1], db);
  operators::View3<fv_real> d2 = to_arena_view3(arena2.arrays[5], db);
  operators::View3<fv_real> fx2 = to_arena_view3(arena2.arrays[3], db);
  operators::View3<fv_real> fy2 = to_arena_view3(arena2.arrays[4], db);

  fv3_gpu_status st = dgrid::launch_deln_flux_transport_cuda(
      b, /*nord=*/0, /*damp=*/0.02, /*da_min=*/1.0e14, q, del6_u, del6_v, rarea_v,
      fx, fy, d2, fx2, fy2, x.submission.stream);
  if (st != FV3_GPU_SUCCESS) {
    printf("FAIL: launch_deln_flux_transport_cuda returned %d\n", (int)st);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }
  cudaStreamSynchronize(static_cast<cudaStream_t>(c->compute_stream));

  const int64_t h = static_cast<int64_t>(db.halo_width);
  const int64_t arena_ni = (db.ie + h) - (db.is - h) + 1;
  const int64_t arena_nj = (db.je + h) - (db.js - h) + 1;
  const size_t arena_count = (size_t)arena_ni * arena_nj * (db.npz + 1);
  std::vector<fv_real> h_fx(arena_count), h_fy(arena_count), h_d2(arena_count);
  cudaMemcpy(h_fx.data(), arena2.arrays[0], arena_count * sizeof(fv_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fy.data(), arena2.arrays[1], arena_count * sizeof(fv_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_d2.data(), arena2.arrays[5], arena_count * sizeof(fv_real), cudaMemcpyDeviceToHost);

  bool part2_pass = check_finite(h_fx.data(), h_fx.size(), "fx (deln_flux)");
  part2_pass = check_finite(h_fy.data(), h_fy.size(), "fy (deln_flux)") && part2_pass;
  part2_pass = check_finite(h_d2.data(), h_d2.size(), "d2 (deln_flux)") && part2_pass;
  printf("  launch succeeded, outputs finite: %s\n", part2_pass ? "YES" : "NO");
  printf("  Part 2: %s\n\n", part2_pass ? "PASS" : "FAIL");

  // ========== Part 3: launch_update_dz_d (WP-C) cross-WP integration ==========
  // Proves WP-C's launch_update_dz_d genuinely calls into the new WP-B
  // dgrid_height_transport kernel end-to-end, not just WP-B in isolation.
  // launch_update_dz_d requires FV3_GPU_H_DSCALAR ready and a nonzero rdt in
  // DispatchParams; ContextData's members are plain public fields (see
  // fv3_gpu_context.hpp), so direct mutation through the reinterpret_cast'd
  // `c` (already used above to reach compute_stream/state.registry) is the
  // same pattern used to install rdt here -- there is no public setter for
  // DispatchParams, unlike NumericalConfig's set_numerical_config.
  printf("Part 3: launch_update_dz_d (WP-C) cross-WP integration via production dispatch\n");
  rc = mark_halo_ready(ctx, FV3_GPU_H_DSCALAR, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: mark_halo_ready(H_DSCALAR) returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }
  c->dispatch_params.rdt = 1.0 / 300.0;

  NumericalDispatch dispatch{};
  dispatch.contract_version = internal_contract_version;
  dispatch.dgrid_height_transport = &dgrid::launch_dgrid_height_transport_cuda;

  const std::vector<fv_real> h_zh_before_part3 = h_zh;
  rc = vertical::launch_update_dz_d(x, dispatch, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: launch_update_dz_d returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }
  cudaStreamSynchronize(static_cast<cudaStream_t>(c->compute_stream));
  printf("  launch succeeded, compute stream synchronized\n\n");

  fv3_gpu_field_mask mask3{};
  mask3.abi_version = FV3_GPU_ABI_VERSION;
  mask3.struct_bytes = sizeof(mask3);
  mask3.words[static_cast<uint32_t>(FieldId::zh) / 64] |=
      (1ULL << (static_cast<uint32_t>(FieldId::zh) % 64));
  mask3.words[static_cast<uint32_t>(FieldId::delz) / 64] |=
      (1ULL << (static_cast<uint32_t>(FieldId::delz) % 64));
  mask3.words[static_cast<uint32_t>(FieldId::ws) / 64] |=
      (1ULL << (static_cast<uint32_t>(FieldId::ws) % 64));
  rc = fv3_gpu_sync_to_host(ctx, &mask3, &ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: fv3_gpu_sync_to_host (Part 3) returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }
  rc = fv3_gpu_event_wait(ctx, ev, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: fv3_gpu_event_wait (Part 3 D2H) returned %d: %s\n", rc, err.message);
    fv3_gpu_finalize(&ctx, &err);
    return 1;
  }

  bool part3_pass = check_finite(h_zh.data(), h_zh.size(), "zh (after update_dz_d)");
  part3_pass = check_finite(h_delz.data(), h_delz.size(), "delz (after update_dz_d)") && part3_pass;
  part3_pass = check_finite(h_ws.data(), h_ws.size(), "ws (after update_dz_d)") && part3_pass;
  bool zh_changed_part3 = any_changed(h_zh_before_part3.data(), h_zh.data(), h_zh.size());
  printf("  zh changed via launch_update_dz_d (WP-C) -> dgrid_height_transport (WP-B): %s\n",
         zh_changed_part3 ? "YES" : "NO");
  part3_pass = part3_pass && zh_changed_part3;
  printf("  Part 3: %s\n\n", part3_pass ? "PASS" : "FAIL");

  rc = fv3_gpu_finalize(&ctx, &err);
  if (rc != FV3_GPU_SUCCESS) {
    printf("FAIL: fv3_gpu_finalize returned %d: %s\n", rc, err.message);
    return 1;
  }

  printf("=== Summary ===\n");
  printf("  Part 1 (dgrid_height_transport): %s\n", part1_pass ? "PASS" : "FAIL");
  printf("  Part 2 (deln_flux_transport, standalone): %s\n", part2_pass ? "PASS" : "FAIL");
  printf("  Part 3 (launch_update_dz_d cross-WP integration): %s\n", part3_pass ? "PASS" : "FAIL");
  printf("  Numerical validation: NOT RUN (no authoritative Fortran reference available)\n");

  if (part1_pass && part2_pass && part3_pass) {
    printf("\nWP-B CUDA smoke test: PASS\n");
    return 0;
  }
  printf("\nWP-B CUDA smoke test: FAIL\n");
  return 1;
}
