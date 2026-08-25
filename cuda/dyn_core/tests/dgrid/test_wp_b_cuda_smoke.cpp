// Milestone 3A.4 WP-B height-transport CUDA behavior test.
//
// Supersedes the Milestone 3A.3 version of this file, which exercised a
// provisional dgrid_height_transport (additive flux divergence, no
// edge_profile, no copy_corners, no del6_vt_flux, symmetric ra_x/ra_y) and
// whose fixture no longer satisfies the authoritative binder contract.
//
// Structure: one parametric Fixture per scenario, then focused parts:
//   A  interior order-6 transport, no panel edge, no corner
//   B  panel-edge PPM behavior (cubed-sphere seam vs. interior stencils)
//   C  copy_corners rotation, both directions, against a host transcription
//   D  edge_profile tridiagonal, against a host transcription
//   E  undamped branch (damp <= 1.E-5)
//   F  damped branch (damp > 1.E-5), including a mixed per-level profile
//   G  del6_vt_flux behavior (raw damp linearity + nord=0 host reference)
//   H  authoritative area-weighted zh update, both branches, host reference
//   I  cross-WP: WP-C launch_update_dz_d -> corrected WP-B transport
//   J  contract rejection: damping_ndif element width, absent npx/npy
//
// VALIDATION LEVEL: GPU EXECUTED plus targeted host-reference agreement for
// individually transcribed stages (edge_profile, copy_corners, del6 nord=0, zh
// update). This is NOT authoritative numerical validation: no Fortran-generated
// savepoint is compared. See docs/agent_handoff/milestone_3a4_wp_b_height_transport_repair.md.

#include "fv3_gpu_api.h"
#include "fv3_gpu_types.hpp"
#include "fv3_gpu_internal.hpp"
#include "fv3_gpu_context.hpp"
#include "fv3_gpu_errors.hpp"
#include "dgrid/wp_b_cuda.cuh"
#include "dgrid/wp_b_contract_cuda.hpp"
#include "vertical/fv3_vertical_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace fv3::gpu;

// Arena slot indices. COUPLED to bind_wp_b_height_transport's documented
// ordering in dgrid/wp_b_contract_cuda.cpp -- if that mapping changes, these
// must change with it.
enum : size_t {
  A_FX = 0, A_FY = 1, A_QI = 2, A_QJ = 3, A_FX2_TP = 4, A_FY2_TP = 5,
  A_RA_X = 6, A_RA_Y = 7, A_Z2 = 8, A_D2 = 9, A_FX2_D6 = 10, A_FY2_D6 = 11,
  A_CRX_ADV = 12, A_CRY_ADV = 13, A_XFX_ADV = 14, A_YFX_ADV = 15, A_GAM = 16
};

static const double TOL = sizeof(fv_real) == 4 ? 2e-5 : 1e-11;

static bool close_rel(double a, double b, double tol = TOL) {
  const double scale = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
  return std::fabs(a - b) <= tol * scale;
}

static bool check_finite(const std::vector<fv_real> &v, const char *name) {
  size_t bad = 0;
  for (double d : v)
    if (std::isnan(d) || std::isinf(d)) ++bad;
  if (bad) { printf("    %s: FAIL (%zu non-finite of %zu)\n", name, bad, v.size()); return false; }
  return true;
}

static bool bitwise_equal(const std::vector<fv_real> &a, const std::vector<fv_real> &b) {
  if (a.size() != b.size()) return false;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(fv_real)) == 0;
}

// ---------------------------------------------------------------- Spec/Fixture

struct Spec {
  int64_t is = 1, ie = 12, js = 1, je = 12, npz = 8;
  uint32_t halo = 3;
  int64_t npx = 13, npy = 13;
  uint8_t cubed = 1, nested = 0;
  uint8_t sw = 0, se = 0, ne = 0, nw = 0;
  int32_t nord_v = 2;
  // 0 = all levels undamped, 1 = all damped, 2 = damped on even k only.
  int damp_mode = 1;
  fv_real damp_value = fv_real(0.01);
  fv_real ndif_value = fv_real(2);
  // Zero means "native fv_real width". Nonzero deliberately registers a
  // mismatched storage width for the damping_ndif contract test.
  uint32_t ndif_bytes = 0;
  // Omit npx/npy from the TopologyView (contract-rejection test).
  bool skip_npxy = false;
};

struct Fixture {
  Spec s;
  fv3_gpu_context *ctx = nullptr;
  ContextData *c = nullptr;
  ExecutionContext x{};
  fv3_gpu_error err{};
  int64_t h = 0, ni = 0, nj = 0, npz1 = 0, fni = 0, fnj = 0;
  size_t n3 = 0, nface = 0, n2 = 0, ndelz = 0, narena = 0;
  std::vector<fv_real> zh, crx, cry, xfx, yfx, area, rarea, dxa, dya;
  std::vector<fv_real> del6_u, del6_v, delz, ws, zs, dp_ref, damp_v, ndif;
  std::vector<fv_real> zh0;
  std::vector<fv3_gpu_field_desc> fields;

  int64_t lo_i() const { return s.is - h; }
  int64_t lo_j() const { return s.js - h; }
  size_t i3(int64_t i, int64_t j, int64_t k) const {
    return (size_t)((i - lo_i()) + (j - lo_j()) * ni + k * ni * nj);
  }
  size_t iface(int64_t i, int64_t j, int64_t k) const {
    return (size_t)((i - lo_i()) + (j - lo_j()) * fni + k * fni * fnj);
  }
  size_t i2(int64_t i, int64_t j) const { return (size_t)((i - lo_i()) + (j - lo_j()) * ni); }
  // Arena arrays share zh's horizontal box with npz+1 levels.
  size_t ia(int64_t i, int64_t j, int64_t k) const { return i3(i, j, k); }

  ~Fixture() { if (ctx) fv3_gpu_finalize(&ctx, &err); }

  fv3_gpu_field_desc desc(FieldId id, void *base, int rank, const int64_t *lower,
                          const uint64_t *extent, const int64_t *stride, uint64_t bytes,
                          uint32_t elem_bytes_override = 0) {
    fv3_gpu_field_desc d{};
    d.abi_version = FV3_GPU_ABI_VERSION;
    d.struct_bytes = sizeof(d);
    d.field_id = static_cast<uint32_t>(id);
    d.base = base;
    d.allocation_bytes = bytes;
    const uint32_t eb = elem_bytes_override ? elem_bytes_override : (uint32_t)sizeof(fv_real);
    d.element_bytes = eb;
    // fv3_gpu_validate_field requires element_type and element_bytes to agree,
    // so an override must move both -- that is exactly the hazard the binder's
    // width guard exists to catch.
    d.element_type = eb == 4 ? FV3_GPU_ELEMENT_F32 : FV3_GPU_ELEMENT_F64;
    d.rank = rank;
    d.memory_location = FV3_GPU_MEMORY_HOST;
    const auto *sp = field_spec(id);
    d.stagger = sp ? sp->stagger : FV3_GPU_STAGGER_A;
    d.access = sp ? sp->access : FV3_GPU_ACCESS_INOUT;
    for (int n = 0; n < rank; ++n) {
      d.lower[n] = lower[n];
      d.extent[n] = extent[n];
      d.stride_elements[n] = stride[n];
      d.compute_lower[n] = lower[n];
      d.compute_upper[n] = lower[n] + (int64_t)extent[n] - 1;
    }
    return d;
  }

  // Physically plausible, non-degenerate, non-symmetric synthetic state. No
  // constant fields: a constant area/del6/crx would let a wrong stencil or a
  // wrong extent pass by cancellation.
  void fill() {
    h = (int64_t)s.halo;
    ni = (s.ie + h) - (s.is - h) + 1;
    nj = (s.je + h) - (s.js - h) + 1;
    npz1 = s.npz + 1;
    fni = ni + 1; fnj = nj + 1;
    n3 = (size_t)ni * nj * npz1;
    nface = (size_t)fni * fnj * npz1;
    n2 = (size_t)ni * nj;
    ndelz = (size_t)ni * nj * s.npz;
    narena = n3;

    zh.assign(n3, 0); crx.assign(nface, 0); cry.assign(nface, 0);
    xfx.assign(nface, 0); yfx.assign(nface, 0);
    area.assign(n2, 0); rarea.assign(n2, 0); dxa.assign(n2, 0); dya.assign(n2, 0);
    del6_u.assign(n2, 0); del6_v.assign(n2, 0);
    delz.assign(ndelz, fv_real(-999)); ws.assign(n2, 0); zs.assign(n2, 0);
    dp_ref.assign(s.npz, 0); damp_v.assign(s.npz, 0);
    // Two elements of headroom per level: the J2 contract case deliberately
    // registers damping_ndif with the other precision's element width, and
    // initialize_state copies allocation_bytes = npz * that width out of this
    // buffer. Oversizing keeps that a device-side contract violation to be
    // rejected, not a host out-of-bounds read inside the test itself.
    ndif.assign((size_t)s.npz * 2, 0);

    for (int64_t j = lo_j(); j <= s.je + h; ++j)
      for (int64_t i = lo_i(); i <= s.ie + h; ++i) {
        const double a = 1.0e8 * (1.0 + 0.02 * std::sin(0.4 * i) + 0.03 * std::cos(0.3 * j));
        area[i2(i, j)] = (fv_real)a;
        rarea[i2(i, j)] = (fv_real)(1.0 / a);
        dxa[i2(i, j)] = (fv_real)(1.0e4 * (1.0 + 0.05 * std::sin(0.25 * i)));
        dya[i2(i, j)] = (fv_real)(1.0e4 * (1.0 + 0.04 * std::cos(0.22 * j)));
        del6_u[i2(i, j)] = (fv_real)(0.30 * (1.0 + 0.10 * std::sin(0.3 * i + 0.2 * j)));
        del6_v[i2(i, j)] = (fv_real)(0.28 * (1.0 + 0.12 * std::cos(0.27 * i - 0.19 * j)));
      }

    for (int64_t k = 0; k < npz1; ++k)
      for (int64_t j = lo_j(); j <= s.je + h; ++j)
        for (int64_t i = lo_i(); i <= s.ie + h; ++i)
          zh[i3(i, j, k)] = (fv_real)(12000.0 - 1200.0 * k + 60.0 * std::sin(0.35 * i) *
                                      std::cos(0.28 * j) + 15.0 * std::cos(0.2 * (i + j)));

    for (int64_t k = 0; k < npz1; ++k)
      for (int64_t j = lo_j(); j <= s.je + h + 1; ++j)
        for (int64_t i = lo_i(); i <= s.ie + h + 1; ++i) {
          const double cx = 0.12 * (1.0 + 0.15 * std::sin(0.3 * i) * std::cos(0.25 * j) + 0.04 * k);
          const double cy = 0.10 * (1.0 + 0.13 * std::cos(0.28 * i) * std::sin(0.22 * j) + 0.03 * k);
          crx[iface(i, j, k)] = (fv_real)cx;
          cry[iface(i, j, k)] = (fv_real)cy;
          xfx[iface(i, j, k)] = (fv_real)(cx * 1.0e6 * (1.0 + 0.10 * std::cos(0.2 * i)));
          yfx[iface(i, j, k)] = (fv_real)(cy * 1.0e6 * (1.0 + 0.09 * std::sin(0.18 * j)));
        }

    for (int64_t j = lo_j(); j <= s.je + h; ++j)
      for (int64_t i = lo_i(); i <= s.ie + h; ++i)
        zs[i2(i, j)] = zh[i3(i, j, s.npz)] - fv_real(5);

    // dp_ref must be non-uniform: edge_profile's non-uniform branch is the one
    // update_dz_d takes, and a uniform dp0 would make gk == 1 everywhere and
    // hide a wrong coefficient.
    for (int64_t k = 0; k < s.npz; ++k) dp_ref[k] = (fv_real)(700.0 + 900.0 * k);
    set_damping();
    zh0 = zh;
  }

  void set_damping() {
    for (int64_t k = 0; k < s.npz; ++k) {
      bool on = s.damp_mode == 1 || (s.damp_mode == 2 && (k % 2 == 0));
      damp_v[k] = on ? s.damp_value : fv_real(0);
      ndif[k] = s.ndif_value;
    }
  }

  bool create(const char *label) {
    fill();
    fv3_gpu_create_desc cd{};
    cd.abi_version = FV3_GPU_ABI_VERSION;
    cd.struct_bytes = sizeof(cd);
    cd.device_ordinal = 0; cd.mpi_rank = 0; cd.mpi_size = 1;
    cd.precision = sizeof(fv_real) == 4 ? FV3_GPU_PRECISION_R4 : FV3_GPU_PRECISION_R8;
    cd.is = s.is; cd.ie = s.ie; cd.js = s.js; cd.je = s.je;
    cd.npz = s.npz; cd.halo_width = s.halo;
    int rc = fv3_gpu_create(&cd, &ctx, &err);
    if (rc) { printf("    [%s] fv3_gpu_create failed: %s\n", label, err.message); return false; }

    const int64_t lo3[3] = {lo_i(), lo_j(), 0};
    const uint64_t ex3[3] = {(uint64_t)ni, (uint64_t)nj, (uint64_t)npz1};
    const int64_t st3[3] = {1, ni, ni * nj};
    const int64_t exf[3] = {lo_i(), lo_j(), 0};
    const uint64_t exface[3] = {(uint64_t)fni, (uint64_t)fnj, (uint64_t)npz1};
    const int64_t stface[3] = {1, fni, fni * fnj};
    const int64_t lo2[2] = {lo_i(), lo_j()};
    const uint64_t ex2[2] = {(uint64_t)ni, (uint64_t)nj};
    const int64_t st2[2] = {1, ni};
    const uint64_t exdz[3] = {(uint64_t)ni, (uint64_t)nj, (uint64_t)s.npz};
    const int64_t lo1[1] = {0};
    const uint64_t ex1[1] = {(uint64_t)s.npz};
    const int64_t st1[1] = {1};
    const uint64_t b3 = n3 * sizeof(fv_real), bface = nface * sizeof(fv_real);
    const uint64_t b2 = n2 * sizeof(fv_real), bdz = ndelz * sizeof(fv_real);

    fields.clear();
    fields.push_back(desc(FieldId::zh, zh.data(), 3, lo3, ex3, st3, b3));
    fields.push_back(desc(FieldId::crx, crx.data(), 3, exf, exface, stface, bface));
    fields.push_back(desc(FieldId::cry, cry.data(), 3, exf, exface, stface, bface));
    fields.push_back(desc(FieldId::xfx, xfx.data(), 3, exf, exface, stface, bface));
    fields.push_back(desc(FieldId::yfx, yfx.data(), 3, exf, exface, stface, bface));
    fields.push_back(desc(FieldId::area, area.data(), 2, lo2, ex2, st2, b2));
    fields.push_back(desc(FieldId::rarea, rarea.data(), 2, lo2, ex2, st2, b2));
    fields.push_back(desc(FieldId::dxa, dxa.data(), 2, lo2, ex2, st2, b2));
    fields.push_back(desc(FieldId::dya, dya.data(), 2, lo2, ex2, st2, b2));
    fields.push_back(desc(FieldId::del6_u, del6_u.data(), 2, lo2, ex2, st2, b2));
    fields.push_back(desc(FieldId::del6_v, del6_v.data(), 2, lo2, ex2, st2, b2));
    fields.push_back(desc(FieldId::delz, delz.data(), 3, lo3, exdz, st3, bdz));
    fields.push_back(desc(FieldId::ws, ws.data(), 2, lo2, ex2, st2, b2));
    fields.push_back(desc(FieldId::zs, zs.data(), 2, lo2, ex2, st2, b2));
    fields.push_back(desc(FieldId::dp_ref, dp_ref.data(), 1, lo1, ex1, st1,
                          s.npz * sizeof(fv_real)));
    fields.push_back(desc(FieldId::damping_v, damp_v.data(), 1, lo1, ex1, st1,
                          s.npz * sizeof(fv_real)));
    // damping_ndif is the LogicalType::integer field whose storage width the
    // binder must validate.
    {
      const uint32_t eb = s.ndif_bytes ? s.ndif_bytes : (uint32_t)sizeof(fv_real);
      fields.push_back(desc(FieldId::damping_ndif, ndif.data(), 1, lo1, ex1, st1,
                            (uint64_t)s.npz * eb, s.ndif_bytes));
    }

    fv3_gpu_state_desc sd{};
    sd.abi_version = FV3_GPU_ABI_VERSION;
    sd.struct_bytes = sizeof(sd);
    sd.field_count = (uint32_t)fields.size();
    sd.fields = fields.data();
    fv3_gpu_event ev = 0;
    rc = fv3_gpu_initialize_state(ctx, &sd, &ev, &err);
    if (rc) { printf("    [%s] initialize_state failed: %s\n", label, err.message); return false; }
    if (fv3_gpu_event_wait(ctx, ev, &err)) { printf("    [%s] H2D wait failed\n", label); return false; }

    NumericalConfig cfg{};
    cfg.hord_tm = 6; cfg.hord_mt = 6; cfg.hord_vt = 6; cfg.hord_dp = -6;
    cfg.grid_type = 0; cfg.hydrostatic = 0; cfg.a_imp = 1.0; cfg.dz_min = 2.0;
    cfg.nord_v = s.nord_v;
    if (set_numerical_config(ctx, cfg, &err)) {
      printf("    [%s] set_numerical_config failed: %s\n", label, err.message); return false;
    }

    TopologyView topo{};
    topo.contract_version = internal_contract_version;
    topo.global_cubed_sphere = s.cubed;
    topo.nested = s.nested;
    topo.sw_corner = s.sw; topo.se_corner = s.se;
    topo.ne_corner = s.ne; topo.nw_corner = s.nw;
    if (!s.skip_npxy) { topo.npx = s.npx; topo.npy = s.npy; }
    if (set_topology(ctx, topo, &err)) {
      printf("    [%s] set_topology failed: %s\n", label, err.message); return false;
    }

    c = reinterpret_cast<ContextData *>(ctx);
    x = ExecutionContext{};
    x.contract_version = internal_contract_version;
    x.fields = &c->state.registry;
    x.immutable = &c->immutable;
    x.dispatch_params = &c->dispatch_params;
    x.workspace = &c->workspace.layout;
    x.submission = Submission{internal_contract_version, c->compute_stream, StreamId::compute, 1};
    x.dependencies = &c->dependencies;
    x.corner_service = nullptr;
    // corner_service must be non-null for validate_execution_context; WP-B's
    // height path never invokes it (it does its own copy_corners), so a
    // fail-adapter proves it is not silently relied upon.
    x.corner_service = [](const ExecutionContext &, const CornerServiceRequest &,
                          fv3_gpu_error *e) noexcept -> int {
      return fail(e, FV3_GPU_UNSUPPORTED_CONFIGURATION, "corner_service_unused",
                  "WP-B height transport must not call corner_edge_service");
    };
    return true;
  }

  void sync() { cudaStreamSynchronize(static_cast<cudaStream_t>(c->compute_stream)); }

  bool push(FieldId id) {  // host -> device for one field
    fv3_gpu_field_mask m{};
    m.abi_version = FV3_GPU_ABI_VERSION; m.struct_bytes = sizeof(m);
    m.words[(uint32_t)id / 64] |= (1ULL << ((uint32_t)id % 64));
    fv3_gpu_event ev = 0;
    return !fv3_gpu_sync_from_host(ctx, &m, &ev, &err) && !fv3_gpu_event_wait(ctx, ev, &err);
  }

  bool pull(const std::vector<FieldId> &ids) {  // device -> host
    fv3_gpu_field_mask m{};
    m.abi_version = FV3_GPU_ABI_VERSION; m.struct_bytes = sizeof(m);
    for (FieldId id : ids) m.words[(uint32_t)id / 64] |= (1ULL << ((uint32_t)id % 64));
    fv3_gpu_event ev = 0;
    return !fv3_gpu_sync_to_host(ctx, &m, &ev, &err) && !fv3_gpu_event_wait(ctx, ev, &err);
  }

  std::vector<fv_real> get_arena(const dgrid::WpBHeightTransportArena &a, size_t slot) {
    std::vector<fv_real> out(narena, fv_real(0));
    cudaMemcpy(out.data(), a.arrays[slot], narena * sizeof(fv_real), cudaMemcpyDeviceToHost);
    return out;
  }
  void put_arena(const dgrid::WpBHeightTransportArena &a, size_t slot,
                 const std::vector<fv_real> &v) {
    cudaMemcpy(a.arrays[slot], v.data(), narena * sizeof(fv_real), cudaMemcpyHostToDevice);
  }

  // Zero every scratch slot. Needed only by parts that pull a WHOLE slot back
  // with get_arena: the authoritative stages write bounded sub-boxes (e.g.
  // edge_profile_x writes is..ie+1 x jsd..jed), so the untouched remainder of
  // the workspace allocation is uninitialized device memory and a full-slot
  // D2H is a genuine compute-sanitizer initcheck error -- in the TEST, not in
  // the kernels. This is harness hygiene, not a masking workaround: initcheck
  // independently reports zero device-side uninitialized __global__ reads, and
  // the parts that probe "was this level written?" overwrite these zeros with
  // an explicit SENTINEL afterwards, so no assertion is weakened.
  void zero_arena(const dgrid::WpBHeightTransportArena &a) {
    for (size_t i = 0; i < dgrid::WpBHeightTransportArena::array_count; ++i)
      if (a.arrays[i]) cudaMemset(a.arrays[i], 0, narena * sizeof(fv_real));
    cudaDeviceSynchronize();
  }

  dgrid::HeightTransportBounds bounds() const {
    dgrid::HeightTransportBounds b{};
    b.is = s.is; b.ie = s.ie; b.js = s.js; b.je = s.je; b.npz = s.npz;
    b.halo_width = h; b.ng = 3;
    b.npx = s.npx; b.npy = s.npy;
    b.nested = s.nested ? 1 : 0;
    b.cubed_sphere = s.cubed ? 1 : 0;
    b.west_edge = (s.is == 1); b.east_edge = (s.ie + 1 == s.npx);
    b.south_edge = (s.js == 1); b.north_edge = (s.je + 1 == s.npy);
    b.sw_corner = s.sw; b.se_corner = s.se; b.ne_corner = s.ne; b.nw_corner = s.nw;
    return b;
  }
};

template <class T> static operators::View3<T> v3(const DeviceViewBase *v) {
  return operators::View3<T>{static_cast<T *>(v->data), v->lower[0], v->lower[1], v->lower[2],
                             v->stride[0], v->stride[1], v->stride[2]};
}
template <class T> static operators::View2<T> v2(const DeviceViewBase *v) {
  return operators::View2<T>{static_cast<T *>(v->data), v->lower[0], v->lower[1],
                             v->stride[0], v->stride[1]};
}
template <class T> static operators::View1<T> v1(const DeviceViewBase *v) {
  return operators::View1<T>{static_cast<T *>(v->data), v->lower[0], v->stride[0]};
}
static operators::View3<fv_real> av3(void *p, const Fixture &f) {
  return operators::View3<fv_real>{static_cast<fv_real *>(p), f.lo_i(), f.lo_j(), 0, 1,
                                   f.ni, f.ni * f.nj};
}

// ---------------------------------------------------------------------- Part A

static bool part_a() {
  printf("Part A: interior order-6 transport (no panel edge, no corner)\n");
  Fixture f;
  // Interior block of a c32 panel: is != 1 and ie+1 != npx, so no edge flag
  // fires and copy_corners is inert. Every zh write must land in [is,ie]x[js,je].
  f.s.is = 9; f.s.ie = 20; f.s.js = 9; f.s.je = 20;
  f.s.npx = 33; f.s.npy = 33;
  f.s.damp_mode = 1;
  if (!f.create("A")) return false;
  const auto b = f.bounds();
  if (b.west_edge || b.east_edge || b.south_edge || b.north_edge) {
    printf("    FAIL: fixture unexpectedly on a panel edge\n"); return false;
  }

  const std::vector<fv_real> crx_in = f.crx, area_in = f.area, dpr_in = f.dp_ref;
  if (dgrid::launch_dgrid_height_transport_cuda(f.x, &f.err)) {
    printf("    FAIL: launch returned: %s\n", f.err.message); return false;
  }
  f.sync();
  if (cudaGetLastError() != cudaSuccess) { printf("    FAIL: CUDA error after launch\n"); return false; }
  if (!f.pull({FieldId::zh, FieldId::crx, FieldId::area, FieldId::dp_ref})) {
    printf("    FAIL: D2H failed\n"); return false;
  }

  bool ok = check_finite(f.zh, "zh");
  // Read-only inputs must be bit-identical.
  if (!bitwise_equal(crx_in, f.crx)) { printf("    FAIL: crx mutated\n"); ok = false; }
  if (!bitwise_equal(area_in, f.area)) { printf("    FAIL: area mutated\n"); ok = false; }
  if (!bitwise_equal(dpr_in, f.dp_ref)) { printf("    FAIL: dp_ref mutated\n"); ok = false; }

  size_t changed_in = 0, changed_out = 0;
  for (int64_t k = 0; k <= f.s.npz; ++k)
    for (int64_t j = f.lo_j(); j <= f.s.je + f.h; ++j)
      for (int64_t i = f.lo_i(); i <= f.s.ie + f.h; ++i) {
        const size_t n = f.i3(i, j, k);
        const bool interior = i >= f.s.is && i <= f.s.ie && j >= f.s.js && j <= f.s.je;
        if (std::memcmp(&f.zh0[n], &f.zh[n], sizeof(fv_real)) != 0)
          (interior ? changed_in : changed_out)++;
      }
  const size_t expect = (size_t)(f.s.ie - f.s.is + 1) * (f.s.je - f.s.js + 1) * (f.s.npz + 1);
  printf("    zh changed: %zu interior (of %zu), %zu outside [is,ie]x[js,je]\n",
         changed_in, expect, changed_out);
  if (changed_in * 5 < expect * 4) { printf("    FAIL: transport degenerate\n"); ok = false; }
  if (changed_out != 0) { printf("    FAIL: wrote outside the compute box\n"); ok = false; }
  printf("  Part A: %s\n\n", ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------- Part B

static bool part_b() {
  printf("Part B: panel-edge PPM behavior (seam formulas vs. interior stencil)\n");
  // Same inputs, only the cubed-sphere seam gating differs. Corners are OFF in
  // both runs so any difference is attributable to xppm/yppm's one-sided
  // reconstruction alone, not to copy_corners.
  std::vector<fv_real> seam, interior_only;
  for (int pass = 0; pass < 2; ++pass) {
    Fixture f;
    f.s.cubed = pass == 0 ? 1 : 0;
    f.s.damp_mode = 1;
    if (!f.create(pass == 0 ? "B-seam" : "B-interior")) return false;
    const auto b = f.bounds();
    if (pass == 0 && !(b.west_edge && b.east_edge && b.south_edge && b.north_edge)) {
      printf("    FAIL: fixture is not on all four panel edges\n"); return false;
    }
    if (dgrid::launch_dgrid_height_transport_cuda(f.x, &f.err)) {
      printf("    FAIL: launch returned: %s\n", f.err.message); return false;
    }
    f.sync();
    if (!f.pull({FieldId::zh})) { printf("    FAIL: D2H failed\n"); return false; }
    (pass == 0 ? seam : interior_only) = f.zh;
  }

  Fixture g; g.fill();  // geometry only, for indexing
  bool ok = check_finite(seam, "zh (seam)") && check_finite(interior_only, "zh (interior)");
  size_t diff_edge = 0, diff_core = 0;
  for (int64_t k = 0; k <= g.s.npz; ++k)
    for (int64_t j = g.s.js; j <= g.s.je; ++j)
      for (int64_t i = g.s.is; i <= g.s.ie; ++i) {
        const size_t n = g.i3(i, j, k);
        const bool differs = std::memcmp(&seam[n], &interior_only[n], sizeof(fv_real)) != 0;
        // x overrides hit i <= 2 and i >= npx-1 == 12; y likewise in j.
        const bool near_seam = i <= 2 || i >= g.s.npx - 1 || j <= 2 || j >= g.s.npy - 1;
        if (differs) (near_seam ? diff_edge : diff_core)++;
      }
  printf("    zh differs at %zu near-seam points, %zu interior-core points\n",
         diff_edge, diff_core);
  if (diff_edge == 0) {
    printf("    FAIL: seam gating had no effect -- edge formulas not reached\n"); ok = false;
  }
  // The PPM stencil is 3 wide, so a seam override at i<=2 propagates to the
  // flux at i=3..5; requiring exact equality across the whole core would be
  // wrong. Assert instead that the deepest interior column is untouched.
  size_t deep_diff = 0;
  for (int64_t k = 0; k <= g.s.npz; ++k)
    for (int64_t j = 6; j <= 7; ++j)
      for (int64_t i = 6; i <= 7; ++i) {
        const size_t n = g.i3(i, j, k);
        if (std::memcmp(&seam[n], &interior_only[n], sizeof(fv_real)) != 0) ++deep_diff;
      }
  printf("    deepest-interior (i,j in [6,7]) differences: %zu (expect 0)\n", deep_diff);
  if (deep_diff != 0) {
    printf("    FAIL: seam gating perturbed points beyond the PPM stencil reach\n"); ok = false;
  }
  printf("  Part B: %s\n\n", ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------- Part C

static bool part_c() {
  printf("Part C: copy_corners rotation vs. host transcription (dir=1 and dir=2)\n");
  bool ok = true;
  for (int dir = 1; dir <= 2; ++dir) {
    Fixture f;
    f.s.sw = f.s.se = f.s.ne = f.s.nw = 1;
    f.s.damp_mode = 1;
    if (!f.create("C")) return false;
    // Encode the index in the value so a wrong rotation is unmistakable.
    for (int64_t k = 0; k <= f.s.npz; ++k)
      for (int64_t j = f.lo_j(); j <= f.s.je + f.h; ++j)
        for (int64_t i = f.lo_i(); i <= f.s.ie + f.h; ++i)
          f.zh[f.i3(i, j, k)] = (fv_real)(1000.0 * (i + 50) + (j + 50) + 0.25 * k);
    if (!f.push(FieldId::zh)) { printf("    FAIL: H2D failed\n"); return false; }
    const std::vector<fv_real> before = f.zh;

    dgrid::WpBHeightTransportViews vw; dgrid::WpBHeightTransportArena ar;
    if (dgrid::bind_wp_b_height_transport(f.x, vw, ar, &f.err)) {
      printf("    FAIL: bind failed: %s\n", f.err.message); return false;
    }
    dgrid::LevelFilter lf{};
    lf.damp = v1<const fv_real>(vw.damping_v);
    lf.ndif = v1<const fv_real>(vw.damping_ndif);
    lf.npz = f.s.npz; lf.mode = 0;
    const auto b = f.bounds();
    if (dgrid::launch_copy_corners_cuda(b, dir, lf, v3<fv_real>(vw.zh),
                                        f.x.submission.stream) != FV3_GPU_SUCCESS) {
      printf("    FAIL: launch_copy_corners_cuda failed\n"); return false;
    }
    f.sync();
    if (!f.pull({FieldId::zh})) { printf("    FAIL: D2H failed\n"); return false; }

    // Host transcription of tp_core.F90:257-318, reading pre-call values.
    std::vector<fv_real> ref = before;
    const int64_t ng = 3, npx = f.s.npx, npy = f.s.npy;
    auto src = [&](int64_t i, int64_t j, int64_t k) { return before[f.i3(i, j, k)]; };
    for (int64_t k = 0; k <= f.s.npz; ++k)
      for (int64_t t = 0; t < ng; ++t)
        for (int64_t s2 = 0; s2 < ng; ++s2) {
          if (dir == 1) {
            { int64_t i = 1 - ng + t, j = 1 - ng + s2; ref[f.i3(i,j,k)] = src(j, 1 - i, k); }
            { int64_t i = npx + t, j = 1 - ng + s2;    ref[f.i3(i,j,k)] = src(npy - j, i - npx + 1, k); }
            { int64_t i = npx + t, j = npy + s2;       ref[f.i3(i,j,k)] = src(j, 2 * npx - 1 - i, k); }
            { int64_t i = 1 - ng + t, j = npy + s2;    ref[f.i3(i,j,k)] = src(npy - j, i - 1 + npx, k); }
          } else {
            { int64_t i = 1 - ng + t, j = 1 - ng + s2; ref[f.i3(i,j,k)] = src(1 - j, i, k); }
            { int64_t i = npx + t, j = 1 - ng + s2;    ref[f.i3(i,j,k)] = src(npy + j - 1, npx - i, k); }
            { int64_t i = npx + t, j = npy + s2;       ref[f.i3(i,j,k)] = src(2 * npy - 1 - j, i, k); }
            { int64_t i = 1 - ng + t, j = npy + s2;    ref[f.i3(i,j,k)] = src(j + 1 - npx, npy - i, k); }
          }
        }

    size_t mismatch = 0, rotated = 0;
    for (size_t n = 0; n < f.zh.size(); ++n) {
      if (!close_rel((double)f.zh[n], (double)ref[n])) ++mismatch;
      if (std::memcmp(&ref[n], &before[n], sizeof(fv_real)) != 0) ++rotated;
    }
    const size_t expect_rot = (size_t)4 * ng * ng * (f.s.npz + 1);
    printf("    dir=%d: %zu cells rotated (expect %zu), %zu mismatches vs. host reference\n",
           dir, rotated, expect_rot, mismatch);
    if (rotated != expect_rot) { printf("    FAIL: wrong number of rotated cells\n"); ok = false; }
    if (mismatch) { printf("    FAIL: rotation disagrees with tp_core.F90 transcription\n"); ok = false; }
  }
  printf("  Part C: %s\n\n", ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------- Part D

static bool part_d() {
  printf("Part D: edge_profile tridiagonal vs. host transcription\n");
  Fixture f;
  f.s.damp_mode = 1;
  if (!f.create("D")) return false;
  dgrid::WpBHeightTransportViews vw; dgrid::WpBHeightTransportArena ar;
  if (dgrid::bind_wp_b_height_transport(f.x, vw, ar, &f.err)) {
    printf("    FAIL: bind failed: %s\n", f.err.message); return false;
  }
  const auto b = f.bounds();
  f.zero_arena(ar);  // this part pulls whole slots back -- see zero_arena
  auto gam = operators::View1<fv_real>{static_cast<fv_real *>(ar.arrays[A_GAM]), 0, 1};
  if (dgrid::launch_edge_profile_gam_cuda(b, v1<const fv_real>(vw.dp_ref), gam,
                                          f.x.submission.stream) != FV3_GPU_SUCCESS) {
    printf("    FAIL: gam launch failed\n"); return false;
  }
  if (dgrid::launch_edge_profile_x_cuda(b, v3<const fv_real>(vw.crx), v3<const fv_real>(vw.xfx),
                                        v1<const fv_real>(vw.dp_ref),
                                        operators::View1<const fv_real>{gam.data, 0, 1},
                                        av3(ar.arrays[A_CRX_ADV], f), av3(ar.arrays[A_XFX_ADV], f),
                                        f.x.submission.stream) != FV3_GPU_SUCCESS) {
    printf("    FAIL: edge_profile_x launch failed\n"); return false;
  }
  f.sync();
  const std::vector<fv_real> crx_adv = f.get_arena(ar, A_CRX_ADV);
  bool ok = check_finite(crx_adv, "crx_adv");

  // Independent host transcription of nh_utils.F90:1595-1651 for one column.
  const int64_t npz = f.s.npz;
  std::vector<double> hgam(npz, 0.0), qe(npz + 1, 0.0);
  {
    const double g0 = (double)f.dp_ref[1] / (double)f.dp_ref[0];
    double bet = g0 * (g0 + 0.5);
    hgam[0] = (1.0 + g0 * (g0 + 1.5)) / bet;
    for (int64_t k = 1; k < npz; ++k) {
      const double gk = (double)f.dp_ref[k - 1] / (double)f.dp_ref[k];
      bet = 2.0 + 2.0 * gk - hgam[k - 1];
      hgam[k] = gk / bet;
    }
  }
  size_t mismatch = 0, checked = 0;
  for (int64_t j = f.s.js - 3; j <= f.s.je + 3; j += 4)
    for (int64_t i = f.s.is; i <= f.s.ie + 1; i += 5) {
      const double g0 = (double)f.dp_ref[1] / (double)f.dp_ref[0];
      qe[0] = (2.0 * g0 * (g0 + 1.0) * (double)f.crx[f.iface(i, j, 0)] +
               (double)f.crx[f.iface(i, j, 1)]) / (g0 * (g0 + 0.5));
      double gk = 0.0;
      for (int64_t k = 1; k < npz; ++k) {
        gk = (double)f.dp_ref[k - 1] / (double)f.dp_ref[k];
        const double bet = 2.0 + 2.0 * gk - hgam[k - 1];
        qe[k] = (3.0 * ((double)f.crx[f.iface(i, j, k - 1)] +
                        gk * (double)f.crx[f.iface(i, j, k)]) - qe[k - 1]) / bet;
      }
      const double a_bot = 1.0 + gk * (gk + 1.5);
      const double xt1 = 2.0 * gk * (gk + 1.0);
      const double xt2 = gk * (gk + 0.5) - a_bot * hgam[npz - 1];
      qe[npz] = (xt1 * (double)f.crx[f.iface(i, j, npz - 1)] +
                 (double)f.crx[f.iface(i, j, npz - 2)] - a_bot * qe[npz - 1]) / xt2;
      for (int64_t k = npz - 1; k >= 0; --k) qe[k] -= hgam[k] * qe[k + 1];
      for (int64_t k = 0; k <= npz; ++k) {
        ++checked;
        if (!close_rel((double)crx_adv[f.ia(i, j, k)], qe[k], TOL * 100)) ++mismatch;
      }
    }
  printf("    crx_adv agrees with host tridiagonal at %zu/%zu sampled interfaces\n",
         checked - mismatch, checked);
  if (mismatch) { printf("    FAIL: edge_profile disagrees with nh_utils.F90\n"); ok = false; }
  printf("  Part D: %s\n\n", ok ? "PASS" : "FAIL");
  return ok;
}
// ------------------------------------------------------------- Parts E and F

static const fv_real SENTINEL = fv_real(-7.77e5);

// Runs the full transport with slots z2/fx2_d6/fy2_d6 pre-poisoned, then reports
// which levels of each slot were written. Used by E (no level damped) and F
// (all / alternating levels damped) to prove the per-level branch split.
static bool damping_branch_probe(int damp_mode, const char *label,
                                 std::vector<int> &z2_written,
                                 std::vector<int> &fx2_written, bool &zh_changed) {
  Fixture f;
  f.s.damp_mode = damp_mode;
  if (!f.create(label)) return false;
  dgrid::WpBHeightTransportViews vw; dgrid::WpBHeightTransportArena ar;
  if (dgrid::bind_wp_b_height_transport(f.x, vw, ar, &f.err)) {
    printf("    FAIL: bind failed: %s\n", f.err.message); return false;
  }
  // Workspace slices are fixed offsets into one allocation, so the launcher's
  // internal bind hands back these same pointers.
  const std::vector<fv_real> poison(f.narena, SENTINEL);
  f.put_arena(ar, A_Z2, poison);
  f.put_arena(ar, A_FX2_D6, poison);
  f.put_arena(ar, A_FY2_D6, poison);
  if (dgrid::launch_dgrid_height_transport_cuda(f.x, &f.err)) {
    printf("    FAIL: launch returned: %s\n", f.err.message); return false;
  }
  f.sync();
  if (cudaGetLastError() != cudaSuccess) { printf("    FAIL: CUDA error\n"); return false; }
  if (!f.pull({FieldId::zh})) { printf("    FAIL: D2H failed\n"); return false; }
  const std::vector<fv_real> z2 = f.get_arena(ar, A_Z2), fx2 = f.get_arena(ar, A_FX2_D6);

  z2_written.assign(f.s.npz + 1, 0);
  fx2_written.assign(f.s.npz + 1, 0);
  for (int64_t k = 0; k <= f.s.npz; ++k)
    for (int64_t j = f.lo_j(); j <= f.s.je + f.h; ++j)
      for (int64_t i = f.lo_i(); i <= f.s.ie + f.h; ++i) {
        const size_t n = f.ia(i, j, k);
        if (std::memcmp(&z2[n], &SENTINEL, sizeof(fv_real)) != 0) z2_written[k] = 1;
        if (std::memcmp(&fx2[n], &SENTINEL, sizeof(fv_real)) != 0) fx2_written[k] = 1;
      }
  zh_changed = false;
  for (size_t n = 0; n < f.zh.size(); ++n)
    if (std::memcmp(&f.zh0[n], &f.zh[n], sizeof(fv_real)) != 0) { zh_changed = true; break; }
  return check_finite(f.zh, "zh") && check_finite(z2, "z2");
}

static bool part_e() {
  printf("Part E: undamped branch (damp <= 1.E-5 on every level)\n");
  std::vector<int> z2w, fxw; bool changed = false;
  if (!damping_branch_probe(0, "E", z2w, fxw, changed)) return false;
  bool ok = true;
  for (size_t k = 0; k < z2w.size(); ++k) {
    if (z2w[k]) { printf("    FAIL: z2 written on undamped level %zu\n", k); ok = false; }
    if (fxw[k]) { printf("    FAIL: fx2 written on undamped level %zu\n", k); ok = false; }
  }
  if (!changed) { printf("    FAIL: zh unchanged -- undamped transport degenerate\n"); ok = false; }
  printf("    no damped-only workspace touched; zh transported in place: %s\n",
         ok ? "yes" : "no");
  printf("  Part E: %s\n\n", ok ? "PASS" : "FAIL");
  return ok;
}

static bool part_f() {
  printf("Part F: damped branch (damp > 1.E-5), uniform then alternating profile\n");
  bool ok = true;
  {
    std::vector<int> z2w, fxw; bool changed = false;
    if (!damping_branch_probe(1, "F-all", z2w, fxw, changed)) return false;
    size_t nz = 0, nf = 0;
    for (size_t k = 0; k < z2w.size(); ++k) { nz += z2w[k]; nf += fxw[k]; }
    printf("    all-damped: z2 written on %zu/%zu levels, fx2 on %zu/%zu\n",
           nz, z2w.size(), nf, fxw.size());
    if (nz != z2w.size() || nf != fxw.size()) {
      printf("    FAIL: damped branch skipped a level\n"); ok = false;
    }
    if (!changed) { printf("    FAIL: zh unchanged\n"); ok = false; }
  }
  {
    // damp nonzero on even k only. Level npz reads damp at min(k,npz-1) == npz-1
    // (odd for npz == 8), reproducing Fortran's damp(km+1) = damp(km).
    std::vector<int> z2w, fxw; bool changed = false;
    if (!damping_branch_probe(2, "F-mixed", z2w, fxw, changed)) return false;
    for (size_t k = 0; k < z2w.size(); ++k) {
      const size_t kk = k < z2w.size() - 1 ? k : z2w.size() - 2;
      const int expect = (kk % 2 == 0) ? 1 : 0;
      if (z2w[k] != expect || fxw[k] != expect) {
        printf("    FAIL: level %zu: z2=%d fx2=%d, expected %d\n", k, z2w[k], fxw[k], expect);
        ok = false;
      }
    }
    printf("    alternating profile: per-level branch selection as expected inside one launch\n");
    if (!changed) { printf("    FAIL: zh unchanged\n"); ok = false; }
  }
  printf("  Part F: %s\n\n", ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------- Part G

static bool part_g() {
  printf("Part G: del6_vt_flux -- raw-damp linearity and nord=0 host reference\n");
  Fixture f;
  f.s.damp_mode = 1;
  f.s.ndif_value = fv_real(0);   // del-2: no corner rotation, no iteration
  if (!f.create("G")) return false;
  dgrid::WpBHeightTransportViews vw; dgrid::WpBHeightTransportArena ar;
  if (dgrid::bind_wp_b_height_transport(f.x, vw, ar, &f.err)) {
    printf("    FAIL: bind failed: %s\n", f.err.message); return false;
  }
  const auto b = f.bounds();
  dgrid::LevelFilter lf{};
  lf.damp = v1<const fv_real>(vw.damping_v);
  lf.ndif = v1<const fv_real>(vw.damping_ndif);
  lf.npz = f.s.npz; lf.mode = 1;

  auto run = [&]() -> bool {
    return dgrid::launch_del6_vt_flux_cuda(b, 2, lf, v3<fv_real>(vw.zh),
                                           v2<const fv_real>(vw.del6_u),
                                           v2<const fv_real>(vw.del6_v),
                                           v2<const fv_real>(vw.rarea),
                                           av3(ar.arrays[A_D2], f),
                                           av3(ar.arrays[A_FX2_D6], f),
                                           av3(ar.arrays[A_FY2_D6], f),
                                           f.x.submission.stream) == FV3_GPU_SUCCESS;
  };
  if (!run()) { printf("    FAIL: del6 launch (damp = c) failed\n"); return false; }
  f.sync();
  const std::vector<fv_real> fx2_c = f.get_arena(ar, A_FX2_D6);
  const std::vector<fv_real> fy2_c = f.get_arena(ar, A_FY2_D6);

  bool ok = check_finite(fx2_c, "fx2") && check_finite(fy2_c, "fy2");

  // G1: exact linearity in damp. del6_vt_flux uses the raw damp(k); tp_core's
  // deln_flux uses (damp_c*da_min)**(nord+1), which is NOT linear -- so this
  // check is what separates the two Fortran routines.
  for (int64_t k = 0; k < f.s.npz; ++k) f.damp_v[k] = f.s.damp_value * fv_real(2);
  if (!f.push(FieldId::damping_v)) { printf("    FAIL: damping_v re-upload failed\n"); return false; }
  if (!run()) { printf("    FAIL: del6 launch (damp = 2c) failed\n"); return false; }
  f.sync();
  const std::vector<fv_real> fx2_2c = f.get_arena(ar, A_FX2_D6);
  size_t nonlinear = 0, nonzero = 0;
  for (size_t n = 0; n < fx2_c.size(); ++n) {
    if (std::fabs((double)fx2_c[n]) > 0) ++nonzero;
    if (!close_rel(2.0 * (double)fx2_c[n], (double)fx2_2c[n])) ++nonlinear;
  }
  printf("    G1 linearity in damp: %zu deviations over %zu nonzero flux cells\n",
         nonlinear, nonzero);
  if (nonzero == 0) { printf("    FAIL: fx2 identically zero -- degenerate\n"); ok = false; }
  if (nonlinear) { printf("    FAIL: fx2 is not linear in damp (deln_flux coefficient?)\n"); ok = false; }

  // G2: nord == 0 host reference, fx2 = del6_v*(d2(i-1,j)-d2(i,j)), d2 = damp*q
  // over j=js..je, i=is..ie+1 (sw_core.F90:1589-1601).
  for (int64_t k = 0; k < f.s.npz; ++k) f.damp_v[k] = f.s.damp_value;
  if (!f.push(FieldId::damping_v)) { printf("    FAIL: damping_v restore failed\n"); return false; }
  if (!run()) { printf("    FAIL: del6 relaunch failed\n"); return false; }
  f.sync();
  const std::vector<fv_real> fx2_r = f.get_arena(ar, A_FX2_D6);
  const std::vector<fv_real> fy2_r = f.get_arena(ar, A_FY2_D6);
  const double dv = (double)f.s.damp_value;
  size_t mismatch = 0, checked = 0;
  for (int64_t k = 0; k <= f.s.npz; ++k)
    for (int64_t j = f.s.js; j <= f.s.je; ++j)
      for (int64_t i = f.s.is; i <= f.s.ie + 1; ++i) {
        const double ref = (double)f.del6_v[f.i2(i, j)] *
                           (dv * (double)f.zh0[f.i3(i - 1, j, k)] -
                            dv * (double)f.zh0[f.i3(i, j, k)]);
        ++checked;
        if (!close_rel((double)fx2_r[f.ia(i, j, k)], ref, TOL * 10)) ++mismatch;
      }
  for (int64_t k = 0; k <= f.s.npz; ++k)
    for (int64_t j = f.s.js; j <= f.s.je + 1; ++j)
      for (int64_t i = f.s.is; i <= f.s.ie; ++i) {
        const double ref = (double)f.del6_u[f.i2(i, j)] *
                           (dv * (double)f.zh0[f.i3(i, j - 1, k)] -
                            dv * (double)f.zh0[f.i3(i, j, k)]);
        ++checked;
        if (!close_rel((double)fy2_r[f.ia(i, j, k)], ref, TOL * 10)) ++mismatch;
      }
  printf("    G2 nord=0 host reference: %zu/%zu flux faces agree\n", checked - mismatch, checked);
  if (mismatch) { printf("    FAIL: del6 del-2 fluxes disagree with sw_core.F90\n"); ok = false; }

  // zh is del6_vt_flux's `q` (intent(inout) in Fortran only because of the
  // corner rotations, which nord=0 skips): it must come back untouched.
  if (!f.pull({FieldId::zh})) { printf("    FAIL: D2H failed\n"); return false; }
  if (!bitwise_equal(f.zh0, f.zh)) { printf("    FAIL: del6 mutated q at nord=0\n"); ok = false; }

  // G3: del-6 must differ from del-2 -- otherwise the nord grouping is inert.
  {
    Fixture g6;
    g6.s.damp_mode = 1; g6.s.ndif_value = fv_real(2);
    if (!g6.create("G-nord2")) return false;
    dgrid::WpBHeightTransportViews w6; dgrid::WpBHeightTransportArena a6;
    if (dgrid::bind_wp_b_height_transport(g6.x, w6, a6, &g6.err)) {
      printf("    FAIL: bind (nord=2) failed: %s\n", g6.err.message); return false;
    }
    dgrid::LevelFilter l6{};
    l6.damp = v1<const fv_real>(w6.damping_v);
    l6.ndif = v1<const fv_real>(w6.damping_ndif);
    l6.npz = g6.s.npz; l6.mode = 1;
    if (dgrid::launch_del6_vt_flux_cuda(g6.bounds(), 2, l6, v3<fv_real>(w6.zh),
                                        v2<const fv_real>(w6.del6_u),
                                        v2<const fv_real>(w6.del6_v),
                                        v2<const fv_real>(w6.rarea),
                                        av3(a6.arrays[A_D2], g6),
                                        av3(a6.arrays[A_FX2_D6], g6),
                                        av3(a6.arrays[A_FY2_D6], g6),
                                        g6.x.submission.stream) != FV3_GPU_SUCCESS) {
      printf("    FAIL: del6 launch (nord=2) failed\n"); return false;
    }
    g6.sync();
    const std::vector<fv_real> fx2_n2 = g6.get_arena(a6, A_FX2_D6);
    if (!check_finite(fx2_n2, "fx2 (nord=2)")) ok = false;
    size_t differ = 0;
    for (int64_t k = 0; k <= f.s.npz; ++k)
      for (int64_t j = f.s.js; j <= f.s.je; ++j)
        for (int64_t i = f.s.is; i <= f.s.ie + 1; ++i)
          if (!close_rel((double)fx2_r[f.ia(i, j, k)], (double)fx2_n2[f.ia(i, j, k)]))
            ++differ;
    printf("    G3 nord=2 vs nord=0: %zu differing flux faces\n", differ);
    if (differ == 0) { printf("    FAIL: nord grouping inert\n"); ok = false; }
  }
  printf("  Part G: %s\n\n", ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------- Part H

static bool part_h() {
  printf("Part H: authoritative area-weighted zh update vs. host reference\n");
  Fixture f;
  f.s.damp_mode = 2;   // both branches fire inside one launch
  if (!f.create("H")) return false;
  dgrid::WpBHeightTransportViews vw; dgrid::WpBHeightTransportArena ar;
  if (dgrid::bind_wp_b_height_transport(f.x, vw, ar, &f.err)) {
    printf("    FAIL: bind failed: %s\n", f.err.message); return false;
  }

  std::vector<fv_real> z2(f.narena), fx(f.narena), fy(f.narena),
      fx2(f.narena), fy2(f.narena), rax(f.narena), ray(f.narena);
  for (int64_t k = 0; k <= f.s.npz; ++k)
    for (int64_t j = f.lo_j(); j <= f.s.je + f.h; ++j)
      for (int64_t i = f.lo_i(); i <= f.s.ie + f.h; ++i) {
        const size_t n = f.ia(i, j, k);
        const double a = (double)f.area[f.i2(i, j)];
        z2[n] = (fv_real)(8000.0 + 30.0 * std::sin(0.31 * i) * std::cos(0.27 * j) - 25.0 * k);
        fx[n] = (fv_real)(1.0e10 * (1.0 + 0.11 * std::sin(0.3 * i) + 0.05 * std::cos(0.2 * j) + 0.02 * k));
        fy[n] = (fv_real)(0.9e10 * (1.0 + 0.09 * std::cos(0.26 * i) + 0.07 * std::sin(0.23 * j) + 0.03 * k));
        fx2[n] = (fv_real)(1.0e9 * (1.0 + 0.13 * std::sin(0.19 * i + 0.11 * j) + 0.02 * k));
        fy2[n] = (fv_real)(0.8e9 * (1.0 + 0.15 * std::cos(0.17 * i - 0.13 * j) + 0.04 * k));
        // Keep (ra_x+ra_y)-area comfortably away from zero.
        rax[n] = (fv_real)(a * (1.0 + 0.03 * std::sin(0.21 * i)));
        ray[n] = (fv_real)(a * (1.0 + 0.04 * std::cos(0.18 * j)));
      }
  f.put_arena(ar, A_Z2, z2); f.put_arena(ar, A_FX, fx); f.put_arena(ar, A_FY, fy);
  f.put_arena(ar, A_FX2_D6, fx2); f.put_arena(ar, A_FY2_D6, fy2);
  f.put_arena(ar, A_RA_X, rax); f.put_arena(ar, A_RA_Y, ray);

  dgrid::LevelFilter lf{};
  lf.damp = v1<const fv_real>(vw.damping_v);
  lf.ndif = v1<const fv_real>(vw.damping_ndif);
  lf.npz = f.s.npz; lf.mode = 0;
  auto cav3 = [&](void *p) {
    return operators::View3<const fv_real>{static_cast<const fv_real *>(p), f.lo_i(),
                                           f.lo_j(), 0, 1, f.ni, f.ni * f.nj};
  };
  if (dgrid::launch_zh_update_cuda(f.bounds(), lf, cav3(ar.arrays[A_Z2]),
                                   cav3(ar.arrays[A_FX]), cav3(ar.arrays[A_FY]),
                                   cav3(ar.arrays[A_FX2_D6]), cav3(ar.arrays[A_FY2_D6]),
                                   cav3(ar.arrays[A_RA_X]), cav3(ar.arrays[A_RA_Y]),
                                   v2<const fv_real>(vw.area), v2<const fv_real>(vw.rarea),
                                   v3<fv_real>(vw.zh),
                                   f.x.submission.stream) != FV3_GPU_SUCCESS) {
    printf("    FAIL: zh_update launch failed\n"); return false;
  }
  f.sync();
  if (!f.pull({FieldId::zh})) { printf("    FAIL: D2H failed\n"); return false; }
  bool ok = check_finite(f.zh, "zh");

  size_t mismatch = 0, checked = 0, damped_seen = 0, undamped_seen = 0;
  for (int64_t k = 0; k <= f.s.npz; ++k) {
    const int64_t kk = k < f.s.npz ? k : f.s.npz - 1;
    const bool damped = (double)f.damp_v[kk] > 1.0e-5;
    (damped ? damped_seen : undamped_seen)++;
    for (int64_t j = f.s.js; j <= f.s.je; ++j)
      for (int64_t i = f.s.is; i <= f.s.ie; ++i) {
        const double a = (double)f.area[f.i2(i, j)];
        const double denom = ((double)rax[f.ia(i, j, k)] + (double)ray[f.ia(i, j, k)]) - a;
        const double fxd = (double)fx[f.ia(i, j, k)] - (double)fx[f.ia(i + 1, j, k)];
        const double fyd = (double)fy[f.ia(i, j, k)] - (double)fy[f.ia(i, j + 1, k)];
        double ref;
        if (damped) {
          const double diff = ((double)fx2[f.ia(i, j, k)] - (double)fx2[f.ia(i + 1, j, k)] +
                               ((double)fy2[f.ia(i, j, k)] - (double)fy2[f.ia(i, j + 1, k)])) *
                              (double)f.rarea[f.i2(i, j)];
          ref = ((double)z2[f.ia(i, j, k)] * a + fxd + fyd) / denom + diff;
        } else {
          ref = ((double)f.zh0[f.i3(i, j, k)] * a + fxd + fyd) / denom;
        }
        ++checked;
        if (!close_rel((double)f.zh[f.i3(i, j, k)], ref, TOL * 10)) ++mismatch;
      }
  }
  printf("    %zu/%zu updated cells agree (%zu damped levels, %zu undamped)\n",
         checked - mismatch, checked, damped_seen, undamped_seen);
  if (damped_seen == 0 || undamped_seen == 0) {
    printf("    FAIL: only one branch exercised\n"); ok = false;
  }
  if (mismatch) {
    printf("    FAIL: zh update disagrees with nh_utils.F90:288-303\n"); ok = false;
  }
  // Halo must be untouched: the update writes is..ie x js..je only.
  size_t halo_touched = 0;
  for (int64_t k = 0; k <= f.s.npz; ++k)
    for (int64_t j = f.lo_j(); j <= f.s.je + f.h; ++j)
      for (int64_t i = f.lo_i(); i <= f.s.ie + f.h; ++i) {
        if (i >= f.s.is && i <= f.s.ie && j >= f.s.js && j <= f.s.je) continue;
        const size_t n = f.i3(i, j, k);
        if (std::memcmp(&f.zh0[n], &f.zh[n], sizeof(fv_real)) != 0) ++halo_touched;
      }
  if (halo_touched) { printf("    FAIL: %zu halo cells written\n", halo_touched); ok = false; }
  printf("  Part H: %s\n\n", ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------- Part I

static bool part_i() {
  printf("Part I: cross-WP -- WP-C update_dz_d consumes the corrected WP-B service\n");
  Fixture f;
  f.s.damp_mode = 1;
  if (!f.create("I")) return false;
  NumericalDispatch d{};
  d.dgrid_height_transport = &dgrid::launch_dgrid_height_transport_cuda;
  f.c->dispatch_params.rdt = 1.0 / 300.0;
  if (mark_halo_ready(f.ctx, FV3_GPU_H_DSCALAR, &f.err)) {
    printf("    FAIL: mark_halo_ready failed: %s\n", f.err.message); return false;
  }
  const std::vector<fv_real> delz0 = f.delz;
  if (vertical::launch_update_dz_d(f.x, d, &f.err)) {
    printf("    FAIL: launch_update_dz_d returned: %s\n", f.err.message); return false;
  }
  f.sync();
  if (cudaGetLastError() != cudaSuccess) { printf("    FAIL: CUDA error\n"); return false; }
  if (!f.pull({FieldId::zh, FieldId::ws, FieldId::delz})) {
    printf("    FAIL: D2H failed\n"); return false;
  }
  bool ok = check_finite(f.zh, "zh") && check_finite(f.ws, "ws");
  size_t changed = 0;
  for (size_t n = 0; n < f.zh.size(); ++n)
    if (std::memcmp(&f.zh0[n], &f.zh[n], sizeof(fv_real)) != 0) ++changed;
  size_t ws_written = 0;
  for (int64_t j = f.s.js; j <= f.s.je; ++j)
    for (int64_t i = f.s.is; i <= f.s.ie; ++i)
      if (std::fabs((double)f.ws[f.i2(i, j)]) > 0) ++ws_written;
  printf("    zh changed at %zu cells; ws written at %zu of %lld compute cells\n", changed,
         ws_written, (long long)((f.s.ie - f.s.is + 1) * (f.s.je - f.s.js + 1)));
  if (!changed) { printf("    FAIL: WP-C did not reach WP-B transport\n"); ok = false; }
  if (!ws_written) { printf("    FAIL: ws never written\n"); ok = false; }
  // Section 9: update_dz_d's Fortran counterpart declares delz intent(out) but
  // never assigns it -- the Riemann solver derives delz from the final zh. So
  // launch_update_dz_d passes write_delz = false and delz must keep its
  // pre-call sentinel. Do not relax this to make a test convenient.
  if (!bitwise_equal(delz0, f.delz)) {
    printf("    FAIL: delz was written; write_delz=false is not being honored\n"); ok = false;
  } else {
    printf("    delz retains its pre-call sentinel (write_delz = false honored)\n");
  }
  // Monotonicity closure (nh_utils.F90:319) must hold after WP-C's boundary pass.
  const double dz_min = 2.0;   // matches NumericalConfig::dz_min set in create()
  size_t violations = 0;
  for (int64_t j = f.s.js; j <= f.s.je; ++j)
    for (int64_t i = f.s.is; i <= f.s.ie; ++i)
      for (int64_t k = 0; k < f.s.npz; ++k)
        if ((double)f.zh[f.i3(i, j, k)] <
            (double)f.zh[f.i3(i, j, k + 1)] + dz_min - 1e-3)
          ++violations;
  printf("    height monotonicity (zh(k) >= zh(k+1) + dz_min): %zu violations\n", violations);
  if (violations) { printf("    FAIL: monotonicity closure not applied\n"); ok = false; }
  printf("  Part I: %s\n\n", ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------- Part J

// Storage-contract and precondition rejection. These are contract tests, not
// numerical ones: each case must fail BEFORE any kernel launch, with a specific
// status, rather than reinterpreting memory or silently applying interior
// stencils.
static bool expect_reject(Spec s, const char *label, int expect_status) {
  Fixture f; f.s = s;
  if (!f.create(label)) { printf("    FAIL: fixture setup failed for %s\n", label); return false; }
  cudaGetLastError();
  const int rc = dgrid::launch_dgrid_height_transport_cuda(f.x, &f.err);
  const cudaError_t ce = cudaGetLastError();
  if (rc != expect_status) {
    printf("    FAIL [%s]: expected status %d, got %d (%s)\n", label, expect_status, rc,
           rc ? f.err.message : "success");
    return false;
  }
  if (ce != cudaSuccess) {
    printf("    FAIL [%s]: a kernel was launched before the rejection\n", label); return false;
  }
  printf("    %s rejected with status %d: %s\n", label, rc, f.err.message);
  return true;
}

static bool part_j() {
  printf("Part J: contract rejection (damping_ndif storage width, npx/npy, corner flags)\n");
  bool ok = true;

  // J1: the native width must be ACCEPTED (no false rejection).
  {
    Fixture f; f.s.damp_mode = 1;
    if (!f.create("J1")) return false;
    if (dgrid::launch_dgrid_height_transport_cuda(f.x, &f.err)) {
      printf("    FAIL: native damping_ndif width (%zu bytes) was rejected: %s\n",
             sizeof(fv_real), f.err.message);
      ok = false;
    } else {
      f.sync();
      printf("    J1 native %zu-byte damping_ndif accepted\n", sizeof(fv_real));
    }
  }

  // J2: the OTHER width must be rejected. It passes fv3_gpu_validate_field
  // (element_type and element_bytes agree) and make_device_view (whose
  // precision check is a tautology for LogicalType::integer), so the binder's
  // element_bytes guard is the only thing standing between a mismatched
  // registration and a View1<const fv_real> over foreign-width storage.
  {
    Spec s; s.damp_mode = 1;
    s.ndif_bytes = sizeof(fv_real) == 4 ? 8u : 4u;
    if (!expect_reject(s, "J2 mismatched damping_ndif width", FV3_GPU_INVALID_ARGUMENT))
      ok = false;
  }

  // J3: absent npx/npy would silently apply interior PPM stencils at a seam.
  { Spec s; s.damp_mode = 1; s.skip_npxy = true;
    if (!expect_reject(s, "J3 absent npx/npy", FV3_GPU_INVALID_ARGUMENT)) ok = false; }

  // J4: a panel corner asserted on a non-cubed-sphere configuration is
  // self-contradictory (fv_grid_utils.F90:288-298 only sets corner flags when
  // grid_type < 3 .and. .not. nested).
  { Spec s; s.damp_mode = 1; s.cubed = 0; s.sw = 1;
    if (!expect_reject(s, "J4 corner flag without cubed sphere", FV3_GPU_INVALID_ARGUMENT))
      ok = false; }

  // J5: nord_v outside [0,2] cannot be produced by dyn_core.F90:713 and would
  // reach past the three rotated corner rings.
  { Spec s; s.damp_mode = 1; s.nord_v = 3;
    if (!expect_reject(s, "J5 nord_v = 3", FV3_GPU_UNSUPPORTED_CONFIGURATION)) ok = false; }

  printf("  Part J: %s\n\n", ok ? "PASS" : "FAIL");
  return ok;
}

// -------------------------------------------------------------------- driver

int main() {
  printf("WP-B CUDA height-transport behavior test (%s)\n",
         sizeof(fv_real) == 4 ? "R4" : "R8");
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    printf("no CUDA device available -- this test requires a GPU (SKIP)\n");
    return 77;   // matches SKIP_RETURN_CODE in cuda/dyn_core/CMakeLists.txt
  }
  struct { const char *name; bool (*fn)(); } parts[] = {
      {"A", part_a}, {"B", part_b}, {"C", part_c}, {"D", part_d}, {"E", part_e},
      {"F", part_f}, {"G", part_g}, {"H", part_h}, {"I", part_i}, {"J", part_j}};
  int failed = 0;
  std::string failed_names;
  for (const auto &p : parts) {
    if (!p.fn()) { ++failed; failed_names += failed_names.empty() ? "" : ", "; failed_names += p.name; }
  }
  if (failed) {
    printf("RESULT: FAIL (%d of %zu parts failed: %s)\n", failed,
           sizeof(parts) / sizeof(parts[0]), failed_names.c_str());
    return 1;
  }
  printf("RESULT: PASS (all %zu parts)\n", sizeof(parts) / sizeof(parts[0]));
  return 0;
}
