#pragma once

#include "fv3_cgrid_view.hpp"

namespace fv3::gpu::cgrid {

// Interpolation constants from authoritative sw_core.F90
inline constexpr fv_real a1 = fv_real(0.5625);
inline constexpr fv_real a2 = fv_real(-0.0625);
inline constexpr fv_real c1 = fv_real(-2.0 / 14.0);
inline constexpr fv_real c2 = fv_real(11.0 / 14.0);
inline constexpr fv_real c3 = fv_real(5.0 / 14.0);

// Edge interpolation formula from sw_core.F90:3142-3153 (edge_interpolate4)
// ua: 4 consecutive values, dxa: 4 consecutive grid spacings
__device__ __forceinline__ fv_real edge_interpolate4(
    fv_real ua0, fv_real ua1, fv_real ua2, fv_real ua3,
    fv_real dxa0, fv_real dxa1, fv_real dxa2, fv_real dxa3) {
  const fv_real t1 = dxa0 + dxa1;
  const fv_real t2 = dxa2 + dxa3;
  return fv_real(0.5) * (((t1 + dxa1) * ua1 - dxa1 * ua0) / t1 +
                         ((t2 + dxa2) * ua2 - dxa2 * ua3) / t2);
}

struct D2A2CArgs {
  Bounds bounds{};

  // Halo width, i.e. the Fortran `ng` used to form bd%isd/ied/jsd/jed:
  //   isd = is-ng, ied = ie+ng, jsd = js-ng, jed = je+ng.
  // REPAIR (D1): d2a2c_vect operates on the FULL halo box, not on is-1:ie+1.
  // The utmp/vtmp workspace is declared (isd:ied,jsd:jed) at sw_core.F90:2813,
  // the 4th-order fill loops at :2893-2902 run to isd/ied/jsd/jed, and the
  // downstream A->C stencils read utmp(i-2,j) at i=is-1 (=> is-3) and
  // vtmp(i,j-2) at j=js-1 (=> js-3). This member replaces the previous
  // hardcoded `isd = is-1` and MUST be populated by the caller from
  // DomainBounds::halo_width (fv3_gpu_internal.hpp:131). It intentionally has
  // no usable default: launch_d2a2c_vect_level rejects ng < 3 rather than
  // silently substituting a workable value.
  // Numerical semantics changed: NO.
  int32_t ng{};

  int32_t npx{}, npy{};
  uint8_t grid_type{};
  uint8_t nested{};
  uint8_t dord4{};
  uint8_t sw_corner{}, se_corner{}, ne_corner{}, nw_corner{};

  // bd%isd/ied/jsd/jed (sw_core.F90:2827-2830).
#if defined(__CUDACC__)
  __host__ __device__
#endif
  constexpr int64_t isd() const { return bounds.is - static_cast<int64_t>(ng); }
#if defined(__CUDACC__)
  __host__ __device__
#endif
  constexpr int64_t ied() const { return bounds.ie + static_cast<int64_t>(ng); }
#if defined(__CUDACC__)
  __host__ __device__
#endif
  constexpr int64_t jsd() const { return bounds.js - static_cast<int64_t>(ng); }
#if defined(__CUDACC__)
  __host__ __device__
#endif
  constexpr int64_t jed() const { return bounds.je + static_cast<int64_t>(ng); }

  // Input D-grid winds
  FieldView<const fv_real, 3> u;   // (isd:ied, jsd:jed+1, k)
  FieldView<const fv_real, 3> v;   // (isd:ied+1, jsd:jed, k)

  // Output A-grid winds
  FieldView<fv_real, 3> ua;        // (isd:ied, jsd:jed, k)
  FieldView<fv_real, 3> va;        // (isd:ied, jsd:jed, k)

  // Output C-grid winds
  FieldView<fv_real, 3> uc;        // (isd:ied+1, jsd:jed, k)
  FieldView<fv_real, 3> vc;        // (isd:ied, jsd:jed+1, k)

  // Output transport velocities
  FieldView<fv_real, 3> ut;        // (isd:ied, jsd:jed, k)
  FieldView<fv_real, 3> vt;        // (isd:ied, jsd:jed, k)

  // C-grid geometry for transport velocity
  FieldView<const fv_real, 2> cosa_u;  // D-u stagger
  FieldView<const fv_real, 2> cosa_v;  // D-v stagger
  FieldView<const fv_real, 2> rsin_u;  // D-u stagger
  FieldView<const fv_real, 2> rsin_v;  // D-v stagger

  // A-grid geometry for D-to-A interpolation
  FieldView<const fv_real, 2> cosa_s;  // A stagger
  FieldView<const fv_real, 2> rsin2;   // A stagger

  // Edge interpolation geometry
  FieldView<const fv_real, 2> dxa;     // A stagger
  FieldView<const fv_real, 2> dya;     // A stagger

  // sin_sg for edge flux direction
  FieldView<const fv_real, 3> sin_sg;  // B stagger (i,j,component)

  // Workspace for utmp/vtmp (from C-grid workspace). Must cover the full
  // declared Fortran extent (isd:ied, jsd:jed) -- sw_core.F90:2813 -- because
  // :2855-2856 prefills all of it with big_number and the corner reflections
  // at :2960-2979 / :3053-3072 read and write its outermost halo rings.
  FieldView<fv_real, 3> utmp;
  FieldView<fv_real, 3> vtmp;
};

// Launch d2a2c_vect for one vertical level. Returns status.
//
// Returns FV3_GPU_UNSUPPORTED_CONFIGURATION for:
//   - ng < 3       (the stencils reach 3 cells into the halo; see D2A2CArgs::ng)
//   - nested != 0  (the nested branch, sw_core.F90:2858-2887, is a distinct
//                   algorithm and is NOT implemented here -- it is rejected
//                   rather than silently mis-handled)
fv3_gpu_status launch_d2a2c_vect_level(const D2A2CArgs &args, int64_t k,
                                       void *stream);

}  // namespace fv3::gpu::cgrid
