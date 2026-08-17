#pragma once

#include "fv3_cgrid_view.hpp"

namespace fv3::gpu::cgrid {

// Interpolation constants from authoritative sw_core.F90
inline constexpr fv_real a1 = fv_real(0.5625);
inline constexpr fv_real a2 = fv_real(-0.0625);
inline constexpr fv_real c1 = fv_real(-2.0 / 14.0);
inline constexpr fv_real c2 = fv_real(11.0 / 14.0);
inline constexpr fv_real c3 = fv_real(5.0 / 14.0);

// Edge interpolation formula from sw_core.F90:3174-3185
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
  int32_t npx{}, npy{};
  uint8_t grid_type{};
  uint8_t nested{};
  uint8_t dord4{};
  uint8_t sw_corner{}, se_corner{}, ne_corner{}, nw_corner{};

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

  // Workspace for utmp/vtmp (from C-grid workspace)
  FieldView<fv_real, 3> utmp;
  FieldView<fv_real, 3> vtmp;
};

// Launch d2a2c_vect for one vertical level. Returns status.
fv3_gpu_status launch_d2a2c_vect_level(const D2A2CArgs &args, int64_t k,
                                       void *stream);

// Full 3D dispatch adapter for integration
int launch_d2a2c_vect_v3(const ExecutionContext &context, fv3_gpu_error *error);

}  // namespace fv3::gpu::cgrid
