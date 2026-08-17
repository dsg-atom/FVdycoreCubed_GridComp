#pragma once

#include "fv3_cgrid_view.hpp"

// Include all C-grid component headers
#include "fv3_cgrid_d2a2c.hpp"
#include "fv3_cgrid_csw.hpp"
#include "fv3_cgrid_divergence.hpp"
#include "fv3_cgrid_dz.hpp"
#include "fv3_cgrid_predictor.hpp"

namespace fv3::gpu::cgrid {

// Exact normal-nonhydrostatic p_grad_c inputs. All views are device views
// obtained from WP-E DeviceField descriptors. stream is the WP-E-owned compute
// stream; this package neither creates nor synchronizes it.
struct PressureGradientCArgs {
  Bounds bounds{};
  fv_real dt2{};
  FieldView<const fv_real, 3> delpc;
  FieldView<const fv_real, 3> pkc;
  FieldView<const fv_real, 3> gz;
  FieldView<fv_real, 3> uc;
  FieldView<fv_real, 3> vc;
  FieldView<const fv_real, 2> rdxc;
  FieldView<const fv_real, 2> rdyc;
};

// Enqueues p_grad_c on stream. It performs no allocation, transfer, MPI call,
// or stream synchronization. The caller owns dependency/event recording.
fv3_gpu_status launch_pressure_gradient_c(const PressureGradientCArgs &args,
                                          void *stream);

// WP-E v3 NumericalDispatch adapter. Integration registers this function in
// NumericalDispatch::cgrid_pressure.
int launch_cgrid_pressure_v3(const ExecutionContext &context,
                             fv3_gpu_error *error);

} // namespace fv3::gpu::cgrid
