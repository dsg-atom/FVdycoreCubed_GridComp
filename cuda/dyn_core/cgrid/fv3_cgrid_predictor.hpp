#pragma once

#include "fv3_cgrid_view.hpp"

namespace fv3::gpu::cgrid {

// WP-E v3.1 NumericalDispatch adapter for the complete C-grid predictor.
// This orchestrates d2a2c_vect, c_sw, divergence_corner, and corner_service
// calls in the correct Fortran-derived order.
//
// Registered as NumericalDispatch::cgrid_predictor.
int launch_cgrid_predictor_v3(const ExecutionContext &context, fv3_gpu_error *error);

}  // namespace fv3::gpu::cgrid
