#pragma once
#include "fv3_gpu_api.h"

namespace fv3::gpu {
#if defined(FV3_GPU_FV_REAL_IS_FLOAT)
using fv_real = float;
inline constexpr fv3_gpu_precision active_precision = FV3_GPU_PRECISION_R4;
#elif defined(FV3_GPU_FV_REAL_IS_DOUBLE)
using fv_real = double;
inline constexpr fv3_gpu_precision active_precision = FV3_GPU_PRECISION_R8;
#else
#error "Define exactly one of FV3_GPU_FV_REAL_IS_FLOAT or FV3_GPU_FV_REAL_IS_DOUBLE"
#endif
using accumulator_real = double;
static_assert(sizeof(accumulator_real) == 8, "FV3 accumulator ABI requires real64");
}
