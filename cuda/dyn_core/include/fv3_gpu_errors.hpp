#pragma once
#include "fv3_gpu_api.h"
#include <string>

namespace fv3::gpu {
int fail(fv3_gpu_error *error, fv3_gpu_status code, const char *api,
         const std::string &message, int native_code = 0, int rank = -1,
         int device = -1, uint64_t sequence = 0) noexcept;
void clear_error(fv3_gpu_error *error) noexcept;
}
