#pragma once
#include "fv3_gpu_api.h"
#include "fv3_gpu_internal.hpp"
#include <cstdint>
#include <vector>

namespace fv3::gpu {
struct DeviceField {
  fv3_gpu_field_desc host{};
  void *device = nullptr;
  uint64_t host_epoch = 0;
  uint64_t device_epoch = 0;
};
struct State { std::vector<DeviceField> fields; FieldRegistry registry; };
struct Workspace { void *allocation = nullptr; WorkspaceLayout layout{}; };
}
