#pragma once
#include "fv3_gpu_state.hpp"
#include <cstdint>

namespace fv3::gpu {
enum class ContextPhase { created, initialized, failed };
struct ContextData {
  fv3_gpu_create_desc create{};
  State state;
  Workspace workspace;
  void *compute_stream = nullptr;
  void *transfer_stream = nullptr;
  uint64_t next_event = 1;
  uint64_t last_event = 0;
  ContextPhase phase = ContextPhase::created;
};
}
