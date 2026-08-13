#pragma once

#include "fv3_gpu_internal.hpp"

namespace fv3::gpu::dgrid {

struct ContractViews {
  const DeviceViewBase *u, *v, *w, *pt, *delp, *uc, *vc;
  const DeviceViewBase *mfx, *mfy, *cx, *cy, *dpx;
  const DeviceViewBase *divgd, *crx, *cry, *xfx, *yfx, *heat_source;
  const DeviceViewBase *area, *rarea, *dx, *dy;
  const WorkspaceSlice *transport;
  fv3_gpu_event cwind_event, divgd_event;
};

// Resolves only WP-E-owned identities and checks the coarse dependencies.  It
// does not launch, synchronize, allocate, or mutate dependency state.
int resolve_contract_v2(const ExecutionContext &, ContractViews &,
                        fv3_gpu_error *) noexcept;

} // namespace fv3::gpu::dgrid
