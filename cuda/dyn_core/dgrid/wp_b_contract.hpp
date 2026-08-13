#pragma once

#include "fv3_gpu_internal.hpp"

namespace fv3::gpu::dgrid {

struct ContractViews {
  const DeviceViewBase *u, *v, *w, *pt, *delp, *uc, *vc;
  const DeviceViewBase *mfx, *mfy, *cx, *cy, *dpx;
  const DeviceViewBase *divgd, *crx, *cry, *xfx, *yfx, *heat_source;
  const DeviceViewBase *area, *rarea, *dx, *dy;
  const DeviceViewBase *rdx, *rdy, *rdxa, *rdya;
  const DeviceViewBase *cosa_u, *cosa_v, *cosa_s, *sina_u, *sina_v;
  const DeviceViewBase *rsin_u, *rsin_v, *rsin2, *cosa, *rsina, *sin_sg;
  const DeviceViewBase *f0, *divg_u, *divg_v, *del6_u, *del6_v;
  const DeviceViewBase *d2_bg, *d4_bg, *damp_v, *damp_w, *damp_t, *kgb;
  const DeviceViewBase *ndif;
  const WorkspaceSlice *transport;
  const NumericalConfig *config;
  const DispatchParams *params;
  const TopologyView *topology;
  fv3_gpu_event cwind_event, divgd_event;
};

// Resolves only WP-E-owned v3 identities and checks the coarse dependencies. It
// does not launch, synchronize, allocate, or mutate dependency state.
int resolve_contract_v3(const ExecutionContext &, ContractViews &,
                        fv3_gpu_error *) noexcept;

} // namespace fv3::gpu::dgrid
