# WP-B D-grid translation

This directory is owned by WP-B. `wp_b_kernels.hpp` contains traceable,
non-owning translations of four contiguous pieces of the authoritative
Fortran dataflow: face flux/Courant preparation, real64 flux-capacitor
accumulation, mass/scalar conservative application, and vertically ordered
`mix_dp` repair. The routines consume WP-E field descriptors and allocate no
state.

They are currently host-callable reference implementations, not submitted CUDA
kernels. WP-E v2 now supplies field/grid registries, workspace slices,
submission, and dependencies; `wp_b_contract.*` connects WP-B to those shared
facilities. The remaining immutable metric and damping/order configuration gap
is recorded in `docs/agent_handoff/wp-b.md`. Building a private replacement
would violate Gate A.
