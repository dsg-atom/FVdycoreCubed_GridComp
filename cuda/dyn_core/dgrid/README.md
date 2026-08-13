# WP-B D-grid translation

This directory is owned by WP-B. `wp_b_kernels.hpp` contains traceable,
non-owning translations of four contiguous pieces of the authoritative
Fortran dataflow: face flux/Courant preparation, real64 flux-capacitor
accumulation, mass/scalar conservative application, and vertically ordered
`mix_dp` repair. The routines consume WP-E field descriptors and allocate no
state.

They are currently host-callable reference implementations, not submitted CUDA
kernels. This is deliberate: WP-E v1 has no internal numerical-package API for
field lookup, immutable grid/config views, workspace slices, stream submission,
or dependency events. Building a private replacement would violate Gate A.
See `docs/agent_handoff/wp-b.md` for the blocking contract request and the full
Fortran/PACE/kernel map.
