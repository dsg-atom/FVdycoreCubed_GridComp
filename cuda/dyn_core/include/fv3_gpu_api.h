#ifndef FV3_GPU_API_H
#define FV3_GPU_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FV3_GPU_ABI_VERSION 1u
#define FV3_GPU_MAX_RANK 4u
#define FV3_GPU_ERROR_MESSAGE_BYTES 256u
#define FV3_GPU_FIELD_NAME_BYTES 32u

typedef struct fv3_gpu_context fv3_gpu_context;
typedef uint64_t fv3_gpu_event;

typedef enum fv3_gpu_status {
  FV3_GPU_SUCCESS = 0,
  FV3_GPU_INVALID_ARGUMENT = 1,
  FV3_GPU_UNSUPPORTED_CONFIGURATION = 2,
  FV3_GPU_CUDA_ERROR = 3,
  FV3_GPU_MPI_ERROR = 4,
  FV3_GPU_OUT_OF_MEMORY = 5,
  FV3_GPU_STATE_CONFLICT = 6,
  FV3_GPU_INTERNAL_ERROR = 7
} fv3_gpu_status;

typedef enum fv3_gpu_element_type {
  FV3_GPU_ELEMENT_F32 = 1,
  FV3_GPU_ELEMENT_F64 = 2
} fv3_gpu_element_type;

typedef enum fv3_gpu_memory_location {
  FV3_GPU_MEMORY_HOST = 1,
  FV3_GPU_MEMORY_DEVICE = 2
} fv3_gpu_memory_location;

typedef enum fv3_gpu_stagger {
  FV3_GPU_STAGGER_A = 1,
  FV3_GPU_STAGGER_C_U = 2,
  FV3_GPU_STAGGER_C_V = 3,
  FV3_GPU_STAGGER_D_U = 4,
  FV3_GPU_STAGGER_D_V = 5,
  FV3_GPU_STAGGER_B = 6,
  FV3_GPU_STAGGER_INTERFACE = 7,
  FV3_GPU_STAGGER_X_FACE = 8,
  FV3_GPU_STAGGER_Y_FACE = 9
} fv3_gpu_stagger;

typedef enum fv3_gpu_access {
  FV3_GPU_ACCESS_INPUT = 1,
  FV3_GPU_ACCESS_OUTPUT = 2,
  FV3_GPU_ACCESS_INOUT = 3
} fv3_gpu_access;

typedef enum fv3_gpu_precision {
  FV3_GPU_PRECISION_R4 = 1,
  FV3_GPU_PRECISION_R8 = 2
} fv3_gpu_precision;

typedef struct fv3_gpu_error {
  uint32_t abi_version;
  uint32_t struct_bytes;
  int32_t code;
  int32_t native_code;
  int32_t rank;
  int32_t device;
  uint64_t sequence;
  char api[FV3_GPU_FIELD_NAME_BYTES];
  char message[FV3_GPU_ERROR_MESSAGE_BYTES];
} fv3_gpu_error;

typedef struct fv3_gpu_field_desc {
  uint32_t abi_version;
  uint32_t struct_bytes;
  uint32_t field_id;
  char name[FV3_GPU_FIELD_NAME_BYTES];
  void *base;
  uint64_t allocation_bytes;
  uint32_t element_type;
  uint32_t element_bytes;
  uint32_t rank;
  uint32_t memory_location;
  int64_t lower[FV3_GPU_MAX_RANK];
  uint64_t extent[FV3_GPU_MAX_RANK];
  int64_t stride_elements[FV3_GPU_MAX_RANK];
  int64_t compute_lower[FV3_GPU_MAX_RANK];
  int64_t compute_upper[FV3_GPU_MAX_RANK];
  uint32_t halo_lower[FV3_GPU_MAX_RANK];
  uint32_t halo_upper[FV3_GPU_MAX_RANK];
  uint32_t stagger;
  uint32_t access;
  uint32_t alignment_bytes;
  uint32_t reserved;
} fv3_gpu_field_desc;

typedef struct fv3_gpu_create_desc {
  uint32_t abi_version;
  uint32_t struct_bytes;
  int32_t device_ordinal;
  int32_t mpi_rank;
  int32_t mpi_size;
  uint32_t precision;
  int64_t is, ie, js, je, npz;
  uint32_t halo_width;
  uint32_t flags;
  uintptr_t communicator_token;
} fv3_gpu_create_desc;

typedef struct fv3_gpu_state_desc {
  uint32_t abi_version;
  uint32_t struct_bytes;
  uint32_t field_count;
  uint32_t reserved;
  const fv3_gpu_field_desc *fields;
} fv3_gpu_state_desc;

typedef struct fv3_gpu_field_mask {
  uint32_t abi_version;
  uint32_t struct_bytes;
  uint64_t words[4];
} fv3_gpu_field_mask;

typedef struct fv3_gpu_step_desc {
  uint32_t abi_version;
  uint32_t struct_bytes;
  double bdt;
  int32_t k_split;
  int32_t n_split;
  uint8_t init_step;
  uint8_t end_step;
  uint16_t reserved0;
  uint64_t model_step;
  uint64_t acoustic_sequence;
  uint64_t validation_mask;
  uint64_t physics_tail_mask;
  fv3_gpu_event dependency;
} fv3_gpu_step_desc;

typedef struct fv3_gpu_capabilities {
  uint32_t abi_version;
  uint32_t struct_bytes;
  uint32_t compiled_with_cuda;
  uint32_t device_count;
  uint32_t precision_mask;
  uint32_t communication_stage;
  uint64_t feature_mask;
} fv3_gpu_capabilities;

/* Prompt 14/15 resumption: the production first-run numerical configuration
 * (dyn_core.F90 namelist/derived values -- hord_*, nord*, akap, akap_exner,
 * rdgas, grav, m_split, ...). Field names and order mirror
 * fv3::gpu::NumericalConfig (fv3_gpu_internal.hpp) exactly, one-for-one, so
 * the runtime's translation is a plain member copy with no reinterpretation.
 * Before fv3_gpu_set_numerical_config existed, the ONLY way to install this
 * struct was the internal, non-ABI C++ setter set_numerical_config(), reached
 * only from benchmark harness code -- there was no public entry point a
 * Fortran caller (or any other non-C++ caller) could use. This struct and its
 * setter close that gap; the internal NumericalConfig type itself, and every
 * numerical formula that reads it, are unchanged. */
typedef struct fv3_gpu_numerical_config {
  uint32_t abi_version;
  uint32_t struct_bytes;
  double ptop, akap, a_imp, p_fac, scale_m, dz_min;
  double akap_exner;
  double rdgas, grav;
  double beta, d_ext, dddmp, d_con, lim_fac, zvir;
  double da_min, da_min_c, delt_max;
  double rayleigh_tau, rayleigh_ztop;
  int32_t hord_tr, hord_mt, hord_vt, hord_tm, hord_dp;
  int32_t hord_height, a2b_order;
  int32_t nord, nord_v, nord_w, nord_t;
  int32_t nf_ke, n_con, vertical_solver;
  int32_t m_split;
  uint8_t use_cond, moist_cappa, use_logp, inline_q;
  uint8_t hydrostatic, rot3, do_f3d, do_skeb;
  uint8_t use_sg, rayleigh_enabled, grid_type, reserved0;
  uint8_t fuse_corner_fill;
  uint8_t fuse_halo_edges;
} fv3_gpu_numerical_config;

int fv3_gpu_validate_field(const fv3_gpu_field_desc *, fv3_gpu_error *);

/* Prompt 77: bounds/topology for fv3_gpu_fv_tp_2d_hord8, one call per
 * (k, iq) tracer-transport slice. Mirrors TransportBounds
 * (wp_b_cuda.cuh) exactly, minus the fields that batched multi-level
 * launches need and this single-slice bridge does not (npz/nk are
 * fixed at 1 internally; halo_width/ng are fixed at the FV3 ng==3
 * ghost-zone convention). is/ie/js/je/isd/ied/jsd/jed are the Fortran
 * fv_grid_bounds_type compute/memory bounds for this tile; npx/npy are
 * the tile corner counts. hord is the transport order actually used at
 * the call site (only 6 and 8 are implemented downstream; any other
 * value is rejected before any kernel launch, matching
 * launch_fv_tp_2d_cuda's own dispatch-gate rejection). */
typedef struct fv3_gpu_tracer_transport_bounds {
  uint32_t abi_version;
  uint32_t struct_bytes;
  int64_t is, ie, js, je;
  int64_t isd, ied, jsd, jed;
  int64_t npx, npy;
  int32_t hord;
  int32_t cubed_sphere;
  int32_t nested;
  int32_t west_edge, east_edge, south_edge, north_edge;
  int32_t sw_corner, se_corner, ne_corner, nw_corner;
} fv3_gpu_tracer_transport_bounds;

/* Prompt 77: production dispatch bridge for tp_core.F90::fv_tp_2d at
 * fv_tracer2d.F90::tracer_2d_1L's nsplt==1 tracer-transport call site,
 * reusing the already GPU-executed/validated
 * fv3::gpu::dgrid::launch_fv_tp_2d_cuda entry point (hord==6/hord==8;
 * see p75/p76 handoffs). Deliberately context-free and
 * FieldRegistry-free: tracer temporaries (q, crx/cry, xfx/yfx,
 * mfx/mfy, ra_x/ra_y) have no FieldRegistry identity, so this takes
 * plain host array pointers laid out exactly as fv_tp_2d's own dummy
 * arguments (tp_core.F90:108-135) -- q/crx/cry/xfx/yfx/ra_x/ra_y/mfx/
 * mfy/area/dxa/dya sized per fv_tp_2d's own bound declarations for the
 * tile described by `b`, column-major (Fortran) order.
 *
 * lim_fac is NOT an argument: the CUDA kernel implements only the
 * lim_fac==1.0 production-default PPM limiter behaviour, so the caller
 * MUST verify lim_fac==1.0 itself (flagstruct%lim_fac) before calling --
 * this entry does not check it. mfx/mfy are REQUIRED and unconditional
 * (the tracer call site never omits them); fv_tp_2d's mfx/mfy-absent
 * branch is out of scope for this bridge.
 *
 * q is intent(inout), exactly as in Fortran: copy_corners mutates
 * ghost/corner cells of q in place. fx_out/fy_out are intent(out), sized
 * (is:ie+1, js:je) / (is:ie, js:je+1) respectively, matching fv_tp_2d's
 * own fx/fy bounds.
 *
 * Not part of fv3_gpu_context's lifecycle -- no fv3_gpu_create/
 * fv3_gpu_finalize interaction. Maintains its own persistent device
 * workspace and caches the invariant area/dxa/dya grid metrics across
 * calls (re-uploaded only when `b`'s bounds change from the previous
 * call), mirroring the Prompt 68 device-workspace-reuse precedent; all
 * other arrays are re-uploaded every call since they change per (k,iq).
 * Returns FV3_GPU_UNSUPPORTED_CONFIGURATION if b->hord is not 6 or 8. */
int fv3_gpu_fv_tp_2d_hord8(const fv3_gpu_tracer_transport_bounds *b,
                            double *q, const double *crx,
                            const double *cry, const double *xfx,
                            const double *yfx, const double *mfx,
                            const double *mfy, const double *ra_x,
                            const double *ra_y, const double *area,
                            const double *dxa, const double *dya,
                            double *fx_out, double *fy_out,
                            fv3_gpu_error *);

int fv3_gpu_create(const fv3_gpu_create_desc *, fv3_gpu_context **, fv3_gpu_error *);
/* Install the production numerical configuration. Must be called after
 * fv3_gpu_create and before the first fv3_gpu_dyn_core_step; may be called
 * again later only while no graph has been captured on this context (same
 * rule fv3_gpu_set_cuda_graph_mode already enforces on the dispatch table).
 * Returns FV3_GPU_INVALID_ARGUMENT if the header (abi_version/struct_bytes)
 * does not match this build. */
int fv3_gpu_set_numerical_config(fv3_gpu_context *, const fv3_gpu_numerical_config *, fv3_gpu_error *);
/* Install the fixed production stage dispatch table (WP-A/B/C/D kernel
 * launchers). This table has no caller-supplied parameters -- it is not
 * configuration, it is which already-validated implementation of each stage
 * to call, and it is the same table the benchmark installs via
 * fv3_build_production_dispatch(). Must be called once, after
 * fv3_gpu_set_numerical_config and before the first fv3_gpu_dyn_core_step. */
int fv3_gpu_install_production_dispatch(fv3_gpu_context *, fv3_gpu_error *);
int fv3_gpu_initialize_state(fv3_gpu_context *, const fv3_gpu_state_desc *, fv3_gpu_event *, fv3_gpu_error *);
int fv3_gpu_sync_from_host(fv3_gpu_context *, const fv3_gpu_field_mask *, fv3_gpu_event *, fv3_gpu_error *);
/* Milestone 3A.10: supply the cubed-sphere decomposition.
 *
 * Until this existed there was NO way for a caller to describe its topology --
 * set_topology is a C++-only internal setter with no production caller, and the
 * Fortran bridge never mentioned tile, npx or npy. The runtime consequently ran
 * with a default-constructed TopologyView, which the halo vacuity test read as
 * "nothing to exchange" and which therefore produced halo readiness for free.
 * That is the fail-open defect this milestone closes, so the caller must now be
 * able to say what its decomposition actually is.
 *
 *   tile          1..6, FMS/FV3 mosaic tile numbering (Atm%tile).
 *   npx, npy      FV3 corner counts (Atm%npx, Atm%npy); owned cells are
 *                 npx-1 by npy-1. Must be equal -- a cube has square tiles
 *                 (CubeHalo.F90:103).
 *   rank_of_tile  6 entries, MPI rank owning tiles 1..6, indexed 0..5.
 *
 * This entry describes the ONE-RANK-PER-TILE decomposition. A rank owning only
 * part of a tile has same-tile neighbours on some edges and panel seams on
 * others; that case is deliberately not expressible here and is rejected rather
 * than approximated. See the milestone 3A.10 handoff for what remains for
 * general multi-rank MPI.
 *
 * Returns FV3_GPU_INVALID_ARGUMENT on any inconsistent argument, without
 * modifying the context. */
int fv3_gpu_set_cubed_sphere_topology(fv3_gpu_context *, int32_t tile,
                                      int64_t npx, int64_t npy,
                                      const int32_t *rank_of_tile,
                                      fv3_gpu_error *);
int fv3_gpu_dyn_core_step(fv3_gpu_context *, const fv3_gpu_step_desc *, fv3_gpu_event *, fv3_gpu_error *);
/* Prompt 52C-IMPLEMENT: fv_mapz_mod's production pt/delz vertical remap
 * (fv3::gpu::vertical::launch_pt_delz_remap_ctx), reached through the
 * NumericalDispatch pt_delz_remap slot installed by
 * fv3_gpu_install_production_dispatch. Runs OUTSIDE the acoustic-substep
 * loop, once per dyn_core call, at fv_mapz.F90's Lagrangian_to_Eulerian call
 * site -- NOT part of fv3_gpu_dyn_core_step. The caller is responsible for
 * fv3_gpu_sync_from_host before this call (mask covering pe, peln, pt, delz,
 * delp, pk, pkz, ak, bk) and fv3_gpu_sync_to_host with the same mask after --
 * this entry only orders the compute-stream launch after `dependency` and
 * returns the kernel-completion event; it neither transfers nor blocks. */
int fv3_gpu_pt_delz_remap(fv3_gpu_context *, fv3_gpu_event dependency, fv3_gpu_event *, fv3_gpu_error *);
/* Prompt 71: fv_mapz_mod's production u/v vertical remap (REMAP_UV)
 * (fv3::gpu::vertical::launch_remap_uv_ctx), reached through the
 * NumericalDispatch remap_uv slot installed by
 * fv3_gpu_install_production_dispatch. Runs OUTSIDE the acoustic-substep
 * loop, once per dyn_core call, at fv_mapz.F90's Lagrangian_to_Eulerian
 * call site (timer 'Remap_UV') -- NOT part of fv3_gpu_dyn_core_step. Must
 * be dispatched AFTER fv3_gpu_pt_delz_remap within the same dyn_core call
 * (reads pe only after PT/DELZ's own closing update). The caller is
 * responsible for fv3_gpu_sync_from_host before this call (mask covering
 * u, v, pe, ak, bk) and fv3_gpu_sync_to_host with the same mask after --
 * this entry only orders the compute-stream launch after `dependency` and
 * returns the kernel-completion event; it neither transfers nor blocks. */
int fv3_gpu_remap_uv(fv3_gpu_context *, fv3_gpu_event dependency, fv3_gpu_event *, fv3_gpu_error *);
/* Prompt 85/86B/87: fv_mapz_mod's production tracer vertical remap
 * (REMAP_Q, mapn_tracer nq=2: sphum kord=9, cld_amt kord=9 -- corrected by
 * Prompt 87 from the originally assumed kord=[8,9]; FV_StateMod.F90:563
 * unconditionally forces flagstruct%kord_tr=9 in every GEOS-integrated
 * runtime, so sphum's effective kord is 9, not fv_arrays.F90:333's bare
 * default of 8)
 * (fv3::gpu::vertical::launch_remap_q_ctx), reached through the
 * NumericalDispatch remap_q slot installed by
 * fv3_gpu_install_production_dispatch. Runs OUTSIDE the acoustic-substep
 * loop, once per dyn_core call, at fv_mapz.F90's Lagrangian_to_Eulerian
 * call site (timer 'Remap_Q') -- NOT part of fv3_gpu_dyn_core_step. Must be
 * dispatched AFTER fv3_gpu_pt_delz_remap within the same dyn_core call
 * (reads pe only after PT/DELZ's own closing update).
 *
 * Unlike fv3_gpu_pt_delz_remap/fv3_gpu_remap_uv, q has no FieldRegistry
 * identity, so THIS call -- not a separate fv3_gpu_sync_from_host/
 * fv3_gpu_sync_to_host pair -- owns the one H2D (before both kernels) and
 * one D2H (after both kernels) that move it, via a persistent context-owned
 * device buffer, for the FULL nq-tracer block in one contiguous transfer
 * each way. `q` must point to exactly `q_elements` host doubles laid out as
 * Fortran's own q(isd:ied,jsd:jed,km,nq) for this tile (i fastest, then j,
 * then k, then nq slowest/outermost); `q_elements` must equal
 * (ied-isd+1)*(jed-jsd+1)*npz*nq for this context's current bounds or the
 * call fails closed with FV3_GPU_INVALID_ARGUMENT. `nq`/`kord1`/`kord2`
 * must be exactly 2/9/9 -- the only validated production tracer schedule
 * (fv_dynamics.F90:941-944 combined with FV_StateMod.F90:563's
 * unconditional kord_tr=9 override; corrected by Prompt 87) -- or the call
 * fails closed with
 * FV3_GPU_UNSUPPORTED_CONFIGURATION. `dependency`/`*ev` follow the same
 * one-call-in, one-event-out contract as fv3_gpu_pt_delz_remap/
 * fv3_gpu_remap_uv -- the caller must fv3_gpu_event_wait on `*ev` before
 * reading `q` again. R8-only (matches this port's validated scope); do not
 * call from an R4 build.
 *
 * Prompt 88B: `fill` is flagstruct%fill (FV_StateMod.F90:568's unconditional
 * override to .true., confirmed universal across every runnable config --
 * see docs/performance/p88a_remap_q_fill_semantics_2026-09-25.md), a plain
 * 0/1 flag (not `bool`, matching this ABI's existing int-flag convention,
 * e.g. fv3_gpu_set_cuda_graph_mode's `enabled`). Applied identically to
 * both tracers -- fillz has no per-tracer branching. Both 0 and 1 are
 * valid/accepted; there is no rejection based on this value -- fill=0 must
 * remain a working, regression-tested no-op path, matching every REMAP_Q
 * call site before this prompt. Does not change the H2D/D2H transfer
 * contract above -- fillz runs entirely inside the existing kernel launches
 * on already-resident scratch. */
int fv3_gpu_remap_q(fv3_gpu_context *, fv3_gpu_event dependency, double *q,
                    int64_t q_elements, int32_t nq, int32_t kord1, int32_t kord2,
                    int32_t fill, fv3_gpu_event *, fv3_gpu_error *);
/* Prompt 92: fv_mapz_mod's production pressure-variable vertical remap
 * (REMAP_PRESSUREVARS: pe/peln/pk/delp unconditional recompute followed by
 * remap_t's MOIST_CAPPA branch -- cappa/pkz, fv_mapz.F90:1239-1516)
 * (fv3::gpu::vertical::launch_remap_pressurevars_ctx), reached through the
 * NumericalDispatch remap_pressurevars slot installed by
 * fv3_gpu_install_production_dispatch. Runs OUTSIDE the acoustic-substep
 * loop, once per dyn_core call, at fv_mapz.F90's Lagrangian_to_Eulerian
 * call site (timer 'Remap_PressureVars') -- NOT part of
 * fv3_gpu_dyn_core_step. Must be dispatched AFTER fv3_gpu_pt_delz_remap
 * within the same dyn_core call (reads pe only after PT/DELZ's own closing
 * update); ordering relative to fv3_gpu_remap_uv/fv3_gpu_remap_q is not
 * otherwise constrained (disjoint field sets).
 *
 * The caller is responsible for fv3_gpu_sync_from_host before this call
 * (mask covering pe, peln, pk, pt, delz, ak, bk) and fv3_gpu_sync_to_host
 * with a mask covering pe, peln, pk, pkz, delp after -- this entry does NOT
 * itself move those fields. It DOES own its own transfer for the two
 * fields with no FieldRegistry identity here: `q_sphum` (read-only input,
 * H2D only) and `cappa` (round-tripped H2D then D2H, so halo bytes outside
 * the tile's own is:ie x js:je interior pass through this call byte-
 * identical -- the kernel never touches them). Both must point to exactly
 * `elements` host doubles laid out as Fortran's own (isd:ied,jsd:jed,km)
 * memory domain for this tile (i fastest, then j, then k) -- `q_sphum` is
 * a single q(:,:,:,sphum) tracer plane (contiguous because nq is the
 * Fortran-outermost dimension), `cappa` is its own standalone array of
 * the same shape. `elements` must equal (ied-isd+1)*(jed-jsd+1)*npz for
 * this context's current bounds, or the call fails closed with
 * FV3_GPU_INVALID_ARGUMENT. `dependency`/`*ev` follow the same one-call-in,
 * one-event-out contract as fv3_gpu_pt_delz_remap/fv3_gpu_remap_uv/
 * fv3_gpu_remap_q -- the caller must fv3_gpu_event_wait on `*ev` before
 * reading `cappa` again (this function does not block the host). R8-only
 * (matches this port's validated scope); do not call from an R4 build. */
int fv3_gpu_remap_pressurevars(fv3_gpu_context *, fv3_gpu_event dependency,
                               double *q_sphum, double *cappa,
                               int64_t elements, fv3_gpu_event *,
                               fv3_gpu_error *);
/* Prompt 99: FV_StateMod.F90's production State_To_FV, Phase A half
 * (fv3::gpu::vertical::launch_state_to_fv_phase_a_ctx), reached through the
 * NumericalDispatch state_to_fv_phase_a slot installed by
 * fv3_gpu_install_production_dispatch. Runs OUTSIDE the acoustic-substep
 * loop, once per k_split sub-iteration, immediately before fv_dynamics() --
 * NOT part of fv3_gpu_dyn_core_step. Covers everything State_To_FV does
 * BEFORE its one mpp_get_boundary call: u(:,:,:)=tiny_number/
 * v(:,:,:)=tiny_number full-buffer fills followed by the
 * u(isc:iec,jsc:jec,:)=STATE%VARS%U / v(...)=STATE%VARS%V interior copies.
 *
 * `state_u`/`state_v` have no FieldRegistry identity (MAPL State fields, not
 * FV_Atm fields) -- this call owns their own ONE-contiguous-H2D-each
 * transfer via a persistent context-owned device buffer, same convention as
 * fv3_gpu_remap_q's `q`. Each must point to exactly `elements` host doubles
 * laid out as Fortran's own STATE%VARS%U/V(isc:iec,jsc:jec,km) for this tile
 * (i fastest, then j, then k -- NO halo, unlike remap_q/remap_pressurevars'
 * isd:ied x jsd:jed memory-domain layout). `elements` must equal
 * (ie-is+1)*(je-js+1)*npz for this context's current bounds, or the call
 * fails closed with FV3_GPU_INVALID_ARGUMENT.
 *
 * This call does NOT perform mpp_get_boundary or the nbuffer/ebuffer edge-
 * row scatter -- the caller (cuda/interface/fv3_gpu_dyncore_bridge.F90's
 * fv3_gpu_interface_state_to_fv_phase_a) is responsible for following this
 * call with fv3_gpu_event_wait, then fv3_gpu_sync_to_host (u/v-only mask)
 * to bring u/v to the host for the UNCHANGED Fortran mpp_get_boundary call
 * and edge scatter, then fv3_gpu_sync_from_host (same mask) before calling
 * fv3_gpu_state_to_fv_phase_c. `dependency`/`*ev` follow the same one-call-
 * in, one-event-out contract as fv3_gpu_remap_q/fv3_gpu_remap_pressurevars.
 * R8-only (matches this port's validated scope); do not call from an R4
 * build. */
int fv3_gpu_state_to_fv_phase_a(fv3_gpu_context *, fv3_gpu_event dependency,
                                double *state_u, double *state_v,
                                int64_t elements, fv3_gpu_event *,
                                fv3_gpu_error *);
/* Prompt 99: FV_StateMod.F90's production State_To_FV, Phase C half
 * (fv3::gpu::vertical::launch_state_to_fv_phase_c_ctx), reached through the
 * NumericalDispatch state_to_fv_phase_c slot installed by
 * fv3_gpu_install_production_dispatch. Runs immediately after the caller's
 * post-mpp_get_boundary fv3_gpu_sync_from_host (u/v-only mask) described at
 * fv3_gpu_state_to_fv_phase_a above, still before fv_dynamics() -- NOT part
 * of fv3_gpu_dyn_core_step. Covers everything State_To_FV does AFTER
 * mpp_get_boundary: the w interior copy (hydrostatic=false required, or the
 * call fails closed with FV3_GPU_UNSUPPORTED_CONFIGURATION), the pe(:,:,:)=
 * tiny_number fill followed by the full pe transpose/peln/pk recompute over
 * ALL km+1 levels (no boundary pass-through, unlike REMAP_PRESSUREVARS),
 * the ps pack, the delp(:,:,:)=tiny_number fill and delp=pe(k+1)-pe(k)
 * recompute, the pt(:,:,:)=tiny_number fill and pt=STATE%VARS%PT*
 * STATE%VARS%PKZ recompute, and the delz/pkz interior copies.
 *
 * `state_w`/`state_pe`/`state_pt`/`state_pkz`/`state_dz` have no
 * FieldRegistry identity -- this call owns their own ONE-contiguous-H2D-
 * each transfer via persistent context-owned device buffers, same
 * convention as fv3_gpu_state_to_fv_phase_a's state_u/state_v. Each of
 * state_w/state_pt/state_pkz/state_dz must point to exactly `elements` host
 * doubles laid out as Fortran's own STATE%VARS%*(isc:iec,jsc:jec,km) for
 * this tile (i fastest, then j, then k -- no halo); `state_pe` must point to
 * exactly `elements_pe` host doubles laid out as STATE%VARS%PE(isc:iec,
 * jsc:jec,km+1) (same i/j/k order, one extra level). `elements` must equal
 * (ie-is+1)*(je-js+1)*npz and `elements_pe` must equal
 * (ie-is+1)*(je-js+1)*(npz+1) for this context's current bounds, or the
 * call fails closed with FV3_GPU_INVALID_ARGUMENT.
 *
 * `ps` also has no FieldRegistry identity and (unlike the inputs above) has
 * no downstream on-device consumer -- this call packs it into its own
 * persistent context-owned scratch buffer and D2H's that buffer directly
 * into `ps` after the kernel, ONE contiguous transfer, sized implicitly to
 * (ie-is+1)*(je-js+1) host doubles laid out as Fortran's own ps(isc:iec,
 * jsc:jec) (i fastest, then j) -- no separate element count parameter for
 * `ps` is needed or accepted.
 *
 * `dependency`/`*ev` follow the same one-call-in, one-event-out contract as
 * fv3_gpu_state_to_fv_phase_a -- the caller must fv3_gpu_event_wait on `*ev`
 * before reading `ps` again (this function does not block the host). R8-
 * only (matches this port's validated scope); do not call from an R4
 * build. */
int fv3_gpu_state_to_fv_phase_c(fv3_gpu_context *, fv3_gpu_event dependency,
                                double *state_w, double *state_pe,
                                double *state_pt, double *state_pkz,
                                double *state_dz, double *ps,
                                int64_t elements, int64_t elements_pe,
                                fv3_gpu_event *, fv3_gpu_error *);
int fv3_gpu_sync_to_host(fv3_gpu_context *, const fv3_gpu_field_mask *, fv3_gpu_event *, fv3_gpu_error *);
int fv3_gpu_event_wait(fv3_gpu_context *, fv3_gpu_event, fv3_gpu_error *);
int fv3_gpu_get_capabilities(fv3_gpu_context *, fv3_gpu_capabilities *);
/* Prompt 8 (CUDA Graphs prototype). Opt-in only; a context never enables graph
 * capture unless this is called with enabled != 0. Default execution (this
 * never called) is unchanged. Must be called before the first
 * fv3_gpu_dyn_core_step on this context -- capture happens on that stage's
 * first occurrence. Scope is deliberately narrow: only the dgrid_dynamics
 * stage is captured/replayed; every other stage and all cross-tile halo
 * communication run exactly as in default mode. See ContextData's
 * cuda_graph_enabled field for why this stage was chosen. */
int fv3_gpu_set_cuda_graph_mode(fv3_gpu_context *, int enabled, fv3_gpu_error *);
int fv3_gpu_finalize(fv3_gpu_context **, fv3_gpu_error *);

#ifdef __cplusplus
}
#endif
#endif
