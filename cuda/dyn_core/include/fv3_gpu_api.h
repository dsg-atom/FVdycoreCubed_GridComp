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
