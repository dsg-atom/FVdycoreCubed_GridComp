#ifndef FV3_GPU_COMMUNICATION_H
#define FV3_GPU_COMMUNICATION_H
#include "fv3_gpu_api.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum fv3_gpu_halo_group {
  FV3_GPU_H_DWIND_IN=1, FV3_GPU_H_W_IN, FV3_GPU_H_GZ_FIRST,
  FV3_GPU_H_Q_INLINE, FV3_GPU_H_DIVGD, FV3_GPU_H_CWIND,
  FV3_GPU_H_DSCALAR, FV3_GPU_H_ZH, FV3_GPU_H_PKC2,
  FV3_GPU_H_HEAT, FV3_GPU_H_DWIND_NEXT,
  /* E11: dyn_core.F90:1178 mpp_get_boundary(u, v, ..., gridtype=DGRID_NE),
   * gated it == n_split. Reconciles the redundantly computed SHARED STAGGERED
   * SEAM LINE (ring h = 0), not a halo ring; see halo_group_spec(). Appended
   * last so every existing group's numeric value is unchanged. */
  FV3_GPU_H_DWIND_SEAM
} fv3_gpu_halo_group;
/* Highest valid group. Use this, not the name of whichever group happens to be
 * last, for array sizing and group iteration. */
#define FV3_GPU_H_GROUP_LAST FV3_GPU_H_DWIND_SEAM
/* Milestone 3A.15 §10 -- which transport moves a halo group's packed bytes
 * between tiles. The numerical content of an exchange does not depend on this
 * choice; only the mechanism does. See include/fv3_gpu_halo_transport.hpp. */
typedef enum fv3_gpu_halo_transport_kind {
  FV3_GPU_HALO_TRANSPORT_NONE = 0,
  /* Model B: six tiles as six host threads in ONE process, device-to-device
   * copies, std::mutex rendezvous. Attached through the C++-only
   * attach_halo_peer_group(); a single-process caller keeps using that. */
  FV3_GPU_HALO_TRANSPORT_IN_PROCESS_PEER = 1,
  /* One tile per MPI rank: D2H of the packed send buffer, MPI point-to-point,
   * H2D into the recv buffer. This is what the real GEOS/FV3 production
   * decomposition (FV_NX=1, FV_NY=6, six separate ranks) requires. */
  FV3_GPU_HALO_TRANSPORT_MPI_HOST_STAGED = 2,
  /* Prompt 43 (Stage 2 of the Communication Evolution model): one tile per MPI
   * rank, same as MPI_HOST_STAGED, but MPI_Isend/MPI_Irecv are issued directly
   * on the packed DEVICE send/recv buffers -- no D2H/H2D payload staging.
   * Requires a CUDA-aware MPI runtime (Intel MPI 2021.13 with I_MPI_OFFLOAD=1,
   * confirmed in Prompt 42); the attach call fails clearly if that capability
   * is not present rather than silently falling back. */
  FV3_GPU_HALO_TRANSPORT_MPI_CUDA_AWARE = 3
} fv3_gpu_halo_transport_kind;

/* Attach an MPI halo transport to `context` for 1-based cubed-sphere `tile`,
 * which must equal the installed TopologyView::tile.
 *
 * `transport_kind` selects the backend and must be either
 * FV3_GPU_HALO_TRANSPORT_MPI_HOST_STAGED (the default, pinned-host-staged D2H/
 * H2D path) or FV3_GPU_HALO_TRANSPORT_MPI_CUDA_AWARE (Prompt 43: device
 * pointers passed directly to MPI, requires a CUDA-aware MPI runtime). Any
 * other value returns FV3_GPU_INVALID_ARGUMENT.
 *
 * THE COMMUNICATOR IS PASSED AS ITS FORTRAN INTEGER HANDLE, deliberately. MPI_Comm
 * is `int` in one MPI implementation and an opaque pointer in another, so it cannot
 * cross a stable C ABI that Fortran also calls through; the Fortran handle is an
 * INTEGER in every implementation and is exactly what an ISO_C_BINDING caller
 * already holds. A C caller converts with MPI_Comm_c2f(comm); a Fortran caller
 * passes its communicator directly. The transport converts back with MPI_Comm_f2c.
 *
 * The caller owns MPI initialization and finalization. This function does not call
 * MPI_Init, and detaching does not call MPI_Finalize.
 *
 * Returns FV3_GPU_UNSUPPORTED_CONFIGURATION if this build was configured without
 * FV3_DYN_CORE_ENABLE_MPI_HALO, or if FV3_GPU_HALO_TRANSPORT_MPI_CUDA_AWARE was
 * requested and the linked MPI runtime does not report CUDA-buffer support, so a
 * caller gets a diagnosable refusal rather than a link error or a silent
 * fallback. */
int fv3_gpu_attach_mpi_halo_transport(fv3_gpu_context *context,
                                      int communicator_fortran_handle,
                                      int32_t tile, int32_t transport_kind,
                                      fv3_gpu_error *error);

/* Detach whichever transport is attached and release the halo buffers. Safe to
 * call when none is attached. Every participant must be past its last exchange
 * before any of them detaches -- the same obligation MPI_Finalize imposes. */
int fv3_gpu_detach_halo_transport(fv3_gpu_context *context, fv3_gpu_error *error);

/* Which transport is attached, as fv3_gpu_halo_transport_kind. Reports
 * FV3_GPU_HALO_TRANSPORT_NONE for a null context or an unattached one, so a caller
 * can verify that it got the transport it asked for instead of assuming it. */
int fv3_gpu_halo_transport_kind_of(const fv3_gpu_context *context);

typedef int (*fv3_gpu_halo_begin_fn)(void *token, uint32_t group,
                                    void *send_buffer, uint64_t send_bytes,
                                    void *recv_buffer, uint64_t recv_bytes);
typedef int (*fv3_gpu_halo_complete_fn)(void *token, uint32_t group);
typedef struct fv3_gpu_halo_callbacks {
  uint32_t abi_version;
  uint32_t struct_bytes;
  void *token;
  fv3_gpu_halo_begin_fn begin;
  fv3_gpu_halo_complete_fn complete;
} fv3_gpu_halo_callbacks;
#ifdef __cplusplus
}
#endif
#endif
