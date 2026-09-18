#pragma once
/* Milestone 3A.15 §4-§7 -- construction of the host-staged MPI halo transport.
 *
 * This header exposes NO MPI type and includes no MPI header, deliberately. It is
 * reached from runtime/fv3_gpu_runtime.cpp, which is compiled into the same library
 * whether or not MPI is available, and from a public C entry point that Fortran
 * calls through; leaking MPI_Comm (an `int` in MPICH, an opaque pointer in Open MPI)
 * into either would make the build's MPI choice part of this project's ABI.
 *
 * The communicator therefore crosses as its FORTRAN INTEGER handle, which is an
 * INTEGER in every implementation. The transport converts with MPI_Comm_f2c
 * internally; that conversion is the only place an MPI handle type appears.
 *
 * WHAT THE BACKEND IS AND WHAT IT COSTS
 *
 * "Stage 1" of project_architecture.md's Communication Evolution: pinned host
 * staging, not GPU-direct. Per halo group per tile it does exactly
 *
 *     D2H(whole packed send region)  ->  MPI_Irecv x(4*fields) + MPI_Isend x(4*fields)
 *       ->  MPI_Waitall  ->  H2D(whole recv region)
 *
 * The D2H must complete before the sends can be posted, so this backend contains
 * ONE host wait per exchange (on its own event -- never a stream or device
 * synchronize). That wait is the intrinsic cost of host staging and is the reason
 * Stage 2 (CUDA-aware MPI, sending the device pointers directly) exists as a later
 * step. It does not violate persistent residency: only the packed EDGE buffers move,
 * never model state, and the device-resident state is never round-tripped.
 *
 * Everything numerical -- which cells, which ring, which rotation, which sign,
 * which donor slot a rotated seam reads -- comes from fv3_gpu_halo_exchange.hpp and
 * is shared verbatim with the in-process backend. See fv3_gpu_halo_transport.hpp.
 *
 * Prompt 43 adds "Stage 2": TransportMode::cuda_aware. Same tag scheme, same
 * message count, same rank<->tile mapping, same slot_base() offsets into the
 * SAME device send/recv buffers -- the only difference is that MPI_Isend/
 * MPI_Irecv address those device buffers directly, skipping the D2H/H2D pair.
 * Requires a CUDA-aware MPI runtime (verified for Intel MPI 2021.13 with
 * I_MPI_OFFLOAD=1 in Prompt 42); construction fails clearly if the runtime does
 * not report that capability rather than silently falling back to host-staged.
 * ------------------------------------------------------------------------- */

#include "fv3_gpu_halo_transport.hpp"

#include <string>

namespace fv3::gpu::halo {

/* Prompt 43: which MPI halo transport variant to construct. Both share all
 * topology/pack/unpack/tag logic; only the payload path differs. */
enum class TransportMode { host_staged, cuda_aware };

/* Construct an MPI halo transport of the given `mode` for 1-based cubed-sphere
 * `tile` on the communicator given by its Fortran handle. Returns nullptr on
 * failure, with a caller-facing explanation written to `why` when it is
 * non-null. For TransportMode::host_staged the reasons are configuration
 * errors (MPI not initialized, communicator not of size 6, this rank's tile
 * not matching the rank->tile identity the production decomposition
 * requires). For TransportMode::cuda_aware, construction additionally fails
 * with an explanation if the linked MPI runtime does not report CUDA-buffer
 * support (queried via MPIX_Query_cuda_support(), the exact discriminator
 * validated against a real A100 in Prompt 42) -- there is no silent fallback
 * to host_staged.
 *
 * Does NOT call MPI_Init. The caller owns MPI initialization and finalization.
 *
 * IS COLLECTIVE over `comm`: it queries size/rank and verifies the tile mapping
 * across ranks, so every rank of the communicator must call it. */
HaloTransport *create_mpi_transport(int comm_fortran_handle, int32_t tile,
                                    TransportMode mode, std::string *why) noexcept;

/* Backward-compatible wrapper: create_mpi_transport(..., TransportMode::host_staged, ...). */
HaloTransport *create_mpi_host_staged_transport(int comm_fortran_handle, int32_t tile,
                                               std::string *why) noexcept;

/* Destroy a transport returned by either constructor above. Does not call
 * MPI_Finalize and does not free the communicator, which the caller still owns. */
void destroy_mpi_host_staged_transport(HaloTransport *t) noexcept;

}  // namespace fv3::gpu::halo
