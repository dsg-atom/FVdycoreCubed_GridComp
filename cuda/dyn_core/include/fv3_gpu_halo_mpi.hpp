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
 * ------------------------------------------------------------------------- */

#include "fv3_gpu_halo_transport.hpp"

#include <string>

namespace fv3::gpu::halo {

/* Construct the host-staged MPI transport for 1-based cubed-sphere `tile` on the
 * communicator given by its Fortran handle. Returns nullptr on failure, with a
 * caller-facing explanation written to `why` when it is non-null -- the reasons are
 * all configuration errors the caller can act on (MPI not initialized, communicator
 * not of size 6, this rank's tile not matching the rank->tile identity the
 * production decomposition requires).
 *
 * Does NOT call MPI_Init. The caller owns MPI initialization and finalization.
 *
 * IS COLLECTIVE over `comm`: it queries size/rank and verifies the tile mapping
 * across ranks, so every rank of the communicator must call it. */
HaloTransport *create_mpi_host_staged_transport(int comm_fortran_handle, int32_t tile,
                                               std::string *why) noexcept;

/* Destroy a transport returned above. Does not call MPI_Finalize and does not free
 * the communicator, which the caller still owns. */
void destroy_mpi_host_staged_transport(HaloTransport *t) noexcept;

}  // namespace fv3::gpu::halo
