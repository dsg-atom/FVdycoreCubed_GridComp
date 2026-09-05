/* Milestone 3A.10 §5/§6/§7/§9 -- GPU cubed-sphere halo pack, transfer, unpack.
 *
 * The declarative half (group field tables, the vector sign rule and the reasons
 * it is keyed on argument slot rather than staggering) is in
 * include/fv3_gpu_halo_exchange.hpp. Read that first; this file is only the
 * machinery.
 *
 * §5 requires the packing to be numerical-data movement on the device and
 * forbids host-side packing and full-state D2H/H2D. Accordingly:
 *
 *   - pack   : a device kernel reading the owner's field arrays, writing a
 *              device-resident send buffer;
 *   - transfer: cudaMemcpyAsync DeviceToDevice, send buffer -> peer recv buffer,
 *              on the runtime's transfer stream. Under the Model B Gate C1
 *              decomposition every tile is in one process's device memory, so
 *              this is a real device-to-device copy with no host staging at any
 *              point. It is the same call an MPI implementation would replace
 *              with a CUDA-aware Isend/Irecv over the identical buffers.
 *   - unpack : a device kernel reading the recv buffer and writing the owner's
 *              halo cells, applying the along-seam reversal, the component
 *              selection and the sign.
 *
 * Nothing here touches host memory. There is no cudaMemcpy to or from the host
 * in this file, and no cudaDeviceSynchronize.
 *
 * ---------------------------------------------------------------------------
 * WHAT AN EDGE STRIP IS, AND WHAT IT IS NOT (§7/§8)
 * ---------------------------------------------------------------------------
 *
 * Each edge strip spans the ALONG-SEAM range of the owner's own block only:
 * 1..edge_seam_extent(edge, nx, ny, stagger). It therefore stops short of the
 * corner halo blocks, where i and j are BOTH outside the owned range.
 *
 * That is deliberate and it is what FMS does: the four edge overlaps are
 * disjoint, so no halo cell is written twice and there is no precedence question
 * to resolve between them, and no dependence on kernel completion order. §7's
 * "define corner behavior explicitly when two edge exchanges meet" is satisfied
 * by their not meeting.
 *
 * The corner halo blocks themselves are a SEPARATE question -- on a cubed sphere
 * three tiles meet at a cube vertex and there is no fourth cell -- and are the
 * subject of §8. This file does not write them and does not claim to. Any stage
 * that reads a corner halo block is not covered by this exchange.
 *
 * The shared staggered seam LINE is a different operation again, and it IS now
 * handled here -- as its own group, FV3_GPU_H_DWIND_SEAM (exchange E11 of the §1
 * map), riding this same machinery at ring h = 0. Correction to what this comment
 * previously asserted: mpp_get_boundary does NOT stop at the 1-D
 * ebuffer/nbuffer. dyn_core.F90:1181-1188 copies both buffers straight back into
 * u and v, so the destination really is the field array -- owned cells, not halo
 * cells. See halo_group_spec(FV3_GPU_H_DWIND_SEAM) for the map and for the
 * evidence that it reproduces the authoritative result exactly.
 *
 * Ring indexing below is therefore ring_begin..halo_width, not 1..halo_width.
 * Every ordinary group has ring_begin == 1 and is bit-identical to before.
 */
#include "fv3_gpu_halo_exchange.hpp"
#include "fv3_gpu_errors.hpp"
#include "fv3_gpu_types.hpp"
#ifdef FV3_GPU_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace fv3::gpu::halo {

#ifdef FV3_GPU_ENABLE_CUDA
namespace {

FV3_HALO_HD inline int64_t view_offset(const HaloStripView &v, int64_t i, int64_t j,
                                       int64_t k) noexcept {
  return (i - v.lo_i) * v.st_i + (j - v.lo_j) * v.st_j + (k - v.lo_k) * v.st_k;
}

/* PACK. Entirely donor-local: no knowledge of any neighbour, no rotation, no
 * sign. Reads the `width` innermost owned lines along `edge` of the owner's own
 * array, in the owner's own index order, using the field's OWN staggering.
 *
 * This is why one pack serves every consumer of that edge. The receiver on a
 * rotated seam reads this same buffer as the partner component, which is exactly
 * right: source_shift(receiver_stagger, reversed) equals the packed field's own
 * staggering, so the strip it wants is the strip already sitting here. */
__global__ void pack_edge_kernel(HaloStripView src, fv_real *buf, int64_t along_extent,
                                 int64_t nk, int64_t nx, int64_t ny, int32_t ring_begin,
                                 int32_t width, TileEdge edge, StaggerShift stagger) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t t = blockIdx.y + 1;                       /* 1..along_extent */
  const int32_t h = ring_begin + static_cast<int32_t>(blockIdx.z); /* ring_begin..width */
  if (k >= nk || t > along_extent || h > width) return;
  int64_t i = 0, j = 0;
  halo_src_index(edge, t, h, nx, ny, stagger, i, j);
  buf[slot_offset(along_extent, nk, h - ring_begin + 1, t, k)] =
      src.data[view_offset(src, i, j, src.lo_k + k)];
}

/* UNPACK, and with it the whole seam transform (§6/§7).
 *
 * `sign` and `reversed` are computed on the host from the receiver's own edge and
 * its donor's edge, so this kernel embodies no connectivity policy of its own --
 * it applies a permutation and a sign it was handed. The destination indices come
 * from halo_dest_index in the receiver's frame with the receiver's own
 * staggering, so a staggered destination keeps its own extents and the write
 * never lands inside the owned block (asserted for every position, tile and edge
 * by T9 of tests/runtime/test_halo_topology.cpp). */
__global__ void unpack_edge_kernel(HaloStripView dst, const fv_real *buf,
                                   int64_t along_extent, int64_t donor_extent,
                                   int64_t nk, int64_t nx, int64_t ny, int32_t ring_begin,
                                   int32_t width, TileEdge edge, StaggerShift stagger,
                                   bool reversed, int8_t sign) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t t = blockIdx.y + 1;
  const int32_t h = ring_begin + static_cast<int32_t>(blockIdx.z);
  if (k >= nk || t > along_extent || h > width) return;
  const int64_t tp = seam_along_index(t, donor_extent, reversed);
  int64_t i = 0, j = 0;
  halo_dest_index(edge, t, h, nx, ny, stagger, i, j);
  const fv_real v = buf[slot_offset(donor_extent, nk, h - ring_begin + 1, tp, k)];
  dst.data[view_offset(dst, i, j, dst.lo_k + k)] =
      (sign < 0) ? static_cast<fv_real>(-v) : v;
}

inline dim3 strip_grid(int64_t nk, int64_t along_extent, int32_t width,
                       int &threads) noexcept {
  threads = (nk >= 64) ? 64 : 32;
  return dim3(static_cast<unsigned>((nk + threads - 1) / threads),
              static_cast<unsigned>(along_extent), static_cast<unsigned>(width));
}

/* Prompt 11 fusion: pack/unpack all four TileEdge values of one (group, field)
 * in a single launch, since their strips write/read fully disjoint slot
 * ranges (slot_base's `field*4 + edge` term) with no producer/consumer
 * dependency between edges of the same field within one resolve() call --
 * unlike the copy_corners dir=1/dir=2 pair characterized in Prompt 10, which
 * had a real intervening consumer. blockIdx.z is widened to
 * eidx*width + (h - ring_begin); everything else is the exact per-edge index
 * math above, just parameterized by a 4-entry table indexed by
 * TileEdge (west=0, east=1, south=2, north=3) instead of a host-side loop. A
 * skipped edge (E11's dest_edges mask) is represented by along_extent = 0,
 * which makes the existing `t > along_extent` guard reject every thread for
 * that edge -- identical to the per-field `continue` it replaces. */
inline dim3 strip_grid4(int64_t nk, int64_t max_along_extent, int32_t width,
                        int &threads) noexcept {
  threads = (nk >= 64) ? 64 : 32;
  return dim3(static_cast<unsigned>((nk + threads - 1) / threads),
              static_cast<unsigned>(max_along_extent), static_cast<unsigned>(width) * 4u);
}

__global__ void pack_edge4_kernel(HaloStripView src, fv_real *field_base,
                                  int64_t elems_per_edge, int64_t along0, int64_t along1,
                                  int64_t along2, int64_t along3, int64_t nk, int64_t nx,
                                  int64_t ny, int32_t ring_begin, int32_t ring_count,
                                  int32_t width, StaggerShift stagger) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t t = blockIdx.y + 1;
  const unsigned eidx = blockIdx.z / static_cast<unsigned>(ring_count);
  const int32_t h =
      ring_begin + static_cast<int32_t>(blockIdx.z % static_cast<unsigned>(ring_count));
  const int64_t along_extent =
      (eidx == 0) ? along0 : (eidx == 1) ? along1 : (eidx == 2) ? along2 : along3;
  if (k >= nk || t > along_extent || h > width) return;
  const TileEdge edge = static_cast<TileEdge>(eidx);
  int64_t i = 0, j = 0;
  halo_src_index(edge, t, h, nx, ny, stagger, i, j);
  field_base[static_cast<int64_t>(eidx) * elems_per_edge +
             slot_offset(along_extent, nk, h - ring_begin + 1, t, k)] =
      src.data[view_offset(src, i, j, src.lo_k + k)];
}

__global__ void unpack_edge4_kernel(HaloStripView dst, const fv_real *field_base,
                                    int64_t elems_per_edge, int64_t along0, int64_t along1,
                                    int64_t along2, int64_t along3, int64_t donor0,
                                    int64_t donor1, int64_t donor2, int64_t donor3, int64_t nk,
                                    int64_t nx, int64_t ny, int32_t ring_begin,
                                    int32_t ring_count, int32_t width, StaggerShift stagger,
                                    uint8_t reversed_mask, int8_t sign0, int8_t sign1,
                                    int8_t sign2, int8_t sign3) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t t = blockIdx.y + 1;
  const unsigned eidx = blockIdx.z / static_cast<unsigned>(ring_count);
  const int32_t h =
      ring_begin + static_cast<int32_t>(blockIdx.z % static_cast<unsigned>(ring_count));
  const int64_t along_extent =
      (eidx == 0) ? along0 : (eidx == 1) ? along1 : (eidx == 2) ? along2 : along3;
  if (k >= nk || t > along_extent || h > width) return;
  const int64_t donor_extent =
      (eidx == 0) ? donor0 : (eidx == 1) ? donor1 : (eidx == 2) ? donor2 : donor3;
  const int8_t sign = (eidx == 0) ? sign0 : (eidx == 1) ? sign1 : (eidx == 2) ? sign2 : sign3;
  const bool reversed = (reversed_mask >> eidx) & 1u;
  const TileEdge edge = static_cast<TileEdge>(eidx);
  const int64_t tp = seam_along_index(t, donor_extent, reversed);
  int64_t i = 0, j = 0;
  halo_dest_index(edge, t, h, nx, ny, stagger, i, j);
  const fv_real v = field_base[static_cast<int64_t>(eidx) * elems_per_edge +
                               slot_offset(donor_extent, nk, h - ring_begin + 1, tp, k)];
  dst.data[view_offset(dst, i, j, dst.lo_k + k)] =
      (sign < 0) ? static_cast<fv_real>(-v) : v;
}

/* field_slot() used to live here. It is now fv3_gpu_halo_exchange.hpp's, so that
 * a transport backend resolves a rotated seam's donor slot with the same function
 * launch_halo_transfer does rather than a second copy of the rule. */

} // namespace

/* Pack every (field, edge) of one group for ONE tile into that tile's send
 * buffer. Enqueued on `stream`; no synchronization of any kind. */
int launch_halo_pack(const HaloGroupSpec &spec, const HaloGeometry &g,
                     const HaloTileFields &f, fv_real *send, void *stream,
                     fv3_gpu_error *e, bool fuse_edges) noexcept {
  clear_error(e);
  if (!spec.live) return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                              "halo group is not live; nothing may be packed for it");
  if (!send) return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "null send buffer");
  const int32_t ring_count = halo_ring_count(spec);
  for (int n = 0; n < spec.field_count; ++n) {
    if (!f.present[n])
      return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                  "a field required by this halo group is not registered");
    const HaloFieldSpec &fs = spec.fields[n];
    const HaloStripView &v = f.view[n];
    if (fuse_edges) {
      int64_t along[4];
      int64_t max_along = 0;
      for (int q = 0; q < 4; ++q) {
        along[q] = edge_seam_extent(static_cast<TileEdge>(q), g.nx, g.ny, fs.stagger);
        if (along[q] > max_along) max_along = along[q];
      }
      int threads = 0;
      const dim3 grid = strip_grid4(v.nk, max_along, ring_count, threads);
      pack_edge4_kernel<<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
          v, send + slot_base(g, ring_count, n, TileEdge::west),
          group_slot_elements(g, ring_count), along[0], along[1], along[2], along[3], v.nk,
          g.nx, g.ny, spec.ring_begin, ring_count, spec.halo_width, fs.stagger);
    } else {
      for (int q = 0; q < 4; ++q) {
        const TileEdge edge = static_cast<TileEdge>(q);
        const int64_t along = edge_seam_extent(edge, g.nx, g.ny, fs.stagger);
        int threads = 0;
        const dim3 grid = strip_grid(v.nk, along, ring_count, threads);
        pack_edge_kernel<<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            v, send + slot_base(g, ring_count, n, edge), along, v.nk, g.nx, g.ny,
            spec.ring_begin, spec.halo_width, edge, fs.stagger);
      }
    }
  }
  const cudaError_t rc = cudaGetLastError();
  if (rc != cudaSuccess)
    return fail(e, FV3_GPU_CUDA_ERROR, __func__, cudaGetErrorString(rc), (int)rc);
  return FV3_GPU_SUCCESS;
}

/* Move each of my four edge slots from the DONOR's send buffer into my recv
 * buffer, device to device.
 *
 * `peer_send` is indexed by tile in 1..6 (entry 0 unused), which is the Model B
 * shape: one process holds every tile's buffers. An MPI build replaces exactly
 * this loop -- the buffers, the slot arithmetic, the pack and the unpack are
 * unchanged, which is the property §15 asks to be preserved.
 *
 * The copy is per (field, edge) rather than one bulk copy per edge because slots
 * are strided by field, and correctness comes first (§ "Performance Work Comes
 * Later"). */
int launch_halo_transfer(const HaloGroupSpec &spec, const HaloGeometry &g,
                         const TopologyView &topo, fv_real *const *peer_send,
                         fv_real *recv, void *stream, fv3_gpu_error *e) noexcept {
  clear_error(e);
  if (!spec.live) return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "halo group is not live");
  if (!peer_send || !recv)
    return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "null buffer table");
  const int64_t slot = group_slot_elements(g, halo_ring_count(spec));
  for (int q = 0; q < 4; ++q) {
    const TileEdge mine = static_cast<TileEdge>(q);
    const EdgeNeighbor &nb = topo.neighbors[q];
    if (nb.tile < 1 || nb.tile > cubed_sphere_tiles)
      return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                  "edge has no neighbour tile; a connected topology is required "
                  "before any halo group may be exchanged");
    const fv_real *src = peer_send[nb.tile];
    if (!src) return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                          "neighbour tile has no send buffer");
    /* The donor packed the strip under ITS edge, and under the staggering of
     * whichever of its fields the seam makes me read. On a rotated seam that is
     * my partner, which occupies the OTHER slot of this same group -- so the
     * source slot index differs from mine and must be resolved, not assumed. */
    const bool reversed = seam_is_reversed(mine, nb.edge);
    for (int n = 0; n < spec.field_count; ++n) {
      const HaloFieldSpec &fs = spec.fields[n];
      int sn = n;
      if (reversed && fs.role != VectorRole::scalar) {
        sn = field_slot(spec, fs.partner);
        if (sn < 0)
          return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                      "vector field's seam partner is not a member of its own halo group");
      }
      const cudaError_t rc = cudaMemcpyAsync(
          recv + slot_base(g, halo_ring_count(spec), n, mine),
          src + slot_base(g, halo_ring_count(spec), sn, nb.edge),
          static_cast<size_t>(slot) * sizeof(fv_real), cudaMemcpyDeviceToDevice,
          static_cast<cudaStream_t>(stream));
      if (rc != cudaSuccess)
        return fail(e, FV3_GPU_CUDA_ERROR, __func__, cudaGetErrorString(rc), (int)rc);
    }
  }
  return FV3_GPU_SUCCESS;
}

/* Write my halo from my recv buffer, applying the seam transform.
 *
 * Note that the recv slot is already indexed by MY field slot -- the transfer
 * did the partner remap -- so the component SELECTION has happened by the time
 * we get here and what remains is the reversal and the sign. Splitting it that
 * way keeps this kernel free of any group-structure knowledge. */
int launch_halo_unpack(const HaloGroupSpec &spec, const HaloGeometry &g,
                       const TopologyView &topo, const HaloTileFields &f,
                       const fv_real *recv, void *stream, fv3_gpu_error *e,
                       bool fuse_edges) noexcept {
  clear_error(e);
  if (!spec.live) return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "halo group is not live");
  if (!recv) return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "null recv buffer");
  const int32_t ring_count = halo_ring_count(spec);
  if (fuse_edges) {
    /* Per-edge quantities that were shared across fields in the edge-outer
     * loop below are here recomputed per field; they are cheap host-side
     * index arithmetic, not device work, so re-deriving them field-outer
     * costs nothing but a few extra scalar ops. */
    for (int n = 0; n < spec.field_count; ++n) {
      if (!f.present[n])
        return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                    "a field required by this halo group is not registered");
      const HaloFieldSpec &fs = spec.fields[n];
      const HaloStripView &v = f.view[n];
      int64_t along[4]{}, donor[4]{};
      int8_t sign[4] = {1, 1, 1, 1};
      uint8_t reversed_mask = 0;
      int64_t max_along = 0;
      for (int q = 0; q < 4; ++q) {
        const TileEdge mine = static_cast<TileEdge>(q);
        /* E11's destinations are owned cells, so a field may legally decline
         * an edge; see HaloFieldSpec::dest_edges. Ordinary groups pass 0xF.
         * along[q] = 0 makes the kernel's `t > along_extent` guard reject
         * every thread for this edge, reproducing the `continue` below. */
        if (!(fs.dest_edges & (1u << q))) continue;
        const EdgeNeighbor &nb = topo.neighbors[q];
        if (nb.tile < 1 || nb.tile > cubed_sphere_tiles)
          return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "edge has no neighbour tile");
        const bool reversed = seam_is_reversed(mine, nb.edge);
        const SeamRotation rot = seam_rotation(mine, nb.edge);
        const StaggerShift src = source_shift(fs.stagger, reversed);
        along[q] = edge_seam_extent(mine, g.nx, g.ny, fs.stagger);
        donor[q] = donor_seam_extent(nb.edge, g.nx, g.ny, src);
        if (along[q] != donor[q])
          return fail(e, FV3_GPU_INTERNAL_ERROR, __func__,
                      "seam extents disagree; index map and staggering are inconsistent");
        sign[q] = seam_component_sign(fs.role, rot);
        if (reversed) reversed_mask |= static_cast<uint8_t>(1u << q);
        if (along[q] > max_along) max_along = along[q];
      }
      int threads = 0;
      const dim3 grid = strip_grid4(v.nk, max_along, ring_count, threads);
      unpack_edge4_kernel<<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
          v, recv + slot_base(g, ring_count, n, TileEdge::west),
          group_slot_elements(g, ring_count), along[0], along[1], along[2], along[3], donor[0],
          donor[1], donor[2], donor[3], v.nk, g.nx, g.ny, spec.ring_begin, ring_count,
          spec.halo_width, fs.stagger, reversed_mask, sign[0], sign[1], sign[2], sign[3]);
    }
  } else {
    for (int q = 0; q < 4; ++q) {
      const TileEdge mine = static_cast<TileEdge>(q);
      const EdgeNeighbor &nb = topo.neighbors[q];
      if (nb.tile < 1 || nb.tile > cubed_sphere_tiles)
        return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__, "edge has no neighbour tile");
      const bool reversed = seam_is_reversed(mine, nb.edge);
      const SeamRotation rot = seam_rotation(mine, nb.edge);
      for (int n = 0; n < spec.field_count; ++n) {
        if (!f.present[n])
          return fail(e, FV3_GPU_INVALID_ARGUMENT, __func__,
                      "a field required by this halo group is not registered");
        const HaloFieldSpec &fs = spec.fields[n];
        /* E11's destinations are owned cells, so a field may legally decline an
         * edge; see HaloFieldSpec::dest_edges. Ordinary groups pass 0xF. */
        if (!(fs.dest_edges & (1u << q))) continue;
        const HaloStripView &v = f.view[n];
        const StaggerShift src = source_shift(fs.stagger, reversed);
        const int64_t along = edge_seam_extent(mine, g.nx, g.ny, fs.stagger);
        const int64_t donor = donor_seam_extent(nb.edge, g.nx, g.ny, src);
        /* T9 proves these are equal for every position, tile and edge. If they
         * ever are not, the reversal is undefined, so refuse rather than write
         * something plausible. */
        if (along != donor)
          return fail(e, FV3_GPU_INTERNAL_ERROR, __func__,
                      "seam extents disagree; index map and staggering are inconsistent");
        const int8_t sign = seam_component_sign(fs.role, rot);
        int threads = 0;
        const dim3 grid = strip_grid(v.nk, along, ring_count, threads);
        unpack_edge_kernel<<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            v, recv + slot_base(g, ring_count, n, mine), along, donor, v.nk, g.nx, g.ny,
            spec.ring_begin, spec.halo_width, mine, fs.stagger, reversed, sign);
      }
    }
  }
  const cudaError_t rc = cudaGetLastError();
  if (rc != cudaSuccess)
    return fail(e, FV3_GPU_CUDA_ERROR, __func__, cudaGetErrorString(rc), (int)rc);
  return FV3_GPU_SUCCESS;
}
#endif  // FV3_GPU_ENABLE_CUDA

} // namespace fv3::gpu::halo
