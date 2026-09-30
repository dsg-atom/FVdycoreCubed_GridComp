#pragma once
/* Milestone 3A.10 §5/§6/§7 -- cubed-sphere halo exchange plan.
 *
 * This header is the DECLARATIVE half of the exchange: what each halo group
 * contains, where each field's edge strip lives, and what transform a panel
 * seam applies to it. The kernels that act on it are in
 * runtime/fv3_gpu_halo_exchange.cu. Everything here is constexpr and host-side
 * testable, deliberately, because the transform table is the part that is easy
 * to get plausibly wrong and impossible to notice afterwards.
 *
 * The index algebra it builds on lives in fv3_gpu_halo_topology.hpp and is
 * validated against FMS's own cubic-grid halo oracle by T9 of
 * tests/runtime/test_halo_topology.cpp.
 *
 * ---------------------------------------------------------------------------
 * THE VECTOR SIGN RULE, AND WHY IT IS NOT KEYED ON STAGGERING
 * ---------------------------------------------------------------------------
 *
 * §6 warns: "Do not assume all vector seams use one transform," and "Where
 * authoritative behavior uses different transforms for C-grid and D-grid winds,
 * preserve the distinction." Both warnings are earned. Traced through FMS
 * 2024.03 (FV3 uses the GROUP update path, mpp_group_update.fh, not
 * mpp_update_domains):
 *
 *   1. The rotation tagged on an overlap is a property of the SEAM, not of the
 *      staggering. define_contact_point runs for CENTER only
 *      (mpp_domains_define.inc:1512); the EAST/NORTH/CORNER specs are derived
 *      from it by set_contact_point, and set_single_overlap copies rotation
 *      verbatim (:5262) while shifting only the index box. So the EAST spec and
 *      the NORTH spec of one seam carry the SAME rotation.
 *
 *   2. The sign is therefore selected by which ARGUMENT SLOT an array occupies,
 *      x or y -- never by its staggering. group_update_pack.inc:
 *          FIELD_X  at NINETY -> +fieldy   (:128)   at MINUS_NINETY -> -fieldy (:115)
 *          FIELD_Y  at NINETY -> -fieldx   (:206)   at MINUS_NINETY -> +fieldx (:181)
 *          FIELD_S  at either -> +field    (:63/:51)   (scalars: no sign, ever)
 *      There is no `gridtype` test anywhere in that file.
 *
 *   3. The slots are bound by the CALL, and the two grids bind them opposite
 *      ways round in staggering terms. mpp_group_update.fh:356-371 /
 *      mpp_domains_util.inc:1950-1974:
 *          CGRID_NE:  position_x = EAST,  position_y = NORTH
 *          DGRID_NE:  position_x = NORTH, position_y = EAST
 *      and fv_mp_mod.F90:880,930 pass (u_cmpt, v_cmpt) through unswapped, so
 *      dyn_core.F90:402/:1109 `(u, v, DGRID_NE)` puts the NORTH-shaped u in
 *      slot x, while :618 `(uc, vc, CGRID_NE)` puts the EAST-shaped uc there.
 *
 * Consequence, stated the two ways §6 asks for:
 *
 *   By argument slot      -- ONE transform. C-grid and D-grid are identical.
 *   By staggering         -- TWO transforms, transposed relative to each other.
 *                            At a NINETY seam C-grid gives EAST_halo=+NORTH_nbr
 *                            but D-grid gives NORTH_halo=+EAST_nbr.
 *
 * So this implementation keys the sign on VectorRole (the slot), which is the
 * universal form, and records each field's role explicitly in its group table
 * rather than deriving it from stagger. Deriving it from stagger with a single
 * table is precisely the bug §6 is warning about.
 *
 * The rotation used is the RECEIVER's, which is what makes a receive-side
 * transform legitimate: mpp_domains_define.inc:5459/:5486/:5513/:5540 set
 * rotateSend and rotateRecv in opposite pairs, so the rotation stamped on a send
 * overlap equals the rotation of the destination halo it feeds. Reading the pack
 * table by the receiver's own halo-edge rotation is therefore exact, not an
 * approximation.
 *
 * ---------------------------------------------------------------------------
 * STRUCTURAL DIFFERENCE FROM FMS (declare for §21)
 * ---------------------------------------------------------------------------
 *
 * FMS signs and transposes at PACK; unpack is a verbatim copy
 * (group_update_unpack.inc:41-64 has no sign and no index transform; its local
 * `rotation` is in the OMP private list at :22 and never read).
 *
 * This implementation is the mirror image: pack is a plain, donor-local copy of
 * an edge strip in the donor's own index order, and UNPACK applies the
 * along-seam reversal, the component selection and the sign. That is a
 * structural difference and must be reported as
 * "MATCHES WITH EXPLAINED STRUCTURAL DIFFERENCE", not as a match.
 *
 * It is numerically identical -- the composition of copy-then-transform equals
 * transform-then-copy for a permutation and a sign -- and it is chosen because
 * the donor cannot know a receiver's rotation without consulting the receiver's
 * connectivity, whereas the receiver knows both edges by construction. It also
 * makes every send buffer independent of who reads it, which is what lets one
 * pack serve a neighbour under Model B and, later, an MPI peer.
 */
#include "fv3_gpu_halo_topology.hpp"
#include "fv3_gpu_internal.hpp"
#include "fv3_gpu_types.hpp"
#include "fv3_gpu_communication.h"

namespace fv3::gpu::halo {

/* Which slot of an FMS vector update a field occupies. `scalar` is FIELD_S. */
enum class VectorRole : uint8_t { scalar, slot_x, slot_y };

/* The sign this field's value acquires when it crosses a rotated seam, read
 * straight off the group_update_pack.inc table above. Scalars never acquire a
 * sign (FIELD_S is `+field` at every rotation), which is why divgd may legally
 * ride a rotated seam at position=CORNER.
 *
 * Returns +1 on an unrotated seam for every role, since no component swap
 * happens there either. */
FV3_HALO_HD inline constexpr int8_t seam_component_sign(VectorRole role,
                                                        SeamRotation r) noexcept {
  if (role == VectorRole::scalar || r == SeamRotation::zero) return 1;
  const bool plus = (r == SeamRotation::plus_ninety);
  /* FIELD_X: +fieldy at NINETY, -fieldy at MINUS_NINETY.
   * FIELD_Y: -fieldx at NINETY, +fieldx at MINUS_NINETY. */
  if (role == VectorRole::slot_x) return plus ? +1 : -1;
  return plus ? -1 : +1;
}

/* One field inside a halo group.
 *
 * `partner` is the OTHER member of the vector pair, and is what this field is
 * read FROM on a rotated seam. It is FieldId::invalid for scalars. Note that the
 * partner is also exactly the field whose own stagger equals
 * source_shift(stagger, true) -- i.e. the transposed shift -- so the index
 * algebra and the slot algebra independently agree on which array is read,
 * and only the sign comes
 * from the slot. That agreement is asserted in the tests rather than assumed. */
struct HaloFieldSpec {
  FieldId id = FieldId::invalid;
  StaggerShift stagger{};
  VectorRole role = VectorRole::scalar;
  FieldId partner = FieldId::invalid;
  /* Bitmask over TileEdge of the edges this field is UNPACKED into; bit q is
   * 1 << q for TileEdge q. 0xF (all four) for every ordinary halo group, where
   * the four edge overlaps are disjoint halo cells and writing all of them is
   * the whole point.
   *
   * It exists for E11, whose destinations are OWNED cells, not halo cells: the
   * authoritative write-back (dyn_core.F90:1181-1188) touches u's NORTH line and
   * v's EAST line and nothing else. Writing the matching SOUTH/WEST ring-0 lines
   * as well would overwrite owned values the authoritative code never replaces.
   * Applied on the unpack side only -- pack and transfer stay edge-complete,
   * which costs a few unread slots and keeps the donor side free of any
   * knowledge of what its consumers intend to keep. */
  uint8_t dest_edges = 0xF;
};

/* Largest field count in any group below; sized for FV3_GPU_H_DSCALAR's four. */
inline constexpr int max_halo_group_fields = 4;

struct HaloGroupSpec {
  uint32_t group = 0;                 /* fv3_gpu_halo_group */
  int32_t halo_width = 0;             /* OUTERMOST ring exchanged, in cells */
  int32_t field_count = 0;
  /* INNERMOST ring exchanged. 1 for every ordinary halo group -- ring 1 is the
   * first cell outside the owned block. 0 only for E11, whose "ring" IS the
   * outermost owned staggered line (see halo_dest_index at h = 0). Rings
   * ring_begin..halo_width are exchanged, so the count is halo_ring_count(). */
  int32_t ring_begin = 1;
  HaloFieldSpec fields[max_halo_group_fields]{};
  bool live = false;                  /* false => nothing to exchange (see notes) */
};

/* Number of rings a group exchanges. Buffer slots are sized and addressed by
 * this, never by halo_width directly, so a group whose innermost ring is 0
 * still gets a correctly sized slot. */
FV3_HALO_HD inline constexpr int32_t halo_ring_count(const HaloGroupSpec &s) noexcept {
  return s.halo_width - s.ring_begin + 1;
}

/* 1-based ring ordinal of ring `h` within its group's slot, which is what
 * slot_offset's `h` argument means. Identity for an ordinary group. */
FV3_HALO_HD inline constexpr int32_t halo_ring_ordinal(const HaloGroupSpec &s,
                                                       int32_t h) noexcept {
  return h - s.ring_begin + 1;
}

/* Index of a group field by FieldId; -1 if the group does not contain it.
 * Used to resolve a rotated seam's partner to a slot in the SAME group, which is
 * the only place a partner can legally live.
 *
 * Milestone 3A.15: promoted out of runtime/fv3_gpu_halo_exchange.cu's anonymous
 * namespace so that a transport backend (the MPI one in particular) resolves the
 * donor's slot with the SAME function launch_halo_transfer uses. A second copy of
 * this remap living in a transport would be a silent divergence: it would put the
 * wrong wind component on a rotated seam and still deliver a plausible number. */
FV3_HALO_HD inline constexpr int field_slot(const HaloGroupSpec &s, FieldId id) noexcept {
  for (int n = 0; n < s.field_count; ++n)
    if (s.fields[n].id == id) return n;
  return -1;
}

/* ---------------------------------------------------------------------------
 * GROUP FIELD TABLES
 *
 * Derived in §1 from GFDL_atmos_cubed_sphere ca30e26 (tag geos/v3.0.0)
 * model/dyn_core.F90. Staggerings from model/fv_arrays.F90 ~1462-1496.
 *
 * Halo widths are the widths the CONSUMER reads, not fv_mp_mod.F90:108's
 * compile-time ng=3 domain allocation, and not the registered halo_width.
 * Where a consumer's reach was not provable from source the authoritative
 * ng = 3 is used, because under-exchanging is a silent wrong answer while
 * over-exchanging is only wasted bandwidth.
 * ------------------------------------------------------------------------- */
FV3_HALO_HD inline constexpr HaloGroupSpec halo_group_spec(uint32_t g) noexcept {
  HaloGroupSpec s{};
  s.group = g;
  switch (g) {
    /* dyn_core.F90:402 and :1109 -- start_group_halo_update(i_pack(8), u, v,
     * gridtype=DGRID_NE). u is NORTH-shaped, v is EAST-shaped, and u is the
     * FIRST argument, hence slot_x. :402 is the entry exchange, :1109 the one
     * that prepares the next substep; same group, same fields. */
    case FV3_GPU_H_DWIND_IN:
    case FV3_GPU_H_DWIND_NEXT:
      s.halo_width = fv3_domain_halo_width;
      s.field_count = 2;
      s.fields[0] = {FieldId::u, stagger_north, VectorRole::slot_x, FieldId::v};
      s.fields[1] = {FieldId::v, stagger_east,  VectorRole::slot_y, FieldId::u};
      s.live = true;
      break;

    /* dyn_core.F90:618 -- start_group_halo_update(i_pack(9), uc, vc,
     * gridtype=CGRID_NE). uc is EAST-shaped and is the first argument, so here
     * the EAST-shaped member is slot_x. This is the transposition that makes the
     * staggering-keyed formulation wrong. */
    case FV3_GPU_H_CWIND:
      s.halo_width = fv3_domain_halo_width;
      s.field_count = 2;
      s.fields[0] = {FieldId::uc, stagger_east,  VectorRole::slot_x, FieldId::vc};
      s.fields[1] = {FieldId::vc, stagger_north, VectorRole::slot_y, FieldId::uc};
      s.live = true;
      break;

    /* dyn_core.F90:427 -- w, scalar, default position (CENTER). */
    case FV3_GPU_H_W_IN:
      s.halo_width = fv3_domain_halo_width;
      s.field_count = 1;
      s.fields[0] = {FieldId::w, stagger_center, VectorRole::scalar, FieldId::invalid};
      s.live = true;
      break;

    /* dyn_core.F90:459 -- gz, scalar CENTER, inside the `if (it == 1)` guard.
     * §11 re-audits the readiness lifecycle of this group; the FIELD content is
     * not in doubt. */
    case FV3_GPU_H_GZ_FIRST:
      s.halo_width = fv3_domain_halo_width;
      s.field_count = 1;
      s.fields[0] = {FieldId::gz, stagger_center, VectorRole::scalar, FieldId::invalid};
      s.live = true;
      break;

    /* dyn_core.F90:528 -- divgd, scalar, position=CORNER, guarded by nord > 0.
     *
     * CORNER on a rotated seam is legal: mpp_group_update.fh:89-95 FATALs only
     * for position EAST or NORTH under rotated_ninety, and a corner scalar takes
     * the FIELD_S branches, which transpose the traversal without any sign. */
    case FV3_GPU_H_DIVGD:
      s.halo_width = fv3_domain_halo_width;
      s.field_count = 1;
      s.fields[0] = {FieldId::divgd, stagger_corner, VectorRole::scalar, FieldId::invalid};
      s.live = true;
      break;

    /* dyn_core.F90:844-847 -- q_con, delp, pt, scalars at CENTER.
     *
     * cappa rides the same group only on the it == 1 pass (MOIST_CAPPA). It is
     * listed here because under-exchanging is the dangerous direction and cappa
     * is a CENTER scalar exchanged over the same seams; the per-substep pass
     * simply refreshes one field more than the Fortran does, which changes no
     * arithmetic. Recorded as a structural difference for §21. */
    case FV3_GPU_H_DSCALAR:
      s.halo_width = fv3_domain_halo_width;
      s.field_count = 4;
      s.fields[0] = {FieldId::q_con, stagger_center, VectorRole::scalar, FieldId::invalid};
      s.fields[1] = {FieldId::delp,  stagger_center, VectorRole::scalar, FieldId::invalid};
      s.fields[2] = {FieldId::pt,    stagger_center, VectorRole::scalar, FieldId::invalid};
      s.fields[3] = {FieldId::cappa, stagger_center, VectorRole::scalar, FieldId::invalid};
      s.live = true;
      break;

    /* dyn_core.F90:931/:934 -- zh, scalar CENTER. */
    case FV3_GPU_H_ZH:
      s.halo_width = fv3_domain_halo_width;
      s.field_count = 1;
      s.fields[0] = {FieldId::zh, stagger_center, VectorRole::scalar, FieldId::invalid};
      s.live = true;
      break;

    /* dyn_core.F90:932/:935 -- pkc, scalar CENTER.
     *
     * The enum name says PKC2 and the width is 3, not 2. The name is a misnomer
     * inherited from the WP-E scaffolding; implementing 2 rings because of it
     * would silently under-exchange. Width comes from the authoritative domain
     * halo, as for every other CENTER scalar here. */
    case FV3_GPU_H_PKC2:
      s.halo_width = fv3_domain_halo_width;
      s.field_count = 1;
      s.fields[0] = {FieldId::pkc, stagger_center, VectorRole::scalar, FieldId::invalid};
      s.live = true;
      break;

    /* dyn_core.F90 heat_source, scalar CENTER, live only when d_con > 0. The
     * guard is a dispatch-time condition, not a table property. */
    case FV3_GPU_H_HEAT:
      s.halo_width = fv3_domain_halo_width;
      s.field_count = 1;
      s.fields[0] = {FieldId::heat_source, stagger_center, VectorRole::scalar, FieldId::invalid};
      s.live = true;
      break;

    /* E11 -- dyn_core.F90:1176-1189.
     *
     *   if( it==n_split .and. gridstruct%grid_type<4 .and. .not.gridstruct%nested)
     *      call mpp_get_boundary(u, v, domain, ebuffery=ebuffer,
     *                            nbufferx=nbuffer, gridtype=DGRID_NE)
     *      u(is:ie, je+1, k) = nbuffer(i-is+1,k)
     *      v(ie+1, js:je, k) = ebuffer(j-js+1,k)
     *
     * This is NOT a halo exchange and must not be described as one. For a
     * staggered field the outermost owned line is shared with the neighbour and
     * BOTH tiles compute it independently; the two results differ at round-off.
     * FV3 replaces one copy with the other so the discrepancy cannot accumulate
     * over substeps (":1177 Prevent accumulation of rounding errors at
     * overlapped domain edges"). Destination cells are OWNED, not halo.
     *
     * The existing seam algebra represents it exactly at ring h = 0:
     *   halo_dest_index(north, t, 0, nx, ny, stagger_north) == (t, ny+1)   == u(is:ie, je+1)
     *   halo_dest_index(east,  t, 0, nx, ny, stagger_east)  == (nx+1, t)   == v(ie+1, js:je)
     * with edge_seam_extent giving nx and ny respectively -- exactly the
     * authoritative i and j ranges, and no corner cell in either. The donor side
     * needs no special case either: an aligned seam reads the neighbour's own
     * shared line (src.j = 1 / src.i = 1) and a rotated seam reads the partner
     * component through the standard source_shift/seam_component_sign pair.
     *
     * VERIFIED, not inferred: tools/savepoints/probe_e11_map.py predicts every
     * tile's post-E11 sp2c line from the donor tile's pre-E11 sp2b image using
     * this map and reproduces the authoritative Fortran result with residual
     * 0.000e+00 on all 6 tiles, both fields and all 10 levels, against a genuine
     * change of 3.9e-14..3.9e-13. u/v only, and DGRID_NE only: uc/vc are never
     * reconciled this way, so this group must not be reused for them.
     *
     * Field order matches i_pack(8): u first, hence slot_x, as in H_DWIND_IN. */
    case FV3_GPU_H_DWIND_SEAM:
      s.ring_begin = 0;
      s.halo_width = 0;               /* the single ring h = 0 */
      s.field_count = 2;
      s.fields[0] = {FieldId::u, stagger_north, VectorRole::slot_x, FieldId::v,
                     static_cast<uint8_t>(1u << static_cast<int>(TileEdge::north))};
      s.fields[1] = {FieldId::v, stagger_east,  VectorRole::slot_y, FieldId::u,
                     static_cast<uint8_t>(1u << static_cast<int>(TileEdge::east))};
      s.live = true;
      break;

    /* dyn_core.F90:1201/:419 exchange q, but the supported production first-run
     * path runs tracer advection OUTSIDE the acoustic loop, so no stage in this
     * runtime consumes a q halo. Declared NOT LIVE rather than given an empty
     * field list, so that a future consumer trips the `live` check instead of
     * silently exchanging nothing. */
    case FV3_GPU_H_Q_INLINE:
    default:
      s.live = false;
      break;
  }
  return s;
}

/* --------------------------------------------------------------------------
 * BUFFER GEOMETRY
 *
 * One send buffer and one receive buffer per (group, field, edge). Layout is
 * [h][along][k], h slowest, k fastest, so that the k-contiguous inner run is
 * the one the unpack kernel walks -- the destination is k-strided too, and
 * matching them keeps both sides coalesced.
 *
 * `along` runs 1..edge_seam_extent(edge, nx, ny, stagger) in the OWNER's own
 * frame. A rotated seam makes the donor's extent differ from the receiver's in
 * general (nx+1 vs ny for a staggered field), which is why the receiver sizes
 * its read by donor_seam_extent and not by its own.
 *
 * Buffers are sized for the LARGEST extent over the four edges and both
 * staggering components, so one allocation serves every edge of a field and the
 * transfer is a fixed-stride copy. The waste is at most one along-line per edge.
 * ------------------------------------------------------------------------ */
struct HaloGeometry {
  int64_t nx = 0, ny = 0;               /* OWNED cells: npx-1, npy-1 */
  int64_t npz = 0;
};

/* A field's device storage reduced to what the seam kernels need. Deliberately a
 * small POD passed by value into a kernel; chasing DeviceViewBase's std::array
 * members on the device would drag the registry contract into device code for no
 * benefit. `nk` is the field's own vertical extent, because interface-level
 * fields (gz, zh, pkc) carry npz+1 while cell-centred ones carry npz. */
struct HaloStripView {
  fv_real *data = nullptr;
  int64_t lo_i = 0, lo_j = 0, lo_k = 0;
  int64_t st_i = 0, st_j = 0, st_k = 0;
  int64_t nk = 0;
};

/* One tile's views for one group, indexed as that group's field list. */
struct HaloTileFields {
  HaloStripView view[max_halo_group_fields]{};
  uint8_t present[max_halo_group_fields]{};
};

/* Uniform per-(field, edge) slot, so a slot address is arithmetic and the
 * transfer is one fixed-size copy. Sized for the longest along-extent any edge
 * and staggering can produce and for interface-level fields (npz+1).
 *
 * The unused tail of a slot is never read, because both sides derive the SAME
 * along-extent from the donor edge and the SOURCE field's staggering -- not from
 * this bound. One uniform slot rather than a per-field packed layout is the
 * single source of truth for buffer addressing; correctness before density.
 *
 * It IS copied, though. launch_halo_transfer moves a whole slot so that a slot
 * address stays pure arithmetic, so the tail crosses the wire even though nobody
 * reads it. That made initcheck report 24587 uninitialized cudaMemcpy sources
 * before the buffers were zeroed once at attach (milestone 3A.10 §20; see the
 * cudaMemset in attach_halo_peer_group). Never-read is not the same as
 * never-touched, and initcheck is right to distinguish them. */
FV3_HALO_HD inline constexpr int64_t group_slot_elements(const HaloGeometry &g,
                                                         int32_t width) noexcept {
  const int64_t along = ((g.nx > g.ny) ? g.nx : g.ny) + 1;
  return static_cast<int64_t>(width) * along * (g.npz + 1);
}

FV3_HALO_HD inline constexpr int64_t slot_base(const HaloGeometry &g, int32_t width,
                                               int32_t field, TileEdge e) noexcept {
  return (static_cast<int64_t>(field) * 4 + static_cast<int64_t>(e)) *
         group_slot_elements(g, width);
}

/* Elements one tile needs for ONE group's send (or recv) buffer. */
FV3_HALO_HD inline constexpr int64_t group_buffer_elements(const HaloGroupSpec &s,
                                                           const HaloGeometry &g) noexcept {
  return static_cast<int64_t>(s.field_count) * 4 *
         group_slot_elements(g, halo_ring_count(s));
}

/* Offset of one value inside a slot. `along_extent` is the extent of the frame
 * the value was PACKED in, which the reader recomputes rather than being told. */
FV3_HALO_HD inline constexpr int64_t slot_offset(int64_t along_extent, int64_t nk,
                                                 int32_t h, int64_t along,
                                                 int64_t k) noexcept {
  return ((static_cast<int64_t>(h - 1) * along_extent) + (along - 1)) * nk + k;
}

/* Implemented in runtime/fv3_gpu_halo_exchange.cu; declared unconditionally so a
 * CPU-only translation unit still sees the contract. */
int launch_halo_pack(const HaloGroupSpec &, const HaloGeometry &, const HaloTileFields &,
                     fv_real *send, void *stream, fv3_gpu_error *,
                     bool fuse_edges = false) noexcept;
/* Prompt 112: peer_send is indexed by EdgeNeighbor.rank (transport peer
 * identity, 0..peer_count-1), not by EdgeNeighbor.tile (physical cube face,
 * 1..6) -- see fv3_gpu_halo_subtile_view.hpp's field-mapping comment for the
 * .tile-vs-.rank contract. peer_count is the caller-owned size of the
 * peer_send array; production callers pass cubed_sphere_tiles (6) exactly as
 * before, so production behavior is unchanged, but the primitive itself can
 * now index a larger, test-only peer table. */
int launch_halo_transfer(const HaloGroupSpec &, const HaloGeometry &, const TopologyView &,
                         fv_real *const *peer_send_by_rank, int32_t peer_count, fv_real *recv,
                         void *stream, fv3_gpu_error *) noexcept;
int launch_halo_unpack(const HaloGroupSpec &, const HaloGeometry &, const TopologyView &,
                       const HaloTileFields &, const fv_real *recv, void *stream,
                       fv3_gpu_error *, bool fuse_edges = false) noexcept;

}  // namespace fv3::gpu::halo
