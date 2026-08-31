#pragma once
//===========================================================================
// WP-E: cubed-sphere halo connectivity and panel-seam index/vector transforms.
//
// Authoritative sources of truth for everything in this header:
//
//   1. Mosaic connectivity (which tile edge touches which tile edge, and in
//      which index direction) -- the 12-contact cubed-sphere specification,
//      read from TWO independent sources that agree contact for contact:
//        FVdycoreCubed_GridComp/CubeHalo.F90:213-261
//        GFDL_atmos_cubed_sphere (geos/v3.0.0, ca30e26)
//          tools/fv_mp_mod.F90:577-625   domain_decomp, case ( 6 )
//      Both feed mpp_define_mosaic with symmetry=.true. and
//      shalo=nhalo=whalo=ehalo=ng, ng being the compile-time PARAMETER 3 at
//      tools/fv_mp_mod.F90:108.
//
//   2. Rotation at a panel seam -- GFDL FMS 2024.03 (the FMS that GEOSgcm
//      actually links, pinned via FMS_DIR into ESMA-Baselibs-8.32.0):
//        mpp/include/mpp_domains_define.inc:5438-5547   rotateSend/rotateRecv
//        mpp/include/mpp_do_updateV.fh:477-647          component sign table
//
// This header is deliberately PURE: no CUDA calls, no allocation, no runtime
// state. Every function is a total function of its arguments so that the whole
// seam algebra is unit-testable on the host, without a GPU, and so that the
// same code produces the indices on the device.
//===========================================================================

#include "fv3_gpu_internal.hpp"
#include <cstdint>

#if defined(__CUDACC__)
#define FV3_HALO_HD __host__ __device__
#else
#define FV3_HALO_HD
#endif

namespace fv3::gpu::halo {

/* Number of tiles on a global cubed sphere (CubeHalo.F90:102 ntiles = 6,
 * fv_mp_mod.F90:578 ntiles = 6). */
inline constexpr int cubed_sphere_tiles = 6;

/* Number of mosaic contact lines (CubeHalo.F90:103 num_contact = 12). */
inline constexpr int cubed_sphere_contacts = 12;

/* FV3's halo width. This is the compile-time PARAMETER at
 * tools/fv_mp_mod.F90:108, handed to mpp_define_mosaic as all four of
 * shalo/nhalo/whalo/ehalo (fv_mp_mod.F90:641-644). Every dyn_core exchange that
 * passes no explicit halo arguments therefore exchanges 3 rings.
 *
 * Note this is NOT the same quantity as CubeHalo.F90:315's local `ng = 1`, which
 * is that harness's own halo request, nor the same as a field's registered
 * halo_width. It is the authoritative FV3 domain halo. */
inline constexpr int fv3_domain_halo_width = 3;

/* Rotation applied to a vector pair when crossing a panel seam.
 *
 * These are FMS's ZERO / NINETY / MINUS_NINETY (mpp_parameter.F90:120).
 * ONE_HUNDRED_EIGHTY is deliberately absent: it is reachable only through a
 * tripolar fold, and set_single_overlap FATALs on any rotation that is not one
 * of these three (mpp_domains_define.inc:5283-5284). A cubed sphere never
 * produces it. */
enum class SeamRotation : int8_t { zero = 0, plus_ninety = 1, minus_ninety = -1 };

/* ---------------------------------------------------------------------------
 * Edge axis.
 *
 * WEST and EAST are the x-normal edges (their along-seam coordinate is j);
 * SOUTH and NORTH are the y-normal edges (along-seam coordinate is i).
 * ------------------------------------------------------------------------- */
FV3_HALO_HD inline constexpr bool edge_is_x_normal(TileEdge e) noexcept {
  return e == TileEdge::west || e == TileEdge::east;
}

/* ---------------------------------------------------------------------------
 * ORIENTATION IS DERIVABLE, and is derived here rather than trusted.
 *
 * Over all 12 cubed-sphere contacts the following two statements are
 * equivalent, and each holds without exception:
 *
 *   (a) the seam transposes (a row of one tile abuts a column of the other);
 *   (b) the along-seam index order reverses.
 *
 * There is no aligned-but-reversed contact and no transposed-but-forward
 * contact on this topology. Consequently a single bit suffices, and it is a
 * function of the two edge identities alone: the seam is ALIGNED exactly when
 * my edge and the donor edge lie on the SAME axis.
 *
 * Worked against the mosaic table:
 *   contact 1  T1 EAST  <-> T2 WEST   both x-normal -> aligned, j:1->ny both
 *   contact 4  T1 SOUTH <-> T6 NORTH  both y-normal -> aligned, i:1->nx both
 *   contact 2  T1 NORTH <-> T3 WEST   y vs x        -> reversed, i:1->nx / j:ny->1
 *   contact 3  T1 WEST  <-> T5 NORTH  x vs y        -> reversed, j:1->ny / i:nx->1
 * ------------------------------------------------------------------------- */
FV3_HALO_HD inline constexpr bool seam_is_reversed(TileEdge mine, TileEdge donor) noexcept {
  return edge_is_x_normal(mine) != edge_is_x_normal(donor);
}

FV3_HALO_HD inline constexpr EdgeOrientation seam_orientation(TileEdge mine,
                                                    TileEdge donor) noexcept {
  /* EdgeOrientation::swap_xy and ::swap_xy_reversed exist in the contract for
   * topologies this project does not yet support; on grid_type = 0 they are
   * unreachable, because transposition and reversal are the same bit (above).
   * Reporting `reversed` for a transposed seam is therefore exact, not a
   * rounding: the transposition is already implied by the donor edge's axis,
   * which travels alongside in EdgeNeighbor::edge. */
  return seam_is_reversed(mine, donor) ? EdgeOrientation::reversed
                                       : EdgeOrientation::aligned;
}

/* ---------------------------------------------------------------------------
 * Vector rotation at a seam.
 *
 * FMS rotateRecv, mpp_domains_define.inc (defaults ZERO at :5438):
 *
 *   my edge   donor edge   rotateRecv        line
 *   EAST      WEST         ZERO              5451
 *   EAST      SOUTH        NINETY            5459
 *   SOUTH     NORTH        ZERO              5478
 *   SOUTH     EAST         MINUS_NINETY      5486
 *   WEST      EAST         ZERO              5505
 *   WEST      NORTH        NINETY            5513
 *   NORTH     SOUTH        ZERO              5532
 *   NORTH     WEST         MINUS_NINETY      5540
 *
 * Reading the four nonzero rows: on a rotated seam the rotation is fixed by MY
 * edge alone -- +90 on the x-normal edges (WEST, EAST), -90 on the y-normal
 * edges (SOUTH, NORTH). The donor edge only decides WHETHER the seam rotates.
 *
 * This reproduces the per-tile table independently derived from the mosaic:
 * odd tiles (1,3,5) rotate on WEST (+90) and NORTH (-90); even tiles (2,4,6)
 * rotate on EAST (+90) and SOUTH (-90).
 * ------------------------------------------------------------------------- */
FV3_HALO_HD inline constexpr SeamRotation seam_rotation(TileEdge mine,
                                              TileEdge donor) noexcept {
  if (!seam_is_reversed(mine, donor)) return SeamRotation::zero;
  return edge_is_x_normal(mine) ? SeamRotation::plus_ninety
                                : SeamRotation::minus_ninety;
}

/* ---------------------------------------------------------------------------
 * Component sign table, mpp_do_updateV.fh (CGRID branch; signs are applied by
 * the SENDER during pack, and the rotation used there is rotateSend, which for
 * a given pairing equals the receiver's rotateRecv -- e.g. donor NORTH -> my
 * WEST has rotateSend = NINETY at :5540's mirror row :5513's rotateRecv):
 *
 *   rotation        x-component        y-component
 *   ZERO            +fieldx  (:477)    +fieldy  (:579)
 *   MINUS_NINETY    -fieldy  (:504)    +fieldx  (:592)
 *   NINETY          +fieldy  (:518)    -fieldx  (:619)
 *
 * i.e. exactly a rotation matrix:
 *   +90 : (u,v) <- (+v_src, -u_src)
 *   -90 : (u,v) <- (-v_src, +u_src)
 *
 * SCALAR_PAIR (BTEST(update_flags, SCALAR_BIT)) selects the branches at :491,
 * :606 that omit the sign flip. That is how the metric terms ride the staggered
 * machinery without acquiring a vector sign: divg_v/divg_u
 * (fv_grid_utils.F90:821-822), del6_v/del6_u (:823-824), dy/dx and dxc/dyc
 * (fv_grid_tools.F90:816,824-825,1001-1002). Nothing in dyn_core.F90 passes
 * SCALAR_PAIR -- grep for it there returns nothing -- so no dyn_core exchange
 * takes the scalar-pair branch, and `scalar_pair` below stays false for every
 * group in the supported path. It is a parameter rather than a hard-coded
 * `false` because the grid-metric exchanges will need it when they move onto
 * the device.
 *
 * BOTH branches are corroborated independently by FMS's own halo oracle,
 * test_fms/mpp/test_mpp_domains.F90, which precomputes the expected halo with
 * explicit signs and then checksums mpp_update_domains against it:
 *
 *   :1552-1553   vector CGRID_NE on a cubic grid
 *                  x-component (ioff,joff)=(1,0), (sign1,sign2)=(+1,-1)
 *                  y-component (ioff,joff)=(0,1), (sign1,sign2)=(-1,+1)
 *                sign1 is applied on the odd tiles' WEST and the even tiles'
 *                EAST halo, sign2 on the odd tiles' NORTH and the even tiles'
 *                SOUTH -- i.e. exactly the four rotated seams. Substituting
 *                gives u <- +v, v <- -u on a WEST/EAST seam (+90) and
 *                u <- -v, v <- +u on a NORTH/SOUTH seam (-90), reproducing the
 *                rotateRecv reading above from a completely different direction.
 *
 *   :1272-1277,  BGRID_NE at CORNER on a cubic grid is tested as
 *   :1367-1368     update_flags = SCALAR_PAIR, (sign1,sign2)=(+1,+1) for both
 *                components -- the pure swap, confirming the `scalar_pair`
 *                parameter is a sign suppression and nothing else.
 *
 * Returns the coefficients of the destination pair in terms of the source pair:
 *   u_dst = cxx*u_src + cxy*v_src
 *   v_dst = cyx*u_src + cyy*v_src
 * ------------------------------------------------------------------------- */
struct SeamVectorTransform {
  int8_t cxx = 1, cxy = 0;
  int8_t cyx = 0, cyy = 1;
  FV3_HALO_HD inline constexpr bool swaps_components() const noexcept {
    return cxy != 0 || cyx != 0;
  }
};

FV3_HALO_HD inline constexpr SeamVectorTransform
seam_vector_transform(SeamRotation r, bool scalar_pair = false) noexcept {
  const int8_t s = scalar_pair ? int8_t(1) : int8_t(-1);
  switch (r) {
    case SeamRotation::plus_ninety:  return SeamVectorTransform{0, 1, s, 0};
    case SeamRotation::minus_ninety: return SeamVectorTransform{0, s, 1, 0};
    case SeamRotation::zero:
    default:                         return SeamVectorTransform{1, 0, 0, 1};
  }
}

/* ---------------------------------------------------------------------------
 * Destination / source index algebra, CELL-CENTERED (FMS position = CENTER).
 *
 * Local edge coordinates. `h` is depth into the halo, h = 1 being the ring
 * nearest my own interior; `t` is the along-seam coordinate. Owned cell centers
 * are i = 1..nx, j = 1..ny.
 *
 *   my WEST  halo point = (1-h,    t),  t = j
 *   my EAST  halo point = (nx+h,   t),  t = j
 *   my SOUTH halo point = (t,    1-h),  t = i
 *   my NORTH halo point = (t,   ny+h),  t = i
 *
 * The donor's source point, on whichever of ITS edges the contact names, at
 * along-seam t' and the same depth h (h' = h; the ring nearest the donor's
 * contacted edge feeds the ring nearest mine):
 *
 *   donor WEST  edge = (h,        t')
 *   donor EAST  edge = (nx+1-h,   t')
 *   donor SOUTH edge = (t',       h)
 *   donor NORTH edge = (t',  ny+1-h)
 *
 * with t' = t on an aligned seam and t' = n+1-t on a reversed one, n being the
 * along-seam extent.
 *
 * Verified against four contacts spanning both orientations and all four of my
 * edges:
 *   contact 1 (T1 EAST  <-> T2 WEST ):  T1(nx+h, j) = T2(h,        j)
 *   contact 4 (T1 SOUTH <-> T6 NORTH):  T1(i,  1-h) = T6(i,   ny+1-h)
 *   contact 2 (T1 NORTH <-> T3 WEST ):  T1(i, ny+h) = T3(h,   ny+1-i)
 *   contact 3 (T1 WEST  <-> T5 NORTH):  T1(1-h,  j) = T5(nx+1-j, ny+1-h)
 *
 * STAGGERED POSITIONS (EAST/NORTH/CORNER, i.e. C-grid uc/vc, D-grid u/v, divgd)
 * are the general case of the above, established in the STAGGERING section that
 * follows. These two _center functions are wrappers on the general ones with
 * ishift = jshift = 0.
 * ------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------------
 * STAGGERING. The generalisation of the above to FMS's four positions.
 *
 * FMS builds CENTER first and then derives the other three by adding integer
 * shifts (mpp_domains_define.inc:1512-1517 calls set_contact_point for CORNER,
 * EAST and NORTH; define_contact_point FATALs for anything but CENTER at
 * :5332-5333). The shifts come from mpp_domains_util.inc:811-832 and are applied
 * ONLY when domain%symmetry, which FV3's cubed sphere sets:
 *
 *   position   ishift  jshift    owned extent        FV3 fields
 *   CENTER       0       0       1..nx , 1..ny       delp pt w q gz zh pkc divg
 *   EAST         1       0       1..nx+1, 1..ny      uc, and v (D-grid)
 *   NORTH        0       1       1..nx , 1..ny+1     vc, and u (D-grid)
 *   CORNER       1       1       1..nx+1, 1..ny+1    divgd
 *
 * `u` is NORTH-shaped and `v` is EAST-shaped (model/fv_arrays.F90:1462-1496);
 * DGRID_NE sets exchange_uv and then updates as CGRID_NE with the two arrays
 * passed in swapped order (mpp_update_domains2D.fh:483-490, 581-586), which is
 * why the same index algebra serves both wind staggerings.
 *
 * THE SHIFT USED ON THE DONOR SIDE IS THE CROSS SHIFT ON A ROTATED SEAM.
 *
 * On an aligned seam the source is the same component array, so it carries my
 * own (I,J). On a rotated seam the source is the OTHER component array, whose
 * shifts are transposed, so the source carries (J,I). This is not a guess: it is
 * read off FMS's own halo oracle, test_fms/mpp/test_mpp_domains.F90:4092-4126
 * (`fill_cubic_grid_halo`), whose eight branches are reproduced exactly by the
 * functions below. Two of them, for the odd tiles:
 *
 *   :4119  halo_data(nx+1+ioff : nx+ehalo+ioff, 1:ny+joff, :)
 *              = data1_all(1+ioff : ehalo+ioff, 1:ny+joff, :, le)
 *          my EAST halo <- donor WEST, ALIGNED: i' = ioff + h, my own ishift.
 *
 *   :4123  halo_data(1:nx+ioff, ny+i+joff, :)
 *              = sign2 * data2_all(i+joff, ny+ioff:1:-1, :, ln)
 *          my NORTH halo <- donor WEST, ROTATED: i' = JOFF + h, the cross shift,
 *          reading the other component, and reversing over n = ny + ioff.
 *
 * Note also that the DEPTH formula on the donor's EAST and NORTH edges carries no
 * shift at all (nx+1-h, ny+1-h), because in the symmetric convention the shared
 * seam line already sits at index nx+1 / ny+1 there, whereas on the donor's WEST
 * and SOUTH edges it sits at index 1 and the shift is needed to step past it.
 *
 * THE SHARED SEAM LINE IS NOT EXCHANGED BY A HALO GROUP. A staggered destination
 * strip at h >= 1 starts strictly outside my own outermost owned line --
 * (nx+I)+h, not nx+I -- so both tiles keep their own copy of the physically
 * identical line and neither overwrites the other. FV3 reconciles those
 * duplicates separately via mpp_get_boundary, with the stated purpose "Prevent
 * accumulation of rounding errors at overlapped domain edges". That is exchange
 * E11 in the halo call map, and it is a distinct operation -- but it is expressed
 * by these same functions evaluated at h = 0, and it is now implemented as the
 * FV3_GPU_H_DWIND_SEAM group.
 *
 * TWO CORRECTIONS to what this comment previously asserted. (1) The call is at
 * dyn_core.F90:1178, not :1094-1095. (2) mpp_get_boundary does NOT stop at the
 * 1-D ebuffer/nbuffer: :1181-1188 copies both buffers straight back into u and v,
 * so E11's destination IS the field array -- specifically owned cells
 * u(is:ie,je+1,:) and v(ie+1,js:je,:). Any reasoning that relied on "never into
 * the field" was wrong.
 *
 * A further inherited claim, that ring h = 0 is "structurally unreachable" in
 * this API, was never tested and is false: halo_dest_index at h = 0 lands exactly
 * on those two owned lines, which is precisely why E11 needs no new index algebra.
 * ------------------------------------------------------------------------- */
struct StaggerShift {
  int8_t i = 0, j = 0;
  FV3_HALO_HD inline constexpr bool operator==(const StaggerShift &o) const noexcept {
    return i == o.i && j == o.j;
  }
};

inline constexpr StaggerShift stagger_center{0, 0};
inline constexpr StaggerShift stagger_east{1, 0};   /* uc, D-grid v */
inline constexpr StaggerShift stagger_north{0, 1};  /* vc, D-grid u */
inline constexpr StaggerShift stagger_corner{1, 1}; /* divgd */

/* The shifts of the array the DONOR is read from. Transposed on a rotated seam
 * because the rotated source is the other component. */
FV3_HALO_HD inline constexpr StaggerShift source_shift(StaggerShift mine,
                                                       bool reversed) noexcept {
  return reversed ? StaggerShift{mine.j, mine.i} : mine;
}

/* Along-seam extent of MY halo strip on edge `e`: the number of destination
 * points in one ring. W/E strips run over j, S/N strips over i. */
FV3_HALO_HD inline constexpr int64_t edge_seam_extent(TileEdge e, int64_t nx, int64_t ny,
                                                      StaggerShift s = stagger_center) noexcept {
  return edge_is_x_normal(e) ? (ny + s.j) : (nx + s.i);
}

/* Along-seam extent of the DONOR's edge, i.e. the modulus of the reversal. Its
 * shifts are `src` = source_shift(mine, reversed). */
FV3_HALO_HD inline constexpr int64_t donor_seam_extent(TileEdge donor, int64_t nx, int64_t ny,
                                                       StaggerShift src) noexcept {
  return edge_is_x_normal(donor) ? (ny + src.j) : (nx + src.i);
}

FV3_HALO_HD inline constexpr void halo_dest_index(TileEdge mine, int64_t t, int32_t h,
                                                  int64_t nx, int64_t ny, StaggerShift s,
                                                  int64_t &i, int64_t &j) noexcept {
  switch (mine) {
    case TileEdge::west:  i = 1 - h;          j = t;              break;
    case TileEdge::east:  i = (nx + s.i) + h; j = t;              break;
    case TileEdge::south: i = t;              j = 1 - h;          break;
    case TileEdge::north: i = t;              j = (ny + s.j) + h; break;
  }
}

FV3_HALO_HD inline constexpr void halo_src_index(TileEdge donor, int64_t tprime, int32_t h,
                                                 int64_t nx, int64_t ny, StaggerShift src,
                                                 int64_t &i, int64_t &j) noexcept {
  switch (donor) {
    case TileEdge::west:  i = src.i + h;  j = tprime;     break;
    case TileEdge::east:  i = nx + 1 - h; j = tprime;     break;
    case TileEdge::south: i = tprime;     j = src.j + h;  break;
    case TileEdge::north: i = tprime;     j = ny + 1 - h; break;
  }
}

/* t -> t' for one seam. `n` is the DONOR's along-seam extent
 * (donor_seam_extent). */
FV3_HALO_HD inline constexpr int64_t seam_along_index(int64_t t, int64_t n,
                                            bool reversed) noexcept {
  return reversed ? (n + 1 - t) : t;
}

/* CENTER wrappers, kept because the cell-centered contacts above are quoted
 * against them and because most groups are cell-centered. They are wrappers
 * rather than copies so that there is one implementation of the algebra. */
FV3_HALO_HD inline constexpr void halo_dest_index_center(TileEdge mine, int64_t t, int32_t h,
                                               int64_t nx, int64_t ny, int64_t &i,
                                               int64_t &j) noexcept {
  halo_dest_index(mine, t, h, nx, ny, stagger_center, i, j);
}

FV3_HALO_HD inline constexpr void halo_src_index_center(TileEdge donor, int64_t tprime,
                                              int32_t h, int64_t nx, int64_t ny,
                                              int64_t &i, int64_t &j) noexcept {
  halo_src_index(donor, tprime, h, nx, ny, stagger_center, i, j);
}

/* ---------------------------------------------------------------------------
 * Builder.
 *
 * Produces the TopologyView for one tile of a global cubed sphere, with every
 * one of the four EdgeNeighbor descriptors fully populated -- neighbour rank,
 * neighbour tile, the neighbour's OWN edge that abuts mine, the derived
 * orientation, and the derived vector signs.
 *
 * `tile` is 1-based (1..6), matching FMS/FV3 mosaic tile numbering. `npx`/`npy`
 * are the FV3 corner counts, so owned cells are nx = npx-1 by ny = npy-1;
 * CubeHalo.F90:103 requires npx == npy on a cube and this builder enforces it.
 * `rank_of_tile` is indexed 0..5 for tiles 1..6.
 *
 * Returns false and leaves `out` untouched on any inconsistent argument. It
 * never partially fills, because a partially filled topology is exactly the
 * fail-open case this milestone is removing.
 * ------------------------------------------------------------------------- */
bool make_cubed_sphere_topology(int tile, int64_t npx, int64_t npy,
                                const int32_t *rank_of_tile,
                                TopologyView &out) noexcept;

/* The mosaic connectivity itself, exposed so tests can assert it directly
 * against CubeHalo.F90:213-261 without going through TopologyView.
 *
 * `tile` is 1-based. Writes the donor tile (1-based) and the donor's own edge
 * for my edge `mine`. Returns false for an out-of-range tile. */
bool cubed_sphere_neighbor(int tile, TileEdge mine, int &donor_tile,
                           TileEdge &donor_edge) noexcept;

} // namespace fv3::gpu::halo
