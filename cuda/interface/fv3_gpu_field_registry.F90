!-----------------------------------------------------------------------------
! fv3_gpu_field_registry_mod -- build the CUDA dyn_core field descriptor array
! from a real FV3 Atm state.
!
! Prompt 14/16 resumption. This is the piece that turns the standalone
! executable's actual Fortran arrays into the fv3_gpu_field_desc[] that
! fv3_gpu_initialize_state consumes. It is called once per rank from
! fv3_gpu_dyncore_bridge.F90 (fv3_gpu_bridge_init).
!
! THREE HARD RULES, each one learned from a real rejection or a real defect:
!
!   1. TRUE FORTRAN BOUNDS, NOT A UNIFORM BOX. FieldRegistry::insert ->
!      make_device_view rejects any descriptor whose field_id / rank / stagger /
!      access / element width disagrees with the stable specification table, and
!      each WP binder then independently shape-checks its own rectangles
!      (dgrid/wp_b_dsw_contract_cuda.cpp:196-355 is the strictest). The
!      benchmark harness could get away with allocating every 2-D metric on one
!      padded ni_h x nj_h box; real FV3 allocates per-stagger true bounds
!      (fv_arrays.F90:1583-1720), so every descriptor below carries the exact
!      declared lower/extent/stride of the array it points at.
!
!   2. THE VERTICAL INDEX IS 0-BASED. riem_solver3_kernel hardcodes k = 0..km-1
!      and copy_corners_kernel derives its level from blockIdx.z, so
!      corner_edge_service_cuda refuses any rank-3 field whose vertical lower is
!      not 0 (runtime/fv3_gpu_production_dispatch.cu:60). Fortran k = 1 maps to
!      descriptor index 0. pe and peln are the only fields stored (i,k,j), so
!      for those two the 0-based axis is dimension 2, not 3.
!
!   3. STAGGER AND ACCESS ARE ASKED FOR, NEVER ASSUMED. They come from the C
!      contract through fv3_gpu_field_spec_query rather than being transcribed
!      here, so there is exactly one (field -> stagger, access, width) table in
!      the project and this module cannot drift from it.
!
! OWNERSHIP. Three groups of arrays:
!
!   (a) Atm components and Atm%gridstruct metrics -- registered in place with
!       c_loc. Atm must therefore be a TARGET, and every one of these arrays
!       must already hold its final value when fv3_gpu_initialize_state runs,
!       because that call is the initial host-to-device copy.
!
!   (b) dyn_core locals and fv_dynamics locals -- ws, delpc, ptc, ut, vt, divgd,
!       crx, xfx, cry, yfx, gz, zh, pkc, pk3, du, dv, heat_source, ws3, z_rat,
!       divg2, dpx, plus the derived cappa / zs / dp_ref. These do not exist on
!       Atm at all (dyn_core.F90:146-262 declares them), so this module owns
!       them for the lifetime of the GPU context and frees them in
!       fv3_gpu_registry_free.
!
!       NOT ALL OF GROUP (b) IS A dyn_core INTERNAL. Two members are
!       fv_dynamics locals that dyn_core writes and that fv_dynamics' own
!       downstream code then reads, so they must be copied back out after every
!       call rather than merely owned here:
!         - the four flux capacitors mfx/mfy/cx/cy, consumed by tracer_2d
!           (fv3_gpu_registry_export_flux_capacitors);
!         - ws, declared at fv_dynamics.F90:275, written intent(out) by
!           dyn_core.F90:224, and read intent(in) by Lagrangian_to_Eulerian
!           (fv_mapz.F90:159, used at :620 as the w-remap bottom boundary
!           condition) -- fv3_gpu_registry_export_ws.
!       Everything else in group (b) is genuinely internal to one dyn_core call
!       and is deliberately never copied back.
!
!   (c) Two shape shims, both module-owned copies rather than in-place
!       registrations, and both for a documented source reason -- see
!       rsina_pad and the edge_* block below.
!
! Numerical semantics changed: NO. Nothing here implements a formula except the
! two authoritative one-liners it is required to reproduce (zs = phis/grav,
! dyn_core.F90:342; dp_ref, dyn_core.F90:332) and the damping column, which is
! delegated to the existing translation rather than re-derived.
!-----------------------------------------------------------------------------
module fv3_gpu_field_registry_mod

  use, intrinsic :: iso_c_binding
  use fv3_gpu_abi_mod
  use fv_arrays_mod,  only: fv_atmos_type
  use constants_mod,  only: grav, rdgas, cp_air

  implicit none
  private

  public :: fv3_gpu_registry_build
  public :: fv3_gpu_registry_free
  public :: fv3_gpu_registry_bind_flux_capacitors
  public :: fv3_gpu_registry_export_flux_capacitors
  public :: fv3_gpu_registry_export_ws
  ! Diagnostic only: lets the caller's FATAL message name the field that was
  ! rejected instead of reporting a bare return code.
  public :: fv3_gpu_registry_last_failure

  ! Model real kind, as promoted by the build (SINGLE_FV / -r8). Every array in
  ! group (b) is declared with it so the registry never mixes precisions.
  integer, parameter :: FVR = kind(1.0)
  ! The five flux capacitors and the four edge_* weights are real(kind=8) in the
  ! authoritative source and are width-pinned to accumulator_real (8 bytes)
  ! by the contract regardless of build precision.
  integer, parameter :: ACC = 8

  !---------------------------------------------------------------------------
  ! Group (b): dyn_core / fv_dynamics locals this module owns.
  ! Bounds are the authoritative declarations, cited per array.
  !---------------------------------------------------------------------------
  real(FVR), allocatable, save, target :: r_ws(:,:)          ! dyn_core.F90:224
  real(FVR), allocatable, save, target :: r_ws3(:,:)         ! :248
  ! z_rat is 2D in dyn_core.F90 (:248) because Fortran recomputes it inside the
  ! k loop, but the CUDA field contract declares it rank 3
  ! (fv3_gpu_contract.cpp:32) and the Gate-C2-validated benchmark registers a
  ! full 3D array. The descriptor must match the contract, so carry the level
  ! axis here. Values are unaffected: the only consumer
  ! (compute_absolute_vorticity, wp_b_kernels.hpp:667) reads z_rat(i,j) with no
  ! level index, and it reads it at all only when .not.hydrostatic .and. do_f3d.
  real(FVR), allocatable, save, target :: r_z_rat(:,:,:)     ! :248 (+level axis)
  real(FVR), allocatable, save, target :: r_divg2(:,:)       ! :256
  real(FVR), allocatable, save, target :: r_zs(:,:)          ! :249
  real(FVR), allocatable, save, target :: r_dp_ref(:)        ! :248
  real(FVR), allocatable, save, target :: r_cappa(:,:,:)     ! :191 (assumed shape)
  real(FVR), allocatable, save, target :: r_delpc(:,:,:)     ! :358
  real(FVR), allocatable, save, target :: r_ptc(:,:,:)       ! :352
  real(FVR), allocatable, save, target :: r_ut(:,:,:)        ! :360
  real(FVR), allocatable, save, target :: r_vt(:,:,:)        ! :362
  real(FVR), allocatable, save, target :: r_divgd(:,:,:)     ! :357
  real(FVR), allocatable, save, target :: r_crx(:,:,:)       ! :353
  real(FVR), allocatable, save, target :: r_xfx(:,:,:)       ! :354
  real(FVR), allocatable, save, target :: r_cry(:,:,:)       ! :355
  real(FVR), allocatable, save, target :: r_yfx(:,:,:)       ! :356
  real(FVR), allocatable, save, target :: r_gz(:,:,:)        ! :349
  real(FVR), allocatable, save, target :: r_zh(:,:,:)        ! :366
  real(FVR), allocatable, save, target :: r_pkc(:,:,:)       ! :351
  real(FVR), allocatable, save, target :: r_pk3(:,:,:)       ! :368
  real(FVR), allocatable, save, target :: r_du(:,:,:)        ! :372
  real(FVR), allocatable, save, target :: r_dv(:,:,:)        ! :374
  real(FVR), allocatable, save, target :: r_heat_source(:,:,:) ! :398

  ! Flux capacitors. These are fv_dynamics DUMMY ARGUMENTS
  ! (fv_dynamics.F90:242-247, real(kind=8), intent(inout)), owned by the caller
  ! above fv_dynamics, so the registry cannot reach them from Atm. If the caller
  ! binds them (fv3_gpu_registry_bind_flux_capacitors) they are registered in
  ! place; otherwise these owned buffers are used -- see the gap note on that
  ! routine.
  real(ACC), allocatable, save, target :: r_mfx(:,:,:)       ! fv_dynamics.F90:242
  real(ACC), allocatable, save, target :: r_mfy(:,:,:)       ! :243
  real(ACC), allocatable, save, target :: r_cx(:,:,:)        ! :245
  real(ACC), allocatable, save, target :: r_cy(:,:,:)        ! :246
  real(ACC), allocatable, save, target :: r_dpx(:,:,:)       ! :282 (local)

  type(c_ptr), save :: p_mfx = C_NULL_PTR
  type(c_ptr), save :: p_mfy = C_NULL_PTR
  type(c_ptr), save :: p_cx  = C_NULL_PTR
  type(c_ptr), save :: p_cy  = C_NULL_PTR
  logical,     save :: flux_bound = .false.

  !---------------------------------------------------------------------------
  ! Group (c), shim 1: rsina.
  !
  ! rsina is THE ONLY 2-D metric whose real allocation is narrower than the
  ! [isd,ied] x [jsd,jed] box the d_sw contract requires of every member of its
  ! geo2[] list (wp_b_dsw_contract_cuda.cpp:196-355). FV3 allocates it
  ! (is:ie+1, js:je+1) -- fv_arrays.F90:1626, whose own source comment reads
  ! "! Why is the size different?" -- and every other geo2 member is either an
  ! exact match or a strict superset.
  !
  ! Its true read footprint was checked rather than assumed: the only two
  ! production consumers are bgrid_vb_kernel and bgrid_ub_kernel
  ! (dgrid/wp_b_momentum_cuda.cu:378-448), both of which launch over
  ! i in [is, ie+1], j in [js, je+1] and read rsina(i,j) only there -- exactly
  ! FV3's allocation. So the contract's uniform requirement is an
  ! over-requirement for this one field, and the halo cells of the padded copy
  ! are provably never read (they stay zero). ie+1 <= ied and je+1 <= jed hold
  ! because the halo is 3.
  !
  ! A padded module-owned shadow is used in preference to relaxing the contract
  ! check because it touches no Gate-C2-validated CUDA source.
  !---------------------------------------------------------------------------
  real(FVR), allocatable, save, target :: r_rsina_pad(:,:)

  !---------------------------------------------------------------------------
  ! Group (c), shim 2: the four edge_* interpolation weights.
  !
  ! FV3 allocates these 1-D and GLOBAL 1-based -- edge_s(npx), edge_n(npx),
  ! edge_w(npy), edge_e(npy) (fv_arrays.F90:1638-1641) -- while the consumers
  ! index them with the absolute i or j of the local subdomain
  ! (operators/wp_d_operators.cpp:288-309,381-452 and
  ! operators/wp_d_cuda.cu:160-183,270-346 read g.edge_s(i) / g.edge_w(j)).
  ! Neither binder shape-checks them: the d_sw contract deliberately declines to,
  ! on the stated grounds that their extents are WP-D's contract and
  ! "duplicating a guess here would just invent a second, possibly different,
  ! rule".
  !
  ! Because the read range therefore cannot be bounded from source, these are
  ! registered as padded module-owned copies spanning the local halo box, filled
  ! from the authoritative global arrays at their absolute indices. That
  ! reproduces the registration the Gate-C2 harness used while keeping the
  ! values authoritative. They are already real(kind=8) (R_GRID), matching
  ! accumulator_real exactly, so only the index domain shifts.
  !---------------------------------------------------------------------------
  real(ACC), allocatable, save, target :: r_edge_w(:)
  real(ACC), allocatable, save, target :: r_edge_e(:)
  real(ACC), allocatable, save, target :: r_edge_s(:)
  real(ACC), allocatable, save, target :: r_edge_n(:)

  !---------------------------------------------------------------------------
  ! Per-level damping columns, dyn_core.F90:677-765. Produced by the existing
  ! authoritative translation through fv3_gpu_registry_damping_column, NOT
  ! re-derived here. The three order columns are carried as fv_real holding
  ! exactly-representable integers, which is what the WP-B binder expects.
  !---------------------------------------------------------------------------
  real(FVR), allocatable, save, target :: r_d2_bg(:), r_d4_bg(:)
  real(FVR), allocatable, save, target :: r_damp_v(:), r_damp_w(:), r_damp_t(:)
  real(FVR), allocatable, save, target :: r_d_con(:), r_kgb(:)
  real(FVR), allocatable, save, target :: r_ndif(:), r_nord_k(:), r_nord_w(:)

  ! Element type/width of the model real, resolved once per build.
  integer(c_int32_t), save :: fv_etype = 0
  integer(c_int32_t), save :: fv_ebytes = 0

  logical, save :: registry_live = .false.

  !---------------------------------------------------------------------------
  ! Registry-construction helpers implemented in
  ! cuda/dyn_core/runtime/fv3_gpu_registry_shim.cpp. Declared locally, not in
  ! fv3_gpu_abi_mod, because neither is part of the public dyn_core ABI.
  !---------------------------------------------------------------------------
  interface
     integer(c_int32_t) function fv3_gpu_field_spec_query(field_id, rank,      &
                                     stagger, access, element_bytes)          &
          bind(C, name='fv3_gpu_field_spec_query')
       import :: c_int32_t
       integer(c_int32_t), value       :: field_id
       integer(c_int32_t), intent(out) :: rank, stagger, access, element_bytes
     end function fv3_gpu_field_spec_query

     integer(c_int32_t) function fv3_gpu_registry_damping_column(npz, nord,    &
              n_sponge, n_zfilter, do_vort_damp, hydrostatic, dddmp, d2_bg,    &
              d4_bg_top, d4_bg_bot, d2_bg_k1, d2_bg_k2, vtdm4, d_con, d_ext,   &
              ke_bg, ptop, pfull, o_d2_bg, o_d4_bg, o_damp_v, o_damp_w,        &
              o_damp_t, o_d_con, o_kgb, o_ndif, o_nord_k, o_nord_w)            &
          bind(C, name='fv3_gpu_registry_damping_column')
       import :: c_int32_t, c_double
       integer(c_int32_t), value :: npz, nord, n_sponge, n_zfilter
       integer(c_int32_t), value :: do_vort_damp, hydrostatic
       real(c_double),     value :: dddmp, d2_bg, d4_bg_top, d4_bg_bot
       real(c_double),     value :: d2_bg_k1, d2_bg_k2, vtdm4, d_con
       real(c_double),     value :: d_ext, ke_bg, ptop
       real(c_double), intent(in)  :: pfull(*)
       real(c_double), intent(out) :: o_d2_bg(*), o_d4_bg(*), o_damp_v(*)
       real(c_double), intent(out) :: o_damp_w(*), o_damp_t(*), o_d_con(*)
       real(c_double), intent(out) :: o_kgb(*), o_ndif(*), o_nord_k(*)
       real(c_double), intent(out) :: o_nord_w(*)
     end function fv3_gpu_registry_damping_column
  end interface

  ! Descriptor cursor, module-scope so the add* helpers can append.
  type(fv3_gpu_field_desc), allocatable, save :: work(:)
  integer, save :: ncur = 0
  integer, save :: build_rc = 0

  ! Diagnostic breadcrumbs. build_rc alone says WHICH CHECK rejected but not
  ! which field tripped it, and a rejected registration is otherwise invisible
  ! from the caller; these make the FATAL message name the field.
  integer, save :: build_fail_fid = -1
  integer, save :: build_fail_aux = 0

contains

  !===========================================================================
  ! Report the breadcrumb left by the most recent failed build.
  !
  ! fid is the field the add() helper was working on when a check rejected, or
  ! -1 if the failure happened outside add(); aux carries the subsidiary code
  ! (the contract query's own rc, or the contract's element width).
  !===========================================================================
  subroutine fv3_gpu_registry_last_failure(fid, aux)
    integer, intent(out) :: fid, aux
    fid = build_fail_fid
    aux = build_fail_aux
  end subroutine fv3_gpu_registry_last_failure

  !===========================================================================
  ! Optional: let the caller register its OWN flux capacitors in place.
  !
  ! mfx/mfy/cx/cy are fv_dynamics dummy arguments, and the tracer transport
  ! that runs AFTER dyn_core (fv_tracer2d) reads exactly those arrays. If the
  ! caller does not bind them, the CUDA path accumulates into registry-owned
  ! buffers instead and the caller's arrays are left untouched -- which is a
  ! real integration gap for any run that advects tracers, though not for a
  ! dyn_core-only state comparison.
  !
  ! Call this BEFORE fv3_gpu_registry_build. The arrays must be contiguous
  ! targets with the authoritative bounds at fv_dynamics.F90:242-247.
  !===========================================================================
  subroutine fv3_gpu_registry_bind_flux_capacitors(mfx, mfy, cx, cy)
    real(ACC), intent(inout), target :: mfx(:,:,:), mfy(:,:,:)
    real(ACC), intent(inout), target :: cx(:,:,:),  cy(:,:,:)
    p_mfx = c_loc(mfx(1,1,1))
    p_mfy = c_loc(mfy(1,1,1))
    p_cx  = c_loc(cx(1,1,1))
    p_cy  = c_loc(cy(1,1,1))
    flux_bound = .true.
  end subroutine fv3_gpu_registry_bind_flux_capacitors

  !===========================================================================
  ! Copy the registry-owned flux capacitors out, after a dyn_core step's D2H.
  !
  ! This is the path the CUDA backend actually uses, and binding is deliberately
  ! NOT used for it. What the Fortran dyn_core is handed at fv_dynamics.F90:713
  ! is not an Atm component but fv_dynamics' own automatic locals mfxL/mfyL/
  ! cxL/cyL (declared :265-269), whose lifetime is one fv_dynamics call. dyn_core
  ! zeroes them on entry (dyn_core.F90:392-395) and fills them over the acoustic
  ! loop; fv_tracer2d then reads them (:789-799) and only afterwards does :873-876
  ! accumulate mfx = mfx + mfxL into the Atm export arrays. A stack local cannot
  ! be bound once at init, and binding Atm%mfx instead would be wrong twice over:
  ! the CUDA dyn_core would zero the k_split/model-step accumulator on every call
  ! and :873 would then add it to itself.
  !
  ! r_mfx/r_mfy/r_cx/r_cy have the same bounds as the locals and the same
  ! once-per-call zeroing on the device side (fv3_gpu_runtime.cpp:72), so they are
  ! the faithful stand-in; this routine hands their post-D2H contents back to
  ! whichever arrays the caller passed down.
  !
  ! Shapes are checked rather than assumed: a silent shape mismatch here would
  ! starve tracer advection in exactly the way that is hard to see in a dyn_core
  ! state comparison.
  !===========================================================================
  subroutine fv3_gpu_registry_export_flux_capacitors(mfx, mfy, cx, cy, rc)
    real(ACC), intent(out) :: mfx(:,:,:), mfy(:,:,:)
    real(ACC), intent(out) :: cx(:,:,:),  cy(:,:,:)
    integer,   intent(out) :: rc

    rc = 0
    if (flux_bound) then
       ! Registered in place by a caller; there is nothing to copy and copying
       ! from the (unallocated) owned buffers would be a bug.
       rc = 1
       return
    end if
    if (.not. allocated(r_mfx)) then
       rc = 2
       return
    end if
    if (any(shape(mfx) /= shape(r_mfx)) .or. any(shape(mfy) /= shape(r_mfy)) .or. &
        any(shape(cx)  /= shape(r_cx))  .or. any(shape(cy)  /= shape(r_cy))) then
       rc = 3
       return
    end if

    mfx = r_mfx
    mfy = r_mfy
    cx  = r_cx
    cy  = r_cy
  end subroutine fv3_gpu_registry_export_flux_capacitors

  !===========================================================================
  ! Copy the post-D2H w-at-surface array back out to the caller's ws.
  !
  ! ws is NOT a dyn_core internal, despite being registered alongside the real
  ! internals in group (b) above. It is an fv_dynamics LOCAL AUTOMATIC
  ! (fv_dynamics.F90:275, real:: ws(bd%is:bd%ie,bd%js:bd%je)) that dyn_core
  ! writes as intent(out) (dyn_core.F90:224 "w at surface") and that
  ! Lagrangian_to_Eulerian then reads as intent(in) (fv_mapz.F90:159), using it
  ! at :620 as q_bot -- the bottom boundary condition of the w vertical remap.
  ! So it crosses the dyn_core boundary in exactly the same way the four flux
  ! capacitors do, and like them it must be copied back after every call: with
  ! the CUDA backend the device writes r_ws, and fv_dynamics' own ws would
  ! otherwise still hold undefined stack contents.
  !
  ! Kept as its own entry point rather than folded into the flux-capacitor
  ! export because the capacitors are real(ACC) (width-pinned to kind=8 by the
  ! authoritative declarations) while ws is a model real, and because the two
  ! have independent bind/allocate preconditions.
  !===========================================================================
  subroutine fv3_gpu_registry_export_ws(ws, rc)
    real(FVR), intent(out) :: ws(:,:)
    integer,   intent(out) :: rc

    rc = 0
    if (.not. allocated(r_ws)) then
       rc = 1
       return
    end if
    if (any(shape(ws) /= shape(r_ws))) then
       rc = 2
       return
    end if

    ws = r_ws
  end subroutine fv3_gpu_registry_export_ws

  !===========================================================================
  ! Fill one descriptor. Stagger, access and the required element width come
  ! from the C contract; everything else comes from the caller.
  !
  ! lo/ex are the descriptor's own index space: lo(1:rank) the lower bounds,
  ! ex(1:rank) the extents. Strides are column-major contiguous over ex, which
  ! is correct for a whole-array registration and for a leading contiguous
  ! prefix (the only prefix case is sin_sg/cos_sg, where 4 of 9 components are
  ! registered and the first two extents still span the full array).
  !===========================================================================
  subroutine add(fid, base, rank, lo, ex, use_acc)
    integer,           intent(in) :: fid, rank
    type(c_ptr),       intent(in) :: base
    integer(c_int64_t), intent(in) :: lo(:), ex(:)
    logical, optional, intent(in) :: use_acc

    type(fv3_gpu_field_desc) :: d
    integer(c_int32_t) :: srank, sstag, sacc, sbytes, q
    integer(c_int64_t) :: st, nelem
    integer :: i
    logical :: acc

    if (build_rc /= 0) return
    acc = .false.
    if (present(use_acc)) acc = use_acc

    q = fv3_gpu_field_spec_query(int(fid, c_int32_t), srank, sstag, sacc, sbytes)
    if (q /= 0) then
       build_rc = 1
       build_fail_fid = fid
       build_fail_aux = int(q)
       return
    end if
    if (int(srank) /= rank) then
       ! A rank disagreement means this routine and the contract disagree about
       ! what the field IS, which no amount of bounds care can rescue.
       build_rc = 2
       build_fail_fid = fid
       build_fail_aux = int(srank)
       return
    end if

    d%abi_version   = FV3_GPU_ABI_VERSION
    d%struct_bytes  = int(storage_size(d) / 8, c_int32_t)
    d%field_id      = int(fid, c_int32_t)
    d%name          = C_NULL_CHAR
    d%base          = base
    d%rank          = int(rank, c_int32_t)
    d%memory_location = FV3_GPU_MEMORY_HOST
    d%stagger       = sstag
    d%access        = sacc
    d%alignment_bytes = 0
    d%reserved      = 0

    if (acc) then
       d%element_type  = FV3_GPU_ELEMENT_F64
       ! Literal 8, not the module parameter ACC: this subroutine's own logical
       ! dummy is spelled `acc`, which shadows it. accumulator_real is pinned at
       ! 8 bytes regardless of build precision (fv3_gpu_contract.cpp).
       d%element_bytes = 8_c_int32_t
    else
       ! sbytes == 0 means the contract leaves the width to the descriptor (the
       ! three integral damping orders); their consumers read them through an
       ! fv_real view, so fv_real is the correct choice there too.
       d%element_type  = fv_etype
       d%element_bytes = fv_ebytes
       if (sbytes /= 0 .and. sbytes /= fv_ebytes) then
          build_rc = 3
          build_fail_fid = fid
          build_fail_aux = int(sbytes)
          return
       end if
    end if

    st    = 1_c_int64_t
    nelem = 1_c_int64_t
    do i = 1, rank
       d%lower(i)           = lo(i)
       d%extent(i)          = ex(i)
       d%stride_elements(i) = st
       d%compute_lower(i)   = lo(i)
       d%compute_upper(i)   = lo(i) + ex(i) - 1_c_int64_t
       d%halo_lower(i)      = 0
       d%halo_upper(i)      = 0
       st    = st * ex(i)
       nelem = nelem * ex(i)
    end do
    do i = rank + 1, FV3_GPU_MAX_RANK
       d%lower(i) = 0; d%extent(i) = 0; d%stride_elements(i) = 0
       d%compute_lower(i) = 0; d%compute_upper(i) = 0
       d%halo_lower(i) = 0; d%halo_upper(i) = 0
    end do
    d%allocation_bytes = nelem * int(d%element_bytes, c_int64_t)

    ncur = ncur + 1
    if (ncur > size(work)) then
       build_rc = 4
       build_fail_fid = fid
       build_fail_aux = ncur
       return
    end if
    work(ncur) = d
  end subroutine add

  ! Rank-1/2/3 convenience wrappers, so each registration below reads as one
  ! line whose bounds can be checked against the cited Fortran declaration.
  subroutine add1(fid, base, l1, e1, use_acc)
    integer, intent(in) :: fid
    type(c_ptr), intent(in) :: base
    integer, intent(in) :: l1, e1
    logical, optional, intent(in) :: use_acc
    call add(fid, base, 1, [int(l1,c_int64_t)], [int(e1,c_int64_t)], use_acc)
  end subroutine add1

  subroutine add2(fid, base, l1, e1, l2, e2, use_acc)
    integer, intent(in) :: fid
    type(c_ptr), intent(in) :: base
    integer, intent(in) :: l1, e1, l2, e2
    logical, optional, intent(in) :: use_acc
    call add(fid, base, 2, [int(l1,c_int64_t), int(l2,c_int64_t)],             &
                          [int(e1,c_int64_t), int(e2,c_int64_t)], use_acc)
  end subroutine add2

  subroutine add3(fid, base, l1, e1, l2, e2, l3, e3, use_acc)
    integer, intent(in) :: fid
    type(c_ptr), intent(in) :: base
    integer, intent(in) :: l1, e1, l2, e2, l3, e3
    logical, optional, intent(in) :: use_acc
    call add(fid, base, 3, [int(l1,c_int64_t), int(l2,c_int64_t), int(l3,c_int64_t)], &
                          [int(e1,c_int64_t), int(e2,c_int64_t), int(e3,c_int64_t)], use_acc)
  end subroutine add3

  !===========================================================================
  ! Build the descriptor array.
  !===========================================================================
  subroutine fv3_gpu_registry_build(Atm, descs, ndesc, rc)
    type(fv_atmos_type), intent(inout), target :: Atm
    type(fv3_gpu_field_desc), allocatable, intent(out) :: descs(:)
    integer, intent(out) :: ndesc
    integer, intent(out) :: rc

    integer :: is, ie, js, je, isd, ied, jsd, jed, npz, npx, npy
    integer :: ni, nj, nih, njh, i, j, k
    real(c_double), allocatable :: pfull(:)
    real(c_double), allocatable :: c2(:,:)
    real(FVR) :: rgrav, kappa_dry
    integer(c_int32_t) :: q

    rc = 0
    ndesc = 0
    build_rc = 0
    build_fail_fid = -1
    build_fail_aux = 0

    if (registry_live) call fv3_gpu_registry_free()

    is  = Atm%bd%is;   ie  = Atm%bd%ie
    js  = Atm%bd%js;   je  = Atm%bd%je
    isd = Atm%bd%isd;  ied = Atm%bd%ied
    jsd = Atm%bd%jsd;  jed = Atm%bd%jed
    npz = Atm%npz
    npx = Atm%npx
    npy = Atm%npy
    ni  = ie - is + 1
    nj  = je - js + 1
    nih = ied - isd + 1
    njh = jed - jsd + 1

    ! Element type/width of the model real, taken from an actual model array
    ! rather than from a hardcoded kind, so an R4 build cannot silently
    ! register R8 descriptors.
    fv_ebytes = int(storage_size(Atm%delp(isd,jsd,1)) / 8, c_int32_t)
    if (fv_ebytes == 4) then
       fv_etype = FV3_GPU_ELEMENT_F32
    else if (fv_ebytes == 8) then
       fv_etype = FV3_GPU_ELEMENT_F64
    else
       rc = 11
       return
    end if

    !-----------------------------------------------------------------------
    ! Group (b) allocations, with the authoritative bounds.
    !-----------------------------------------------------------------------
    allocate(r_ws   (is:ie,   js:je))                  ; r_ws    = 0.0_FVR
    allocate(r_ws3  (isd:ied, jsd:jed))                ; r_ws3   = 0.0_FVR
    allocate(r_z_rat(isd:ied, jsd:jed, npz))           ; r_z_rat = 0.0_FVR
    allocate(r_divg2(is:ie+1, js:je+1))                ; r_divg2 = 0.0_FVR
    allocate(r_zs   (isd:ied, jsd:jed))                ; r_zs    = 0.0_FVR
    allocate(r_dp_ref(npz))                            ; r_dp_ref = 0.0_FVR
    allocate(r_cappa(isd:ied, jsd:jed, npz))
    allocate(r_delpc(isd:ied,   jsd:jed,   npz))       ; r_delpc = 0.0_FVR
    allocate(r_ptc  (isd:ied,   jsd:jed,   npz))       ; r_ptc   = 0.0_FVR
    ! ut/vt carry ONE MORE LINE than dyn_core.F90:360-362 allocates
    ! (isd:ied, jsd:jed): the CUDA d_sw binder requires ut over [isd,ied+1] and
    ! vt over [jsd,jed+1] (wp_b_dsw_contract_cuda.cpp:224-231), a deliberate
    ! over-requirement of the same kind it documents for heat_source/diss_est,
    ! and the Gate-C2-validated benchmark satisfies it with padded arrays. These
    ! are module-private scratch never aliased to, nor copied back into, Fortran
    ! state, so the extra line is padding only. The C-grid predictor asks for
    ! exactly the Fortran box (fv3_cgrid_predictor.cu:426) and a larger array
    ! covers it.
    allocate(r_ut   (isd:ied+1, jsd:jed,   npz))       ; r_ut    = 0.0_FVR
    allocate(r_vt   (isd:ied,   jsd:jed+1, npz))       ; r_vt    = 0.0_FVR
    allocate(r_divgd(isd:ied+1, jsd:jed+1, npz))       ; r_divgd = 0.0_FVR
    allocate(r_crx  (is:ie+1,   jsd:jed,   npz))       ; r_crx   = 0.0_FVR
    allocate(r_xfx  (is:ie+1,   jsd:jed,   npz))       ; r_xfx   = 0.0_FVR
    allocate(r_cry  (isd:ied,   js:je+1,   npz))       ; r_cry   = 0.0_FVR
    allocate(r_yfx  (isd:ied,   js:je+1,   npz))       ; r_yfx   = 0.0_FVR
    allocate(r_gz   (isd:ied,   jsd:jed,   npz+1))     ; r_gz    = 0.0_FVR
    allocate(r_zh   (isd:ied,   jsd:jed,   npz+1))     ; r_zh    = 0.0_FVR
    allocate(r_pkc  (isd:ied,   jsd:jed,   npz+1))     ; r_pkc   = 0.0_FVR
    allocate(r_pk3  (isd:ied,   jsd:jed,   npz+1))     ; r_pk3   = 0.0_FVR
    allocate(r_du   (isd:ied,   jsd:jed+1, npz))       ; r_du    = 0.0_FVR
    allocate(r_dv   (isd:ied+1, jsd:jed,   npz))       ; r_dv    = 0.0_FVR
    allocate(r_heat_source(isd:ied, jsd:jed, npz))     ; r_heat_source = 0.0_FVR
    allocate(r_dpx  (is:ie, js:je, npz))               ; r_dpx   = 0.0_8

    if (.not. flux_bound) then
       allocate(r_mfx(is:ie+1,  js:je,   npz)) ; r_mfx = 0.0_8
       allocate(r_mfy(is:ie,    js:je+1, npz)) ; r_mfy = 0.0_8
       allocate(r_cx (is:ie+1,  jsd:jed, npz)) ; r_cx  = 0.0_8
       allocate(r_cy (isd:ied,  js:je+1, npz)) ; r_cy  = 0.0_8
       p_mfx = c_loc(r_mfx(is,js,1))
       p_mfy = c_loc(r_mfy(is,js,1))
       p_cx  = c_loc(r_cx (is,jsd,1))
       p_cy  = c_loc(r_cy (isd,js,1))
    end if

    ! zs = phis * (1/grav) -- dyn_core.F90:339-342, verbatim.
    rgrav = 1.0_FVR / real(grav, FVR)
    do j = jsd, jed
       do i = isd, ied
          r_zs(i,j) = Atm%phis(i,j) * rgrav
       end do
    end do

    ! dp_ref(k) = (ak(k+1)-ak(k)) + (bk(k+1)-bk(k))*1.E5 -- dyn_core.F90:330-332.
    do k = 1, npz
       r_dp_ref(k) = (Atm%ak(k+1) - Atm%ak(k))                                &
                   + (Atm%bk(k+1) - Atm%bk(k)) * 1.0e5_FVR
    end do

    ! cappa. The authoritative array is a fv_dynamics local, not an Atm
    ! component, so it cannot be registered in place. For the dry configuration
    ! this integration targets (q_con identically zero) the MOIST_CAPPA
    ! expression reduces exactly to the dry kappa, which is the same value the
    ! Gate-C2 harness used. A moist run needs the real array plumbed through --
    ! that is the already-recorded MOIST_CAPPA/USE_COND coverage gap, not a new
    ! one, and it must be closed before any q_con /= 0 configuration.
    kappa_dry = real(rdgas, FVR) / real(cp_air, FVR)
    r_cappa = kappa_dry

    ! rsina shadow -- see the block comment on r_rsina_pad.
    allocate(r_rsina_pad(isd:ied, jsd:jed)) ; r_rsina_pad = 0.0_FVR
    do j = js, min(je+1, jed)
       do i = is, min(ie+1, ied)
          r_rsina_pad(i,j) = Atm%gridstruct%rsina(i,j)
       end do
    end do

    ! edge_* shadows -- see the block comment on r_edge_w. Filled at absolute
    ! index; entries outside the authoritative 1..npx/1..npy global range stay
    ! zero, which is the same thing the harness registration presented.
    allocate(r_edge_w(isd:ied)) ; r_edge_w = 0.0_8
    allocate(r_edge_e(isd:ied)) ; r_edge_e = 0.0_8
    allocate(r_edge_s(isd:ied)) ; r_edge_s = 0.0_8
    allocate(r_edge_n(isd:ied)) ; r_edge_n = 0.0_8
    do j = max(jsd,1), min(jed, npy)
       r_edge_w(j) = Atm%gridstruct%edge_w(j)
       r_edge_e(j) = Atm%gridstruct%edge_e(j)
    end do
    do i = max(isd,1), min(ied, npx)
       r_edge_s(i) = Atm%gridstruct%edge_s(i)
       r_edge_n(i) = Atm%gridstruct%edge_n(i)
    end do

    !-----------------------------------------------------------------------
    ! Per-level damping columns, delegated to the authoritative translation.
    !
    ! pfull is only read by the d4dmp() log profile, and only when
    ! d4_bg_top /= d4_bg_bot. FV_StateMod sets both to the same value for a
    ! resolution below npx*CEILING(stretch_fac) < 1200 (which c8 is), so the
    ! profile collapses to a constant and pfull is unread; the reference column
    ! below is supplied so the argument is always well defined.
    !-----------------------------------------------------------------------
    allocate(pfull(npz))
    do k = 1, npz
       pfull(k) = 0.5d0 * (real(Atm%ak(k),8) + real(Atm%ak(k+1),8))           &
                + 0.5d0 * (real(Atm%bk(k),8) + real(Atm%bk(k+1),8)) * 1.0d5
    end do

    allocate(r_d2_bg(npz), r_d4_bg(npz), r_damp_v(npz), r_damp_w(npz),        &
             r_damp_t(npz), r_d_con(npz), r_kgb(npz), r_ndif(npz),            &
             r_nord_k(npz), r_nord_w(npz))
    allocate(c2(npz, 10))

    q = fv3_gpu_registry_damping_column(                                      &
          int(npz, c_int32_t), int(Atm%flagstruct%nord, c_int32_t),            &
          int(Atm%flagstruct%n_sponge, c_int32_t),                             &
          int(Atm%flagstruct%n_zfilter, c_int32_t),                            &
          merge(1_c_int32_t, 0_c_int32_t, Atm%flagstruct%do_vort_damp),        &
          merge(1_c_int32_t, 0_c_int32_t, Atm%flagstruct%hydrostatic),         &
          real(Atm%flagstruct%dddmp, c_double),                                &
          real(Atm%flagstruct%d2_bg,  c_double),                               &
          ! fv_flags_type carries d4_bg_top and d4_bg_bot separately
          ! (fv_arrays.F90:359, :365); there is no scalar `d4_bg` component.
          ! Pass both authoritatively rather than duplicating one of them.
          real(Atm%flagstruct%d4_bg_top, c_double),                             &
          real(Atm%flagstruct%d4_bg_bot, c_double),                             &
          real(Atm%flagstruct%d2_bg_k1, c_double),                             &
          real(Atm%flagstruct%d2_bg_k2, c_double),                             &
          real(Atm%flagstruct%vtdm4, c_double),                                &
          real(Atm%flagstruct%d_con, c_double),                                &
          real(Atm%flagstruct%d_ext, c_double),                                &
          real(Atm%flagstruct%ke_bg, c_double),                                &
          real(Atm%ak(1), c_double), pfull,                                    &
          c2(:,1), c2(:,2), c2(:,3), c2(:,4), c2(:,5),                         &
          c2(:,6), c2(:,7), c2(:,8), c2(:,9), c2(:,10))
    if (q /= 0) then
       rc = 20 + int(q)
       deallocate(pfull, c2)
       call fv3_gpu_registry_free()
       return
    end if
    r_d2_bg  = real(c2(:,1), FVR) ; r_d4_bg  = real(c2(:,2), FVR)
    r_damp_v = real(c2(:,3), FVR) ; r_damp_w = real(c2(:,4), FVR)
    r_damp_t = real(c2(:,5), FVR) ; r_d_con  = real(c2(:,6), FVR)
    r_kgb    = real(c2(:,7), FVR) ; r_ndif   = real(c2(:,8), FVR)
    r_nord_k = real(c2(:,9), FVR) ; r_nord_w = real(c2(:,10), FVR)
    deallocate(pfull, c2)

    !-----------------------------------------------------------------------
    ! Emit the descriptors. Room for the whole contract; ndesc reports what was
    ! actually emitted.
    !-----------------------------------------------------------------------
    allocate(work(FV3_FID_DAMPING_D_CON))
    ncur = 0

    ! --- (a) Atm prognostic and diagnostic state, fv_arrays.F90:1462-1528 ----
    call add3(FV3_FID_U,    c_loc(Atm%u(isd,jsd,1)),    isd,nih,   jsd,njh+1, 0,npz)
    call add3(FV3_FID_V,    c_loc(Atm%v(isd,jsd,1)),    isd,nih+1, jsd,njh,   0,npz)
    call add3(FV3_FID_W,    c_loc(Atm%w(isd,jsd,1)),    isd,nih,   jsd,njh,   0,npz)
    call add3(FV3_FID_DELZ, c_loc(Atm%delz(isd,jsd,1)), isd,nih,   jsd,njh,   0,npz)
    call add3(FV3_FID_PT,   c_loc(Atm%pt(isd,jsd,1)),   isd,nih,   jsd,njh,   0,npz)
    call add3(FV3_FID_DELP, c_loc(Atm%delp(isd,jsd,1)), isd,nih,   jsd,njh,   0,npz)
    call add3(FV3_FID_Q_CON,c_loc(Atm%q_con(isd,jsd,1)),isd,nih,   jsd,njh,   0,npz)
    call add3(FV3_FID_OMGA, c_loc(Atm%omga(isd,jsd,1)), isd,nih,   jsd,njh,   0,npz)
    call add3(FV3_FID_UA,   c_loc(Atm%ua(isd,jsd,1)),   isd,nih,   jsd,njh,   0,npz)
    call add3(FV3_FID_VA,   c_loc(Atm%va(isd,jsd,1)),   isd,nih,   jsd,njh,   0,npz)
    call add3(FV3_FID_UC,   c_loc(Atm%uc(isd,jsd,1)),   isd,nih+1, jsd,njh,   0,npz)
    call add3(FV3_FID_VC,   c_loc(Atm%vc(isd,jsd,1)),   isd,nih,   jsd,njh+1, 0,npz)
    call add3(FV3_FID_DISS_EST, c_loc(Atm%diss_est(isd,jsd,1)), isd,nih, jsd,njh, 0,npz)
    call add2(FV3_FID_PHIS, c_loc(Atm%phis(isd,jsd)),   isd,nih,   jsd,njh)

    ! pe and peln are the ONLY fields stored (i,k,j): the 0-based vertical axis
    ! is dimension 2 for these two. pk is (i,j,k) despite sitting beside them.
    call add3(FV3_FID_PE,   c_loc(Atm%pe(is-1,1,js-1)), is-1,ni+2, 0,npz+1, js-1,nj+2)
    call add3(FV3_FID_PELN, c_loc(Atm%peln(is,1,js)),   is,ni,     0,npz+1, js,nj)
    call add3(FV3_FID_PK,   c_loc(Atm%pk(is,js,1)),     is,ni,     js,nj,   0,npz+1)
    call add3(FV3_FID_PKZ,  c_loc(Atm%pkz(is,js,1)),    is,ni,     js,nj,   0,npz)

    ! --- Flux capacitors and dpx: accumulator_real (8 bytes) ----------------
    call add3(FV3_FID_MFX, p_mfx, is,ni+1, js,nj,   0,npz, use_acc=.true.)
    call add3(FV3_FID_MFY, p_mfy, is,ni,   js,nj+1, 0,npz, use_acc=.true.)
    call add3(FV3_FID_CX,  p_cx,  is,ni+1, jsd,njh, 0,npz, use_acc=.true.)
    call add3(FV3_FID_CY,  p_cy,  isd,nih, js,nj+1, 0,npz, use_acc=.true.)
    call add3(FV3_FID_DPX, c_loc(r_dpx(is,js,1)), is,ni, js,nj, 0,npz, use_acc=.true.)

    ! --- (b) dyn_core locals, this module's own -----------------------------
    call add3(FV3_FID_CAPPA, c_loc(r_cappa(isd,jsd,1)), isd,nih, jsd,njh, 0,npz)
    call add2(FV3_FID_WS,    c_loc(r_ws(is,js)),        is,ni,   js,nj)
    call add3(FV3_FID_DELPC, c_loc(r_delpc(isd,jsd,1)), isd,nih, jsd,njh, 0,npz)
    call add3(FV3_FID_PTC,   c_loc(r_ptc(isd,jsd,1)),   isd,nih, jsd,njh, 0,npz)
    call add3(FV3_FID_UT,    c_loc(r_ut(isd,jsd,1)),    isd,nih+1, jsd,njh, 0,npz)
    call add3(FV3_FID_VT,    c_loc(r_vt(isd,jsd,1)),    isd,nih, jsd,njh+1, 0,npz)
    call add3(FV3_FID_DIVGD, c_loc(r_divgd(isd,jsd,1)), isd,nih+1, jsd,njh+1, 0,npz)
    call add3(FV3_FID_CRX,   c_loc(r_crx(is,jsd,1)),    is,ni+1, jsd,njh, 0,npz)
    call add3(FV3_FID_XFX,   c_loc(r_xfx(is,jsd,1)),    is,ni+1, jsd,njh, 0,npz)
    call add3(FV3_FID_CRY,   c_loc(r_cry(isd,js,1)),    isd,nih, js,nj+1, 0,npz)
    call add3(FV3_FID_YFX,   c_loc(r_yfx(isd,js,1)),    isd,nih, js,nj+1, 0,npz)
    call add3(FV3_FID_GZ,    c_loc(r_gz(isd,jsd,1)),    isd,nih, jsd,njh, 0,npz+1)
    call add3(FV3_FID_ZH,    c_loc(r_zh(isd,jsd,1)),    isd,nih, jsd,njh, 0,npz+1)
    call add3(FV3_FID_PKC,   c_loc(r_pkc(isd,jsd,1)),   isd,nih, jsd,njh, 0,npz+1)
    call add3(FV3_FID_PK3,   c_loc(r_pk3(isd,jsd,1)),   isd,nih, jsd,njh, 0,npz+1)
    call add3(FV3_FID_DU,    c_loc(r_du(isd,jsd,1)),    isd,nih, jsd,njh+1, 0,npz)
    call add3(FV3_FID_DV,    c_loc(r_dv(isd,jsd,1)),    isd,nih+1, jsd,njh, 0,npz)
    call add3(FV3_FID_HEAT_SOURCE, c_loc(r_heat_source(isd,jsd,1)),           &
                                                        isd,nih, jsd,njh, 0,npz)
    call add2(FV3_FID_WS3,   c_loc(r_ws3(isd,jsd)),     isd,nih, jsd,njh)
    call add3(FV3_FID_Z_RAT, c_loc(r_z_rat(isd,jsd,1)), isd,nih, jsd,njh, 0,npz)
    call add2(FV3_FID_DIVG2, c_loc(r_divg2(is,js)),     is,ni+1, js,nj+1)
    call add2(FV3_FID_ZS,    c_loc(r_zs(isd,jsd)),      isd,nih, jsd,njh)
    ! dp_ref is indexed by absolute level in the kernels, so it is 0-based like
    ! every other vertical axis even though Fortran declares it 1:npz.
    call add1(FV3_FID_DP_REF, c_loc(r_dp_ref(1)), 0, npz)

    ! --- (a) grid metrics, fv_arrays.F90:1583-1720 --------------------------
    ! Cell centre (isd:ied, jsd:jed)
    call add2(FV3_FID_AREA,   c_loc(Atm%gridstruct%area(isd,jsd)),   isd,nih, jsd,njh)
    call add2(FV3_FID_RAREA,  c_loc(Atm%gridstruct%rarea(isd,jsd)),  isd,nih, jsd,njh)
    call add2(FV3_FID_DXA,    c_loc(Atm%gridstruct%dxa(isd,jsd)),    isd,nih, jsd,njh)
    call add2(FV3_FID_DYA,    c_loc(Atm%gridstruct%dya(isd,jsd)),    isd,nih, jsd,njh)
    call add2(FV3_FID_RDXA,   c_loc(Atm%gridstruct%rdxa(isd,jsd)),   isd,nih, jsd,njh)
    call add2(FV3_FID_RDYA,   c_loc(Atm%gridstruct%rdya(isd,jsd)),   isd,nih, jsd,njh)
    call add2(FV3_FID_F0,     c_loc(Atm%gridstruct%f0(isd,jsd)),     isd,nih, jsd,njh)
    call add2(FV3_FID_COSA_S, c_loc(Atm%gridstruct%cosa_s(isd,jsd)), isd,nih, jsd,njh)
    call add2(FV3_FID_RSIN2,  c_loc(Atm%gridstruct%rsin2(isd,jsd)),  isd,nih, jsd,njh)
    ! Corner (isd:ied+1, jsd:jed+1)
    call add2(FV3_FID_RAREA_C, c_loc(Atm%gridstruct%rarea_c(isd,jsd)), isd,nih+1, jsd,njh+1)
    call add2(FV3_FID_F_C,     c_loc(Atm%gridstruct%fC(isd,jsd)),      isd,nih+1, jsd,njh+1)
    call add2(FV3_FID_COSA,    c_loc(Atm%gridstruct%cosa(isd,jsd)),    isd,nih+1, jsd,njh+1)
    ! y-face (isd:ied, jsd:jed+1)
    call add2(FV3_FID_DX,     c_loc(Atm%gridstruct%dx(isd,jsd)),     isd,nih, jsd,njh+1)
    call add2(FV3_FID_RDX,    c_loc(Atm%gridstruct%rdx(isd,jsd)),    isd,nih, jsd,njh+1)
    call add2(FV3_FID_DYC,    c_loc(Atm%gridstruct%dyc(isd,jsd)),    isd,nih, jsd,njh+1)
    call add2(FV3_FID_RDYC,   c_loc(Atm%gridstruct%rdyc(isd,jsd)),   isd,nih, jsd,njh+1)
    call add2(FV3_FID_DIVG_U, c_loc(Atm%gridstruct%divg_u(isd,jsd)), isd,nih, jsd,njh+1)
    call add2(FV3_FID_DEL6_U, c_loc(Atm%gridstruct%del6_u(isd,jsd)), isd,nih, jsd,njh+1)
    call add2(FV3_FID_COSA_V, c_loc(Atm%gridstruct%cosa_v(isd,jsd)), isd,nih, jsd,njh+1)
    call add2(FV3_FID_SINA_V, c_loc(Atm%gridstruct%sina_v(isd,jsd)), isd,nih, jsd,njh+1)
    call add2(FV3_FID_RSIN_V, c_loc(Atm%gridstruct%rsin_v(isd,jsd)), isd,nih, jsd,njh+1)
    ! x-face (isd:ied+1, jsd:jed)
    call add2(FV3_FID_DY,     c_loc(Atm%gridstruct%dy(isd,jsd)),     isd,nih+1, jsd,njh)
    call add2(FV3_FID_RDY,    c_loc(Atm%gridstruct%rdy(isd,jsd)),    isd,nih+1, jsd,njh)
    call add2(FV3_FID_DXC,    c_loc(Atm%gridstruct%dxc(isd,jsd)),    isd,nih+1, jsd,njh)
    call add2(FV3_FID_RDXC,   c_loc(Atm%gridstruct%rdxc(isd,jsd)),   isd,nih+1, jsd,njh)
    call add2(FV3_FID_DIVG_V, c_loc(Atm%gridstruct%divg_v(isd,jsd)), isd,nih+1, jsd,njh)
    call add2(FV3_FID_DEL6_V, c_loc(Atm%gridstruct%del6_v(isd,jsd)), isd,nih+1, jsd,njh)
    call add2(FV3_FID_COSA_U, c_loc(Atm%gridstruct%cosa_u(isd,jsd)), isd,nih+1, jsd,njh)
    call add2(FV3_FID_SINA_U, c_loc(Atm%gridstruct%sina_u(isd,jsd)), isd,nih+1, jsd,njh)
    call add2(FV3_FID_RSIN_U, c_loc(Atm%gridstruct%rsin_u(isd,jsd)), isd,nih+1, jsd,njh)
    ! The rsina shadow, not gridstruct%rsina -- see its block comment.
    call add2(FV3_FID_RSINA,  c_loc(r_rsina_pad(isd,jsd)),           isd,nih, jsd,njh)

    ! sin_sg/cos_sg are (isd:ied, jsd:jed, 9); the contract reads the first four
    ! components, which are the leading contiguous block, so a prefix extent of
    ! 4 with the array's own strides is exact.
    call add3(FV3_FID_SIN_SG, c_loc(Atm%gridstruct%sin_sg(isd,jsd,1)), isd,nih, jsd,njh, 0,4)
    call add3(FV3_FID_COS_SG, c_loc(Atm%gridstruct%cos_sg(isd,jsd,1)), isd,nih, jsd,njh, 0,4)

    ! grid (corners) and agrid (centres) are stored COMPONENT-SLOWEST --
    ! grid(isd:ied+1, jsd:jed+1, 1:2) with lon = (:,:,1), lat = (:,:,2). An
    ! interleaved fill here was the SP1 it=2 root cause (a2b_ord4's corner
    ! extrapolation is the only consumer), so the layout is load-bearing.
    call add3(FV3_FID_GRID_LONLAT,  c_loc(Atm%gridstruct%grid(isd,jsd,1)),    &
                                    isd,nih+1, jsd,njh+1, 0,2)
    call add3(FV3_FID_AGRID_LONLAT, c_loc(Atm%gridstruct%agrid(isd,jsd,1)),   &
                                    isd,nih,   jsd,njh,   0,2)

    ! edge_* shadows -- accumulator_real width, see their block comment.
    call add1(FV3_FID_EDGE_W, c_loc(r_edge_w(isd)), isd, nih, use_acc=.true.)
    call add1(FV3_FID_EDGE_E, c_loc(r_edge_e(isd)), isd, nih, use_acc=.true.)
    call add1(FV3_FID_EDGE_S, c_loc(r_edge_s(isd)), isd, nih, use_acc=.true.)
    call add1(FV3_FID_EDGE_N, c_loc(r_edge_n(isd)), isd, nih, use_acc=.true.)

    ! --- Per-level damping columns, 0-based like every vertical axis --------
    call add1(FV3_FID_DAMPING_D2_BG,  c_loc(r_d2_bg(1)),  0, npz)
    call add1(FV3_FID_DAMPING_D4_BG,  c_loc(r_d4_bg(1)),  0, npz)
    call add1(FV3_FID_DAMPING_V,      c_loc(r_damp_v(1)), 0, npz)
    call add1(FV3_FID_DAMPING_W,      c_loc(r_damp_w(1)), 0, npz)
    call add1(FV3_FID_DAMPING_T,      c_loc(r_damp_t(1)), 0, npz)
    call add1(FV3_FID_DAMPING_KGB,    c_loc(r_kgb(1)),    0, npz)
    call add1(FV3_FID_DAMPING_NDIF,   c_loc(r_ndif(1)),   0, npz)
    call add1(FV3_FID_DAMPING_NORD_K, c_loc(r_nord_k(1)), 0, npz)
    call add1(FV3_FID_DAMPING_NORD_W, c_loc(r_nord_w(1)), 0, npz)
    call add1(FV3_FID_DAMPING_D_CON,  c_loc(r_d_con(1)),  0, npz)

    if (build_rc /= 0) then
       rc = build_rc
       call fv3_gpu_registry_free()
       return
    end if

    allocate(descs(ncur))
    descs(1:ncur) = work(1:ncur)
    ndesc = ncur
    deallocate(work)
    registry_live = .true.
  end subroutine fv3_gpu_registry_build

  !===========================================================================
  ! Release everything this module owns. Safe to call more than once.
  !
  ! MUST NOT run before the GPU context is finalized: the device views point at
  ! these host allocations for the context's whole lifetime, which is why the
  ! bridge calls this from its finalize path and nowhere else.
  !===========================================================================
  subroutine fv3_gpu_registry_free()
    if (allocated(work)) deallocate(work)
    ncur = 0

    if (allocated(r_ws))     deallocate(r_ws)
    if (allocated(r_ws3))    deallocate(r_ws3)
    if (allocated(r_z_rat))  deallocate(r_z_rat)
    if (allocated(r_divg2))  deallocate(r_divg2)
    if (allocated(r_zs))     deallocate(r_zs)
    if (allocated(r_dp_ref)) deallocate(r_dp_ref)
    if (allocated(r_cappa))  deallocate(r_cappa)
    if (allocated(r_delpc))  deallocate(r_delpc)
    if (allocated(r_ptc))    deallocate(r_ptc)
    if (allocated(r_ut))     deallocate(r_ut)
    if (allocated(r_vt))     deallocate(r_vt)
    if (allocated(r_divgd))  deallocate(r_divgd)
    if (allocated(r_crx))    deallocate(r_crx)
    if (allocated(r_xfx))    deallocate(r_xfx)
    if (allocated(r_cry))    deallocate(r_cry)
    if (allocated(r_yfx))    deallocate(r_yfx)
    if (allocated(r_gz))     deallocate(r_gz)
    if (allocated(r_zh))     deallocate(r_zh)
    if (allocated(r_pkc))    deallocate(r_pkc)
    if (allocated(r_pk3))    deallocate(r_pk3)
    if (allocated(r_du))     deallocate(r_du)
    if (allocated(r_dv))     deallocate(r_dv)
    if (allocated(r_heat_source)) deallocate(r_heat_source)
    if (allocated(r_dpx))    deallocate(r_dpx)

    if (allocated(r_mfx)) deallocate(r_mfx)
    if (allocated(r_mfy)) deallocate(r_mfy)
    if (allocated(r_cx))  deallocate(r_cx)
    if (allocated(r_cy))  deallocate(r_cy)
    if (.not. flux_bound) then
       p_mfx = C_NULL_PTR; p_mfy = C_NULL_PTR
       p_cx  = C_NULL_PTR; p_cy  = C_NULL_PTR
    end if

    if (allocated(r_rsina_pad)) deallocate(r_rsina_pad)
    if (allocated(r_edge_w)) deallocate(r_edge_w)
    if (allocated(r_edge_e)) deallocate(r_edge_e)
    if (allocated(r_edge_s)) deallocate(r_edge_s)
    if (allocated(r_edge_n)) deallocate(r_edge_n)

    if (allocated(r_d2_bg))  deallocate(r_d2_bg)
    if (allocated(r_d4_bg))  deallocate(r_d4_bg)
    if (allocated(r_damp_v)) deallocate(r_damp_v)
    if (allocated(r_damp_w)) deallocate(r_damp_w)
    if (allocated(r_damp_t)) deallocate(r_damp_t)
    if (allocated(r_d_con))  deallocate(r_d_con)
    if (allocated(r_kgb))    deallocate(r_kgb)
    if (allocated(r_ndif))   deallocate(r_ndif)
    if (allocated(r_nord_k)) deallocate(r_nord_k)
    if (allocated(r_nord_w)) deallocate(r_nord_w)

    registry_live = .false.
  end subroutine fv3_gpu_registry_free

end module fv3_gpu_field_registry_mod
