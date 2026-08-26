!------------------------------------------------------------------------------
! FV3 GPU Interface Module
!------------------------------------------------------------------------------
! Fortran ISO_C_BINDING wrapper for the WP-E CUDA dyn_core subsystem.
!
! This module provides:
!   - fv3_gpu_interface_init: Create GPU context, register fields
!   - fv3_gpu_interface_step: Execute one dyn_core time step
!   - fv3_gpu_interface_sync: Synchronize selected fields to host
!   - fv3_gpu_interface_finalize: Release GPU resources
!
! The module maps FV3 state arrays to the stable WP-E C ABI and manages
! GPU context lifetime.
!------------------------------------------------------------------------------

module fv3_gpu_interface_mod

  use iso_c_binding
  use fv_arrays_mod, only: fv_atmos_type, FVPRC, REAL4, REAL8, R_GRID

  implicit none
  private

  public :: fv3_gpu_interface_init
  public :: fv3_gpu_interface_step
  public :: fv3_gpu_interface_sync
  public :: fv3_gpu_interface_finalize
  public :: fv3_gpu_interface_available

  ! ABI constants (must match fv3_gpu_api.h)
  integer(c_int32_t), parameter :: FV3_GPU_ABI_VERSION = 1
  integer(c_int32_t), parameter :: FV3_GPU_MAX_RANK = 4
  integer(c_int32_t), parameter :: FV3_GPU_ERROR_MESSAGE_BYTES = 256
  integer(c_int32_t), parameter :: FV3_GPU_FIELD_NAME_BYTES = 32

  ! Status codes
  integer(c_int32_t), parameter, public :: FV3_GPU_SUCCESS = 0
  integer(c_int32_t), parameter, public :: FV3_GPU_INVALID_ARGUMENT = 1
  integer(c_int32_t), parameter, public :: FV3_GPU_UNSUPPORTED_CONFIGURATION = 2
  integer(c_int32_t), parameter, public :: FV3_GPU_CUDA_ERROR = 3
  integer(c_int32_t), parameter, public :: FV3_GPU_MPI_ERROR = 4
  integer(c_int32_t), parameter, public :: FV3_GPU_OUT_OF_MEMORY = 5
  integer(c_int32_t), parameter, public :: FV3_GPU_STATE_CONFLICT = 6
  integer(c_int32_t), parameter, public :: FV3_GPU_INTERNAL_ERROR = 7

  ! Element types
  integer(c_int32_t), parameter :: FV3_GPU_ELEMENT_F32 = 1
  integer(c_int32_t), parameter :: FV3_GPU_ELEMENT_F64 = 2

  ! Memory locations
  integer(c_int32_t), parameter :: FV3_GPU_MEMORY_HOST = 1
  integer(c_int32_t), parameter :: FV3_GPU_MEMORY_DEVICE = 2

  ! Stagger types
  integer(c_int32_t), parameter :: FV3_GPU_STAGGER_A = 1
  integer(c_int32_t), parameter :: FV3_GPU_STAGGER_C_U = 2
  integer(c_int32_t), parameter :: FV3_GPU_STAGGER_C_V = 3
  integer(c_int32_t), parameter :: FV3_GPU_STAGGER_D_U = 4
  integer(c_int32_t), parameter :: FV3_GPU_STAGGER_D_V = 5
  integer(c_int32_t), parameter :: FV3_GPU_STAGGER_B = 6
  integer(c_int32_t), parameter :: FV3_GPU_STAGGER_INTERFACE = 7
  integer(c_int32_t), parameter :: FV3_GPU_STAGGER_X_FACE = 8
  integer(c_int32_t), parameter :: FV3_GPU_STAGGER_Y_FACE = 9

  ! Access modes
  integer(c_int32_t), parameter :: FV3_GPU_ACCESS_INPUT = 1
  integer(c_int32_t), parameter :: FV3_GPU_ACCESS_OUTPUT = 2
  integer(c_int32_t), parameter :: FV3_GPU_ACCESS_INOUT = 3

  ! Precision modes
  integer(c_int32_t), parameter :: FV3_GPU_PRECISION_R4 = 1
  integer(c_int32_t), parameter :: FV3_GPU_PRECISION_R8 = 2

  ! FieldId enumeration (must match fv3_gpu_internal.hpp)
  integer(c_int32_t), parameter :: FIELD_U = 1
  integer(c_int32_t), parameter :: FIELD_V = 2
  integer(c_int32_t), parameter :: FIELD_W = 3
  integer(c_int32_t), parameter :: FIELD_DELZ = 4
  integer(c_int32_t), parameter :: FIELD_PT = 5
  integer(c_int32_t), parameter :: FIELD_DELP = 6
  integer(c_int32_t), parameter :: FIELD_Q = 7
  integer(c_int32_t), parameter :: FIELD_Q_CON = 8
  integer(c_int32_t), parameter :: FIELD_CAPPA = 9
  integer(c_int32_t), parameter :: FIELD_PE = 10
  integer(c_int32_t), parameter :: FIELD_PK = 11
  integer(c_int32_t), parameter :: FIELD_PELN = 12
  integer(c_int32_t), parameter :: FIELD_PKZ = 13
  integer(c_int32_t), parameter :: FIELD_PHIS = 14
  integer(c_int32_t), parameter :: FIELD_OMGA = 15
  integer(c_int32_t), parameter :: FIELD_UA = 16
  integer(c_int32_t), parameter :: FIELD_VA = 17
  integer(c_int32_t), parameter :: FIELD_UC = 18
  integer(c_int32_t), parameter :: FIELD_VC = 19
  integer(c_int32_t), parameter :: FIELD_MFX = 20
  integer(c_int32_t), parameter :: FIELD_MFY = 21
  integer(c_int32_t), parameter :: FIELD_CX = 22
  integer(c_int32_t), parameter :: FIELD_CY = 23
  integer(c_int32_t), parameter :: FIELD_DPX = 24
  integer(c_int32_t), parameter :: FIELD_DISS_EST = 25
  integer(c_int32_t), parameter :: FIELD_AK = 44
  integer(c_int32_t), parameter :: FIELD_BK = 45
  integer(c_int32_t), parameter :: FIELD_AREA = 48
  integer(c_int32_t), parameter :: FIELD_RAREA = 49

  ! C interoperable types
  type, bind(c) :: fv3_gpu_error_t
    integer(c_int32_t) :: abi_version
    integer(c_int32_t) :: struct_bytes
    integer(c_int32_t) :: code
    integer(c_int32_t) :: native_code
    integer(c_int32_t) :: rank
    integer(c_int32_t) :: device
    integer(c_int64_t) :: sequence
    character(kind=c_char) :: api(FV3_GPU_FIELD_NAME_BYTES)
    character(kind=c_char) :: message(FV3_GPU_ERROR_MESSAGE_BYTES)
  end type fv3_gpu_error_t

  type, bind(c) :: fv3_gpu_field_desc_t
    integer(c_int32_t) :: abi_version
    integer(c_int32_t) :: struct_bytes
    integer(c_int32_t) :: field_id
    character(kind=c_char) :: name(FV3_GPU_FIELD_NAME_BYTES)
    type(c_ptr) :: base
    integer(c_int64_t) :: allocation_bytes
    integer(c_int32_t) :: element_type
    integer(c_int32_t) :: element_bytes
    integer(c_int32_t) :: rank
    integer(c_int32_t) :: memory_location
    integer(c_int64_t) :: lower(FV3_GPU_MAX_RANK)
    integer(c_int64_t) :: extent(FV3_GPU_MAX_RANK)
    integer(c_int64_t) :: stride_elements(FV3_GPU_MAX_RANK)
    integer(c_int64_t) :: compute_lower(FV3_GPU_MAX_RANK)
    integer(c_int64_t) :: compute_upper(FV3_GPU_MAX_RANK)
    integer(c_int32_t) :: halo_lower(FV3_GPU_MAX_RANK)
    integer(c_int32_t) :: halo_upper(FV3_GPU_MAX_RANK)
    integer(c_int32_t) :: stagger
    integer(c_int32_t) :: access
    integer(c_int32_t) :: alignment_bytes
    integer(c_int32_t) :: reserved
  end type fv3_gpu_field_desc_t

  type, bind(c) :: fv3_gpu_create_desc_t
    integer(c_int32_t) :: abi_version
    integer(c_int32_t) :: struct_bytes
    integer(c_int32_t) :: device_ordinal
    integer(c_int32_t) :: mpi_rank
    integer(c_int32_t) :: mpi_size
    integer(c_int32_t) :: precision
    integer(c_int64_t) :: is, ie, js, je, npz
    integer(c_int32_t) :: halo_width
    integer(c_int32_t) :: flags
    integer(c_intptr_t) :: communicator_token
  end type fv3_gpu_create_desc_t

  type, bind(c) :: fv3_gpu_state_desc_t
    integer(c_int32_t) :: abi_version
    integer(c_int32_t) :: struct_bytes
    integer(c_int32_t) :: field_count
    integer(c_int32_t) :: reserved
    type(c_ptr) :: fields
  end type fv3_gpu_state_desc_t

  type, bind(c) :: fv3_gpu_field_mask_t
    integer(c_int32_t) :: abi_version
    integer(c_int32_t) :: struct_bytes
    integer(c_int64_t) :: words(4)
  end type fv3_gpu_field_mask_t

  type, bind(c) :: fv3_gpu_step_desc_t
    integer(c_int32_t) :: abi_version
    integer(c_int32_t) :: struct_bytes
    real(c_double) :: bdt
    integer(c_int32_t) :: k_split
    integer(c_int32_t) :: n_split
    integer(c_int8_t) :: init_step
    integer(c_int8_t) :: end_step
    integer(c_int16_t) :: reserved0
    integer(c_int64_t) :: model_step
    integer(c_int64_t) :: acoustic_sequence
    integer(c_int64_t) :: validation_mask
    integer(c_int64_t) :: physics_tail_mask
    integer(c_int64_t) :: dependency
  end type fv3_gpu_step_desc_t

  type, bind(c) :: fv3_gpu_capabilities_t
    integer(c_int32_t) :: abi_version
    integer(c_int32_t) :: struct_bytes
    integer(c_int32_t) :: compiled_with_cuda
    integer(c_int32_t) :: device_count
    integer(c_int32_t) :: precision_mask
    integer(c_int32_t) :: communication_stage
    integer(c_int64_t) :: feature_mask
  end type fv3_gpu_capabilities_t

  ! C function interfaces
  interface

    function fv3_gpu_create(desc, ctx_ptr, err) bind(c, name='fv3_gpu_create')
      import :: c_int, c_ptr, fv3_gpu_create_desc_t, fv3_gpu_error_t
      type(fv3_gpu_create_desc_t), intent(in) :: desc
      type(c_ptr), intent(out) :: ctx_ptr
      type(fv3_gpu_error_t), intent(out) :: err
      integer(c_int) :: fv3_gpu_create
    end function

    function fv3_gpu_initialize_state(ctx, state_desc, event, err) &
        bind(c, name='fv3_gpu_initialize_state')
      import :: c_int, c_ptr, c_int64_t, fv3_gpu_state_desc_t, fv3_gpu_error_t
      type(c_ptr), value :: ctx
      type(fv3_gpu_state_desc_t), intent(in) :: state_desc
      integer(c_int64_t), intent(out) :: event
      type(fv3_gpu_error_t), intent(out) :: err
      integer(c_int) :: fv3_gpu_initialize_state
    end function

    function fv3_gpu_sync_from_host(ctx, mask, event, err) &
        bind(c, name='fv3_gpu_sync_from_host')
      import :: c_int, c_ptr, c_int64_t, fv3_gpu_field_mask_t, fv3_gpu_error_t
      type(c_ptr), value :: ctx
      type(fv3_gpu_field_mask_t), intent(in) :: mask
      integer(c_int64_t), intent(out) :: event
      type(fv3_gpu_error_t), intent(out) :: err
      integer(c_int) :: fv3_gpu_sync_from_host
    end function

    ! Milestone 3A.10: describe the cubed-sphere decomposition to the runtime.
    ! rank_of_tile is a 6-element table, entry n giving the MPI rank that owns
    ! mosaic tile n.
    function fv3_gpu_set_cubed_sphere_topology(ctx, tile, npx, npy, rank_of_tile, err) &
        bind(c, name='fv3_gpu_set_cubed_sphere_topology')
      import :: c_int, c_ptr, c_int32_t, c_int64_t, fv3_gpu_error_t
      type(c_ptr), value :: ctx
      integer(c_int32_t), value :: tile
      integer(c_int64_t), value :: npx
      integer(c_int64_t), value :: npy
      integer(c_int32_t), intent(in) :: rank_of_tile(6)
      type(fv3_gpu_error_t), intent(out) :: err
      integer(c_int) :: fv3_gpu_set_cubed_sphere_topology
    end function

    function fv3_gpu_dyn_core_step(ctx, step_desc, event, err) &
        bind(c, name='fv3_gpu_dyn_core_step')
      import :: c_int, c_ptr, c_int64_t, fv3_gpu_step_desc_t, fv3_gpu_error_t
      type(c_ptr), value :: ctx
      type(fv3_gpu_step_desc_t), intent(in) :: step_desc
      integer(c_int64_t), intent(out) :: event
      type(fv3_gpu_error_t), intent(out) :: err
      integer(c_int) :: fv3_gpu_dyn_core_step
    end function

    function fv3_gpu_sync_to_host(ctx, mask, event, err) &
        bind(c, name='fv3_gpu_sync_to_host')
      import :: c_int, c_ptr, c_int64_t, fv3_gpu_field_mask_t, fv3_gpu_error_t
      type(c_ptr), value :: ctx
      type(fv3_gpu_field_mask_t), intent(in) :: mask
      integer(c_int64_t), intent(out) :: event
      type(fv3_gpu_error_t), intent(out) :: err
      integer(c_int) :: fv3_gpu_sync_to_host
    end function

    function fv3_gpu_event_wait(ctx, event, err) &
        bind(c, name='fv3_gpu_event_wait')
      import :: c_int, c_ptr, c_int64_t, fv3_gpu_error_t
      type(c_ptr), value :: ctx
      integer(c_int64_t), value :: event
      type(fv3_gpu_error_t), intent(out) :: err
      integer(c_int) :: fv3_gpu_event_wait
    end function

    function fv3_gpu_get_capabilities(ctx, caps) &
        bind(c, name='fv3_gpu_get_capabilities')
      import :: c_int, c_ptr, fv3_gpu_capabilities_t
      type(c_ptr), value :: ctx
      type(fv3_gpu_capabilities_t), intent(out) :: caps
      integer(c_int) :: fv3_gpu_get_capabilities
    end function

    function fv3_gpu_finalize(ctx_ptr, err) bind(c, name='fv3_gpu_finalize')
      import :: c_int, c_ptr, fv3_gpu_error_t
      type(c_ptr), intent(inout) :: ctx_ptr
      type(fv3_gpu_error_t), intent(out) :: err
      integer(c_int) :: fv3_gpu_finalize
    end function

  end interface

  ! Module state
  type(c_ptr), save :: gpu_context = c_null_ptr
  logical, save :: context_initialized = .false.
  integer(c_int64_t), save :: model_step_counter = 0

  ! Field descriptor storage (max 128 fields)
  integer, parameter :: MAX_FIELDS = 128
  type(fv3_gpu_field_desc_t), allocatable, save :: field_descs(:)
  integer, save :: field_count = 0

contains

  !----------------------------------------------------------------------------
  ! Check if GPU interface is available (compiled with CUDA support)
  !----------------------------------------------------------------------------
  function fv3_gpu_interface_available() result(available)
    logical :: available
#ifdef USE_FV3_CUDA
    available = .true.
#else
    available = .false.
#endif
  end function fv3_gpu_interface_available

  !----------------------------------------------------------------------------
  ! Initialize GPU context and register all fields
  !----------------------------------------------------------------------------
  subroutine fv3_gpu_interface_init(Atm, comm, mpi_rank, mpi_size, rc)
    type(fv_atmos_type), intent(in) :: Atm
    integer, intent(in) :: comm
    integer, intent(in) :: mpi_rank
    integer, intent(in) :: mpi_size
    integer, intent(out) :: rc

    type(fv3_gpu_create_desc_t) :: create_desc
    type(fv3_gpu_state_desc_t) :: state_desc
    type(fv3_gpu_error_t) :: err
    integer(c_int64_t) :: event
    integer :: status
    integer(c_int32_t) :: rank_of_tile(6)
    integer :: t

    rc = FV3_GPU_SUCCESS

#ifndef USE_FV3_CUDA
    rc = FV3_GPU_UNSUPPORTED_CONFIGURATION
    return
#endif

    if (context_initialized) then
      rc = FV3_GPU_STATE_CONFLICT
      return
    endif

    ! Allocate field descriptor storage
    if (.not. allocated(field_descs)) then
      allocate(field_descs(MAX_FIELDS))
    endif
    field_count = 0

    ! Fill create descriptor
    create_desc%abi_version = FV3_GPU_ABI_VERSION
    create_desc%struct_bytes = c_sizeof(create_desc)
    create_desc%device_ordinal = 0
    create_desc%mpi_rank = mpi_rank
    create_desc%mpi_size = mpi_size
#ifdef SINGLE_FV
    create_desc%precision = FV3_GPU_PRECISION_R4
#else
    create_desc%precision = FV3_GPU_PRECISION_R8
#endif
    create_desc%is = Atm%bd%isc
    create_desc%ie = Atm%bd%iec
    create_desc%js = Atm%bd%jsc
    create_desc%je = Atm%bd%jec
    create_desc%npz = Atm%npz
    create_desc%halo_width = Atm%ng
    create_desc%flags = 0
    create_desc%communicator_token = comm

    ! Create GPU context
    status = fv3_gpu_create(create_desc, gpu_context, err)
    if (status /= 0) then
      rc = err%code
      return
    endif

    ! Milestone 3A.10: tell the runtime where this rank's tile sits on the cube.
    !
    ! This must happen before any step, because the runtime now refuses to
    ! declare a halo group ready when it has not been told its own connectivity.
    ! Before 3A.10 nothing supplied this, and the runtime's default topology was
    ! read as "no neighbours, nothing to exchange" -- free halo readiness on a
    ! decomposition that in fact has four panel seams.
    !
    ! Only the one-rank-per-tile decomposition is supported. Anything else is
    ! rejected HERE, before any device state is created or transferred, rather
    ! than silently mapped onto a connectivity that does not describe it.
    if (mpi_size /= 6) then
      rc = FV3_GPU_UNSUPPORTED_CONFIGURATION
      return
    endif
    if (Atm%tile /= mpi_rank + 1) then
      ! The seam algebra below assumes FMS's one-PE-per-tile assignment,
      ! pe = tile-1. If this run does not use it, the rank table would be a
      ! guess, so refuse instead of guessing.
      rc = FV3_GPU_UNSUPPORTED_CONFIGURATION
      return
    endif
    do t = 1, 6
      rank_of_tile(t) = int(t - 1, c_int32_t)
    enddo
    status = fv3_gpu_set_cubed_sphere_topology(gpu_context, int(Atm%tile, c_int32_t), &
                                               int(Atm%npx, c_int64_t), int(Atm%npy, c_int64_t), &
                                               rank_of_tile, err)
    if (status /= 0) then
      rc = err%code
      return
    endif

    ! Register all required fields
    call register_mutable_state_fields(Atm, status)
    if (status /= 0) then
      rc = status
      return
    endif

    call register_immutable_grid_fields(Atm, status)
    if (status /= 0) then
      rc = status
      return
    endif

    ! Initialize state (performs H2D transfer)
    state_desc%abi_version = FV3_GPU_ABI_VERSION
    state_desc%struct_bytes = c_sizeof(state_desc)
    state_desc%field_count = field_count
    state_desc%reserved = 0
    state_desc%fields = c_loc(field_descs(1))

    status = fv3_gpu_initialize_state(gpu_context, state_desc, event, err)
    if (status /= 0) then
      rc = err%code
      return
    endif

    ! Wait for initialization to complete
    status = fv3_gpu_event_wait(gpu_context, event, err)
    if (status /= 0) then
      rc = err%code
      return
    endif

    context_initialized = .true.
    model_step_counter = 0

  end subroutine fv3_gpu_interface_init

  !----------------------------------------------------------------------------
  ! Execute one dyn_core time step on GPU
  !----------------------------------------------------------------------------
  subroutine fv3_gpu_interface_step(Atm, bdt, k_split, n_split, &
                                    init_step, end_step, rc)
    type(fv_atmos_type), intent(inout) :: Atm
    real(FVPRC), intent(in) :: bdt
    integer, intent(in) :: k_split, n_split
    logical, intent(in) :: init_step, end_step
    integer, intent(out) :: rc

    type(fv3_gpu_step_desc_t) :: step_desc
    type(fv3_gpu_error_t) :: err
    integer(c_int64_t) :: event
    integer :: status

    rc = FV3_GPU_SUCCESS

#ifndef USE_FV3_CUDA
    rc = FV3_GPU_UNSUPPORTED_CONFIGURATION
    return
#endif

    if (.not. context_initialized) then
      rc = FV3_GPU_STATE_CONFLICT
      return
    endif

    ! Fill step descriptor
    step_desc%abi_version = FV3_GPU_ABI_VERSION
    step_desc%struct_bytes = c_sizeof(step_desc)
    step_desc%bdt = real(bdt, c_double)
    step_desc%k_split = k_split
    step_desc%n_split = n_split
    if (init_step) then
      step_desc%init_step = 1_c_int8_t
    else
      step_desc%init_step = 0_c_int8_t
    endif
    if (end_step) then
      step_desc%end_step = 1_c_int8_t
    else
      step_desc%end_step = 0_c_int8_t
    endif
    step_desc%reserved0 = 0
    step_desc%model_step = model_step_counter
    step_desc%acoustic_sequence = 0
    step_desc%validation_mask = 0
    step_desc%physics_tail_mask = 0
    step_desc%dependency = 0

    ! Execute dyn_core step
    status = fv3_gpu_dyn_core_step(gpu_context, step_desc, event, err)
    if (status /= 0) then
      rc = err%code
      return
    endif

    ! Wait for step to complete
    status = fv3_gpu_event_wait(gpu_context, event, err)
    if (status /= 0) then
      rc = err%code
      return
    endif

    model_step_counter = model_step_counter + 1

  end subroutine fv3_gpu_interface_step

  !----------------------------------------------------------------------------
  ! Synchronize fields back to host
  !----------------------------------------------------------------------------
  subroutine fv3_gpu_interface_sync(sync_all, rc)
    logical, intent(in) :: sync_all
    integer, intent(out) :: rc

    type(fv3_gpu_field_mask_t) :: mask
    type(fv3_gpu_error_t) :: err
    integer(c_int64_t) :: event
    integer :: status

    rc = FV3_GPU_SUCCESS

#ifndef USE_FV3_CUDA
    rc = FV3_GPU_UNSUPPORTED_CONFIGURATION
    return
#endif

    if (.not. context_initialized) then
      rc = FV3_GPU_STATE_CONFLICT
      return
    endif

    ! Build field mask (all bits set for sync_all)
    mask%abi_version = FV3_GPU_ABI_VERSION
    mask%struct_bytes = c_sizeof(mask)
    if (sync_all) then
      mask%words(1) = -1_c_int64_t
      mask%words(2) = -1_c_int64_t
      mask%words(3) = -1_c_int64_t
      mask%words(4) = -1_c_int64_t
    else
      mask%words(1) = 0
      mask%words(2) = 0
      mask%words(3) = 0
      mask%words(4) = 0
    endif

    status = fv3_gpu_sync_to_host(gpu_context, mask, event, err)
    if (status /= 0) then
      rc = err%code
      return
    endif

    status = fv3_gpu_event_wait(gpu_context, event, err)
    if (status /= 0) then
      rc = err%code
      return
    endif

  end subroutine fv3_gpu_interface_sync

  !----------------------------------------------------------------------------
  ! Finalize GPU context
  !----------------------------------------------------------------------------
  subroutine fv3_gpu_interface_finalize(rc)
    integer, intent(out) :: rc

    type(fv3_gpu_error_t) :: err
    integer :: status

    rc = FV3_GPU_SUCCESS

#ifndef USE_FV3_CUDA
    rc = FV3_GPU_UNSUPPORTED_CONFIGURATION
    return
#endif

    if (.not. context_initialized) then
      return
    endif

    status = fv3_gpu_finalize(gpu_context, err)
    if (status /= 0) then
      rc = err%code
    endif

    gpu_context = c_null_ptr
    context_initialized = .false.
    model_step_counter = 0

    if (allocated(field_descs)) then
      deallocate(field_descs)
    endif
    field_count = 0

  end subroutine fv3_gpu_interface_finalize

  !----------------------------------------------------------------------------
  ! Internal: Register mutable state fields
  !----------------------------------------------------------------------------
  subroutine register_mutable_state_fields(Atm, rc)
    type(fv_atmos_type), intent(in), target :: Atm
    integer, intent(out) :: rc

    integer :: isd, ied, jsd, jed, npz, ng
    integer :: isc, iec, jsc, jec

    rc = 0

    isd = Atm%bd%isd
    ied = Atm%bd%ied
    jsd = Atm%bd%jsd
    jed = Atm%bd%jed
    isc = Atm%bd%isc
    iec = Atm%bd%iec
    jsc = Atm%bd%jsc
    jec = Atm%bd%jec
    npz = Atm%npz
    ng = Atm%ng

    ! u: D-grid u-wind (isd:ied, jsd:jed+1, npz)
    call add_field_3d(FIELD_U, 'u', c_loc(Atm%u), &
        isd, ied, jsd, jed+1, 1, npz, &
        isc, iec, jsc, jec+1, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_D_U, FV3_GPU_ACCESS_INOUT)

    ! v: D-grid v-wind (isd:ied+1, jsd:jed, npz)
    call add_field_3d(FIELD_V, 'v', c_loc(Atm%v), &
        isd, ied+1, jsd, jed, 1, npz, &
        isc, iec+1, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_D_V, FV3_GPU_ACCESS_INOUT)

    ! w: vertical velocity (isd:ied, jsd:jed, npz)
    call add_field_3d(FIELD_W, 'w', c_loc(Atm%w), &
        isd, ied, jsd, jed, 1, npz, &
        isc, iec, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INOUT)

    ! delz: layer thickness (isd:ied, jsd:jed, npz)
    call add_field_3d(FIELD_DELZ, 'delz', c_loc(Atm%delz), &
        isd, ied, jsd, jed, 1, npz, &
        isc, iec, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INOUT)

    ! pt: potential temperature (isd:ied, jsd:jed, npz)
    call add_field_3d(FIELD_PT, 'pt', c_loc(Atm%pt), &
        isd, ied, jsd, jed, 1, npz, &
        isc, iec, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INOUT)

    ! delp: pressure thickness (isd:ied, jsd:jed, npz)
    call add_field_3d(FIELD_DELP, 'delp', c_loc(Atm%delp), &
        isd, ied, jsd, jed, 1, npz, &
        isc, iec, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INOUT)

    ! q_con: condensate (isd:ied, jsd:jed, npz)
    call add_field_3d(FIELD_Q_CON, 'q_con', c_loc(Atm%q_con), &
        isd, ied, jsd, jed, 1, npz, &
        isc, iec, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INOUT)

    ! omga: omega (isd:ied, jsd:jed, npz)
    call add_field_3d(FIELD_OMGA, 'omga', c_loc(Atm%omga), &
        isd, ied, jsd, jed, 1, npz, &
        isc, iec, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INOUT)

    ! ua: A-grid u-wind (isd:ied, jsd:jed, npz)
    call add_field_3d(FIELD_UA, 'ua', c_loc(Atm%ua), &
        isd, ied, jsd, jed, 1, npz, &
        isc, iec, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INOUT)

    ! va: A-grid v-wind (isd:ied, jsd:jed, npz)
    call add_field_3d(FIELD_VA, 'va', c_loc(Atm%va), &
        isd, ied, jsd, jed, 1, npz, &
        isc, iec, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INOUT)

    ! uc: C-grid u-wind (isd:ied+1, jsd:jed, npz)
    call add_field_3d(FIELD_UC, 'uc', c_loc(Atm%uc), &
        isd, ied+1, jsd, jed, 1, npz, &
        isc, iec+1, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_C_U, FV3_GPU_ACCESS_INOUT)

    ! vc: C-grid v-wind (isd:ied, jsd:jed+1, npz)
    call add_field_3d(FIELD_VC, 'vc', c_loc(Atm%vc), &
        isd, ied, jsd, jed+1, 1, npz, &
        isc, iec, jsc, jec+1, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_C_V, FV3_GPU_ACCESS_INOUT)

    ! diss_est: dissipation estimate (isd:ied, jsd:jed, npz)
    call add_field_3d(FIELD_DISS_EST, 'diss_est', c_loc(Atm%diss_est), &
        isd, ied, jsd, jed, 1, npz, &
        isc, iec, jsc, jec, 1, npz, &
        ng, ng, ng, ng, 0, 0, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INOUT)

  end subroutine register_mutable_state_fields

  !----------------------------------------------------------------------------
  ! Internal: Register immutable grid fields
  !----------------------------------------------------------------------------
  subroutine register_immutable_grid_fields(Atm, rc)
    type(fv_atmos_type), intent(in), target :: Atm
    integer, intent(out) :: rc

    integer :: isd, ied, jsd, jed, npz

    rc = 0

    isd = Atm%bd%isd
    ied = Atm%bd%ied
    jsd = Atm%bd%jsd
    jed = Atm%bd%jed
    npz = Atm%npz

    ! ak: hybrid A coefficient (npz+1)
    call add_field_1d(FIELD_AK, 'ak', c_loc(Atm%ak), &
        1, npz+1, FV3_GPU_STAGGER_INTERFACE, FV3_GPU_ACCESS_INPUT)

    ! bk: hybrid B coefficient (npz+1)
    call add_field_1d(FIELD_BK, 'bk', c_loc(Atm%bk), &
        1, npz+1, FV3_GPU_STAGGER_INTERFACE, FV3_GPU_ACCESS_INPUT)

    ! phis: surface geopotential (isd:ied, jsd:jed)
    call add_field_2d(FIELD_PHIS, 'phis', c_loc(Atm%phis), &
        isd, ied, jsd, jed, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INPUT)

    ! area: cell area (isd:ied, jsd:jed)
    call add_field_2d(FIELD_AREA, 'area', c_loc(Atm%gridstruct%area), &
        isd, ied, jsd, jed, &
        FV3_GPU_STAGGER_A, FV3_GPU_ACCESS_INPUT)

  end subroutine register_immutable_grid_fields

  !----------------------------------------------------------------------------
  ! Internal: Add 1D field descriptor
  !----------------------------------------------------------------------------
  subroutine add_field_1d(id, name, ptr, lo, hi, stagger, access)
    integer(c_int32_t), intent(in) :: id
    character(*), intent(in) :: name
    type(c_ptr), intent(in) :: ptr
    integer, intent(in) :: lo, hi
    integer(c_int32_t), intent(in) :: stagger, access

    integer :: n, i

    field_count = field_count + 1
    n = field_count

    field_descs(n)%abi_version = FV3_GPU_ABI_VERSION
    field_descs(n)%struct_bytes = c_sizeof(field_descs(n))
    field_descs(n)%field_id = id

    ! Copy name
    do i = 1, min(len_trim(name), FV3_GPU_FIELD_NAME_BYTES-1)
      field_descs(n)%name(i) = name(i:i)
    enddo
    field_descs(n)%name(min(len_trim(name)+1, FV3_GPU_FIELD_NAME_BYTES)) = c_null_char

    field_descs(n)%base = ptr
    field_descs(n)%allocation_bytes = 0
#ifdef SINGLE_FV
    field_descs(n)%element_type = FV3_GPU_ELEMENT_F32
    field_descs(n)%element_bytes = 4
#else
    field_descs(n)%element_type = FV3_GPU_ELEMENT_F64
    field_descs(n)%element_bytes = 8
#endif
    field_descs(n)%rank = 1
    field_descs(n)%memory_location = FV3_GPU_MEMORY_HOST

    field_descs(n)%lower(1) = lo
    field_descs(n)%extent(1) = hi - lo + 1
    field_descs(n)%stride_elements(1) = 1

    field_descs(n)%lower(2:4) = 0
    field_descs(n)%extent(2:4) = 0
    field_descs(n)%stride_elements(2:4) = 0

    field_descs(n)%compute_lower(1) = lo
    field_descs(n)%compute_upper(1) = hi
    field_descs(n)%compute_lower(2:4) = 0
    field_descs(n)%compute_upper(2:4) = 0

    field_descs(n)%halo_lower(:) = 0
    field_descs(n)%halo_upper(:) = 0

    field_descs(n)%stagger = stagger
    field_descs(n)%access = access
    field_descs(n)%alignment_bytes = 8
    field_descs(n)%reserved = 0

  end subroutine add_field_1d

  !----------------------------------------------------------------------------
  ! Internal: Add 2D field descriptor
  !----------------------------------------------------------------------------
  subroutine add_field_2d(id, name, ptr, i_lo, i_hi, j_lo, j_hi, stagger, access)
    integer(c_int32_t), intent(in) :: id
    character(*), intent(in) :: name
    type(c_ptr), intent(in) :: ptr
    integer, intent(in) :: i_lo, i_hi, j_lo, j_hi
    integer(c_int32_t), intent(in) :: stagger, access

    integer :: n, i

    field_count = field_count + 1
    n = field_count

    field_descs(n)%abi_version = FV3_GPU_ABI_VERSION
    field_descs(n)%struct_bytes = c_sizeof(field_descs(n))
    field_descs(n)%field_id = id

    do i = 1, min(len_trim(name), FV3_GPU_FIELD_NAME_BYTES-1)
      field_descs(n)%name(i) = name(i:i)
    enddo
    field_descs(n)%name(min(len_trim(name)+1, FV3_GPU_FIELD_NAME_BYTES)) = c_null_char

    field_descs(n)%base = ptr
    field_descs(n)%allocation_bytes = 0
#ifdef SINGLE_FV
    field_descs(n)%element_type = FV3_GPU_ELEMENT_F32
    field_descs(n)%element_bytes = 4
#else
    field_descs(n)%element_type = FV3_GPU_ELEMENT_F64
    field_descs(n)%element_bytes = 8
#endif
    field_descs(n)%rank = 2
    field_descs(n)%memory_location = FV3_GPU_MEMORY_HOST

    field_descs(n)%lower(1) = i_lo
    field_descs(n)%lower(2) = j_lo
    field_descs(n)%extent(1) = i_hi - i_lo + 1
    field_descs(n)%extent(2) = j_hi - j_lo + 1
    field_descs(n)%stride_elements(1) = 1
    field_descs(n)%stride_elements(2) = i_hi - i_lo + 1

    field_descs(n)%lower(3:4) = 0
    field_descs(n)%extent(3:4) = 0
    field_descs(n)%stride_elements(3:4) = 0

    field_descs(n)%compute_lower(1) = i_lo
    field_descs(n)%compute_lower(2) = j_lo
    field_descs(n)%compute_upper(1) = i_hi
    field_descs(n)%compute_upper(2) = j_hi
    field_descs(n)%compute_lower(3:4) = 0
    field_descs(n)%compute_upper(3:4) = 0

    field_descs(n)%halo_lower(:) = 0
    field_descs(n)%halo_upper(:) = 0

    field_descs(n)%stagger = stagger
    field_descs(n)%access = access
    field_descs(n)%alignment_bytes = 8
    field_descs(n)%reserved = 0

  end subroutine add_field_2d

  !----------------------------------------------------------------------------
  ! Internal: Add 3D field descriptor
  !----------------------------------------------------------------------------
  subroutine add_field_3d(id, name, ptr, &
      i_lo, i_hi, j_lo, j_hi, k_lo, k_hi, &
      ci_lo, ci_hi, cj_lo, cj_hi, ck_lo, ck_hi, &
      hi_lo, hi_hi, hj_lo, hj_hi, hk_lo, hk_hi, &
      stagger, access)
    integer(c_int32_t), intent(in) :: id
    character(*), intent(in) :: name
    type(c_ptr), intent(in) :: ptr
    integer, intent(in) :: i_lo, i_hi, j_lo, j_hi, k_lo, k_hi
    integer, intent(in) :: ci_lo, ci_hi, cj_lo, cj_hi, ck_lo, ck_hi
    integer, intent(in) :: hi_lo, hi_hi, hj_lo, hj_hi, hk_lo, hk_hi
    integer(c_int32_t), intent(in) :: stagger, access

    integer :: n, i
    integer(c_int64_t) :: ni, nj

    field_count = field_count + 1
    n = field_count

    field_descs(n)%abi_version = FV3_GPU_ABI_VERSION
    field_descs(n)%struct_bytes = c_sizeof(field_descs(n))
    field_descs(n)%field_id = id

    do i = 1, min(len_trim(name), FV3_GPU_FIELD_NAME_BYTES-1)
      field_descs(n)%name(i) = name(i:i)
    enddo
    field_descs(n)%name(min(len_trim(name)+1, FV3_GPU_FIELD_NAME_BYTES)) = c_null_char

    field_descs(n)%base = ptr
    field_descs(n)%allocation_bytes = 0
#ifdef SINGLE_FV
    field_descs(n)%element_type = FV3_GPU_ELEMENT_F32
    field_descs(n)%element_bytes = 4
#else
    field_descs(n)%element_type = FV3_GPU_ELEMENT_F64
    field_descs(n)%element_bytes = 8
#endif
    field_descs(n)%rank = 3
    field_descs(n)%memory_location = FV3_GPU_MEMORY_HOST

    ni = i_hi - i_lo + 1
    nj = j_hi - j_lo + 1

    field_descs(n)%lower(1) = i_lo
    field_descs(n)%lower(2) = j_lo
    field_descs(n)%lower(3) = k_lo
    field_descs(n)%extent(1) = ni
    field_descs(n)%extent(2) = nj
    field_descs(n)%extent(3) = k_hi - k_lo + 1
    field_descs(n)%stride_elements(1) = 1
    field_descs(n)%stride_elements(2) = ni
    field_descs(n)%stride_elements(3) = ni * nj

    field_descs(n)%lower(4) = 0
    field_descs(n)%extent(4) = 0
    field_descs(n)%stride_elements(4) = 0

    field_descs(n)%compute_lower(1) = ci_lo
    field_descs(n)%compute_lower(2) = cj_lo
    field_descs(n)%compute_lower(3) = ck_lo
    field_descs(n)%compute_upper(1) = ci_hi
    field_descs(n)%compute_upper(2) = cj_hi
    field_descs(n)%compute_upper(3) = ck_hi
    field_descs(n)%compute_lower(4) = 0
    field_descs(n)%compute_upper(4) = 0

    field_descs(n)%halo_lower(1) = hi_lo
    field_descs(n)%halo_lower(2) = hj_lo
    field_descs(n)%halo_lower(3) = hk_lo
    field_descs(n)%halo_upper(1) = hi_hi
    field_descs(n)%halo_upper(2) = hj_hi
    field_descs(n)%halo_upper(3) = hk_hi
    field_descs(n)%halo_lower(4) = 0
    field_descs(n)%halo_upper(4) = 0

    field_descs(n)%stagger = stagger
    field_descs(n)%access = access
    field_descs(n)%alignment_bytes = 8
    field_descs(n)%reserved = 0

  end subroutine add_field_3d

end module fv3_gpu_interface_mod
