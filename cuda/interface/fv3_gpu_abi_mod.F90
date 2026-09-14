module fv3_gpu_abi_mod

! ISO_C_BINDING mirror of the stable CUDA dyn_core C ABI.
!
! Every derived type here is BIND(C) and is a member-for-member image of the
! corresponding struct in cuda/dyn_core/include/fv3_gpu_api.h (and, for the halo
! transport selection, include/fv3_gpu_communication.h). Nothing in this module
! interprets, defaults or reorders anything -- it exists so that a Fortran caller
! can reach the same entry points the C++ benchmark harness reaches, and no
! further. In particular:
!
!   * No C++ type and no MPI type appears anywhere in the ABI. The GEOS/FV3
!     communicator crosses as the Fortran INTEGER handle and is converted on the
!     far side with MPI_Comm_f2c, because MPI_Comm is an int in MPICH but an
!     opaque pointer in Open MPI and so cannot appear in a stable C ABI.
!   * The field_id space is the identity map over fv3::gpu::FieldId
!     (fv3_gpu_internal.hpp:23-59); see field_id_from_abi at :53-56.
!
! Keep this file in lockstep with fv3_gpu_api.h. If a struct there gains a
! member, add it here in the same position -- struct_bytes is checked on the
! far side precisely so a silent layout drift becomes a loud error.

   use iso_c_binding

   implicit none
   private

   ! ---- scalar constants (fv3_gpu_api.h) --------------------------------------

   integer, parameter, public :: FV3_GPU_ABI_VERSION       = 1
   integer, parameter, public :: FV3_GPU_MAX_RANK          = 4
   integer, parameter, public :: FV3_GPU_FIELD_NAME_BYTES  = 32
   integer, parameter, public :: FV3_GPU_ERROR_MSG_BYTES   = 256

   ! fv3_gpu_status
   integer, parameter, public :: FV3_GPU_SUCCESS                   = 0
   integer, parameter, public :: FV3_GPU_INVALID_ARGUMENT          = 1
   integer, parameter, public :: FV3_GPU_UNSUPPORTED_CONFIGURATION = 2
   integer, parameter, public :: FV3_GPU_CUDA_ERROR                = 3
   integer, parameter, public :: FV3_GPU_MPI_ERROR                 = 4
   integer, parameter, public :: FV3_GPU_OUT_OF_MEMORY             = 5
   integer, parameter, public :: FV3_GPU_STATE_CONFLICT            = 6
   integer, parameter, public :: FV3_GPU_INTERNAL_ERROR            = 7

   integer, parameter, public :: FV3_GPU_ELEMENT_F32 = 1
   integer, parameter, public :: FV3_GPU_ELEMENT_F64 = 2

   integer, parameter, public :: FV3_GPU_MEMORY_HOST   = 1
   integer, parameter, public :: FV3_GPU_MEMORY_DEVICE = 2

   integer, parameter, public :: FV3_GPU_STAGGER_A         = 1
   integer, parameter, public :: FV3_GPU_STAGGER_C_U       = 2
   integer, parameter, public :: FV3_GPU_STAGGER_C_V       = 3
   integer, parameter, public :: FV3_GPU_STAGGER_D_U       = 4
   integer, parameter, public :: FV3_GPU_STAGGER_D_V       = 5
   integer, parameter, public :: FV3_GPU_STAGGER_B         = 6
   integer, parameter, public :: FV3_GPU_STAGGER_INTERFACE = 7
   integer, parameter, public :: FV3_GPU_STAGGER_X_FACE    = 8
   integer, parameter, public :: FV3_GPU_STAGGER_Y_FACE    = 9

   integer, parameter, public :: FV3_GPU_ACCESS_INPUT  = 1
   integer, parameter, public :: FV3_GPU_ACCESS_OUTPUT = 2
   integer, parameter, public :: FV3_GPU_ACCESS_INOUT  = 3

   integer, parameter, public :: FV3_GPU_PRECISION_R4 = 1
   integer, parameter, public :: FV3_GPU_PRECISION_R8 = 2

   ! Halo transport selection (fv3_gpu_communication.h, Prompt 15).
   integer, parameter, public :: FV3_GPU_HALO_TRANSPORT_NONE            = 0
   integer, parameter, public :: FV3_GPU_HALO_TRANSPORT_IN_PROCESS_PEER = 1
   integer, parameter, public :: FV3_GPU_HALO_TRANSPORT_MPI_HOST_STAGED = 2

   ! ---- structs ---------------------------------------------------------------

   type, bind(C), public :: fv3_gpu_error
      integer(c_int32_t)     :: abi_version
      integer(c_int32_t)     :: struct_bytes
      integer(c_int32_t)     :: code
      integer(c_int32_t)     :: native_code
      integer(c_int32_t)     :: rank
      integer(c_int32_t)     :: device
      integer(c_int64_t)     :: sequence
      character(kind=c_char) :: api(FV3_GPU_FIELD_NAME_BYTES)
      character(kind=c_char) :: message(FV3_GPU_ERROR_MSG_BYTES)
   end type fv3_gpu_error

   type, bind(C), public :: fv3_gpu_field_desc
      integer(c_int32_t)     :: abi_version
      integer(c_int32_t)     :: struct_bytes
      integer(c_int32_t)     :: field_id
      character(kind=c_char) :: name(FV3_GPU_FIELD_NAME_BYTES)
      type(c_ptr)            :: base
      integer(c_int64_t)     :: allocation_bytes
      integer(c_int32_t)     :: element_type
      integer(c_int32_t)     :: element_bytes
      integer(c_int32_t)     :: rank
      integer(c_int32_t)     :: memory_location
      integer(c_int64_t)     :: lower(FV3_GPU_MAX_RANK)
      integer(c_int64_t)     :: extent(FV3_GPU_MAX_RANK)
      integer(c_int64_t)     :: stride_elements(FV3_GPU_MAX_RANK)
      integer(c_int64_t)     :: compute_lower(FV3_GPU_MAX_RANK)
      integer(c_int64_t)     :: compute_upper(FV3_GPU_MAX_RANK)
      integer(c_int32_t)     :: halo_lower(FV3_GPU_MAX_RANK)
      integer(c_int32_t)     :: halo_upper(FV3_GPU_MAX_RANK)
      integer(c_int32_t)     :: stagger
      integer(c_int32_t)     :: access
      integer(c_int32_t)     :: alignment_bytes
      integer(c_int32_t)     :: reserved
   end type fv3_gpu_field_desc

   type, bind(C), public :: fv3_gpu_create_desc
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
   end type fv3_gpu_create_desc

   type, bind(C), public :: fv3_gpu_state_desc
      integer(c_int32_t) :: abi_version
      integer(c_int32_t) :: struct_bytes
      integer(c_int32_t) :: field_count
      integer(c_int32_t) :: reserved
      type(c_ptr)        :: fields
   end type fv3_gpu_state_desc

   type, bind(C), public :: fv3_gpu_field_mask
      integer(c_int32_t) :: abi_version
      integer(c_int32_t) :: struct_bytes
      integer(c_int64_t) :: words(4)
   end type fv3_gpu_field_mask

   type, bind(C), public :: fv3_gpu_step_desc
      integer(c_int32_t) :: abi_version
      integer(c_int32_t) :: struct_bytes
      real(c_double)     :: bdt
      integer(c_int32_t) :: k_split
      integer(c_int32_t) :: n_split
      integer(c_int8_t)  :: init_step
      integer(c_int8_t)  :: end_step
      integer(c_int16_t) :: reserved0
      integer(c_int64_t) :: model_step
      integer(c_int64_t) :: acoustic_sequence
      integer(c_int64_t) :: validation_mask
      integer(c_int64_t) :: physics_tail_mask
      integer(c_int64_t) :: dependency
   end type fv3_gpu_step_desc

   type, bind(C), public :: fv3_gpu_capabilities
      integer(c_int32_t) :: abi_version
      integer(c_int32_t) :: struct_bytes
      integer(c_int32_t) :: compiled_with_cuda
      integer(c_int32_t) :: device_count
      integer(c_int32_t) :: precision_mask
      integer(c_int32_t) :: communication_stage
      integer(c_int64_t) :: feature_mask
   end type fv3_gpu_capabilities

   ! Member-for-member image of fv3_gpu_numerical_config (fv3_gpu_api.h:163-182),
   ! which itself mirrors fv3::gpu::NumericalConfig one-for-one so the runtime's
   ! translation is a plain member copy. Order and grouping are preserved exactly
   ! as declared there; the u8 tail is a set of logical flags carried as bytes.
   type, bind(C), public :: fv3_gpu_numerical_config
      integer(c_int32_t) :: abi_version
      integer(c_int32_t) :: struct_bytes
      real(c_double)     :: ptop, akap, a_imp, p_fac, scale_m, dz_min
      real(c_double)     :: akap_exner
      real(c_double)     :: rdgas, grav
      real(c_double)     :: beta, d_ext, dddmp, d_con, lim_fac, zvir
      real(c_double)     :: da_min, da_min_c, delt_max
      real(c_double)     :: rayleigh_tau, rayleigh_ztop
      integer(c_int32_t) :: hord_tr, hord_mt, hord_vt, hord_tm, hord_dp
      integer(c_int32_t) :: hord_height, a2b_order
      integer(c_int32_t) :: nord, nord_v, nord_w, nord_t
      integer(c_int32_t) :: nf_ke, n_con, vertical_solver
      integer(c_int32_t) :: m_split
      integer(c_int8_t)  :: use_cond, moist_cappa, use_logp, inline_q
      integer(c_int8_t)  :: hydrostatic, rot3, do_f3d, do_skeb
      integer(c_int8_t)  :: use_sg, rayleigh_enabled, grid_type, reserved0
      integer(c_int8_t)  :: fuse_corner_fill
      integer(c_int8_t)  :: fuse_halo_edges
   end type fv3_gpu_numerical_config

   ! ---- field identifiers -----------------------------------------------------
   !
   ! Identity map over fv3::gpu::FieldId (fv3_gpu_internal.hpp:23-59), in the
   ! order the enum declares them, starting at invalid = 0. Only the ids this
   ! bridge actually registers are named; add more here if a future path needs
   ! them, keeping the ordinal in step with the enum.

   integer, parameter, public :: FV3_FID_U        = 1
   integer, parameter, public :: FV3_FID_V        = 2
   integer, parameter, public :: FV3_FID_W        = 3
   integer, parameter, public :: FV3_FID_DELZ     = 4
   integer, parameter, public :: FV3_FID_PT       = 5
   integer, parameter, public :: FV3_FID_DELP     = 6
   integer, parameter, public :: FV3_FID_Q        = 7
   integer, parameter, public :: FV3_FID_Q_CON    = 8
   integer, parameter, public :: FV3_FID_CAPPA    = 9
   integer, parameter, public :: FV3_FID_PE       = 10
   integer, parameter, public :: FV3_FID_PK       = 11
   integer, parameter, public :: FV3_FID_PELN     = 12
   integer, parameter, public :: FV3_FID_PKZ      = 13
   integer, parameter, public :: FV3_FID_PHIS     = 14
   integer, parameter, public :: FV3_FID_OMGA     = 15
   integer, parameter, public :: FV3_FID_UA       = 16
   integer, parameter, public :: FV3_FID_VA       = 17
   integer, parameter, public :: FV3_FID_UC       = 18
   integer, parameter, public :: FV3_FID_VC       = 19
   integer, parameter, public :: FV3_FID_MFX      = 20
   integer, parameter, public :: FV3_FID_MFY      = 21
   integer, parameter, public :: FV3_FID_CX       = 22
   integer, parameter, public :: FV3_FID_CY       = 23
   integer, parameter, public :: FV3_FID_DPX      = 24
   integer, parameter, public :: FV3_FID_DISS_EST = 25
   integer, parameter, public :: FV3_FID_WS       = 26
   integer, parameter, public :: FV3_FID_DELPC    = 27
   integer, parameter, public :: FV3_FID_PTC      = 28
   integer, parameter, public :: FV3_FID_UT       = 29
   integer, parameter, public :: FV3_FID_VT       = 30
   integer, parameter, public :: FV3_FID_DIVGD    = 31
   integer, parameter, public :: FV3_FID_CRX      = 32
   integer, parameter, public :: FV3_FID_XFX      = 33
   integer, parameter, public :: FV3_FID_CRY      = 34
   integer, parameter, public :: FV3_FID_YFX      = 35
   integer, parameter, public :: FV3_FID_GZ       = 36
   integer, parameter, public :: FV3_FID_ZH       = 37
   integer, parameter, public :: FV3_FID_PKC      = 38
   integer, parameter, public :: FV3_FID_PK3      = 39
   integer, parameter, public :: FV3_FID_DU       = 40
   integer, parameter, public :: FV3_FID_DV       = 41
   integer, parameter, public :: FV3_FID_HEAT_SOURCE = 42
   integer, parameter, public :: FV3_FID_WS3      = 43
   integer, parameter, public :: FV3_FID_Z_RAT    = 44
   integer, parameter, public :: FV3_FID_DIVG2    = 45

   integer, parameter, public :: FV3_FID_AK       = 46
   integer, parameter, public :: FV3_FID_BK       = 47
   integer, parameter, public :: FV3_FID_PFULL    = 48
   integer, parameter, public :: FV3_FID_DP_REF   = 49
   integer, parameter, public :: FV3_FID_ZS       = 50
   integer, parameter, public :: FV3_FID_AREA     = 51
   integer, parameter, public :: FV3_FID_RAREA    = 52
   integer, parameter, public :: FV3_FID_DX       = 53
   integer, parameter, public :: FV3_FID_DY       = 54
   integer, parameter, public :: FV3_FID_DXC      = 55
   integer, parameter, public :: FV3_FID_DYC      = 56
   integer, parameter, public :: FV3_FID_RDX      = 57
   integer, parameter, public :: FV3_FID_RDY      = 58
   integer, parameter, public :: FV3_FID_RDXC     = 59
   integer, parameter, public :: FV3_FID_RDYC     = 60
   integer, parameter, public :: FV3_FID_COSA_U   = 61
   integer, parameter, public :: FV3_FID_COSA_V   = 62
   integer, parameter, public :: FV3_FID_SINA_U   = 63
   integer, parameter, public :: FV3_FID_SINA_V   = 64
   integer, parameter, public :: FV3_FID_RSIN_U   = 65
   integer, parameter, public :: FV3_FID_RSIN_V   = 66
   integer, parameter, public :: FV3_FID_SIN_SG   = 67
   integer, parameter, public :: FV3_FID_COS_SG   = 68
   integer, parameter, public :: FV3_FID_F0       = 69
   integer, parameter, public :: FV3_FID_DEL6_U   = 70
   integer, parameter, public :: FV3_FID_DEL6_V   = 71
   integer, parameter, public :: FV3_FID_COSA_S   = 72
   integer, parameter, public :: FV3_FID_RSIN2    = 73
   integer, parameter, public :: FV3_FID_DXA      = 74
   integer, parameter, public :: FV3_FID_DYA      = 75
   integer, parameter, public :: FV3_FID_RAREA_C  = 76
   integer, parameter, public :: FV3_FID_AREA_C   = 77
   integer, parameter, public :: FV3_FID_F_C      = 78
   integer, parameter, public :: FV3_FID_RDXA     = 79
   integer, parameter, public :: FV3_FID_RDYA     = 80
   integer, parameter, public :: FV3_FID_COSA     = 81
   integer, parameter, public :: FV3_FID_RSINA    = 82
   integer, parameter, public :: FV3_FID_DIVG_U   = 83
   integer, parameter, public :: FV3_FID_DIVG_V   = 84
   integer, parameter, public :: FV3_FID_GRID_LONLAT  = 85
   integer, parameter, public :: FV3_FID_AGRID_LONLAT = 86
   integer, parameter, public :: FV3_FID_EDGE_W   = 87
   integer, parameter, public :: FV3_FID_EDGE_E   = 88
   integer, parameter, public :: FV3_FID_EDGE_S   = 89
   integer, parameter, public :: FV3_FID_EDGE_N   = 90

   integer, parameter, public :: FV3_FID_DAMPING_D2_BG  = 91
   integer, parameter, public :: FV3_FID_DAMPING_D4_BG  = 92
   integer, parameter, public :: FV3_FID_DAMPING_V      = 93
   integer, parameter, public :: FV3_FID_DAMPING_W      = 94
   integer, parameter, public :: FV3_FID_DAMPING_T      = 95
   integer, parameter, public :: FV3_FID_DAMPING_KGB    = 96
   integer, parameter, public :: FV3_FID_DAMPING_NDIF   = 97
   integer, parameter, public :: FV3_FID_DAMPING_NORD_K = 98
   integer, parameter, public :: FV3_FID_DAMPING_NORD_W = 99
   integer, parameter, public :: FV3_FID_DAMPING_D_CON  = 100

   ! ---- the C ABI ------------------------------------------------------------
   !
   ! Lifecycle order is fixed and is NOT the declaration order below:
   !   create -> initialize_state -> set_numerical_config
   !          -> install_production_dispatch -> set_cubed_sphere_topology
   !          -> [attach_mpi_halo_transport] -> dyn_core_step ...
   !          -> [detach_halo_transport] -> finalize
   ! initialize_state performs the one initial H2D, so every registered array
   ! must hold its final value before it is called -- in particular the derived
   ! per-level damping arrays.

   public :: fv3_gpu_create
   public :: fv3_gpu_initialize_state
   public :: fv3_gpu_set_numerical_config
   public :: fv3_gpu_install_production_dispatch
   public :: fv3_gpu_set_cubed_sphere_topology
   public :: fv3_gpu_sync_from_host
   public :: fv3_gpu_dyn_core_step
   public :: fv3_gpu_sync_to_host
   public :: fv3_gpu_event_wait
   public :: fv3_gpu_get_capabilities
   public :: fv3_gpu_set_cuda_graph_mode
   public :: fv3_gpu_finalize
   public :: fv3_gpu_attach_mpi_halo_transport
   public :: fv3_gpu_detach_halo_transport
   public :: fv3_gpu_halo_transport_kind_of
   public :: fv3_gpu_error_text

   interface

      integer(c_int) function fv3_gpu_create(desc, ctx, err) &
                              bind(C, name="fv3_gpu_create")
         import :: c_int, c_ptr, fv3_gpu_create_desc, fv3_gpu_error
         type(fv3_gpu_create_desc), intent(in) :: desc
         type(c_ptr), intent(out)              :: ctx
         type(fv3_gpu_error), intent(inout)    :: err
      end function fv3_gpu_create

      integer(c_int) function fv3_gpu_initialize_state(ctx, state, ev, err) &
                              bind(C, name="fv3_gpu_initialize_state")
         import :: c_int, c_int64_t, c_ptr, fv3_gpu_state_desc, fv3_gpu_error
         type(c_ptr), value                   :: ctx
         type(fv3_gpu_state_desc), intent(in) :: state
         integer(c_int64_t), intent(out)      :: ev
         type(fv3_gpu_error), intent(inout)   :: err
      end function fv3_gpu_initialize_state

      ! Prompt 16 addition. Previously reachable only through the internal,
      ! non-ABI C++ setter set_numerical_config().
      integer(c_int) function fv3_gpu_set_numerical_config(ctx, cfg, err) &
                              bind(C, name="fv3_gpu_set_numerical_config")
         import :: c_int, c_ptr, fv3_gpu_error, fv3_gpu_numerical_config
         type(c_ptr), value                        :: ctx
         type(fv3_gpu_numerical_config), intent(in) :: cfg
         type(fv3_gpu_error), intent(inout)        :: err
      end function fv3_gpu_set_numerical_config

      ! Prompt 16 addition. Defined in fv3_gpu_production_dispatch.cu rather
      ! than fv3_gpu_runtime.cpp to avoid a static-library dependency cycle.
      integer(c_int) function fv3_gpu_install_production_dispatch(ctx, err) &
                              bind(C, name="fv3_gpu_install_production_dispatch")
         import :: c_int, c_ptr, fv3_gpu_error
         type(c_ptr), value                 :: ctx
         type(fv3_gpu_error), intent(inout) :: err
      end function fv3_gpu_install_production_dispatch

      integer(c_int) function fv3_gpu_set_cubed_sphere_topology( &
                                 ctx, tile, npx, npy, rank_of_tile, err) &
                              bind(C, name="fv3_gpu_set_cubed_sphere_topology")
         import :: c_int, c_int32_t, c_int64_t, c_ptr, fv3_gpu_error
         type(c_ptr), value                 :: ctx
         integer(c_int32_t), value          :: tile
         integer(c_int64_t), value          :: npx, npy
         integer(c_int32_t), intent(in)     :: rank_of_tile(*)
         type(fv3_gpu_error), intent(inout) :: err
      end function fv3_gpu_set_cubed_sphere_topology

      integer(c_int) function fv3_gpu_sync_from_host(ctx, mask, ev, err) &
                              bind(C, name="fv3_gpu_sync_from_host")
         import :: c_int, c_int64_t, c_ptr, fv3_gpu_field_mask, fv3_gpu_error
         type(c_ptr), value                    :: ctx
         type(fv3_gpu_field_mask), intent(in)  :: mask
         integer(c_int64_t), intent(out)       :: ev
         type(fv3_gpu_error), intent(inout)    :: err
      end function fv3_gpu_sync_from_host

      integer(c_int) function fv3_gpu_dyn_core_step(ctx, step, ev, err) &
                              bind(C, name="fv3_gpu_dyn_core_step")
         import :: c_int, c_int64_t, c_ptr, fv3_gpu_step_desc, fv3_gpu_error
         type(c_ptr), value                   :: ctx
         type(fv3_gpu_step_desc), intent(in)  :: step
         integer(c_int64_t), intent(out)      :: ev
         type(fv3_gpu_error), intent(inout)   :: err
      end function fv3_gpu_dyn_core_step

      integer(c_int) function fv3_gpu_sync_to_host(ctx, mask, ev, err) &
                              bind(C, name="fv3_gpu_sync_to_host")
         import :: c_int, c_int64_t, c_ptr, fv3_gpu_field_mask, fv3_gpu_error
         type(c_ptr), value                    :: ctx
         type(fv3_gpu_field_mask), intent(in)  :: mask
         integer(c_int64_t), intent(out)       :: ev
         type(fv3_gpu_error), intent(inout)    :: err
      end function fv3_gpu_sync_to_host

      integer(c_int) function fv3_gpu_event_wait(ctx, ev, err) &
                              bind(C, name="fv3_gpu_event_wait")
         import :: c_int, c_int64_t, c_ptr, fv3_gpu_error
         type(c_ptr), value                 :: ctx
         integer(c_int64_t), value          :: ev
         type(fv3_gpu_error), intent(inout) :: err
      end function fv3_gpu_event_wait

      integer(c_int) function fv3_gpu_get_capabilities(ctx, caps) &
                              bind(C, name="fv3_gpu_get_capabilities")
         import :: c_int, c_ptr, fv3_gpu_capabilities
         type(c_ptr), value                        :: ctx
         type(fv3_gpu_capabilities), intent(inout) :: caps
      end function fv3_gpu_get_capabilities

      ! Prompt 8 (CUDA Graphs prototype), exposed via Prompt 29. Opt-in only;
      ! must be called before the first fv3_gpu_dyn_core_step on this context
      ! -- callers set this once, right after fv3_gpu_create. Rejects being
      ! toggled after a graph has already been captured (FV3_GPU_STATE_CONFLICT).
      ! Scope is deliberately narrow: only the dgrid_dynamics stage is
      ! captured/replayed; every other stage and all cross-tile halo
      ! communication run exactly as in default mode.
      integer(c_int) function fv3_gpu_set_cuda_graph_mode(ctx, enabled, err) &
                              bind(C, name="fv3_gpu_set_cuda_graph_mode")
         import :: c_int, c_ptr, fv3_gpu_error
         type(c_ptr), value                 :: ctx
         integer(c_int), value              :: enabled
         type(fv3_gpu_error), intent(inout) :: err
      end function fv3_gpu_set_cuda_graph_mode

      ! Takes fv3_gpu_context ** -- it nulls the caller's handle -- so the dummy
      ! is a non-VALUE type(c_ptr), i.e. a pointer to the handle variable.
      integer(c_int) function fv3_gpu_finalize(ctx, err) &
                              bind(C, name="fv3_gpu_finalize")
         import :: c_int, c_ptr, fv3_gpu_error
         type(c_ptr), intent(inout)         :: ctx
         type(fv3_gpu_error), intent(inout) :: err
      end function fv3_gpu_finalize

      ! Prompt 15 halo transport selection. comm_f is the Fortran INTEGER
      ! communicator handle; the backend converts it with MPI_Comm_f2c. The
      ! communicator must have exactly 6 ranks with one tile per rank -- the
      ! backend verifies the rank<->tile map at attach and refuses otherwise.
      integer(c_int) function fv3_gpu_attach_mpi_halo_transport(ctx, comm_f, tile, err) &
                              bind(C, name="fv3_gpu_attach_mpi_halo_transport")
         import :: c_int, c_int32_t, c_ptr, fv3_gpu_error
         type(c_ptr), value                 :: ctx
         integer(c_int), value              :: comm_f
         integer(c_int32_t), value          :: tile
         type(fv3_gpu_error), intent(inout) :: err
      end function fv3_gpu_attach_mpi_halo_transport

      integer(c_int) function fv3_gpu_detach_halo_transport(ctx, err) &
                              bind(C, name="fv3_gpu_detach_halo_transport")
         import :: c_int, c_ptr, fv3_gpu_error
         type(c_ptr), value                 :: ctx
         type(fv3_gpu_error), intent(inout) :: err
      end function fv3_gpu_detach_halo_transport

      integer(c_int) function fv3_gpu_halo_transport_kind_of(ctx) &
                              bind(C, name="fv3_gpu_halo_transport_kind_of")
         import :: c_int, c_ptr
         type(c_ptr), value :: ctx
      end function fv3_gpu_halo_transport_kind_of

   end interface

contains

   ! Render fv3_gpu_error's C message into a Fortran string for mpp_error.
   function fv3_gpu_error_text(err) result(txt)
      type(fv3_gpu_error), intent(in) :: err
      character(len=FV3_GPU_ERROR_MSG_BYTES) :: txt
      integer :: i
      txt = ' '
      do i = 1, FV3_GPU_ERROR_MSG_BYTES
         if (err%message(i) == c_null_char) exit
         txt(i:i) = err%message(i)
      end do
   end function fv3_gpu_error_text

end module fv3_gpu_abi_mod
