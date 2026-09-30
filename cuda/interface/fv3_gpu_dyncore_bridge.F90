module fv3_gpu_dyncore_bridge

! Lifecycle bridge between GEOS/FV3 Fortran and the CUDA dyn_core subsystem.
!
! This module owns the persistent GPU context for this MPI rank. It is the only
! place that calls the C ABI, and it exposes exactly three things to production
! Fortran:
!
!   fv3_gpu_bridge_init        once, from FV_StateMod, where Atm and the ESMF VM
!                              communicator are both in scope
!   fv3_gpu_backend_enabled    a pure query, so fv_dynamics can branch without
!                              knowing anything about the backend
!   fv3_gpu_interface_step_dyn one dyn_core call, from inside fv_dynamics' n_map
!                              loop, replacing the Fortran dyn_core call
!   fv3_gpu_bridge_final       once, at teardown
!
! WHAT CROSSES, AND WHY IT IS NOT AN ARCHITECTURAL REGRESSION.
! The context and every device allocation are created once and reused for the
! life of the run (WP-E persistent residency). Model state is NOT re-registered
! per call -- the descriptors point at the same FV_Atm arrays for the whole run.
! What does cross per call is one H2D of the mutable prognostic state before the
! step and one D2H after it. That is required, not convenient: between two
! successive dyn_core calls fv_dynamics runs tracer advection and the
! Lagrangian-to-Eulerian remap, both of which rewrite delp/pt/u/v/w/delz/pe/pk/
! pkz on the host. Remapping is one of the synchronization boundaries
! project_architecture.md names explicitly. Nothing is transferred BETWEEN work
! packages inside a step; the acoustic loop stays entirely on the device.
!
! HALO OWNERSHIP. The CUDA backend performs its own six-tile cubed-sphere halo
! exchange on the device (Prompt 15 host-staged MPI transport, one tile per
! rank), so fv_dynamics does not start the FMS group halo updates i_pack(1) and
! i_pack(8) when this backend is active. See the COMM_TOTAL block in
! fv_dynamics.F90 for the full argument and for why the starts must be skipped
! rather than started and abandoned. i_pack(10) -- the tracer halo update
! dyn_core deliberately leaves outstanding across its own exit -- is DISTINCT
! and is still issued by the CUDA branch.

   use iso_c_binding
   use mpi

   use fv3_gpu_abi_mod
   use fv3_gpu_field_registry_mod, only: fv3_gpu_registry_build, fv3_gpu_registry_free, &
                                         fv3_gpu_registry_export_flux_capacitors,        &
                                         fv3_gpu_registry_export_ws,                    &
                                         fv3_gpu_registry_last_failure

   use fv_arrays_mod, only: fv_atmos_type
   use mpp_mod,       only: mpp_error, FATAL, NOTE

   ! rdgas/cp_air/grav are the same constants_mod symbols dyn_core.F90:106 and
   ! nh_utils.F90:53 import, so the closure kappa below is byte-for-byte the one
   ! the authoritative Fortran uses. MAPL_KAPPA is separately the single-precision
   ! Exner kappa FV_StateMod.F90:501 assigns to the `akap` dummy dyn_core
   ! receives -- the two-kappa split recorded in CURRENT_STATE.md section 4 #8.
   use constants_mod,     only: rdgas, cp_air, grav
   use MAPL_ConstantsMod, only: MAPL_KAPPA, MAPL_RGAS, MAPL_RVAP

   implicit none
   private

   public :: fv3_gpu_bridge_init
   public :: fv3_gpu_backend_enabled
   public :: fv3_gpu_device_count
   public :: fv3_gpu_interface_step_dyn
   public :: fv3_gpu_interface_pe_halo
   public :: fv3_gpu_interface_pt_delz_remap
   public :: fv3_gpu_interface_remap_uv
   public :: fv3_gpu_interface_remap_q
   public :: fv3_gpu_interface_remap_pressurevars
   public :: fv3_gpu_interface_state_to_fv_phase_a
   public :: fv3_gpu_interface_state_to_fv_phase_c
   public :: fv3_gpu_interface_fv_tp_2d_hord8
   public :: fv3_gpu_bridge_final
   public :: env_flag_enabled

   ! ---- persistent per-rank state ---------------------------------------------

   logical,     save :: bridge_ready = .false.
   type(c_ptr), save :: ctx          = C_NULL_PTR
   integer,     save :: bridge_comm  = MPI_COMM_NULL
   integer,     save :: bridge_tile  = 0
   integer(c_int64_t), save :: model_step = 0

   ! Process-local CUDA device count (cudaGetDeviceCount, so it already
   ! respects CUDA_VISIBLE_DEVICES), cached once in fv3_gpu_bridge_init from
   ! the fv3_gpu_get_capabilities call it already makes. Exposed read-only so
   ! callers (e.g. State_To_FV's density-aware default selector) do not need
   ! their own ABI call. 0 until the bridge has initialized.
   integer, save :: cached_device_count = 0

   ! The registered descriptors must outlive fv3_gpu_initialize_state only as
   ! data, but the ARRAYS they point at must outlive the context, so the registry
   ! module owns those. These are kept saved anyway so a later re-registration
   ! (not currently used) does not have to rebuild them.
   ! TARGET is required: c_loc(descs(1)) at :189 hands the descriptor array to
   ! fv3_gpu_initialize_state through the C ABI, and c_loc demands POINTER or
   ! TARGET (F2018 18.2.3.6). ALLOCATABLE alone is not sufficient.
   type(fv3_gpu_field_desc), allocatable, save, target :: descs(:)
   integer, save :: ndesc = 0

   ! H2D/D2H mask: the mutable prognostic state, field ids 1..26 of the FieldId
   ! enum (u .. ws). Bits for ids this configuration does not register are simply
   ! never matched -- the runtime iterates its registered fields, not the mask.
   integer, parameter :: mask_id_first = FV3_FID_U
   integer, parameter :: mask_id_last  = FV3_FID_WS

contains

! =============================================================================
! Query
! =============================================================================

   logical function fv3_gpu_backend_enabled()
      fv3_gpu_backend_enabled = bridge_ready
   end function fv3_gpu_backend_enabled

   ! Process-local CUDA device count visible to this rank (see
   ! cached_device_count above). Returns 0 if the bridge never initialized
   ! (backend disabled, or init failed before the capabilities query).
   integer function fv3_gpu_device_count()
      fv3_gpu_device_count = cached_device_count
   end function fv3_gpu_device_count

! =============================================================================
! Initialization
! =============================================================================

! comm is the Fortran INTEGER communicator handle for the six-rank FV3
! decomposition (one tile per rank). enable is the caller's backend selection --
! FV_StateMod reads it from the RUN_CUDA_FV3 MAPL resource, exactly as it
! already reads RUN_GTFV3 for the other non-Fortran backend. When enable is
! .false. this routine does nothing at all: no context, no device query, no
! allocation.
   subroutine fv3_gpu_bridge_init(Atm, comm, enable, rc)
      type(fv_atmos_type), intent(inout), target :: Atm
      integer,             intent(in)  :: comm
      logical,             intent(in)  :: enable
      integer,             intent(out) :: rc

      type(fv3_gpu_create_desc)      :: cd
      type(fv3_gpu_state_desc)       :: sd
      type(fv3_gpu_numerical_config) :: cfg
      type(fv3_gpu_capabilities)     :: caps
      type(fv3_gpu_error)            :: err
      integer(c_int64_t)             :: ev
      integer(c_int32_t)             :: rank_of_tile(6)
      integer(c_int32_t)             :: mpi_transport_kind
      ! Prompt 114: FMS decomposition layout metadata, wired through to the
      ! ABI/runtime for storage/diagnostics only -- see fv3_gpu_bridge_init's
      ! "layout" comment block below. Does not gate or alter any topology,
      ! transport, or dispatch behavior.
      integer(c_int32_t)             :: layout_x, layout_y
      integer :: crc, ierr, comm_rank, comm_size, dev
      integer :: local_rank, local_size, split_ierr
      integer :: fail_fid, fail_aux
      character(len=256) :: msg
      character(len=MPI_MAX_PROCESSOR_NAME) :: procname
      integer :: namelen

      rc = 0
      if (.not. enable) return
      if (bridge_ready) return

      bridge_comm = comm
      call MPI_Comm_rank(comm, comm_rank, ierr)
      call MPI_Comm_size(comm, comm_size, ierr)

      if (comm_size /= 6) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: the CUDA dyn_core backend '// &
              'requires exactly 6 ranks with one cubed-sphere tile per rank')
      endif

      ! ---- FMS decomposition layout metadata (Prompt 114) -----------------------
      ! Atm%layout(1:2) is FMS' own authoritative within-tile decomposition
      ! (Atm%layout(1)=layout_x, Atm%layout(2)=layout_y), pointer-aliased from
      ! fv_control.F90 and already resident on Atm, which this subroutine
      ! already receives -- no new plumbing above this call site. Today's
      ! production configuration is always layout=(1,1) (one rank per tile);
      ! the comm_size==6 gate above remains the SOLE authority for that, this
      ! is storage/diagnostics only. A value <1 would mean FMS itself failed
      ! to set up its own decomposition, which is a genuine FATAL, not a
      ! "default and continue" case like an omitted ABI field.
      layout_x = int(Atm%layout(1), c_int32_t)
      layout_y = int(Atm%layout(2), c_int32_t)
      if (layout_x < 1 .or. layout_y < 1) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: Atm%layout(1:2) reports '// &
              'a non-positive within-tile decomposition; refusing to proceed')
      endif

      ! ---- node-local rank, for GPU binding on multi-node allocations -----------
      call fv3_gpu_node_local_rank(comm, local_rank, local_size, split_ierr)
      if (split_ierr /= 0) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: failed to derive a '// &
              'node-local rank via MPI_Comm_split_type(MPI_COMM_TYPE_SHARED); '// &
              'refusing to guess a GPU binding')
      endif

      ! ---- capabilities, and which device this rank drives ---------------------
      caps%abi_version  = FV3_GPU_ABI_VERSION
      caps%struct_bytes = int(storage_size(caps)/8, c_int32_t)
      crc = fv3_gpu_get_capabilities(C_NULL_PTR, caps)
      if (crc /= FV3_GPU_SUCCESS .or. caps%compiled_with_cuda == 0) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: this build of the CUDA '// &
              'dyn_core library was compiled without CUDA support')
      endif
      if (caps%device_count <= 0) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: no CUDA device is visible '// &
              'to this rank')
      endif
      ! Device binding uses the rank's position within its own node (Prompt 107),
      ! not its global rank, so multi-node allocations bind correctly even when
      ! global rank is not congruent to node-local rank mod device_count. On a
      ! single node local_rank == comm_rank, so this preserves prior behavior
      ! exactly. The halo-transport topology below still uses the global
      ! communicator/rank unchanged -- only device selection is node-local.
      dev = mod(local_rank, int(caps%device_count))
      cached_device_count = int(caps%device_count)

      if (env_flag_enabled('RUN_CUDA_FV3_LOG_GPU_BINDING')) then
         call MPI_Get_processor_name(procname, namelen, ierr)
         write(msg,'(a,a,a,i0,a,i0,a,i0,a,i0,a,i0)') &
              'FV3_GPU_BIND host=', trim(procname(1:namelen)), &
              ' global=', comm_rank, ' local=', local_rank, '/', local_size, &
              ' devices=', int(caps%device_count), ' selected=', dev
         ! mpp_error(NOTE,...) only prints from the FMS root PE by default --
         ! write(0,...) is the existing per-rank diagnostic pattern in this
         ! codebase (see FV_StateMod.F90's P99_EXIT_DUMP), used here so every
         ! rank's binding line actually reaches the log, not just PE 0's.
         write(0,'(A)') trim(msg)

         ! Prompt 114: additive diagnostic only, same env gate as the
         ! FV3_GPU_BIND line above -- no unconditional production log added.
         write(msg,'(a,i0,a,i0,a,i0,a,i0)') &
              'FV3_GPU_LAYOUT layout=', layout_x, 'x', layout_y, &
              ' tile=', Atm%tile, ' rank=', comm_rank
         write(0,'(A)') trim(msg)
      endif

      ! ---- create -------------------------------------------------------------
      cd%abi_version        = FV3_GPU_ABI_VERSION
      cd%struct_bytes       = int(storage_size(cd)/8, c_int32_t)
      cd%device_ordinal     = int(dev, c_int32_t)
      cd%mpi_rank           = int(comm_rank, c_int32_t)
      cd%mpi_size           = int(comm_size, c_int32_t)
#ifdef OVERLOAD_R4
      cd%precision          = FV3_GPU_PRECISION_R4
#else
      cd%precision          = FV3_GPU_PRECISION_R8
#endif
      cd%is                 = int(Atm%bd%is,  c_int64_t)
      cd%ie                 = int(Atm%bd%ie,  c_int64_t)
      cd%js                 = int(Atm%bd%js,  c_int64_t)
      cd%je                 = int(Atm%bd%je,  c_int64_t)
      cd%npz                = int(Atm%npz,    c_int64_t)
      cd%halo_width         = int(Atm%bd%ng,  c_int32_t)
      cd%flags              = 0
      cd%layout_x           = layout_x
      cd%layout_y           = layout_y
      ! Deliberately zero. The communicator does NOT cross here: this field has
      ! never been wired to anything, and MPI_Comm cannot appear in a stable C
      ! ABI. It crosses as the Fortran INTEGER handle at
      ! fv3_gpu_attach_mpi_halo_transport below.
      cd%communicator_token = 0

      crc = fv3_gpu_create(cd, ctx, err)
      call check(crc, err, 'fv3_gpu_create')

      ! Prompt 29: minimal opt-in override to enable the already-validated
      ! dgrid_dynamics CUDA Graph capture/replay path (Prompt 8/13), reusing
      ! the same env_flag_enabled() helper as the two Prompt 28 fusion flags.
      ! Must be set before the first fv3_gpu_dyn_core_step; right after
      ! fv3_gpu_create satisfies that and matches the runtime's own
      ! documented contract. Default OFF -- unchanged behavior when unset.
      crc = fv3_gpu_set_cuda_graph_mode(ctx, &
           merge(1_c_int, 0_c_int, env_flag_enabled('RUN_CUDA_FV3_GRAPH')), err)
      call check(crc, err, 'fv3_gpu_set_cuda_graph_mode')

      ! ---- register state, and perform the one initial H2D --------------------
      ! Every registered array must already hold its final value here, because
      ! fv3_gpu_initialize_state is the initial host-to-device copy. That
      ! includes the derived per-level damping arrays, which the registry builds.
      ! The four flux capacitors are deliberately NOT bound here, and Atm%mfx/
      ! mfy/cx/cy in particular must NOT be bound. What dyn_core receives is
      ! fv_dynamics' own LOCAL mfxL/mfyL/cxL/cyL (fv_dynamics.F90:265-269),
      ! which dyn_core zeroes on entry (dyn_core.F90:392-395) and fills over the
      ! acoustic loop; fv_tracer2d then consumes those locals
      ! (fv_dynamics.F90:789-799) and only afterwards does :873-876 accumulate
      ! mfx = mfx + mfxL into Atm's export arrays. Binding Atm%mfx would
      ! therefore have the CUDA dyn_core zero the k_split/model-step accumulator
      ! every call and then have :873 add it to itself.
      ! The registry's own buffers play exactly the role of the locals -- same
      ! bounds, same once-per-call zeroing (fv3_gpu_runtime.cpp:72) -- so the
      ! integration is a copy OUT of them after each step, into whichever arrays
      ! the caller passed down; see fv3_gpu_interface_step_dyn below.
      call fv3_gpu_registry_build(Atm, descs, ndesc, rc)
      if (rc /= 0) then
         ! Name the field and the subsidiary code. rc 1-4 come from add(): 1 the
         ! contract has no such field, 2 rank disagreement, 3 element-width
         ! disagreement, 4 descriptor overflow. rc 11 is an unsupported model
         ! real width; rc 20+n is the damping-column translation's own n.
         call fv3_gpu_registry_last_failure(fail_fid, fail_aux)
         write(msg,'(a,i0,a,i0,a,i0)')                                          &
              'fv3_gpu_bridge_init: field registry construction failed, rc=',   &
              rc, ' field_id=', fail_fid, ' aux=', fail_aux
         call mpp_error(FATAL, trim(msg))
      endif

      sd%abi_version  = FV3_GPU_ABI_VERSION
      sd%struct_bytes = int(storage_size(sd)/8, c_int32_t)
      sd%field_count  = int(ndesc, c_int32_t)
      sd%reserved     = 0
      sd%fields       = c_loc(descs(1))

      crc = fv3_gpu_initialize_state(ctx, sd, ev, err)
      call check(crc, err, 'fv3_gpu_initialize_state')

      ! ---- numerical configuration and the production dispatch table ---------
      call fv3_gpu_fill_numerical_config(Atm, cfg)
      crc = fv3_gpu_set_numerical_config(ctx, cfg, err)
      call check(crc, err, 'fv3_gpu_set_numerical_config')

      crc = fv3_gpu_install_production_dispatch(ctx, err)
      call check(crc, err, 'fv3_gpu_install_production_dispatch')

      ! ---- topology ----------------------------------------------------------
      ! The rank<->tile map is VERIFIED, not assumed. It is nominally the
      ! identity (rank r owns tile r+1) for FV_NX=1/FV_NY=6, but a launcher or
      ! layout change that broke it would otherwise produce plausible-looking
      ! wrong halo data instead of a failure, so the map is allgathered and the
      ! runtime cross-checks it against this context's own mpi_rank.
      bridge_tile = Atm%tile
      rank_of_tile = -1
      call build_rank_of_tile(comm, comm_rank, bridge_tile, rank_of_tile, &
                              layout_x, layout_y)

      crc = fv3_gpu_set_cubed_sphere_topology(ctx, int(bridge_tile, c_int32_t), &
                                              int(Atm%npx, c_int64_t),          &
                                              int(Atm%npy, c_int64_t),          &
                                              rank_of_tile,                     &
                                              layout_x, layout_y, err)
      call check(crc, err, 'fv3_gpu_set_cubed_sphere_topology')

      ! ---- halo transport ----------------------------------------------------
      ! The communicator crosses as its Fortran INTEGER handle; the transport
      ! converts back with MPI_Comm_f2c. Prompt 15's host-staged MPI backend
      ! remains the default; Prompt 43 adds an opt-in cuda_aware backend
      ! (device send/recv buffers passed directly to MPI, no D2H/H2D staging),
      ! selected via RUN_CUDA_FV3_MPI_TRANSPORT (host_staged|cuda_aware,
      ! default host_staged -- unset/unrecognized values are host_staged).
      mpi_transport_kind = mpi_transport_kind_from_env()
      crc = fv3_gpu_attach_mpi_halo_transport(ctx, int(comm, c_int), &
                                              int(bridge_tile, c_int32_t), &
                                              mpi_transport_kind, err)
      call check(crc, err, 'fv3_gpu_attach_mpi_halo_transport')

      if (fv3_gpu_halo_transport_kind_of(ctx) /= mpi_transport_kind) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: the requested MPI halo '// &
              'transport did not attach')
      endif

      ! Verified, not assumed: a silent fallback would invalidate any A/B
      ! between the two transports. Printed once per rank, not per message.
      if (mpi_transport_kind == FV3_GPU_HALO_TRANSPORT_MPI_CUDA_AWARE) then
         write(msg,'(a,i0,a)') 'fv3_gpu_bridge_init: rank ', comm_rank, &
              ' halo transport = cuda_aware'
      else
         write(msg,'(a,i0,a)') 'fv3_gpu_bridge_init: rank ', comm_rank, &
              ' halo transport = host_staged'
      endif
      call mpp_error(NOTE, trim(msg))

      bridge_ready = .true.
      model_step   = 0

      write(msg,'(a,i0,a,i0,a,i0)') 'fv3_gpu_bridge_init: CUDA dyn_core backend ready, rank ', &
           comm_rank, ' owns tile ', bridge_tile, ', device ', dev
      call mpp_error(NOTE, trim(msg))

   end subroutine fv3_gpu_bridge_init

! =============================================================================
! One dyn_core call
! =============================================================================

! bdt here is dyn_core's own bdt argument, i.e. fv_dynamics' mdt = bdt/k_split.
! init_step is (n_map == 1) and end_step is last_step, matching exactly what the
! Fortran dyn_core is passed at the same call site.
   subroutine fv3_gpu_interface_step_dyn(bdt, k_split, n_split, init_step, end_step, &
                                         mfx, mfy, cx, cy, ws, rc)
      real,    intent(in)  :: bdt
      integer, intent(in)  :: k_split, n_split
      logical, intent(in)  :: init_step, end_step
      ! The caller's flux capacitors for THIS dyn_core call, i.e. exactly the
      ! arrays the Fortran dyn_core would have been handed at the same call site
      ! (fv_dynamics.F90:711-713). Filled from the registry after the D2H.
      real(c_double), intent(out) :: mfx(:,:,:), mfy(:,:,:)
      real(c_double), intent(out) :: cx(:,:,:),  cy(:,:,:)
      ! The caller's ws. Like the capacitors this is an fv_dynamics local that
      ! dyn_core writes intent(out) (dyn_core.F90:224) and that
      ! Lagrangian_to_Eulerian reads intent(in) (fv_mapz.F90:159, used at :620 as
      ! the w-remap bottom boundary condition), so it must cross back per call.
      real,    intent(out) :: ws(:,:)
      integer, intent(out) :: rc

      type(fv3_gpu_step_desc)  :: sp
      type(fv3_gpu_field_mask) :: mask
      type(fv3_gpu_error)      :: err
      integer(c_int64_t)       :: ev_h2d, ev_step, ev_d2h
      integer :: crc
      character(len=160) :: msg

      rc = 0
      if (.not. bridge_ready) then
         call mpp_error(FATAL, 'fv3_gpu_interface_step_dyn: called before '// &
              'fv3_gpu_bridge_init succeeded')
      endif

      call mutable_state_mask(mask)

      ! H2D: pick up whatever tracer advection and the Lagrangian-to-Eulerian
      ! remap wrote on the host since the previous call.
      crc = fv3_gpu_sync_from_host(ctx, mask, ev_h2d, err)
      call check(crc, err, 'fv3_gpu_sync_from_host')

      sp%abi_version       = FV3_GPU_ABI_VERSION
      sp%struct_bytes      = int(storage_size(sp)/8, c_int32_t)
      sp%bdt               = real(bdt, c_double)
      sp%k_split           = int(k_split, c_int32_t)
      sp%n_split           = int(n_split, c_int32_t)
      sp%init_step         = merge(1_c_int8_t, 0_c_int8_t, init_step)
      sp%end_step          = merge(1_c_int8_t, 0_c_int8_t, end_step)
      sp%reserved0         = 0
      sp%model_step        = model_step
      sp%acoustic_sequence = 0
      sp%validation_mask   = 0
      sp%physics_tail_mask = 0
      ! Make the acoustic loop wait for the upload rather than blocking the host.
      sp%dependency        = ev_h2d

      crc = fv3_gpu_dyn_core_step(ctx, sp, ev_step, err)
      call check(crc, err, 'fv3_gpu_dyn_core_step')

      ! D2H, then the one host wait per dyn_core call. Everything after this
      ! point in fv_dynamics reads the Fortran arrays, so the wait is a defined
      ! external boundary, not a convenience sync.
      crc = fv3_gpu_sync_to_host(ctx, mask, ev_d2h, err)
      call check(crc, err, 'fv3_gpu_sync_to_host')

      crc = fv3_gpu_event_wait(ctx, ev_d2h, err)
      call check(crc, err, 'fv3_gpu_event_wait')

      ! Only now are the registry's flux capacitors on the host. Hand them to the
      ! caller's per-call arrays so fv_tracer2d and the k_split accumulation see
      ! what this dyn_core call actually produced.
      call fv3_gpu_registry_export_flux_capacitors(mfx, mfy, cx, cy, crc)
      if (crc /= 0) then
         write(msg, '(a,i0)') 'fv3_gpu_interface_step_dyn: flux capacitor '// &
              'export failed, rc=', crc
         call mpp_error(FATAL, trim(msg))
      endif

      call fv3_gpu_registry_export_ws(ws, crc)
      if (crc /= 0) then
         write(msg, '(a,i0)') 'fv3_gpu_interface_step_dyn: ws export failed, rc=', crc
         call mpp_error(FATAL, trim(msg))
      endif

      model_step = model_step + 1

   end subroutine fv3_gpu_interface_step_dyn

! =============================================================================
! fv_mapz_mod production pt/delz vertical remap (Prompt 52C-IMPLEMENT)
! =============================================================================

   ! Called from Lagrangian_to_Eulerian in place of the Fortran remap_t/delz
   ! path, ONLY for the validated production subset (remap_option==0,
   ! gmao_remap==0, kord_mt==kord_wz==kord_tm==9, nwat==0, R8) -- gating that
   ! subset is the CALLER's responsibility (fv_mapz.F90), not this
   ! subroutine's; this entry trusts it has already been checked, exactly as
   ! launch_pt_delz_remap_ctx's own header comment documents.
   !
   ! By this call site fv3_gpu_interface_step_dyn's own D2H has already made
   ! the host arrays authoritative for this dyn_core call (Stage 6/52C-prep
   ! finding), so this call stages a fresh, narrow H2D of exactly the 9 fields
   ! the kernel reads (pe, peln, pt, delz, delp, pk, pkz, ak, bk), launches the
   ! kernel, D2Hs the same 9 fields back, and blocks once -- the same
   ! one-host-wait-per-call shape as fv3_gpu_interface_step_dyn above, for the
   ! same reason: everything after this point in Lagrangian_to_Eulerian reads
   ! the Fortran arrays.
   ! Prompt 65: cuda_owns_pt (default .true., preserving all pre-existing
   ! call sites/behavior unchanged) selects between the original CUDA_PT_DELZ
   ! mode (both fields' H2D+D2H, mask unchanged) and the new hybrid
   ! CUDA_DELZ_FORTRAN_PT mode (.false.): H2D still uses the full 9-field
   ! mask (the kernel requires a valid device pt to read, even though this
   ! mode never uses its own pt/pkz output), but D2H uses the 8-field
   ! delz-only mask so host `pt` is left untouched for fv_mapz.F90's own,
   ! immediately-following Fortran map_scalar call to consume.
   subroutine fv3_gpu_interface_pt_delz_remap(rc, cuda_owns_pt)
      integer, intent(out) :: rc
      logical, intent(in), optional :: cuda_owns_pt

      type(fv3_gpu_field_mask) :: mask_h2d, mask_d2h
      type(fv3_gpu_error)      :: err
      integer(c_int64_t)       :: ev_h2d, ev_kernel, ev_d2h
      integer :: crc
      logical :: owns_pt

      rc = 0
      if (.not. bridge_ready) then
         call mpp_error(FATAL, 'fv3_gpu_interface_pt_delz_remap: called before '// &
              'fv3_gpu_bridge_init succeeded')
      endif

      owns_pt = .true.
      if (present(cuda_owns_pt)) owns_pt = cuda_owns_pt

      call pt_delz_remap_state_mask(mask_h2d)
      if (owns_pt) then
         mask_d2h = mask_h2d
      else
         call pt_delz_remap_state_mask_delz_only_d2h(mask_d2h)
      endif

      crc = fv3_gpu_sync_from_host(ctx, mask_h2d, ev_h2d, err)
      call check(crc, err, 'fv3_gpu_sync_from_host')

      crc = fv3_gpu_pt_delz_remap(ctx, ev_h2d, ev_kernel, err)
      call check(crc, err, 'fv3_gpu_pt_delz_remap')

      ! D2H, then the one host wait for this call. fv3_gpu_sync_to_host orders
      ! the transfer stream after whatever is outstanding on the compute
      ! stream (event_compute_done), which the kernel above was enqueued on,
      ! so this is correctly ordered after it without needing ev_kernel
      ! threaded through explicitly -- same mechanism
      ! fv3_gpu_interface_step_dyn relies on for its own D2H.
      crc = fv3_gpu_sync_to_host(ctx, mask_d2h, ev_d2h, err)
      call check(crc, err, 'fv3_gpu_sync_to_host')

      crc = fv3_gpu_event_wait(ctx, ev_d2h, err)
      call check(crc, err, 'fv3_gpu_event_wait')

   end subroutine fv3_gpu_interface_pt_delz_remap

! =============================================================================
! fv_mapz_mod production u/v vertical remap (Prompt 71, REMAP_UV)
! =============================================================================

   ! Called from Lagrangian_to_Eulerian in place of the Fortran u/v
   ! map_scalar path, ONLY for the validated production subset (iv=-1,
   ! kord_mt==9, R8) -- gating that subset is the CALLER's responsibility
   ! (fv_mapz.F90), not this subroutine's; this entry trusts it has already
   ! been checked, exactly as launch_remap_uv_ctx's own header comment
   ! documents.
   !
   ! MUST be called AFTER fv3_gpu_interface_pt_delz_remap within the same
   ! dyn_core call: REMAP_UV reads pe only after PT/DELZ's own closing
   ! update has finished it (see fv3_remap_uv.cu's header comment). Unlike
   ! pt_delz_remap, there is no partial-ownership hybrid mode here -- u and
   ! v are always both read and written, so a single combined H2D/D2H mask
   ! suffices (no owns_pt-style split).
   subroutine fv3_gpu_interface_remap_uv(rc)
      integer, intent(out) :: rc

      type(fv3_gpu_field_mask) :: mask
      type(fv3_gpu_error)      :: err
      integer(c_int64_t)       :: ev_h2d, ev_kernel, ev_d2h
      integer :: crc

      rc = 0
      if (.not. bridge_ready) then
         call mpp_error(FATAL, 'fv3_gpu_interface_remap_uv: called before '// &
              'fv3_gpu_bridge_init succeeded')
      endif

      call remap_uv_state_mask(mask)

      crc = fv3_gpu_sync_from_host(ctx, mask, ev_h2d, err)
      call check(crc, err, 'fv3_gpu_sync_from_host')

      crc = fv3_gpu_remap_uv(ctx, ev_h2d, ev_kernel, err)
      call check(crc, err, 'fv3_gpu_remap_uv')

      crc = fv3_gpu_sync_to_host(ctx, mask, ev_d2h, err)
      call check(crc, err, 'fv3_gpu_sync_to_host')

      crc = fv3_gpu_event_wait(ctx, ev_d2h, err)
      call check(crc, err, 'fv3_gpu_event_wait')

   end subroutine fv3_gpu_interface_remap_uv

! =============================================================================
! fv_mapz_mod production tracer vertical remap (Prompt 85, REMAP_Q)
! =============================================================================

   ! Called from Lagrangian_to_Eulerian in place of the Fortran mapn_tracer
   ! path, ONLY for the validated production subset (nq==2, kord_tr(1)==9
   ! [sphum], cld_amt's own kord==9, R8) -- gating that subset is the
   ! CALLER's responsibility (fv_mapz.F90), not this subroutine's; this
   ! entry trusts it has already been checked, exactly as
   ! launch_remap_q_ctx's own header comment documents. (Prompt 86B: an
   ! earlier version of this comment read "kord_tr(1)==9 per this
   ! codebase's established '+1' kord convention" -- a copy-paste error from
   ! the neighboring PT_DELZ/REMAP_UV blocks, which DO use that convention;
   ! REMAP_Q's own selector has always checked kord_tr(1) directly, with no
   ! offset. Prompt 87 found the real Fortran value of kord_tr(1) is 9, not
   ! the 8 Prompts 85/86A/86B assumed -- FV_StateMod.F90:563 unconditionally
   ! overrides fv_arrays.F90:333's bare default -- so kord_tr(1)==9 is now
   ! correct here for a genuinely different reason than the "+1" convention
   ! this comment previously (correctly) disclaimed; see
   ! docs/performance/p87_remap_q_actual_kord9_production_2026-09-25.md.)
   !
   ! MUST be called AFTER fv3_gpu_interface_pt_delz_remap within the same
   ! dyn_core call: REMAP_Q reads pe only after PT/DELZ's own closing
   ! update has finished it (same ordering constraint as REMAP_UV; see
   ! fv3_remap_q.cu's header comment). Unlike pt_delz_remap/remap_uv, q has
   ! no FieldRegistry identity, so there is no mask/sync_from_host/
   ! sync_to_host pair here -- fv3_gpu_remap_q owns its own single H2D
   ! (before the kernel) and single D2H (after) directly against the `q`
   ! actual argument, via a persistent context-owned device buffer.
   ! `q_elements` must equal (ied-isd+1)*(jed-jsd+1)*km*nq for this tile's
   ! current bounds, matching fv3_gpu_remap_q's own contract -- the call
   ! fails closed (FV3_GPU_INVALID_ARGUMENT) on any mismatch. `nq`/`kord1`/
   ! `kord2` are passed through as literal 2/9/9 (Prompt 86B/87 -- corrected
   ! from the originally assumed 2/8/9) -- the only validated production
   ! schedule; fv3_gpu_remap_q itself re-validates this as defense-in-depth.
   ! `dependency`
   ! is passed as 0: unlike pt_delz_remap/remap_uv (whose dependency is the
   ! caller's own external sync_from_host event), q's H2D is internal to
   ! fv3_gpu_remap_q itself, and both it and the kernel launch are enqueued
   ! on this context's existing transfer/compute streams in call order, so
   ! single-stream issue ordering already sequences this correctly after
   ! PT/DELZ's kernel and after any prior call's D2H of the same buffer --
   ! no separate host-visible event needs to be threaded through.
   !
   ! `q` is plain `real` (the caller's build-native kind, R4 or R8) rather
   ! than real(c_double), matching fv3_gpu_interface_fv_tp_2d_hord8's own
   ! precedent (fv3_gpu_api.h's fv_tp_2d_hord8 comment) -- this keeps this
   ! subroutine's explicit interface compile-clean in an R4 build too (this
   ! repo's own build_r4 target), even though the caller (fv_mapz.F90's
   ! use_cuda_remap_q gate) only ever DISPATCHES to this call in an R8
   ! build. The local q_d conversion below is therefore a no-op value copy
   ! whenever this actually runs (R8 in, real(c_double) kind=8 out, same
   ! bit layout) -- one bulk array assignment each way, not the per-field
   ! scalar marshaling loop Prompt 82 found responsible for the TRACER_2D
   ! regression.
   !
   ! Prompt 88B: `fill` is the caller's flagstruct%fill (logical), converted
   ! to the plain 0/1 integer fv3_gpu_remap_q's C ABI expects. Applied
   ! identically to both tracers by fv3_gpu_remap_q/launch_remap_q_ctx --
   ! there is no per-tracer fill argument here because flagstruct%fill is
   ! one global value, not per-tracer (see fv3_remap_q.cuh header comment).
   subroutine fv3_gpu_interface_remap_q(rc, q, q_elements, fill)
      integer, intent(out) :: rc
      real, intent(inout)            :: q(*)
      integer(c_int64_t), intent(in) :: q_elements
      logical, intent(in)            :: fill

      real(c_double), allocatable :: q_d(:)
      type(fv3_gpu_error) :: err
      integer(c_int64_t)  :: ev_kernel
      integer(c_int32_t)  :: fill_flag
      integer :: crc

      rc = 0
      if (.not. bridge_ready) then
         call mpp_error(FATAL, 'fv3_gpu_interface_remap_q: called before '// &
              'fv3_gpu_bridge_init succeeded')
      endif

      allocate(q_d(q_elements))
      q_d = real(q(1:q_elements), c_double)

      fill_flag = 0_c_int32_t
      if (fill) fill_flag = 1_c_int32_t

      crc = fv3_gpu_remap_q(ctx, 0_c_int64_t, q_d, q_elements, &
                            2_c_int32_t, 9_c_int32_t, 9_c_int32_t, fill_flag, &
                            ev_kernel, err)
      call check(crc, err, 'fv3_gpu_remap_q')

      crc = fv3_gpu_event_wait(ctx, ev_kernel, err)
      call check(crc, err, 'fv3_gpu_event_wait')

      q(1:q_elements) = real(q_d, kind(q))
      deallocate(q_d)

   end subroutine fv3_gpu_interface_remap_q

! =============================================================================
! fv_mapz_mod production pressure-variable vertical remap
! (Prompt 92, REMAP_PRESSUREVARS)
! =============================================================================

   ! Called from Lagrangian_to_Eulerian in place of the Fortran
   ! pe2/pn2/pk2 recompute + delp/cappa/pkz block (fv_mapz.F90:1239-1516,
   ! timer 'Remap_PressureVars') -- gating the validated production subset
   ! (nwat==0, R8) is the CALLER's responsibility (fv_mapz.F90), not this
   ! subroutine's; this entry trusts it has already been checked, exactly
   ! as launch_remap_pressurevars_ctx's own header comment documents.
   !
   ! MUST be called AFTER fv3_gpu_interface_pt_delz_remap (and, when both
   ! run, after fv3_gpu_interface_remap_uv/fv3_gpu_interface_remap_q)
   ! within the same dyn_core call: this entry reads pe only after PT/
   ! DELZ's own closing update has finished it (see
   ! fv3_remap_pressurevars.cu's header comment).
   !
   ! q(sphum)/cappa have no FieldRegistry identity (like remap_q's q), so
   ! fv3_gpu_remap_pressurevars owns their own H2D (q_sphum read-only,
   ! cappa round-tripped) and cappa's own D2H directly against the actual
   ! arguments below, via two persistent context-owned device buffers --
   ! there is no mask/sync_from_host/sync_to_host pair for THOSE two
   ! fields. pe/peln/pk/delp/pkz/pt/delz/ak/bk remain ordinary
   ! FieldRegistry fields and DO go through the usual mask/sync_from_host/
   ! sync_to_host pair here, split into a narrower H2D-only mask (ak, bk,
   ! pe, pt, delz -- the fields this kernel reads as input) and a
   ! narrower D2H-only mask (pe, peln, pk, pkz, delp -- the fields this
   ! kernel writes), rather than one combined mask covering all 9: pt and
   ! delz are read-only here (never rewritten by this kernel) and pkz is
   ! write-only (never read), so a combined mask would move bytes neither
   ! direction needs, contrary to project_architecture.md's "avoid
   ! unnecessary full-state host/device transfers".
   !
   ! `q_sphum`/`cappa` are plain `real` (the caller's build-native kind,
   ! R4 or R8), matching fv3_gpu_interface_remap_q's own precedent -- this
   ! keeps this subroutine's explicit interface compile-clean in an R4
   ! build too, even though the caller only ever dispatches to this call
   ! in an R8 build. `elements` must equal (ied-isd+1)*(jed-jsd+1)*npz for
   ! this tile's current bounds; the call fails closed
   ! (FV3_GPU_INVALID_ARGUMENT) on any mismatch.
   subroutine fv3_gpu_interface_remap_pressurevars(rc, q_sphum, cappa, elements)
      integer, intent(out) :: rc
      real, intent(in)               :: q_sphum(*)
      real, intent(inout)            :: cappa(*)
      integer(c_int64_t), intent(in) :: elements

      real(c_double), allocatable :: q_sphum_d(:), cappa_d(:)
      type(fv3_gpu_field_mask) :: mask_h2d, mask_d2h
      type(fv3_gpu_error)      :: err
      integer(c_int64_t)       :: ev_h2d, ev_kernel, ev_d2h
      integer :: crc

      rc = 0
      if (.not. bridge_ready) then
         call mpp_error(FATAL, 'fv3_gpu_interface_remap_pressurevars: '// &
              'called before fv3_gpu_bridge_init succeeded')
      endif

      allocate(q_sphum_d(elements), cappa_d(elements))
      q_sphum_d = real(q_sphum(1:elements), c_double)
      cappa_d   = real(cappa(1:elements),   c_double)

      call pressurevars_state_mask_h2d(mask_h2d)
      call pressurevars_state_mask_d2h(mask_d2h)

      crc = fv3_gpu_sync_from_host(ctx, mask_h2d, ev_h2d, err)
      call check(crc, err, 'fv3_gpu_sync_from_host')

      crc = fv3_gpu_remap_pressurevars(ctx, ev_h2d, q_sphum_d, cappa_d, &
                                       elements, ev_kernel, err)
      call check(crc, err, 'fv3_gpu_remap_pressurevars')

      crc = fv3_gpu_sync_to_host(ctx, mask_d2h, ev_d2h, err)
      call check(crc, err, 'fv3_gpu_sync_to_host')

      crc = fv3_gpu_event_wait(ctx, ev_d2h, err)
      call check(crc, err, 'fv3_gpu_event_wait')

      ! ev_kernel is not separately waited on: it orders cappa's own D2H
      ! (inside fv3_gpu_remap_pressurevars) after the kernel on the same
      ! compute/transfer stream pair, and ev_d2h above (the FieldRegistry
      ! D2H for pe/peln/pk/pkz/delp) is issued and waited on strictly after
      ! that call returns -- same reliance on single-stream issue order as
      ! fv3_gpu_interface_remap_q's own ev_kernel handling.
      cappa(1:elements) = real(cappa_d, kind(cappa))
      deallocate(q_sphum_d, cappa_d)

   end subroutine fv3_gpu_interface_remap_pressurevars

! =============================================================================
! FV_StateMod.F90 production State_To_FV, Phase A/C halves (Prompt 99)
! =============================================================================

   ! Called from FV_StateMod.F90's State_To_FV in place of the Fortran
   ! u(:,:,:)=tiny_number / v(:,:,:)=tiny_number full-buffer fills and the
   ! FV_Atm(1)%u(isc:iec,jsc:jec,:)=STATE%VARS%U / v(...)=STATE%VARS%V
   ! interior copies (FV_StateMod.F90:2571-2588, the grid_type<4 production
   ! branch) -- gating that production subset (grid_type<4, R8,
   ! ADJUST_SPLITS=.false. -- see below) is the CALLER's responsibility, not
   ! this subroutine's.
   !
   ! This entry does NOT call mpp_get_boundary or perform the nbuffer/
   ! ebuffer edge-row scatter -- those stay verbatim, host-side, in
   ! FV_StateMod.F90. This entry's own D2H (below) is what makes the
   ! UNCHANGED mpp_get_boundary call able to read this kernel's output: it
   ! brings the just-computed FV_Atm(1)%u/v interior values to the host,
   ! exactly where the Fortran source's own assignment would have left
   ! them. The caller must call mpp_get_boundary/do the edge scatter on
   ! FV_Atm(1)%u/v (host) immediately after this call returns, and then
   ! call fv3_gpu_interface_state_to_fv_phase_c (which brings the halo-
   ! updated host u/v back to the device as ITS first action) before
   ! continuing.
   !
   ! ADJUST_SPLITS SCOPE NOTE: FV_StateMod.F90:2685-2694 calls
   ! adjust_fv3_splits(...) using FV_Atm(1)%u/v/w/delz directly as HOST
   ! arrays, AFTER Phase C would normally leave w/delz device-resident
   ! only (see fv3_gpu_interface_state_to_fv_phase_c below). Neither this
   ! entry nor Phase C perform the D2H adjust_fv3_splits would need. The
   ! CUDA selector in FV_StateMod.F90 MUST reject ADJUST_SPLITS=.true.
   ! (fall back to Fortran) rather than silently feeding adjust_fv3_splits
   ! stale host w/delz -- ADJUST_SPLITS defaults to .false. and is absent
   ! from the production first-run configuration project_architecture.md
   ! documents, so this is a real but narrow scope boundary, not a defect.
   !
   ! `state_u`/`state_v` are plain `real` (build-native kind), matching
   ! fv3_gpu_interface_remap_q's own precedent. `elements` must equal
   ! (ie-is+1)*(je-js+1)*npz for this tile's current bounds; the call
   ! fails closed (FV3_GPU_INVALID_ARGUMENT) on any mismatch.
   subroutine fv3_gpu_interface_state_to_fv_phase_a(rc, state_u, state_v, elements)
      integer, intent(out) :: rc
      real, intent(in)               :: state_u(*), state_v(*)
      integer(c_int64_t), intent(in) :: elements

      real(c_double), allocatable :: state_u_d(:), state_v_d(:)
      type(fv3_gpu_field_mask) :: mask_uv
      type(fv3_gpu_error)      :: err
      integer(c_int64_t)       :: ev_kernel, ev_d2h
      integer :: crc

      rc = 0
      if (.not. bridge_ready) then
         call mpp_error(FATAL, 'fv3_gpu_interface_state_to_fv_phase_a: '// &
              'called before fv3_gpu_bridge_init succeeded')
      endif

      allocate(state_u_d(elements), state_v_d(elements))
      state_u_d = real(state_u(1:elements), c_double)
      state_v_d = real(state_v(1:elements), c_double)

      crc = fv3_gpu_state_to_fv_phase_a(ctx, 0_c_int64_t, state_u_d, state_v_d, &
                                        elements, ev_kernel, err)
      call check(crc, err, 'fv3_gpu_state_to_fv_phase_a')

      call state_to_fv_uv_mask(mask_uv)
      crc = fv3_gpu_sync_to_host(ctx, mask_uv, ev_d2h, err)
      call check(crc, err, 'fv3_gpu_sync_to_host')

      crc = fv3_gpu_event_wait(ctx, ev_d2h, err)
      call check(crc, err, 'fv3_gpu_event_wait')

      deallocate(state_u_d, state_v_d)

   end subroutine fv3_gpu_interface_state_to_fv_phase_a

   ! Called from FV_StateMod.F90's State_To_FV immediately after the
   ! caller's own UNCHANGED mpp_get_boundary call and edge-row scatter (see
   ! fv3_gpu_interface_state_to_fv_phase_a above), in place of the Fortran
   ! w interior copy (hydrostatic-gated), the pe(:,:,:)=tiny_number fill +
   ! full pe transpose/peln/pk recompute over all km+1 levels, the ps pack,
   ! the delp(:,:,:)=tiny_number fill + recompute, the pt(:,:,:)=
   ! tiny_number fill + recompute, and the delz/pkz interior copies
   ! (FV_StateMod.F90:2604-2680, the SW_DYNAMICS=.false. production
   ! branch).
   !
   ! This entry's FIRST action is fv3_gpu_sync_from_host on the u/v-only
   ! mask -- bringing the host u/v the caller's mpp_get_boundary call just
   ! updated back onto the device, closing the device -> host -> device
   ! round trip Phase A's header comment documents. Phase C's own kernel
   ! does not read u/v; this exists solely so the FieldRegistry's device-
   ! resident u/v stay consistent with the host before fv_dynamics() (the
   ! next production call) consumes them from the device.
   !
   ! w/pe/peln/pk/delp/pt/delz/pkz are left DEVICE-RESIDENT ONLY after this
   ! call -- no D2H for them here, by design: their consumer is the
   ! already-GPU-resident fv_dynamics() CUDA path, not host code (see
   ! docs/performance/p98_state_to_fv_cuda_port_plan_2026-09-28.md's
   ! residency-direction finding). See fv3_gpu_interface_state_to_fv_
   ! phase_a's ADJUST_SPLITS scope note above -- the CUDA selector must
   ! reject ADJUST_SPLITS=.true., which is the one host consumer of w/delz
   ! this design does not serve.
   !
   ! `state_w`/`state_pe`/`state_pt`/`state_pkz`/`state_dz` are plain
   ! `real`. `elements` must equal (ie-is+1)*(je-js+1)*npz and
   ! `elements_pe` must equal (ie-is+1)*(je-js+1)*(npz+1); the call fails
   ! closed (FV3_GPU_INVALID_ARGUMENT) on any mismatch. `ps` is returned to
   ! the caller directly (sized elements_pe - elements == (ie-is+1)*
   ! (je-js+1), the one contiguous D2H fv3_gpu_state_to_fv_phase_c's own
   ! ABI comment documents).
   subroutine fv3_gpu_interface_state_to_fv_phase_c(rc, state_w, state_pe, state_pt, &
                                                     state_pkz, state_dz, ps, &
                                                     elements, elements_pe)
      integer, intent(out) :: rc
      real, intent(in)               :: state_w(*), state_pe(*), state_pt(*), &
                                         state_pkz(*), state_dz(*)
      real, intent(out)              :: ps(*)
      integer(c_int64_t), intent(in) :: elements, elements_pe

      real(c_double), allocatable :: state_w_d(:), state_pe_d(:), state_pt_d(:), &
                                      state_pkz_d(:), state_dz_d(:), ps_d(:)
      type(fv3_gpu_field_mask) :: mask_uv
      type(fv3_gpu_error)      :: err
      integer(c_int64_t)       :: ev_h2d_uv, ev_kernel
      integer(c_int64_t)       :: elements_ps
      integer :: crc

      rc = 0
      if (.not. bridge_ready) then
         call mpp_error(FATAL, 'fv3_gpu_interface_state_to_fv_phase_c: '// &
              'called before fv3_gpu_bridge_init succeeded')
      endif

      call state_to_fv_uv_mask(mask_uv)
      crc = fv3_gpu_sync_from_host(ctx, mask_uv, ev_h2d_uv, err)
      call check(crc, err, 'fv3_gpu_sync_from_host')

      crc = fv3_gpu_event_wait(ctx, ev_h2d_uv, err)
      call check(crc, err, 'fv3_gpu_event_wait')

      elements_ps = elements_pe - elements

      allocate(state_w_d(elements), state_pe_d(elements_pe), state_pt_d(elements), &
               state_pkz_d(elements), state_dz_d(elements), ps_d(elements_ps))

      state_w_d   = real(state_w(1:elements),     c_double)
      state_pe_d  = real(state_pe(1:elements_pe), c_double)
      state_pt_d  = real(state_pt(1:elements),    c_double)
      state_pkz_d = real(state_pkz(1:elements),   c_double)
      state_dz_d  = real(state_dz(1:elements),    c_double)

      crc = fv3_gpu_state_to_fv_phase_c(ctx, 0_c_int64_t, state_w_d, state_pe_d, &
                                        state_pt_d, state_pkz_d, state_dz_d, ps_d, &
                                        elements, elements_pe, ev_kernel, err)
      call check(crc, err, 'fv3_gpu_state_to_fv_phase_c')

      crc = fv3_gpu_event_wait(ctx, ev_kernel, err)
      call check(crc, err, 'fv3_gpu_event_wait')

      ps(1:elements_ps) = real(ps_d, kind(ps))

      deallocate(state_w_d, state_pe_d, state_pt_d, state_pkz_d, state_dz_d, ps_d)

   end subroutine fv3_gpu_interface_state_to_fv_phase_c

   !==========================================================================
   ! Prompt 77: single (k, iq) tracer-transport slice for
   ! fv_tracer2d.F90::tracer_2d_1L's nsplt==1 fv_tp_2d call site. Context-free
   ! and FieldRegistry-free, mirroring fv3_gpu_fv_tp_2d_hord8's own design
   ! (fv3_gpu_api.h:210-249) -- no ctx argument, no fv3_gpu_create/finalize
   ! interaction; tracer temporaries have no FieldRegistry identity.
   !
   ! Dummy arguments use plain `real` (the caller's build-native kind, R4 or
   ! R8) and shapes matching fv_tp_2d's own dummy-argument bounds exactly
   ! (tp_core.F90:109-138), so a caller can substitute this call one-for-one.
   ! Every array is converted to real(c_double) before crossing the ABI
   ! (fixed at double regardless of build precision -- see fv3_gpu_api.h's
   ! fv3_gpu_fv_tp_2d_hord8 comment) and fx/fy are converted back on return.
   ! This subroutine does not itself gate on build precision or hord; the
   ! caller (fv_tracer2d.F90's use_cuda_tp2d selector) is responsible for
   ! calling this only when eligible.
   subroutine fv3_gpu_interface_fv_tp_2d_hord8( &
         is, ie, js, je, isd, ied, jsd, jed, npx, npy, hord, &
         cubed_sphere, nested, west_edge, east_edge, south_edge, north_edge, &
         sw_corner, se_corner, ne_corner, nw_corner, &
         q, crx, cry, xfx, yfx, mfx, mfy, ra_x, ra_y, area, dxa, dya, fx, fy)
      integer, intent(in) :: is, ie, js, je, isd, ied, jsd, jed, npx, npy, hord
      logical, intent(in) :: cubed_sphere, nested
      logical, intent(in) :: west_edge, east_edge, south_edge, north_edge
      logical, intent(in) :: sw_corner, se_corner, ne_corner, nw_corner
      real,    intent(inout) :: q(isd:ied, jsd:jed)
      real,    intent(in)    :: crx(is:ie+1, jsd:jed), xfx(is:ie+1, jsd:jed)
      real,    intent(in)    :: cry(isd:ied, js:je+1), yfx(isd:ied, js:je+1)
      real,    intent(in)    :: mfx(is:ie+1, js:je), mfy(is:ie, js:je+1)
      real,    intent(in)    :: ra_x(is:ie, jsd:jed), ra_y(isd:ied, js:je)
      real,    intent(in)    :: area(isd:ied, jsd:jed), dxa(isd:ied, jsd:jed), dya(isd:ied, jsd:jed)
      real,    intent(out)   :: fx(is:ie+1, js:je), fy(is:ie, js:je+1)

      real(c_double) :: q_d(isd:ied, jsd:jed)
      real(c_double) :: crx_d(is:ie+1, jsd:jed), xfx_d(is:ie+1, jsd:jed)
      real(c_double) :: cry_d(isd:ied, js:je+1), yfx_d(isd:ied, js:je+1)
      real(c_double) :: mfx_d(is:ie+1, js:je), mfy_d(is:ie, js:je+1)
      real(c_double) :: ra_x_d(is:ie, jsd:jed), ra_y_d(isd:ied, js:je)
      real(c_double) :: area_d(isd:ied, jsd:jed), dxa_d(isd:ied, jsd:jed), dya_d(isd:ied, jsd:jed)
      real(c_double) :: fx_d(is:ie+1, js:je), fy_d(is:ie, js:je+1)

      type(fv3_gpu_tracer_transport_bounds) :: b
      type(fv3_gpu_error) :: err
      integer :: crc

      if (.not. bridge_ready) then
         call mpp_error(FATAL, 'fv3_gpu_interface_fv_tp_2d_hord8: called before '// &
              'fv3_gpu_bridge_init succeeded')
      endif

      b%abi_version  = FV3_GPU_ABI_VERSION
      b%struct_bytes = int(storage_size(b)/8, c_int32_t)
      b%is  = int(is,  c_int64_t); b%ie  = int(ie,  c_int64_t)
      b%js  = int(js,  c_int64_t); b%je  = int(je,  c_int64_t)
      b%isd = int(isd, c_int64_t); b%ied = int(ied, c_int64_t)
      b%jsd = int(jsd, c_int64_t); b%jed = int(jed, c_int64_t)
      b%npx = int(npx, c_int64_t); b%npy = int(npy, c_int64_t)
      b%hord = int(hord, c_int32_t)
      b%cubed_sphere = merge(1_c_int32_t, 0_c_int32_t, cubed_sphere)
      b%nested       = merge(1_c_int32_t, 0_c_int32_t, nested)
      b%west_edge  = merge(1_c_int32_t, 0_c_int32_t, west_edge)
      b%east_edge  = merge(1_c_int32_t, 0_c_int32_t, east_edge)
      b%south_edge = merge(1_c_int32_t, 0_c_int32_t, south_edge)
      b%north_edge = merge(1_c_int32_t, 0_c_int32_t, north_edge)
      b%sw_corner = merge(1_c_int32_t, 0_c_int32_t, sw_corner)
      b%se_corner = merge(1_c_int32_t, 0_c_int32_t, se_corner)
      b%ne_corner = merge(1_c_int32_t, 0_c_int32_t, ne_corner)
      b%nw_corner = merge(1_c_int32_t, 0_c_int32_t, nw_corner)

      q_d    = real(q,    c_double)
      crx_d  = real(crx,  c_double); xfx_d = real(xfx, c_double)
      cry_d  = real(cry,  c_double); yfx_d = real(yfx, c_double)
      mfx_d  = real(mfx,  c_double); mfy_d = real(mfy, c_double)
      ra_x_d = real(ra_x, c_double); ra_y_d = real(ra_y, c_double)
      area_d = real(area, c_double)
      dxa_d  = real(dxa,  c_double); dya_d = real(dya, c_double)

      crc = fv3_gpu_fv_tp_2d_hord8(b, q_d, crx_d, cry_d, xfx_d, yfx_d, &
                                   mfx_d, mfy_d, ra_x_d, ra_y_d, &
                                   area_d, dxa_d, dya_d, fx_d, fy_d, err)
      call check(crc, err, 'fv3_gpu_fv_tp_2d_hord8')

      q  = real(q_d,  kind(q))
      fx = real(fx_d, kind(fx))
      fy = real(fy_d, kind(fy))

   end subroutine fv3_gpu_interface_fv_tp_2d_hord8

   !==========================================================================
   ! Complete pe's one halo ring at the dyn_core exit boundary.
   !
   ! WHY THIS IS NEEDED, and why it is a boundary completion rather than a
   ! numerical change. pe is declared pe(is-1:ie+1, 1:npz+1, js-1:je+1) -- the
   ! owned box plus exactly one ring -- and the authoritative Fortran fills that
   ! ring inside dyn_core with pe_halo (dyn_core.F90:1504-1532), called at
   ! :1011 under "if (remap_step)", which is true at it==n_split; no delp write
   ! follows it inside dyn_core, so the exit ring is exactly the cumulative sum
   ! of the exit delp ring starting from ptop. The CUDA dyn_core never writes
   ! that ring (a coverage gap already on record), so with the CUDA backend the
   ! D2H hands the ring straight back to the host holding whatever it was
   ! uploaded with -- i.e. the PREVIOUS call's values.
   !
   ! That ring is NOT inert. Lagrangian_to_Eulerian's D-grid wind and flux
   ! vertical remap reads it directly:
   !   fv_mapz.F90:686  pe1(i,k) = 0.5*(pe(i,k,j-1)+pe(i,k,j))   , j=js..je+1
   !   fv_mapz.F90:693  pe2 likewise from pe(i,km+1,j-1)
   !   fv_mapz.F90:729  pe0(i,k) = 0.5*(pe(i-1,k,j)+pe(i,k,j))   , i=is..ie+1
   !   fv_mapz.F90:740  pe3 likewise from pe(i-1,km+1,j)
   ! so u/mfy/cy at j==js and j==je+1 are remapped against pe(:,:,js-1) and
   ! pe(:,:,je+1), and v/mfx/cx at i==is and i==ie+1 against pe(is-1,:,:) and
   ! pe(ie+1,:,:). Measured signature with the ring stale: u differs from the
   ! Fortran backend at exactly j=js and j=je+1, v at exactly i=is and i=ie+1,
   ! O(1) magnitude, everything else in the owned box at round-off -- which is
   ! this footprint and no other. (The uniform ~1e-4 pt/pkz difference that
   ! follows is the consv_te=1.0 energy fixer redistributing the resulting
   ! total-energy error.)
   !
   ! Done on the host rather than on the device deliberately: delp's own halo
   ! ring is already host-resident after the D2H (delp is registered over
   ! isd:ied,jsd:jed), the arithmetic is a k-cumulative sum in the same order
   ! and the same precision as pe_halo, so the result is bit-for-bit what the
   ! Fortran produces, and no host/device transfer and no device kernel are
   ! added. This is the same category as the ws and flux-capacitor copy-outs
   ! above: completing state that crosses the dyn_core boundary, not changing
   ! any numerical algorithm.
   !==========================================================================
   subroutine fv3_gpu_interface_pe_halo(is, ie, js, je, isd, ied, jsd, jed, &
                                        npz, ptop, pe, delp)
      integer, intent(in) :: is, ie, js, je, isd, ied, jsd, jed, npz
      real,    intent(in) :: ptop
      real,    intent(in)    :: delp(isd:ied, jsd:jed, npz)
      real,    intent(inout) :: pe(is-1:ie+1, npz+1, js-1:je+1)

      integer :: i, j, k

      if (.not. bridge_ready) then
         call mpp_error(FATAL, 'fv3_gpu_interface_pe_halo: called before '// &
              'fv3_gpu_bridge_init succeeded')
      endif

      ! Verbatim structure of dyn_core.F90:1512-1530 (west/east columns first,
      ! then south/north rows over the already-filled i range).
      do j = js, je
         pe(is-1,1,j) = ptop
         pe(ie+1,1,j) = ptop
         do k = 1, npz
            pe(is-1,k+1,j) = pe(is-1,k,j) + delp(is-1,j,k)
            pe(ie+1,k+1,j) = pe(ie+1,k,j) + delp(ie+1,j,k)
         enddo
      enddo

      do i = is-1, ie+1
         pe(i,1,js-1) = ptop
         pe(i,1,je+1) = ptop
         do k = 1, npz
            pe(i,k+1,js-1) = pe(i,k,js-1) + delp(i,js-1,k)
            pe(i,k+1,je+1) = pe(i,k,je+1) + delp(i,je+1,k)
         enddo
      enddo

   end subroutine fv3_gpu_interface_pe_halo

! =============================================================================
! Teardown
! =============================================================================

   subroutine fv3_gpu_bridge_final(rc)
      integer, intent(out) :: rc
      type(fv3_gpu_error) :: err
      integer :: crc

      rc = 0
      if (.not. bridge_ready) return

      ! Every rank must be past its last exchange before any rank detaches --
      ! the same obligation MPI_Finalize imposes.
      crc = fv3_gpu_detach_halo_transport(ctx, err)
      call check(crc, err, 'fv3_gpu_detach_halo_transport')

      crc = fv3_gpu_finalize(ctx, err)
      call check(crc, err, 'fv3_gpu_finalize')

      call fv3_gpu_registry_free()
      if (allocated(descs)) deallocate(descs)
      ndesc        = 0
      bridge_ready = .false.
      ctx          = C_NULL_PTR

   end subroutine fv3_gpu_bridge_final

! =============================================================================
! Helpers
! =============================================================================

! -----------------------------------------------------------------------------
! Numerical configuration
! -----------------------------------------------------------------------------
!
! Every member is taken from the authoritative Fortran quantity the corresponding
! dyn_core code reads, NOT from the standalone benchmark harness. That distinction
! is deliberate: CURRENT_STATE.md section 4 records thirteen input-fidelity
! defects found in Gate C2, of which four (n_sponge, n_con, nf_ke, cv_air) were
! harness values with plausible-looking in-source justifications. The harness is a
! cross-check here, not the authority. Two values below differ from it for exactly
! that reason and are flagged inline.
!
! Where dyn_core DERIVES a quantity rather than reading a flag, the derivation is
! reproduced from the authoritative line rather than the result being hardcoded,
! so a different namelist still yields the right answer.
   subroutine fv3_gpu_fill_numerical_config(Atm, cfg)
      type(fv_atmos_type),            intent(in)  :: Atm
      type(fv3_gpu_numerical_config), intent(out) :: cfg
      integer :: nord_h, n_con_l

      cfg%abi_version  = FV3_GPU_ABI_VERSION
      cfg%struct_bytes = int(storage_size(cfg)/8, c_int32_t)

      ! ---- scalars the acoustic loop reads ------------------------------------
      cfg%ptop  = real(Atm%ptop, c_double)
      ! Two kappas, deliberately. cfg%akap is the CLOSURE kappa driving cappa(i,j,k)
      ! -- FMS double dry rdgas/cp_air, exactly 2/7. cfg%akap_exner is the Exner
      ! kappa dyn_core receives as its `akap` dummy, which FV_StateMod.F90:501 sets
      ! from the SINGLE-precision MAPL_KAPPA. Collapsing them regresses SP1.
      cfg%akap       = real(rdgas/cp_air, c_double)
      cfg%akap_exner = real(MAPL_KAPPA,   c_double)
      cfg%rdgas      = real(rdgas, c_double)
      cfg%grav       = real(grav,  c_double)
      ! FV_StateMod.F90:504's own expression, not a literal.
      cfg%zvir       = real(MAPL_RVAP/MAPL_RGAS - 1.0, c_double)

      cfg%a_imp   = real(Atm%flagstruct%a_imp,   c_double)
      cfg%p_fac   = real(Atm%flagstruct%p_fac,   c_double)
      ! CUDA's scale_m IS Fortran's scale_z: dyn_core.F90:583/:922 pass scale_z
      ! into the argument the port named scale_m. The harness uses 1.0; the
      ! authoritative value is fv_arrays.F90:337's 0.0.
      cfg%scale_m = real(Atm%flagstruct%scale_z, c_double)
      cfg%dz_min  = real(Atm%flagstruct%dz_min,  c_double)
      cfg%beta    = real(Atm%flagstruct%beta,    c_double)
      cfg%d_ext   = real(Atm%flagstruct%d_ext,   c_double)
      cfg%dddmp   = real(Atm%flagstruct%dddmp,   c_double)
      cfg%d_con   = real(Atm%flagstruct%d_con,   c_double)
      cfg%lim_fac = real(Atm%flagstruct%lim_fac, c_double)
      cfg%delt_max = real(Atm%flagstruct%delt_max, c_double)

      ! Grid-derived areas: scalars on gridstruct, computed by fv_grid_tools from
      ! the real grid, so they are authoritative by construction here rather than
      ! needing the override path the validation harness used.
      cfg%da_min   = real(Atm%gridstruct%da_min,   c_double)
      cfg%da_min_c = real(Atm%gridstruct%da_min_c, c_double)

      cfg%rayleigh_tau  = real(Atm%flagstruct%tau, c_double)
      ! No fv_flags_type counterpart exists. rf_cutoff is a PRESSURE in Pa, so it
      ! is not this quantity under another name; left zero rather than guessed.
      ! Inert here: rayleigh_enabled below is false.
      cfg%rayleigh_ztop = 0.0_c_double

      ! ---- transport / damping orders -----------------------------------------
      cfg%hord_mt = int(Atm%flagstruct%hord_mt, c_int32_t)
      cfg%hord_vt = int(Atm%flagstruct%hord_vt, c_int32_t)
      cfg%hord_tm = int(Atm%flagstruct%hord_tm, c_int32_t)
      cfg%hord_dp = int(Atm%flagstruct%hord_dp, c_int32_t)   ! sign is load-bearing
      cfg%hord_tr = int(Atm%flagstruct%hord_tr, c_int32_t)
      ! No Fortran counterpart. kord_wz is a vertical-REMAP order, a different
      ! operator, so it is not claimed as the equivalent.
      cfg%hord_height = 0
      cfg%a2b_order   = int(Atm%flagstruct%a2b_ord, c_int32_t)

      nord_h      = Atm%flagstruct%nord
      cfg%nord    = int(nord_h, c_int32_t)
      ! dyn_core.F90:743-745 -- nord_v is min(2,nord) and nord_w/nord_t follow it.
      cfg%nord_v  = int(min(2, nord_h), c_int32_t)
      cfg%nord_w  = int(min(2, nord_h), c_int32_t)
      cfg%nord_t  = int(min(2, nord_h), c_int32_t)
      ! dyn_core.F90:1310. Derived, never pinned -- pinning this to 1 was defect
      ! #13, and it masked the missing corner service that was defect #12.
      cfg%nf_ke   = int(min(3, nord_h + 1), c_int32_t)

      ! dyn_core.F90:405-412, reproduced as the same ladder. n_con bounds how many
      ! LEVELS of the post-loop pt update run; hardcoding it to 1 was defect #11.
      if (Atm%flagstruct%convert_ke .or. Atm%flagstruct%vtdm4 > 1.e-4) then
         n_con_l = Atm%npz
      else if (Atm%flagstruct%d2_bg_k1 < 1.e-3) then
         n_con_l = 0
      else if (Atm%flagstruct%d2_bg_k2 < 1.e-3) then
         n_con_l = 1
      else
         n_con_l = 2
      endif
      cfg%n_con = int(n_con_l, c_int32_t)

      ! No enum exists on either side; the vertical solver is selected from a_imp
      ! alone, in Fortran at nh_core.F90:160-176 and identically in CUDA. Left
      ! zero, meaning "no override", not "SIM1".
      cfg%vertical_solver = 0

      ! Riemann sub-splitting. fv_dynamics.F90:616 recomputes flagstruct%m_split
      ! from bdt/k_split/n_split immediately before the n_map loop, i.e. AFTER this
      ! routine runs at initialization, so the value here is still fv_arrays'
      ! zero and the runtime's m_split >= 1 fence would reject it. Clamping is
      ! safe ONLY because a_imp = 1.0 selects SIM1_solver, which takes neither ms
      ! nor m_split (nh_core.F90:160-176, and sim1_column likewise on the CUDA
      ! side). The a_imp guard below makes that precondition loud instead of
      ! assumed: any a_imp <= 0.999 configuration must install the config after
      ! fv_dynamics.F90:616 has run, not here.
      cfg%m_split = int(max(1, Atm%flagstruct%m_split), c_int32_t)

      ! ---- flags --------------------------------------------------------------
      ! USE_COND and MOIST_CAPPA are ON in this Fortran build
      ! (@fvdycore/CMakeLists.txt) but the CUDA runtime's production fence rejects
      ! them, so 0 is what must be sent -- and that is a REAL, recorded coverage
      ! gap, not a translation. It is round-off-neutral for this fixture only
      ! because q_con is identically zero on all six tiles; it must be closed
      ! before any moist configuration.
      cfg%use_cond    = 0
      cfg%moist_cappa = 0
      ! Compile-time in the Fortran, and absent from this build's flags.
      cfg%rot3   = 0
      cfg%use_sg = 0

      cfg%use_logp     = merge(1_c_int8_t, 0_c_int8_t, Atm%flagstruct%use_logp)
      cfg%inline_q     = merge(1_c_int8_t, 0_c_int8_t, Atm%flagstruct%inline_q)
      cfg%hydrostatic  = merge(1_c_int8_t, 0_c_int8_t, Atm%flagstruct%hydrostatic)
      cfg%do_f3d       = merge(1_c_int8_t, 0_c_int8_t, Atm%flagstruct%do_f3d)
      cfg%do_skeb      = merge(1_c_int8_t, 0_c_int8_t, Atm%flagstruct%do_skeb)
      ! dyn_core.F90:1128's gate verbatim, not just the flag.
      cfg%rayleigh_enabled = merge(1_c_int8_t, 0_c_int8_t, &
           Atm%flagstruct%RF_fast .and. Atm%flagstruct%tau > 0.0)
      cfg%grid_type    = int(Atm%flagstruct%grid_type, c_int8_t)
      cfg%reserved0    = 0

      ! Both launch-fusion optimizations default OFF -- the default path is
      ! the one validated bitwise (Prompts 9/11). Either can be opted into at
      ! run time via environment variable, with no rebuild and no change to
      ! the numerical kernels themselves; this only selects which
      ! already-validated launch pattern the runtime uses (Prompt 28).
      cfg%fuse_corner_fill = merge(1_c_int8_t, 0_c_int8_t, &
           env_flag_enabled('RUN_CUDA_FV3_FUSE_CORNER_FILL'))
      cfg%fuse_halo_edges  = merge(1_c_int8_t, 0_c_int8_t, &
           env_flag_enabled('RUN_CUDA_FV3_FUSE_HALO_EDGES'))

      if (Atm%flagstruct%a_imp <= 0.999) then
         call mpp_error(FATAL, 'fv3_gpu_fill_numerical_config: the CUDA dyn_core '// &
              'backend is validated only for a_imp > 0.999 (SIM1_solver). At or '// &
              'below that threshold m_split and scale_m become live and this '// &
              'routine''s initialization-time m_split is not yet valid.')
      endif

   end subroutine fv3_gpu_fill_numerical_config

   ! Prompt 28: minimal opt-in override for the two already-validated launch-
   ! fusion modes (RUN_CUDA_FV3_FUSE_HALO_EDGES / RUN_CUDA_FV3_FUSE_CORNER_FILL,
   ! YES/1/TRUE, case-insensitive). Unset or any other value keeps the default
   ! OFF behavior identical to every prior run.
   function env_flag_enabled(name) result(enabled)
      character(len=*), intent(in) :: name
      logical :: enabled
      character(len=16) :: val
      integer :: vlen, istat
      call get_environment_variable(trim(name), value=val, length=vlen, status=istat)
      enabled = .false.
      if (istat == 0 .and. vlen > 0) then
         select case (trim(adjustl(val)))
         case ('YES','yes','Yes','1','TRUE','true','True')
            enabled = .true.
         end select
      endif
   end function env_flag_enabled

   ! Prompt 43: MPI halo transport selector. RUN_CUDA_FV3_MPI_TRANSPORT=cuda_aware
   ! (case-insensitive) opts into the device-pointer transport; unset or any other
   ! value (including "host_staged") keeps the default host-staged behavior
   ! identical to every prior run. A string selector, not env_flag_enabled's
   ! boolean, because there are two named backends to choose between, not an
   ! on/off switch.
   function mpi_transport_kind_from_env() result(kind)
      integer(c_int32_t) :: kind
      character(len=16) :: val
      integer :: vlen, istat
      call get_environment_variable('RUN_CUDA_FV3_MPI_TRANSPORT', value=val, &
           length=vlen, status=istat)
      kind = FV3_GPU_HALO_TRANSPORT_MPI_HOST_STAGED
      if (istat == 0 .and. vlen > 0) then
         select case (trim(adjustl(val)))
         case ('cuda_aware','CUDA_AWARE','Cuda_Aware')
            kind = FV3_GPU_HALO_TRANSPORT_MPI_CUDA_AWARE
         end select
      endif
   end function mpi_transport_kind_from_env

   subroutine check(crc, err, api)
      integer,             intent(in) :: crc
      type(fv3_gpu_error), intent(in) :: err
      character(len=*),    intent(in) :: api
      if (crc == FV3_GPU_SUCCESS) return
      call mpp_error(FATAL, 'CUDA dyn_core backend: '//trim(api)//' failed: '// &
           trim(fv3_gpu_error_text(err)))
   end subroutine check

   subroutine mutable_state_mask(mask)
      type(fv3_gpu_field_mask), intent(out) :: mask
      integer :: id
      mask%abi_version  = FV3_GPU_ABI_VERSION
      mask%struct_bytes = int(storage_size(mask)/8, c_int32_t)
      mask%words = 0_c_int64_t
      do id = mask_id_first, mask_id_last
         mask%words(id/64 + 1) = ibset(mask%words(id/64 + 1), mod(id, 64))
      end do
   end subroutine mutable_state_mask

   ! Narrow mask for fv3_gpu_interface_pt_delz_remap -- exactly the 9 fields
   ! launch_pt_delz_remap_ctx reads/writes (fv3_pt_delz_remap.cu), not the
   ! full mutable_state_mask range above.
   subroutine pt_delz_remap_state_mask(mask)
      type(fv3_gpu_field_mask), intent(out) :: mask
      integer, parameter :: ids(9) = &
           (/ FV3_FID_PE, FV3_FID_PELN, FV3_FID_PT, FV3_FID_DELZ, FV3_FID_DELP, &
              FV3_FID_PK, FV3_FID_PKZ, FV3_FID_AK, FV3_FID_BK /)
      integer :: k
      mask%abi_version  = FV3_GPU_ABI_VERSION
      mask%struct_bytes = int(storage_size(mask)/8, c_int32_t)
      mask%words = 0_c_int64_t
      do k = 1, size(ids)
         mask%words(ids(k)/64 + 1) = ibset(mask%words(ids(k)/64 + 1), mod(ids(k), 64))
      end do
   end subroutine pt_delz_remap_state_mask

   ! Prompt 65: D2H-only companion mask for the CUDA-delz/Fortran-pt hybrid
   ! mode -- identical to pt_delz_remap_state_mask above MINUS FV3_FID_PT.
   ! The kernel still needs a valid on-device pt to read as H2D input (it
   ! internally computes a pt pretransform/remap it never returns to the
   ! host in this mode), but that device-side pt result must never be
   ! copied back: fv_mapz.F90's own Fortran map_scalar call for pt runs
   ! immediately after this dispatch and requires host `pt` to still hold
   ! the pre-remap (pretransformed) value it wrote before the dispatch, not
   ! whatever the kernel's own (thrown-away) remap produced. Omitting PT
   ! from the D2H mask is the correctness mechanism, not a convenience --
   ! see fv_mapz.F90's cuda_owns_pt comments.
   subroutine pt_delz_remap_state_mask_delz_only_d2h(mask)
      type(fv3_gpu_field_mask), intent(out) :: mask
      integer, parameter :: ids(8) = &
           (/ FV3_FID_PE, FV3_FID_PELN, FV3_FID_DELZ, FV3_FID_DELP, &
              FV3_FID_PK, FV3_FID_PKZ, FV3_FID_AK, FV3_FID_BK /)
      integer :: k
      mask%abi_version  = FV3_GPU_ABI_VERSION
      mask%struct_bytes = int(storage_size(mask)/8, c_int32_t)
      mask%words = 0_c_int64_t
      do k = 1, size(ids)
         mask%words(ids(k)/64 + 1) = ibset(mask%words(ids(k)/64 + 1), mod(ids(k), 64))
      end do
   end subroutine pt_delz_remap_state_mask_delz_only_d2h

   ! Narrow combined H2D/D2H mask for fv3_gpu_interface_remap_uv -- exactly
   ! the 5 fields launch_remap_uv_ctx reads/writes (fv3_remap_uv.cu).
   subroutine remap_uv_state_mask(mask)
      type(fv3_gpu_field_mask), intent(out) :: mask
      integer, parameter :: ids(5) = &
           (/ FV3_FID_U, FV3_FID_V, FV3_FID_PE, FV3_FID_AK, FV3_FID_BK /)
      integer :: k
      mask%abi_version  = FV3_GPU_ABI_VERSION
      mask%struct_bytes = int(storage_size(mask)/8, c_int32_t)
      mask%words = 0_c_int64_t
      do k = 1, size(ids)
         mask%words(ids(k)/64 + 1) = ibset(mask%words(ids(k)/64 + 1), mod(ids(k), 64))
      end do
   end subroutine remap_uv_state_mask

   ! H2D-only mask for fv3_gpu_interface_remap_pressurevars -- the 5 fields
   ! launch_remap_pressurevars_ctx reads as input (fv3_remap_pressurevars.cu):
   ! ak/bk (level coefficients), pe (surface pressure at k=npz+1, plus
   ! boundary pe(1)/pe(npz+1) pass-through), pt/delz (pkz's inputs). None of
   ! these need their PRE-kernel host value copied back, so they are absent
   ! from the D2H mask below.
   subroutine pressurevars_state_mask_h2d(mask)
      type(fv3_gpu_field_mask), intent(out) :: mask
      integer, parameter :: ids(5) = &
           (/ FV3_FID_AK, FV3_FID_BK, FV3_FID_PE, FV3_FID_PT, FV3_FID_DELZ /)
      integer :: k
      mask%abi_version  = FV3_GPU_ABI_VERSION
      mask%struct_bytes = int(storage_size(mask)/8, c_int32_t)
      mask%words = 0_c_int64_t
      do k = 1, size(ids)
         mask%words(ids(k)/64 + 1) = ibset(mask%words(ids(k)/64 + 1), mod(ids(k), 64))
      end do
   end subroutine pressurevars_state_mask_h2d

   ! D2H-only mask for fv3_gpu_interface_remap_pressurevars -- the 5 fields
   ! launch_remap_pressurevars_ctx writes (fv3_remap_pressurevars.cu): pe/
   ! peln/pk (Pass 1's pe2/pn2/pk2 recompute) and delp/pkz (Pass 2). pt/
   ! delz/ak/bk are read-only in this kernel and are therefore absent here
   ! -- copying them back would move bytes the caller never changed.
   subroutine pressurevars_state_mask_d2h(mask)
      type(fv3_gpu_field_mask), intent(out) :: mask
      integer, parameter :: ids(5) = &
           (/ FV3_FID_PE, FV3_FID_PELN, FV3_FID_PK, FV3_FID_PKZ, FV3_FID_DELP /)
      integer :: k
      mask%abi_version  = FV3_GPU_ABI_VERSION
      mask%struct_bytes = int(storage_size(mask)/8, c_int32_t)
      mask%words = 0_c_int64_t
      do k = 1, size(ids)
         mask%words(ids(k)/64 + 1) = ibset(mask%words(ids(k)/64 + 1), mod(ids(k), 64))
      end do
   end subroutine pressurevars_state_mask_d2h

   ! u/v-only mask for the host halo round trip between
   ! fv3_gpu_interface_state_to_fv_phase_a (D2H, before the caller's
   ! UNCHANGED mpp_get_boundary call and edge-row scatter) and
   ! fv3_gpu_interface_state_to_fv_phase_c (H2D, immediately after) --
   ! see fv3_gpu_state_to_fv_phase_a's ABI header comment
   ! (fv3_gpu_api.h) for the full contract this mask serves.
   subroutine state_to_fv_uv_mask(mask)
      type(fv3_gpu_field_mask), intent(out) :: mask
      integer, parameter :: ids(2) = (/ FV3_FID_U, FV3_FID_V /)
      integer :: k
      mask%abi_version  = FV3_GPU_ABI_VERSION
      mask%struct_bytes = int(storage_size(mask)/8, c_int32_t)
      mask%words = 0_c_int64_t
      do k = 1, size(ids)
         mask%words(ids(k)/64 + 1) = ibset(mask%words(ids(k)/64 + 1), mod(ids(k), 64))
      end do
   end subroutine state_to_fv_uv_mask

   subroutine build_rank_of_tile(comm, comm_rank, tile, rank_of_tile, layout_x, layout_y)
      integer,            intent(in)  :: comm, comm_rank, tile
      integer(c_int32_t), intent(out) :: rank_of_tile(6)
      ! Prompt 114 Stage 7: reuses this routine's existing per-rank
      ! MPI_Allgather rather than adding a second init-only collective.
      ! Cross-checks that every rank in the (already-required) 6-rank
      ! communicator agrees on the FMS within-tile layout -- a disagreement
      ! here would mean ranks were launched against inconsistent Atm state,
      ! which is a genuine FATAL, independent of the comm_size==6 gate.
      integer(c_int32_t), intent(in)  :: layout_x, layout_y
      integer :: tile_of_rank(6), r, ierr
      integer :: layout_of_rank(2,6), my_layout(2)
      character(len=160) :: msg

      tile_of_rank = -1
      call MPI_Allgather(tile, 1, MPI_INTEGER, tile_of_rank, 1, MPI_INTEGER, comm, ierr)
      if (ierr /= MPI_SUCCESS) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: MPI_Allgather of the tile map failed')
      endif

      my_layout = (/ int(layout_x), int(layout_y) /)
      call MPI_Allgather(my_layout, 2, MPI_INTEGER, layout_of_rank, 2, MPI_INTEGER, comm, ierr)
      if (ierr /= MPI_SUCCESS) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: MPI_Allgather of the layout map failed')
      endif
      do r = 1, 6
         if (layout_of_rank(1,r) /= int(layout_x) .or. layout_of_rank(2,r) /= int(layout_y)) then
            write(msg,'(a,i0,a,i0,a,i0,a,i0,a,i0)') &
                 'fv3_gpu_bridge_init: rank ', r-1, ' reports layout ', &
                 layout_of_rank(1,r), 'x', layout_of_rank(2,r), &
                 ' which disagrees with rank ', comm_rank
            call mpp_error(FATAL, trim(msg))
         endif
      end do

      rank_of_tile = -1
      do r = 1, 6
         if (tile_of_rank(r) < 1 .or. tile_of_rank(r) > 6) then
            write(msg,'(a,i0,a,i0)') 'fv3_gpu_bridge_init: rank ', r-1, &
                 ' reported an out-of-range tile ', tile_of_rank(r)
            call mpp_error(FATAL, trim(msg))
         endif
         if (rank_of_tile(tile_of_rank(r)) /= -1) then
            write(msg,'(a,i0,a)') 'fv3_gpu_bridge_init: tile ', tile_of_rank(r), &
                 ' is claimed by more than one rank'
            call mpp_error(FATAL, trim(msg))
         endif
         rank_of_tile(tile_of_rank(r)) = int(r-1, c_int32_t)
      end do

      if (rank_of_tile(tile) /= int(comm_rank, c_int32_t)) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: the allgathered tile map '// &
              'disagrees with this rank''s own tile')
      endif
   end subroutine build_rank_of_tile

   ! Derive this rank's position among the ranks that share its physical node
   ! (memory domain), independent of its position in the global/model
   ! communicator. Used solely to pick a node-local GPU index -- it does not
   ! change tile ownership, halo topology, or any MPI communication path.
   ! On a single-node run local_rank == the rank's index in comm, so callers
   ! that previously used comm_rank directly for device selection see no
   ! change in behavior.
   subroutine fv3_gpu_node_local_rank(comm, local_rank, local_size, ierr)
      integer, intent(in)  :: comm
      integer, intent(out) :: local_rank, local_size, ierr
      integer :: local_comm, split_ierr, rank_ierr, size_ierr, free_ierr

      local_rank = -1
      local_size = -1
      ierr = 0

      call MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &
           local_comm, split_ierr)
      if (split_ierr /= MPI_SUCCESS) then
         ierr = split_ierr
         return
      endif

      call MPI_Comm_rank(local_comm, local_rank, rank_ierr)
      call MPI_Comm_size(local_comm, local_size, size_ierr)
      call MPI_Comm_free(local_comm, free_ierr)

      if (rank_ierr /= MPI_SUCCESS .or. size_ierr /= MPI_SUCCESS) then
         ierr = merge(rank_ierr, size_ierr, rank_ierr /= MPI_SUCCESS)
         return
      endif
      if (free_ierr /= MPI_SUCCESS) then
         ierr = free_ierr
         return
      endif
   end subroutine fv3_gpu_node_local_rank

end module fv3_gpu_dyncore_bridge
