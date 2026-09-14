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
   public :: fv3_gpu_interface_step_dyn
   public :: fv3_gpu_interface_pe_halo
   public :: fv3_gpu_bridge_final

   ! ---- persistent per-rank state ---------------------------------------------

   logical,     save :: bridge_ready = .false.
   type(c_ptr), save :: ctx          = C_NULL_PTR
   integer,     save :: bridge_comm  = MPI_COMM_NULL
   integer,     save :: bridge_tile  = 0
   integer(c_int64_t), save :: model_step = 0

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
      integer :: crc, ierr, comm_rank, comm_size, dev
      integer :: fail_fid, fail_aux
      character(len=256) :: msg

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
      ! Six ranks share whatever GPUs the node exposes; the Prompt 15 transport
      ! is a single-node design and its test carries the same assumption.
      dev = mod(comm_rank, int(caps%device_count))

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
      call build_rank_of_tile(comm, comm_rank, bridge_tile, rank_of_tile)

      crc = fv3_gpu_set_cubed_sphere_topology(ctx, int(bridge_tile, c_int32_t), &
                                              int(Atm%npx, c_int64_t),          &
                                              int(Atm%npy, c_int64_t),          &
                                              rank_of_tile, err)
      call check(crc, err, 'fv3_gpu_set_cubed_sphere_topology')

      ! ---- halo transport ----------------------------------------------------
      ! The communicator crosses as its Fortran INTEGER handle; the transport
      ! converts back with MPI_Comm_f2c. This is the Prompt 15 host-staged MPI
      ! backend, validated bitwise against the in-process peer transport.
      crc = fv3_gpu_attach_mpi_halo_transport(ctx, int(comm, c_int), &
                                              int(bridge_tile, c_int32_t), err)
      call check(crc, err, 'fv3_gpu_attach_mpi_halo_transport')

      if (fv3_gpu_halo_transport_kind_of(ctx) /= FV3_GPU_HALO_TRANSPORT_MPI_HOST_STAGED) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: the host-staged MPI halo '// &
              'transport did not attach')
      endif

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

   subroutine build_rank_of_tile(comm, comm_rank, tile, rank_of_tile)
      integer,            intent(in)  :: comm, comm_rank, tile
      integer(c_int32_t), intent(out) :: rank_of_tile(6)
      integer :: tile_of_rank(6), r, ierr
      character(len=160) :: msg

      tile_of_rank = -1
      call MPI_Allgather(tile, 1, MPI_INTEGER, tile_of_rank, 1, MPI_INTEGER, comm, ierr)
      if (ierr /= MPI_SUCCESS) then
         call mpp_error(FATAL, 'fv3_gpu_bridge_init: MPI_Allgather of the tile map failed')
      endif

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

end module fv3_gpu_dyncore_bridge
