module Held_Suarez_ForcingMod
!
! Held & Suarez (1994, Bull. Amer. Meteor. Soc., 75, 1825-1830) idealized
! dry dynamical-core forcing: Newtonian relaxation of temperature toward a
! prescribed equilibrium profile Teq(lat,sigma), plus low-level Rayleigh
! friction on the horizontal winds. Canonical published parameters
! (Table 1 of the paper), unmodified.
!
! Backend-neutral by construction: operates only on the common FV3 state
! (pt, u, v; pe/ps read-only for the sigma coordinate) held in FV_Atm(1),
! called from FV_StateMod's FV_Run strictly BEFORE the Fortran/CUDA
! dyn_core dispatch. Does not reference RUN_CUDA_FV3 or any backend flag.
!
! pt is physical temperature (K) at this interface (confirmed from
! fv_dynamics.F90's own dummy-argument declaration, not potential
! temperature), so no theta<->T conversion is required here.

   use fv_arrays_mod, only: fv_grid_bounds_type

   implicit none
   private

   public :: Held_Suarez_Forcing

contains

   subroutine Held_Suarez_Forcing(u, v, pt, pe, ps, agrid, npz, dt, kappa, bd)

      type(fv_grid_bounds_type), intent(in)    :: bd
      integer,                   intent(in)    :: npz
      real,                      intent(inout) :: u (bd%isd:bd%ied,   bd%jsd:bd%jed+1, npz)
      real,                      intent(inout) :: v (bd%isd:bd%ied+1, bd%jsd:bd%jed,   npz)
      real,                      intent(inout) :: pt(bd%isd:bd%ied,   bd%jsd:bd%jed,   npz)
      real,                      intent(in)    :: pe(bd%is-1:bd%ie+1, npz+1, bd%js-1:bd%je+1)
      real,                      intent(in)    :: ps(bd%isd:bd%ied,   bd%jsd:bd%jed)
      real,                      intent(in)    :: agrid(bd%isd:bd%ied, bd%jsd:bd%jed, 2)
      real,                      intent(in)    :: dt     ! forcing timestep (s); same dt as this dyn_core call
      real,                      intent(in)    :: kappa  ! R_dry/cp_dry, the SAME constant the active run uses

      ! Canonical Held & Suarez (1994) Table 1 parameters, unmodified.
      real, parameter :: HS_SIGMA_B   = 0.7d0
      real, parameter :: HS_KA        = 1.0d0/(40.0d0*86400.0d0)  ! 1/40 day^-1
      real, parameter :: HS_KS        = 1.0d0/(4.0d0*86400.0d0)   ! 1/4  day^-1
      real, parameter :: HS_KF        = 1.0d0/86400.0d0           ! 1    day^-1
      real, parameter :: HS_DELTA_TY  = 60.0d0                    ! K
      real, parameter :: HS_DELTA_THZ = 10.0d0                    ! K
      real, parameter :: HS_T_MIN     = 200.0d0                   ! K
      real, parameter :: HS_T0        = 315.0d0                   ! K

      integer :: i, j, k
      real :: pfull, sigma, lat, sinlat2, coslat2
      real :: teq, kt, kv, sigma_ratio

      do k = 1, npz
         do j = bd%js, bd%je
            do i = bd%is, bd%ie

               pfull = 0.5d0*(pe(i,k,j) + pe(i,k+1,j))
               sigma = pfull/ps(i,j)
               lat   = agrid(i,j,2)
               sinlat2 = sin(lat)*sin(lat)
               coslat2 = cos(lat)*cos(lat)

               teq = (HS_T0 - HS_DELTA_TY*sinlat2 - HS_DELTA_THZ*log(sigma)*coslat2) * sigma**kappa
               teq = max(HS_T_MIN, teq)

               sigma_ratio = max(0.0d0, (sigma - HS_SIGMA_B)/(1.0d0 - HS_SIGMA_B))
               kt = HS_KA + (HS_KS - HS_KA)*sigma_ratio*coslat2*coslat2
               kv = HS_KF*sigma_ratio

               ! Implicit-in-time update (standard treatment for this forcing;
               ! stable regardless of dt*kt, dt*kv magnitude).
               pt(i,j,k) = (pt(i,j,k) + dt*kt*teq)/(1.0d0 + dt*kt)
               u(i,j,k)  = u(i,j,k)/(1.0d0 + dt*kv)
               v(i,j,k)  = v(i,j,k)/(1.0d0 + dt*kv)

            end do
         end do
      end do

   end subroutine Held_Suarez_Forcing

end module Held_Suarez_ForcingMod
