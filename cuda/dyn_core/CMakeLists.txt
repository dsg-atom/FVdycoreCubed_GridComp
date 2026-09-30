cmake_minimum_required(VERSION 3.21)
project(FV3DynCoreGPU LANGUAGES C CXX)

option(FV3_DYN_CORE_ENABLE_CUDA "Build the CUDA-backed runtime" OFF)
# Milestone 3A.11: manually-run wall-clock timing tool, not a correctness
# test -- off by default so it never adds time to the default ctest run.
option(BUILD_FV3_GPU_PERF_BENCH "Build the 3A.11 small-scale perf benchmark" OFF)
# Milestone 3A.15: the real cross-process halo transport (one cubed-sphere tile per
# MPI rank), which the production GEOS/FV3 decomposition (FV_NX=1, FV_NY=6) requires
# and which the in-process HaloPeerGroup ("Model B") cannot serve. OFF by default so
# that every existing target -- the 22 ctests and the perf benchmark included --
# builds and runs exactly as before on a machine with no MPI: with this off, not one
# MPI symbol, header or compile definition enters any target.
option(FV3_DYN_CORE_ENABLE_MPI_HALO "Build the cross-process MPI halo transport" OFF)
set(FV3_DYN_CORE_PRECISION "R8" CACHE STRING "FV precision: R4, R4R8, or R8")
set_property(CACHE FV3_DYN_CORE_PRECISION PROPERTY STRINGS R4 R4R8 R8)

# Core runtime library (WP-E)
# Milestone 3A.10 adds runtime/fv3_gpu_halo_topology.cpp: the cubed-sphere
# mosaic connectivity builder (CubeHalo.F90:213-261 / fv_mp_mod.F90:577-625) and
# the panel-seam index/vector algebra. It is host-only and CUDA-free so the seam
# transforms are unit-testable without a GPU; the same inline functions are
# compiled into the device kernels from the header.
#
# Milestone 3A.11V Task 2 adds runtime/fv3_damping_translation.cpp: the single
# place where damping configuration scalars become the ten per-level
# device-resident damping arrays the storage contract already declares
# (dyn_core.F90:677-765). It lives in the core library so the benchmark, the
# unit tests and the eventual GEOS bridge all consume one translation instead
# of inventing their own tables. Host-only and CUDA-free, for the same reason
# fv3_gpu_halo_topology.cpp is.
add_library(fv3_dyn_core_gpu
  runtime/fv3_gpu_runtime.cpp
  runtime/fv3_gpu_contract.cpp
  runtime/fv3_gpu_halo_topology.cpp
  runtime/fv3_damping_translation.cpp
  # Registry-construction helpers for the Fortran field registry
  # (cuda/interface/fv3_gpu_field_registry.F90). NOT part of the public ABI:
  # they exist so the Fortran side can query the one field specification table
  # and reuse the one damping translation instead of transcribing either.
  runtime/fv3_gpu_registry_shim.cpp
)
target_include_directories(fv3_dyn_core_gpu PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(fv3_dyn_core_gpu PUBLIC cxx_std_17)

# WP-D operators library (CPU-visible formulas)
add_library(fv3_wp_d_operators STATIC operators/wp_d_operators.cpp operators/wp_d_contract.cpp)
target_include_directories(fv3_wp_d_operators PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/operators
)
target_link_libraries(fv3_wp_d_operators PUBLIC fv3_dyn_core_gpu)
target_compile_features(fv3_wp_d_operators PUBLIC cxx_std_17)

# WP-B D-grid library (CPU-visible formulas)
add_library(fv3_wp_b_dgrid STATIC dgrid/wp_b_contract.cpp)
target_include_directories(fv3_wp_b_dgrid PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/dgrid
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(fv3_wp_b_dgrid PUBLIC fv3_dyn_core_gpu)
target_compile_features(fv3_wp_b_dgrid PUBLIC cxx_std_17)
if(FV3_DYN_CORE_PRECISION STREQUAL "R4" OR FV3_DYN_CORE_PRECISION STREQUAL "R4R8")
  target_compile_definitions(fv3_dyn_core_gpu PUBLIC FV3_GPU_FV_REAL_IS_FLOAT=1)
elseif(FV3_DYN_CORE_PRECISION STREQUAL "R8")
  target_compile_definitions(fv3_dyn_core_gpu PUBLIC FV3_GPU_FV_REAL_IS_DOUBLE=1)
else()
  message(FATAL_ERROR "FV3_DYN_CORE_PRECISION must be R4, R4R8, or R8")
endif()

if(FV3_DYN_CORE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(fv3_dyn_core_gpu PRIVATE FV3_GPU_ENABLE_CUDA=1)
  target_link_libraries(fv3_dyn_core_gpu PRIVATE CUDA::cudart)

  # WP-A C-grid CUDA sources
  add_library(fv3_wp_a_cgrid STATIC
    cgrid/fv3_cgrid.cu
    cgrid/fv3_cgrid_d2a2c.cu
    cgrid/fv3_cgrid_csw.cu
    cgrid/fv3_cgrid_divergence.cu
    cgrid/fv3_cgrid_dz.cu
    cgrid/fv3_cgrid_predictor.cu
    # Milestone 3A.9: sw_core.F90's fill_4corners / fill2_4corners, the A-grid
    # scalar corner halo fills c_sw and update_dz_c need between their x and y
    # flux phases (sw_core.F90:206/237/257/284, nh_utils.F90:162/174).
    cgrid/fv3_cgrid_corners.cu
  )
  target_include_directories(fv3_wp_a_cgrid PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cgrid
  )
  target_link_libraries(fv3_wp_a_cgrid PUBLIC fv3_dyn_core_gpu CUDA::cudart)
  set_target_properties(fv3_wp_a_cgrid PROPERTIES CUDA_STANDARD 17)

  # WP-C vertical CUDA sources
  # Milestone 3A.9 adds vertical/fv3_riem_solver_c.cu: nh_utils.F90's
  # Riem_Solver_c, the C-grid Riemann solve (dyn_core.F90:583, dt2). It is a
  # genuinely distinct routine from Riem_Solver3, not an alias, but it reuses the
  # shared sim1_column device solver rather than forking it.
  # Prompt 52C adds vertical/fv3_pt_delz_remap.cu: fv_mapz_mod's production
  # pt/delz vertical-remap entry (Lagrangian_to_Eulerian, remap_option=0,
  # gmao_remap=0, kord=9, nwat=0). Reuses map_scalar_column (Prompt 52B) and
  # the RiemCViewC3/RiemViewC3 view adapters (fv3_riem_solver_c.cuh) rather
  # than duplicating either.
  # Prompt 71 adds vertical/fv3_remap_uv.cu: fv_mapz_mod's production u/v
  # vertical-remap entry (Lagrangian_to_Eulerian, timer 'Remap_UV', iv=-1,
  # kord=kord_mt=9). Reuses map_scalar_column (Prompt 52B, extended with
  # iv=-1 branches this prompt) and the RiemViewC3/PtDelzView1 view adapters
  # rather than duplicating either.
  # Prompt 85 adds vertical/fv3_remap_q.cu: fv_mapz_mod's production tracer
  # vertical-remap entry (mapn_tracer, nq=1, timer 'Remap_Q', iv=0,
  # kord=kord_tr=8). Reuses map_scalar_column (Prompt 85, extended with the
  # missing abs(kord)<9 branch) and the RiemViewC3/PtDelzView1 view adapters;
  # q itself has no FieldRegistry identity (unlike pt_delz_remap/remap_uv's
  # fields) so it is threaded through via a dedicated device buffer, not a
  # field-registry view -- see fv3_remap_q.cuh's header comment.
  # Prompt 92 adds vertical/fv3_remap_pressurevars.cu: fv_mapz_mod's
  # production pressure-variable vertical-remap entry (Lagrangian_to_
  # Eulerian, timer 'Remap_PressureVars'). A direct closed-form recompute
  # (pe2/pn2/pk2 from ak/bk/pe(surface), then delp/cappa/pkz) -- does NOT
  # call map_scalar_column/scalar_profile_column, unlike pt_delz_remap/
  # remap_uv/remap_q above. q(sphum)/cappa have no FieldRegistry identity
  # (like remap_q's q) so they are threaded through via dedicated device
  # buffers -- see fv3_remap_pressurevars.cuh's header comment.
  # Prompt 99 adds vertical/fv3_state_to_fv.cu: FV_StateMod.F90's production
  # State_To_FV Phase A (u/v copy-in) and Phase C (w/pe/peln/pk/delp/pt/
  # delz/pkz local compute) halves; Phase B (mpp_get_boundary) stays host-
  # only per this file's own header comment. STATE%VARS%* has no
  # FieldRegistry identity (like remap_q's q / remap_pressurevars's
  # q(sphum)/cappa) so it is threaded through via dedicated device buffers
  # -- see fv3_state_to_fv.cuh's header comment.
  add_library(fv3_wp_c_vertical STATIC
    vertical/fv3_vertical_kernels.cu
    vertical/fv3_riem_solver_c.cu
    vertical/fv3_pt_delz_remap.cu
    vertical/fv3_remap_uv.cu
    vertical/fv3_remap_q.cu
    vertical/fv3_remap_pressurevars.cu
    vertical/fv3_state_to_fv.cu
  )
  target_include_directories(fv3_wp_c_vertical PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical
    # Milestone 3A.9 §4: WP-C owns the C-grid vertical ENTRY sequencing
    # (dyn_core.F90:543-583), but the update_dz_c kernels and the A-grid corner
    # fills that sequence drives are WP-A property -- project_architecture.md
    # gives WP-A "C-grid dz updates". WP-C therefore CALLS
    # cgrid::launch_update_dz_c_interface / launch_fill_4corners instead of
    # transcribing a second copy, exactly as WP-B's smag_corner calls WP-D's
    # a2b_ord4 (see fv3_wp_b_cuda below). No cycle: fv3_wp_a_cgrid depends only
    # on the WP-E runtime, never on fv3_wp_c_vertical.
    ${CMAKE_CURRENT_SOURCE_DIR}/cgrid
  )
  target_link_libraries(fv3_wp_c_vertical PUBLIC
    fv3_dyn_core_gpu
    fv3_wp_a_cgrid
    CUDA::cudart)
  set_target_properties(fv3_wp_c_vertical PROPERTIES CUDA_STANDARD 17)

  # Prompt 64 Stage 7: disable FMA (fused-multiply-add) contraction for
  # fv3_pt_delz_remap.cu ONLY (not fv3_riem_solver_c.cu / fv3_vertical_
  # kernels.cu, which also live in this library but are untouched -- no
  # Riemann-solver arithmetic changes). Root cause (Stages 0-6): nvcc's
  # default FMA contraction in map_scalar_column/scalar_profile_column's
  # (and the closing pe/peln/pk/delp update's) a*b+c expressions rounds
  # differently than Fortran/ifort's -no-fma map_scalar. Diagnostic twin
  # build (whole-library -fmad=false, fv3_wp_c_pt_delz_remap_gpu_smoke vs.
  # a temporary nofma variant) proved this exactly: pt/delz/delp/dpeO went
  # from round-off (~1e-15/1e-16 rel) to bitwise (0.0) across all 6
  # fixtures, including case6_realistic_c512, with zero algorithm change.
  # pkz keeps a separate, much smaller (~1e-16, 1-2 ULP) residual in 2/6
  # cases -- a distinct, unrelated intrinsic-evaluation difference, not
  # FMA contraction (this flag does not and should not eliminate it).
  set_source_files_properties(vertical/fv3_pt_delz_remap.cu PROPERTIES
    COMPILE_OPTIONS "-fmad=false")

  # Prompt 71: fv3_remap_uv.cu calls the IDENTICAL map_scalar_column/
  # scalar_profile_column arithmetic path as fv3_pt_delz_remap.cu (same
  # a*b+c expressions the Prompt-64 diagnostic above traced the round-off
  # to), so the same -fmad=false scoping is applied here proactively ahead
  # of Stage 8's bitwise C32 A/B, rather than re-deriving the identical
  # root cause a second time. Not yet independently re-confirmed with its
  # own twin-build diagnostic against REMAP_UV's own fixtures -- if Stage 8
  # still shows non-round-off residuals, revisit this assumption rather
  # than assuming the flag alone suffices.
  set_source_files_properties(vertical/fv3_remap_uv.cu PROPERTIES
    COMPILE_OPTIONS "-fmad=false")

  # Prompt 85: fv3_remap_q.cu calls the IDENTICAL map_scalar_column/
  # scalar_profile_column arithmetic path as fv3_pt_delz_remap.cu/
  # fv3_remap_uv.cu (same a*b+c expressions the Prompt-64 diagnostic traced
  # the round-off to), so the same -fmad=false scoping is applied here
  # proactively ahead of Stage 7's bitwise fixture comparison, rather than
  # re-deriving the identical root cause a third time.
  set_source_files_properties(vertical/fv3_remap_q.cu PROPERTIES
    COMPILE_OPTIONS "-fmad=false")

  # Prompt 92: fv3_remap_pressurevars.cu does not call map_scalar_column,
  # but it has the same a*b+c FMA-contraction-prone shape the Prompt-64
  # diagnostic traced the round-off to (pe2 = ak + bk*pe_old_km; cappa's
  # rdgas/(rdgas + cv_air/(1+r_vir*q)); pkz's exp(cappa*log(...))), so the
  # same -fmad=false scoping is applied here proactively ahead of Stage 8's
  # bitwise/round-off fixture classification, rather than re-deriving the
  # identical root cause a fourth time.
  set_source_files_properties(vertical/fv3_remap_pressurevars.cu PROPERTIES
    COMPILE_OPTIONS "-fmad=false")

  # Prompt 99: fv3_state_to_fv.cu's Phase C peln=log(pe)/pk=exp(akap*peln)
  # pair has the same FMA-contraction-prone shape (exp(cappa*log(...))) as
  # fv3_remap_pressurevars.cu's pkz, so the same -fmad=false scoping is
  # applied here proactively ahead of Stage 9's fixture classification,
  # rather than re-deriving the identical root cause a fifth time.
  set_source_files_properties(vertical/fv3_state_to_fv.cu PROPERTIES
    COMPILE_OPTIONS "-fmad=false")

  # WP-D operators CUDA sources
  add_library(fv3_wp_d_cuda STATIC operators/wp_d_cuda.cu operators/wp_d_contract_cuda.cpp)
  target_include_directories(fv3_wp_d_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/operators
  )
  target_link_libraries(fv3_wp_d_cuda PUBLIC fv3_wp_d_operators CUDA::cudart)
  set_target_properties(fv3_wp_d_cuda PROPERTIES CUDA_STANDARD 17)

  # Milestone 3A.3/3A.5: WP-B D-grid height + scalar transport CUDA sources.
  # Milestone 3A.6 adds dgrid/wp_b_momentum_cuda.cu (WP-B.3 momentum
  # transport); Milestone 3A.7 adds dgrid/wp_b_divergence_cuda.cu (WP-B.4
  # divergence damping / smag_corner / d_con heating); Milestone 3A.8 adds
  # dgrid/wp_b_dsw_cuda.cu (WP-B.5, the complete d_sw driver that orchestrates
  # all of the above). All four .cu files share dgrid/wp_b_ppm_shared.cuh so the
  # PPM constants, seam predicates and launch helpers have one definition.
  add_library(fv3_wp_b_cuda STATIC
    dgrid/wp_b_cuda.cu
    dgrid/wp_b_momentum_cuda.cu
    dgrid/wp_b_divergence_cuda.cu
    dgrid/wp_b_dsw_cuda.cu
    dgrid/wp_b_contract_cuda.cpp
    dgrid/wp_b_dsw_contract_cuda.cpp
    dgrid/fv3_tracer_transport_bridge.cu)
  target_include_directories(fv3_wp_b_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/dgrid
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  # fv3_wp_d_cuda: Milestone 3A.7's smag_corner REUSES WP-D's
  # operators::launch_a2b_ord4_global_cuda rather than transcribing a second
  # a2b_ord4 (project no-duplication rule). This is the only cross-work-package
  # call in WP-B and introduces no cycle -- fv3_wp_d_cuda does not depend on
  # fv3_wp_b_cuda.
  target_link_libraries(fv3_wp_b_cuda PUBLIC fv3_wp_b_dgrid fv3_wp_d_cuda CUDA::cudart)
  set_target_properties(fv3_wp_b_cuda PROPERTIES CUDA_STANDARD 17)

  # Milestone 3A.10 (WP-E): the cubed-sphere halo exchange kernels -- device
  # pack, device-to-device transfer, and the seam-transforming unpack.
  #
  # WP-E, not WP-A/B/C/D: project_architecture.md assigns "halo/communication
  # interfaces" to WP-E, and this is shared infrastructure with no numerical
  # algorithm of its own. It moves values and applies the panel-seam permutation
  # and sign; it computes nothing.
  # These kernels belong to fv3_dyn_core_gpu itself rather than to a separate
  # library, because the runtime's resolve_halo() calls them: a standalone library
  # linking PUBLIC against fv3_dyn_core_gpu while fv3_dyn_core_gpu called into it
  # would be a dependency cycle. Adding the source here also gives it
  # FV3_GPU_ENABLE_CUDA, which is PRIVATE on this target -- the first attempt used
  # a separate library WITHOUT that define, so every definition was preprocessed
  # away and the archive linked empty, surfacing only as an undefined reference at
  # the first consumer rather than as a compile error.
  target_sources(fv3_dyn_core_gpu PRIVATE runtime/fv3_gpu_halo_exchange.cu)
  # Milestone 3A.11T: the huge_r constant-fill used to restore dyn_core.F90's
  # init_step poison of gz/pk3 (see poison_init_step_transients). Same target and
  # same reasoning as the halo kernels above -- the runtime itself calls it, and
  # it needs FV3_GPU_ENABLE_CUDA, which is PRIVATE here. A .cu because
  # cudaMemsetAsync fills bytes and so cannot write huge_r; this needs a kernel.
  target_sources(fv3_dyn_core_gpu PRIVATE runtime/fv3_gpu_state_poison.cu)
  # Prompt 6B: the per-substep heat_source accumulation of dyn_core.F90:853-861,
  # which the port was missing. Same target and same reasoning as the two sources
  # above -- fv3_gpu_dyn_core_step calls it directly, and it needs
  # FV3_GPU_ENABLE_CUDA, which is PRIVATE here.
  target_sources(fv3_dyn_core_gpu PRIVATE runtime/fv3_gpu_heat_accumulator.cu)
  # Milestone 3A.15 §4-§7: the host-staged MPI halo transport. Added to THIS target
  # rather than to a library of its own for the same two reasons the halo kernels
  # above are: the runtime calls it (a separate library linking PUBLIC against
  # fv3_dyn_core_gpu while fv3_dyn_core_gpu called into it would be a dependency
  # cycle), and it needs FV3_GPU_ENABLE_CUDA, which is PRIVATE here -- the
  # separate-library attempt recorded above preprocessed every definition away and
  # linked an empty archive.
  #
  # Guarded so the DEFAULT build is byte-for-byte unaffected: with
  # FV3_DYN_CORE_ENABLE_MPI_HALO off, this source is not compiled, no MPI package is
  # searched for, and fv3_gpu_attach_mpi_halo_transport still EXISTS but refuses with
  # FV3_GPU_UNSUPPORTED_CONFIGURATION naming this option -- a diagnosable refusal
  # instead of an undefined reference.
  #
  # MPI is linked PRIVATE because the transport is entirely internal: no MPI type
  # crosses this project's public ABI (the communicator arrives as its Fortran
  # INTEGER handle -- see include/fv3_gpu_communication.h). A static library's
  # PRIVATE link dependencies still reach a consumer's link line as LINK_ONLY, so
  # the six-rank test links without naming MPI twice.
  if(FV3_DYN_CORE_ENABLE_MPI_HALO)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_sources(fv3_dyn_core_gpu PRIVATE runtime/fv3_gpu_halo_mpi.cpp)
    target_compile_definitions(fv3_dyn_core_gpu PRIVATE FV3_DYN_CORE_ENABLE_MPI_HALO=1)
    target_link_libraries(fv3_dyn_core_gpu PRIVATE MPI::MPI_CXX)
  endif()
  set_target_properties(fv3_dyn_core_gpu PROPERTIES CUDA_STANDARD 17)

  # Milestone 3A.1: production NumericalDispatch (wires WP-A/WP-C/WP-D CUDA
  # adapters into a real dispatch table consumed by fv3_gpu_dyn_core_step).
  # .cu (not .cpp): it transitively includes cgrid/fv3_cgrid_d2a2c.hpp, which
  # declares __device__ functions that only nvcc can parse.
  add_library(fv3_production_dispatch STATIC runtime/fv3_gpu_production_dispatch.cu)
  target_include_directories(fv3_production_dispatch PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(fv3_production_dispatch PUBLIC
    fv3_dyn_core_gpu
    fv3_wp_a_cgrid
    fv3_wp_c_vertical
    fv3_wp_d_cuda
    fv3_wp_b_cuda
  )
  target_compile_features(fv3_production_dispatch PUBLIC cxx_std_17)
  set_target_properties(fv3_production_dispatch PROPERTIES CUDA_STANDARD 17)
endif()

include(CTest)
if(BUILD_TESTING)
  # WP-E runtime tests
  add_executable(fv3_gpu_descriptor_test tests/runtime/test_descriptor.cpp)
  target_link_libraries(fv3_gpu_descriptor_test PRIVATE fv3_dyn_core_gpu)
  add_test(NAME fv3_gpu_descriptor COMMAND fv3_gpu_descriptor_test)

  add_executable(fv3_gpu_api_test tests/runtime/test_api.cpp)
  target_link_libraries(fv3_gpu_api_test PRIVATE fv3_dyn_core_gpu)
  add_test(NAME fv3_gpu_api COMMAND fv3_gpu_api_test)

  add_executable(fv3_gpu_c_header_test tests/runtime/test_c_header.c)
  target_link_libraries(fv3_gpu_c_header_test PRIVATE fv3_dyn_core_gpu)
  set_target_properties(fv3_gpu_c_header_test PROPERTIES LINKER_LANGUAGE CXX)
  add_test(NAME fv3_gpu_c_header COMMAND fv3_gpu_c_header_test)

  add_executable(fv3_gpu_contract_test tests/runtime/test_contract.cpp)
  target_link_libraries(fv3_gpu_contract_test PRIVATE fv3_dyn_core_gpu)
  add_test(NAME fv3_gpu_contract COMMAND fv3_gpu_contract_test)

  # WP-A C-grid tests (CPU-visible indexing)
  add_executable(fv3_wp_a_view_test tests/cgrid/test_cgrid_view.cpp)
  target_include_directories(fv3_wp_a_view_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cgrid)
  target_link_libraries(fv3_wp_a_view_test PRIVATE fv3_dyn_core_gpu)
  add_test(NAME fv3_wp_a_view COMMAND fv3_wp_a_view_test)

  # WP-B D-grid tests (CPU-visible kernels)
  add_executable(fv3_wp_b_kernels_test tests/dgrid/test_wp_b_kernels.cpp)
  target_include_directories(fv3_wp_b_kernels_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(fv3_wp_b_kernels_test PRIVATE fv3_dyn_core_gpu)
  add_test(NAME fv3_wp_b_kernels COMMAND fv3_wp_b_kernels_test)

  add_executable(fv3_wp_b_contract_test tests/dgrid/test_wp_b_contract.cpp)
  target_include_directories(fv3_wp_b_contract_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(fv3_wp_b_contract_test PRIVATE fv3_wp_b_dgrid)
  add_test(NAME fv3_wp_b_contract COMMAND fv3_wp_b_contract_test)

  # WP-C vertical tests (CPU-visible column primitives)
  add_executable(fv3_wp_c_column_test tests/vertical/test_vertical_column.cpp)
  target_include_directories(fv3_wp_c_column_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vertical)
  target_link_libraries(fv3_wp_c_column_test PRIVATE fv3_dyn_core_gpu)
  add_test(NAME fv3_wp_c_column COMMAND fv3_wp_c_column_test)

  # WP-C fv_mapz vertical remap tests (Prompt 52B, CPU-visible host path of
  # the __host__ __device__ map_scalar_column/scalar_profile_column header).
  # Fixture dir is the repo-root Prompt 52A baseline snapshot, not under
  # cuda/dyn_core/ -- pass its absolute path in at configure time since no
  # existing SOURCE_DIR-style macro convention exists in this file to reuse.
  add_executable(fv3_wp_c_map_scalar_test tests/vertical/test_map_scalar.cpp)
  target_include_directories(fv3_wp_c_map_scalar_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vertical)
  target_link_libraries(fv3_wp_c_map_scalar_test PRIVATE fv3_dyn_core_gpu)
  target_compile_definitions(fv3_wp_c_map_scalar_test PRIVATE
    FV3_MAP_SCALAR_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fortran_baseline/fv_mapz/fixtures")
  add_test(NAME fv3_wp_c_map_scalar COMMAND fv3_wp_c_map_scalar_test)

  # WP-C fv_mapz REMAP_Q (mapn_tracer, nq=1) host-path regression test
  # (Prompt 85, Stages 1/7): reuses map_scalar_column directly with
  # iv=0/kord=8/gmao_bot=gmao_top=false against the Prompt 84B tracer
  # fixtures (case14-22; case23 is nq=3, out of scope). Same fixture dir as
  # fv3_wp_c_map_scalar_test above.
  add_executable(fv3_wp_c_remap_q_tracer_test tests/vertical/test_remap_q_tracer.cpp)
  target_include_directories(fv3_wp_c_remap_q_tracer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vertical)
  target_link_libraries(fv3_wp_c_remap_q_tracer_test PRIVATE fv3_dyn_core_gpu)
  target_compile_definitions(fv3_wp_c_remap_q_tracer_test PRIVATE
    FV3_MAP_SCALAR_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fortran_baseline/fv_mapz/fixtures")
  add_test(NAME fv3_wp_c_remap_q_tracer COMMAND fv3_wp_c_remap_q_tracer_test)

  # WP-C fv_mapz REMAP_Q nq=2 scope-proof test (Prompt 86A, Stage 8):
  # host-only (no CUDA), synthetic production-shaped sphum-like (kord=8) +
  # cld_amt-like (kord=9) profiles on a shared pe1/pe2 grid, checking mass
  # conservation and positivity via the existing map_scalar_column path.
  # Documentation-only investigation task -- see
  # docs/performance/p86a_remap_q_nq2_scope_2026-09-25.md.
  add_executable(fv3_wp_c_remap_q_nq2_scope_test tests/vertical/test_remap_q_nq2_scope.cpp)
  target_include_directories(fv3_wp_c_remap_q_nq2_scope_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vertical)
  target_link_libraries(fv3_wp_c_remap_q_nq2_scope_test PRIVATE fv3_dyn_core_gpu)
  add_test(NAME fv3_wp_c_remap_q_nq2_scope COMMAND fv3_wp_c_remap_q_nq2_scope_test)

  # WP-C fv_mapz REMAP_Q fill=true golden-reference test (Prompt 88A, Stage
  # 6): host-only (no CUDA, no project headers -- a standalone mechanical
  # port of the authoritative fv_fill.F90 fillz), covering fill-branch
  # activation, no-op-on-positive-q, near-minimum q, top/interior/bottom
  # borrow boundary behavior, the final non-local normalize-and-clip path
  # (both its success and its documented sum0<=0 inertness), and a batched
  # nq=2 call confirming fillz is applied identically/independently per
  # tracer. See docs/performance/p88a_remap_q_fill_semantics_2026-09-25.md.
  add_executable(fv3_wp_c_fillz_host_reference_test tests/vertical/test_fillz_host_reference.cpp)
  add_test(NAME fv3_wp_c_fillz_host_reference COMMAND fv3_wp_c_fillz_host_reference_test)

  # WP-C REMAP_Q fill=true CUDA device-helper vs. host-oracle bitwise
  # comparison (Prompt 88B, Stage 5): host-compiled instantiation of the new
  # __host__ __device__ fillz_column<T> (fv3_fillz_column.hpp) against
  # fillz_host()'s Prompt 88A oracle, same 9 fixtures (case 8 split into two
  # single-tracer sub-cases to match fillz_column<T>'s single-tracer
  # signature). No project headers/CUDA runtime required -- host-only.
  add_executable(fv3_wp_c_fillz_cuda_vs_host_test tests/vertical/test_fillz_cuda_vs_host.cpp)
  add_test(NAME fv3_wp_c_fillz_cuda_vs_host COMMAND fv3_wp_c_fillz_cuda_vs_host_test)

  # WP-C REMAP_PRESSUREVARS host-only golden-reference harness (Prompt 91,
  # Stage 10/13): mechanical transcription of fv_mapz.F90:1239-1516's
  # production-active branches only (hydrostatic=.false., remap_t=.true.,
  # MOIST_CAPPA nwat=0 case-default, do_omega=.false.). Test-only
  # transcription (Option C) because Remap_PressureVars is inline code
  # inside Lagrangian_to_Eulerian, not a standalone callable routine. No
  # CUDA source added/modified; host-only, same pattern as
  # fv3_wp_c_fillz_host_reference_test above. See
  # docs/performance/p91_remap_pressurevars_cuda_port_plan_2026-09-27.md.
  add_executable(fv3_wp_c_remap_pressurevars_host_reference_test
    tests/vertical/test_remap_pressurevars_host_reference.cpp)
  add_test(NAME fv3_wp_c_remap_pressurevars_host_reference
    COMMAND fv3_wp_c_remap_pressurevars_host_reference_test)

  # WP-C REMAP_TOTALENERGYCON host-only golden-reference harness (Prompt 95,
  # Stage 11/13/15): mechanical transcription of fv_mapz.F90:1612-1864's
  # production-active branches only (hydrostatic=.false.,
  # consv_te>consv_min, nwat=0 MOIST_CAPPA case-default, adiabatic=.true.
  # post-reduction correction). Test-only transcription (Option C), same
  # convention as fv3_wp_c_remap_pressurevars_host_reference_test above --
  # Remap_TotalEnergyConsv is inline code inside Lagrangian_to_Eulerian, not
  # a standalone callable routine. No CUDA source added/modified; host-only.
  # Deliberately excludes the two mpp_global_sum reduction calls themselves
  # (measured to dominate the timer, see the doc below) -- covers only the
  # pre-reduction per-column compute (Unit A), the post-reduction global-
  # scalar bridge formula, and the post-reduction per-point pt correction
  # (Unit B). See
  # docs/performance/p95_remap_totalenergycon_cuda_port_plan_2026-09-28.md.
  add_executable(fv3_wp_c_remap_totalenergycon_host_reference_test
    tests/vertical/test_remap_totalenergycon_host_reference.cpp)
  add_test(NAME fv3_wp_c_remap_totalenergycon_host_reference
    COMMAND fv3_wp_c_remap_totalenergycon_host_reference_test)

  # WP-E/WP-C FV_To_State host-only golden-reference harness (Prompt 97,
  # Stage 11/13): mechanical transcription of FV_StateMod.F90:2704-2780's
  # production-active branches only (hydrostatic=.false., SW_DYNAMICS=
  # .false.), including the inlined fv_getPKZ(pkz,pe) dry-akap formula
  # (FV_StateMod.F90:3078-3108). Test-only transcription (Option C) because
  # a direct call (Option A) requires a live MAPL_MetaComp/T_FVDYCORE_STATE
  # environment; no MAPL infrastructure is tested here. No CUDA source
  # added/modified -- host-only, same pattern as
  # fv3_wp_c_remap_pressurevars_host_reference_test above. See
  # docs/performance/p97_fv_to_state_cuda_port_plan_2026-09-28.md.
  add_executable(fv3_wp_c_fv_to_state_host_reference_test
    tests/host/test_fv_to_state_host_reference.cpp)
  add_test(NAME fv3_wp_c_fv_to_state_host_reference
    COMMAND fv3_wp_c_fv_to_state_host_reference_test)

  # WP-E/WP-C State_To_FV host-only golden-reference harness (Prompt 98,
  # Stage 13/14): mechanical transcription of FV_StateMod.F90:2519-2702's
  # production-active branches only (grid_type=0 interior-only scope,
  # hydrostatic=.false., SW_DYNAMICS=.false.), including the derived
  # peln/pk/ps/delp fields. Test-only transcription (Option C), same
  # rationale as fv3_wp_c_fv_to_state_host_reference_test above -- a direct
  # call (Option A) requires a live MAPL_MetaComp/T_FVDYCORE_STATE
  # environment; no MAPL infrastructure is tested here. mpp_get_boundary
  # (Category D communication) is out of scope -- only the interior U/V
  # copy is transcribed. No CUDA source added/modified -- host-only. See
  # docs/performance/p98_state_to_fv_cuda_port_plan_2026-09-28.md.
  add_executable(fv3_wp_c_state_to_fv_host_reference_test
    tests/host/test_state_to_fv_host_reference.cpp)
  add_test(NAME fv3_wp_c_state_to_fv_host_reference
    COMMAND fv3_wp_c_state_to_fv_host_reference_test)

  # WP-C fv_mapz GPU smoke test (Prompt 52B, Stage 15): the same
  # map_scalar_column/scalar_profile_column device functions launched from a
  # real __global__ kernel on a real GPU. GPU-execution + bitwise sanity
  # check only, plus a rough non-authoritative CPU-vs-GPU timing sanity
  # check; not registered as a ctest (requires an actual GPU, unlike every
  # other test in this block) -- run manually via srun.
  add_executable(fv3_wp_c_map_scalar_gpu_smoke vertical/fv3_map_scalar_kernel.cu)
  target_include_directories(fv3_wp_c_map_scalar_gpu_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fv3_wp_c_map_scalar_gpu_smoke PRIVATE fv3_dyn_core_gpu)

  # Diagnostic-only twin build with FMA contraction disabled, to localize
  # whether GPU FMA reassociation explains any non-bitwise mismatch above
  # (Prompt 52B Stage 11: root-cause before relaxing any tolerance).
  add_executable(fv3_wp_c_map_scalar_gpu_smoke_nofma vertical/fv3_map_scalar_kernel.cu)
  target_include_directories(fv3_wp_c_map_scalar_gpu_smoke_nofma PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fv3_wp_c_map_scalar_gpu_smoke_nofma PRIVATE fv3_dyn_core_gpu)
  target_compile_definitions(fv3_wp_c_map_scalar_gpu_smoke_nofma PRIVATE
    FV3_MAP_SCALAR_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fortran_baseline/fv_mapz/fixtures")
  target_compile_options(fv3_wp_c_map_scalar_gpu_smoke_nofma PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
  target_compile_definitions(fv3_wp_c_map_scalar_gpu_smoke PRIVATE
    FV3_MAP_SCALAR_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fortran_baseline/fv_mapz/fixtures")

  # WP-C fv_mapz pt/delz production-remap GPU smoke test (Prompt 52C, Stage
  # 8-9): fv3_pt_delz_remap.cu's real __global__ kernel launched on a real
  # GPU, single column, compared against the pt_delz-driver Fortran
  # fixtures. GPU-execution + tolerance-comparison check only; not
  # registered as a ctest (requires an actual GPU) -- run manually via srun.
  add_executable(fv3_wp_c_pt_delz_remap_gpu_smoke vertical/fv3_pt_delz_remap_smoke.cu)
  target_include_directories(fv3_wp_c_pt_delz_remap_gpu_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fv3_wp_c_pt_delz_remap_gpu_smoke PRIVATE fv3_dyn_core_gpu fv3_wp_c_vertical)
  target_compile_definitions(fv3_wp_c_pt_delz_remap_gpu_smoke PRIVATE
    FV3_PT_DELZ_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fortran_baseline/fv_mapz/pt_delz_fixtures")

  # WP-C fv_mapz pt/delz PRODUCTION-SHAPE GPU smoke test (Prompt 55, Stage 8):
  # unlike the single-column test above, this builds multi-dimensional
  # RiemViewC3 views with the SAME lower/stride layout the real
  # fv3_gpu_field_registry.F90 registration produces for a C32-shaped tile
  # (pe/peln with a 1-cell halo and (i,k,j) physical axis order; pt/delz/delp
  # with the real 3-cell halo; pk/pkz with no halo), then launches the real
  # 2D-grid kernel -- exactly the launch shape that triggered the Prompt 54
  # out-of-bounds finding. GPU-execution + tolerance-comparison check only;
  # not registered as a ctest (requires an actual GPU) -- run manually via
  # srun, ideally under compute-sanitizer memcheck.
  add_executable(fv3_wp_c_pt_delz_remap_prod_shape_smoke vertical/fv3_pt_delz_remap_prod_shape_smoke.cu)
  target_include_directories(fv3_wp_c_pt_delz_remap_prod_shape_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fv3_wp_c_pt_delz_remap_prod_shape_smoke PRIVATE fv3_dyn_core_gpu fv3_wp_c_vertical)
  target_compile_definitions(fv3_wp_c_pt_delz_remap_prod_shape_smoke PRIVATE
    FV3_PT_DELZ_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fortran_baseline/fv_mapz/pt_delz_fixtures")

  # WP-C fv_mapz REMAP_UV PRODUCTION-SHAPE GPU smoke test (Prompt 71, Stage
  # 7): same methodology as the PT/DELZ prod-shape smoke test above, but for
  # launch_remap_uv -- multi-dimensional RiemViewC3 views matching the real
  # fv3_gpu_field_registry.F90 registration for pe/u/v at C32 shape, launched
  # over the real union-bounding-box 2D grid. Correctness oracle: the Prompt
  # 71 Stage 2 iv=-1 fixtures (case10-13), reproduced exactly at every column
  # via a broadcast pe field plus an ak=pe2/bk=0 construction. Also exercises
  # the device-workspace path (bitwise-vs-local-scratch cross-check). GPU-
  # execution + tolerance-comparison check only; not registered as a ctest
  # (requires an actual GPU) -- run manually via srun, ideally under
  # compute-sanitizer memcheck.
  add_executable(fv3_wp_c_remap_uv_prod_shape_smoke vertical/fv3_remap_uv_prod_shape_smoke.cu)
  target_include_directories(fv3_wp_c_remap_uv_prod_shape_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fv3_wp_c_remap_uv_prod_shape_smoke PRIVATE fv3_dyn_core_gpu fv3_wp_c_vertical)
  target_compile_definitions(fv3_wp_c_remap_uv_prod_shape_smoke PRIVATE
    FV3_MAP_SCALAR_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fortran_baseline/fv_mapz/fixtures")

  # WP-C fv_mapz REMAP_Q PRODUCTION-SHAPE GPU smoke test (Prompt 85, Stage
  # 9): same methodology as the REMAP_UV prod-shape smoke test above, but for
  # launch_remap_q -- pe/ak/bk views matching the real
  # fv3_gpu_field_registry.F90 registration for pe at C32 shape, plus a
  # dedicated device buffer for q (which has no FieldRegistry view of its
  # own), launched over the plain interior is:ie x js:je grid. Correctness
  # oracle: the Prompt 84B mapn_tracer fixtures (case14-22), reproduced
  # exactly at every column via a broadcast pe field plus an ak=pe2/bk=0
  # construction. Also exercises the device-workspace path (bitwise-vs-
  # local-scratch cross-check). GPU-execution + tolerance-comparison check
  # only; not registered as a ctest (requires an actual GPU) -- run manually
  # via srun, ideally under compute-sanitizer memcheck.
  add_executable(fv3_wp_c_remap_q_prod_shape_smoke vertical/fv3_remap_q_prod_shape_smoke.cu)
  target_include_directories(fv3_wp_c_remap_q_prod_shape_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fv3_wp_c_remap_q_prod_shape_smoke PRIVATE fv3_dyn_core_gpu fv3_wp_c_vertical)
  target_compile_definitions(fv3_wp_c_remap_q_prod_shape_smoke PRIVATE
    FV3_MAP_SCALAR_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fortran_baseline/fv_mapz/fixtures")

  # WP-C fv_mapz REMAP_Q nq=2 production-SHAPE GPU smoke test (Prompt 86B,
  # Stage 7; kord corrected by Prompt 87): extends the nq=1 smoke test
  # above to the real production nq=2 schedule (sphum kord=9, cld_amt
  # kord=9) via two sequential
  # launch_remap_q calls sharing one H2D/D2H over a contiguous 2-tracer-
  # plane buffer -- exactly launch_remap_q_ctx's Stage 4 mechanism.
  # Correctness oracle: test_remap_q_nq2_scope.cpp's synthetic sphum-like/
  # cld_amt-like profiles via the same host map_scalar_column path. GPU-
  # execution + tolerance-comparison check only; not registered as a ctest
  # (requires an actual GPU) -- run manually via srun, ideally under
  # compute-sanitizer memcheck.
  add_executable(fv3_wp_c_remap_q_nq2_prod_shape_smoke vertical/fv3_remap_q_nq2_prod_shape_smoke.cu)
  target_include_directories(fv3_wp_c_remap_q_nq2_prod_shape_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fv3_wp_c_remap_q_nq2_prod_shape_smoke PRIVATE fv3_dyn_core_gpu fv3_wp_c_vertical)

  # WP-C fv_mapz REMAP_PRESSUREVARS PRODUCTION-SHAPE GPU smoke test (Prompt
  # 92, Stage 3/8/9): same broadcast methodology as the REMAP_UV/REMAP_Q
  # prod-shape smoke tests above, but for launch_remap_pressurevars --
  # hand-built RiemViewC3/PtDelzView1/PressureVarsPlaneView views over a
  # 3-cell-halo C32-ish tile, launched over the plain interior is:ie x
  # js:je grid spanning multiple thread blocks. Correctness oracle: the
  # exact 10 fixtures from test_remap_pressurevars_host_reference.cpp
  # (Prompt 91), duplicated into this file (no header-level export exists
  # to share them from). GPU-execution + bitwise/round-off-tolerance
  # comparison check only; not registered as a ctest (requires an actual
  # GPU) -- run manually via srun, ideally under compute-sanitizer memcheck.
  add_executable(fv3_wp_c_remap_pressurevars_prod_shape_smoke
    vertical/fv3_remap_pressurevars_prod_shape_smoke.cu)
  target_include_directories(fv3_wp_c_remap_pressurevars_prod_shape_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fv3_wp_c_remap_pressurevars_prod_shape_smoke PRIVATE
    fv3_dyn_core_gpu fv3_wp_c_vertical)

  # WP-C/WP-E FV_StateMod.F90 State_To_FV PRODUCTION-SHAPE GPU smoke test
  # (Prompt 99, Stage 9): same broadcast/tile methodology as the
  # REMAP_PRESSUREVARS prod-shape smoke test above, but for
  # launch_state_to_fv_phase_a/_phase_c -- hand-built RiemViewC3/
  # StateInputView/PsPackView views over a 3-cell-halo tile sized to match
  # test_state_to_fv_host_reference.cpp's fixtures (ni=6, nj=5, npz=10)
  # exactly. Correctness oracle: the exact 8 fixtures from that file
  # (Prompt 98), duplicated into this file (no header-level export exists
  # to share them from), including the mandatory
  # sentinel_layout_transposition_check fixture. GPU-execution +
  # bitwise/round-off-tolerance comparison check only; not registered as a
  # ctest (requires an actual GPU) -- run manually via srun, ideally under
  # compute-sanitizer memcheck.
  add_executable(fv3_wp_c_state_to_fv_prod_shape_smoke
    vertical/fv3_state_to_fv_prod_shape_smoke.cu)
  target_include_directories(fv3_wp_c_state_to_fv_prod_shape_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fv3_wp_c_state_to_fv_prod_shape_smoke PRIVATE
    fv3_dyn_core_gpu fv3_wp_c_vertical)

  # WP-C fv_fill_mod fillz dedicated device-kernel GPU smoke test (Prompt
  # 88B, Stage 6): GPU-executes the exact fillz_column<T> template (already
  # host-bitwise-validated by fv3_wp_c_fillz_cuda_vs_host) from a real
  # __global__ kernel against Prompt 88A's 9 adversarial fixtures. See the
  # file's own header comment for why fillz's corrective branches are
  # exercised here rather than only through the full REMAP_Q pipeline. GPU-
  # execution + bitwise-comparison check only; not registered as a ctest
  # (requires an actual GPU) -- run manually via srun.
  add_executable(fv3_wp_c_fillz_device_kernel_smoke vertical/fv3_fillz_device_kernel_smoke.cu)
  target_include_directories(fv3_wp_c_fillz_device_kernel_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical)

  # WP-C fv_mapz REMAP_Q nq=2/kord=(9,9)/fill=TRUE production-SHAPE GPU
  # smoke test (Prompt 88B, Stage 6): the fv3_wp_c_remap_q_nq2_prod_shape_
  # smoke sibling above, with RemapQArgs::fill=true, compared against a
  # host oracle running map_scalar_column then fillz_column (the real
  # mapn_tracer -> fillz production sequence). GPU-execution + tolerance-
  # comparison check only; not registered as a ctest (requires an actual
  # GPU) -- run manually via srun, ideally under compute-sanitizer memcheck.
  add_executable(fv3_wp_c_remap_q_nq2_fill_true_prod_shape_smoke
    vertical/fv3_remap_q_nq2_fill_true_prod_shape_smoke.cu)
  target_include_directories(fv3_wp_c_remap_q_nq2_fill_true_prod_shape_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(fv3_wp_c_remap_q_nq2_fill_true_prod_shape_smoke PRIVATE
    fv3_dyn_core_gpu fv3_wp_c_vertical)

  # WP-D operators tests (CPU-visible formulas)
  add_executable(fv3_wp_d_operators_test tests/operators/test_wp_d.cpp)
  target_link_libraries(fv3_wp_d_operators_test PRIVATE fv3_wp_d_operators)
  add_test(NAME fv3_wp_d_operators COMMAND fv3_wp_d_operators_test)

  add_executable(fv3_wp_d_contract_test tests/operators/test_wp_d_contract.cpp)
  target_link_libraries(fv3_wp_d_contract_test PRIVATE fv3_wp_d_operators)
  add_test(NAME fv3_wp_d_contract COMMAND fv3_wp_d_contract_test)

  # Milestone 3A.10: cubed-sphere connectivity and panel-seam algebra. Host-only
  # by design -- the seam index map and vector transforms are pure functions, so
  # they are provable without a GPU, and the same inline code is what the device
  # kernels compile.
  add_executable(fv3_halo_topology_test tests/runtime/test_halo_topology.cpp)
  target_link_libraries(fv3_halo_topology_test PRIVATE fv3_dyn_core_gpu)
  add_test(NAME fv3_halo_topology COMMAND fv3_halo_topology_test)

  # Prompt 109 Phases 1-4: generalized sub-tile (multi-rank-per-face) halo
  # topology. Host-only for the same reason as fv3_halo_topology_test above --
  # every function in fv3_gpu_halo_subtile_topology.hpp is a pure, total
  # function of its arguments, ADDITIVE to (never modifying) the legacy
  # topology this test regression-checks against at layout=1x1.
  add_executable(fv3_halo_subtile_topology_test tests/runtime/test_halo_subtile_topology.cpp)
  target_link_libraries(fv3_halo_subtile_topology_test PRIVATE fv3_dyn_core_gpu)
  add_test(NAME fv3_halo_subtile_topology COMMAND fv3_halo_subtile_topology_test)

  # Milestone 3A.11V Task 2: the per-level damping translation. Host-only for
  # the same reason as the topology test -- translate_damping() is a total
  # function of its arguments, so the entire authoritative npz=10 production
  # column can be asserted value-for-value without a GPU. This is the test that
  # would have caught the fabricated (d4_bg=0.16, nord_k=2) benchmark table.
  add_executable(fv3_damping_translation_test tests/runtime/test_damping_translation.cpp)
  target_link_libraries(fv3_damping_translation_test PRIVATE fv3_dyn_core_gpu)
  add_test(NAME fv3_damping_translation COMMAND fv3_damping_translation_test)

  # Milestone 3A.11V Task 1: the globally single-valued cubed-sphere geometry the
  # benchmark IC is built from. Deliberately a SEPARATE library and deliberately
  # NOT linked into fv3_dyn_core_gpu -- it is benchmark/test infrastructure that
  # ports FV3's grid construction for the harness, not production numerics, and
  # nothing in the shipped runtime may come to depend on it.
  add_library(fv3_cube_geometry STATIC benchmarks/fv3_cube_geometry.cpp)
  target_include_directories(fv3_cube_geometry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
  )

  # The seam-continuity test required by Task 1. Host-only, and that is the
  # point: single-valuedness of the initial state is a property of the geometry,
  # so it is provable on a login node with no GPU and no CUDA. Three of its six
  # checks (T2 against FV3's own 13 seam copies, T4 by brute-force geometric
  # cell matching, T5 by brute-force edge matching) never consult the halo index
  # map, so they cannot pass by agreeing with the code under test.
  add_executable(fv3_cube_seam_test tests/benchmarks/test_cube_seam_continuity.cpp)
  target_link_libraries(fv3_cube_seam_test PRIVATE fv3_cube_geometry)
  add_test(NAME fv3_cube_seam COMMAND fv3_cube_seam_test)

  # Prompt 5B.1 Integrated smoke test (requires CUDA and GPU)
  if(FV3_DYN_CORE_ENABLE_CUDA)
    add_executable(fv3_integrated_smoke_test tests/integration/test_integrated_smoke.cpp)
    target_include_directories(fv3_integrated_smoke_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/operators
    )
    target_link_libraries(fv3_integrated_smoke_test PRIVATE
      fv3_dyn_core_gpu
      fv3_wp_d_cuda
      CUDA::cudart
    )
    # This test requires a GPU to run - mark as needing GPU label
    add_test(NAME fv3_integrated_smoke COMMAND fv3_integrated_smoke_test)
    set_tests_properties(fv3_integrated_smoke PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Milestone 3A.1 production dispatch smoke test (requires CUDA and GPU)
    add_executable(fv3_dispatch_smoke_test tests/integration/test_dispatch_smoke.cpp)
    target_include_directories(fv3_dispatch_smoke_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    )
    # Threads::Threads is for Phase E (Milestone 3A.10 §14), which runs the six
    # cubed-sphere tiles on six std::threads because a halo exchange is
    # two-sided: a tile's transfer must not read a neighbour's send buffer
    # before that neighbour has packed it. See include/fv3_gpu_halo_peer.hpp.
    find_package(Threads REQUIRED)
    target_link_libraries(fv3_dispatch_smoke_test PRIVATE
      fv3_dyn_core_gpu
      fv3_production_dispatch
      CUDA::cudart
      Threads::Threads
    )
    # This test requires a GPU to run - mark as needing GPU label
    add_test(NAME fv3_dispatch_smoke COMMAND fv3_dispatch_smoke_test)
    set_tests_properties(fv3_dispatch_smoke PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Prompt 109 Phases 1-4, Stage 15: GPU-EXECUTED smoke test for the
    # generalized sub-tile halo topology. Proves the new FV3_HALO_HD
    # topology functions (fv3_gpu_halo_subtile_topology.hpp) actually
    # execute on a real device and match the host results bit-for-bit --
    # NOT a halo-fill/data-movement kernel (that is Phase 5+, out of this
    # task's change budget). Reports 77 (SKIP) with no GPU, matching the
    # convention used by every other GPU-requiring test above.
    add_executable(fv3_halo_subtile_topology_cuda_test
                   tests/runtime/test_halo_subtile_topology_cuda.cu)
    target_link_libraries(fv3_halo_subtile_topology_cuda_test PRIVATE
      fv3_dyn_core_gpu
      CUDA::cudart
    )
    set_target_properties(fv3_halo_subtile_topology_cuda_test PROPERTIES CUDA_STANDARD 17)
    add_test(NAME fv3_halo_subtile_topology_cuda COMMAND fv3_halo_subtile_topology_cuda_test)
    set_tests_properties(fv3_halo_subtile_topology_cuda PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Milestone 3A.10 §13: permanent integrated MULTI-TILE halo exchange test.
    # All six cubed-sphere tiles, exchanged on the device through the real
    # pack -> D2D transfer -> seam-transforming unpack path, with every halo
    # value decoded back to its originating tile, index and sign.
    add_executable(fv3_halo_exchange_multitile_test
                   tests/integration/test_halo_exchange_multitile.cu)
    target_link_libraries(fv3_halo_exchange_multitile_test PRIVATE
      fv3_dyn_core_gpu
      CUDA::cudart
    )
    set_target_properties(fv3_halo_exchange_multitile_test PROPERTIES CUDA_STANDARD 17)
    add_test(NAME fv3_halo_exchange_multitile COMMAND fv3_halo_exchange_multitile_test)
    set_tests_properties(fv3_halo_exchange_multitile PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Prompt 111 (Phase 5, edges-only slice): sub-tile halo FIELD-DATA
    # exchange test. Generalizes the multitile test above to layout=1x1 (N=6,
    # a bitwise regression against the legacy builder's already-proven
    # behavior) and layout=2x2 (N=24 synthetic sub-ranks), using the new
    # additive fv3_gpu_halo_subtile_view.hpp builder in place of
    # make_cubed_sphere_topology -- exercises the REAL, UNMODIFIED
    # launch_halo_pack/launch_halo_transfer/launch_halo_unpack kernel chain.
    # Edges only; no corner message. Reports 77 (SKIP) with no GPU.
    add_executable(fv3_halo_subtile_exchange_test
                   tests/integration/test_halo_subtile_exchange.cu)
    target_link_libraries(fv3_halo_subtile_exchange_test PRIVATE
      fv3_dyn_core_gpu
      CUDA::cudart
    )
    set_target_properties(fv3_halo_subtile_exchange_test PROPERTIES CUDA_STANDARD 17)
    add_test(NAME fv3_halo_subtile_exchange COMMAND fv3_halo_subtile_exchange_test)
    set_tests_properties(fv3_halo_subtile_exchange PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Milestone 3A.15 §11-§13: permanent SIX-RANK MPI halo transport test.
    # The cross-process counterpart of the multi-tile test above: one tile per MPI
    # process, exchanged through pack -> host-staged MPI transfer -> unpack, with
    # every received value decoded back to its originating tile, index and sign.
    #
    # Only built when the MPI transport itself is built, and registered with
    # MPIEXEC so ctest runs it on exactly six ranks -- it refuses any other rank
    # count, because one rank holds exactly one cubed-sphere tile. It needs a GPU
    # visible to all six ranks, so like the other integrated tests it reports 77
    # (SKIP) rather than failing where no device exists.
    if(FV3_DYN_CORE_ENABLE_MPI_HALO)
      add_executable(fv3_halo_mpi_transport_test
                     tests/integration/test_halo_mpi_transport.cu)
      target_link_libraries(fv3_halo_mpi_transport_test PRIVATE
        fv3_dyn_core_gpu
        CUDA::cudart
        MPI::MPI_CXX
      )
      set_target_properties(fv3_halo_mpi_transport_test PROPERTIES CUDA_STANDARD 17)
      add_test(NAME fv3_halo_mpi_transport
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 6
                       ${MPIEXEC_PREFLAGS} $<TARGET_FILE:fv3_halo_mpi_transport_test>
                       ${MPIEXEC_POSTFLAGS})
      set_tests_properties(fv3_halo_mpi_transport PROPERTIES
        LABELS "gpu;mpi"
        SKIP_RETURN_CODE 77
      )

      # Prompt 109 Phases 1-4, Stage 14: permanent 24-RANK sub-tile halo MPI
      # ROUTING test (fv3_gpu_halo_subtile_topology.hpp). Host-only, same
      # reason as fv3_halo_subtile_topology_test above -- pure topology/tag
      # logic, no CUDA field ever moves here -- so it needs MPI::MPI_CXX
      # only, not CUDA::cudart, and does not need a GPU visible at runtime.
      # Reuses the same MPI::MPI_CXX target found above; kept nested here
      # rather than hoisted out because that target only exists when both
      # FV3_DYN_CORE_ENABLE_CUDA and FV3_DYN_CORE_ENABLE_MPI_HALO are ON.
      add_executable(fv3_halo_subtile_mpi_routing_test
                     tests/integration/test_halo_subtile_mpi_routing.cpp)
      target_link_libraries(fv3_halo_subtile_mpi_routing_test PRIVATE
        fv3_dyn_core_gpu
        MPI::MPI_CXX
      )
      add_test(NAME fv3_halo_subtile_mpi_routing
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 24
                       ${MPIEXEC_PREFLAGS} $<TARGET_FILE:fv3_halo_subtile_mpi_routing_test>
                       ${MPIEXEC_POSTFLAGS})
      set_tests_properties(fv3_halo_subtile_mpi_routing PROPERTIES
        LABELS "mpi"
      )

      # Prompt 112, Stage 13: synthetic 24-RANK REAL-MPI sub-tile halo
      # TRANSPORT existence proof (test_halo_subtile_mpi_transport.cu).
      # Cross-process counterpart of fv3_halo_subtile_exchange_test's
      # layout=2x2 in-process case, exercising MpiHaloTransport's Prompt 112
      # Stage 7 arbitrary-size storage over a REAL 24-rank communicator via
      # the TEST-ONLY create_mpi_transport_for_testing() factory.
      # create_mpi_transport()'s production comm_size==6 gate is never
      # called by this target. Needs a GPU visible to all 24 ranks; reports
      # 77 (SKIP) rather than failing where no device exists.
      add_executable(fv3_halo_subtile_mpi_transport_test
                     tests/integration/test_halo_subtile_mpi_transport.cu)
      target_link_libraries(fv3_halo_subtile_mpi_transport_test PRIVATE
        fv3_dyn_core_gpu
        CUDA::cudart
        MPI::MPI_CXX
      )
      set_target_properties(fv3_halo_subtile_mpi_transport_test PROPERTIES CUDA_STANDARD 17)
      add_test(NAME fv3_halo_subtile_mpi_transport
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 24
                       ${MPIEXEC_PREFLAGS} $<TARGET_FILE:fv3_halo_subtile_mpi_transport_test>
                       ${MPIEXEC_POSTFLAGS})
      set_tests_properties(fv3_halo_subtile_mpi_transport PROPERTIES
        LABELS "gpu;mpi"
        SKIP_RETURN_CODE 77
      )
    endif()

    # Milestone 3A.11T §4: permanent LAYOUT-SENSITIVE pe/peln test.
    # pe and peln are the FV3 axis-order exception -- authoritative FV3 declares
    # pe(is-1:ie+1, npz+1, js-1:je+1), i.e. (i,k,j), while pk/gz/zh/pkc/pk3 in
    # the same routines are (i,j,k). Both share FV3_GPU_STAGGER_INTERFACE, so
    # stagger carries no axis information and the order is expressible only
    # through descriptor strides. This test seeds values that uniquely encode
    # all three coordinates, so any i/j/k permutation aliases into a detectable
    # mismatch rather than round-tripping silently.
    add_executable(fv3_pe_peln_layout_test
                   tests/integration/test_pe_peln_layout.cu)
    target_link_libraries(fv3_pe_peln_layout_test PRIVATE
      fv3_dyn_core_gpu
      CUDA::cudart
    )
    set_target_properties(fv3_pe_peln_layout_test PROPERTIES CUDA_STANDARD 17)
    add_test(NAME fv3_pe_peln_layout COMMAND fv3_pe_peln_layout_test)
    set_tests_properties(fv3_pe_peln_layout PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Milestone 3A.11: small-scale Fortran-vs-CUDA performance benchmark.
    # PRE-VALIDATION PERFORMANCE CHARACTERIZATION -- see
    # docs/agent_handoff/milestone_3a11_small_perf.md. Not registered as a
    # ctest: it is a manually-invoked timing tool with CLI-selectable grid
    # size/mode/repeats, not a pass/fail correctness check.
    if(BUILD_FV3_GPU_PERF_BENCH)
      add_executable(fv3_gpu_perf_bench benchmarks/run_small_perf_bench.cpp)
      target_include_directories(fv3_gpu_perf_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime
      )
      find_package(Threads REQUIRED)
      target_link_libraries(fv3_gpu_perf_bench PRIVATE
        fv3_dyn_core_gpu
        fv3_production_dispatch
        fv3_cube_geometry
        CUDA::cudart
        Threads::Threads
      )
      # Milestone 3A.15 SS12: --mpi-halo (six MPI processes, one tile per rank).
      #
      # The definition has to be repeated HERE. FV3_DYN_CORE_ENABLE_MPI_HALO=1 is
      # PRIVATE on fv3_dyn_core_gpu, so it does not reach this translation unit; a
      # bench compiled without it would preprocess every --mpi-halo line away and
      # then report "requires a build configured with ...ON" in a build that has it
      # on. That is the same class of silent-preprocess-away hazard already
      # recorded at CMakeLists.txt:164-168, so it is guarded explicitly rather
      # than inherited.
      if(FV3_DYN_CORE_ENABLE_MPI_HALO)
        target_compile_definitions(fv3_gpu_perf_bench PRIVATE
                                   FV3_DYN_CORE_ENABLE_MPI_HALO=1)
        target_link_libraries(fv3_gpu_perf_bench PRIVATE MPI::MPI_CXX)
      endif()
    endif()

    # Milestone 3A.4 WP-B height-transport CUDA behavior test (requires CUDA and GPU)
    # Parts A-J: interior order-6 transport, panel-edge PPM, copy_corners,
    # edge_profile, both damping branches, del6_vt_flux, the area-weighted zh
    # update, the cross-WP path through WP-C's launch_update_dz_d
    # (fv3_wp_c_vertical), and storage-contract rejection cases.
    add_executable(fv3_wp_b_cuda_smoke_test tests/dgrid/test_wp_b_cuda_smoke.cpp)
    target_include_directories(fv3_wp_b_cuda_smoke_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/dgrid
      ${CMAKE_CURRENT_SOURCE_DIR}/vertical
    )
    target_link_libraries(fv3_wp_b_cuda_smoke_test PRIVATE
      fv3_dyn_core_gpu
      fv3_wp_b_cuda
      fv3_wp_c_vertical
      CUDA::cudart
    )
    # This test requires a GPU to run - mark as needing GPU label
    add_test(NAME fv3_wp_b_cuda_smoke COMMAND fv3_wp_b_cuda_smoke_test)
    set_tests_properties(fv3_wp_b_cuda_smoke PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Milestone 3A.5 WP-B.2 arbitrary scalar transport / deln_flux CUDA behavior
    # test (requires CUDA and GPU). Kept as a separate executable from
    # fv3_wp_b_cuda_smoke_test because the scalar transport contract is distinct
    # from the update_dz_d height contract (nk == npz cell levels, no per-level
    # damping split, deln_flux instead of del6_vt_flux, no arena).
    # Parts A-K: interior order-6, panel-edge PPM, corner rotation, deln_flux
    # enabled/disabled, nord dependence, the (damp_c*da_min)**(nord+1)
    # coefficient law, mass weighting and hord<0 positivity, R8/R4 precision,
    # and unsupported-configuration rejection. Part L (Prompt 76): hord==8
    # interior/seam/corner fv_tp_2d against a bitwise device-mirroring host
    # reference.
    add_executable(fv3_wp_b2_scalar_transport_test
      tests/dgrid/test_wp_b2_scalar_transport_cuda.cpp)
    target_include_directories(fv3_wp_b2_scalar_transport_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/dgrid
    )
    target_link_libraries(fv3_wp_b2_scalar_transport_test PRIVATE
      fv3_dyn_core_gpu
      fv3_wp_b_cuda
      CUDA::cudart
    )
    # This test requires a GPU to run - mark as needing GPU label
    add_test(NAME fv3_wp_b2_scalar_transport
      COMMAND fv3_wp_b2_scalar_transport_test)
    set_tests_properties(fv3_wp_b2_scalar_transport PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Prompt 76 hord==8 fv_tp_2d CUDA extension correctness test (requires CUDA
    # and GPU). Kept as its own executable rather than folded into
    # fv3_wp_b2_scalar_transport_test: it exercises a NEW dispatch branch
    # (opt.hord == 8) against an INDEPENDENT host transcription of tp_core.F90's
    # iord>=7/hord==8 xppm/yppm branch, cross-checked by construction against
    # both the CUDA device functions and the separate Prompt-75 host harness
    # (tests/host/test_fv_tp_2d_hord8_hostharness.cpp).
    # Parts A-D: interior hord==8, x-direction panel edge, y-direction panel
    # edge, all-four-corners C512-like field. Part E: confirms the hord==6
    # dispatch path is not perturbed by the new hord==8 branch.
    add_executable(fv3_wp_b2_hord8_transport_test
      tests/dgrid/test_wp_b2_hord8_transport_cuda.cpp)
    target_include_directories(fv3_wp_b2_hord8_transport_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/dgrid
    )
    target_link_libraries(fv3_wp_b2_hord8_transport_test PRIVATE
      fv3_dyn_core_gpu
      fv3_wp_b_cuda
      CUDA::cudart
    )
    # This test requires a GPU to run - mark as needing GPU label
    add_test(NAME fv3_wp_b2_hord8_transport
      COMMAND fv3_wp_b2_hord8_transport_test)
    set_tests_properties(fv3_wp_b2_hord8_transport PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Prompt 76 Stage 1: dedicated pert_ppm(iv=1) device-helper unit test,
    # independent of the hord==8 face-flux fixtures above. Five named
    # cases (monotonic, local extremum, sign change, boundary, near-flat)
    # compared bitwise against an independent host transcription of
    # tp_core.F90:1238-1252 (iv/=0 branch).
    add_executable(fv3_pert_ppm_iv1_test
      tests/dgrid/test_pert_ppm_iv1_cuda.cu)
    target_include_directories(fv3_pert_ppm_iv1_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/dgrid
    )
    target_link_libraries(fv3_pert_ppm_iv1_test PRIVATE
      fv3_dyn_core_gpu
      CUDA::cudart
    )
    add_test(NAME fv3_pert_ppm_iv1
      COMMAND fv3_pert_ppm_iv1_test)
    set_tests_properties(fv3_pert_ppm_iv1 PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Milestone 3A.6 WP-B.3 authoritative momentum transport CUDA behavior test
    # (requires CUDA and GPU). Every part compares the CUDA result against an
    # independent host transcription written directly from sw_core.F90 -- the
    # GPU code is never compared against itself.
    # Parts A-J: xtp_u interior, ytp_v interior, xtp_u panel edge, ytp_v panel
    # edge, vb/ub construction + D-wind update, kinetic-energy terms including
    # the four-corner fix, cubed-sphere seam/corner liveness, a
    # precision-sensitive near-tie case, hord_mt sign equivalence, and
    # unsupported-configuration rejection.
    add_executable(fv3_wp_b3_momentum_test
      tests/dgrid/test_wp_b3_momentum_cuda.cpp)
    target_include_directories(fv3_wp_b3_momentum_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/dgrid
    )
    target_link_libraries(fv3_wp_b3_momentum_test PRIVATE
      fv3_dyn_core_gpu
      fv3_wp_b_cuda
      CUDA::cudart
    )
    # This test requires a GPU to run - mark as needing GPU label
    add_test(NAME fv3_wp_b3_momentum COMMAND fv3_wp_b3_momentum_test)
    set_tests_properties(fv3_wp_b3_momentum PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Milestone 3A.7 WP-B.4 authoritative divergence-damping / corner slice CUDA
    # behavior test (requires CUDA and GPU). Host references are transcribed from
    # sw_core.F90 d_sw :1277-1534 and smag_corner :1857-1947 in the Fortran's own
    # multi-pass structure, so agreement is evidence rather than tautology.
    # Parts A-L: branch L (nord == 0) at the seam and in the interior,
    # fill_corners BGRID scalar and DGRID vector rotations, the branch-H
    # iterate, vort zeroing + the high-order finalize, smag_corner (which reuses
    # WP-D's a2b_ord4 rather than duplicating it), the d_con heat-source chain,
    # the damp_v wind update, bitwise identity of the vorticity-damping adapter
    # with launch_del6_vt_flux_cuda, unsupported-configuration rejection, and the
    # DivergenceLevels selector semantics.
    #
    # fv3_wp_d_cuda is already PUBLIC on fv3_wp_b_cuda (smag_corner calls
    # launch_a2b_ord4_global_cuda); it is repeated here because this test also
    # calls that launcher DIRECTLY to build its reference sh.
    add_executable(fv3_wp_b4_divergence_test
      tests/dgrid/test_wp_b4_divergence_cuda.cpp)
    target_include_directories(fv3_wp_b4_divergence_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/dgrid
      ${CMAKE_CURRENT_SOURCE_DIR}/operators
    )
    target_link_libraries(fv3_wp_b4_divergence_test PRIVATE
      fv3_dyn_core_gpu
      fv3_wp_b_cuda
      fv3_wp_d_cuda
      CUDA::cudart
    )
    # This test requires a GPU to run - mark as needing GPU label
    add_test(NAME fv3_wp_b4_divergence COMMAND fv3_wp_b4_divergence_test)
    set_tests_properties(fv3_wp_b4_divergence PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

    # Milestone 3A.8 WP-B.5 authoritative d_sw driver test (requires CUDA and
    # GPU). Parts A-H are host transcriptions of the DRIVER-OWNED kernels from
    # sw_core.F90 d_sw (contravariant/edge/corner winds, advective fluxes and
    # ra_x/ra_y, the joint pt/delp update, the dpx accumulation, the w chain, the
    # flux capacitors, the heat_source reset, and the derived per-level
    # coefficient arrays). Parts I-M exercise the COMPLETE launch_d_sw_cuda:
    # mixed per-level damping branches inside ONE invocation compared against the
    # equivalent uniform-parameter runs level by level, flux-capacitor
    # accumulation across two invocations, end-to-end anti-dependence
    # invariants, unsupported-configuration fences, and direct GPU execution at
    # the seam and in the interior.
    add_executable(fv3_wp_b5_dsw_test
      tests/dgrid/test_wp_b5_dsw_cuda.cpp)
    target_include_directories(fv3_wp_b5_dsw_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/dgrid
      ${CMAKE_CURRENT_SOURCE_DIR}/operators
    )
    target_link_libraries(fv3_wp_b5_dsw_test PRIVATE
      fv3_dyn_core_gpu
      fv3_wp_b_cuda
      fv3_wp_d_cuda
      CUDA::cudart
    )
    # This test requires a GPU to run - mark as needing GPU label
    add_test(NAME fv3_wp_b5_dsw COMMAND fv3_wp_b5_dsw_test)
    set_tests_properties(fv3_wp_b5_dsw PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )
  endif()
endif()
