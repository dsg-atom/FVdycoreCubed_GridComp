cmake_minimum_required(VERSION 3.21)
project(FV3DynCoreGPU LANGUAGES C CXX)

option(FV3_DYN_CORE_ENABLE_CUDA "Build the CUDA-backed runtime" OFF)
set(FV3_DYN_CORE_PRECISION "R8" CACHE STRING "FV precision: R4, R4R8, or R8")
set_property(CACHE FV3_DYN_CORE_PRECISION PROPERTY STRINGS R4 R4R8 R8)

# Core runtime library (WP-E)
add_library(fv3_dyn_core_gpu runtime/fv3_gpu_runtime.cpp runtime/fv3_gpu_contract.cpp)
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
  )
  target_include_directories(fv3_wp_a_cgrid PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cgrid
  )
  target_link_libraries(fv3_wp_a_cgrid PUBLIC fv3_dyn_core_gpu CUDA::cudart)
  set_target_properties(fv3_wp_a_cgrid PROPERTIES CUDA_STANDARD 17)

  # WP-C vertical CUDA sources
  add_library(fv3_wp_c_vertical STATIC vertical/fv3_vertical_kernels.cu)
  target_include_directories(fv3_wp_c_vertical PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vertical
  )
  target_link_libraries(fv3_wp_c_vertical PUBLIC fv3_dyn_core_gpu CUDA::cudart)
  set_target_properties(fv3_wp_c_vertical PROPERTIES CUDA_STANDARD 17)

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
  # transport). Both .cu files share dgrid/wp_b_ppm_shared.cuh so the PPM
  # constants, seam predicates and launch helpers have one definition.
  add_library(fv3_wp_b_cuda STATIC
    dgrid/wp_b_cuda.cu
    dgrid/wp_b_momentum_cuda.cu
    dgrid/wp_b_contract_cuda.cpp)
  target_include_directories(fv3_wp_b_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/dgrid
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(fv3_wp_b_cuda PUBLIC fv3_wp_b_dgrid CUDA::cudart)
  set_target_properties(fv3_wp_b_cuda PROPERTIES CUDA_STANDARD 17)

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

  # WP-D operators tests (CPU-visible formulas)
  add_executable(fv3_wp_d_operators_test tests/operators/test_wp_d.cpp)
  target_link_libraries(fv3_wp_d_operators_test PRIVATE fv3_wp_d_operators)
  add_test(NAME fv3_wp_d_operators COMMAND fv3_wp_d_operators_test)

  add_executable(fv3_wp_d_contract_test tests/operators/test_wp_d_contract.cpp)
  target_link_libraries(fv3_wp_d_contract_test PRIVATE fv3_wp_d_operators)
  add_test(NAME fv3_wp_d_contract COMMAND fv3_wp_d_contract_test)

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
    target_link_libraries(fv3_dispatch_smoke_test PRIVATE
      fv3_dyn_core_gpu
      fv3_production_dispatch
      CUDA::cudart
    )
    # This test requires a GPU to run - mark as needing GPU label
    add_test(NAME fv3_dispatch_smoke COMMAND fv3_dispatch_smoke_test)
    set_tests_properties(fv3_dispatch_smoke PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
    )

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
    # and unsupported-configuration rejection.
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
  endif()
endif()
