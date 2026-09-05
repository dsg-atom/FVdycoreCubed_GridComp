# GEOS FV3 CUDA Interface
#
# This library provides the Fortran-to-CUDA bridge for the WP-E dyn_core
# subsystem. It is optional and only built when BUILD_FV3_CUDA is enabled.

cmake_minimum_required(VERSION 3.18)

# Interface library sources.
#
# ORDER IS THE MODULE DEPENDENCY ORDER, which CMake's Fortran dependency
# scanner will honour anyway but which is spelled out here because it is not
# obvious: the ABI type/parameter module first, then the field registry that
# builds fv3_gpu_field_desc arrays out of it, then the bridge that calls the
# registry and drives the C ABI.
#
# NOTE: this directory is not currently reached by any add_subdirectory() -- the
# top-level CMakeLists.txt adds only cuda/dyn_core, and the standalone
# integration compiles these .F90 sources directly into the fixture's
# @FVdycoreCubed_GridComp target. The list is kept accurate so that configuring
# this directory produces the right library rather than a stale one.
set(INTERFACE_SRCS
  fv3_gpu_abi_mod.F90
  fv3_gpu_field_registry.F90
  fv3_gpu_dyncore_bridge.F90
  fv3_gpu_interface_mod.F90
)

# Create the interface library
add_library(fv3_cuda_interface STATIC ${INTERFACE_SRCS})

# Require the dyn_core CUDA library. The target is fv3_dyn_core_gpu
# (cuda/dyn_core/CMakeLists.txt:32); the fv3_dyn_core_cuda name previously
# written here has never existed, which is consistent with this directory never
# having been configured. fv3_production_dispatch supplies
# fv3_gpu_install_production_dispatch, which the bridge calls.
target_link_libraries(fv3_cuda_interface
  PUBLIC
    fv3_dyn_core_gpu
    fv3_production_dispatch
)

# Include directories for the public API headers
target_include_directories(fv3_cuda_interface
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../dyn_core/include>
    $<INSTALL_INTERFACE:include/fv3_cuda>
)

# Fortran module output directory
set_target_properties(fv3_cuda_interface PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod
)

target_include_directories(fv3_cuda_interface
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
    $<INSTALL_INTERFACE:include/fv3_cuda/mod>
)

# Compile definitions for Fortran
target_compile_definitions(fv3_cuda_interface
  PUBLIC
    USE_FV3_CUDA
)

# Precision handling
if(FV_PRECISION STREQUAL "R4")
  target_compile_definitions(fv3_cuda_interface PRIVATE SINGLE_FV)
endif()

# Installation
install(TARGETS fv3_cuda_interface
  EXPORT fv3_cuda_interface-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mod/
  DESTINATION include/fv3_cuda/mod
  FILES_MATCHING PATTERN "*.mod"
)

# Test subdirectory
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
