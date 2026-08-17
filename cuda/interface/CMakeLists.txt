# GEOS FV3 CUDA Interface
#
# This library provides the Fortran-to-CUDA bridge for the WP-E dyn_core
# subsystem. It is optional and only built when BUILD_FV3_CUDA is enabled.

cmake_minimum_required(VERSION 3.18)

# Interface library sources
set(INTERFACE_SRCS
  fv3_gpu_interface_mod.F90
)

# Create the interface library
add_library(fv3_cuda_interface STATIC ${INTERFACE_SRCS})

# Require the dyn_core CUDA library
target_link_libraries(fv3_cuda_interface
  PUBLIC
    fv3_dyn_core_cuda
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
