# FV3 CUDA Interface Tests

# Interface C ABI test (no Fortran required)
add_executable(fv3_interface_c_test test_interface_c.cpp)
target_link_libraries(fv3_interface_c_test PRIVATE fv3_dyn_core_gpu)
target_include_directories(fv3_interface_c_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../dyn_core/include
)
add_test(NAME fv3_interface_c COMMAND fv3_interface_c_test)
