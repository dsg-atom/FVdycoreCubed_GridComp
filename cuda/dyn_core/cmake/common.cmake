# Platform-neutral defaults. Toolchain files may set CMAKE_CUDA_ARCHITECTURES;
# this runtime deliberately does not select a GPU generation.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard for FV3 GPU runtime")
