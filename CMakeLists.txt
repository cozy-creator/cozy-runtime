# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.26)
project(cozy_runtime_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# The reviewed upstream v0.2.31 CUDA 13 Linux wheel fleet policy. CMake initializes
# CMAKE_CUDA_ARCHITECTURES to nvcc's default before this point, so this must be
# explicit rather than guarded by a truthiness check.
set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;89;90a-real;100f;120f")

find_package(CUDAToolkit 13.0 REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

add_library(rms_rope_kernel OBJECT native/rms_rope.cu)
target_include_directories(rms_rope_kernel PRIVATE native ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(rms_rope_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda;--use_fast_math;-O3>
)

nanobind_add_module(_C NB_STATIC STABLE_ABI LTO native/binding.cpp)
target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x030C0000 NB_STATIC)
target_include_directories(_C PRIVATE native ${CUDAToolkit_INCLUDE_DIRS})
target_sources(_C PRIVATE $<TARGET_OBJECTS:rms_rope_kernel>)
target_link_libraries(_C PRIVATE CUDA::cudart_static ${CMAKE_DL_LIBS})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Static libstdc++/libgcc is what makes the wheel PUBLISHABLE. Linked dynamically, the
    # module references its build gcc's GLIBCXX_3.4.31/CXXABI_1.3.13, which pins the wheel to
    # manylinux_2_39 and excludes every glibc 2.34-2.38 host. Absorbed, the only external
    # versioned symbols left are glibc's own (max GLIBC_2.34), so auditwheel certifies
    # manylinux_2_34_x86_64. nanobind is already NB_STATIC, so no C++ ABI crosses this .so.
    target_link_options(_C PRIVATE -Wl,--build-id=none -static-libstdc++ -static-libgcc)
endif()
set_target_properties(_C PROPERTIES SUFFIX ".abi3.so")
install(TARGETS _C LIBRARY DESTINATION cozy_runtime/_kernels)
