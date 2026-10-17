# SPDX-License-Identifier: Apache-2.0
# cozy-kernels: one module per kernel group, every GPU target we serve inside it
# (`cozy_kernels.ARCHS`), the CUDA 13.0 runtime linked static, nanobind on Python's stable ABI.
# Each unit is compiled for its own targets only: an arch-specific instruction (wgmma, tcgen05)
# never meets a card it cannot run on.
cmake_minimum_required(VERSION 3.26)
project(cozy_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)

find_package(CUDAToolkit 13.0 REQUIRED)
find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# CUTLASS's headers at Runtime's pin (`kernel_sources.PINS`), or a tree already on the machine.
set(CUTLASS_COMMIT 098de2a652cf8f00fd70b2df54051c7eccbb855a)
set(CUTLASS_DIR "" CACHE PATH "An unpacked CUTLASS tree at ${CUTLASS_COMMIT}")
if(NOT CUTLASS_DIR)
    set(archive "${CMAKE_BINARY_DIR}/cutlass-${CUTLASS_COMMIT}.tar.gz")
    file(DOWNLOAD "https://github.com/NVIDIA/cutlass/archive/${CUTLASS_COMMIT}.tar.gz" "${archive}"
         EXPECTED_HASH SHA256=a8a1b952c7f6179bc8076f20d45859f18bd81f21f3d22b6766d7f3f0cceb5d54)
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${CMAKE_BINARY_DIR}")
    set(CUTLASS_DIR "${CMAKE_BINARY_DIR}/cutlass-${CUTLASS_COMMIT}")
endif()

set(INT8 "${CMAKE_CURRENT_SOURCE_DIR}/../src/cozy_runtime/internal/int8_op")
set(isolated $<$<PLATFORM_ID:Linux>:-fno-gnu-unique>)

# One 8-bit unit for `targets`, its family named by its defines (`int8.h`).
function(gemm_unit name source targets)
    add_library(${name} OBJECT "${INT8}/${source}")
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "${targets}")
    target_include_directories(${name} PRIVATE "${INT8}" "${CUTLASS_DIR}/include"
                               "${CUTLASS_DIR}/tools/util/include")
    # The GEMM waits on its quantize kernel by grid dependency (programmatic launch).
    target_compile_definitions(${name} PRIVATE CUTLASS_ENABLE_GDC_FOR_SM90
                               CUTLASS_ENABLE_GDC_FOR_SM100 ${ARGN})
    target_compile_options(${name} PRIVATE --expt-relaxed-constexpr
                           $<$<PLATFORM_ID:Linux>:-Xcompiler=-fno-gnu-unique>)
endfunction()

# mma.sync carries PTX (`80` is SASS and PTX), which the driver compiles for a card none names.
gemm_unit(rotate rotate.cu "80;86-real;89-real;90-real;100-real;103-real;120-real;121-real")
gemm_unit(int8_mma_sync gemm_sm80.cu "80;86-real;89-real;120-real;121-real" COZY_MMA_SYNC)
gemm_unit(int8_wgmma gemm_tma.cu "90a-real" COZY_WGMMA)
gemm_unit(int8_tcgen05 gemm_tma.cu "100a-real" COZY_TCGEN05)
gemm_unit(fp8_wgmma gemm_fp8.cu "90a-real" COZY_WGMMA)
gemm_unit(fp8_tcgen05 gemm_fp8.cu "100a-real;103a-real" COZY_TCGEN05)
gemm_unit(fp8_mma_sync gemm_fp8.cu "120a-real;121a-real" COZY_MMA_SYNC)

nanobind_add_module(_gemm NB_STATIC STABLE_ABI LTO csrc/gemm.cpp)
target_include_directories(_gemm PRIVATE "${INT8}" ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(_gemm PRIVATE ${isolated})
foreach(unit rotate int8_mma_sync int8_wgmma int8_tcgen05 fp8_wgmma fp8_tcgen05 fp8_mma_sync)
    target_sources(_gemm PRIVATE $<TARGET_OBJECTS:${unit}>)
endforeach()
target_link_libraries(_gemm PRIVATE CUDA::cudart_static ${CMAKE_DL_LIBS})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Static libstdc++/libgcc keep the wheel at its manylinux floor: only glibc's versioned
    # symbols stay external. -Bsymbolic binds the module's CUTLASS instantiations to itself,
    # never to another library's of the same name (comfy-kitchen's) in the process.
    target_link_options(_gemm PRIVATE -Wl,--build-id=none -Wl,-Bsymbolic -static-libstdc++
                        -static-libgcc)
    set_target_properties(_gemm PROPERTIES SUFFIX ".abi3.so")
endif()
install(TARGETS _gemm LIBRARY DESTINATION cozy_kernels)
