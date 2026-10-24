# SPDX-License-Identifier: Apache-2.0
# Cozy's prebuilt kernels: one module per kernel group, every GPU target we serve inside it
# (`cozy_runtime._kernels.ARCHS`), the CUDA 13.0 runtime linked static, nanobind on Python's
# stable ABI. Installed under `cozy_runtime/_kernels`, they go into the native cozy-runtime wheels
# (kernels.yaml builds them, keyed by this directory: toolchain.env pins the toolchain).
# Each unit is compiled for its own targets only: an arch-specific instruction (wgmma, tcgen05)
# never meets a card it cannot run on.
cmake_minimum_required(VERSION 3.26)
project(cozy_runtime_kernels LANGUAGES CXX)
# CUDA 13.0 takes MSVC through 19.4x (Visual Studio 2022); a newer one builds only past nvcc's check.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.50)
    message(WARNING "MSVC ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER}) is newer than CUDA 13.0 "
                    "supports: building with -allow-unsupported-compiler")
    string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
endif()
enable_language(CUDA)
# The C and C++ runtimes linked static, as NVIDIA's cudart_static is (LIBCMT): a Windows module
# needs no Visual C++ redistributable (MSVCP140), as a Linux one links libstdc++ static.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
# MSVC stamps build times into objects and the module unless /Brepro: the same sources then build
# the same bytes, as on Linux, so a rebuild can be compared with the cached build it replaces.
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Brepro> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Brepro>)
    add_link_options(/Brepro)
endif()

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

# A GitHub source tree at a pinned commit and digest (Runtime's `kernel_sources.PINS`) as
# `${var}`, unless the caller names a tree already on the machine (`-D${var}=...`).
function(pinned var repository commit sha256)
    set(${var} "" CACHE PATH "${repository} at ${commit}")
    if(NOT ${var})
        get_filename_component(name "${repository}" NAME)
        set(archive "${CMAKE_BINARY_DIR}/${name}-${commit}.tar.gz")
        file(DOWNLOAD "https://github.com/${repository}/archive/${commit}.tar.gz" "${archive}"
             EXPECTED_HASH SHA256=${sha256})
        file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${CMAKE_BINARY_DIR}")
        set(${var} "${CMAKE_BINARY_DIR}/${name}-${commit}" PARENT_SCOPE)
    endif()
endfunction()

pinned(CUTLASS_DIR NVIDIA/cutlass 098de2a652cf8f00fd70b2df54051c7eccbb855a
       a8a1b952c7f6179bc8076f20d45859f18bd81f21f3d22b6766d7f3f0cceb5d54)

set(INT8 "${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm")
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
                           $<$<PLATFORM_ID:Linux>:-Xcompiler=-fno-gnu-unique>
                           $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/Zc:__cplusplus>)
endfunction()

# mma.sync carries PTX (`80` is SASS and PTX), which the driver compiles for a card none names.
set(EVERY "80;86-real;89-real;90-real;100-real;103-real;120-real;121-real")
gemm_unit(rotate rotate.cu "${EVERY}")
gemm_unit(int8_mma_sync gemm_sm80.cu "80;86-real;89-real;120-real;121-real" COZY_MMA_SYNC)
# W4A8's mma.sync GEMM, on every card with int8 MMA (Hopper too: design four-bit-lanes §5.2).
gemm_unit(w4a8 w4a8.cu "80;86-real;89-real;90-real;100-real;120-real;121-real")
# svdq.nvfp4's linear: block-scaled FP4 mma.sync, RTX 50 and DGX Spark only (`svdq.cu`).
gemm_unit(svdq svdq.cu "120a-real;121a-real")
set(units rotate int8_mma_sync w4a8 svdq)
# Every other GEMM is CUTLASS's TMA kernel: its 128-byte-aligned parameters are passed by value,
# which MSVC refuses (C2719), so a Windows build carries none (cr-201).
if(MSVC)
    set(tma OFF)
else()
    set(tma ON)
endif()
option(COZY_TMA "CUTLASS's TMA GEMMs: wgmma, tcgen05 and RTX 50's fp8" ${tma})
if(COZY_TMA)
    gemm_unit(int8_wgmma gemm_tma.cu "90a-real" COZY_WGMMA)
    gemm_unit(int8_tcgen05 gemm_tma.cu "100a-real" COZY_TCGEN05)
    gemm_unit(fp8_wgmma gemm_fp8.cu "90a-real" COZY_WGMMA)
    gemm_unit(fp8_tcgen05 gemm_fp8.cu "100a-real;103a-real" COZY_TCGEN05)
    gemm_unit(fp8_mma_sync gemm_fp8.cu "120a-real;121a-real" COZY_MMA_SYNC)
    list(APPEND units int8_wgmma int8_tcgen05 fp8_wgmma fp8_tcgen05 fp8_mma_sync)
endif()

# FlashAttention-3's Hopper forward: upstream's instantiation units for head dims 64 and 128 in
# bf16 and fp16, unmodified, with its own switches turning off what `fa3.cpp` never reaches.
# Neither MSVC (its TMA parameters, as above) nor ptxas on an arm64 host (signal 11, cr-200)
# builds them.
if(MSVC OR CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(fa3 OFF)
else()
    set(fa3 ON)
endif()
option(COZY_FA3 "FlashAttention-3's forward (_fa3)" ${fa3})
if(COZY_FA3)
# FlashAttention-3 (v2.8.3.post1) and the CUTLASS its submodule pins.
pinned(FLASH_ATTENTION_DIR Dao-AILab/flash-attention a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2
       e06c229e24ad61bcb31eea9446a5e6484cd8b3b5d10d90b0f8f48cc2b0d4c431)
pinned(FA3_CUTLASS_DIR NVIDIA/cutlass dc4817921edda44a549197ff3a9dcf5df0636e7b
       f2a3a9df5e6f010c8b02716aa2644a6f071827fafa606fac5f5241cab6a1ab56)
set(FA3 "${FLASH_ATTENTION_DIR}/hopper")
add_library(fa3_units OBJECT "${FA3}/flash_prepare_scheduler.cu"
            "${FA3}/instantiations/flash_fwd_hdim64_bf16_sm90.cu"
            "${FA3}/instantiations/flash_fwd_hdim64_fp16_sm90.cu"
            "${FA3}/instantiations/flash_fwd_hdim128_bf16_sm90.cu"
            "${FA3}/instantiations/flash_fwd_hdim128_fp16_sm90.cu")
set_target_properties(fa3_units PROPERTIES CUDA_ARCHITECTURES "90a-real")
target_include_directories(fa3_units PRIVATE "${FA3}" "${FA3_CUTLASS_DIR}/include")
target_compile_definitions(fa3_units PRIVATE CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
    CUTLASS_ENABLE_GDC_FOR_SM90 CUTLASS_DEBUG_TRACE_LEVEL=0 FLASHATTENTION_DISABLE_BACKWARD
    FLASHATTENTION_DISABLE_SPLIT FLASHATTENTION_DISABLE_PAGEDKV FLASHATTENTION_DISABLE_APPENDKV
    FLASHATTENTION_DISABLE_LOCAL FLASHATTENTION_DISABLE_SOFTCAP FLASHATTENTION_DISABLE_PACKGQA
    FLASHATTENTION_DISABLE_FP8 FLASHATTENTION_DISABLE_VARLEN FLASHATTENTION_DISABLE_SM8x
    $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
# Upstream builds through torch's extension builder, which adds its own nvcc flags: the same here.
target_compile_options(fa3_units PRIVATE --use_fast_math --expt-relaxed-constexpr
                       -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
                       -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
                       $<$<PLATFORM_ID:Linux>:-Xcompiler=-fno-gnu-unique>
                       $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/Zc:__cplusplus>)
endif()

# One module: its binding, its units' objects, the static CUDA runtime.
function(kernel_module name binding)
    nanobind_add_module(${name} NB_STATIC STABLE_ABI LTO "csrc/${binding}")
    target_include_directories(${name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_options(${name} PRIVATE ${isolated})
    foreach(unit ${ARGN})
        target_sources(${name} PRIVATE $<TARGET_OBJECTS:${unit}>)
    endforeach()
    target_link_libraries(${name} PRIVATE CUDA::cudart_static ${CMAKE_DL_LIBS})
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Static libstdc++/libgcc keep the wheel at its manylinux floor: only glibc's versioned
        # symbols stay external. -Bsymbolic binds the module's CUTLASS instantiations to itself,
        # never to another library's of the same name (comfy-kitchen's) in the process. The module
        # exports PyInit_* alone: an exported (GNU-unique) libstdc++ symbol binds process-wide
        # and crashed torch imported after it (0.28.0's blocker).
        target_link_options(${name} PRIVATE -Wl,--build-id=none -Wl,-Bsymbolic
                            -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL
                            -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/modules.map)
        set_target_properties(${name} PROPERTIES SUFFIX ".abi3.so")
    endif()
    install(TARGETS ${name} LIBRARY DESTINATION cozy_runtime/_kernels)
endfunction()

kernel_module(_gemm gemm.cpp ${units})
target_include_directories(_gemm PRIVATE "${INT8}")
target_compile_definitions(_gemm PRIVATE $<$<BOOL:${COZY_TMA}>:COZY_TMA>)
if(COZY_FA3)
    kernel_module(_fa3 fa3.cpp fa3_units)
    target_include_directories(_fa3 PRIVATE "${FA3}" "${FA3_CUTLASS_DIR}/include")
endif()

# RMSNorm plus split-half RoPE (comfy-kitchen v0.2.31's, `csrc/norm/UPSTREAM.md`).
add_library(rms_rope OBJECT csrc/norm/rms_rope.cu)
set_target_properties(rms_rope PROPERTIES CUDA_ARCHITECTURES "${EVERY}")
target_compile_options(rms_rope PRIVATE --expt-extended-lambda --use_fast_math)
kernel_module(_norm norm.cpp rms_rope)
