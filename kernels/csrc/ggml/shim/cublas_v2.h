// SPDX-License-Identifier: Apache-2.0
// What ggml-cuda's `common.cuh` names of cuBLAS, declared only: `_ggml` builds its matmuls without cuBLAS and
// never calls one of these (its context's cuBLAS handles stay unused), so the module links no cuBLAS.
#pragma once

#define CUBLAS_VER_MAJOR 13
#define CUBLAS_VER_MINOR 0

typedef struct cublasContext* cublasHandle_t;
typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
} cublasStatus_t;
typedef enum { CUBLAS_DEFAULT_MATH = 0, CUBLAS_TENSOR_OP_MATH = 1, CUBLAS_TF32_TENSOR_OP_MATH = 3 } cublasMath_t;

const char* cublasGetStatusString(cublasStatus_t status);
cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* workspace, size_t bytes);
