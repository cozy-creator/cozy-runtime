// SPDX-License-Identifier: Apache-2.0
// cuBLASLt as ggml-cuda's `common.cuh` names it, declared only (`cublas_v2.h` here says why).
#pragma once

#include "cublas_v2.h"

typedef struct cublasLtContext* cublasLtHandle_t;
cublasStatus_t cublasLtCreate(cublasLtHandle_t* handle);
