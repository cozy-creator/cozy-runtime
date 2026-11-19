// SPDX-License-Identifier: Apache-2.0
// What llama.cpp's ggml-cuda matmul sources (`ggml/`, b12466, MIT, unmodified) take from the rest of ggml, for
// `_kernels._ggml`: the device table, one context per device whose stream is the caller's and whose pool is the
// caller's allocator, ggml's tensor arithmetic for the types `_ggml` runs, and failures as C++ exceptions. No ggml
// graph, allocator or backend: ggml-cuda.cu and ggml.c are not built.
#include "shim.h"

#include "ggml/src/ggml-cuda/graph.cuh"

#include <cstdarg>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

std::string format(const char* fmt, va_list args) {
    char line[1024];
    std::vsnprintf(line, sizeof line, fmt, args);
    return line;
}

thread_local cozy_ggml::Allocate* current = nullptr;

// Temporaries (quantized activations, stream-K partials) from the caller's stream-ordered allocator, held until
// the call returns; nothing is freed early, so a captured graph keeps its addresses.
struct CallerPool : ggml_cuda_pool {
    void* alloc(size_t size, size_t* actual_size) override {
        if (current == nullptr) throw std::runtime_error("ggml: a temporary outside a call");
        void* ptr = current->fn(size, current->context);
        if (ptr == nullptr) throw std::bad_alloc();
        *actual_size = size;
        return ptr;
    }
    void free(void*, size_t) override {}
};

}  // namespace

extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const std::string what = format(fmt, args);
    va_end(args);
    throw std::runtime_error("ggml: " + what + " (" + file + ":" + std::to_string(line) + ")");
}

extern "C" void ggml_log_internal(enum ggml_log_level level, const char* fmt, ...) {
    if (level != GGML_LOG_LEVEL_ERROR) return;
    va_list args;
    va_start(args, fmt);
    std::fputs(("ggml: " + format(fmt, args)).c_str(), stderr);
    va_end(args);
}

// ------------------------------------------------------------------- tensors (ggml.c's arithmetic)

extern "C" int64_t ggml_blck_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16: case GGML_TYPE_I32: return 1;
        case GGML_TYPE_Q8_0: return QK8_0;
        case GGML_TYPE_Q8_1: return QK8_1;
        case GGML_TYPE_Q5_0: return QK5_0;
        case GGML_TYPE_Q5_1: return QK5_1;
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: return QK_K;
        default: GGML_ABORT("type %d is not one _ggml runs", int(type));
    }
}

extern "C" size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_I32: return 4;
        case GGML_TYPE_F16: case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
        case GGML_TYPE_Q8_1: return sizeof(block_q8_1);
        case GGML_TYPE_Q5_0: return sizeof(block_q5_0);
        case GGML_TYPE_Q5_1: return sizeof(block_q5_1);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        default: GGML_ABORT("type %d is not one _ggml runs", int(type));
    }
}

extern "C" bool ggml_is_quantized(enum ggml_type type) { return ggml_blck_size(type) > 1; }

extern "C" size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    return ggml_type_size(type) * ne / ggml_blck_size(type);
}

extern "C" size_t ggml_element_size(const struct ggml_tensor* tensor) { return ggml_type_size(tensor->type); }

extern "C" int64_t ggml_nelements(const struct ggml_tensor* tensor) {
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

extern "C" size_t ggml_nbytes(const struct ggml_tensor* tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) return 0;
    }
    size_t nbytes;
    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
    } else {
        nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
    }
    return nbytes;
}

static bool contiguous_n(const struct ggml_tensor* tensor, int n) {
    size_t next_nb = ggml_type_size(tensor->type);
    if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) return false;
    next_nb *= tensor->ne[0] / ggml_blck_size(tensor->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1) {
            if (i > n) {
                if (tensor->nb[i] != next_nb) return false;
                next_nb *= tensor->ne[i];
            } else {
                next_nb = tensor->ne[i] * tensor->nb[i];
            }
        }
    }
    return true;
}

extern "C" bool ggml_is_contiguous(const struct ggml_tensor* tensor) { return contiguous_n(tensor, 0); }

extern "C" bool ggml_is_contiguously_allocated(const struct ggml_tensor* tensor) {
    return ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type) / ggml_blck_size(tensor->type);
}

extern "C" bool ggml_is_matrix(const struct ggml_tensor* tensor) { return tensor->ne[2] == 1 && tensor->ne[3] == 1; }

extern "C" bool ggml_are_same_shape(const struct ggml_tensor* a, const struct ggml_tensor* b) {
    return a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1] && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3];
}

extern "C" bool ggml_are_same_stride(const struct ggml_tensor* a, const struct ggml_tensor* b) {
    return a->nb[0] == b->nb[0] && a->nb[1] == b->nb[1] && a->nb[2] == b->nb[2] && a->nb[3] == b->nb[3];
}

extern "C" enum ggml_prec ggml_prec_get_src(const struct ggml_tensor* a, int idx) {
    if ((a->op == GGML_OP_MUL_MAT || a->op == GGML_OP_MUL_MAT_ID) && idx == 1) {
        return (enum ggml_prec) a->op_params[2 + idx];
    }
    return GGML_PREC_UNDEFINED;
}

// The caller's buffers are weights or inputs, never a compute buffer whose padding ggml clears.
extern "C" enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) {
    return GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

extern "C" size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t, const struct ggml_tensor* tensor) {
    return ggml_nbytes(tensor);
}

// ------------------------------------------------------------------- CUDA (ggml-cuda.cu's, NVIDIA's part)

[[noreturn]] void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line, const char* msg) {
    throw std::runtime_error(std::string("ggml: CUDA error ") + msg + " in " + func + " at " + file + ":" +
                             std::to_string(line) + ": " + stmt);
}

void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (device != current_device) CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

bool ggml_cuda_pool_debug_enabled() { return false; }

const ggml_cuda_device_info& ggml_cuda_info() {
    static const ggml_cuda_device_info info = [] {
        ggml_cuda_device_info info = {};
        if (cudaGetDeviceCount(&info.physical_device_count) != cudaSuccess) return info;
        info.device_count = std::min(info.physical_device_count, GGML_CUDA_MAX_DEVICES);
        for (int id = 0; id < info.device_count; ++id) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
            auto& d = info.devices[id];
            d.physical_device = id;
            d.physical_share_count = 1;
            d.integrated = false;
            d.nsm = prop.multiProcessorCount;
            d.smpb = prop.sharedMemPerBlock;
            d.smpbo = prop.sharedMemPerBlockOptin;
            d.warp_size = prop.warpSize;
            d.total_vram = prop.totalGlobalMem;
            d.cc = 100 * prop.major + 10 * prop.minor;
            int coop = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, id));
            d.supports_cooperative_launch = coop != 0;
        }
        return info;
    }();
    return info;
}

#if defined(GGML_CUDA_USE_PDL)
// ggml-cuda.cu's rule: a kernel launches with programmatic dependent launch when its loaded code is PTX 9.0 or newer
// (its device-side PDL waits are only there).
bool ggml_cuda_kernel_can_use_pdl(const void* kernel) {
    static std::mutex lock;
    static std::map<std::pair<int, const void*>, bool> known;
    const std::pair<int, const void*> key{ggml_cuda_get_device(), kernel};
    const std::lock_guard<std::mutex> held(lock);
    const auto it = known.find(key);
    if (it != known.end()) return it->second;
    cudaFuncAttributes attr = {};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    return known[key] = attr.ptxVersion >= 90;
}
#endif

ggml_backend_cuda_context::ggml_backend_cuda_context(int device)
    : device(device),
      name("CUDA" + std::to_string(device)),
      concurrent_stream_context(std::make_unique<ggml_cuda_stream_context>()) {}

// The streams are the callers' (`cozy_ggml::Call`): nothing here owns one.
ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(ggml_backend_cuda_context&, int,
                                                                               int) {
    return std::make_unique<CallerPool>();
}

namespace cozy_ggml {

Call::Call(int device, cudaStream_t stream, Allocate* allocate) {
    static std::mutex lock;
    static ggml_backend_cuda_context* contexts[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= ggml_cuda_info().device_count) throw std::runtime_error("ggml: no such device");
    {
        const std::lock_guard<std::mutex> held(lock);
        if (contexts[device] == nullptr) contexts[device] = new ggml_backend_cuda_context(device);
    }
    ctx_ = contexts[device];
    ctx_->curr_stream_no = 0;
    ctx_->streams[device][0] = stream;
    current = allocate;
}

Call::~Call() {
    ctx_->streams[ctx_->device][0] = nullptr;
    current = nullptr;
}

}  // namespace cozy_ggml
