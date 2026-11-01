// SPDX-License-Identifier: Apache-2.0
// Prepared, caller-buffer FP16 biased linears. Query is outside compute; execute never
// selects an algorithm, allocates workspace, or retries. See research/varena-vae-dense-contract.md.
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace {
uint64_t process() {
#ifdef _WIN32
    return static_cast<uint64_t>(_getpid());
#else
    return static_cast<uint64_t>(getpid());
#endif
}

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
struct VendorError : std::runtime_error {
    const char* domain;
    int status;
    VendorError(const char* domain_, int status_, const std::string& message)
        : std::runtime_error(message), domain(domain_), status(status_) {}
};

void cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw VendorError("cuda", status, std::string("prepared dense CUDA: ") +
                                               cudaGetErrorString(status));
}
void driver(CUresult status) {
    if (status != CUDA_SUCCESS)
        throw VendorError("driver", status, "prepared dense CUDA context error " +
                                            std::to_string(status));
}
void blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw VendorError("cublas", status, status == CUBLAS_STATUS_ALLOC_FAILED
            ? "prepared dense CUBLAS_STATUS_ALLOC_FAILED"
            : "prepared dense cuBLASLt error " + std::to_string(status));
}
void not_capturing(uintptr_t stream) {
    cudaStreamCaptureStatus status;
    cuda(cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(stream), &status));
    require(status == cudaStreamCaptureStatusNone, "prepared dense initial provider is eager only");
}

struct Context {
    const uint64_t pid = process();
    const int device;
    CUcontext context = nullptr;
    cublasLtHandle_t handle = nullptr;
    size_t plans = 0;
    int sm_count = 0;
    int sm_major = 0;

    Context(int device_, uintptr_t stream) : device(device_) {
        int actual = -1;
        cuda(cudaGetDevice(&actual));
        require(actual == device, "prepared dense requires the caller's current device");
        driver(cuCtxGetCurrent(&context));
        require(context != nullptr, "prepared dense requires an initialized caller context");
        not_capturing(stream);
        cuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
        cuda(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
        blas(cublasLtCreate(&handle));
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    void current(uintptr_t stream) const {
        require(pid == process(), "prepared dense context belongs to another process");
        require(handle != nullptr, "prepared dense context is closed");
        CUcontext actual = nullptr;
        driver(cuCtxGetCurrent(&actual));
        require(actual == context, "prepared dense CUDA context changed");
        not_capturing(stream);
    }
    void close(uintptr_t stream) {
        require(pid == process(), "prepared dense context belongs to another process");
        if (!handle) return;
        current(stream);
        require(plans == 0, "prepared dense context still has retained plans");
        // cublasLtDestroy synchronizes. Runtime calls this only at the device owner's
        // already-drained final teardown, never per projection or variant.
        blas(cublasLtDestroy(handle));
        handle = nullptr;
    }
    ~Context() {
        if (!handle || pid != process()) return;
        // A Python owner may be released with another device current. Restore the
        // caller's context; if ours was already destroyed, its device resources are gone.
        if (cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
        cublasLtDestroy(handle);
        CUcontext restored = nullptr;
        cuCtxPopCurrent(&restored);
    }
};

struct Range {
    uintptr_t pointer;
    size_t size;
};
Range range(uintptr_t pointer, size_t capacity, size_t size) {
    require(capacity >= size, "prepared dense caller buffer is too small");
    if (size) {
        require(pointer != 0 && pointer % 256 == 0,
                "prepared dense caller buffer must be 256-byte aligned");
        require(size <= std::numeric_limits<uintptr_t>::max() - pointer,
                "prepared dense caller address range overflows");
    }
    return {pointer, size};
}
bool overlaps(Range a, Range b) {
    return a.size && b.size && a.pointer < b.pointer + b.size && b.pointer < a.pointer + a.size;
}
size_t matrix_bytes(int rows, int columns) {
    require(rows > 1 && columns > 1, "prepared dense dimensions must exceed one");
    const auto elements = static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns);
    require(elements <= std::numeric_limits<size_t>::max() / 2,
            "prepared dense matrix byte size overflows");
    return static_cast<size_t>(elements) * 2;
}

template <typename T>
void attribute(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t key, T value) {
    blas(cublasLtMatmulDescSetAttribute(desc, key, &value, sizeof(value)));
}
template <typename T>
void preference(cublasLtMatmulPreference_t desc, cublasLtMatmulPreferenceAttributes_t key, T value) {
    blas(cublasLtMatmulPreferenceSetAttribute(desc, key, &value, sizeof(value)));
}

struct Plan {
    const std::shared_ptr<Context> owner;
    const int rows, inputs, outputs;
    const bool half_accumulation;
    const int reduction, sm_target;
    const uintptr_t bias;
    const size_t bias_capacity, workspace_limit;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, result = nullptr;
    cublasLtMatmulPreference_t prefs = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t heuristic_workspace = 0, checked_workspace = 0, workspace = 0;
    const size_t x_bytes, weight_bytes, output_bytes, bias_bytes;

    Plan(std::shared_ptr<Context> owner_, int rows_, int inputs_, int outputs_, uintptr_t bias_,
         size_t bias_capacity_, bool half_, int reduction_, int sm_target_, size_t limit_)
        : owner(std::move(owner_)), rows(rows_), inputs(inputs_), outputs(outputs_),
          half_accumulation(half_), reduction(reduction_), sm_target(sm_target_), bias(bias_),
          bias_capacity(bias_capacity_), workspace_limit(limit_),
          x_bytes(matrix_bytes(rows, inputs)), weight_bytes(matrix_bytes(outputs, inputs)),
          output_bytes(matrix_bytes(rows, outputs)), bias_bytes(static_cast<size_t>(outputs) * 2) {
        ++owner->plans;
    }
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    ~Plan() {
        // Reject inherited objects before even entering vendor metadata destruction.
        if (owner->pid == process()) {
            if (prefs) cublasLtMatmulPreferenceDestroy(prefs);
            if (result) cublasLtMatrixLayoutDestroy(result);
            if (b) cublasLtMatrixLayoutDestroy(b);
            if (a) cublasLtMatrixLayoutDestroy(a);
            if (operation) cublasLtMatmulDescDestroy(operation);
        }
        --owner->plans;
    }

    void prepare(uintptr_t stream) {
        owner->current(stream);
        range(bias, bias_capacity, bias_bytes);
        require(reduction >= 0 && reduction <= 2, "invalid prepared dense reduction policy");
        require(sm_target >= 0 && sm_target <= owner->sm_count, "invalid prepared dense SM target");
        require(!half_accumulation || owner->sm_major >= 7, "FP16 accumulation needs SM70+");
        const auto compute = half_accumulation ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
        const auto scalar = half_accumulation ? CUDA_R_16F : CUDA_R_32F;
        blas(cublasLtMatmulDescCreate(&operation, compute, scalar));
        attribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
        attribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);
        attribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_BIAS);
        attribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, reinterpret_cast<void*>(bias));
        if (sm_target) attribute(operation, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, sm_target);
        // Torch's contiguous row-major linear is Y.T = W @ X.T + bias. Physical
        // column-major A is [K,N] transposed; B is [K,M]; C and D share [N,M].
        blas(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, inputs, outputs, inputs));
        blas(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, inputs, rows, inputs));
        blas(cublasLtMatrixLayoutCreate(&result, CUDA_R_16F, outputs, rows, outputs));
        blas(cublasLtMatmulPreferenceCreate(&prefs));
        preference(prefs, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_limit);
        for (auto key : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                         CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                         CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                         CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
            preference(prefs, key, uint32_t{256});
        // 0: allow reduced precision + split-K (vendor default); 1: disallow reduced
        // precision, allow split-K; 2: disallow both. The fourth combination is invalid.
        if (reduction) {
            uint32_t mask = CUBLASLT_REDUCTION_SCHEME_NONE;
            if (reduction == 1) mask |= CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
            preference(prefs, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, mask);
        }
        cublasLtMatmulHeuristicResult_t found{};
        int count = 0;
        blas(cublasLtMatmulAlgoGetHeuristic(owner->handle, operation, a, b, result, result,
                                           prefs, 1, &found, &count));
        if (count != 1) blas(CUBLAS_STATUS_NOT_SUPPORTED);
        blas(found.state);
        algorithm = found.algo;
        heuristic_workspace = found.workspaceSize;
        cublasLtMatmulHeuristicResult_t checked{};
        blas(cublasLtMatmulAlgoCheck(owner->handle, operation, a, b, result, result,
                                    &algorithm, &checked));
        blas(checked.state);
        checked_workspace = checked.workspaceSize;
        workspace = std::max(heuristic_workspace, checked_workspace);
        require(workspace <= workspace_limit, "selected dense workspace exceeds effective policy");
    }

    void validate(uintptr_t x, size_t x_capacity, uintptr_t weight, size_t weight_capacity,
                uintptr_t output, size_t output_capacity, uintptr_t scratch,
                size_t scratch_capacity, uintptr_t stream) const {
        owner->current(stream);
        std::array<Range, 5> ranges = {
            range(x, x_capacity, x_bytes), range(weight, weight_capacity, weight_bytes),
            range(output, output_capacity, output_bytes), range(bias, bias_capacity, bias_bytes),
            range(scratch, scratch_capacity, workspace)};
        for (size_t i = 0; i != ranges.size(); ++i)
            for (size_t j = i + 1; j != ranges.size(); ++j)
                require(!overlaps(ranges[i], ranges[j]), "prepared dense operands overlap");
    }

    void execute(uintptr_t x, size_t x_capacity, uintptr_t weight, size_t weight_capacity,
                 uintptr_t output, size_t output_capacity, uintptr_t scratch,
                 size_t scratch_capacity, uintptr_t stream) const {
        validate(x, x_capacity, weight, weight_capacity, output, output_capacity, scratch,
                 scratch_capacity, stream);
        const float one = 1.0f, zero = 0.0f;
        const uint16_t half_one = 0x3c00, half_zero = 0;
        const void* alpha = half_accumulation ? static_cast<const void*>(&half_one) : &one;
        const void* beta = half_accumulation ? static_cast<const void*>(&half_zero) : &zero;
        // No heuristic, descriptor mutation, allocation or fallback here. The caller
        // retains all owners through completion; every vendor failure preserves its status.
        blas(cublasLtMatmul(owner->handle, operation, alpha, reinterpret_cast<void*>(weight),
                              a, reinterpret_cast<void*>(x), b, beta,
                              reinterpret_cast<void*>(output), result,
                              reinterpret_cast<void*>(output), result, &algorithm,
                              workspace ? reinterpret_cast<void*>(scratch) : nullptr, workspace,
                              reinterpret_cast<cudaStream_t>(stream)));
    }

    nb::dict info() const {
        require(owner->pid == process(), "prepared dense plan belongs to another process");
        nb::dict info;
        info["device"] = owner->device;
        info["rows"] = rows; info["inputs"] = inputs; info["outputs"] = outputs;
        info["workspace_bytes"] = workspace;
        info["heuristic_workspace_bytes"] = heuristic_workspace;
        info["checked_workspace_bytes"] = checked_workspace;
        info["workspace_limit_bytes"] = workspace_limit;
        info["alignment"] = 256;
        info["bias_address"] = bias;
        info["half_accumulation"] = half_accumulation;
        info["reduction"] = reduction; info["sm_target"] = sm_target;
        info["library_version"] = cublasLtGetVersion();
        info["header_version"] = CUBLAS_VERSION;
        nb::dict config;
        for (auto item : {std::pair{"id", CUBLASLT_ALGO_CONFIG_ID},
                          std::pair{"tile", CUBLASLT_ALGO_CONFIG_TILE_ID},
                          std::pair{"split_k", CUBLASLT_ALGO_CONFIG_SPLITK_NUM},
                          std::pair{"reduction", CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME},
                          std::pair{"swizzle", CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING},
                          std::pair{"custom", CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION},
                          std::pair{"stages", CUBLASLT_ALGO_CONFIG_STAGES_ID}}) {
            int value = 0;
            size_t written = 0;
            blas(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, item.second, &value,
                                                      sizeof(value), &written));
            require(written == sizeof(value), "unexpected dense algorithm metadata size");
            config[nb::str(item.first)] = value;
        }
        info["algorithm"] = config;
        return info;
    }
};

std::shared_ptr<Plan> query(std::shared_ptr<Context> context, int rows, int inputs, int outputs,
                            uintptr_t bias, size_t bias_capacity, bool half_accumulation,
                            int reduction, int sm_target, size_t workspace_limit, uintptr_t stream) {
    require(context != nullptr, "prepared dense needs a context");
    context->current(stream);
    // Build the owner first: partial preparation failure still destroys every descriptor.
    auto plan = std::make_shared<Plan>(std::move(context), rows, inputs, outputs, bias,
                                        bias_capacity, half_accumulation, reduction, sm_target,
                                        workspace_limit);
    plan->prepare(stream);
    return plan;
}
}  // namespace

NB_MODULE(_dense, m) {
    m.attr("API") = 1;
    m.attr("NOT_SUPPORTED") = static_cast<int>(CUBLAS_STATUS_NOT_SUPPORTED);
    nb::exception<VendorError> error(m, "Error", PyExc_RuntimeError);
    nb::register_exception_translator(
        [](const std::exception_ptr& pointer, void* payload) {
            try {
                std::rethrow_exception(pointer);
            } catch (const VendorError& e) {
                auto type = nb::borrow<nb::object>(static_cast<PyObject*>(payload));
                auto value = type(e.what());
                value.attr("domain") = e.domain;
                value.attr("status") = e.status;
                PyErr_SetObject(type.ptr(), value.ptr());
            }
        }, error.ptr());
    nb::class_<Context>(m, "Context")
        .def(nb::init<int, uintptr_t>(), "device"_a, "stream"_a)
        .def("close", &Context::close, "stream"_a)
        .def("query", &query, "rows"_a, "inputs"_a, "outputs"_a, "bias"_a,
             "bias_capacity"_a, "half_accumulation"_a, "reduction"_a, "sm_target"_a,
             "workspace_limit"_a, "stream"_a);
    nb::class_<Plan>(m, "Plan")
        .def("info", &Plan::info)
        .def("validate", &Plan::validate, "x"_a, "x_capacity"_a, "weight"_a,
             "weight_capacity"_a, "output"_a, "output_capacity"_a, "workspace"_a,
             "workspace_capacity"_a, "stream"_a)
        .def("execute", &Plan::execute, "x"_a, "x_capacity"_a, "weight"_a,
             "weight_capacity"_a, "output"_a, "output_capacity"_a, "workspace"_a,
             "workspace_capacity"_a, "stream"_a);
}
