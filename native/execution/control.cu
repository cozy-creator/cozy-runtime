// Original CUDA graph control, built ahead of time. No Python or host nodes in a loop body.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <new>
#include <vector>

template<class Body> static int on_device(int device, Body body) {
    int previous = 0;
    auto selected = cudaGetDevice(&previous);
    if (selected != cudaSuccess) return selected;
    if (previous != device) {
        selected = cudaSetDevice(device);
        if (selected != cudaSuccess) return selected;
    }
    int result;
    try { result = body(); }
    catch (const std::bad_alloc &) { result = cudaErrorMemoryAllocation; }
    catch (...) { result = cudaErrorUnknown; }
    const auto restored = previous == device ? cudaSuccess : cudaSetDevice(previous);
    return result == cudaSuccess ? restored : result;
}

struct Control { unsigned completed; unsigned cancel; };
struct Snapshot { unsigned completed; unsigned cancel; };
struct Repeat {
    int device = 0;
    unsigned steps = 0;
    unsigned nodes = 0;
    unsigned validated_nodes = 0;
    char inspection[256]{};
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaStream_t work = nullptr;
    cudaStream_t control_stream = nullptr;
    cudaEvent_t done = nullptr;
    cudaEvent_t producer_ready = nullptr;
    cudaEvent_t initialized = nullptr;
    cudaEvent_t sampled = nullptr;
    Control *control = nullptr;
    Snapshot *device_snapshot = nullptr;
    Snapshot *host_snapshot = nullptr;
    int64_t *index = nullptr;
    Snapshot observed{};
    bool launched = false;
    bool pending_sample = false;
};

__global__ void advance(Control *state, int64_t *index, unsigned steps,
                        cudaGraphConditionalHandle condition) {
    const unsigned completed = atomicAdd(&state->completed, 1u) + 1u;
    *index = static_cast<int64_t>(completed);
    const unsigned canceled = atomicAdd(&state->cancel, 0u);
    cudaGraphSetConditional(condition, completed < steps && !canceled);
}

__global__ void cancel_repeat(Control *state) { atomicExch(&state->cancel, 1u); }

__global__ void snapshot(Control *state, Snapshot *out) {
    out->completed = atomicAdd(&state->completed, 0u);
    out->cancel = atomicAdd(&state->cancel, 0u);
}

// Preserve only the first refusal, without pointers or another lifetime/ownership mechanism.
static cudaError_t ineligible(Repeat *run, const char *stage, const char *reason,
                             int type, unsigned depth, size_t count, unsigned total,
                             int copy_kind = -1, int pointer_type = -1, int pointer_device = -1) {
    if (!run->inspection[0]) {
        std::snprintf(run->inspection, sizeof(run->inspection),
                      "stage=%s reason=%s type=%d depth=%u local=%zu total=%u "
                      "copy_kind=%d pointer_type=%d pointer_device=%d expected_device=%d",
                      stage, reason, type, depth, count, total, copy_kind,
                      pointer_type, pointer_device, run->device);
    }
    return static_cast<cudaError_t>(-100);
}

// Reject every operation whose ownership/side effects cannot be represented by a resident body.
// cudaGraphAddChildGraphNode clones the accepted definition, never an executable handle.
static cudaError_t inspect(cudaGraph_t graph, unsigned &total, int device,
                           Repeat *run, const char *stage, unsigned depth = 0) {
    if (depth > 32) return ineligible(run, stage, "depth_limit", -1, depth, 0, total);
    size_t count = 0;
    auto error = cudaGraphGetNodes(graph, nullptr, &count);
    if (error != cudaSuccess) return error;
    if (count > 1000000 || total > 1000000 - count)
        return ineligible(run, stage, "node_limit", -1, depth, count, total);
    std::vector<cudaGraphNode_t> nodes(count);
    error = cudaGraphGetNodes(graph, nodes.data(), &count);
    if (error != cudaSuccess) return error;
    total += static_cast<unsigned>(count);
    for (auto node : nodes) {
        cudaGraphNodeType type;
        error = cudaGraphNodeGetType(node, &type);
        if (error != cudaSuccess) return error;
        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child;
            error = cudaGraphChildGraphNodeGetGraph(node, &child);
            if (error == cudaSuccess) error = inspect(child, total, device, run, stage, depth + 1);
            if (error != cudaSuccess) return error;
        } else if (type == cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms parameters{};
            error = cudaGraphMemcpyNodeGetParams(node, &parameters);
            if (error != cudaSuccess) return error;
            if (parameters.srcArray || parameters.dstArray || !parameters.srcPtr.ptr || !parameters.dstPtr.ptr
                || (parameters.kind != cudaMemcpyDeviceToDevice && parameters.kind != cudaMemcpyDefault))
                return ineligible(run, stage, "memcpy_layout_or_kind", type, depth, count, total,
                                  parameters.kind);
            unsigned endpoint = 0;
            for (const void *pointer : {parameters.srcPtr.ptr, parameters.dstPtr.ptr}) {
                cudaPointerAttributes attributes{};
                error = cudaPointerGetAttributes(&attributes, pointer);
                if (error != cudaSuccess) return error;
                if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
                    return ineligible(run, stage, endpoint ? "memcpy_destination" : "memcpy_source",
                                      type, depth, count, total, parameters.kind,
                                      attributes.type, attributes.device);
                ++endpoint;
            }
        } else if (type == cudaGraphNodeTypeMemset) {
            cudaMemsetParams parameters{};
            error = cudaGraphMemsetNodeGetParams(node, &parameters);
            if (error != cudaSuccess) return error;
            cudaPointerAttributes attributes{};
            error = cudaPointerGetAttributes(&attributes, parameters.dst);
            if (error != cudaSuccess) return error;
            if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
                return ineligible(run, stage, "memset_destination", type, depth, count, total,
                                  -1, attributes.type, attributes.device);
        } else if (type != cudaGraphNodeTypeKernel && type != cudaGraphNodeTypeEmpty
                   && type != cudaGraphNodeTypeMemcpy && type != cudaGraphNodeTypeMemset) {
            return ineligible(run, stage, "node_type", type, depth, count, total);
        }
    }
    return cudaSuccess;
}

static cudaError_t cleanup(Repeat *run) {
    // Stop at an uncertain completion. The caller retains this object and all source owners.
    cudaError_t error;
    if (run->work && (error = cudaStreamSynchronize(run->work)) != cudaSuccess) return error;
    if (run->control_stream && (error = cudaStreamSynchronize(run->control_stream)) != cudaSuccess) return error;
    if (run->executable && (error = cudaGraphExecDestroy(run->executable)) != cudaSuccess) return error;
    run->executable = nullptr;
    if (run->graph && (error = cudaGraphDestroy(run->graph)) != cudaSuccess) return error;
    run->graph = nullptr;
    if (run->control && (error = cudaFree(run->control)) != cudaSuccess) return error;
    run->control = nullptr;
    if (run->device_snapshot && (error = cudaFree(run->device_snapshot)) != cudaSuccess) return error;
    run->device_snapshot = nullptr;
    if (run->host_snapshot && (error = cudaFreeHost(run->host_snapshot)) != cudaSuccess) return error;
    run->host_snapshot = nullptr;
    if (run->done && (error = cudaEventDestroy(run->done)) != cudaSuccess) return error;
    run->done = nullptr;
    if (run->producer_ready && (error = cudaEventDestroy(run->producer_ready)) != cudaSuccess) return error;
    run->producer_ready = nullptr;
    if (run->initialized && (error = cudaEventDestroy(run->initialized)) != cudaSuccess) return error;
    run->initialized = nullptr;
    if (run->sampled && (error = cudaEventDestroy(run->sampled)) != cudaSuccess) return error;
    run->sampled = nullptr;
    if (run->work && (error = cudaStreamDestroy(run->work)) != cudaSuccess) return error;
    run->work = nullptr;
    if (run->control_stream && (error = cudaStreamDestroy(run->control_stream)) != cudaSuccess) return error;
    run->control_stream = nullptr;
    return cudaSuccess;
}

static int create(uintptr_t source, uintptr_t index, int device, unsigned steps, Repeat **out) {
    *out = nullptr;
    if (!source || !index || !steps) return cudaErrorInvalidValue;
    cudaError_t error;
    auto run = new (std::nothrow) Repeat;
    if (!run) return cudaErrorMemoryAllocation;
    run->device = device; run->steps = steps; run->index = reinterpret_cast<int64_t *>(index);
    // Return ownership even on failure so Rust can retain the source after uncertain cleanup.
    *out = run;
    auto checked = [&](cudaError_t result) { return result; };
#define CHECK(call) do { error = checked(call); if (error != cudaSuccess) return error; } while (0)
    CHECK(inspect(reinterpret_cast<cudaGraph_t>(source), run->nodes, device, run, "source"));
    if (!run->nodes) return -101;
    CHECK(cudaStreamCreateWithFlags(&run->work, cudaStreamNonBlocking));
    int lowest_priority, highest_priority;
    CHECK(cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority));
    CHECK(cudaStreamCreateWithPriority(&run->control_stream, cudaStreamNonBlocking, highest_priority));
    CHECK(cudaEventCreateWithFlags(&run->done, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&run->producer_ready, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&run->initialized, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&run->sampled, cudaEventDisableTiming));
    // A first control-kernel launch must never trigger lazy loading while the repeat runs.
    cudaFuncAttributes attributes;
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(advance)));
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(cancel_repeat)));
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(snapshot)));
    CHECK(cudaMalloc(&run->control, sizeof(Control)));
    CHECK(cudaMalloc(&run->device_snapshot, sizeof(Snapshot)));
    CHECK(cudaMallocHost(&run->host_snapshot, sizeof(Snapshot)));
    CHECK(cudaGraphCreate(&run->graph, 0));
    cudaGraphConditionalHandle condition;
    CHECK(cudaGraphConditionalHandleCreate(&condition, run->graph, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams conditional{};
    conditional.type = cudaGraphNodeTypeConditional;
    conditional.conditional.handle = condition;
    conditional.conditional.type = cudaGraphCondTypeWhile;
    conditional.conditional.size = 1;
    cudaGraphNode_t loop;
    CHECK(cudaGraphAddNode(&loop, run->graph, nullptr, nullptr, 0, &conditional));
    auto body = conditional.conditional.phGraph_out[0];
    cudaGraphNode_t compute;
    CHECK(cudaGraphAddChildGraphNode(&compute, body, nullptr, 0, reinterpret_cast<cudaGraph_t>(source)));
    void *args[] = {&run->control, &run->index, &run->steps, &condition};
    cudaKernelNodeParams kernel{};
    kernel.func = reinterpret_cast<void *>(advance);
    kernel.gridDim = dim3(1); kernel.blockDim = dim3(1); kernel.kernelParams = args;
    cudaGraphNode_t boundary;
    CHECK(cudaGraphAddKernelNode(&boundary, body, &compute, 1, &kernel));
    // Inspect the actual assembled body, including the cloned source child definition.
    CHECK(inspect(body, run->validated_nodes, device, run, "assembled"));
    CHECK(cudaGraphInstantiate(&run->executable, run->graph, 0));
    CHECK(cudaGraphUpload(run->executable, run->work));
    CHECK(cudaStreamSynchronize(run->work));
    return cudaSuccess;
#undef CHECK
}

extern "C" int cozy_repeat_create(uintptr_t source, uintptr_t index, int device,
                                  unsigned steps, Repeat **out) {
    *out = nullptr;
    return on_device(device, [&] { return create(source, index, device, steps, out); });
}

static int arm(Repeat *run, uintptr_t producer) {
    cudaError_t error;
    cudaStreamCaptureStatus capturing;
    error = cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(producer), &capturing);
    if (error != cudaSuccess) return error;
    if (capturing != cudaStreamCaptureStatusNone) return -102;
    if (run->launched) {
        error = cudaEventQuery(run->done); if (error != cudaSuccess) return error;
        error = cudaStreamSynchronize(run->control_stream); if (error != cudaSuccess) return error;
    }
    // Record on the caller's OS thread: a per-thread default stream cannot be moved to a worker.
    return cudaEventRecord(run->producer_ready, reinterpret_cast<cudaStream_t>(producer));
}

static int launch_armed(Repeat *run) {
    // The caller-ready event was already recorded before the native read worker was started.
    cudaError_t error = cudaStreamWaitEvent(run->work, run->producer_ready, 0);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(run->control, 0, sizeof(Control), run->work);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(run->index, 0, sizeof(int64_t), run->work);
    if (error != cudaSuccess) return error;
    error = cudaEventRecord(run->initialized, run->work);
    if (error != cudaSuccess) return error;
    error = cudaStreamWaitEvent(run->control_stream, run->initialized, 0);
    if (error != cudaSuccess) return error;
    run->observed = {}; run->pending_sample = false;
    run->launched = true;
    error = cudaGraphLaunch(run->executable, run->work);
    if (error != cudaSuccess) return error;
    error = cudaEventRecord(run->done, run->work);
    if (error != cudaSuccess) return error;
    return cudaSuccess;
}

static int launch(Repeat *run, uintptr_t producer) {
    int status = arm(run, producer);
    if (status != cudaSuccess) return status;
    status = launch_armed(run);
    if (status != cudaSuccess) return status;
    // Pure resident submit is immediate. Composed async loads instead require an explicit join.
    return cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(producer), run->done, 0);
}

static int ready(Repeat *run, int *done) {
    cudaError_t error;
    if (!run->launched) return cudaErrorInvalidValue;
    error = cudaEventQuery(run->done);
    *done = error == cudaSuccess;
    return error == cudaErrorNotReady ? cudaSuccess : error;
}

static int poll(Repeat *run, unsigned *completed, unsigned *canceled, int *done) {
    cudaError_t error;
    if (!run->launched) return cudaErrorInvalidValue;
    auto finished = cudaEventQuery(run->done);
    if (finished != cudaSuccess && finished != cudaErrorNotReady) return finished;
    *done = finished == cudaSuccess;
    if (run->pending_sample) {
        error = cudaEventQuery(run->sampled);
        if (error == cudaSuccess) { run->observed = *run->host_snapshot; run->pending_sample = false; }
        else if (error != cudaErrorNotReady) return error;
    }
    if (!run->pending_sample) {
        snapshot<<<1, 1, 0, run->control_stream>>>(run->control, run->device_snapshot);
        error = cudaGetLastError(); if (error != cudaSuccess) return error;
        error = cudaMemcpyAsync(run->host_snapshot, run->device_snapshot, sizeof(Snapshot), cudaMemcpyDeviceToHost, run->control_stream);
        if (error != cudaSuccess) return error;
        error = cudaEventRecord(run->sampled, run->control_stream); if (error != cudaSuccess) return error;
        run->pending_sample = true;
    }
    *completed = run->observed.completed;
    *canceled = run->observed.cancel && run->observed.completed < run->steps;
    return cudaSuccess;
}

static int cancel(Repeat *run) {
    cancel_repeat<<<1, 1, 0, run->control_stream>>>(run->control);
    return cudaGetLastError();
}

static int wait(Repeat *run, unsigned *completed, unsigned *canceled) {
    cudaError_t error;
    if (!run->launched) return cudaErrorInvalidValue;
    error = cudaEventSynchronize(run->done); if (error != cudaSuccess) return error;
    error = cudaStreamSynchronize(run->control_stream); if (error != cudaSuccess) return error;
    Control state;
    error = cudaMemcpy(&state, run->control, sizeof(state), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return error;
    *completed = state.completed; *canceled = state.cancel && state.completed < run->steps;
    return cudaSuccess;
}

extern "C" unsigned cozy_repeat_nodes(Repeat *run) { return run->nodes; }
extern "C" unsigned cozy_repeat_validated_nodes(Repeat *run) { return run->validated_nodes; }

static int destroy(Repeat *run) {
    if (!run) return cudaSuccess;
    cudaError_t error;
    if (run->launched) {
        cancel_repeat<<<1, 1, 0, run->control_stream>>>(run->control);
        error = cudaGetLastError(); if (error != cudaSuccess) return error;
    }
    error = cleanup(run);
    if (error == cudaSuccess) run->launched = false;
    return error;
}

extern "C" int cozy_repeat_launch(Repeat *run, uintptr_t producer) {
    return on_device(run->device, [&] { return launch(run, producer); });
}
extern "C" int cozy_repeat_arm(Repeat *run, uintptr_t producer) {
    return on_device(run->device, [&] { return arm(run, producer); });
}
extern "C" int cozy_repeat_launch_armed(Repeat *run) {
    return on_device(run->device, [&] { return launch_armed(run); });
}
extern "C" int cozy_repeat_ready(Repeat *run, int *done) {
    return on_device(run->device, [&] { return ready(run, done); });
}
extern "C" int cozy_repeat_poll(Repeat *run, unsigned *completed, unsigned *canceled, int *done) {
    return on_device(run->device, [&] { return poll(run, completed, canceled, done); });
}
extern "C" int cozy_repeat_cancel(Repeat *run) {
    return on_device(run->device, [&] { return cancel(run); });
}
extern "C" int cozy_repeat_wait(Repeat *run, unsigned *completed, unsigned *canceled) {
    return on_device(run->device, [&] { return wait(run, completed, canceled); });
}
extern "C" const char *cozy_repeat_inspection(Repeat *run) { return run->inspection; }
extern "C" int cozy_repeat_destroy(Repeat *run) {
    if (!run) return cudaSuccess;
    const int code = on_device(run->device, [&] { return destroy(run); });
    if (code == cudaSuccess) delete run;
    return code;
}
