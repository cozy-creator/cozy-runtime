// Original CUDA graph control, built ahead of time. No Python or host nodes in a loop body.
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <new>
#include <vector>

template<class Body> static int switched(int device, Body body) {
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

// A failure reaches the caller as the return value. This module links its own cudart, so leaving
// it as this thread's last error (a refused create's out-of-memory) would make a later launch check
// here, such as destroy's, report it as its own.
template<class Body> static int on_device(int device, Body body) {
    const int result = switched(device, body);
    if (result != cudaSuccess) (void)cudaGetLastError();
    return result;
}

struct Control {
    unsigned completed, cancel, suffix_completed, body_entered, input_invalid;
    unsigned long long input_error;
};
struct Snapshot { unsigned completed, cancel, suffix_completed, body_entered, input_invalid; };
struct Repeat {
    int device = 0;
    unsigned steps = 0;
    unsigned nodes = 0;
    unsigned validated_nodes = 0;
    unsigned prefix_nodes = 0;
    unsigned suffix_nodes = 0;
    bool has_suffix = false;
    char inspection[256]{};
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaStream_t work = nullptr;
    cudaStream_t control_stream = nullptr;
    cudaEvent_t done = nullptr;
    cudaEvent_t producer_ready = nullptr;
    cudaEvent_t initialized = nullptr;
    cudaEvent_t sampled = nullptr;
    std::array<cudaEvent_t, 4> input_ready{};
    Control *control = nullptr;
    Snapshot *device_snapshot = nullptr;
    Snapshot *host_snapshot = nullptr;
    int64_t *index = nullptr;
    Snapshot observed{};
    bool launched = false;
    bool pending_sample = false;
    bool input_checked = false;
};

__global__ void set_input(Control *state, unsigned long long error) { state->input_error = error; }

__global__ void select_input(Control *state, cudaGraphConditionalHandle condition) {
    const auto error = reinterpret_cast<const int *>(state->input_error);
    const unsigned invalid = error && *error != 0;
    atomicExch(&state->input_invalid, invalid);
    cudaGraphSetConditional(condition, !invalid && !atomicAdd(&state->cancel, 0u));
}

__global__ void begin_body(Control *state, int64_t *index) {
    *index = 0;
    atomicExch(&state->body_entered, 1u);
}

__global__ void advance(Control *state, int64_t *index, unsigned steps,
                        cudaGraphConditionalHandle condition) {
    const unsigned completed = atomicAdd(&state->completed, 1u) + 1u;
    *index = static_cast<int64_t>(completed);
    const unsigned canceled = atomicAdd(&state->cancel, 0u);
    cudaGraphSetConditional(condition, completed < steps && !canceled);
}

__global__ void cancel_repeat(Control *state) { atomicExch(&state->cancel, 1u); }

// Gate entry after a prefix, and gate the suffix after the whole loop. No host decides
// whether another phase runs, and a cancel before the suffix boundary skips it.
__global__ void select_phase(Control *state, unsigned steps,
                             cudaGraphConditionalHandle condition, bool finishing) {
    const unsigned canceled = atomicAdd(&state->cancel, 0u);
    const unsigned completed = atomicAdd(&state->completed, 0u);
    cudaGraphSetConditional(condition, !canceled && (!finishing || completed == steps));
}

__global__ void finish_suffix(Control *state) { atomicExch(&state->suffix_completed, 1u); }

__global__ void snapshot(Control *state, Snapshot *out) {
    out->completed = atomicAdd(&state->completed, 0u);
    out->cancel = atomicAdd(&state->cancel, 0u);
    out->suffix_completed = atomicAdd(&state->suffix_completed, 0u);
    out->body_entered = atomicAdd(&state->body_entered, 0u);
    out->input_invalid = atomicAdd(&state->input_invalid, 0u);
}

// A body's memset and memcpy nodes, as kernels (`lower`). `word` repeats the memset value to four
// bytes; `wide` moves 16 bytes a thread when every address, pitch and row is 16-aligned.
__global__ void fill_rows(unsigned char *dst, size_t pitch, size_t row_bytes, size_t rows,
                          unsigned word, bool wide) {
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (wide) {
        const uint4 value = make_uint4(word, word, word, word);
        const size_t units = row_bytes / 16, total = units * rows;
        for (; i < total; i += stride)
            reinterpret_cast<uint4 *>(dst + (i / units) * pitch)[i % units] = value;
        return;
    }
    for (const size_t total = row_bytes * rows; i < total; i += stride) {
        const size_t column = i % row_bytes;
        dst[(i / row_bytes) * pitch + column] = static_cast<unsigned char>(word >> (8 * (column % 4)));
    }
}

__global__ void copy_rows(unsigned char *dst, const unsigned char *src, size_t dst_pitch,
                          size_t dst_slice, size_t src_pitch, size_t src_slice, size_t width,
                          size_t height, size_t depth, bool wide) {
    const size_t unit = wide ? 16 : 1, per_row = width / unit, total = per_row * height * depth;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        const size_t row = i / per_row, column = (i % per_row) * unit;
        const size_t y = row % height, z = row / height;
        unsigned char *to = dst + z * dst_slice + y * dst_pitch + column;
        const unsigned char *from = src + z * src_slice + y * src_pitch + column;
        if (wide) *reinterpret_cast<uint4 *>(to) = *reinterpret_cast<const uint4 *>(from);
        else *to = *from;
    }
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

// The driver runs a conditional body on the device. Replace each memset and memcpy node of an
// embedded body copy with a `fill_rows`/`copy_rows` kernel node on the same edges, so a body
// holds only kernel, empty and child-graph nodes. `inspect` already accepted every endpoint.
static cudaError_t lower(cudaGraph_t graph, Repeat *run, const char *stage, unsigned depth = 0) {
    if (depth > 32) return ineligible(run, stage, "depth_limit", -1, depth, 0, 0);
    size_t count = 0;
    auto error = cudaGraphGetNodes(graph, nullptr, &count);
    if (error != cudaSuccess) return error;
    std::vector<cudaGraphNode_t> nodes(count);
    error = cudaGraphGetNodes(graph, nodes.data(), &count);
    if (error != cudaSuccess) return error;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        if ((error = cudaGraphNodeGetType(node, &type)) != cudaSuccess) return error;
        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child;
            if ((error = cudaGraphChildGraphNodeGetGraph(node, &child)) != cudaSuccess) return error;
            if ((error = lower(child, run, stage, depth + 1)) != cudaSuccess) return error;
            continue;
        }
        if (type != cudaGraphNodeTypeMemset && type != cudaGraphNodeTypeMemcpy) continue;
        unsigned char *dst = nullptr;
        const unsigned char *src = nullptr;
        size_t dst_pitch = 0, dst_slice = 0, src_pitch = 0, src_slice = 0, width = 0, height = 0;
        size_t depth_rows = 1, units = 0;
        unsigned word = 0;
        bool wide = false;
        void *fill_args[] = {&dst, &dst_pitch, &width, &height, &word, &wide};
        void *copy_args[] = {&dst, &src, &dst_pitch, &dst_slice, &src_pitch, &src_slice,
                             &width, &height, &depth_rows, &wide};
        cudaKernelNodeParams kernel{};
        if (type == cudaGraphNodeTypeMemset) {
            cudaMemsetParams set{};
            if ((error = cudaGraphMemsetNodeGetParams(node, &set)) != cudaSuccess) return error;
            if (set.elementSize != 1 && set.elementSize != 2 && set.elementSize != 4)
                return ineligible(run, stage, "memset_element", type, depth, count, 0);
            dst = static_cast<unsigned char *>(set.dst);
            width = set.width * set.elementSize;
            height = set.height;
            dst_pitch = height > 1 ? set.pitch : width;
            word = set.elementSize == 1 ? (set.value & 0xffu) * 0x01010101u
                 : set.elementSize == 2 ? (set.value & 0xffffu) * 0x00010001u : set.value;
            wide = ((reinterpret_cast<uintptr_t>(dst) | dst_pitch | width) & 15) == 0;
            units = (wide ? width / 16 : width) * height;
            kernel.func = reinterpret_cast<void *>(fill_rows);
            kernel.kernelParams = fill_args;
        } else {
            cudaMemcpy3DParms copy{};
            if ((error = cudaGraphMemcpyNodeGetParams(node, &copy)) != cudaSuccess) return error;
            if (copy.srcArray || copy.dstArray || !copy.srcPtr.ptr || !copy.dstPtr.ptr)
                return ineligible(run, stage, "memcpy_layout_or_kind", type, depth, count, 0,
                                  copy.kind);
            width = copy.extent.width;
            height = copy.extent.height ? copy.extent.height : 1;
            depth_rows = copy.extent.depth ? copy.extent.depth : 1;
            src_pitch = height > 1 || depth_rows > 1 ? copy.srcPtr.pitch : width;
            dst_pitch = height > 1 || depth_rows > 1 ? copy.dstPtr.pitch : width;
            src_slice = src_pitch * copy.srcPtr.ysize;
            dst_slice = dst_pitch * copy.dstPtr.ysize;
            src = static_cast<const unsigned char *>(copy.srcPtr.ptr) + copy.srcPos.z * src_slice
                + copy.srcPos.y * src_pitch + copy.srcPos.x;
            dst = static_cast<unsigned char *>(copy.dstPtr.ptr) + copy.dstPos.z * dst_slice
                + copy.dstPos.y * dst_pitch + copy.dstPos.x;
            wide = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst) | width
                     | (height > 1 || depth_rows > 1 ? src_pitch | dst_pitch | src_slice | dst_slice : 0))
                    & 15) == 0;
            units = (wide ? width / 16 : width) * height * depth_rows;
            kernel.func = reinterpret_cast<void *>(copy_rows);
            kernel.kernelParams = copy_args;
        }
        const size_t blocks = (units + 255) / 256;
        kernel.gridDim = dim3(static_cast<unsigned>(blocks < 1 ? 1 : blocks > 4096 ? 4096 : blocks));
        kernel.blockDim = dim3(256);
        size_t inputs = 0, outputs = 0;
        if ((error = cudaGraphNodeGetDependencies(node, nullptr, nullptr, &inputs)) != cudaSuccess) return error;
        if ((error = cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &outputs)) != cudaSuccess) return error;
        std::vector<cudaGraphNode_t> before(inputs), after(outputs);
        std::vector<cudaGraphEdgeData> before_data(inputs), after_data(outputs);
        if ((error = cudaGraphNodeGetDependencies(node, before.data(), before_data.data(), &inputs)) != cudaSuccess
            || (error = cudaGraphNodeGetDependentNodes(node, after.data(), after_data.data(), &outputs)) != cudaSuccess)
            return error;
        for (const auto *edges : {&before_data, &after_data})
            for (const auto &edge : *edges)
                if (edge.type != cudaGraphDependencyTypeDefault || edge.from_port || edge.to_port)
                    return ineligible(run, stage, "lowered_edge", type, depth, count, 0);
        cudaGraphNode_t replacement;
        if ((error = cudaGraphAddKernelNode(&replacement, graph, before.data(), inputs, &kernel)) != cudaSuccess)
            return error;
        for (size_t i = 0; i < outputs; ++i)
            if ((error = cudaGraphAddDependencies(graph, &replacement, &after[i], nullptr, 1)) != cudaSuccess)
                return error;
        if ((error = cudaGraphDestroyNode(node)) != cudaSuccess) return error;
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
    for (auto &event : run->input_ready) {
        if (event && (error = cudaEventDestroy(event)) != cudaSuccess) return error;
        event = nullptr;
    }
    if (run->work && (error = cudaStreamDestroy(run->work)) != cudaSuccess) return error;
    run->work = nullptr;
    if (run->control_stream && (error = cudaStreamDestroy(run->control_stream)) != cudaSuccess) return error;
    run->control_stream = nullptr;
    return cudaSuccess;
}

static int create(uintptr_t source, uintptr_t index, int device, unsigned steps,
                  uintptr_t prefix, uintptr_t suffix, Repeat **out) {
    *out = nullptr;
    if (!source || !index || !steps) return cudaErrorInvalidValue;
    cudaError_t error;
    auto run = new (std::nothrow) Repeat;
    if (!run) return cudaErrorMemoryAllocation;
    run->device = device; run->steps = steps; run->index = reinterpret_cast<int64_t *>(index);
    run->has_suffix = suffix != 0;
    // Return ownership even on failure so Rust can retain the source after uncertain cleanup.
    *out = run;
    auto checked = [&](cudaError_t result) { return result; };
#define CHECK(call) do { error = checked(call); if (error != cudaSuccess) return error; } while (0)
    CHECK(inspect(reinterpret_cast<cudaGraph_t>(source), run->nodes, device, run, "source"));
    if (!run->nodes) return -101;
    unsigned total = run->nodes;
    if (prefix) {
        CHECK(inspect(reinterpret_cast<cudaGraph_t>(prefix), total, device, run, "prefix"));
        run->prefix_nodes = total - run->nodes;
    }
    if (suffix) {
        const unsigned before = total;
        CHECK(inspect(reinterpret_cast<cudaGraph_t>(suffix), total, device, run, "suffix"));
        run->suffix_nodes = total - before;
    }
    CHECK(cudaStreamCreateWithFlags(&run->work, cudaStreamNonBlocking));
    int lowest_priority, highest_priority;
    CHECK(cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority));
    CHECK(cudaStreamCreateWithPriority(&run->control_stream, cudaStreamNonBlocking, highest_priority));
    CHECK(cudaEventCreateWithFlags(&run->done, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&run->producer_ready, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&run->initialized, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&run->sampled, cudaEventDisableTiming));
    for (auto &event : run->input_ready)
        CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    // A first control-kernel launch must never trigger lazy loading while the repeat runs.
    cudaFuncAttributes attributes;
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(advance)));
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(cancel_repeat)));
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(snapshot)));
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(set_input)));
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(select_input)));
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(begin_body)));
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(fill_rows)));
    CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(copy_rows)));
    if (prefix || suffix) CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(select_phase)));
    if (suffix) CHECK(cudaFuncGetAttributes(&attributes, reinterpret_cast<const void *>(finish_suffix)));
    CHECK(cudaMalloc(&run->control, sizeof(Control)));
    CHECK(cudaMalloc(&run->device_snapshot, sizeof(Snapshot)));
    CHECK(cudaMallocHost(&run->host_snapshot, sizeof(Snapshot)));
    CHECK(cudaGraphCreate(&run->graph, 0));
    cudaGraphConditionalHandle accepted;
    CHECK(cudaGraphConditionalHandleCreate(&accepted, run->graph, 0, cudaGraphCondAssignDefault));
    void *input_args[] = {&run->control, &accepted};
    cudaKernelNodeParams input_gate{};
    input_gate.func = reinterpret_cast<void *>(select_input);
    input_gate.gridDim = dim3(1); input_gate.blockDim = dim3(1); input_gate.kernelParams = input_args;
    cudaGraphNode_t input_node;
    CHECK(cudaGraphAddKernelNode(&input_node, run->graph, nullptr, 0, &input_gate));
    cudaGraphNodeParams input_if{};
    input_if.type = cudaGraphNodeTypeConditional;
    input_if.conditional.handle = accepted;
    input_if.conditional.type = cudaGraphCondTypeIf;
    input_if.conditional.size = 1;
    cudaGraphNode_t accepted_node;
    CHECK(cudaGraphAddNode(&accepted_node, run->graph, &input_node, nullptr, 1, &input_if));
    auto program = input_if.conditional.phGraph_out[0];
    void *begin_args[] = {&run->control, &run->index};
    cudaKernelNodeParams begin{};
    begin.func = reinterpret_cast<void *>(begin_body);
    begin.gridDim = dim3(1); begin.blockDim = dim3(1); begin.kernelParams = begin_args;
    cudaGraphNode_t predecessor;
    CHECK(cudaGraphAddKernelNode(&predecessor, program, nullptr, 0, &begin));
    cudaGraphConditionalHandle condition;
    CHECK(cudaGraphConditionalHandleCreate(&condition, run->graph, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams conditional{};
    conditional.type = cudaGraphNodeTypeConditional;
    conditional.conditional.handle = condition;
    conditional.conditional.type = cudaGraphCondTypeWhile;
    conditional.conditional.size = 1;
    if (prefix) {
        cudaGraphNode_t compute;
        CHECK(cudaGraphAddChildGraphNode(&compute, program, &predecessor, 1, reinterpret_cast<cudaGraph_t>(prefix)));
        cudaGraph_t cloned;
        CHECK(cudaGraphChildGraphNodeGetGraph(compute, &cloned));
        CHECK(lower(cloned, run, "assembled_prefix"));
        CHECK(inspect(cloned, run->validated_nodes, device, run, "assembled_prefix"));
        bool finishing = false;
        void *args[] = {&run->control, &run->steps, &condition, &finishing};
        cudaKernelNodeParams gate{};
        gate.func = reinterpret_cast<void *>(select_phase);
        gate.gridDim = dim3(1); gate.blockDim = dim3(1); gate.kernelParams = args;
        CHECK(cudaGraphAddKernelNode(&predecessor, program, &compute, 1, &gate));
    }
    cudaGraphNode_t loop;
    CHECK(cudaGraphAddNode(&loop, program, &predecessor, nullptr, 1, &conditional));
    auto body = conditional.conditional.phGraph_out[0];
    cudaGraphNode_t compute;
    CHECK(cudaGraphAddChildGraphNode(&compute, body, nullptr, 0, reinterpret_cast<cudaGraph_t>(source)));
    cudaGraph_t embedded;
    CHECK(cudaGraphChildGraphNodeGetGraph(compute, &embedded));
    CHECK(lower(embedded, run, "assembled"));
    void *args[] = {&run->control, &run->index, &run->steps, &condition};
    cudaKernelNodeParams kernel{};
    kernel.func = reinterpret_cast<void *>(advance);
    kernel.gridDim = dim3(1); kernel.blockDim = dim3(1); kernel.kernelParams = args;
    cudaGraphNode_t boundary;
    CHECK(cudaGraphAddKernelNode(&boundary, body, &compute, 1, &kernel));
    // Inspect the actual assembled body, including the cloned source child definition.
    CHECK(inspect(body, run->validated_nodes, device, run, "assembled"));
    if (suffix) {
        cudaGraphConditionalHandle finish;
        CHECK(cudaGraphConditionalHandleCreate(&finish, run->graph, 0, cudaGraphCondAssignDefault));
        bool finishing = true;
        void *args[] = {&run->control, &run->steps, &finish, &finishing};
        cudaKernelNodeParams gate{};
        gate.func = reinterpret_cast<void *>(select_phase);
        gate.gridDim = dim3(1); gate.blockDim = dim3(1); gate.kernelParams = args;
        cudaGraphNode_t decide;
        CHECK(cudaGraphAddKernelNode(&decide, program, &loop, 1, &gate));
        cudaGraphNodeParams conditional{};
        conditional.type = cudaGraphNodeTypeConditional;
        conditional.conditional.handle = finish;
        conditional.conditional.type = cudaGraphCondTypeIf;
        conditional.conditional.size = 1;
        cudaGraphNode_t tail;
        CHECK(cudaGraphAddNode(&tail, program, &decide, nullptr, 1, &conditional));
        auto final = conditional.conditional.phGraph_out[0];
        cudaGraphNode_t compute, complete;
        CHECK(cudaGraphAddChildGraphNode(&compute, final, nullptr, 0, reinterpret_cast<cudaGraph_t>(suffix)));
        cudaGraph_t embedded_suffix;
        CHECK(cudaGraphChildGraphNodeGetGraph(compute, &embedded_suffix));
        CHECK(lower(embedded_suffix, run, "assembled_suffix"));
        void *done_args[] = {&run->control};
        cudaKernelNodeParams done{};
        done.func = reinterpret_cast<void *>(finish_suffix);
        done.gridDim = dim3(1); done.blockDim = dim3(1); done.kernelParams = done_args;
        CHECK(cudaGraphAddKernelNode(&complete, final, &compute, 1, &done));
        CHECK(inspect(final, run->validated_nodes, device, run, "assembled_suffix"));
    }
    CHECK(cudaGraphInstantiate(&run->executable, run->graph, 0));
    CHECK(cudaGraphUpload(run->executable, run->work));
    CHECK(cudaStreamSynchronize(run->work));
    return cudaSuccess;
#undef CHECK
}

extern "C" int cozy_repeat_create(uintptr_t source, uintptr_t index, int device,
                                  unsigned steps, uintptr_t prefix, uintptr_t suffix, Repeat **out) {
    *out = nullptr;
    return on_device(device, [&] { return create(source, index, device, steps, prefix, suffix, out); });
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
    run->input_checked = false;
    return cudaEventRecord(run->producer_ready, reinterpret_cast<cudaStream_t>(producer));
}

static int launch_armed(Repeat *run, const uintptr_t *streams = nullptr,
                        unsigned count = 0, unsigned long long input_error = 0) {
    if (count > run->input_ready.size() || (count && !streams) || (input_error && !count))
        return cudaErrorInvalidValue;
    for (unsigned i = 0; i < count; ++i) {
        if (!streams[i] || streams[i] == reinterpret_cast<uintptr_t>(cudaStreamLegacy)
            || streams[i] == reinterpret_cast<uintptr_t>(cudaStreamPerThread)) return cudaErrorInvalidValue;
        auto error = cudaEventRecord(run->input_ready[i], reinterpret_cast<cudaStream_t>(streams[i]));
        if (error != cudaSuccess) return error;
    }
    // The caller-ready event was already recorded before the native read worker was started.
    cudaError_t error = cudaStreamWaitEvent(run->work, run->producer_ready, 0);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(run->control, 0, sizeof(Control), run->work);
    if (error != cudaSuccess) return error;
    if (input_error) {
        set_input<<<1, 1, 0, run->work>>>(run->control, input_error);
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
    }
    error = cudaEventRecord(run->initialized, run->work);
    if (error != cudaSuccess) return error;
    error = cudaStreamWaitEvent(run->control_stream, run->initialized, 0);
    if (error != cudaSuccess) return error;
    for (unsigned i = 0; i < count; ++i) {
        error = cudaStreamWaitEvent(run->work, run->input_ready[i], 0);
        if (error != cudaSuccess) return error;
    }
    run->observed = {}; run->pending_sample = false; run->input_checked = false;
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
    *canceled = run->observed.cancel && (run->observed.completed < run->steps
                 || (run->has_suffix && !run->observed.suffix_completed));
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
    run->observed = {state.completed, state.cancel, state.suffix_completed,
                     state.body_entered, state.input_invalid};
    run->input_checked = true;
    *completed = state.completed;
    *canceled = state.cancel && (state.completed < run->steps
                 || (run->has_suffix && !state.suffix_completed));
    return cudaSuccess;
}

extern "C" unsigned cozy_repeat_nodes(Repeat *run) { return run->nodes; }
extern "C" unsigned cozy_repeat_validated_nodes(Repeat *run) { return run->validated_nodes; }
extern "C" unsigned cozy_repeat_prefix_nodes(Repeat *run) { return run->prefix_nodes; }
extern "C" unsigned cozy_repeat_suffix_nodes(Repeat *run) { return run->suffix_nodes; }

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
extern "C" int cozy_repeat_launch_after(Repeat *run, const uintptr_t *streams,
                                        unsigned count, unsigned long long input_error) {
    return on_device(run->device, [&] { return launch_armed(run, streams, count, input_error); });
}
// Host-only facts: zero is not a checked negative observation until input_checked is true.
extern "C" void cozy_repeat_input_status(Repeat *run, unsigned *entered, unsigned *invalid, int *checked) {
    *entered = run->observed.body_entered;
    *invalid = run->observed.input_invalid;
    *checked = run->input_checked;
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
