// SPDX-License-Identifier: Apache-2.0
// `_kernels._engine`: the inference engine's step launcher (xs-213). A Segment runs a captured step graph as
// the body of a CUDA WHILE node, so one launch runs steps until a step writes 0 to its `go` flag. `report`,
// a step's last op, copies a record into a ring in pinned, device-mapped host memory; a Reader takes the
// records in order. Raw pointers and streams, so neither torch nor the Python version is a build input.
#include <nanobind/nanobind.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/loop.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

class OnDevice {
  public:
    explicit OnDevice(int device) {
        error_ = cudaGetDevice(&previous_);
        if (error_ == cudaSuccess && previous_ != device) error_ = cudaSetDevice(device);
    }
    ~OnDevice() {
        int now = -1;
        if (error_ == cudaSuccess && cudaGetDevice(&now) == cudaSuccess && now != previous_)
            cudaSetDevice(previous_);
    }
    cudaError_t error() const { return error_; }

  private:
    int previous_ = -1;
    cudaError_t error_ = cudaSuccess;
};

std::string said(const char* call, cudaError_t code) {
    return std::string(call) + ": " + cudaGetErrorName(code) + " (" + std::to_string(int(code)) + ")";
}

void check(const char* call, cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(said(call, code));
}

const char* kind(cudaGraphNodeType type) {
    switch (type) {
        case cudaGraphNodeTypeKernel: return "kernel";
        case cudaGraphNodeTypeMemcpy: return "memcpy";
        case cudaGraphNodeTypeMemset: return "memset";
        case cudaGraphNodeTypeHost: return "host";
        case cudaGraphNodeTypeGraph: return "child graph";
        case cudaGraphNodeTypeEmpty: return "empty";
        case cudaGraphNodeTypeWaitEvent: return "event wait";
        case cudaGraphNodeTypeEventRecord: return "event record";
        case cudaGraphNodeTypeExtSemaphoreSignal: return "external semaphore signal";
        case cudaGraphNodeTypeExtSemaphoreWait: return "external semaphore wait";
        case cudaGraphNodeTypeMemAlloc: return "memory allocation";
        case cudaGraphNodeTypeMemFree: return "memory free";
        case cudaGraphNodeTypeConditional: return "conditional";
        default: return "unknown";
    }
}

std::vector<cudaGraphNode_t> nodes(cudaGraph_t graph) {
    size_t count = 0;
    check("cudaGraphGetNodes", cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> found(count);
    if (count) check("cudaGraphGetNodes", cudaGraphGetNodes(graph, found.data(), &count));
    found.resize(count);
    return found;
}

// The first node the step graph holds that a WHILE body's child graph may not: a body holds kernel,
// empty, memcpy, memset, child graph and conditional nodes (programming guide, conditional node body
// graph requirements), and a child graph no conditional (`cudaChildGraphNodeParams`).
// A memcpy node's direction, from its kind or, where it is cudaMemcpyDefault, its pointers.
const char* direction(cudaGraphNode_t node) {
    cudaMemcpy3DParms p = {};
    if (cudaGraphMemcpyNodeGetParams(node, &p) != cudaSuccess) return "memcpy";
    cudaMemcpyKind kind = p.kind;
    if (kind == cudaMemcpyDefault) {
        auto on_card = [](const void* ptr) {
            cudaPointerAttributes at = {};
            if (cudaPointerGetAttributes(&at, ptr) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            return at.type == cudaMemoryTypeDevice || at.type == cudaMemoryTypeManaged;
        };
        const void* src = p.srcArray ? nullptr : static_cast<const char*>(p.srcPtr.ptr) + p.srcPos.x;
        const void* dst = p.dstArray ? nullptr : static_cast<const char*>(p.dstPtr.ptr) + p.dstPos.x;
        const bool from = p.srcArray || on_card(src), to = p.dstArray || on_card(dst);
        kind = from ? (to ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost)
                    : (to ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost);
    }
    switch (kind) {
        case cudaMemcpyHostToDevice: return "memcpy host-to-device";
        case cudaMemcpyDeviceToDevice: return "memcpy device-to-device";
        case cudaMemcpyDeviceToHost: return "memcpy device-to-host";
        case cudaMemcpyHostToHost: return "memcpy host-to-host";
        default: return "memcpy";
    }
}

// Every node kind of `graph`, a child graph's as its own nodes', a memcpy by its direction.
void kinds_of(cudaGraph_t graph, std::vector<std::string>& found) {
    for (cudaGraphNode_t node : nodes(graph)) {
        cudaGraphNodeType type;
        check("cudaGraphNodeGetType", cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child;
            check("cudaGraphChildGraphNodeGetGraph", cudaGraphChildGraphNodeGetGraph(node, &child));
            kinds_of(child, found);
            continue;
        }
        const std::string name = type == cudaGraphNodeTypeMemcpy ? direction(node) : kind(type);
        bool seen = false;
        for (const std::string& had : found) seen = seen || had == name;
        if (!seen) found.push_back(name);
    }
}

nb::list kinds(uintptr_t graph) {
    if (!graph) throw std::invalid_argument("kinds: no graph");
    std::vector<std::string> found;
    kinds_of(reinterpret_cast<cudaGraph_t>(graph), found);
    nb::list out;
    for (const std::string& name : found) out.append(nb::str(name.c_str()));  // no stl caster is included
    return out;
}

void memset_async(uintptr_t ptr, int value, size_t nbytes, uintptr_t stream) {
    if (!ptr) throw std::invalid_argument("memset: no pointer");
    check("cudaMemsetAsync", cudaMemsetAsync(reinterpret_cast<void*>(ptr), value, nbytes,
                                             reinterpret_cast<cudaStream_t>(stream)));
}

const char* refused(cudaGraph_t graph) {
    for (cudaGraphNode_t node : nodes(graph)) {
        cudaGraphNodeType type;
        check("cudaGraphNodeGetType", cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child;
            check("cudaGraphChildGraphNodeGetGraph", cudaGraphChildGraphNodeGetGraph(node, &child));
            if (const char* inner = refused(child)) return inner;
        } else if (type != cudaGraphNodeTypeKernel && type != cudaGraphNodeTypeMemcpy &&
                   type != cudaGraphNodeTypeMemset && type != cudaGraphNodeTypeEmpty) {
            return kind(type);
        }
    }
    return nullptr;
}

// A body's 1-D copy, lowered to a `copy_rows` kernel node (`lower`): its node, and its operands as captured.
struct Copy {
    cudaGraphNode_t node;
    uintptr_t dst, src;
    size_t bytes;
};

// The address a kernel reads or writes `pointer` at: its own on this card, a mapped host range's device
// address (the K/V host tier registers its blocks so: CU_MEMHOSTREGISTER_DEVICEMAP; pinned memory under
// unified addressing is mapped too). Anything else (pageable memory, another card's) the lowering refuses.
const void* on_card(const void* pointer, int device) {
    cudaPointerAttributes at = {};
    if (cudaPointerGetAttributes(&at, pointer) != cudaSuccess) {
        cudaGetLastError();
        at = {};
    }
    if ((at.type == cudaMemoryTypeDevice || at.type == cudaMemoryTypeManaged) && at.device == device) return pointer;
    if (at.type == cudaMemoryTypeHost && at.devicePointer) return at.devicePointer;
    throw std::invalid_argument("Segment: a copy in the step reads or writes memory this card does not address");
}

unsigned blocks(size_t units) {
    const size_t n = (units + 255) / 256;
    return static_cast<unsigned>(n < 1 ? 1 : n > 4096 ? 4096 : n);
}

// One 1-D copy of `bytes` as `copy_rows`'s kernel parameters (`args` holds what they point at).
struct Row1D {
    unsigned char* dst;
    const unsigned char* src;
    size_t dst_pitch, dst_slice, src_pitch, src_slice, width, height = 1, depth = 1;
    bool wide;
    void* args[10];
    cudaKernelNodeParams params = {};

    Row1D(const void* to, const void* from, size_t bytes)
        : dst(static_cast<unsigned char*>(const_cast<void*>(to))), src(static_cast<const unsigned char*>(from)),
          dst_pitch(bytes), dst_slice(bytes), src_pitch(bytes), src_slice(bytes), width(bytes),
          wide(((reinterpret_cast<uintptr_t>(to) | reinterpret_cast<uintptr_t>(from) | bytes) & 15) == 0),
          args{&dst, &src, &dst_pitch, &dst_slice, &src_pitch, &src_slice, &width, &height, &depth, &wide} {
        params.func = const_cast<void*>(cozy_engine::copy_rows_kernel());
        params.gridDim = dim3(blocks(wide ? bytes / 16 : bytes));
        params.blockDim = dim3(256);
        params.kernelParams = args;
    }
};

// cozy-runtime #1860's lowering, the one in Cozy (its ResidentRepeat is gone): each memset and memcpy node of
// a WHILE body, through child graphs, replaced by a `fill_rows`/`copy_rows` kernel node on the same edges, so
// the body holds kernel, empty and child-graph nodes only. A 1-D copy is kept in `copies` for its edits.
void lower(cudaGraph_t graph, int device, std::vector<Copy>& copies, unsigned depth = 0) {
    if (depth > 32) throw std::invalid_argument("Segment: the step's graphs nest deeper than 32");
    for (cudaGraphNode_t node : nodes(graph)) {
        cudaGraphNodeType type;
        check("cudaGraphNodeGetType", cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child;
            check("cudaGraphChildGraphNodeGetGraph", cudaGraphChildGraphNodeGetGraph(node, &child));
            lower(child, device, copies, depth + 1);
            continue;
        }
        if (type != cudaGraphNodeTypeMemset && type != cudaGraphNodeTypeMemcpy) continue;
        unsigned char* dst = nullptr;
        const unsigned char* src = nullptr;
        size_t dst_pitch = 0, dst_slice = 0, src_pitch = 0, src_slice = 0, width = 0, height = 0, depth_rows = 1,
               units = 0;
        unsigned word = 0;
        bool wide = false;
        void* fill_args[] = {&dst, &dst_pitch, &width, &height, &word, &wide};
        void* copy_args[] = {&dst, &src, &dst_pitch, &dst_slice, &src_pitch, &src_slice, &width, &height, &depth_rows,
                             &wide};
        cudaKernelNodeParams kernel = {};
        bool one_d = false;
        uintptr_t captured_dst = 0, captured_src = 0;
        if (type == cudaGraphNodeTypeMemset) {
            cudaMemsetParams set = {};
            check("cudaGraphMemsetNodeGetParams", cudaGraphMemsetNodeGetParams(node, &set));
            if (set.elementSize != 1 && set.elementSize != 2 && set.elementSize != 4)
                throw std::invalid_argument("Segment: a memset of " + std::to_string(set.elementSize) + "-byte elements");
            dst = static_cast<unsigned char*>(const_cast<void*>(on_card(set.dst, device)));
            width = set.width * set.elementSize;
            height = set.height ? set.height : 1;
            dst_pitch = height > 1 ? set.pitch : width;
            word = set.elementSize == 1   ? (set.value & 0xffu) * 0x01010101u
                   : set.elementSize == 2 ? (set.value & 0xffffu) * 0x00010001u
                                          : set.value;
            wide = ((reinterpret_cast<uintptr_t>(dst) | dst_pitch | width) & 15) == 0;
            units = (wide ? width / 16 : width) * height;
            kernel.func = const_cast<void*>(cozy_engine::fill_rows_kernel());
            kernel.kernelParams = fill_args;
        } else {
            cudaMemcpy3DParms copy = {};
            check("cudaGraphMemcpyNodeGetParams", cudaGraphMemcpyNodeGetParams(node, &copy));
            if (copy.srcArray || copy.dstArray || !copy.srcPtr.ptr || !copy.dstPtr.ptr)
                throw std::invalid_argument("Segment: a copy of a CUDA array, which a WHILE body's kernel cannot make");
            width = copy.extent.width;
            height = copy.extent.height ? copy.extent.height : 1;
            depth_rows = copy.extent.depth ? copy.extent.depth : 1;
            src_pitch = height > 1 || depth_rows > 1 ? copy.srcPtr.pitch : width;
            dst_pitch = height > 1 || depth_rows > 1 ? copy.dstPtr.pitch : width;
            src_slice = src_pitch * copy.srcPtr.ysize;
            dst_slice = dst_pitch * copy.dstPtr.ysize;
            const auto* from = static_cast<const unsigned char*>(copy.srcPtr.ptr) + copy.srcPos.z * src_slice +
                               copy.srcPos.y * src_pitch + copy.srcPos.x;
            auto* to = static_cast<unsigned char*>(copy.dstPtr.ptr) + copy.dstPos.z * dst_slice +
                       copy.dstPos.y * dst_pitch + copy.dstPos.x;
            captured_dst = reinterpret_cast<uintptr_t>(to);
            captured_src = reinterpret_cast<uintptr_t>(from);
            src = static_cast<const unsigned char*>(on_card(from, device));
            dst = static_cast<unsigned char*>(const_cast<void*>(on_card(to, device)));
            wide = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst) | width |
                     (height > 1 || depth_rows > 1 ? src_pitch | dst_pitch | src_slice | dst_slice : 0)) &
                    15) == 0;
            units = (wide ? width / 16 : width) * height * depth_rows;
            kernel.func = const_cast<void*>(cozy_engine::copy_rows_kernel());
            kernel.kernelParams = copy_args;
            one_d = height == 1 && depth_rows == 1;
        }
        kernel.gridDim = dim3(blocks(units));
        kernel.blockDim = dim3(256);
        size_t inputs = 0, outputs = 0;
        check("cudaGraphNodeGetDependencies", cudaGraphNodeGetDependencies(node, nullptr, nullptr, &inputs));
        check("cudaGraphNodeGetDependentNodes", cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &outputs));
        std::vector<cudaGraphNode_t> before(inputs), after(outputs);
        std::vector<cudaGraphEdgeData> before_data(inputs), after_data(outputs);
        check("cudaGraphNodeGetDependencies", cudaGraphNodeGetDependencies(node, before.data(), before_data.data(), &inputs));
        check("cudaGraphNodeGetDependentNodes",
              cudaGraphNodeGetDependentNodes(node, after.data(), after_data.data(), &outputs));
        for (const auto* edges : {&before_data, &after_data})
            for (const auto& edge : *edges)
                if (edge.type != cudaGraphDependencyTypeDefault || edge.from_port || edge.to_port)
                    throw std::invalid_argument("Segment: a copy or memset on a programmatic edge");
        cudaGraphNode_t replacement;
        check("cudaGraphAddKernelNode", cudaGraphAddKernelNode(&replacement, graph, before.data(), inputs, &kernel));
        for (size_t i = 0; i < outputs; ++i)
            check("cudaGraphAddDependencies", cudaGraphAddDependencies(graph, &replacement, &after[i], nullptr, 1));
        check("cudaGraphDestroyNode", cudaGraphDestroyNode(node));
        if (one_d) copies.push_back({replacement, captured_dst, captured_src, width});
    }
}

class Segment {
  public:
    Segment(uintptr_t step, uintptr_t go, int device) : device_(device) {
        if (!step) throw std::invalid_argument("Segment: no step graph");
        if (!go || go % alignof(int)) throw std::invalid_argument("Segment: go is no aligned int32 pointer");
        const OnDevice guard(device);
        check("cudaSetDevice", guard.error());
        const auto graph = reinterpret_cast<cudaGraph_t>(step);
        if (const char* bad = refused(graph))
            throw std::invalid_argument(std::string("Segment: the step graph holds a ") + bad +
                                        " node, which a WHILE body's child graph may not");
        try {
            build(graph, reinterpret_cast<const int*>(go));
        } catch (...) {
            release();
            throw;
        }
    }
    ~Segment() { release(); }
    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;

    void launch(uintptr_t stream) {
        nb::gil_scoped_release free;
        std::lock_guard<std::mutex> held(lock_);
        if (!exec_) throw std::runtime_error("Segment: closed");
        const OnDevice guard(device_);
        check("cudaSetDevice", guard.error());
        check("cudaGraphLaunch", cudaGraphLaunch(exec_, reinterpret_cast<cudaStream_t>(stream)));
    }

    // A launch in flight finishes: the driver frees its executable once it completes (cuGraphExecDestroy).
    void close() {
        nb::gil_scoped_release free;
        std::lock_guard<std::mutex> held(lock_);
        release();
    }

    nb::list copies() {
        std::lock_guard<std::mutex> held(lock_);
        nb::list out;
        for (const Copy& copy : copies_) out.append(nb::make_tuple(copy.dst, copy.src, copy.bytes));
        return out;
    }

    // Through the parent's executable, for later launches.
    void set_copy(size_t index, uintptr_t dst, uintptr_t src, size_t nbytes) {
        std::lock_guard<std::mutex> held(lock_);
        const Copy& copy = body_copy(index);
        const OnDevice guard(device_);
        check("cudaSetDevice", guard.error());
        const Row1D row(on_card(reinterpret_cast<const void*>(dst), device_),
                        on_card(reinterpret_cast<const void*>(src), device_), nbytes);
        check("cudaGraphExecKernelNodeSetParams", cudaGraphExecKernelNodeSetParams(exec_, copy.node, &row.params));
    }

    void enable_copy(size_t index, bool on) {
        std::lock_guard<std::mutex> held(lock_);
        const Copy& copy = body_copy(index);
        const OnDevice guard(device_);
        check("cudaSetDevice", guard.error());
        check("cudaGraphNodeSetEnabled", cudaGraphNodeSetEnabled(exec_, copy.node, on ? 1 : 0));
    }

  private:
    // parent: WHILE (handle default 1, reset each launch) { step, as a child graph -> again(handle, go) }
    void build(cudaGraph_t step, const int* go) {
        check("cudaGraphCreate", cudaGraphCreate(&parent_, 0));
        cudaGraphConditionalHandle handle;
        const cudaError_t made = cudaGraphConditionalHandleCreate(&handle, parent_, 1, cudaGraphCondAssignDefault);
        if (made != cudaSuccess)
            throw std::runtime_error(said("cudaGraphConditionalHandleCreate", made) +
                                     ": this driver and CUDA runtime make no conditional graph nodes");
        cudaGraphNodeParams loop = {};
        loop.type = cudaGraphNodeTypeConditional;
        loop.conditional.handle = handle;
        loop.conditional.type = cudaGraphCondTypeWhile;
        loop.conditional.size = 1;
        cudaGraphNode_t node;
        check("cudaGraphAddNode (WHILE)", cudaGraphAddNode(&node, parent_, nullptr, nullptr, 0, &loop));
        cudaGraph_t body = loop.conditional.phGraph_out[0];

        cudaGraphNodeParams child = {};
        child.type = cudaGraphNodeTypeGraph;
        child.graph.graph = step;
        child.graph.ownership = cudaGraphChildGraphOwnershipClone;
        cudaGraphNode_t stepped;
        check("cudaGraphAddNode (step)", cudaGraphAddNode(&stepped, body, nullptr, nullptr, 0, &child));

        void* args[] = {&handle, &go};
        cudaGraphNodeParams again = {};
        again.type = cudaGraphNodeTypeKernel;
        again.kernel.func = const_cast<void*>(cozy_engine::again_kernel());
        again.kernel.gridDim.x = again.kernel.gridDim.y = again.kernel.gridDim.z = 1;
        again.kernel.blockDim.x = again.kernel.blockDim.y = again.kernel.blockDim.z = 1;
        again.kernel.kernelParams = args;
        cudaGraphNode_t setter;
        check("cudaGraphAddNode (condition)", cudaGraphAddNode(&setter, body, &stepped, nullptr, 1, &again));

        cudaGraph_t cloned;
        check("cudaGraphChildGraphNodeGetGraph", cudaGraphChildGraphNodeGetGraph(stepped, &cloned));
        lower(cloned, device_, copies_);  // the body as kernels only: one path on every driver

        cudaGraphInstantiateParams params = {};
        const cudaError_t code = cudaGraphInstantiateWithParams(&exec_, parent_, &params);
        if (code != cudaSuccess) {
            exec_ = nullptr;
            std::string why = said("cudaGraphInstantiateWithParams", code) + ", result " +
                              std::to_string(int(params.result_out));
            cudaGraphNodeType type;
            if (params.errNode_out && cudaGraphNodeGetType(params.errNode_out, &type) == cudaSuccess)
                why += std::string(", at a ") + kind(type) + " node";
            throw std::runtime_error(why);
        }
    }

    const Copy& body_copy(size_t index) const {
        if (!exec_) throw std::runtime_error("Segment: closed");
        if (index >= copies_.size()) throw std::out_of_range("Segment: no such copy in the body");
        return copies_[index];
    }

    void release() {
        const OnDevice guard(device_);
        if (exec_) cudaGraphExecDestroy(exec_);
        if (parent_) cudaGraphDestroy(parent_);
        exec_ = nullptr;
        parent_ = nullptr;
        copies_.clear();
    }

    int device_;
    cudaGraph_t parent_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    std::vector<Copy> copies_;
    std::mutex lock_;
};

void ring_shape(int64_t slots, int64_t slot_bytes) {
    if (slots <= 0) throw std::invalid_argument("a ring has at least one slot");
    if (slot_bytes <= cozy_engine::kHeader || slot_bytes % 16)
        throw std::invalid_argument("a ring slot is a multiple of 16 bytes, more than its 16-byte header");
}

void report(uintptr_t ring, int64_t slots, int64_t slot_bytes, uintptr_t counter, uintptr_t src, int64_t nbytes,
            uintptr_t stream) {
    ring_shape(slots, slot_bytes);
    if (!ring || ring % 16) throw std::invalid_argument("report: the ring is not 16-byte aligned");
    if (!counter || counter % 8) throw std::invalid_argument("report: the counter is no aligned int64 pointer");
    if (nbytes < 0 || nbytes > slot_bytes - cozy_engine::kHeader)
        throw std::invalid_argument("report: " + std::to_string(nbytes) + " bytes do not fit a " +
                                    std::to_string(slot_bytes) + "-byte slot");
    if (nbytes && !src) throw std::invalid_argument("report: no source");
    check("report's launch", cozy_engine::report(reinterpret_cast<void*>(ring), slots, slot_bytes,
                                                 reinterpret_cast<unsigned long long*>(counter),
                                                 reinterpret_cast<const void*>(src), nbytes,
                                                 reinterpret_cast<cudaStream_t>(stream)));
}

// The records `report` writes, read as a seqlock: a slot's sequence number (acquire), its record, then the
// number again; a record is taken only when both reads are the one expected.
class Reader {
  public:
    Reader(uintptr_t ring, int64_t slots, int64_t slot_bytes)
        : ring_(reinterpret_cast<const unsigned char*>(ring)), slots_(uint64_t(slots)),
          slot_bytes_(uint64_t(slot_bytes)) {
        ring_shape(slots, slot_bytes);
        if (!ring || ring % 16) throw std::invalid_argument("Reader: the ring is not 16-byte aligned");
    }

    // Every record since the last poll, in order. Records the device overwrote before they were read end a
    // poll's records; the next poll raises OverflowError naming them, and the poll after resumes past them.
    nb::list poll() {
        nb::list out;
        const uint64_t room = slot_bytes_ - uint64_t(cozy_engine::kHeader);
        std::vector<unsigned char> record(room);
        for (;;) {
            const unsigned char* slot = ring_ + ((next_ - 1) % slots_) * slot_bytes_;
            const volatile uint64_t* head = reinterpret_cast<const volatile uint64_t*>(slot);
            const uint64_t seq = head[0];
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq < next_) break;  // not written yet (0: being written)
            if (seq == next_) {
                const uint64_t nbytes = head[1];
                if (nbytes <= room) std::memcpy(record.data(), slot + cozy_engine::kHeader, nbytes);
                std::atomic_thread_fence(std::memory_order_acquire);
                if (head[0] != seq) continue;  // overwritten as it was read: look again
                if (nbytes > room) throw std::runtime_error("Reader: a record longer than its slot");
                out.append(nb::bytes(reinterpret_cast<const char*>(record.data()), nbytes));
                ++next_;
                continue;
            }
            if ((seq - next_) % slots_)
                throw std::runtime_error("Reader: slot " + std::to_string((next_ - 1) % slots_) + " holds record " +
                                         std::to_string(seq) + ", which `report` never writes there");
            if (out.size()) break;  // hand over what precedes the gap first
            // Gone: every record a full ring older than the newest the device has written.
            uint64_t newest = seq;
            for (uint64_t i = 0; i < slots_; ++i) {
                const uint64_t held = *reinterpret_cast<const volatile uint64_t*>(ring_ + i * slot_bytes_);
                if (held > newest) newest = held;
            }
            const uint64_t first = next_, last = newest - slots_;
            next_ = last + 1;
            throw std::overflow_error("Reader: records " + std::to_string(first) + " to " + std::to_string(last) +
                                      " were overwritten before they were read");
        }
        return out;
    }

    uint64_t next() const { return next_; }

  private:
    const unsigned char* ring_;
    uint64_t slots_, slot_bytes_;
    uint64_t next_ = 1;
};

}  // namespace

NB_MODULE(_engine, m) {
    m.attr("HEADER") = cozy_engine::kHeader;
    nb::class_<Segment>(m, "Segment")
        .def(nb::init<uintptr_t, uintptr_t, int>(), "step"_a, "go"_a, "device"_a)
        .def("launch", &Segment::launch, "stream"_a)
        .def("close", &Segment::close)
        .def_prop_ro("copies", &Segment::copies)
        .def("set_copy", &Segment::set_copy, "index"_a, "dst"_a, "src"_a, "nbytes"_a)
        .def("enable_copy", &Segment::enable_copy, "index"_a, "on"_a);
    m.def("report", &report, "ring"_a, "slots"_a, "slot_bytes"_a, "counter"_a, "src"_a, "nbytes"_a, "stream"_a);
    m.def("kinds", &kinds, "graph"_a);
    m.def("memset", &memset_async, "ptr"_a, "value"_a, "nbytes"_a, "stream"_a);
    nb::class_<Reader>(m, "Reader")
        .def(nb::init<uintptr_t, int64_t, int64_t>(), "ring"_a, "slots"_a, "slot_bytes"_a)
        .def("poll", &Reader::poll)
        .def_prop_ro("next", &Reader::next);
}
