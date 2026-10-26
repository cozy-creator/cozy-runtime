#ifndef COZY_EXECUTION_V1_H
#define COZY_EXECUTION_V1_H
#include <stdint.h>

/* Private same-process capability, exported as this exact named PyCapsule. */
#define COZY_EXECUTION_CAPSULE_V1 "cozy.execution.continuation.v1"
#define COZY_EXECUTION_ABI_V1 1u
#define COZY_EXECUTION_NATIVE_ONLY 1ull
#define COZY_EXECUTION_CALLER_READY 2ull
#define COZY_EXECUTION_RETAINED_OWNERS 4ull
#define COZY_EXECUTION_REQUIRED_FLAGS 7ull
#define COZY_EXECUTION_EVENT_INPUT 8ull
#define COZY_EXECUTION_REUSABLE_INPUT 16ull

/* Every function returns a value; no exception/unwind crosses this C ABI. */
#define COZY_EXECUTION_OK 0u
#define COZY_EXECUTION_CANCELED 1u
#define COZY_EXECUTION_INVALID 2u
#define COZY_EXECUTION_CAPACITY 3u
#define COZY_EXECUTION_CUDA 4u
#define COZY_EXECUTION_INTERNAL 5u
#define COZY_EXECUTION_UNPROVEN 6u

typedef struct CozyExecutionStatusV1 {
    uint32_t kind;
    int32_t code; /* CUDA code when applicable; 999 for internal/unproven failures. */
    uint32_t detail_len;
    uint8_t detail[244]; /* bounded UTF-8 diagnostic; never an allocated foreign string */
} CozyExecutionStatusV1;

#define COZY_EXECUTION_ARMED 0u
#define COZY_EXECUTION_RUNNING 1u
#define COZY_EXECUTION_DONE 2u
#define COZY_EXECUTION_FAILED 3u

typedef struct CozyExecutionProgressV1 {
    uint32_t phase;
    uint32_t completed_steps;
    uint32_t canceled;
    uint32_t done;
    uint64_t epoch;
} CozyExecutionProgressV1;

typedef CozyExecutionStatusV1 (*CozyExecutionOpV1)(void *context);
typedef CozyExecutionStatusV1 (*CozyExecutionObserveV1)(
    void *context, CozyExecutionProgressV1 *out);
typedef CozyExecutionStatusV1 (*CozyExecutionStartAfterV1)(
    void *context, const uintptr_t *streams, uint32_t count, uint64_t error_device_ptr);

typedef struct CozyExecutionContinuationV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t flags;
    int32_t device;
    uint32_t reserved; /* zero */
    uint64_t pid;
    void *context; /* opaque: allocated, retained and released only by these functions */
    CozyExecutionOpV1 retain;
    CozyExecutionOpV1 release;
    CozyExecutionOpV1 start;
    CozyExecutionObserveV1 poll;
    CozyExecutionOpV1 cancel;
    CozyExecutionObserveV1 wait;
    /* Optional tail: require EVENT_INPUT and a struct_size covering this pointer. */
    CozyExecutionStartAfterV1 start_after;
    /* Optional REUSABLE_INPUT tail. Checked DONE -> ARMED; cancellation stays sticky. */
    CozyExecutionOpV1 rearm;
    /* Checked DONE -> terminal; releases definition reservation deterministically. */
    CozyExecutionOpV1 finish;
} CozyExecutionContinuationV1;

/* Import checks capsule name, ABI, full struct size, required flags, pid and device.
   Retain once before storing context; release exactly once after checked completion.
   Keep the provider capsule/module alive through all calls, including release.
   start/cancel linearize through an atomic gate: cancel is nonblocking (no mutex, driver
   or Python call). Before start it prevents every GPU launch; after start it sets a flag
   which the native worker wait/poll delivers after launch initialization. wait never
   holds the observation mutex while parked.
   start is one-shot. poll/wait before start report ARMED; cancel-before-start reports
   DONE/canceled without a GPU launch. Read failure must cancel before releasing.
   UNPROVEN or CUDA failure requires retaining actual read/Fill/Pin/graph owners until
   checked cleanup succeeds or process exit. No Rust types or Python callbacks are in this ABI.
   Caller-ready is recorded before export on the submitting OS thread. start uses an
   explicitly owned native CUDA stream; callers must wait for composed completion. */
/* start_after copies 1..4 native producer stream handles during the call and records
   owned events after all source work was enqueued. No PTDS/legacy stream handles.
   The caller retains the destination, producer resources and optional i32 device
   error word through checked composed completion. Zero means no decoder error word.
   An encoded error may submit a graph but skips its complete authored body, including
   index reset and prefix. Late device/ownership failures do NOT prove zero execution:
   discard outputs, retain uncertain owners and retire. cancel before enqueue commitment
   prevents submission; cancellation after commitment is cooperative execution cancel. */
/* REUSABLE_INPUT exclusively reserves the graph definition until this capability is
   released. Rearm requires checked completion and preserves the first caller-ready
   event. It never drops storage pins. Native banks may change bytes only after their
   previous consumer completed. A non-reusable continuation retains one-shot behavior. */
#endif
