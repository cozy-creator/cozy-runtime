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
#endif
