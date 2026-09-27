# Release GPUs early

A request holds its GPUs from its first GPU call until it ends, so the Python gap
between two calls never lets another request in. When the rest of the request is CPU
or network work, say so:

```python
async def long_form(ctx: Context, payload: Input, out: Outputs) -> Output:
    for shot in payload.shots:
        videos.append(await render_shot(...))
    ctx.release_gpus()              # the next request's denoise starts now
    video = assemble(videos, out)   # CPU work, without GPUs
    ...
```

- The request's lease ends at once and its GPUs go to the oldest waiter on the next
  scheduling tick. The request keeps running.
- A GPU call still in flight keeps its GPUs until it returns; they then go to the pool.
- A GPU call made afterwards is a new demand. It waits in line at the request's
  original priority and starts a new lease; it never preempts whoever was granted the
  GPUs meanwhile. Call `release_gpus()` again when that work is done.
- Idempotent. Where there is no lease it does nothing: a request that made no GPU call,
  `warm`, or a GPU handler (its GPUs end when it returns).
- A request that is itself one GPU call (a direct serving or device-job request) needs
  nothing: its lease ends at its device exit, before its outputs are encoded and sent.
- The lease is in memory only. A worker restart, a retry, a pause or a cancel ends it
  anyway; the next attempt starts with no lease and holds normally until it releases.

The root's journal records `gpu.lease {ordinals: [], cause: "released_by_root"}` (or
`"device_exit"`), and its `MachineExecutionState.gpu.phase` becomes `none`.

Requires a `cozy-runtime` newer than 0.18.55 in the package. A worker older than the SDK
declines the release and the lease ends with the request, as before.

Deferred: running other queued work inside a request's gaps between GPU calls.
