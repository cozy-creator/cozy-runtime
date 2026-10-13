# Release GPUs early

A request holds its GPUs (an allocation, as in Slurm) from its first GPU call until it
ends, so the gaps between its calls never let another request in. Requests wait in
acceptance order. Before a long CPU-only stretch, usually the end, say so:

```python
async def long_form(ctx: Context, payload: Input, out: Outputs) -> Output:
    for shot in payload.shots:
        videos.append(await render_shot(...))
    ctx.release_gpus()              # the next request's GPU work starts now
    video = assemble(videos, out)   # CPU work, without GPUs
    ...
```

- The allocation ends at once and its GPUs go to the oldest waiting request. The request
  keeps running.
- A GPU call still in flight keeps its GPUs until it returns.
- A GPU call awaited later waits in line at the request's original priority (its
  acceptance) and never preempts: it starts when the GPUs it needs are free. Release only
  before a long stretch; a short gap given away can cost a whole younger request's run.
- Idempotent. Where there is no allocation it does nothing: a request that made no GPU
  call, a nested job, or a GPU handler (its GPU ends when it returns).
- The allocation is the machine's, in memory: a pause, cancel, failure or machine restart
  ends it, and `cozy run show` lists each grant, wait and release.
