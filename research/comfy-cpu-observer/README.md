Run an existing local ComfyUI server through an ordinary Cozy CPU job:

```sh
cozy run ./research/comfy-cpu-observer/run --rental=RENTAL --input=observer-input.json --await
```

`observer-input.json` supplies `graph_json` (the serialized Comfy API graph),
the worker's `output_root`, optional `port`, `timeout_s` (1–7200 seconds), and
`trace`. Tracing additionally requires the worker's `trace_root`. Graph inputs are sent unchanged;
benchmark callers must set the text-encoder cache options in the graph themselves.
Starting/stopping the server remains separate owned maintenance.
An exclusive benchmark GPU window is still required: this CPU job observes the
external server and does not itself reserve its GPU against other submissions.

The job is explicitly `accelerator=False`, imports no Torch, and waits for actual
Comfy completion. The accepted CPU request remains active on the worker while the
client is disconnected. It does not acquire a GPU parent context or use a dummy
sleep to keep a rental alive. HTTP history polling recovers from a broken event
socket. A caller-provided UUID and resumable Scratch receipt ensure a lost POST
response or resumed observation never blindly resubmits a graph.

The result's `status` describes the **Comfy render**. A completed observation can
report `failed` or `timed_out`; only `success` has an eligible render duration.
The retained tree contains request, acceptance, events, history, summary, output
files, and complete trace files when requested. File copies use bounded chunks.
Profiler timings remain diagnostic. Timeout/cancellation detaches observation;
the observer does not send a global Comfy interrupt or change server/rental state.

Source proof: Runtime `Executor.run_job` initializes Torch/CUDA only when the job's
admitted budget has a positive GPU count. Static/imported package interfaces and
the real CPU execution path must be qualified before rental use. A GPU Comfy render
has not been exercised through this wrapper yet.
