# Prepared input failure and checked completion

A failed result and unfinished GPU work are different states. A native storage read can fail
after TensorFS has proved producer/continuation drain and released its private GPU ownership.
That request must preserve the storage error without retiring a healthy executor.

`Execution.completion_proven()` observes this independently of the result. `poll()` and `wait()`
continue raising the original operation error. The request scope uses the completion certificate
only for errors that are neither sticky nor device faults; older providers without this
observation keep the conservative polling path.

Completion is published after every model-use claim is released. Each claim stays in the owned
list until its release succeeds. Normal output copies/readout synchronization sit outside the
input-error handler: a successful input certificate cannot certify a failed output readout.
An operation whose completion cannot be proved retains the existing executor-retirement behavior.

This distinction does not retry a submitted computation or implement storage re-fetch/requeue.
Source availability refusal before submission remains separate from a source failure afterward.

## Proof

Real default-home CLI requests on RTX4070 Laptop8GiB, Torch2.14.1+cu130, on2026-10-26:

| Cases | Runs | Observed result |
|---|---|---|
| Baseline cold loading |18580|Exact eager parity, successful native execution|
| Post-capture input read failure, events/checked |18581,18583|Original StorageError; Runtime/native completion certificates true; zero graph launches; zero model users; healthy native memory and initial pinned-byte count|
| Successors after those failures |18582,18584|Correct inference on the same executor PID1341383|
| Stale source/layout refusal |18585,18587|Zero-launch request failures; no executor retirement|
| Successors after stale refusal |18586,18588|Correct inference on PID1341383|
| SDXL source unavailable before preparation |18589|Pre-submit fallback signal, without submitting work|
| SDXL source truncated after capture |18590|Original StorageError, checked completion and cleanup, no eager retry after submission|
| SDXL successor |18591|Exact eager parity on PID1341383|
| Real CUDA replay assertion |18600|Actual Indexing.cu indexSelectSmallIndex assertion; machine observed Xid43 and ended PID1359050|
| Successor after GPU fault |18601|Correct native inference on replacement PID1359519, same installed SDK|

The first12 gates passed together: seven requests completed successfully and five expected
request failures had their refusal/cleanup outcomes verified. The two GPU-fault gates passed
in a separate window on the same frozen cohort. The machine observer ended the unsafe executor
before Python's error handler reported a completion certificate. This proves system-level fault
retirement and replacement, **not** an observed false Runtime certificate or that observer's
own device-error branch. No missing telemetry is interpreted as a completion result.

The real fault used a graph captured with a valid torch.index_select input, then changed that
device input out of bounds before replay. Storage failures modified only independent private
checkpoint copies; production/cache-store chunks were never changed. Request-scope admission,
native execution and ordinary CLI remained in the path.

The exact tested Runtime source was `80793e9ce1ff725e5f0aa80938bbd2da14859323`. Its full source tree
is identical to the subsequently rebased `c42e28733eac7ed9c66209fb07d4bf4a789ac96f`:
`9f3419a242850cfaf54426f4241e7df2bba29a4b`. The added proof document does not change executable code.
Paired Creator/TensorFS source was6862dee81. Artifact SHA256:

- Runtime: `6836678e51bcfc02480236538c4c7410c909b84057a6db3373e04623f6764aa4`.
- TensorFS: `0673680c2286481163b873b91aaf8e6194afdc706f6ac4ae063fd24f0cd74a85`.
- Cozy: `dbe800873e4e0fcfad1fe3f6182c664afb4e6054af19d2f96040a1f9aaad009a`.

Hosted pure build40356122481 and the repository wheel recipe passed. All216 packaged source
files and embedded commit provenance matched; all46 native/build inputs matched the previously
qualified4d carrier and its four native modules were unchanged. All229 native-wheel RECORD
entries verified. Actual Runtime/TensorFS/native imports passed before GPU qualification.

Workspace receipts are under `outputs/stock-torch-20261025/codex-resume-20261026/`:
`completion-{session,cells}-v2` and `completion-{session,cells}-fault-v4`. Independent reviewed
receipts and hashes are in adjacent `loading-proof-independent-20261026/GPU-REVIEW-V2.json`
and `GPU-REVIEW-FAULT-V4.json`. Exact prior CLI/SDK hashes and idle state were restored after
both windows (16:18:51 and16:24:22UTC respectively).

Earlier failed fixture/controller receipts remain preserved: oversized telemetry was clamped
in the first window; a Cozy Device-to-torch conversion mistake prevented the first fault
attempt; a later genuine fault exposed the machine observer's earlier retirement boundary.
None was relabeled as a passing Runtime fault-observer test.

This is focused lifecycle qualification. It is not a full-model throughput benchmark,
generic oversized streaming proof, async source re-fetch proof, or overall Varena completion.
