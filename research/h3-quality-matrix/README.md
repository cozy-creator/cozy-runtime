Build the local quality-review index without modifying the run manifest:

```sh
python3 research/h3-quality-matrix/review_matrix.py MANIFEST.json --out REVIEW_DIRECTORY
```

The manifest contains `runs` with `id`, `kernel`, `schedule`, `scene`, `seed`,
`video`, `run_show`, `status` and `timing_note`. Optional `kernel_label`,
`arithmetic_note` and `attention_selector` make the displayed precision and
observed selector explicit. Root owns the manifest; this helper writes only its
separate review directory.

Completed clips receive a full software AV decode, SHA256, geometry, 362-frame,
24fps, 32kHz stereo and duration checks. The run receipt must establish eight
denoising steps and execution time. On this one-GPU H3 matrix, `dense4-sol4`
requires 192 sparse calls and `dense8` requires zero. Pending/failed/missing
artifacts remain visible and do not count as verified. Existing matching AV
receipts are reused; original videos are hard-linked or copied into `media`.

`index.html` is static and needs no server, framework, network fonts or autoplay.
`index.json` retains all run metadata and check results. AV verification does not
approve visual quality, audio quality, or defaults; the user reviews those.

The manifest owner can execute the matrix serially through the ordinary CLI:

```sh
python3 research/h3-quality-matrix/batch_matrix.py MANIFEST.json
```

This driver mutates the owner's manifest atomically. Do not edit the manifest
concurrently: every manifest writer must hold the same `MANIFEST.batch.lock`
advisory lock, or wait until the driver has stopped. The driver also detects
external changes before writes, but that best-effort check is not a filesystem
compare-and-swap guarantee. Replacement of the complete JSON file is atomic. `--prepare-only` adds missing warmup
rows and prints the plan without invoking Cozy. The order is kernel-major: two
untimed full mixed-schedule scene warmups, then that kernel's four timed cases.
Existing manual `warmups` are matched by kernel/scene/schedule. The manual fields
`events_path`, `result_path`, `request_id` or `run_number` are supported;
`source_run_id` is merely a matrix-cell label, never a Cozy request ID.

Each submission uses literal argv, the pinned input file, its own output directory,
a persistent idempotency key, and `cozy run --await --json`. Intent is persisted
before process creation. Every command retains argv, stdout, stderr, process and
exit receipts. The driver uses precise `cozy run show REQUEST_ID --json`
`execution_ms`; CLI's rounded display duration is not a measurement. Completion
requires the pinned GPU UUID, all eight denoising steps, the exact observed
attention selector, expected sparse-call count, and a retained video with matching
exported bytes/SHA256. Full AV and visual inspection remain separate review steps.

On restart, completed cases are skipped only after rechecking their evidence and
files. Known accepted work is recovered only with `run show` and `run watch`.
An interrupted submission with no request ID stops for owner reconciliation;
the driver never blindly resubmits, including with the same idempotency key.
Any failed/blocked/unknown state or CLI error stops the batch. Inspect the retained
logs, record the actual request ID if recovered manually, then restart this same
command. Do not erase an attempted row to make it pending.

Create `PAUSE` beside the manifest to pause **between requests**; remove it to
continue. Create `STOP` to detach the current CLI observer and stop the batch;
accepted remote work continues. Remove `STOP` before read-only recovery/resume.
SIGINT/SIGTERM also detach. `--observer-seconds` defaults to 7200 and only bounds
observation. The driver deliberately omits Cozy's `--timeout`, which is an active
request-cancellation deadline. It never invokes run cancel or provider commands.
The rental's ordinary keepalive policy remains the owner's responsibility.

CPU-only regression tests use an executable fake CLI with durable request state:

```sh
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover \
  -s research/h3-quality-matrix -p test_batch_matrix.py -v
```

Before each **new** submission, a separate read-only `cozy rental show` receipt
must confirm the manifest's `rental_id`, `runtime_wheel.version`, `tensorfs_version`
and `driver`, plus ready status, one device, and no running/queued requests.
Preflight receipts do not mark a submission attempted; a failed check can be
corrected and retried without entering uncertain-submission recovery. Known
accepted/completed requests never require this fresh-rental check.

Every video, including manual/recovered files, must match the completed canonical
`run show` `output[]` video receipt's SHA256 and length. A `watch` result's `saved[]`
receipt must agree when present; when absent, the canonical output's retained
local path supplies the file. Missing canonical output evidence always refuses.

The manifest also pins `hub_url`; every new submission uses that explicit
`--tensorhub` value and includes it in its persisted request identity.
