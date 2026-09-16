# Progress for Python composition

An ordinary function call uses the caller's telemetry. A managed child has its own
attempt and progress journal. Runtime returns that child's latest measured progress
with the existing local call poll, and the caller relays it into its own telemetry.
No additional workflow language, protocol field or result identity is introduced.

Use a scope to name one part of a known plan:

```python
with tel.scope("Shot 2 of 7", overall_range=(0.2, 0.4)):
    result = await render_shot(payload=shot)
```

The child can report `denoise`, step 3 of 8, and its own overall fraction. The parent
reports `Shot 2 of 7 / denoise`, preserves the child's step count, and maps the
fraction into 0.2–0.4. Scope metadata is captured when the call is reserved, so
other asynchronous tasks cannot relabel it. Repeated polls of the same journal
sequence emit no duplicate progress. Retry uses only the child's current attempt;
old attempts, unrelated parents and other owners cannot supply progress.

Scopes also map ordinary local telemetry, including assembly callbacks. Existing
`tel.stage` timing brackets keep their current semantics. A scope closes at its
endpoint only after successful exit; a failed scope adds no completion credit.
Without a declared scope range, child stage/step progress remains visible but does
not become whole-request progress. Fractions represent authored planned work, not
elapsed time or an ETA. Concurrent scopes keep their labels and bounds isolated;
authors must not use ordered ranges to imply completion of unfinished parallel work.

Preparation begins with a visible stage. "Preparing model files" includes checking
cached files; "Downloading model" appears only while the actual TensorFS pull runs.
"Checking model compatibility" describes the model/placement checks, and "Checking
model inputs" describes input admission. None of these names claims that GPU model
loading or denoising has begun. Timing logs mark whether a phase completed or raised.

The parent and child journals remain available for reconnecting clients. The
managed-call progress relay currently consumes machine-owned children; progress
for calls dispatched to another machine remains the external coordinator's owner.
This is a display feature, not a checkpoint, scheduling or memoization boundary.

Tracking: [Runtime #552](https://github.com/cozy-creator/cozy-runtime/issues/552),
[H3 #242](https://github.com/cozy-creator/packages-v2/issues/242), and
[Creator #594](https://github.com/cozy-creator/cozy/issues/594).
