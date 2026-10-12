# Retained H3 generation and refinement

`cozy_runtime.models.minimax_h3.refinement` defines `H3Latents`, a completed
generation with normalized video latents, clean audio latents, decoded stereo PCM,
text embeddings/tags, and encoded reference conditioning. It contains no raw prompt,
model weights or process-local handles. Ordinary `Outputs.save_bytes` and `FileAsset`
carry `encode_latents(value)` / `decode_latents(bytes)`; the carrier is bounded to 2 GiB.

After base denoising and audio decoding, capture once:

```python
from cozy_runtime.models.minimax_h3.refinement import capture_latents, encode_latents

retained = capture_latents(task, state, checkpoint=model.checkpoint_ref, seed=actual_seed)
artifact = out.save_bytes(encode_latents(retained), media_type="application/octet-stream")
```

The learned upscaler consumes the already normalized `1×24×T×H×W` video directly.
It must preserve time, audio, PCM and conditioning. FL2VA's spatial keyframe latents
in `conditions` must be upscaled to the same target dimensions; Ref2VA references
keep their own geometry. Use `dataclasses.replace` to make an independent result.
`height` and `width` derive from the new video tensor and are on the 32-pixel grid.

Refinement needs only the corresponding H3 DiT and video VAE, with no new text or
reference encoding. `H3Model.refine_fl2va` / `refine_ref2va` take the retained value
and keyword arguments `generator`, `on_step`, `cancel`, `checks`, and `steps=2`.
They return `(state, ScheduleFacts)` for the ordinary decode path.

Refinement always uses a suffix of the standard 30-step schedule, even if the base
generation was Turbo. The default two evaluations start at video sigma 0.461538 and
continue through 0.292683 to zero. They use the already available modulation-table
rows; no off-grid schedule or Turbo overlay is invented. Audio rows condition at the
clean timestep and remain fixed between forwards. `decode_audio` returns the retained
PCM, so refinement cannot rewrite the soundtrack.

`model.decode_video(..., output_size=(width, height))` optionally applies a standard
Torchvision bicubic, antialiased fit and zero padding to each decoded chunk. It preserves
aspect ratio and frame order and can deliver exact 1080p/1440p/2160p dimensions after
sampling an aligned internal canvas. This is final canvas fitting, not the learned
upscaler or the detail-producing refinement pass.

Keep a single-GPU learned upscaler in a separate child operation. Adding its model
slot to H3's multi-GPU construction would restrict that construction's declared width.
An omitted global GPU count allows each child to choose its supported width; an explicit
count remains exact for every GPU child.

CPU tests execute real upstream blocks and both full and AdaLN-pruned transformer
tables for text-only, first-frame and reference workflows. They cover the two-step
schedule, unchanged audio, input immutability, artifact validation and output fitting.
They do not establish high-resolution GPU memory fit or visual quality; those require
the complete package workflow on the chosen hardware.
