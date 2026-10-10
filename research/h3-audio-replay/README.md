This research entrypoint replays a saved normalized H3 audio latent without
conditioning, denoising, or video decoding. Its single model slot selects only
`audio_vae` from the same published H3 base lane and invokes the existing
`H3TurboBase.decode_audio` component method. Official mean/std normalization and
decoder arithmetic remain in that method.

The experiment makes three baseline calls, then three calls inside PyTorch's
`cudnn.flags` context with `deterministic=True` and `benchmark=False`. Every other
context argument—including `enabled`, `allow_tf32`, precision, benchmark limit,
and depthwise kernel choice—is preserved explicitly. Flags before, during, and
after the context are recorded; the context restores them even on exceptions.
No production default or persistent numerical setting changes.

Input is an existing raw latent FileAsset with exact SHA256, shape, dtype and
byte-order metadata. Six bounded raw FP32 waveform assets and a JSON report
retain repeat identity, input custody, finiteness, timings, memory and device
evidence. The first sample in each case may contain cold setup work; these are
decoder measurements, not video-generation benchmarks.

Selecting only `audio_vae` prevents other components from executing or becoming
resident for this entrypoint. The pinned TensorD may still complete the remaining
checkpoint in the background, so this declaration alone does not guarantee that
only audio files transfer from storage.

CPU controls verify the public model/component interface, the minimum official
decode block state, exact latent reconstruction, corruption rejection, six-call
input custody and flag restoration on success and failure. They are not GPU
determinism or performance proof.
