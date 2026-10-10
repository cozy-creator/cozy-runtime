Research diagnostic for the H100 audio difference between runs 5331 and 5332.

Add `audio_boundary.py` to the frozen local H3 package and point its application
entrypoint at `audio_boundary:app`. This separate research app exposes only the
diagnostic; the captured `h3.py` and published H3 functions are unchanged.
The new callable uses the same text-only preparation, conditioning and full
eight-step Turbo method, with the same four-dense/four-sparse schedule. It omits
video decode and encoding, so its duration is not a video generation benchmark.

Outputs are the final normalized audio latent, the original canonical FP32
waveform, one same-state decode replay waveform, and JSON metadata. Raw byte
files retain their declared dtype, shape, byte order and contiguous layout.
Latent hashes before and after each decode detect mutation. Replay does not
change attention, precision, cuDNN settings or sampling. It runs inside the
ordinary model component method and ordinary Cozy CLI request.

The final latent is sampled after `denoise.after_denoise`, before the audio
decoder's mean/std denormalization. Equal latent bytes but unequal decoded audio
locate the difference at decoding or its execution settings. Unequal latent
bytes require investigating upstream denoising operations; identical RGB8 video
alone does not prove the final video latents were equal.

CPU controls exercise strided BF16 byte preservation, reuse of the decoder's
waveform buffer, and detection of latent mutation. They do not qualify GPU
determinism or video quality.
