# Cold image VAE planning

Tracker: cozy-creator/tracker#320. Owner: Codex.

The fresh 1.5 GiB rental completed the original 1024-square SDXL request's twenty
denoise steps, then failed decoding: a 512 MiB nearest-upsample output needed more
than the observed 503 MB free. Runtime de40 correctly refused whole-decode replay.
An unknown first decode currently chooses untiled; no completed untiled calibration
exists. The owner's subsequently recorded 1,008,372,224-byte partial peak is also
not a completed untiled calibration. Successful tiled peaks have the same distinction.

Before the first call, estimate native AutoencoderKL working memory from its actual
decoder channels and expansion, activation/parameter dtype, batch and slicing.
Account for overlapping feature buffers, the live latent and accumulated outputs.
This estimate selects a strategy; it neither proves fit nor refuses a request.
Local completed untiled measurements remain separate from owner stage estimates.
Choose tiles before calling decode when the estimate exceeds available plus
evictable memory, remeasure after eviction, and preserve the large-room fast path.
There is one decode invocation and capacity failures retain the existing poison rule.

CPU proof must cover cold native geometry, dtype and slicing, partial/tiled owner
seeds, unchanged large-room behavior and one-call state safety. The original rental
failure remains evidence. A new ordinary attempt must preserve its authored request,
fixed holder and source identity, save an image and meet the declared quality rule.
All twelve 1.5 GiB grouped/alternating requests and smaller pools remain GPU gates.
