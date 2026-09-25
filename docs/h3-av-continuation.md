# H3 completed-segment AV continuation

This is a model-local implementation and qualification plan. GPU continuity, speech,
long-take drift and Turbo quality remain unqualified until matched real renders pass.
It does not checkpoint an unfinished denoising loop. The application owns private
artifact persistence; Runtime owns the bounded AV representation and inference math.

## Source audit (2026-09-25)

Director source was pinned to
[`4b4fe6e1a7e68b17f4a36c8b3e942667defab61f`](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/tree/4b4fe6e1a7e68b17f4a36c8b3e942667defab61f).
Its default guide mode carries a previous segment's AV tail as fixed conditioning.
Supported context windows are 5, 22, 39 and 56 frames; default22 is about0.92 seconds
at24fps. Its nearest-window selector maps32 to39. Same-size latent reuse is preferred;
decoded pixels can be re-encoded when the canvas changes. A phase-aligned latent
window may end before the previously exported end, requiring a prior-tail trim.
See [motion context](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/4b4fe6e1a7e68b17f4a36c8b3e942667defab61f/director/h3_motion_context.py).

This is not merely adding a video reference. The Director changes interior keyframe
rotary times, adjusts them for the target's origin after reference blocks, and merges
keyframe and reference latent payloads. Its patches refuse incompatible foreign wrappers.
The internal layout self-test checks stock endpoints and increasing interior times;
it is not a visual continuity benchmark.
See [layout/payload patches](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/4b4fe6e1a7e68b17f4a36c8b3e942667defab61f/director/h3_context_patches.py).

The executor first builds official Comfy conditioning, retaining fixed references in
Ref2VA. It adds motion context afterward, samples extra frames, removes the generated
context prefix, and crops to the intended export length. Its independent latent-continue
mode instead writes previous AV into the target latent and applies masks/remasking.
That second algorithm must not be confused with fixed conditioning or silently combined
with it. We do not adopt its redraw, SelfLift, upscaler, face-refine or semantic-bridge
features. See [executor](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/4b4fe6e1a7e68b17f4a36c8b3e942667defab61f/director/executor_core.py)
and [latent-continue](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/4b4fe6e1a7e68b17f4a36c8b3e942667defab61f/director/h3_latent_continue.py).

Director's guide mode can pin generated audio; its default audio window is24 frames
independently of the22-frame video window. Its audio handling uses40 latent ticks/s,
32kHz waveform output, aligned prefix removal and per-segment trim/pad before
concatenation. That establishes a timing mechanism, not guarantees of uninterrupted
speech, stable voice or a seamless musical phrase.
See [audio assembly](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/4b4fe6e1a7e68b17f4a36c8b3e942667defab61f/director/audio_export.py).

It uses FL2VA for text/first/last-frame tasks and Ref2VA for persistent multimedia
references. There is no special Director checkpoint. Its generic sampler can accept a
Turbo-patched model, but its default25-step res_multistep recipe does not prove parity
with our separately qualified eight-evaluation PDD bank. The only checked-in Actions
workflow found publishes the node; no end-to-end GPU acceptance suite was found.
See [sampling](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/4b4fe6e1a7e68b17f4a36c8b3e942667defab61f/director/core_sampling.py)
and [publisher](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/4b4fe6e1a7e68b17f4a36c8b3e942667defab61f/.github/workflows/publish.yml).

Easy Media's separately reviewed
[`ed20432dcb744895c94c8b5b0eed90f36234a55b`](https://github.com/yolain/ComfyUI-Easy-Media/tree/ed20432dcb744895c94c8b5b0eed90f36234a55b)
provides another approach: reserve context space and rebuild the next carry from the
actual delivered span, avoiding discarded future frames. Its optional hard-prefix masks
are a separate comparison arm, not evidence that every conditioning-only path is equivalent.
See [its native context implementation](https://github.com/yolain/ComfyUI-Easy-Media/blob/ed20432dcb744895c94c8b5b0eed90f36234a55b/modules/motion_context/core.py).

## Native API and ownership

`continuation.py` exports `plan_continuation`, `ContinuationPlan`, `AVContext`,
`encode_context`, `decode_context` and `audio_sample_range`. Contexts hold22/39/56
RGB8 frames and exactly aligned stereo32kHz PCM, bounded by256MiB including the
safetensors carrier. They record their source frame interval, schema and canonical
renderer provenance. Consumers must compare current model/base-adapter manifests,
native inference-code identity and relevant software. Parent workflow/script identity
is not a model compatibility fence.

`OfficialH3Pipeline.start_ref2va` accepts optional `context`, `delivery` and
`expected_context_provenance`. With no delivery/context its old path remains unchanged.
Persistent user references are encoded normally. The private temporal condition is
added after text encoding, so it is not a new numbered Picture/Video/Audio reference.
Native VAE encoding and native packed layout supply its rows; their times are aligned
to the target prefix. Native scheduling holds those condition rows fixed. This is the
Director-style guide path, not a claim that target-prefix pixels are hard-copied.

The video decoder streams only the new delivery window and retains a ring of its last56
RGB8 frames. Audio uses integer frame-to-sample boundaries. `export_completed_av_tail`
is valid only after complete video and audio decode. The package serializes that bounded
context as a private child artifact. Only its final MP4 and final PNG are public outputs.
Re-encoding the actual delivered tail avoids latent phase/overshoot mistakes; retaining
native latent state is a later optimization requiring separate custody and parity proof.

## Duration contract and gates

Requested duration counts NEW frames, excluding context and VAE padding. Native sampled
frames remain on the17n+5 grid and at or below362. For240 delivered frames (10 seconds):

| Context | Sampled | Prefix discarded | Tail padding discarded | Delivered |
| --- | --- | --- | --- | --- |
| none | 243 | 0 | 3 | 240 |
|22 |277 |22 |15 |240 |
|39 |294 |39 |15 |240 |
|56 |311 |56 |15 |240 |

A15-second new segment plus22 context needs396 sampled frames and is refused. Likewise,
13 seconds plus56 needs379. No overlapping frames count twice, no silent shortening,
and no expansion of the qualified model envelope. A first live qualification can use
five10-second segments with identical refs/prompts/seeds, comparing22/39/56 contexts.

CPU gates cover codec refusal, exact audio clocks, no padding leakage, complete-only
exports, reference-preserving native rotary layout and unchanged conditioning at each
real scheduler step. Required GPU gates remain: real VAE/model preparation, standard
and PDD Turbo comparison,2/4-GPU equivalence, seam motion and identity, spoken-audio
continuity, exact output frame/sample counts, cancellation/retry with private artifact
custody, and at least45 seconds of a coherent continuous take. Current cuts remain an
independent camera-cut choice, not an implementation of motion continuation.
