H3 continuation carries the previous segment's denoised latents instead of re-encoding its
delivered RGB and audio: a picture window of whole VAE cycles and/or the last second (40 ticks)
of sound, chosen independently per segment (`Carry`); sound alone rides a one-second regenerated
head. The sound ends on the new clip's timeline exactly where it ended on the predecessor's;
segments deliver their whole sample after the regenerated head. Context schema 3 (≤ 16 MiB); `H3Model.export_completed_av_tail` is replaced
by `OfficialH3Pipeline.export_context`, called after denoise.
The `media` extra requires PyAV 19.0.1 or newer: video orientation still fails on 18.1.
