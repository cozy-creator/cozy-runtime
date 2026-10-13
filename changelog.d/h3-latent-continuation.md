H3 continuation carries the previous segment's denoised latents instead of re-encoding its
delivered RGB and audio: a picture window of whole VAE cycles and/or the last second (40 ticks)
of sound, chosen independently per segment (`Carry`). The sound ends on the new clip's timeline
exactly where it ended on the predecessor's; segments deliver their whole sample after the
regenerated head. Context schema 3 (≤ 16 MiB); `H3Model.export_completed_av_tail` is replaced
by `OfficialH3Pipeline.export_context`, called after denoise.
