# Fragmented MP4 audio clocks

The streaming writer previously disabled edit lists and shifted submitted PCM
timestamps by 1024 samples. That avoided stretching the first video frame, but
made AAC encoder priming part of the decoded soundtrack. A 165600-sample, 32 kHz
input became 166624 valid decoded samples. H3's 5-second assembly rejected the
added frame; the package's clock guard was correct.

The writer now preserves the zero-based PCM timestamps, delays the initial MP4
metadata until the first fragment, and enables edit lists. FFmpeg can then signal
encoder priming without moving the video clock. Runtime's output probe requires
the exact submitted audio sample count. Decoder rules and package tolerances are
unchanged. Older files without priming metadata are not reinterpreted.

The CPU regression uses real Outputs and MediaDecoder services at 32 and 48 kHz,
with 5- and 15-second timelines plus H3's aligned frame/audio geometries. It checks
interleaved and late audio, exact decoded sample counts and timestamps, fragmented
writes before EOF, and sample correlation to catch a false fix that cuts the tail
while retaining leading priming.

FFmpeg describes the required metadata behavior in its
[delay_moov implementation note](https://ffmpeg.org/pipermail/ffmpeg-cvslog/2015-January/084883.html).
The owned H100 failure remains recorded as
[packages-v2 issue 227](https://github.com/cozy-creator/packages-v2/issues/227).
