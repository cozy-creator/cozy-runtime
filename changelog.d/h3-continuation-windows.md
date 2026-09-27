H3 continuation exports only the context windows its caller names and reads the tail from the
delivered RGB8 buffer instead of copying every decoded chunk; `export_completed_av_tail` takes
`frames` and `windows`, and an `AVContext` may hold any non-empty window subset. The video VAE
encode spreads its 17-frame clips over the group like decode. PNG outputs encode at zlib level 1
(10-16% larger, 10x faster: level 9 took 2.9 s per 1344x768 frame before an outcome).
