Torch's allocator splits every cached block (no `max_split_size_mb`), so a capture needs no allocator switch: H3 Turbo 15 s on a 4090 is 13.3 s faster, byte-identical.
