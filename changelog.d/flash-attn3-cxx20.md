FlashAttention-3's boot compile builds `flash_api.cpp`, its one unit that includes Torch, as C++20, which Torch 2.14's headers require. The recipe change rekeys the artifact.
