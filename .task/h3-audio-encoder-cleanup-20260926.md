Owner: /root/h3_reference_astra
Purpose: extend PR635's computed WeightNorm cleanup to every actual audio-VAE encoding/decoding scope, including Ref2VA conditioning and continuation export.
Branch: fix/h3-audio-encoder-cleanup-20260926
Base: 46d76fb13f22c5b3ba2bc7d205d346099bc1523d (fresh origin/master, PR635 merged)

Read-only checkpoint accounting: 135 decoder weight_v tensors use 259,161,824 FP32 bytes (259,166,720 after individual 512-byte rounding); 37 encoder tensors use 294,651,648 bytes (294,651,904 rounded). No other audio-VAE weight_v families exist. Both conditioning and continuation export use the shared vae_encoder block, and output audio uses decode.audio.

No math, hook, registered parameter or ledger change. Source review only; user prohibits tests and CI. Root owns wheel build and live qualification. Existing five-run capture remains untouched.
