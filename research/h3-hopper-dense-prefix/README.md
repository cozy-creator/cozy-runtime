# Hopper dense prefix partition control

Additive, bounded single-H100 diagnostic using public Runtime0.23.7. It does not change the separately frozen Sage2/CuTe/Kitchen package. Supported explicit choices: `flash-attn3`, `sdpa`, and the experimental `flash-attn3-fp8`; default is FA3 BF16 with 257 query rows and109225 K/V rows. Optional full self-attention uses109225 query rows.

Local head groupings2/4/7/8 are arithmetic controls on one GPU, not real multi-GPU tests. Every call records the actual Runtime implementation. SDPA math selection is reported unsupported at this long sequence length rather than risking a huge attention matrix or changing the selected backend. FA3 already fixes num_splits=1 in the public adapter; this control does not change that policy.

FA3 FP8 has historical quality errors and remains an explicit research choice. Byte-identical partitions would not make it quality-approved. No GPU run or rental is authorized by this source. Normal CLI capture/describe and independent review are required before a paid request.
