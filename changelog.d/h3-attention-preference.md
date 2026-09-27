`Model.choose_attention` may return an ordered preference instead of one name: Runtime takes,
per site and per rank, the first kernel the site, device, environment and numerical probe admit,
and records why each earlier one was skipped; a pin still overrides it. MiniMax H3 prefers
`sol-attn`, `sageattention`, `flash-attn3`, `sdpa` for both DiTs (owner decision after H100
Ulysses-4 runs 1240-1252); `flash-attn3-fp8` is never auto-selected and the VAEs keep Runtime's
ranking. A group now serves rank 0's prepared selection: each follower takes rank 0's kernel per
component before the group installs, and a rank that cannot serve it refuses the prepare as
`rank_divergence` instead of every later request refusing.
