A GPU group states each GPU's own memory: a follower's replies carry its plane facts, and rank
0's replies carry every follower's as `rank_planes` (rank 1 first; null until stated), so the
owner charges each GPU its own process, context and pinned tier.
