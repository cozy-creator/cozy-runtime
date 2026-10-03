A GPU group's followers obey their own GPU's memory cap (`rank_cells/1`). When the owner sets
`Start.rank_cells`, rank 0 creates one budget cell per follower at formation, hands it to the
owner (`BudgetCell.rank`) and to the follower with its `join`; each follower's weight plane reads
its own cell, so a floor breach on any GPU lowers that GPU's process. `Load`, `Budget` and
`Invoke` take `rank_caps` (one cap per rank). Owners that set neither are unchanged.
