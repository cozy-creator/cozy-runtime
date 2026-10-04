GPU executors stage only what streams (`host_tiers.staged/1`, when the machine says
`Load.staged_tiers`): each sealed layout comes with
nothing staged, its holes read from the verified object files the machine hands over
(`ObjectFiles`, through the page cache), and a stage that streams asks the machine to stage those
regions (`SealedStage`) and pins what was staged. Resident blocks and idle models no longer hold
a sealed host copy. CPU executors keep whole layouts.
