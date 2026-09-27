Cancelling a running request no longer kills its executor after a flat 3 s grace. The request
stops at its next safe point (a telemetry step, stage or progress call, or a Ulysses group's
next mirrored call, before any rank is commanded), and the executor and every rank of its group
stay warm for the next request. The executor is killed only when it has shown no progress for
eight times the longest gap between its own frames, and never sooner than the Runtime's noise
floor. The kill is reported with that measurement. `--grace` is accepted and ignored.
