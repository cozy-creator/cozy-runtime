MiniMax H3's intake attention-length statement no longer runs on a detached daemon thread. Each
pipeline runs statements on one non-daemon worker: a request settles its own before its first
DiT step, so the statement's kernel row lands in that request; a newer statement cancels an older
one still queued; `OfficialH3Pipeline.close()` cancels what is queued and waits for the running
one. A short-lived process that exits with statements in flight (the H3 reference-fidelity proof)
now exits 0 instead of aborting (134) while torch finalized beneath a statement.
