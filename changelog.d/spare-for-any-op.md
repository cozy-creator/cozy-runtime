An op that runs out of device memory by itself (a block's first run, a kernel between blocks) now
also has the stage's components that are not in use give their weights back, when unmapping to
the stage's floor gave nothing: every component but those the op reads, those with a block in its
forward and the one whose block ran last (runs 4360-4369: Anima in a 1 GiB pool on an RTX 3070).
A weight set closes after its pinned tier gave back regions its fill had asked for; that was
reported as `region 80 left the pinned tier` and failed the run (run 4365).
