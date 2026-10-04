# Private LoRA regions and Degree2 custody

[Tracker #320](https://github.com/cozy-creator/tracker/issues/320) records this separate
GPU comparison arm; it is not part of CPU-qualified Runtime candidate6e3cc11e.

The inherited U branch's two-line idea is supported by source: per-step regions have
`bakes` but are not `baked`, so the old export test admits them. Export immediately
makes the producer mapping read-only/shared. When custody refuses or sees a duplicate,
Runtime releases the exported mapping; another stage then refills it. This explains a
possible repeated-copy mechanism but does not independently prove the reported134GB
movement or2.4x slowdown.

Keep any potentially LoRA-mutated region private in both strategies. Also exclude
already baked/dirty bytes so a removed hook cannot make a modified region appear to
be the canonical layout. Plain regions stay attachable/exportable. This is a tactical
whole-region choice, not speculative segmented copy-on-write or a fused kernel.

Seven CPU metadata tests cover the actual shared selection predicate and the per-step
red arm. Required rental sequence: none→baked→per_step→per_step→baked at unchanged
inputs and seed; compare per-step output with a fresh per-step control, record mode,
misses/H2D bytes/operator time, and verify plain regions still share. Compare this arm
against candidate6e3cc11e, then combine only if correct and the copied bytes/time improve.
