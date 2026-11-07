# Running out of device memory

Paul, 2026-11-07: keep trying while any lever can make it fit, and give up only as the last resort.
varena decides; the Runtime only runs what varena admits or runs again what varena says to. Packages
never catch a refusal.

## Levers, in varena's order

1. Free or evict: completed working pages, idle pools, idle weights, owner caches, idle peers.
2. Spill retained state to its host copy (the host tier's room is the host owner's number).
3. Wait a bounded time (2 s) for memory another process gives back.
4. Split finer: the next level of a cut.
5. Stream instead of hold: run eagerly, weights filled per use, rather than recorded or held.

An allocation runs levers 1–3 before it is refused. What happens next depends on where it was.

## Where each answer comes from

| Where | Asks | Answers |
|---|---|---|
| A cut (FFN tokens, attention pieces, norm and conv bands, a package `split`, `windows`) | `vt.Site.enter` before its pieces; `Leaf.exit` after a refusal inside | the level, its room made; **Retry** (a Runtime-owned pure op runs again at the same level once if room now covers the refusal, else finer) or **Final** (a package's op, or no level fits) |
| A recording (`denoise_loop`, a component's graph, an LM's decode graphs) | `varena.admit` with each way to run: with `after`, without, eager | the first way varena holds, its weights pinned and pool mapped |
| A replay | `varena.admit` with its weights and pool, then eager | held, or the call runs as written |
| An LM's decode set | `varena.admit`: model and draft head, model alone, neither | held, undrafted, or Transformers' loop streams it |
| An LM call's keys and state | `vt.reserve`: one block for every slot, cut into them | the block, **wait** (fits once rows decoding now leave), or Final |
| A program's own copies of a call's tensors | `vt.reserve` | the block, or no program kept |
| An embedding table | `varena.admit`: on the card, else on the host | held, or its rows are gathered on the host |

## Rules

- Each attempt changes something. A Retry at the same level happens once, and only when room grew.
- The smallest unit runs again: one cut's leaf. Finished steps never run again, and package code
  never runs twice.
- A Final names the next step: `again` (varena learned the level's need; the next call splits
  finer), `wait`, or `smaller` (a smaller input, or `cozy run … --on new` for a larger card).
- A refusal nothing answered teaches its process: the loop's next pass of that shape is refused at
  its first forward unless the card has more room (`vt.escapes_take`). varena decides this itself,
  in every rank and executor.
- No Runtime code reads `room()` to decide anything. It is a report for run records.
