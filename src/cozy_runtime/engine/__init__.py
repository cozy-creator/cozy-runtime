"""The inference engine's own modules (xs-213): what its step loop is built from. None imports
`cozy_runtime.lm`.

- `rungs`: which rung a step takes and how much of a call's K/V the card holds, judged from a
  measured first step.
- `kv`: the rows' K/V blocks and state slots in two varena Caches, with the device tables a step's
  graph reads.
"""
