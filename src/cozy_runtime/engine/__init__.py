"""The inference engine's own modules (xs-213): what its step loop is built from. None imports
`cozy_runtime.lm`.

- `rungs`: a call's declared K/V and state need (`KvNeed`), which `kv` states; the judge that
  reads it comes with its caller.
- `kv`: the rows' K/V blocks and state slots in two varena Caches, with the device tables a step's
  graph reads.
"""
