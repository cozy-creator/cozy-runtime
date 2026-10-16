Less host work per op and per block:
- A stage leases its resident blocks once, where its plan binds them, so a block runs with no acquire, no eviction drain and no release. Any unmapping lets those leases go first.
- A component's own call (`unet(...)`) runs with no Python dispatch layer once its first call showed it changes nothing outside itself. An out-of-memory outside its blocks makes room and runs the call again; its ops then recover one by one.
- The dispatch mode no longer sits behind torch's Dynamo-disable wrapper.
- Attention names its implementation once per operand shape, not on every call.
- A stage's exit no longer empties torch's cache; the next stage's plan trims what it needs.
