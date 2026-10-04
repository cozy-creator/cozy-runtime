Out-of-memory recovery no longer replays a block's complete forward after an observed
input-pure call. Successful operations, module state, inference tensor mutations and
random draws remain committed. Automatic retries are limited to non-mutating ATen
operations without random or possible hidden mutation tags. Other failed operations
report `recovery_contract_unavailable`, identifying the missing rollback contract;
this does not establish that the model exceeds device capacity.

CPU dispatcher fault tests cover mutation, random and opaque custom operators. CUDA
low-memory recovery and the dispatch overhead on inference workloads still need GPU
qualification before release.
