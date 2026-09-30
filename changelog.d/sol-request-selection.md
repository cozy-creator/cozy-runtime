Sol attention now freezes published logical variant availability at the request boundary.
A cold or previously unseen length uses the configured dense fallback for the entire
request, including retries. Compilation continues asynchronously and a later request can
adopt the published sparse variant. This intentionally changes cold-request numerical
behavior: a request no longer switches from dense to approximate sparse attention midway.

GPU groups intersect logical variant coverage across ranks, independent of their
architecture-specific artifact keys, and acknowledge the choice before authored execution.
A selected artifact that is missing, corrupt, or unloadable refuses the request instead of
silently changing its route. Sparse internal calls require an explicit selection scope;
authored dense warmup is unchanged.

CPU route and Gloo agreement checks cover selection mechanics. This change does not claim
H3 GPU output equality, cross-architecture numerical identity, or unchanged cuDNN engine
selection. Those remain separate qualification gates.
