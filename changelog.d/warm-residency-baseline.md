Component staging during model warm-up no longer becomes a permanent construction
capacity limit. Runtime preserves the components that initially fitted, restores
warm-up evictions when measured capacity permits, and keeps them resident across
subsequent requests. Components actually parked by the construction budget remain
subject to the existing staged-execution limits.
