The machine agent serves TensorFS's transport decision log through `PodHost.ReadMachineLog`
(wire 72): one line per hedge, lane grant, win and pull walk, read from the store's
`logs/transport.log` and its rotated `.1`, oldest first, optionally just the newest bytes. The
agent reads it from disk, so it answers while the Runtime is stopped.
