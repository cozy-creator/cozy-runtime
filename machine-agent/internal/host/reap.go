package host

import "sync"

// A child os/exec waits for is never reaped by the guardian; spawnMu covers its start, so its
// pid is known before it can exit.
var (
	spawnMu sync.Mutex
	spawned = map[int]bool{}
)
