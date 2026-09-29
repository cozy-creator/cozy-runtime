//go:build !linux

package host

// Machines run on Linux; elsewhere the daemon adopts no orphans.
func adoptOrphans() {}

func reapOrphans() {}
