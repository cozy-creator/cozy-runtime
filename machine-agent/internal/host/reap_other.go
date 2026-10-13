//go:build !linux

package host

import "os"

func guardianExecutable() (string, error) { return os.Executable() }

// Machines run on Linux; elsewhere the daemon adopts no orphans.
func adoptOrphans() {}

func reapOrphans() {}
