package host

import (
	"fmt"
	"golang.org/x/sys/unix"
	"os"
	"path/filepath"
)

// lockMachine is process-held exclusion, not durable owner history. File contents
// and stale PIDs do not authorize taking over a running machine.
func lockMachine(state string) (func(), error) {
	if err := os.MkdirAll(state, 0755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(filepath.Join(state, "agent.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB); err != nil {
		f.Close()
		return nil, fmt.Errorf("another machine agent owns this root: %w", err)
	}
	return func() { _ = unix.Flock(int(f.Fd()), unix.LOCK_UN); _ = f.Close() }, nil
}
