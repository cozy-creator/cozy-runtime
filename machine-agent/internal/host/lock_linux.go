package host

import (
	"fmt"
	"golang.org/x/sys/unix"
	"os"
	"path/filepath"
)

// lockMachine is process-held exclusion, not durable owner history. File contents
// and stale PIDs do not authorize taking over a running machine.
func lockMachine(state string) (*machineLease, error) {
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
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	return &machineLease{lock: f, lockPath: filepath.Join(state, "agent.lock"), lockInfo: info}, nil
}

// Hold the same process lease as Runtime while checking retained work and
// changing its software. A surviving Runtime or executor is never interrupted.
func lockStartupWorker(root string) (func(), error) {
	dir := filepath.Join(root, "run/cozy/worker")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(filepath.Join(dir, "worker.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB); err != nil {
		f.Close()
		return nil, err
	}
	return func() { _ = unix.Flock(int(f.Fd()), unix.LOCK_UN); _ = f.Close() }, nil
}

func syncInstallation(root string) error {
	f, err := os.Open(filepath.Join(root, "opt/cozy/python"))
	if err != nil {
		return err
	}
	defer f.Close()
	return unix.Syncfs(int(f.Fd()))
}
