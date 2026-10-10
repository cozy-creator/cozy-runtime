package host

import (
	"errors"
	"fmt"
	"os"
)

var errMachineOwnershipLost = errors.New("machine root ownership was removed or replaced")

// The open descriptors pin the identities actually owned by this process. A
// pathname recreated by cleanup or another installer cannot inherit that lease.
type machineLease struct {
	lock     *os.File
	lockPath string
	lockInfo os.FileInfo
	root     *os.File
	rootPath string
	rootInfo os.FileInfo
}

func (l *machineLease) watchRoot(path string) error {
	root, err := os.Open(path)
	if err != nil {
		return err
	}
	info, err := root.Stat()
	if err != nil {
		root.Close()
		return err
	}
	l.root, l.rootPath, l.rootInfo = root, path, info
	return l.check()
}

func (l *machineLease) check() error {
	if l == nil {
		return nil
	}
	for _, owned := range []struct {
		path string
		info os.FileInfo
	}{{l.rootPath, l.rootInfo}, {l.lockPath, l.lockInfo}} {
		if owned.info == nil {
			continue
		}
		current, err := os.Stat(owned.path)
		if errors.Is(err, os.ErrNotExist) || err == nil && !os.SameFile(owned.info, current) {
			return fmt.Errorf("%w: %s", errMachineOwnershipLost, owned.path)
		}
		// Permission or transient I/O errors are not evidence of replacement.
	}
	return nil
}

func (l *machineLease) close() {
	if l.root != nil {
		_ = l.root.Close()
	}
	if l.lock != nil {
		_ = l.lock.Close()
	}
}
