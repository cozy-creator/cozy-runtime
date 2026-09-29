package host

import (
	"bytes"
	"errors"
	"os/exec"
	"sync"
)

// A child os/exec waits for is never reaped by the guardian; spawnMu covers its start, so its
// pid is known before it can exit.
var (
	spawnMu sync.Mutex
	spawned = map[int]bool{}
)

func startChild(cmd *exec.Cmd) error {
	spawnMu.Lock()
	defer spawnMu.Unlock()
	if err := cmd.Start(); err != nil {
		return err
	}
	spawned[cmd.Process.Pid] = true
	return nil
}

func waitChild(cmd *exec.Cmd) error {
	err := cmd.Wait()
	spawnMu.Lock()
	delete(spawned, cmd.Process.Pid)
	spawnMu.Unlock()
	reapOrphans()
	return err
}

func runChild(cmd *exec.Cmd) error {
	if err := startChild(cmd); err != nil {
		return err
	}
	return waitChild(cmd)
}

// All helper processes participate in reaping ownership, including short-lived
// public journal reads and maintenance. The subreaper may only wait for orphans.
func childOutput(cmd *exec.Cmd, combined bool) ([]byte, error) {
	var out, stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if combined {
		cmd.Stderr = &out
	}
	err := runChild(cmd)
	var exited *exec.ExitError
	if errors.As(err, &exited) && !combined {
		exited.Stderr = stderr.Bytes()
	}
	return out.Bytes(), err
}
