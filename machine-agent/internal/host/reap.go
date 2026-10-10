package host

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
)

// A child os/exec waits for is never reaped by the guardian; spawnMu covers its start, so its
// pid is known before it can exit.
var (
	spawnMu sync.Mutex
	spawned = map[int]bool{}
)

func startChild(cmd *exec.Cmd) error {
	if cmd.Env == nil {
		cmd.Env = []string{}
		for _, name := range []string{"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"} {
			if value, present := os.LookupEnv(name); present {
				cmd.Env = append(cmd.Env, name+"="+value)
			}
		}
	}
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

// Helpers own no accepted workload; bound stalls using actual CPU/output progress.
func progressChildOutput(cmd *exec.Cmd, combined bool) ([]byte, error) {
	var out, stderr bytes.Buffer
	var progress atomic.Uint64
	var writing sync.Mutex
	cmd.Stdout = writerFunc(func(p []byte) (int, error) {
		writing.Lock()
		defer writing.Unlock()
		progress.Add(uint64(len(p)))
		return out.Write(p)
	})
	cmd.Stderr = writerFunc(func(p []byte) (int, error) {
		writing.Lock()
		defer writing.Unlock()
		progress.Add(uint64(len(p)))
		return stderr.Write(p)
	})
	if combined {
		cmd.Stderr = cmd.Stdout
	}
	prepareMaintenanceProcess(cmd)
	if err := startChild(cmd); err != nil {
		return nil, err
	}
	done := make(chan struct{})
	go killWhenStill(done, func() uint64 { return processTreeTicks(cmd.Process.Pid) + progress.Load() }, func() { killMaintenanceProcess(cmd) })
	err := waitChild(cmd)
	close(done)
	var exited *exec.ExitError
	if errors.As(err, &exited) && !combined {
		exited.Stderr = stderr.Bytes()
	}
	return out.Bytes(), err
}
