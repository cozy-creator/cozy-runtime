package host

import (
	"os"
	"os/signal"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

// The guardian is the subreaper of the Runtime's process tree: an executor orphaned by a
// Runtime that died is adopted and reaped here.
func adoptOrphans() {
	_ = unix.Prctl(unix.PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
	signals := make(chan os.Signal, 16)
	signal.Notify(signals, syscall.SIGCHLD)
	go func() {
		for range signals {
			reapOrphans()
		}
	}()
}

func reapOrphans() {
	spawnMu.Lock()
	defer spawnMu.Unlock()
	for {
		var info unix.Siginfo
		if unix.Waitid(unix.P_ALL, 0, &info, unix.WEXITED|unix.WNOHANG|unix.WNOWAIT, nil) != nil {
			return
		}
		pid := int(*(*int32)(unsafe.Add(unsafe.Pointer(&info), 16))) // si_pid on 64-bit Linux
		if pid <= 0 || spawned[pid] {
			return
		}
		var status unix.WaitStatus
		if _, err := unix.Wait4(pid, &status, unix.WNOHANG, nil); err != nil {
			return
		}
	}
}
