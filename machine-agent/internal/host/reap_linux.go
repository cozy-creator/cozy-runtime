package host

import (
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

// A Runtime wheel can replace the installed agent pathname during startup.
// Its guardian must still execute this process's inode, preserving their private IPC.
func guardianExecutable() (string, error) { return "/proc/self/exe", nil }

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

// A maintenance helper has no accepted workload. Its descendants share a group
// so a measured stall cannot leave uv/probe children holding output pipes open.
func prepareMaintenanceProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if cmd.Cancel != nil {
		cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	}
}
func killMaintenanceProcess(cmd *exec.Cmd) { _ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }

func requestPlaneCredential(cmd *exec.Cmd) {
	if os.Geteuid() == 0 {
		cmd.SysProcAttr = &syscall.SysProcAttr{Credential: &syscall.Credential{Uid: machineUID, Gid: machineUID, Groups: []uint32{}}}
	}
}
