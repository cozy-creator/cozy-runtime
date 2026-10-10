//go:build !linux

package host

import (
	"os"
	"os/exec"
)

func guardianExecutable() (string, error) { return os.Executable() }

// Machines run on Linux; elsewhere the daemon adopts no orphans.
func adoptOrphans() {}

func reapOrphans() {}

func prepareMaintenanceProcess(*exec.Cmd)  {}
func killMaintenanceProcess(cmd *exec.Cmd) { _ = cmd.Process.Kill() }

func requestPlaneCredential(*exec.Cmd) {}
