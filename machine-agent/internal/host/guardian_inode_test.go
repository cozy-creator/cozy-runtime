package host

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// This helper replaces its own executable pathname, exactly as uv can replace
// a wheel-owned cozy-machine while the old agent is still mapped and running.
func TestGuardianExecutingInodeHelper(t *testing.T) {
	root := os.Getenv("COZY_GUARDIAN_INODE_ROOT")
	if root == "" {
		return
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	replacement := filepath.Join(root, "replacement")
	executable(t, replacement, "exit 87")
	if err := os.Rename(replacement, self); err != nil {
		t.Fatal(err)
	}
	source, err := guardianExecutable()
	if err != nil {
		t.Fatal(err)
	}
	readControl, writeControl, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	readStatus, writeStatus, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer readStatus.Close()
	child := exec.Command(source, "-test.run=^TestGuardianBoundaryHelper$")
	child.Env = append(os.Environ(), "COZY_GUARDIAN_BOUNDARY_HELPER=1", "COZY_GUARDIAN_BOUNDARY_ROOT="+root)
	child.ExtraFiles = []*os.File{readControl, writeStatus}
	if err := startChild(child); err != nil {
		t.Fatal(err)
	}
	readControl.Close()
	writeStatus.Close()
	if err := json.NewEncoder(writeControl).Encode(guardianCommand{Op: "pair"}); err != nil {
		t.Fatal(err)
	}
	var response guardianStatus
	if err := json.NewDecoder(readStatus).Decode(&response); err != nil {
		t.Fatalf("replacement image cannot serve original guardian IPC: %v", err)
	}
	if response.Phase != "maintained" || response.Error != "" {
		t.Fatalf("original guardian response: %+v", response)
	}
	writeControl.Close()
	if err := waitChild(child); err != nil {
		t.Fatal(err)
	}
}

func TestGuardianUsesExecutingInodeAfterInstalledPathReplacement(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux executable inode semantics")
	}
	root := t.TempDir()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	in, err := os.Open(self)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	path := filepath.Join(root, "agent")
	out, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0755)
	if err != nil {
		t.Fatal(err)
	}
	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		t.Fatal(copyErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, path, "-test.run=^TestGuardianExecutingInodeHelper$")
	command.Env = append(os.Environ(), "COZY_GUARDIAN_INODE_ROOT="+root)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("executing inode handoff: %v\n%s", err, output)
	}
}
