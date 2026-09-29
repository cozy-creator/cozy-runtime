package host

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRequestPlaneDisconnectKeepsParentOwnedRuntime(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "runtime")
	executable(t, path, "exec cat <&3 >/dev/null")
	d := &directLauncher{path: path, root: root, env: []string{"PATH=/usr/bin:/bin"}, out: io.Discard}
	p, err := d.launch("accepted-runtime")
	if err != nil {
		t.Fatal(err)
	}
	processes := &guardianProcesses{}
	processes.set(p)
	defer processes.stop("")()
	report := func(guardianStatus) {}
	if err = serveGuardianCommands(strings.NewReader(""), d, processes, report); err != nil {
		t.Fatal(err)
	}
	if p.exited() || d.current != p {
		t.Fatal("public transport EOF stopped or replaced Runtime")
	}
	// A replacement control channel still addresses the same incarnation.
	if err = serveGuardianCommands(strings.NewReader(`{"op":"stop","incarnation":"accepted-runtime"}`+"\n"), d, processes, report); err != nil {
		t.Fatal(err)
	}
	select {
	case <-p.done:
	case <-time.After(5 * time.Second):
		t.Fatal("replacement channel could not stop its Runtime")
	}
}

func TestApplicationRetentionUsesRunningInode(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "application")
	if err := os.WriteFile(path, []byte("running explicit application"), 0755); err != nil {
		t.Fatal(err)
	}
	held, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer held.Close()
	if err = os.Rename(path, path+".old"); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, []byte("new wheel application"), 0755); err != nil {
		t.Fatal(err)
	}
	retained, err := retainApplication(root, fmt.Sprintf("/proc/self/fd/%d", held.Fd()))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(retained)
	if err != nil || string(raw) != "running explicit application" {
		t.Fatalf("retained pathname replacement: %s %v", raw, err)
	}
}
