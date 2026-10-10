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

func TestUnsupportedInstalledBundleCannotReplaceRepairAgent(t *testing.T) {
	root := t.TempDir()
	m := &Machine{layout: NewLayout(&Grant{Root: root}), log: io.Discard}
	executable(t, filepath.Join(root, "opt/cozy/python/bin/cozy-machine"), `echo '{"name":"cozy-machine","wire_minor":68,"minimum_wire_minor":64,"capabilities":["hub-access/1","runtime-update/1"]}'`)
	d := &directLauncher{root: root, out: io.Discard}
	answer, err := d.maintain("agent-replace", []string{"running-repair-agent"})
	if err != nil || answer != "" || d.replacement != "" {
		t.Fatalf("unsupported installed bundle displaced repair agent: %q %v", answer, err)
	}
	if err = m.startupSave(startupStatus{State: "boot_pending"}); err != nil {
		t.Fatal(err)
	}
	if _, err = d.maintain("agent-replace", []string{"running-repair-agent"}); err == nil {
		t.Fatal("uncommitted unsupported application was not refused")
	}
}

func TestApplicationReplacementRequiresTransactionEvenWhenBundleSupportsABI(t *testing.T) {
	root := t.TempDir()
	executable(t, filepath.Join(root, "opt/cozy/python/bin/cozy-machine"), `echo '{"name":"cozy-machine","wire_minor":68,"minimum_wire_minor":64,"capabilities":["hub-access/1","runtime-update/1","machine-bootstrap/1"]}'`)
	d := &directLauncher{root: root, out: io.Discard}
	answer, err := d.maintain("agent-replace", []string{"another-running-agent"})
	if err != nil || answer != "" || d.replacement != "" {
		t.Fatalf("unjournaled application replacement: %q %v", answer, err)
	}
}
