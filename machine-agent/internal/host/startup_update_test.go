package host

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestStartupMetadataAndPublicRetention(t *testing.T) {
	python := os.Getenv("COZY_RUNTIME_TEST_PYTHON")
	if python == "" {
		python = "python3"
	}
	if out, err := exec.Command(python, "-B", "startup_update_test.py").CombinedOutput(); err != nil {
		t.Fatalf("metadata/public journal: %v\n%s", err, out)
	}
}

func TestInterruptedStartupCannotBootWithoutIntactRollback(t *testing.T) {
	m := &Machine{grant: &Grant{StartupUpdate: "off"}, layout: NewLayout(&Grant{Root: t.TempDir()}), log: io.Discard}
	status := startupStatus{State: "installing", Before: pairVersions{Runtime: "0.18.85", TensorFS: "0.3.78"}, Previous: []startupWheel{{File: "cozy_runtime-0.18.85-cp312-abi3-manylinux_2_28_x86_64.whl", SHA256: strings.Repeat("a", 64)}}}
	if err := m.startupSave(status); err != nil {
		t.Fatal(err)
	}
	for path, mode := range map[string]os.FileMode{m.startupPath("status.json"): 0600, filepath.Join(m.layout.State, "startup-update-status.json"): 0644} {
		info, err := os.Stat(path)
		if err != nil || info.Mode().Perm() != mode {
			t.Fatalf("journal/report permissions: %s %v", path, err)
		}
	}
	if err := m.startupUpdate(context.Background()); err == nil {
		t.Fatal("off policy allowed an interrupted mutation without verified rollback")
	}
	raw, err := os.ReadFile(m.startupPath("status.json"))
	if err != nil || !strings.Contains(string(raw), "installing") {
		t.Fatal("failed recovery erased its pending journal")
	}
}

func TestStartupWorkerLeaseNeverTakesOverLiveWorker(t *testing.T) {
	root := t.TempDir()
	unlocked, err := lockStartupWorker(root)
	if err != nil {
		t.Fatal(err)
	}
	defer unlocked()
	m := &Machine{grant: &Grant{}, layout: NewLayout(&Grant{Root: root}), log: io.Discard}
	if err := m.startupUpdate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(m.startupPath("status.json")); !os.IsNotExist(err) {
		t.Fatal("live worker caused update mutation")
	}
}

func TestStartupRollbackKeepsVerifiedPrivateCopies(t *testing.T) {
	m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()})}
	source := filepath.Join(t.TempDir(), "cozy_runtime-0.18.85-cp312-abi3-manylinux_2_28_x86_64.whl")
	if err := os.WriteFile(source, []byte("retained immutable wheel"), 0600); err != nil {
		t.Fatal(err)
	}
	kept, err := m.startupPrevious([]string{source})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("changed source"), 0600); err != nil {
		t.Fatal(err)
	}
	sha, err := hashFile(m.startupPath("previous", kept[0].File))
	if err != nil || sha != kept[0].SHA256 {
		t.Fatal("rollback depended on mutable source")
	}
	if strings.HasPrefix(m.startupPath(), m.layout.State+string(os.PathSeparator)) {
		t.Fatal("privileged rollback journal would be handed to unprivileged agent")
	}
}

func TestStartupCommitRequiresItsLiveIncarnation(t *testing.T) {
	p := &runtimeProcess{incarnation: "candidate", done: make(chan struct{})}
	if _, err := maintain(t.TempDir(), p, "startup-commit", []string{"older"}); err == nil {
		t.Fatal("stale Runtime committed candidate")
	}
	if _, err := maintain(t.TempDir(), p, "startup-rollback", nil); err == nil {
		t.Fatal("rollback replaced live Runtime")
	}
}

func TestExplicitRemoteWheelPinAppliesOnlyToItsInstalledPair(t *testing.T) {
	m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()})}
	pair := pairVersions{Runtime: "0.18.86", TensorFS: "0.3.78"}
	if !(updateStatus{State: "rolled_back", From: pair, PreviouslyPinned: true}).pins(pair) {
		t.Fatal("failed maintenance erased the prior explicit selection")
	}
	for _, row := range []struct {
		state     string
		pinned    bool
		installed pairVersions
		want      bool
	}{
		{"succeeded", true, pair, true}, {"succeeded", false, pair, false}, {"failed", true, pair, false}, {"succeeded", true, pairVersions{Runtime: "0.18.87", TensorFS: "0.3.78"}, false},
	} {
		raw, err := json.Marshal(updateStatus{State: row.state, Pinned: row.pinned, To: pair})
		if err != nil {
			t.Fatal(err)
		}
		if err := writeAtomic(m.updatePath(), raw, 0600); err != nil {
			t.Fatal(err)
		}
		if got := m.explicitlyPinnedPair(row.installed); got != row.want {
			t.Fatalf("pin %s/%v: %v", row.state, row.pinned, got)
		}
	}
}

func TestStartupCandidateDoesNotTrustMutableStageOrFollowTargetSymlink(t *testing.T) {
	m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()})}
	source := filepath.Join(t.TempDir(), "candidate.whl")
	if err := os.WriteFile(source, []byte("verified bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	sha, err := hashFile(source)
	if err != nil {
		t.Fatal(err)
	}
	dir := m.startupPath("candidate")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "untouched")
	if err := os.WriteFile(outside, []byte("outside bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, filepath.Base(source))); err != nil {
		t.Fatal(err)
	}
	private, err := m.startupCopyWheel(source, "candidate", sha)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("changed stage"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := hashFile(private)
	if err != nil || got != sha {
		t.Fatal("candidate still depended on agent-writable staging")
	}
	if raw, err := os.ReadFile(outside); err != nil || string(raw) != "outside bytes" {
		t.Fatal("private copy followed target symlink")
	}
	if _, err := m.startupCopyWheel(source, "candidate", sha); err == nil {
		t.Fatal("changed stage passed expected release digest")
	}
}
