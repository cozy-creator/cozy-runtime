package host

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestPolicyCommitAndRollbackOwnOnlyTheirChanges(t *testing.T) {
	for _, operatorEdit := range []bool{false, true} {
		t.Run(map[bool]string{false: "rollback", true: "later operator"}[operatorEdit], func(t *testing.T) {
			m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()}), log: io.Discard}
			before := softwarePolicy{"off", "explicit"}
			after := softwarePolicy{"auto", "bundled"}
			raw, _ := json.Marshal(before)
			if err := writeAtomic(policyPath(m.layout.Root), raw, 0644); err != nil {
				t.Fatal(err)
			}
			entry := filepath.Join(m.layout.Root, "usr/local/bin/cozy-machine")
			if err := writeAtomic(entry, []byte("explicit binary"), 0755); err != nil {
				t.Fatal(err)
			}
			var journal startupStatus
			if err := m.snapshotPolicy(&journal, &after); err != nil {
				t.Fatal(err)
			}
			m.preparePolicyEntry(&journal)
			if err := m.commitSoftwarePolicy(&journal); err != nil {
				t.Fatal(err)
			}
			target, err := os.Readlink(entry)
			if err != nil || target != filepath.Join(m.layout.Root, "opt/cozy/python/bin/cozy-machine") {
				t.Fatalf("entry: %s %v", target, err)
			}
			if operatorEdit {
				raw, _ = json.Marshal(softwarePolicy{"off", "bundled"})
				if err = writeAtomic(policyPath(m.layout.Root), raw, 0644); err != nil {
					t.Fatal(err)
				}
			}
			if err = m.rollbackSoftwarePolicy(journal); err != nil {
				t.Fatal(err)
			}
			policy, _, err := readSoftwarePolicy(m.layout.Root)
			if err != nil {
				t.Fatal(err)
			}
			if operatorEdit {
				if policy != (softwarePolicy{"off", "bundled"}) {
					t.Fatal("operator edit overwritten")
				}
			} else {
				if policy != before {
					t.Fatalf("policy not restored: %+v", policy)
				}
				body, err := os.ReadFile(entry)
				if err != nil || string(body) != "explicit binary" {
					t.Fatalf("explicit entry not restored: %s %v", body, err)
				}
			}
		})
	}
}

func TestPolicyChangeBeforeCommitRefusesMutation(t *testing.T) {
	m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()})}
	var journal startupStatus
	if err := m.snapshotPolicy(&journal, &softwarePolicy{"auto", "bundled"}); err != nil {
		t.Fatal(err)
	}
	raw := []byte(`{"startup_update":"off","agent":"explicit"}`)
	if err := writeAtomic(policyPath(m.layout.Root), raw, 0644); err != nil {
		t.Fatal(err)
	}
	m.preparePolicyEntry(&journal)
	if err := m.commitSoftwarePolicy(&journal); err == nil {
		t.Fatal("changed operator policy was overwritten")
	}
}

func TestManualPolicyReplacesInvalidPolicyWithoutBlockingRepair(t *testing.T) {
	m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()})}
	original := []byte(`{"startup_update":"future","agent":"future"}`)
	if err := writeAtomic(policyPath(m.layout.Root), original, 0644); err != nil {
		t.Fatal(err)
	}
	var journal startupStatus
	if err := m.snapshotPolicy(&journal, &softwarePolicy{"off", "explicit"}); err != nil {
		t.Fatal(err)
	}
	if err := m.commitSoftwarePolicy(&journal); err != nil {
		t.Fatal(err)
	}
	if err := m.rollbackSoftwarePolicy(journal); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(policyPath(m.layout.Root))
	if err != nil || string(got) != string(original) {
		t.Fatalf("prior operator policy not retained: %s %v", got, err)
	}
}

func TestAutomaticExplicitAgentSelectionUsesRunningApplication(t *testing.T) {
	m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()}), log: io.Discard}
	if err := writeAtomic(policyPath(m.layout.Root), []byte(`{"startup_update":"auto","agent":"explicit"}`), 0644); err != nil {
		t.Fatal(err)
	}
	retained := filepath.Join(m.layout.Root, "retained-running-agent")
	if err := m.startupSave(startupStatus{State: "boot_pending", AgentBefore: retained}); err != nil {
		t.Fatal(err)
	}
	if got := m.selectedApplication(); got != retained {
		t.Fatalf("explicit application changed to pathname %s", got)
	}
}
