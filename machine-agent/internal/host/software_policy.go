package host

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

type softwarePolicy struct {
	StartupUpdate string `json:"startup_update"`
	Agent         string `json:"agent"`
}

func policyPath(root string) string { return filepath.Join(root, "etc/cozy/software-policy.json") }
func readPolicyBytes(root string) ([]byte, error) {
	raw, err := os.ReadFile(policyPath(root))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	return raw, err
}
func readSoftwarePolicy(root string) (softwarePolicy, []byte, error) {
	raw, err := os.ReadFile(policyPath(root))
	if errors.Is(err, os.ErrNotExist) {
		return softwarePolicy{"auto", "bundled"}, nil, nil
	}
	if err != nil {
		return softwarePolicy{}, nil, err
	}
	var policy softwarePolicy
	if err = json.Unmarshal(raw, &policy); err != nil {
		return policy, raw, err
	}
	if (policy.StartupUpdate != "auto" && policy.StartupUpdate != "off") || (policy.Agent != "bundled" && policy.Agent != "explicit") {
		return policy, raw, errors.New("software policy requires startup_update auto|off and agent bundled|explicit")
	}
	return policy, raw, nil
}

func (m *Machine) transactionAgent() string {
	var status startupStatus
	if raw, err := os.ReadFile(m.startupPath("status.json")); err == nil && json.Unmarshal(raw, &status) == nil && (status.State == "installing" || status.State == "boot_pending") && status.PolicyAfter != nil {
		return status.PolicyAfter.Agent
	}
	policy, _, err := readSoftwarePolicy(m.layout.Root)
	if err != nil {
		return "explicit"
	}
	return policy.Agent
}

func (m *Machine) preparePolicyEntry(status *startupStatus) {
	policy, _, _ := readSoftwarePolicy(m.layout.Root)
	if status.PolicyAfter != nil {
		policy = *status.PolicyAfter
	}
	if policy.Agent == "bundled" {
		status.EntryAfter = filepath.Join(m.layout.Root, "opt/cozy/python/bin/cozy-machine")
	} else {
		status.EntryAfter = status.AgentBefore
	}
}

func (m *Machine) commitSoftwarePolicy(status *startupStatus) error {
	current, err := readPolicyBytes(m.layout.Root)
	if err != nil {
		return err
	}
	after := status.PolicyBefore
	if status.PolicyAfter != nil {
		after, _ = json.Marshal(status.PolicyAfter)
	}
	if !bytes.Equal(current, status.PolicyBefore) && !bytes.Equal(current, after) {
		return errors.New("software policy changed during the update; preserving operator selection")
	}
	if status.PolicyAfter != nil {
		if err = os.MkdirAll(filepath.Dir(policyPath(m.layout.Root)), 0755); err != nil {
			return err
		}
		if err = writeAtomic(policyPath(m.layout.Root), after, 0644); err != nil {
			return err
		}
	}
	if status.EntryAfter != "" {
		return writeAgentEntry(m.layout.Root, status.EntryAfter)
	}
	return nil
}

func writeAgentEntry(root, target string) error {
	entry := filepath.Join(root, "usr/local/bin/cozy-machine")
	if err := os.MkdirAll(filepath.Dir(entry), 0755); err != nil {
		return err
	}
	temp, err := os.CreateTemp(filepath.Dir(entry), ".agent-entry-")
	if err != nil {
		return err
	}
	name := temp.Name()
	temp.Close()
	os.Remove(name)
	defer os.Remove(name)
	if err = os.Symlink(target, name); err != nil {
		return err
	}
	if err = os.Rename(name, entry); err != nil {
		return err
	}
	return syncDir(filepath.Dir(entry))
}

func (m *Machine) snapshotPolicy(status *startupStatus, next *softwarePolicy) error {
	if err := os.MkdirAll(m.startupPath(), 0700); err != nil {
		return err
	}
	raw, err := readPolicyBytes(m.layout.Root)
	if err != nil {
		return err
	}
	status.PolicyBefore, status.PolicyAfter = raw, next
	entry := filepath.Join(m.layout.Root, "usr/local/bin/cozy-machine")
	info, err := os.Lstat(entry)
	if errors.Is(err, os.ErrNotExist) {
		status.EntryMissing = true
		return nil
	}
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		status.EntryBefore, err = os.Readlink(entry)
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("agent entry is not a regular file or symlink")
	}
	raw, err = os.ReadFile(entry)
	if err != nil {
		return err
	}
	status.EntryBefore = m.startupPath("previous-agent")
	status.EntryFile = true
	return writeAtomic(status.EntryBefore, raw, 0700)
}

func (m *Machine) rollbackSoftwarePolicy(status startupStatus) error {
	if status.PolicyAfter != nil {
		after, _ := json.Marshal(status.PolicyAfter)
		current, err := readPolicyBytes(m.layout.Root)
		if err != nil {
			return err
		}
		if bytes.Equal(current, after) {
			if status.PolicyBefore == nil {
				err = os.Remove(policyPath(m.layout.Root))
			} else {
				err = writeAtomic(policyPath(m.layout.Root), status.PolicyBefore, 0644)
			}
			if err != nil && !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
	}
	entry := filepath.Join(m.layout.Root, "usr/local/bin/cozy-machine")
	target, err := os.Readlink(entry)
	if err != nil || target != status.EntryAfter || status.EntryAfter == "" {
		return nil
	}
	if status.EntryMissing {
		err = os.Remove(entry)
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if status.EntryFile {
		raw, e := os.ReadFile(status.EntryBefore)
		if e != nil {
			return e
		}
		return writeAtomic(entry, raw, 0755)
	}
	return writeAgentEntry(m.layout.Root, status.EntryBefore)
}

// Retain the descriptor-selected application before uv can unlink its pathname.
// This is a normal executable, never setuid, and contains no grant or secret.
func retainApplication(root, source string) (string, error) {
	if source == "" {
		return "", nil
	}
	hash, err := hashFile(source)
	if err != nil {
		return "", err
	}
	target := filepath.Join(root, "usr/local/lib/cozy-machine", hash, "cozy-machine")
	if got, err := hashFile(target); err == nil && got == hash {
		return target, nil
	}
	raw, err := os.ReadFile(source)
	if err != nil {
		return "", err
	}
	if err = os.MkdirAll(filepath.Dir(target), 0755); err != nil {
		return "", err
	}
	return target, writeAtomic(target, raw, 0755)
}

func (m *Machine) selectedApplication() string {
	var status startupStatus
	raw, err := os.ReadFile(m.startupPath("status.json"))
	if err == nil && json.Unmarshal(raw, &status) == nil {
		if (status.State == "installing" || status.State == "boot_pending") && m.transactionAgent() == "explicit" && status.AgentBefore != "" {
			return status.AgentBefore
		}
	}
	if m.transactionAgent() == "bundled" {
		return filepath.Join(m.layout.Root, "opt/cozy/python/bin/cozy-machine")
	}
	return filepath.Join(m.layout.Root, "usr/local/bin/cozy-machine")
}

func (m *Machine) restoredApplication() string {
	var status startupStatus
	if raw, err := os.ReadFile(m.startupPath("status.json")); err == nil && json.Unmarshal(raw, &status) == nil && status.State == "rolled_back" && status.AgentBefore != "" {
		return status.AgentBefore
	}
	return m.selectedApplication()
}
