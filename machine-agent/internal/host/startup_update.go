package host

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/build"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
)

//go:embed startup_update.py
var startupUpdateScript string

type startupProbe struct {
	pairVersions
	WireMinor        uint32   `json:"wire_minor"`
	MinimumWireMinor uint32   `json:"minimum_wire_minor"`
	Capabilities     []string `json:"capabilities"`
}

func (m *Machine) startupVerify(ctx context.Context, overlay ...string) (pairVersions, error) {
	var probe startupProbe
	if err := m.startupPython(ctx, &probe, append([]string{"probe"}, overlay...)...); err != nil {
		return pairVersions{}, err
	}
	if probe.WireMinor < pb.MinCompatibleWireMinor || probe.MinimumWireMinor > pb.WireMinor || !slices.Contains(probe.Capabilities, "machine-supervisor/1") || !slices.Contains(probe.Capabilities, "machine-public-reads/1") {
		return pairVersions{}, errors.New("Runtime does not implement this agent's supported protocol/capabilities")
	}
	return probe.pairVersions, nil
}

type startupWheel struct {
	File, SHA256, Version, URL string
}

type startupStatus struct {
	State         string `json:"state"` // installing | succeeded | rolled_back | skipped
	Detail        string `json:"detail,omitempty"`
	Before, After pairVersions
	Previous      []startupWheel `json:"previous,omitempty"`
}

func (m *Machine) startupPath(names ...string) string {
	// This sibling stays owned by the launcher, unlike the agent's writable State.
	return filepath.Join(append([]string{m.layout.Root, "var/lib/cozy/startup-update"}, names...)...)
}

func (m *Machine) startupSave(status startupStatus) error {
	if err := os.MkdirAll(m.startupPath(), 0o700); err != nil {
		return err
	}
	if err := syncDir(filepath.Dir(m.startupPath())); err != nil {
		return err
	}
	raw, err := json.Marshal(status)
	if err != nil {
		return err
	}
	if err := writeAtomic(m.startupPath("status.json"), raw, 0o600); err != nil {
		return err
	}
	// This secrets-free report remains readable after the agent drops privilege;
	// only the private journal above is used to authorize recovery.
	return writeAtomic(filepath.Join(m.layout.State, "startup-update-status.json"), raw, 0o644)
}

func (m *Machine) startupPython(ctx context.Context, into any, arguments ...string) error {
	args := append([]string{"-I", "-c", startupUpdateScript}, arguments...)
	cmd := exec.CommandContext(ctx, filepath.Join(m.layout.Root, "opt/cozy/python/bin/python"), args...)
	out, err := cmd.Output()
	if err != nil {
		if failed, ok := err.(*exec.ExitError); ok {
			return fmt.Errorf("startup %s: %s", arguments[0], strings.TrimSpace(string(failed.Stderr)))
		}
		return err
	}
	return json.Unmarshal(out, into)
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// A successful explicit wheel operation owns its particular installed pair.
// This file can only defer discovery; it never authorizes privileged mutation.
func (m *Machine) explicitlyPinnedPair(installed pairVersions) bool {
	var previous updateStatus
	raw, err := os.ReadFile(m.updatePath())
	return err == nil && json.Unmarshal(raw, &previous) == nil && previous.pins(installed)
}

func (m *Machine) startupPrevious(wheels []string) ([]startupWheel, error) {
	var kept []startupWheel
	for _, source := range wheels {
		sha, err := hashFile(source)
		if err != nil {
			return nil, err
		}
		if _, err := m.startupCopyWheel(source, "previous", sha); err != nil {
			return nil, err
		}
		kept = append(kept, startupWheel{File: filepath.Base(source), SHA256: sha})
	}
	return kept, nil
}

// Existing maintenance staging is writable by the unprivileged agent. Root
// installation and native probes consume only a verified copy in private custody.
func (m *Machine) startupCopyWheel(source, folder, expected string) (string, error) {
	dir := m.startupPath(folder)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	path := filepath.Join(dir, filepath.Base(source))
	if err := copyWheel(source, path); err != nil {
		return "", err
	}
	got, err := hashFile(path)
	if err != nil || got != expected {
		return "", errors.New("private wheel copy differs from its expected digest")
	}
	return path, syncDir(dir)
}

func (m *Machine) startupRollback(ctx context.Context, status startupStatus) error {
	var wheels []string
	for _, wheel := range status.Previous {
		if wheelDistribution(wheel.File) == "" || !sha256Hex.MatchString(wheel.SHA256) {
			return errors.New("invalid startup rollback identity")
		}
		path := m.startupPath("previous", wheel.File)
		sha, err := hashFile(path)
		if err != nil || sha != wheel.SHA256 {
			return fmt.Errorf("startup rollback wheel %s is missing or changed", wheel.File)
		}
		wheels = append(wheels, path)
	}
	if len(wheels) != 2 || !samePair(wheels, status.Before) {
		return errors.New("startup rollback pair is incomplete")
	}
	if err := installPair(m.layout.Root, wheels); err != nil {
		return err
	}
	var got pairVersions
	if err := m.startupPython(ctx, &got, "probe"); err != nil {
		return err
	}
	if got != status.Before {
		return errors.New("startup rollback did not restore the verified pair")
	}
	if err := selectPair(m.layout.Root, wheels); err != nil {
		return err
	}
	status.State = "rolled_back"
	return m.startupSave(status)
}

// startupUpdate runs once, under the machine and worker-root locks, before any
// launcher or new-work admission. Recovery precedes policy: a half-installed pair
// is never used merely because discovery is offline or updates were disabled.
func (m *Machine) startupUpdate(ctx context.Context) error {
	var status startupStatus
	raw, err := os.ReadFile(m.startupPath("status.json"))
	if err == nil {
		if err := json.Unmarshal(raw, &status); err != nil {
			return fmt.Errorf("unreadable startup update journal: %w", err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	unlock, err := lockStartupWorker(m.layout.Root)
	if err != nil {
		if status.State == "installing" || status.State == "boot_pending" {
			return fmt.Errorf("cannot recover interrupted update while worker ownership is unresolved: %w", err)
		}
		fmt.Fprintln(m.log, "cozy machine: startup update deferred: worker root is held or unreadable")
		return nil
	}
	defer unlock()
	if status.State == "installing" || status.State == "boot_pending" {
		status.Detail = "recovering interrupted startup update"
		if err := m.startupRollback(ctx, status); err != nil {
			return fmt.Errorf("interrupted startup update could not be repaired: %w", err)
		}
		fmt.Fprintln(m.log, "cozy machine: restored the verified pair after interrupted startup update")
		return nil
	}
	if m.grant.StartupUpdate == "off" || strings.ContainsAny(build.Version, "-+") {
		fmt.Fprintln(m.log, "cozy machine: startup update skipped: explicitly pinned or development agent")
		return nil
	}
	skip := func(detail string) error {
		fmt.Fprintln(m.log, "cozy machine: startup update retained installed pair:", detail)
		return m.startupSave(startupStatus{State: "skipped", Detail: detail})
	}
	var installed struct {
		pairVersions
		Eligible bool
	}
	if err := m.startupPython(ctx, &installed, "inspect", m.layout.Store); err != nil {
		return skip("installed metadata or public work journal is unreadable; update deferred")
	}
	if !installed.Eligible {
		return skip("development pair or unfinished retained work; update deferred")
	}
	if m.explicitlyPinnedPair(installed.pairVersions) {
		return skip("installed pair is owned by an explicit wheel update")
	}
	verified, err := m.startupVerify(ctx)
	if err != nil {
		return fmt.Errorf("installed pair failed startup verification: %w", err)
	}
	var selected map[string]startupWheel
	if err := m.startupPython(ctx, &selected, "plan"); err != nil {
		return skip("release discovery unavailable or no compatible published pair; verified installed pair retained")
	}
	next := pairVersions{Runtime: selected["cozy-runtime"].Version, TensorFS: selected["tensorfs"].Version}
	if next == verified {
		return skip("installed pair is current within its compatible major lines")
	}
	if err := os.MkdirAll(m.startupPath(), 0o700); err != nil {
		return err
	}
	var candidate []string
	for _, name := range []string{"cozy-runtime", "tensorfs"} {
		wheel := selected[name]
		if wheelDistribution(wheel.File) != name || wheelVersion(wheel.File) != wheel.Version || !sha256Hex.MatchString(wheel.SHA256) || !strings.HasPrefix(wheel.URL, "https://files.pythonhosted.org/") {
			return skip("published wheel identity is invalid")
		}
		path, err := m.fetchWheel(ctx, name, wheel.Version)
		if err != nil {
			return skip("published wheel download failed; verified installed pair retained")
		}
		if filepath.Base(path) != wheel.File {
			return skip("published wheel digest or platform changed during discovery")
		}
		path, err = m.startupCopyWheel(path, "candidate", wheel.SHA256)
		if err != nil {
			return skip("candidate wheel failed private-custody integrity verification")
		}
		candidate = append(candidate, path)
	}
	previous, err := keptPair(m.layout.Root)
	if err != nil {
		return skip("rollback wheels are unreadable")
	}
	if !samePair(previous, verified) {
		previous = nil
		for name, version := range map[string]string{"cozy-runtime": verified.Runtime, "tensorfs": verified.TensorFS} {
			path, err := m.fetchWheel(ctx, name, version)
			if err != nil {
				return skip("verified installed wheels cannot be retained for offline rollback")
			}
			previous = append(previous, path)
		}
	}
	kept, err := m.startupPrevious(previous)
	if err != nil {
		return err
	}
	// Stage an overlay: uv touches no existing dependency or interpreter. Python
	// imports the two candidate distributions from it against the actual base deps.
	overlay, err := os.MkdirTemp(m.startupPath(), "probe-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(overlay)
	args := append([]string{"pip", "install", "--no-config", "--offline", "--no-deps", "--target", overlay, "--python", filepath.Join(m.layout.Root, "opt/cozy/python/bin/python")}, candidate...)
	if _, err := exec.CommandContext(ctx, filepath.Join(m.layout.Root, "usr/local/bin/uv"), args...).CombinedOutput(); err != nil {
		return skip("candidate staging failed; verified installed pair retained")
	}
	probed, err := m.startupVerify(ctx, overlay)
	if err != nil || probed != next {
		return skip("candidate failed native/dependency preflight; verified installed pair retained")
	}
	status = startupStatus{State: "installing", Before: verified, After: next, Previous: kept}
	if err := m.startupSave(status); err != nil {
		return err
	}
	err = installPair(m.layout.Root, candidate)
	if err == nil {
		probed, err = m.startupVerify(ctx)
	}
	if err == nil && probed != next {
		err = errors.New("installed pair differs from staged pair")
	}
	if err == nil {
		err = selectPair(m.layout.Root, candidate)
	}
	if err != nil {
		status.Detail = "candidate installation or probe failed"
		if rollback := m.startupRollback(ctx, status); rollback != nil {
			return fmt.Errorf("startup update failed: %v; rollback failed: %w", err, rollback)
		}
		fmt.Fprintln(m.log, "cozy machine: startup update rolled back to verified installed pair")
		return nil
	}
	status.State = "boot_pending"
	if err := m.startupSave(status); err != nil {
		return err
	}
	fmt.Fprintf(m.log, "cozy machine: startup selected Runtime %s / TensorFS %s\n", next.Runtime, next.TensorFS)
	m.mu.Lock()
	m.grant.startupPending = true
	m.mu.Unlock()
	return nil
}

// Only the existing privileged launcher can commit or repair its private update
// journal after the public agent has dropped privilege.
func startupMaintenance(root, operation string) error {
	m := &Machine{layout: NewLayout(&Grant{Root: root}), log: io.Discard}
	raw, err := os.ReadFile(m.startupPath("status.json"))
	var status startupStatus
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, &status); err != nil {
		return err
	}
	if operation == "startup-rollback" {
		status.Detail = "candidate failed authenticated Runtime startup"
		return m.startupRollback(context.Background(), status)
	}
	if status.State != "boot_pending" {
		return errors.New("no startup candidate awaits readiness")
	}
	got, err := m.startupVerify(context.Background())
	if err != nil || got != status.After {
		return errors.New("ready Runtime does not match startup candidate")
	}
	status.State = "succeeded"
	return m.startupSave(status)
}
