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

type softwareState struct {
	pairVersions
	Eligible     bool
	Quiescent    bool
	AcceptedWork bool   `json:"accepted_work"`
	Workspace    string `json:"workspace_id"`
}

type pairTransaction struct {
	Before, After       pairVersions
	Candidate, Previous []string
	Store, Lifetime     string
}

type startupProbe struct {
	pairVersions
	WireMinor        uint32   `json:"wire_minor"`
	MinimumWireMinor uint32   `json:"minimum_wire_minor"`
	Capabilities     []string `json:"capabilities"`
}

func (m *Machine) startupVerify(ctx context.Context, overlay ...string) (pairVersions, error) {
	if len(overlay) != 0 {
		if err := m.verifyBundledAgent(ctx, overlay[0]); err != nil {
			return pairVersions{}, err
		}
	}
	var probe startupProbe
	if err := m.startupPython(ctx, &probe, append([]string{"probe"}, overlay...)...); err != nil {
		return pairVersions{}, err
	}
	if probe.WireMinor < pb.MinCompatibleWireMinor || probe.MinimumWireMinor > pb.WireMinor || !slices.Contains(probe.Capabilities, "machine-supervisor/1") || !slices.Contains(probe.Capabilities, "machine-public-reads/1") {
		return pairVersions{}, errors.New("Runtime does not implement this agent's supported protocol/capabilities")
	}
	return probe.pairVersions, nil
}

// Every default update carries the next boot's agent. Its advertised contracts
// and protocol range are checked before uv can replace the installed pair.
func (m *Machine) verifyBundledAgent(ctx context.Context, overlay string) error {
	cmd := exec.CommandContext(ctx, filepath.Join(overlay, "bin/cozy-machine"), "version", "--json")
	cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + filepath.Join(m.layout.Root, "tmp")}
	raw, err := progressChildOutput(cmd, false)
	if err != nil {
		return fmt.Errorf("candidate Runtime requires a working bundled machine agent: %w", err)
	}
	var identity struct {
		Name             string   `json:"name"`
		WireMinor        uint32   `json:"wire_minor"`
		MinimumWireMinor uint32   `json:"minimum_wire_minor"`
		Capabilities     []string `json:"capabilities"`
	}
	if err := json.Unmarshal(raw, &identity); err != nil {
		return fmt.Errorf("candidate bundled agent identity: %w", err)
	}
	if identity.Name != "cozy-machine" || identity.WireMinor < pb.MinCompatibleWireMinor || identity.MinimumWireMinor > pb.WireMinor {
		return errors.New("candidate bundled agent has an unsupported identity or wire range")
	}
	for _, required := range build.Capabilities() {
		if !slices.Contains(identity.Capabilities, required) {
			return fmt.Errorf("candidate bundled agent lacks %s", required)
		}
	}
	return nil
}

type startupWheel struct {
	File, SHA256, Version, URL string
}

type startupStatus struct {
	Workspace        string `json:"workspace,omitempty"`
	Pinned           bool   `json:"pinned,omitempty"`
	PreviouslyPinned bool   `json:"previously_pinned,omitempty"`
	State            string `json:"state"` // installing | succeeded | rolled_back | skipped
	Detail           string `json:"detail,omitempty"`
	Before, After    pairVersions
	Previous         []startupWheel `json:"previous,omitempty"`
	Failed           []pairVersions `json:"failed,omitempty"`
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
	if err := writeAtomic(filepath.Join(m.layout.State, "startup-update-status.json"), raw, 0o644); err != nil {
		fmt.Fprintln(m.log, "cozy machine: startup report unavailable:", err)
	}
	return nil
}

func (m *Machine) startupPython(ctx context.Context, into any, arguments ...string) error {
	args := append([]string{"-I", "-c", startupUpdateScript}, arguments...)
	cmd := exec.CommandContext(ctx, filepath.Join(m.layout.Root, "opt/cozy/python/bin/python"), args...)
	cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + filepath.Join(m.layout.Root, "tmp")}
	out, err := progressChildOutput(cmd, false)
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
	var transaction startupStatus
	if raw, err := os.ReadFile(m.startupPath("status.json")); err == nil && json.Unmarshal(raw, &transaction) == nil {
		if transaction.State == "succeeded" && transaction.After == installed && transaction.Pinned || transaction.State == "rolled_back" && transaction.Before == installed && transaction.PreviouslyPinned {
			return true
		}
	}
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
	if !slices.Contains(status.Failed, status.After) {
		status.Failed = append(status.Failed, status.After)
	}
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
		// Metadata identifies the installation independently of the health probe.
		// A later explicit installation supersedes this transaction even if off.
		if installed, err := m.installedPair(); err == nil {
			if installed != status.After {
				status.State = "superseded"
				if installed == status.Before {
					status.State = "rolled_back"
					if !slices.Contains(status.Failed, status.After) {
						status.Failed = append(status.Failed, status.After)
					}
				}
				if err := m.startupSave(status); err != nil {
					fmt.Fprintln(m.log, "cozy machine: recovery report:", err)
				}
				return nil
			}
		}
		status.Detail = "recovering interrupted startup update"
		if err := m.startupRollback(ctx, status); err != nil {
			return fmt.Errorf("interrupted startup update could not be repaired: %w", err)
		}
		fmt.Fprintln(m.log, "cozy machine: restored the verified pair after interrupted startup update")
		return nil
	}
	automatic, policyErr := automaticSoftwareUpdates(m.layout.Root)
	if policyErr != nil {
		fmt.Fprintln(m.log, "cozy machine: startup update policy deferred:", policyErr)
	}
	if !automatic || strings.ContainsAny(build.Version, "-+") {
		fmt.Fprintln(m.log, "cozy machine: startup update skipped: explicitly pinned or development agent")
		return nil
	}
	skip := func(detail string) error {
		fmt.Fprintln(m.log, "cozy machine: startup update retained installed pair:", detail)
		if status.State == "" {
			status.State = "skipped"
		}
		status.Detail = detail
		if err := m.startupSave(status); err != nil {
			fmt.Fprintln(m.log, "cozy machine: startup report:", err)
		}
		return nil
	}
	var installed softwareState
	if err := m.startupPython(ctx, &installed, "inspect", m.layout.Store); err != nil {
		return skip("installed metadata or public work journal is unreadable; update deferred")
	}
	if !installed.Eligible || !installed.Quiescent || m.grant.Lifetime != "persistent" && installed.AcceptedWork {
		return skip("development pair, retained work, or previously used rental; update deferred")
	}
	if m.explicitlyPinnedPair(installed.pairVersions) {
		return skip("installed pair is owned by an explicit wheel update")
	}
	verified, err := m.startupVerify(ctx)
	if err != nil {
		return skip("installed pair failed verification; repair API remains available: " + err.Error())
	}
	var selected map[string]startupWheel
	if err := m.startupPython(ctx, &selected, "plan"); err != nil {
		return skip("release discovery unavailable or no compatible published pair; verified installed pair retained")
	}
	next := pairVersions{Runtime: selected["cozy-runtime"].Version, TensorFS: selected["tensorfs"].Version}
	if slices.Contains(status.Failed, next) {
		return skip("candidate previously failed; waiting for a different published pair")
	}
	if next == verified {
		return skip("installed pair is current within its compatible major lines")
	}
	if err := os.MkdirAll(m.startupPath(), 0o700); err != nil {
		return skip("cannot create update staging: " + err.Error())
	}
	downloads := &Machine{layout: Layout{State: m.startupPath("downloads")}}
	var candidate []string
	for _, name := range []string{"cozy-runtime", "tensorfs"} {
		wheel := selected[name]
		if wheelDistribution(wheel.File) != name || wheelVersion(wheel.File) != wheel.Version || !sha256Hex.MatchString(wheel.SHA256) || !strings.HasPrefix(wheel.URL, "https://files.pythonhosted.org/") {
			return skip("published wheel identity is invalid")
		}
		path, err := downloads.fetchWheel(ctx, name, wheel.Version)
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
			path, err := downloads.fetchWheel(ctx, name, version)
			if err != nil {
				return skip("verified installed wheels cannot be retained for offline rollback")
			}
			previous = append(previous, path)
		}
	}
	status.Workspace = installed.Workspace
	if err := m.preparePair(ctx, verified, next, candidate, previous, status, false); err != nil {
		fmt.Fprintln(m.log, "cozy machine: update preparation failed; repair remains available:", err)
		if m.transactionPending() {
			return err
		}
	}
	return nil
}

// preparePair is the sole stage/install/recovery engine for automatic and
// explicit updates. Both finish through the same authenticated readiness gate.
func (m *Machine) preparePair(ctx context.Context, verified, next pairVersions, candidate, previous []string, status startupStatus, pinned bool) error {
	if !samePair(candidate, next) || !samePair(previous, verified) {
		return errors.New("update pair is incomplete")
	}
	var private []string
	for _, path := range candidate {
		digest, err := hashFile(path)
		if err != nil {
			return err
		}
		copy, err := m.startupCopyWheel(path, "candidate", digest)
		if err != nil {
			return err
		}
		private = append(private, copy)
	}
	candidate = private
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
	cmd := exec.CommandContext(ctx, filepath.Join(m.layout.Root, "usr/local/bin/uv"), args...)
	cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + filepath.Join(m.layout.Root, "tmp")}
	if _, err := progressChildOutput(cmd, true); err != nil {
		return fmt.Errorf("candidate staging failed: %w", err)
	}
	probed, err := m.startupVerify(ctx, overlay)
	if err != nil || probed != next {
		if !slices.Contains(status.Failed, next) {
			status.Failed = append(status.Failed, next)
		}
		if status.State == "" {
			status.State = "skipped"
		}
		status.Detail = "candidate failed native/dependency preflight"
		_ = m.startupSave(status)
		return fmt.Errorf("candidate failed native/dependency preflight: %v", err)
	}
	status.PreviouslyPinned, status.Pinned = m.explicitlyPinnedPair(verified), pinned
	status.State, status.Before, status.After, status.Previous = "installing", verified, next, kept
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
func startupMaintenance(root, operation string, workspace string) error {
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
		if status.State != "installing" && status.State != "boot_pending" {
			return errors.New("no interrupted update awaits rollback")
		}
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
	if status.Workspace != "" && status.Workspace != workspace {
		return errors.New("ready Runtime changed the execution workspace")
	}
	status.State = "succeeded"
	return m.startupSave(status)
}

func (m *Machine) transactionPending() bool {
	var status startupStatus
	raw, err := os.ReadFile(m.startupPath("status.json"))
	return err == nil && json.Unmarshal(raw, &status) == nil && (status.State == "installing" || status.State == "boot_pending")
}

// The installer's typed machine policy is independent of transport credentials
// and Runtime configuration. It controls discovery, never explicit maintenance.
func automaticSoftwareUpdates(root string) (bool, error) {
	raw, err := os.ReadFile(filepath.Join(root, "etc/cozy/software-policy.json"))
	if errors.Is(err, os.ErrNotExist) {
		return true, nil
	}
	if err != nil {
		return false, err
	}
	var policy struct {
		StartupUpdate string `json:"startup_update"`
	}
	if err := json.Unmarshal(raw, &policy); err != nil {
		return false, err
	}
	switch policy.StartupUpdate {
	case "auto":
		return true, nil
	case "off":
		return false, nil
	}
	return false, errors.New("software-policy.json requires startup_update auto or off")
}
