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
	Operation           string
	Pin                 bool
	Agent               string
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

type agentIdentity struct {
	Name             string   `json:"name"`
	WireMinor        uint32   `json:"wire_minor"`
	MinimumWireMinor uint32   `json:"minimum_wire_minor"`
	Capabilities     []string `json:"capabilities"`
}

func (m *Machine) startupVerify(ctx context.Context, overlay ...string) (pairVersions, error) {
	var agent agentIdentity
	selection := m.transactionAgent()
	if m.grant != nil && m.grant.agentSelection != "" {
		selection = m.grant.agentSelection
	}
	if len(overlay) == 0 && !m.transactionPending() && m.grant != nil && m.grant.agentBefore != "" {
		var err error
		agent, err = m.readAgentIdentity(ctx, m.grant.agentBefore)
		if err != nil {
			return pairVersions{}, err
		}
	} else if selection == "bundled" {
		location := filepath.Join(m.layout.Root, "opt/cozy/python")
		if len(overlay) > 0 {
			location = overlay[0]
		}
		var err error
		agent, err = m.bundledIdentity(ctx, location)
		if err != nil {
			return pairVersions{}, err
		}
	} else {
		path := m.selectedApplication()
		if m.grant != nil && m.grant.agentBefore != "" {
			path = m.grant.agentBefore
		}
		var err error
		agent, err = m.readAgentIdentity(ctx, path)
		if err != nil {
			return pairVersions{}, err
		}
	}
	var probe startupProbe
	if err := m.startupPython(ctx, &probe, append([]string{"probe"}, overlay...)...); err != nil {
		return pairVersions{}, err
	}
	if probe.WireMinor < agent.MinimumWireMinor || probe.MinimumWireMinor > agent.WireMinor || !slices.Contains(probe.Capabilities, "machine-supervisor/1") || !slices.Contains(probe.Capabilities, "machine-public-reads/1") {
		return pairVersions{}, errors.New("Runtime does not implement the selected agent's supported protocol/capabilities")
	}
	return probe.pairVersions, nil
}

func (m *Machine) bundledIdentity(ctx context.Context, overlay string) (agentIdentity, error) {
	return m.readAgentIdentity(ctx, filepath.Join(overlay, "bin/cozy-machine"))
}
func (m *Machine) readAgentIdentity(ctx context.Context, path string) (agentIdentity, error) {
	cmd := exec.CommandContext(ctx, path, "version", "--json")
	cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + filepath.Join(m.layout.Root, "tmp")}
	raw, err := progressChildOutput(cmd, false)
	var identity agentIdentity
	if err != nil {
		return identity, fmt.Errorf("candidate Runtime requires a working bundled machine agent: %w", err)
	}
	if err = json.Unmarshal(raw, &identity); err != nil {
		return identity, err
	}
	if identity.Name != "cozy-machine" || identity.MinimumWireMinor > identity.WireMinor {
		return identity, errors.New("candidate bundled agent has invalid identity or wire range")
	}
	for _, required := range []string{"hub-access/1", "runtime-update/1", bootstrapABI} {
		if !slices.Contains(identity.Capabilities, required) {
			return identity, fmt.Errorf("candidate bundled agent lacks %s", required)
		}
	}
	return identity, nil
}

func (m *Machine) verifyBundledAgent(ctx context.Context, overlay string) error {
	_, err := m.bundledIdentity(ctx, overlay)
	return err
}

type startupWheel struct {
	File, SHA256, Version, URL string
}

type startupStatus struct {
	Dependencies     []dependencyWheel `json:"dependencies,omitempty"`
	EntryAfter       string            `json:"entry_after,omitempty"`
	AgentBefore      string            `json:"agent_before,omitempty"`
	PolicyBefore     []byte            `json:"policy_before,omitempty"`
	PolicyAfter      *softwarePolicy   `json:"policy_after,omitempty"`
	EntryBefore      string            `json:"entry_before,omitempty"`
	EntryFile        bool              `json:"entry_file,omitempty"`
	EntryMissing     bool              `json:"entry_missing,omitempty"`
	Workspace        string            `json:"workspace,omitempty"`
	Pinned           bool              `json:"pinned,omitempty"`
	PreviouslyPinned bool              `json:"previously_pinned,omitempty"`
	State            string            `json:"state"` // installing | succeeded | rolled_back | skipped
	Detail           string            `json:"detail,omitempty"`
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

// A successful explicit maintenance selection owns its particular installed pair.
// This file can only defer discovery; it never authorizes privileged mutation.
func (m *Machine) explicitlyPinnedPair(_ pairVersions) bool {
	policy, _, err := readSoftwarePolicy(m.layout.Root)
	return err == nil && policy.StartupUpdate == "off"
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
	if err := m.removeDependencies(ctx, status.Dependencies); err != nil {
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
	if err := m.rollbackSoftwarePolicy(status); err != nil {
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
		// uv replaces distributions separately. Before, After, and their mixed
		// pairs can all be interrupted states, including partly written payloads
		// or additive dependencies while metadata still reports Before. Only a
		// version outside this transaction proves a foreign installation. A later
		// update through this engine owns its own journal, even for the same pair.
		if installed, err := m.installedPair(); err == nil {
			foreignRuntime := installed.Runtime != status.Before.Runtime && installed.Runtime != status.After.Runtime
			foreignTensorFS := installed.TensorFS != status.Before.TensorFS && installed.TensorFS != status.After.TensorFS
			if foreignRuntime || foreignTensorFS {
				status.State = "superseded"
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
	// A durable explicit candidate owns selection across request-plane restarts.
	var update updateStatus
	if raw, err := os.ReadFile(m.updatePath()); err == nil && json.Unmarshal(raw, &update) == nil && update.Request != nil && !update.terminal() {
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
		return skip("installed pair is owned by an explicit update")
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
	if next == verified && m.applicationMatchesBundle() {
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
	status.PolicyAfter = nil
	status.PolicyBefore = nil
	status.EntryBefore = ""
	status.EntryFile = false
	status.EntryMissing = false
	if err := m.preparePair(ctx, verified, next, candidate, previous, status, false); err != nil {
		fmt.Fprintln(m.log, "cozy machine: update preparation failed; repair remains available:", err)
		if m.transactionPending() {
			return err
		}
	}
	return nil
}

// stagePair installs and verifies the candidate in a private overlay, leaving
// the live interpreter untouched until guarded activation.
func (m *Machine) stagePair(ctx context.Context, verified, next pairVersions, candidate, previous []string, operation ...string) (*preparedPair, error) {
	if !samePair(candidate, next) || !samePair(previous, verified) {
		return nil, errors.New("update pair is incomplete")
	}
	var private []string
	for _, path := range candidate {
		digest, err := hashFile(path)
		if err != nil {
			return nil, err
		}
		copy, err := m.startupCopyWheel(path, "candidate", digest)
		if err != nil {
			return nil, err
		}
		private = append(private, copy)
	}
	candidate = private
	kept, err := m.startupPrevious(previous)
	if err != nil {
		return nil, err
	}
	additions, err := m.stageDependencies(ctx, candidate)
	if err != nil {
		return nil, err
	}
	complete := append([]string{}, candidate...)
	for _, wheel := range additions {
		complete = append(complete, wheel.Path)
	}
	// Stage an overlay: uv touches no existing dependency or interpreter. Python
	// imports the two candidate distributions from it against the actual base deps.
	overlay, err := os.MkdirTemp(m.startupPath(), "probe-")
	if err != nil {
		return nil, err
	}
	retained := false
	defer func() {
		if !retained {
			_ = os.RemoveAll(overlay)
		}
	}()
	args := append([]string{"pip", "install", "--no-config", "--offline", "--no-deps", "--target", overlay, "--python", filepath.Join(m.layout.Root, "opt/cozy/python/bin/python")}, complete...)
	cmd := exec.CommandContext(ctx, filepath.Join(m.layout.Root, "usr/local/bin/uv"), args...)
	cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + filepath.Join(m.layout.Root, "tmp")}
	if _, err := progressChildOutput(cmd, true); err != nil {
		return nil, fmt.Errorf("candidate staging failed: %w", err)
	}
	probed, err := m.startupVerify(ctx, overlay)
	if err != nil || probed != next {
		return nil, fmt.Errorf("candidate failed native/dependency preflight: %v", err)
	}
	// Persist every overlay entry before publishing the prepared manifest.
	if err := filepath.Walk(overlay, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		return f.Sync()
	}); err != nil {
		return nil, err
	}
	if err := syncDir(m.startupPath()); err != nil {
		return nil, err
	}
	identities := map[string]string{}
	for _, wheel := range candidate {
		digest, err := hashFile(wheel)
		if err != nil {
			return nil, err
		}
		identities[wheel] = digest
	}
	prepared := &preparedPair{CandidateHashes: identities, Before: verified, After: next, Candidate: candidate, Previous: kept, Dependencies: additions, Overlay: overlay}
	if len(operation) > 0 {
		prepared.Operation = operation[0]
	}
	var old preparedPair
	if raw, err := os.ReadFile(m.startupPath("prepared.json")); err == nil {
		_ = json.Unmarshal(raw, &old)
	}
	raw, _ := json.Marshal(prepared)
	if err := writeAtomic(m.startupPath("prepared.json"), raw, 0o600); err != nil {
		return nil, err
	}
	retained = true
	if old.Overlay != overlay {
		m.removePreparedOverlay(old.Overlay)
	}
	return prepared, nil
}

// preparedPair is separate from the installation journal: preparing it never mutates
// the running interpreter or starts rollback recovery.
type preparedPair struct {
	CandidateHashes map[string]string
	Operation       string
	Before, After   pairVersions
	Candidate       []string
	Previous        []startupWheel
	Dependencies    []dependencyWheel
	Overlay         string
}

func (m *Machine) removePreparedOverlay(path string) {
	if filepath.Dir(path) == m.startupPath() && strings.HasPrefix(filepath.Base(path), "probe-") {
		_ = os.RemoveAll(path)
	}
}

func (m *Machine) preparePair(ctx context.Context, verified, next pairVersions, candidate, previous []string, status startupStatus, pinned bool) error {
	prepared, err := m.stagePair(ctx, verified, next, candidate, previous)
	if err != nil {
		return err
	}
	return m.activatePair(ctx, prepared, status, pinned)
}

func (m *Machine) activatePair(ctx context.Context, prepared *preparedPair, status startupStatus, pinned bool) error {
	verified, next, candidate, kept, additions := prepared.Before, prepared.After, prepared.Candidate, prepared.Previous, prepared.Dependencies
	if len(candidate) != 2 || len(prepared.CandidateHashes) != 2 {
		return errors.New("prepared candidate has no complete wheel identities")
	}
	for _, wheel := range candidate {
		digest, err := hashFile(wheel)
		if err != nil || !sha256Hex.MatchString(prepared.CandidateHashes[wheel]) || digest != prepared.CandidateHashes[wheel] {
			return errors.New("prepared candidate wheel changed")
		}
	}
	for _, wheel := range additions {
		digest, err := hashFile(wheel.Path)
		if err != nil || !sha256Hex.MatchString(wheel.SHA256) || digest != wheel.SHA256 {
			return errors.New("prepared dependency wheel changed")
		}
	}

	complete := append([]string{}, candidate...)
	for _, wheel := range additions {
		complete = append(complete, wheel.Path)
	}
	probed, err := m.startupVerify(ctx, prepared.Overlay)
	if err != nil || probed != next {
		return fmt.Errorf("prepared candidate failed revalidation: %v", err)
	}

	if m.grant != nil {
		status.AgentBefore = m.grant.agentBefore
	}
	if status.PolicyAfter == nil {
		if err := m.snapshotPolicy(&status, nil); err != nil {
			return err
		}
	}
	m.preparePolicyEntry(&status)
	status.Dependencies = additions
	status.PreviouslyPinned, status.Pinned = m.explicitlyPinnedPair(verified), pinned
	status.State, status.Before, status.After, status.Previous = "installing", verified, next, kept
	if err := m.startupSave(status); err != nil {
		return err
	}
	err = installPair(m.layout.Root, complete)
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
func startupMaintenance(root, operation, workspace, detail string) error {
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
		if detail != "" {
			status.Detail = detail
		} else if status.Detail == "" {
			status.Detail = "candidate failed authenticated Runtime startup"
		}
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
	if err := m.commitSoftwarePolicy(&status); err != nil {
		return err
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
	policy, _, err := readSoftwarePolicy(root)
	return err == nil && policy.StartupUpdate == "auto", err
}

// An application-only change still needs the same rollback/readiness transaction,
// even when both Python distribution versions are already current.
func (m *Machine) applicationMatchesBundle() bool {
	if m.grant == nil || m.grant.agentBefore == "" || m.transactionAgent() != "bundled" {
		return true
	}
	current, err := hashFile(m.grant.agentBefore)
	if err != nil {
		return false
	}
	selected, err := hashFile(filepath.Join(m.layout.Root, "opt/cozy/python/bin/cozy-machine"))
	return err == nil && current == selected
}
