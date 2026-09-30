package host

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/build"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/hostruntime"
)

// A guarded Runtime update replaces the machine's Runtime and TensorFS wheels in place. It
// waits until the Runtime accepts a guarded restart, which it refuses while any work runs,
// installs the new pair, relaunches the Runtime and checks it answers at the new versions.
// A Runtime that does not is replaced by the pair it had. The owner drives it over the
// machine endpoint with a maintenance capability:
//
//	GET  /v1/machine/runtime                 the installed pair and the last update
//	PUT  /v1/machine/runtime/wheels/{file}   stage one wheel
//	POST /v1/machine/runtime/update          {operation, pin, runtime, tensorfs}: each pair member a staged
//	                                         {file, sha256}, a {version} to fetch, or omitted
//
// Staged wheels serve the next accepted operation. Uploads are refused while an
// operation owns staging; its terminal status is published only after cleanup.
//
// A machine without these routes predates native updates.

// UpdateCapability is what GET /v1/machine/runtime advertises.
const UpdateCapability = "runtime-update/1"

const maxWheelBytes = 256 << 20

type wheelChoice struct {
	File    string `json:"file,omitempty"`
	SHA256  string `json:"sha256,omitempty"`
	Version string `json:"version,omitempty"`
}

type updateRequest struct {
	Operation string       `json:"operation"`
	Agent     string       `json:"agent"`
	Pin       *bool        `json:"pin"`
	Runtime   *wheelChoice `json:"runtime,omitempty"`
	TensorFS  *wheelChoice `json:"tensorfs,omitempty"`
}

type pairVersions struct {
	Runtime  string `json:"runtime"`
	TensorFS string `json:"tensorfs"`
}

type updateStatus struct {
	Operation        string         `json:"operation"`
	Request          *updateRequest `json:"request,omitempty"`
	State            string         `json:"state"` // waiting | preparing | waiting_activation | installing | starting | rolling_back | succeeded | rolled_back | failed
	Error            string         `json:"error,omitempty"`
	From             pairVersions   `json:"from"`
	To               pairVersions   `json:"to"`
	Pinned           bool           `json:"pinned,omitempty"`            // the caller explicitly selected this installed pair
	PreviouslyPinned bool           `json:"previously_pinned,omitempty"` // failed maintenance retains the previous selection
}

func (s updateStatus) terminal() bool {
	return s.State == "succeeded" || s.State == "rolled_back" || s.State == "failed"
}

type runtimeUpdates struct {
	staging sync.Mutex // uploads, admission and cleanup; never needed for observation
	mu      sync.Mutex
	status  *updateStatus
	active  string // held through cleanup, even after the installer has finished
}

var (
	operationName = regexp.MustCompile(`^[A-Za-z0-9._-]{1,128}$`)
	sha256Hex     = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

func (m *Machine) updatePath() string { return filepath.Join(m.layout.State, "runtime-update.json") }
func (m *Machine) stagePath(names ...string) string {
	return filepath.Join(append([]string{m.layout.State, "runtime-update"}, names...)...)
}

// openUpdates resumes a prepared request after initialization. Interrupted shared
// installation is recovered by the private journal before Runtime admission.
func (m *Machine) openUpdates() {
	var status updateStatus
	if raw, err := os.ReadFile(m.updatePath()); err == nil && json.Unmarshal(raw, &status) == nil {
		if status.Request != nil && (status.State == "waiting" || status.State == "preparing" || status.State == "waiting_activation") {
			m.updates.status, m.updates.active = &status, status.Operation
			go func() { <-m.initialized; m.runUpdate(*status.Request, status) }()
			return
		}

		if !status.terminal() && (m.grant.bootstrap == nil || !m.grant.bootstrap.Pending) {
			status.State, status.Error = "failed", "the machine restarted during its Runtime update; installation recovery runs before admission"
			if m.grant.bootstrap != nil && m.grant.bootstrap.CommittedPair != nil && *m.grant.bootstrap.CommittedPair == status.To {
				status.State, status.Error = "succeeded", ""
			}
			if m.grant.bootstrap != nil && m.grant.bootstrap.RolledBack {
				status.State, status.Error = "rolled_back", "candidate request plane exited; previous installation restored"
			}
			m.saveUpdate(status)
		}
		m.updates.status = &status
	}
}

func (m *Machine) saveUpdate(status updateStatus) error {
	m.updates.mu.Lock()
	defer m.updates.mu.Unlock()
	if err := m.writeUpdate(status); err != nil {
		return err
	}
	m.updates.status = &status
	return nil
}

// writeUpdate persists a status without publishing it. Admission must succeed
// durably before the operation can start or replace the previous status.
func (m *Machine) writeUpdate(status updateStatus) error {
	raw, _ := json.Marshal(status)
	if err := writeAtomic(m.updatePath(), raw, 0o600); err != nil {
		fmt.Fprintln(m.log, "cozy machine: the Runtime update state was not recorded:", err)
		return err
	}
	return nil
}

func (m *Machine) updating() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.inUpdate
}

func (m *Machine) setUpdating(on bool) {
	m.mu.Lock()
	m.inUpdate = on
	m.mu.Unlock()
}

// maintenance admits a request carrying the owner's maintenance capability.
func (m *Machine) maintenance(w http.ResponseWriter, r *http.Request) bool {
	token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Cozy-Cap ")
	grant, err := capability.Verify(token, m.grant.WorkerID, m.claims.authorizedKeys(), time.Now(), "")
	if err == nil && !grant.Permits(capability.Maintenance) {
		err = capability.ErrScope
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return false
	}
	if !m.launcherReady() {
		runtimeStarting(w, "starting")
		return false
	}
	return true
}

func (m *Machine) serveRuntimeState(w http.ResponseWriter, r *http.Request) {
	if !m.maintenance(w, r) {
		return
	}
	installed, err := m.installedPair()
	m.updates.mu.Lock()
	status := m.updates.status
	m.updates.mu.Unlock()
	answer := map[string]any{"capabilities": []string{UpdateCapability, "runtime-update/2"}, "runtime": installed.Runtime,
		"tensorfs": installed.TensorFS, "update": status, "phase": m.phase()}
	digest, _ := hashFile("/proc/self/exe")
	selection, repairFallback := m.runningAgentSelection(digest)
	if m.grant.bootstrap != nil {
		answer["bootstrap"] = map[string]any{"abi": bootstrapABI, "version": m.grant.bootstrap.BootstrapVersion, "update_boundary": "image-or-service-bootstrap"}
	}
	answer["agent"] = map[string]any{"version": build.Version, "sha256": digest, "selection": selection, "capabilities": build.Capabilities(), "repair_fallback": repairFallback}
	var startup startupStatus
	if raw, err := os.ReadFile(filepath.Join(m.layout.State, "startup-update-status.json")); err == nil && json.Unmarshal(raw, &startup) == nil {
		answer["startup_update"] = startup
	}
	if err != nil {
		answer["error"] = err.Error()
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(answer)
}

func (m *Machine) serveStageWheel(w http.ResponseWriter, r *http.Request) {
	if !m.maintenance(w, r) {
		return
	}
	if m.startupPending() {
		http.Error(w, "startup update is awaiting Runtime readiness", http.StatusServiceUnavailable)
		return
	}
	// Staging belongs to the next accepted update. Do not acknowledge an upload
	// while that update owns the directory or is still cleaning it up.
	m.updates.staging.Lock()
	defer m.updates.staging.Unlock()
	if m.updating() {
		http.Error(w, "software maintenance still owns Runtime admission", http.StatusConflict)
		return
	}
	m.updates.mu.Lock()
	busy := m.updates.active != "" || m.updates.status != nil && !m.updates.status.terminal()
	m.updates.mu.Unlock()
	if busy {
		http.Error(w, "the machine is running a Runtime update; stage wheels after it finishes", http.StatusConflict)
		return
	}
	file := r.PathValue("file")
	if wheelDistribution(file) == "" || wheelVersion(file) == "" {
		http.Error(w, "stage a cozy_runtime or tensorfs wheel", http.StatusBadRequest)
		return
	}
	sha, size, err := m.stageWheel(file, http.MaxBytesReader(w, r.Body, maxWheelBytes))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"file": file, "sha256": sha, "length": size})
}

// stageWheel keeps a wheel under the staging directory as <sha256>/<file>.
func (m *Machine) stageWheel(file string, body io.Reader) (string, int64, error) {
	if err := os.MkdirAll(m.stagePath(), 0o755); err != nil {
		return "", 0, err
	}
	temp, err := os.CreateTemp(m.stagePath(), ".upload-*")
	if err != nil {
		return "", 0, err
	}
	defer os.Remove(temp.Name())
	sum := sha256.New()
	size, err := io.Copy(io.MultiWriter(temp, sum), body)
	if err == nil {
		err = temp.Chmod(0o644)
	}
	if closeErr := temp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return "", 0, err
	}
	digest := hex.EncodeToString(sum.Sum(nil))
	dir := m.stagePath(digest)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", 0, err
	}
	return digest, size, os.Rename(temp.Name(), filepath.Join(dir, file))
}

// Report the running application, including a retained repair fallback.
func (m *Machine) runningAgentSelection(digest string) (string, bool) {
	selection := "explicit"
	if bundledAgentPolicy(m.layout.Root) {
		selection = "bundled"
	}
	if m.grant.bootstrap != nil && m.startupPending() {
		selection = m.grant.bootstrap.Selection
	}
	if selection == "bundled" {
		bundleHash, err := hashFile(filepath.Join(m.layout.Root, "opt/cozy/python/bin/cozy-machine"))
		if err != nil || bundleHash != digest {
			return "explicit", true
		}
	}
	return selection, false
}

func (m *Machine) serveUpdate(w http.ResponseWriter, r *http.Request) {
	if !m.maintenance(w, r) {
		return
	}
	var request updateRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&request); err != nil || !operationName.MatchString(request.Operation) ||
		(request.Agent != "" && request.Agent != "bundled" && request.Agent != "explicit") || request.Runtime == nil && request.TensorFS == nil {
		http.Error(w, "an update requires operation, valid agent selection, and a Runtime or TensorFS", http.StatusBadRequest)
		return
	}
	// Older clients omit selection intent. Preserve their unpinned update and
	// the currently running agent instead of requiring synchronized upgrades.
	if request.Pin == nil {
		pin := false
		request.Pin = &pin
	}
	if request.Agent == "" {
		if digest, err := hashFile("/proc/self/exe"); err == nil {
			request.Agent, _ = m.runningAgentSelection(digest)
		} else {
			request.Agent = m.transactionAgent()
		}
		if request.Agent != "bundled" && request.Agent != "explicit" {
			request.Agent = "explicit"
		}
	}
	m.updates.staging.Lock()
	defer m.updates.staging.Unlock()
	m.updates.mu.Lock()
	current := m.updates.status
	busy := m.updates.active != "" || current != nil && !current.terminal()
	m.updates.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	switch {
	case current != nil && current.Operation == request.Operation:
		_ = json.NewEncoder(w).Encode(current) // the same operation, asked again
		return
	case busy:
		http.Error(w, "the machine is running another Runtime update: "+current.Operation, http.StatusConflict)
		return
	}
	if m.startupPending() {
		runtimeStarting(w, "booting")
		return
	}
	if m.updating() {
		// A resumed rollback can persist its terminal report before exiting
		// into the restored application. That report does not release admission.
		http.Error(w, "software maintenance still owns Runtime admission", http.StatusConflict)
		return
	}
	m.mu.Lock()
	process, ready := m.proc, m.ready
	m.mu.Unlock()
	if process != nil && !process.exited() {
		select {
		case <-ready:
		default:
			runtimeStarting(w, "booting")
			return
		}
	}
	installed, err := m.installedPair()
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	pinned := *request.Pin
	status := updateStatus{Operation: request.Operation, State: "waiting", From: installed, Pinned: pinned, Request: &request}
	status.PreviouslyPinned = m.explicitlyPinnedPair(installed)
	if err := m.writeUpdate(status); err != nil {
		http.Error(w, "cannot record the Runtime update: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	m.updates.mu.Lock()
	m.updates.status, m.updates.active = &status, request.Operation
	m.updates.mu.Unlock()
	go m.runUpdate(request, status)
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(status)
}

// installedPair reads the Runtime and TensorFS versions opt/cozy/python holds.
func (m *Machine) installedPair() (pairVersions, error) {
	out, err := childOutput(exec.Command(filepath.Join(m.layout.Root, "opt/cozy/python/bin/python"), "-I", "-c",
		"import importlib.metadata as m; print(m.version('"+hostruntime.Distribution+"')); print(m.version('tensorfs'))"), false)
	lines := strings.Fields(string(out))
	if err != nil || len(lines) != 2 {
		return pairVersions{}, fmt.Errorf("the machine's Runtime environment is unreadable: %v", err)
	}
	return pairVersions{Runtime: lines[0], TensorFS: lines[1]}, nil
}

// runUpdate performs one update to its end, then records it.
func (m *Machine) runUpdate(request updateRequest, status updateStatus) {
	defer func() {
		m.updates.staging.Lock()
		defer m.updates.staging.Unlock()
		m.updates.mu.Lock()
		owned := m.updates.active == request.Operation
		m.updates.mu.Unlock()
		if owned {
			m.completeUpdateFiles(status)
			m.setUpdating(false)
			m.updates.mu.Lock()
			m.updates.status, m.updates.active = &status, ""
			m.updates.mu.Unlock()
		}
	}()
	fail := func(state string, err error) {
		status.State, status.Error = state, err.Error()
		fmt.Fprintln(m.log, "cozy machine: Runtime update", status.Operation, state+":", err)
	}
	ctx := context.Background()
	var previous, candidate []string
	var err error
	if status.State != "waiting_activation" {
		previous, err = m.previousPair(ctx, status.From)
		if err != nil {
			fail("failed", err)
			return
		}
		candidate, err = m.candidatePair(ctx, request, previous)
		if err != nil {
			fail("failed", err)
			return
		}
		for _, wheel := range candidate {
			if wheelDistribution(filepath.Base(wheel)) == hostruntime.Distribution {
				status.To.Runtime = wheelVersion(wheel)
			} else {
				status.To.TensorFS = wheelVersion(wheel)
			}
		}
	}
	payload, _ := json.Marshal(pairTransaction{Operation: request.Operation, Before: status.From, After: status.To, Candidate: candidate, Previous: previous, Store: m.layout.Store, Lifetime: m.grant.Lifetime, Pin: status.Pinned, Agent: request.Agent})
	legacyGuardian := false
	if status.State != "waiting_activation" {
		status.State = "preparing"
		if err := m.saveUpdate(status); err != nil {
			fail("failed", err)
			return
		}
		if _, err := m.launcher.maintain("update-stage", []string{string(payload)}); err != nil {
			if err.Error() != `unknown maintenance op "update-stage"` {
				fail("failed", err)
				return
			}
			// The stable guardian can outlive the application that requested this
			// update. Older guardians safely prepare only after guarded shutdown.
			legacyGuardian = true
			payload, _ = json.Marshal(pairTransaction{Before: status.From, After: status.To, Candidate: candidate, Previous: previous, Store: m.layout.Store, Lifetime: m.grant.Lifetime, Pin: status.Pinned, Agent: request.Agent})
		}
	}
	waiting := func() error {
		phase := "waiting_activation"
		if legacyGuardian {
			phase = "waiting"
			status.Error = "this machine bootstrap prepares updates only after work drains; active preparation and paused activation require replacing its service or image bootstrap"
		}
		if status.State == phase {
			return nil
		}
		status.State = phase
		return m.saveUpdate(status)
	}
	if err := m.quiesce(ctx, waiting); err != nil {
		fail("failed", err)
		return
	}
	status.State, status.Error = "installing", ""
	if err := m.saveUpdate(status); err != nil {
		fail("failed", err)
		if launchErr := m.launch(); launchErr != nil {
			m.restarts.fail(launchErr)
		}
		return
	}
	answer, err := m.launcher.maintain("update-prepare", []string{string(payload)})
	if err != nil || answer != "boot_pending" {
		if err == nil {
			err = errors.New("candidate installation rolled back")
		}
		fail("failed", err)
		if answer == "recovery_required" {
			m.restarts.fail(&launchRefusal{"runtime_update_recovery_required", err.Error()})
			return
		}
		if launchErr := m.launch(); launchErr != nil {
			m.restarts.fail(launchErr)
		}
		return
	}
	m.mu.Lock()
	m.grant.startupPending = true
	m.mu.Unlock()
	err = m.replaceApplication()
	if err == nil {
		err = m.launch()
	}
	if err == nil {
		m.mu.Lock()
		p, ready := m.proc, m.ready
		m.mu.Unlock()
		select {
		case <-ready:
		case <-p.done:
		}
		// Exit may race the reply to a durable readiness commit. Let that
		// incarnation finish its claim before deciding whether rollback is needed.
		<-p.claimDone
		select {
		case <-ready:
			status.State, status.Error = "succeeded", ""
			_ = m.restarts.clear()
			return
		default:
			if p.startupCommitted {
				fail("failed", fmt.Errorf("Runtime installation committed but authenticated readiness failed: %s", m.restarts.reason()))
				return
			}
			err = fmt.Errorf("candidate exited before authenticated readiness: %s", exitText(p.err))
		}
	}
	status.State, status.Error = "rolling_back", err.Error()
	m.saveUpdate(status)
	if rollback := m.rollbackStartup(ctx, err); rollback != nil {
		fail("failed", fmt.Errorf("%v; rollback failed: %w", err, rollback))
		return
	}
	fail("rolled_back", err)
}

// Both the original request plane and its replacement retain completion ownership
// until terminal status is durable. The caller holds staging through publication.
func (m *Machine) completeUpdateFiles(status updateStatus) {
	for delay := time.Second; ; delay = min(2*delay, 10*time.Second) {
		if err := m.writeUpdate(status); err != nil {
			fmt.Fprintln(m.log, "cozy machine: cannot persist completed update; retaining its candidate and retrying:", err)
			m.updates.mu.Lock()
			observed := *m.updates.status
			observed.Error = "cannot persist completed update; retaining its candidate and retrying: " + err.Error()
			m.updates.status = &observed
			m.updates.mu.Unlock()
			time.Sleep(delay)
			continue
		}
		break
	}
	_, _ = m.launcher.maintain("update-cleanup", []string{status.Operation})
	if err := os.RemoveAll(m.stagePath()); err != nil {
		fmt.Fprintln(m.log, "cozy machine: cannot remove completed update files:", err)
	}
}

// quiesce holds the machine for the update once its Runtime accepts a guarded restart. A busy
// Runtime is asked again; running work is never interrupted.
func (m *Machine) quiesce(ctx context.Context, onWaiting ...func() error) error {
	waiting := func() error {
		if len(onWaiting) > 0 {
			return onWaiting[0]()
		}
		return nil
	}
	wait := time.Second
	recovered := false
	for {
		m.launchMu.Lock()
		m.setUpdating(true)
		m.mu.Lock()
		process, ready, preparing := m.proc, m.ready, m.active > 0
		m.mu.Unlock()
		if preparing {
			m.launchMu.Unlock()
			if err := waiting(); err != nil {
				return err
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(wait):
			}
			continue
		}
		if process == nil || process.exited() {
			// A stopped PID does not prove its accepted work or descendants have
			// settled. Let the installed Runtime recover its journal and establish
			// a fresh guarded verdict before changing software under retained work.
			raw, err := m.launcher.maintain("software-state", nil)
			var state softwareState
			if err == nil {
				err = json.Unmarshal([]byte(raw), &state)
			}
			if err != nil {
				m.launchMu.Unlock()
				return fmt.Errorf("cannot establish retained Runtime work before activation: %w", err)
			}
			if !state.Quiescent {
				if recovered {
					m.launchMu.Unlock()
					return errors.New("the installed Runtime exited while recovering retained work; repair its startup before activating the candidate")
				}
				if err := m.launch(); err != nil {
					m.launchMu.Unlock()
					return fmt.Errorf("cannot recover the installed Runtime's retained work before activation: %w", err)
				}
				recovered = true
				m.launchMu.Unlock()
				if err := waiting(); err != nil {
					return err
				}
				continue
			}
		}
		if process != nil && !process.exited() {
			select {
			case <-ready:
			default:
				m.setUpdating(false)
				m.launchMu.Unlock()
				if err := waiting(); err != nil {
					return err
				}
				select {
				case <-ready:
				case <-process.done:
				case <-ctx.Done():
					return ctx.Err()
				}
				continue
			}
		}
		answer, err := m.launcher.maintain("restart", nil)
		if err == nil && answer == "accepted" {
			m.stopRuntime() // the Runtime exited; the guardian's report settles it here
			m.launchMu.Unlock()
			return nil
		}
		m.setUpdating(false)
		m.launchMu.Unlock()
		if err != nil {
			return err
		}
		if err := waiting(); err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}
		wait = min(2*wait, 10*time.Second)
	}
}

// previousPair is the installed pair's wheels, to roll back to: the kept pair, or the same
// versions fetched from the package index.
func (m *Machine) previousPair(ctx context.Context, installed pairVersions) ([]string, error) {
	answer, err := m.launcher.maintain("pair", nil)
	if err != nil {
		return nil, err
	}
	var kept []string
	if answer != "" {
		kept = strings.Split(answer, "\n")
	}
	if len(kept) == 2 && samePair(kept, installed) {
		return kept, nil
	}
	var pair []string
	for name, version := range map[string]string{hostruntime.Distribution: installed.Runtime, "tensorfs": installed.TensorFS} {
		wheel, err := m.fetchWheel(ctx, name, version)
		if err != nil {
			return nil, fmt.Errorf("the installed %s %s cannot be kept for rollback: %w", name, version, err)
		}
		pair = append(pair, wheel)
	}
	return pair, nil
}

func samePair(wheels []string, want pairVersions) bool {
	got := pairVersions{}
	for _, wheel := range wheels {
		if wheelDistribution(filepath.Base(wheel)) == hostruntime.Distribution {
			got.Runtime = wheelVersion(wheel)
		} else {
			got.TensorFS = wheelVersion(wheel)
		}
	}
	return got == want
}

// candidatePair is the requested pair's wheels; an omitted member stays as installed.
func (m *Machine) candidatePair(ctx context.Context, request updateRequest, previous []string) ([]string, error) {
	var pair []string
	for name, choice := range map[string]*wheelChoice{hostruntime.Distribution: request.Runtime, "tensorfs": request.TensorFS} {
		switch {
		case choice == nil:
			for _, wheel := range previous {
				if wheelDistribution(filepath.Base(wheel)) == name {
					pair = append(pair, wheel)
				}
			}
		case choice.File != "":
			path := m.stagePath(choice.SHA256, choice.File)
			if wheelDistribution(choice.File) != name || !sha256Hex.MatchString(choice.SHA256) {
				return nil, fmt.Errorf("%s names no staged %s wheel", choice.File, name)
			}
			if _, err := os.Stat(path); err != nil {
				return nil, fmt.Errorf("%s was not staged: %w", choice.File, err)
			}
			pair = append(pair, path)
		case choice.Version != "":
			wheel, err := m.fetchWheel(ctx, name, choice.Version)
			if err != nil {
				return nil, err
			}
			pair = append(pair, wheel)
		default:
			return nil, fmt.Errorf("the update names no %s wheel or version", name)
		}
	}
	if len(pair) != 2 {
		return nil, errors.New("the update does not name one Runtime and one TensorFS")
	}
	return pair, nil
}

// fetchWheel stages a published wheel for this machine: the release's native wheel for its
// Python and architecture, verified against the index's digest.
func (m *Machine) fetchWheel(ctx context.Context, name, version string) (string, error) {
	var release struct {
		URLs []struct {
			Filename    string            `json:"filename"`
			URL         string            `json:"url"`
			PackageType string            `json:"packagetype"`
			Yanked      bool              `json:"yanked"`
			Digests     map[string]string `json:"digests"`
		} `json:"urls"`
	}
	if err := getJSON(ctx, "https://pypi.org/pypi/"+name+"/"+version+"/json", &release); err != nil {
		return "", fmt.Errorf("read %s %s from the package index: %w", name, version, err)
	}
	arch := map[string]string{"amd64": "x86_64", "arm64": "aarch64"}[runtime.GOARCH]
	for _, file := range release.URLs {
		tags := strings.Split(strings.TrimSuffix(file.Filename, ".whl"), "-")
		if file.PackageType != "bdist_wheel" || file.Yanked || len(tags) < 5 || !strings.Contains(tags[len(tags)-3], "cp312") && tags[len(tags)-3] != "py3" ||
			!strings.Contains(tags[len(tags)-1], "manylinux") || !strings.HasSuffix(tags[len(tags)-1], arch) {
			continue
		}
		if path := m.stagePath(file.Digests["sha256"], file.Filename); fileExists(path) {
			return path, nil
		}
		response, err := httpGet(ctx, file.URL)
		if err != nil {
			return "", err
		}
		digest, _, err := m.stageWheel(file.Filename, response.Body)
		response.Body.Close()
		if err != nil {
			return "", err
		}
		if digest != file.Digests["sha256"] {
			return "", fmt.Errorf("%s does not match the index's digest", file.Filename)
		}
		return m.stagePath(digest, file.Filename), nil
	}
	return "", fmt.Errorf("%s %s has no native wheel for this machine (cp312, %s)", name, version, arch)
}

func fileExists(path string) bool { _, err := os.Stat(path); return err == nil }

func httpGet(ctx context.Context, url string) (*http.Response, error) {
	ctx, cancel := context.WithCancel(ctx)
	progress := &progressBody{cancel: cancel, done: make(chan struct{})}
	go killWhenStill(progress.done, progress.bytes.Load, cancel)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		progress.Close()
		return nil, err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		progress.Close()
		return nil, err
	}
	progress.ReadCloser = response.Body
	response.Body = progress
	if response.StatusCode != http.StatusOK {
		response.Body.Close()
		return nil, fmt.Errorf("%s answered HTTP %d", url, response.StatusCode)
	}
	return response, nil
}

func getJSON(ctx context.Context, url string, into any) error {
	response, err := httpGet(ctx, url)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	return json.NewDecoder(io.LimitReader(response.Body, 32<<20)).Decode(into)
}

func (m *Machine) manualUpdateActive() bool {
	m.updates.mu.Lock()
	defer m.updates.mu.Unlock()
	return m.updates.active != ""
}

// Transfers reset the stall observation on every received byte; there is no
// maximum transfer duration while useful progress continues.
type progressBody struct {
	io.ReadCloser
	bytes  atomic.Uint64
	cancel context.CancelFunc
	done   chan struct{}
	once   sync.Once
}

func (p *progressBody) Read(b []byte) (int, error) {
	n, err := p.ReadCloser.Read(b)
	p.bytes.Add(uint64(n))
	return n, err
}
func (p *progressBody) Close() error {
	p.once.Do(func() { close(p.done); p.cancel() })
	if p.ReadCloser != nil {
		return p.ReadCloser.Close()
	}
	return nil
}

// This response is emitted only before new update admission. Clients may observe
// the read-only Runtime state and retry the same operation when phase is ready.
func runtimeStarting(w http.ResponseWriter, phase string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusServiceUnavailable)
	_ = json.NewEncoder(w).Encode(struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Phase   string `json:"phase"`
	}{"runtime_starting", "Runtime has not finished guarded readiness; observe GET /v1/machine/runtime", phase})
}
