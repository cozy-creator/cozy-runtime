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
	"time"

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
//	POST /v1/machine/runtime/update          {operation, runtime, tensorfs}: each a staged
//	                                         {file, sha256}, a {version} to fetch, or omitted
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
	Runtime   *wheelChoice `json:"runtime,omitempty"`
	TensorFS  *wheelChoice `json:"tensorfs,omitempty"`
}

type pairVersions struct {
	Runtime  string `json:"runtime"`
	TensorFS string `json:"tensorfs"`
}

type updateStatus struct {
	Operation        string       `json:"operation"`
	State            string       `json:"state"` // waiting | installing | starting | rolling_back | succeeded | rolled_back | failed
	Error            string       `json:"error,omitempty"`
	From             pairVersions `json:"from"`
	To               pairVersions `json:"to"`
	Pinned           bool         `json:"pinned,omitempty"`            // an explicitly supplied wheel owns this installed pair
	PreviouslyPinned bool         `json:"previously_pinned,omitempty"` // failed maintenance retains the previous selection
}

func (s updateStatus) terminal() bool {
	return s.State == "succeeded" || s.State == "rolled_back" || s.State == "failed"
}

func (s updateStatus) pins(pair pairVersions) bool {
	if s.State == "succeeded" {
		return s.Pinned && s.To == pair
	}
	return s.PreviouslyPinned && s.From == pair
}

type runtimeUpdates struct {
	mu     sync.Mutex
	status *updateStatus
}

var (
	operationName = regexp.MustCompile(`^[A-Za-z0-9._-]{1,128}$`)
	sha256Hex     = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

func (m *Machine) updatePath() string { return filepath.Join(m.layout.State, "runtime-update.json") }
func (m *Machine) stagePath(names ...string) string {
	return filepath.Join(append([]string{m.layout.State, "runtime-update"}, names...)...)
}

// openUpdates reads the last update. One this daemon was part-way through when it stopped
// failed: the operator reads the installed pair and runs it again.
func (m *Machine) openUpdates() {
	var status updateStatus
	if raw, err := os.ReadFile(m.updatePath()); err == nil && json.Unmarshal(raw, &status) == nil {
		if !status.terminal() {
			status.State, status.Error = "failed", "the machine restarted during its Runtime update; run it again"
			m.saveUpdate(status)
		}
		m.updates.status = &status
	}
}

func (m *Machine) saveUpdate(status updateStatus) {
	m.updates.mu.Lock()
	m.updates.status = &status
	m.updates.mu.Unlock()
	raw, _ := json.Marshal(status)
	if err := writeAtomic(m.updatePath(), raw, 0o600); err != nil {
		fmt.Fprintln(m.log, "cozy machine: the Runtime update state was not recorded:", err)
	}
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
		http.Error(w, "machine_starting: the process launcher is not ready", http.StatusServiceUnavailable)
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
	answer := map[string]any{"capabilities": []string{UpdateCapability}, "runtime": installed.Runtime,
		"tensorfs": installed.TensorFS, "update": status}
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

func (m *Machine) serveUpdate(w http.ResponseWriter, r *http.Request) {
	if !m.maintenance(w, r) {
		return
	}
	if m.startupPending() {
		http.Error(w, "startup update is awaiting Runtime readiness", http.StatusServiceUnavailable)
		return
	}
	var request updateRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&request); err != nil || !operationName.MatchString(request.Operation) ||
		request.Runtime == nil && request.TensorFS == nil {
		http.Error(w, "an update names its operation and a Runtime or TensorFS", http.StatusBadRequest)
		return
	}
	m.updates.mu.Lock()
	current := m.updates.status
	busy := current != nil && !current.terminal()
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
	installed, err := m.installedPair()
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	pinned := request.Runtime != nil && request.Runtime.File != "" || request.TensorFS != nil && request.TensorFS.File != ""
	status := updateStatus{Operation: request.Operation, State: "waiting", From: installed, Pinned: pinned}
	status.PreviouslyPinned = current != nil && current.pins(installed)
	m.saveUpdate(status)
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
	fail := func(state string, err error) {
		status.State, status.Error = state, err.Error()
		fmt.Fprintln(m.log, "cozy machine: Runtime update", status.Operation, state+":", err)
		m.saveUpdate(status)
	}
	ctx := context.Background()
	defer os.RemoveAll(m.stagePath()) // staged wheels serve one update; the kept pair is elsewhere
	previous, err := m.previousPair(ctx, status.From)
	if err != nil {
		fail("failed", err)
		return
	}
	candidate, err := m.candidatePair(ctx, request, previous)
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
	m.saveUpdate(status)
	if err := m.quiesce(ctx); err != nil {
		fail("failed", err)
		return
	}
	defer m.setUpdating(false)
	status.State = "installing"
	m.saveUpdate(status)
	err = m.replacePair(ctx, candidate, status.To)
	if err == nil {
		if _, err = m.launcher.maintain("select", candidate); err == nil {
			status.State, status.Error = "succeeded", ""
			_ = m.restarts.clear()
			m.saveUpdate(status)
			return
		}
	}
	status.State, status.Error = "rolling_back", err.Error()
	m.saveUpdate(status)
	if rollback := m.replacePair(ctx, previous, status.From); rollback != nil {
		fail("failed", fmt.Errorf("%v; rolling back failed too: %v", err, rollback))
		return
	}
	fail("rolled_back", err)
}

// quiesce holds the machine for the update once its Runtime accepts a guarded restart. A busy
// Runtime is asked again; running work is never interrupted.
func (m *Machine) quiesce(ctx context.Context) error {
	wait := time.Second
	for {
		m.setUpdating(true)
		answer, err := m.launcher.maintain("restart", nil)
		if err == nil && answer == "accepted" {
			m.stopRuntime() // the Runtime exited; the guardian's report settles it here
			return nil
		}
		m.setUpdating(false)
		if err != nil {
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

// replacePair installs a pair, relaunches the Runtime and checks it answers at versions.
func (m *Machine) replacePair(ctx context.Context, wheels []string, want pairVersions) error {
	m.stopRuntime()
	if _, err := m.launcher.maintain("install", wheels); err != nil {
		return err
	}
	if err := m.launch(); err != nil {
		return err
	}
	m.mu.Lock()
	p, ready := m.proc, m.ready
	m.mu.Unlock()
	select {
	case <-ready:
	case <-p.done:
		return fmt.Errorf("the new Runtime exited before it was ready: %s", exitText(p.err))
	}
	if err := m.requireRuntimeRange(ctx); err != nil {
		return err
	}
	got, err := m.installedPair()
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("the machine runs Runtime %s / TensorFS %s, not %s / %s", got.Runtime, got.TensorFS, want.Runtime, want.TensorFS)
	}
	return nil
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
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, err
	}
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
