package host

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// This opt-in test uses real Runtime/TensorFS wheels and a disposable copy of an
// installed Python environment. It exercises candidate staging and late imports,
// without starting a GPU worker or modifying the owner's installed environment.
func TestRealCandidatePreparationPreservesRunningPythonLateImports(t *testing.T) {
	fixture := os.Getenv("COZY_UPDATE_TEST_PYTHON_ENV")
	beforeWheel := os.Getenv("COZY_UPDATE_TEST_RUNTIME_BEFORE")
	afterWheel := os.Getenv("COZY_UPDATE_TEST_RUNTIME_AFTER")
	tensorfsWheel := os.Getenv("COZY_UPDATE_TEST_TENSORFS")
	if fixture == "" || beforeWheel == "" || afterWheel == "" || tensorfsWheel == "" {
		t.Skip("set COZY_UPDATE_TEST_PYTHON_ENV and the three COZY_UPDATE_TEST wheel paths")
	}
	root := t.TempDir()
	for _, dir := range []string{"opt/cozy", "usr/local/bin"} {
		if err := os.MkdirAll(filepath.Join(root, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(fixture, filepath.Join(root, "opt/cozy/python")); err != nil {
		t.Fatal(err)
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(uv, filepath.Join(root, "usr/local/bin/uv")); err != nil {
		t.Fatal(err)
	}
	grant := &Grant{Root: root, Lifetime: "persistent", agentSelection: "bundled"}
	m := &Machine{grant: grant, layout: NewLayout(grant), log: io.Discard}
	before := pairVersions{Runtime: wheelVersion(beforeWheel), TensorFS: wheelVersion(tensorfsWheel)}
	after := pairVersions{Runtime: wheelVersion(afterWheel), TensorFS: before.TensorFS}
	if before.Runtime == after.Runtime {
		t.Fatal("the real-wheel fixture must change Runtime versions")
	}
	installed, err := m.installedPair()
	if err != nil || installed != before {
		t.Fatalf("disposable environment must contain the previous pair: %+v %v", installed, err)
	}
	const lateImport = `
import hashlib, importlib.metadata as metadata, json, pathlib, sys
assert 'cozy_runtime.internal.worker.session' not in sys.modules
site = pathlib.Path(sys.prefix) / ('lib/python%d.%d/site-packages' % sys.version_info[:2])
source = site / 'cozy_runtime/internal/worker/session.py'
before = hashlib.sha256(source.read_bytes()).hexdigest()
version = metadata.version('cozy-runtime')
print('ready', flush=True)
assert sys.stdin.readline() == 'continue\n'
from cozy_runtime.internal.worker import session
print(json.dumps({'before': before, 'after': hashlib.sha256(pathlib.Path(session.__file__).read_bytes()).hexdigest(), 'initial_version': version, 'late_version': metadata.version('cozy-runtime'), 'module': session.__file__}), flush=True)
`
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	child := exec.CommandContext(ctx, filepath.Join(fixture, "bin/python"), "-I", "-c", lateImport)
	stdin, err := child.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := child.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	child.Stderr = &stderr
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	waited := false
	defer func() {
		cancel()
		if !waited {
			_ = child.Wait()
		}
	}()
	reader := bufio.NewReader(stdout)
	ready, err := reader.ReadString('\n')
	if err != nil || ready != "ready\n" {
		t.Fatalf("old interpreter did not reach the late-import barrier: %q %v %s", ready, err, stderr.String())
	}
	prepared, err := m.stagePair(t.Context(), before, after,
		[]string{afterWheel, tensorfsWheel}, []string{beforeWheel, tensorfsWheel})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(m.startupPath("prepared.json")); err != nil {
		t.Fatalf("candidate preparation was not durable: %v", err)
	}
	if _, err := os.Stat(m.startupPath("status.json")); !os.IsNotExist(err) {
		t.Fatalf("preparation entered the installation/rollback transaction: %v", err)
	}
	probed, err := m.startupVerify(t.Context(), prepared.Overlay)
	if err != nil || probed != after {
		t.Fatalf("separate candidate did not load the new pair: %+v %v", probed, err)
	}
	if _, err := io.WriteString(stdin, "continue\n"); err != nil {
		t.Fatal(err)
	}
	_ = stdin.Close()
	answer, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("old interpreter could not perform its late import: %v %s", err, stderr.String())
	}
	var got struct {
		Before, After  string
		InitialVersion string `json:"initial_version"`
		LateVersion    string `json:"late_version"`
		Module         string
	}
	if err := json.Unmarshal([]byte(answer), &got); err != nil {
		t.Fatal(err)
	}
	if err := child.Wait(); err != nil {
		t.Fatalf("old process failed after preparation: %v %s", err, stderr.String())
	}
	waited = true
	if got.Before != got.After || got.InitialVersion != before.Runtime || got.LateVersion != before.Runtime ||
		!strings.HasPrefix(got.Module, fixture+string(os.PathSeparator)) {
		t.Fatalf("candidate changed the running process's late import: %+v", got)
	}
	installed, err = m.installedPair()
	if err != nil || installed != before {
		t.Fatalf("preparation changed the selected installation: %+v %v", installed, err)
	}
	t.Logf("old Runtime %s kept its late import; separately prepared %s", before.Runtime, after.Runtime)
}
