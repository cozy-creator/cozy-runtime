package host

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
)

// The fixture never installs software or launches a Runtime. HTTP, signatures,
// version-probe subprocesses, staging and durable status files are real; the
// maintenance boundary deliberately fails installation to exercise rollback.
type updateLauncher struct {
	pair            []string
	pairReady       <-chan struct{}
	rollback        chan struct{}
	rollbackRelease chan struct{}
	rollbackDone    chan struct{}
	installs        atomic.Int32
}

func (*updateLauncher) launch(string) (*runtimeProcess, error) {
	return nil, errors.New("fixture must not launch a Runtime")
}
func (*updateLauncher) close() {}
func (l *updateLauncher) maintain(op string, _ []string) (string, error) {
	switch op {
	case "software-state":
		return `{"quiescent":true}`, nil
	case "pair":
		if l.pairReady != nil {
			<-l.pairReady
		}
		if len(l.pair) == 0 {
			return "", errors.New("fixture previous pair is unavailable")
		}
		return strings.Join(l.pair, "\n"), nil
	case "restart":
		return "accepted", nil
	case "update-stage":
		return "prepared", nil
	case "update-prepare":
		if l.installs.Add(1) == 1 && l.rollback != nil {
			close(l.rollback)
			<-l.rollbackRelease
			close(l.rollbackDone)
		}
		return "", errors.New("fixture wheel installation failed")
	}
	return "", fmt.Errorf("unexpected fixture maintenance: %s", op)
}

func updateMachine(t *testing.T) (*Machine, string, *http.ServeMux) {
	t.Helper()
	public, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	claims, err := newClaims("test-machine", "boot", make([]byte, 32), []ed25519.PublicKey{public})
	if err != nil {
		t.Fatal(err)
	}
	layout := NewLayout(&Grant{Root: t.TempDir()})
	if err := os.MkdirAll(layout.State, 0700); err != nil {
		t.Fatal(err)
	}
	m := &Machine{grant: &Grant{WorkerID: "test-machine"}, layout: layout, claims: claims,
		launcher: &updateLauncher{}, log: io.Discard, restarts: &restarts{}}
	executable(t, filepath.Join(layout.Root, "opt/cozy/python/bin/python"), "printf '0.18.86\\n0.3.78\\n'")
	token, err := capability.Mint(key, capability.Grant{Machine: "test-machine", Action: capability.Maintenance, Expires: time.Now().Add(time.Hour).Unix()})
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/machine/runtime/update", m.serveUpdate)
	mux.HandleFunc("PUT /v1/machine/runtime/wheels/{file}", m.serveStageWheel)
	return m, token, mux
}

type updateReply struct {
	code   int
	body   string
	status updateStatus
	err    error
}

func updateCall(client *http.Client, method, address, token string, body io.Reader) updateReply {
	r, err := http.NewRequest(method, address, body)
	if err != nil {
		return updateReply{err: err}
	}
	r.Header.Set("Authorization", "Cozy-Cap "+token)
	response, err := client.Do(r)
	if err != nil {
		return updateReply{err: err}
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	reply := updateReply{code: response.StatusCode, body: string(raw), err: err}
	_ = json.Unmarshal(raw, &reply.status)
	return reply
}

func postUpdate(server *httptest.Server, token, operation string, choice *wheelChoice) updateReply {
	if choice == nil {
		choice = &wheelChoice{Version: "0.18.87"}
	}
	pin := true
	raw, _ := json.Marshal(updateRequest{Operation: operation, Runtime: choice, Pin: &pin, Agent: "bundled"})
	return updateCall(server.Client(), http.MethodPost, server.URL+"/v1/machine/runtime/update", token, bytes.NewReader(raw))
}

func awaitUpdate(t *testing.T, server *httptest.Server, token, operation string) updateStatus {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		reply := postUpdate(server, token, operation, nil)
		if reply.err != nil || reply.code != http.StatusOK {
			t.Fatalf("read operation %s: %+v", operation, reply)
		}
		if reply.status.terminal() {
			return reply.status
		}
		if time.Now().After(deadline) {
			t.Fatalf("operation %s did not settle: %+v", operation, reply.status)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestRuntimeUpdateConcurrentAdmissionAndStaging(t *testing.T) {
	m, token, mux := updateMachine(t)
	pairReady := make(chan struct{})
	m.launcher = &updateLauncher{pairReady: pairReady}
	var release sync.Once
	defer release.Do(func() { close(pairReady) })
	// All requests enter admission together, while a real version-probe process
	// makes the pre-admission window observable in the unprotected implementation.
	executable(t, filepath.Join(m.layout.Root, "opt/cozy/python/bin/python"), "sleep 0.05\nprintf '0.18.86\\n0.3.78\\n'")
	const callers = 8
	arrived, start := make(chan struct{}, callers), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("concurrent") != "" {
			arrived <- struct{}{}
			<-start
		}
		mux.ServeHTTP(w, r)
	}))
	defer server.Close()
	replies := make(chan updateReply, callers)
	for n := range callers {
		go func() {
			raw := fmt.Sprintf(`{"operation":"update-%d","pin":false,"agent":"bundled","runtime":{"version":"0.18.87"}}`, n)
			replies <- updateCall(server.Client(), http.MethodPost, server.URL+"/v1/machine/runtime/update?concurrent=1", token, strings.NewReader(raw))
		}()
	}
	for range callers {
		<-arrived
	}
	close(start)
	accepted, operation := 0, ""
	for range callers {
		reply := <-replies
		if reply.err != nil {
			t.Fatal(reply.err)
		}
		switch reply.code {
		case http.StatusAccepted:
			accepted++
			operation = reply.status.Operation
		case http.StatusConflict:
		default:
			t.Fatalf("unexpected admission: %+v", reply)
		}
	}
	if accepted != 1 {
		t.Fatalf("concurrent requests admitted %d operations, want one", accepted)
	}
	if reply := postUpdate(server, token, operation, nil); reply.code != http.StatusOK || reply.status.State != "waiting" {
		t.Fatalf("same operation did not retain its identity: %+v", reply)
	}
	reply := updateCall(server.Client(), http.MethodPut, server.URL+"/v1/machine/runtime/wheels/cozy_runtime-0.18.87-py3-none-any.whl", token, strings.NewReader("next upload"))
	if reply.code != http.StatusConflict {
		t.Fatalf("active update acknowledged an upload it can delete: %+v", reply)
	}
	release.Do(func() { close(pairReady) })
	awaitUpdate(t, server, token, operation)
}

func TestRuntimeUpdatePublishesTerminalOnlyAfterCleanup(t *testing.T) {
	m, token, mux := updateMachine(t)
	l := &updateLauncher{rollback: make(chan struct{}), rollbackRelease: make(chan struct{}), rollbackDone: make(chan struct{})}
	for _, name := range []string{"cozy_runtime-0.18.86-py3-none-any.whl", "tensorfs-0.3.78-py3-none-any.whl"} {
		path := filepath.Join(m.layout.Root, name)
		if err := os.WriteFile(path, []byte(name), 0600); err != nil {
			t.Fatal(err)
		}
		l.pair = append(l.pair, path)
	}
	m.launcher = l
	server := httptest.NewServer(mux)
	defer server.Close()
	file := "cozy_runtime-0.18.87-py3-none-any.whl"
	sha, _, err := m.stageWheel(file, strings.NewReader("candidate"))
	if err != nil {
		t.Fatal(err)
	}
	if reply := postUpdate(server, token, "first", &wheelChoice{File: file, SHA256: sha}); reply.code != http.StatusAccepted {
		t.Fatalf("first update: %+v", reply)
	}
	<-l.rollback
	// Resetting inUpdate must complete before publishing a terminal response.
	// Hold that exact state transition, without touching any live process.
	m.mu.Lock()
	locked := true
	defer func() {
		if locked {
			m.mu.Unlock()
		}
	}()
	close(l.rollbackRelease)
	<-l.rollbackDone
	replied := make(chan updateReply, 1)
	go func() { replied <- postUpdate(server, token, "first", nil) }()
	select {
	case reply := <-replied:
		if reply.status.terminal() {
			t.Fatalf("terminal status preceded cleanup/reset: %+v", reply)
		}
	case <-time.After(100 * time.Millisecond):
	}
	m.mu.Unlock()
	locked = false
	status := awaitUpdate(t, server, token, "first")
	if status.State != "failed" || !strings.Contains(status.Error, "fixture wheel installation failed") {
		t.Fatalf("rollback result: %+v", status)
	}
	if m.updating() {
		t.Fatal("terminal operation still holds the Runtime")
	}
	if _, err := os.Stat(m.stagePath()); !os.IsNotExist(err) {
		t.Fatalf("terminal operation left staging behind: %v", err)
	}
	upload := "next operation's wheel"
	reply := updateCall(server.Client(), http.MethodPut, server.URL+"/v1/machine/runtime/wheels/"+file, token, strings.NewReader(upload))
	if reply.code != http.StatusOK {
		t.Fatalf("next upload: %+v", reply)
	}
	digest := sha256.Sum256([]byte(upload))
	path := m.stagePath(hex.EncodeToString(digest[:]), file)
	if raw, err := os.ReadFile(path); err != nil || string(raw) != upload {
		t.Fatalf("previous cleanup consumed the next upload: %q %v", raw, err)
	}
}

func TestRuntimeUpdateFailedAdmissionPreservesPreviousStatus(t *testing.T) {
	for _, failure := range []string{"version probe", "status write"} {
		t.Run(failure, func(t *testing.T) {
			m, token, mux := updateMachine(t)
			previous := updateStatus{Operation: "previous", State: "succeeded", From: pairVersions{Runtime: "0.18.85", TensorFS: "0.3.78"}}
			m.updates.status = &previous
			python := filepath.Join(m.layout.Root, "opt/cozy/python/bin/python")
			if failure == "version probe" {
				executable(t, python, "exit 1")
			} else if err := os.Mkdir(m.updatePath(), 0700); err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(mux)
			defer server.Close()
			if reply := postUpdate(server, token, "next", nil); reply.code != http.StatusServiceUnavailable {
				t.Fatalf("failed admission: %+v", reply)
			}
			if reply := postUpdate(server, token, "previous", nil); reply.code != http.StatusOK || reply.status != previous {
				t.Fatalf("failed admission replaced previous status: %+v", reply)
			}
			if failure == "version probe" {
				executable(t, python, "printf '0.18.86\\n0.3.78\\n'")
			} else if err := os.Remove(m.updatePath()); err != nil {
				t.Fatal(err)
			}
			if reply := postUpdate(server, token, "next", nil); reply.code != http.StatusAccepted {
				t.Fatalf("failed admission retained a reservation: %+v", reply)
			}
			awaitUpdate(t, server, token, "next")
		})
	}
}

func TestRuntimeUpdateWaitsForInProgressUpload(t *testing.T) {
	m, token, mux := updateMachine(t)
	server := httptest.NewServer(mux)
	defer server.Close()
	file := "cozy_runtime-0.18.87-py3-none-any.whl"
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	uploaded := make(chan updateReply, 1)
	go func() {
		uploaded <- updateCall(server.Client(), http.MethodPut, server.URL+"/v1/machine/runtime/wheels/"+file, token, reader)
	}()
	if _, err := writer.Write([]byte("first")); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		paths, _ := filepath.Glob(m.stagePath(".upload-*"))
		if len(paths) > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("upload did not enter staging")
		}
		time.Sleep(time.Millisecond)
	}
	admitted := make(chan updateReply, 1)
	go func() { admitted <- postUpdate(server, token, "after-upload", nil) }()
	select {
	case reply := <-admitted:
		t.Fatalf("update passed an incomplete upload: %+v", reply)
	case <-time.After(100 * time.Millisecond):
	}
	if _, err := writer.Write([]byte("last")); err != nil {
		t.Fatal(err)
	}
	writer.Close()
	if reply := <-uploaded; reply.code != http.StatusOK {
		t.Fatalf("upload failed: %+v", reply)
	}
	if reply := <-admitted; reply.code != http.StatusAccepted {
		t.Fatalf("update did not follow the upload: %+v", reply)
	}
	awaitUpdate(t, server, token, "after-upload")
}

func TestCopyWheelReadFailurePreservesDestination(t *testing.T) {
	directory := t.TempDir()
	target := filepath.Join(directory, "kept.whl")
	if err := os.WriteFile(target, []byte("known good wheel"), 0600); err != nil {
		t.Fatal(err)
	}
	// Opening a directory succeeds, but reading it as a wheel fails on Linux.
	if err := copyWheel(t.TempDir(), target); err == nil {
		t.Fatal("wheel read failure was discarded")
	}
	if raw, err := os.ReadFile(target); err != nil || string(raw) != "known good wheel" {
		t.Fatalf("failed copy replaced retained wheel: %q %v", raw, err)
	}
	if temporary, _ := filepath.Glob(filepath.Join(directory, ".wheel-*")); len(temporary) != 0 {
		t.Fatalf("failed copy retained temporary files: %v", temporary)
	}
}

func TestNativeUpdateDefersUnreadyRuntimeBeforeAdmission(t *testing.T) {
	m, token, mux := updateMachine(t)
	m.proc = &runtimeProcess{done: make(chan struct{})}
	m.ready = make(chan struct{})
	server := httptest.NewServer(mux)
	defer server.Close()
	reply := postUpdate(server, token, "boot-race", nil)
	if reply.code != http.StatusServiceUnavailable {
		t.Fatalf("unready Runtime admitted update: %+v", reply)
	}
	if m.updates.status != nil || m.updates.active != "" {
		t.Fatal("temporary startup wrote update admission")
	}
	if _, err := os.Stat(m.updatePath()); !os.IsNotExist(err) {
		t.Fatal("temporary startup persisted an update")
	}
	// A lost response for an already admitted operation is still an idempotent read.
	m.updates.status = &updateStatus{Operation: "already-admitted", State: "waiting"}
	reply = postUpdate(server, token, "already-admitted", nil)
	if reply.code != http.StatusOK || reply.status.State != "waiting" {
		t.Fatalf("readiness hid accepted operation: %+v", reply)
	}
}

func TestNativeUpdateDefaultsLegacyPinIntent(t *testing.T) {
	for _, decision := range []string{"omitted", "null", "false", "true"} {
		t.Run(decision, func(t *testing.T) {
			m, token, mux := updateMachine(t)
			gate := make(chan struct{})
			m.launcher = &updateLauncher{pairReady: gate}
			server := httptest.NewServer(mux)
			defer server.Close()
			body := `{"operation":"pin-proof","runtime":{"version":"0.18.87"}`
			if decision != "omitted" {
				body += `,"pin":` + decision
			}
			body += "}"
			reply := updateCall(server.Client(), http.MethodPost, server.URL+"/v1/machine/runtime/update", token, strings.NewReader(body))
			if reply.code != http.StatusAccepted || reply.status.Pinned != (decision == "true") {
				close(gate)
				t.Fatalf("pin intent changed: %+v", reply)
			}
			if reply.status.Request == nil || reply.status.Request.Agent != "explicit" {
				close(gate)
				t.Fatalf("legacy request changed the running agent selection: %+v", reply.status.Request)
			}
			close(gate)
			awaitUpdate(t, server, token, "pin-proof")
		})
	}
}
