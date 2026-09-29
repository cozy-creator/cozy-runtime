package host

import (
	"context"
	"crypto/tls"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// A replacement request plane resumes the private transaction without the old
// process's runUpdate goroutine or in-memory operation owner.
func resumedCompletionMachine(t *testing.T, recorded bool) (*Machine, *completionLauncher, string, *http.ServeMux) {
	t.Helper()
	m, token, mux := updateMachine(t)
	m.conn = completionConnection(t, &updateCompletionPeer{})
	m.owned, m.inUpdate, m.grant.startupPending = true, true, true
	m.grant.bootstrap = &bootstrapChild{Pending: true, Resume: true}
	m.id = &Identity{BootID: "boot", Leaf: tls.Certificate{Certificate: [][]byte{[]byte("leaf")}}}
	m.idle = &idle{persistent: true}
	if recorded {
		m.updates.status = &updateStatus{Operation: "resumed", State: "installing"}
	}
	launcher := &completionLauncher{machine: m, committed: make(chan struct{}), release: make(chan struct{})}
	m.launcher = launcher
	return m, launcher, token, mux
}
func awaitCompletionSignal(t *testing.T, ch <-chan struct{}, reason string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(3 * time.Second):
		t.Fatal(reason)
	}
}
func waitPrivateCommitResult(t *testing.T, m *Machine) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for m.startupPending() {
		if time.Now().After(deadline) {
			t.Fatal("private commit did not reach receipt/finalization")
		}
		time.Sleep(time.Millisecond)
	}
}
func TestResumedExitWaitsForInFlightReadinessCommit(t *testing.T) {
	m, launcher, _, _ := resumedCompletionMachine(t, true)
	if err := m.launch(); err != nil {
		t.Fatal(err)
	}
	awaitCompletionSignal(t, launcher.committed, "candidate did not reach private commit")
	launcher.process.stop()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	supervised := make(chan error, 1)
	go func() { supervised <- m.supervise(ctx) }()
	var release sync.Once
	defer release.Do(func() { close(launcher.release) })
	// The committed reply is deliberately withheld after the candidate exits.
	time.Sleep(100 * time.Millisecond)
	premature := launcher.rollbacks.Load()
	release.Do(func() { close(launcher.release) })
	awaitCompletionSignal(t, launcher.process.claimDone, "resumed claim did not settle")
	cancel()
	select {
	case err := <-supervised:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("supervisor did not finish")
	}
	m.updates.mu.Lock()
	result := *m.updates.status
	m.updates.mu.Unlock()
	if premature != 0 || launcher.rollbacks.Load() != 0 || result.State != "succeeded" {
		t.Fatalf("resumed commit raced rollback: early=%d total=%d status=%+v", premature, launcher.rollbacks.Load(), result)
	}
}

type admissionDuringReceiptLauncher struct {
	*completionLauncher
	releasePair chan struct{}
	pairDone    chan struct{}
}

func (l *admissionDuringReceiptLauncher) maintain(op string, args []string) (string, error) {
	if op == "pair" {
		<-l.releasePair
		close(l.pairDone)
		return "", errors.New("unexpected admitted update stopped at fixture boundary")
	}
	return l.completionLauncher.maintain(op, args)
}
func TestResumedCommitKeepsAdmissionThroughReceipt(t *testing.T) {
	m, launcher, token, mux := resumedCompletionMachine(t, false)
	blocked := &admissionDuringReceiptLauncher{completionLauncher: launcher, releasePair: make(chan struct{}), pairDone: make(chan struct{})}
	m.launcher = blocked
	m.owned = false
	m.receipt = &receipt{standalone: true}
	m.receipt.mu.Lock()
	var receiptRelease sync.Once
	defer receiptRelease.Do(m.receipt.mu.Unlock)
	if err := m.launch(); err != nil {
		t.Fatal(err)
	}
	awaitCompletionSignal(t, launcher.committed, "candidate did not commit")
	close(launcher.release)
	waitPrivateCommitResult(t, m)
	launcher.process.stop()
	server := httptest.NewServer(mux)
	defer server.Close()
	held := m.updating()
	uploaded := updateCall(server.Client(), http.MethodPut, server.URL+"/v1/machine/runtime/wheels/cozy_runtime-0.18.87-py3-none-any.whl", token, strings.NewReader("next candidate"))
	reply := postUpdate(server, token, "next", nil)
	receiptRelease.Do(m.receipt.mu.Unlock)
	awaitCompletionSignal(t, launcher.process.claimDone, "receipt completion did not settle")
	m.updates.mu.Lock()
	current := m.updates.status
	m.updates.mu.Unlock()
	close(blocked.releasePair)
	if reply.code == http.StatusAccepted {
		awaitCompletionSignal(t, blocked.pairDone, "unexpected admission did not reach fixture boundary")
		awaitUpdate(t, server, token, "next")
	}
	if !held || uploaded.code != http.StatusConflict || reply.code != http.StatusConflict || current != nil {
		t.Fatalf("resumed finalization allowed new staging/admission: held=%v upload=%d update=%d status=%+v", held, uploaded.code, reply.code, current)
	}
}
func TestResumedCommitKeepsLaunchOwnershipThroughCleanup(t *testing.T) {
	m, launcher, _, _ := resumedCompletionMachine(t, true)
	if err := m.launch(); err != nil {
		t.Fatal(err)
	}
	awaitCompletionSignal(t, launcher.committed, "candidate did not commit")
	m.updates.staging.Lock()
	close(launcher.release)
	waitPrivateCommitResult(t, m)
	held := m.updating()
	select {
	case <-m.ready:
		t.Error("API readiness opened before resumed cleanup")
	default:
	}
	m.updates.staging.Unlock()
	awaitCompletionSignal(t, launcher.process.claimDone, "resumed cleanup did not settle")
	if !held || m.updating() {
		t.Fatalf("resumed cleanup ownership: held=%v after=%v", held, m.updating())
	}
}

func TestResumedExitBeforeCommitStillRollsBack(t *testing.T) {
	m, launcher, _, _ := resumedCompletionMachine(t, true)
	peer := &stalledCompletionPeer{entered: make(chan struct{}), canceled: make(chan struct{})}
	m.conn = completionConnection(t, peer)
	if err := m.launch(); err != nil {
		t.Fatal(err)
	}
	awaitCompletionSignal(t, peer.entered, "candidate never entered claim")
	launcher.process.stop()
	awaitCompletionSignal(t, launcher.process.claimDone, "exited candidate retained its claim")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := m.supervise(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	m.updates.mu.Lock()
	result := *m.updates.status
	m.updates.mu.Unlock()
	if launcher.rollbacks.Load() != 1 || result.State != "failed" || !strings.Contains(result.Error, "offline rollback failed") {
		t.Fatalf("uncommitted resumed exit did not attempt rollback: count=%d status=%+v", launcher.rollbacks.Load(), result)
	}
}

func TestResumedCompletionRetriesPersistenceBeforeReadinessAndCleanup(t *testing.T) {
	m, launcher, _, _ := resumedCompletionMachine(t, true)
	log := &completionRetryLog{failed: make(chan struct{})}
	m.log = log
	if err := os.Mkdir(m.updatePath(), 0700); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(m.updatePath())
	if err := os.MkdirAll(m.stagePath(), 0700); err != nil {
		t.Fatal(err)
	}
	if err := m.launch(); err != nil {
		t.Fatal(err)
	}
	defer launcher.process.stop()
	awaitCompletionSignal(t, launcher.committed, "candidate did not reach private commit")
	close(launcher.release)
	awaitCompletionSignal(t, log.failed, "resumed completion did not observe its persistence failure")
	if !m.updating() || launcher.cleanups.Load() != 0 {
		t.Fatal("undurable resumed completion released admission or removed the candidate")
	}
	select {
	case <-m.ready:
		t.Fatal("resumed readiness preceded durable completion")
	default:
	}
	if _, err := os.Stat(m.stagePath()); err != nil {
		t.Fatal("resumed completion lost staging before status persisted", err)
	}
	if err := os.Remove(m.updatePath()); err != nil {
		t.Fatal(err)
	}
	awaitCompletionSignal(t, launcher.process.claimDone, "resumed persistence did not recover without another restart")
	if m.updating() || m.grant.bootstrap.Pending || launcher.cleanups.Load() != 1 {
		t.Fatalf("resumed completion did not settle once: updating=%v pending=%v cleanups=%d", m.updating(), m.grant.bootstrap.Pending, launcher.cleanups.Load())
	}
	m.updates.mu.Lock()
	result := *m.updates.status
	m.updates.mu.Unlock()
	if result.State != "succeeded" {
		t.Fatalf("resumed completion lost its committed result: %+v", result)
	}
}
