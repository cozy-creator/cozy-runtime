package host

import (
	"context"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type updateCompletionPeer struct{ startupClosurePeer }

func (*updateCompletionPeer) Control(stream grpc.BidiStreamingServer[pb.RecordOwnerFrame, pb.WorkerFrame]) error {
	if _, err := stream.Recv(); err != nil {
		return err
	}
	return stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{Accepted: true}}})
}
func completionConnection(t *testing.T, peer pb.WorkerControlServer) *grpc.ClientConn {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	pb.RegisterWorkerControlServer(server, peer)
	go server.Serve(listener)
	t.Cleanup(server.Stop)
	conn, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return conn
}
func TestManualUpdateCommitKeepsLaunchOwnership(t *testing.T) {
	m, _, _ := updateMachine(t)
	m.conn = completionConnection(t, &updateCompletionPeer{})
	m.owned, m.inUpdate, m.grant.startupPending = true, true, true
	m.updates.active = "manual"
	m.launcher = &startupCommitLauncher{receipt: &receipt{}}
	p := &runtimeProcess{incarnation: "candidate", done: make(chan struct{})}
	if err := m.commitStartup(p, nil); err != nil {
		t.Fatal(err)
	}
	if !m.updating() {
		t.Fatal("authenticated commit released manual launch ownership before update cleanup")
	}
}
func TestManualUpdateOwnsExitedCandidateUntilCleanup(t *testing.T) {
	m, _, _ := updateMachine(t)
	m.owned, m.inUpdate = true, true
	m.idle = &idle{persistent: true}
	m.updates.active = "manual"
	p := &runtimeProcess{incarnation: "committed", done: make(chan struct{}), err: errors.New("exit after commit")}
	close(p.done)
	m.proc = p
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := m.supervise(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if m.proc != p {
		t.Fatal("supervisor cleared the candidate while its manual update still owned completion")
	}
}

type completionLauncher struct {
	machine            *Machine
	previous           []string
	process            *runtimeProcess
	committed, release chan struct{}
	rollbacks          atomic.Int32
	cleanups           atomic.Int32
}

func (*completionLauncher) close() {}
func (l *completionLauncher) launch(incarnation string) (*runtimeProcess, error) {
	p := &runtimeProcess{incarnation: incarnation, done: make(chan struct{}), err: errors.New("candidate exited during commit response")}
	var stopped sync.Once
	p.stop = func() { stopped.Do(func() { close(p.done) }) }
	l.process = p
	path := l.machine.layout.boot("readiness-payload")
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	raw, _ := json.Marshal(map[string]string{"process_incarnation": incarnation, "pod_boot_id": "boot", "tls_certificate_der_base64": base64.StdEncoding.EncodeToString([]byte("leaf"))})
	if err := os.WriteFile(path, raw, 0600); err != nil {
		return nil, err
	}
	return p, nil
}
func (l *completionLauncher) maintain(op string, _ []string) (string, error) {
	switch op {
	case "update-cleanup":
		l.cleanups.Add(1)
		return "", nil
	case "software-state":
		return `{"quiescent":true}`, nil
	case "pair":
		return strings.Join(l.previous, "\n"), nil
	case "restart":
		return "accepted", nil
	case "update-stage":
		return "prepared", nil
	case "update-prepare":
		return "boot_pending", nil
	case "capabilities":
		return `{"capabilities":["machine-supervisor/1"]}`, nil
	case "startup-commit":
		// The private commit is durable, but its reply has not reached the agent.
		if err := l.machine.startupSave(startupStatus{State: "succeeded"}); err != nil {
			return "", err
		}
		close(l.committed)
		<-l.release
		return "", nil
	case "startup-rollback":
		l.rollbacks.Add(1)
		return "", errors.New("no interrupted update awaits rollback")
	}
	return "", errors.New("unexpected fixture maintenance: " + op)
}
func startCompletionUpdate(t *testing.T, peer pb.WorkerControlServer, configure ...func(*Machine)) (*Machine, *completionLauncher, <-chan struct{}) {
	t.Helper()
	m, _, _ := updateMachine(t)
	m.conn = completionConnection(t, peer)
	m.owned = true
	m.id = &Identity{BootID: "boot", Leaf: tls.Certificate{Certificate: [][]byte{[]byte("leaf")}}}
	l := &completionLauncher{machine: m, committed: make(chan struct{}), release: make(chan struct{})}
	m.launcher = l
	for _, name := range []string{"cozy_runtime-0.18.86-py3-none-any.whl", "tensorfs-0.3.78-py3-none-any.whl"} {
		path := filepath.Join(m.layout.Root, name)
		if err := os.WriteFile(path, []byte(name), 0600); err != nil {
			t.Fatal(err)
		}
		l.previous = append(l.previous, path)
	}
	file := "cozy_runtime-0.18.87-py3-none-any.whl"
	digest, _, err := m.stageWheel(file, strings.NewReader("candidate"))
	if err != nil {
		t.Fatal(err)
	}
	request := updateRequest{Operation: "completion", Runtime: &wheelChoice{File: file, SHA256: digest}}
	status := updateStatus{Operation: request.Operation, State: "waiting", From: pairVersions{Runtime: "0.18.86", TensorFS: "0.3.78"}}
	m.updates.status, m.updates.active = &status, request.Operation
	for _, configure := range configure {
		configure(m)
	}
	finished := make(chan struct{})
	go func() { m.runUpdate(request, status); close(finished) }()
	return m, l, finished
}

func TestManualUpdateExitWaitsForInFlightReadinessCommit(t *testing.T) {
	m, l, finished := startCompletionUpdate(t, &updateCompletionPeer{})
	var release sync.Once
	defer release.Do(func() { close(l.release) })
	select {
	case <-l.committed:
	case <-time.After(3 * time.Second):
		t.Fatal("candidate never reached authenticated health commit")
	}
	l.process.stop()
	early := false
	select {
	case <-finished:
		early = true
	case <-time.After(100 * time.Millisecond):
	}
	release.Do(func() { close(l.release) })
	select {
	case <-finished:
	case <-time.After(3 * time.Second):
		t.Fatal("completed readiness commit did not settle the update")
	}
	m.updates.mu.Lock()
	result := *m.updates.status
	m.updates.mu.Unlock()
	if early || l.rollbacks.Load() != 0 || result.State != "succeeded" {
		t.Fatalf("Runtime exit rolled back a committed candidate: early=%v rollbacks=%d status=%+v", early, l.rollbacks.Load(), result)
	}
}

// A dead candidate must also unblock a Claim whose peer never replies. Waiting
// for the readiness attempt is safe only when its RPC lifetime follows that child.
type stalledCompletionPeer struct {
	startupClosurePeer
	entered, canceled chan struct{}
}

func (p *stalledCompletionPeer) Control(stream grpc.BidiStreamingServer[pb.RecordOwnerFrame, pb.WorkerFrame]) error {
	if _, err := stream.Recv(); err != nil {
		return err
	}
	close(p.entered)
	<-stream.Context().Done()
	close(p.canceled)
	return stream.Context().Err()
}
func TestManualUpdateExitCancelsUnansweredClaim(t *testing.T) {
	peer := &stalledCompletionPeer{entered: make(chan struct{}), canceled: make(chan struct{})}
	m, launcher, finished := startCompletionUpdate(t, peer)
	select {
	case <-peer.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("candidate did not enter its authenticated claim")
	}
	launcher.process.stop()
	select {
	case <-finished:
	case <-time.After(3 * time.Second):
		t.Fatal("dead candidate retained its unanswered Claim and update ownership")
	}
	select {
	case <-peer.canceled:
	case <-time.After(time.Second):
		t.Fatal("dead candidate left an orphaned Control stream")
	}
	if launcher.rollbacks.Load() != 1 || m.updating() {
		t.Fatalf("uncommitted exit did not reach rollback cleanup: rollbacks=%d updating=%v", launcher.rollbacks.Load(), m.updating())
	}
}

func TestManualUpdateRollbackKeepsLaunchOwnership(t *testing.T) {
	m, _, _ := updateMachine(t)
	m.owned, m.inUpdate, m.grant.startupPending = true, true, true
	m.updates.active = "manual"
	m.launcher = &completionLauncher{}
	if err := m.rollbackStartup(context.Background(), errors.New("candidate failed")); err == nil {
		t.Fatal("fixture rollback failure was lost")
	}
	if !m.updating() {
		t.Fatal("rollback released manual launch ownership before update cleanup")
	}
}

func TestCommittedUpdateReceiptFailureDoesNotRollback(t *testing.T) {
	m, launcher, finished := startCompletionUpdate(t, &updateCompletionPeer{}, func(m *Machine) {
		m.owned = false
		m.receipt = &receipt{path: filepath.Join(m.layout.State, "receipt.json"), key: make([]byte, 32)}
		if err := os.Mkdir(m.receipt.path, 0700); err != nil {
			t.Fatal(err)
		}
	})
	var release sync.Once
	defer release.Do(func() { close(launcher.release) })
	select {
	case <-launcher.committed:
	case <-time.After(3 * time.Second):
		t.Fatal("candidate did not commit")
	}
	release.Do(func() { close(launcher.release) })
	select {
	case <-finished:
	case <-time.After(3 * time.Second):
		t.Fatal("receipt failure did not settle the update")
	}
	m.updates.mu.Lock()
	result := *m.updates.status
	m.updates.mu.Unlock()
	if launcher.rollbacks.Load() != 0 || result.State != "failed" || !strings.Contains(result.Error, "committed") || m.startupPending() {
		t.Errorf("receipt failure confused committed software with pending rollback: rollbacks=%d status=%+v pending=%v", launcher.rollbacks.Load(), result, m.startupPending())
	}
	var journal startupStatus
	raw, err := os.ReadFile(m.startupPath("status.json"))
	if err != nil || json.Unmarshal(raw, &journal) != nil || journal.State != "succeeded" {
		t.Fatalf("committed installation journal changed: %s %v", raw, err)
	}
	select {
	case <-m.ready:
		t.Fatal("failed receipt opened API readiness")
	default:
	}
	if m.receipt.Envelope() != nil || len(m.receipt.key) != 32 {
		t.Fatal("failed receipt write advertised readiness or spent the only signing key")
	}
	// Repair the failed persistence target and retry with the same authority.
	if err := os.Remove(m.receipt.path); err != nil {
		t.Fatal(err)
	}
	raw, err = os.ReadFile(m.layout.boot("readiness-payload"))
	if err != nil {
		t.Fatal(err)
	}
	if err := m.attestRuntime(raw); err != nil {
		t.Fatalf("receipt retry after storage repair: %v", err)
	}
	if m.receipt.Envelope() == nil || len(m.receipt.key) != 0 {
		t.Fatal("persisted receipt did not consume its authority exactly once")
	}
}
