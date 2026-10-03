package host

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"testing"
	"time"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

// reporter is a Runtime's activity reporter: a separate process that atomically replaces the
// report every tick, stamped by the test's clock (offset in ns), as the Runtime's reporter does.
const reporter = `d=$1; n=0
while :; do
  n=$((n+1))
  printf '{"version":2,"sequence":%d,"active_work":%s,"holding":[]}\n' "$n" "$(cat "$d/active")" >"$d/.next"
  t=$(( $(date +%s%N) + $(cat "$d/offset") ))
  touch -d "@$((t / 1000000000)).$(printf %09d $((t % 1000000000)))" "$d/.next"
  mv "$d/.next" "$d/worker-activity"
  sleep 0.05
done`

// idleRental is a rental machine on the real request plane, idle ledger, supervisor tick and Hub
// client, with a reporting Runtime process and a Runtime gRPC service that accepts runs.
type idleRental struct {
	m        *Machine
	dir      string
	work     *activity
	log      *lockedLog
	offset   time.Duration
	runtime  *os.Process
	client   pb.WorkerControlClient
	claim    *pb.Claim
	nextAsk  time.Time
	mu       sync.Mutex
	releases []string
}

func newIdleRental(t *testing.T) *idleRental {
	t.Helper()
	h := &idleRental{dir: t.TempDir()}
	h.write(t, "active", "false")
	h.write(t, "offset", "0")
	cmd := exec.Command("/bin/sh", "-c", reporter, "reporter", h.dir)
	if err := startChild(cmd); err != nil {
		t.Fatal(err)
	}
	h.runtime = cmd.Process
	p := &runtimeProcess{pid: cmd.Process.Pid, done: make(chan struct{})}
	p.stop = func() { _ = h.runtime.Kill() }
	go func() { p.err = waitChild(cmd); close(p.done) }()
	t.Cleanup(func() { p.stop(); <-p.done })

	hub := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		h.releases = append(h.releases, r.Method+" "+r.URL.Path+" "+r.Header.Get("X-Cozy-Worker-ID")+" "+r.Header.Get("X-Cozy-Worker-Token"))
		h.mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(hub.Close)

	public, private, _ := ed25519.GenerateKey(rand.Reader)
	claims, err := newClaims("rental-1", "boot", make([]byte, 32), []ed25519.PublicKey{public})
	if err != nil {
		t.Fatal(err)
	}
	state, err := openIdle(filepath.Join(h.dir, "idle.json"), true, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	ready := make(chan struct{})
	close(ready)
	h.log = &lockedLog{}
	grant := &Grant{WorkerID: "rental-1", Lifetime: "rental", HubOrigin: hub.URL, HubCA: hub.Certificate().Raw, WorkerToken: "worker-token"}
	h.m = &Machine{grant: grant, id: &Identity{BootID: "boot"}, claims: claims, hub: newHubClient(grant),
		idle: state, restarts: &restarts{}, log: h.log, proc: p, ready: ready,
		runtimeAck: &pb.ClaimAck{Accepted: true, WireMinor: pb.WireMinor}, wake: make(chan struct{}, 1)}
	h.m.conn = runtimeService(t, h)
	h.work = &activity{path: filepath.Join(h.dir, "worker-activity"), log: h.log}

	listener := bufconn.Listen(1 << 20)
	server := grpc.NewServer(grpc.UnknownServiceHandler(h.m.serveRPC))
	go server.Serve(listener)
	t.Cleanup(server.Stop)
	conn, err := grpc.NewClient("passthrough:///machine", grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	h.client = pb.NewWorkerControlClient(conn)
	transcript, _ := claims.transcript(1)
	h.claim = &pb.Claim{WorkerId: "rental-1", WorkerBootId: "boot", RecordOwnerEpoch: 1, WireMinor: pb.WireMinor, Proof: ed25519.Sign(private, transcript)}
	h.awaitReport(t, false)
	return h
}

type lockedLog struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *lockedLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.Write(p)
}
func (l *lockedLog) has(line string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return bytes.Contains(l.b.Bytes(), []byte(line))
}

type stubWorkerControl struct {
	pb.UnimplementedWorkerControlServer
	h *idleRental
}

// SubmitMachineExecution accepts the run as the Runtime does: its unit starts and the
// activity report says so before the receipt is returned.
func (s *stubWorkerControl) SubmitMachineExecution(_ context.Context, in *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	if err := writeAtomic(filepath.Join(s.h.dir, "active"), []byte("true"), 0o600); err != nil || !s.h.report(true) {
		return nil, status.Error(codes.Internal, "the run's unit was never reported")
	}
	return &pb.MachineExecutionReceipt{RequestId: "run-" + in.SubmissionId, SubmissionId: in.SubmissionId}, nil
}

type stubPreparation struct {
	pb.UnimplementedRuntimePreparationServer
}

func (stubPreparation) ProtocolInfo(context.Context, *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
	return &pb.ProtocolInfoResult{WireMinor: pb.WireMinor, MinimumWireMinor: pb.MinCompatibleWireMinor}, nil
}

func runtimeService(t *testing.T, h *idleRental) *grpc.ClientConn {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	pb.RegisterWorkerControlServer(server, &stubWorkerControl{h: h})
	pb.RegisterRuntimePreparationServer(server, stubPreparation{})
	go server.Serve(listener)
	t.Cleanup(server.Stop)
	conn, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return conn
}

func (h *idleRental) write(t *testing.T, name, value string) {
	t.Helper()
	if err := writeAtomic(filepath.Join(h.dir, name), []byte(value), 0o600); err != nil {
		t.Fatal(err)
	}
}

func (h *idleRental) now() time.Time { return time.Now().Add(h.offset) }

// advance moves the test clock; the reporter stamps its next reports with it.
func (h *idleRental) advance(t *testing.T, d time.Duration) {
	h.offset += d
	h.write(t, "offset", strconv.FormatInt(h.offset.Nanoseconds(), 10))
}

func (h *idleRental) setActive(t *testing.T, active bool) {
	h.write(t, "active", strconv.FormatBool(active))
	h.awaitReport(t, active)
}

// report waits for a fresh report of active.
func (h *idleRental) report(active bool) bool {
	for limit := time.Now().Add(10 * time.Second); time.Now().Before(limit); time.Sleep(10 * time.Millisecond) {
		if busy, _, err := h.work.observe(h.now()); err == nil && busy == active {
			return true
		}
	}
	return false
}

func (h *idleRental) awaitReport(t *testing.T, active bool) {
	t.Helper()
	if !h.report(active) {
		t.Fatalf("the Runtime never reported active_work=%v", active)
	}
}

func (h *idleRental) submit(t *testing.T, id string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	if _, err := h.client.SubmitMachineExecution(ctx, &pb.MachineExecutionSubmit{Claim: h.claim, SubmissionId: id}); err != nil {
		t.Fatal(err)
	}
}

// tick runs one supervisor idle step at now; true means the Hub accepted the release.
func (h *idleRental) tick(t *testing.T, now time.Time) bool {
	t.Helper()
	done, err := h.m.idleTick(t.Context(), h.work, now, &h.nextAsk)
	if err != nil {
		t.Fatal(err)
	}
	return done
}

// finish ends the run on the Runtime and ticks until its reports settle it.
func (h *idleRental) finish(t *testing.T) {
	t.Helper()
	h.setActive(t, false)
	for limit := time.Now().Add(10 * time.Second); h.m.idle.unsettled(); time.Sleep(20 * time.Millisecond) {
		if time.Now().After(limit) {
			t.Fatal("the Runtime's idle reports never settled the accepted run")
		}
		h.tick(t, h.now())
	}
}

func (h *idleRental) signal(t *testing.T, s syscall.Signal) {
	t.Helper()
	if err := h.runtime.Signal(s); err != nil {
		t.Fatal(err)
	}
}

func (h *idleRental) released(t *testing.T) []string {
	t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.releases...)
}

func (h *idleRental) ledger(t *testing.T) idleState {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(h.dir, "idle.json"))
	if err != nil {
		t.Fatal(err)
	}
	var s idleState
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatal(err)
	}
	return s
}

// togepi: a run completed, then the Runtime was stopped (alive, unreachable, its report
// stale). No accepted run is unfinished, so the rental releases after the normal idle window.
func TestUnreachableRuntimeWithoutUnsettledRunReleasesAfterIdleWindow(t *testing.T) {
	h := newIdleRental(t)
	h.submit(t, "a")
	if !h.m.idle.unsettled() {
		t.Fatal("the agent kept no record of the run the Runtime accepted")
	}
	lastBusy := h.now()
	h.tick(t, lastBusy)
	h.finish(t)

	h.signal(t, syscall.SIGSTOP)
	stale := h.now().Add(activityFreshness + time.Second)
	if h.tick(t, stale) || h.ledger(t).Unknown {
		t.Fatal("unreadable activity with no unsettled run held the rental")
	}
	deadline := h.m.idle.deadline()
	if deadline.UnixMilli() < lastBusy.Add(idleGrace).UnixMilli() || deadline.After(h.now().Add(idleGrace)) {
		t.Fatalf("deadline %v is not one idle window after the last busy report %v", deadline, lastBusy)
	}
	if h.tick(t, deadline.Add(-time.Second)) || len(h.released(t)) != 0 {
		t.Fatal("released before the idle window ended")
	}
	if !h.tick(t, deadline.Add(time.Second)) {
		t.Fatal("an idle rental with an unreachable Runtime did not release")
	}
	if got := h.released(t); len(got) != 1 || got[0] != "POST /v1/worker/rental/release rental-1 worker-token" {
		t.Fatalf("Hub saw %q", got)
	}
	if !h.ledger(t).Released {
		t.Fatal("the release claim is not durable")
	}
	if !h.log.has("releasing with the Runtime's activity unreadable") {
		t.Fatal("the release does not say the Runtime was unreadable")
	}
}

// A run the Runtime accepted has not finished when the Runtime stops answering: the rental is
// never released while it stays unreachable. When it comes back it is supervised as usual.
func TestUnreachableRuntimeWithUnsettledRunHoldsUntilItReturns(t *testing.T) {
	h := newIdleRental(t)
	h.submit(t, "a")
	h.tick(t, h.now())
	h.signal(t, syscall.SIGSTOP)
	outage := h.now()
	for _, after := range []time.Duration{activityFreshness + time.Second, idleGrace + time.Minute, 6 * time.Hour} {
		now := outage.Add(after)
		if h.tick(t, now) || !h.m.idle.deadline().After(now) {
			t.Fatalf("released %v into the outage with an unfinished run", after)
		}
	}
	if s := h.ledger(t); !s.Unknown || s.Unsettled == 0 || len(h.released(t)) != 0 {
		t.Fatalf("an unfinished run did not hold: %+v releases=%q", s, h.released(t))
	}
	if !h.log.has("idle held: the Runtime's activity is unreadable and a run it accepted has not finished") {
		t.Fatal("the hold does not say why")
	}

	h.advance(t, 6*time.Hour+time.Minute)
	h.signal(t, syscall.SIGCONT)
	h.awaitReport(t, true)
	back := h.now()
	if h.tick(t, back) || h.ledger(t).Unknown {
		t.Fatal("the returned Runtime's busy report did not replace unknown activity")
	}
	h.finish(t)
	deadline := h.m.idle.deadline()
	if deadline.UnixMilli() < back.Add(idleGrace).UnixMilli() || deadline.After(h.now().Add(idleGrace)) {
		t.Fatalf("deadline %v is not one idle window after the last busy report %v", deadline, back)
	}
	// An idle, reachable Runtime releases on the ordinary clock.
	h.advance(t, deadline.Sub(h.now())-time.Second)
	h.awaitReport(t, false)
	if h.tick(t, h.now()) || len(h.released(t)) != 0 {
		t.Fatal("released before the idle window ended")
	}
	h.advance(t, 2*time.Second)
	h.awaitReport(t, false)
	if !h.tick(t, h.now()) || len(h.released(t)) != 1 {
		t.Fatal("an idle reachable Runtime did not release on the ordinary clock")
	}
}

// One idle report written after an acceptance could have been computed before it; the
// agent settles a run only on a second one.
func TestOneIdleReportDoesNotSettleAnAcceptedRun(t *testing.T) {
	now := time.Now()
	i, err := openIdle(filepath.Join(t.TempDir(), "idle.json"), true, now)
	if err != nil {
		t.Fatal(err)
	}
	if err := i.work(now, true); err != nil {
		t.Fatal(err)
	}
	report := now.Add(time.Millisecond)
	for range 3 { // the same report read again is still one report
		if _, err := i.observe(report, sample{known: true, at: report}); err != nil || !i.unsettled() {
			t.Fatal("one idle report settled the run", err)
		}
	}
	if _, err := i.observe(now, sample{known: true, at: now.Add(-time.Second)}); err != nil || !i.unsettled() {
		t.Fatal("a report older than the acceptance settled the run", err)
	}
	next := report.Add(50 * time.Millisecond)
	if _, err := i.observe(next, sample{known: true, at: next}); err != nil || i.unsettled() {
		t.Fatal("a second idle report did not settle the run", err)
	}
}

// An older agent's ledger records no runs: after work, unknown activity keeps holding until
// the Runtime reports idle twice.
func TestOlderAgentLedgerHoldsUntilTheRuntimeReportsIdle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "idle.json")
	now := time.Now()
	legacy, _ := json.Marshal(map[string]any{"deadline_ms": now.Add(time.Minute).UnixMilli(), "work_observed": true})
	if err := os.WriteFile(path, legacy, 0o600); err != nil {
		t.Fatal(err)
	}
	i, err := openIdle(path, false, now)
	if err != nil {
		t.Fatal(err)
	}
	if deadline, _ := i.observe(now.Add(time.Hour), sample{}); !deadline.After(now.Add(time.Hour)) {
		t.Fatal("an older agent's work lost its unknown-activity hold")
	}
	unused := filepath.Join(t.TempDir(), "idle.json")
	never, _ := json.Marshal(map[string]any{"deadline_ms": now.Add(time.Minute).UnixMilli()})
	if err := os.WriteFile(unused, never, 0o600); err != nil {
		t.Fatal(err)
	}
	if i, err = openIdle(unused, false, now); err != nil || i.unsettled() {
		t.Fatal("an older ledger with no work invented an unsettled run", err)
	}
}
