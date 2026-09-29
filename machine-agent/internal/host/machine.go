package host

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/protobuf/proto"
)

// Machine is one running machine role.
type Machine struct {
	grant    *Grant
	layout   Layout
	id       *Identity
	claims   *claims
	receipt  *receipt
	tfs      *tensorFS
	hub      *hubClient
	idle     *idle
	restarts *restarts
	log      io.Writer
	words    *lastWords // the Runtime's output, remembering its last refusal
	started  time.Time
	owned    bool // an owned machine keeps serving after it releases; a rental ends

	childAddr string // the Runtime's loopback worker listener
	mediaAddr string // where the Runtime proves the media listener
	conn      *grpc.ClientConn
	launcher  launcher

	mu         sync.Mutex
	proc       *runtimeProcess
	ready      chan struct{} // closed once the current Runtime is claimed
	runtimeAck *pb.ClaimAck
	releasing  bool
	control    uint64     // Control streams answered
	keeping    sync.Mutex // one pass of keepFinished at a time
	wake       chan struct{}
	preparing  sync.WaitGroup
	active     int
	inUpdate   bool // a Runtime update holds the machine: no Runtime launches but its own
	updates    runtimeUpdates
}

// Run boots the machine and serves until ctx ends or a rental's release is accepted.
func Run(ctx context.Context, g *Grant, log io.Writer) error {
	for _, name := range g.Ignored {
		fmt.Fprintf(log, "cozy machine: ignoring %s, which this machine does not read\n", name)
	}
	for _, line := range g.Skipped {
		fmt.Fprintf(log, "cozy machine: ignoring %s\n", line)
	}
	layout := NewLayout(g)
	id, err := prepare(g, layout)
	if err != nil {
		return err
	}
	rec, err := openReceipt(layout.boot("readiness-envelope.json"), g.receiptKey)
	if err != nil {
		return err
	}
	g.receiptKey = nil
	if rec.retained != nil && bootIDOf(rec.retained) != id.BootID {
		return errors.New("the retained readiness envelope names another boot than this machine root")
	}
	c, err := newClaims(g.WorkerID, id.BootID, id.Digest, g.Authorized)
	if err != nil {
		return err
	}
	now := time.Now()
	state, err := openIdle(filepath.Join(layout.State, "idle.json"), rec.retained == nil, now)
	if err != nil {
		return err
	}
	m := &Machine{grant: g, layout: layout, id: id, claims: c, receipt: rec, hub: newHubClient(g), idle: state,
		restarts: &restarts{path: filepath.Join(layout.State, "runtime-restart")}, log: log, started: now,
		owned: strings.HasPrefix(g.WorkerID, "om-"), ready: make(chan struct{}), wake: make(chan struct{}, 1)}
	if rec.retained == nil {
		if err := m.restarts.clear(); err != nil {
			return err
		}
	}
	m.openUpdates()
	m.tfs = &tensorFS{bin: layout.TFS, store: layout.Store, repoCache: g.RepoCacheRoot, observe: m.observeCache}
	if m.childAddr, err = freeLoopback(); err != nil {
		return err
	}
	// The endpoint binds before anything slow: any HTTP answer tells the Hub the machine is up.
	listeners, err := m.listen()
	if err != nil {
		return err
	}
	defer listeners.close()
	if err := m.serveWebRTC(ctx); err != nil {
		return err
	}
	if m.conn, err = m.dialRuntime(); err != nil {
		return err
	}
	defer m.conn.Close()
	if err := m.prepareLauncher(); err != nil {
		return err
	}
	defer m.launcher.close()
	if err := m.tfs.initStore(ctx); err != nil {
		return err
	}
	if err := m.launch(); err != nil {
		return err
	}
	errs := make(chan error, 1)
	go func() { errs <- m.supervise(ctx) }()
	select {
	case err := <-errs:
		m.keepAndStop()
		return err
	case err := <-listeners.failed:
		m.keepAndStop()
		return err
	}
}

func freeLoopback() (string, error) {
	listener, err := listen("127.0.0.1", 0)
	if err != nil {
		return "", err
	}
	defer listener.Close()
	return listener.Addr().String(), nil
}

// runtimeEnvironment is the Runtime's whole launch contract. It trusts only the daemon's key.
func (m *Machine) runtimeEnvironment() []string {
	g, l := m.grant, m.layout
	tokens := []string{strings.Repeat("0", 64)}
	if g.ObservedAuth != nil && len(g.ObservedAuth.MediaTokens) > 0 {
		tokens = g.ObservedAuth.MediaTokens
	}
	auth, _ := json.Marshal(OwnerAuth{ControlKey: m.claims.ownKey(), MediaTokens: tokens})
	_, childPort, _ := net.SplitHostPort(m.childAddr)
	_, mediaPort, _ := net.SplitHostPort(m.mediaAddr)
	env := []string{"PATH=" + filepath.Join(l.Root, "usr/local/bin") + ":/usr/bin:/bin", "TMPDIR=" + l.Tmp,
		"COZY_MACHINE_ROOT=" + l.Root, "COZY_WORKER_ID=" + g.WorkerID, "COZY_WORKER_INTERNAL_PORT=" + childPort,
		"COZY_MEDIA_INTERNAL_PORT=" + mediaPort, "COZY_RECORD_OWNER_AUTH_JSON=" + string(auth),
		"TENSORHUB_OBJECT_STORAGE_HOSTS=" + strings.Join(g.ObjectHosts, ",")}
	if g.StoreRoot != "" {
		env = append(env, "COZY_TENSORFS_ROOT="+g.StoreRoot)
	}
	return append(env, g.inherited...)
}

// prepareLauncher readies Runtime launches. A root machine starts its guardian and then
// becomes the machine uid; everything after this runs unprivileged.
func (m *Machine) prepareLauncher() error {
	env := m.runtimeEnvironment()
	m.words = &lastWords{Writer: m.log}
	if os.Geteuid() == 0 {
		if m.grant.Development && m.grant.DeveloperKey != "" {
			if err := startSSH(m.grant.DeveloperKey, m.log); err != nil {
				return err
			}
		}
		g, err := startGuardian(m.layout.Runtime, m.layout.Root, env, m.words)
		if err != nil {
			return err
		}
		m.launcher = g
		if err := os.MkdirAll(m.layout.Store, 0o755); err != nil {
			return err
		}
		if err := os.Lchown(m.layout.Store, machineUID, machineUID); err != nil {
			return err
		}
		if err := dropPrivilege(m.layout.Bootstrap, m.layout.State, filepath.Join(m.layout.Installs, ".stage")); err != nil {
			return err
		}
	} else {
		m.launcher = &directLauncher{path: m.layout.Runtime, root: m.layout.Root, env: env, out: m.words}
	}
	return nil
}

func (m *Machine) launch() error {
	// The payload is this launch's readiness: an earlier Runtime's must not stand for it.
	if err := os.Remove(m.layout.boot("readiness-payload")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	p, err := m.launcher.launch()
	if err != nil {
		return err
	}
	m.mu.Lock()
	m.proc, m.ready = p, make(chan struct{})
	m.mu.Unlock()
	go m.claimRuntime(p)
	return nil
}

func (m *Machine) dialRuntime() (*grpc.ClientConn, error) {
	roots := x509.NewCertPool()
	roots.AddCert(mustLeaf(m.id.Leaf))
	return grpc.NewClient(m.childAddr, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{
		MinVersion: tls.VersionTLS13, RootCAs: roots, ServerName: ServerName})),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxMessageBytes), grpc.MaxCallSendMsgSize(maxMessageBytes)))
}

func mustLeaf(c tls.Certificate) *x509.Certificate {
	if c.Leaf != nil {
		return c.Leaf
	}
	leaf, _ := x509.ParseCertificate(c.Certificate[0])
	return leaf
}

// claimRuntime waits for the Runtime's readiness payload, seals the receipt with it, and
// records the daemon as the Runtime's owner.
func (m *Machine) claimRuntime(p *runtimeProcess) {
	payload := m.layout.boot("readiness-payload")
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		if raw, err := os.ReadFile(payload); err == nil && len(raw) > 0 {
			if err := m.receipt.seal(raw, m.grant.WorkerPort, m.grant.ObservedAuth, m.webrtcReceipt()); err != nil {
				fmt.Fprintln(m.log, "cozy machine: the Runtime's readiness was refused:", err)
				p.stop()
				return
			}
			break
		}
		select {
		case <-p.done:
			return
		case <-tick.C:
		}
	}
	ready := func(ack *pb.ClaimAck) {
		m.mu.Lock()
		if m.proc == p {
			m.runtimeAck = ack
			close(m.ready)
		}
		m.mu.Unlock()
	}
	if ack := m.heldClaim(); ack != nil { // this boot is claimed already: a relaunched Runtime keeps its owner
		ready(ack)
		return
	}
	for {
		ack, err := m.controlClaim()
		if err == nil && ack.Accepted {
			if raw, err := proto.Marshal(ack); err == nil {
				_ = writeAtomic(filepath.Join(m.layout.State, "runtime-claim"), raw, 0o600)
			}
			ready(ack)
			return
		}
		if err == nil && ack.Rejection != pb.ClaimRejection_CLAIM_REJECTION_UNDURABLE {
			fmt.Fprintln(m.log, "cozy machine: the Runtime refused this machine's Claim:", ack.Rejection)
			p.stop()
			return
		}
		select {
		case <-p.done:
			return
		case <-time.After(200 * time.Millisecond):
		}
	}
}

// controlClaim records the daemon as the Runtime's owner on one Control stream.
func (m *Machine) controlClaim() (*pb.ClaimAck, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := pb.NewWorkerControlClient(m.conn).Control(ctx)
	if err != nil {
		return nil, err
	}
	if err := stream.Send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_Claim{Claim: m.claims.claim}}); err != nil {
		return nil, err
	}
	for {
		frame, err := stream.Recv()
		if err != nil {
			return nil, err
		}
		if failure := frame.GetBootFailure(); failure != nil {
			return nil, fmt.Errorf("the Runtime failed to boot: %s", failure.GetDetail())
		}
		if ack := frame.GetClaimAck(); ack != nil {
			_ = stream.CloseSend()
			return ack, nil
		}
	}
}

// runtime answers the claimed Runtime connection, launching a stopped Runtime on an owned
// machine. It waits while the Runtime boots.
func (m *Machine) runtime(ctx context.Context) (*grpc.ClientConn, error) {
	for {
		if m.idle.releasedNow() && !m.owned {
			return nil, unavailable("machine_released", "this rental released itself after its idle deadline")
		}
		m.mu.Lock()
		p, ready := m.proc, m.ready
		m.mu.Unlock()
		if p == nil || p.exited() {
			if m.updating() {
				return nil, unavailable("runtime_updating", "this machine is updating its Runtime; ask again when it is done")
			}
			if m.restarts.gone() {
				return nil, refusal(codes.FailedPrecondition, "runtime_gone", m.runtimeGone())
			}
			m.poke()
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(100 * time.Millisecond):
			}
			continue
		}
		select {
		case <-ready:
			return m.conn, nil
		case <-p.done:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

// poke asks the supervisor to launch a Runtime a call needs.
// runtimeGone says why the Runtime cannot start. An owned machine then forgets it, so the
// next run launches the Runtime again; a rental stays known idle.
func (m *Machine) runtimeGone() string {
	reason := "this machine's Runtime exited twice without completing work"
	if words := m.words.last(); words != "" {
		reason += ": " + words
	}
	if m.owned {
		_ = m.restarts.clear()
	}
	return reason
}

func (m *Machine) poke() {
	select {
	case m.wake <- struct{}{}:
	default:
	}
}

func (m *Machine) stopRuntime() {
	m.mu.Lock()
	p := m.proc
	m.mu.Unlock()
	if p != nil && !p.exited() {
		p.stop()
		<-p.done
	}
}

func (m *Machine) observeCache(o cacheObservation) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := m.hub.observeCache(ctx, o); err != nil {
		fmt.Fprintln(m.log, "cozy machine: cache observation not recorded:", err)
	}
}

// supervise is the machine's one loop: Runtime exits, activity, and the idle release.
func (m *Machine) supervise(ctx context.Context) error {
	work := &activity{path: m.layout.boot("worker-activity"), log: m.log}
	wasBusy := false
	var nextAsk time.Time
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		now := time.Now()
		m.mu.Lock()
		p := m.proc
		m.mu.Unlock()
		if p != nil && p.exited() {
			if err := m.relaunchAfter(p); err != nil {
				return err
			}
		}
		busy, err := false, error(nil)
		if m.running() {
			busy, err = work.observe(now)
		}
		if wasBusy && !busy && err == nil {
			go m.keepSettled()
		}
		wasBusy = busy && err == nil
		if err := m.restarts.observe(busy, err); err != nil {
			return err
		}
		if m.preparations() || m.updating() {
			busy, err = true, nil
		}
		deadline, observeErr := m.idle.observe(now, busy, err == nil)
		if observeErr != nil && !errors.Is(observeErr, errReleased) {
			return observeErr
		}
		if !now.Before(deadline) && !now.Before(nextAsk) {
			released, err := m.idle.claim(now)
			if err != nil {
				return err
			}
			if released {
				done, err := m.release(ctx)
				if err != nil {
					fmt.Fprintln(m.log, "cozy machine: idle release not accepted; retrying without extending the deadline:", err)
					nextAsk = now.Add(5 * time.Second)
				} else if done {
					return nil
				}
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-tick.C:
		case <-m.wake:
			m.mu.Lock()
			idle := m.proc == nil
			m.mu.Unlock()
			if idle && !m.updating() && !m.restarts.gone() && (m.owned || !m.idle.releasedNow()) {
				m.launchOrIdle()
			}
		}
	}
}

func (m *Machine) running() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.proc != nil && !m.proc.exited()
}

func (m *Machine) relaunchAfter(p *runtimeProcess) error {
	m.mu.Lock()
	if m.proc != p {
		m.mu.Unlock()
		return nil
	}
	m.proc = nil
	updating := m.inUpdate
	m.mu.Unlock()
	if p.stopped.Load() || updating { // the update relaunches it itself
		return nil
	}
	again, err := m.restarts.exited()
	if err != nil {
		return err
	}
	if !again {
		fmt.Fprintln(m.log, "cozy machine: the Runtime exited again without completing work; it is not relaunched and the machine is known idle:", exitText(p.err))
		return nil
	}
	fmt.Fprintln(m.log, "cozy machine: relaunching the Runtime:", exitText(p.err))
	m.launchOrIdle()
	return nil
}

// launchOrIdle launches a Runtime; one that cannot start leaves the machine known idle,
// still serving, rather than ending it.
func (m *Machine) launchOrIdle() {
	if err := m.launch(); err != nil {
		fmt.Fprintln(m.log, "cozy machine: the Runtime could not be launched; the machine is known idle:", err)
		_ = writeAtomic(m.restarts.path, []byte("gone\n"), 0o600)
	}
}

// release ends an idle session. The Runtime stops first, which frees every GPU byte; a
// rental then asks the Hub to end it, and an owned machine keeps serving.
func (m *Machine) release(ctx context.Context) (bool, error) {
	m.mu.Lock()
	m.releasing = true
	m.mu.Unlock()
	m.keepAndStop()
	call, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	err := m.hub.release(call)
	if !m.owned {
		if err == nil {
			fmt.Fprintln(m.log, "cozy machine: idle deadline passed; Tensorhub accepted the release")
		}
		return err == nil, err
	}
	if err != nil {
		fmt.Fprintln(m.log, "cozy machine: the release was not recorded at Tensorhub:", err)
	}
	fmt.Fprintln(m.log, "cozy machine: idle; the Runtime is stopped until the next call needs it")
	if err := m.restarts.clear(); err != nil {
		return false, err
	}
	m.mu.Lock()
	m.releasing = false
	m.mu.Unlock()
	return false, m.idle.reset(time.Now())
}

func (m *Machine) phase() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	switch {
	case m.releasing:
		return "releasing"
	case m.proc == nil || m.proc.exited():
		return "idle"
	case m.runtimeAck == nil:
		return "booting"
	}
	select {
	case <-m.ready:
		return "ready"
	default:
		return "booting"
	}
}

func (m *Machine) preparations() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.active > 0
}

func (m *Machine) beginPreparation() func() {
	m.mu.Lock()
	m.active++
	m.mu.Unlock()
	return func() {
		m.mu.Lock()
		m.active--
		m.mu.Unlock()
	}
}

// heldClaim is the ClaimAck this boot's Runtime gave the daemon, when the Runtime's durable
// ownership record still names the daemon for this boot; otherwise nil, and the daemon claims.
func (m *Machine) heldClaim() *pb.ClaimAck {
	raw, err := os.ReadFile(filepath.Join(m.layout.Root, "run/cozy/worker/ownership.json"))
	if err != nil {
		return nil
	}
	var owner struct {
		Boot   string `json:"worker_boot_id"`
		Epoch  uint64 `json:"record_owner_epoch"`
		ID     string `json:"record_owner_id"`
		Stream uint64 `json:"control_stream_epoch"`
	}
	if json.Unmarshal(raw, &owner) != nil || owner.Boot != m.id.BootID || owner.ID != runtimeOwnerID ||
		owner.Epoch != runtimeOwnerEpoch || owner.Stream == 0 {
		return nil
	}
	saved, err := os.ReadFile(filepath.Join(m.layout.State, "runtime-claim"))
	ack := &pb.ClaimAck{}
	if err != nil || proto.Unmarshal(saved, ack) != nil || ack.WorkerBootId != m.id.BootID {
		return nil
	}
	return ack
}

// keepAndStop records finished runs, then stops the Runtime.
func (m *Machine) keepAndStop() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	m.keepFinished(ctx)
	cancel()
	m.stopRuntime()
}

// keepSettled records finished runs once work settles; one pass at a time.
func (m *Machine) keepSettled() {
	if !m.keeping.TryLock() {
		return
	}
	defer m.keeping.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	m.keepFinished(ctx)
}
