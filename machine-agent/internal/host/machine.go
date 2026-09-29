package host

import (
	"context"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
)

// Machine is one running machine role.
type Machine struct {
	lease    *machineLease
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
	owned    bool // persistent machines have no Hub allocation lifecycle

	childAddr   string // the Runtime's loopback worker listener
	mediaAddr   string // where the Runtime proves the media listener
	conn        *grpc.ClientConn
	launcher    launcher
	initialized chan struct{} // closed after the process launcher is installed
	launchMu    sync.Mutex    // normal launches and guarded update acquisition

	mu          sync.Mutex
	proc        *runtimeProcess
	ready       chan struct{} // closed once the current Runtime is claimed
	runtimeAck  *pb.ClaimAck
	runtimeCaps []string
	releasing   bool
	control     uint64 // Control streams answered
	wake        chan struct{}
	preparing   sync.WaitGroup
	active      int
	inUpdate    bool // a Runtime update holds the machine: no Runtime launches but its own
	updates     runtimeUpdates
	accessMu    sync.Mutex
}

// Run boots the machine and serves until ctx ends or a rental's release is accepted.
func Run(ctx context.Context, g *Grant, log io.Writer) error {
	for _, name := range g.Ignored {
		fmt.Fprintf(log, "cozy machine: ignoring %s, which this machine does not read\n", name)
	}

	layout := NewLayout(g)
	lease, err := requestPlaneLease(g, layout.State)
	if err != nil {
		return err
	}
	defer lease.close()
	if err := lease.watchRoot(layout.Root); err != nil {
		return err
	}
	id, err := requestPlaneIdentity(g, layout)
	if err != nil {
		return err
	}
	rec := &receipt{standalone: true}
	if g.Lifetime == "rental" || len(g.receiptKey) != 0 {
		rec, err = openReceipt(layout.boot("readiness-envelope.json"), g.receiptKey)
	}
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
	if g.bootstrap != nil {
		if err := c.useOwn(g.bootstrap.Own); err != nil {
			return err
		}
	}
	now := time.Now()
	state, err := openIdle(filepath.Join(layout.State, "idle.json"), rec.retained == nil && (g.bootstrap == nil || !g.bootstrap.Resume), now)
	if err != nil {
		return err
	}
	restartPath := filepath.Join(layout.State, "runtime-restarts.json")
	restartState, err := openRestarts(restartPath)
	if err != nil {
		// Unreadable bookkeeping blocks launches, never the authenticated repair API.
		restartState = &restarts{path: restartPath, blocked: true, safeCode: "runtime_restart_state_unavailable", detail: err.Error()}
	}
	m := &Machine{lease: lease, grant: g, layout: layout, id: id, claims: c, receipt: rec, hub: newHubClient(g), idle: state,
		restarts: restartState, log: log, started: now,
		inUpdate: true, // listener liveness is available while startup admission remains held
		owned:    g.Lifetime == "persistent", ready: make(chan struct{}), wake: make(chan struct{}, 1), initialized: make(chan struct{})}
	state.persistent = m.owned
	if m.owned && (g.bootstrap == nil || !g.bootstrap.Resume) {
		if err := state.reset(now); err != nil {
			return err
		}
	}
	m.openUpdates()
	m.tfs = &tensorFS{bin: layout.TFS, store: layout.Store, repoCache: g.RepoCacheRoot}
	if g.bootstrap != nil {
		m.childAddr = g.bootstrap.ChildAddr
	} else if m.childAddr, err = freeLoopback(); err != nil {
		return err
	}
	if err := lease.check(); err != nil {
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
	if m.owned && m.receipt.Envelope() == nil {
		identity, _ := json.Marshal(map[string]any{"kind": "machine_identity", "pod_boot_id": m.id.BootID,
			"tls_certificate_der_base64": base64.StdEncoding.EncodeToString(m.id.Leaf.Certificate[0])})
		if err := m.receipt.seal(identity, g.WorkerPort, g.ObservedAuth, m.webrtcReceipt()); err != nil {
			return err
		}
	}
	if m.conn, err = m.dialRuntime(); err != nil {
		return err
	}
	defer m.conn.Close()
	if err := m.prepareLauncher(); err != nil {
		return err
	}
	defer m.launcher.close()
	// The privileged guardian prepares software after the network agent drops root.
	startup, startupErr := m.prepareApplication()
	m.mu.Lock()
	g.startupPending = startup == "boot_pending"
	m.mu.Unlock()
	m.setUpdating(m.startupPending())
	if startupErr == nil && (g.bootstrap == nil || g.bootstrap.Incarnation == "") {
		if err := m.replaceApplication(startup == "rolled_back"); err != nil {
			if m.startupPending() {
				return err
			} // the stable parent restores the failed candidate
			startupErr = err
		}
	}
	if err := lease.check(); err != nil {
		return err
	}
	m.launchMu.Lock() // initial boot shares update/supervisor launch admission
	if g.bootstrap != nil && g.bootstrap.Incarnation != "" {
		g := m.launcher.(*guardianLauncher)
		g.mu.Lock()
		m.proc = g.current
		g.mu.Unlock()
		m.setUpdating(false)
		if m.proc != nil {
			if err := m.loadRuntimeCapabilities(); err != nil {
				fmt.Fprintln(m.log, "cozy machine: retained Runtime capabilities:", err)
			}
			m.proc.claimDone = make(chan struct{})
			go m.claimRuntime(m.proc)
		}
	} else if startupErr != nil {
		m.restarts.fail(&launchRefusal{"runtime_update_recovery_required", startupErr.Error()})
	} else if m.restarts.gone() && !m.startupPending() {
		fmt.Fprintln(m.log, "cozy machine: persisted Runtime failure requires repair:", m.restarts.reason())
	} else if err := m.tfs.initStore(ctx); err != nil {
		if m.startupPending() {
			if err := m.rollbackStartup(ctx, err); err != nil {
				fmt.Fprintln(m.log, "cozy machine: repair required:", err)
			}
		} else {
			m.restarts.fail(fmt.Errorf("TensorFS initialization failed: %w", err))
		}
	} else {
		if err := m.launch(); err != nil {
			if m.startupPending() && !m.manualUpdateActive() {
				if err := m.rollbackStartup(ctx, err); err != nil {
					fmt.Fprintln(m.log, "cozy machine: repair required:", err)
				}
			} else {
				m.restarts.fail(err)
			}
		}
	}
	m.launchMu.Unlock()
	close(m.initialized)
	errs := make(chan error, 1)
	go func() { errs <- m.supervise(ctx) }()
	select {
	case err := <-errs:
		if g.bootstrap == nil {
			m.stopRuntime()
		}
		return err
	case err := <-listeners.failed:
		if g.bootstrap == nil {
			m.stopRuntime()
		}
		return err
	}
}

func (m *Machine) launcherReady() bool {
	if m.initialized == nil {
		return true
	}
	select {
	case <-m.initialized:
		return true
	default:
		return false
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
		"COZY_RUNTIME_CONTROL_MODE=supervisor", "COZY_MACHINE_ROOT=" + l.Root, "COZY_WORKER_ID=" + g.WorkerID, "COZY_WORKER_INTERNAL_PORT=" + childPort,
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
	if m.grant.bootstrap != nil {
		return m.adoptBootstrap()
	}
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
		adoptOrphans()
		m.launcher = &directLauncher{path: m.layout.Runtime, root: m.layout.Root, env: env, out: m.words}
	}
	return nil
}

// launch is called under launchMu by the supervisor, or while an update owns
// inUpdate after acquiring that same lock in quiesce.
func (m *Machine) loadRuntimeCapabilities() error {
	raw, err := m.launcher.maintain("capabilities", nil)
	if err != nil {
		return err
	}
	var advertised struct {
		Capabilities []string `json:"capabilities"`
	}
	if err := json.Unmarshal([]byte(raw), &advertised); err != nil {
		return err
	}
	m.mu.Lock()
	m.runtimeCaps = advertised.Capabilities
	m.mu.Unlock()
	return nil
}

func (m *Machine) launch() error {
	if err := m.loadRuntimeCapabilities(); err != nil {
		return err
	}
	// The payload is this launch's readiness: an earlier Runtime's must not stand for it.
	if err := os.Remove(m.layout.boot("readiness-payload")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	random := make([]byte, 24)
	if _, err := rand.Read(random); err != nil {
		return err
	}
	p, err := m.launcher.launch(hex.EncodeToString(random))
	if err != nil {
		return err
	}
	p.claimDone = make(chan struct{})
	m.mu.Lock()
	m.proc, m.ready, m.runtimeAck = p, make(chan struct{}), nil
	m.mu.Unlock()
	go m.claimRuntime(p)
	return nil
}

func (m *Machine) supportsRuntime(capability string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Contains(m.runtimeCaps, capability)
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

// claimRuntime verifies this child's fresh readiness and opens its private session.
func (m *Machine) claimRuntime(p *runtimeProcess) {
	defer close(p.claimDone)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		select {
		case <-p.done:
			cancel()
		case <-ctx.Done():
		}
	}()
	payload := m.layout.boot("readiness-payload")
	var pendingPayload []byte
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		if raw, err := os.ReadFile(payload); err == nil && len(raw) > 0 {
			var ready struct {
				Incarnation string `json:"process_incarnation"`
				BootID      string `json:"pod_boot_id"`
				Leaf        string `json:"tls_certificate_der_base64"`
			}
			if json.Unmarshal(raw, &ready) != nil || ready.Incarnation != p.incarnation || ready.BootID != m.id.BootID || ready.Leaf != base64.StdEncoding.EncodeToString(m.id.Leaf.Certificate[0]) {
				fmt.Fprintln(m.log, "cozy machine: Runtime readiness does not name this process incarnation")
				m.restarts.fail(&launchRefusal{"runtime_readiness_invalid", "this process did not prove readiness"})
				p.stop()
				return
			}
			if m.startupPending() {
				pendingPayload = raw
			} else if err := m.attestRuntime(raw); err != nil {
				fmt.Fprintln(m.log, "cozy machine: the Runtime's readiness was refused:", err)
				m.restarts.fail(&launchRefusal{"runtime_readiness_invalid", err.Error()})
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
	ready := func(ack *pb.ClaimAck) error {
		m.mu.Lock()
		defer m.mu.Unlock()
		if m.proc == p {
			if err := m.restarts.clear(); err != nil {
				return err
			}
			m.runtimeAck = ack
			close(m.ready)
		}
		return nil
	}
	for {
		ack, err := m.controlClaim(ctx)
		if err == nil && ack.Accepted {
			if m.startupPending() {
				if err := m.commitStartup(p, pendingPayload); err != nil {
					m.restarts.fail(&launchRefusal{"startup_update_readiness_failed", err.Error()})
					p.stop()
					return
				}
			}
			if err := ready(ack); err != nil {
				m.restarts.fail(&launchRefusal{"runtime_restart_state_unavailable", err.Error()})
				p.stop()
				return
			}
			return
		}
		if err == nil && ack.Rejection != pb.ClaimRejection_CLAIM_REJECTION_UNDURABLE {
			fmt.Fprintln(m.log, "cozy machine: the Runtime refused this machine's Claim:", ack.Rejection)
			m.restarts.fail(&launchRefusal{"runtime_control_refused", ack.Rejection.String()})
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

func (m *Machine) attestRuntime(raw []byte) error {
	if m.owned {
		return nil
	} // its identity receipt is independent of Runtime availability
	return m.receipt.seal(raw, m.grant.WorkerPort, m.grant.ObservedAuth, m.webrtcReceipt())
}

// controlClaim records the daemon as the Runtime's owner on one Control stream.
func (m *Machine) controlClaim(parent context.Context) (*pb.ClaimAck, error) {
	ctx, cancel := context.WithCancel(parent)
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

// runtime waits for the resident coordinator, requesting a paced restart if needed.
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
				return nil, refusal(codes.FailedPrecondition, m.restarts.code(), m.runtimeGone())
			}
			if !m.restarts.due(time.Now()) {
				return nil, unavailable("runtime_restarting", m.runtimeGone())
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

// runtimeGone reports the retained launch failure without changing machine state.
func (m *Machine) runtimeGone() string {
	reason := m.restarts.reason()
	if m.launcherReady() {
		if words := m.words.last(); words != "" {
			reason += ": " + words
		}
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

// supervise is the machine's one loop: Runtime exits, activity, and the idle release.
func (m *Machine) supervise(ctx context.Context) error {
	work := &activity{path: m.layout.boot("worker-activity"), log: m.log}
	var nextAsk time.Time
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		if err := m.lease.check(); err != nil {
			return err
		}
		now := time.Now()
		m.mu.Lock()
		p := m.proc
		m.mu.Unlock()
		if p != nil && p.exited() && !(m.updating() && m.manualUpdateActive()) {
			select {
			case <-p.claimDone:
				// A resumed request plane has no manual update goroutine. Its
				// readiness result still decides whether this candidate needs rollback.
				if m.startupPending() && !m.manualUpdateActive() {
					if err := m.rollbackStartup(ctx, fmt.Errorf("candidate Runtime exited before readiness: %s; %s", exitText(p.err), m.restarts.reason())); err != nil {
						fmt.Fprintln(m.log, "cozy machine: repair required:", err)
					}
					continue
				}
				if err := m.relaunchAfter(p); err != nil {
					return err
				}
			default: // keep observing rental idleness while an in-flight commit settles
			}
		}
		busy, known := m.observeActivity(work, now)
		deadline, observeErr := m.idle.observe(now, busy, known)
		if observeErr != nil && !errors.Is(observeErr, errReleased) {
			return observeErr
		}
		if !m.owned && !now.Before(deadline) && !now.Before(nextAsk) {
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
			m.mu.Lock()
			stopped := m.proc == nil
			m.mu.Unlock()
			if stopped && !m.updating() && !m.idle.releasedNow() && m.restarts.due(time.Now()) {
				m.launchOrIdle()
			}
		case <-m.wake:
			m.mu.Lock()
			idle := m.proc == nil
			m.mu.Unlock()
			if idle && !m.updating() && m.restarts.due(time.Now()) && (m.owned || !m.idle.releasedNow()) {
				m.launchOrIdle()
			}
		}
	}
}

// A live process protects unknown activity. Maintenance without a Runtime is not
// accepted work and cannot renew a rental's idle deadline.
func (m *Machine) observeActivity(work *activity, now time.Time) (busy, known bool) {
	if !m.running() {
		return false, true
	}
	if m.preparations() {
		return true, true
	}
	if m.phase() != "ready" {
		return false, false
	}
	busy, err := work.observe(now)
	return busy, err == nil
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
	wasReady := m.runtimeAck != nil
	m.proc = nil
	updating := m.inUpdate
	m.mu.Unlock()
	if p.stopped.Load() || updating { // the update relaunches it itself
		return nil
	}
	if !m.restarts.exited(p.err, wasReady) {
		fmt.Fprintln(m.log, "cozy machine: the Runtime failed structurally; repair or update it before retrying:", exitText(p.err))
		return nil
	}
	fmt.Fprintln(m.log, "cozy machine: Runtime exited; restart scheduled:", exitText(p.err))
	return nil
}

// launchOrIdle records a launch failure while the API keeps serving repair and status calls.
func (m *Machine) launchOrIdle() {
	m.launchMu.Lock()
	defer m.launchMu.Unlock()
	// The supervisor's earlier observations are only a wake-up hint. An update
	// may have acquired the Runtime, or another launch may have finished, since.
	if m.updating() || m.running() || !m.restarts.due(time.Now()) || !m.owned && m.idle.releasedNow() {
		return
	}
	if err := m.launch(); err != nil {
		fmt.Fprintln(m.log, "cozy machine: the Runtime could not be launched:", err)
		m.restarts.fail(err)
	}
}

// release ends a rental allocation only. Persistent machines have no Hub lifecycle.
func (m *Machine) release(ctx context.Context) (bool, error) {
	if m.owned {
		return false, nil
	}
	m.mu.Lock()
	m.releasing = true
	m.mu.Unlock()
	m.stopRuntime()
	call, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	err := m.hub.release(call)
	if err == nil {
		fmt.Fprintln(m.log, "cozy machine: idle deadline passed; Tensorhub accepted the release")
	}
	return err == nil, err
}

func (m *Machine) phase() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	switch {
	case m.releasing:
		return "releasing"
	case m.restarts.gone():
		return "failed"
	case m.proc == nil || m.proc.exited():
		return "starting"
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
	if m.inUpdate {
		m.mu.Unlock()
		return nil
	}
	m.active++
	m.mu.Unlock()
	return func() {
		m.mu.Lock()
		m.active--
		m.mu.Unlock()
	}
}
