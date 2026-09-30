package host

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func guardianFixture(t *testing.T) (*guardianLauncher, string, <-chan struct{}) {
	t.Helper()
	root := t.TempDir()
	executable(t, filepath.Join(root, "runtime"), "exec cat <&3 >/dev/null")
	controlRead, controlWrite, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	statusRead, statusWrite, err := os.Pipe()
	if err != nil {
		controlRead.Close()
		controlWrite.Close()
		t.Fatal(err)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	cmd := exec.CommandContext(ctx, self, "-test.run=^TestGuardianBoundaryHelper$")
	cmd.Env = append(os.Environ(), "COZY_GUARDIAN_BOUNDARY_HELPER=1", "COZY_GUARDIAN_BOUNDARY_ROOT="+root)
	cmd.ExtraFiles = []*os.File{controlRead, statusWrite}
	if err := startChild(cmd); err != nil {
		cancel()
		controlRead.Close()
		controlWrite.Close()
		statusRead.Close()
		statusWrite.Close()
		t.Fatal(err)
	}
	controlRead.Close()
	statusWrite.Close()
	g := &guardianLauncher{control: controlWrite, cmd: cmd}
	readDone := make(chan struct{})
	go func() { g.read(statusRead); close(readDone) }()
	t.Cleanup(func() {
		defer cancel()
		controlWrite.Close()
		if err := waitChild(cmd); err != nil {
			t.Errorf("fixture guardian exit: %v", err)
		}
		<-readDone
	})
	return g, root, readDone
}

func TestGuardianConcurrentLaunchRefusesWithoutReplacing(t *testing.T) {
	g, _, _ := guardianFixture(t)
	type result struct {
		process *runtimeProcess
		err     error
	}
	results := make(chan result, 2)
	start := make(chan struct{})
	for _, incarnation := range []string{"normal-relaunch", "update-replacement"} {
		go func() {
			<-start
			p, err := g.launch(incarnation)
			results <- result{p, err}
		}()
	}
	close(start)
	var running *runtimeProcess
	refused := 0
	for range 2 {
		select {
		case r := <-results:
			if r.err != nil {
				refused++
			} else if running != nil {
				t.Fatal("two concurrent Runtime launches succeeded")
			} else {
				running = r.process
			}
		case <-time.After(2 * time.Second):
			t.Fatal("concurrent guardian launch lost its reply")
		}
	}
	if refused != 1 || running == nil || running.exited() {
		t.Fatalf("expected one live child and one refusal, got %d refusals", refused)
	}
	// The child-side boundary must also refuse a malformed caller that bypasses
	// guardianLauncher's reservation, rather than stopping the healthy child.
	if err := g.send("launch", "unreserved-replacement"); err != nil {
		t.Fatal(err)
	}
	if _, err := g.maintain("pair", nil); err != nil {
		t.Fatal(err)
	}
	if running.exited() {
		t.Fatal("a duplicate guardian command stopped the active Runtime")
	}
	running.stop()
	<-running.done
}

func TestGuardianLaunchRecoversAfterRefusalAndClosesAfterEOF(t *testing.T) {
	g, root, readDone := guardianFixture(t)
	if err := os.Remove(filepath.Join(root, "runtime")); err != nil {
		t.Fatal(err)
	}
	if _, err := g.launch("missing-executable"); err == nil {
		t.Fatal("missing Runtime executable was accepted")
	}
	executable(t, filepath.Join(root, "runtime"), "exec cat <&3 >/dev/null")
	p, err := g.launch("repaired-executable")
	if err != nil {
		t.Fatalf("refused launch retained the reservation: %v", err)
	}
	p.stop()
	<-p.done
	g.control.Close()
	<-readDone
	if _, err := g.launch("after-guardian-exit"); err == nil {
		t.Fatal("launch accepted after guardian status channel ended")
	}
	if _, err := g.maintain("pair", nil); err == nil {
		t.Fatal("maintenance accepted after guardian status channel ended")
	}
}

func TestDirectConcurrentLaunchKeepsOneRuntime(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "runtime")
	executable(t, path, "exec cat <&3 >/dev/null")
	d := &directLauncher{path: path, root: root, env: os.Environ(), out: io.Discard}
	processes := make(chan *runtimeProcess, 2)
	start := make(chan struct{})
	for _, incarnation := range []string{"first", "second"} {
		go func() {
			<-start
			p, _ := d.launch(incarnation)
			processes <- p
		}()
	}
	close(start)
	accepted := 0
	for range 2 {
		p := <-processes
		if p != nil {
			accepted++
			t.Cleanup(func() { p.stop(); <-p.done })
		}
	}
	if accepted != 1 {
		t.Fatalf("direct concurrent launches started %d Runtimes", accepted)
	}
}

type launchBarrier struct {
	probeEntered chan struct{}
	probeRelease chan struct{}
	restarted    chan bool
	probes       atomic.Int32
	mu           sync.Mutex
	process      *runtimeProcess
}

func (l *launchBarrier) launch(incarnation string) (*runtimeProcess, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	p := &runtimeProcess{incarnation: incarnation, done: make(chan struct{})}
	var stopped sync.Once
	p.stop = func() { stopped.Do(func() { close(p.done) }) }
	l.process = p
	return p, nil
}
func (*launchBarrier) close() {}
func (l *launchBarrier) maintain(op string, _ []string) (string, error) {
	if op == "capabilities" {
		if l.probes.Add(1) == 1 {
			close(l.probeEntered)
		}
		<-l.probeRelease
		return `{"capabilities":["machine-supervisor/1"]}`, nil
	}
	l.mu.Lock()
	p := l.process
	l.mu.Unlock()
	l.restarted <- p != nil
	if p != nil {
		p.stop()
	}
	return "accepted", nil
}

func TestGuardedUpdateWaitsForNormalLaunch(t *testing.T) {
	l := &launchBarrier{probeEntered: make(chan struct{}), probeRelease: make(chan struct{}), restarted: make(chan bool, 1)}
	m := &Machine{owned: true, layout: NewLayout(&Grant{Root: t.TempDir()}), launcher: l,
		restarts: &restarts{}, log: io.Discard}
	launched, quiesced := make(chan struct{}), make(chan error, 1)
	go func() { m.launchOrIdle(); close(launched) }()
	<-l.probeEntered
	var released sync.Once
	t.Cleanup(func() {
		released.Do(func() { close(l.probeRelease) })
		<-launched
		m.stopRuntime()
	})
	go func() { quiesced <- m.quiesce(context.Background()) }()
	select {
	case err := <-quiesced:
		t.Fatalf("update acquired Runtime before in-flight launch settled: %v", err)
	case <-time.After(100 * time.Millisecond):
	}
	released.Do(func() { close(l.probeRelease) })
	<-launched
	select {
	case <-l.restarted:
		t.Fatal("guarded restart ran before Runtime readiness")
	case <-time.After(100 * time.Millisecond):
	}
	m.mu.Lock()
	close(m.ready)
	m.mu.Unlock()
	if err := <-quiesced; err != nil {
		t.Fatal(err)
	}
	if !<-l.restarted {
		t.Fatal("guarded restart missed the normally launched process")
	}
	m.launchOrIdle()
	if l.probes.Load() != 1 {
		t.Fatal("normal launch entered an update-owned Runtime environment")
	}
}

func TestBootCrashBackoffReturnsVisibleUnavailable(t *testing.T) {
	m := &Machine{owned: true, idle: &idle{}, restarts: &restarts{}, wake: make(chan struct{}, 1)}
	m.restarts.exited(runtimeExit{detail: "boot configuration failed", code: 1}, false)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan error, 1)
	go func() { _, err := m.runtime(ctx); result <- err }()
	select {
	case err := <-result:
		if status.Code(err) != codes.Unavailable || !strings.Contains(err.Error(), "runtime_restarting") || !strings.Contains(err.Error(), "boot configuration failed") {
			t.Fatalf("boot fault lost its visible recovery reason: %v", err)
		}
	case <-time.After(500 * time.Millisecond):
		cancel()
		<-result
		t.Fatal("boot crash held dispatch through its retry backoff")
	}
	if m.restarts.gone() {
		t.Fatal("reporting backoff made a transient crash permanent")
	}
}
