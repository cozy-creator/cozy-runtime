package host

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// The Runtime is `cozy-runtime-worker`, started with no argv and its launch contract in the
// environment: fd 3 is a pipe whose EOF means stop. It arms PDEATHSIG on the thread that
// started it, so that thread stays locked until it exits.

// runtimeProcess is one launched Runtime.
type runtimeProcess struct {
	incarnation      string
	done             chan struct{}
	claimDone        chan struct{} // the authenticated readiness attempt has settled
	startupCommitted bool          // private installation journal committed; read after claimDone
	err              error
	guardedStopped   atomic.Bool // the live Worker accepted restart and completed shutdown
	stopped          atomic.Bool // this daemon asked it to stop
	stop             func()      // cooperative: EOF on fd 3, then SIGKILL once it stops moving
	pid              int         // known where this process started it
}

func (p *runtimeProcess) exited() bool {
	select {
	case <-p.done:
		return true
	default:
		return false
	}
}

// launcher starts Runtimes and maintains them (maintenance.go) as the Runtime's own user. A
// root machine launches through a root guardian so this daemon can drop privilege; any other
// machine launches directly.
type launcher interface {
	launch(incarnation string) (*runtimeProcess, error)
	maintain(op string, args []string) (string, error)
	close()
}

// stallWindow is how long a stopping Runtime may show no CPU time and no output before it
// is killed (#871). A Runtime that is still moving is never killed.
const stallWindow = 5 * time.Second

// lastWords passes the Runtime's output through and keeps its last refusal line.
type lastWords struct {
	io.Writer
	mu   sync.Mutex
	line string
}

func (l *lastWords) Write(p []byte) (int, error) {
	for _, line := range strings.Split(string(p), "\n") {
		if strings.HasPrefix(line, "error(") {
			l.mu.Lock()
			l.line = strings.TrimSpace(line)
			l.mu.Unlock()
		}
	}
	return l.Writer.Write(p)
}

func (l *lastWords) last() string {
	if l == nil {
		return ""
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.line
}

type directLauncher struct {
	replacement     string // next application selected by a stopped-runtime handoff
	agentExecutable string // parent-held descriptor naming the running application ELF
	path, root      string // the Runtime entrypoint and the machine root
	env             []string
	out             io.Writer
	mu              sync.Mutex
	current         *runtimeProcess
}

func (d *directLauncher) close() {}

func (d *directLauncher) maintain(op string, args []string) (string, error) {
	if op == "public-read" || op == "software-state" {
		store := filepath.Join(d.root, "var/lib/tensorfs")
		for _, entry := range d.env {
			if value, ok := strings.CutPrefix(entry, "COZY_TENSORFS_ROOT="); ok && value != "" {
				store = value
			}
		}
		if op == "software-state" {
			m := &Machine{layout: NewLayout(&Grant{Root: d.root, StoreRoot: store})}
			var state softwareState
			if err := m.startupPython(context.Background(), &state, "inspect", store); err != nil {
				return "", err
			}
			raw, err := json.Marshal(state)
			return string(raw), err
		}
		return readPublic(d.root, store, args)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if op == "agent-replace" {
		if d.current != nil && !d.current.exited() {
			return "", errors.New("agent replacement requires a guarded stopped Runtime")
		}
		if len(args) < 1 {
			return "", errors.New("current application identity required")
		}
		m := &Machine{layout: NewLayout(&Grant{Root: d.root})}
		restored := len(args) == 2 && args[1] == "rollback"
		if !restored && !m.transactionPending() {
			return "", nil
		}
		selected := m.selectedApplication()
		if restored {
			selected = m.restoredApplication()
		}
		hash, err := hashFile(selected)
		if err != nil {
			return "", err
		}
		if hash == args[0] {
			return "", nil
		}
		if _, err := m.readAgentIdentity(context.Background(), selected); err != nil {
			if m.transactionPending() {
				return "", err
			}
			fmt.Fprintln(d.out, "cozy machine: installed application cannot attach to this bootstrap; retaining repair agent:", err)
			return "", nil
		}
		d.replacement = selected
		return "replace", nil
	}
	return maintainWithAgent(d.root, d.current, op, args, d.agentExecutable, d.out)
}

func (d *directLauncher) launch(incarnation string) (*runtimeProcess, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.current != nil && !d.current.exited() {
		return nil, errors.New("a Runtime process is already running")
	}
	stopRead, stopWrite, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	var output atomic.Uint64
	counted := writerFunc(func(p []byte) (int, error) { output.Add(uint64(len(p))); return d.out.Write(p) })
	cmd := exec.Command(d.path)
	cmd.Env, cmd.ExtraFiles, cmd.Stdout, cmd.Stderr = append(append([]string{}, d.env...), "COZY_RUNTIME_PROCESS_INCARNATION="+incarnation), []*os.File{stopRead}, counted, counted
	p := &runtimeProcess{done: make(chan struct{}), incarnation: incarnation}
	var once sync.Once
	closeStop := func() { once.Do(func() { _ = stopWrite.Close() }) }
	started := make(chan error, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		if err := startChild(cmd); err != nil {
			stopRead.Close()
			closeStop()
			started <- err
			return
		}
		pid := cmd.Process.Pid
		stopRead.Close()
		p.pid = pid
		p.stop = func() {
			p.stopped.Store(true)
			closeStop()
			go killWhenStill(p.done, func() uint64 { return cpuTicks(pid) + output.Load() }, func() { _ = cmd.Process.Kill() })
		}
		started <- nil
		p.err = waitChild(cmd)
		closeStop()
		close(p.done)
	}()
	if err := <-started; err != nil {
		return nil, fmt.Errorf("start the Runtime: %w", err)
	}
	d.current = p
	// Only a journaled, uncommitted candidate is disposable on measured stalls.
	// Accepted work is never monitored by this preparation-only guard.
	journal := filepath.Join(d.root, "var/lib/cozy/startup-update/status.json")
	var state startupStatus
	if raw, err := os.ReadFile(journal); err == nil && json.Unmarshal(raw, &state) == nil && state.State == "boot_pending" {
		go killWhenStill(p.done, func() uint64 { return processTreeTicks(p.pid) + output.Load() }, func() {
			d.mu.Lock()
			defer d.mu.Unlock()
			var current startupStatus
			if raw, err := os.ReadFile(journal); err == nil && json.Unmarshal(raw, &current) == nil && current.State == "boot_pending" {
				_ = cmd.Process.Kill()
			}
		})
	}
	return p, nil
}

func killWhenStill(done <-chan struct{}, progress func() uint64, kill func()) {
	last, since := progress(), time.Now()
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-done:
			return
		case now := <-tick.C:
			if current := progress(); current != last {
				last, since = current, now
			} else if now.Sub(since) >= stallWindow {
				kill()
				return
			}
		}
	}
}

// cpuTicks is the process's own and reaped children's CPU time.
func processTreeTicks(pid int) uint64 {
	total := cpuTicks(pid)
	raw, _ := os.ReadFile(fmt.Sprintf("/proc/%d/task/%d/children", pid, pid))
	for _, word := range strings.Fields(string(raw)) {
		if child, err := strconv.Atoi(word); err == nil {
			total += processTreeTicks(child)
		}
	}
	return total
}

func cpuTicks(pid int) uint64 {
	raw, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	end := bytes.LastIndexByte(raw, ')')
	if err != nil || end < 0 {
		return 0
	}
	fields := strings.Fields(string(raw[end+1:]))
	var total uint64
	for _, i := range []int{11, 12, 13, 14} {
		if i < len(fields) {
			n, _ := strconv.ParseUint(fields[i], 10, 64)
			total += n
		}
	}
	return total
}

type writerFunc func([]byte) (int, error)

func (f writerFunc) Write(p []byte) (int, error) { return f(p) }

// The guardian is a root child that outlives this daemon's privilege drop. It launches and
// stops Runtimes on command and reports each start and exit, one JSON line each.
const guardianProcessName = "cozy-machine-guardian"

// GuardianProcess reports whether argv0 names the guardian entrypoint.
func GuardianProcess(argv0 string) bool { return filepath.Base(argv0) == guardianProcessName }

type guardianCommand struct {
	Incarnation string   `json:"incarnation,omitempty"`
	Op          string   `json:"op"` // launch | stop | a maintenance op (maintenance.go)
	Args        []string `json:"args,omitempty"`
}

type guardianStatus struct {
	Incarnation string `json:"incarnation,omitempty"`
	Phase       string `json:"phase"` // started | exited | refused | maintained
	Error       string `json:"error,omitempty"`
	Answer      string `json:"answer,omitempty"` // a maintenance op's answer
	Code        string `json:"code,omitempty"`
	ExitCode    int    `json:"exit_code,omitempty"`
}

// A public read can return one full RPC payload encoded as base64 inside JSON.
// Commands also wrap an already-encoded public query in a JSON string.
const maxGuardianFrameBytes = 2 * maxMessageBytes

type guardianLauncher struct {
	control  *os.File
	mu       sync.Mutex
	current  *runtimeProcess
	started  chan error
	answered chan guardianStatus
	calls    sync.Mutex // one maintenance op at a time
	cmd      *exec.Cmd
	failed   error // the status stream has ended; no future command can be answered
}

func startGuardian(runtimePath, root string, env []string, out io.Writer) (*guardianLauncher, error) {
	self, err := guardianExecutable()
	if err != nil {
		return nil, err
	}
	controlRead, controlWrite, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	statusRead, statusWrite, err := os.Pipe()
	if err != nil {
		controlRead.Close()
		controlWrite.Close()
		return nil, err
	}
	cmd := &exec.Cmd{Path: self, Args: []string{guardianProcessName, runtimePath, root}, Env: env,
		ExtraFiles: []*os.File{controlRead, statusWrite}, Stdout: out, Stderr: out}
	if err := startChild(cmd); err != nil {
		controlRead.Close()
		controlWrite.Close()
		statusRead.Close()
		statusWrite.Close()
		return nil, fmt.Errorf("start the Runtime guardian: %w", err)
	}
	controlRead.Close()
	statusWrite.Close()
	g := &guardianLauncher{control: controlWrite, cmd: cmd}
	go g.read(statusRead)
	return g, nil
}

func (g *guardianLauncher) read(status io.ReadCloser) {
	scanner := bufio.NewScanner(status)
	scanner.Buffer(make([]byte, 64<<10), maxGuardianFrameBytes)
	for scanner.Scan() {
		var s guardianStatus
		if json.Unmarshal(scanner.Bytes(), &s) != nil {
			continue
		}
		g.mu.Lock()
		switch s.Phase {
		case "started":
			if g.current != nil && g.current.incarnation == s.Incarnation && g.started != nil {
				g.started <- nil
				g.started = nil
			}
		case "refused":
			if g.current != nil && g.current.incarnation == s.Incarnation && g.started != nil {
				g.current.err = errors.New(s.Error)
				g.started <- g.current.err
				g.started = nil
				close(g.current.done)
				g.current = nil
			}
		case "exited":
			if g.current != nil && g.current.incarnation == s.Incarnation {
				g.current.err = runtimeExit{detail: s.Error, code: s.ExitCode}
				close(g.current.done)
				g.current = nil
			}
		case "maintained":
			if g.answered != nil {
				g.answered <- s
				g.answered = nil
			}
		}
		g.mu.Unlock()
	}
	readError := errors.New("the Runtime guardian exited")
	if err := scanner.Err(); err != nil {
		readError = fmt.Errorf("the Runtime guardian status stream failed: %w", err)
	}
	status.Close()
	g.mu.Lock()
	g.failed = readError
	if g.answered != nil {
		g.answered <- guardianStatus{Error: readError.Error()}
		g.answered = nil
	}
	if g.started != nil {
		g.started <- readError
		g.started = nil
	}
	if g.current != nil {
		g.current.err = readError
		close(g.current.done)
		g.current = nil
	}
	g.mu.Unlock()
}

func (g *guardianLauncher) send(op, incarnation string) error {
	body, _ := json.Marshal(guardianCommand{Op: op, Incarnation: incarnation})
	_, err := g.control.Write(append(body, '\n'))
	return err
}

func (g *guardianLauncher) launch(incarnation string) (*runtimeProcess, error) {
	started := make(chan error, 1)
	p := &runtimeProcess{done: make(chan struct{}), incarnation: incarnation}
	p.stop = func() { p.stopped.Store(true); _ = g.send("stop", incarnation) }
	g.mu.Lock()
	if g.failed != nil {
		err := g.failed
		g.mu.Unlock()
		return nil, err
	}
	if g.current != nil && !g.current.exited() {
		g.mu.Unlock()
		return nil, errors.New("a Runtime process is already running or launching")
	}
	g.current, g.started = p, started
	g.mu.Unlock()
	if err := g.send("launch", incarnation); err != nil {
		g.mu.Lock()
		if g.current == p {
			p.err = err
			close(p.done)
			g.current, g.started = nil, nil
		}
		g.mu.Unlock()
		return nil, err
	}
	if err := <-started; err != nil {
		return nil, err
	}
	return p, nil
}

// maintain runs one maintenance op in the guardian, as root.
func (g *guardianLauncher) maintain(op string, args []string) (string, error) {
	g.calls.Lock()
	defer g.calls.Unlock()
	answered := make(chan guardianStatus, 1)
	g.mu.Lock()
	if g.failed != nil {
		err := g.failed
		g.mu.Unlock()
		return "", err
	}
	g.answered = answered
	g.mu.Unlock()
	body, _ := json.Marshal(guardianCommand{Op: op, Args: args})
	if _, err := g.control.Write(append(body, '\n')); err != nil {
		return "", err
	}
	s := <-answered
	if s.Error != "" {
		if s.Code != "" {
			return s.Answer, &launchRefusal{s.Code, s.Error}
		}
		return s.Answer, errors.New(s.Error)
	}
	return s.Answer, nil
}

func (g *guardianLauncher) close() {
	g.control.Close()
	if g.cmd != nil {
		_ = waitChild(g.cmd)
	}
}

type guardianProcesses struct {
	mu      sync.Mutex
	current *runtimeProcess
}

func (g *guardianProcesses) set(p *runtimeProcess) {
	g.mu.Lock()
	g.current = p
	g.mu.Unlock()
}

// stop selects a process before its returned action may be scheduled. An old
// stop command must never follow current into a replacement Runtime. Empty
// incarnation selects the current process for shutdown.
func (g *guardianProcesses) stop(incarnation string) func() {
	g.mu.Lock()
	p := g.current
	if p != nil && incarnation != "" && p.incarnation != incarnation {
		p = nil
	}
	g.mu.Unlock()
	return func() {
		if p != nil && !p.exited() {
			p.stop()
			<-p.done
		}
	}
}

// RunGuardian is the guardian process: it launches the Runtime at args[1], with its own
// environment, on "launch", stops it on "stop", runs a maintenance op on the machine rooted at
// args[2] as root, and stops the Runtime and exits when its control pipe closes. It adopts the
// orphans of the Runtime's process tree and reaps them.
func RunGuardian(args []string) int {
	if len(args) != 3 {
		return 2
	}
	syscall.CloseOnExec(3)
	syscall.CloseOnExec(4)
	control, status := os.NewFile(3, "control"), os.NewFile(4, "status")
	var reporting sync.Mutex
	report := func(s guardianStatus) {
		reporting.Lock()
		defer reporting.Unlock()
		body, _ := json.Marshal(s)
		_, _ = status.Write(append(body, '\n'))
	}
	direct := &directLauncher{path: args[1], root: args[2], env: os.Environ(), out: os.Stderr}
	adoptOrphans()
	processes := &guardianProcesses{}
	err := serveGuardianCommands(control, direct, processes, report)
	processes.stop("")()
	if err != nil {
		fmt.Fprintln(os.Stderr, "cozy machine: guardian control stream failed:", err)
		return 1
	}
	return 0
}

func serveGuardianCommands(control io.Reader, direct *directLauncher, processes *guardianProcesses, report func(guardianStatus)) error {
	var ops sync.WaitGroup
	defer ops.Wait()
	scanner := bufio.NewScanner(control)
	scanner.Buffer(make([]byte, 64<<10), maxGuardianFrameBytes)
	for scanner.Scan() {
		var c guardianCommand
		if json.Unmarshal(scanner.Bytes(), &c) != nil {
			continue
		}
		switch c.Op {
		case "launch":
			p, err := direct.launch(c.Incarnation)
			if err != nil {
				report(guardianStatus{Phase: "refused", Incarnation: c.Incarnation, Error: err.Error()})
				continue
			}
			processes.set(p)
			report(guardianStatus{Phase: "started", Incarnation: p.incarnation})
			go func() {
				<-p.done
				code := 0
				var exited interface{ ExitCode() int }
				if errors.As(p.err, &exited) {
					code = exited.ExitCode()
				}
				report(guardianStatus{Phase: "exited", Incarnation: p.incarnation, Error: exitText(p.err), ExitCode: code})
			}()
		case "stop":
			stop := processes.stop(c.Incarnation)
			go stop()
		default:
			ops.Add(1)
			go func(c guardianCommand) {
				defer ops.Done()
				answer, err := direct.maintain(c.Op, c.Args)
				s := guardianStatus{Phase: "maintained", Answer: answer}
				if err != nil {
					s.Error = err.Error()
					var refused *launchRefusal
					if errors.As(err, &refused) {
						s.Code = refused.code
					}
				}
				report(s)
			}(c)
		}
	}
	return scanner.Err()
}

type runtimeExit struct {
	detail string
	code   int
}

func (e runtimeExit) Error() string { return e.detail }
func (e runtimeExit) ExitCode() int { return e.code }

func exitText(err error) string {
	if err == nil {
		return "exit status 0"
	}
	return err.Error()
}

// dropPrivilege hands the listed paths to the machine uid and becomes it: a root machine's
// request plane runs unprivileged while its guardian keeps root for the Runtime.
const machineUID = 65532

func dropPrivilege(paths ...string) error {
	for _, path := range paths {
		if err := filepath.Walk(path, func(p string, _ os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			return os.Lchown(p, machineUID, machineUID)
		}); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("hand %s to the machine uid: %w", path, err)
		}
	}
	if err := syscall.Setgroups(nil); err != nil {
		return err
	}
	if err := syscall.Setgid(machineUID); err != nil {
		return err
	}
	if err := syscall.Setuid(machineUID); err != nil {
		return err
	}
	if os.Geteuid() != machineUID {
		return errors.New("the privilege drop did not hold")
	}
	return nil
}
