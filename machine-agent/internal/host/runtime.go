package host

import (
	"bufio"
	"bytes"
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
	done    chan struct{}
	err     error
	stopped atomic.Bool // this daemon asked it to stop
	stop    func()      // cooperative: EOF on fd 3, then SIGKILL once it stops moving
	pid     int         // known where this process started it
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
	launch() (*runtimeProcess, error)
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
	path, root string // the Runtime entrypoint and the machine root
	env        []string
	out        io.Writer
	mu         sync.Mutex
	current    *runtimeProcess
}

func (d *directLauncher) close() {}

func (d *directLauncher) maintain(op string, args []string) (string, error) {
	d.mu.Lock()
	p := d.current
	d.mu.Unlock()
	return maintain(d.root, p, op, args)
}

func (d *directLauncher) launch() (*runtimeProcess, error) {
	stopRead, stopWrite, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	var output atomic.Uint64
	counted := writerFunc(func(p []byte) (int, error) { output.Add(uint64(len(p))); return d.out.Write(p) })
	cmd := exec.Command(d.path)
	cmd.Env, cmd.ExtraFiles, cmd.Stdout, cmd.Stderr = d.env, []*os.File{stopRead}, counted, counted
	p := &runtimeProcess{done: make(chan struct{})}
	var once sync.Once
	closeStop := func() { once.Do(func() { _ = stopWrite.Close() }) }
	started := make(chan error, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		spawnMu.Lock()
		if err := cmd.Start(); err != nil {
			spawnMu.Unlock()
			stopRead.Close()
			closeStop()
			started <- err
			return
		}
		pid := cmd.Process.Pid
		spawned[pid] = true
		spawnMu.Unlock()
		stopRead.Close()
		p.pid = pid
		p.stop = func() {
			p.stopped.Store(true)
			closeStop()
			go killWhenStill(p.done, func() uint64 { return cpuTicks(pid) + output.Load() }, func() { _ = cmd.Process.Kill() })
		}
		started <- nil
		p.err = cmd.Wait()
		spawnMu.Lock()
		delete(spawned, pid)
		spawnMu.Unlock()
		reapOrphans()
		closeStop()
		close(p.done)
	}()
	if err := <-started; err != nil {
		return nil, fmt.Errorf("start the Runtime: %w", err)
	}
	d.mu.Lock()
	d.current = p
	d.mu.Unlock()
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
	Op   string   `json:"op"` // launch | stop | a maintenance op (maintenance.go)
	Args []string `json:"args,omitempty"`
}

type guardianStatus struct {
	Phase  string `json:"phase"` // started | exited | refused | maintained
	Error  string `json:"error,omitempty"`
	Answer string `json:"answer,omitempty"` // a maintenance op's answer
}

type guardianLauncher struct {
	control  *os.File
	mu       sync.Mutex
	current  *runtimeProcess
	started  chan error
	answered chan guardianStatus
	calls    sync.Mutex // one maintenance op at a time
	cmd      *exec.Cmd
}

func startGuardian(runtimePath, root string, env []string, out io.Writer) (*guardianLauncher, error) {
	self, err := os.Executable()
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
	if err := cmd.Start(); err != nil {
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
	for scanner.Scan() {
		var s guardianStatus
		if json.Unmarshal(scanner.Bytes(), &s) != nil {
			continue
		}
		g.mu.Lock()
		switch s.Phase {
		case "started":
			if g.started != nil {
				g.started <- nil
				g.started = nil
			}
		case "refused":
			if g.started != nil {
				g.started <- errors.New(s.Error)
				g.started = nil
			}
		case "exited":
			if g.current != nil {
				g.current.err = errors.New(s.Error)
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
	status.Close()
	g.mu.Lock()
	if g.answered != nil {
		g.answered <- guardianStatus{Error: "the Runtime guardian exited"}
		g.answered = nil
	}
	if g.started != nil {
		g.started <- errors.New("the Runtime guardian exited")
		g.started = nil
	}
	if g.current != nil {
		g.current.err = errors.New("the Runtime guardian exited")
		close(g.current.done)
		g.current = nil
	}
	g.mu.Unlock()
}

func (g *guardianLauncher) send(op string) error {
	body, _ := json.Marshal(guardianCommand{Op: op})
	_, err := g.control.Write(append(body, '\n'))
	return err
}

func (g *guardianLauncher) launch() (*runtimeProcess, error) {
	started := make(chan error, 1)
	p := &runtimeProcess{done: make(chan struct{})}
	p.stop = func() { p.stopped.Store(true); _ = g.send("stop") }
	g.mu.Lock()
	g.current, g.started = p, started
	g.mu.Unlock()
	if err := g.send("launch"); err != nil {
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
	g.answered = answered
	g.mu.Unlock()
	body, _ := json.Marshal(guardianCommand{Op: op, Args: args})
	if _, err := g.control.Write(append(body, '\n')); err != nil {
		return "", err
	}
	s := <-answered
	if s.Error != "" {
		return s.Answer, errors.New(s.Error)
	}
	return s.Answer, nil
}

func (g *guardianLauncher) close() {
	g.control.Close()
	_ = g.cmd.Wait()
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
	report := func(s guardianStatus) { body, _ := json.Marshal(s); _, _ = status.Write(append(body, '\n')) }
	direct := &directLauncher{path: args[1], root: args[2], out: os.Stderr}
	adoptOrphans()
	var mu sync.Mutex
	var current *runtimeProcess
	stop := func() {
		mu.Lock()
		p := current
		mu.Unlock()
		if p != nil && !p.exited() {
			p.stop()
			<-p.done
		}
	}
	scanner := bufio.NewScanner(control)
	for scanner.Scan() {
		var c guardianCommand
		if json.Unmarshal(scanner.Bytes(), &c) != nil {
			continue
		}
		switch c.Op {
		case "launch":
			stop()
			p, err := direct.launch()
			if err != nil {
				report(guardianStatus{Phase: "refused", Error: err.Error()})
				continue
			}
			mu.Lock()
			current = p
			mu.Unlock()
			report(guardianStatus{Phase: "started"})
			go func() {
				<-p.done
				report(guardianStatus{Phase: "exited", Error: exitText(p.err)})
			}()
		case "stop":
			go stop()
		default:
			go func(c guardianCommand) {
				answer, err := direct.maintain(c.Op, c.Args)
				s := guardianStatus{Phase: "maintained", Answer: answer}
				if err != nil {
					s.Error = err.Error()
				}
				report(s)
			}(c)
		}
	}
	stop()
	return 0
}

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
