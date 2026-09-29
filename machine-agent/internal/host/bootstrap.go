package host

import (
	"context"
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/build"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// The bootstrap ABI belongs to the image/service, not the replaceable application
// agent. Its parent process owns the Runtime and survives a request-plane crash.
const bootstrapABI = "machine-bootstrap/1"
const requestPlaneName = "cozy-machine-request-plane"
const replaceAgentExit = 75
const releasedAgentExit = 76

type bootstrapChild struct {
	BootstrapVersion     string
	Selection            string
	RolledBack           bool
	CommittedPair        *pairVersions
	ABI                  string
	Grant                *Grant
	ReceiptKey           []byte
	Inherited            []string
	Own                  ed25519.PrivateKey
	ChildAddr, MediaAddr string
	Incarnation          string
	Resume, Pending      bool
	RecoveryError        string
}

func RequestPlaneProcess(name string) bool { return filepath.Base(name) == requestPlaneName }

func RunRequestPlane(log io.Writer) int {
	input := os.NewFile(5, "bootstrap-contract")
	var state bootstrapChild
	err := json.NewDecoder(io.LimitReader(input, 1<<20)).Decode(&state)
	input.Close()
	if err != nil || state.ABI != bootstrapABI || state.Grant == nil || len(state.Own) != ed25519.PrivateKeySize {
		fmt.Fprintln(log, "cozy machine: invalid private bootstrap contract", err)
		return 2
	}
	for _, fd := range []int{3, 4} {
		syscall.CloseOnExec(fd)
	}
	state.Grant.receiptKey, state.Grant.inherited, state.Grant.bootstrap = state.ReceiptKey, state.Inherited, &state
	return runGranted(state.Grant, log)
}

// RunBootstrap uses the existing guardian as the stable parent. Only this parent
// owns Runtime stop control; losing the public child never means cancel work.
func RunBootstrap(ctx context.Context, grant *Grant, log io.Writer) error {
	layout := NewLayout(grant)
	lease, err := lockMachine(layout.State)
	if err != nil {
		return err
	}
	defer lease.close()
	if err = lease.watchRoot(layout.Root); err != nil {
		return err
	}
	id, err := prepare(grant, layout)
	if err != nil {
		return err
	}
	claims, err := newClaims(grant.WorkerID, id.BootID, id.Digest, grant.Authorized)
	if err != nil {
		return err
	}
	childAddr, err := freeLoopback()
	if err != nil {
		return err
	}
	mediaAddr := net.JoinHostPort("127.0.0.1", strconv.Itoa(grant.MediaPort))
	if grant.MediaPort == 0 {
		mediaAddr, err = freeLoopback()
		if err != nil {
			return err
		}
	}
	m := &Machine{grant: grant, layout: layout, id: id, claims: claims, childAddr: childAddr, mediaAddr: mediaAddr, log: log}
	if os.Geteuid() == 0 {
		if grant.Development && grant.DeveloperKey != "" {
			if err = startSSH(grant.DeveloperKey, log); err != nil {
				return err
			}
		}
		if err = os.MkdirAll(layout.Store, 0755); err != nil {
			return err
		}
		if err = os.Lchown(layout.Store, machineUID, machineUID); err != nil {
			return err
		}
		for _, path := range []string{layout.Bootstrap, layout.State, filepath.Join(layout.Installs, ".stage")} {
			if err = filepath.Walk(path, func(p string, _ os.FileInfo, e error) error {
				if e != nil {
					return e
				}
				return os.Lchown(p, machineUID, machineUID)
			}); err != nil {
				return err
			}
		}
	}
	adoptOrphans()
	direct := &directLauncher{path: layout.Runtime, root: layout.Root, env: m.runtimeEnvironment(), out: log}
	processes := &guardianProcesses{}
	defer func() { processes.stop("")() }()
	// The reporter can detach from a dead child without closing Runtime control.
	var reportMu sync.Mutex
	var destination io.Writer
	report := func(s guardianStatus) {
		reportMu.Lock()
		defer reportMu.Unlock()
		if destination != nil {
			body, _ := json.Marshal(s)
			_, _ = destination.Write(append(body, '\n'))
		}
	}
	state := bootstrapChild{ABI: bootstrapABI, BootstrapVersion: build.Version, Selection: m.transactionAgent(), Grant: grant, ReceiptKey: grant.receiptKey, Inherited: grant.inherited, Own: claims.own, ChildAddr: childAddr, MediaAddr: mediaAddr}
	self, err := guardianExecutable()
	if err != nil {
		return err
	}
	executable := self
	watchOwnership := time.NewTicker(250 * time.Millisecond)
	defer watchOwnership.Stop()
	for {
		if err = lease.check(); err != nil {
			return err
		}
		direct.mu.Lock()
		current := direct.current
		direct.mu.Unlock()
		state.Incarnation = ""
		if current != nil && !current.exited() {
			state.Incarnation = current.incarnation
		}
		controlRead, controlWrite, e := os.Pipe()
		if e != nil {
			return e
		}
		statusRead, statusWrite, e := os.Pipe()
		if e != nil {
			controlRead.Close()
			controlWrite.Close()
			return e
		}
		contractRead, contractWrite, e := os.Pipe()
		if e != nil {
			controlRead.Close()
			controlWrite.Close()
			statusRead.Close()
			statusWrite.Close()
			return e
		}
		application, openErr := os.Open(executable)
		if openErr != nil {
			controlRead.Close()
			controlWrite.Close()
			statusRead.Close()
			statusWrite.Close()
			contractRead.Close()
			contractWrite.Close()
			return openErr
		}
		direct.agentExecutable = fmt.Sprintf("/proc/self/fd/%d", application.Fd())
		var output atomic.Uint64
		counted := writerFunc(func(p []byte) (int, error) { output.Add(uint64(len(p))); return log.Write(p) })
		cmd := &exec.Cmd{Path: executable, Args: []string{requestPlaneName}, ExtraFiles: []*os.File{controlWrite, statusRead, contractRead}, Stdout: counted, Stderr: counted}
		requestPlaneCredential(cmd)
		if e = startChild(cmd); e != nil {
			controlRead.Close()
			controlWrite.Close()
			statusRead.Close()
			statusWrite.Close()
			contractRead.Close()
			contractWrite.Close()
			application.Close()
			if m.transactionPending() {
				recoverCandidateApplication(m, direct, processes, &state, e, log)
				executable = self
				if state.RecoveryError == "" {
					executable = m.restoredApplication()
				}
				state.Resume = true
				continue
			}

			return e
		}
		controlWrite.Close()
		statusRead.Close()
		contractRead.Close()
		go func(snapshot bootstrapChild) {
			_ = json.NewEncoder(contractWrite).Encode(snapshot)
			contractWrite.Close()
		}(state)
		reportMu.Lock()
		destination = statusWrite
		reportMu.Unlock()
		served := make(chan error, 1)
		go func() { served <- serveGuardianCommands(controlRead, direct, processes, report) }()
		if current != nil && current.exited() && state.Incarnation != "" {
			report(guardianStatus{Phase: "exited", Incarnation: state.Incarnation, Error: exitText(current.err)})
		}
		done := make(chan error, 1)
		exitedSignal := make(chan struct{})
		go func() { result := waitChild(cmd); close(exitedSignal); done <- result }()
		if state.Pending {
			go killWhenStill(exitedSignal, func() uint64 {
				direct.mu.Lock()
				p := direct.current
				direct.mu.Unlock()
				ticks := processTreeTicks(cmd.Process.Pid) + output.Load()
				if p != nil && !p.exited() {
					ticks += processTreeTicks(p.pid)
				}
				return ticks
			}, func() {
				direct.mu.Lock()
				defer direct.mu.Unlock()
				if m.transactionPending() {
					_ = cmd.Process.Kill()
				}
			})
		}
		var childErr error
		waiting := true
		for waiting {
			select {
			case childErr = <-done:
				waiting = false
			case <-ctx.Done():
				_ = cmd.Process.Kill()
				<-done
				controlRead.Close()
				<-served
				statusWrite.Close()
				application.Close()
				return ctx.Err()
			case <-watchOwnership.C:
				if err = lease.check(); err != nil {
					_ = cmd.Process.Kill()
					<-done
					controlRead.Close()
					<-served
					statusWrite.Close()
					application.Close()
					return err
				}
			}
		}
		controlRead.Close()
		<-served
		application.Close()
		reportMu.Lock()
		destination = nil
		statusWrite.Close()
		reportMu.Unlock()
		var exited *exec.ExitError
		intentional := errors.As(childErr, &exited) && exited.ExitCode() == replaceAgentExit
		state.Resume = true
		state.Pending = m.transactionPending()
		var transaction startupStatus
		if raw, err := os.ReadFile(m.startupPath("status.json")); err == nil && json.Unmarshal(raw, &transaction) == nil && transaction.State == "succeeded" {
			pair := transaction.After
			state.CommittedPair = &pair
		} else {
			state.CommittedPair = nil
		}
		if !state.Pending && errors.As(childErr, &exited) && exited.ExitCode() == releasedAgentExit {
			return nil
		}
		if intentional {
			state.RecoveryError = ""
			state.RolledBack = false
		}
		if !intentional && state.Pending {
			recoverCandidateApplication(m, direct, processes, &state, childErr, log)
		} else if !intentional {
			direct.mu.Lock()
			live := direct.current != nil && !direct.current.exited()
			direct.mu.Unlock()
			if !live {
				return fmt.Errorf("request plane exited outside a candidate transaction: %v", childErr)
			}
		}

		executable = self
		state.Selection = m.transactionAgent()
		if state.RecoveryError == "" {
			executable = m.selectedApplication()
			if state.RolledBack {
				executable = m.restoredApplication()
			}
			if intentional && direct.replacement != "" {
				executable = direct.replacement
			}
		}
	}
}

func recoverCandidateApplication(m *Machine, direct *directLauncher, processes *guardianProcesses, state *bootstrapChild, cause error, log io.Writer) {
	processes.stop("")()
	_, err := direct.maintain("startup-rollback", []string{fmt.Sprintf("candidate request plane exited before readiness: %v", cause)})
	state.Pending = false
	state.RolledBack = err == nil
	if err != nil {
		state.RecoveryError = fmt.Sprintf("candidate agent exited and rollback failed: %v", err)
		fmt.Fprintln(log, "cozy machine:", state.RecoveryError)
	} else {
		fmt.Fprintln(log, "cozy machine: restored previous application and pair after candidate exit:", cause)
	}
}
