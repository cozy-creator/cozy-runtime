package host

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestActivationDrainsPreparationBeforeGuardedRestart(t *testing.T) {
	m, _, _ := updateMachine(t)
	done := m.beginPreparation()
	if done == nil {
		t.Fatal("initial preparation refused")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	observed := false
	err := m.quiesce(ctx, func() error {
		observed = true
		if next := m.beginPreparation(); next != nil {
			next()
			t.Fatal("preparation crossed activation fence")
		}
		if !m.preparations() {
			t.Fatal("active preparation was discarded")
		}
		done()
		cancel()
		return nil
	})
	if !observed || err != context.Canceled {
		t.Fatalf("drain observation: %v %v", observed, err)
	}
	if m.preparations() {
		t.Fatal("completed preparation retained")
	}
}

type legacyUpdateLauncher struct {
	updateLauncher
	guardian   *guardianLauncher
	stageError error
	prepared   *pairTransaction
	restarted  bool
}

func (l *legacyUpdateLauncher) maintain(op string, args []string) (string, error) {
	switch op {
	case "update-stage":
		if l.guardian != nil {
			return l.guardian.maintain(op, args)
		}
		return "", l.stageError
	case "restart":
		l.restarted = true
	case "update-prepare":
		l.prepared = &pairTransaction{}
		if err := json.Unmarshal([]byte(args[0]), l.prepared); err != nil {
			return "", err
		}
		if l.guardian != nil {
			return l.guardian.maintain(op, args)
		}
	}
	return l.updateLauncher.maintain(op, args)
}

func legacyFixtureGuardian(t *testing.T, path, root string) *guardianLauncher {
	t.Helper()
	controlRead, controlWrite, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	statusRead, statusWrite, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd := &exec.Cmd{Path: path, Args: []string{guardianProcessName, filepath.Join(root, "runtime"), root},
		ExtraFiles: []*os.File{controlRead, statusWrite}}
	if err := startChild(cmd); err != nil {
		t.Fatal(err)
	}
	controlRead.Close()
	statusWrite.Close()
	g := &guardianLauncher{control: controlWrite, cmd: cmd}
	go g.read(statusRead)
	t.Cleanup(g.close)
	return g
}

func TestUpdateFallsBackOnlyForUnsupportedGuardianStaging(t *testing.T) {
	for _, arm := range []string{"old_guardian", "resumed_old_guardian", "staging_failed", "published_guardian"} {
		t.Run(arm, func(t *testing.T) {
			oldAgent := os.Getenv("COZY_UPDATE_TEST_OLD_AGENT")
			if arm == "published_guardian" && oldAgent == "" {
				t.Skip("set COZY_UPDATE_TEST_OLD_AGENT to the published 0.18.88 agent")
			}
			m, _, _ := updateMachine(t)
			l := &legacyUpdateLauncher{stageError: errors.New(`unknown maintenance op "update-stage"`)}
			m.launcher = l
			if arm == "staging_failed" {
				l.stageError = errors.New("candidate digest mismatch")
			}
			if arm == "published_guardian" {
				l.guardian = legacyFixtureGuardian(t, oldAgent, m.layout.Root)
				// This is a real old guardian. Its legacy prepare route must be reached
				// with the complete request before this fixture refuses installation.
				executable(t, filepath.Join(m.layout.Root, "opt/cozy/python/bin/python"), `exit 42`)
			}
			for _, name := range []string{"cozy_runtime-0.18.86-py3-none-any.whl", "tensorfs-0.3.78-py3-none-any.whl"} {
				path := filepath.Join(m.layout.Root, name)
				if err := os.WriteFile(path, []byte(name), 0600); err != nil {
					t.Fatal(err)
				}
				l.pair = append(l.pair, path)
			}
			file := "cozy_runtime-0.18.87-py3-none-any.whl"
			digest, _, err := m.stageWheel(file, strings.NewReader("candidate"))
			if err != nil {
				t.Fatal(err)
			}
			request := updateRequest{Operation: "guardian-skew", Runtime: &wheelChoice{File: file, SHA256: digest}}
			status := updateStatus{Operation: request.Operation, State: "waiting", Request: &request, From: pairVersions{Runtime: "0.18.86", TensorFS: "0.3.78"}}
			m.updates.status, m.updates.active = &status, request.Operation
			if arm == "resumed_old_guardian" {
				// Legacy waiting is not the prepared-overlay state. After restarting
				// the request plane, the frozen request must still supply both pairs.
				status.To = pairVersions{Runtime: "0.18.87", TensorFS: "0.3.78"}
				if err := m.writeUpdate(status); err != nil {
					t.Fatal(err)
				}
				m.initialized = make(chan struct{})
				m.openUpdates()
				close(m.initialized)
				deadline := time.Now().Add(5 * time.Second)
				for {
					m.updates.mu.Lock()
					finished := m.updates.active == ""
					m.updates.mu.Unlock()
					if finished {
						break
					}
					if time.Now().After(deadline) {
						t.Fatal("legacy waiting update did not resume")
					}
					runtime.Gosched()
				}
			} else {
				m.runUpdate(request, status)
			}
			if arm == "staging_failed" {
				if l.prepared != nil || l.restarted || m.updates.status.Error != "candidate digest mismatch" {
					t.Fatal("real staging failure fell back to legacy activation")
				}
				return
			}
			if !l.restarted || l.prepared == nil || l.prepared.Operation != "" || len(l.prepared.Candidate) != 2 || len(l.prepared.Previous) != 2 {
				t.Fatalf("old guardian did not receive the guarded complete legacy pair: %+v", l.prepared)
			}
			if strings.Contains(m.updates.status.Error, "unknown maintenance op") {
				t.Fatal("old guardian could not handle legacy preparation", m.updates.status.Error)
			}
		})
	}
}

type retainedRecoveryLauncher struct {
	completionLauncher
	bootError         error
	launches, guarded int
}

func (l *retainedRecoveryLauncher) launch(incarnation string) (*runtimeProcess, error) {
	l.launches++
	if l.bootError != nil {
		return nil, l.bootError
	}
	p, err := l.completionLauncher.launch(incarnation)
	if err == nil {
		p.err = nil
	}
	return p, err
}

func (l *retainedRecoveryLauncher) maintain(op string, args []string) (string, error) {
	switch op {
	case "software-state":
		return `{"quiescent":false,"accepted_work":true,"workspace_id":"retained"}`, nil
	case "restart":
		l.machine.mu.Lock()
		ready := l.machine.runtimeAck != nil
		l.machine.mu.Unlock()
		if !ready {
			return "", errors.New("restart was asked before retained work recovered")
		}
		l.guarded++
		l.process.stop()
		return "accepted", nil
	}
	return l.completionLauncher.maintain(op, args)
}

func TestColdRetainedWorkRecoversInstalledRuntimeBeforeActivation(t *testing.T) {
	for _, failedBoot := range []bool{false, true} {
		t.Run(map[bool]string{false: "recovered", true: "startup_failed"}[failedBoot], func(t *testing.T) {
			m, _, _ := updateMachine(t)
			m.owned = true
			m.conn = completionConnection(t, &updateCompletionPeer{})
			m.id = &Identity{BootID: "boot", Leaf: tls.Certificate{Certificate: [][]byte{[]byte("leaf")}}}
			l := &retainedRecoveryLauncher{completionLauncher: completionLauncher{machine: m}}
			if failedBoot {
				l.bootError = errors.New("installed Runtime cannot start")
			}
			m.launcher = l
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			err := m.quiesce(ctx)
			if l.launches != 1 {
				t.Fatalf("retained custody did not recover the installed Runtime once: %d", l.launches)
			}
			if failedBoot {
				if err == nil || !strings.Contains(err.Error(), "cannot recover") || l.guarded != 0 {
					t.Fatalf("failed recovery authorized replacement: %v guards=%d", err, l.guarded)
				}
			} else if err != nil || l.guarded != 1 || !l.process.exited() {
				t.Fatalf("recovery did not use the live guard: %v guards=%d", err, l.guarded)
			}
		})
	}
}

type completionRetryLog struct {
	once   sync.Once
	failed chan struct{}
}

func (l *completionRetryLog) Write(p []byte) (int, error) {
	if strings.Contains(string(p), "cannot persist completed update") {
		l.once.Do(func() { close(l.failed) })
	}
	return len(p), nil
}

func TestCompletedUpdateRetriesPersistenceBeforeReleasingCandidate(t *testing.T) {
	m, _, _ := updateMachine(t)
	log := &completionRetryLog{failed: make(chan struct{})}
	m.log = log
	request := updateRequest{Operation: "retry-terminal", Runtime: &wheelChoice{Version: "0.18.87"}}
	status := updateStatus{Operation: request.Operation, State: "waiting", Request: &request}
	m.updates.status, m.updates.active = &status, request.Operation
	if err := m.writeUpdate(status); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(m.updatePath(), m.updatePath()+".accepted"); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(m.updatePath(), 0700); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(m.updatePath())
	if err := os.MkdirAll(m.stagePath(), 0700); err != nil {
		t.Fatal(err)
	}
	finished := make(chan struct{})
	go func() { m.runUpdate(request, status); close(finished) }()
	select {
	case <-log.failed:
	case <-time.After(5 * time.Second):
		t.Fatal("completion did not reach the failed persistence boundary")
	}
	m.updates.mu.Lock()
	owned, observed := m.updates.active == request.Operation, *m.updates.status
	m.updates.mu.Unlock()
	if !owned || observed.terminal() {
		t.Fatal("undurable completion released the operation")
	}
	if _, err := os.Stat(m.stagePath()); err != nil {
		t.Fatal("candidate was deleted before terminal persistence", err)
	}
	if err := os.Remove(m.updatePath()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("persistence recovery required restarting the request plane")
	}
	raw, err := os.ReadFile(m.updatePath())
	if err != nil {
		t.Fatal(err)
	}
	var terminal updateStatus
	if err := json.Unmarshal(raw, &terminal); err != nil || !terminal.terminal() || m.updates.active != "" {
		t.Fatalf("completion was not durable after retry: %+v %v", terminal, err)
	}
}

func TestPendingCandidateSurvivesRequestPlaneRestart(t *testing.T) {
	m, _, _ := updateMachine(t)
	request := updateRequest{Operation: "pending", Runtime: &wheelChoice{Version: "0.18.87"}}
	status := updateStatus{Operation: request.Operation, State: "waiting_activation", Request: &request,
		From: pairVersions{Runtime: "0.18.86", TensorFS: "0.3.78"}, To: pairVersions{Runtime: "0.18.87", TensorFS: "0.3.78"}}
	if err := m.writeUpdate(status); err != nil {
		t.Fatal(err)
	}
	m.initialized = make(chan struct{})
	m.openUpdates()
	if m.updates.status.State != "waiting_activation" || m.updates.active != request.Operation {
		t.Fatalf("lost prepared operation: %+v", m.updates)
	}
	raw, err := os.ReadFile(m.updatePath())
	if err != nil {
		t.Fatal(err)
	}
	var retained updateStatus
	if err := json.Unmarshal(raw, &retained); err != nil {
		t.Fatal(err)
	}
	if retained.State != status.State || retained.To != status.To {
		t.Fatalf("replaced pending selection: %+v", retained)
	}
	// Closing initialization resumes against a deliberately failing fixture.
	close(m.initialized)
	for {
		m.updates.mu.Lock()
		finished := m.updates.active == ""
		m.updates.mu.Unlock()
		if finished {
			break
		}
		runtime.Gosched()
	}
	if m.launcher.(*updateLauncher).installs.Load() != 1 {
		t.Fatal("resumed candidate tried to rediscover its previous or candidate wheels")
	}
}

func TestPreparedCandidateRejectsSameVersionChangedBytes(t *testing.T) {
	m, _, _ := updateMachine(t)
	prepared := &preparedPair{Before: pairVersions{Runtime: "0.18.86", TensorFS: "0.3.78"}, After: pairVersions{Runtime: "0.18.87", TensorFS: "0.3.78"}, CandidateHashes: map[string]string{}}
	for _, name := range []string{"cozy_runtime-0.18.87-py3-none-any.whl", "tensorfs-0.3.78-py3-none-any.whl"} {
		path := filepath.Join(m.layout.Root, name)
		if err := os.WriteFile(path, []byte("verified wheel"), 0600); err != nil {
			t.Fatal(err)
		}
		digest, err := hashFile(path)
		if err != nil {
			t.Fatal(err)
		}
		prepared.Candidate = append(prepared.Candidate, path)
		prepared.CandidateHashes[path] = digest
	}
	if err := os.WriteFile(prepared.Candidate[0], []byte("different wheel with identical version"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := m.activatePair(context.Background(), prepared, startupStatus{}, false); err == nil || !strings.Contains(err.Error(), "candidate wheel changed") {
		t.Fatalf("changed candidate reached activation: %v", err)
	}
	if m.transactionPending() {
		t.Fatal("rejected candidate started installation")
	}
}
