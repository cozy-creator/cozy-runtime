package host

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
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
