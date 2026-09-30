package host

import (
	"errors"
	"io"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestPreReadyCrashLoopPersistsUntilRepair(t *testing.T) {
	path := filepath.Join(t.TempDir(), "restarts.json")
	r, err := openRestarts(path)
	if err != nil {
		t.Fatal(err)
	}
	if !r.exited(errors.New("first boot failure"), false) {
		t.Fatal("first failure should retry")
	}
	r, err = openRestarts(path)
	if err != nil {
		t.Fatal(err)
	}
	if r.exited(errors.New("second boot failure"), false) || !r.gone() {
		t.Fatal("second unready exit did not block")
	}
	r, err = openRestarts(path)
	if err != nil {
		t.Fatal(err)
	}
	if !r.gone() || r.due(time.Now().Add(24*time.Hour)) {
		t.Fatal("restart or elapsed time cleared failure")
	}
	if err := r.clear(); err != nil {
		t.Fatal(err)
	}
	r, err = openRestarts(path)
	if err != nil {
		t.Fatal(err)
	}
	if r.gone() || !r.exited(errors.New("first failure after repair"), false) {
		t.Fatal("repair did not reset readiness streak")
	}
}

func TestUsedRentalDeadRuntimeDoesNotRenewDeadline(t *testing.T) {
	now := time.Now()
	i, err := openIdle(filepath.Join(t.TempDir(), "idle.json"), true, now)
	if err != nil {
		t.Fatal(err)
	}
	if err := i.work(now); err != nil {
		t.Fatal(err)
	}
	_, err = i.observe(now, false, true)
	if err != nil {
		t.Fatal(err)
	}
	unknownDeadline, err := i.observe(now, false, false)
	if err != nil {
		t.Fatal(err)
	}
	for n := 1; n <= 3; n++ {
		next, err := i.observe(unknownDeadline.Add(time.Duration(n)*time.Second), false, false)
		if err != nil {
			t.Fatal(err)
		}
		unknownDeadline = next
	}
	knownDead, err := i.observe(unknownDeadline.Add(idleGrace), false, true)
	if err != nil || !knownDead.Equal(unknownDeadline) {
		t.Fatal("dead Runtime renewed deadline", knownDead, err)
	}
	ok, err := i.claim(knownDead.Add(time.Second))
	if err != nil || !ok {
		t.Fatal("used dead rental cannot release", ok, err)
	}
}

func TestLiveUnreadyProcessProtectsUsedRentalUntilActualExit(t *testing.T) {
	cmd := exec.Command("/bin/sh", "-c", "read line; exit 1")
	input, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := startChild(cmd); err != nil {
		t.Fatal(err)
	}
	p := &runtimeProcess{pid: cmd.Process.Pid, done: make(chan struct{})}
	go func() { p.err = waitChild(cmd); close(p.done) }()
	defer func() { _ = input.Close(); <-p.done }()
	m := &Machine{proc: p, restarts: &restarts{}, ready: make(chan struct{}), inUpdate: true}
	now := time.Now()
	i, err := openIdle(filepath.Join(t.TempDir(), "idle.json"), true, now)
	if err != nil {
		t.Fatal(err)
	}
	_ = i.work(now)
	work := &activity{path: filepath.Join(t.TempDir(), "absent"), log: io.Discard}
	busy, known := m.observeActivity(work, now)
	if busy || known {
		t.Fatal("live unready Runtime lost unknown-work protection")
	}
	deadline, err := i.observe(now, busy, known)
	if err != nil {
		t.Fatal(err)
	}
	if ok, _ := i.claim(deadline.Add(time.Second)); ok {
		t.Fatal("live unknown work released")
	}
	_ = input.Close()
	<-p.done
	m.active = 1 // an update/preparation still holds admission, not workload
	busy, known = m.observeActivity(work, now)
	if busy || !known {
		t.Fatal("dead Runtime still held rental")
	}
	deadline, err = i.observe(now, busy, known)
	if err != nil {
		t.Fatal(err)
	}
	if ok, err := i.claim(deadline.Add(time.Second)); err != nil || !ok {
		t.Fatal("dead used rental failed to release", ok, err)
	}
}

func TestReadyExitDoesNotCountAsPreReadyFailure(t *testing.T) {
	r := &restarts{}
	if !r.exited(errors.New("ready process exited"), true) || r.failures != 0 {
		t.Fatal("ready exit counted as pre-ready")
	}
	if !r.exited(errors.New("first boot failure"), false) || r.gone() {
		t.Fatal("first pre-ready failure blocked")
	}
	if r.exited(errors.New("second boot failure"), false) || !r.gone() {
		t.Fatal("second pre-ready failure not blocked")
	}
}
