package host

import (
	"sync/atomic"
	"testing"
)

func TestGuardianDelayedStopDoesNotStopReplacement(t *testing.T) {
	old := &runtimeProcess{incarnation: "old", done: make(chan struct{})}
	replacement := &runtimeProcess{incarnation: "replacement", done: make(chan struct{})}
	var replacementStops atomic.Int32
	replacement.stop = func() {
		replacementStops.Add(1)
		close(replacement.done)
	}
	processes := &guardianProcesses{}
	processes.set(old)
	stop := processes.stop(old.incarnation)

	// Hold the selected stop before it runs. The old Runtime exits independently,
	// and its replacement starts before this already-accepted command is scheduled.
	runStop, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		<-runStop
		stop()
		close(stopped)
	}()
	close(old.done)
	processes.set(replacement)
	close(runStop)
	<-stopped
	if replacementStops.Load() != 0 || replacement.exited() {
		t.Fatal("a delayed old-incarnation stop killed the replacement Runtime")
	}

	// A late command for the old incarnation is also inert, while shutdown still
	// deliberately selects and stops the current Runtime.
	processes.stop(old.incarnation)()
	if replacementStops.Load() != 0 {
		t.Fatal("a stale incarnation selected the replacement Runtime")
	}
	processes.stop("")()
	if replacementStops.Load() != 1 || !replacement.exited() {
		t.Fatal("guardian shutdown did not stop its current Runtime")
	}
}

func TestGuardianSelectedStopRetainsItsProcess(t *testing.T) {
	old := &runtimeProcess{incarnation: "old", done: make(chan struct{})}
	replacement := &runtimeProcess{incarnation: "replacement", done: make(chan struct{})}
	old.stop = func() { close(old.done) }
	processes := &guardianProcesses{}
	processes.set(old)
	stop := processes.stop(old.incarnation)
	processes.set(replacement)
	stop()
	if !old.exited() || replacement.exited() {
		t.Fatal("selected stop did not retain its original process")
	}
}
