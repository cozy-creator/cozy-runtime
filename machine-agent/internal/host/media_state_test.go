package host

import (
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"testing"
)

func TestPausedAttemptDoesNotFinishOutputLog(t *testing.T) {
	log := newRunLog(1)
	log.add(&pb.MachineExecutionEvent{Sequence: 7, AttemptOrdinal: 1, Outcome: &pb.AttemptOutcome{}})
	log.observeState(&pb.MachineExecutionState{State: "paused", AttemptOrdinal: 1, Sequence: 7})
	if log.terminal || len(log.entries) != 0 {
		t.Fatal("paused attempt finalized the run", log.entries)
	}
	log.add(&pb.MachineExecutionEvent{Sequence: 9, AttemptOrdinal: 2, Outcome: &pb.AttemptOutcome{}})
	log.observeState(&pb.MachineExecutionState{State: "running", AttemptOrdinal: 2, Sequence: 9})
	if log.terminal {
		t.Fatal("attempt outcome outranked current run state")
	}
	log.observeState(&pb.MachineExecutionState{State: "succeeded", AttemptOrdinal: 2, Sequence: 10})
	if log.terminal {
		t.Fatal("output snapshot finalized before reading the terminal sequence")
	}
	log.add(&pb.MachineExecutionEvent{Sequence: 10, AttemptOrdinal: 2})
	log.observeState(&pb.MachineExecutionState{State: "succeeded", AttemptOrdinal: 2, Sequence: 10})
	if !log.terminal || len(log.entries) != 1 || log.entries[0].Status != "completed" {
		t.Fatal(log.entries)
	}
}
