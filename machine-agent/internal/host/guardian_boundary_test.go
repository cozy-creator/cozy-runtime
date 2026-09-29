package host

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestGuardianBoundaryHelper(t *testing.T) {
	if os.Getenv("COZY_GUARDIAN_BOUNDARY_HELPER") != "1" {
		return
	}
	root := os.Getenv("COZY_GUARDIAN_BOUNDARY_ROOT")
	os.Exit(RunGuardian([]string{guardianProcessName, filepath.Join(root, "runtime"), root}))
}

func TestGuardianLargeCommandKeepsRuntimeAndControlAlive(t *testing.T) {
	root := t.TempDir()
	executable(t, filepath.Join(root, "runtime"), "exec cat <&3 >/dev/null")
	controlRead, controlWrite, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer controlWrite.Close()
	statusRead, statusWrite, err := os.Pipe()
	if err != nil {
		controlRead.Close()
		t.Fatal(err)
	}
	defer statusRead.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	self, err := os.Executable()
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	cmd := exec.CommandContext(ctx, self, "-test.run=^TestGuardianBoundaryHelper$")
	cmd.Env = append(os.Environ(), "COZY_GUARDIAN_BOUNDARY_HELPER=1", "COZY_GUARDIAN_BOUNDARY_ROOT="+root)
	cmd.ExtraFiles = []*os.File{controlRead, statusWrite}
	var diagnostics bytes.Buffer
	cmd.Stderr = &diagnostics
	if err := startChild(cmd); err != nil {
		cancel()
		controlRead.Close()
		statusWrite.Close()
		t.Fatal(err)
	}
	controlRead.Close()
	statusWrite.Close()
	t.Cleanup(func() {
		defer cancel()
		controlWrite.Close()
		if err := waitChild(cmd); err != nil {
			t.Errorf("guardian exit: %v: %s", err, diagnostics.String())
		}
	})
	decoder := json.NewDecoder(statusRead)
	call := func(command guardianCommand) guardianStatus {
		t.Helper()
		if err := json.NewEncoder(controlWrite).Encode(command); err != nil {
			t.Fatal(err)
		}
		var answer guardianStatus
		if err := decoder.Decode(&answer); err != nil {
			t.Fatalf("guardian lost control after %s: %v", command.Op, err)
		}
		return answer
	}
	if answer := call(guardianCommand{Op: "launch", Incarnation: "large-command"}); answer.Phase != "started" {
		t.Fatalf("launch: %+v", answer)
	}
	query, _ := json.Marshal(map[string]string{"op": "get", "request": strings.Repeat("\\", 20<<10)})
	command := guardianCommand{Op: "public-read", Args: []string{string(query)}}
	framed, _ := json.Marshal(command)
	if len(query) >= 64<<10 || len(framed) <= 64<<10 {
		t.Fatal("fixture does not cross only the outer JSON framing boundary")
	}
	// This fixture has no installed journal reader. The operation must return its
	// typed refusal while the guardian and already-running Runtime stay alive.
	if answer := call(command); answer.Phase != "maintained" || answer.Code != "runtime_public_reads_unavailable" {
		t.Fatalf("public read: %+v", answer)
	}
	if answer := call(guardianCommand{Op: "pair"}); answer.Phase != "maintained" || answer.Error != "" {
		t.Fatalf("subsequent maintenance: %+v", answer)
	}
	if answer := call(guardianCommand{Op: "stop", Incarnation: "large-command"}); answer.Phase != "exited" || answer.Incarnation != "large-command" {
		t.Fatalf("explicit Runtime stop: %+v", answer)
	}
}

func TestGuardianLargePublicResponseKeepsStatusDispatchAlive(t *testing.T) {
	body, _ := json.Marshal(map[string]string{"message": strings.Repeat("x", 100<<10)})
	page, err := proto.Marshal(&pb.MachineExecutionEventPage{HeadSequence: 1, Events: []*pb.MachineExecutionEvent{{
		Sequence: 1, AttemptOrdinal: 1, Kind: "log", BodyCanonicalBytes: body,
	}}})
	if err != nil {
		t.Fatal(err)
	}
	answer := base64.StdEncoding.EncodeToString(page)
	reader, writer := io.Pipe()
	defer writer.Close()
	process := &runtimeProcess{incarnation: "large-response", done: make(chan struct{})}
	answered, started := make(chan guardianStatus, 1), make(chan error, 1)
	g := &guardianLauncher{current: process, answered: answered, started: started}
	readDone := make(chan struct{})
	go func() { g.read(reader); close(readDone) }()
	send := func(status guardianStatus) {
		t.Helper()
		if err := json.NewEncoder(writer).Encode(status); err != nil {
			t.Fatal(err)
		}
	}
	send(guardianStatus{Phase: "maintained", Answer: answer})
	if received := <-answered; received.Error != "" || received.Answer != answer {
		t.Fatal("large public event page was not delivered intact")
	}
	send(guardianStatus{Phase: "started", Incarnation: process.incarnation})
	if err := <-started; err != nil {
		t.Fatalf("status reader did not survive the public response: %v", err)
	}
	send(guardianStatus{Phase: "exited", Incarnation: process.incarnation, ExitCode: 0})
	<-process.done
	writer.Close()
	<-readDone
	if exited, ok := process.err.(interface{ ExitCode() int }); !ok || exited.ExitCode() != 0 {
		t.Fatalf("subsequent Runtime exit was lost: %v", process.err)
	}
}
