package host

import (
	"context"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type observationStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (s observationStream) Context() context.Context { return s.ctx }
func (s observationStream) RecvMsg(any) error        { return nil }
func (s observationStream) SendMsg(any) error        { return nil }

func TestUnauthenticatedObservationDoesNotWakeRuntime(t *testing.T) {
	m := &Machine{receipt: &receipt{}, restarts: &restarts{}, idle: &idle{}, owned: true, wake: make(chan struct{}, 1)}
	response := httptest.NewRecorder()
	m.serveReceipt(response, httptest.NewRequest("GET", "/v1/bootstrap/receipt", nil))
	if response.Code != http.StatusTooEarly {
		t.Fatal(response.Code)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	if err := m.protocolInfo(observationStream{ctx: ctx}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-m.wake:
		t.Fatal("unauthenticated observation requested a Runtime launch")
	default:
	}
}

func TestAuthenticatedWorkspaceObservationBeforeLauncherIsTransient(t *testing.T) {
	m := &Machine{grant: &Grant{WorkerID: "machine"}, id: &Identity{BootID: "boot"}, initialized: make(chan struct{})}
	_, handled, err := m.publicFallback(t.Context(), &pb.MachineExecutionWorkspaceQuery{})
	if !handled || status.Code(err) != codes.Unavailable {
		t.Fatalf("initial workspace absence became permanent refusal: handled=%v err=%v", handled, err)
	}
}
