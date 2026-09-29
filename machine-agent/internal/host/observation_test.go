package host

import (
	"context"
	"google.golang.org/grpc"
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
