package host

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

type rollbackBarrier struct {
	entered, release chan struct{}
	probes           atomic.Int32
}

func (*rollbackBarrier) close() {}
func (*rollbackBarrier) launch(string) (*runtimeProcess, error) {
	return nil, errors.New("fixture must not launch Runtime")
}
func (l *rollbackBarrier) maintain(op string, _ []string) (string, error) {
	switch op {
	case "startup-rollback":
		return "", nil
	case "agent-replace":
		close(l.entered)
		<-l.release
		return "", errors.New("fixture application handoff unavailable")
	case "capabilities":
		l.probes.Add(1)
		return "", errors.New("supervisor entered rollback")
	default:
		return "", errors.New("unexpected fixture maintenance")
	}
}

func TestRollbackOwnsAdmissionThroughApplicationHandoff(t *testing.T) {
	for _, boot := range []bool{false, true} {
		name := "manual"
		if boot {
			name = "boot-lock-already-held"
		}
		t.Run(name, func(t *testing.T) {
			m, token, mux := updateMachine(t)
			barrier := &rollbackBarrier{entered: make(chan struct{}), release: make(chan struct{})}
			m.launcher = barrier
			m.owned = true
			m.setUpdating(true)
			m.grant.startupPending = true
			m.grant.bootstrap = &bootstrapChild{Pending: true}
			m.updates.status = &updateStatus{Operation: "recovering", State: "installing"}
			done := make(chan error, 1)
			go func() {
				if boot {
					m.launchMu.Lock()
					defer m.launchMu.Unlock()
				}
				done <- m.rollbackStartup(context.Background(), errors.New("fixture candidate failed"))
			}()
			<-barrier.entered // no non-reentrant lock acquisition inside rollback
			server := httptest.NewServer(mux)
			defer server.Close()
			if !boot {
				m.launchOrIdle()
			}
			probes := barrier.probes.Load()
			held := m.updating()
			reply := postUpdate(server, token, "second-update", &wheelChoice{Version: "0.18.87"})
			close(barrier.release)
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			if !held || probes != 0 {
				t.Fatalf("rollback lost launch ownership before handoff: held=%v probes=%d", held, probes)
			}
			if reply.code != http.StatusConflict {
				t.Fatalf("terminal rollback status admitted another update before handoff: %+v", reply)
			}
			if m.updating() || m.restarts.gone() {
				t.Fatal("failed handoff did not release ownership to paced repair service")
			}
		})
	}
}
