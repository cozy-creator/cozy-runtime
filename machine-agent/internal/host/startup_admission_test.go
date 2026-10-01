package host

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/status"
)

// A caller reaching a booting machine before its startup check finishes waits for it: with no
// update pending the Runtime is starting, and the caller gets it once it is ready. Only a boot
// whose check found an update pending answers "updating" (the CLI then reads the Hub instead).
func TestABootAnswersUpdatingOnlyForAPendingUpdate(t *testing.T) {
	for _, pending := range []bool{false, true} {
		state, err := openIdle(filepath.Join(t.TempDir(), "idle.json"), true, time.Now())
		if err != nil {
			t.Fatal(err)
		}
		m := &Machine{idle: state, owned: true, restarts: &restarts{}, ready: make(chan struct{}),
			initialized: make(chan struct{}), inUpdate: true}
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		answered := make(chan error, 1)
		go func() { _, err := m.runtime(ctx); answered <- err }()
		select {
		case err := <-answered:
			t.Fatalf("answered before the startup check finished (pending=%v): %v", pending, err)
		case <-time.After(200 * time.Millisecond):
		}
		// The startup check decides; boot launches the Runtime unless an update holds it.
		m.setUpdating(pending)
		if !pending {
			m.mu.Lock()
			m.proc = &runtimeProcess{done: make(chan struct{})}
			m.mu.Unlock()
			close(m.ready)
		}
		close(m.initialized)
		err = <-answered
		cancel()
		updating := err != nil && strings.HasPrefix(status.Convert(err).Message(), "runtime_updating:")
		if pending != updating {
			t.Fatalf("pending=%v answered %v", pending, err)
		}
		if !pending && err != nil {
			t.Fatalf("a booted Runtime was not handed to its caller: %v", err)
		}
	}
}
