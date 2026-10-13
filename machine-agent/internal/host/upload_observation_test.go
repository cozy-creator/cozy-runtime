package host

import (
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestWheelUploadKeepsUpdateObservationAvailable(t *testing.T) {
	m, token, mux := updateMachine(t)
	mux.HandleFunc("GET /v1/machine/runtime", m.serveRuntimeState)
	server := httptest.NewServer(mux)
	server.Client().Timeout = 5 * time.Second
	defer server.Close()
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	uploaded := make(chan updateReply, 1)
	go func() {
		uploaded <- updateCall(server.Client(), http.MethodPut, server.URL+"/v1/machine/runtime/wheels/cozy_runtime-0.18.87-py3-none-any.whl", token, reader)
	}()
	if _, err := writer.Write([]byte("first")); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		paths, _ := filepath.Glob(m.stagePath(".upload-*"))
		if len(paths) > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("upload never entered staging")
		}
		runtime.Gosched()
	}
	// The body remains open with its next byte controlled by this test. Its
	// staging file proves the handler has entered the transfer, without relying
	// on a sleep to infer that independent observation is blocked.
	if !m.updates.mu.TryLock() {
		t.Fatal("upload body owns the update-state mutex")
	}
	m.updates.mu.Unlock()
	if reply := updateCall(server.Client(), http.MethodGet, server.URL+"/v1/machine/runtime", token, nil); reply.code != http.StatusOK {
		t.Fatalf("status during upload: %+v", reply)
	}
	if m.manualUpdateActive() {
		t.Fatal("staging a wheel admitted an update")
	}
	if _, err := writer.Write([]byte("last")); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if reply := <-uploaded; reply.code != http.StatusOK {
		t.Fatalf("upload: %+v", reply)
	}
}
