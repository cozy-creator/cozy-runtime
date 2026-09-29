// Package webrtctest is test support for the machine's WebRTC listener: a machine whose
// outputs are real files rewritten in place under a directory, and a pion ICE-TCP client
// that connects the way a browser does.
package webrtctest

import (
	"context"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"fmt"
	"math/big"
	"net"
	"net/netip"
	"os"
	"path/filepath"
	"runtime/pprof"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host/outputs"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host/webrtc"
)

// Machine is an outputs.Source over files: each revision's bytes are written (appended in
// place, or replaced by rename) before its entry is journaled, as the Runtime does. Like the
// machine's fold, revisions carry their sha256 once the run's terminal exists.
type Machine struct {
	dir     string
	mu      sync.Mutex
	logs    map[uint64][]outputs.Entry
	current map[string]int // the log index of each output's latest entry, by path
	changed chan struct{}  // closed on every entry
	keys    []ed25519.PublicKey
	rekeyed chan struct{} // closed when the keys change
}

func NewMachine(dir string, keys ...ed25519.PublicKey) *Machine {
	return &Machine{dir: dir, logs: map[uint64][]outputs.Entry{}, current: map[string]int{},
		changed: make(chan struct{}), keys: keys, rekeyed: make(chan struct{})}
}

func (m *Machine) path(run uint64, output string, index int) string {
	return filepath.Join(m.dir, fmt.Sprintf("%d-%s-%d", run, output, index))
}

// latest is an output's current entry; ok is false before its first. Callers hold mu.
func (m *Machine) latest(run uint64, path string) (outputs.Entry, bool) {
	i, ok := m.current[path]
	if !ok {
		return outputs.Entry{}, false
	}
	return m.logs[run][i], true
}

// Append adds data, which plays for durationUS, to an output in place.
func (m *Machine) Append(run uint64, output string, index int, data []byte, durationUS uint64) outputs.Entry {
	path := m.path(run, output, index)
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o600)
	if err == nil {
		_, err = file.Write(data)
		file.Close()
	}
	if err != nil {
		panic(err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	previous, _ := m.latest(run, path)
	from := uint64(previous.Length)
	return m.journal(run, output, index, path, previous.Length+int64(len(data)), &from, previous.DurationUS+durationUS)
}

// Replace rewrites an output's bytes by rename, so readers of the old revision keep them.
func (m *Machine) Replace(run uint64, output string, index int, data []byte, durationUS uint64) outputs.Entry {
	path := m.path(run, output, index)
	if err := os.WriteFile(path+".new", data, 0o600); err != nil {
		panic(err)
	}
	if err := os.Rename(path+".new", path); err != nil {
		panic(err)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.journal(run, output, index, path, int64(len(data)), nil, durationUS)
}

// journal records a revision; durationUS is the whole output's, as the fold's. Callers hold mu.
func (m *Machine) journal(run uint64, output string, index int, path string, length int64, appended *uint64, durationUS uint64) outputs.Entry {
	previous, _ := m.latest(run, path)
	e := outputs.Entry{Seq: uint64(len(m.logs[run]) + 1), Output: output, Index: index, Rev: previous.Rev + 1,
		Length: length, AppendedFrom: appended, DurationUS: durationUS, MediaType: "video/mp4", Label: output}
	m.current[path] = len(m.logs[run])
	m.logs[run] = append(m.logs[run], e)
	m.notify()
	return e
}

// End journals the run's terminal: every output's current revision becomes final.
func (m *Machine) End(run uint64, status string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for path, i := range m.current {
		if !strings.HasPrefix(filepath.Base(path), fmt.Sprintf("%d-", run)) {
			continue
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			panic(err)
		}
		sum := sha256.Sum256(raw)
		m.logs[run][i].SHA256 = "sha256:" + hex.EncodeToString(sum[:])
	}
	m.logs[run] = append(m.logs[run], outputs.Entry{Seq: uint64(len(m.logs[run]) + 1), Index: -1, Status: status})
	m.notify()
}

func (m *Machine) notify() {
	close(m.changed)
	m.changed = make(chan struct{})
}

func (m *Machine) Open(run uint64, output string, index int) (outputs.Snapshot, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	path := m.path(run, output, index)
	e, ok := m.latest(run, path)
	if !ok {
		return outputs.Snapshot{}, outputs.ErrNotFound
	}
	file, err := os.Open(path)
	if err != nil {
		return outputs.Snapshot{}, err
	}
	return outputs.Snapshot{Body: file, Length: e.Length, Rev: e.Rev, SHA256: e.SHA256, Final: e.SHA256 != ""}, nil
}

func (m *Machine) Entries(ctx context.Context, run uint64, after uint64) ([]outputs.Entry, error) {
	for {
		m.mu.Lock()
		log, changed := slices.Clone(m.logs[run]), m.changed
		m.mu.Unlock()
		if n := uint64(len(log)); after < n {
			return log[after:], nil
		} else if n > 0 && log[n-1].Status != "" {
			return log[n-1:], nil
		}
		select {
		case <-changed:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func (m *Machine) Keys() ([]ed25519.PublicKey, <-chan struct{}) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.keys), m.rekeyed
}

// Revoke stops authorizing a key.
func (m *Machine) Revoke(key ed25519.PublicKey) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.keys = slices.DeleteFunc(m.keys, func(k ed25519.PublicKey) bool { return k.Equal(key) })
	close(m.rekeyed)
	m.rekeyed = make(chan struct{})
}

// Server is a listener serving a Machine.
type Server struct {
	Addr        netip.AddrPort
	Fingerprint string // the leaf's pin, "sha-256 AB:…"
	Machine     string // the worker id capabilities name
}

// Serve runs the listener for m on host (an IP) until the test ends.
func Serve(t testing.TB, host string, m *Machine) Server {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{SerialNumber: big.NewInt(1), NotBefore: time.Now().Add(-time.Hour),
		NotAfter: time.Date(9999, 12, 31, 0, 0, 0, 0, time.UTC), KeyUsage: x509.KeyUsageDigitalSignature}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	ln, err := net.Listen("tcp", net.JoinHostPort(host, "0")) //cozy:allow the listener under test, on a test port
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	s := Server{Addr: ln.Addr().(*net.TCPAddr).AddrPort(), Fingerprint: webrtc.Fingerprint(der), Machine: "wk-test"}
	go pprof.Do(ctx, pprof.Labels("side", "machine"), func(ctx context.Context) { // profiles tell the ends apart
		done <- webrtc.Serve(ctx, ln, webrtc.Config{Machine: s.Machine, Source: m,
			Leaf: tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}})
	})
	t.Cleanup(func() {
		cancel()
		if err := <-done; err != nil {
			t.Error(err)
		}
	})
	return s
}

func (m *Machine) RefreshKeys(ctx context.Context) error { return ctx.Err() }
