// Package webrtc serves a machine's run outputs to browsers over WebRTC-direct: ICE-lite,
// passive ICE-TCP on one granted port, one data channel speaking cozy/1. There is no
// signaling server, ICE server, proxy or per-machine certificate: the browser pins the
// machine leaf's fingerprint and presents a capability.
package webrtc

import (
	"context"
	"crypto/ed25519"
	"crypto/tls"
	"net"
	"net/netip"
	"slices"
	"sync"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host/outputs"
)

// Config is one machine's listener.
type Config struct {
	Machine string          // the worker id every capability must name
	Leaf    tls.Certificate // the machine leaf, whose sha-256 browsers pin
	Source  outputs.Source
}

// Limits are by capacity, never by timer: an arrival over a limit evicts the oldest session
// still pending (before a valid hello), its own IP's first. Authenticated sessions end only
// on their own key's limit, a revocation, or the peer.
const (
	maxConns      = 64
	maxConnsPerIP = 8
	maxPerKey     = 16
	maxStreams    = 8
	maxHello      = 8 << 10
	maxMessage    = 64 << 10
	endedUfrags   = 4096
)

type server struct {
	cfg      Config
	mu       sync.Mutex
	sessions []*session     // in arrival order
	live     map[string]int // connections per ufrag
	ended    map[string]bool
	endedAt  []string // ended ufrags, oldest first, at most endedUfrags
}

// Serve answers clients on ln until ctx ends; authenticated sessions then get bye{shutdown}.
func Serve(ctx context.Context, ln net.Listener, cfg Config) error {
	s := &server{cfg: cfg, live: map[string]int{}, ended: map[string]bool{}}
	stop := context.AfterFunc(ctx, func() {
		ln.Close()
		s.mu.Lock()
		defer s.mu.Unlock()
		for _, c := range s.sessions {
			if c.authed {
				c.bye("shutdown", "the machine is stopping")
			} else {
				c.abort()
			}
		}
	})
	defer stop()
	go s.watchKeys(ctx)
	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		c := newSession(s, conn.(*net.TCPConn))
		if s.admit(c) {
			go c.run()
		} else {
			c.abort()
		}
	}
}

func (s *server) admit(c *session) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for {
		own := 0
		for _, o := range s.sessions {
			if o.ip == c.ip {
				own++
			}
		}
		if own < maxConnsPerIP && len(s.sessions) < maxConns {
			s.sessions = append(s.sessions, c)
			return true
		}
		victim := s.oldestPending(c.ip)
		if victim == nil && own < maxConnsPerIP {
			victim = s.oldestPending(netip.Addr{})
		}
		if victim == nil {
			return false
		}
		s.dropLocked(victim)
		victim.abort()
	}
}

func (s *server) oldestPending(ip netip.Addr) *session {
	for _, o := range s.sessions {
		if !o.authed && (!ip.IsValid() || o.ip == ip) {
			return o
		}
	}
	return nil
}

// A ufrag names one session for its whole life. After a dropped TCP connection a browser
// reconnects to the same candidate with the same ufrag and resumes DTLS records no new
// session could read; closing that connection unanswered fails its PeerConnection, so the
// client reconnects afresh and resumes at its cursor. The most recent ended ufrags are kept
// by capacity, never by time.
func (s *server) claim(ufrag string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ended[ufrag] {
		return false
	}
	s.live[ufrag]++
	return true
}

func (s *server) release(ufrag string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.live[ufrag]--; s.live[ufrag] > 0 {
		return
	}
	delete(s.live, ufrag)
	if len(s.endedAt) == endedUfrags {
		delete(s.ended, s.endedAt[0])
		s.endedAt = s.endedAt[1:]
	}
	s.ended[ufrag] = true
	s.endedAt = append(s.endedAt, ufrag)
}

// authenticate admits a session under its key, which must still be authorized, and evicts
// that key's oldest session beyond its limit.
func (s *server) authenticate(c *session, key string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	keys, _ := s.cfg.Source.Keys()
	if !authorized(keys, key) || !slices.Contains(s.sessions, c) {
		return false
	}
	var same []*session
	for _, o := range s.sessions {
		if o.authed && o.key == key {
			same = append(same, o)
		}
	}
	if len(same) >= maxPerKey {
		same[0].authed = false
		same[0].bye("evicted", "this key opened more sessions than a machine keeps")
	}
	c.authed, c.key = true, key
	return true
}

// watchKeys ends every session of a key the machine stops authorizing.
func (s *server) watchKeys(ctx context.Context) {
	for {
		keys, changed := s.cfg.Source.Keys()
		s.mu.Lock()
		for _, c := range s.sessions {
			if c.authed && !authorized(keys, c.key) {
				c.authed = false
				c.bye("revoked", "the key that opened this session was revoked")
			}
		}
		s.mu.Unlock()
		select {
		case <-changed:
		case <-ctx.Done():
			return
		}
	}
}

func (s *server) drop(c *session) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.dropLocked(c)
}

func (s *server) dropLocked(c *session) {
	s.sessions = slices.DeleteFunc(s.sessions, func(o *session) bool { return o == c })
}

func authorized(keys []ed25519.PublicKey, id string) bool {
	return slices.ContainsFunc(keys, func(k ed25519.PublicKey) bool { return capability.KeyID(k) == id })
}
