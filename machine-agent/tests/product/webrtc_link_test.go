package producttest

import (
	"io"
	"net"
	"net/netip"
	"sync"
	"testing"
	"time"
)

// mediaLink is a path between client and machine: it forwards each TCP connection with a
// one-way delay each way (RTT = 2 × delay), optionally through a bottleneck of rate bytes/s
// with an unbounded queue, and can kill every connection at once.
type mediaLink struct {
	ln       net.Listener
	delay    time.Duration
	rate     float64
	mu       sync.Mutex
	conns    []*net.TCPConn
	maxQueue time.Duration // the longest any byte waited at the bottleneck
}

func newMediaRelay(t testing.TB, target netip.AddrPort, delay time.Duration) *mediaLink {
	return newMediaLink(t, target, delay, 0)
}

func newMediaLink(t testing.TB, target netip.AddrPort, delay time.Duration, rate float64) *mediaLink {
	ln, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow a test path between client and listener
	if err != nil {
		t.Fatal(err)
	}
	r := &mediaLink{ln: ln, delay: delay, rate: rate}
	t.Cleanup(func() { ln.Close(); r.kill() })
	go func() {
		for {
			in, err := ln.Accept()
			if err != nil {
				return
			}
			out, err := net.Dial("tcp", target.String())
			if err != nil {
				in.Close()
				continue
			}
			r.mu.Lock()
			r.conns = append(r.conns, in.(*net.TCPConn), out.(*net.TCPConn))
			r.mu.Unlock()
			go r.forward(out, in)
			go r.forward(in, out)
		}
	}()
	return r
}

func (r *mediaLink) addr() netip.AddrPort { return r.ln.Addr().(*net.TCPAddr).AddrPort() }

// kill resets every connection, as a dropped path does.
func (r *mediaLink) kill() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, c := range r.conns {
		c.SetLinger(0)
		c.Close()
	}
	r.conns = nil
}

func (r *mediaLink) forward(dst, src net.Conn) {
	defer dst.Close()
	if r.delay == 0 && r.rate == 0 {
		io.Copy(dst, src)
		return
	}
	type chunk struct {
		at   time.Time
		data []byte
	}
	line := make(chan chunk, 1<<16)
	go func() {
		defer close(line)
		buf := make([]byte, 16<<10)
		var free time.Time // when the bottleneck has sent everything queued
		for {
			n, err := src.Read(buf)
			if err != nil {
				return
			}
			at := time.Now()
			if r.rate > 0 {
				if free.Before(at) {
					free = at
				}
				free = free.Add(time.Duration(float64(n) / r.rate * float64(time.Second)))
				r.mu.Lock()
				r.maxQueue = max(r.maxQueue, free.Sub(at))
				r.mu.Unlock()
				at = free
			}
			line <- chunk{at.Add(r.delay), append([]byte(nil), buf[:n]...)}
		}
	}()
	for c := range line {
		time.Sleep(time.Until(c.at))
		if _, err := dst.Write(c.data); err != nil {
			return
		}
	}
}
