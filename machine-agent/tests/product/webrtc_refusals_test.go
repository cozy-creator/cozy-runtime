package producttest

import (
	"bufio"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host/webrtc"
	"github.com/cozy-creator/cozy-runtime/machine-agent/tests/product/webrtctest"
	"github.com/pion/stun/v4"
)

func TestWebRTCHelloAdmitsOnlyAValidCapability(t *testing.T) {
	h := newMediaHarness(t)
	_, stranger, _ := ed25519.GenerateKey(rand.Reader)
	signed := func(key ed25519.PrivateKey, payload string) string { // any payload, signed as Mint signs
		sig := ed25519.Sign(key, append([]byte("cozy-capability/1\x00"), payload...))
		return base64.RawURLEncoding.EncodeToString([]byte(payload)) + "." + base64.RawURLEncoding.EncodeToString(sig)
	}
	for _, tc := range []struct {
		name  string
		hello func(c *webrtctest.Client) any
		want  string
	}{
		{"the first message is not hello", func(*webrtctest.Client) any {
			return map[string]any{"t": "get", "id": 1, "run": "7", "output": "video"}
		}, "bye auth"},
		{"an unauthorized signer", func(*webrtctest.Client) any {
			token, _ := capability.Mint(stranger, capability.Grant{Machine: h.srv.Machine, Run: "7", Expires: time.Now().Add(time.Hour).Unix()})
			return map[string]any{"t": "hello", "v": 1, "cap": token}
		}, "bye auth"},
		{"an expired capability", func(*webrtctest.Client) any {
			return map[string]any{"t": "hello", "v": 1, "cap": h.grant(func(g *capability.Grant) { g.Expires = time.Now().Unix() - 1 })}
		}, "bye expired"},
		{"another machine", func(*webrtctest.Client) any {
			return map[string]any{"t": "hello", "v": 1, "cap": h.grant(func(g *capability.Grant) { g.Machine = "wk-other" })}
		}, "bye auth"},
		{"x names another DTLS certificate", func(*webrtctest.Client) any {
			return map[string]any{"t": "hello", "v": 1, "cap": h.grant(func(g *capability.Grant) { g.Binding = webrtc.Fingerprint([]byte("other")) })}
		}, "bye auth"},
		{"x names this client's certificate", func(c *webrtctest.Client) any {
			return map[string]any{"t": "hello", "v": 1, "cap": h.grant(func(g *capability.Grant) { g.Binding = c.Cert })}
		}, "welcome"},
		{"a member this machine does not know", func(*webrtctest.Client) any {
			payload := `{"m":"` + h.srv.Machine + `","r":"7","e":` + strconv.FormatInt(time.Now().Add(time.Hour).Unix(), 10) + `,"k":"` +
				capability.KeyID(h.key.Public().(ed25519.PublicKey)) + `","z":1}`
			return map[string]any{"t": "hello", "v": 1, "cap": signed(h.key, payload)}
		}, "bye auth"},
		{"a hello over 8 KiB", func(*webrtctest.Client) any {
			return map[string]any{"t": "hello", "v": 1, "cap": h.grant(nil), "pad": strings.Repeat("x", 8<<10)}
		}, "bye auth"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := h.dial()
			h.send(c, tc.hello(c))
			m, _ := c.Recv(h.ctx)
			if got := strings.TrimSpace(m.T + " " + m.Code); got != tc.want {
				t.Fatalf("got %s, want %s", m.Raw, tc.want)
			}
			if m.T == "bye" {
				if m, ok := c.Recv(h.ctx); ok {
					t.Fatalf("the session outlived its bye: %s", m.Raw)
				}
			}
		})
	}
}

func TestWebRTCRequestsOutsideTheGrantAreRefused(t *testing.T) {
	h := newMediaHarness(t)
	h.m.Append(7, "video", -1, mediaSegment(1000, 1), 1_000_000)
	h.m.End(7, "completed")
	c := h.dial()
	h.send(c, map[string]any{"t": "hello", "v": 1, "cap": h.grant(func(g *capability.Grant) { g.Outputs = []string{"references/2"} })})
	h.expect(c, "welcome")
	h.send(c, map[string]any{"t": "credit", "bytes": 1 << 20})
	for _, request := range []map[string]any{
		{"t": "get", "id": 1, "run": "7", "output": "video"},
		{"t": "follow", "id": 2, "run": "7", "output": "references", "index": 1},
		{"t": "get", "id": 3, "run": "8", "output": "references", "index": 2},
	} {
		h.send(c, request)
		if m := h.expect(c, "error"); m.Code != "scope" {
			t.Fatalf("%v got %s", request, m.Raw)
		}
	}
	h.send(c, map[string]any{"t": "get", "id": 4, "run": "7", "output": "references", "index": 2})
	if m := h.expect(c, "error"); m.Code != "not_found" {
		t.Fatalf("a granted, absent output got %s", m.Raw)
	}
}

func TestWebRTCAMessageOver64KiBEndsTheSession(t *testing.T) {
	h := newMediaHarness(t)
	c, err := webrtctest.Dial(h.ctx, h.srv.Addr, h.srv.Fingerprint, webrtctest.Options{Oversend: true})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	h.send(c, map[string]any{"t": "hello", "v": 1, "cap": h.grant(nil)})
	h.expect(c, "welcome")
	if err := c.SendRaw(make([]byte, 65<<10), false); err != nil {
		t.Fatal(err)
	}
	if m, ok := c.Recv(h.ctx); ok {
		t.Fatalf("the session survived: %s", m.Raw)
	}
}

func TestWebRTCRevokingTheKeyEndsItsSessions(t *testing.T) {
	h := newMediaHarness(t)
	c := h.open(64 << 20)
	h.follow(c, 0, 0)
	var f mediaFollower
	h.m.Append(7, "video", -1, mediaSegment(50_000, 1), 1_000_000)
	h.until(c, &f, func() bool { return len(f.got) == 50_000 })
	h.m.Revoke(h.key.Public().(ed25519.PublicKey))
	if m := h.expect(c, "bye"); m.Code != "revoked" {
		t.Fatalf("got %s", m.Raw)
	}
	again := h.dial()
	h.send(again, map[string]any{"t": "hello", "v": 1, "cap": h.grant(nil)})
	if m := h.expect(again, "bye"); m.Code != "auth" {
		t.Fatalf("a revoked key's capability got %s", m.Raw)
	}
}

// iceTCPConn speaks raw ICE-TCP: framed STUN, as a browser's first packet.
func iceTCPConn(t *testing.T, h *mediaHarness) net.Conn {
	conn, err := net.Dial("tcp", h.srv.Addr.String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return conn
}

func sendBinding(t *testing.T, conn net.Conn, username, pwd string) {
	m, err := stun.Build(stun.TransactionID, stun.BindingRequest, stun.NewUsername(username),
		stun.NewShortTermIntegrity(pwd), stun.Fingerprint)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Write(append(binary.BigEndian.AppendUint16(nil, uint16(len(m.Raw))), m.Raw...)); err != nil {
		t.Fatal(err)
	}
}

func TestWebRTCSTUNAnswersOnlyTheClientChosenCredential(t *testing.T) {
	h := newMediaHarness(t)
	ufrag := webrtctest.Ufrag()

	good := iceTCPConn(t, h)
	sendBinding(t, good, ufrag+":browser", ufrag)
	r := bufio.NewReader(good)
	var size [2]byte
	if _, err := io.ReadFull(r, size[:]); err != nil {
		t.Fatal(err)
	}
	reply := &stun.Message{Raw: make([]byte, binary.BigEndian.Uint16(size[:]))}
	if _, err := io.ReadFull(r, reply.Raw); err != nil || reply.Decode() != nil {
		t.Fatal("no Binding success")
	}
	var mapped stun.XORMappedAddress
	if reply.Type != stun.BindingSuccess || mapped.GetFrom(reply) != nil || mapped.Port != good.LocalAddr().(*net.TCPAddr).Port ||
		stun.NewShortTermIntegrity(ufrag).Check(reply) != nil || stun.Fingerprint.Check(reply) != nil {
		t.Fatalf("reply %v", reply)
	}

	for name, pair := range map[string][2]string{
		"signed with another password": {ufrag + ":browser", ufrag + "x"},
		"a ufrag without the prefix":   {"published:browser", "published"},
	} {
		conn := iceTCPConn(t, h)
		sendBinding(t, conn, pair[0], pair[1])
		conn.SetReadDeadline(time.Now().Add(10 * time.Second))
		if n, err := conn.Read(make([]byte, 1)); n != 0 || !errors.Is(err, io.EOF) && !errors.Is(err, net.ErrClosed) && !connReset(err) {
			t.Fatalf("%s: read %d, %v", name, n, err)
		}
	}
}

// ufragRefused says whether a new connection's Binding request under ufrag is closed
// unanswered; false means it was answered.
func ufragRefused(t *testing.T, h *mediaHarness, ufrag string) bool {
	conn := iceTCPConn(t, h)
	defer conn.Close()
	sendBinding(t, conn, ufrag+":browser", ufrag)
	conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	n, err := conn.Read(make([]byte, 1))
	if n == 0 && !errors.Is(err, io.EOF) && !connReset(err) {
		t.Fatalf("a reused ufrag got neither an answer nor a close: %v", err)
	}
	return n == 0
}

func connReset(err error) bool {
	return err != nil && strings.Contains(err.Error(), "connection reset")
}

func TestWebRTCTheNinthConnectionFromOneIPEvictsTheOldestPending(t *testing.T) {
	h := newMediaHarness(t)
	h.m.Append(7, "video", -1, mediaSegment(1000, 1), 1_000_000)
	h.m.End(7, "completed")
	c := h.open(1 << 20) // authenticated: never evicted for pending ones
	var pending []net.Conn
	for range 8 { // 1 + 7 fill the IP's 8; the 8th pending arrival is the 9th connection
		pending = append(pending, iceTCPConn(t, h))
	}
	pending[0].SetReadDeadline(time.Now().Add(10 * time.Second))
	if _, err := pending[0].Read(make([]byte, 1)); !errors.Is(err, io.EOF) && !connReset(err) {
		t.Fatalf("the oldest pending connection was kept: %v", err)
	}
	for _, conn := range pending[1:] {
		conn.SetReadDeadline(time.Now().Add(50 * time.Millisecond))
		if _, err := conn.Read(make([]byte, 1)); !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Fatalf("a newer pending connection ended: %v", err)
		}
	}
	h.send(c, map[string]any{"t": "get", "id": 1, "run": "7", "output": "video"})
	if got, _ := h.body(c); len(got) != 1000 {
		t.Fatal("the authenticated session stopped serving")
	}
}
