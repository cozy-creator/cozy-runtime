package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"slices"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
	"github.com/cozy-creator/cozy-runtime/machine-agent/tests/product/webrtctest"
)

// Every test runs a real listener on loopback and a real pion ICE-TCP client over real TCP;
// outputs are real files, appended in place or replaced by rename, before their entries.

type mediaHarness struct {
	t   testing.TB
	ctx context.Context
	m   *webrtctest.Machine
	srv webrtctest.Server
	key ed25519.PrivateKey
}

func newMediaHarness(t testing.TB) *mediaHarness {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	t.Cleanup(cancel)
	public, key, _ := ed25519.GenerateKey(rand.Reader)
	m := webrtctest.NewMachine(t.TempDir(), public)
	return &mediaHarness{t: t, ctx: ctx, m: m, srv: webrtctest.Serve(t, "127.0.0.1", m), key: key}
}

// grant mints a capability for run 7 of this machine, valid for an hour unless edit says otherwise.
func (h *mediaHarness) grant(edit func(*capability.Grant)) string {
	g := capability.Grant{Machine: h.srv.Machine, Run: "7", Expires: time.Now().Add(time.Hour).Unix()}
	if edit != nil {
		edit(&g)
	}
	token, err := capability.Mint(h.key, g)
	if err != nil {
		h.t.Fatal(err)
	}
	return token
}

func (h *mediaHarness) dial() *webrtctest.Client {
	c, err := webrtctest.Dial(h.ctx, h.srv.Addr, h.srv.Fingerprint, webrtctest.Options{})
	if err != nil {
		h.t.Fatal(err)
	}
	h.t.Cleanup(func() { c.Close() })
	return c
}

// open is a welcomed session with credit.
func (h *mediaHarness) open(credit uint64) *webrtctest.Client {
	c := h.dial()
	h.send(c, map[string]any{"t": "hello", "v": 1, "cap": h.grant(nil)})
	h.expect(c, "welcome")
	h.send(c, map[string]any{"t": "credit", "bytes": credit})
	return c
}

func (h *mediaHarness) send(c *webrtctest.Client, record any) {
	if err := c.Send(record); err != nil {
		h.t.Fatal(err)
	}
}

func (h *mediaHarness) expect(c *webrtctest.Client, t string) webrtctest.Message {
	h.t.Helper()
	m, _ := c.Recv(h.ctx)
	if m.T != t {
		h.t.Fatalf("expected %s, got %s %s", t, m.T, m.Raw)
	}
	return m
}

// body reads one stream's binary messages from its open to its end, checking every offset.
func (h *mediaHarness) body(c *webrtctest.Client) ([]byte, webrtctest.Message) {
	h.t.Helper()
	open := h.expect(c, "open")
	var got []byte
	for {
		m, _ := c.Recv(h.ctx)
		switch m.T {
		case "data":
			if m.Stream != open.Stream || m.Offset != open.Offset+uint64(len(got)) {
				h.t.Fatalf("data at %d of stream %d; expected %d of %d", m.Offset, m.Stream, open.Offset+uint64(len(got)), open.Stream)
			}
			got = append(got, m.Data...)
		case "end":
			if uint64(len(got)) != open.Length {
				h.t.Fatalf("%d bytes before end; open announced %d", len(got), open.Length)
			}
			return got, m
		default:
			h.t.Fatalf("unexpected %s", m.Raw)
		}
	}
}

func mediaSegment(n int, fill byte) []byte { return bytes.Repeat([]byte{fill}, n) }

func TestWebRTCGetServesARangeOfTheCurrentBytes(t *testing.T) {
	h := newMediaHarness(t)
	seg := [][]byte{mediaSegment(100_000, 1), mediaSegment(150_000, 2), mediaSegment(70_000, 3)}
	h.m.Append(7, "video", -1, seg[0], 1_000_000)
	h.m.Append(7, "video", -1, seg[1], 1_000_000)
	h.m.Append(7, "video", -1, seg[2], 1_000_000)
	h.m.End(7, "completed")
	c := h.open(1 << 20)

	h.send(c, map[string]any{"t": "get", "id": 1, "run": "7", "output": "video", "offset": 100_000, "length": 150_000})
	got, end := h.body(c)
	if !bytes.Equal(got, seg[1]) {
		t.Fatal("the middle mediaSegment differs")
	}
	whole := bytes.Join(seg, nil)
	if end.Length != uint64(len(whole)) || end.SHA256 != digestOf(whole) {
		t.Fatalf("end %s", end.Raw)
	}

	h.send(c, map[string]any{"t": "get", "id": 2, "run": "7", "output": "video", "offset": 0, "etag": "r2"})
	if m := h.expect(c, "error"); m.Code != "changed" {
		t.Fatalf("a stale etag got %s", m.Raw)
	}
	h.send(c, map[string]any{"t": "get", "id": 3, "run": "7", "output": "video", "offset": 0, "etag": "r3"})
	if got, _ := h.body(c); !bytes.Equal(got, whole) {
		t.Fatal("the whole output differs")
	}
}

// mediaFollower is a client's copy of a followed output and its cursor (seq, len(got)).
type mediaFollower struct {
	got     []byte
	seq     uint64 // the last entry whose bytes it holds in full
	entries []webrtctest.Message
	resets  int
	end     *webrtctest.Message
}

// until applies messages until done says so, checking there is never a gap or a duplicate.
func (h *mediaHarness) until(c *webrtctest.Client, f *mediaFollower, done func() bool) {
	h.t.Helper()
	for !done() {
		m, ok := c.Recv(h.ctx)
		if !ok {
			h.t.Fatalf("the session ended: %s", m.T)
		}
		switch m.T {
		case "entry":
			f.entries = append(f.entries, m)
		case "reset": // entries are eager: those from the reset on are the replacement's
			f.entries = slices.DeleteFunc(f.entries, func(e webrtctest.Message) bool { return e.Seq < m.Seq })
			f.got, f.seq = nil, 0
			f.resets++
		case "open", "data":
			if m.Offset != uint64(len(f.got)) {
				h.t.Fatalf("%s at %d while holding %d", m.T, m.Offset, len(f.got))
			}
			f.got = append(f.got, m.Data...)
		case "end":
			f.end = &m
		default:
			h.t.Fatalf("unexpected %s", m.Raw)
		}
		for _, e := range f.entries {
			if e.Length <= uint64(len(f.got)) {
				f.seq = e.Seq
			}
		}
	}
}

func (h *mediaHarness) follow(c *webrtctest.Client, after, offset uint64) {
	h.send(c, map[string]any{"t": "follow", "id": "v", "run": "7", "output": "video", "after": after, "offset": offset})
}

func TestWebRTCFollowStreamsEachAppendAsItLands(t *testing.T) {
	h := newMediaHarness(t)
	c := h.open(64 << 20)
	h.follow(c, 0, 0)
	var f mediaFollower
	var whole []byte
	for k := range 3 {
		seg := mediaSegment(200_000+k*50_000, byte(k+1))
		whole = append(whole, seg...)
		e := h.m.Append(7, "video", -1, seg, 1_000_000)
		h.until(c, &f, func() bool { return len(f.got) == len(whole) && f.seq == e.Seq })
	}
	h.m.End(7, "completed")
	h.until(c, &f, func() bool { return f.end != nil })
	if !bytes.Equal(f.got, whole) || f.end.Status != "completed" || f.end.SHA256 != digestOf(whole) || f.end.Length != uint64(len(whole)) {
		t.Fatalf("end %s after %d bytes", f.end.Raw, len(f.got))
	}
	if len(f.entries) != 3 || *f.entries[2].AppendedFrom != uint64(len(whole)-300_000) || f.entries[2].Rev != 3 ||
		f.entries[2].DurationUS != 3_000_000 {
		t.Fatalf("entries %+v", f.entries)
	}
}

func TestWebRTCFollowResumesAfterTheConnectionDies(t *testing.T) {
	h := newMediaHarness(t)
	r := newMediaRelay(t, h.srv.Addr, 0)
	seg := [][]byte{mediaSegment(300_000, 1), mediaSegment(400_000, 2), mediaSegment(250_000, 3)}
	c, err := webrtctest.Dial(h.ctx, r.addr(), h.srv.Fingerprint, webrtctest.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	h.send(c, map[string]any{"t": "hello", "v": 1, "cap": h.grant(nil)})
	h.expect(c, "welcome")
	held := uint64(len(seg[0]) + len(seg[1])/2) // credit ends mid-segment
	h.send(c, map[string]any{"t": "credit", "bytes": held})
	h.follow(c, 0, 0)
	var f mediaFollower
	h.m.Append(7, "video", -1, seg[0], 1_000_000)
	h.m.Append(7, "video", -1, seg[1], 1_000_000)
	h.until(c, &f, func() bool { return uint64(len(f.got)) == held })
	r.kill()

	// A browser reconnects TCP with the dropped session's ufrag: closed unanswered, so its
	// PeerConnection fails and it starts afresh. Once the machine has seen the drop.
	for !ufragRefused(t, h, c.Ufrag) {
		time.Sleep(10 * time.Millisecond)
	}

	h.m.Append(7, "video", -1, seg[2], 1_000_000)
	h.m.End(7, "completed")
	resumed := h.open(64 << 20)
	if f.seq != 1 {
		t.Fatalf("the cursor is at entry %d", f.seq)
	}
	h.follow(resumed, f.seq, uint64(len(f.got)))
	h.until(resumed, &f, func() bool { return f.end != nil })
	whole := bytes.Join(seg, nil)
	if !bytes.Equal(f.got, whole) || f.end.SHA256 != digestOf(whole) || f.resets != 0 {
		t.Fatalf("resumed to %d bytes, %d resets; end %s", len(f.got), f.resets, f.end.Raw)
	}
}

func TestWebRTCFollowResetsWhenTheOutputIsReplaced(t *testing.T) {
	h := newMediaHarness(t)
	c := h.open(64 << 20)
	h.follow(c, 0, 0)
	var f mediaFollower
	h.m.Append(7, "video", -1, mediaSegment(100_000, 1), 1_000_000)
	h.until(c, &f, func() bool { return len(f.got) == 100_000 })
	cursor := f
	replaced := mediaSegment(80_000, 9)
	h.m.Replace(7, "video", -1, replaced, 1_000_000)
	h.until(c, &f, func() bool { return f.resets == 1 && bytes.Equal(f.got, replaced) })

	// A client that held the first revision resumes into the replacement: reset, then all of it.
	late := h.open(64 << 20)
	h.follow(late, cursor.seq, uint64(len(cursor.got)))
	h.until(late, &cursor, func() bool { return cursor.resets == 1 && bytes.Equal(cursor.got, replaced) })

	h.m.End(7, "canceled")
	h.until(c, &f, func() bool { return f.end != nil })
	if f.end.Status != "canceled" || f.end.Length != uint64(len(replaced)) || f.end.SHA256 != digestOf(replaced) {
		t.Fatalf("end %s", f.end.Raw)
	}
}
