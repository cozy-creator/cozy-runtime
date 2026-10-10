package webrtc

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/netip"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host/outputs"
)

// cozy/1: JSON text messages, one typed record per `t` (unknown fields ignored), and binary
// data messages [u32 stream][u64 offset][payload] of at most 64 KiB.
const (
	header    = 12
	minWindow = 256 << 10
)

// session is one client connection: pending until its hello verifies, then serving follow
// and get requests over its one channel.
type session struct {
	srv    *server
	tcp    *net.TCPConn
	ip     netip.Addr
	ctx    context.Context
	abort  context.CancelFunc
	authed bool   // guarded by srv.mu
	key    string // guarded by srv.mu
	ufrag  string // claimed by the first Binding request

	ch    *channel
	grant capability.Grant // the reader's

	mu      sync.Mutex
	wake    chan struct{}
	out     []any     // session messages, sent before any stream's: welcome, error, bye
	streams []*stream // in turn order
	ending  bool      // a bye is queued: nothing else is sent or accepted
	closed  bool      // the sender stopped
	credit  uint64    // binary payload the client granted, cumulative
	sent    uint64    // binary payload sent
	number  uint32    // the last stream number

	// the sender's measurement of the path
	written, sampleAcked uint64
	sampleAt             time.Time
	minRTT               float64 // seconds
	rates                [8]float64
	sample               int
}

type stream struct {
	id     json.RawMessage
	number uint32
	get    bool
	items  []item
	stop   any // set when the stream ends early: what the sender sends in its place
	ctx    context.Context
	cancel context.CancelFunc
}

// item is a control message, or bytes [from, to) of body.
type item struct {
	msg      any
	last     bool // the stream ends with this message
	body     io.ReaderAt
	from, to int64
}

type request struct {
	T      string          `json:"t"`
	ID     json.RawMessage `json:"id"`
	Run    string          `json:"run"`
	Output string          `json:"output"`
	Index  *int            `json:"index"`
	Offset int64           `json:"offset"`
	Length int64           `json:"length"`
	After  uint64          `json:"after"`
	ETag   string          `json:"etag"`
	Bytes  uint64          `json:"bytes"`
}

type (
	welcomeMsg struct {
		T       string   `json:"t"`
		V       int      `json:"v"`
		Machine string   `json:"machine"`
		Run     string   `json:"run"`
		Outputs []string `json:"outputs,omitempty"`
		Expires int64    `json:"expires"`
	}
	entryMsg struct {
		T            string          `json:"t"`
		ID           json.RawMessage `json:"id"`
		Seq          uint64          `json:"seq"`
		Output       string          `json:"output"`
		Index        *int            `json:"index,omitempty"`
		Rev          uint64          `json:"rev"`
		Length       int64           `json:"length"`
		AppendedFrom *uint64         `json:"appended_from,omitempty"`
		DurationUS   uint64          `json:"duration_us,omitempty"`
		SHA256       string          `json:"sha256,omitempty"`
		MediaType    string          `json:"media_type,omitempty"`
		Label        string          `json:"label,omitempty"`
	}
	openMsg struct {
		T      string          `json:"t"`
		ID     json.RawMessage `json:"id"`
		Stream uint32          `json:"stream"`
		Offset int64           `json:"offset"`
		Length int64           `json:"length"`
		ETag   string          `json:"etag"`
	}
	resetMsg struct {
		T   string          `json:"t"`
		ID  json.RawMessage `json:"id"`
		Seq uint64          `json:"seq"`
	}
	endMsg struct {
		T      string          `json:"t"`
		ID     json.RawMessage `json:"id"`
		Status string          `json:"status,omitempty"`
		Length *int64          `json:"length,omitempty"`
		SHA256 string          `json:"sha256,omitempty"`
	}
	errorMsg struct {
		T       string          `json:"t"`
		ID      json.RawMessage `json:"id,omitempty"`
		Code    string          `json:"code"`
		Message string          `json:"message"`
	}
	byeMsg struct {
		T       string `json:"t"`
		Code    string `json:"code"`
		Message string `json:"message"`
	}
)

func newSession(s *server, tcp *net.TCPConn) *session {
	ctx, abort := context.WithCancel(context.Background())
	context.AfterFunc(ctx, func() { tcp.Close() })
	return &session{srv: s, tcp: tcp, ip: tcp.RemoteAddr().(*net.TCPAddr).AddrPort().Addr().Unmap(), ctx: ctx,
		abort: abort, wake: make(chan struct{}, 1)}
}

func (c *session) run() {
	defer c.srv.drop(c)
	defer c.abort()
	defer func() {
		if c.ufrag != "" {
			c.srv.release(c.ufrag)
		}
	}()
	ch, err := connect(c.ctx, c.tcp, c.srv.cfg.Leaf, func(ufrag string) bool {
		if !c.srv.claim(ufrag) {
			return false
		}
		c.ufrag = ufrag
		return true
	})
	if err != nil {
		return
	}
	c.ch, c.sampleAt = ch, time.Now()
	ch.dc.OnBufferedAmountLow(c.signal)
	go c.read()
	c.send()
	ch.close(c.ctx)
}

func (c *session) signal() {
	select {
	case c.wake <- struct{}{}:
	default:
	}
}

func (c *session) queue(msg any) {
	c.mu.Lock()
	c.out = append(c.out, msg)
	c.mu.Unlock()
	c.signal()
}

// bye ends the session once the message is sent.
func (c *session) bye(code, message string) {
	c.mu.Lock()
	if !c.ending {
		c.ending = true
		c.out = append(c.out, byeMsg{"bye", code, message})
	}
	c.mu.Unlock()
	c.signal()
}

// read handles the client's messages; one over 64 KiB ends the session.
func (c *session) read() {
	defer c.abort()
	buf := make([]byte, maxMessage)
	for {
		n, text, err := c.ch.dc.ReadDataChannel(buf)
		if err != nil {
			c.ch.assoc.Abort(err.Error()) // an SCTP ABORT tells a live peer at once
			return
		}
		c.mu.Lock()
		ending := c.ending
		c.mu.Unlock()
		switch {
		case ending:
		case c.grant.Machine == "":
			c.hello(buf[:n], text)
		default:
			c.handle(buf[:n], text)
		}
	}
}

// hello verifies the capability; nothing is read, listed or opened before it does.
func (c *session) hello(raw []byte, text bool) {
	var h struct {
		T   string `json:"t"`
		V   int    `json:"v"`
		Cap string `json:"cap"`
	}
	if !text || len(raw) > maxHello || json.Unmarshal(raw, &h) != nil || h.T != "hello" || h.V != 1 {
		c.bye("auth", "the first message must be hello{v: 1, cap}")
		return
	}
	if err := c.srv.cfg.Source.RefreshKeys(c.ctx); err != nil {
		c.bye("authority_unavailable", "current machine authority could not be verified")
		return
	}
	keys, _ := c.srv.cfg.Source.Keys()
	g, err := capability.Verify(h.Cap, c.srv.cfg.Machine, keys, time.Now(), c.ch.binding)
	switch {
	case errors.Is(err, capability.ErrExpired):
		c.bye("expired", err.Error())
	case err != nil:
		c.bye("auth", err.Error())
	case !c.srv.authenticate(c, g.Key):
		c.bye("auth", capability.ErrInvalid.Error())
	default:
		c.grant = g
		c.queue(welcomeMsg{"welcome", 1, g.Machine, g.Run, g.Outputs, g.Expires})
	}
}

func (c *session) handle(raw []byte, text bool) {
	var req request
	if !text || json.Unmarshal(raw, &req) != nil {
		c.queue(errorMsg{T: "error", Code: "bad_request", Message: "requests are JSON text messages"})
		return
	}
	switch req.T {
	case "credit":
		c.mu.Lock()
		c.credit = max(c.credit, req.Bytes)
		c.mu.Unlock()
		c.signal()
	case "cancel":
		c.mu.Lock()
		if i := slices.IndexFunc(c.streams, func(st *stream) bool { return string(st.id) == string(req.ID) }); i >= 0 && c.streams[i].stop == nil {
			c.streams[i].stop = endMsg{T: "end", ID: c.streams[i].id}
		}
		c.mu.Unlock()
		c.signal()
	case "follow", "get":
		c.request(req)
	default:
		c.queue(errorMsg{"error", req.ID, "bad_request", "unknown message type " + strconv.Quote(req.T)})
	}
}

func (c *session) request(req request) {
	if time.Now().Unix() >= c.grant.Expires {
		c.bye("expired", capability.ErrExpired.Error())
		return
	}
	index := -1
	if req.Index != nil {
		index = *req.Index
	}
	fail := func(code, message string) { c.queue(errorMsg{"error", req.ID, code, message}) }
	run, err := strconv.ParseUint(req.Run, 10, 64)
	switch {
	case len(req.ID) == 0 || string(req.ID) == "null" || err != nil || req.Output == "" || req.Index != nil && index < 0 ||
		req.Offset < 0 || req.Length < 0:
		fail("bad_request", "a request needs an id, a run number, an output, and no negative index, offset or length")
		return
	case !c.grant.Allows(req.Run, req.Output, index):
		fail("scope", capability.ErrScope.Error())
		return
	}
	c.mu.Lock()
	switch {
	case slices.ContainsFunc(c.streams, func(st *stream) bool { return string(st.id) == string(req.ID) }):
		c.mu.Unlock()
		fail("bad_request", "a request with this id is open")
		return
	case len(c.streams) >= maxStreams:
		c.mu.Unlock()
		fail("limit", "a session keeps at most 8 open requests")
		return
	}
	c.number++
	st := &stream{id: req.ID, number: c.number, get: req.T == "get"}
	st.ctx, st.cancel = context.WithCancel(c.ctx)
	c.streams = append(c.streams, st)
	c.mu.Unlock()
	serve := c.follow // off the reader: opening an output may wait on the Runtime
	if st.get {
		serve = c.get
	}
	go serve(st, run, req, index)
}

// push appends to a stream, or closes what it carries when the stream has ended. Entries
// go out as soon as they are journaled, ahead of every stream's bytes, so a follower learns
// an output's whole map at once; everything else keeps byte order.
func (c *session) push(st *stream, items ...item) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if st.stop != nil || c.closed {
		for _, it := range items {
			closeBody(it.body)
		}
		return false
	}
	for _, it := range items {
		if _, eager := it.msg.(entryMsg); eager {
			c.out = append(c.out, it.msg)
		} else {
			st.items = append(st.items, it)
		}
	}
	c.signal()
	return true
}

func (c *session) stop(st *stream, msg any) {
	c.mu.Lock()
	if st.stop == nil {
		st.stop = msg
	}
	c.mu.Unlock()
	c.signal()
}

// get serves one byte range of an output's current bytes; a stale etag gets `changed`.
func (c *session) get(st *stream, run uint64, req request, index int) {
	snap, err := c.srv.cfg.Source.Open(run, req.Output, index)
	if err != nil {
		c.stop(st, sourceError(st.id, err))
		return
	}
	etag := "r" + strconv.FormatUint(snap.Rev, 10)
	end := snap.Length
	if req.Length > 0 && req.Length < end-req.Offset {
		end = req.Offset + req.Length
	}
	switch {
	case req.ETag != "" && req.ETag != etag:
		closeBody(snap.Body)
		c.stop(st, errorMsg{"error", st.id, "changed", "the output is at " + etag})
		return
	case req.Offset > snap.Length:
		closeBody(snap.Body)
		c.stop(st, errorMsg{"error", st.id, "bad_request", "the offset is past the output's length"})
		return
	}
	items := []item{{msg: openMsg{"open", st.id, st.number, req.Offset, end - req.Offset, etag}}}
	if end > req.Offset {
		items = append(items, item{body: snap.Body, from: req.Offset, to: end})
	} else {
		closeBody(snap.Body)
	}
	c.push(st, append(items, item{msg: endMsg{T: "end", ID: st.id, Length: &snap.Length, SHA256: snap.SHA256}, last: true})...)
}

// follow streams an output from the client's cursor (after, offset) as the log announces its
// revisions, until the run's terminal. The pod never reads past the length the latest entry
// committed.
func (c *session) follow(st *stream, run uint64, req request, index int) {
	src := c.srv.cfg.Source
	after, held := req.After, req.Offset
	var pending []outputs.Entry
	for {
		batch, err := src.Entries(st.ctx, run, after)
		if err != nil {
			c.stop(st, sourceError(st.id, err))
			return
		}
		var terminal *outputs.Entry
		for i, e := range batch {
			after = max(after, e.Seq)
			if e.Status != "" {
				terminal = &batch[i]
				break
			}
			if e.Output == req.Output && e.Index == index {
				pending = append(pending, e)
			}
		}
		if len(pending) > 0 {
			last := pending[len(pending)-1]
			snap, err := src.Open(run, req.Output, index)
			if err != nil {
				c.stop(st, sourceError(st.id, err))
				return
			}
			if snap.Rev > last.Rev && terminal == nil { // the log holds newer entries: read them first
				closeBody(snap.Body)
				continue
			}
			if snap.Rev != last.Rev {
				closeBody(snap.Body)
				c.stop(st, errorMsg{"error", st.id, "unavailable", "the output's bytes and its log disagree"})
				return
			}
			var items []item
			items, held = emit(st, pending, snap, held)
			pending = nil
			if !c.push(st, items...) {
				return
			}
		}
		if terminal != nil {
			end := endMsg{T: "end", ID: st.id, Status: terminal.Status}
			if snap, err := src.Open(run, req.Output, index); err == nil {
				closeBody(snap.Body)
				end.Length, end.SHA256 = &snap.Length, snap.SHA256
			}
			c.push(st, item{msg: end, last: true})
			return
		}
	}
}

// emit turns the followed output's new entries into what the client lacks: a reset when a
// replacement voids the bytes it holds, the entries still current, then the bytes past held.
func emit(st *stream, pending []outputs.Entry, snap outputs.Snapshot, held int64) ([]item, int64) {
	from, replaced := 0, false
	for i, e := range pending {
		if e.AppendedFrom == nil {
			from, replaced = i, true
		}
	}
	last := pending[len(pending)-1]
	var items []item
	if replaced && held > 0 || held > last.Length {
		items, held = append(items, item{msg: resetMsg{"reset", st.id, pending[from].Seq}}), 0
	}
	for _, e := range pending[from:] {
		m := entryMsg{"entry", st.id, e.Seq, e.Output, nil, e.Rev, e.Length, e.AppendedFrom, e.DurationUS, e.SHA256, e.MediaType, e.Label}
		if e.Index >= 0 {
			m.Index = &e.Index
		}
		items = append(items, item{msg: m})
	}
	if held == last.Length {
		closeBody(snap.Body)
		return items, held
	}
	etag := "r" + strconv.FormatUint(last.Rev, 10)
	return append(items, item{msg: openMsg{"open", st.id, st.number, held, last.Length - held, etag}},
		item{body: snap.Body, from: held, to: last.Length}), last.Length
}

func sourceError(id json.RawMessage, err error) errorMsg {
	code := "unavailable"
	switch {
	case errors.Is(err, outputs.ErrNotFound):
		code = "not_found"
	case errors.Is(err, outputs.ErrUpdateRequired):
		code = "runtime_update_required"
	}
	return errorMsg{"error", id, code, err.Error()}
}

func closeBody(body io.ReaderAt) {
	if closer, ok := body.(io.Closer); ok {
		closer.Close()
	}
}

// work is one write of the sender.
type work struct {
	msg    any
	st     *stream
	body   io.ReaderAt
	from   int64
	n      int64
	finish bool // the body is done after this chunk
}

// send is the session's one writer. Session messages go first, then streams' control
// messages, then one chunk of data at a time while credit and the window allow: gets before
// follows, and streams of a class take turns.
func (c *session) send() {
	defer c.shut()
	buf := make([]byte, maxMessage)
	for {
		w, ok := c.next()
		if !ok {
			return
		}
		var err error
		if w.msg != nil {
			raw, _ := json.Marshal(w.msg)
			_, err = c.ch.dc.WriteDataChannel(raw, true)
			c.written += uint64(len(raw))
			if _, bye := w.msg.(byeMsg); bye {
				return
			}
		} else {
			binary.BigEndian.PutUint32(buf, w.st.number)
			binary.BigEndian.PutUint64(buf[4:], uint64(w.from))
			n, readErr := w.body.ReadAt(buf[header:header+w.n], w.from)
			if w.finish {
				closeBody(w.body)
			}
			if int64(n) < w.n {
				c.stop(w.st, sourceError(w.st.id, readErr))
				continue
			}
			_, err = c.ch.dc.WriteDataChannel(buf[:header+w.n], false)
			c.written += uint64(header + w.n)
		}
		if err != nil {
			return
		}
	}
}

func (c *session) next() (work, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for {
		c.streams = slices.DeleteFunc(c.streams, func(st *stream) bool {
			if st.stop == nil {
				return false
			}
			for _, it := range st.items {
				closeBody(it.body)
			}
			st.cancel()
			c.out = append(c.out, st.stop)
			return true
		})
		if len(c.out) > 0 {
			msg := c.out[0]
			c.out = c.out[1:]
			return work{msg: msg}, true
		}
		if !c.ending {
			for i, st := range c.streams {
				if len(st.items) > 0 && st.items[0].body == nil {
					it := st.items[0]
					st.items = st.items[1:]
					if it.last {
						c.streams = slices.Delete(c.streams, i, i+1)
						st.cancel()
					}
					return work{msg: it.msg}, true
				}
			}
			if st := c.pick(); st != nil && c.sent < c.credit && c.windowOpen() {
				it := &st.items[0]
				n := min(maxMessage-header, it.to-it.from, int64(min(c.credit-c.sent, maxMessage)))
				w := work{st: st, body: it.body, from: it.from, n: n}
				it.from += n
				c.sent += uint64(n)
				if it.from == it.to {
					w.finish = true
					st.items = st.items[1:]
				}
				c.streams = append(slices.DeleteFunc(c.streams, func(o *stream) bool { return o == st }), st)
				return w, true
			}
		}
		c.mu.Unlock()
		select {
		case <-c.wake:
			c.mu.Lock()
		case <-c.ctx.Done():
			c.mu.Lock()
			return work{}, false
		}
	}
}

// pick is the next stream with data at its head: gets first, in turn order.
func (c *session) pick() *stream {
	var follow *stream
	for _, st := range c.streams {
		if len(st.items) == 0 || st.items[0].body == nil {
			continue
		}
		if st.get {
			return st
		}
		if follow == nil {
			follow = st
		}
	}
	return follow
}

// windowOpen says whether the sender may queue more. The channel's unacknowledged bytes stay
// under max(256 KiB, 2 × delivery rate × min SRTT), measured as acknowledgements arrive, so
// the queue in this machine's TCP buffer never outgrows the path: over TCP, SCTP sees no
// loss, and a queue longer than its RTO would collapse it.
func (c *session) windowOpen() bool {
	buffered := c.ch.dc.BufferedAmount()
	if rtt := c.ch.assoc.SRTT() / 1000; rtt > 0 && (c.minRTT == 0 || rtt < c.minRTT) {
		c.minRTT = rtt
	}
	if elapsed := time.Since(c.sampleAt).Seconds(); c.minRTT > 0 && elapsed >= c.minRTT {
		acked := c.written - buffered
		c.rates[c.sample%len(c.rates)] = float64(acked-c.sampleAcked) / elapsed
		c.sample++
		c.sampleAt, c.sampleAcked = time.Now(), acked
	}
	window := max(minWindow, uint64(2*slices.Max(c.rates[:])*c.minRTT))
	c.ch.dc.SetBufferedAmountLowThreshold(window - maxMessage) // before the check: a drop after it signals
	return c.ch.dc.BufferedAmount() < window
}

// shut stops the sender: nothing more is queued, and every body is closed.
func (c *session) shut() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	for _, st := range c.streams {
		for _, it := range st.items {
			closeBody(it.body)
		}
		st.cancel()
	}
	c.streams = nil
}
