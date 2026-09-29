package webrtctest

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/netip"
	"strings"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host/webrtc"
	"github.com/pion/datachannel"
	"github.com/pion/ice/v4"
	pion "github.com/pion/webrtc/v4"
)

// Client is a full ICE agent reaching a machine as a browser does: active ICE-TCP, with the
// machine's answer synthesized from its address and fingerprint.
type Client struct {
	pc    *pion.PeerConnection
	dc    datachannel.ReadWriteCloser
	in    chan Message
	Cert  string // this client's DTLS certificate, "sha-256 AB:…", which a bound capability names
	Ufrag string // the ICE credential it chose, ufrag = pwd
}

// Message is one message from the machine: a cozy/1 record, or binary data.
type Message struct {
	T            string          `json:"t"`
	ID           json.RawMessage `json:"id"`
	Code         string          `json:"code"`
	Message      string          `json:"message"`
	Run          string          `json:"run"`
	Output       string          `json:"output"`
	Seq          uint64          `json:"seq"`
	Rev          uint64          `json:"rev"`
	Offset       uint64          `json:"offset"`
	Length       uint64          `json:"length"`
	AppendedFrom *uint64         `json:"appended_from"`
	DurationUS   uint64          `json:"duration_us"`
	SHA256       string          `json:"sha256"`
	ETag         string          `json:"etag"`
	Status       string          `json:"status"`
	Stream       uint32          `json:"stream"`
	Raw          string          `json:"-"`
	Data         []byte          `json:"-"` // a binary message's payload, at Offset of Stream
}

// Options tune the client's side of the path.
type Options struct {
	ReceiveBuffer uint32 // the SCTP receive window; 0 keeps pion's
	Oversend      bool   // ignore the machine's max-message-size, as a misbehaving client
}

// Dial connects to a machine's listener at addr, pinning its leaf fingerprint.
func Dial(ctx context.Context, addr netip.AddrPort, fingerprint string, opts Options) (*Client, error) {
	var se pion.SettingEngine
	se.SetNetworkTypes([]pion.NetworkType{pion.NetworkTypeTCP4})
	se.SetInterfaceFilter(func(name string) bool { return name == "lo" })
	se.SetIncludeLoopbackCandidate(true)
	se.SetICEMulticastDNSMode(ice.MulticastDNSModeDisabled)
	se.DetachDataChannels()
	if opts.ReceiveBuffer > 0 {
		se.SetSCTPMaxReceiveBufferSize(opts.ReceiveBuffer)
	}
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, err
	}
	cert, err := pion.GenerateCertificate(key)
	if err != nil {
		return nil, err
	}
	fingerprints, err := cert.GetFingerprints()
	if err != nil {
		return nil, err
	}
	pc, err := pion.NewAPI(pion.WithSettingEngine(se)).NewPeerConnection(pion.Configuration{Certificates: []pion.Certificate{*cert}})
	if err != nil {
		return nil, err
	}
	c := &Client{pc: pc, in: make(chan Message, 64), Cert: "sha-256 " + strings.ToUpper(fingerprints[0].Value), Ufrag: Ufrag()}
	if err := c.connect(ctx, addr, fingerprint, opts.Oversend); err != nil {
		pc.Close()
		return nil, err
	}
	go c.receive()
	return c, nil
}

func (c *Client) connect(ctx context.Context, addr netip.AddrPort, fingerprint string, oversend bool) error {
	protocol := "cozy/1"
	dc, err := c.pc.CreateDataChannel("cozy", &pion.DataChannelInit{Protocol: &protocol})
	if err != nil {
		return err
	}
	opened := make(chan error, 1)
	dc.OnOpen(func() {
		var err error
		c.dc, err = dc.Detach()
		opened <- err
	})
	offer, err := c.pc.CreateOffer(nil)
	if err != nil {
		return err
	}
	gathered := pion.GatheringCompletePromise(c.pc)
	if err := c.pc.SetLocalDescription(offer); err != nil {
		return err
	}
	<-gathered
	answer := Answer(offer.SDP, addr, fingerprint, c.Ufrag)
	if oversend {
		answer = strings.Replace(answer, "max-message-size:65536", "max-message-size:1048576", 1)
	}
	if err := c.pc.SetRemoteDescription(pion.SessionDescription{Type: pion.SDPTypeAnswer, SDP: answer}); err != nil {
		return err
	}
	select {
	case err := <-opened:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Ufrag is a client-chosen ICE credential, ufrag = pwd; base32 is within ICE's alphabet.
func Ufrag() string { return webrtc.UfragPrefix + rand.Text()[:22] }

// Answer synthesizes the machine's SDP answer to an offer, as the player page does.
func Answer(offer string, addr netip.AddrPort, fingerprint, ufrag string) string {
	mid := "0"
	for line := range strings.Lines(offer) {
		if value, ok := strings.CutPrefix(strings.TrimRight(line, "\r\n"), "a=mid:"); ok {
			mid = value
			break
		}
	}
	return strings.Join([]string{"v=0", "o=- 1 1 IN IP4 0.0.0.0", "s=-", "t=0 0", "a=group:BUNDLE " + mid, "a=ice-lite",
		"m=application 9 UDP/DTLS/SCTP webrtc-datachannel", "c=IN IP4 0.0.0.0", "a=mid:" + mid,
		"a=ice-ufrag:" + ufrag, "a=ice-pwd:" + ufrag, "a=fingerprint:" + fingerprint, "a=setup:passive",
		"a=sctp-port:5000", "a=max-message-size:65536",
		fmt.Sprintf("a=candidate:1 1 tcp 2130706431 %s %d typ host tcptype passive", addr.Addr(), addr.Port()),
		"a=end-of-candidates", ""}, "\r\n")
}

func (c *Client) receive() {
	defer close(c.in)
	buf := make([]byte, 1<<20)
	for {
		n, text, err := c.dc.ReadDataChannel(buf)
		if err != nil {
			return
		}
		var m Message
		switch {
		case text:
			m.Raw = string(buf[:n])
			if json.Unmarshal(buf[:n], &m) != nil {
				m.T = "?"
			}
		case n >= 12:
			m.T, m.Stream, m.Offset = "data", binary.BigEndian.Uint32(buf), binary.BigEndian.Uint64(buf[4:])
			m.Data = append([]byte(nil), buf[12:n]...)
		}
		c.in <- m
	}
}

// Recv is the next message; ok is false once the channel has closed.
func (c *Client) Recv(ctx context.Context) (Message, bool) {
	select {
	case m, ok := <-c.in:
		return m, ok
	case <-ctx.Done():
		return Message{T: "timeout"}, false
	}
}

// Send writes one cozy/1 record.
func (c *Client) Send(record any) error {
	raw, err := json.Marshal(record)
	if err != nil {
		return err
	}
	_, err = c.dc.WriteDataChannel(raw, true)
	return err
}

// SendRaw writes one message as it is.
func (c *Client) SendRaw(raw []byte, text bool) error {
	_, err := c.dc.WriteDataChannel(raw, text)
	return err
}

func (c *Client) Close() error { return c.pc.Close() }
