package webrtc

import (
	"bufio"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/pion/datachannel"
	"github.com/pion/dtls/v3"
	"github.com/pion/logging"
	"github.com/pion/sctp"
	"github.com/pion/stun/v4"
)

// One TCP connection is one ICE-lite, passive ICE-TCP session: RFC 4571 frames told apart by
// their first byte (RFC 7983), a STUN responder, DTLS 1.2 with the machine's leaf, SCTP, and
// exactly one data channel. The browser synthesizes this end's SDP; nothing here parses SDP.

// UfragPrefix starts the ICE credentials every client chooses: ufrag = pwd.
const UfragPrefix = "cozy+webrtc+v1/"

var (
	errFraming = errors.New("the connection is not ICE-TCP from a cozy client")
	quiet      = &logging.DefaultLoggerFactory{Writer: io.Discard, DefaultLogLevel: logging.LogLevelDisabled}
)

// iceConn is the packet conn DTLS reads: its DTLS frames, with STUN answered on the way.
type iceConn struct {
	tcp   *net.TCPConn
	r     *bufio.Reader
	pwd   string                  // fixed by the first Binding request
	claim func(ufrag string) bool // admits the first Binding's ufrag to this connection
	wmu   sync.Mutex
}

// frame reads one RFC 4571 frame into p.
func (c *iceConn) frame(p []byte) (int, error) {
	var header [2]byte
	if _, err := io.ReadFull(c.r, header[:]); err != nil {
		return 0, err
	}
	n := int(binary.BigEndian.Uint16(header[:]))
	if n == 0 || n > len(p) {
		return 0, errFraming
	}
	return io.ReadFull(c.r, p[:n])
}

func (c *iceConn) ReadFrom(p []byte) (int, net.Addr, error) {
	for {
		n, err := c.frame(p)
		switch {
		case err != nil:
			return 0, nil, err
		case p[0] >= 20 && p[0] <= 63:
			return n, c.tcp.RemoteAddr(), nil
		case p[0] > 3:
			return 0, nil, errFraming
		}
		if err := c.answer(p[:n]); err != nil {
			return 0, nil, err
		}
	}
}

// answer replies to a Binding request (the first frame, then consent checks) keyed by the
// client-chosen password, which equals the local part of USERNAME.
func (c *iceConn) answer(frame []byte) error {
	m := &stun.Message{Raw: frame}
	var user stun.Username
	if m.Decode() != nil || m.Type != stun.BindingRequest || user.GetFrom(m) != nil {
		return errFraming
	}
	local, _, _ := strings.Cut(user.String(), ":")
	integrity := stun.NewShortTermIntegrity(local)
	if !strings.HasPrefix(local, UfragPrefix) || c.pwd != "" && local != c.pwd ||
		integrity.Check(m) != nil || stun.Fingerprint.Check(m) != nil || c.pwd == "" && !c.claim(local) {
		return errFraming
	}
	c.pwd = local
	peer := c.tcp.RemoteAddr().(*net.TCPAddr)
	reply, err := stun.Build(stun.NewTransactionIDSetter(m.TransactionID), stun.BindingSuccess,
		&stun.XORMappedAddress{IP: peer.IP, Port: peer.Port}, integrity, stun.Fingerprint)
	if err != nil {
		return err
	}
	_, err = c.WriteTo(reply.Raw, nil)
	return err
}

func (c *iceConn) WriteTo(p []byte, _ net.Addr) (int, error) {
	if len(p) > 0xffff {
		return 0, errFraming
	}
	header := binary.BigEndian.AppendUint16(nil, uint16(len(p)))
	c.wmu.Lock()
	defer c.wmu.Unlock()
	buffers := net.Buffers{header, p}
	if _, err := buffers.WriteTo(c.tcp); err != nil {
		return 0, err
	}
	return len(p), nil
}

func (c *iceConn) Close() error                       { return c.tcp.Close() }
func (c *iceConn) LocalAddr() net.Addr                { return c.tcp.LocalAddr() }
func (c *iceConn) SetDeadline(t time.Time) error      { return c.tcp.SetDeadline(t) }
func (c *iceConn) SetReadDeadline(t time.Time) error  { return c.tcp.SetReadDeadline(t) }
func (c *iceConn) SetWriteDeadline(t time.Time) error { return c.tcp.SetWriteDeadline(t) }

// channel is an open session's transport.
type channel struct {
	dtls    *dtls.Conn
	assoc   *sctp.Association
	dc      *datachannel.DataChannel
	binding string // the client's DTLS certificate, "sha-256 AB:…"
}

// connect runs one connection up to its open data channel. A first frame that is not a
// valid Binding request, or whose ufrag claim refuses, closes it unanswered, before any DTLS
// state exists.
func connect(ctx context.Context, tcp *net.TCPConn, leaf tls.Certificate, claim func(string) bool) (*channel, error) {
	ice := &iceConn{tcp: tcp, r: bufio.NewReaderSize(tcp, 64<<10), claim: claim}
	first := make([]byte, 1500)
	n, err := ice.frame(first)
	if err != nil || first[0] > 3 {
		return nil, errFraming
	}
	if err := ice.answer(first[:n]); err != nil {
		return nil, err
	}
	d, err := dtls.ServerWithOptions(ice, tcp.RemoteAddr(),
		dtls.WithCertificates(leaf),
		dtls.WithCipherSuites(dtls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256, dtls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384),
		dtls.WithExtendedMasterSecret(dtls.RequireExtendedMasterSecret),
		dtls.WithClientAuth(dtls.RequireAnyClientCert),
		// Browsers always offer use_srtp; a data-only server must still name a profile.
		dtls.WithSRTPProtectionProfiles(dtls.SRTP_AEAD_AES_128_GCM, dtls.SRTP_AES128_CM_HMAC_SHA1_80),
		// TCP and STUN have proven the address; a cookie round trip would add nothing.
		dtls.WithInsecureSkipVerifyHello(true),
		dtls.WithLoggerFactory(quiet))
	if err != nil {
		return nil, err
	}
	if err := d.HandshakeContext(ctx); err != nil {
		d.Close()
		return nil, err
	}
	state, _ := d.ConnectionState()
	if len(state.PeerCertificates) == 0 {
		d.Close()
		return nil, errFraming
	}
	a, err := sctp.ClientContext(ctx, sctp.WithNetConn(d), sctp.WithLoggerFactory(quiet), sctp.WithMTU(1191),
		sctp.WithMaxMessageSize(maxMessage), sctp.WithMaxReceiveBufferSize(4*maxMessage))
	if err != nil {
		d.Close()
		return nil, err
	}
	config := &datachannel.Config{LoggerFactory: quiet}
	dc, err := datachannel.Accept(a, config)
	if err != nil || dc.Config.Label != "cozy" || dc.Config.Protocol != "cozy/1" {
		a.Close()
		d.Close()
		return nil, fmt.Errorf("the client opened a channel other than cozy/1")
	}
	go func() { // exactly one channel: a second one ends the session
		if _, err := datachannel.Accept(a, config, dc); err == nil {
			a.Abort("a session has one channel")
		}
	}()
	return &channel{dtls: d, assoc: a, dc: dc, binding: Fingerprint(state.PeerCertificates[0])}, nil
}

// Fingerprint spells a certificate's pin as SDP's a=fingerprint does: "sha-256 AB:CD:…".
func Fingerprint(der []byte) string {
	sum := sha256.Sum256(der)
	var b strings.Builder
	b.WriteString("sha-256 ")
	for i, octet := range sum {
		if i > 0 {
			b.WriteByte(':')
		}
		fmt.Fprintf(&b, "%02X", octet)
	}
	return b.String()
}

// close ends the channel after what was written is acknowledged, or at once when ctx ends.
func (c *channel) close(ctx context.Context) {
	_ = c.assoc.Shutdown(ctx)
	_ = c.assoc.Close()
	_ = c.dtls.Close()
}
