package host

import (
	"net"
	"strconv"
)

// listen is the machine role's one bind site. It binds the machine endpoint (TLS with the
// machine's pinned leaf; every call carries a Claim or a capability), the receipt-only port a
// Hub older than M6 probes, the granted WebRTC port, and a loopback port for its own Runtime.
func listen(host string, port int) (net.Listener, error) {
	return net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port)))
}
