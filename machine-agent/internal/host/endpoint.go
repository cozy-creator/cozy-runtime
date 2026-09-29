package host

import (
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"google.golang.org/grpc"
)

// The one machine endpoint: gRPC and HTTPS on one TLS listener with the machine's leaf,
// told apart by content type. A Hub older than M6 also probes the receipt on the media port,
// and the Runtime proves that port refuses a foreign bearer, so that port serves only those.

const maxMessageBytes = 16 << 20

type listeners struct {
	servers []*http.Server
	failed  chan error
}

func (l *listeners) close() {
	for _, s := range l.servers {
		_ = s.Close()
	}
}

func (m *Machine) listen() (*listeners, error) {
	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{m.id.Leaf}}
	endpoint, err := listen(m.grant.ListenHost, m.grant.WorkerPort)
	if err != nil {
		return nil, fmt.Errorf("bind the machine endpoint: %w", err)
	}
	mediaHost, mediaPort := m.grant.ListenHost, m.grant.MediaPort
	if mediaPort == 0 {
		mediaHost = "127.0.0.1"
	}
	media, err := listen(mediaHost, mediaPort)
	if err != nil {
		endpoint.Close()
		return nil, fmt.Errorf("bind the receipt listener: %w", err)
	}
	m.mediaAddr = net.JoinHostPort("127.0.0.1", strconv.Itoa(media.Addr().(*net.TCPAddr).Port))
	grpcServer := grpc.NewServer(grpc.MaxRecvMsgSize(maxMessageBytes), grpc.MaxSendMsgSize(maxMessageBytes),
		grpc.UnknownServiceHandler(m.serveRPC))
	routes := http.NewServeMux()
	routes.HandleFunc("GET /v1/bootstrap/receipt", m.serveReceipt)
	routes.HandleFunc("GET /v1/health", serveHealth)
	routes.HandleFunc("GET /v1/runs/{run}/outputs/{output}", m.serveOutput)
	routes.HandleFunc("GET /v1/runs/{run}/outputs/{output}/{index}", m.serveOutput)
	routes.HandleFunc("GET /v1/machine/runtime", m.serveRuntimeState)
	routes.HandleFunc("PUT /v1/machine/runtime/wheels/{file}", m.serveStageWheel)
	routes.HandleFunc("POST /v1/machine/runtime/update", m.serveUpdate)
	main := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ProtoMajor == 2 && strings.HasPrefix(r.Header.Get("Content-Type"), "application/grpc") {
			grpcServer.ServeHTTP(w, r)
			return
		}
		routes.ServeHTTP(w, r)
	})
	receiptOnly := http.NewServeMux()
	receiptOnly.HandleFunc("GET /v1/bootstrap/receipt", m.serveReceipt)
	receiptOnly.HandleFunc("GET /v1/health", serveHealth)
	errorLog := log.New(m.log, "cozy machine: ", 0)
	l := &listeners{failed: make(chan error, 2)}
	for _, s := range []struct {
		listener net.Listener
		handler  http.Handler
	}{{endpoint, main}, {media, receiptOnly}} {
		server := &http.Server{Handler: s.handler, TLSConfig: tlsConfig.Clone(), ErrorLog: errorLog, ReadHeaderTimeout: time.Minute}
		l.servers = append(l.servers, server)
		go func(listener net.Listener) {
			if err := server.ServeTLS(listener, "", ""); err != nil && !errors.Is(err, http.ErrServerClosed) {
				l.failed <- fmt.Errorf("the machine endpoint stopped serving: %w", err)
			}
		}(s.listener)
	}
	return l, nil
}

// serveReceipt answers 425 until the envelope is sealed, launching a stopped Runtime as a call
// would, and 503 once the Runtime cannot start. Any answer tells the Hub the machine is up;
// the sealed envelope tells it the Runtime is ready.
func (m *Machine) serveReceipt(w http.ResponseWriter, _ *http.Request) {
	envelope := m.receipt.Envelope()
	w.Header().Set("Cache-Control", "no-store")
	if envelope == nil && m.restarts.gone() {
		body, _ := json.Marshal(map[string]any{"error": map[string]string{"code": "machine.runtime_gone", "message": m.runtimeGone()}})
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write(body)
		return
	}
	if envelope == nil {
		m.poke()
		http.Error(w, `{"error":{"code":"media.bootstrap_pending","message":"the machine is booting"}}`, http.StatusTooEarly)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(envelope)
}

// serveHealth carries no data. A request presenting a credential is refused: this machine
// has no bearer, and the Runtime's readiness proves exactly that.
func serveHealth(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Authorization") != "" {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
