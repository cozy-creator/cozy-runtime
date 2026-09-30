package host

import (
	"context"
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

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
	"google.golang.org/grpc"
)

// The one machine endpoint: gRPC and HTTPS on one TLS listener with the machine's leaf,
// told apart by content type. A Hub older than M6 also probes the receipt on the media port,
// and the Runtime proves that port refuses a foreign bearer, so that port serves only those.

const maxMessageBytes = 16 << 20

type listeners struct {
	cancelAuthority context.CancelFunc
	servers         []*http.Server
	failed          chan error
}

func (l *listeners) close() {
	if l.cancelAuthority != nil {
		l.cancelAuthority()
	}
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
	if m.grant.bootstrap != nil {
		_, port, _ := net.SplitHostPort(m.grant.bootstrap.MediaAddr)
		mediaPort, _ = strconv.Atoi(port)
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
	routes.HandleFunc("POST /v1/hubs/access", m.serveHubAccess)
	routes.HandleFunc("DELETE /v1/hubs/access", m.serveForgetHubAccess)
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
		m.serveAuthorizedHTTP(routes, w, r)
	})
	receiptOnly := http.NewServeMux()
	receiptOnly.HandleFunc("GET /v1/bootstrap/receipt", m.serveReceipt)
	receiptOnly.HandleFunc("GET /v1/health", serveHealth)
	errorLog := log.New(m.log, "cozy machine: ", 0)
	authorityContext, cancelAuthority := context.WithCancel(context.Background())
	m.startAuthority(authorityContext)
	l := &listeners{failed: make(chan error, 2), cancelAuthority: cancelAuthority}
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

// serveReceipt observes readiness without waking or launching a Runtime. It answers
// 425 until the envelope is sealed and 503 once the Runtime cannot start.
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

// Stop copying output bytes after this reader loses authority; execution remains durable.
type authorityWriter struct {
	http.ResponseWriter
	ctx context.Context
}

func (w *authorityWriter) Write(data []byte) (int, error) {
	if err := w.ctx.Err(); err != nil {
		return 0, err
	}
	return w.ResponseWriter.Write(data)
}
func (w *authorityWriter) Flush() {
	if w.ctx.Err() != nil {
		return
	}
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (m *Machine) serveAuthorizedHTTP(routes http.Handler, w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/v1/bootstrap/receipt" && r.URL.Path != "/v1/health" {
		if err := m.claims.checkAuthority(); err != nil {
			http.Error(w, "rental_authority_unavailable: current rental authority could not be verified", http.StatusServiceUnavailable)
			return
		}
		token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Cozy-Cap ")
		grant, err := capability.Verify(token, m.grant.WorkerID, m.claims.authorizedKeys(), time.Now(), "")
		if err != nil {
			http.Error(w, "a current machine capability is required", http.StatusForbidden)
			return
		}
		call, cancel := m.claims.watchAuthority(r.Context(), grant.Key)
		defer cancel()
		r = r.WithContext(call)
		w = &authorityWriter{ResponseWriter: w, ctx: call}
		go func() { <-call.Done(); _ = r.Body.Close() }()
	}
	routes.ServeHTTP(w, r)
}
