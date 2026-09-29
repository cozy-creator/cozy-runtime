package host

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
)

// hubClient speaks the Hub's worker API as this machine: its worker id and token in the
// X-Cozy-Worker-* headers, JSON bodies, 204 for acceptance.
type hubClient struct {
	origin, workerID, token string
	http                    *http.Client
}

func newHubClient(g *Grant) *hubClient {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	if len(g.HubCA) > 0 {
		roots, err := x509.SystemCertPool()
		if err != nil || roots == nil {
			roots = x509.NewCertPool()
		}
		if ca, err := x509.ParseCertificate(g.HubCA); err == nil {
			roots.AddCert(ca)
		}
		transport.TLSClientConfig = &tls.Config{RootCAs: roots, MinVersion: tls.VersionTLS12}
	}
	return &hubClient{origin: g.HubOrigin, workerID: g.WorkerID, token: g.WorkerToken, http: &http.Client{Transport: transport}}
}

var refusalCode = regexp.MustCompile(`^[a-z][a-z0-9_.-]{0,127}$`)

func (h *hubClient) post(ctx context.Context, path string, body any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, h.origin+path, bytes.NewReader(raw))
	if err != nil {
		return err
	}
	request.Header.Set("X-Cozy-Worker-ID", h.workerID)
	request.Header.Set("X-Cozy-Worker-Token", h.token)
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", "application/json")
	response, err := h.http.Do(request)
	if err != nil {
		return fmt.Errorf("reach Tensorhub: %w", err)
	}
	defer response.Body.Close()
	answer, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
	if response.StatusCode == http.StatusNoContent {
		return nil
	}
	var refusal struct {
		Error struct{ Code, Message string } `json:"error"`
	}
	if json.Unmarshal(answer, &refusal) == nil && refusalCode.MatchString(refusal.Error.Code) {
		return fmt.Errorf("%s: %.512s (HTTP %d)", refusal.Error.Code, refusal.Error.Message, response.StatusCode)
	}
	return fmt.Errorf("Tensorhub answered HTTP %d", response.StatusCode)
}

func (h *hubClient) release(ctx context.Context) error {
	return h.post(ctx, "/v1/worker/rental/release", map[string]string{"worker_id": h.workerID})
}

func (h *hubClient) observeCache(ctx context.Context, o cacheObservation) error {
	if o.Scope == "" {
		o.Scope = "runtime"
	}
	return h.post(ctx, "/v1/worker/rental/cache-observations", o)
}

// modelSource is where a run of hub fetches its Models: "" or this machine's own Hub, or
// another Hub it is registered at (wire 67). A Hub it is not registered at has none.
func (m *Machine) modelSource(hub string) (hubSource, bool) {
	g, same := m.grant, func(a, b string) bool { return strings.TrimSuffix(a, "/") == strings.TrimSuffix(b, "/") }
	if hub == "" || same(hub, g.HubOrigin) {
		source := hubSource{hub: g.PublicOrigin, credential: "worker " + g.WorkerID + " " + g.WorkerToken, allowHosts: g.ObjectHosts, own: true}
		if len(g.HubCA) > 0 {
			source.caFile = m.layout.boot("tensorhub-ca.crt")
		}
		return source, true
	}
	for i, other := range g.Hubs {
		if same(hub, other.Origin) {
			source := hubSource{hub: other.PublicOrigin, credential: "worker " + other.WorkerID + " " + other.WorkerToken, allowHosts: other.ObjectHosts}
			if len(other.CA) > 0 {
				source.caFile = m.layout.boot(fmt.Sprintf("hub-ca-%d.crt", i+1))
			}
			return source, true
		}
	}
	return hubSource{}, false
}
