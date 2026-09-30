package host

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Rental authority is a Hub lease refreshed in the background; calls read only the
// cached lease, so a warm call never contacts the Hub. A transport failure neither
// revokes nor extends the last lease; an explicit Hub denial revokes at once.
// Losing authority never cancels work.
var errAuthorityDenied = errors.New("rental authority denied")

type rentalAuthority struct {
	gate       chan struct{}
	mu         sync.Mutex
	fetch      func(context.Context) ([]ed25519.PublicKey, time.Duration, error)
	timer      *time.Timer
	expiresAt  time.Time
	generation uint64
}

func (m *Machine) startAuthority(ctx context.Context) {
	if m.grant.Lifetime != "rental" {
		return
	}
	c := m.claims
	c.authority = &rentalAuthority{fetch: m.hub.authorizedKeys}
	c.replaceKeys(nil)
	go func() {
		delay := time.Duration(0)
		for {
			timer := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				// Shutdown proves nothing about the lease; its timer still ends it.
				timer.Stop()
				return
			case <-timer.C:
			}
			lease, err := c.refreshAuthority(ctx)
			if err != nil {
				delay = time.Second
			} else {
				delay = lease / 2
			}
		}
	}()
}

func (h *hubClient) authorizedKeys(ctx context.Context) ([]ed25519.PublicKey, time.Duration, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, h.origin+"/v1/worker/rental/authorized-keys", nil)
	if err != nil {
		return nil, 0, err
	}
	request.Header.Set("X-Cozy-Worker-ID", h.workerID)
	request.Header.Set("X-Cozy-Worker-Token", h.token)
	client := *h.http
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	response, err := client.Do(request)
	if err != nil {
		return nil, 0, err
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusUnauthorized || response.StatusCode == http.StatusForbidden {
		return nil, 0, errAuthorityDenied
	}
	if response.StatusCode != http.StatusOK {
		return nil, 0, fmt.Errorf("rental authority returned HTTP %d", response.StatusCode)
	}
	var document struct {
		WorkerID     string   `json:"worker_id"`
		Keys         []string `json:"authorized_keys"`
		LeaseSeconds int64    `json:"lease_seconds"`
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, (64<<10)+1))
	if err != nil || len(raw) > 64<<10 {
		return nil, 0, fmt.Errorf("rental authority metadata is unreadable or too large")
	}
	if json.Unmarshal(raw, &document) != nil || document.WorkerID != h.workerID || document.LeaseSeconds < 1 || document.LeaseSeconds > 3600 {
		return nil, 0, fmt.Errorf("rental authority metadata is invalid")
	}
	keys := make([]ed25519.PublicKey, 0, len(document.Keys))
	for _, encoded := range document.Keys {
		key, err := base64.RawURLEncoding.Strict().DecodeString(encoded)
		if err != nil || len(key) != ed25519.PublicKeySize {
			return nil, 0, fmt.Errorf("rental authority public key is invalid")
		}
		keys = append(keys, ed25519.PublicKey(key))
	}
	return keys, time.Duration(document.LeaseSeconds) * time.Second, nil
}

func (c *claims) replaceKeys(keys []ed25519.PublicKey) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if slices.EqualFunc(c.authorized, keys, func(a, b ed25519.PublicKey) bool { return slices.Equal(a, b) }) {
		return
	}
	c.authorized = slices.Clone(keys)
	close(c.changed)
	c.changed = make(chan struct{})
}

func (c *claims) refreshAuthority(ctx context.Context) (time.Duration, error) {
	if c.authority == nil {
		return 0, nil
	}
	a := c.authority
	a.mu.Lock()
	if a.gate == nil {
		a.gate = make(chan struct{}, 1)
		a.gate <- struct{}{}
	}
	gate := a.gate
	a.mu.Unlock()
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	case <-gate:
	}
	defer func() { gate <- struct{}{} }()
	// This is a bounded authority-read transport budget, never an execution limit.
	call, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	started := time.Now()
	keys, lease, err := a.fetch(call)
	lease -= time.Since(started)
	a.mu.Lock()
	defer a.mu.Unlock()
	denied := errors.Is(err, errAuthorityDenied)
	if err == nil && lease <= 0 {
		err = fmt.Errorf("rental authority lease expired in transit")
	}
	if err != nil && !denied {
		// A transport failure cannot revoke keys or extend their last valid lease.
		return 0, status.Error(codes.Unavailable, "rental_authority_unavailable: current rental authority could not be verified")
	}
	a.generation++
	generation := a.generation
	if a.timer != nil {
		a.timer.Stop()
	}
	if denied {
		a.expiresAt = time.Time{}
		c.replaceKeys(nil)
		return 0, status.Error(codes.Unavailable, "rental_authority_unavailable: the Hub denied rental authority")
	}
	c.replaceKeys(keys)
	a.expiresAt = time.Now().Add(lease)
	a.timer = time.AfterFunc(lease, func() {
		a.mu.Lock()
		defer a.mu.Unlock()
		if a.generation == generation {
			a.expiresAt = time.Time{}
			c.replaceKeys(nil)
		}
	})
	return lease, nil
}

// checkAuthority admits a new control from the cached lease; it never contacts the Hub.
func (c *claims) checkAuthority() error {
	if c.authority == nil {
		return nil
	}
	c.authority.mu.Lock()
	valid := !c.authority.expiresAt.IsZero() && time.Now().Before(c.authority.expiresAt)
	c.authority.mu.Unlock()
	if !valid {
		return status.Error(codes.Unavailable, "rental_authority_unavailable: no current rental authority lease")
	}
	return nil
}

// watchAuthority cancels only this client's transport when its key is removed.
func (c *claims) watchAuthority(ctx context.Context, keyID string) (context.Context, context.CancelFunc) {
	call, cancel := context.WithCancel(ctx)
	if c.authority == nil {
		return call, cancel
	}
	go func() {
		for {
			c.mu.Lock()
			present := slices.ContainsFunc(c.authorized, func(k ed25519.PublicKey) bool { return capability.KeyID(k) == keyID })
			changed := c.changed
			c.mu.Unlock()
			if !present {
				cancel()
				return
			}
			select {
			case <-call.Done():
				return
			case <-changed:
			}
		}
	}()
	return call, cancel
}
