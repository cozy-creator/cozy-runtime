package host

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
)

func accessToken(t *testing.T, issuer, subject, nonce string) string {
	t.Helper()
	header, _ := json.Marshal(map[string]any{"typ": "delegated-access+jwt", "alg": "EdDSA"})
	payload, _ := json.Marshal(map[string]any{"iss": issuer, "delegated_sub": subject, "permissions": []string{"cozy.execution-access"}, "jti": nonce})
	return base64.RawURLEncoding.EncodeToString(header) + "." + base64.RawURLEncoding.EncodeToString(payload) + ".fixture-signature"
}

func TestAccessRefreshIsScopedAndContainsNoWorkerRegistration(t *testing.T) {
	public, key, _ := ed25519.GenerateKey(rand.Reader)
	claims, err := newClaims("local-machine", "boot", make([]byte, 32), []ed25519.PublicKey{public})
	if err != nil {
		t.Fatal(err)
	}
	m := &Machine{grant: &Grant{WorkerID: "local-machine"}, claims: claims, layout: NewLayout(&Grant{Root: t.TempDir()})}
	access := hubAccess{Origin: "https://example.test", Token: accessToken(t, "issuer", "account-a", "first"), ExpiresAt: time.Now().Add(time.Hour).Unix(), Environment: map[string]string{"TENSORHUB_ORIGIN": "https://example.test", "FUTURE_HUB_SETTING": "ignored", "PATH": "/untrusted"}}
	send := func(action string, a hubAccess) *httptest.ResponseRecorder {
		cap, _ := capability.Mint(key, capability.Grant{Machine: "local-machine", Action: action, Expires: time.Now().Add(time.Hour).Unix()})
		raw, _ := json.Marshal(a)
		r := httptest.NewRequest(http.MethodPost, "/v1/hubs/access", bytes.NewReader(raw))
		r.Header.Set("Authorization", "Cozy-Cap "+cap)
		w := httptest.NewRecorder()
		m.serveHubAccess(w, r)
		return w
	}
	if w := send(capability.Maintenance, access); w.Code != http.StatusForbidden {
		t.Fatalf("maintenance grant acquired Hub access: %d", w.Code)
	}
	if w := send(hubAccessAction, access); w.Code != 200 || strings.Contains(w.Body.String(), access.Token) {
		t.Fatalf("grant response: %d %s", w.Code, w.Body.String())
	}
	access.Token = accessToken(t, "issuer", "account-a", "refreshed")
	if w := send(hubAccessAction, access); w.Code != 200 {
		t.Fatalf("refresh: %s", w.Body.String())
	}
	raw, err := os.ReadFile(m.layout.boot("machine-hubs.json"))
	if err != nil {
		t.Fatal(err)
	}
	var projected struct {
		Hubs []map[string]any `json:"hubs"`
	}
	if err := json.Unmarshal(raw, &projected); err != nil {
		t.Fatal(err)
	}
	if len(projected.Hubs) != 1 || projected.Hubs[0]["access_token"] != access.Token || projected.Hubs[0]["worker_token"] != nil {
		t.Fatalf("bad projection: %s", raw)
	}
	if strings.Contains(string(raw), "/untrusted") || strings.Contains(string(raw), "FUTURE_HUB_SETTING") {
		t.Fatal("unknown settings leaked into Runtime environment")
	}
	if info, err := os.Stat(m.accessPath()); err != nil || info.Mode().Perm() != 0600 {
		t.Fatal("access configuration is not private")
	}
	retained, _ := os.ReadFile(m.accessPath())
	for _, tc := range []struct{ name, origin, token string }{
		{"account", access.Origin, accessToken(t, "issuer", "account-b", "changed")},
		{"issuer", access.Origin, accessToken(t, "other-issuer", "account-a", "changed")},
		{"origin-alias", "https://EXAMPLE.test:443", accessToken(t, "issuer", "account-b", "changed")},
		{"opaque", access.Origin, "different-opaque-token"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed := access
			changed.Origin, changed.Token = tc.origin, tc.token
			changed.Environment = map[string]string{"TENSORHUB_ORIGIN": tc.origin}
			w := send(hubAccessAction, changed)
			if w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), "hub_access_principal_conflict") || strings.Contains(w.Body.String(), changed.Token) {
				t.Fatalf("account switch was not safely refused: %d %s", w.Code, w.Body.String())
			}
			state, _ := os.ReadFile(m.accessPath())
			projection, _ := os.ReadFile(m.layout.boot("machine-hubs.json"))
			if !bytes.Equal(state, retained) || !bytes.Equal(projection, raw) {
				t.Fatal("refused account switch changed the retained grant")
			}
		})
	}
	// Expiry never turns old accepted work into permission to use another account.
	file, _ := readHubAccess(m.accessPath())
	file.Hubs[0].ExpiresAt = 1
	expired, _ := json.Marshal(file)
	if err := writeAtomic(m.accessPath(), expired, 0600); err != nil {
		t.Fatal(err)
	}
	changed := access
	changed.Token = accessToken(t, "issuer", "account-b", "expired-replacement")
	if w := send(hubAccessAction, changed); w.Code != http.StatusConflict {
		t.Fatal("expired old grant allowed another account")
	}
	if w := send(hubAccessAction, access); w.Code != http.StatusOK {
		t.Fatal("original account could not renew expired access")
	}
	alias := access
	alias.Origin = "https://example.test:443"
	alias.Environment = map[string]string{"TENSORHUB_ORIGIN": alias.Origin}
	if w := send(hubAccessAction, alias); w.Code != http.StatusOK {
		t.Fatal("same-account origin alias could not refresh")
	}
	file, _ = readHubAccess(m.accessPath())
	if len(file.Hubs) != 1 || file.Hubs[0].Origin != access.Origin {
		t.Fatal("an equivalent origin changed the retained Hub slot")
	}
	opaque := hubAccess{Origin: "https://other.test", Token: "opaque-one", ExpiresAt: access.ExpiresAt, Environment: map[string]string{"TENSORHUB_ORIGIN": "https://other.test"}}
	for range 2 {
		if w := send(hubAccessAction, opaque); w.Code != http.StatusOK {
			t.Fatal("opaque same-token replay was refused")
		}
	}
	opaque.Token = "opaque-two"
	if w := send(hubAccessAction, opaque); w.Code != http.StatusConflict {
		t.Fatal("opaque replacement invented authority continuity")
	}
	access.ExpiresAt = time.Now().Add(-time.Hour).Unix()
	if w := send(hubAccessAction, access); w.Code != 400 {
		t.Fatal("expired access accepted")
	}
	if _, err := os.Stat(filepath.Join(m.layout.State, "worker-registration.json")); !os.IsNotExist(err) {
		t.Fatal("access registered machine")
	}
}

func TestOpaqueAccessDoesNotInventAccountContinuity(t *testing.T) {
	if hubAccessPrincipal("opaque-one") != hubAccessPrincipal("opaque-one") || hubAccessPrincipal("opaque-one") == hubAccessPrincipal("opaque-two") {
		t.Fatal("opaque tokens lost exact-token continuity")
	}
	for _, token := range []string{"x.y.z", "e30.e30.signature", "..", strings.Repeat("x", 33<<10)} {
		if !strings.HasPrefix(hubAccessPrincipal(token), "opaque:") {
			t.Fatal("unknown JWT acquired delegated continuity")
		}
	}
}

func TestAccessProjectionCannotResurrectLegacyRegistration(t *testing.T) {
	l := NewLayout(&Grant{Root: t.TempDir()})
	if err := writeAtomic(l.boot("machine-hubs.json"), []byte(`{"version":1,"hubs":[{"origin":"https://example.test","worker_id":"om-old","worker_token":"retired"}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	err := projectHubAccess(hubAccessFile{Version: 1, Hubs: []hubAccess{{Origin: "https://example.test", Token: "new", ExpiresAt: 100}}}, l)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(l.boot("machine-hubs.json"))
	if strings.Contains(string(raw), "worker_token") || !strings.Contains(string(raw), `"access_token":"new"`) {
		t.Fatal("legacy machine registration shadowed access refresh")
	}
}

func TestOfflineAccessProjectionHasEmptyList(t *testing.T) {
	l := NewLayout(&Grant{Root: t.TempDir()})
	if err := projectHubAccess(hubAccessFile{Version: 1}, l); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(l.boot("machine-hubs.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `{"version":1,"hubs":[]}` {
		t.Fatalf("invalid offline access document: %s", raw)
	}
}

func TestForgetAccessRefusesRetainedWorkAndAllowsAccountSwitchAfterReset(t *testing.T) {
	public, key, _ := ed25519.GenerateKey(rand.Reader)
	claims, _ := newClaims("local-machine", "boot", make([]byte, 32), []ed25519.PublicKey{public})
	launcher := &idleMutationLauncher{before: false, after: false, answer: "accepted"}
	m := &Machine{grant: &Grant{WorkerID: "local-machine", Lifetime: "persistent"}, claims: claims, layout: NewLayout(&Grant{Root: t.TempDir()}), launcher: launcher, restarts: &restarts{}, log: io.Discard}
	old := hubAccess{Origin: "https://example.test", Token: accessToken(t, "issuer", "old-account", "old"), ExpiresAt: 1, Environment: map[string]string{"TENSORHUB_ORIGIN": "https://example.test"}}
	file := hubAccessFile{Version: 1, Hubs: []hubAccess{old}}
	before, _ := json.Marshal(file)
	if err := writeAtomic(m.accessPath(), before, 0600); err != nil {
		t.Fatal(err)
	}
	if err := projectHubAccess(file, m.layout); err != nil {
		t.Fatal(err)
	}
	request := func(action string) *httptest.ResponseRecorder {
		token, _ := capability.Mint(key, capability.Grant{Machine: "local-machine", Action: action, Expires: time.Now().Add(time.Hour).Unix()})
		r := httptest.NewRequest(http.MethodDelete, "/v1/hubs/access", strings.NewReader(`{"origin":"https://example.test"}`))
		r.Header.Set("Authorization", "Cozy-Cap "+token)
		w := httptest.NewRecorder()
		m.serveForgetHubAccess(w, r)
		return w
	}
	if w := request(capability.Maintenance); w.Code != 403 {
		t.Fatal("wrong scope", w.Code)
	}
	if w := request(hubAccessAction); w.Code != 409 || !strings.Contains(w.Body.String(), "machine_busy") {
		t.Fatal(w.Code, w.Body.String())
	}
	after, _ := os.ReadFile(m.accessPath())
	if !bytes.Equal(before, after) || launcher.restarts != 0 {
		t.Fatal("busy reset mutated retained access")
	}
	launcher.before, launcher.after = true, true
	// The fixture refuses relaunch: removal remains durable and the endpoint reports
	// that failure honestly. A real-process proof covers successful 204/relaunch.
	if w := request(hubAccessAction); w.Code != 503 {
		t.Fatal(w.Code, w.Body.String())
	}
	held, err := readHubAccess(m.accessPath())
	if err != nil || len(held.Hubs) != 0 {
		t.Fatal(held, err)
	}
	projection, _ := os.ReadFile(m.layout.boot("machine-hubs.json"))
	if strings.Contains(string(projection), old.Token) {
		t.Fatal("projection retained old account")
	}
	next := old
	next.Token = accessToken(t, "issuer", "new-account", "new")
	next.ExpiresAt = time.Now().Add(time.Hour).Unix()
	token, _ := capability.Mint(key, capability.Grant{Machine: "local-machine", Action: hubAccessAction, Expires: time.Now().Add(time.Hour).Unix()})
	raw, _ := json.Marshal(next)
	r := httptest.NewRequest(http.MethodPost, "/v1/hubs/access", bytes.NewReader(raw))
	r.Header.Set("Authorization", "Cozy-Cap "+token)
	w := httptest.NewRecorder()
	m.serveHubAccess(w, r)
	if w.Code != 200 {
		t.Fatal("reset did not permit account switch", w.Code, w.Body.String())
	}
}
