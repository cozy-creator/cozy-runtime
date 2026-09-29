package host

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
)

func TestAccessRefreshIsScopedAndContainsNoWorkerRegistration(t *testing.T) {
	public, key, _ := ed25519.GenerateKey(rand.Reader)
	claims, err := newClaims("local-machine", "boot", make([]byte, 32), []ed25519.PublicKey{public})
	if err != nil {
		t.Fatal(err)
	}
	m := &Machine{grant: &Grant{WorkerID: "local-machine"}, claims: claims, layout: NewLayout(&Grant{Root: t.TempDir()})}
	access := hubAccess{Origin: "https://example.test", Token: "scoped-credential", ExpiresAt: time.Now().Add(time.Hour).Unix(), Environment: map[string]string{"TENSORHUB_ORIGIN": "https://example.test", "FUTURE_HUB_SETTING": "ignored", "PATH": "/untrusted"}}
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
	access.Token = "refreshed-credential"
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
	access.ExpiresAt = time.Now().Add(-time.Hour).Unix()
	if w := send(hubAccessAction, access); w.Code != 400 {
		t.Fatal("expired access accepted")
	}
	if _, err := os.Stat(filepath.Join(m.layout.State, "worker-registration.json")); !os.IsNotExist(err) {
		t.Fatal("access registered machine")
	}
}

func TestAccessProjectionReplacesLegacyCredentialForSameOrigin(t *testing.T) {
	l := NewLayout(&Grant{Root: t.TempDir()})
	err := projectHubAccess([]HubGrant{{Origin: "https://example.test", WorkerID: "om-old", WorkerToken: "old"}}, hubAccessFile{Version: 1, Hubs: []hubAccess{{Origin: "https://example.test", Token: "new", ExpiresAt: 100}}}, l)
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
	if err := projectHubAccess(nil, hubAccessFile{Version: 1}, l); err != nil {
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
