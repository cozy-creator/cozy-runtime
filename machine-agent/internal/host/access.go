package host

import (
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
)

const hubAccessAction = "hub-access"

type hubAccess struct {
	Origin      string            `json:"origin"`
	Token       string            `json:"token"`
	ExpiresAt   int64             `json:"expires_at"`
	Environment map[string]string `json:"environment"`
	CA          string            `json:"ca_der_b64url,omitempty"`
}

type hubAccessFile struct {
	Version int         `json:"version"`
	Hubs    []hubAccess `json:"hubs"`
}

// Access grants are scoped Hub credentials delegated to this machine's TLS leaf.
// They are independent of machine registration and can refresh without stopping work.
func (m *Machine) serveHubAccess(w http.ResponseWriter, r *http.Request) {
	token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Cozy-Cap ")
	grant, err := capability.Verify(token, m.grant.WorkerID, m.claims.authorizedKeys(), time.Now(), "")
	if err != nil || !grant.Permits(hubAccessAction) {
		http.Error(w, "a hub-access capability is required", http.StatusForbidden)
		return
	}
	var access hubAccess
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&access); err != nil {
		http.Error(w, "invalid Hub access grant", http.StatusBadRequest)
		return
	}
	access.Origin = strings.TrimSuffix(access.Origin, "/")
	if err := validateHubAccess(access, time.Now()); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	m.accessMu.Lock()
	defer m.accessMu.Unlock()
	file, err := readHubAccess(m.accessPath())
	if err != nil {
		http.Error(w, "cannot read the retained Hub access grants", http.StatusServiceUnavailable)
		return
	}
	replaced := false
	for i, old := range file.Hubs {
		if hubOriginKey(old.Origin) == hubOriginKey(access.Origin) {
			if hubAccessPrincipal(old.Token) != hubAccessPrincipal(access.Token) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusConflict)
				_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
					"code":    "hub_access_principal_conflict",
					"message": "this machine holds another account at this Hub; remove its access with DELETE /v1/hubs/access once accepted work is complete",
				}})
				return
			}
			file.Hubs[i] = access
			file.Hubs[i].Origin = old.Origin
			replaced = true
			break
		}
	}
	if !replaced {
		if len(file.Hubs) >= 64 {
			http.Error(w, "too many Hub access grants", http.StatusBadRequest)
			return
		}
		file.Hubs = append(file.Hubs, access)
	}
	raw, err := json.Marshal(file)
	if err == nil {
		err = writeAtomic(m.accessPath(), raw, 0600)
	}
	if err == nil {
		err = projectHubAccess(file, m.layout)
	}
	if err != nil {
		http.Error(w, "cannot retain the Hub access grant", http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"origin": access.Origin, "expires_at": access.ExpiresAt})
}

func hubOriginKey(origin string) string {
	u, err := url.Parse(origin)
	if err != nil {
		return origin
	}
	port := u.Port()
	if port == "" {
		port = "443"
		if u.Scheme == "http" {
			port = "80"
		}
	} else if number, err := strconv.Atoi(port); err == nil {
		port = strconv.Itoa(number)
	}
	return strings.ToLower(u.Scheme) + "://" + strings.ToLower(u.Hostname()) + ":" + port
}

// This is a continuity check, not JWT authentication: Tensorhub still verifies
// each request. Opaque or unfamiliar tokens may only refresh with identical bytes.
// The persisted grant itself retains the binding, including after it expires.
func hubAccessPrincipal(token string) string {
	fallback := func() string {
		digest := sha256.Sum256([]byte(token))
		return "opaque:" + hex.EncodeToString(digest[:])
	}
	parts := strings.Split(token, ".")
	if len(token) > 32<<10 || len(parts) != 3 || parts[2] == "" {
		return fallback()
	}
	header, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return fallback()
	}
	var kind struct {
		Type string `json:"typ"`
	}
	if json.Unmarshal(header, &kind) != nil || !strings.EqualFold(strings.TrimSpace(kind.Type), "delegated-access+jwt") {
		return fallback()
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return fallback()
	}
	var claims struct {
		Issuer      string   `json:"iss"`
		Subject     string   `json:"delegated_sub"`
		User        string   `json:"sub"`
		Permissions []string `json:"permissions"`
	}
	if json.Unmarshal(payload, &claims) != nil || claims.Issuer == "" || len(claims.Issuer) > 2048 || claims.Subject == "" || len(claims.Subject) > 256 || claims.User != "" || len(claims.Permissions) != 1 || claims.Permissions[0] != "cozy.execution-access" {
		return fallback()
	}
	identity, _ := json.Marshal([2]string{claims.Issuer, claims.Subject})
	return "delegated:" + string(identity)
}

func (m *Machine) accessPath() string { return filepath.Join(m.layout.State, "hub-access.json") }

func readHubAccess(path string) (hubAccessFile, error) {
	file := hubAccessFile{Version: 1}
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return file, nil
	}
	if err != nil {
		return file, err
	}
	if len(raw) > 4<<20 {
		return file, fmt.Errorf("Hub access file too large")
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		return file, err
	}
	return file, nil
}

func validateHubAccess(a hubAccess, now time.Time) error {
	if err := validateHubOrigin(a.Origin); err != nil {
		return err
	}
	if a.Token == "" || len(a.Token) > 32<<10 || strings.TrimSpace(a.Token) != a.Token || strings.ContainsAny(a.Token, "\r\n") {
		return fmt.Errorf("Hub access token is invalid")
	}
	if a.ExpiresAt <= now.Unix() {
		return fmt.Errorf("Hub access grant has expired")
	}
	if got := a.Environment["TENSORHUB_ORIGIN"]; strings.TrimSuffix(got, "/") != strings.TrimSuffix(a.Origin, "/") {
		return fmt.Errorf("Hub access environment names another origin")
	}
	// Project only the settings consumed below. Unknown Hub additions are never
	// applied to the process environment and do not invalidate an access grant.
	if a.CA != "" {
		der, err := base64.RawURLEncoding.DecodeString(a.CA)
		if err != nil {
			return fmt.Errorf("invalid Hub CA encoding")
		}
		if _, err := x509.ParseCertificate(der); err != nil {
			return fmt.Errorf("invalid Hub CA certificate")
		}
	}
	return nil
}

// projectHubAccess writes the Runtime's hot-readable grant document. State is the
// restartable configuration source; this projection never contains user login tokens.
func projectHubAccess(access hubAccessFile, layout Layout) error {
	var doc struct {
		Version int               `json:"version"`
		Hubs    []json.RawMessage `json:"hubs"`
	}
	doc.Version = 1
	doc.Hubs = make([]json.RawMessage, 0, len(access.Hubs))
	// Rebuild solely from scoped durable access; retired registration grants
	// must never reappear after reset or restart.
	for _, a := range access.Hubs {
		row := map[string]any{"origin": a.Origin, "access_token": a.Token, "expires_at": a.ExpiresAt, "object_storage_hosts": []string{}}
		if hosts := a.Environment["TENSORHUB_OBJECT_STORAGE_HOSTS"]; hosts != "" {
			row["object_storage_hosts"] = strings.Split(hosts, ",")
		}
		if public := a.Environment["TENSORHUB_PUBLIC_ORIGIN"]; public != "" {
			row["public_origin"] = public
		}
		if a.CA != "" {
			der, err := base64.RawURLEncoding.DecodeString(a.CA)
			if err != nil {
				return err
			}
			digest := sha256.Sum256([]byte(a.Origin))
			name := "hub-access-" + hex.EncodeToString(digest[:12]) + ".crt"
			if err := writeAtomic(layout.boot(name), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0600); err != nil {
				return err
			}
			row["ca"] = name
		}
		raw, err := json.Marshal(row)
		if err != nil {
			return err
		}
		doc.Hubs = append(doc.Hubs, raw)
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	return writeAtomic(layout.boot("machine-hubs.json"), raw, 0600)
}

func validateHubOrigin(value string) error {
	origin, err := url.Parse(value)
	if err != nil || origin.Host == "" || origin.User != nil || origin.RawQuery != "" || origin.Fragment != "" || origin.Path != "" && origin.Path != "/" || origin.Scheme != "https" && !(origin.Scheme == "http" && (origin.Hostname() == "localhost" || origin.Hostname() == "127.0.0.1" || origin.Hostname() == "::1")) {
		return fmt.Errorf("Hub origin must be an HTTPS origin or loopback HTTP origin")
	}

	return nil
}

// Explicit owner removal never invalidates access beneath accepted work. The
// Runtime closes admission before this callback changes either retained copy.
func (m *Machine) serveForgetHubAccess(w http.ResponseWriter, r *http.Request) {
	token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Cozy-Cap ")
	grant, err := capability.Verify(token, m.grant.WorkerID, m.claims.authorizedKeys(), time.Now(), "")
	if err != nil || !grant.Permits(hubAccessAction) {
		http.Error(w, "a hub-access capability is required", http.StatusForbidden)
		return
	}
	var request struct {
		Origin string `json:"origin"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
	if decoder.Decode(&request) != nil || decoder.Decode(new(any)) != io.EOF || validateHubOrigin(request.Origin) != nil {
		http.Error(w, "send one valid Hub origin", http.StatusBadRequest)
		return
	}
	if !m.launcherReady() || m.launcher == nil {
		http.Error(w, "machine_starting: maintenance is not ready", http.StatusServiceUnavailable)
		return
	}
	err = m.mutateIdleRuntime(r.Context(), func() error {
		m.accessMu.Lock()
		defer m.accessMu.Unlock()
		file, err := readHubAccess(m.accessPath())
		if err != nil {
			return err
		}
		file.Hubs = slices.DeleteFunc(file.Hubs, func(a hubAccess) bool { return hubOriginKey(a.Origin) == hubOriginKey(request.Origin) })
		raw, err := json.Marshal(file)
		if err != nil {
			return err
		}
		if err := writeAtomic(m.accessPath(), raw, 0600); err != nil {
			return err
		}
		return projectHubAccess(file, m.layout)
	})
	if err != nil {
		code, httpStatus := "hub_access_reset_failed", http.StatusServiceUnavailable
		message := status.Convert(err).Message()
		if status.Code(err) == codes.FailedPrecondition {
			prefix, _, _ := strings.Cut(message, ":")
			if prefix == "machine_busy" || prefix == "quiescence_unavailable" {
				code, httpStatus = prefix, http.StatusConflict
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(httpStatus)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"code": code, "message": message}})
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
