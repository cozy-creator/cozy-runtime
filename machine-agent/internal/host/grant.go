// Package host is the machine agent: it supervises this machine's Runtime,
// proves readiness to the Hub, releases the machine when it idles, and serves the one
// machine endpoint that clients, the Hub and players reach. A rented pod runs it as PID 1;
// this computer runs the same code.
package host

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// Grant is the machine's launch contract: the environment a Hub or this computer's launcher
// gives it. Only the names read here matter; any other is logged and ignored, so a newer Hub
// cannot stop an older machine from booting.
type Grant struct {
	agentBefore    string
	agentSelection string
	startupPending bool
	bootstrap      *bootstrapChild
	Lifetime       string // persistent (standalone) or rental (provider-owned allocation)
	Root           string // COZY_MACHINE_ROOT; the image layout is rooted here
	StoreRoot      string // COZY_TENSORFS_ROOT; empty keeps <root>/var/lib/tensorfs
	ListenHost     string
	WorkerID       string
	WorkerToken    string
	WorkerPort     int
	MediaPort      int // a pre-M6 Hub probes the receipt here; 0 when none was granted
	WebRTCPort     int // browsers reach media over WebRTC here; 0 when none was granted
	HubOrigin      string
	HubCA          []byte // DER
	PublicOrigin   string
	ObjectHosts    []string
	RepoCacheRoot  string
	// Authorized are the device keys whose Claims and capabilities this machine accepts.
	Authorized []ed25519.PublicKey
	// ObservedAuth is the auth document a pre-M6 Hub froze; the receipt must repeat it.
	ObservedAuth *OwnerAuth
	// Development enables SSH with DeveloperKey.
	Development  bool
	DeveloperKey string
	receiptKey   []byte
	inherited    []string // locale and trust-store settings the Runtime keeps
	Ignored      []string
}

// HubGrant is this machine's registration at another Hub.
type HubGrant struct {
	Origin, PublicOrigin, WorkerID, WorkerToken string
	CA                                          []byte // DER
	ObjectHosts                                 []string
}

// OwnerAuth is COZY_RECORD_OWNER_AUTH_JSON.
type OwnerAuth struct {
	ControlKey  string   `json:"control_public_key_ed25519_b64url"`
	MediaTokens []string `json:"media_token_sha256"`
}

const receiptKeyName = "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL"
const receiptKeyFileName = "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE"

var grantNames = map[string]bool{
	"COZY_MACHINE_LIFETIME": true, "COZY_MACHINE_ROOT": true, "COZY_TENSORFS_ROOT": true, "COZY_LISTEN_HOST": true,
	"COZY_WORKER_ID": true, "COZY_WORKER_AUTH_TOKEN": true, "COZY_WORKER_INTERNAL_PORT": true,
	"COZY_MEDIA_INTERNAL_PORT": true, "COZY_WEBRTC_INTERNAL_PORT": true, "COZY_RECORD_OWNER_AUTH_JSON": true, "COZY_AUTHORIZED_KEYS": true,
	"COZY_REPO_CACHE_ROOT": true, receiptKeyName: true, receiptKeyFileName: true, "TENSORHUB_ORIGIN": true,
	"TENSORHUB_CA_DER_B64URL": true, "TENSORHUB_PUBLIC_ORIGIN": true, "TENSORHUB_OBJECT_STORAGE_HOSTS": true,
}

// HasGrant says whether the environment names a machine at all.
func HasGrant(environ []string) bool {
	for _, entry := range environ {
		if strings.HasPrefix(entry, "COZY_WORKER_ID=") {
			return true
		}
	}
	return false
}

// ReadGrant parses the environment. It unsets the readiness key the moment it is read: the
// key signs one envelope and must not reach any child.
func ReadGrant(environ []string) (*Grant, error) {
	env := map[string]string{}
	g := &Grant{}
	for _, entry := range environ {
		name, value, _ := strings.Cut(entry, "=")
		env[name] = value
		if (strings.HasPrefix(name, "COZY_") || strings.HasPrefix(name, "TENSORHUB_")) && !grantNames[name] {
			g.Ignored = append(g.Ignored, name)
		}
	}
	for _, name := range []string{receiptKeyName, receiptKeyFileName} {
		if err := os.Unsetenv(name); err != nil {
			return nil, err
		}
	}
	var err error
	g.Root = cmpOr(env["COZY_MACHINE_ROOT"], "/")
	if g.StoreRoot = env["COZY_TENSORFS_ROOT"]; g.StoreRoot != "" && !cleanAbs(g.StoreRoot) {
		return nil, fmt.Errorf("COZY_TENSORFS_ROOT must be one clean absolute directory")
	}
	if !cleanAbs(g.Root) {
		return nil, fmt.Errorf("COZY_MACHINE_ROOT must be one clean absolute directory")
	}
	switch g.ListenHost = cmpOr(env["COZY_LISTEN_HOST"], "0.0.0.0"); g.ListenHost {
	case "0.0.0.0", "127.0.0.1":
	default:
		return nil, fmt.Errorf("COZY_LISTEN_HOST must be 0.0.0.0 or 127.0.0.1")
	}
	if g.WorkerID = env["COZY_WORKER_ID"]; g.WorkerID == "" || len(g.WorkerID) > 256 || strings.TrimSpace(g.WorkerID) != g.WorkerID {
		return nil, fmt.Errorf("COZY_WORKER_ID must be one identifier of at most 256 bytes")
	}
	g.Lifetime = cmpOr(env["COZY_MACHINE_LIFETIME"], "rental")
	if g.Lifetime != "persistent" && g.Lifetime != "rental" {
		return nil, fmt.Errorf("COZY_MACHINE_LIFETIME must be persistent or rental")
	}
	if g.Lifetime == "rental" {
		g.WorkerToken = env["COZY_WORKER_AUTH_TOKEN"]
		if len(decode64(g.WorkerToken)) != 32 {
			return nil, fmt.Errorf("COZY_WORKER_AUTH_TOKEN must be 32 bytes of unpadded base64url")
		}
	}
	if g.WorkerPort, err = port(env, "COZY_WORKER_INTERNAL_PORT", true); err != nil {
		return nil, err
	}
	if g.MediaPort, err = port(env, "COZY_MEDIA_INTERNAL_PORT", false); err != nil {
		return nil, err
	}
	if g.WebRTCPort, err = port(env, "COZY_WEBRTC_INTERNAL_PORT", false); err != nil {
		return nil, err
	}
	if g.MediaPort == g.WorkerPort {
		return nil, fmt.Errorf("the worker and media ports must differ")
	}
	if g.Lifetime == "rental" {
		own, err := hubGrant(env, true)
		if err != nil {
			return nil, err
		}
		g.HubOrigin, g.PublicOrigin, g.HubCA, g.ObjectHosts = own.Origin, own.PublicOrigin, own.CA, own.ObjectHosts
	}
	// Persistent machines never consume copied legacy registration settings.
	// Content access is explicitly attached as a scoped bearer.
	g.RepoCacheRoot = env["COZY_REPO_CACHE_ROOT"]
	if text := env["COZY_RECORD_OWNER_AUTH_JSON"]; text != "" {
		g.ObservedAuth = &OwnerAuth{}
		if err := json.Unmarshal([]byte(text), g.ObservedAuth); err != nil {
			return nil, fmt.Errorf("COZY_RECORD_OWNER_AUTH_JSON is not an auth document: %w", err)
		}
		key, err := publicKey(g.ObservedAuth.ControlKey)
		if err != nil {
			return nil, fmt.Errorf("COZY_RECORD_OWNER_AUTH_JSON: %w", err)
		}
		g.Authorized = append(g.Authorized, key)
	}
	if text := env["COZY_AUTHORIZED_KEYS"]; text != "" {
		g.Authorized = nil
		for _, spelled := range strings.Split(text, ",") {
			key, err := publicKey(spelled)
			if err != nil {
				return nil, fmt.Errorf("COZY_AUTHORIZED_KEYS: %w", err)
			}
			g.Authorized = append(g.Authorized, key)
		}
	}
	if len(g.Authorized) == 0 {
		return nil, fmt.Errorf("the grant authorizes no key: COZY_AUTHORIZED_KEYS or COZY_RECORD_OWNER_AUTH_JSON is required")
	}
	key := env[receiptKeyName]
	if path := env[receiptKeyFileName]; path != "" {
		if key != "" {
			return nil, fmt.Errorf("specify only one bootstrap receipt key source")
		}
		if !cleanAbs(path) {
			return nil, fmt.Errorf("%s must be an absolute private file", receiptKeyFileName)
		}
		info, err := os.Lstat(path)
		if err != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0077 != 0 {
			return nil, fmt.Errorf("%s requires a private regular file", receiptKeyFileName)
		}
		file, err := os.Open(path)
		if err != nil {
			return nil, fmt.Errorf("open receipt key file: %w", err)
		}
		opened, statErr := file.Stat()
		if statErr != nil || !os.SameFile(info, opened) {
			file.Close()
			return nil, fmt.Errorf("receipt key file changed while opening")
		}
		raw, readErr := io.ReadAll(io.LimitReader(file, 256))
		file.Close()
		if readErr != nil {
			return nil, fmt.Errorf("read receipt key file: %w", readErr)
		}
		key = strings.TrimSpace(string(raw))
		if key == "" {
			return nil, fmt.Errorf("receipt key file is empty")
		}
	}
	if key != "" {
		if g.receiptKey = decode64(key); len(g.receiptKey) != 32 {
			return nil, fmt.Errorf("bootstrap receipt key must be 32 bytes of unpadded base64url")
		}
	}
	switch env["WORKER_MODE"] {
	case "development":
		g.Development, g.DeveloperKey = true, env["PUBLIC_KEY"]
	}
	slices.Sort(g.Ignored)
	return g, nil
}

// hubGrant reads one Hub's TENSORHUB_* settings.
func hubGrant(env map[string]string, required bool) (HubGrant, error) {
	var h HubGrant
	var err error
	if h.Origin, err = origin(env, "TENSORHUB_ORIGIN", required); err != nil {
		return h, err
	}
	if h.PublicOrigin, err = origin(env, "TENSORHUB_PUBLIC_ORIGIN", false); err != nil {
		return h, err
	}
	if text := env["TENSORHUB_CA_DER_B64URL"]; text != "" {
		if h.CA = decode64(text); h.CA == nil {
			return h, fmt.Errorf("TENSORHUB_CA_DER_B64URL must be unpadded base64url DER")
		}
	}
	if hosts := env["TENSORHUB_OBJECT_STORAGE_HOSTS"]; hosts != "" {
		h.ObjectHosts = strings.Split(hosts, ",")
	}
	return h, nil
}

func publicKey(spelled string) (ed25519.PublicKey, error) {
	raw := decode64(spelled)
	if len(raw) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("%q is not a 32-byte unpadded base64url Ed25519 key", spelled)
	}
	return ed25519.PublicKey(raw), nil
}

func decode64(text string) []byte {
	raw, err := base64.RawURLEncoding.Strict().DecodeString(text)
	if err != nil || len(raw) == 0 {
		return nil
	}
	return raw
}

func port(env map[string]string, name string, required bool) (int, error) {
	text := env[name]
	if text == "" && !required {
		return 0, nil
	}
	n, err := strconv.Atoi(text)
	if err != nil || n <= 0 || n > 65535 || strconv.Itoa(n) != text {
		return 0, fmt.Errorf("%s must be one decimal TCP port", name)
	}
	return n, nil
}

func origin(env map[string]string, name string, required bool) (string, error) {
	text := env[name]
	if text == "" && !required {
		return "", nil
	}
	parsed, err := url.Parse(text)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.Path != "" || parsed.RawQuery != "" {
		return "", fmt.Errorf("%s must be https://host[:port]", name)
	}
	return text, nil
}

func cleanAbs(path string) bool { return filepath.IsAbs(path) && filepath.Clean(path) == path }

func cmpOr(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
