// Package capability is the signed grant a client presents to one machine: an authorized
// key's permission to read a run's outputs, or to maintain the machine, until an expiry. It travels only in the
// `Authorization: Cozy-Cap` header or a WebRTC session's hello. A machine verifies it offline
// and stores no bearer.
package capability

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"slices"
	"strconv"
	"strings"
	"time"
)

const domain = "cozy-capability/1\x00"

// Maintenance is the action that replaces a machine's Runtime.
const Maintenance = "runtime-update"

// Grant is what a capability allows. A member this verifier does not know refuses the whole
// capability: a restriction an older machine cannot read must not be dropped.
type Grant struct {
	Machine string   `json:"m"`           // the machine's worker id
	Run     string   `json:"r,omitempty"` // the run's number on that machine
	Action  string   `json:"a,omitempty"` // or a maintenance action (Maintenance), never both
	Outputs []string `json:"p,omitempty"` // "name" (every index) or "name/i"; empty: every output
	Expires int64    `json:"e"`           // unix seconds
	Key     string   `json:"k"`           // the signer: KeyID of its public key
	Binding string   `json:"x,omitempty"` // the client's DTLS certificate, "sha-256 AB:…"
}

// KeyID names a public key compactly.
func KeyID(key ed25519.PublicKey) string {
	sum := sha256.Sum256(key)
	return base64.RawURLEncoding.EncodeToString(sum[:16])
}

// Mint signs a grant.
func Mint(key ed25519.PrivateKey, g Grant) (string, error) {
	return MintSigned(key.Public().(ed25519.PublicKey), func(message []byte) []byte { return ed25519.Sign(key, message) }, g)
}

// MintSigned signs a grant with a key held elsewhere, such as a device identity.
func MintSigned(public ed25519.PublicKey, sign func([]byte) []byte, g Grant) (string, error) {
	g.Key = KeyID(public)
	payload, err := json.Marshal(g)
	if err != nil {
		return "", err
	}
	signature := sign(append([]byte(domain), payload...))
	return base64.RawURLEncoding.EncodeToString(payload) + "." + base64.RawURLEncoding.EncodeToString(signature), nil
}

var (
	ErrInvalid = errors.New("the capability is malformed or not signed by an authorized key")
	ErrExpired = errors.New("the capability has expired")
	ErrScope   = errors.New("the capability does not grant this")
)

// Verify admits a token for machine at now, signed by one of keys. binding is the client's
// DTLS certificate fingerprint on WebRTC and empty on HTTPS: a capability that names a binding
// holds only there, and one that names none holds anywhere.
func Verify(token, machine string, keys []ed25519.PublicKey, now time.Time, binding string) (Grant, error) {
	var g Grant
	encoded, sig, ok := strings.Cut(token, ".")
	payload, err1 := base64.RawURLEncoding.DecodeString(encoded)
	signature, err2 := base64.RawURLEncoding.DecodeString(sig)
	if !ok || err1 != nil || err2 != nil || len(signature) != ed25519.SignatureSize {
		return g, ErrInvalid
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&g) != nil || decoder.More() || g.Machine == "" || (g.Run == "") == (g.Action == "") || g.Expires == 0 {
		return Grant{}, ErrInvalid
	}
	signed := append([]byte(domain), payload...)
	if !slices.ContainsFunc(keys, func(k ed25519.PublicKey) bool { return KeyID(k) == g.Key && ed25519.Verify(k, signed, signature) }) {
		return g, ErrInvalid
	}
	if g.Machine != machine || g.Binding != "" && g.Binding != binding {
		return g, ErrInvalid
	}
	if now.Unix() >= g.Expires {
		return g, ErrExpired
	}
	return g, nil
}

// Allows says whether the grant covers one output of a run; index < 0 names an output that
// has no index.
func (g Grant) Allows(run, output string, index int) bool {
	if run != g.Run {
		return false
	}
	if len(g.Outputs) == 0 {
		return true
	}
	return slices.Contains(g.Outputs, output) || index >= 0 && slices.Contains(g.Outputs, output+"/"+strconv.Itoa(index))
}

// Permits says whether the grant allows one maintenance action.
func (g Grant) Permits(action string) bool { return g.Run == "" && g.Action == action }
