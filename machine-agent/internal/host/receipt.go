package host

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/build"
)

// ReceiptDomain separates the readiness receipt's HMAC from any other use of its key.
const ReceiptDomain = "cozy.pod-readiness/1\x00"

// The readiness receipt is the Hub's proof that this machine booted: the payload the Runtime
// measured, sealed once under the per-attempt key and served at /v1/bootstrap/receipt.
const (
	receiptDomain   = ReceiptDomain
	maxReceiptBytes = 64 << 10
)

type envelope struct {
	Payload []byte `json:"payload"`
	HMAC    string `json:"hmac_sha256"`
}

// receipt holds the one envelope this boot publishes and the one-shot key that seals it.
type receipt struct {
	path       string
	mu         sync.Mutex
	key        []byte
	pending    []byte // one signed envelope awaiting a successful durable write
	retained   []byte // the payload of an envelope an earlier process of this boot sealed
	sealed     []byte
	standalone bool
}

// openReceipt reads a retained envelope. A provider replaying the original environment on a
// container restart hands the key again; it may only verify what was sealed, never re-sign.
func openReceipt(path string, key []byte) (*receipt, error) {
	r := &receipt{path: path, key: key}
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		if len(key) == 0 {
			return nil, fmt.Errorf("%s is required when no readiness envelope is retained", receiptKeyName)
		}
		return r, nil
	}
	if err != nil {
		return nil, err
	}
	env, err := verifyEnvelope(raw, key)
	if err != nil {
		return nil, err
	}
	clear(key)
	r.key, r.retained, r.sealed = nil, env.Payload, raw
	return r, nil
}

func verifyEnvelope(raw, key []byte) (envelope, error) {
	var env envelope
	if len(raw) > maxReceiptBytes || json.Unmarshal(raw, &env) != nil || len(env.Payload) == 0 {
		return env, errors.New("the retained readiness envelope is unreadable")
	}
	if len(key) != 0 && !hmac.Equal(mac(key, env.Payload), mustHex(env.HMAC)) {
		return env, errors.New("the retained readiness envelope does not verify under the granted key")
	}
	return env, nil
}

// Envelope is the sealed bytes, or nil before readiness.
func (r *receipt) Envelope() []byte {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.sealed
}

// seal publishes the Runtime's payload: the worker port becomes the public one, the machine
// names its own release (machine_version, the cozy binary's stamp) and, for a Hub that froze
// an auth document, the observed auth repeats it. Everything else passes through, including
// members this daemon does not know. A later Runtime of the same boot must attest the same
// facts; the original envelope is kept byte for byte.
func (r *receipt) seal(raw []byte, workerPort int, observed *OwnerAuth, webrtc map[string]any) error {
	payload, err := rewritePayload(raw, workerPort, observed, webrtc)
	if err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.standalone {
		body, err := json.Marshal(envelope{Payload: payload})
		if err != nil {
			return err
		}
		r.sealed, r.retained = body, payload
		return nil
	}
	if r.retained != nil {
		return sameAttestation(r.retained, payload)
	}
	if r.sealed != nil {
		return nil
	}
	if len(r.key) == 0 {
		return errors.New("the readiness key was spent; this boot signs nothing again")
	}
	body := r.pending
	if body == nil {
		// Rename can succeed even when syncing its directory fails. Recover that
		// signature before considering a new one; it still cannot be advertised.
		var readErr error
		body, readErr = os.ReadFile(r.path)
		if readErr != nil && !errors.Is(readErr, os.ErrNotExist) {
			return readErr
		}
		if errors.Is(readErr, os.ErrNotExist) {
			sum := hex.EncodeToString(mac(r.key, payload))
			body, err = json.Marshal(envelope{Payload: payload, HMAC: sum})
			if err != nil || len(body) > maxReceiptBytes {
				return fmt.Errorf("the readiness envelope exceeds %d bytes", maxReceiptBytes)
			}
		}
	}
	env, err := verifyEnvelope(body, r.key)
	if err != nil {
		return err
	}
	r.pending = body
	if err := sameAttestation(env.Payload, payload); err != nil {
		return err
	}
	if err := writeAtomic(r.path, r.pending, 0o444); err != nil {
		return err
	}
	clear(r.key)
	r.key = nil
	r.sealed, r.retained, r.pending = r.pending, env.Payload, nil
	return nil
}

func rewritePayload(raw []byte, workerPort int, observed *OwnerAuth, webrtc map[string]any) ([]byte, error) {
	var members map[string]json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil {
		return nil, fmt.Errorf("the Runtime readiness payload is not one object: %w", err)
	}
	if members == nil {
		return nil, errors.New("the Runtime readiness payload is not one object")
	}
	delete(members, "process_incarnation") // private readiness is never a retained public fact
	var err error
	if members["worker_internal_port"], err = json.Marshal(workerPort); err != nil {
		return nil, err
	}
	if members["machine_version"], err = json.Marshal(build.Version); err != nil {
		return nil, err
	}
	members["machine_capabilities"], _ = json.Marshal(build.Capabilities())
	if observed == nil { // the Hub froze no auth document: the daemon's own key is no fact about this boot
		delete(members, "observed_record_owner_auth")
	} else if members["observed_record_owner_auth"], err = json.Marshal(observed); err != nil {
		return nil, err
	}
	if webrtc != nil {
		if members["webrtc"], err = json.Marshal(webrtc); err != nil {
			return nil, err
		}
	}
	return json.Marshal(members)
}

type attested struct {
	BootID string `json:"pod_boot_id"`
	Leaf   string `json:"tls_certificate_der_base64"`
	Auth   struct {
		Key    string   `json:"control_public_key_ed25519_b64url"`
		Tokens []string `json:"media_token_sha256"`
	} `json:"observed_record_owner_auth"`
	GPUs []struct {
		UUID string `json:"device_uuid"`
	} `json:"runtime_gpus"`
}

func (a attested) facts() []string {
	out := []string{a.BootID, a.Leaf, a.Auth.Key}
	tokens := slices.Sorted(slices.Values(a.Auth.Tokens))
	out = append(out, tokens...)
	var gpus []string
	for _, gpu := range a.GPUs {
		gpus = append(gpus, gpu.UUID)
	}
	slices.Sort(gpus)
	return append(out, gpus...)
}

func sameAttestation(before, after []byte) error {
	var a, b attested
	if json.Unmarshal(before, &a) != nil || json.Unmarshal(after, &b) != nil || !slices.Equal(a.facts(), b.facts()) {
		return errors.New("a restarted Runtime attested a different boot, leaf, owner or GPU set than this boot sealed")
	}
	return nil
}

func bootIDOf(payload []byte) string {
	var a attested
	_ = json.Unmarshal(payload, &a)
	return a.BootID
}

func mac(key, payload []byte) []byte {
	m := hmac.New(sha256.New, key)
	m.Write([]byte(receiptDomain))
	m.Write(payload)
	return m.Sum(nil)
}

func mustHex(text string) []byte {
	raw, _ := hex.DecodeString(text)
	return raw
}
