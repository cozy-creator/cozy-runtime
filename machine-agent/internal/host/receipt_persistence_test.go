package host

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func receiptPayload(t *testing.T, change string) []byte {
	t.Helper()
	fields := map[string]any{"pod_boot_id": "boot-before", "tls_certificate_der_base64": "leaf-before", "runtime_gpus": []map[string]string{{"device_uuid": "gpu-before"}}, "runtime_version": "before"}
	switch change {
	case "boot":
		fields["pod_boot_id"] = "boot-after"
	case "leaf":
		fields["tls_certificate_der_base64"] = "leaf-after"
	case "gpu":
		fields["runtime_gpus"] = []map[string]string{{"device_uuid": "gpu-after"}}
	case "harmless":
		fields["runtime_version"] = "after"
		fields["process_incarnation"] = "fresh-child"
	}
	raw, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func signedReceipt(t *testing.T, key, raw []byte) []byte {
	t.Helper()
	payload, err := rewritePayload(raw, 9443, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.MarshalIndent(envelope{Payload: payload, HMAC: hex.EncodeToString(mac(key, payload))}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(body, '\n')
}

func TestReceiptPostRenameFailureRetainsOneSignedEnvelope(t *testing.T) {
	for _, change := range []string{"boot", "leaf", "gpu", "harmless"} {
		t.Run(change, func(t *testing.T) {
			key := bytes.Repeat([]byte{0x42}, 32)
			original := signedReceipt(t, key, receiptPayload(t, ""))
			path := filepath.Join(t.TempDir(), "receipt.json")
			if err := os.WriteFile(path, original, 0444); err != nil {
				t.Fatal(err)
			}
			// Exact observable state after Rename succeeds but directory sync reports
			// failure: the real file exists, the key is unspent, and readiness is closed.
			r := &receipt{path: path, key: key}
			err := r.seal(receiptPayload(t, change), 9443, nil, nil)
			if change == "harmless" {
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(r.Envelope(), original) || len(r.key) != 0 {
					t.Fatal("retry did not publish the original envelope and spend its key")
				}
			} else {
				if err == nil {
					t.Fatal("post-rename retry signed a different attestation")
				}
				if r.Envelope() != nil || len(r.key) != 32 {
					t.Fatal("refused retry published readiness or spent authority")
				}
				if err := r.seal(receiptPayload(t, "harmless"), 9443, nil, nil); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(r.Envelope(), original) {
					t.Fatal("matching retry replaced the persisted envelope")
				}
			}
			actual, err := os.ReadFile(path)
			if err != nil || !bytes.Equal(actual, original) {
				t.Fatal("retry changed persisted envelope bytes")
			}
		})
	}
}

func TestReceiptPreWriteFailureRetriesTheOriginalEnvelope(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing", "receipt.json")
	blocker := filepath.Dir(path)
	if err := os.WriteFile(blocker, []byte("not a directory"), 0600); err != nil {
		t.Fatal(err)
	}
	key := bytes.Repeat([]byte{0x24}, 32)
	r := &receipt{path: path, key: key}
	if err := r.seal(receiptPayload(t, ""), 9443, nil, nil); err == nil {
		t.Fatal("blocked persistence succeeded")
	}
	if r.Envelope() != nil || len(r.key) != 32 {
		t.Fatal("pre-write failure published readiness or spent authority")
	}
	if err := os.Remove(blocker); err != nil {
		t.Fatal(err)
	}
	if err := r.seal(receiptPayload(t, ""), 9443, nil, nil); err != nil {
		t.Fatal(err)
	}
	if r.Envelope() == nil || len(r.key) != 0 {
		t.Fatal("storage repair did not permit one successful seal")
	}
	before := bytes.Clone(r.Envelope())
	if err := r.seal(receiptPayload(t, "gpu"), 9443, nil, nil); err == nil {
		t.Fatal("completed receipt accepted changed device facts")
	}
	if !bytes.Equal(before, r.Envelope()) {
		t.Fatal("refused retry replaced completed receipt")
	}
}

func TestReceiptUnspentKeyRefusesUnverifiablePersistedEnvelope(t *testing.T) {
	for _, raw := range [][]byte{[]byte("broken JSON"), signedReceipt(t, bytes.Repeat([]byte{1}, 32), receiptPayload(t, ""))} {
		path := filepath.Join(t.TempDir(), "receipt.json")
		if err := os.WriteFile(path, raw, 0444); err != nil {
			t.Fatal(err)
		}
		r := &receipt{path: path, key: bytes.Repeat([]byte{2}, 32)}
		if err := r.seal(receiptPayload(t, ""), 9443, nil, nil); err == nil {
			t.Fatal("unverifiable persisted envelope was overwritten")
		}
		after, err := os.ReadFile(path)
		if err != nil || !bytes.Equal(raw, after) || r.Envelope() != nil || len(r.key) != 32 {
			t.Fatal("refusal mutated persisted receipt or signing authority")
		}
	}
}

func TestReceiptInvalidOrOversizedInputDoesNotPublish(t *testing.T) {
	for _, raw := range [][]byte{[]byte("null"), []byte("{"), []byte("[]"), []byte(`{"large":"` + strings.Repeat("x", maxReceiptBytes) + `"}`)} {
		path := filepath.Join(t.TempDir(), "receipt.json")
		r := &receipt{path: path, key: bytes.Repeat([]byte{3}, 32)}
		if err := r.seal(raw, 9443, nil, nil); err == nil {
			t.Fatal("invalid or oversized input was sealed")
		}
		if r.Envelope() != nil || len(r.key) != 32 {
			t.Fatal("invalid input published readiness or spent authority")
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatal("invalid input left a receipt")
		}
		if err := r.seal(receiptPayload(t, ""), 9443, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
}

func TestReceiptPendingSignatureSurvivesPersistenceFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := os.Mkdir(path, 0700); err != nil {
		t.Fatal(err)
	}
	key := bytes.Repeat([]byte{4}, 32)
	original := signedReceipt(t, key, receiptPayload(t, ""))
	r := &receipt{path: path, key: key, pending: bytes.Clone(original)}
	// A pending signature is retained before writeAtomic. A directory at the
	// target gives a real rename failure without relying on filesystem privileges.
	if err := r.seal(receiptPayload(t, ""), 9443, nil, nil); err == nil {
		t.Fatal("rename onto directory succeeded")
	}
	if r.Envelope() != nil || len(r.key) != 32 || !bytes.Equal(r.pending, original) {
		t.Fatal("failed persistence changed pending authority")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := r.seal(receiptPayload(t, "leaf"), 9443, nil, nil); err == nil {
		t.Fatal("retry signed changed facts after storage repair")
	}
	if err := r.seal(receiptPayload(t, "harmless"), 9443, nil, nil); err != nil {
		t.Fatal(err)
	}
	actual, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(actual, original) || !bytes.Equal(r.Envelope(), original) || r.pending != nil || len(r.key) != 0 {
		t.Fatal("repaired persistence did not publish original bytes exactly once")
	}
}
