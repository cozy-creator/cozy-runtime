package host

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestReceiptKeyFileIngressIsPrivateAndNotInherited(t *testing.T) {
	public, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	path := filepath.Join(root, "receipt.key")
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		t.Fatal(err)
	}
	encoded := base64.RawURLEncoding.EncodeToString(key)
	env := []string{"COZY_MACHINE_ROOT=" + root, "COZY_MACHINE_LIFETIME=persistent", "COZY_WORKER_ID=machine", "COZY_WORKER_INTERNAL_PORT=9443", "COZY_AUTHORIZED_KEYS=" + base64.RawURLEncoding.EncodeToString(public), receiptKeyFileName + "=" + path}
	if err := os.WriteFile(path, []byte(encoded+"\n"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(receiptKeyFileName, path)
	grant, err := ReadGrant(env)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(grant.receiptKey, key) || os.Getenv(receiptKeyFileName) != "" {
		t.Fatal("key was not read privately and removed from process environment")
	}
	if _, err := ReadGrant(append(env, receiptKeyName+"="+encoded)); err == nil {
		t.Fatal("ambiguous key sources accepted")
	}
	if err := os.Chmod(path, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadGrant(env); err == nil {
		t.Fatal("public key-file permissions accepted")
	}
	if err := os.Chmod(path, 0600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "key-link")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	env[len(env)-1] = receiptKeyFileName + "=" + link
	if _, err := ReadGrant(env); err == nil {
		t.Fatal("symlink key-file accepted")
	}
}

func TestReadinessAdvertisesAgentContracts(t *testing.T) {
	raw, err := rewritePayload([]byte(`{"pod_boot_id":"boot"}`), 9443, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var payload struct {
		Capabilities []string `json:"machine_capabilities"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	for _, contract := range []string{"hub-access/1", "runtime-update/1"} {
		if !slices.Contains(payload.Capabilities, contract) {
			t.Fatalf("missing contract %s", contract)
		}
	}
}
