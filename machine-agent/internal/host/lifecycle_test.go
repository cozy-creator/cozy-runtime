package host

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func executable(t *testing.T, path, script string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+script+"\n"), 0755); err != nil {
		t.Fatal(err)
	}
}

func TestCapabilityProbeDoesNotLaunchUnsupportedRuntime(t *testing.T) {
	dir := t.TempDir()
	marker, bin := filepath.Join(dir, "launched"), filepath.Join(dir, "runtime")
	executable(t, bin, `if [ "$1" = capabilities ]; then echo '{"capabilities":["unrelated/2"]}'; exit 0; fi`+"\ntouch "+marker)
	_, err := probeRuntimeCapabilities(bin)
	var refused *launchRefusal
	if !errors.As(err, &refused) || refused.code != "runtime_supervisor_unsupported" {
		t.Fatalf("probe: %v", err)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("capability probe launched execution")
	}
	executable(t, bin, `echo '{"capabilities":["machine-supervisor/1","future/2"]}'`)
	if _, err := probeRuntimeCapabilities(bin); err != nil {
		t.Fatal(err)
	}
}

func TestReadyCrashesRemainFailuresAndRetryWithBackoff(t *testing.T) {
	r := &restarts{}
	for i := 0; i < 4; i++ {
		if !r.exited(runtimeExit{"exit status 1", 1}, true) || r.gone() || r.due(time.Now()) {
			t.Fatal("exit after readiness became idle/blocked or skipped backoff")
		}
	}
	if r.exited(runtimeExit{"exit status 6", 6}, false) || !r.gone() || r.code() != "runtime_launch_refused" {
		t.Fatal("structural refusal was retried")
	}
	if err := r.clear(); err != nil || !r.due(time.Now()) {
		t.Fatal("repair did not clear failure")
	}
}

func TestPersistentMachineNeverReleasesToHub(t *testing.T) {
	now := time.Now()
	i, err := openIdle(filepath.Join(t.TempDir(), "idle.json"), true, now)
	if err != nil {
		t.Fatal(err)
	}
	i.persistent = true
	if claimed, err := i.claim(now.Add(24 * time.Hour)); err != nil || claimed {
		t.Fatal("persistent machine claimed rental release")
	}
	if _, err := i.admit(context.Background(), now.Add(24*time.Hour)); err != nil {
		t.Fatal(err)
	}
	m := &Machine{owned: true} // deliberately no Hub client
	if ended, err := m.release(context.Background()); err != nil || ended {
		t.Fatal("persistent release reached Hub")
	}
}

func TestMachineLockRejectsConcurrentAgentButNotRestart(t *testing.T) {
	root := t.TempDir()
	unlock, err := lockMachine(root)
	if err != nil {
		t.Fatal(err)
	}
	if second, err := lockMachine(root); err == nil {
		second.close()
		t.Fatal("two agents acquired same machine")
	}
	unlock.close()
	unlock, err = lockMachine(root)
	if err != nil {
		t.Fatal(err)
	}
	unlock.close()
}

func TestStandaloneGrantNeedsNoHub(t *testing.T) {
	key, _, _ := ed25519.GenerateKey(rand.Reader)
	g, err := ReadGrant([]string{"COZY_MACHINE_LIFETIME=persistent", "COZY_MACHINE_ROOT=" + t.TempDir(), "COZY_WORKER_ID=local-stable", "COZY_WORKER_INTERNAL_PORT=12345", "COZY_AUTHORIZED_KEYS=" + base64.RawURLEncoding.EncodeToString(key), "COZY_WORKER_AUTH_TOKEN=obsolete", "TENSORHUB_ORIGIN=obsolete", "COZY_MACHINE_HUBS_JSON=obsolete"})
	if err != nil {
		t.Fatal(err)
	}
	if g.HubOrigin != "" || g.WorkerToken != "" {
		t.Fatal("local grant acquired Hub authority")
	}
}

func TestAgentIdentitySurvivesUnsupportedRuntimeAndRestart(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("in-process privilege dropping test requires unprivileged user")
	}
	root := t.TempDir()
	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	key := make([]byte, 32)
	_, _ = rand.Read(key)
	port, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	number := port.Addr().(*net.TCPAddr).Port
	port.Close()
	g := &Grant{Lifetime: "persistent", Root: root, WorkerID: "local-stable", ListenHost: "127.0.0.1", WorkerPort: number, Authorized: []ed25519.PublicKey{pub}, receiptKey: append([]byte{}, key...)}
	l := NewLayout(g)
	executable(t, l.TFS, "exit 0")
	executable(t, l.Runtime, "exit 2")
	client := &http.Client{Transport: &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}}, Timeout: time.Second}
	defer client.CloseIdleConnections()
	var identity []byte
	for round := 0; round < 2; round++ {
		g.receiptKey = append([]byte{}, key...)
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan error, 1)
		go func() { done <- Run(ctx, g, io.Discard) }()
		deadline := time.Now().Add(10 * time.Second)
		var receipt envelope
		for {
			response, err := client.Get("https://127.0.0.1:" + strings.TrimPrefix(port.Addr().String(), "127.0.0.1:") + "/v1/bootstrap/receipt")
			if err == nil {
				raw, _ := io.ReadAll(response.Body)
				response.Body.Close()
				if response.StatusCode == 200 && json.Unmarshal(raw, &receipt) == nil {
					break
				}
			}
			if time.Now().After(deadline) {
				cancel()
				t.Fatal("machine API unavailable when Runtime unsupported")
			}
			time.Sleep(20 * time.Millisecond)
		}
		if !strings.Contains(string(receipt.Payload), `"kind":"machine_identity"`) || len(mustHex(receipt.HMAC)) != 32 {
			cancel()
			t.Fatal("no authenticated identity receipt")
		}
		if round == 0 {
			identity = receipt.Payload
		} else if string(identity) != string(receipt.Payload) {
			cancel()
			t.Fatal("restart changed machine identity")
		}
		cancel()
		if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	}
}
