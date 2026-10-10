package host

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestMachineLeaseRecognizesOnlyRemovedOrReplacedOwnership(t *testing.T) {
	for _, mutation := range []string{"contents", "removed lock", "replaced lock", "replaced root"} {
		t.Run(mutation, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), "root")
			state := filepath.Join(root, "state")
			lease, err := lockMachine(state)
			if err != nil {
				t.Fatal(err)
			}
			defer lease.close()
			if err := lease.watchRoot(root); err != nil {
				t.Fatal(err)
			}
			lock := filepath.Join(state, "agent.lock")
			switch mutation {
			case "contents":
				err = os.WriteFile(lock, []byte("new diagnostic contents"), 0600)
			case "removed lock":
				err = os.Remove(lock)
			case "replaced lock":
				if err = os.Rename(lock, lock+".old"); err == nil {
					err = os.WriteFile(lock, nil, 0600)
				}
			case "replaced root":
				if err = os.Rename(root, root+".old"); err == nil {
					err = os.Mkdir(root, 0755)
				}
			}
			if err != nil {
				t.Fatal(err)
			}
			err = lease.check()
			if mutation == "contents" {
				if err != nil {
					t.Fatal("same inode lost ownership:", err)
				}
			} else if !errors.Is(err, errMachineOwnershipLost) {
				t.Fatal("ownership loss was not observed:", err)
			}
		})
	}
}

func TestMachineLeaseDoesNotInterpretPermissionFailureAsRemoval(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("permission refusal requires unprivileged user")
	}
	parent := t.TempDir()
	root := filepath.Join(parent, "root")
	lease, err := lockMachine(filepath.Join(root, "state"))
	if err != nil {
		t.Fatal(err)
	}
	defer lease.close()
	if err := lease.watchRoot(root); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(parent, 0000); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(parent, 0700)
	if err := lease.check(); err != nil {
		t.Fatal("unreadable identity was fabricated as removal:", err)
	}
}

func TestRunningAgentStopsAfterOwnedRootOrLockReplacement(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("in-process privilege dropping requires unprivileged user")
	}
	for _, target := range []string{"root", "lock"} {
		t.Run(target, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), "root")
			pub, _, err := ed25519.GenerateKey(rand.Reader)
			if err != nil {
				t.Fatal(err)
			}
			g := &Grant{Root: root, Lifetime: "persistent", WorkerID: "lease-proof", ListenHost: "127.0.0.1", Authorized: []ed25519.PublicKey{pub}}
			layout := NewLayout(g)
			marker := filepath.Join(root, "runtime-started")
			executable(t, layout.TFS, "exit 0")
			executable(t, layout.Runtime, fmt.Sprintf("if [ \"$1\" = capabilities ]; then echo '{\"capabilities\":[\"machine-supervisor/1\"]}'; exit 0; fi\ntouch '%s'\nexec cat <&3 >/dev/null", marker))
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan error, 1)
			go func() { done <- Run(ctx, g, io.Discard) }()
			// This is a disposable test observation budget, never an execution timeout.
			deadline := time.After(10 * time.Second)
			for {
				if _, err := os.Stat(marker); err == nil {
					break
				}
				select {
				case err := <-done:
					t.Fatalf("agent exited before replacement: %v", err)
				case <-deadline:
					t.Fatal("fixture Runtime did not start")
				case <-time.After(10 * time.Millisecond):
				}
			}
			lock := filepath.Join(layout.State, "agent.lock")
			if err := os.WriteFile(lock, []byte("same live lease"), 0600); err != nil {
				t.Fatal(err)
			}
			select {
			case err := <-done:
				t.Fatalf("content change ended owned machine: %v", err)
			case <-time.After(300 * time.Millisecond):
			}
			if target == "root" {
				if err := os.Rename(root, root+".old"); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(root, 0755); err != nil {
					t.Fatal(err)
				}
			} else {
				if err := os.Rename(lock, lock+".old"); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(lock, nil, 0600); err != nil {
					t.Fatal(err)
				}
			}
			select {
			case err := <-done:
				if !errors.Is(err, errMachineOwnershipLost) {
					t.Fatalf("wrong exit cause: %v", err)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("agent survived removed ownership")
			}
		})
	}
}
