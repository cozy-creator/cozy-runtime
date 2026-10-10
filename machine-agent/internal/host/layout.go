package host

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"time"
)

// ServerName is the one SAN of every machine leaf; clients pin the leaf itself.
const ServerName = "cozy-worker"

// Layout is the image's filesystem, rooted: "/" on a pod, a directory on an owned machine.
// The Runtime derives the same paths from the same root.
type Layout struct {
	Root, Bootstrap, State, Store, Installs, Tmp, Runtime, TFS string
}

func NewLayout(g *Grant) Layout {
	at := func(path string) string { return filepath.Join(g.Root, path) }
	l := Layout{Root: g.Root, Bootstrap: at("run/cozy/bootstrap"), State: at("var/lib/cozy/machine"),
		Store: at("var/lib/tensorfs"), Installs: at("var/lib/cozy/installs"), Tmp: at("tmp"),
		Runtime: at("opt/cozy/bin/cozy-runtime-worker"), TFS: at("usr/local/bin/tfs")}
	if g.StoreRoot != "" {
		l.Store = g.StoreRoot
	}
	return l
}

func (l Layout) boot(name string) string { return filepath.Join(l.Bootstrap, name) }

// Stage holds uploaded local package carriers, per operation.
func (l Layout) Stage(operation string) string {
	return filepath.Join(l.Installs, ".stage", operation, "wheels")
}

// Identity is this machine lifetime: its boot id and TLS leaf.
type Identity struct {
	BootID string
	Leaf   tls.Certificate
	Digest []byte // sha256 of the leaf DER, which every ClaimProof signs
}

// prepare lays the machine out and mints or reopens its identity. The boot id and leaf
// persist with the root, so a restarted daemon is the same machine lifetime.
func prepare(g *Grant, l Layout) (*Identity, error) {
	for _, dir := range []string{l.Bootstrap, l.State, filepath.Join(l.Tmp, "cozy"), filepath.Join(l.Installs, ".stage")} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	stale, _ := filepath.Glob(l.boot("hub-ca-*.crt"))
	for _, name := range append(stale, l.boot("readiness-payload"), l.boot("tensorhub-ca.crt"), l.boot("worker-activity"), l.boot("machine-hubs.json")) {
		if err := os.Remove(name); err != nil && !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
	}
	bootID, err := persistent(filepath.Join(l.State, "boot-id"), func() ([]byte, error) {
		// A root the tensorhub Host booted keeps that lifetime; its Runtime's history names it.
		if raw, err := os.ReadFile(l.boot("pod-boot-id")); err == nil && len(bytes.TrimSpace(raw)) > 0 {
			return bytes.TrimSpace(raw), nil
		}
		raw := make([]byte, 32)
		_, err := rand.Read(raw)
		return []byte(base64.RawURLEncoding.EncodeToString(raw)), err
	})
	if err != nil {
		return nil, err
	}
	if err := writeAtomic(l.boot("pod-boot-id"), append(bootID, '\n'), 0o444); err != nil {
		return nil, err
	}
	if len(g.HubCA) > 0 {
		if err := writeAtomic(l.boot("tensorhub-ca.crt"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: g.HubCA}), 0o444); err != nil {
			return nil, err
		}
	}
	if g.HubOrigin != "" && g.WorkerToken != "" {
		authority, _ := json.Marshal(map[string]any{"version": 2, "origin": g.HubOrigin, "worker_id": g.WorkerID, "worker_token": g.WorkerToken})
		if err := writeAtomic(l.boot("machine-publication-authority.json"), authority, 0o400); err != nil {
			return nil, err
		}
	} else if err := os.Remove(l.boot("machine-publication-authority.json")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	access, err := readHubAccess(filepath.Join(l.State, "hub-access.json"))
	if err != nil {
		return nil, err
	}
	if err := projectHubAccess(access, l); err != nil {
		return nil, err
	}
	leaf, err := leaf(l.boot("tls.crt"), l.boot("tls.key"))
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(leaf.Certificate[0])
	return &Identity{BootID: string(bootID), Leaf: leaf, Digest: sum[:]}, nil
}

func persistent(path string, mint func() ([]byte, error)) ([]byte, error) {
	if raw, err := os.ReadFile(path); err == nil && len(raw) > 0 {
		return raw, nil
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	raw, err := mint()
	if err != nil {
		return nil, err
	}
	return raw, writeAtomic(path, raw, 0o400)
}

func leaf(certPath, keyPath string) (tls.Certificate, error) {
	if pair, err := tls.LoadX509KeyPair(certPath, keyPath); err == nil {
		return pair, nil
	} else if _, statErr := os.Stat(certPath); !errors.Is(statErr, os.ErrNotExist) {
		return tls.Certificate{}, fmt.Errorf("reopen the machine TLS identity: %w", err)
	}
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return tls.Certificate{}, err
	}
	serial, _ := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	template := &x509.Certificate{SerialNumber: serial, Subject: pkix.Name{CommonName: "cozy-machine"},
		NotBefore: time.Now().Add(-5 * time.Minute), NotAfter: time.Date(9999, 12, 31, 23, 59, 59, 0, time.UTC),
		KeyUsage: x509.KeyUsageDigitalSignature, DNSNames: []string{ServerName},
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth}}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		return tls.Certificate{}, err
	}
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return tls.Certificate{}, err
	}
	if err := writeAtomic(keyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o400); err != nil {
		return tls.Certificate{}, err
	}
	if err := writeAtomic(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o444); err != nil {
		return tls.Certificate{}, err
	}
	return tls.LoadX509KeyPair(certPath, keyPath)
}

func writeAtomic(target string, data []byte, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	temp, err := os.CreateTemp(filepath.Dir(target), ".write-*")
	if err != nil {
		return err
	}
	defer os.Remove(temp.Name())
	_, err = temp.Write(data)
	if err == nil {
		err = temp.Chmod(mode)
	}
	if err == nil {
		err = temp.Sync()
	}
	if closeErr := temp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err := os.Rename(temp.Name(), target); err != nil {
		return err
	}
	dir, err := os.Open(filepath.Dir(target))
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}
