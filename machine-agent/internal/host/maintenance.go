package host

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/hostruntime"
)

// Maintenance ops run as the Runtime's own user: the root guardian on a pod, this daemon
// elsewhere. The machine's Runtime and TensorFS are one wheel pair installed into
// opt/cozy/python and kept for rollback: in var/lib/cozy/dev/current, which opt/cozy/wheels
// follows, on an image, or in opt/cozy/wheels itself on this computer's machine.

// maintain runs one op against the machine rooted at root; p is its current Runtime.
func maintain(root string, p *runtimeProcess, op string, args []string) (string, error) {
	switch op {
	case "startup-commit":
		if p == nil || p.exited() || len(args) != 1 || args[0] != p.incarnation {
			return "", errors.New("startup commit requires the live Runtime incarnation")
		}
		return "", startupMaintenance(root, op)
	case "startup-rollback":
		if p != nil && !p.exited() {
			return "", errors.New("startup rollback requires a stopped candidate")
		}
		return "", startupMaintenance(root, op)
	case "capabilities":
		return probeRuntimeCapabilities(filepath.Join(root, "opt/cozy/bin/cozy-runtime-worker"))
	case "restart":
		return guardedRestart(root, p)
	case "pair":
		pair, err := keptPair(root)
		return strings.Join(pair, "\n"), err
	case "install":
		return "", installPair(root, args)
	case "select":
		return "", selectPair(root, args)
	}
	return "", fmt.Errorf("unknown maintenance op %q", op)
}

// restartAnswerBound is how long a Runtime has to answer a guarded restart request: it
// answers from its supervisor thread at once, busy or accepted.
const restartAnswerBound = time.Minute

type restartFact struct {
	PID      int    `json:"pid"`
	Sequence int    `json:"sequence"`
	Status   string `json:"status"` // accepted | busy
	Guarded  bool   `json:"supports_guarded_restart"`
}

func readRestartFact(path string) (restartFact, bool) {
	var fact restartFact
	raw, err := os.ReadFile(path)
	return fact, err == nil && json.Unmarshal(raw, &fact) == nil
}

// guardedRestart asks the Runtime to stop only if nothing runs: it answers "busy" and keeps
// serving, or "accepted" and exits, keeping its durable work for the next Runtime.
func guardedRestart(root string, p *runtimeProcess) (string, error) {
	if p == nil || p.exited() {
		return "accepted", nil // nothing runs
	}
	worker := filepath.Join(root, "run/cozy/worker")
	if capable, ok := readRestartFact(filepath.Join(worker, "restart-capability.json")); !ok || capable.PID != p.pid || !capable.Guarded {
		return "", errors.New("the running Runtime has not established guarded restarts; it predates in-place updates")
	}
	before, _ := readRestartFact(filepath.Join(worker, "restart-status.json"))
	if err := syscall.Kill(p.pid, syscall.SIGHUP); err != nil {
		return "", fmt.Errorf("signal the Runtime: %w", err)
	}
	deadline := time.Now().Add(restartAnswerBound)
	for {
		fact, ok := readRestartFact(filepath.Join(worker, "restart-status.json"))
		if ok && fact.PID == p.pid && (before.PID != p.pid || fact.Sequence > before.Sequence) {
			if fact.Status == "accepted" {
				<-p.done
			}
			return fact.Status, nil
		}
		if p.exited() {
			return "", errors.New("the Runtime exited without answering its guarded restart")
		}
		if time.Now().After(deadline) {
			return "", errors.New("the Runtime did not answer its guarded restart")
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// keptDir is where the installed pair's wheels are kept: the directory var/lib/cozy/dev/current
// names on an image, else opt/cozy/wheels.
func keptDir(root string) (string, bool) {
	current := filepath.Join(root, "var/lib/cozy/dev/current")
	if info, err := os.Lstat(current); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return current, true
	}
	return filepath.Join(root, "opt/cozy/wheels"), false
}

// keptPair is the kept Runtime and TensorFS wheels, or none when the pair is not kept whole.
func keptPair(root string) ([]string, error) {
	dir, _ := keptDir(root)
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var pair []string
	seen := map[string]bool{}
	for _, entry := range entries {
		name := wheelDistribution(entry.Name())
		if name == "" || seen[name] || !entry.Type().IsRegular() {
			continue
		}
		seen[name] = true
		pair = append(pair, filepath.Join(dir, entry.Name()))
	}
	if len(pair) != 2 {
		return nil, nil
	}
	return pair, nil
}

// wheelDistribution is the Runtime's or TensorFS' distribution for one of the pair's wheel files, else "".
func wheelDistribution(file string) string {
	if !strings.HasSuffix(file, ".whl") || strings.ContainsAny(file, "/\x00") {
		return ""
	}
	switch {
	case strings.HasPrefix(file, "cozy_runtime-"):
		return hostruntime.Distribution
	case strings.HasPrefix(file, "tensorfs-"):
		return "tensorfs"
	}
	return ""
}

// wheelVersion is the version a wheel's file name spells.
func wheelVersion(file string) string {
	parts := strings.Split(strings.TrimSuffix(filepath.Base(file), ".whl"), "-")
	if len(parts) < 5 {
		return ""
	}
	return parts[1]
}

// installPair replaces the pair in opt/cozy/python: exactly these wheels, nothing resolved.
func installPair(root string, wheels []string) error {
	if len(wheels) == 0 {
		return errors.New("no wheels to install")
	}
	bin := filepath.Join(root, "usr/local/bin")
	cmd := exec.Command(filepath.Join(bin, "uv"), append([]string{"pip", "install", "--no-config", "--no-cache",
		"--python", filepath.Join(root, "opt/cozy/python/bin/python"), "--break-system-packages",
		"--offline", "--no-deps", "--reinstall"}, wheels...)...)
	cmd.Env = []string{"PATH=" + bin + ":/usr/bin:/bin", "HOME=" + filepath.Join(root, "tmp")}
	if out, err := childOutput(cmd, true); err != nil {
		return fmt.Errorf("install %s: %v: %s", strings.Join(wheels, ", "), err, strings.TrimSpace(string(out)))
	}
	return nil
}

// selectPair keeps these wheels as the installed pair: a new directory that current names on an
// image, or opt/cozy/wheels' contents on this computer's machine.
func selectPair(root string, wheels []string) error {
	dir, image := keptDir(root)
	if !image {
		return keepFiles(dir, wheels)
	}
	sum := sha256.New()
	for _, wheel := range slices.Sorted(slices.Values(wheels)) {
		sum.Write([]byte(filepath.Base(wheel) + "\x00"))
	}
	dev := filepath.Dir(dir)
	named := "pair-" + hex.EncodeToString(sum.Sum(nil))[:16]
	if _, err := os.Stat(filepath.Join(dev, named)); errors.Is(err, os.ErrNotExist) {
		staged, err := os.MkdirTemp(dev, ".pair-")
		if err != nil {
			return err
		}
		if err := keepFiles(staged, wheels); err != nil {
			_ = os.RemoveAll(staged)
			return err
		}
		if err := os.Rename(staged, filepath.Join(dev, named)); err != nil {
			_ = os.RemoveAll(staged)
			return err
		}
	}
	next := filepath.Join(dev, ".current-new")
	_ = os.Remove(next)
	if err := os.Symlink(named, next); err != nil {
		return err
	}
	if err := os.Rename(next, dir); err != nil {
		return err
	}
	return syncDir(dev)
}

// keepFiles makes dir hold exactly these wheels.
func keepFiles(dir string, wheels []string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	keep := map[string]bool{}
	for _, wheel := range wheels {
		name := filepath.Base(wheel)
		keep[name] = true
		if filepath.Dir(wheel) == dir {
			continue
		}
		if err := copyWheel(wheel, filepath.Join(dir, name)); err != nil {
			return err
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if !keep[entry.Name()] {
			if err := os.RemoveAll(filepath.Join(dir, entry.Name())); err != nil {
				return err
			}
		}
	}
	return syncDir(dir)
}

func copyWheel(source, target string) error {
	in, err := os.Open(source)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.CreateTemp(filepath.Dir(target), ".wheel-*")
	if err != nil {
		return err
	}
	defer os.Remove(out.Name())
	if _, err := io.Copy(out, in); err == nil {
		err = out.Chmod(0o644)
	}
	if err == nil {
		err = out.Sync()
	}
	if closeErr := out.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(out.Name(), target)
}

func syncDir(dir string) error {
	f, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}
