package host

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
)

// Invoked reports whether this process runs the machine role: `cozy daemon`, or the binary
// started under the name older launchers and images exec, `pod-supervisor`.
func Invoked(args []string) bool {
	if len(args) == 1 && filepath.Base(args[0]) == "pod-supervisor" {
		return true
	}
	return len(args) == 2 && args[1] == "daemon"
}

// Main runs the machine role in the foreground from the grant in its environment, until
// SIGTERM or, on a rental, until the Hub accepts its idle release.
func Main(log io.Writer) int {
	environ := os.Environ()
	if !HasGrant(environ) {
		fmt.Fprintln(log, "cozy daemon: this process has no machine grant (COZY_WORKER_ID and the rest of a pod's environment)")
		return 2
	}
	g, err := ReadGrant(environ)
	if err != nil {
		fmt.Fprintln(log, "cozy daemon:", err)
		return 2
	}
	g.inherited = inherited(environ)
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	if err := Run(ctx, g, log); err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintln(log, "cozy daemon:", err)
		return 1
	}
	return 0
}

// inherited are the locale and trust-store settings a Runtime may take from this process.
func inherited(environ []string) []string {
	var out []string
	for _, entry := range environ {
		switch name, _, _ := cut(entry); name {
		case "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR":
			out = append(out, entry)
		}
	}
	return out
}

func cut(entry string) (string, string, bool) {
	for i := 0; i < len(entry); i++ {
		if entry[i] == '=' {
			return entry[:i], entry[i+1:], true
		}
	}
	return entry, "", false
}

// startSSH serves a development rental's SSH with the granted key. It runs as root, before
// the daemon drops privilege, and ends with the container.
func startSSH(key string, log io.Writer) error {
	if err := os.MkdirAll("/root/.ssh", 0o700); err != nil {
		return err
	}
	if err := os.WriteFile("/root/.ssh/authorized_keys", []byte(key+"\n"), 0o600); err != nil {
		return err
	}
	if err := os.MkdirAll("/run/sshd", 0o755); err != nil {
		return err
	}
	if out, err := exec.Command("/usr/bin/ssh-keygen", "-A").CombinedOutput(); err != nil {
		return fmt.Errorf("ssh host keys: %v: %s", err, out)
	}
	sshd := exec.Command("/usr/sbin/sshd", "-D", "-e")
	sshd.Stdout, sshd.Stderr = log, log
	if err := sshd.Start(); err != nil {
		return fmt.Errorf("start sshd: %w", err)
	}
	go func() { _ = sshd.Wait() }()
	return nil
}
