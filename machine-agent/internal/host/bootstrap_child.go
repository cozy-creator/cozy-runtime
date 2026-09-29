package host

import (
	"crypto/sha256"
	"crypto/tls"
	"errors"
	"os"
	"path/filepath"
)

func requestPlaneLease(g *Grant, state string) (*machineLease, error) {
	if g.bootstrap == nil {
		return lockMachine(state)
	}
	lock, err := os.Open(filepath.Join(state, "agent.lock"))
	if err != nil {
		return nil, err
	}
	info, err := lock.Stat()
	if err != nil {
		return nil, err
	}
	return &machineLease{lock: lock, lockInfo: info, lockPath: filepath.Join(state, "agent.lock")}, nil
}

func requestPlaneIdentity(g *Grant, l Layout) (*Identity, error) {
	if g.bootstrap == nil {
		return prepare(g, l)
	}
	boot, err := os.ReadFile(filepath.Join(l.State, "boot-id"))
	if err != nil {
		return nil, err
	}
	leaf, err := tls.LoadX509KeyPair(l.boot("tls.crt"), l.boot("tls.key"))
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(leaf.Certificate[0])
	return &Identity{BootID: string(boot), Leaf: leaf, Digest: digest[:]}, nil
}

func (m *Machine) adoptBootstrap() error {
	g := &guardianLauncher{control: os.NewFile(3, "guardian-control")}
	if incarnation := m.grant.bootstrap.Incarnation; incarnation != "" {
		p := &runtimeProcess{incarnation: incarnation, done: make(chan struct{})}
		p.stop = func() { p.stopped.Store(true); _ = g.send("stop", incarnation) }
		g.current = p
	}
	m.words = &lastWords{Writer: m.log}
	m.launcher = g
	go g.read(os.NewFile(4, "guardian-status"))
	return nil
}

func (m *Machine) prepareApplication() (string, error) {
	state := m.grant.bootstrap
	if state != nil && state.Resume {
		if state.RecoveryError != "" {
			return "", errors.New(state.RecoveryError)
		}
		if state.Pending {
			return "boot_pending", nil
		}
		return "", nil
	}
	return m.launcher.maintain("startup-prepare", []string{m.grant.Lifetime, m.layout.Store})
}

func bundledAgentPolicy(root string) bool {
	policy, _, err := readSoftwarePolicy(root)
	return err == nil && policy.Agent == "bundled"
}

func (m *Machine) replaceApplication(restored ...bool) error {
	if m.grant.bootstrap == nil {
		return nil
	}
	current, err := hashFile("/proc/self/exe")
	if err != nil {
		return err
	}
	args := []string{current}
	if len(restored) > 0 && restored[0] {
		args = append(args, "rollback")
	}
	answer, err := m.launcher.maintain("agent-replace", args)
	if err != nil {
		return err
	}
	if answer != "replace" {
		return nil
	}
	os.Exit(replaceAgentExit)
	return nil
}

func (m *Machine) finishResumedUpdate(state, detail string) {
	if m.grant.bootstrap == nil || !m.grant.bootstrap.Pending {
		return
	}
	m.updates.staging.Lock()
	defer m.updates.staging.Unlock()
	m.updates.mu.Lock()
	current := m.updates.status
	m.updates.mu.Unlock()
	if current != nil && !current.terminal() {
		status := *current
		status.State, status.Error = state, detail
		m.completeUpdateFiles(status)
		m.updates.mu.Lock()
		m.updates.status, m.updates.active = &status, ""
		m.updates.mu.Unlock()
	}
	m.grant.bootstrap.Pending = false
}
