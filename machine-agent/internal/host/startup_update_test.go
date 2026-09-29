package host

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/build"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStartupMetadataAndPublicRetention(t *testing.T) {
	python := os.Getenv("COZY_RUNTIME_TEST_PYTHON")
	if python == "" {
		python = "python3"
	}
	if out, err := exec.Command(python, "-B", "startup_update_test.py").CombinedOutput(); err != nil {
		t.Fatalf("metadata/public journal: %v\n%s", err, out)
	}
}

func TestInterruptedStartupCannotBootWithoutIntactRollback(t *testing.T) {
	m := &Machine{grant: &Grant{}, layout: NewLayout(&Grant{Root: t.TempDir()}), log: io.Discard}
	status := startupStatus{State: "installing", Before: pairVersions{Runtime: "0.18.85", TensorFS: "0.3.78"}, Previous: []startupWheel{{File: "cozy_runtime-0.18.85-cp312-abi3-manylinux_2_28_x86_64.whl", SHA256: strings.Repeat("a", 64)}}}
	if err := m.startupSave(status); err != nil {
		t.Fatal(err)
	}
	for path, mode := range map[string]os.FileMode{m.startupPath("status.json"): 0600, filepath.Join(m.layout.State, "startup-update-status.json"): 0644} {
		info, err := os.Stat(path)
		if err != nil || info.Mode().Perm() != mode {
			t.Fatalf("journal/report permissions: %s %v", path, err)
		}
	}
	if err := m.startupUpdate(context.Background()); err == nil {
		t.Fatal("off policy allowed an interrupted mutation without verified rollback")
	}
	raw, err := os.ReadFile(m.startupPath("status.json"))
	if err != nil || !strings.Contains(string(raw), "installing") {
		t.Fatal("failed recovery erased its pending journal")
	}
}

func TestStartupWorkerLeaseNeverTakesOverLiveWorker(t *testing.T) {
	root := t.TempDir()
	unlocked, err := lockStartupWorker(root)
	if err != nil {
		t.Fatal(err)
	}
	defer unlocked()
	m := &Machine{grant: &Grant{}, layout: NewLayout(&Grant{Root: root}), log: io.Discard}
	if err := m.startupUpdate(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(m.startupPath("status.json")); !os.IsNotExist(err) {
		t.Fatal("live worker caused update mutation")
	}
}

func TestStartupRollbackKeepsVerifiedPrivateCopies(t *testing.T) {
	m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()})}
	source := filepath.Join(t.TempDir(), "cozy_runtime-0.18.85-cp312-abi3-manylinux_2_28_x86_64.whl")
	if err := os.WriteFile(source, []byte("retained immutable wheel"), 0600); err != nil {
		t.Fatal(err)
	}
	kept, err := m.startupPrevious([]string{source})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("changed source"), 0600); err != nil {
		t.Fatal(err)
	}
	sha, err := hashFile(m.startupPath("previous", kept[0].File))
	if err != nil || sha != kept[0].SHA256 {
		t.Fatal("rollback depended on mutable source")
	}
	if strings.HasPrefix(m.startupPath(), m.layout.State+string(os.PathSeparator)) {
		t.Fatal("privileged rollback journal would be handed to unprivileged agent")
	}
}

func TestStartupCommitRequiresItsLiveIncarnation(t *testing.T) {
	p := &runtimeProcess{incarnation: "candidate", done: make(chan struct{})}
	if _, err := maintain(t.TempDir(), p, "startup-commit", []string{"older"}); err == nil {
		t.Fatal("stale Runtime committed candidate")
	}
	if _, err := maintain(t.TempDir(), p, "startup-rollback", nil); err == nil {
		t.Fatal("rollback replaced live Runtime")
	}
}

func TestOperatorPolicyOverridesHistoricalPin(t *testing.T) {
	m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()}), log: io.Discard}
	pair := pairVersions{Runtime: "0.18.88", TensorFS: "0.3.78"}
	for _, mode := range []string{"auto", "off", "auto"} {
		raw, _ := json.Marshal(softwarePolicy{mode, "bundled"})
		if err := writeAtomic(policyPath(m.layout.Root), raw, 0644); err != nil {
			t.Fatal(err)
		}
		if err := m.startupSave(startupStatus{State: "succeeded", After: pair, Pinned: mode == "auto"}); err != nil {
			t.Fatal(err)
		}
		if got := m.explicitlyPinnedPair(pair); got != (mode == "off") {
			t.Fatalf("operator mode %s overridden by historical journal", mode)
		}
	}
}

func TestStartupCandidateDoesNotTrustMutableStageOrFollowTargetSymlink(t *testing.T) {
	m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()})}
	source := filepath.Join(t.TempDir(), "candidate.whl")
	if err := os.WriteFile(source, []byte("verified bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	sha, err := hashFile(source)
	if err != nil {
		t.Fatal(err)
	}
	dir := m.startupPath("candidate")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "untouched")
	if err := os.WriteFile(outside, []byte("outside bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, filepath.Base(source))); err != nil {
		t.Fatal(err)
	}
	private, err := m.startupCopyWheel(source, "candidate", sha)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("changed stage"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := hashFile(private)
	if err != nil || got != sha {
		t.Fatal("candidate still depended on agent-writable staging")
	}
	if raw, err := os.ReadFile(outside); err != nil || string(raw) != "outside bytes" {
		t.Fatal("private copy followed target symlink")
	}
	if _, err := m.startupCopyWheel(source, "candidate", sha); err == nil {
		t.Fatal("changed stage passed expected release digest")
	}
}

func TestStaleStartupJournalNeverReplacesExplicitInstallation(t *testing.T) {
	for _, installed := range []pairVersions{
		{Runtime: "0.18.99", TensorFS: "0.3.78"},
		{Runtime: "0.18.85", TensorFS: "0.3.99"},
	} {
		t.Run(installed.Runtime+"/"+installed.TensorFS, func(t *testing.T) {
			m := &Machine{grant: &Grant{}, layout: NewLayout(&Grant{Root: t.TempDir()}), log: io.Discard}
			state := startupStatus{State: "installing", Before: pairVersions{Runtime: "0.18.85", TensorFS: "0.3.78"}, After: pairVersions{Runtime: "0.18.86", TensorFS: "0.3.78"}}
			if err := m.startupSave(state); err != nil {
				t.Fatal(err)
			}
			executable(t, filepath.Join(m.layout.Root, "opt/cozy/python/bin/python"), fmt.Sprintf("printf '%s\\n%s\\n'", installed.Runtime, installed.TensorFS))
			if err := m.startupUpdate(context.Background()); err != nil {
				t.Fatal(err)
			}
			raw, _ := os.ReadFile(m.startupPath("status.json"))
			if err := json.Unmarshal(raw, &state); err != nil {
				t.Fatal(err)
			}
			if state.State != "superseded" {
				t.Fatalf("stale journal: %+v", state)
			}
		})
	}
}

func TestBrokenInstalledProbeAndReportDoNotPreventRepair(t *testing.T) {
	previous := build.Version
	build.Version = "0.1.2"
	defer func() { build.Version = previous }()
	for _, blockedReport := range []bool{false, true} {
		m := &Machine{grant: &Grant{Lifetime: "persistent"}, layout: NewLayout(&Grant{Root: t.TempDir()}), log: io.Discard}
		executable(t, filepath.Join(m.layout.Root, "opt/cozy/python/bin/python"), `case "$4" in inspect) echo '{"runtime":"0.18.86","tensorfs":"0.3.78","eligible":true,"quiescent":true}' ;; *) exit 1 ;; esac`)
		if blockedReport {
			if err := os.MkdirAll(m.startupPath("status.json"), 0700); err != nil {
				t.Fatal(err)
			}
			os.Remove(m.startupPath("status.json"))
			os.Chmod(m.startupPath(), 0500)
		}
		if err := m.startupUpdate(context.Background()); err != nil {
			t.Fatalf("pre-mutation failure closed repair API: %v", err)
		}
		if m.startupPending() {
			t.Fatal("failed preflight admitted a candidate")
		}
	}
}

func TestUsedRentalDefersUpdateBeforeProbing(t *testing.T) {
	previous := build.Version
	build.Version = "0.1.2"
	defer func() { build.Version = previous }()
	m := &Machine{grant: &Grant{Lifetime: "rental"}, layout: NewLayout(&Grant{Root: t.TempDir()}), log: io.Discard}
	executable(t, filepath.Join(m.layout.Root, "opt/cozy/python/bin/python"), `case "$4" in inspect) echo '{"runtime":"0.18.86","tensorfs":"0.3.78","eligible":true,"quiescent":true,"accepted_work":true}' ;; *) echo unexpected >&2; exit 1 ;; esac`)
	if err := m.startupUpdate(context.Background()); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(m.startupPath("status.json"))
	if !strings.Contains(string(raw), "previously used rental") {
		t.Fatalf("rental pin not honored: %s", raw)
	}
}

func TestAutomaticAndExplicitUpdatesShareRecoveryAndCommit(t *testing.T) {
	for _, mode := range []struct {
		explicit bool
		lifetime string
		guarded  bool
	}{{false, "persistent", false}, {true, "persistent", false}, {true, "rental", false}, {true, "persistent", true}} {
		explicit := mode.explicit
		t.Run(fmt.Sprint(explicit)+"/"+mode.lifetime+fmt.Sprint(mode.guarded), func(t *testing.T) {
			root := t.TempDir()
			m := &Machine{grant: &Grant{Root: root, Lifetime: "persistent"}, layout: NewLayout(&Grant{Root: root}), log: io.Discard}
			before := pairVersions{Runtime: "0.18.86", TensorFS: "0.3.78"}
			after := pairVersions{Runtime: "0.18.87", TensorFS: "0.3.78"}
			var previous, candidate []string
			for _, pair := range []pairVersions{before, after} {
				var wheels []string
				for _, name := range []string{"cozy_runtime-" + pair.Runtime + "-py3-none-any.whl", "tensorfs-" + pair.TensorFS + "-py3-none-any.whl"} {
					path := filepath.Join(root, name)
					if err := os.WriteFile(path, []byte(name), 0600); err != nil {
						t.Fatal(err)
					}
					wheels = append(wheels, path)
				}
				if pair == before {
					previous = wheels
				} else {
					candidate = wheels
				}
			}
			for name, pair := range map[string]pairVersions{"before": before, "after": after, "current": before} {
				body, _ := json.Marshal(startupProbe{pairVersions: pair, WireMinor: pb.WireMinor, MinimumWireMinor: pb.MinCompatibleWireMinor, Capabilities: []string{"machine-supervisor/1", "machine-public-reads/1"}})
				if err := os.WriteFile(filepath.Join(root, name+".json"), body, 0600); err != nil {
					t.Fatal(err)
				}
			}
			executable(t, filepath.Join(root, "opt/cozy/python/bin/python"), fmt.Sprintf(`case "$4" in
 inspect) echo '{"runtime":"0.18.86","tensorfs":"0.3.78","eligible":true,"quiescent":true,"accepted_work":true,"workspace_id":"original"}' ;;
 dependencies) echo '[]' ;;
 probe) if [ -n "$5" ]; then cat '%s/after.json'; else cat '%s/current.json'; fi ;;
 *) printf '0.18.86\n0.3.78\n' ;;
 esac`, root, root))
			executable(t, filepath.Join(root, "bundled-agent"), fmt.Sprintf(`echo '{"name":"cozy-machine","wire_minor":%d,"minimum_wire_minor":%d,"capabilities":["hub-access/1","runtime-update/1","machine-bootstrap/1"]}'`, pb.WireMinor, pb.MinCompatibleWireMinor))
			body, err := os.ReadFile(filepath.Join(root, "bundled-agent"))
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(filepath.Join(root, "opt/cozy/python/bin/cozy-machine"), body, 0755); err != nil {
				t.Fatal(err)
			}
			executable(t, filepath.Join(root, "usr/local/bin/uv"), fmt.Sprintf(`case "$*" in
 *--target*)
 for value in "$@"; do
  if [ "$last" = --target ]; then mkdir -p "$value/bin"; cp "%s/bundled-agent" "$value/bin/cozy-machine"; exit 0; fi
  last="$value"
 done ;;
 *cozy_runtime-0.18.87-*) cp '%s/after.json' '%s/current.json' ;;
 *) cp '%s/before.json' '%s/current.json' ;;
 esac`, root, root, root, root, root))
			if explicit {
				request, _ := json.Marshal(pairTransaction{Before: before, After: after, Previous: previous, Candidate: candidate, Store: m.layout.Store, Lifetime: mode.lifetime, Pin: true, Agent: "bundled"})
				var stopped *runtimeProcess
				if mode.guarded {
					// Paused/retained records are not live work. A live guarded shutdown
					// authorizes handoff; an unexplained cold exit does not.
					path := filepath.Join(root, "opt/cozy/python/bin/python")
					script, err := os.ReadFile(path)
					if err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(path, []byte(strings.ReplaceAll(string(script), `"quiescent":true`, `"quiescent":false`)), 0755); err != nil {
						t.Fatal(err)
					}
					if _, err := maintain(root, nil, "update-prepare", []string{string(request)}); err == nil {
						t.Fatal("cold retained work authorized activation")
					}
					stopped = &runtimeProcess{done: make(chan struct{})}
					close(stopped.done)
					stopped.guardedStopped.Store(true)
					// Old request planes omit update-stage and must still upgrade safely.
				} else if _, err := maintain(root, nil, "update-stage", []string{string(request)}); err != nil {
					t.Fatal(err)
				}
				answer, err := maintain(root, stopped, "update-prepare", []string{string(request)})
				if err != nil || answer != "boot_pending" {
					t.Fatalf("explicit preparation: %s %v", answer, err)
				}

			} else if err := m.preparePair(context.Background(), before, after, candidate, previous, startupStatus{Workspace: "original"}, false); err != nil {
				t.Fatal(err)
			}
			if !m.transactionPending() {
				t.Fatal("install did not retain recovery authority until health commit")
			}
			process := &runtimeProcess{incarnation: "candidate", done: make(chan struct{})}
			if _, err := maintain(root, process, "startup-commit", []string{"candidate", "different"}); err == nil {
				t.Fatal("candidate changed workspace identity")
			}
			if _, err := maintain(root, nil, "startup-rollback", nil); err != nil {
				t.Fatal(err)
			}
			raw, _ := os.ReadFile(m.startupPath("status.json"))
			var state startupStatus
			if err := json.Unmarshal(raw, &state); err != nil {
				t.Fatal(err)
			}
			if state.State != "rolled_back" || len(state.Failed) != 1 || state.Failed[0] != after {
				t.Fatalf("missing shared rollback/failure history: %+v", state)
			}
			got, err := m.startupVerify(context.Background())
			if err != nil || got != before {
				t.Fatalf("rollback pair: %+v %v", got, err)
			}
			// Restoring the old pair does not close repair service if its store
			// helper cannot relaunch it. The failure is paced, not structural.
			if err := m.preparePair(context.Background(), before, after, candidate, previous, state, explicit); err != nil {
				t.Fatal(err)
			}
			m.launcher = &rollbackObservationLauncher{directLauncher{root: root}}
			m.tfs = &tensorFS{bin: filepath.Join(root, "missing-tfs")}
			m.restarts = &restarts{}
			cause := errors.New("initialize TensorFS Store: permission denied at fixture-store")
			var rollbackLog bytes.Buffer
			m.log = &rollbackLog
			m.grant.bootstrap = &bootstrapChild{Pending: true}
			m.updates.status = &updateStatus{Operation: "fixture-update", State: "installing"}
			if err := m.rollbackStartup(context.Background(), cause); err != nil {
				t.Fatalf("repair service lost on relaunch failure: %v", err)
			}
			raw, _ = os.ReadFile(m.startupPath("status.json"))
			var recovered startupStatus
			if json.Unmarshal(raw, &recovered) != nil || recovered.Detail != cause.Error() || !strings.Contains(rollbackLog.String(), cause.Error()) {
				t.Fatalf("rollback discarded startup failure: %s %s", raw, rollbackLog.String())
			}
			m.openUpdates() // replacement request plane must retain the specific terminal error
			if m.updates.status.State != "rolled_back" || m.updates.status.Error != cause.Error() {
				t.Fatalf("replacement lost rollback cause: %+v", m.updates.status)
			}
			m.grant.bootstrap = nil
			if m.startupPending() || m.restarts.gone() || m.restarts.due(time.Now()) {
				t.Fatal("rollback relaunch did not retain paced repair service")
			}
			// An explicit retry is allowed; successful commit uses the same private journal.
			if err := m.preparePair(context.Background(), before, after, candidate, previous, state, explicit); err != nil {
				t.Fatal(err)
			}
			if _, err := maintain(root, process, "startup-commit", []string{"candidate", "original"}); err != nil {
				t.Fatal(err)
			}
			if m.transactionPending() {
				t.Fatal("health commit left the transaction pending")
			}
			if m.explicitlyPinnedPair(after) != explicit {
				t.Fatal("private commit did not retain explicit pin before API publication")
			}
		})
	}
}

func TestMaintenanceStallCannotLeaveDescendantsHoldingOutput(t *testing.T) {
	cmd := exec.Command("sh", "-c", "sleep 300 & wait")
	if _, err := progressChildOutput(cmd, true); err == nil {
		t.Fatal("stalled helper was not stopped")
	}
}

func TestCapabilityProbeFailureRemainsTransient(t *testing.T) {
	path := filepath.Join(t.TempDir(), "worker")
	executable(t, path, "exit 1")
	_, err := probeRuntimeCapabilities(path)
	if err == nil {
		t.Fatal("failed probe succeeded")
	}
	restarts := &restarts{}
	restarts.fail(err)
	if restarts.gone() || restarts.due(time.Now()) {
		t.Fatal("probe crash became structural or unpaced")
	}
}

func TestTypedSoftwarePolicyDefersUnknownWithoutEnvironmentFallback(t *testing.T) {
	root := t.TempDir()
	if automatic, err := automaticSoftwareUpdates(root); err != nil || !automatic {
		t.Fatal(automatic, err)
	}
	path := filepath.Join(root, "etc/cozy/software-policy.json")
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{`{"startup_update":"off"}`, `{"startup_update":"future"}`, `broken`} {
		if err := os.WriteFile(path, []byte(value), 0600); err != nil {
			t.Fatal(err)
		}
		if automatic, _ := automaticSoftwareUpdates(root); automatic {
			t.Fatal("invalid or pinned policy enabled discovery")
		}
	}
}

func TestCandidateBundleRequiresContractsWithoutReleaseNumberGate(t *testing.T) {
	m := &Machine{layout: NewLayout(&Grant{Root: t.TempDir()})}
	overlay := t.TempDir()
	if err := m.verifyBundledAgent(context.Background(), overlay); err == nil {
		t.Fatal("unbundled candidate accepted")
	}
	binary := filepath.Join(overlay, "bin/cozy-machine")
	for _, row := range []struct {
		caps string
		ok   bool
	}{
		{`["runtime-update/1"]`, false},
		{`["hub-access/1","runtime-update/1","machine-bootstrap/1","future/2"]`, true},
	} {
		executable(t, binary, fmt.Sprintf(`echo '{"name":"cozy-machine","version":"future.dev","wire_minor":%d,"minimum_wire_minor":%d,"capabilities":%s}'`, pb.WireMinor, pb.MinCompatibleWireMinor, row.caps))
		if err := m.verifyBundledAgent(context.Background(), overlay); (err == nil) != row.ok {
			t.Fatalf("contract admission: %v", err)
		}
	}
}

func TestMaintenanceProgressCapturesConcurrentOutputStreams(t *testing.T) {
	cmd := exec.Command("sh", "-c", `i=0; while [ "$i" -lt 1000 ]; do echo output; echo diagnostic >&2; i=$((i+1)); done`)
	raw, err := progressChildOutput(cmd, true)
	if err != nil || strings.Count(string(raw), "output\n") != 1000 || strings.Count(string(raw), "diagnostic\n") != 1000 {
		t.Fatalf("maintenance output lost: %v (%d bytes)", err, len(raw))
	}
}

func TestKeptImagePairDistinguishesExplicitSameVersionBytes(t *testing.T) {
	root := t.TempDir()
	dev := filepath.Join(root, "var/lib/cozy/dev")
	if err := os.MkdirAll(filepath.Join(dev, "initial"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("initial", filepath.Join(dev, "current")); err != nil {
		t.Fatal(err)
	}
	var wheels []string
	for _, name := range []string{"cozy_runtime-0.18.88-py3-none-any.whl", "tensorfs-0.3.78-py3-none-any.whl"} {
		path := filepath.Join(root, name)
		if err := os.WriteFile(path, []byte("first"), 0600); err != nil {
			t.Fatal(err)
		}
		wheels = append(wheels, path)
	}
	if err := selectPair(root, wheels); err != nil {
		t.Fatal(err)
	}
	first, err := os.Readlink(filepath.Join(dev, "current"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(wheels[0], []byte("explicit replacement bytes"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := selectPair(root, wheels); err != nil {
		t.Fatal(err)
	}
	second, err := os.Readlink(filepath.Join(dev, "current"))
	if err != nil {
		t.Fatal(err)
	}
	if first == second {
		t.Fatal("same filename reused older wheel custody")
	}
	raw, err := os.ReadFile(filepath.Join(dev, "current", filepath.Base(wheels[0])))
	if err != nil || string(raw) != "explicit replacement bytes" {
		t.Fatalf("kept explicit bytes: %s %v", raw, err)
	}
}

func TestDefaultHelpersDoNotInheritAgentCredentials(t *testing.T) {
	t.Setenv("COZY_WORKER_AUTH_TOKEN", "fixture-worker-secret")
	t.Setenv("TENSORHUB_PRIVATE_TOKEN", "fixture-hub-secret")
	t.Setenv("UNRELATED_API_KEY", "fixture-other-secret")
	raw, err := childOutput(exec.Command("env"), false)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "fixture-") {
		t.Fatal("default helper inherited credentials")
	}
	cmd := exec.Command("env")
	cmd.Env = []string{"COZY_EXPLICIT_RUNTIME_GRANT=fixture-explicit"}
	raw, err = childOutput(cmd, false)
	if err != nil || string(raw) != "COZY_EXPLICIT_RUNTIME_GRANT=fixture-explicit\n" {
		t.Fatal("explicit Runtime launch grant was changed")
	}
}

func TestCandidatePairUsesCandidateWireRange(t *testing.T) {
	root := t.TempDir()
	m := &Machine{layout: NewLayout(&Grant{Root: root})}
	agent := fmt.Sprintf(`echo '{"name":"cozy-machine","wire_minor":%d,"minimum_wire_minor":%d,"capabilities":["hub-access/1","runtime-update/1","machine-bootstrap/1"]}'`, pb.WireMinor+20, pb.WireMinor+10)
	executable(t, filepath.Join(root, "opt/cozy/python/bin/cozy-machine"), agent)
	for _, floor := range []uint32{pb.WireMinor + 10, pb.WireMinor + 21} {
		executable(t, filepath.Join(root, "opt/cozy/python/bin/python"), fmt.Sprintf(`echo '{"runtime":"0.18.88","tensorfs":"0.3.78","wire_minor":%d,"minimum_wire_minor":%d,"capabilities":["machine-supervisor/1","machine-public-reads/1"]}'`, pb.WireMinor+22, floor))
		_, err := m.startupVerify(context.Background())
		if (err == nil) != (floor == pb.WireMinor+10) {
			t.Fatalf("candidate range floor=%d: %v", floor, err)
		}
	}
}

func TestInstalledPairProbeUsesActualRunningApplication(t *testing.T) {
	root := t.TempDir()
	running := filepath.Join(root, "running-agent")
	m := &Machine{grant: &Grant{Root: root, agentBefore: running}, layout: NewLayout(&Grant{Root: root}), log: io.Discard}
	executable(t, running, fmt.Sprintf(`echo '{"name":"cozy-machine","wire_minor":%d,"minimum_wire_minor":%d,"capabilities":["hub-access/1","runtime-update/1","machine-bootstrap/1"]}'`, pb.WireMinor, pb.MinCompatibleWireMinor))
	executable(t, filepath.Join(root, "opt/cozy/python/bin/python"), fmt.Sprintf(`echo '{"runtime":"0.18.87","tensorfs":"0.3.78","wire_minor":%d,"minimum_wire_minor":%d,"capabilities":["machine-supervisor/1","machine-public-reads/1"]}'`, pb.WireMinor, pb.MinCompatibleWireMinor))
	if _, err := m.startupVerify(context.Background()); err != nil {
		t.Fatalf("missing installed bundle bricked working repair agent: %v", err)
	}
	if err := m.startupSave(startupStatus{State: "boot_pending"}); err != nil {
		t.Fatal(err)
	}
	if _, err := m.startupVerify(context.Background()); err == nil {
		t.Fatal("running repair agent stood in for a missing candidate")
	}
}

func TestCurrentPairStillNeedsTransactionForDifferentApplication(t *testing.T) {
	root := t.TempDir()
	running := filepath.Join(root, "running")
	m := &Machine{grant: &Grant{Root: root, agentBefore: running}, layout: NewLayout(&Grant{Root: root})}
	if err := writeAtomic(running, []byte("running bytes"), 0755); err != nil {
		t.Fatal(err)
	}
	bundle := filepath.Join(root, "opt/cozy/python/bin/cozy-machine")
	if err := writeAtomic(bundle, []byte("candidate bytes"), 0755); err != nil {
		t.Fatal(err)
	}
	if m.applicationMatchesBundle() {
		t.Fatal("same distribution versions hid different application")
	}
	if err := writeAtomic(bundle, []byte("running bytes"), 0755); err != nil {
		t.Fatal(err)
	}
	if !m.applicationMatchesBundle() {
		t.Fatal("identical current application needs no replacement")
	}
}

// Distribution metadata is not an atomic pair commit: uv can be interrupted
// before either metadata replacement, between them, or after both. Even the
// original versions can accompany partly installed payloads or dependencies.
func TestInterruptedPairRecoversEveryOwnedInstallationState(t *testing.T) {
	before := pairVersions{Runtime: "0.18.85", TensorFS: "0.3.77"}
	after := pairVersions{Runtime: "0.18.86", TensorFS: "0.3.78"}
	for _, row := range []struct {
		name             string
		installed, after pairVersions
	}{
		{"before", before, after},
		{"runtime-first", pairVersions{Runtime: after.Runtime, TensorFS: before.TensorFS}, after},
		{"tensorfs-first", pairVersions{Runtime: before.Runtime, TensorFS: after.TensorFS}, after},
		{"after", after, after},
		{"application-and-dependencies-only", before, before},
	} {
		for _, phase := range []string{"installing", "boot_pending"} {
			t.Run(row.name+"/"+phase, func(t *testing.T) {
				root := t.TempDir()
				m := &Machine{grant: &Grant{}, layout: NewLayout(&Grant{Root: root}), log: io.Discard}
				var previous []string
				for _, name := range []string{"cozy_runtime-0.18.85-py3-none-any.whl", "tensorfs-0.3.77-py3-none-any.whl"} {
					path := filepath.Join(root, name)
					if err := os.WriteFile(path, []byte("retained test wheel"), 0600); err != nil {
						t.Fatal(err)
					}
					previous = append(previous, path)
				}
				kept, err := m.startupPrevious(previous)
				if err != nil {
					t.Fatal(err)
				}
				state := startupStatus{State: phase, Before: before, After: row.after, Previous: kept,
					Dependencies: []dependencyWheel{{Name: "owned-dependency", Version: "1.0"}}}
				state.PolicyBefore = []byte(`{"startup_update":"auto","agent":"bundled"}`)
				state.PolicyAfter = &softwarePolicy{"off", "explicit"}
				state.EntryBefore, state.EntryAfter = "/previous-agent", "/candidate-agent"
				currentPolicy, _ := json.Marshal(state.PolicyAfter)
				if err := os.MkdirAll(filepath.Dir(policyPath(root)), 0755); err != nil {
					t.Fatal(err)
				}
				if err := writeAtomic(policyPath(root), currentPolicy, 0644); err != nil {
					t.Fatal(err)
				}
				if err := writeAgentEntry(root, state.EntryAfter); err != nil {
					t.Fatal(err)
				}
				if err := m.startupSave(state); err != nil {
					t.Fatal(err)
				}
				log := filepath.Join(root, "installer.log")
				// These are inert shell doubles, never Python or a Runtime process.
				executable(t, filepath.Join(root, "usr/local/bin/uv"), fmt.Sprintf("echo \"$1 $2\" >> %q", log))
				executable(t, filepath.Join(root, "opt/cozy/python/bin/python"), fmt.Sprintf(`
case "$4" in
 probe) echo '{"runtime":"%s","tensorfs":"%s"}' ;;
 removable-dependencies) echo '["owned-dependency"]' ;;
 *) printf '%s\n%s\n' ;;
esac`, before.Runtime, before.TensorFS, row.installed.Runtime, row.installed.TensorFS))
				if got, err := m.installedPair(); err != nil || got != row.installed {
					t.Fatalf("metadata double: %+v %v", got, err)
				}
				answer, err := maintainWithAgent(root, nil, "startup-prepare", []string{"persistent", m.layout.Store}, "")
				if err != nil || answer != "rolled_back" {
					t.Fatalf("fresh recovery did not request application restoration: %q %v", answer, err)
				}
				answer, err = maintainWithAgent(root, nil, "startup-prepare", []string{"persistent", m.layout.Store}, "")
				if err != nil || answer != "" {
					t.Fatalf("historical rollback overrode a later bootstrap: %q %v", answer, err)
				}
				raw, err := os.ReadFile(m.startupPath("status.json"))
				if err != nil || json.Unmarshal(raw, &state) != nil || state.State != "rolled_back" {
					t.Fatalf("incomplete transaction escaped recovery: %s (%v)", raw, err)
				}
				if len(state.Failed) != 1 || state.Failed[0] != row.after {
					t.Fatalf("failed candidate was not remembered: %+v", state.Failed)
				}
				selected, err := keptPair(root)
				if err != nil || !samePair(selected, before) {
					t.Fatalf("retained pair not restored: %v %v", selected, err)
				}
				policy, err := readPolicyBytes(root)
				if err != nil || string(policy) != string(state.PolicyBefore) {
					t.Fatalf("policy not restored: %s %v", policy, err)
				}
				entry, err := os.Readlink(filepath.Join(root, "usr/local/bin/cozy-machine"))
				if err != nil || entry != state.EntryBefore {
					t.Fatalf("application entry not restored: %s %v", entry, err)
				}
				raw, err = os.ReadFile(log)
				if err != nil || string(raw) != "pip install\npip uninstall\n" {
					t.Fatalf("pair and owned dependencies not restored: %q (%v)", raw, err)
				}
			})
		}
	}
}

// Keep the test process alive while exercising real rollback journal/status writes.
type rollbackObservationLauncher struct{ directLauncher }

func (l *rollbackObservationLauncher) maintain(op string, args []string) (string, error) {
	if op == "agent-replace" {
		return "", nil
	}
	return l.directLauncher.maintain(op, args)
}
