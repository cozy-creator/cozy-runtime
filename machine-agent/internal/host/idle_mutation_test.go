package host

import (
	"context"
	"errors"
	"io"
	"strings"
	"syscall"
	"testing"
)

type idleMutationLauncher struct {
	reads, restarts int
	before, after   bool
	answer          string
}

func (l *idleMutationLauncher) maintain(op string, _ []string) (string, error) {
	switch op {
	case "software-state":
		l.reads++
		quiescent := l.before
		if l.reads > 1 {
			quiescent = l.after
		}
		if quiescent {
			return `{"quiescent":true}`, nil
		}
		return `{"quiescent":false}`, nil
	case "restart":
		l.restarts++
		return l.answer, nil
	case "capabilities":
		return `{"capabilities":["machine-supervisor/1"]}`, nil
	}
	return "", errors.New("unexpected operation")
}
func (*idleMutationLauncher) launch(string) (*runtimeProcess, error) { return nil, syscall.EAGAIN }
func (*idleMutationLauncher) close()                                 {}

func TestIdleMutationRequiresRuntimeQuiescenceAndDoesNotWaitForBusyWork(t *testing.T) {
	for _, arm := range []struct {
		name          string
		before, after bool
		answer        string
		mutates       bool
	}{
		{"retained", false, false, "accepted", false},
		{"busy admission", true, true, "busy", false},
		{"retained after stop", true, false, "accepted", false},
		{"idle", true, true, "accepted", true},
	} {
		t.Run(arm.name, func(t *testing.T) {
			launcher := &idleMutationLauncher{before: arm.before, after: arm.after, answer: arm.answer}
			m := &Machine{grant: &Grant{}, layout: NewLayout(&Grant{Root: t.TempDir()}), launcher: launcher, restarts: &restarts{}, log: io.Discard}
			mutated := false
			err := m.mutateIdleRuntime(context.Background(), func() error { mutated = true; return nil })
			if mutated != arm.mutates {
				t.Fatal("mutation did not respect Runtime evidence")
			}
			if launcher.restarts > 1 {
				t.Fatal("busy work was polled instead of refused")
			}
			if m.updating() {
				t.Fatal("mutation leaked its admission reservation")
			}
			if !arm.mutates && (err == nil || !strings.Contains(err.Error(), "machine_busy")) {
				t.Fatalf("missing busy refusal: %v", err)
			}
			if arm.mutates && (!errors.Is(err, syscall.EAGAIN) || m.restarts.gone()) {
				t.Fatalf("relaunch failure closed repair or hid mutation: %v", err)
			}
		})
	}
}
