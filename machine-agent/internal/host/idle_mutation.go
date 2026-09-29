package host

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/grpc/codes"
)

// mutateIdleRuntime performs an explicit owner mutation only after Runtime has
// atomically closed its admission. Busy work is refused immediately, never
// canceled or polled until it becomes idle. launchMu also excludes updates.
func (m *Machine) mutateIdleRuntime(ctx context.Context, mutate func() error) (err error) {
	m.launchMu.Lock()
	defer m.launchMu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if m.updating() || m.manualUpdateActive() {
		return refusal(codes.FailedPrecondition, "machine_busy", "Runtime maintenance already owns admission")
	}
	quiescent := func() error {
		raw, err := m.launcher.maintain("software-state", nil)
		if err != nil {
			return refusal(codes.FailedPrecondition, "quiescence_unavailable", err.Error())
		}
		var state softwareState
		if err := json.Unmarshal([]byte(raw), &state); err != nil {
			return refusal(codes.FailedPrecondition, "quiescence_unavailable", err.Error())
		}
		if !state.Quiescent {
			return refusal(codes.FailedPrecondition, "machine_busy", "Runtime retains accepted work or native activity")
		}
		return nil
	}
	if err := quiescent(); err != nil {
		return err
	}
	m.setUpdating(true)
	defer m.setUpdating(false)
	answer, err := m.launcher.maintain("restart", nil)
	if err != nil {
		return refusal(codes.FailedPrecondition, "quiescence_unavailable", err.Error())
	}
	if answer != "accepted" {
		return refusal(codes.FailedPrecondition, "machine_busy", "Runtime refused an idle mutation while work is active")
	}
	m.stopRuntime()
	defer func() {
		if launchErr := m.launch(); launchErr != nil {
			m.restarts.fail(launchErr)
			err = errors.Join(err, fmt.Errorf("Runtime relaunch after idle mutation: %w", launchErr))
		}
	}()
	if err := quiescent(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return mutate()
}
