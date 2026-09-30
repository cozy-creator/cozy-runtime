package host

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
)

// idleGrace is the fixed idle window: a machine no one uses for this long releases itself.
const idleGrace = time.Duration(pb.RentalIdleTimeoutSeconds) * time.Second

var errReleased = errors.New("this machine's idle release is due or committed")

// idleState is the machine's idle ledger, rewritten only when a fact changes.
type idleState struct {
	Deadline     int64               `json:"deadline_ms"`
	Released     bool                `json:"released"` // the claim is irreversible for a rental
	WorkObserved bool                `json:"work_observed"`
	Unknown      bool                `json:"unknown"` // activity unreadable after work: it holds
	Keepalives   map[string][2]int64 `json:"keepalives,omitempty"`
}

type idle struct {
	persistent bool
	path       string
	mu         sync.Mutex
	s          idleState
	renewed    int64
	admissions map[*admission]struct{}
}

type admission struct {
	i    *idle
	call context.Context
}

func openIdle(path string, fresh bool, now time.Time) (*idle, error) {
	i := &idle{path: path, admissions: map[*admission]struct{}{}}
	if raw, err := os.ReadFile(path); err == nil && !fresh {
		if err := json.Unmarshal(raw, &i.s); err != nil {
			return nil, fmt.Errorf("the idle ledger is unreadable: %w", err)
		}
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if i.s.Deadline == 0 || fresh {
		i.s = idleState{Deadline: now.Add(idleGrace).UnixMilli()}
		if err := i.save(); err != nil {
			return nil, err
		}
	}
	return i, nil
}

func (i *idle) save() error {
	raw, _ := json.Marshal(i.s)
	return writeAtomic(i.path, raw, 0o600)
}

// observe folds one activity sample into the deadline. Busy work renews it; activity that
// cannot be read after work was seen holds it; an idle machine lets it run out.
func (i *idle) observe(now time.Time, busy, known bool) (time.Time, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.s.Released {
		return now, errReleased
	}
	hadWork := i.s.WorkObserved || known && busy
	unknown := !known && hadWork
	// A live Runtime with unreadable activity holds the deadline. Once the
	// Runtime is known dead, its prior unknown sample must not start a fresh
	// idle window.
	if known && busy || unknown {
		i.renewed = max(i.renewed, now.Add(idleGrace).UnixMilli())
	}
	target := max(i.s.Deadline, i.renewed)
	if unknown == i.s.Unknown && hadWork == i.s.WorkObserved && target-i.s.Deadline < idleGrace.Milliseconds()/2 {
		return time.UnixMilli(target), nil
	}
	i.s.Deadline, i.s.Unknown, i.s.WorkObserved = target, unknown, hadWork
	return time.UnixMilli(target), i.save()
}

// work renews the deadline for accepted work.
func (i *idle) work(now time.Time) error {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.s.Released {
		return errReleased
	}
	i.s.Deadline, i.s.WorkObserved = max(i.s.Deadline, now.Add(idleGrace).UnixMilli()), true
	return i.save()
}

// admit holds idle release while a call that may start work is in flight.
func (i *idle) admit(call context.Context, now time.Time) (*admission, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if !i.persistent && (i.s.Released || now.UnixMilli() >= max(i.s.Deadline, i.renewed)) {
		return nil, errReleased
	}
	a := &admission{i: i, call: call}
	i.admissions[a] = struct{}{}
	return a, nil
}

func (a *admission) finish(work bool, now time.Time) error {
	a.i.mu.Lock()
	delete(a.i.admissions, a)
	a.i.mu.Unlock()
	if !work {
		return nil
	}
	return a.i.work(now)
}

// claim takes the idle release once its deadline passed and nothing holds it.
func (i *idle) claim(now time.Time) (bool, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	for a := range i.admissions {
		if a.call.Err() != nil {
			delete(i.admissions, a)
		}
	}
	if i.persistent {
		return false, nil
	}
	if i.s.Released {
		return true, nil
	}
	if len(i.admissions) > 0 || i.s.Unknown || now.UnixMilli() < max(i.s.Deadline, i.renewed) {
		return false, nil
	}
	i.s.Released = true
	return true, i.save()
}

// reset begins a new idle session: an owned machine keeps serving after it releases.
func (i *idle) reset(now time.Time) error {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.renewed = 0
	i.s = idleState{Deadline: now.Add(idleGrace).UnixMilli()}
	return i.save()
}

// keepalive is an explicit owner action, idempotent per request id.
func (i *idle) keepalive(id string, now time.Time) (time.Time, time.Time, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if !i.persistent && (i.s.Released || now.UnixMilli() >= max(i.s.Deadline, i.renewed)) {
		return time.Time{}, time.Time{}, errReleased
	}
	if seen, ok := i.s.Keepalives[id]; ok {
		return time.UnixMilli(seen[0]), time.UnixMilli(seen[1]), nil
	}
	if i.s.Keepalives == nil || len(i.s.Keepalives) >= 256 {
		i.s.Keepalives = map[string][2]int64{}
	}
	deadline := now.Add(idleGrace).UnixMilli()
	i.s.Keepalives[id] = [2]int64{now.UnixMilli(), deadline}
	i.s.Deadline = max(i.s.Deadline, deadline)
	return now, time.UnixMilli(deadline), i.save()
}

func (i *idle) deadline() time.Time {
	i.mu.Lock()
	defer i.mu.Unlock()
	return time.UnixMilli(max(i.s.Deadline, i.renewed))
}

// activity reads Runtime demand. Quiet counters do not prove that active work ended.
type activity struct {
	path   string
	log    io.Writer
	said   string
	saidAt time.Time
}

const activityFreshness = 10 * time.Second

func (a *activity) observe(now time.Time) (busy bool, err error) {
	file, err := os.Open(a.path)
	if err != nil {
		return false, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return false, err
	}
	if now.Sub(info.ModTime()) > activityFreshness || info.ModTime().After(now.Add(activityFreshness)) {
		return false, errors.New("the Runtime's activity is stale")
	}
	var facts struct {
		ActiveWork *bool    `json:"active_work"`
		Holding    []string `json:"holding"`
	}
	if err := json.NewDecoder(io.LimitReader(file, 4096)).Decode(&facts); err != nil || facts.ActiveWork == nil {
		return false, errors.New("the Runtime's activity does not report active_work")
	}
	if !*facts.ActiveWork {
		return false, nil
	}
	holders := strings.Join(facts.Holding, ", ")
	a.say(now, "cozy machine: idle held by the Runtime: "+holders)
	return true, nil
}

func (a *activity) say(now time.Time, line string) {
	if line == a.said && now.Sub(a.saidAt) < time.Minute {
		return
	}
	fmt.Fprintln(a.log, line)
	a.said, a.saidAt = line, now
}

// restarts paces transient failures and exposes structural failures until repair/update.
// Pre-readiness failures survive request-plane and machine restarts; only a proved
// readiness or explicit repair clears them. They are not workload observations.
type restarts struct {
	path     string
	mu       sync.Mutex
	failures uint
	retryAt  time.Time
	blocked  bool
	detail   string
	safeCode string
}

type restartState struct {
	Failures uint   `json:"pre_ready_failures"`
	Blocked  bool   `json:"blocked"`
	Detail   string `json:"detail,omitempty"`
	Code     string `json:"code,omitempty"`
}

func openRestarts(path string) (*restarts, error) {
	r := &restarts{path: path}
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return r, nil
	}
	if err != nil {
		return nil, err
	}
	var s restartState
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("Runtime restart state unreadable: %w", err)
	}
	r.failures, r.blocked, r.detail, r.safeCode = s.Failures, s.Blocked, s.Detail, s.Code
	return r, nil
}

func (r *restarts) save() error {
	if r.path == "" {
		return nil
	}
	raw, _ := json.Marshal(restartState{r.failures, r.blocked, r.detail, r.safeCode})
	return writeAtomic(r.path, raw, 0o600)
}

func (r *restarts) retain() {
	if err := r.save(); err != nil {
		r.blocked = true
		r.safeCode = "runtime_restart_state_unavailable"
		r.detail = fmt.Sprintf("cannot persist Runtime restart state: %v", err)
	}
}

func (r *restarts) gone() bool { r.mu.Lock(); defer r.mu.Unlock(); return r.blocked }
func (r *restarts) due(now time.Time) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return !r.blocked && !now.Before(r.retryAt)
}
func (r *restarts) code() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return cmpOr(r.safeCode, "runtime_failed")
}
func (r *restarts) reason() string { r.mu.Lock(); defer r.mu.Unlock(); return r.detail }
func (r *restarts) clear() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.failures = 0
	r.blocked = false
	r.detail, r.safeCode = "", ""
	r.retryAt = time.Time{}
	return r.save()
}
func (r *restarts) fail(err error) {
	var refusal *launchRefusal
	if !errors.As(err, &refusal) {
		r.exited(err, false)
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.blocked = true
	r.detail = err.Error()
	if errors.As(err, &refusal) {
		r.safeCode = refusal.code
	}
	r.retain()
}

// exited paces a relaunch. An exit after readiness is progress and relaunches;
// a second consecutive exit before readiness blocks until readiness or repair.
func (r *restarts) exited(err error, wasReady bool) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.detail = "Runtime exited: " + exitText(err)
	var exited interface{ ExitCode() int }
	if errors.As(err, &exited) && exited.ExitCode() == 6 {
		r.safeCode = "runtime_launch_refused"
		r.blocked = true
	} else if wasReady {
		r.failures = 0
	} else if r.failures++; r.failures >= 2 {
		r.safeCode = "runtime_readiness_failed"
		r.blocked = true
	}
	r.retain()
	if r.blocked {
		return false
	}
	r.retryAt = time.Now().Add(time.Second)
	return true
}

func (i *idle) releasedNow() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.s.Released
}
