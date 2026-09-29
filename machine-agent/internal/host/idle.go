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
	if known && busy || unknown || i.s.Unknown {
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
	if i.s.Released || now.UnixMilli() >= max(i.s.Deadline, i.renewed) {
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
	if i.s.Released || now.UnixMilli() >= max(i.s.Deadline, i.renewed) {
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

// activity reads the Runtime's activity file. Work that holds is fresh, active, and moving:
// a meter that has not changed for the Runtime's own still window no longer holds (#872).
type activity struct {
	path   string
	log    io.Writer
	meter  string
	moved  time.Time
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
		Meter      string   `json:"meter"`
		Still      float64  `json:"still_floor_seconds"`
	}
	if err := json.NewDecoder(io.LimitReader(file, 4096)).Decode(&facts); err != nil || facts.ActiveWork == nil {
		return false, errors.New("the Runtime's activity does not report active_work")
	}
	if !*facts.ActiveWork {
		a.meter, a.moved = "", time.Time{}
		return false, nil
	}
	if facts.Meter != a.meter || a.moved.IsZero() {
		a.meter, a.moved = facts.Meter, now
	}
	holders := strings.Join(facts.Holding, ", ")
	if still := time.Duration(facts.Still * float64(time.Second)); facts.Meter != "" && still > 0 && now.Sub(a.moved) >= still {
		a.say(now, "cozy machine: the Runtime's work has not moved for "+still.String()+"; it no longer holds the machine: "+holders)
		return false, nil
	}
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

// restarts is the crash rule (#865/#869): relaunch without limit after the Runtime completed
// work; relaunch once after an exit without work; a second consecutive one leaves the
// machine known idle. The fact survives daemon restarts; work is never re-run.
type restarts struct {
	path             string
	worked, progress bool
}

func (r *restarts) fact() string {
	raw, _ := os.ReadFile(r.path)
	return strings.TrimSpace(string(raw))
}

func (r *restarts) gone() bool { return r.fact() == "gone" }

// observe records progress: work seen, then the machine idle again.
func (r *restarts) observe(busy bool, err error) error {
	if err != nil || r.progress {
		return nil
	}
	if busy {
		r.worked = true
		return nil
	}
	if !r.worked {
		return nil
	}
	r.progress = true
	return r.clear()
}

func (r *restarts) clear() error {
	if err := os.Remove(r.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// exited decides whether the Runtime is launched again.
func (r *restarts) exited() (bool, error) {
	progressed := r.progress
	r.worked, r.progress = false, false
	if progressed {
		return true, nil
	}
	next := "once"
	if r.fact() != "" {
		next = "gone"
	}
	if err := writeAtomic(r.path, []byte(next+"\n"), 0o600); err != nil {
		return false, err
	}
	return next == "once", nil
}

func (i *idle) releasedNow() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.s.Released
}
