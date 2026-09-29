package host

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"sync"
)

// TensorFS owns model bytes. The machine asks `tfs` for them and reads only its typed JSON
// events and exit status, never text meant for people.
type tensorFS struct {
	bin, store, repoCache string
}

// run execs tfs with an empty environment plus env; every stdout and stderr line goes to lines.
func (t *tensorFS) run(ctx context.Context, env []string, lines func(string), args ...string) error {
	cmd := exec.CommandContext(ctx, t.bin, args...)
	cmd.Env = append([]string{}, env...)
	out, errs := &lineWriter{line: lines}, &lineWriter{line: lines}
	cmd.Stdout, cmd.Stderr = out, errs
	err := runChild(cmd)
	out.flush()
	errs.flush()
	if err == nil {
		return nil
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return &fetchRefusal{Code: errs.refusal(), Detail: fmt.Sprintf("tfs %s: %v", args[0], err)}
}

// initStore makes sure the Store exists and records its repo-cache binding.
func (t *tensorFS) initStore(ctx context.Context) error {
	if t.run(ctx, nil, nil, "store", "info", t.store) != nil {
		if err := t.run(ctx, nil, nil, "store", "init", t.store); err != nil {
			return fmt.Errorf("initialize the TensorFS Store at %s: %w", t.store, err)
		}
	}
	bind := []string{"store", "ensure", t.store, "--no-repo-cache"}
	if t.repoCache != "" {
		bind = []string{"store", "ensure", t.store, "--repo-cache", t.repoCache}
	}
	if err := t.run(ctx, nil, nil, bind...); err != nil {
		return fmt.Errorf("bind the TensorFS Store's repo cache: %w", err)
	}
	return t.run(ctx, nil, nil, "store", "prepare-readers", t.store)
}

// fetchRefusal is a classified fetch outcome. A resumable one leaves the request valid.
type fetchRefusal struct {
	Code, Detail string
	Resumable    bool
}

func (r *fetchRefusal) Error() string { return r.Code + ": " + r.Detail }

// lineWriter splits a stream into lines and remembers a `REFUSED CODE: …` line's code.
type lineWriter struct {
	mu      sync.Mutex
	partial []byte
	line    func(string)
	code    string
}

func (w *lineWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.partial = append(w.partial, p...)
	for {
		at := bytes.IndexByte(w.partial, '\n')
		if at < 0 {
			break
		}
		w.handle(strings.TrimRight(string(w.partial[:at]), "\r"))
		w.partial = w.partial[at+1:]
	}
	if len(w.partial) > 1<<20 {
		w.partial = w.partial[:0]
	}
	return len(p), nil
}

func (w *lineWriter) handle(line string) {
	if rest, ok := strings.CutPrefix(strings.TrimSpace(line), "REFUSED "); ok {
		if code, _, _ := strings.Cut(rest, ":"); code != "" && !strings.ContainsAny(code, " \t") {
			w.code = code
		}
	}
	if w.line != nil {
		w.line(line)
	}
}

func (w *lineWriter) flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.partial) > 0 {
		w.handle(string(w.partial))
		w.partial = nil
	}
}

func (w *lineWriter) refusal() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.code
}
