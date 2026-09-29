package host

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// TensorFS owns model bytes. The machine asks `tfs` for them and reads only its typed JSON
// events and exit status, never text meant for people.
type tensorFS struct {
	bin, store string
	repoCache  string
	streams    atomic.Int64 // the stream level the last pull settled at
	observe    func(cacheObservation)
	capsOnce   sync.Once
	caps       map[string]bool
}

// hubSource is where a fetch reads Models: one Hub's public origin, this machine's worker
// capability there, its private CA and the hosts its bytes live on. Cache observations are
// reported to the machine's own Hub only.
type hubSource struct {
	hub, caFile, credential string
	allowHosts              []string
	own                     bool
}

// run execs tfs with an empty environment plus env; every stdout and stderr line goes to lines.
func (t *tensorFS) run(ctx context.Context, env []string, lines func(string), args ...string) error {
	cmd := exec.CommandContext(ctx, t.bin, args...)
	cmd.Env = append([]string{}, env...)
	out, errs := &lineWriter{line: lines}, &lineWriter{line: lines}
	cmd.Stdout, cmd.Stderr = out, errs
	err := cmd.Run()
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

// capable reports a capability of the installed tfs; an older tfs has none.
func (t *tensorFS) capable(ctx context.Context, name string) bool {
	t.capsOnce.Do(func() {
		t.caps = map[string]bool{}
		output, err := exec.CommandContext(ctx, t.bin, "capabilities").Output()
		if err != nil {
			return
		}
		for _, line := range strings.Split(string(output), "\n") {
			t.caps[strings.TrimSpace(line)] = true
		}
	})
	return t.caps[name]
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

// terminalFetch are the old `tfs fetch` codes that retrying cannot change.
var terminalFetch = map[string]bool{"OBJECT_ID_MISMATCH": true, "LENGTH_MISMATCH": true, "WHOLE_DIGEST_MISMATCH": true,
	"OBJECT_CORRUPT": true, "MALFORMED_DIGEST": true, "SOURCE_NOT_ALLOWED": true, "REDIRECT_REFUSED": true,
	"PERMISSION_DENIED": true, "STORE_ERA": true, "TLS_UNTRUSTED": true}

type tfsEvent struct {
	Event       string `json:"event"`
	TotalBytes  int64  `json:"total_bytes"`
	HeldBytes   int64  `json:"held_bytes"`
	OriginBytes int64  `json:"origin_bytes"`
	CacheBytes  int64  `json:"cache_bytes"`
	BytesDone   int64  `json:"bytes_done"`
	BytesTotal  int64  `json:"bytes_total"`
	Streams     int64  `json:"streams"`
	Code        string `json:"code"`
	Detail      string `json:"detail"`
	Resumable   bool   `json:"resumable"`
}

type cacheObservation struct {
	Model        string `json:"model"`
	Manifest     string `json:"manifest"`
	Scope        string `json:"scope"`
	TotalBytes   int64  `json:"total_bytes"`
	ReadBytes    int64  `json:"cache_bytes"`
	WrittenBytes int64  `json:"cache_written_bytes"`
}

// downloadModel is one row of a download set; adapters flatten into their own rows.
type downloadModel struct {
	Model, Release, Manifest, Lane, Package, Slot string
	Adapters                                      []downloadModel
}

func (m downloadModel) refspec() string {
	spec := m.Model
	for _, pin := range []string{m.Release, m.Manifest} {
		if pin != "" {
			spec += "@" + pin
		}
	}
	return spec
}

func modelsOf(desired []byte) ([]downloadModel, error) {
	var set struct{ Models []downloadModel }
	if err := json.Unmarshal(desired, &set); err != nil {
		return nil, fmt.Errorf("the download set is not a JSON document: %w", err)
	}
	var out []downloadModel
	seen := map[string]bool{}
	for _, model := range set.Models {
		for _, row := range append([]downloadModel{model}, model.Adapters...) {
			key := row.refspec() + "\x00" + row.Lane
			if row.Model == "" || strings.HasPrefix(row.Lane, "-") || seen[key] {
				continue
			}
			seen[key] = true
			row.Package, row.Slot, row.Adapters = model.Package, model.Slot, nil
			out = append(out, row)
		}
	}
	return out, nil
}

// fetch lands every model the download set names through `tfs ensure`, or `tfs fetch` on a
// TensorFS that predates it, reporting landed and total bytes as they move.
func (t *tensorFS) fetch(ctx context.Context, source hubSource, desired []byte, progress func(landed, total int64)) error {
	models, err := modelsOf(desired)
	if err != nil {
		return &fetchRefusal{Code: "model_fetch_set_invalid", Detail: err.Error()}
	}
	if len(models) > 0 && source.hub == "" {
		return &fetchRefusal{Code: "model_fetch_origin_absent", Detail: "this machine was granted no TENSORHUB_PUBLIC_ORIGIN"}
	}
	verb := "fetch"
	if t.capable(ctx, "ensure/1") {
		verb = "ensure"
	}
	var doneBytes int64
	for _, model := range models {
		args := []string{verb, t.store, model.refspec(), "--hub", source.hub}
		if source.caFile != "" {
			args = append(args, "--ca-file", source.caFile)
		}
		if model.Lane != "" {
			args = append(args, "--lane", model.Lane)
		}
		if len(source.allowHosts) > 0 {
			args = append(args, "--allow-hosts", strings.Join(source.allowHosts, ","))
		}
		if streams := t.streams.Load(); streams > 0 {
			args = append(args, "--streams-start", strconv.FormatInt(streams, 10))
		}
		var landed, total int64
		var refused *fetchRefusal
		lines := func(line string) {
			var event tfsEvent
			if !strings.HasPrefix(line, "{") || json.Unmarshal([]byte(line), &event) != nil {
				return
			}
			switch event.Event {
			case "pull.progress":
				landed, total = event.HeldBytes+event.OriginBytes+event.CacheBytes, event.TotalBytes
			case "ensure.progress":
				landed, total = event.BytesDone, event.BytesTotal
			case "pull.streams":
				if event.Streams > 0 {
					t.streams.Store(event.Streams)
				}
			case "pull.cache":
				var seen cacheObservation
				if t.observe != nil && source.own && json.Unmarshal([]byte(line), &seen) == nil && seen.Model == model.Model {
					t.observe(seen)
				}
				return
			case "ensure.refused":
				refused = &fetchRefusal{Code: "model_fetch_" + strings.ToLower(event.Code),
					Detail: model.refspec() + ": " + event.Detail, Resumable: event.Resumable}
				return
			default:
				return
			}
			if progress != nil {
				progress(doneBytes+landed, doneBytes+total)
			}
		}
		if err := t.run(ctx, []string{"TFS_CREDENTIAL=" + source.credential}, lines, args...); err != nil {
			if refused != nil {
				return refused
			}
			if old, ok := err.(*fetchRefusal); ok {
				return &fetchRefusal{Code: "model_fetch_" + strings.ToLower(cmpOr(old.Code, "failed")),
					Detail: model.refspec() + ": " + old.Detail, Resumable: !terminalFetch[old.Code]}
			}
			return err
		}
		doneBytes += total
		if progress != nil {
			progress(doneBytes, doneBytes)
		}
	}
	return nil
}

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
