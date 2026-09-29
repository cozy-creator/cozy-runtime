package host

import (
	"context"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host/outputs"
)

// serveOutput is GET /v1/runs/<n>/outputs/<output>[/<i>]: the output's current bytes with
// full Range semantics. The ETag is the revision, so If-None-Match and If-Range hold across
// rewrites; the digest appears only on the final revision. A capability in the Authorization
// header (`Cozy-Cap <token>`) grants the run's outputs.
func (m *Machine) serveOutput(w http.ResponseWriter, r *http.Request) {
	run, err := strconv.ParseUint(r.PathValue("run"), 10, 64)
	output, index := r.PathValue("output"), -1
	if text := r.PathValue("index"); text != "" && err == nil {
		index, err = strconv.Atoi(text)
		if index < 1 {
			err = errors.New("a list index is 1-based")
		}
	}
	if err != nil || run == 0 || output == "" {
		http.Error(w, "the path names no run output", http.StatusNotFound)
		return
	}
	token, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Cozy-Cap ")
	grant, err := capability.Verify(token, m.grant.WorkerID, m.claims.authorizedKeys(), time.Now(), "")
	if err == nil && !grant.Allows(r.PathValue("run"), output, index) {
		err = capability.ErrScope
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}
	snapshot, err := m.Open(run, output, index)
	switch {
	case errors.Is(err, outputs.ErrNotFound):
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	case errors.Is(err, outputs.ErrUnavailable):
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	case errors.Is(err, outputs.ErrUpdateRequired):
		http.Error(w, "capability_unavailable: "+err.Error(), http.StatusNotImplemented)
		return
	case errors.Is(err, context.DeadlineExceeded):
		http.Error(w, "the machine's Runtime did not answer", http.StatusServiceUnavailable)
		return
	case err != nil:
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer snapshot.Close()
	if m.phase() != "ready" {
		w.Header().Set("Cozy-Snapshot-Source", "journal")
		w.Header().Set("Cozy-Runtime-State", m.phase())
		w.Header().Set("Cozy-Runtime-Available", "false")
	}
	w.Header().Set("ETag", `"r`+strconv.FormatUint(snapshot.Rev, 10)+`"`)
	w.Header().Set("Cache-Control", "private, no-cache")
	if snapshot.MediaType != "" {
		w.Header().Set("Content-Type", snapshot.MediaType)
	} else {
		w.Header().Set("Content-Type", "application/octet-stream")
	}
	if hexDigest, final := strings.CutPrefix(snapshot.SHA256, "sha256:"); final && snapshot.Final {
		if raw, err := hex.DecodeString(hexDigest); err == nil {
			w.Header().Set("Repr-Digest", "sha-256=:"+base64.StdEncoding.EncodeToString(raw)+":")
		}
	}
	http.ServeContent(w, r, "", time.Time{}, io.NewSectionReader(snapshot.Body, 0, snapshot.Length))
}
