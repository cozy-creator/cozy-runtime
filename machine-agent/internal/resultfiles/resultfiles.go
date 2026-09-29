// Package resultfiles is how a result file is NAMED in its durable directory — the
// package's own store under outputs/ or a caller's --out — and how that directory is
// proven writable before a run is queued. The worker writes the file there itself under
// this same name (cozy-runtime `author._media.EXTENSIONS` spells the identical table);
// the daemon recomputes the name from the accepted manifest and reads exactly it, so
// execution never composes a path from a terminal document.
package resultfiles

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/exit"
)

// Filename names one verified file by its own content digest and media type:
// `<sha256 hex>.<ext>`. The same bytes always get the same name, so a regenerated file
// lands on itself and two different files can never collide.
func Filename(digest, mediaType string) (string, *exit.Error) {
	hex, ok := strings.CutPrefix(digest, "sha256:")
	if !ok || len(hex) != sha256.Size*2 || strings.Trim(hex, "0123456789abcdef") != "" {
		return "", exit.Named(exit.Validation, "output_export_digest_malformed",
			"output digest %q is not sha256 over 64 lowercase hexadecimal characters", digest)
	}
	return hex + Extension(mediaType), nil
}

// Extension is the closed media-to-filename projection, equal to cozy-runtime's
// `author._media.EXTENSIONS`. Unknown bytes get no guessed suffix.
func Extension(mediaType string) string {
	switch mediaType {
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	case "video/mp4":
		return ".mp4"
	case "audio/wav":
		return ".wav"
	case "application/json":
		return ".json"
	default:
		return ""
	}
}

// Preflight proves the destination can accept a published file BEFORE a request is
// queued: create-if-absent, then a touch-and-unlink probe in the final directory. It
// runs in the daemon — the process that later publishes — so it proves the writer's own
// access, not a client's. A probe that passes is not a reservation; the durable export
// still handles a destination that dies later.
func Preflight(directory string) *exit.Error {
	if !filepath.IsAbs(directory) || filepath.Clean(directory) != directory {
		return exit.Named(exit.Validation, "output_export_directory_malformed",
			"output directory %q is not one canonical absolute path", directory)
	}
	if err := os.MkdirAll(directory, 0o755); err != nil {
		return unwritable(directory, err)
	}
	info, err := os.Lstat(directory)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return exit.Named(exit.Conflict, "output_export_directory_substituted",
			"output directory %s is not a real directory", directory)
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return unwritable(directory, err)
	}
	defer root.Close()
	name := ".cozy-probe-" + randomSuffix()
	probe, err := root.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return unwritable(directory, err)
	}
	problem := probe.Close()
	if err := root.Remove(name); problem == nil {
		problem = err
	}
	if problem != nil {
		return unwritable(directory, problem)
	}
	return nil
}

func randomSuffix() string {
	var raw [12]byte
	if _, err := rand.Read(raw[:]); err != nil {
		panic(fmt.Sprintf("crypto/rand: %s", err))
	}
	return hex.EncodeToString(raw[:])
}

func unwritable(directory string, err error) *exit.Error {
	return exit.Named(exit.Validation, "output_destination_unwritable",
		"output destination %s is not writable: %s", directory, ioCause(err)).
		WithRemedy("fix permissions on %s (or pick another --out) and resubmit; nothing was queued", directory)
}

// ioCause strips the probe's own random filename out of the refusal: the caller's
// remediable fact is the directory and the errno, not a name that never existed.
func ioCause(err error) error {
	var pathError *fs.PathError
	if errors.As(err, &pathError) {
		return pathError.Err
	}
	return err
}
