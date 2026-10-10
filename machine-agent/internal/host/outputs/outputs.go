// Package outputs is how a machine hands out a run's outputs: the current bytes of each
// (output, index), addressed by revision, and the run's log as typed entries. The machine
// implements Source; its byte route and its WebRTC media both read through it.
package outputs

import (
	"context"
	"crypto/ed25519"
	"errors"
	"io"
)

// Snapshot is one consistent view of an output's current bytes. SHA256 is set only once the
// output is final.
type Snapshot struct {
	Body      io.ReaderAt // the current bytes: one store object, or the parts in order
	Length    int64
	Rev       uint64 // the 1-based ordinal of this (output, index)'s product entries in the log
	SHA256    string
	Final     bool
	MediaType string
}

// Entry is one entry of a run's log.
type Entry struct {
	Seq          uint64
	Output       string
	Index        int // a list item's 1-based index; -1: the output is not a list
	Rev          uint64
	Length       int64
	AppendedFrom *uint64 // the byte offset an append continued from
	DurationUS   uint64
	SHA256       string // sha256:<hex>, on the final revision only
	MediaType    string
	Label        string
	Status       string // "" for a product; completed, failed or canceled on the terminal entry
}

var (
	ErrNotFound       = errors.New("this machine has no such run or output")
	ErrUpdateRequired = errors.New("this machine's Runtime predates run outputs; update the machine's Runtime")
	ErrUnavailable    = errors.New("this run is live or unrecorded and the machine's Runtime is stopped")
)

// Source is a machine's outputs and the keys that may read them.
type Source interface {
	// RefreshKeys checks authoritative access before admitting a new session.
	RefreshKeys(context.Context) error
	Open(run uint64, output string, index int) (Snapshot, error)
	// Entries answers the entries after `after`, waiting at the tail; at or past the terminal
	// entry it answers the terminal entry again.
	Entries(ctx context.Context, run uint64, after uint64) ([]Entry, error)
	// Keys is the authorized key set now and a channel closed when it changes.
	Keys() ([]ed25519.PublicKey, <-chan struct{})
}

// Close releases the descriptors a snapshot's body holds.
func (s Snapshot) Close() error {
	if closer, ok := s.Body.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}
