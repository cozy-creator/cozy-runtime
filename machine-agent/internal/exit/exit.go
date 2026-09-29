// Package exit is the detailed domain/API refusal vocabulary. The CLI output
// boundary preserves these names while projecting process exits to 0/1/2.
package exit

import (
	"fmt"
	"strings"
)

type Code int

const (
	OK          Code = 0
	Internal    Code = 1
	Usage       Code = 2
	Validation  Code = 3
	NotFound    Code = 4
	Credential  Code = 5
	Structural  Code = 6
	Confirm     Code = 7
	OfflineMiss Code = 8
	Unavailable Code = 9
	Deadline    Code = 10
	Failed      Code = 11
	Canceled    Code = 12
	Conflict    Code = 13
	Capacity    Code = 14
)

type Row struct {
	Code    Code
	Name    string
	Meaning string
}

// Matrix is the shared exit matrix, verbatim.
var Matrix = []Row{
	{OK, "ok", "success, including idempotent no-ops"},
	{Internal, "internal", "unexpected fault — a bug, never a user condition"},
	{Usage, "usage", "bad invocation: unknown flag, malformed `key=value`, malformed target"},
	{Validation, "validation", "typed payload/schema/bounds refusal; verifier refusal at ingest"},
	{NotFound, "not_found", "unknown ref/function/package/attempt; hub 404 rendered verbatim"},
	{Credential, "credential", "private/gated source without a credential — names the credential to add"},
	{Structural, "structural", "structural incompatibility — never a fit shortfall (a degradable shortfall degrades and exits 0; a below-floor shortfall is 14)"},
	{Confirm, "confirm", "destructive op without `--yes`"},
	{OfflineMiss, "offline_miss", "`--offline` and the bytes are not in the CAS"},
	{Unavailable, "unavailable", "socket / local server / hub unreachable"},
	{Deadline, "deadline", "`--timeout` or request deadline exceeded"},
	{Failed, "failed", "attempt/job failure terminal (remedy verbatim)"},
	{Canceled, "canceled", "canceled terminal"},
	{Conflict, "conflict", "target exists / concurrent writer / failed replacement kept the working state"},
	{Capacity, "capacity", "no proven plan fits BELOW the physical floor — quantified shortfall (needed N, had M, short by N−M for X); fires only after the ladder's deepest authorized rung, never exit 6, never a silent 0"},
}

func (c Code) Name() string {
	for _, r := range Matrix {
		if r.Code == c {
			return r.Name
		}
	}
	return "internal"
}

func (c Code) Valid() bool {
	for _, r := range Matrix {
		if r.Code == c {
			return true
		}
	}
	return false
}

// JobTerminal maps a job/attempt terminal state onto the shared matrix:
// succeeded 0 · failed 11 · canceled 12 · deadline 10. The one home for that
// mapping; `cozy help job follow` renders it from here.
func JobTerminal(state string) Code {
	switch strings.ToLower(strings.TrimSpace(state)) {
	case "succeeded", "completed", "ok":
		return OK
	case "failed":
		return Failed
	case "canceled", "cancelled":
		return Canceled
	case "deadline", "timeout", "timed_out":
		return Deadline
	}
	return Internal
}

// Error is the one typed CLI error. Name may be more specific than the matrix
// name (for example not_implemented under usage), but Code is always a matrix code.
type Error struct {
	Code    Code     `json:"code"`
	Name    string   `json:"name,omitempty"`
	Message string   `json:"message"`
	Remedy  string   `json:"remedy,omitempty"`
	Next    []string `json:"next,omitempty"`
	// Cause is the stable code of the component that originated the failure (a pod's
	// `insufficient_storage`, a Runtime's `wheel_download_failed`) when that is not Creator.
	Cause string `json:"cause,omitempty"`
	// Details are the structured facts behind Message, for machine readers.
	Details map[string]any `json:"details,omitempty"`
}

func (e *Error) Error() string { return fmt.Sprintf("error(%s): %s", e.ErrName(), e.Message) }

func (e *Error) ErrName() string {
	if e.Name != "" {
		return e.Name
	}
	return e.Code.Name()
}

// New builds a typed error whose name is the matrix name for the code.
func New(code Code, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}

// Named builds a typed error with a specific name under a matrix code.
func Named(code Code, name, format string, args ...any) *Error {
	return &Error{Code: code, Name: name, Message: fmt.Sprintf(format, args...)}
}

func (e *Error) WithRemedy(format string, args ...any) *Error {
	e.Remedy = fmt.Sprintf(format, args...)
	return e
}

// WithCause records the originating component's stable code.
func (e *Error) WithCause(code string) *Error {
	e.Cause = code
	return e
}

// WithNext attaches suggestion lines; at most two survive rendering.
func (e *Error) WithNext(next ...string) *Error {
	e.Next = append(e.Next, next...)
	if len(e.Next) > 2 {
		e.Next = e.Next[:2]
	}
	return e
}

func Usagef(format string, args ...any) *Error       { return New(Usage, format, args...) }
func Unavailablef(format string, args ...any) *Error { return New(Unavailable, format, args...) }
func Internalf(format string, args ...any) *Error    { return New(Internal, format, args...) }

// As returns err as a typed *Error, wrapping an untyped one as internal.
func As(err error) *Error {
	if err == nil {
		return nil
	}
	if e, ok := err.(*Error); ok {
		return e
	}
	return Internalf("%s", err.Error())
}
