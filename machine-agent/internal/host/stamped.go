package host

import (
	"bytes"
	"io"
	"sync"
	"time"
)

// Stamped prefixes each line written to out with its UTC wall time, to the millisecond:
// the Runtime's and executors' output reaches the machine log through it, so a boot, an
// admission or an executor start reads as a timeline from the log alone.
func Stamped(out io.Writer) io.Writer { return &stamped{out: out, fresh: true} }

type stamped struct {
	mu    sync.Mutex
	out   io.Writer
	fresh bool // the next byte starts a line
}

func (s *stamped) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var b []byte
	for rest := p; len(rest) > 0; {
		if s.fresh {
			b = time.Now().UTC().AppendFormat(b, "2006-01-02T15:04:05.000Z ")
		}
		line, after, ended := bytes.Cut(rest, []byte{'\n'})
		b = append(b, line...)
		if ended {
			b = append(b, '\n')
		}
		s.fresh, rest = ended, after
	}
	if _, err := s.out.Write(b); err != nil {
		return 0, err
	}
	return len(p), nil
}
