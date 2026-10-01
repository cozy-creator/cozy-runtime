package host

import (
	"bytes"
	"regexp"
	"testing"
)

// Lines are stamped where they start, however the writes that carry them are split.
func TestStampedLogStampsEachLineOnce(t *testing.T) {
	var out bytes.Buffer
	log := Stamped(&out)
	for _, chunk := range []string{"[worker] boot: ", "started\n[worker] warm: a\nb", "\n", "\n"} {
		if n, err := log.Write([]byte(chunk)); err != nil || n != len(chunk) {
			t.Fatalf("write %q: %d, %v", chunk, n, err)
		}
	}
	stamp := `\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z `
	want := regexp.MustCompile(`^` + stamp + `\[worker\] boot: started\n` + stamp + `\[worker\] warm: a\n` +
		stamp + `b\n` + stamp + `\n$`)
	if !want.Match(out.Bytes()) {
		t.Fatalf("stamped log:\n%s", out.String())
	}
}
