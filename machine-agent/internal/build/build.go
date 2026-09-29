// Package build is this binary's identity, one stamp for every role: `cozy --version`, the
// Hub's client revision, and the machine role's DescribeMachine and readiness receipt.
package build

import "runtime/debug"

// Version and Commit are stamped with -ldflags -X at release (scripts/release.sh).
var (
	Version = "0.0.0-dev"
	Commit  = ""
)

// Revision is the stamped commit, else Go's VCS stamp, to 12 characters, and whether the
// tree it was built from was modified.
func Revision() (string, bool) {
	revision, dirty := Commit, false
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				if revision == "" {
					revision = setting.Value
				}
			case "vcs.modified":
				dirty = setting.Value == "true"
			}
		}
	}
	if revision == "" {
		revision = "unknown"
	}
	return revision[:min(len(revision), 12)], dirty
}

// Capabilities name implemented agent contracts, independently of release numbers.
func Capabilities() []string {
	return []string{"hub-access/1", "runtime-update/1", "runtime-update/2", "machine-bootstrap/1"}
}
