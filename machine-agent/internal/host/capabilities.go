package host

import (
	"encoding/json"
	"os/exec"
	"slices"
)

type launchRefusal struct {
	code, detail string
}

func (r *launchRefusal) Error() string { return r.code + ": " + r.detail }

// Probe without supervisor environment or fd 3. An older Runtime must never be
// launched with an unknown mode and accidentally enter its durable owner protocol.
func probeRuntimeCapabilities(path string) (string, error) {
	out, err := childOutput(exec.Command(path, "capabilities", "--json"), false)
	var reply struct {
		Capabilities []string `json:"capabilities"`
	}
	if err != nil || json.Unmarshal(out, &reply) != nil || !slices.Contains(reply.Capabilities, "machine-supervisor/1") {
		return "", &launchRefusal{"runtime_supervisor_unsupported", "the installed Runtime does not support machine supervision; update its Runtime wheel"}
	}
	return string(out), nil
}
