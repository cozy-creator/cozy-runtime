package host

import (
	"encoding/json"
	"fmt"
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
	out, err := progressChildOutput(exec.Command(path, "capabilities", "--json"), false)
	var reply struct {
		Capabilities []string `json:"capabilities"`
	}
	if err != nil {
		return "", fmt.Errorf("Runtime capability probe failed: %w", err)
	}
	if err := json.Unmarshal(out, &reply); err != nil {
		return "", fmt.Errorf("Runtime capability response is unreadable: %w", err)
	}
	if !slices.Contains(reply.Capabilities, "machine-supervisor/1") {
		return "", &launchRefusal{"runtime_supervisor_unsupported", "the installed Runtime does not support machine supervision; update its Runtime wheel"}
	}
	return string(out), nil
}
