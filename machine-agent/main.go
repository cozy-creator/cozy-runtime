// cozy-machine is the machine server, independent of any client controller.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/build"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host"
	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
)

func main() {
	if host.RequestPlaneProcess(os.Args[0]) {
		os.Exit(host.RunRequestPlane(os.Stderr))
	}
	if host.GuardianProcess(os.Args[0]) {
		os.Exit(host.RunGuardian(os.Args))
	}
	if len(os.Args) >= 2 && (os.Args[1] == "version" || os.Args[1] == "--version") {
		revision, dirty := build.Revision()
		answer := struct {
			Name             string   `json:"name"`
			Version          string   `json:"version"`
			Revision         string   `json:"revision"`
			Dirty            bool     `json:"dirty"`
			WireMinor        uint32   `json:"wire_minor"`
			MinimumWireMinor uint32   `json:"minimum_wire_minor"`
			Capabilities     []string `json:"capabilities"`
		}{"cozy-machine", build.Version, revision, dirty, pb.WireMinor, pb.MinCompatibleWireMinor, build.Capabilities()}
		if err := json.NewEncoder(os.Stdout).Encode(answer); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if len(os.Args) > 2 || len(os.Args) == 2 && os.Args[1] != "run" {
		fmt.Fprintln(os.Stderr, "usage: cozy-machine [run | version --json]")
		os.Exit(2)
	}
	os.Exit(host.Main(host.Stamped(os.Stderr)))
}
