package host

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
)

type dependencyWheel struct {
	Name    string   `json:"name"`
	Version string   `json:"version"`
	File    string   `json:"file"`
	SHA256  string   `json:"sha256"`
	URL     string   `json:"url"`
	Path    string   `json:"path"`
	Paths   []string `json:"paths"`
}

// Stage only missing distributions. The resolver treats every installed version
// as fixed; uv later receives concrete private wheels with resolution disabled.
func (m *Machine) stageDependencies(ctx context.Context, pair []string) ([]dependencyWheel, error) {
	var selected []dependencyWheel
	if err := m.startupPython(ctx, &selected, append([]string{"dependencies"}, pair...)...); err != nil {
		return nil, fmt.Errorf("additive dependency preflight: %w", err)
	}
	downloads := &Machine{layout: Layout{State: m.startupPath("downloads")}}
	paths := map[string]bool{}
	for i := range selected {
		wheel := &selected[i]
		if wheel.Name == "" || filepath.Base(wheel.File) != wheel.File || !strings.HasSuffix(wheel.File, ".whl") || !sha256Hex.MatchString(wheel.SHA256) || !strings.HasPrefix(wheel.URL, "https://files.pythonhosted.org/") {
			return nil, errors.New("dependency wheel has invalid index identity")
		}
		response, err := httpGet(ctx, wheel.URL)
		if err != nil {
			return nil, err
		}
		hash, _, err := downloads.stageWheel(wheel.File, response.Body)
		response.Body.Close()
		if err != nil {
			return nil, err
		}
		if hash != wheel.SHA256 {
			return nil, errors.New("dependency wheel differs from index digest")
		}
		wheel.Path, err = m.startupCopyWheel(downloads.stagePath(hash, wheel.File), filepath.Join("dependencies", hash), hash)
		if err != nil {
			return nil, err
		}
		var inventory []dependencyWheel
		if err = m.startupPython(ctx, &inventory, "dependency-files", wheel.Path); err != nil {
			return nil, err
		}
		if len(inventory) != 1 || inventory[0].Name != wheel.Name || inventory[0].Version != wheel.Version {
			return nil, errors.New("dependency wheel metadata differs from index")
		}
		wheel.Paths = inventory[0].Paths
		for _, path := range wheel.Paths {
			if paths[path] {
				return nil, fmt.Errorf("dependency wheels overlap at %s", path)
			}
			paths[path] = true
		}
	}
	return selected, nil
}

func (m *Machine) removeDependencies(ctx context.Context, dependencies []dependencyWheel) error {
	if len(dependencies) == 0 {
		return nil
	}
	raw, _ := json.Marshal(dependencies)
	var names []string
	if err := m.startupPython(ctx, &names, "removable-dependencies", string(raw)); err != nil {
		return err
	}
	if len(names) == 0 {
		return nil
	}
	args := append([]string{"pip", "uninstall", "--no-config", "--python", filepath.Join(m.layout.Root, "opt/cozy/python/bin/python"), "--break-system-packages"}, names...)
	cmd := exec.CommandContext(ctx, filepath.Join(m.layout.Root, "usr/local/bin/uv"), args...)
	cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + filepath.Join(m.layout.Root, "tmp")}
	if out, err := progressChildOutput(cmd, true); err != nil {
		return fmt.Errorf("remove transaction-owned dependencies: %v: %s", err, out)
	}
	return syncInstallation(m.layout.Root)
}
