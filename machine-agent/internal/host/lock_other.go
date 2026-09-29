//go:build !linux

package host

import "errors"

func lockMachine(string) (*machineLease, error) {
	return nil, errors.New("cozy-machine requires Linux process containment")
}

func lockStartupWorker(string) (func(), error) {
	return nil, errors.New("startup updates require Linux worker ownership")
}

func syncInstallation(string) error { return errors.New("installation durability requires Linux") }
