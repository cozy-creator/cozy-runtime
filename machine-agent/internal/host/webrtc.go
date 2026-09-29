package host

import (
	"context"
	"fmt"

	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/host/webrtc"
)

// serveWebRTC serves browser media on the granted port. Without the grant nothing listens.
func (m *Machine) serveWebRTC(ctx context.Context) error {
	if m.grant.WebRTCPort == 0 {
		return nil
	}
	listener, err := listen(m.grant.ListenHost, m.grant.WebRTCPort)
	if err != nil {
		return fmt.Errorf("bind the WebRTC port: %w", err)
	}
	go func() {
		if err := webrtc.Serve(ctx, listener, webrtc.Config{Machine: m.grant.WorkerID, Leaf: m.id.Leaf, Source: m}); err != nil {
			fmt.Fprintln(m.log, "cozy machine: WebRTC stopped serving:", err)
		}
	}()
	return nil
}

// webrtcReceipt is the receipt's additive member naming the WebRTC port, when served.
func (m *Machine) webrtcReceipt() map[string]any {
	if m.grant.WebRTCPort == 0 {
		return nil
	}
	return map[string]any{"port": m.grant.WebRTCPort}
}
