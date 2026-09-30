package host

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/cozy-creator/cozy-runtime/machine-agent/internal/capability"
	"io"
	"strings"
	"time"

	pb "github.com/cozy-creator/cozy-runtime/machine-agent/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

// Every RPC reaches serveRPC. A method the daemon answers itself has a handler; any other
// the daemon forwards to the Runtime after checking its Claim, rewriting the Claim to the
// daemon's own; a method with no route is not served.

type route struct {
	method  protoreflect.MethodDescriptor
	target  string // the Runtime method, "/cozy.worker.v1.Service/Method"
	handler func(*Machine, grpc.ServerStream) error
	work    bool // new work: the wire floor, the Runtime's range and an idle admission apply
}

const (
	podHost     = "cozy.worker.v1.PodHost"
	control     = "cozy.worker.v1.WorkerControl"
	preparation = "cozy.worker.v1.RuntimePreparation"
)

var routes = func() map[string]*route {
	file := pb.File_cozy_worker_v1_worker_proto
	service := func(name string) protoreflect.ServiceDescriptor {
		return file.Services().ByName(protoreflect.FullName(name).Name())
	}
	out := map[string]*route{}
	add := func(from, method, to, toMethod string) {
		md := service(from).Methods().ByName(protoreflect.Name(method))
		target := service(to).Methods().ByName(protoreflect.Name(toMethod))
		if md == nil || target == nil || md.Input() != target.Input() || md.Output() != target.Output() {
			panic("host route " + from + "/" + method + " does not match " + to + "/" + toMethod)
		}
		out["/"+from+"/"+method] = &route{method: md, target: "/" + to + "/" + toMethod}
	}
	execution := []string{"GetMachineExecutionWorkspace", "SubmitMachineExecution", "GetMachineExecution",
		"ListMachineExecutionEvents", "ControlMachineExecution", "CollectMachineExecution",
		"AcknowledgeMachineExecutionCollection", "ReadMachineExecutionTriage", "ListMachineExecutions", "ListPackages", "CloseMachineSubmission"}
	for _, from := range []string{podHost, control} {
		for _, method := range execution {
			add(from, method, control, method)
		}
	}
	for _, method := range []string{"RetainDerivedResult", "ReleaseDerivedRetention", "ReleaseDerivedResult",
		"RetainByteTree", "ReleaseByteTree", "ReadByteTreeObject", "NativeArtifactTransfer", "ForgetPackage"} {
		add(podHost, method, preparation, "Workspace"+method)
	}
	for _, method := range []string{"ImportInputTree", "RecordOperationResult", "LookupOperation", "PruneOperationCache"} {
		add(podHost, method, preparation, method)
	}
	out["/"+podHost+"/SubmitMachineExecution"].work = true
	out["/"+control+"/SubmitMachineExecution"].work = true
	own := map[string]func(*Machine, grpc.ServerStream) error{
		"KeepRentalAlive": (*Machine).keepRentalAlive, "ProtocolInfo": (*Machine).protocolInfo,
		"DescribeMachine": (*Machine).describeMachine, "ListModels": (*Machine).listModels,
		"PreparePackageSet": (*Machine).preparePackageSet, "PrepareLocalPackage": (*Machine).prepareLocalPackage,
		"PreparePrivatePlacement": (*Machine).preparePrivatePlacement, "LocalPackageUpload": (*Machine).localPackageUpload,
	}
	for method, handler := range own {
		out["/"+podHost+"/"+method] = &route{method: service(podHost).Methods().ByName(protoreflect.Name(method)), handler: handler}
	}
	for _, method := range []string{"DescribeMachine", "ListModels"} {
		out["/"+control+"/"+method] = &route{method: service(control).Methods().ByName(protoreflect.Name(method)), handler: own[method]}
	}
	out["/"+control+"/Control"] = &route{method: service(control).Methods().ByName("Control"), handler: (*Machine).answerControl}
	return out
}()

func (m *Machine) serveRPC(_ any, stream grpc.ServerStream) (err error) {
	if err := m.claims.checkAuthority(); err != nil {
		return err
	}
	call, cancel := context.WithCancel(stream.Context())
	defer cancel()
	scoped := &authorityStream{ServerStream: stream, ctx: call, cancel: cancel, claims: m.claims}
	stream = scoped

	name, _ := grpc.MethodFromServerStream(stream)
	r := routes[name]
	switch {
	case r == nil:
		return status.Errorf(codes.Unimplemented, "this machine does not serve %s", name)
	case r.handler != nil:
		err = r.handler(m, stream)
	default:
		err = m.forward(stream, r)
	}
	if err != nil && status.Code(err) != codes.Canceled {
		fmt.Fprintf(m.log, "cozy machine: %s: %v\n", name[strings.LastIndexByte(name, '/')+1:], err)
	}
	return err
}

func newMessage(d protoreflect.MessageDescriptor) proto.Message {
	t, err := protoregistry.GlobalTypes.FindMessageByName(d.FullName())
	if err != nil {
		panic(err)
	}
	return t.New().Interface()
}

// forward carries one call to the Runtime. Its trailers and unknown fields pass through;
// a Runtime that predates a method refuses only that call, naming the side that is behind.
func (m *Machine) forward(stream grpc.ServerStream, r *route) (err error) {
	ctx := stream.Context()
	first := newMessage(r.method.Input())
	if err := stream.RecvMsg(first); err != nil {
		return err
	}
	client, err := m.admitCall(stream, first)
	if err != nil {
		return err
	}
	if r.work || isResume(first) {
		if err := admitWork(client, "new work"); err != nil {
			return err
		}
		if err := m.requireRuntimeRange(ctx); err != nil {
			return err
		}
		a, refused := m.idle.admit(ctx, time.Now())
		if refused != nil {
			return status.Error(codes.FailedPrecondition, "this machine's idle release is due or committed")
		}
		defer func() { _ = a.finish(err == nil, time.Now()) }()
	}
	phase := m.phase()
	if (phase == "starting" || phase == "booting") && publicReadMethod(string(r.method.Name())) {
		// An authenticated caller may wait for the coordinator it is using.
		// The journal's read-only workspace does not advertise live mutation
		// capabilities, so it cannot qualify a new submission during boot.
		if _, err := m.runtime(ctx); err != nil {
			return err
		}
	} else if phase != "ready" && publicReadMethod(string(r.method.Name())) {
		if answer, handled, err := m.publicFallback(ctx, first); handled {
			if err != nil {
				return err
			}
			_ = stream.SendHeader(metadata.Pairs("cozy-snapshot-source", "journal", "cozy-runtime-state", m.phase(), "cozy-runtime-available", "false"))
			return stream.SendMsg(answer)
		}
	}
	conn, err := m.runtime(ctx)
	if err != nil {
		return err
	}
	if submit, ok := first.(*pb.MachineExecutionSubmit); ok {
		if err := m.requireModelOverrides(submit); err != nil {
			return err
		}
	}
	if submit, ok := first.(*pb.MachineExecutionSubmit); ok && submit.Hub != "" {
		peer, err := pb.NewRuntimePreparationClient(conn).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
		if err != nil {
			return err
		}
		if peer.GetWireMinor() < 69 {
			return status.Error(codes.FailedPrecondition, pb.CapabilityUnavailableCode+": this Runtime cannot preserve captured execution Hub scope; update the Runtime")
		}
	}
	m.claims.forRuntime(first, client.GetWireMinor())
	desc := &grpc.StreamDesc{ServerStreams: r.method.IsStreamingServer(), ClientStreams: r.method.IsStreamingClient()}
	out, err := conn.NewStream(ctx, desc, r.target)
	if err != nil {
		return err
	}
	if err := out.SendMsg(first); err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	if desc.ClientStreams {
		for {
			next := newMessage(r.method.Input())
			if err := stream.RecvMsg(next); errors.Is(err, io.EOF) {
				break
			} else if err != nil {
				return err
			}
			m.claims.forRuntime(next, client.GetWireMinor())
			if err := out.SendMsg(next); err != nil {
				break
			}
		}
	}
	if err := out.CloseSend(); err != nil {
		return err
	}
	if header, err := out.Header(); err == nil && len(header) > 0 {
		_ = stream.SendHeader(header)
	}
	for {
		reply := newMessage(r.method.Output())
		err := out.RecvMsg(reply)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			stream.SetTrailer(out.Trailer())
			return behind(err, r)
		}
		m.claims.answer(reply, client)
		if err := stream.SendMsg(reply); err != nil {
			return err
		}
	}
	stream.SetTrailer(out.Trailer())
	return nil
}

func (m *Machine) requireModelOverrides(request *pb.MachineExecutionSubmit) error {
	requested := false
	for _, choice := range request.GetReleaseRoot().GetModels() {
		requested = requested || len(choice.Adapters) > 0 || strings.Contains(choice.Parameter, ".models.")
	}
	if len(request.CaptureCanonicalBytes) > 0 {
		var capture struct {
			ModelChoices []json.RawMessage `json:"model_choices"`
		}
		if err := json.Unmarshal(request.CaptureCanonicalBytes, &capture); err != nil {
			return status.Error(codes.InvalidArgument, "invalid captured execution document")
		}
		requested = requested || len(capture.ModelChoices) > 0
	}
	if requested && !m.supportsRuntime("runtime-model-overrides/1") {
		return status.Error(codes.FailedPrecondition, pb.CapabilityUnavailableCode+": this Runtime cannot preserve model overrides; update the Runtime")
	}
	return nil
}

func publicReadMethod(name string) bool {
	switch name {
	case "GetMachineExecutionWorkspace", "GetMachineExecution", "ListMachineExecutions", "ListMachineExecutionEvents":
		return true
	}
	return false
}

// admitCall verifies the client Claim a message carries.
func (m *Machine) admitCall(stream grpc.ServerStream, message proto.Message) (*pb.Claim, error) {
	var claim *pb.Claim
	walk(message.ProtoReflect(), func(msg protoMessage) {
		if c, ok := msg.Interface().(*pb.Claim); ok && claim == nil {
			claim = c
		}
	})
	key, err := m.claims.verify(claim)
	if err != nil {
		return nil, err
	}
	if scoped, ok := stream.(*authorityStream); ok {
		scoped.bind(key)
	}
	return proto.Clone(claim).(*pb.Claim), nil
}

func isResume(message proto.Message) bool {
	control, ok := message.(*pb.MachineExecutionControl)
	return ok && control.Action == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_RESUME
}

// behind turns a Runtime that lacks a method into a refusal of this call alone.
func behind(err error, r *route) error {
	if status.Code(err) != codes.Unimplemented {
		return err
	}
	method := r.target[strings.LastIndexByte(r.target, '/')+1:]
	return refusal(codes.FailedPrecondition, pb.CapabilityUnavailableCode,
		fmt.Sprintf("this machine's Runtime predates %s; update the machine's Runtime", method))
}

func refusal(code codes.Code, safeCode, message string) error {
	return status.Error(code, safeCode+": "+message)
}

func unavailable(safeCode, message string) error {
	return refusal(codes.Unavailable, safeCode, message)
}

// requireRuntimeRange refuses new work when the Runtime shares no protocol minor with this
// machine: the refusal names the Runtime as the side to update.
func (m *Machine) requireRuntimeRange(ctx context.Context) error {
	conn, err := m.runtime(ctx)
	if err != nil {
		return err
	}
	peer, err := pb.NewRuntimePreparationClient(conn).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	if err != nil {
		return err
	}
	if _, ok := sharedRange(peer); !ok {
		return refusal(codes.FailedPrecondition, pb.CapabilityUnavailableCode, fmt.Sprintf(
			"this machine's Runtime speaks worker protocol %d-%d and the machine %d-%d; update the machine's Runtime",
			peer.GetMinimumWireMinor(), peer.GetWireMinor(), pb.MinCompatibleWireMinor, pb.WireMinor))
	}
	return nil
}

func sharedRange(peer *pb.ProtocolInfoResult) (*pb.ProtocolInfoResult, bool) {
	if peer == nil || peer.MinimumWireMinor == 0 || peer.MinimumWireMinor > peer.WireMinor {
		return nil, false
	}
	shared := &pb.ProtocolInfoResult{WireMinor: min(peer.WireMinor, pb.WireMinor), MinimumWireMinor: max(peer.MinimumWireMinor, pb.MinCompatibleWireMinor)}
	return shared, shared.MinimumWireMinor <= shared.WireMinor
}

// recvClaimed reads one message a daemon-served method takes and verifies its Claim.
func recvClaimed[T proto.Message](stream grpc.ServerStream, m *Machine, message T) (*pb.Claim, error) {
	if err := stream.RecvMsg(message); err != nil {
		return nil, err
	}
	return m.admitCall(stream, message)
}

func (m *Machine) keepRentalAlive(stream grpc.ServerStream) error {
	call := &pb.KeepRentalAliveRequest{}
	if _, err := recvClaimed(stream, m, call); err != nil {
		return err
	}
	if call.RequestId == "" || len(call.RequestId) > pb.MaxRentalKeepaliveRequestIDBytes {
		return status.Error(codes.InvalidArgument, "a keepalive names one bounded request_id")
	}
	acknowledged, deadline, err := m.idle.keepalive(call.RequestId, time.Now())
	if err != nil {
		return status.Error(codes.FailedPrecondition, "this machine's idle release is already due or committed")
	}
	return stream.SendMsg(&pb.KeepRentalAliveResult{RequestId: call.RequestId, WorkerId: m.grant.WorkerID,
		WorkerBootId: m.id.BootID, AcknowledgedAtUnixMs: acknowledged.UnixMilli(), IdleDeadlineUnixMs: deadline.UnixMilli()})
}

// protocolInfo is the intersection of the machine's range and its Runtime's, or the
// machine's own when they share none or the Runtime is down, so maintenance still works.
func (m *Machine) protocolInfo(stream grpc.ServerStream) error {
	if err := stream.RecvMsg(&pb.ProtocolInfoRequest{}); err != nil {
		return err
	}
	own := &pb.ProtocolInfoResult{WireMinor: pb.WireMinor, MinimumWireMinor: pb.MinCompatibleWireMinor}
	ctx, cancel := context.WithTimeout(stream.Context(), 10*time.Second)
	defer cancel()
	if conn, err := m.liveRuntime(ctx); err == nil {
		if peer, err := pb.NewRuntimePreparationClient(conn).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{}); err == nil {
			if shared, ok := sharedRange(peer); ok {
				own = shared
			}
		}
	}
	return stream.SendMsg(own)
}

// answerControl answers a client's Control Claim itself: the daemon holds the Runtime's
// only session, so a client's stream records nothing and fences nothing.
func (m *Machine) answerControl(stream grpc.ServerStream) error {
	first := &pb.RecordOwnerFrame{}
	if err := stream.RecvMsg(first); err != nil {
		return err
	}
	claim := first.GetClaim()
	ack := &pb.ClaimAck{RecordOwnerEpoch: claim.GetRecordOwnerEpoch(), WorkerBootId: m.id.BootID, WorkerId: m.grant.WorkerID,
		WireMinor: pb.WireMinor}
	key, err := m.claims.verify(claim)
	if err != nil {
		ack.Rejection = pb.ClaimRejection_CLAIM_REJECTION_UNAUTHENTICATED
		return stream.SendMsg(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: ack}})
	}
	if scoped, ok := stream.(*authorityStream); ok {
		scoped.bind(key)
	}
	m.mu.Lock()
	runtimeAck := m.runtimeAck
	m.control++
	ack.Accepted, ack.ControlStreamEpoch = true, m.control
	m.mu.Unlock()
	if runtimeAck != nil {
		ack.WireMinor = min(pb.WireMinor, runtimeAck.WireMinor)
		ack.WorkerInstanceId, ack.WorkerReleaseId = runtimeAck.WorkerInstanceId, runtimeAck.WorkerReleaseId
		ack.ControlRuntimeDigest, ack.GitCommit, ack.Resources = runtimeAck.ControlRuntimeDigest, runtimeAck.GitCommit, runtimeAck.Resources
	}
	if err := stream.SendMsg(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: ack}}); err != nil {
		return err
	}
	for { // the stream stays open until the client closes it; nothing it sends is a command
		if err := stream.RecvMsg(&pb.RecordOwnerFrame{}); err != nil {
			return nil
		}
	}
}

// authorityStream propagates revocation to the forwarded Runtime observer while
// leaving its already-accepted durable work untouched.
type authorityStream struct {
	grpc.ServerStream
	ctx    context.Context
	cancel context.CancelFunc
	claims *claims
}

func (s *authorityStream) Context() context.Context { return s.ctx }
func (s *authorityStream) bind(key []byte) {
	call, stop := s.claims.watchAuthority(s.ctx, capability.KeyID(key))
	go func() { defer stop(); <-call.Done(); s.cancel() }()
}
func (s *authorityStream) RecvMsg(message any) error {
	if err := s.ctx.Err(); err != nil {
		return err
	}
	result := make(chan error, 1)
	go func() { result <- s.ServerStream.RecvMsg(message) }()
	select {
	case <-s.ctx.Done():
		return s.ctx.Err()
	case err := <-result:
		if s.ctx.Err() != nil {
			return s.ctx.Err()
		}
		return err
	}
}
func (s *authorityStream) SendMsg(message any) error {
	if err := s.ctx.Err(); err != nil {
		return err
	}
	result := make(chan error, 1)
	go func() { result <- s.ServerStream.SendMsg(message) }()
	select {
	case <-s.ctx.Done():
		return s.ctx.Err()
	case err := <-result:
		return err
	}
}
