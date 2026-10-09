"""Real CPU executor and loopback protocol tests; no Comfy/GPU inference claim."""

from __future__ import annotations

import asyncio
import gzip
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
from aiohttp import web

from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import JobBudget, RunJob
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel

PROJECT = Path(__file__).parents[1] / "research/comfy-cpu-observer"
sys.path.insert(0, str(PROJECT))
from comfy_cpu_observer.runner import bounded_copy, observe, retain_outputs  # noqa: E402


class ProtocolServer:
    """Small real HTTP/websocket service exercising the client's wire contract."""

    def __init__(self, root: Path, *, lost_ack: bool = False, finish: bool = True):
        self.root, self.lost_ack, self.finish = root, lost_ack, finish
        self.posts = []
        self.history = {}
        self.sockets = {}
        self.ready = threading.Event()
        self.loop = asyncio.new_event_loop()

    async def queue(self, request):
        return web.json_response({"queue_running": [], "queue_pending": []})

    async def get_history(self, request):
        key = request.match_info["prompt_id"]
        return web.json_response({key: self.history[key]} if key in self.history else {})

    async def events(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sockets[request.query["clientId"]] = ws
        async for _ in ws:
            pass
        return ws

    async def submit(self, request):
        body = await request.json()
        self.posts.append(body)
        prompt_id = body["prompt_id"]
        if self.finish:
            (self.root / "receipt.bin").write_bytes(b"actual protocol output\x00\xff")
            self.history[prompt_id] = {
                "status": {
                    "completed": True,
                    "status_str": "success",
                    "messages": [
                        ["execution_start", {"timestamp": 1000}],
                        ["execution_success", {"timestamp": 1500}],
                    ],
                },
                "outputs": {
                    "save": {
                        "files": [
                            {"filename": "receipt.bin", "subfolder": "", "type": "output"},
                        ]
                    }
                },
            }
        ws = self.sockets.get(body["client_id"])
        if ws is not None:
            if self.lost_ack:
                await ws.close()
            else:
                await ws.send_json({"type": "execution_success", "data": {"prompt_id": prompt_id}})
        if self.lost_ack:
            request.transport.close()
        return web.json_response({"prompt_id": prompt_id, "node_errors": {}})

    def __enter__(self):
        async def start():
            app = web.Application()
            app.router.add_get("/queue", self.queue)
            app.router.add_get("/history/{prompt_id}", self.get_history)
            app.router.add_get("/ws", self.events)
            app.router.add_post("/prompt", self.submit)
            self.runner = web.AppRunner(app, access_log=None)
            await self.runner.setup()
            site = web.TCPSite(self.runner, "127.0.0.1", 0)
            await site.start()
            self.port = site._server.sockets[0].getsockname()[1]
            self.ready.set()

        def serve():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(start())
            self.loop.run_forever()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        assert self.ready.wait(10)
        return self

    def __exit__(self, *args):
        async def stop():
            for ws in self.sockets.values():
                await ws.close()
            await self.runner.cleanup()

        asyncio.run_coroutine_threadsafe(stop(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)
        self.loop.close()


def options(tmp_path, server, timeout=5):
    return dict(
        graph_json=json.dumps({"test": {"class_type": "ProtocolContract", "inputs": {}}}),
        port=server.port,
        timeout_s=timeout,
        trace=False,
        output_root=str(tmp_path),
        trace_root=str(tmp_path / "traces"),
        directory=tmp_path / "scratch",
        deadline=time.monotonic() + 30,
        check_cancelled=lambda: None,
    )


def test_lost_submission_and_event_socket_recover_without_duplicate_post(tmp_path):
    with ProtocolServer(tmp_path, lost_ack=True) as server:
        first = observe(**options(tmp_path, server))
        second = observe(**options(tmp_path, server))
        assert first["status"] == second["status"] == "success"
        assert first["prompt_id"] == second["prompt_id"]
        assert len(server.posts) == 1
        assert first["server_execution_seconds"] == 0.5
        assert (
            tmp_path / "scratch/artifacts/outputs/receipt.bin"
        ).read_bytes() == b"actual protocol output\x00\xff"
        changed = options(tmp_path, server)
        changed["graph_json"] = '{"different": {}}'
        with pytest.raises(ValueError, match="resumed observer cannot change"):
            observe(**changed)


def test_timeout_retains_prompt_identity_and_does_not_cancel_or_resubmit(tmp_path):
    with ProtocolServer(tmp_path, finish=False) as server:
        start = time.monotonic()
        result = observe(**options(tmp_path, server, timeout=1))
        assert result["status"] == "timed_out"
        assert result["server_execution_seconds"] is None
        assert not result["cancellation_sent"] and not result["submission_repeated"]
        assert len(server.posts) == 1 and time.monotonic() - start < 3
        assert (tmp_path / "scratch/artifacts/accepted.json").is_file()


def test_retained_outputs_cannot_escape_configured_directory(tmp_path):
    history = {"outputs": [{"filename": "../escape", "type": "output"}]}
    with pytest.raises(ValueError, match="escapes"):
        retain_outputs(history, tmp_path, tmp_path / "artifacts")


def test_bounded_copy_credits_previous_attempt_bytes_and_refuses_overflow(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"abc")
    target.write_bytes(b"abc")
    assert bounded_copy(source, target, 0) == 0
    source.write_bytes(b"abcde")
    assert bounded_copy(source, target, 2) == 2
    source.write_bytes(b"too many bytes")
    with pytest.raises(ValueError, match=r"byte|bound"):
        bounded_copy(source, target, 0)


def test_trace_export_timeout_resumes_to_retained_files_without_resubmission(tmp_path):
    with ProtocolServer(tmp_path) as server:
        request = options(tmp_path, server, timeout=1)
        request["trace"] = True
        pending = observe(**request)
        assert pending["status"] == "timed_out"
        assert "trace export" in pending["detail"]
        trace = tmp_path / "traces" / pending["prompt_id"]
        trace.mkdir(parents=True)
        # This is protocol/artifact retention, not a GPU trace qualification.
        data = gzip.compress(b'{"traceEvents":[]}')
        (trace / "trace-000.json.gz").write_bytes(data)
        (trace / "summary.json").write_text(
            json.dumps(
                {
                    "steps_observed": 8,
                    "expected_steps": 8,
                    "segments": [{"file": "trace-000.json.gz"}],
                }
            )
        )
        finished = observe(**request)
        assert finished["status"] == "success" and len(server.posts) == 1
        assert (tmp_path / "scratch/artifacts/trace/trace-000.json.gz").read_bytes() == data


def test_cpu_interface_and_real_executor_need_no_torch_or_cuda(tmp_path):
    assert "torch" not in sys.modules
    found = discover(PROJECT)
    static = static_interface.build(PROJECT)
    imported = package_interface.build(found)
    assert package_interface.canonical_bytes(static) == package_interface.canonical_bytes(imported)
    assert static["jobs"][0]["accelerator"] is False
    with ProtocolServer(tmp_path) as server:
        # Use the actual execution kernel and its real channel/spool, not a fake Context.
        left, right = socket.socketpair()
        with left, right:
            peer = Channel(right)
            frames = []

            def receive():
                while (frame := peer.recv()) is not None:
                    frames.append(frame)

            reader = threading.Thread(target=receive, daemon=True)
            reader.start()
            executor = Executor(Channel(left), tmp_path)
            executor.discovered = found
            executor.surface_names = {surface.name: surface for surface in found.surfaces}
            spool = tmp_path / "spool"
            result = executor.run_job(
                RunJob(
                    job="run",
                    request_id="comfy-cpu-contract",
                    payload={
                        "graph_json": '{"test":{"class_type":"ProtocolContract","inputs":{}}}',
                        "port": server.port,
                        "timeout_s": 10,
                        "output_root": str(tmp_path),
                    },
                    deadline_s=30,
                    spool=str(spool),
                    scratch=str(tmp_path / "job-scratch"),
                    application="",
                    package_interface="",
                    budget=JobBudget(gpu_count=0),
                    max_output_bytes=64 << 30,
                )
            )
            assert result["outcome"]["terminal"] == "succeeded", result
            document = json.loads((spool / RESULT_DOCUMENT).read_bytes())
            assert document["status"] == "success"
            assert document["cpu_device"] == "cpu" and not document["torch_imported"]
            assert "torch" not in sys.modules and executor.torch is None
            assert any(
                p.read_bytes() == b"actual protocol output\x00\xff"
                for p in spool.rglob("receipt.bin")
            )
            left.shutdown(socket.SHUT_WR)
            reader.join(5)
            assert not reader.is_alive()
