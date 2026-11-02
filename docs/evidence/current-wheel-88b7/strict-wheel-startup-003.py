"""Actual command decoding and framed pre-GPU refusal; no simulated CUDA success."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--tensorfs-python', type=Path, required=True)
parser.add_argument('--torch-site', type=Path, required=True)
parser.add_argument('--sender-frames', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
runtime = Path('/home/fidika/cozy/.worktrees/cozy-runtime/xs138-v2-current-core/.task/runtime-wheel-003')
args.out.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(runtime), str(args.tensorfs_python), str(args.torch_site)]
torch_spec = importlib.util.find_spec('torch')
assert torch_spec is not None, 'the successful preflight requires a real Torch package'

import msgspec  # noqa: E402 -- select the exact source/native cohort before imports

from cozy_runtime.internal import cooperation, executor_commands  # noqa: E402


def intent(index: int) -> dict:
    return {'uuid': f'GPU-13800000-0000-0000-0000-{index:012d}',
            'directory': '/dev/shm/varena-rank-metadata-only', 'shared': False,
            'priority': index}


# Every real decoded rank selects its own immutable intent. These UUIDs are metadata,
# never hardware identity evidence; no native device constructor is called here.
rank_selections = []
sender = json.loads(args.sender_frames.read_text())
assert sender['passed'] and [f['sequence_parallel_degree'] for f in sender['frames']] == [1, 2, 4]
for frame in sender['frames']:
    degree = frame['sequence_parallel_degree']
    decoded = executor_commands.decode(frame)
    assert isinstance(decoded, executor_commands.Start)
    assert executor_commands.decode(executor_commands.encode(decoded)) == decoded
    for rank in range(degree):
        selected = cooperation.Config.for_rank(decoded.cooperation, rank, degree)
        assert selected is not None and selected.uuid == intent(rank)['uuid']
        assert selected.priority == rank
        rank_selections.append({'degree': degree, 'rank': rank, 'uuid': selected.uuid})
for old_form in (intent(0), None):
    try:
        executor_commands.decode({'cmd': 'start', 'cooperation': old_form})
    except msgspec.ValidationError:
        pass
    else:
        raise AssertionError('old single-object/null cooperation form was accepted')

child_script = args.out / 'child.py'
child_script.write_text('''import hashlib, json, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[2], sys.argv[6]]
from cozy_runtime.internal import executor, cooperation
from tensorfs import _ext
torch_on_import = "torch" in sys.modules
code = executor.main(["--socket", sys.argv[3], "--root", sys.argv[4]])
maps = Path("/proc/self/maps").read_text()
state = {"exit_code": code, "torch_imported": "torch" in sys.modules,
 "torch_on_import": torch_on_import,
 "cuda_initialized": bool("torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()),
 "driver_loaded": any(x in maps for x in ("libcuda.so", "libnvidia-ml.so")),
 "capable": cooperation.capable(), "executor": executor.__file__,
 "executor_sha256": hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest(),
 "extension_sha256": hashlib.sha256(Path(_ext.__file__).read_bytes()).hexdigest()}
Path(sys.argv[5]).write_text(json.dumps(state, indent=2)+"\\n")
raise SystemExit(code)
''')
header = struct.Struct('!I')


def receive(connection: socket.socket, count: int) -> bytes:
    found = b''
    while len(found) < count:
        part = connection.recv(count - len(found))
        assert part, 'executor closed the protocol unexpectedly'
        found += part
    return found


def exchange(connection: socket.socket, frame: dict) -> dict:
    body = json.dumps(frame).encode()
    connection.sendall(header.pack(len(body)) + body)
    size, = header.unpack(receive(connection, header.size))
    assert size < 4 * 1024 * 1024
    return json.loads(receive(connection, size))


cases = [
    ('missing', 2, [], {}, 'cooperation_required'),
    ('short', 2, [intent(0)], {}, 'cooperation_invalid'),
    ('duplicate', 2, [intent(0), intent(0)], {}, 'cooperation_invalid'),
    ('relative_follower', 2, [intent(0), {**intent(1), 'directory': 'relative'}], {},
     'cooperation_invalid'),
    ('single_preflight', 1, [intent(0)], {}, 'package_prepare_path_invalid'),
    ('two_rank_preflight', 2, [intent(0), intent(1)], {}, 'package_prepare_path_invalid'),
    ('four_rank_preflight', 4, [intent(i) for i in range(4)], {},
     'package_prepare_path_invalid'),
    ('uncovered_sealed_device', 2, [intent(0)], {'sequence_parallel_degree': 1},
     'sequence_parallel_degree_mismatch'),
    ('import_only', 2, [intent(0), intent(1)], {'import_only': True},
     'cooperation_unsupported'),
]
reports = []
executor_source = runtime / 'cozy_runtime/internal/executor.py'
executor_hash = hashlib.sha256(executor_source.read_bytes()).hexdigest()
extension, = (args.tensorfs_python / 'tensorfs').glob('_ext*.so')
extension_hash = hashlib.sha256(extension.read_bytes()).hexdigest()
for name, degree, values, extra, expected in cases:
    output = args.out / name
    output.mkdir()
    (output / 'root').mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('COZY_', 'CUDA_', 'TENSORFS_', 'PYTHON'))}
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(degree)))
    env['OPENBLAS_NUM_THREADS'] = '1'
    report = {'case': name, 'passed': False}
    child = None
    try:
        with tempfile.TemporaryDirectory(prefix='vr-rank-', dir='/dev/shm') as directory:
            address = str(Path(directory) / 'executor.sock')
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(address)
                listener.listen(1)
                listener.settimeout(30)  # This host protocol has an explicit bounded deadline.
                command = [sys.executable, '-I', '-B', str(child_script), str(runtime),
                           str(args.tensorfs_python), address, str(output / 'root'),
                           str(output / 'child.json'), str(args.torch_site)]
                with (
                    (output / 'stdout').open('w') as stdout,
                    (output / 'stderr').open('w') as stderr,
                ):
                    child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(30)
                        hello = exchange(connection, {'cmd': 'hello'})
                        assert hello['ok'] and hello['cooperative_owner']
                        assert hello['pid'] == child.pid
                        frame = {'cmd': 'start', 'devices': env['CUDA_VISIBLE_DEVICES'],
                                 'sequence_parallel_degree': degree,
                                 'package_interface': '/must-not-exist-varena-rank-preflight',
                                 'cooperation': values, **extra}
                        refusal = exchange(connection, frame)
                        assert not refusal['ok'] and refusal['code'] == expected, refusal
                        assert exchange(connection, {'cmd': 'shutdown'})['ok']
                    child.wait(timeout=30)
                state = json.loads((output / 'child.json').read_text())
                assert child.returncode == state['exit_code'] == 0
                # Existing stepcache imports Torch when installed. Refused Start must add no
                # Torch import or device initialization; CPU Torch presence is not CUDA proof.
                assert state['capable'] and state['torch_imported'] == state['torch_on_import']
                assert not state['driver_loaded'] and not state['cuda_initialized']
                assert state['executor_sha256'] == executor_hash
                assert state['extension_sha256'] == extension_hash
                report.update(passed=True, input=frame, refusal=refusal, child=state)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    reports.append(report)

result = {'passed': True, 'scope': 'source Runtime with exact native wheel; CPU protocol only',
          'torch_package': torch_spec.origin,
          'sender_frames_sha256': hashlib.sha256(args.sender_frames.read_bytes()).hexdigest(),
          'rank_selections': rank_selections, 'cases': reports, 'gpu_qualified': False}
(args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'passed': True, 'rank_selections': len(rank_selections),
                  'framed_cases': len(reports), 'gpu_qualified': False}))
