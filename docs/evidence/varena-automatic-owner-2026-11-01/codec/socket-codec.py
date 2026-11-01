"""Run the actual executor over its local framed socket; Hello must remain CPU-only.

The caller supplies a task-owned native/Python overlay and an immutable Runtime source tree.
Nothing is installed into the SDK, and no Start/GPU operation is sent.
"""
import argparse
import hashlib
import json
import os
import socket
import struct
import subprocess
import tempfile
from pathlib import Path

import tomllib

parser = argparse.ArgumentParser()
parser.add_argument('--python', type=Path, required=True)
parser.add_argument('--runtime-source', type=Path, required=True)
parser.add_argument('--overlay', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--expected-capability', type=int, choices=(0, 1), default=1)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
project = tomllib.loads((args.runtime_source.parent / 'pyproject.toml').read_text())['project']
metadata = args.out / 'metadata' / f"cozy_runtime-{project['version']}.dist-info"
metadata.mkdir(parents=True)
(metadata / 'METADATA').write_text(f"Metadata-Version: 2.1\nName: cozy-runtime\nVersion: {project['version']}\n")
launcher = args.out / 'child.py'
launcher.write_text('''from pathlib import Path
import hashlib, json, sys, threading
sys.path[:0] = [sys.argv[1], sys.argv[2], str(Path(__file__).parent / 'metadata')]
from cozy_runtime.internal import executor, cooperation
main = executor.main
code = main(['--socket', sys.argv[3], '--root', sys.argv[4]])
from tensorfs import _ext, varena
maps = Path('/proc/self/maps').read_text()
torch = sys.modules.get('torch')
result = {'exit_code': code, 'torch_imported': torch is not None,
          'cuda_initialized': bool(torch is not None and torch.cuda.is_initialized()),
          'driver_loaded': 'libcuda.so' in maps or 'libnvidia-ml.so' in maps,
          'cooperation_capability': getattr(getattr(varena, 'Cooperation', None), 'capability', lambda: 0)(),
          'extension': _ext.__file__, 'native_version': _ext.__version__,
          'runtime_executor_source': executor.__file__,
          'runtime_executor_sha256': hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest(),
          'runtime_owner_sha256': hashlib.sha256(Path(cooperation.__file__).read_bytes()).hexdigest(),
          'threads': [t.name for t in threading.enumerate()]}
Path(sys.argv[5]).write_text(json.dumps(result, indent=2) + '\\n')
raise SystemExit(code)
''')
header = struct.Struct('!I')
def read_exact(connection, count):
    data = b''
    while len(data) < count:
        part = connection.recv(count - len(data))
        assert part, 'executor ended before complete frame'
        data += part
    return data

def exchange(connection, value):
    body = json.dumps(value, separators=(',', ':')).encode()
    connection.sendall(header.pack(len(body)) + body)
    size, = header.unpack(read_exact(connection, header.size))
    assert size <= 4 * 1024 * 1024
    return json.loads(read_exact(connection, size))

report = {'passed': False, 'scope': 'actual CPU-only executor Hello/Shutdown socket, no Start or GPU work'}
child = None
try:
    with tempfile.TemporaryDirectory(prefix='owner-hello-', dir=args.out) as directory:
        directory = Path(directory)
        # Use a short, private socket pathname within Linux AF_UNIX's length bound.
        with tempfile.TemporaryDirectory(prefix='vr-hello-', dir='/dev/shm') as sockets:
            address = str(Path(sockets) / 'executor.sock')
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(address); listener.listen(1); listener.settimeout(30)
                command = [str(args.python), '-I', '-B', str(launcher), str(args.overlay.resolve()),
                           str(args.runtime_source.resolve()), address, str(directory),
                           str(args.out / 'child-result.json')]
                env = {k:v for k,v in os.environ.items() if not k.startswith(('COZY_', 'CUDA_', 'TENSORFS_', 'PYTHON'))}
                with (args.out / 'child.stdout').open('w') as stdout, (args.out / 'child.stderr').open('w') as stderr:
                    child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(30)
                        hello = exchange(connection, {'cmd':'hello'})
                        assert hello['ok'] and hello['reply'] == 'hello', hello
                        assert hello['cooperative_owner'] is bool(args.expected_capability), hello
                        assert hello['pid'] == child.pid
                        rejected = exchange(connection, {'cmd': 'start', 'devices': '0',
                            'import_only': True, 'sequence_parallel_degree': 1,
                            'cooperation': {'uuid': 'GPU-13800000-0000-0000-0000-000000000003',
                                            'directory': '/dev/shm/not-opened', 'shared': False, 'priority': 0}})
                        assert rejected.get('code') == 'cooperation_unsupported', rejected
                        shutdown = exchange(connection, {'cmd':'shutdown'})
                        assert shutdown['ok'] and shutdown['reply'] == 'shutdown', shutdown
                    child.wait(timeout=30)
                state = json.loads((args.out / 'child-result.json').read_text())
                assert child.returncode == 0 and state['exit_code'] == 0
                assert not state['cuda_initialized'] and not state['driver_loaded'], state
                assert state['cooperation_capability'] == args.expected_capability
                assert hello['runtime_version'] == project['version']
                assert Path(state['runtime_executor_source']).resolve() == (args.runtime_source / 'cozy_runtime/internal/executor.py').resolve()
                assert state['threads'] == ['MainThread'], state
                report.update(passed=True, command=command, hello=hello, shutdown=shutdown,
                              child=state, pre_cuda_invalid_start=rejected, extension_sha256=hashlib.sha256(Path(state['extension']).read_bytes()).hexdigest())
except BaseException as error:
    report['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    if child is not None and child.poll() is None:
        child.kill(); child.wait(timeout=10)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
