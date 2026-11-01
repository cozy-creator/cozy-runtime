from pathlib import Path
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
Path(sys.argv[5]).write_text(json.dumps(result, indent=2) + '\n')
raise SystemExit(code)
